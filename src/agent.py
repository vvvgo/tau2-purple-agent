import json
import logging
import os
import re

from openai import AsyncOpenAI
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text

from prompts import get_system_prompt, get_periodic_reminder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_tools_from_prompt(text: str) -> list[dict]:
    """Extract OpenAI tool schemas from the first message (green agent prompt)."""
    # The green agent embeds tool schemas as a JSON array in the prompt
    # Find the array between "tools you can use" and the respond tool
    try:
        # Find the JSON array of tools
        start = text.find('[{"type"')
        if start == -1:
            start = text.find('[{\n')
        if start == -1:
            return []

        # Find matching end bracket
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '[':
                depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        tools_json = json.loads(text[start:end])
        # Convert to OpenAI tools format
        tools = []
        for t in tools_json:
            if isinstance(t, dict):
                if "type" in t and "function" in t:
                    tools.append(t)
                elif "name" in t:
                    # Already in function format, wrap it
                    tools.append({"type": "function", "function": t})
        return tools
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse tools: {e}")
        return []


def extract_policy(text: str) -> str:
    """Extract the domain policy from the first message."""
    # Policy is at the beginning, before "Here's a list of tools"
    marker = "Here's a list of tools"
    idx = text.find(marker)
    if idx > 0:
        return text[:idx].strip()
    return text[:3000]  # fallback


def extract_tool_names(text: str) -> set[str]:
    """Extract tool names from the first message using regex."""
    names = set()
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        name = m.group(1)
        if not any(c in name for c in [" ", ".", ","]) and len(name) < 50:
            names.add(name)
    return names


# ---------------------------------------------------------------------------
# Agent — native function calling with proper message history
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.messages: list[dict] = []  # OpenAI chat messages
        self.tools: list[dict] = []  # OpenAI tool schemas
        self.tool_names: set[str] = set()
        self.turn_count: int = 0
        self._transfer_done: bool = False
        self._last_tool_sigs: list[str] = []
        self._pending_tool_call_id: str | None = None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_text = get_message_text(message)
        if not user_text:
            return

        # First message: parse tools and policy from green agent prompt
        if self.turn_count == 0:
            self._init_from_first_message(user_text)
        else:
            # Subsequent messages: tool results or user messages
            self._add_incoming_message(user_text)

        self.turn_count += 1

        # If transfer already done, respond immediately
        if self._transfer_done:
            response_text = json.dumps({
                "name": "respond",
                "arguments": {"content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."}
            })
            self.messages.append({"role": "assistant", "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."})
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response_text))],
                name="response",
            )
            return

        # Inject periodic reminder
        if self.turn_count >= 5 and self.turn_count % 6 == 0:
            reminder = get_periodic_reminder(None)
            if reminder:
                self.messages.append({"role": "system", "content": reminder})

        self._trim_history()

        # Generate response using native function calling
        action_json = await self._generate_action()

        # Track transfer
        if action_json.get("name") == "transfer_to_human_agents":
            self._transfer_done = True

        response_text = json.dumps(action_json)

        # Add to history based on what happened
        action_name = action_json.get("name", "respond")
        if action_name == "respond":
            # Text response — store as assistant content
            self.messages.append({
                "role": "assistant",
                "content": action_json.get("arguments", {}).get("content", "")
            })
        else:
            # Tool call — store as assistant with tool_calls
            tool_call_id = f"call_{self.turn_count}"
            self.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": action_name,
                        "arguments": json.dumps(action_json.get("arguments", {}))
                    }
                }]
            })
            # We'll get the tool result in the next message
            self._pending_tool_call_id = tool_call_id

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="response",
        )

    def _init_from_first_message(self, text: str):
        """Parse first message: extract policy, tools, and user's first message."""
        # Extract tools from the green agent prompt
        self.tools = parse_tools_from_prompt(text)
        self.tool_names = extract_tool_names(text)
        logger.info(f"Parsed {len(self.tools)} tools: {[t.get('function', {}).get('name', '') for t in self.tools]}")

        # Extract just the policy
        policy = extract_policy(text)

        # Build system prompt: our instructions + the domain policy
        system_content = get_system_prompt("") + "\n\n## DOMAIN POLICY (from environment)\n" + policy

        self.messages = [{"role": "system", "content": system_content}]

        # Extract user messages from the prompt
        # Green agent sends: agent_prompt + "\n\nNow here are the user messages:\n" + messages
        user_marker = "Now here are the user messages:\n"
        idx = text.find(user_marker)
        if idx >= 0:
            user_content = text[idx + len(user_marker):].strip()
            if user_content:
                self.messages.append({"role": "user", "content": user_content})
        else:
            # Fallback: use everything after the tool definitions
            last_example = text.rfind('"content": "Hello, how can I help you today?"')
            if last_example > 0:
                after = text[last_example:]
                end_of_examples = after.find('\n\n')
                if end_of_examples > 0:
                    remaining = after[end_of_examples:].strip()
                    if remaining:
                        self.messages.append({"role": "user", "content": remaining})

    def _add_incoming_message(self, text: str):
        """Add incoming message to history with proper role."""
        # If we have a pending tool call, this must be the tool result
        pending_id = getattr(self, '_pending_tool_call_id', None)
        if pending_id:
            self.messages.append({
                "role": "tool",
                "tool_call_id": pending_id,
                "content": text
            })
            self._pending_tool_call_id = None
        else:
            # User message
            self.messages.append({"role": "user", "content": text})

    async def _generate_action(self) -> dict:
        """Call LLM with native function calling and return action dict."""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0,
            "max_completion_tokens": 2048,
        }

        # Add tools if we have them
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]
                msg = choice.message

                # Case 1: Model wants to call a tool
                if msg.tool_calls and len(msg.tool_calls) > 0:
                    tc = msg.tool_calls[0]  # Take first tool call only
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    action = {"name": tc.function.name, "arguments": args}

                    # Duplicate guard
                    sig = json.dumps(action, sort_keys=True)
                    if len(self._last_tool_sigs) >= 3 and all(s == sig for s in self._last_tool_sigs[-3:]):
                        self._last_tool_sigs = []
                        return {"name": "respond", "arguments": {
                            "content": "I've attempted this action multiple times. Could you clarify your request?"
                        }}
                    self._last_tool_sigs.append(sig)

                    return action

                # Case 2: Model responds with text (no tool call)
                content = msg.content or ""
                self._last_tool_sigs = []

                # Check if the text content is actually a JSON action (fallback)
                if content.strip().startswith("{"):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                            return parsed
                    except json.JSONDecodeError:
                        pass

                return {"name": "respond", "arguments": {"content": content}}

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return {"name": "respond", "arguments": {
                        "content": "I apologize, I'm experiencing technical difficulties. Please try again."
                    }}
                import asyncio
                await asyncio.sleep(min(2 ** attempt, 10))

        return {"name": "respond", "arguments": {"content": "I'm having trouble processing your request."}}

    def _trim_history(self, max_msgs: int = 60):
        """Keep history manageable."""
        if len(self.messages) <= max_msgs:
            return
        # Keep system message + recent messages
        marker = {"role": "system", "content": "[Earlier conversation omitted. Continue from recent context.]"}
        self.messages = self.messages[:1] + [marker] + self.messages[-(max_msgs - 2):]
