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
    """Extract OpenAI tool schemas from the green agent prompt.

    The green agent embeds tools as: json.dumps([tool.openai_schema for tool in tools], indent=2)
    Format: [{\n  "type": "function",\n  "function": {\n    "name": "...", ...}\n}, ...]
    """
    try:
        # Try multiple patterns to find the tool array
        # Pattern 1: indented format [{\n  "type"
        for pattern in [r'\[\s*\{\s*"type"\s*:\s*"function"', r'\[\s*\{\s*\n\s*"type"']:
            match = re.search(pattern, text)
            if match:
                start = match.start()
                # Find matching end bracket
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == '[':
                        depth += 1
                    elif text[i] == ']':
                        depth -= 1
                        if depth == 0:
                            chunk = text[start:i + 1]
                            tools_json = json.loads(chunk)
                            tools = []
                            for t in tools_json:
                                if isinstance(t, dict) and "type" in t and "function" in t:
                                    tools.append(t)
                            if tools:
                                return tools
                            break
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse tools from prompt: {e}")
        return []


def extract_policy(text: str) -> str:
    """Extract the domain policy from the first message."""
    marker = "Here's a list of tools"
    idx = text.find(marker)
    if idx > 0:
        return text[:idx].strip()
    return text[:4000]


def extract_tool_names(text: str) -> set[str]:
    """Extract tool names via regex (fallback for JSON mode)."""
    names = set()
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        name = m.group(1)
        if not any(c in name for c in [" ", ".", ","]) and len(name) < 50:
            names.add(name)
    names.add("respond")
    return names


PLACEHOLDER_RE = re.compile(
    r"\b(placeholder|unknown|n/a|dummy|fake|example|xxx|your_|my_)\b", re.IGNORECASE
)


def extract_json(text: str) -> dict | None:
    """Extract JSON action from text (for JSON mode fallback)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed:
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            return parsed[0] if isinstance(parsed, list) and parsed else parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group())
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Agent — native function calling with JSON mode fallback
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.messages: list[dict] = []
        self.tools: list[dict] = []  # OpenAI native tool schemas
        self.tool_names: set[str] = set()  # For JSON mode fallback
        self.use_native_tools: bool = False
        self.turn_count: int = 0
        self._transfer_done: bool = False
        self._last_tool_sigs: list[str] = []
        self._pending_tool_call_id: str | None = None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_text = get_message_text(message)
        if not user_text:
            return

        if self.turn_count == 0:
            self._init_from_first_message(user_text)
        else:
            self._add_incoming_message(user_text)

        self.turn_count += 1

        # Transfer shortcut
        if self._transfer_done:
            response_text = json.dumps({
                "name": "respond",
                "arguments": {"content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."}
            })
            self.messages.append({"role": "assistant", "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."})
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response_text))], name="response")
            return

        # Periodic reminder
        if self.turn_count >= 5 and self.turn_count % 6 == 0:
            reminder = get_periodic_reminder(None)
            if reminder:
                self.messages.append({"role": "system", "content": reminder})

        self._trim_history()

        # Generate action
        if self.use_native_tools:
            action_json = await self._generate_native()
        else:
            action_json = await self._generate_json_mode()

        logger.info(f"Action: {json.dumps(action_json)[:300]}")

        if action_json.get("name") == "transfer_to_human_agents":
            self._transfer_done = True

        response_text = json.dumps(action_json)

        # Update history
        action_name = action_json.get("name", "respond")
        if self.use_native_tools:
            if action_name == "respond":
                self.messages.append({
                    "role": "assistant",
                    "content": action_json.get("arguments", {}).get("content", "")
                })
            else:
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
                self._pending_tool_call_id = tool_call_id
        else:
            # JSON mode: store raw JSON as assistant content
            self.messages.append({"role": "assistant", "content": response_text})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))], name="response")

    def _init_from_first_message(self, text: str):
        """Parse first message from green agent."""
        # Try to parse native tools
        self.tools = parse_tools_from_prompt(text)
        self.tool_names = extract_tool_names(text)

        if self.tools:
            self.use_native_tools = True
            logger.info(f"Native tools mode: {len(self.tools)} tools parsed")
        else:
            self.use_native_tools = False
            logger.info(f"JSON mode fallback: {len(self.tool_names)} tool names found")

        # Extract policy
        policy = extract_policy(text)

        # Build system prompt
        system_content = get_system_prompt("")
        if not self.use_native_tools:
            # In JSON mode, append response format instructions
            system_content += """

## RESPONSE FORMAT
Output ONLY a raw JSON object. No markdown, no code fences, no extra text.
- Tool call: {"name": "tool_name", "arguments": {"param": "value"}}
- User reply: {"name": "respond", "arguments": {"content": "message"}}
"""

        self.messages = [{"role": "system", "content": system_content + "\n\n## DOMAIN POLICY (from environment)\n" + policy}]

        # Extract user messages
        user_marker = "Now here are the user messages:\n"
        idx = text.find(user_marker)
        if idx >= 0:
            user_content = text[idx + len(user_marker):].strip()
            if user_content:
                self.messages.append({"role": "user", "content": user_content})
        else:
            # Fallback: entire text as user message (green agent sends everything)
            self.messages.append({"role": "user", "content": text})

    def _add_incoming_message(self, text: str):
        """Add incoming message with proper role."""
        if self._pending_tool_call_id:
            self.messages.append({
                "role": "tool",
                "tool_call_id": self._pending_tool_call_id,
                "content": text
            })
            self._pending_tool_call_id = None
        else:
            self.messages.append({"role": "user", "content": text})

    # --- Native function calling ---

    async def _generate_native(self) -> dict:
        """Use OpenAI native tool calling."""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0,
            "max_completion_tokens": 2048,
            "tools": self.tools,
            "tool_choice": "auto",
        }

        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message

                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    action = {"name": tc.function.name, "arguments": args}
                    return self._dedup_check(action)

                content = msg.content or ""
                self._last_tool_sigs = []

                # Fallback: check if text is JSON action
                if content.strip().startswith("{"):
                    parsed = extract_json(content)
                    if parsed and "name" in parsed and "arguments" in parsed:
                        return self._dedup_check(parsed) if parsed["name"] != "respond" else parsed

                return {"name": "respond", "arguments": {"content": content}}

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return {"name": "respond", "arguments": {"content": "Technical difficulties. Please try again."}}
                import asyncio
                await asyncio.sleep(min(2 ** attempt, 10))

        return {"name": "respond", "arguments": {"content": "Processing error."}}

    # --- JSON mode fallback ---

    async def _generate_json_mode(self) -> dict:
        """Fallback: JSON mode (like v9)."""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0,
            "max_completion_tokens": 4096,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                action = extract_json(text)

                if action and self._is_valid_action(action):
                    return self._dedup_check(action) if action["name"] != "respond" else action

                # Retry once
                if attempt == 0:
                    retry_resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=self.messages + [{"role": "user", "content":
                            "Invalid response. Output ONE JSON: {\"name\": \"<tool>\", \"arguments\": {...}}. "
                            "Available: " + ", ".join(sorted(self.tool_names))}],
                        temperature=0,
                        max_completion_tokens=4096,
                        response_format={"type": "json_object"},
                    )
                    action = extract_json(retry_resp.choices[0].message.content or "")
                    if action and self._is_valid_action(action):
                        return self._dedup_check(action) if action["name"] != "respond" else action

                return {"name": "respond", "arguments": {"content": text[:500]}}

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return {"name": "respond", "arguments": {"content": "Technical difficulties."}}
                import asyncio
                await asyncio.sleep(min(2 ** attempt, 10))

        return {"name": "respond", "arguments": {"content": "Processing error."}}

    def _is_valid_action(self, action: dict) -> bool:
        if "name" not in action or "arguments" not in action:
            return False
        name, args = action["name"], action["arguments"]
        if not isinstance(name, str) or not isinstance(args, dict):
            return False
        if name not in self.tool_names:
            return False
        if name != "respond":
            for v in args.values():
                if isinstance(v, str) and PLACEHOLDER_RE.search(v):
                    return False
        return True

    def _dedup_check(self, action: dict) -> dict:
        """Block after 3 identical consecutive tool calls."""
        sig = json.dumps(action, sort_keys=True)
        if len(self._last_tool_sigs) >= 3 and all(s == sig for s in self._last_tool_sigs[-3:]):
            self._last_tool_sigs = []
            return {"name": "respond", "arguments": {
                "content": "I've attempted this action multiple times. Could you clarify?"
            }}
        self._last_tool_sigs.append(sig)
        return action

    def _trim_history(self, max_msgs: int = 60):
        if len(self.messages) <= max_msgs:
            return
        marker = {"role": "system", "content": "[Earlier conversation omitted.]"}
        self.messages = self.messages[:1] + [marker] + self.messages[-(max_msgs - 2):]
