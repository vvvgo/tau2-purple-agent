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

# Marker that separates green agent prompt from user messages
USER_MARKER = "Now here are the user messages:\n"
TOOLS_START = "Here's a list of tools"
TOOLS_END = "Additionally, you can respond"


def parse_first_message(text: str) -> tuple[str, list[dict], str]:
    """Parse the green agent's first message into (policy, tools, user_text).

    Green agent format:
      {policy}\n\nHere's a list of tools...\n[...tools JSON...]\n\nAdditionally...\n\n...\n\nNow here are the user messages:\n{user messages}
    """
    # 1. Extract user messages
    user_text = ""
    um_idx = text.find(USER_MARKER)
    if um_idx >= 0:
        user_text = text[um_idx + len(USER_MARKER):].strip()

    # 2. Extract policy (before tools section)
    ts_idx = text.find(TOOLS_START)
    policy = text[:ts_idx].strip() if ts_idx > 0 else text[:3000]

    # 3. Extract tools JSON array
    tools = []
    # Find the [ that starts the array (after "tools you can use")
    arr_start = text.find("\n[", ts_idx if ts_idx > 0 else 0)
    if arr_start >= 0:
        arr_start += 1  # skip the \n
        # Find the matching ]
        te_idx = text.find(TOOLS_END)
        if te_idx > arr_start:
            chunk = text[arr_start:te_idx].strip()
            try:
                tools = json.loads(chunk)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tools JSON, len={len(chunk)}")

    return policy, tools, user_text


class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.messages: list[dict] = []  # OpenAI chat format
        self.tools: list[dict] = []  # OpenAI native tools
        self.turn_count: int = 0
        self._transfer_done: bool = False
        self._last_tool_sigs: list[str] = []
        self._pending_tool_call_id: str | None = None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_text = get_message_text(message)
        if not user_text:
            return

        if self.turn_count == 0:
            self._init(user_text)
        else:
            self._add_message(user_text)

        self.turn_count += 1

        # Transfer shortcut
        if self._transfer_done:
            out = json.dumps({"name": "respond", "arguments": {
                "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."}})
            self.messages.append({"role": "assistant",
                "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."})
            await updater.add_artifact(parts=[Part(root=TextPart(text=out))], name="response")
            return

        # Periodic reminder
        if self.turn_count >= 5 and self.turn_count % 6 == 0:
            r = get_periodic_reminder(None)
            if r:
                self.messages.append({"role": "system", "content": r})

        self._trim_history()

        # Call LLM with native tools
        action = await self._call_llm()
        logger.info(f"Action: {json.dumps(action)[:300]}")

        if action.get("name") == "transfer_to_human_agents":
            self._transfer_done = True

        out = json.dumps(action)

        # Update message history
        name = action.get("name", "respond")
        if name == "respond":
            self.messages.append({
                "role": "assistant",
                "content": action.get("arguments", {}).get("content", "")
            })
        else:
            cid = f"call_{self.turn_count}"
            self.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": cid, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(action.get("arguments", {}))}
                }]
            })
            self._pending_tool_call_id = cid

        await updater.add_artifact(parts=[Part(root=TextPart(text=out))], name="response")

    def _init(self, text: str):
        """Parse green agent's first message, set up system prompt and tools."""
        policy, tools, user_text = parse_first_message(text)

        self.tools = tools
        logger.info(f"Parsed {len(self.tools)} native tools")

        # System prompt = our instructions + domain policy
        self.messages = [{"role": "system", "content": get_system_prompt("") + "\n\n" + policy}]

        # First user message(s) from the green agent prompt
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
        else:
            # Shouldn't happen, but fallback
            self.messages.append({"role": "user", "content": text})

    def _add_message(self, text: str):
        """Add incoming message — tool result or user message."""
        if self._pending_tool_call_id:
            self.messages.append({
                "role": "tool",
                "tool_call_id": self._pending_tool_call_id,
                "content": text
            })
            self._pending_tool_call_id = None
        else:
            self.messages.append({"role": "user", "content": text})

    async def _call_llm(self) -> dict:
        """Call OpenAI with native function calling."""
        kwargs = {
            "model": self.model,
            "messages": self.messages,
            "temperature": 0,
            "max_completion_tokens": 2048,
        }
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"

        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message

                # Native tool call
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    action = {"name": tc.function.name, "arguments": args}
                    return self._dedup(action)

                # Text response
                content = msg.content or ""
                self._last_tool_sigs = []
                return {"name": "respond", "arguments": {"content": content}}

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    return {"name": "respond", "arguments": {
                        "content": "Technical difficulties. Please try again."}}
                import asyncio
                await asyncio.sleep(min(2 ** attempt, 10))

        return {"name": "respond", "arguments": {"content": "Processing error."}}

    def _dedup(self, action: dict) -> dict:
        """Block after 3 identical consecutive tool calls."""
        sig = json.dumps(action, sort_keys=True)
        if len(self._last_tool_sigs) >= 3 and all(s == sig for s in self._last_tool_sigs[-3:]):
            self._last_tool_sigs = []
            return {"name": "respond", "arguments": {
                "content": "I've attempted this multiple times. Could you clarify?"}}
        self._last_tool_sigs.append(sig)
        return action

    def _trim_history(self, max_msgs: int = 60):
        if len(self.messages) <= max_msgs:
            return
        marker = {"role": "system", "content": "[Earlier conversation omitted.]"}
        self.messages = self.messages[:1] + [marker] + self.messages[-(max_msgs - 2):]
