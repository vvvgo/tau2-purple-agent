import json
import logging
import os
import re

from openai import AsyncOpenAI
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text

from prompts import (
    get_system_prompt,
    get_plan_prompt,
    get_plan_to_action,
    get_correction_prompt,
    get_periodic_reminder,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Marker that separates green agent prompt from user messages
USER_MARKER = "Now here are the user messages:\n"
TOOLS_START = "Here's a list of tools"
TOOLS_END = "Additionally, you can respond"

MAX_CONTEXT_MESSAGES = 30


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from reasoning model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = strip_thinking(text)
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()


def parse_action(text: str) -> tuple[dict, bool]:
    """Parse LLM response into action dict. Returns (action, was_fallback)."""
    raw = extract_json(text)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            return parsed, False
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    logger.warning(f"Could not parse JSON, falling back to respond: {text[:200]}")
    return {"name": "respond", "arguments": {"content": text}}, True


def parse_first_message(text: str) -> tuple[str, set[str], str]:
    """Parse the green agent's first message into (policy, tool_names, user_text).

    Green agent format:
      {policy}\n\nHere's a list of tools...\n[...tools JSON...]\n\nAdditionally...\n\n...\n\nNow here are the user messages:\n{user messages}

    Returns policy text, set of known tool names, and user messages.
    """
    # 1. Extract user messages
    user_text = ""
    um_idx = text.find(USER_MARKER)
    if um_idx >= 0:
        user_text = text[um_idx + len(USER_MARKER):].strip()

    # 2. Extract policy (before tools section)
    ts_idx = text.find(TOOLS_START)
    policy = text[:ts_idx].strip() if ts_idx > 0 else text[:3000]

    # 3. Extract tool names
    tool_names = {"respond"}
    for match in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        tool_names.add(match.group(1))

    return policy, tool_names, user_text


class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.messages: list[dict] = []  # OpenAI chat format (system/user/assistant text only)
        self.known_tools: set[str] = set()
        self.turn_count: int = 0
        self._transfer_done: bool = False
        self._last_tool_sigs: list[str] = []
        self._cancelled_reservations: set[str] = set()  # prevent cancel loops

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_text = get_message_text(message)
        if not user_text:
            return

        if self.turn_count == 0:
            self._init(user_text)
        else:
            self.messages.append({"role": "user", "content": user_text})

        self.turn_count += 1

        # Transfer shortcut
        if self._transfer_done:
            out = json.dumps({"name": "respond", "arguments": {
                "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."}})
            self.messages.append({"role": "assistant", "content": out})
            await updater.add_artifact(parts=[Part(root=TextPart(text=out))], name="response")
            return

        # Periodic reminder
        if self.turn_count >= 5 and self.turn_count % 8 == 0:
            r = get_periodic_reminder(None)
            if r:
                self.messages.append({"role": "system", "content": r})

        # ── Call 1: Plan (not stored in history) ──
        plan_text = await self._call_plan()

        # ── Call 2: Execute (stored in history) ──
        action = await self._call_execute(plan_text)
        logger.info(f"Action: {json.dumps(action)[:300]}")

        if action.get("name") == "transfer_to_human_agents":
            self._transfer_done = True

        # Cancel loop guard: block re-cancelling same reservation
        if action.get("name") == "cancel_reservation":
            rid = action.get("arguments", {}).get("reservation_id", "")
            if rid in self._cancelled_reservations:
                logger.warning(f"Blocked duplicate cancel for {rid}")
                action = {"name": "respond", "arguments": {
                    "content": f"Reservation {rid} has already been cancelled."}}
            else:
                self._cancelled_reservations.add(rid)

        # Dedup guard
        action = self._dedup(action)

        out = json.dumps(action)
        self.messages.append({"role": "assistant", "content": out})

        await updater.add_artifact(parts=[Part(root=TextPart(text=out))], name="response")

    def _init(self, text: str):
        """Parse green agent's first message, set up system prompt."""
        policy, tool_names, user_text = parse_first_message(text)

        self.known_tools = tool_names
        logger.info(f"Parsed {len(self.known_tools)} tool names: {self.known_tools}")

        # System prompt = our instructions + domain policy
        self.messages = [{"role": "system", "content": get_system_prompt("") + "\n\n" + policy}]

        # First user message(s) from the green agent prompt
        if user_text:
            self.messages.append({"role": "user", "content": user_text})
        else:
            self.messages.append({"role": "user", "content": text})

    def _get_trimmed_messages(self) -> list[dict]:
        """Keep system + first user message + last N turns."""
        if len(self.messages) <= MAX_CONTEXT_MESSAGES:
            return list(self.messages)
        preserved = self.messages[:2]
        preserved.append({
            "role": "user",
            "content": "[Earlier conversation messages omitted. Continue from here.]"
        })
        recent = self.messages[-(MAX_CONTEXT_MESSAGES - 3):]
        return preserved + recent

    async def _call_plan(self) -> str:
        """Call 1: reasoning step. Result is NOT stored in message history."""
        try:
            plan_messages = self._get_trimmed_messages() + [{
                "role": "user",
                "content": get_plan_prompt(),
            }]
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=plan_messages,
                temperature=0,
                max_completion_tokens=1024,
            )
            plan = strip_thinking(resp.choices[0].message.content or "")
            logger.info(f"Plan (first 300): {plan[:300]}")
            return plan
        except Exception as e:
            logger.warning(f"Plan call failed (continuing without plan): {e}")
            return ""

    async def _call_execute(self, plan_text: str) -> dict:
        """Call 2: produce JSON action. Result IS stored in message history."""
        messages = self._get_trimmed_messages()
        if plan_text:
            messages = messages + [{
                "role": "user",
                "content": get_plan_to_action(plan_text),
            }]

        for attempt in range(2):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_completion_tokens=4096,
                )
                reply = resp.choices[0].message.content or ""
                logger.info(f"Execute reply (first 300): {reply[:300]}")

                action, was_fallback = parse_action(reply)

                # Validate tool name
                if not was_fallback and self.known_tools and action["name"] not in self.known_tools:
                    logger.warning(f"Unknown tool '{action['name']}', known: {self.known_tools}")
                    was_fallback = True

                if not was_fallback:
                    return action

                # Retry with correction prompt
                if attempt == 0:
                    logger.warning("Invalid response, retrying with correction prompt")
                    messages = messages + [
                        {"role": "assistant", "content": reply},
                        {"role": "user", "content": get_correction_prompt()},
                    ]
                    continue

                # Final fallback: return as text response
                return {"name": "respond", "arguments": {"content": reply}}

            except Exception as e:
                logger.error(f"LLM error (attempt {attempt + 1}): {e}")
                if attempt == 1:
                    return {"name": "respond", "arguments": {
                        "content": "Technical difficulties. Please try again."}}
                import asyncio
                await asyncio.sleep(2)

        return {"name": "respond", "arguments": {"content": "Processing error."}}

    def _dedup(self, action: dict) -> dict:
        """Block after 3 identical consecutive tool calls."""
        sig = json.dumps(action, sort_keys=True)
        if len(self._last_tool_sigs) >= 3 and all(s == sig for s in self._last_tool_sigs[-3:]):
            self._last_tool_sigs = []
            return {"name": "respond", "arguments": {
                "content": "I've attempted this multiple times. Could you clarify?"}}
        self._last_tool_sigs.append(sig)
        if action.get("name") == "respond":
            self._last_tool_sigs = []
        return action
