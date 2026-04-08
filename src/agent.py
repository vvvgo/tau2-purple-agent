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

def extract_tool_names(text: str) -> set[str]:
    """Extract tool names from the first message using a broad regex."""
    names = set()
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        name = m.group(1)
        if not any(c in name for c in [" ", ".", ","]) and len(name) < 50:
            names.add(name)
    names.add("respond")
    return names


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict | None:
    text = strip_thinking(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 0:
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
            parsed = json.loads(brace.group())
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
    return None


PLACEHOLDER_RE = re.compile(
    r"\b(placeholder|unknown|n/a|dummy|fake|example|xxx|your_|my_)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Agent — single call, JSON mode, with transfer loop protection
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = os.environ.get("AGENT_MODEL", "gpt-4o-mini")
        self.history: list[dict] = []
        self.known_tools: set[str] = set()
        self.turn_count: int = 0
        self.last_tool_sig: str | None = None
        self._dup_count: int = 0
        self._transfer_done: bool = False  # Track if transfer already happened

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_text = get_message_text(message)
        if not user_text:
            return

        # First message: extract tools, set system prompt
        if self.turn_count == 0:
            self.known_tools = extract_tool_names(user_text)
            self.history.append({"role": "system", "content": get_system_prompt("")})
            logger.info(f"Tools ({len(self.known_tools)}): {sorted(self.known_tools)}")

        self.history.append({"role": "user", "content": user_text})
        self.turn_count += 1

        # If transfer already done, just respond with the transfer message
        if self._transfer_done:
            action = {"name": "respond", "arguments": {
                "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
            }}
            response_text = json.dumps(action)
            self.history.append({"role": "assistant", "content": response_text})
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=response_text))],
                name="response",
            )
            return

        # Periodic reminder
        if self.turn_count > 4 and self.turn_count % 8 == 0:
            reminder = get_periodic_reminder(None)
            if reminder:
                self.history.append({"role": "system", "content": reminder})

        self._trim_history()

        # Single LLM call with JSON mode
        action = await self._generate_action()

        # If agent calls transfer_to_human_agents, mark it so next turn we just respond
        if action.get("name") == "transfer_to_human_agents":
            self._transfer_done = True

        response_text = json.dumps(action)
        self.history.append({"role": "assistant", "content": response_text})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="response",
        )

    async def _generate_action(self) -> dict:
        response_text = await self._call_llm(json_mode=True)
        logger.info(f"LLM: {response_text[:200]}")

        action = extract_json(response_text)

        # Validate — retry once if invalid
        if not action or not self._is_valid(action):
            retry = await self._call_llm(
                extra_msg=(
                    "Your previous response was invalid. "
                    "Output EXACTLY one JSON: {\"name\": \"<tool>\", \"arguments\": {...}}. "
                    "Available: " + ", ".join(sorted(self.known_tools))
                ),
                json_mode=True,
            )
            action = extract_json(retry)
            if not action or not self._is_valid(action):
                action = {"name": "respond", "arguments": {"content": response_text[:500]}}

        # Duplicate guard — block after 1 repeat
        sig = json.dumps(action, sort_keys=True)
        if sig == self.last_tool_sig and action["name"] != "respond":
            self._dup_count += 1
            if self._dup_count >= 1:
                action = {"name": "respond", "arguments": {
                    "content": "I've already attempted that action. Let me try a different approach. Could you provide more details or clarify your request?"
                }}
                self._dup_count = 0
        else:
            self._dup_count = 0
        self.last_tool_sig = sig

        return action

    def _is_valid(self, action: dict) -> bool:
        if "name" not in action or "arguments" not in action:
            return False
        name, args = action["name"], action["arguments"]
        if not isinstance(name, str) or not isinstance(args, dict):
            return False
        if name not in self.known_tools:
            logger.warning(f"Unknown tool: {name}, known: {sorted(self.known_tools)}")
            return False
        if name != "respond":
            for v in args.values():
                if isinstance(v, str) and PLACEHOLDER_RE.search(v):
                    logger.warning(f"Placeholder in args: {v}")
                    return False
        if name == "respond":
            content = args.get("content", "")
            if isinstance(content, str) and content.strip().startswith("{"):
                try:
                    inner = json.loads(content)
                    if isinstance(inner, dict) and "name" in inner:
                        return False
                except json.JSONDecodeError:
                    pass
        return True

    async def _call_llm(
        self,
        extra_msg: str | None = None,
        messages: list[dict] | None = None,
        json_mode: bool = False,
        max_retries: int = 3,
    ) -> str:
        msgs = messages or list(self.history)
        if extra_msg:
            msgs = msgs + [{"role": "user", "content": extra_msg}]

        kwargs = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0,
            "max_completion_tokens": 4096,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(max_retries):
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"LLM error (attempt {attempt+1}): {e}")
                if attempt == max_retries - 1:
                    raise
                import asyncio
                await asyncio.sleep(min(2 ** attempt, 10))
        return ""

    def _trim_history(self, max_msgs: int = 30):
        if len(self.history) <= max_msgs:
            return
        marker = {"role": "system", "content": "[Earlier conversation messages omitted. Continue from here.]"}
        self.history = self.history[:2] + [marker] + self.history[-(max_msgs - 3):]
