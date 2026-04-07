import json
import logging
import os
import re

from openai import AsyncOpenAI
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text

from prompts import get_system_prompt, get_periodic_reminder, PLAN_PROMPT, ACT_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_tool_names(text: str) -> set[str]:
    """Extract tool names from the first message using a broad regex."""
    names = set()
    # Broad match: any "name": "..." inside tool definitions
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        name = m.group(1)
        # Filter out non-tool names (descriptions, types, etc.)
        if not any(c in name for c in [" ", ".", ","]) and len(name) < 50:
            names.add(name)
    names.add("respond")
    return names


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_json(text: str) -> dict | None:
    """Robustly extract JSON from LLM output (handles reasoning text around JSON)."""
    text = strip_thinking(text)

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) > 0:
            return parsed[0]
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try code fences
    fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            return parsed[0] if isinstance(parsed, list) and parsed else parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    # Try first {...} block (handles reasoning text before/after JSON)
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
# Agent — plan-then-act (2 calls), NO json_mode on plan, temperature=0
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = os.environ.get("AGENT_MODEL", "gpt-5.4-mini")
        self.history: list[dict] = []
        self.known_tools: set[str] = set()
        self.turn_count: int = 0
        self.last_tool_sig: str | None = None

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

        # Periodic reminder
        if self.turn_count > 4 and self.turn_count % 8 == 0:
            reminder = get_periodic_reminder(None)
            if reminder:
                self.history.append({"role": "system", "content": reminder})

        self._trim_history()

        # Plan-then-act: 2 LLM calls
        action = await self._plan_and_act()

        response_text = json.dumps(action)
        self.history.append({"role": "assistant", "content": response_text})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response_text))],
            name="response",
        )

    async def _plan_and_act(self) -> dict:
        # Step 1: Plan — reasoning in plain text, NOT stored in history
        plan = await self._call_llm(
            extra_msg=PLAN_PROMPT,
            json_mode=False,
            max_tokens=1024,
        )
        logger.info(f"Plan: {plan[:300]}")

        # Step 2: Act — produce JSON action using the plan
        act_prompt = ACT_PROMPT.format(plan=plan)
        act_messages = self.history + [{"role": "user", "content": act_prompt}]

        response_text = await self._call_llm(
            messages=act_messages,
            json_mode=False,  # NO json_mode — let model reason freely
            max_tokens=4096,
        )
        logger.info(f"Act: {response_text[:200]}")

        action = extract_json(response_text)

        # Validate — retry once if invalid
        if not action or not self._is_valid(action):
            retry = await self._call_llm(
                extra_msg=(
                    "Your previous response was not valid JSON. "
                    "Output EXACTLY one JSON object: {\"name\": \"<tool>\", \"arguments\": {...}}. "
                    "Available tools: " + ", ".join(sorted(self.known_tools))
                ),
                json_mode=True,  # Force JSON on retry
            )
            action = extract_json(retry)
            if not action or not self._is_valid(action):
                action = {"name": "respond", "arguments": {"content": response_text[:500]}}

        # Duplicate guard (only block if 3+ consecutive identical calls)
        sig = json.dumps(action, sort_keys=True)
        if sig == self.last_tool_sig and action["name"] != "respond":
            # Allow one repeat, block on second
            if hasattr(self, '_dup_count') and self._dup_count >= 1:
                action = {"name": "respond", "arguments": {
                    "content": "I've attempted this action multiple times. Let me try a different approach."
                }}
                self._dup_count = 0
            else:
                self._dup_count = getattr(self, '_dup_count', 0) + 1
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
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> str:
        msgs = messages or list(self.history)
        if extra_msg:
            msgs = msgs + [{"role": "user", "content": extra_msg}]

        kwargs = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0,
            "max_completion_tokens": max_tokens,
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
        # Keep [0]=system, [1]=first user msg (policy+tools), marker, then last N
        marker = {"role": "system", "content": "[Earlier conversation messages omitted. Continue from here.]"}
        self.history = self.history[:2] + [marker] + self.history[-(max_msgs - 3):]
