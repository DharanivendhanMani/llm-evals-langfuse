"""Arusuvai Kitchen support assistant: a small tool-use agent on Groq (OpenAI-compatible), traced with Langfuse.

Tracing layout (one trace per chat turn, one session per conversation):

    handle-support-message          (agent)  input: user message, output: reply
    ├── generate-response           (generation)  messages, reasoning, tool calls, usage
    ├── get-opening-hours           (tool)
    ├── generate-response           (generation)
    ├── search-menu                 (tool)
    └── generate-response           (generation)

The evals live in evals/ and scripts/; they call `answer()` below.
"""

import contextvars
import difflib
import json
import os
import re
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load env vars BEFORE importing Langfuse so it initializes with the right credentials.
load_dotenv()

from langfuse import Langfuse, get_client, observe, propagate_attributes  # noqa: E402
from langfuse.types import MaskOtelSpansParams, MaskOtelSpansResult, OtelSpanPatch  # noqa: E402

import restaurant_data as rd  # noqa: E402

APP_VERSION = "0.2.0"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
PROMPT_VERSION = os.environ.get("PROMPT_VERSION", "v2")  # prompts/support-<version>.txt
MAX_AGENT_STEPS = 8
REASONING_EFFORT = "medium"
TZ = ZoneInfo(rd.TIMEZONE)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def _redact(value: str) -> str:
    return EMAIL_RE.sub("[REDACTED_EMAIL]", value)


def mask_otel_spans(*, params: MaskOtelSpansParams) -> Optional[MaskOtelSpansResult]:
    """Redact email addresses from every string attribute before export.

    Runs on all spans this client exports; prompts and completions live in span attributes.
    """
    patches = {}
    for identifier, span in params.spans.items():
        redacted = {
            key: _redact(value)
            for key, value in span.attributes.items()
            if isinstance(value, str) and EMAIL_RE.search(value)
        }
        if redacted:
            patches[identifier] = OtelSpanPatch(set_attributes=redacted)
    return MaskOtelSpansResult(span_patches=patches)


# Initialize Langfuse first (so it has the masking hook), then create the traced OpenAI client.
langfuse = Langfuse(mask_otel_spans=mask_otel_spans)

from openai import OpenAI  # noqa: E402

# Groq exposes an OpenAI-compatible API. The plain client is used (not the langfuse.openai
# wrapper) because the wrapper drops the model's `reasoning`, which traces should capture.
client = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1", max_retries=6)


def build_system_prompt(version: Optional[str] = None) -> str:
    path = Path(__file__).parent / "prompts" / f"support-{version or PROMPT_VERSION}.txt"
    return path.read_text().replace("{phone}", rd.PHONE).strip()


# --- Clock and tool-call log (both overridable per eval item so results are reproducible) -------

_NOW_OVERRIDE: contextvars.ContextVar[Optional[datetime]] = contextvars.ContextVar("now_override", default=None)
_TOOL_LOG: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar("tool_log", default=None)


def current_time() -> datetime:
    return _NOW_OVERRIDE.get() or datetime.now(TZ)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_opening_hours",
            "description": "Get the weekly opening hours, whether the restaurant is open right now, and any "
            "holiday notes. Optionally pass a weekday name or an ISO date (YYYY-MM-DD) to look at one day.",
            "parameters": {
                "type": "object",
                "properties": {"day": {"type": "string", "description": "e.g. 'Saturday' or '2026-09-30'"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_menu",
            "description": "Search the menu by dish name or category (e.g. 'dosai', 'desserts', 'fish meals'). "
            "Returns names, prices and availability notes.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_restaurant_info",
            "description": "Get restaurant details: contact (address, phone, website), services (pickup, "
            "delivery, group orders, catering), catering tray prices or delivery fees by distance.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string", "enum": ["contact", "services", "catering", "delivery"]}},
                "required": ["topic"],
            },
        },
    },
]

# --- Tools (each call is its own `tool` observation) -------------------------------------------

DAYS = list(rd.HOURS)  # Monday-first


def _fmt_time(hhmm: str) -> str:
    h, m = map(int, hhmm.split(":"))
    suffix = "AM" if h < 12 else "PM"
    return f"{(h % 12) or 12}{f':{m:02d}' if m else ''} {suffix}"


def _fmt_range(day: str) -> str:
    opens, closes = rd.HOURS[day]
    return f"{_fmt_time(opens)} - {_fmt_time(closes)}"


def _resolve_day(day: str) -> tuple[str, Optional[date]]:
    day = day.strip()
    try:
        d = date.fromisoformat(day[:10])
        return DAYS[d.weekday()], d
    except ValueError:
        pass
    for name in DAYS:
        if name.lower().startswith(day.lower()[:3]) and len(day) >= 3:
            return name, None
    raise ValueError(f"unrecognised day: {day!r}")


@observe(name="get-opening-hours", as_type="tool", capture_input=False)
def get_opening_hours(day: Optional[str] = None) -> dict:
    langfuse.update_current_span(input={"day": day})
    now = current_time()
    today = DAYS[now.weekday()]
    opens, closes = rd.HOURS[today]
    hhmm = now.strftime("%H:%M")
    open_now = opens <= hhmm < closes

    if open_now:
        status = f"Open now, closes today at {_fmt_time(closes)}."
    elif hhmm < opens:
        status = f"Closed right now, opens today at {_fmt_time(opens)}."
    else:
        nxt = DAYS[(now.weekday() + 1) % 7]
        status = f"Closed right now, opens {nxt} at {_fmt_time(rd.HOURS[nxt][0])}."

    result = {
        "now_local": now.strftime(f"%A %Y-%m-%d %H:%M ({rd.TIMEZONE})"),
        "open_now": open_now,
        "status": status,
        "weekly_hours": {d: _fmt_range(d) for d in DAYS},
        "upcoming_special_days": [
            {"date": d.isoformat(), "weekday": DAYS[d.weekday()], "note": note}
            for d, note in sorted(rd.SPECIAL_DAYS.items())
            if timedelta(0) <= d - now.date() <= timedelta(days=14)
        ],
        "disclaimer": rd.HOURS_DISCLAIMER,
    }
    if day:
        weekday, d = _resolve_day(day)
        result["requested"] = {"weekday": weekday, "date": d.isoformat() if d else None, "hours": _fmt_range(weekday)}
        if d and d in rd.SPECIAL_DAYS:
            result["requested"]["special_day_note"] = rd.SPECIAL_DAYS[d]
    return result


def _fold(text: str) -> list[str]:
    """Lowercase tokens with common South-Indian spelling variants folded to the menu's spelling."""
    aliases = {
        "dosa": "dosai", "dosay": "dosai", "idli": "idly", "vada": "vadai", "vadas": "vadai", "vade": "vadai",
        "medu": "medhu", "parota": "parotta", "porotta": "parotta", "paratha": "parotta",
        "biriyani": "biryani", "briyani": "biryani", "uttapam": "uthappam", "uthapam": "uthappam",
        "uttappam": "uthappam", "lollipop": "lollypop", "jamun": "jamoon", "dessert": "desserts",
        "appetizer": "appetizers", "starter": "appetizers", "starters": "appetizers", "meal": "meals",
        "thali": "meals", "drink": "beverages", "drinks": "beverages", "curry": "curries", "soup": "soups",
        "egg": "eggs", "koththu": "kothu", "kotthu": "kothu",
    }
    stop = {"the", "a", "an", "do", "you", "have", "what", "is", "are", "how", "much", "for", "of", "and",
            "your", "any", "on", "in", "i", "me", "can", "get", "does", "it", "to", "with", "cost", "price"}
    tokens = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    return [aliases.get(t, t) for t in tokens if t not in stop]


def _token_score(q: str, targets: list[str]) -> float:
    return max((difflib.SequenceMatcher(None, q, t).ratio() for t in targets), default=0.0)


@observe(name="search-menu", as_type="tool", capture_input=False)
def search_menu(query: str) -> dict:
    langfuse.update_current_span(input={"query": query})
    q_tokens = _fold(query)
    scored = []
    for item in rd.MENU:
        name_t, cat_t = _fold(item["name"]), _fold(item["category"])
        hits = sum(1 for q in q_tokens if _token_score(q, name_t + cat_t) >= 0.84)
        name_hits = sum(1 for q in q_tokens if _token_score(q, name_t) >= 0.84)
        if q_tokens and hits / len(q_tokens) >= 0.5:
            scored.append((hits / len(q_tokens) + 0.01 * name_hits, item))
    scored.sort(key=lambda s: -s[0])

    if not scored:
        cats = list(dict.fromkeys(i["category"] for i in rd.MENU))
        return {"results": [], "note": "No matching menu item.", "categories": cats}
    return {
        "menu_snapshot_date": rd.MENU_SNAPSHOT_DATE,
        "results": [
            {
                "name": i["name"],
                "category": i["category"],
                "price": i["price_text"] or (f"${i['price']:.2f}" if i["price"] is not None else None),
                **({"note": i["note"]} if i["note"] else {}),
            }
            for _, i in scored[:15]
        ],
    }


@observe(name="get-restaurant-info", as_type="tool", capture_input=False)
def get_restaurant_info(topic: str) -> dict:
    langfuse.update_current_span(input={"topic": topic})
    if topic == "contact":
        return {"name": rd.NAME, "address": rd.ADDRESS, "phone": rd.PHONE, "website": rd.WEBSITE, "menu": rd.MENU_URL}
    if topic == "services":
        return {"services": rd.SERVICES, "description": rd.DESCRIPTION}
    if topic == "catering":
        return rd.CATERING
    if topic == "delivery":
        return rd.DELIVERY
    return {"error": f"unknown topic {topic!r}"}


TOOL_FUNCTIONS = {
    "get_opening_hours": get_opening_hours,
    "search_menu": search_menu,
    "get_restaurant_info": get_restaurant_info,
}


def run_tool(name: str, tool_input: dict) -> str:
    """Returns the tool result as text (errors are reported to the model as text)."""
    log = _TOOL_LOG.get()
    if log is not None:
        log.append(name)
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        return json.dumps(fn(**tool_input))
    except Exception as exc:  # the observation records the exception; tell the model too
        return f"Tool error: {exc}"


# --- Agent ------------------------------------------------------------------------------------


def generate_response(messages: list):
    """One model invocation, recorded as its own `generation` observation."""
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="generate-response",  # stable, verb-first (not the model name)
        model=MODEL,
        input=list(messages),  # snapshot: `messages` keeps growing after this call
        model_parameters={"reasoning_effort": REASONING_EFFORT},
        metadata={"tools": [t["function"]["name"] for t in TOOLS]},
    ) as generation:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS, reasoning_effort=REASONING_EFFORT
        )
        message = response.choices[0].message
        usage = response.usage

        output = {"role": "assistant", "content": message.content}
        # Groq returns the model's thinking in a non-standard `reasoning` field.
        reasoning = getattr(message, "reasoning", None)
        if reasoning:
            output["reasoning"] = reasoning
        if message.tool_calls:
            output["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ]

        usage_details = {"input": usage.prompt_tokens, "output": usage.completion_tokens}
        reasoning_tokens = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
        if reasoning_tokens:
            usage_details["output_reasoning_tokens"] = reasoning_tokens
        generation.update(output=output, usage_details=usage_details)
    return message


# Root observation of the trace. capture_input/output are off so the trace shows only the
# user message and the final reply, not the whole `messages` history (which also holds
# thinking blocks and tool payloads; those are on the generation observations).
@observe(name="handle-support-message", as_type="agent", capture_input=False, capture_output=False)
def handle_support_message(messages: list, user_message: str) -> str:
    langfuse.update_current_span(input=user_message)

    messages.append({"role": "user", "content": user_message})
    reply = ""

    for _ in range(MAX_AGENT_STEPS):
        message = generate_response(messages)
        reply = message.content or ""

        assistant_turn = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ]
        messages.append(assistant_turn)

        if not message.tool_calls:
            break

        for call in message.tool_calls:
            result = run_tool(call.function.name, json.loads(call.function.arguments or "{}"))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
    else:
        reply = reply or "Sorry, I could not finish that request."

    langfuse.update_current_span(output=reply)
    return reply


def answer(message: str, *, now: Optional[str] = None, prompt_version: Optional[str] = None) -> dict:
    """Single-turn entry point for evals.

    `now` (ISO local time, Toronto) pins the clock so "are you open right now?" has one right
    answer. Returns the reply plus the tools the agent called, so evaluators can check tool use.
    """
    version = prompt_version or PROMPT_VERSION
    tool_log: list = []
    now_dt = None
    if now:
        now_dt = datetime.fromisoformat(now)
        now_dt = now_dt if now_dt.tzinfo else now_dt.replace(tzinfo=TZ)
    tokens = (_TOOL_LOG.set(tool_log), _NOW_OVERRIDE.set(now_dt))
    try:
        with propagate_attributes(metadata={"prompt_version": version, "model": MODEL}):
            reply = handle_support_message([{"role": "system", "content": build_system_prompt(version)}], message)
    finally:
        _TOOL_LOG.reset(tokens[0])
        _NOW_OVERRIDE.reset(tokens[1])
    return {"reply": reply, "tools_called": tool_log}


def chat(user_id: str, session_id: str) -> None:
    messages: list = [{"role": "system", "content": build_system_prompt()}]
    print(f"{rd.NAME} support assistant (session {session_id}). Empty line to quit.")
    while True:
        try:
            user_message = input("you> ").strip()
        except EOFError:
            break
        if not user_message:
            break
        # One trace per turn; session_id ties the turns of a conversation together.
        with propagate_attributes(
            user_id=user_id,
            session_id=session_id,
            tags=["support-assistant"],
            metadata={"app_version": APP_VERSION, "model": MODEL, "prompt_version": PROMPT_VERSION},
            version=APP_VERSION,
        ):
            print("bot>", handle_support_message(messages, user_message))


if __name__ == "__main__":
    user = os.environ.get("DEMO_USER_ID", "demo-user")
    session = sys.argv[1] if len(sys.argv) > 1 else f"chat-{uuid.uuid4().hex[:8]}"
    try:
        chat(user, session)
    finally:
        # Short-lived script: make sure buffered spans are sent before exit.
        get_client().flush()
