# Auto-generated from chapters/12-api-surface.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import json
import re
from collections.abc import Callable
from functools import partial
from typing import Any

from jsonschema import Draft202012Validator

Message = dict[str, Any]
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


def _http_transport(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST one JSON body to a chat-completions URL without a provider SDK."""
    import httpx

    response = httpx.post(url, json=body, timeout=30.0)
    response.raise_for_status()
    return response.json()


def chat(
    base_url: str,
    model: str,
    messages: list[Message],
    *,
    transport: Transport | None = None,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 256,
    tool_choice: Any = None,
    parallel_tool_calls: bool = False,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat-completions endpoint over raw HTTP.

    The body is the wire contract; every keyword is a field a server versions
    and may interpret differently. We return a small dict rather than raw text
    because the caller must inspect four things a bare string hides: the typed
    assistant ``message`` (which may carry tool calls and null content), the
    ``finish_reason`` (why generation stopped), token ``usage``, and a
    ``request_id`` for tracing.

    Args:
        base_url: Server root; ``/v1/chat/completions`` is appended.
        model: Model identifier the server routes on.
        messages: Ordered role/content turns — the conversation so far.
        transport: A ``(url, body) -> wire_dict`` function; defaults to raw
            HTTP. Injecting it is what lets tests and this chapter run offline.
        tools: Optional tool declarations the model may propose against.
        response_format: Optional structured-output request (see
            :func:`schema_response_format`).
        temperature: Decoding sharpness; ``0.0`` is the reproducible default.
        top_p: Nucleus-sampling mass to keep.
        max_tokens: Hard generation budget; hitting it yields a ``length`` finish.
        tool_choice: Optional forcing control (``auto``/``required``/named/``none``).
        parallel_tool_calls: Whether the server may propose several calls at once.

    Returns:
        A dict with ``message``, ``finish_reason``, ``usage``, and ``request_id``.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if tools is not None:
        body["tools"] = tools
        body["parallel_tool_calls"] = parallel_tool_calls
    if response_format is not None:
        body["response_format"] = response_format
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    wire = (transport or _http_transport)(endpoint, body)
    choice = wire["choices"][0]
    return {
        "message": choice["message"],
        "finish_reason": choice.get("finish_reason"),
        "usage": wire.get("usage", {}),
        "request_id": wire.get("id"),
    }


class ScriptedModel:
    """A deterministic, offline stand-in for a chat-completions server.

    This is not a language model. It replays a fixed list of assistant turns,
    wrapping each in the OpenAI-compatible ``choices``/``finish_reason``/
    ``usage`` envelope so the code under test sees a real wire shape. It also
    remembers the last request body, which lets us print exactly what went out
    on the wire. Every number it reports is a stub value, honest about being
    one; it estimates no model's quality.

    Args:
        turns: Assistant turns to play in order, each a
            ``(message, finish_reason)`` pair. The final turn repeats if asked
            for more calls than were scripted.
        prompt_tokens: Fixed prompt-token count to report in ``usage``.
    """

    def __init__(
        self,
        turns: list[tuple[Message, str]],
        *,
        prompt_tokens: int = 40,
    ) -> None:
        self._turns = list(turns)
        self._prompt_tokens = prompt_tokens
        self.calls = 0
        self.last_body: dict[str, Any] | None = None

    def __call__(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        self.last_body = body
        message, finish_reason = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        completion = max(1, len(json.dumps(message)) // 4)
        return {
            "id": f"stub-{self.calls}",
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": completion,
                "total_tokens": self._prompt_tokens + completion,
            },
        }


def render_chat_template(messages: list[Message], template: dict[str, Any]) -> str:
    """Render messages into one prompt string using a template's role markers.

    A real chat template is tokenizer-owned and far richer, but the mechanism
    is this: each turn is wrapped in role-specific markers and the turns are
    joined, then a generation marker invites the assistant to continue. Two
    templates over identical messages produce different strings — and therefore
    different token counts — which is the whole point.

    Args:
        messages: The conversation turns; string content only, for the demo.
        template: Markers dict with per-role ``open``/``close``, a ``sep``
            between turns, and an ``assistant_open`` generation marker.

    Returns:
        The rendered prompt string a tokenizer would then encode.
    """
    rendered = []
    for turn in messages:
        marker = template["roles"][turn["role"]]
        rendered.append(f"{marker['open']}{turn['content']}{marker['close']}")
    return template["sep"].join(rendered) + template["assistant_open"]


def count_tokens(text: str) -> int:
    """Count tokens with a word-and-punctuation splitter (a tokenizer stand-in)."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def nucleus(probs: list[float], top_p: float) -> list[int]:
    """Return the token indices nucleus (top_p) sampling keeps, most likely first.

    We sort by probability, accumulate mass, and stop as soon as the running
    total reaches ``top_p``. The size of the returned set is a property of the
    distribution, not a fixed fraction of the vocabulary — which is the whole
    reason ``top_p=0.7`` keeps two tokens in one context and many in another.

    Args:
        probs: A probability vector over the candidate tokens; should sum to 1.
        top_p: Cumulative-mass threshold in ``(0, 1]``.

    Returns:
        The kept indices, ordered from most to least probable.
    """
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept, cumulative = [], 0.0
    for index in order:
        kept.append(index)
        cumulative += probs[index]
        if cumulative >= top_p:
            break
    return kept


def schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build the OpenAI-compatible strict JSON-Schema request block.

    Args:
        name: A stable name for the schema, recorded in traces.
        schema: The JSON Schema the decoder should conform output to.

    Returns:
        The ``response_format`` object to place in the request body.
    """
    return {"type": "json_schema",
            "json_schema": {"name": name, "schema": schema, "strict": True}}


REFUND_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "ticket_id": {"type": "string", "pattern": "^T-[0-9]{3}$"},
        "action": {"enum": ["refund", "deny", "escalate"]},
        "order_id": {"type": "string", "pattern": "^O-[0-9]{4}$"},
        "amount_cents": {"type": "integer", "minimum": 0},
        "currency": {"enum": ["USD", "EUR"]},
    },
    "required": ["ticket_id", "action", "order_id", "amount_cents", "currency"],
    "additionalProperties": False,
}


def constrained_renormalize(probs: list[float], legal: set[int]) -> list[float]:
    """Apply @eq-ch12-mask: zero illegal tokens and renormalize the survivors.

    Args:
        probs: The model's unconstrained probability vector.
        legal: Indices the grammar still permits after the current prefix.

    Returns:
        The renormalized distribution over legal tokens; illegal tokens get 0.

    Raises:
        ValueError: If the grammar leaves no legal continuation (empty support).
    """
    masked = [p if i in legal else 0.0 for i, p in enumerate(probs)]
    total = sum(masked)
    if total == 0.0:
        raise ValueError("the grammar left no legal continuation")
    return [p / total for p in masked]


def constraint_tax(probs: list[float], legal: set[int]) -> float:
    """Return the probability mass the grammar diverts from the model's top choice."""
    intended = max(range(len(probs)), key=lambda i: probs[i])
    return 0.0 if intended in legal else probs[intended]


class StructuredOutputFailed(RuntimeError):
    """No valid application object was obtained within the attempt budget."""


class Extraction:
    """A validated object plus the cost of obtaining it.

    Keeping ``attempts`` and ``total_tokens`` beside the value is what makes the
    repair loop measurable: every retry is a countable cost, not a hidden one.
    """

    def __init__(self, value: dict[str, Any], attempts: int, total_tokens: int) -> None:
        self.value = value
        self.attempts = attempts
        self.total_tokens = total_tokens


def parse_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in text, tolerating surrounding prose.

    Tolerance is not enforcement: this happily extracts an object a level-1
    model wrapped in prose, which is why it must be followed by real schema and
    business validation — parsing something is not the same as accepting it.

    Args:
        text: Raw assistant content that should contain one JSON object.

    Returns:
        The first decoded object.

    Raises:
        ValueError: If no JSON object is present or the first value is not one.
    """
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("the first JSON value is not an object")
    return value


def _complaint(
    value: dict[str, Any],
    validator: Draft202012Validator,
    semantic_check: Callable[[dict[str, Any]], str | None] | None,
) -> str | None:
    """Return one specific structural-or-semantic complaint, or None if valid."""
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
    if errors:
        return "; ".join(
            f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors
        )
    return semantic_check(value) if semantic_check else None


def extract(
    invoke: Callable[..., dict[str, Any]],
    messages: list[Message],
    schema: dict[str, Any],
    *,
    max_attempts: int = 3,
    semantic_check: Callable[[dict[str, Any]], str | None] | None = None,
) -> Extraction:
    """Parse, validate, and repair a structured extraction within a token budget.

    Each attempt sends the transcript with a strict-schema request, parses the
    first object, and asks for one specific complaint. A complaint becomes the
    next user turn so the model repairs exactly what failed; success returns a
    typed ``Extraction``; exhaustion raises rather than leaking a half-valid dict.

    Args:
        invoke: A ``chat``-like callable bound to a transport and model.
        messages: The seed conversation (typically one user turn).
        schema: The canonical schema the value must satisfy.
        max_attempts: Maximum model calls before giving up.
        semantic_check: Optional business-state check returning a complaint string.

    Returns:
        The validated value with its attempt count and cumulative token cost.

    Raises:
        StructuredOutputFailed: If no valid object appears within the budget.
    """
    validator = Draft202012Validator(schema)
    transcript = list(messages)
    tokens = 0
    for attempt in range(1, max_attempts + 1):
        result = invoke(
            transcript,
            response_format=schema_response_format("refund_decision", schema),
        )
        tokens += int(result.get("usage", {}).get("total_tokens", 0))
        text = result["message"].get("content") or ""
        try:
            value = parse_object(text)
            complaint = _complaint(value, validator, semantic_check)
        except ValueError as exc:
            complaint = str(exc)
        if complaint is None:
            return Extraction(value=value, attempts=attempt, total_tokens=tokens)
        transcript += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": (
                f"The candidate failed validation: {complaint}. "
                "Return only one object matching the supplied schema.")},
        ]
    raise StructuredOutputFailed(f"no valid object after {max_attempts} attempts")


KNOWN_ORDERS = {f"O-{i:04d}" for i in range(1, 51)}


def semantic_check(value: dict[str, Any]) -> str | None:
    """Reject a schema-legal order id that does not exist in application state.

    Args:
        value: A structurally valid extraction with an ``order_id``.

    Returns:
        ``None`` if the order exists, else a complaint string for the loop.
    """
    return None if value["order_id"] in KNOWN_ORDERS else "order_id does not exist"


def make_tickets(count: int = 50) -> list[dict[str, Any]]:
    """Build source-traceable but deliberately messy refund tickets.

    Args:
        count: How many tickets to generate (ids ``T-001`` upward).

    Returns:
        Ticket dicts carrying a ticket id, order id, amount, currency, and prose.
    """
    tickets = []
    for i in range(1, count + 1):
        currency = "USD" if i % 5 else "EUR"
        tickets.append({
            "ticket_id": f"T-{i:03d}",
            "order_id": f"O-{i:04d}",
            "amount_cents": 900 + 37 * i,
            "currency": currency,
        })
    return tickets


def scripted_candidate(ticket: dict[str, Any], level: int) -> str:
    """Return a guarantee-level's characteristic output for one ticket.

    The injected flaws are the point: level 1 lower-cases some enums and wraps
    some objects in prose, level 2 adds a stray property, level 3 emits a
    legal-shaped but nonexistent order id. They stand in for real model failure
    modes so the boundary has honest, known-bad inputs to reject.

    Args:
        ticket: A ticket from :func:`make_tickets`.
        level: The guarantee level (1, 2, or 3) whose flaw to inject.

    Returns:
        The candidate assistant content as a string.
    """
    index = int(ticket["ticket_id"].split("-")[1])
    value = {
        "ticket_id": ticket["ticket_id"], "action": "refund",
        "order_id": ticket["order_id"], "amount_cents": ticket["amount_cents"],
        "currency": ticket["currency"],
    }
    if level == 1 and index % 4 == 0:
        value["currency"] = value["currency"].lower()   # bad enum
    if level == 2 and index % 11 == 0:
        value["note"] = "not in schema"                 # extra property
    if level == 3 and index % 10 == 0:
        value["order_id"] = f"O-9{index:03d}"           # legal shape, nonexistent
    body = json.dumps(value)
    if level == 1 and index % 7 == 0:
        return f"Here is the extraction: {body}"         # prose wrapper
    return body


def run_validity_experiment(count: int = 50) -> dict[str, Any]:
    """Measure schema validity per level and the level-4 repair cost.

    Args:
        count: Number of tickets to run.

    Returns:
        A report with per-level validity, the count of schema-valid but
        business-invalid records, the repair-invocation rate, and mean tokens
        per successful extraction.
    """
    tickets = make_tickets(count)
    validator = Draft202012Validator(REFUND_SCHEMA)

    def valid(text: str) -> bool:
        try:
            return not list(validator.iter_errors(parse_object(text)))
        except ValueError:
            return False

    validity = {
        level: sum(valid(scripted_candidate(t, level)) for t in tickets) / count
        for level in (1, 2, 3)
    }
    business_invalid = sum(
        valid(scripted_candidate(t, 3))
        and semantic_check(parse_object(scripted_candidate(t, 3))) is not None
        for t in tickets
    )
    extractions = []
    for ticket in tickets:
        stub = ScriptedModel([
            ({"role": "assistant", "content": scripted_candidate(ticket, 2)}, "stop"),
            ({"role": "assistant", "content": json.dumps({
                "ticket_id": ticket["ticket_id"], "action": "refund",
                "order_id": ticket["order_id"], "amount_cents": ticket["amount_cents"],
                "currency": ticket["currency"]})}, "stop"),
        ])
        extractions.append(extract(
            partial(chat, "http://stub", "scripted", transport=stub),
            [{"role": "user", "content": "extract"}],
            REFUND_SCHEMA, semantic_check=semantic_check))
    repaired = sum(e.attempts > 1 for e in extractions)
    return {
        "validity": validity,
        "business_invalid": business_invalid,
        "repair_rate": repaired / count,
        "tokens_per_success": sum(e.total_tokens for e in extractions) / count,
    }


def assert_all_calls_answered(calls: list[dict], results: list[Message]) -> None:
    """Require every proposal id to be answered by exactly one result id.

    Args:
        calls: The assistant's proposed tool calls, each carrying an ``id``.
        results: The tool result messages, each carrying a ``tool_call_id``.

    Raises:
        ValueError: If ids are duplicated or the two id sets do not match.
    """
    proposed = [c["id"] for c in calls]
    answered = [r["tool_call_id"] for r in results]
    if len(proposed) != len(set(proposed)) or sorted(proposed) != sorted(answered):
        raise ValueError("tool results must answer every unique call id exactly once")


def run_tool_turn(
    invoke: Callable[..., dict[str, Any]],
    messages: list[Message],
    tool: dict[str, Any],
    handler: Callable[..., Any],
) -> dict[str, Any]:
    """Run one proposed tool call and return the model's final response.

    The proposal is validated against the tool's own parameter schema *before*
    the handler runs, so malformed arguments raise at the boundary rather than
    inside application code. The complete assistant turn is preserved on the
    second request — reconstructing only name and arguments can drop provider
    metadata the model needs.

    Args:
        invoke: A ``chat``-like callable bound to a transport and model.
        messages: The seed conversation.
        tool: One tool declaration (name, description, parameter schema).
        handler: The Python function that actually performs the action.

    Returns:
        The second-request response, from which the model answers.

    Raises:
        ValueError: On an unexpected finish, wrong call count, disallowed tool,
            or arguments that fail the tool's schema.
    """
    first = invoke(messages, tools=[tool], parallel_tool_calls=False)
    if first["finish_reason"] != "tool_calls":
        raise ValueError("expected a tool_calls finish reason")
    calls = first["message"].get("tool_calls", [])
    if len(calls) != 1:
        raise ValueError("this boundary permits exactly one tool proposal")
    call = calls[0]
    if call["function"]["name"] != tool["function"]["name"]:
        raise ValueError("the proposed tool is not allowed")
    arguments = json.loads(call["function"]["arguments"])
    errors = list(Draft202012Validator(tool["function"]["parameters"]).iter_errors(arguments))
    if errors:
        raise ValueError(f"tool arguments failed validation: {errors[0].message}")
    output = handler(**arguments)
    result = {"role": "tool", "tool_call_id": call["id"],
              "content": json.dumps(output, sort_keys=True)}
    assert_all_calls_answered(calls, [result])
    return invoke([*messages, first["message"], result], tools=[tool],
                  parallel_tool_calls=False)


class IncompleteToolStream(RuntimeError):
    """A streamed tool proposal never reached both close and terminal events."""


def assemble_tool_stream(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assemble streamed tool-call deltas, exposing proposals only when actionable.

    Arguments accumulate per call index; a proposal becomes eligible only after
    its call-close event *and* a response-terminal event. Anything less fails
    closed, because a parseable prefix is not a committed turn.

    Args:
        chunks: Streamed events; a terminal event carries ``terminal: True``,
            others carry a ``tool_call`` delta with ``index`` and partial
            ``arguments``.

    Returns:
        One assembled proposal per call index, each with id, name, and parsed
        arguments.

    Raises:
        IncompleteToolStream: If any call is unclosed or the response never
            reached a terminal event.
    """
    buffers: dict[int, dict[str, Any]] = {}
    terminal = False
    for chunk in chunks:
        if chunk.get("terminal"):
            terminal = True
            continue
        delta = chunk["tool_call"]
        index = int(delta["index"])
        state = buffers.setdefault(index, {"id": None, "name": None, "text": "", "closed": False})
        state["id"] = delta.get("id", state["id"])
        state["name"] = delta.get("name", state["name"])
        state["text"] += delta.get("arguments", "")
        state["closed"] = state["closed"] or bool(delta.get("closed"))
    if not terminal or not buffers or any(not s["closed"] for s in buffers.values()):
        raise IncompleteToolStream("stream ended before close and terminal events")
    return [{"id": s["id"], "name": s["name"], "arguments": json.loads(s["text"])}
            for _, s in sorted(buffers.items())]


def classify_failure(
    *, status: int | None = None, finish_reason: str | None = None,
    error: Exception | None = None,
) -> str:
    """Map a boundary symptom to the layer that owns its recovery.

    Args:
        status: An HTTP status, when the failure was transport-level.
        finish_reason: A completion's finish reason, when one was returned.
        error: A raised exception, when the call did not complete.

    Returns:
        One of ``code_retry``, ``typed_failure``, ``model_retry_without_effect``,
        ``typed_refusal``, or ``accept``.
    """
    if error is not None:
        return "code_retry" if type(error).__name__ in {"ConnectError", "TimeoutException"} else "typed_failure"
    if status is not None and status >= 400:
        return "code_retry" if status in {408, 429, 500, 502, 503, 504} else "typed_failure"
    if finish_reason == "length":
        return "model_retry_without_effect"
    if finish_reason == "content_filter":
        return "typed_refusal"
    return "accept"
