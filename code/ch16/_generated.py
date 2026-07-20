# Auto-generated from chapters/16-agent-anatomy.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    """One tool the harness is willing to execute on the model's behalf.

    The schema maps argument names to Python types; it is the tool's public
    contract, advertised to the model as data and enforced by the harness
    before the handler runs. The handler is ordinary code with real authority,
    which is why the effectful flag marks tools that change the world instead
    of reading it.
    """

    description: str
    schema: dict[str, type]
    handler: Callable[..., Any]
    effectful: bool = False


def make_environment() -> tuple[dict[str, ToolSpec], list[dict[str, Any]]]:
    """Create the support-desk world: three tools and an append-only effect log.

    The effect log is the ground truth of this small world. Reads return data;
    only ``refund_order`` appends a receipt. Every safety claim we make later
    is an assertion on this list, never on anything the model says.

    Returns:
        The tool registry and the (initially empty) effect log.
    """
    orders = {"A-17": {"order_id": "A-17", "status": "damaged", "paid_cents": 4999}}
    effects: list[dict[str, Any]] = []

    def lookup_order(order_id: str) -> dict[str, Any]:
        if order_id not in orders:
            raise KeyError(f"no such order: {order_id}")
        return orders[order_id]

    def read_policy() -> str:
        return "Damaged orders: refund the amount paid, up to 5000 cents automatically."

    def refund_order(order_id: str, amount_cents: int) -> dict[str, Any]:
        receipt = {"order_id": order_id, "refunded_cents": amount_cents}
        effects.append(receipt)
        return receipt

    tools = {
        "lookup_order": ToolSpec(
            "Fetch an order's status and the amount paid.", {"order_id": str}, lookup_order
        ),
        "read_policy": ToolSpec("Read the refund policy.", {}, read_policy),
        "refund_order": ToolSpec(
            "Issue a refund. Moves real money.",
            {"order_id": str, "amount_cents": int},
            refund_order,
            effectful=True,
        ),
    }
    return tools, effects


@dataclass(frozen=True)
class ToolCall:
    """A parsed proposal to run one tool. It carries no authority by itself.

    ``arguments`` is ``None`` when the model's argument string failed to parse
    as JSON — a proposal can be broken before it can be judged.
    """

    call_id: str
    name: str
    arguments: dict[str, Any] | None


def validate_call(call: ToolCall, tools: dict[str, ToolSpec]) -> str | None:
    """Check one proposal against the registry's declared contracts.

    Validation answers "is this call well formed?" — the tool exists, the
    argument names match exactly, and every value has the declared type. It
    deliberately says nothing about whether the call is *permitted*; that is
    the gate's question, asked later.

    Args:
        call: The parsed proposal.
        tools: The registry of declared contracts.

    Returns:
        None when the proposal is well formed, otherwise a reason string the
        model can read and act on.
    """
    if call.arguments is None:
        return "arguments were not valid JSON"
    tool = tools.get(call.name)
    if tool is None:
        return f"unknown tool: {call.name}"
    if set(call.arguments) != set(tool.schema):
        return f"{call.name} expects exactly the fields {sorted(tool.schema)}"
    for arg_name, expected in tool.schema.items():
        if not isinstance(call.arguments[arg_name], expected):
            return f"{arg_name} must be {expected.__name__}"
    return None


def tool_schema_json(tools: dict[str, ToolSpec]) -> list[dict[str, Any]]:
    """Render the registry in the JSON wire shape a chat API expects.

    This is how tools reach the model: as data in the request, one JSON
    schema per tool. The model never receives the handlers — only these
    descriptions — so everything it "knows" about a tool is what we choose
    to advertise here.

    Args:
        tools: The registry to advertise.

    Returns:
        A list of function-tool declarations in OpenAI-compatible form.
    """
    type_names = {str: "string", int: "integer", float: "number", bool: "boolean"}
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        arg: {"type": type_names[kind]} for arg, kind in spec.schema.items()
                    },
                    "required": sorted(spec.schema),
                },
            },
        }
        for name, spec in tools.items()
    ]


def tool_call_message(
    call_id: str, name: str, arguments: dict[str, Any], thought: str | None = None
) -> dict[str, Any]:
    """Build an assistant message that proposes one tool call, wire-shaped.

    The arguments travel as a JSON *string*, exactly as chat APIs deliver
    them: the model emitted text that happens to look like JSON, and the
    harness must parse it — and must survive the parse failing.

    Args:
        call_id: Correlation id the tool result must echo back.
        name: The tool being proposed.
        arguments: The arguments, serialized to the wire string.
        thought: Optional visible reasoning text carried in ``content``.

    Returns:
        An assistant message dict in OpenAI-compatible wire shape.
    """
    return {
        "role": "assistant",
        "content": thought,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def answer_message(text: str) -> dict[str, Any]:
    """Build a plain assistant message that answers and proposes nothing."""
    return {"role": "assistant", "content": text}


@dataclass
class ScriptedModel:
    """A stand-in for a chat model whose entire behavior is a visible script.

    ``chat`` plays the scripted turns in order. Before consuming the next
    turn it checks the most recent tool result against ``reactions``: if a
    key occurs in that result, the paired message is returned instead. That
    one rule lets a script react to what the loop feeds back — retry after
    a denial, for example — while staying deterministic and inspectable.
    Nothing here is a language model; the point is that the loop cannot tell
    the difference, because both speak the same wire format.
    """

    turns: list[dict[str, Any]]
    reactions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Return the next assistant message, given the conversation so far.

        Args:
            messages: The transcript; only the latest tool message is
                consulted, and only to match ``reactions`` keys.

        Returns:
            The next scripted assistant message, a reaction, or a default
            answer once the script is exhausted.
        """
        last = messages[-1]
        if last["role"] == "tool":
            for needle, reply in self.reactions.items():
                if needle in last["content"]:
                    return reply
        if not self.turns:
            return answer_message("I have nothing further to add.")
        return self.turns.pop(0)


def parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Parse the tool calls, if any, out of an assistant message.

    The arguments field is model-generated text, so it can be broken JSON;
    a call whose arguments fail to parse still comes back as a ``ToolCall``,
    with ``arguments=None``, so the loop can reject it through the normal
    validation path instead of crashing.

    Args:
        message: An assistant message in wire shape.

    Returns:
        One ``ToolCall`` per entry in ``tool_calls``; empty for a plain answer.
    """
    calls: list[ToolCall] = []
    for item in message.get("tool_calls") or []:
        function = item["function"]
        try:
            arguments = json.loads(function["arguments"])
        except json.JSONDecodeError:
            arguments = None
        calls.append(ToolCall(item["id"], function["name"], arguments))
    return calls


from enum import Enum


class Stop(str, Enum):
    """The typed ways a run can end. Callers branch on these, not on prose."""

    ANSWERED = "answered"
    STEP_LIMIT = "step_limit"
    TOKEN_LIMIT = "token_limit"
    NO_PROGRESS = "no_progress"


@dataclass(frozen=True)
class Limits:
    """Resource ceilings the loop enforces regardless of what the model wants.

    ``max_turns`` bounds model calls; ``max_prompt_tokens`` bounds the
    estimated context we are willing to send on any single call; and
    ``repeated_denials`` bounds how often the identical proposal may be
    denied before the loop declares no progress.
    """

    max_turns: int = 8
    max_prompt_tokens: int = 4000
    repeated_denials: int = 2


@dataclass(frozen=True)
class Observation:
    """The typed result of attempting one tool call.

    ``kind`` is one of ``result`` (the handler ran), ``invalid`` (validation
    failed), ``denied`` (the gate refused), or ``error`` (the handler raised
    an expected exception). A strategy can branch on the kind; "failed" alone
    invites blind retries.
    """

    call_id: str
    ok: bool
    kind: str
    content: Any


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate the token cost of a message list at four characters per token.

    Crude, but monotone in the thing that matters: the transcript — and with
    it the price of every further turn — only ever grows.

    Args:
        messages: Wire-format messages about to be sent.

    Returns:
        The estimated prompt token count.
    """
    return sum(len(json.dumps(message)) for message in messages) // 4


Gate = Callable[[ToolCall], tuple[bool, str]]


def allow_all(call: ToolCall) -> tuple[bool, str]:
    """A gate that approves everything — the absence of policy, made explicit."""
    return True, "no policy configured"


def execute_call(call: ToolCall, tools: dict[str, ToolSpec], gate: Gate) -> Observation:
    """Carry one proposal across the effect boundary — or refuse to.

    Validation asks whether the call is well formed; the gate asks whether
    this exact call may run now. Only when both pass does the handler
    execute. Expected handler failures come back as typed observations
    instead of ending the run, and a denial is information, not an exception.

    Args:
        call: The parsed proposal.
        tools: The registry of executable contracts.
        gate: The policy consulted immediately before execution.

    Returns:
        A typed, correlated ``Observation``.
    """
    error = validate_call(call, tools)
    if error is not None:
        return Observation(call.call_id, False, "invalid", error)
    allowed, reason = gate(call)
    if not allowed:
        return Observation(call.call_id, False, "denied", reason)
    try:
        result = tools[call.name].handler(**call.arguments)
    except (KeyError, ValueError, TimeoutError) as exc:
        return Observation(call.call_id, False, "error", str(exc))
    return Observation(call.call_id, True, "result", result)


@dataclass
class RunState:
    """Everything the loop knows, as plain serializable data.

    ``messages`` is the transcript in wire format: persist it and the run can
    resume anywhere, because there is no hidden state elsewhere. The typed
    observations and per-turn prompt sizes exist so a verifier can replay
    what happened without re-running anything.
    """

    messages: list[dict[str, Any]]
    observations: list[Observation] = field(default_factory=list)
    prompt_token_log: list[int] = field(default_factory=list)
    turns: int = 0


@dataclass(frozen=True)
class RunResult:
    """A typed ending, the final answer if any, and the full state behind it."""

    stop: Stop
    answer: str | None
    state: RunState


SYSTEM_PROMPT = (
    "You are a support agent. Use the tools to resolve the customer's problem. "
    "A denied call is information: read the reason and adjust."
)


def run_agent(
    task: str,
    model: Any,
    tools: dict[str, ToolSpec],
    gate: Gate,
    limits: Limits = Limits(),
) -> RunResult:
    """Run one task through the propose-gate-execute-observe loop to a typed end.

    Each turn sends the whole transcript to the model, appends whatever it
    returns, executes any proposed calls through ``execute_call``, and appends
    the typed results as tool messages. The model chooses what to propose;
    this function owns everything else — the transcript, the budgets, the
    effect boundary, and the decision to stop.

    Args:
        task: The user's goal. Text only; it grants no authority.
        model: Anything with ``chat(messages) -> assistant message``.
        tools: The registry of executable contracts.
        gate: Policy consulted immediately before every handler call.
        limits: Step, token, and no-progress ceilings.

    Returns:
        A ``RunResult`` with the stop reason, the answer text when the model
        answered, and the complete state for verification.
    """
    state = RunState(messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ])
    denials: dict[str, int] = {}
    while state.turns < limits.max_turns:
        prompt_tokens = estimate_tokens(state.messages)
        if prompt_tokens > limits.max_prompt_tokens:
            return RunResult(Stop.TOKEN_LIMIT, None, state)
        state.prompt_token_log.append(prompt_tokens)
        reply = model.chat(state.messages)
        state.messages.append(reply)
        state.turns += 1
        calls = parse_tool_calls(reply)
        if not calls:
            return RunResult(Stop.ANSWERED, reply.get("content"), state)
        for call in calls:
            observation = execute_call(call, tools, gate)
            state.observations.append(observation)
            state.messages.append({
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": json.dumps({
                    "ok": observation.ok,
                    "kind": observation.kind,
                    "content": observation.content,
                }),
            })
            if observation.kind == "denied":
                key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True)}"
                denials[key] = denials.get(key, 0) + 1
                if denials[key] >= limits.repeated_denials:
                    return RunResult(Stop.NO_PROGRESS, None, state)
    return RunResult(Stop.STEP_LIMIT, None, state)


def print_transcript(state: RunState) -> None:
    """Print a run's transcript, one aligned line per message or tool call.

    Args:
        state: The finished (or in-flight) run state to display.
    """
    labels = {"system": "system |", "user": "user   |", "tool": "tool   |"}
    for message in state.messages:
        if message["role"] == "assistant":
            if message.get("content"):
                print(f"model  | {message['content']}")
            for item in message.get("tool_calls") or []:
                function = item["function"]
                print(f"model  | -> {function['name']} {function['arguments']}")
        else:
            print(f"{labels[message['role']]} {message['content']}")


def overreaching_model() -> ScriptedModel:
    """Build the chapter's over-reaching support model, fresh for each run.

    Its script looks up the order, proposes a 9999-cent refund — twice the
    amount paid — and, if it ever sees a denial, falls back to refunding
    exactly what the order cost. A capable model behaves this way when its
    instructions say to read denials; here the reaction is scripted so every
    run of the book reproduces it.

    Returns:
        A fresh ``ScriptedModel`` with the same three turns and one reaction.
    """
    return ScriptedModel(
        turns=[
            tool_call_message("call_1", "lookup_order", {"order_id": "A-17"}),
            tool_call_message("call_2", "refund_order",
                              {"order_id": "A-17", "amount_cents": 9999}),
            answer_message("I have issued the refund for order A-17."),
        ],
        reactions={
            "denied": tool_call_message("call_3", "refund_order",
                                        {"order_id": "A-17", "amount_cents": 4999}),
        },
    )


AUTO_REFUND_LIMIT_CENTS = 5000


def refund_gate(call: ToolCall) -> tuple[bool, str]:
    """The support desk's policy, applied at the effect boundary.

    It examines the exact arguments about to run — not what the prompt said,
    not what the model intended — and refuses any refund above the automatic
    authority limit. Everything else passes.

    Args:
        call: The validated proposal about to be executed.

    Returns:
        ``(allowed, reason)``; the reason travels back to the model either way.
    """
    if call.name == "refund_order" and call.arguments["amount_cents"] > AUTO_REFUND_LIMIT_CENTS:
        return False, f"refund exceeds automatic authority of {AUTO_REFUND_LIMIT_CENTS} cents"
    return True, "within policy"


@dataclass
class Meter:
    """Measure what a pattern actually spends.

    Wraps model calls and tool executions so every pattern is charged in the
    same three currencies: model calls, tool calls, and estimated prompt
    tokens. The differences between patterns then come from control policy,
    not from accounting.
    """

    model_calls: int = 0
    tool_calls: int = 0
    prompt_tokens: int = 0

    def chat(self, model: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Charge one model call and forward it to the model.

        Args:
            model: The model being consulted.
            messages: The prompt about to be sent; its size is charged.

        Returns:
            The assistant message the model produced.
        """
        self.model_calls += 1
        self.prompt_tokens += estimate_tokens(messages)
        return model.chat(messages)

    def execute(self, call: ToolCall, tools: dict[str, ToolSpec], gate: Gate) -> Observation:
        """Charge one tool call and route it through the shared effect boundary.

        Args:
            call: The proposal to execute.
            tools: The registry.
            gate: The policy gate — the same one every pattern must pass.

        Returns:
            The typed observation from ``execute_call``.
        """
        self.tool_calls += 1
        return execute_call(call, tools, gate)


def run_chain(model: Any, tools: dict[str, ToolSpec], gate: Gate, meter: Meter) -> str:
    """The chain pattern: code owns the whole sequence.

    Look up the order, refund exactly what was paid, then consult the model
    once — for the only genuinely semantic step, drafting the reply. When the
    path is known in advance, this is the cheapest and most testable shape.

    Args:
        model: Drafting model (one call).
        tools: The registry.
        gate: The shared policy gate.
        meter: The accounting for this run.

    Returns:
        The drafted customer reply.
    """
    order = meter.execute(ToolCall("c1", "lookup_order", {"order_id": "A-17"}),
                          tools, gate).content
    receipt = meter.execute(
        ToolCall("c2", "refund_order",
                 {"order_id": "A-17", "amount_cents": order["paid_cents"]}),
        tools, gate).content
    reply = meter.chat(model, [{"role": "user", "content":
                                f"Draft one sentence for the customer citing this receipt: "
                                f"{json.dumps(receipt)}"}])
    return reply["content"]


def run_parallel(model: Any, tools: dict[str, ToolSpec], gate: Gate, meter: Meter) -> str:
    """The parallel fan-out pattern: independent reads dispatched together.

    Code fires both reads, joins the typed results, refunds the amount the
    order reports, and consults the model once with everything in view. The
    fan-out is safe because the reads are independent and effect-free.

    Args:
        model: Drafting model (one call).
        tools: The registry.
        gate: The shared policy gate.
        meter: The accounting for this run.

    Returns:
        The drafted customer reply.
    """
    reads = [ToolCall("p1", "lookup_order", {"order_id": "A-17"}),
             ToolCall("p2", "read_policy", {})]
    joined = {c.name: meter.execute(c, tools, gate).content for c in reads}
    receipt = meter.execute(
        ToolCall("p3", "refund_order",
                 {"order_id": "A-17",
                  "amount_cents": joined["lookup_order"]["paid_cents"]}),
        tools, gate).content
    reply = meter.chat(model, [{"role": "user", "content":
                                f"Order and policy: {json.dumps(joined)}. "
                                f"Draft one sentence citing {json.dumps(receipt)}."}])
    return reply["content"]


def run_router(task: str, model: Any, tools: dict[str, ToolSpec], gate: Gate,
               meter: Meter) -> str:
    """The router pattern: one bounded model decision picks a code-owned branch.

    The model answers a single classification question; code owns everything
    after it. A misroute contaminates the whole branch, which is why the
    branch set stays small and the routing decision is worth logging.

    Args:
        task: The customer request being routed.
        model: The routing (and drafting) model.
        tools: The registry.
        gate: The shared policy gate.
        meter: The accounting for this run.

    Returns:
        The branch's final reply.
    """
    route = meter.chat(model, [{"role": "user", "content":
                                f"Answer 'billing' or 'policy' only. Request: {task}"}])
    branch = route["content"].strip()
    if branch == "policy":
        policy = meter.execute(ToolCall("r1", "read_policy", {}), tools, gate).content
        return f"Our policy: {policy}"
    return run_chain(model, tools, gate, meter)


def run_evaluator(model: Any, tools: dict[str, ToolSpec], gate: Gate, meter: Meter,
                  max_rounds: int = 3) -> str:
    """The evaluator-optimizer pattern: generate, check, revise until it passes.

    Code performs the transaction, then loops the model against an acceptance
    criterion owned by code — here, that the draft states the exact refunded
    amount. The criterion is what makes revision meaningful; a weak one
    rewards cosmetic edits.

    Args:
        model: The drafting model, consulted once per round.
        tools: The registry.
        gate: The shared policy gate.
        meter: The accounting for this run.
        max_rounds: The revision budget.

    Returns:
        The first draft that passes the criterion, or the last draft.
    """
    order = meter.execute(ToolCall("e1", "lookup_order", {"order_id": "A-17"}),
                          tools, gate).content
    receipt = meter.execute(
        ToolCall("e2", "refund_order",
                 {"order_id": "A-17", "amount_cents": order["paid_cents"]}),
        tools, gate).content
    critique = ""
    draft = ""
    for _ in range(max_rounds):
        prompt = f"Draft a reply for this receipt: {json.dumps(receipt)}.{critique}"
        draft = meter.chat(model, [{"role": "user", "content": prompt}])["content"]
        if str(receipt["refunded_cents"]) in draft:
            return draft
        critique = " Critique: the draft must state the exact refunded amount."
    return draft


def execute_plan(plan: list[dict[str, Any]], tools: dict[str, ToolSpec],
                 gate: Gate) -> tuple[list[Observation], int]:
    """Execute a model-proposed plan step by step through the normal gate.

    A plan is a hypothesis about the future, not an execution grant: each
    step is gated at execution time, and the executor stops at the first
    step that fails so the planner can revise with fresh information.

    Args:
        plan: Steps of the form ``{"tool": name, "arguments": {...}}``.
        tools: The registry.
        gate: The same policy gate every other pattern uses.

    Returns:
        The observations for the executed prefix, and the index of the first
        failed step (``len(plan)`` when every step succeeded).
    """
    observations: list[Observation] = []
    for index, step in enumerate(plan):
        observation = execute_call(
            ToolCall(f"s{index}", step["tool"], step["arguments"]), tools, gate)
        observations.append(observation)
        if not observation.ok:
            return observations, index
    return observations, len(plan)


def check_run(result: RunResult, effects: list[dict[str, Any]],
              limits: Limits) -> dict[str, bool]:
    """Check a finished run's mechanical invariants from its own record.

    These are the properties the loop must guarantee no matter how the model
    behaved: every effect was authorized through an allowed observation,
    every tool message correlates to a proposed call id, the turn budget was
    respected, and the ending is one of the typed stops.

    Args:
        result: The finished run.
        effects: The environment's effect log for the same run.
        limits: The limits the run was given.

    Returns:
        A mapping from invariant name to whether it held.
    """
    state = result.state
    proposed_ids = {item["id"]
                    for message in state.messages if message.get("tool_calls")
                    for item in message["tool_calls"]}
    return {
        "effects_authorized": all(
            any(o.ok and o.content == receipt for o in state.observations)
            for receipt in effects),
        "observations_correlated": all(
            message["tool_call_id"] in proposed_ids
            for message in state.messages if message["role"] == "tool"),
        "bounded": state.turns <= limits.max_turns,
        "typed_ending": isinstance(result.stop, Stop),
    }


def verify_refund(effects: list[dict[str, Any]], order_id: str, amount_cents: int) -> bool:
    """Decide task success from the environment, not from the answer text.

    Success is exactly one receipt for the right order and the right amount.
    A polite "your refund is complete" beside an empty effect log fails this
    check, which is the entire point of grounding verification in state the
    model cannot narrate its way around.

    Args:
        effects: The environment's append-only effect log.
        order_id: The order that should have been refunded.
        amount_cents: The exact amount that should have moved.

    Returns:
        True when the log contains exactly the one intended receipt.
    """
    return effects == [{"order_id": order_id, "refunded_cents": amount_cents}]
