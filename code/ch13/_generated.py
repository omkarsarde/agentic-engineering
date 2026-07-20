# Auto-generated from chapters/13-prompting-context-engineering.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from collections import Counter
from dataclasses import dataclass, replace
from typing import Sequence

REFUND_CUES = ("refund", "money back", "reverse the payment", "reverse payment",
               "charge back", "chargeback", "reimburse")
TECH_CUES = ("crash", "error", "freez", "logged me out", "won't load",
             "does not load", "broken", "bug", "500")
STATUS_CUES = ("where", "tracking", "package", "parcel", "shipped", "arrive",
               "delivery", "delivered", "eta")
DEFAULT_PRECEDENCE = ("refund", "status", "technical")


@dataclass(frozen=True)
class Ticket:
    text: str
    expected: str


@dataclass(frozen=True)
class Demo:
    text: str
    label: str


@dataclass(frozen=True)
class Prompt:
    instruction: str
    precedence: tuple[str, ...] = DEFAULT_PRECEDENCE
    demos: tuple[Demo, ...] = ()


def _content_tokens(text: str) -> set[str]:
    return {w.strip(".,?!;:'\"-").lower() for w in text.split() if len(w) > 2}


def _similarity(a: str, b: str) -> float:
    ta, tb = _content_tokens(a), _content_tokens(b)
    return len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0


def _prior_label(text: str, precedence: tuple[str, ...]) -> str:
    lower = text.lower()
    hits = {
        "refund": any(cue in lower for cue in REFUND_CUES),
        "technical": any(cue in lower for cue in TECH_CUES),
        "status": any(cue in lower for cue in STATUS_CUES),
    }
    for label in precedence:
        if hits[label]:
            return label
    return "technical"


def _nearest_demo(text: str, demos: Sequence[Demo]) -> tuple[Demo | None, float]:
    best, best_sim = None, 0.0
    for demo in demos:
        sim = _similarity(text, demo.text)
        if sim > best_sim:
            best, best_sim = demo, sim
    return best, best_sim


def classify(prompt: Prompt, text: str, demo_threshold: float = 0.5) -> str:
    """Route one ticket with the stub proxy: nearest demonstration, else keyword prior.

    The stub imitates two real phenomena cheaply. When a demonstration is close
    enough it completes that demonstration's label (the in-context-learning
    path); otherwise it falls back to keyword cues resolved by a precedence
    order (its "prior" behavior). It is a keyword matcher, not a model: its job
    is to make every downstream score explainable, not to be good at English.

    Args:
        prompt: The prompt configuration — its precedence order and any
            demonstrations both steer the routing.
        text: The customer ticket to classify.
        demo_threshold: Minimum token-overlap similarity at which the nearest
            demonstration overrides the keyword prior.

    Returns:
        One of ``"refund"``, ``"status"``, or ``"technical"``.
    """
    if prompt.demos:
        demo, sim = _nearest_demo(text, prompt.demos)
        if demo is not None and sim >= demo_threshold:
            return demo.label
    return _prior_label(text, prompt.precedence)


DEV_SET = (
    Ticket("Where is order 41? The tracking page has not moved.", "status"),
    Ticket("The desktop app crashes on launch every time.", "technical"),
    Ticket("Please refund the duplicate charge on my card.", "refund"),
    Ticket("My package says delivered, but it is not here.", "status"),
    Ticket("The tracking page crashes whenever I open it.", "technical"),
    Ticket("I want my money back because the parcel is late.", "refund"),
    Ticket("Reset links always return a server error.", "technical"),
    Ticket("Has order 77 shipped yet?", "status"),
    Ticket("Cancel and refund the order that has not shipped.", "refund"),
    Ticket("The delivery screen freezes after sign-in.", "technical"),
    Ticket("You charged me twice for one order.", "refund"),
    Ticket("You charged me twice; refund one of the two order charges.", "refund"),
)

HELDOUT_SET = (
    Ticket("The checkout page throws an error at payment.", "technical"),
    Ticket("Where is my parcel? Tracking stopped three days ago.", "status"),
    Ticket("Refund me; the shipment never arrived.", "refund"),
    Ticket("The delivery map freezes and the app crashes.", "technical"),
    Ticket("Two identical payments left my account this week.", "refund"),
    Ticket("Please reimburse the charge for the broken item.", "refund"),
)

BASELINE = Prompt("Classify the customer ticket. Return one label.")


def evaluate(prompt: Prompt, examples: Sequence[Ticket]) -> dict:
    """Score a prompt on a case set and keep the per-case failure rows.

    Args:
        prompt: The prompt configuration to score.
        examples: The frozen cases to run it against.

    Returns:
        A dict with the exact-match ``score`` in ``[0, 1]``, the list of
        ``failures`` (each an expected/actual/text row), and every ``rows``
        prediction — the failure rows, not the scalar, are the primary artifact.
    """
    rows, failures = [], []
    for item in examples:
        actual = classify(prompt, item.text)
        row = {"text": item.text, "expected": item.expected, "actual": actual}
        rows.append(row)
        if actual != item.expected:
            failures.append(row)
    score = (len(examples) - len(failures)) / len(examples)
    return {"score": score, "failures": failures, "rows": rows}


def _propose_precedence(current: tuple[str, ...], failures: Sequence[dict]) -> tuple[str, ...]:
    """Promote the most-confused expected label above the label it loses to."""
    confusion = Counter((f["actual"], f["expected"]) for f in failures)
    if not confusion:
        return current
    (lost_to, should_be), _ = confusion.most_common(1)[0]
    order = list(current)
    if lost_to in order and should_be in order and order.index(should_be) > order.index(lost_to):
        order.remove(should_be)
        order.insert(order.index(lost_to), should_be)
    return tuple(order)


def _bootstrap_demo(prompt: Prompt, dev: Sequence[Ticket], failures: Sequence[dict]) -> Demo | None:
    """Turn a currently-correct dev case near the first failure into a demonstration."""
    if not failures:
        return None
    fail = failures[0]
    existing = {d.text for d in prompt.demos}
    best, best_sim = None, 0.0
    for item in dev:
        if item.text in existing or item.text == fail["text"]:
            continue
        if item.expected != fail["expected"] or classify(prompt, item.text) != item.expected:
            continue
        sim = _similarity(fail["text"], item.text)
        if sim > best_sim:
            best, best_sim = item, sim
    return Demo(best.text, best.expected) if best is not None and best_sim >= 0.5 else None


def optimize_prompt(dev: Sequence[Ticket], heldout: Sequence[Ticket], rounds: int = 3) -> dict:
    """Search prompt space with a propose -> evaluate -> keep-best loop over a metric.

    Each round generates candidates from the current best prompt's failures — an
    instruction precedence edit and a bootstrapped demonstration — scores them
    all on ``dev``, and keeps the highest (ties break toward the earlier, simpler
    candidate). The held-out score is recorded but never optimized against, so a
    growing dev-minus-held-out gap exposes overfitting to the dev set.

    Args:
        dev: The development split the search is allowed to see.
        heldout: An untouched split, scored only for reporting.
        rounds: How many propose/evaluate rounds to run.

    Returns:
        A dict with the ``winner`` prompt and the per-round ``trajectory`` of
        dev and held-out scores.
    """
    best, trajectory = BASELINE, []
    for step in range(rounds + 1):
        dev_result = evaluate(best, dev)
        trajectory.append({
            "round": step,
            "dev_score": dev_result["score"],
            "heldout_score": evaluate(best, heldout)["score"],
            "n_demos": len(best.demos),
            "precedence": best.precedence,
        })
        if step == rounds:
            break
        candidates = []
        new_prec = _propose_precedence(best.precedence, dev_result["failures"])
        if new_prec != best.precedence:
            edited = best.instruction + f" PRECEDENCE: {' > '.join(new_prec)}."
            candidates.append(replace(best, instruction=edited, precedence=new_prec))
        demo = _bootstrap_demo(best, dev, dev_result["failures"])
        if demo is not None:
            candidates.append(replace(best, demos=best.demos + (demo,)))
        scored = [(evaluate(c, dev)["score"], -i, c) for i, c in enumerate(candidates)]
        scored.append((dev_result["score"], 1, best))
        best = max(scored, key=lambda t: (t[0], t[1]))[2]
    return {"winner": best, "trajectory": trajectory}


def token_count(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class Segment:
    """One typed unit of context with the metadata a policy can act on.

    ``kind`` fixes structural position; ``trust`` records who controls the
    content (provenance, not authorization); ``stable`` marks material that can
    form a reusable prefix; ``priority`` and ``tags`` drive selection.
    """

    kind: str
    trust: str
    content: str
    stable: bool = False
    priority: int = 0
    tags: tuple[str, ...] = ()


ORDER = {"system": 0, "tools": 1, "examples": 2, "retrieved": 3, "history": 4, "query": 5}


def render_context(segments: Sequence[Segment], budget: int) -> str:
    """Render typed segments into one deterministic, trust-labelled sequence.

    Segments are ordered stable-to-volatile by a fixed table, each wrapped with
    its kind and trust. The budget is applied at block boundaries: an optional
    over-budget segment is dropped, but a required system or query segment that
    will not fit raises instead of vanishing, so truncation can never silently
    delete policy or the user's question.

    Args:
        segments: The typed segments to assemble, in any order.
        budget: The whitespace-token proxy budget for the whole input.

    Returns:
        The assembled context string.

    Raises:
        ValueError: If a required (stable or query) segment exceeds the budget.
    """
    ordered = sorted(enumerate(segments), key=lambda pair: (ORDER[pair[1].kind], pair[0]))
    rendered, used = [], 0
    for _, segment in ordered:
        block = f'<{segment.kind} trust="{segment.trust}">\n{segment.content}\n</{segment.kind}>'
        size = token_count(block)
        if used + size > budget and not segment.stable and segment.kind != "query":
            continue
        if used + size > budget:
            raise ValueError(f"required {segment.kind} segment exceeds the {budget}-token budget")
        rendered.append(block)
        used += size
    return "\n".join(rendered)


def select_context(segments: Sequence[Segment], query: str) -> list[Segment]:
    """Keep stable segments, the query, high-priority items, and tag matches."""
    terms = _content_tokens(query)
    return [s for s in segments
            if s.stable or s.kind == "query" or s.priority >= 8 or terms & set(s.tags)]


def common_prefix_bytes(left: str, right: str) -> int:
    """Count identical leading bytes of two strings; the first difference ends reuse.

    A prefix cache can reuse KV state only up to the first byte at which two
    requests diverge, so this count is the caller-side diagnostic for how much
    of a request a stable layout keeps reusable.

    Args:
        left: The previous request's assembled bytes.
        right: The current request's assembled bytes.

    Returns:
        The number of identical leading UTF-8 bytes.
    """
    a, b = left.encode("utf-8"), right.encode("utf-8")
    for index, (one, two) in enumerate(zip(a, b)):
        if one != two:
            return index
    return min(len(a), len(b))


def cache_cost(stable_tokens: int, volatile_tokens: int, read_rate: float = 0.10) -> dict:
    """Illustrative cache economics: a hit reads the stable prefix at a discount."""
    hit = stable_tokens * read_rate + volatile_tokens
    miss = stable_tokens + volatile_tokens
    return {"hit": hit, "miss": miss, "multiplier": miss / hit}


DURABLE_PREFIXES = ("DECISION ", "OPEN ", "ERROR ")


def assert_survival(before: Sequence[str], after: Sequence[str]) -> None:
    """Fail closed unless every pre-compaction decision and open question survives.

    Literal substring survival is stronger than a model grader for these fields
    and far cheaper to debug: a decision or open question that vanished from the
    compacted transcript is a hard error, not a quality regression.

    Args:
        before: The full transcript before compaction.
        after: The compacted transcript.

    Raises:
        AssertionError: If any ``DECISION`` or ``OPEN`` line is missing from
            ``after``.
    """
    joined = "\n".join(after)
    missing = [m for m in before if m.startswith(("DECISION ", "OPEN ")) and m not in joined]
    if missing:
        raise AssertionError(f"compaction lost durable state: {missing}")


def compact_history(messages: Sequence[str], budget: int, trigger: float = 0.65,
                    keep_recent: int = 4) -> tuple[list[str], bool]:
    """Compact a transcript by classifying durable state before summarizing the rest.

    The rule to take away: never summarize first. Extract the state that must
    survive verbatim (decisions, open questions, errors), summarize only the
    expendable middle, keep the recent turns, then assert survival. Compaction
    fires only when utilization crosses ``trigger``, leaving headroom for the
    next turn, tool results, and output.

    Args:
        messages: The full ordered transcript, oldest first.
        budget: The whitespace-token proxy budget for the active window.
        trigger: Fraction of ``budget`` at which compaction fires.
        keep_recent: Number of most-recent messages kept verbatim.

    Returns:
        A tuple ``(compacted, changed)``; ``changed`` is False when the
        transcript already fit and nothing was rewritten.

    Raises:
        AssertionError: If a durable decision or open question would be lost.
    """
    if token_count("\n".join(messages)) <= budget * trigger:
        return list(messages), False
    durable = [m for m in messages if m.startswith(DURABLE_PREFIXES)]
    recent = list(messages[-keep_recent:])
    omitted = len(messages) - len({*durable, *recent})
    summary = f"SUMMARY: {omitted} routine messages folded; no durable state added."
    compacted = [*durable, summary, *[m for m in recent if m not in durable]]
    assert_survival(messages, compacted)
    return compacted, True


def synthetic_history(turns: int = 40) -> list[str]:
    """Build a debugging trace with sparse, high-value durable state to compact.

    Most turns are routine chatter; a handful carry ``DECISION``, ``OPEN``, and
    ``ERROR`` lines, so the compactor has real durable state to preserve.

    Args:
        turns: Total number of messages in the trace.

    Returns:
        The ordered transcript, oldest message first.
    """
    anchors = {
        5: "DECISION D1: preserve typed tool results; correlation IDs are evidence.",
        11: "OPEN Q1: reproduce the timeout under the production proxy.",
        18: "DECISION D2: retry at most twice; duplicate writes are unsafe.",
        24: "ERROR E1: proxy closed the stream before the tool result arrived.",
        30: "OPEN Q2: confirm whether order writes are idempotent.",
        34: "DECISION D3: pin schema version v4; replay depends on it.",
    }
    return [anchors.get(t, f"TURN {t:02d}: inspected a routine trace batch " + "detail " * 10).strip()
            for t in range(1, turns + 1)]


POLICY_TOKENS = 600
STABLE_POLICY = "SYSTEM\n" + "stable-policy-token " * POLICY_TOKENS


def simulate_turns(compaction: bool, stable_prefix: bool, turns: int = 50,
                   budget: int = 1100) -> list[dict]:
    """Grow a transcript turn by turn, recording utilization and reusable-prefix fraction.

    Each turn appends an observation (with occasional durable decision, open
    question, and error lines); with ``compaction`` on, the survival-checked
    ``compact_history`` runs at a clean boundary. With ``stable_prefix`` off, a
    changing timestamp is prepended, which breaks cache reuse at byte 0. The two
    controls are independent: compaction bounds utilization, a stable prefix
    preserves reuse, and neither substitutes for the other.

    Args:
        compaction: Whether to compact when history crosses the trigger.
        stable_prefix: Whether the leading bytes stay identical across turns.
        turns: Number of turns to simulate.
        budget: The whitespace-token proxy budget for the whole window.

    Returns:
        One ledger row per turn with tokens, utilization, reuse fraction, and a
        compaction flag.
    """
    anchors = {8: "DECISION D1: route delivery failures to status, not refund.",
               15: "OPEN Q1: does the refund tool need a second approval?",
               22: "ERROR E1: tool timeout on the refund path.",
               31: "DECISION D2: cap auto-refunds at two per account per day.",
               40: "OPEN Q2: is the shipment webhook idempotent?"}
    history, previous, ledger = [], "", []
    history_budget = budget - token_count(STABLE_POLICY)
    for turn in range(1, turns + 1):
        history.append(anchors.get(turn, f"TURN {turn:02d} observation " + "working-token " * 20))
        did_compact = False
        if compaction:
            history, did_compact = compact_history(history, history_budget,
                                                    trigger=0.65, keep_recent=3)
        body = STABLE_POLICY + "\n" + "\n".join(history)
        prompt = body if stable_prefix else f"NOW=2026-07-20T12:00:{turn:02d}Z\n{body}"
        reusable = common_prefix_bytes(previous, prompt) if previous else 0
        ledger.append({"turn": turn, "tokens": token_count(prompt),
                       "utilization": token_count(prompt) / budget,
                       "reuse": reusable / len(prompt.encode("utf-8")),
                       "compacted": did_compact})
        previous = prompt
    return ledger
