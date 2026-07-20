# Auto-generated from chapters/27-operating-agents.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Link


def build_tracer() -> tuple[Any, InMemorySpanExporter]:
    """Create an isolated in-memory OpenTelemetry pipeline for examples and tests.

    An in-memory exporter keeps the span tree in a Python list instead of
    shipping it to a collector, so the structure a run produces is something
    a test can read back and assert on. Production swaps this one object for
    an OTLP exporter and nothing else changes.

    Returns:
        The tracer to open spans with, and the exporter holding finished spans.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("agentic-book-ch27"), exporter


def emit_linked_run() -> tuple[list[Any], str]:
    """Emit one agent run whose approval pause splits it into two linked traces.

    The first trace covers work up to the suspension and then ends, so the
    days spent waiting for a human never inflate a span. The second trace
    begins on resume and carries a ``Link`` to the first, which is how an
    observability backend reconstructs one journey from two bounded traces.

    Returns:
        The finished spans, and the first trace id as a hex string.
    """
    tracer, exporter = build_tracer()
    with tracer.start_as_current_span(
        "invoke_agent",
        attributes={"gen_ai.operation.name": "invoke_agent",
                    "app.run.id": "run-17", "app.release.hash": "bundle-42"},
    ) as first:
        with tracer.start_as_current_span(
            "chat",
            attributes={"gen_ai.operation.name": "chat",
                        "gen_ai.request.model": "fixture-model",
                        "gen_ai.usage.input_tokens": 20_000,
                        "gen_ai.usage.output_tokens": 2_000,
                        "app.cost": 0.013},
        ):
            pass
        with tracer.start_as_current_span(
            "request_approval", attributes={"app.action.id": "refund:A-17:v1"},
        ):
            pass
        first_context = first.get_span_context()
    with tracer.start_as_current_span(
        "resume_agent",
        links=[Link(first_context, attributes={"app.link.reason": "approval_resume"})],
        attributes={"app.run.id": "run-17", "app.approval.decision": "approve"},
    ):
        with tracer.start_as_current_span(
            "execute_tool",
            attributes={"gen_ai.operation.name": "execute_tool",
                        "app.action.id": "refund:A-17:v1",
                        "app.effect.receipt": "receipt-1"},
        ):
            pass
    return list(exporter.get_finished_spans()), f"{first_context.trace_id:032x}"


def print_span_tree(spans: list[Any]) -> None:
    """Print finished spans as one indented tree per trace, marking links.

    Grouping by trace and ordering by start time turns a flat span list back
    into the causal picture, and annotating the link shows how the resumed
    trace points home. This is the diagnostic view the aggregate dashboards
    threw away.

    Args:
        spans: Finished spans from an in-memory exporter.
    """
    by_trace: dict[int, list[Any]] = {}
    for span in spans:
        by_trace.setdefault(span.context.trace_id, []).append(span)
    ordered = sorted(by_trace.items(), key=lambda kv: min(s.start_time for s in kv[1]))
    number = {trace_id: i for i, (trace_id, _) in enumerate(ordered, start=1)}
    for trace_id, group in ordered:
        by_parent: dict[int | None, list[Any]] = {}
        for span in group:
            by_parent.setdefault(span.parent.span_id if span.parent else None, []).append(span)

        def walk(parent_id: int | None, depth: int) -> None:
            for span in sorted(by_parent.get(parent_id, []), key=lambda s: s.start_time):
                link = (f"   (links to trace {number[span.links[0].context.trace_id]})"
                        if span.links else "")
                print(f"trace {number[trace_id]}  " + "  " * depth + f"{span.name}{link}")
                walk(span.context.span_id, depth + 1)

        walk(None, 0)


def trace_cost(spans: list[Any]) -> float:
    """Sum the diagnostic cost carried on spans; the bill lives in the ledger.

    Reading cost from sampled traces is convenient for attributing spend to a
    trajectory during debugging, but a lost span would silently lower the
    total — so this number is a diagnostic estimate, never the amount a
    customer is charged.

    Args:
        spans: Finished spans that may carry an ``app.cost`` attribute.

    Returns:
        The summed cost, rounded, for diagnostic display only.
    """
    return round(sum(float(s.attributes.get("app.cost", 0.0)) for s in spans), 6)


def replay_execute(name: str, arguments: dict[str, Any], effectful: bool,
                   recorded: dict[str, Any], effect_log: list[Any]) -> dict[str, Any]:
    """Re-run a tool during replay: reads return recorded data, writes no-op.

    Replay reproduces a trajectory to diagnose it, so a read is served from
    what was recorded and a write is *suppressed* rather than executed. The
    empty ``effect_log`` afterward is the proof that diagnosing a run did not
    repeat its consequences.

    Args:
        name: The tool being replayed.
        arguments: The recorded arguments (unused for suppressed writes).
        effectful: Whether the tool changes the world.
        recorded: Recorded read results keyed by tool name.
        effect_log: The append-only effect log, which must stay untouched.

    Returns:
        The recorded read result, or a marker that the write was suppressed.
    """
    if effectful:
        return {"replayed": name, "effect_suppressed": True}
    return recorded[name]


from dataclasses import dataclass


@dataclass(frozen=True)
class RunRecord:
    """One finished journey, reduced to the fields the indicators need.

    ``success`` is the control-flow fact that the agent answered; ``grounded``
    is the separate fact that the answer was supported by authoritative state.
    Keeping them apart is the whole point — a fluent, ungrounded answer is the
    failure mode a task-success counter alone will never show.
    """

    run_id: str
    tenant: str
    success: bool
    grounded: bool
    effect_count: int
    ttft_ms: float
    cost: float
    score: float


import math


def nearest_rank(values: list[float], percentile: float) -> float:
    """Return the nearest-rank percentile, deterministic on small batches.

    Args:
        values: The sample.
        percentile: A fraction in ``(0, 1]``.

    Returns:
        The value at the nearest rank.
    """
    if not values:
        raise ValueError("values cannot be empty")
    return sorted(values)[max(1, math.ceil(percentile * len(values))) - 1]


def compute_slis(records: list[RunRecord]) -> dict[str, float]:
    """Compute journey-level quality, effect, latency, and cost indicators.

    Success counts only runs that are both authoritatively successful *and*
    grounded, and cost-per-task divides every run's cost by the count of
    successful runs so that failing early never flatters the bill. These two
    choices are why the indicators track the user promise rather than the
    endpoint's health.

    Args:
        records: A batch of finished journeys.

    Returns:
        A mapping of indicator name to value.
    """
    if not records:
        raise ValueError("records cannot be empty")
    good = [r for r in records if r.success and r.grounded]
    return {
        "success_and_grounded_rate": len(good) / len(records),
        "exactly_one_effect_rate": sum(r.effect_count == 1 for r in records) / len(records),
        "p95_ttft_ms": nearest_rank([r.ttft_ms for r in records], 0.95),
        "cost_per_successful_task": round(sum(r.cost for r in records) / len(good), 4),
    }


def fixture_records() -> list[RunRecord]:
    """Return a small batch built to make every indicator hand-checkable.

    Returns:
        Four runs: two success-and-grounded, one successful but ungrounded,
        and one plain failure — a mix HTTP status alone would call healthy.
    """
    return [
        RunRecord("r1", "alpha", True, True, 1, 80, 0.010, 0.96),
        RunRecord("r2", "alpha", True, True, 1, 90, 0.012, 0.94),
        RunRecord("r3", "beta", True, False, 1, 110, 0.014, 0.45),
        RunRecord("r4", "beta", False, False, 0, 400, 0.020, 0.30),
    ]


def burn_rate(good: int, total: int, slo_target: float) -> float:
    """Divide the observed bad-event rate by the rate the SLO permits.

    A result of 1 means the budget is being spent exactly as fast as it
    accrues; 6.9 means nearly seven months of allowance are being consumed in
    one, which is why a burn rate turns a bland success percentage into an
    alarm with a deadline.

    Args:
        good: Observed good events.
        total: Eligible events.
        slo_target: The objective, strictly between 0 and 1.

    Returns:
        The burn rate as a multiple of the sustainable rate.
    """
    if not 0 < slo_target < 1 or not 0 <= good <= total or total <= 0:
        raise ValueError("invalid SLO counts or target")
    return (1 - good / total) / (1 - slo_target)


def days_to_exhaustion(window_days: float, remaining_fraction: float, burn: float) -> float:
    """Estimate days until the budget is gone under a constant burn rate.

    Args:
        window_days: The SLO window length in days.
        remaining_fraction: The fraction of budget still unspent.
        burn: The current burn rate.

    Returns:
        Days to exhaustion, or infinity when nothing is burning.
    """
    if burn <= 0:
        return math.inf
    return window_days * remaining_fraction / burn


def budget_remaining(day: float, burn: float, window_days: float = 30.0) -> float:
    """Fraction of a full error budget left after ``day`` days at ``burn``.

    Args:
        day: Days elapsed at constant burn.
        burn: The burn rate.
        window_days: The SLO window length.

    Returns:
        The remaining budget fraction, clamped at zero.
    """
    return max(0.0, 1 - burn * day / window_days)


def window_burn(bad_flags: list[float], slo_target: float) -> float:
    """Burn rate over one window from its per-slot bad fractions.

    Args:
        bad_flags: Per-slot bad fractions in the window.
        slo_target: The objective.

    Returns:
        The window's burn rate, zero for an empty window.
    """
    if not bad_flags:
        return 0.0
    return (sum(bad_flags) / len(bad_flags)) / (1 - slo_target)


def multiwindow_page(long_burn: float, short_burn: float, threshold: float = 14.4) -> bool:
    """Fire only when both windows exceed the threshold together.

    Requiring the long *and* the short window to agree is what buys precision:
    the long window rejects brief noise, and the short window lets the alert
    resolve itself minutes after the incident does instead of ringing for an
    hour.

    Args:
        long_burn: Burn over the long confirmation window.
        short_burn: Burn over the short reaction window.
        threshold: The multiple both windows must exceed.

    Returns:
        Whether to page.
    """
    return long_burn >= threshold and short_burn >= threshold


from statistics import mean


def drift_by_tenant(baseline: list[RunRecord], current: list[RunRecord],
                    threshold: float) -> dict[str, float]:
    """Return per-tenant score drops exceeding a threshold, hiding none in a mean.

    A fleet average is a weighted blend, so one tenant's improvement can mask
    another's regression. Comparing tenants separately is what surfaces the
    concentrated harm a single number launders away.

    Args:
        baseline: Reference-window records.
        current: Live-window records.
        threshold: The minimum mean drop worth alerting on.

    Returns:
        A mapping of tenant to score drop for tenants past the threshold.
    """
    def by_tenant(records: list[RunRecord]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for r in records:
            out.setdefault(r.tenant, []).append(r.score)
        return out

    base, live = by_tenant(baseline), by_tenant(current)
    alerts = {}
    for tenant in sorted(base.keys() & live.keys()):
        drop = mean(base[tenant]) - mean(live[tenant])
        if drop >= threshold:
            alerts[tenant] = round(drop, 6)
    return alerts


from collections import deque


def streaming_drift_delay(reference_mean: float, stream: list[float],
                          window: int, threshold: float) -> int:
    """Return the sample index at which a trailing window first detects a drop.

    A wider window rejects noise but detects later, because its mean cannot
    cross the threshold until enough shifted samples have displaced the old
    ones — the detection delay is the price of the smoothing.

    Args:
        reference_mean: The pre-shift mean to compare against.
        stream: Incoming scores in order.
        window: Trailing-window width.
        threshold: The drop below reference that fires.

    Returns:
        The 1-based sample count at first detection, or -1 if never.
    """
    win: deque[float] = deque(maxlen=window)
    for index, score in enumerate(stream):
        win.append(score)
        if len(win) == window and (reference_mean - mean(win)) >= threshold:
            return index + 1
    return -1


def canary_decision(control_good: int, control_total: int, canary_good: int,
                    canary_total: int, min_count: int = 100, z_crit: float = 1.645) -> dict[str, Any]:
    """Decide ship / hold / rollback for a canary from a two-proportion test.

    A canary that looks worse might just be small, so the rule holds for more
    evidence below ``min_count``, rolls back only when the canary is
    *significantly* worse than control, and ships otherwise — the same
    discipline an A/B test uses, applied to a release gate.

    Args:
        control_good: Good journeys on the current release.
        control_total: Eligible journeys on control.
        canary_good: Good journeys on the canary.
        canary_total: Eligible journeys on the canary.
        min_count: Minimum canary sample before deciding.
        z_crit: One-sided critical value for "significantly worse".

    Returns:
        A decision with the z-statistic and both proportions.
    """
    p_control, p_canary = control_good / control_total, canary_good / canary_total
    if canary_total < min_count:
        return {"decision": "hold", "z": None, "p_control": p_control, "p_canary": p_canary}
    pooled = (control_good + canary_good) / (control_total + canary_total)
    se = math.sqrt(pooled * (1 - pooled) * (1 / control_total + 1 / canary_total))
    z = (p_canary - p_control) / se if se > 0 else 0.0
    decision = "rollback" if z <= -z_crit else "ship"
    return {"decision": decision, "z": round(z, 2), "p_control": p_control, "p_canary": p_canary}


def autonomy_action(burn: float, safety_violation: bool = False) -> str:
    """Map fleet evidence to a bounded runtime posture.

    Rising burn tightens autonomy step by step — ticket, review, read-only —
    so blast radius shrinks before anyone knows the cause. A safety violation
    is not on this ladder at all: it stops new effects regardless of burn,
    because safety is an invariant, not a budget you are allowed to spend.

    Args:
        burn: The current burn rate.
        safety_violation: Whether an invariant (not a quality target) broke.

    Returns:
        The runtime posture to enter.
    """
    if safety_violation:
        return "stop_new_effects"
    if burn >= 14.4:
        return "read_only_and_freeze_rollout"
    if burn >= 6.0:
        return "require_review_and_page"
    if burn >= 1.0:
        return "open_ticket"
    return "normal"


def apply_effect_once(store: dict[str, Any], key: str, receipt: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Apply an effect at most once per idempotency key.

    Keying the write on a stable idempotency token is what makes a retry after
    a crash safe: the second attempt finds the key already present and returns
    the first receipt instead of moving money twice.

    Args:
        store: The effect store keyed by idempotency token.
        key: The idempotency key for this action.
        receipt: The receipt to record on first application.

    Returns:
        The stored receipt and whether this call created a new effect.
    """
    if key in store:
        return store[key], False
    store[key] = receipt
    return receipt, True


def run_chaos_drill(use_idempotency_key: bool) -> int:
    """Kill a worker in the effect gap, retry, and count surviving effects.

    Args:
        use_idempotency_key: Whether the retry reuses a stable key.

    Returns:
        The number of distinct effects after the retry.
    """
    store: dict[str, Any] = {}
    for attempt in (1, 2):  # attempt 1 accepted then killed; attempt 2 is the retry
        key = "refund:A-17:v1" if use_idempotency_key else f"auto-token-{attempt}"
        apply_effect_once(store, key, {"attempt": attempt, "refunded_cents": 4999})
    return len(store)


def incident_timeline(events: list[dict[str, Any]]) -> dict[str, int]:
    """Reduce an incident's event log to the intervals that grade the response.

    Mean time to detect — first harmful event to detection — is singled out
    because it is the interval where harm compounds unseen, and it is measured
    from the first *harmful effect*, not the first alert, so a late alarm
    cannot flatter the number.

    Args:
        events: Records with an integer ``t`` (minutes) and a ``kind``.

    Returns:
        Named intervals in minutes.
    """
    t = {e["kind"]: e["t"] for e in events}
    return {
        "mttd_min": t["detected"] - t["first_harmful"],
        "time_to_contain_min": t["contained"] - t["detected"],
        "harm_window_min": t["last_harmful_effect"] - t["first_harmful"],
        "technical_recovery_min": t["technical_recovered"] - t["detected"],
        "business_recovery_min": t["business_recovered"] - t["detected"],
    }


def as_regression_test(incident_id: str, invariant: str, probe: str) -> dict[str, str]:
    """Freeze a resolved incident into a permanent regression case.

    An incident that leaves only a postmortem can recur; one that leaves a
    runnable probe on the exact invariant it broke cannot recur silently,
    which is why "add the test" is the corrective action and "be more careful"
    is not.

    Args:
        incident_id: The incident identifier.
        invariant: The property that must hold forever after.
        probe: The runnable check that asserts it.

    Returns:
        A regression case wired to the release and chaos gates.
    """
    return {"case_id": f"regression::{incident_id}", "invariant": invariant,
            "probe": probe, "gate": "release + chaos"}


@dataclass(frozen=True)
class ApprovalRecord:
    """One review decision: what was decided and how long it took.

    Latency and decision together are what expose a rubber stamp — near-total
    approval reached in a fraction of a second is not oversight, it is a queue
    whose policy should either automate or supply harder cases.
    """

    action_id: str
    decision: str  # approve | deny | override | modify | abandon
    latency_s: float


def hitl_metrics(records: list[ApprovalRecord]) -> dict[str, Any]:
    """Measure whether a review queue oversees or merely rubber-stamps.

    A 99 percent approval rate reached in 0.2 seconds is flagged not because
    approving is wrong but because a person cannot weigh a consequential action
    that fast — the combined rate-and-latency signal catches the degenerate
    queue a raw approval count would call healthy.

    Args:
        records: Completed and abandoned review decisions.

    Returns:
        Approval, override, and abandonment rates, median decision time, and
        the rubber-stamp signal.
    """
    if not records:
        raise ValueError("approval records cannot be empty")
    completed = [r for r in records if r.decision != "abandon"]
    approvals = [r for r in completed if r.decision == "approve"]
    overrides = [r for r in completed if r.decision in {"override", "modify"}]
    latencies = sorted(r.latency_s for r in completed)
    n = len(latencies)
    median = latencies[n // 2] if n % 2 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2
    approval_rate = len(approvals) / len(completed) if completed else 0.0
    return {
        "approval_rate": round(approval_rate, 4),
        "override_rate": round(len(overrides) / len(completed), 4) if completed else 0.0,
        "abandonment_rate": round((len(records) - len(completed)) / len(records), 4),
        "median_decision_s": median,
        "rubber_stamp_signal": approval_rate >= 0.98 and median < 1.0,
    }


def queue_length(arrival_per_hr: float, service_time_hr: float) -> float:
    """Average items in the system by Little's Law, ``L = lambda * W``."""
    return arrival_per_hr * service_time_hr


def queue_stable(arrival_per_hr: float, reviewers: int, service_time_hr: float) -> bool:
    """Whether review capacity strictly exceeds the arrival rate.

    Args:
        arrival_per_hr: Review items arriving per hour.
        reviewers: Number of reviewers.
        service_time_hr: Average handling time per item in hours.

    Returns:
        True when the queue does not grow without bound.
    """
    return arrival_per_hr < reviewers / service_time_hr


import hashlib
import re

SECRET_SEGMENTS = {"authorization", "api_key", "api-key", "apikey", "access_token",
                   "token", "secret", "password", "cookie", "set-cookie"}
BEARER = re.compile(r"(?i)bearer\s+[a-z0-9._~+/=-]+")


def _secret_key(key: str) -> bool:
    return key.rsplit(".", 1)[-1].strip().lower() in SECRET_SEGMENTS


def sanitize_attributes(attributes: dict[str, Any], capture_content: bool = False) -> dict[str, Any]:
    """Redact secrets and minimize identity and content before span creation.

    Secret-named fields are matched on whole dotted segments, not substrings,
    so ``input_tokens`` is never mistaken for a ``token`` secret — the exact
    false positive that would have deleted benign usage counts. Content is
    dropped unless explicitly captured, tenant ids become short surrogates,
    and bearer values are scrubbed even out of free-text error messages.

    Args:
        attributes: Raw span attributes.
        capture_content: Whether approved content capture is enabled.

    Returns:
        A sanitized copy safe to attach to a span.
    """
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        if _secret_key(key):
            cleaned[key] = "[REDACTED]"
        elif key in {"gen_ai.input.messages", "gen_ai.output.messages"} and not capture_content:
            cleaned[key] = "[CONTENT_DISABLED]"
        elif key == "app.tenant.id":
            cleaned[key] = hashlib.sha256(str(value).encode()).hexdigest()[:12]
        elif isinstance(value, str):
            cleaned[key] = BEARER.sub("Bearer [REDACTED]", value)
        else:
            cleaned[key] = value
    return cleaned


def dp_epsilon(per_query_eps: list[float]) -> dict[str, float]:
    """Compose a differential-privacy budget and its worst-case leakage.

    Under basic composition the epsilons of successive queries add, and the
    total bounds how much one person's presence can change the output
    distribution by a factor of ``e**epsilon`` — so spending budget is literal,
    and after enough queries the guarantee is gone.

    Args:
        per_query_eps: Per-query epsilon spends.

    Returns:
        The total epsilon and its maximum likelihood ratio.
    """
    total = sum(per_query_eps)
    return {"total_epsilon": round(total, 4), "max_likelihood_ratio": round(math.exp(total), 2)}


def federated_average(client_updates: list[tuple[float, int]]) -> float:
    """Weight client summaries by example count without moving raw data.

    Each client contributes a summary and its sample size; the server forms a
    weighted mean, so raw examples never leave the device — though the summary
    itself can still leak and needs secure aggregation and, often, differential
    privacy on top.

    Args:
        client_updates: ``(summary, n_examples)`` per client.

    Returns:
        The example-weighted average.
    """
    total_n = sum(n for _, n in client_updates)
    if total_n == 0:
        raise ValueError("no client examples")
    return round(sum(update * n for update, n in client_updates) / total_n, 6)


def operator_snapshot() -> dict[str, Any]:
    """Assemble the console's headline numbers from the chapter's own functions.

    Every field is produced by a function built earlier, so the snapshot is a
    real reduction of the fleet's runtime evidence — traces, indicators, burn,
    canary, chaos, and review — into the handful of numbers an operator reads
    first.

    Returns:
        A dictionary of the console's headline metrics.
    """
    spans, _ = emit_linked_run()
    b = burn_rate(9_655, 10_000, 0.995)
    reviews = [ApprovalRecord(f"a{i}", "approve", 0.2) for i in range(99)] + \
        [ApprovalRecord("a99", "deny", 0.3)]
    return {
        "spans": len(spans),
        "traces": len({s.context.trace_id for s in spans}),
        "diagnostic_trace_cost": trace_cost(spans),
        "slis": compute_slis(fixture_records()),
        "fleet_burn": round(b, 1),
        "days_to_exhaustion": round(days_to_exhaustion(30, 0.95, b), 1),
        "runtime_action": autonomy_action(b),
        "canary": canary_decision(970, 1000, 900, 1000)["decision"],
        "chaos_exactly_once": run_chaos_drill(True) == 1,
        "rubber_stamp": hitl_metrics(reviews)["rubber_stamp_signal"],
    }
