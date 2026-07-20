"""Invariants for the Chapter 26 production platform.

The chapter authors its teaching code inline in ``{python}`` cells; the reusable
definitions are tangled to ``code/ch26/_generated.py``. These tests import that
generated module under a unique name so the suite can run beside every other
chapter's generated module without a name collision.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path


_GENERATED = Path(__file__).parents[1] / "code" / "ch26" / "_generated.py"
_spec = importlib.util.spec_from_file_location("ch26_generated", _GENERATED)
assert _spec and _spec.loader
ch26 = importlib.util.module_from_spec(_spec)
sys.modules["ch26_generated"] = ch26  # so dataclasses can resolve __module__
_spec.loader.exec_module(ch26)


# --- gateway: identity, admission, routing -------------------------------


def _gateway() -> "ch26.Gateway":
    return ch26.Gateway(
        keys={"key-a": ("tenant-a", frozenset({"chat"}))},
        limits={"tenant-a": 0.01},
        deployments=[
            ch26.Deployment("cheap", frozenset({"chat"}), 1.0),
            ch26.Deployment("rich", frozenset({"chat", "vision"}), 5.0),
        ],
    )


def test_identity_and_cost_derive_from_virtual_key() -> None:
    gate = _gateway()
    context = gate.authenticate("key-a", "req-1", 10.0)
    assert context.tenant == "tenant-a"
    assert gate.route(context, "chat").name == "cheap"  # cheapest eligible
    assert gate.settle(context, ch26.Usage(1_000, 100, 1.0, 2.0)) == 0.0012


def test_unknown_key_is_refused() -> None:
    gate = _gateway()
    try:
        gate.authenticate("nope", "req-1", 10.0)
    except PermissionError:
        pass
    else:
        raise AssertionError("unknown virtual key authenticated")


def test_budget_blocks_whole_journey_before_call() -> None:
    gate = _gateway()
    context = gate.authenticate("key-a", "req-1", 10.0)
    try:
        gate.admit(context, 0.02)
    except ch26.BudgetError:
        pass
    else:
        raise AssertionError("over-budget journey was admitted")


def test_route_requires_a_healthy_capable_deployment() -> None:
    gate = _gateway()
    context = gate.authenticate("key-a", "req-1", 10.0)
    try:
        gate.route(context, "audio")
    except LookupError:
        pass
    else:
        raise AssertionError("routed to a deployment that lacks the capability")


def test_usage_breakdown_includes_tool_and_review_terms() -> None:
    usage = ch26.Usage(1_000_000, 0, 1.0, 0.0, tool_units=2.0, review_units=3.0)
    parts = usage.breakdown()
    assert parts["input"] == 1.0 and parts["tool"] == 2.0 and parts["review"] == 3.0
    assert usage.cost == 6.0


# --- routing vs cascade -------------------------------------------------


def test_cascade_beats_small_and_undercuts_large() -> None:
    small = ch26.Backend("small", 0.55, 1.0, 200.0)
    large = ch26.Backend("large", 0.97, 6.0, 900.0)
    tasks = [i / 400 for i in range(400)]
    single_small = ch26.run_single(small, tasks)
    single_large = ch26.run_single(large, tasks)
    cascade = ch26.run_cascade(small, large, tasks, false_accept=0.05, seed=2)
    assert cascade.accuracy > single_small.accuracy
    assert cascade.total_cost < single_large.total_cost
    assert cascade.escalations > 0


# --- cost engineering ---------------------------------------------------


def test_cache_breaks_even_within_one_reuse() -> None:
    assert ch26.cache_breakeven(1.25, 0.10) < 1.0
    assert ch26.cached_cost(1, 1.25, 0.10) < 2.0  # write + 1 read beats 2 uncached


def test_batch_saving_and_retry_amplification() -> None:
    assert ch26.batch_saving(4.0, 0.5) == 2.0
    assert abs(ch26.retry_multiplier(0.2) - 1.25) < 1e-9
    assert ch26.goodput(120, 0.9) == 108.0


def test_agent_loop_cap_bounds_the_tail() -> None:
    def worst(cap: int) -> int:
        rng = random.Random(7)
        return max(ch26.simulate_agent_loop(rng, 0.5, 500, cap)[0] for _ in range(2000))

    assert worst(4) < worst(50)  # capping amputates the fat tail
    assert worst(4) <= 500 * (1 + 2 + 3 + 4)


def test_token_bucket_applies_backpressure_under_injected_clock() -> None:
    clock = [0.0]
    bucket = ch26.TokenBucket(100, 10, lambda: clock[0])
    assert bucket.allow(80)
    assert not bucket.allow(30)  # drained
    clock[0] = 1.0
    assert bucket.allow(30)  # refilled


# --- reliability --------------------------------------------------------


def test_failure_classification_and_deadline_propagation() -> None:
    assert ch26.classify_failure("policy_refusal") == "terminal"
    assert ch26.classify_failure("overloaded") == "retryable"
    assert ch26.classify_failure("unknown_write_lost") == "ambiguous"
    assert abs(ch26.propagate_deadline(0.8, 0.2) - 0.6) < 1e-9
    assert ch26.propagate_deadline(0.1, 0.5) == 0.0


def test_retry_budget_exhausts_on_deadline() -> None:
    clock = [0.0]
    budget = ch26.RetryBudget(attempts_left=5, deadline_s=0.5, now=lambda: clock[0])
    clock[0] = 0.25
    assert budget.consume("overloaded") == 0.25  # time remains
    clock[0] = 0.5
    try:
        budget.consume("overloaded")
    except TimeoutError:
        pass
    else:
        raise AssertionError("retry allowed past the deadline")


def test_retry_budget_refuses_terminal_failure() -> None:
    budget = ch26.RetryBudget(attempts_left=5, deadline_s=10.0, now=lambda: 0.0)
    try:
        budget.consume("policy_refusal")
    except RuntimeError:
        pass
    else:
        raise AssertionError("terminal failure was retried")


def test_hedge_fires_only_past_threshold() -> None:
    fast, fired = ch26.hedge_latency(150, 800, threshold_ms=400)
    assert fast == 150 and not fired
    slow, fired = ch26.hedge_latency(1500, 180, threshold_ms=400)
    assert slow == 400 + 180 and fired


# --- effect ledger: idempotency across crashes --------------------------


def test_any_crash_count_yields_exactly_one_effect() -> None:
    """Property: no number of crashes in the window produces two effects."""
    rng = random.Random(2026)
    for _ in range(500):
        crashes = rng.randint(0, 12)
        ledger, provider = ch26.EffectLedger(), ch26.IdempotentProvider()
        key, payload = f"refund:{rng.randint(0, 999)}:v1", {"cents": rng.randint(1, 99999)}
        for _ in range(crashes):
            try:
                ch26.execute_once(ledger, provider, key, payload, crash_after_provider=True)
            except ch26.InjectedCrash:
                pass
        ch26.execute_once(ledger, provider, key, payload)
        assert provider.effect_count == 1


def test_retrying_never_increases_effect_count() -> None:
    """Metamorphic: adding a crash-and-recover cannot add an external effect."""
    clean_ledger, clean = ch26.EffectLedger(), ch26.IdempotentProvider()
    ch26.execute_once(clean_ledger, clean, "refund:A:v1", {"cents": 10})

    crash_ledger, crashed = ch26.EffectLedger(), ch26.IdempotentProvider()
    try:
        ch26.execute_once(crash_ledger, crashed, "refund:A:v1", {"cents": 10}, True)
    except ch26.InjectedCrash:
        pass
    ch26.execute_once(crash_ledger, crashed, "refund:A:v1", {"cents": 10})
    assert crashed.effect_count == clean.effect_count == 1


def test_same_key_rejects_a_substituted_payload() -> None:
    ledger, provider = ch26.EffectLedger(), ch26.IdempotentProvider()
    ch26.execute_once(ledger, provider, "refund:A", {"cents": 10})
    try:
        ch26.execute_once(ledger, provider, "refund:A", {"cents": 20})
    except ValueError:
        pass
    else:
        raise AssertionError("a substituted payload reused an effect identity")
    assert provider.effect_count == 1


# --- durable execution: replay and compensation -------------------------


def test_replay_returns_recorded_value_without_re_executing() -> None:
    calls = {"n": 0}

    def flaky() -> int:
        calls["n"] += 1
        return 100 + calls["n"]

    engine = ch26.WorkflowEngine()
    first = engine.activity("a", flaky)
    replayed = engine.activity("a", flaky)
    assert first == replayed  # replay is stable
    assert calls["n"] == 1  # the function ran once, not twice
    assert engine.executed == ["a"]


def test_crash_resume_charges_the_effect_once() -> None:
    state = {"sends": 0}

    def send() -> str:
        state["sends"] += 1
        if state["sends"] == 1:
            raise ch26.InjectedCrash("died sending")
        return "sent"

    provider = ch26.IdempotentProvider()
    engine = ch26.WorkflowEngine()

    def workflow() -> None:
        engine.activity("charge", lambda: provider.apply("charge:1", {"cents": 50}))
        engine.activity("send", send)

    try:
        workflow()
    except ch26.InjectedCrash:
        pass
    workflow()  # resume on the same journal
    assert provider.effect_count == 1
    assert engine.executed == ["charge", "send"]


def test_saga_compensates_committed_steps_in_reverse() -> None:
    provider = ch26.IdempotentProvider()
    engine = ch26.WorkflowEngine()
    try:
        ch26.book_trip(engine, provider, event_available=False)
    except ch26.EffectRejected:
        pass
    keys = list(provider.receipts)
    assert keys == ["flight:T1", "hotel:T1",
                    "cancel_hotel:hotel:T1", "cancel_flight:flight:T1"]
    assert provider.effect_count == 4


def test_successful_saga_creates_no_compensation() -> None:
    provider = ch26.IdempotentProvider()
    engine = ch26.WorkflowEngine()
    ch26.book_trip(engine, provider, event_available=True)
    assert list(provider.receipts) == ["flight:T1", "hotel:T1", "event:T1"]


# --- versioning ---------------------------------------------------------


def test_bundle_digest_is_order_independent_and_change_sensitive() -> None:
    v42 = {"model": "m1", "prompt": "p1", "corpus": "c1"}
    assert ch26.bundle_digest(v42) == ch26.bundle_digest(dict(reversed(list(v42.items()))))
    v43 = {**v42, "corpus": "c2"}
    assert ch26.bundle_digest(v42) != ch26.bundle_digest(v43)
    assert ch26.manifest_diff(v42, v43) == {"corpus"}


# --- delivery: powered canary gate --------------------------------------


def test_canary_gate_flips_on_evidence() -> None:
    assert ch26.canary_gate(98, 100, 0.90).effect == "PROMOTE"
    assert ch26.canary_gate(24, 25, 0.90).effect == "HOLD"  # same rate, too few
    assert ch26.canary_gate(100, 100, 0.90, critical_failures=1).effect == "HOLD"


def test_wilson_bound_and_detection_math() -> None:
    assert ch26.wilson_lower(98, 100) < 98 / 100  # a lower bound, not the rate
    assert 0.71 < ch26.detection_probability(25, 0.05) < 0.73
    assert ch26.canary_sample_size(0.05, 0.80) == 32
    try:
        ch26.wilson_lower(1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("wilson_lower accepted zero trials")
