# Auto-generated from chapters/26-production-platform.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


@dataclass(frozen=True)
class RequestContext:
    """The trusted identity a journey carries, derived only from a virtual key.

    A gateway builds this from a server-side key lookup, never from a
    model-controlled or client-declared field. Everything downstream —
    budgets, routing scopes, effect authorization — reads identity from here,
    which is why the tenant and scopes are fixed at admission and immutable.
    """

    request_id: str
    tenant: str
    scopes: frozenset[str]
    deadline_s: float


@dataclass(frozen=True)
class Usage:
    """Metered tokens and per-call charges for one model call in a journey.

    The cost of a journey is more than input and output tokens: it includes
    per-call tool charges and, for reviewed journeys, a share of human review.
    Carrying all four terms here is what lets ``breakdown`` show where a
    journey's money actually goes instead of hiding it behind a token count.
    """

    input_tokens: int
    output_tokens: int
    input_per_million: float
    output_per_million: float
    tool_units: float = 0.0
    review_units: float = 0.0

    @property
    def cost(self) -> float:
        """Return the total cost of this call in currency units."""
        tokens = (
            self.input_tokens * self.input_per_million
            + self.output_tokens * self.output_per_million
        ) / 1_000_000
        return tokens + self.tool_units + self.review_units

    def breakdown(self) -> dict[str, float]:
        """Return per-component cost so a journey's spend can be attributed."""
        return {
            "input": self.input_tokens * self.input_per_million / 1_000_000,
            "output": self.output_tokens * self.output_per_million / 1_000_000,
            "tool": self.tool_units,
            "review": self.review_units,
        }


class BudgetError(RuntimeError):
    """Raised when a whole journey has no remaining spend authority."""


@dataclass
class Deployment:
    """A model deployment the gateway may route to, with its cost per call."""

    name: str
    capabilities: frozenset[str]
    cost_per_call: float
    healthy: bool = True


class Gateway:
    """Authenticate, admit, route, and settle one agent journey.

    The gateway is the single enforcement point the Friday incident lacked:
    identity comes from a virtual key, a journey above its tenant budget is
    rejected before any call, routing considers only deployments that satisfy
    the requested capability and are healthy, and settlement charges realized
    usage back to the tenant. It deliberately holds provider credentials so an
    application never does.
    """

    def __init__(
        self,
        keys: dict[str, tuple[str, frozenset[str]]],
        limits: dict[str, float],
        deployments: list[Deployment],
    ) -> None:
        self.keys = keys
        self.limits = limits
        self.deployments = deployments
        self.spend = {tenant: 0.0 for tenant in limits}

    def authenticate(
        self, virtual_key: str, request_id: str, deadline_s: float
    ) -> RequestContext:
        """Map a virtual key to a trusted identity, or refuse an unknown key.

        Args:
            virtual_key: The opaque key presented by the caller.
            request_id: An id used to correlate evidence for this request.
            deadline_s: The absolute time budget the journey starts with.

        Returns:
            The ``RequestContext`` whose tenant and scopes come from the
            server-side key table, never from a client-supplied field.

        Raises:
            PermissionError: If the key is not in the table.
        """
        if virtual_key not in self.keys:
            raise PermissionError("unknown virtual key")
        tenant, scopes = self.keys[virtual_key]
        return RequestContext(request_id, tenant, scopes, deadline_s)

    def admit(self, context: RequestContext, estimated_cost: float) -> None:
        """Reject a journey whose estimate would exceed the tenant budget.

        Args:
            context: The authenticated identity for this journey.
            estimated_cost: The whole-journey estimate to reserve.

        Raises:
            BudgetError: If reserving the estimate would exceed the limit.
        """
        limit = self.limits[context.tenant]
        if self.spend[context.tenant] + estimated_cost > limit:
            raise BudgetError("tenant journey budget exhausted")

    def route(self, context: RequestContext, capability: str) -> Deployment:
        """Select the least-cost healthy deployment satisfying a capability.

        Args:
            context: The authenticated identity (its scopes gate capabilities).
            capability: The logical capability the journey needs, e.g. "chat".

        Returns:
            The cheapest eligible ``Deployment``.

        Raises:
            LookupError: If no healthy deployment offers the capability.
        """
        candidates = [
            d for d in self.deployments if capability in d.capabilities and d.healthy
        ]
        if not candidates:
            raise LookupError("no healthy deployment satisfies the capability")
        return min(candidates, key=lambda d: d.cost_per_call)

    def settle(self, context: RequestContext, usage: Usage) -> float:
        """Charge realized usage to the tenant and return the new total spend."""
        self.spend[context.tenant] += usage.cost
        return self.spend[context.tenant]


@dataclass(frozen=True)
class Backend:
    """A stub model deployment with a fixed cost, quality, and latency profile.

    The backend answers a task correctly when its ``strength`` meets the
    task's difficulty. This deterministic rule stands in for a real model's
    accuracy curve so routing and cascading can be compared and priced without
    a network or an accelerator; only the numbers change with a real model.
    """

    name: str
    strength: float
    cost_per_call: float
    latency_ms: float

    def answer(self, difficulty: float) -> bool:
        """Return whether this backend answers a task of the given difficulty."""
        return self.strength >= difficulty


@dataclass(frozen=True)
class WorkloadResult:
    """Aggregate cost, quality, and latency for one routing strategy."""

    strategy: str
    accuracy: float
    total_cost: float
    mean_latency_ms: float
    escalations: int


def run_single(backend: Backend, difficulties: list[float]) -> WorkloadResult:
    """Serve every task with one backend and aggregate the outcome.

    Args:
        backend: The single deployment used for the whole workload.
        difficulties: One difficulty per task in the workload.

    Returns:
        The strategy's accuracy, total cost, mean latency, and (zero)
        escalations.
    """
    correct = sum(backend.answer(d) for d in difficulties)
    n = len(difficulties)
    return WorkloadResult(
        f"single:{backend.name}", correct / n,
        backend.cost_per_call * n, backend.latency_ms, 0,
    )


def run_router(
    small: Backend, large: Backend, difficulties: list[float],
    noise: float, seed: int,
) -> WorkloadResult:
    """Route each task by a noisy difficulty estimate, then serve it once.

    A cheap classifier estimates difficulty as the true value plus Gaussian
    noise; tasks estimated above the small backend's strength go to the large
    backend. Misestimates are the router's characteristic failure: an easy
    task sent to the large model wastes money, a hard task kept on the small
    model loses quality.

    Args:
        small: The cheap, weaker backend.
        large: The expensive, stronger backend.
        difficulties: One difficulty per task.
        noise: Standard deviation of the classifier's estimation error.
        seed: Seed for the deterministic noise stream.

    Returns:
        The routed workload's aggregate result.
    """
    rng = random.Random(seed)
    correct = cost = latency = escalations = 0
    for d in difficulties:
        estimate = d + rng.gauss(0.0, noise)
        chosen = large if estimate > small.strength else small
        escalations += chosen is large
        correct += chosen.answer(d)
        cost += chosen.cost_per_call
        latency += chosen.latency_ms
    n = len(difficulties)
    return WorkloadResult("router", correct / n, cost, latency / n, escalations)


def run_cascade(
    small: Backend, large: Backend, difficulties: list[float],
    false_accept: float, seed: int,
) -> WorkloadResult:
    """Try the cheap backend first, escalate when a grader rejects its answer.

    Every task pays the small backend. A grader (an oracle, except that it
    wrongly accepts a wrong answer with probability ``false_accept``) decides
    whether to escalate; escalated tasks also pay the large backend and its
    latency. The cascade's promise is large-backend quality at a fraction of
    its cost, bought with higher latency on the escalated tail.

    Args:
        small: The cheap backend tried first.
        large: The expensive backend used on escalation.
        difficulties: One difficulty per task.
        false_accept: Probability the grader accepts a wrong cheap answer.
        seed: Seed for the deterministic grader stream.

    Returns:
        The cascade's aggregate result.
    """
    rng = random.Random(seed)
    correct = cost = latency = escalations = 0
    for d in difficulties:
        cost += small.cost_per_call
        small_ok = small.answer(d)
        accept = small_ok or rng.random() < false_accept
        if accept:
            correct += small_ok
            latency += small.latency_ms
        else:
            escalations += 1
            correct += large.answer(d)
            cost += large.cost_per_call
            latency += small.latency_ms + large.latency_ms
    n = len(difficulties)
    return WorkloadResult("cascade", correct / n, cost, latency / n, escalations)


def cache_breakeven(write_multiplier: float, read_multiplier: float) -> float:
    """Return the minimum reuses for a prefix cache to pay for its write.

    Solves @eq-ch26-cache: a cache whose write costs ``write_multiplier`` times
    an uncached call and whose reads cost ``read_multiplier`` times an uncached
    call breaks even after ``(w - 1) / (1 - d)`` reuses.

    Args:
        write_multiplier: Cost of writing the cache, relative to one call.
        read_multiplier: Cost of a cached read, relative to one call.

    Returns:
        The break-even reuse count (a real number; round up for whole reuses).
    """
    return (write_multiplier - 1) / (1 - read_multiplier)


def cached_cost(reuses: int, write_multiplier: float, read_multiplier: float) -> float:
    """Return the cost of one cache write plus ``reuses`` discounted reads."""
    return write_multiplier + reuses * read_multiplier


def batch_saving(synchronous_cost: float, discount: float) -> float:
    """Return the cost of a deferrable workload run through a batch API.

    Providers commonly price asynchronous, latency-tolerant work — nightly
    evaluations, bulk classification, backfills — at a discount to the
    synchronous rate. Moving that work to a batch queue is often the single
    largest cost lever for an agent platform's offline jobs.

    Args:
        synchronous_cost: What the workload would cost at the live rate.
        discount: Fractional discount the batch tier offers, in [0, 1].

    Returns:
        The batch-tier cost of the same workload.
    """
    return synchronous_cost * (1 - discount)


def retry_multiplier(q: float) -> float:
    """Return the expected attempts when each independently retries with prob q.

    Args:
        q: Per-attempt probability that another attempt is needed, in [0, 1).

    Returns:
        The geometric-series expectation ``1 / (1 - q)``.
    """
    return 1.0 / (1.0 - q)


def goodput(throughput: float, qualifying_fraction: float) -> float:
    """Return goodput: the rate of completions that satisfy the task contract."""
    return throughput * qualifying_fraction


def simulate_agent_loop(
    rng: random.Random, p_success: float, per_turn_tokens: int, max_turns: int
) -> tuple[int, int]:
    """Simulate one retry-until-success journey with a growing transcript.

    Each turn resends the whole transcript, so turn ``t`` costs
    ``per_turn_tokens * t``; the journey ends on the first Bernoulli success or
    when ``max_turns`` caps it. The uncapped tail of this distribution is the
    denial-of-wallet pathology a platform must bound.

    Args:
        rng: Seeded random source, so the simulation is reproducible.
        p_success: Per-turn probability the journey completes.
        per_turn_tokens: Base token cost of the first turn.
        max_turns: Hard cap on turns; the lever that bounds the tail.

    Returns:
        A tuple of (total tokens spent, turns taken).
    """
    total = 0
    for turn in range(1, max_turns + 1):
        total += per_turn_tokens * turn
        if rng.random() < p_success:
            return total, turn
    return total, max_turns


class TokenBucket:
    """Rate-limit a resource with an injected clock, so tests need no real time.

    Tokens refill continuously at ``refill_per_s`` up to ``capacity``; a
    request is admitted only if enough tokens are present, and consuming them
    applies backpressure on the constrained dimension (input tokens, say,
    rather than request count). The clock is injected so drain-and-refill
    behavior is exercised deterministically.
    """

    def __init__(self, capacity: float, refill_per_s: float,
                 now: Callable[[], float]) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.now = now
        self.tokens = capacity
        self.updated_at = now()

    def allow(self, requested: float) -> bool:
        """Admit a request of ``requested`` tokens, refilling for elapsed time.

        Args:
            requested: Tokens the request would consume.

        Returns:
            True and consumes the tokens when enough are available; False and
            consumes nothing otherwise.
        """
        current = self.now()
        elapsed = max(0.0, current - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_s)
        self.updated_at = current
        if requested > self.tokens:
            return False
        self.tokens -= requested
        return True


TERMINAL = frozenset({"policy_refusal", "invalid_request", "auth"})
RETRYABLE = frozenset({"overloaded", "connection_reset", "rate_limited"})


def classify_failure(kind: str) -> str:
    """Classify a failure as retryable, terminal, or ambiguous.

    Args:
        kind: A short failure label from the provider or transport.

    Returns:
        "terminal", "retryable", or "ambiguous"; the last covers writes whose
        outcome is unknown and must be reconciled, not blindly retried.
    """
    if kind in TERMINAL:
        return "terminal"
    if kind in RETRYABLE:
        return "retryable"
    return "ambiguous"


def propagate_deadline(remaining_s: float, reserved_s: float) -> float:
    """Return the timeout a downstream call may use, reserving tail time.

    A service that receives 800 ms remaining must not start a fixed two-second
    timeout: it must pass down what is left minus time reserved for validation,
    effect recording, and the response path.

    Args:
        remaining_s: Time left on the journey's absolute deadline.
        reserved_s: Time to hold back for the caller's own tail work.

    Returns:
        The non-negative budget a downstream call may spend.
    """
    return max(0.0, remaining_s - reserved_s)


@dataclass
class RetryBudget:
    """Bound retries by an attempt count and an absolute deadline together.

    A retryable failure consumes one attempt and is allowed only while both
    attempts and deadline remain; a terminal failure stops immediately.
    Separating the two limits is what makes the budget honest: a fast failure
    loop runs out of attempts, a slow one runs out of time.
    """

    attempts_left: int
    deadline_s: float
    now: Callable[[], float]

    def consume(self, kind: str) -> float:
        """Consume one retry for a retryable failure, or refuse to.

        Args:
            kind: The failure label being handled.

        Returns:
            The time remaining after consuming the attempt.

        Raises:
            RuntimeError: On a terminal failure (do not retry).
            TimeoutError: When attempts or the deadline are exhausted.
        """
        remaining = self.deadline_s - self.now()
        if classify_failure(kind) == "terminal":
            raise RuntimeError("terminal failure; do not retry")
        if self.attempts_left <= 0 or remaining <= 0:
            raise TimeoutError("retry budget exhausted")
        self.attempts_left -= 1
        return remaining


def hedge_latency(first_ms: float, second_ms: float, threshold_ms: float
                  ) -> tuple[float, bool]:
    """Return the effective latency and whether a hedge fired.

    If the first attempt finishes by ``threshold_ms``, its latency stands and
    no hedge fires. Otherwise a second attempt starts at the threshold and the
    request finishes when either returns, so the effective latency is the
    smaller of the first attempt and threshold-plus-second.

    Args:
        first_ms: Latency of the primary attempt.
        second_ms: Latency the hedged attempt would take on its own.
        threshold_ms: Delay before the hedge is issued.

    Returns:
        A tuple of (effective latency in ms, whether the hedge fired).
    """
    if first_ms <= threshold_ms:
        return first_ms, False
    return min(first_ms, threshold_ms + second_ms), True


class EffectState(str, Enum):
    """The durable states of one external effect, in recovery order."""

    INTENDED = "INTENDED"
    RESERVED = "RESERVED"
    EXECUTED = "EXECUTED"
    RECORDED = "RECORDED"
    FAILED = "FAILED"


@dataclass
class EffectRecord:
    """One effect's durable identity, digest, state, and authoritative receipt."""

    key: str
    payload_digest: str
    state: EffectState
    receipt: str | None = None


class EffectLedger:
    """Keep a stable, payload-bound effect identity across crash and replay.

    The ledger is the application-owned record that a checkpoint cannot be: it
    binds a business idempotency key to a payload digest before the provider is
    called, so recovery re-asks by that key and a substituted payload is
    refused. It stores intent, not workflow progress.
    """

    def __init__(self) -> None:
        self.records: dict[str, EffectRecord] = {}

    @staticmethod
    def digest(payload: dict[str, Any]) -> str:
        """Return a canonical SHA-256 digest of an effect payload."""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def reserve(self, key: str, payload: dict[str, Any]) -> EffectRecord:
        """Reserve a stable key for an exact payload, or reject a substitution.

        Args:
            key: The business idempotency key (stable across retries).
            payload: The exact effect payload; its digest is bound to the key.

        Returns:
            The record for this key, moved to RESERVED on first reservation.

        Raises:
            ValueError: If the key already exists for a different payload.
        """
        digest = self.digest(payload)
        record = self.records.get(key)
        if record and record.payload_digest != digest:
            raise ValueError("idempotency key reused for a different payload")
        if record is None:
            record = EffectRecord(key, digest, EffectState.INTENDED)
            self.records[key] = record
        if record.state == EffectState.INTENDED:
            record.state = EffectState.RESERVED
        return record


class InjectedCrash(RuntimeError):
    """Simulate worker death after the provider committed, before recording."""


class IdempotentProvider:
    """A provider that commits at most one effect per stable key.

    It returns the same receipt for a repeated key, which is the property that
    makes recovery safe: a worker that crashes and retries with the same key
    sees the original receipt instead of a second effect. ``effect_count`` is
    the ground truth the tests assert on.
    """

    def __init__(self) -> None:
        self.receipts: dict[str, str] = {}
        self.effect_count = 0

    def apply(self, key: str, payload: dict[str, Any]) -> str:
        """Commit the effect once for a new key, else return its receipt."""
        if key not in self.receipts:
            self.effect_count += 1
            self.receipts[key] = f"receipt-{self.effect_count}"
        return self.receipts[key]


def execute_once(
    ledger: EffectLedger,
    provider: IdempotentProvider,
    key: str,
    payload: dict[str, Any],
    crash_after_provider: bool = False,
) -> str:
    """Execute an effect once, or recover it safely after a crash.

    Recovery is a composition, not magic: a durable reservation (the ledger)
    plus a provider that honors a stable key. If the record is already
    RECORDED, the receipt is returned without touching the provider. Otherwise
    the provider is called with the stable key; a crash in the window between
    the call and recording leaves the record RESERVED, so the next attempt
    re-asks the provider by the same key and still sees one effect.

    Args:
        ledger: The durable effect ledger.
        provider: The idempotent external provider.
        key: The stable business idempotency key.
        payload: The exact effect payload.
        crash_after_provider: If True, raise in the dangerous window to model
            a worker dying before it recorded the receipt.

    Returns:
        The authoritative receipt for this effect.

    Raises:
        InjectedCrash: When ``crash_after_provider`` is set, before recording.
    """
    record = ledger.reserve(key, payload)
    if record.state == EffectState.RECORDED:
        return record.receipt  # type: ignore[return-value]
    try:
        receipt = provider.apply(key, payload)
        if crash_after_provider:
            raise InjectedCrash("died before recording provider receipt")
        record.state = EffectState.EXECUTED
        record.receipt = receipt
        record.state = EffectState.RECORDED
        return receipt
    except InjectedCrash:
        raise
    except Exception:
        record.state = EffectState.FAILED
        raise


@dataclass
class WorkflowEngine:
    """A minimal durable-execution engine: journal activities, replay them.

    A workflow calls :meth:`activity` for every non-deterministic or effectful
    step. The first time an activity id is seen its function runs and the
    result is journaled; on any later pass the journaled result is returned
    without running the function again. That is exactly what lets a crashed
    workflow resume — replay reconstructs past decisions from the journal
    rather than re-executing paid, non-deterministic calls.
    """

    journal: dict[str, Any] = field(default_factory=dict)
    executed: list[str] = field(default_factory=list)

    def activity(self, activity_id: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` once and journal it, or replay the journaled result.

        Args:
            activity_id: Stable id naming this step in the workflow history.
            fn: The step body; executed only the first time this id is seen.

        Returns:
            The recorded result for ``activity_id`` — fresh on first
            execution, replayed verbatim thereafter.
        """
        if activity_id in self.journal:
            return self.journal[activity_id]
        result = fn()
        self.journal[activity_id] = result
        self.executed.append(activity_id)
        return result


class EffectRejected(RuntimeError):
    """A provider declined an effect (e.g. an event is sold out)."""


def book_trip(engine: WorkflowEngine, provider: IdempotentProvider,
              event_available: bool) -> list[str]:
    """Book flight, hotel, and event as a saga; compensate on failure.

    Each booking is a durable activity. If the event booking is rejected, the
    already-committed steps are compensated in reverse order, each compensation
    being a new forward effect with its own stable key — because you cannot
    un-charge a card, only issue a matching reversal.

    Args:
        engine: The durable engine journaling each step.
        provider: The idempotent effect provider.
        event_available: Whether the event booking succeeds.

    Returns:
        The provider receipts, in the order effects were committed (bookings
        first, then any compensations).

    Raises:
        EffectRejected: After compensations run, to signal the saga failed.
    """
    compensations: list[tuple[str, str]] = []
    engine.activity("flight", lambda: provider.apply("flight:T1", {"seat": "12A"}))
    compensations.append(("cancel_flight", "flight:T1"))
    engine.activity("hotel", lambda: provider.apply("hotel:T1", {"nights": 2}))
    compensations.append(("cancel_hotel", "hotel:T1"))
    try:
        if not event_available:
            raise EffectRejected("event sold out")
        engine.activity("event", lambda: provider.apply("event:T1", {"seats": 2}))
    except EffectRejected:
        for name, target in reversed(compensations):
            provider.apply(f"{name}:{target}", {"compensates": target})
        raise
    return list(provider.receipts.values())


def bundle_digest(surface: dict[str, str]) -> str:
    """Return a canonical content hash of a complete release bundle.

    Every correctness-bearing surface — model, tokenizer, prompt, tools,
    corpus, embedder, judge — maps to an immutable id here. Canonical JSON
    makes the digest independent of key order, so the same surfaces always
    hash equal and any single change flips the digest.

    Args:
        surface: A map from surface name to its immutable identifier.

    Returns:
        The hex SHA-256 digest addressing this exact bundle.
    """
    encoded = json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def manifest_diff(old: dict[str, str], new: dict[str, str]) -> set[str]:
    """Return the surfaces whose identifier changed between two bundles."""
    return {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}


def wilson_lower(successes: int, trials: int, z: float = 1.96) -> float:
    """Return a Wilson lower confidence bound for a success rate.

    The Wilson interval is well-behaved near 0 and 1 and at small samples,
    where the naive normal interval misbehaves. A canary uses the lower bound —
    not the point estimate — so a good-looking rate from few trials cannot
    promote a release on luck.

    Args:
        successes: Number of successful canary cases.
        trials: Total canary cases (must be positive).
        z: Normal quantile for the confidence level (1.96 ~ 95%).

    Returns:
        The lower confidence bound on the true success rate.

    Raises:
        ValueError: If ``trials`` is not positive.
    """
    if trials <= 0:
        raise ValueError("trials must be positive")
    rate = successes / trials
    denom = 1 + z * z / trials
    center = rate + z * z / (2 * trials)
    radius = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials ** 2))
    return max(0.0, (center - radius) / denom)


def detection_probability(sample_size: int, regression_rate: float) -> float:
    """Return the chance of seeing at least one independent regression."""
    return 1 - (1 - regression_rate) ** sample_size


def canary_sample_size(regression_rate: float, target_prob: float) -> int:
    """Return the independent cases needed to detect a regression with target_prob.

    Args:
        regression_rate: The extra failure fraction to be able to catch.
        target_prob: The detection probability to reach, in (0, 1).

    Returns:
        The smallest case count meeting the target under the one-hit model.
    """
    return math.ceil(math.log(1 - target_prob) / math.log(1 - regression_rate))


@dataclass(frozen=True)
class CanaryDecision:
    """The outcome of a canary gate: promote or hold, with the reasons why."""

    effect: str
    observed_rate: float
    lower_bound: float
    reasons: tuple[str, ...]


def canary_gate(successes: int, trials: int, minimum_rate: float,
                critical_failures: int = 0) -> CanaryDecision:
    """Decide promotion from an uncertainty bound plus a zero-tolerance rule.

    The candidate promotes only if no critical invariant failed and the Wilson
    lower bound clears the release floor; otherwise it holds. Combining a
    statistical quality test with an absolute safety test is what lets one
    unauthorized transfer block a release that otherwise looks excellent.

    Args:
        successes: Successful canary cases.
        trials: Total canary cases.
        minimum_rate: The quality floor the lower bound must clear.
        critical_failures: Count of zero-tolerance failures observed.

    Returns:
        A ``CanaryDecision`` naming the effect and every holding reason.
    """
    lower = wilson_lower(successes, trials)
    reasons: list[str] = []
    if critical_failures:
        reasons.append("critical invariant failed")
    if lower < minimum_rate:
        reasons.append("quality lower bound misses release floor")
    return CanaryDecision("PROMOTE" if not reasons else "HOLD",
                          successes / trials, lower, tuple(reasons))
