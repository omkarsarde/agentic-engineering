# Auto-generated from appendices/a-bridge.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import hashlib
import json
import math
import random
import sqlite3
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count


def posterior(prior: float, sensitivity: float, false_positive: float) -> float:
    """Return the posterior probability of a condition given a positive test.

    Applies Bayes' rule to a binary detector. The point of the exercise is that
    a rare condition stays rare even after a positive result, because the flood
    of false positives from the large negative population swamps the true ones.

    Args:
        prior: Base rate of the condition before any test.
        sensitivity: P(test positive | condition present).
        false_positive: P(test positive | condition absent).

    Returns:
        P(condition present | test positive), between zero and one.
    """

    hit = sensitivity * prior
    return hit / (hit + false_positive * (1 - prior))


def entropy(dist: list[float]) -> float:
    """Return the Shannon entropy of a distribution, in bits.

    Args:
        dist: Probabilities that sum to one.

    Returns:
        Expected surprise in bits; larger means less predictable.
    """

    return -sum(p * math.log2(p) for p in dist if p > 0)


def kl_divergence(p: list[float], q: list[float]) -> float:
    """Return KL(p || q) in bits, the coding penalty for modelling p as q.

    The two orderings differ because KL punishes putting little q-mass where p
    has mass far more than the reverse; swapping the arguments is not a rounding
    detail but a different question.

    Args:
        p: The true distribution (the expectation is taken under it).
        q: The model distribution being charged for.

    Returns:
        A nonnegative divergence in bits, zero only when p equals q.
    """

    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q) if pi > 0)


def best_expected_loss(p_urgent: float, false_negative: float, false_positive: float) -> float:
    """Return the loss of the better of two actions under asymmetric costs.

    Compares *escalate* (which risks a false positive) against *do not escalate*
    (which risks a false negative) and returns whichever expected loss is
    smaller. Asymmetric costs, not the raw probability, choose the action.

    Args:
        p_urgent: Probability the ticket is genuinely urgent.
        false_negative: Cost of failing to escalate an urgent ticket.
        false_positive: Cost of escalating an ordinary ticket.

    Returns:
        The minimum expected loss across the two actions.
    """

    escalate = (1 - p_urgent) * false_positive
    do_not = p_urgent * false_negative
    return min(escalate, do_not)


def value_of_signal(
    prior_urgent: float,
    sensitivity: float,
    specificity: float,
    false_negative: float,
    false_positive: float,
) -> dict[str, float]:
    """Return the expected loss a single binary observation avoids.

    Splits the world into signal-positive and signal-negative, updates the
    urgency belief in each branch by Bayes' rule, takes the best action within
    each branch, and reports the reduction in expected loss versus deciding now.
    The result is nonnegative: information cannot hurt a decision-maker free to
    ignore it. Compare it against the money, latency, and friction of asking.

    Args:
        prior_urgent: Belief the ticket is urgent before the signal.
        sensitivity: P(signal positive | urgent).
        specificity: P(signal negative | not urgent).
        false_negative: Cost of missing an urgent ticket.
        false_positive: Cost of escalating an ordinary one.

    Returns:
        A dict with the prior loss, P(positive), the two branch posteriors, and
        the value of information itself.
    """

    prior_loss = best_expected_loss(prior_urgent, false_negative, false_positive)
    p_pos = prior_urgent * sensitivity + (1 - prior_urgent) * (1 - specificity)
    post_pos = prior_urgent * sensitivity / p_pos
    post_neg = prior_urgent * (1 - sensitivity) / (1 - p_pos)
    observed = p_pos * best_expected_loss(post_pos, false_negative, false_positive) + (
        1 - p_pos
    ) * best_expected_loss(post_neg, false_negative, false_positive)
    return {
        "prior_loss": prior_loss,
        "p_positive": p_pos,
        "posterior_positive": post_pos,
        "posterior_negative": post_neg,
        "voi": prior_loss - observed,
    }


def brier(p: float, y: int) -> float:
    """Return the Brier score (p - y)^2 for probability p and outcome y."""

    return (p - y) ** 2


def log_loss(p: float, y: int) -> float:
    """Return the log loss for probability p and binary outcome y."""

    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def simulate_routing(n: int, seed: int) -> list[tuple[str, int, int]]:
    """Return synthetic (difficulty, action, outcome) logs with confounding.

    Hard tickets are routed to the new agent far more often than easy ones, so
    difficulty is a common cause of treatment and outcome. The new agent's true
    causal effect is a constant +0.10 on resolution probability at every
    difficulty; the confounding is what a naive comparison will get wrong.

    Args:
        n: Number of logged tickets to generate.
        seed: Seed for the deterministic pseudo-random stream.

    Returns:
        A list of (difficulty, action, outcome) triples.
    """

    rng = random.Random(seed)
    route = {"easy": 0.2, "hard": 0.8}
    resolve = {("easy", 0): 0.8, ("easy", 1): 0.9, ("hard", 0): 0.3, ("hard", 1): 0.4}
    rows = []
    for _ in range(n):
        d = "hard" if rng.random() < 0.5 else "easy"
        a = 1 if rng.random() < route[d] else 0
        y = 1 if rng.random() < resolve[(d, a)] else 0
        rows.append((d, a, y))
    return rows


def _rate(rows: list[tuple[str, int, int]]) -> float:
    return sum(y for *_, y in rows) / len(rows)


def naive_effect(rows: list[tuple[str, int, int]]) -> float:
    """Return P(Y|A=1) - P(Y|A=0), the confounded difference of outcome rates."""

    return _rate([r for r in rows if r[1] == 1]) - _rate([r for r in rows if r[1] == 0])


def backdoor_effect(rows: list[tuple[str, int, int]]) -> float:
    """Return the difficulty-adjusted effect, averaging within each stratum.

    Implements backdoor adjustment: estimate the treatment effect separately
    within each level of the confounder D, then average by the prevalence of D.
    Because D no longer varies within a stratum, it can no longer masquerade as
    the agent's effect.

    Args:
        rows: Logged (difficulty, action, outcome) triples.

    Returns:
        The confounding-adjusted estimate of the causal effect.
    """

    effect = 0.0
    for d in ("easy", "hard"):
        stratum = [r for r in rows if r[0] == d]
        weight = len(stratum) / len(rows)
        treated = [r for r in stratum if r[1] == 1]
        control = [r for r in stratum if r[1] == 0]
        effect += weight * (_rate(treated) - _rate(control))
    return effect


def ips_value(logged: list[tuple[int, float, float]]) -> float:
    """Return the inverse-propensity estimate of an always-treat policy.

    Each logged row is (action, reward, behavior-probability). The target policy
    always takes action 1, so rows where it disagrees drop out and surviving
    rows are up-weighted by the inverse of how likely the logging policy was to
    take that action. Rare actions get large weights, which is exactly why the
    estimator has high variance and can exceed the reward range on small logs.

    Args:
        logged: (action, reward, mu) triples from the behavior policy.

    Returns:
        The IPS estimate of the target policy's mean reward.
    """

    return sum((1.0 if a == 1 else 0.0) / mu * r for a, r, mu in logged) / len(logged)


POINTS = {
    "small/direct": (1.0, 0.72),
    "small/RAG": (1.8, 0.83),
    "medium/direct": (2.8, 0.82),
    "medium/RAG": (3.6, 0.91),
    "large/agent": (7.2, 0.93),
    "large/ensemble": (11.0, 0.935),
}


def pareto_names(points: dict[str, tuple[float, float]]) -> set[str]:
    """Return the configurations no cheaper, no-worse point dominates.

    A point is dominated when some other point costs no more and scores no less,
    strictly beating it on at least one axis. The survivors form the Pareto
    frontier — the only candidates worth weighing preferences over, since a
    dominated point is beaten outright.

    Args:
        points: Name to (cost, quality); lower cost and higher quality are better.

    Returns:
        The set of non-dominated configuration names.
    """

    frontier = set()
    for name, (cost, quality) in points.items():
        dominated = any(
            oc <= cost and oq >= quality and (oc < cost or oq > quality)
            for other, (oc, oq) in points.items()
            if other != name
        )
        if not dominated:
            frontier.add(name)
    return frontier


GRID = ["S....", ".###.", ".#...", ".#.#.", "...#G"]


def _cell(mark: str) -> tuple[int, int]:
    return next((r, c) for r, row in enumerate(GRID) for c, ch in enumerate(row) if ch == mark)


def _neighbors(cell: tuple[int, int]):
    r, c = cell
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(GRID) and 0 <= nc < len(GRID[0]) and GRID[nr][nc] != "#":
            yield (nr, nc)


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return the grid (L1) distance, an admissible heuristic for 4-connected moves."""

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(heuristic) -> tuple[list, list]:
    """Return the optimal grid path and the cells expanded to find it.

    Expands the frontier node of least f = g + h until the goal is reached, then
    reconstructs the path from the recorded predecessors. The heuristic argument
    lets us compare an informed search (Manhattan distance) against uninformed
    search (a zero heuristic, which is Dijkstra) by counting expansions.

    Args:
        heuristic: A function from (cell, goal) to an optimistic cost estimate.

    Returns:
        A (path, expanded) pair: the start-to-goal cells and the expansion order.
    """

    start, goal = _cell("S"), _cell("G")
    tie = count()
    frontier = [(heuristic(start, goal), next(tie), start)]
    g_cost = {start: 0}
    came: dict[tuple[int, int], tuple[int, int]] = {}
    expanded: list[tuple[int, int]] = []
    while frontier:
        _, _, cell = heappop(frontier)
        if cell in expanded:
            continue
        expanded.append(cell)
        if cell == goal:
            break
        for nb in _neighbors(cell):
            step = g_cost[cell] + 1
            if step < g_cost.get(nb, math.inf):
                g_cost[nb] = step
                came[nb] = cell
                heappush(frontier, (step + heuristic(nb, goal), next(tie), nb))
    path = [goal]
    while path[-1] != start:
        path.append(came[path[-1]])
    return path[::-1], expanded


def value_iteration(sweeps: int, gamma: float = 0.9) -> list[list[float]]:
    """Return the value estimate after each sweep on a three-state reward chain.

    A minimal MDP: states 0 and 1 move deterministically toward an absorbing
    goal state 2 that pays one unit. Each Bellman backup sets a state's value to
    its reward plus the discounted value of its successor, so the goal's reward
    propagates one step backward per sweep — the mechanism behind every dynamic
    program in RL, shown at a scale you can check by hand.

    Args:
        sweeps: Number of full backups to perform.
        gamma: Discount factor in [0, 1).

    Returns:
        The list of value vectors, one per sweep.
    """

    reward = {0: 0.0, 1: 0.0, 2: 1.0}
    values = {0: 0.0, 1: 0.0, 2: 0.0}
    history = []
    for _ in range(sweeps):
        updated = dict(values)
        for s in (0, 1, 2):
            updated[s] = reward[s] + (0.0 if s == 2 else gamma * values[s + 1])
        values = updated
        history.append([round(values[s], 3) for s in (0, 1, 2)])
    return history


def clarifying_info_gain(prior_a: float, p_yes_given_a: float, p_yes_given_b: float) -> dict[str, float]:
    """Return the expected information gain, in bits, of a clarifying question.

    Models a binary latent intent and a binary answer. Computes the prior
    entropy, the belief after each possible answer, and the entropy remaining on
    average; the reduction is the mutual information between question and intent
    — the information-theoretic value of asking, complementary to the utility
    VOI of the decision-theory section.

    Args:
        prior_a: Prior probability of the first intent.
        p_yes_given_a: P(answer yes | first intent).
        p_yes_given_b: P(answer yes | second intent).

    Returns:
        A dict with P(yes), the two posteriors on the first intent, and the gain.
    """

    def h(p: float) -> float:
        return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

    p_yes = prior_a * p_yes_given_a + (1 - prior_a) * p_yes_given_b
    post_yes = prior_a * p_yes_given_a / p_yes
    post_no = prior_a * (1 - p_yes_given_a) / (1 - p_yes)
    expected_post = p_yes * h(post_yes) + (1 - p_yes) * h(post_no)
    return {
        "p_yes": p_yes,
        "posterior_if_yes": post_yes,
        "posterior_if_no": post_no,
        "info_gain_bits": h(prior_a) - expected_post,
    }


def bandit_regret(strategy: str, seed: int, rounds: int = 300) -> float:
    """Return cumulative regret for a two-armed bandit under a given strategy.

    Two arms pay off with fixed probabilities; the better arm is unknown. The
    ``ucb`` strategy adds an exploration bonus that shrinks as an arm is pulled,
    so it keeps sampling the uncertain arm; the ``greedy`` strategy commits to
    whichever arm looked best after one pull each and can lock onto the loser.
    Regret is the reward given up versus always pulling the best arm.

    Args:
        strategy: Either ``"ucb"`` or ``"greedy"``.
        seed: Seed for the deterministic reward stream.
        rounds: Number of pulls.

    Returns:
        Total regret accumulated over all rounds.
    """

    rng = random.Random(seed)
    means = [0.3, 0.6]
    counts, sums, regret = [0, 0], [0.0, 0.0], 0.0
    for t in range(1, rounds + 1):
        if min(counts) == 0:
            arm = counts.index(0)
        elif strategy == "ucb":
            scores = [sums[i] / counts[i] + math.sqrt(2 * math.log(t) / counts[i]) for i in (0, 1)]
            arm = scores.index(max(scores))
        else:
            arm = 0 if sums[0] / counts[0] >= sums[1] / counts[1] else 1
        reward = 1.0 if rng.random() < means[arm] else 0.0
        counts[arm] += 1
        sums[arm] += reward
        regret += max(means) - means[arm]
    return regret


def error_dynamics(gain: float, delay: int, steps: int = 40, start: float = 1.0) -> list[float]:
    """Return the error trajectory of a proportional controller with lag.

    Each step corrects the error using a correction proportional to a possibly
    stale observation of it, e[t+1] = e[t] - gain * e[t-delay]. With no delay
    and moderate gain the error decays; add delay, or push the gain up, and the
    correction fights yesterday's error, producing sustained or growing
    oscillation — the control-theoretic shape of a retry storm.

    Args:
        gain: Proportional correction strength K.
        delay: Observation staleness in steps.
        steps: Number of steps to simulate.
        start: Initial error.

    Returns:
        The error at each step, starting from ``start``.
    """

    history = [start]
    for t in range(steps):
        observed = history[t - delay] if t - delay >= 0 else start
        history.append(history[-1] - gain * observed)
    return history


class InjectedCrash(RuntimeError):
    """Raised once, after an external effect and before the local commit."""


@dataclass(frozen=True)
class Task:
    """A notification intent carried by an at-least-once queue."""

    payment_id: str
    recipient: str

    @property
    def key(self) -> str:
        """Return an idempotency key derived from intent, not delivery attempt.

        Hashing the canonical intent fields means every redelivery of the same
        notification shares a key, so deduplication can recognize it; a key tied
        to the delivery instead would make each retry look like new work.
        """

        payload = json.dumps(
            {"payment_id": self.payment_id, "recipient": self.recipient}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class CountingProvider:
    """An external effect that counts calls and, optionally, deduplicates them."""

    def __init__(self, deduplicate: bool = False) -> None:
        self.deduplicate = deduplicate
        self.calls = 0
        self.effects = 0
        self.seen: set[str] = set()

    def send(self, task: Task) -> None:
        """Record one call and one externally visible effect, unless deduplicating."""

        self.calls += 1
        if self.deduplicate and task.key in self.seen:
            return
        self.seen.add(task.key)
        self.effects += 1


class Ledger:
    """A local record of pending and committed intent keys.

    The reserve-then-commit shape is the crux of the lab: an intent becomes
    pending before the effect and committed only after, so a crash between the
    two leaves a pending row that cannot, by itself, say whether the external
    effect happened.
    """

    def __init__(self) -> None:
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute("CREATE TABLE effects (key TEXT PRIMARY KEY, status TEXT NOT NULL)")

    def status(self, key: str) -> str | None:
        """Return the recorded status for an intent key, or None if unseen."""

        row = self.connection.execute("SELECT status FROM effects WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def reserve(self, key: str) -> bool:
        """Claim an intent as pending; refuse if it is already committed.

        Returns:
            True if the caller may proceed, False if the work is already done.
        """

        if self.status(key) == "committed":
            return False
        self.connection.execute(
            "INSERT OR IGNORE INTO effects(key, status) VALUES (?, 'pending')", (key,)
        )
        return True

    def commit(self, key: str) -> None:
        """Mark a reserved intent committed."""

        self.connection.execute("UPDATE effects SET status = 'committed' WHERE key = ?", (key,))


def fixture_deliveries(task_count: int = 20) -> list[Task]:
    """Return an at-least-once stream: unique tasks with a duplicate every fifth."""

    unique = [Task(f"payment-{i:02d}", f"user-{i:02d}@example.test") for i in range(task_count)]
    stream: list[Task] = []
    for i, task in enumerate(unique, start=1):
        stream.append(task)
        if i % 5 == 0:
            stream.append(task)
    return stream


def run_naive(deliveries: list[Task], provider: CountingProvider) -> None:
    """Execute every delivery with no ledger — the un-deduplicated baseline."""

    for task in deliveries:
        provider.send(task)


class IdempotentWorker:
    """Reserve, perform, then commit — exposing the effect-before-commit window."""

    def __init__(self, ledger: Ledger, provider: CountingProvider) -> None:
        self.ledger = ledger
        self.provider = provider
        self.crashed: set[str] = set()

    def process(self, task: Task, crash_after_effect_for: str | None = None) -> str:
        """Process one delivery, optionally crashing once in the ambiguous window.

        Reserves the intent, performs the external effect, and commits — but if
        asked, raises exactly once *after* the effect and *before* the commit,
        reproducing the partial failure that a local ledger alone cannot resolve.

        Args:
            task: The delivered intent.
            crash_after_effect_for: Payment id to crash on once, or None.

        Returns:
            ``"committed"`` for new work, ``"duplicate:committed"`` for a skip.
        """

        if not self.ledger.reserve(task.key):
            return "duplicate:committed"
        self.provider.send(task)
        if crash_after_effect_for == task.payment_id and task.key not in self.crashed:
            self.crashed.add(task.key)
            raise InjectedCrash(task.payment_id)
        self.ledger.commit(task.key)
        return "committed"

    def drain(self, deliveries: list[Task], crash_after_effect_for: str | None = None) -> int:
        """Drain the stream, retrying crashed intents, and return the crash count."""

        crashes, retry = 0, []
        for task in deliveries:
            try:
                self.process(task, crash_after_effect_for)
            except InjectedCrash:
                crashes += 1
                retry.append(task)
        for task in retry:
            self.process(task, crash_after_effect_for)
        return crashes


def build_report() -> dict[str, object]:
    """Run the four break-it drills and return their call and effect counts.

    Runs the same twenty-task, four-duplicate stream through the naive worker,
    the ledgered worker with no crash, the ledgered worker crashed once in the
    ambiguous window, and the crashed worker against a deduplicating provider.
    The counts are the lab's whole argument, so they are returned rather than
    printed.

    Returns:
        A nested dict of calls and effects for each of the four drills.
    """

    deliveries = fixture_deliveries()
    naive = CountingProvider()
    run_naive(deliveries, naive)

    local = CountingProvider()
    IdempotentWorker(Ledger(), local).drain(deliveries)

    ambiguous = CountingProvider()
    ambiguous_crashes = IdempotentWorker(Ledger(), ambiguous).drain(deliveries, "payment-07")

    dedup = CountingProvider(deduplicate=True)
    dedup_crashes = IdempotentWorker(Ledger(), dedup).drain(deliveries, "payment-07")

    return {
        "unique_tasks": 20,
        "deliveries": len(deliveries),
        "naive": {"calls": naive.calls, "effects": naive.effects},
        "local_ledger": {"calls": local.calls, "effects": local.effects},
        "ambiguous_window": {"crashes": ambiguous_crashes, "calls": ambiguous.calls, "effects": ambiguous.effects},
        "provider_key": {"crashes": dedup_crashes, "calls": dedup.calls, "effects": dedup.effects},
    }
