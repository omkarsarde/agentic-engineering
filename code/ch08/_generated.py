# Auto-generated from chapters/08-reasoning-test-time-compute.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import math
import random
from dataclasses import dataclass


def apply_op(value: int, op: str, operand: int) -> int:
    """Apply one arithmetic step; the whole task is chains of these."""
    return value + operand if op == "+" else value - operand


@dataclass(frozen=True)
class Problem:
    """A multi-step arithmetic problem carrying its own ground truth.

    The problem is a start value followed by a sequence of ``(op, operand)``
    steps. Because we can compute the gold running value after every step,
    the task supplies both an *outcome* check (does the final answer match?)
    and a *process* check (is each intermediate value right?) — the two
    verifier granularities the chapter contrasts.

    Attributes:
        start: The initial integer the chain operates on.
        ops: The ordered arithmetic steps, e.g. ``(("+", 5), ("-", 2))``.
    """

    start: int
    ops: tuple[tuple[str, int], ...]

    @property
    def gold_trace(self) -> tuple[int, ...]:
        """The correct running value after 0, 1, ... , n steps."""
        value, trace = self.start, [self.start]
        for op, operand in self.ops:
            value = apply_op(value, op, operand)
            trace.append(value)
        return tuple(trace)

    @property
    def gold(self) -> int:
        """The correct final answer (the last running value)."""
        return self.gold_trace[-1]


def make_problem(steps: int, rng: random.Random) -> Problem:
    """Draw a random ``steps``-long problem with single-digit operands."""
    ops = tuple((rng.choice("+-"), rng.randint(1, 9)) for _ in range(steps))
    return Problem(rng.randint(1, 9), ops)


SYSTEMATIC_DELTA = -1  # the off-by-one misconception many attempts share


def sample_trace(problem: Problem, reliability: float, trap: float,
                 rng: random.Random) -> list[tuple[int, bool]]:
    """One stochastic attempt at a problem, step by step.

    At each step the solver writes the correct running value with
    probability ``reliability``. Otherwise it slips: with conditional
    probability ``trap`` it commits the shared systematic off-by-one, else a
    diffuse random error. A slip corrupts the running value and every later
    step inherits it, so per-attempt success probability is ``reliability``
    raised to the number of steps — the knob the rest of the chapter turns.

    Args:
        problem: The problem to attempt.
        reliability: Per-step probability of computing the step correctly.
        trap: Given a slip, the probability it is the shared systematic error.
        rng: Seeded generator; the only source of randomness.

    Returns:
        One ``(value, is_step_correct)`` pair per step; the final value is
        the attempt's answer.
    """
    gold = problem.gold_trace
    value, steps, slipped = problem.start, [], False
    for i, (op, operand) in enumerate(problem.ops):
        if not slipped and rng.random() < reliability:
            value = apply_op(value, op, operand)
        elif not slipped:
            slipped = True
            slip = SYSTEMATIC_DELTA if rng.random() < trap else rng.choice([-3, -2, 2, 3, 4])
            value = apply_op(value, op, operand) + slip
        else:
            value = apply_op(value, op, operand)
        steps.append((value, value == gold[i + 1]))
    return steps


def final_answer(steps: list[tuple[int, bool]]) -> int:
    """Read the attempt's final answer off its trace."""
    return steps[-1][0]


def first_error(steps: list[tuple[int, bool]]) -> int | None:
    """Locate where an attempt first goes wrong — the point a PRM prunes at.

    Because a slip corrupts every later step, the first wrong index marks the
    boundary between an attempt's verified-correct prefix and its doomed
    suffix. A process reward model rejects a branch at exactly this index, and
    verifier-informed revision keeps everything before it and re-samples the
    rest.

    Args:
        steps: An attempt's ``(value, is_step_correct)`` pairs, as returned by
            ``sample_trace``.

    Returns:
        The index of the first incorrect step, or ``None`` when every step is
        correct (the attempt succeeds).
    """
    for i, (_, ok) in enumerate(steps):
        if not ok:
            return i
    return None


DIFFICULTY = {"easy": (2, 0.93), "medium": (4, 0.80), "hard": (8, 0.72)}


def measure_p(steps: int, reliability: float, trap: float = 0.5,
              trials: int = 4000, seed: int = 1) -> float:
    """Empirical per-attempt success rate of the solver on fresh problems.

    Args:
        steps: Problem length (difficulty).
        reliability: Per-step correctness probability.
        trap: Systematic-slip fraction (does not affect the success rate).
        trials: Number of independent problems.
        seed: RNG seed.

    Returns:
        The fraction of single attempts the exact verifier accepts.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        prob = make_problem(steps, rng)
        hits += final_answer(sample_trace(prob, reliability, trap, rng)) == prob.gold
    return hits / trials


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from a pool of n attempts with c correct.

    Rather than sub-sampling k attempts and checking coverage (high
    variance), this computes the exact probability that a random size-k
    subset of the pool contains a correct attempt: one minus the chance of
    drawing k attempts all from the (n - c) wrong ones.

    Args:
        n: Pool size (attempts actually drawn).
        c: Number of correct attempts in the pool.
        k: Sample budget to estimate coverage for (k <= n).

    Returns:
        The unbiased pass@k estimate for one problem.
    """
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def coverage_curve(steps: int, reliability: float, trap: float = 0.5,
                   problems: int = 800, pool: int = 64,
                   seed: int = 7) -> dict[int, float]:
    """Average pass@k over many problems for a fixed difficulty.

    Draws a pool of ``pool`` attempts per problem once and reads pass@k off
    it for every k, so the whole curve costs one pass of sampling.

    Args:
        steps: Problem length.
        reliability: Per-step correctness probability.
        trap: Systematic-slip fraction.
        problems: Number of problems to average over.
        pool: Attempts drawn per problem.
        seed: RNG seed.

    Returns:
        A mapping from sample budget k to dataset pass@k.
    """
    rng = random.Random(seed)
    ks = [1, 2, 4, 8, 16, 32, 64]
    totals = {k: 0.0 for k in ks}
    for _ in range(problems):
        prob = make_problem(steps, rng)
        c = sum(final_answer(sample_trace(prob, reliability, trap, rng)) == prob.gold
                for _ in range(pool))
        for k in ks:
            totals[k] += pass_at_k(pool, c, k)
    return {k: totals[k] / problems for k in ks}


def sample_direct(problem: Problem, p_direct: float, trap: float,
                  rng: random.Random) -> int:
    """A one-shot answer with no intermediate computation.

    Models direct prompting: the solver commits a final answer with a low
    success probability ``p_direct`` and otherwise returns a systematic or
    diffuse wrong answer — the same error structure as the step solver, but
    without the serial computation that a chain of thought buys. The gap
    between this and ``sample_trace`` is precisely the probability mass that
    emitting intermediate steps recovers.

    Args:
        problem: The problem whose gold answer is the target.
        p_direct: Probability the one-shot answer is correct; lower than the
            step solver's pass rate because no scratch computation is done.
        trap: Given a miss, the probability the wrong answer is the shared
            systematic off-by-one rather than a diffuse error.
        rng: Seeded generator; the only source of randomness.

    Returns:
        A single final answer — the gold value on success, otherwise a
        systematic or diffuse wrong value.
    """
    if rng.random() < p_direct:
        return problem.gold
    return problem.gold + (SYSTEMATIC_DELTA if rng.random() < trap
                           else rng.choice([-3, -2, 2, 3, 4]))


def cot_family(steps: int, reliability: float, trap: float = 0.5,
               p_direct: float = 0.22, problems: int = 3000,
               seed: int = 5) -> tuple[float, float, dict[int, float]]:
    """Compare direct, chain-of-thought, and self-consistency accuracy.

    Returns:
        ``(direct_pass@1, cot_pass@1, {k: self_consistency@k})`` where
        self-consistency@k takes the plurality answer over k CoT traces.
    """
    rng = random.Random(seed)
    direct_hits = cot_hits = 0
    sc_hits = {k: 0 for k in (1, 5, 21)}
    for _ in range(problems):
        prob = make_problem(steps, rng)
        direct_hits += sample_direct(prob, p_direct, trap, rng) == prob.gold
        cot_hits += final_answer(sample_trace(prob, reliability, trap, rng)) == prob.gold
        answers = [final_answer(sample_trace(prob, reliability, trap, rng)) for _ in range(21)]
        for k in sc_hits:
            votes = answers[:k]
            sc_hits[k] += max(set(votes), key=votes.count) == prob.gold
    return (direct_hits / problems, cot_hits / problems,
            {k: v / problems for k, v in sc_hits.items()})


def selection_curve(steps: int, reliability: float, trap: float = 0.5,
                    problems: int = 1500, pool: int = 64, seed: int = 9,
                    q: float = 0.15) -> tuple[dict, dict, dict]:
    """Coverage vs plurality vs noisy best-of-n on the same sampled pools.

    The verifier is binary with symmetric error ``q``: it marks a correct
    attempt positive with probability ``1 - q`` and a wrong attempt positive
    with probability ``q``. Best-of-n returns a uniformly random marked
    attempt (or the first attempt if none is marked).

    Args:
        steps, reliability, trap: Solver difficulty parameters.
        problems: Problems to average over.
        pool: Attempts per problem.
        seed: RNG seed.
        q: Symmetric verifier error rate.

    Returns:
        Three ``{k: accuracy}`` maps: coverage, plurality, noisy best-of-n.
    """
    rng = random.Random(seed)
    ks = [1, 2, 4, 8, 16, 32, 64]
    cov = {k: 0.0 for k in ks}
    plur = {k: 0.0 for k in ks}
    bon = {k: 0.0 for k in ks}
    for _ in range(problems):
        prob = make_problem(steps, rng)
        answers, correct = [], []
        for _ in range(pool):
            a = final_answer(sample_trace(prob, reliability, trap, rng))
            answers.append(a)
            correct.append(a == prob.gold)
        marks = [(rng.random() > q) if ok else (rng.random() < q) for ok in correct]
        for k in ks:
            av, cv, mv = answers[:k], correct[:k], marks[:k]
            cov[k] += any(cv)
            plur[k] += max(set(av), key=av.count) == prob.gold
            passed = [i for i in range(k) if mv[i]]
            bon[k] += cv[rng.choice(passed) if passed else 0]
    n = problems
    return ({k: cov[k] / n for k in ks}, {k: plur[k] / n for k in ks},
            {k: bon[k] / n for k in ks})


def precision_ceiling(p: float, q: float) -> float:
    """Precision among candidates a symmetric-error-q verifier marks positive.

    As the pool grows, some marked candidate almost surely exists, but a
    uniformly chosen marked candidate is correct only with this precision,
    not with probability one — the verifier's false positives cap best-of-n.
    This is the Bayesian posterior that a marked candidate is truly correct,
    given base rate ``p`` and a check that fires with the wrong sign a
    fraction ``q`` of the time.

    Args:
        p: Per-attempt success probability — the base rate of correct
            candidates in the pool.
        q: The verifier's symmetric error rate: it marks a wrong attempt
            positive with probability ``q`` and misses a correct one with the
            same probability.

    Returns:
        The precision ceiling pi_v of @eq-ch08-piv — the fraction of marked
        candidates that are actually correct, which best-of-n approaches from
        below and cannot exceed no matter how many samples are drawn.
    """
    return p * (1 - q) / (p * (1 - q) + (1 - p) * q)


def beam_search(problem: Problem, reliability: float, trap: float,
                rng: random.Random, beam: int = 4, fanout: int = 4,
                prm_noise: float = 0.1) -> tuple[int, int]:
    """Beam search over reasoning steps scored by a noisy process checker.

    At each step every prefix is expanded into ``fanout`` candidate
    continuations (drawn from the same slip model as the solver); a noisy
    process checker scores each by whether its running value matches gold
    (flipped with probability ``prm_noise``); the top ``beam`` survive. This
    is the search analogue of coverage: many prefixes are kept alive so a
    correct path survives even when any single attempt would slip.

    Args:
        problem: The problem to solve.
        reliability, trap: Solver slip parameters for expansions.
        rng: Seeded generator.
        beam: Prefixes retained per depth.
        fanout: Candidate continuations per prefix.
        prm_noise: Probability the process checker mislabels a step.

    Returns:
        ``(final_answer, total_expansions)``.
    """
    gold = problem.gold_trace
    frontier = [(problem.start,)]
    expansions = 0
    for op, operand in problem.ops:
        candidates = []
        for prefix in frontier:
            for _ in range(fanout):
                expansions += 1
                if rng.random() < reliability:
                    nv = apply_op(prefix[-1], op, operand)
                else:
                    slip = SYSTEMATIC_DELTA if rng.random() < trap else rng.choice([-3, -2, 2, 3, 4])
                    nv = apply_op(prefix[-1], op, operand) + slip
                candidates.append(prefix + (nv,))

        def score(cand: tuple[int, ...]) -> float:
            correct = cand[-1] == gold[len(cand) - 1]
            return float(correct if rng.random() > prm_noise else not correct)

        candidates.sort(key=score, reverse=True)
        frontier = candidates[:beam]
    return frontier[0][-1], expansions


def compare_search(steps: int, reliability: float, trap: float = 0.5,
                   problems: int = 1500, seed: int = 13) -> dict[str, float]:
    """Greedy vs beam vs equal-budget independent sampling.

    Independent sampling gets the same number of step expansions the beam
    used and *oracle* final-answer selection (accept the first correct
    attempt) — a generous baseline that isolates whether search's structural
    reuse of verified prefixes beats spending the same compute on whole
    independent attempts.

    Returns:
        Accuracies and per-problem expansion counts for all three methods.
    """
    rng = random.Random(seed)
    greedy = beam = indep = 0
    beam_exp = indep_exp = 0
    for _ in range(problems):
        prob = make_problem(steps, rng)
        greedy += final_answer(sample_trace(prob, reliability, trap, rng)) == prob.gold
        ans, exp = beam_search(prob, reliability, trap, rng)
        beam += ans == prob.gold
        beam_exp += exp
        n_indep = max(1, exp // steps)
        hit = False
        for _ in range(n_indep):
            if final_answer(sample_trace(prob, reliability, trap, rng)) == prob.gold:
                hit = True
                break
        indep += hit
        indep_exp += n_indep * steps
    n = problems
    return {"greedy": greedy / n, "beam": beam / n, "indep": indep / n,
            "beam_exp": beam_exp / n, "indep_exp": indep_exp / n}


def prm_vs_orm_checks(steps: int, reliability: float, trap: float = 0.5,
                      problems: int = 3000, seed: int = 17) -> dict[str, float]:
    """How early can a process check reject a doomed attempt?

    Compares checks spent per problem: an ORM always spends one check at the
    end; a PRM checks step by step and stops at the first error. Because a
    slip dooms every later step, stopping there wastes no computation on a
    branch already known to be wrong — the whole economic case for process
    rewards.

    Args:
        steps: Problem length — the number of step-checks an exhaustive PRM
            spends on an attempt that never errs.
        reliability: Per-step correctness probability of the solver.
        trap: Systematic-slip fraction (does not move where errors land).
        problems: Number of independent problems to average over.
        seed: RNG seed.

    Returns:
        A dict with ``wrong`` (attempts that erred), ``caught`` (wrong
        attempts flagged at their first error), ``avg_prm_checks`` (mean
        step-checks spent per problem, well below ``steps`` because doomed
        branches die early), and ``steps`` (the ORM's implicit full-length
        cost, for comparison).
    """
    rng = random.Random(seed)
    wrong = caught = prm_checks = 0
    for _ in range(problems):
        prob = make_problem(steps, rng)
        tr = sample_trace(prob, reliability, trap, rng)
        fe = first_error(tr)
        if fe is None:
            prm_checks += steps
        else:
            wrong += 1
            caught += 1
            prm_checks += fe + 1
    return {"wrong": wrong, "caught": caught,
            "avg_prm_checks": prm_checks / problems, "steps": steps}


def self_correction(steps: int, reliability: float, trap: float = 0.5,
                    problems: int = 4000, seed: int = 21) -> tuple[dict, dict]:
    """Intrinsic vs verifier-informed revision, counting both directions.

    Intrinsic revision re-attempts the problem with no new evidence.
    Verifier-informed revision keeps the verified-correct prefix (up to the
    first error located by a reliable process check) and re-samples the rest.
    Both report the fraction correct before and after and the counts of
    right-to-wrong and wrong-to-right flips — a net score can hide a large
    volume of destroyed-correct answers.

    Returns:
        Two dicts (intrinsic, verifier-informed) with ``before``, ``after``,
        ``r2w`` (right-to-wrong flips), and ``w2r`` (wrong-to-right flips).
    """
    rng = random.Random(seed)
    intr = {"r2w": 0, "w2r": 0, "before": 0, "after": 0}
    vinf = {"r2w": 0, "w2r": 0, "before": 0, "after": 0}
    for _ in range(problems):
        prob = make_problem(steps, rng)
        tr = sample_trace(prob, reliability, trap, rng)
        ok0 = final_answer(tr) == prob.gold

        ok_i = final_answer(sample_trace(prob, reliability, trap, rng)) == prob.gold
        intr["before"] += ok0; intr["after"] += ok_i
        intr["r2w"] += ok0 and not ok_i
        intr["w2r"] += (not ok0) and ok_i

        fe = first_error(tr)
        if fe is None:
            ok_v = True
        else:
            sub = Problem(prob.gold_trace[fe], prob.ops[fe:])
            ok_v = final_answer(sample_trace(sub, reliability, trap, rng)) == prob.gold
        vinf["before"] += ok0; vinf["after"] += ok_v
        vinf["r2w"] += ok0 and not ok_v
        vinf["w2r"] += (not ok0) and ok_v
    n = problems
    scale = lambda d: {k: (v / n if k in ("before", "after") else v) for k, v in d.items()}
    return scale(intr), scale(vinf)


def knee(curve: dict[int, float], threshold: float = 0.03) -> int:
    """First budget whose next doubling gains less than ``threshold``.

    A cheap stand-in for the compute-optimal stopping point: extra samples
    are worth spending while the coverage slope stays above the product's
    minimum acceptable gain, and not after. Because the knee sits at a
    different budget for each difficulty, one fixed sample count cannot be
    optimal across a mixed workload.

    Args:
        curve: A ``{sample budget: coverage}`` mapping, e.g. from
            ``coverage_curve``; its keys are read in increasing order.
        threshold: The smallest per-doubling coverage gain still worth its
            compute — the product's minimum acceptable marginal value.

    Returns:
        The smallest budget k whose next doubling gains less than
        ``threshold``, or the largest budget in the curve when the slope
        never drops that low (coverage still climbing at the sweep's end).
    """
    ks = sorted(curve)
    for i in range(len(ks) - 1):
        if curve[ks[i + 1]] - curve[ks[i]] < threshold:
            return ks[i]
    return ks[-1]


def softmax(logits: list[float]) -> list[float]:
    """Turn logits into a probability distribution (max-shifted for stability).

    Subtracting the largest logit before exponentiating leaves the result
    unchanged but keeps ``exp`` from overflowing — the standard numerically
    safe softmax. This is the map from the RL policy's five logits to the
    strategy probabilities every training step samples from.

    Args:
        logits: Unnormalized scores, one per choice (here, per strategy).

    Returns:
        Probabilities in the same order, each in [0, 1] and summing to one.
    """
    peak = max(logits)
    weights = [math.exp(v - peak) for v in logits]
    total = sum(weights)
    return [w / total for w in weights]


def grpo_advantages(rewards: list[float]) -> list[float]:
    """Group-relative, population-standardized advantages (@eq-ch08-grpo).

    Centers each reward on the group mean and scales by the group standard
    deviation, so the group is its own baseline — no critic. A zero-variance
    group (every sibling tied) yields all-zero advantages: a prompt where the
    policy already agrees with itself teaches nothing.

    Args:
        rewards: One reward per group member.

    Returns:
        Advantages in the same order, summing to zero. The ``eps`` in the
        denominator matches @sec-ch23 and keeps a tied group at zero rather
        than dividing a rounding residual by a near-zero standard deviation.
    """
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    return [(r - mean) / (std + 1e-8) for r in rewards]


STEPS_RLVR = 4
# (name, per-step reliability, response length in tokens); accuracy = reliability**4
STRATEGIES = [
    ("concise-correct", 0.9740, 4),
    ("verbose-correct", 0.9740, 14),
    ("off-by-one", 0.5623, 6),
    ("diffuse-wrong", 0.5623, 6),
    ("verbose-hack", 0.4729, 30),
]
STRAT_ACC = [r ** STEPS_RLVR for _, r, _ in STRATEGIES]
STRAT_TOK = [t for _, _, t in STRATEGIES]


def rollout_reward(action: int, mode: str, rng: random.Random) -> float:
    """Run the solver under a sampled strategy and score it with the verifier.

    ``mode='exact'`` returns the pure outcome-verifier reward (1 if the exact
    answer matches, else 0). ``mode='proxy'`` adds a length bonus — a reward
    with a blind spot the optimizer can exploit. The correctness term is
    always the verifier executed on a freshly sampled candidate, so RLVR's
    reward is earned by real generated work rather than looked up.

    Args:
        action: Index into ``STRATEGIES`` — the strategy sampled from the
            policy, which sets the solver's per-step reliability and length.
        mode: ``'exact'`` for the verifier-only reward, or ``'proxy'`` to add
            the length bonus whose blind spot drives the Goodhart failure.
        rng: Seeded generator; draws a fresh problem and one solver attempt.

    Returns:
        The scalar reward for this rollout: 1.0 or 0.0 from the exact verifier
        under ``'exact'``, plus ``2 * tokens / 30`` under ``'proxy'`` so a
        long wrong answer can out-score a short correct one.
    """
    _, reliability, _ = STRATEGIES[action]
    prob = make_problem(STEPS_RLVR, rng)
    correct = final_answer(sample_trace(prob, reliability, 0.5, rng)) == prob.gold
    base = 1.0 if correct else 0.0
    return base if mode == "exact" else base + 2.0 * STRAT_TOK[action] / 30.0


def policy_stats(logits: list[float], mode: str) -> dict:
    """Closed-form monitoring of the current strategy policy.

    Because per-strategy accuracy and length are known, the policy's true
    accuracy, normalized entropy, expected length, and normalized reward
    objective are exact functions of its probabilities — no sampling needed
    to watch training, which keeps the curves clean. Reporting true accuracy
    and the optimized objective as separate series is what lets the Goodhart
    crossover be seen: the objective can climb while accuracy collapses.

    Args:
        logits: The policy's current logits over strategies.
        mode: ``'exact'`` or ``'proxy'`` — selects which reward the
            ``objective`` is measured against, so the proxy run's objective
            can rise even as true accuracy falls.

    Returns:
        A dict of exact scalar diagnostics: ``true_acc`` (accuracy the exact
        verifier would report), ``entropy`` (normalized to [0, 1]; 1 is
        uniform, 0 is a collapsed policy), ``tokens`` (expected response
        length), ``objective`` (the reward being optimized, normalized to its
        max), and ``probs`` (the rounded strategy distribution).
    """
    probs = softmax(logits)
    true_acc = sum(p * a for p, a in zip(probs, STRAT_ACC))
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs) / math.log(len(probs))
    tokens = sum(p * t for p, t in zip(probs, STRAT_TOK))
    exp_reward = [STRAT_ACC[i] + (0.0 if mode == "exact" else 2.0 * STRAT_TOK[i] / 30.0)
                  for i in range(len(probs))]
    objective = sum(p * r for p, r in zip(probs, exp_reward)) / max(exp_reward)
    return {"true_acc": true_acc, "entropy": entropy, "tokens": tokens,
            "objective": objective, "probs": [round(p, 3) for p in probs]}


def _choice(probs: list[float], rng: random.Random) -> int:
    x, cumulative = rng.random(), 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if x <= cumulative:
            return i
    return len(probs) - 1


def train_grpo(mode: str, steps: int = 150, group_size: int = 8, lr: float = 0.4,
               clip: float = 0.2, beta: float = 0.01, seed: int = 23) -> list[dict]:
    """Train the strategy policy with clipped, group-relative RLVR.

    Each step samples a group of strategies from the old policy, scores each
    by running the solver and calling the verifier, forms group-relative
    advantages (@eq-ch08-grpo), broadcasts each group advantage to its
    strategy, and takes three clipped gradient passes with a small reference
    KL. The clipped surrogate and the KL are PPO's, unchanged from @sec-ch07.

    Args:
        mode: ``'exact'`` (verifier only) or ``'proxy'`` (verifier + length).
        steps, group_size, lr, clip, beta, seed: RL hyperparameters.

    Returns:
        Per-step ``policy_stats`` snapshots (with the logits) for plotting.
    """
    rng = random.Random(seed)
    logits = [0.3, 0.5, 0.0, 0.0, -1.5]  # the hack starts rare, as exploits do
    reference = softmax(logits)
    history = [dict(step=0, **policy_stats(logits, mode))]
    for step in range(1, steps + 1):
        old = softmax(logits)
        actions = [_choice(old, rng) for _ in range(group_size)]
        advantages = grpo_advantages([rollout_reward(a, mode, rng) for a in actions])
        for _ in range(3):
            probs = softmax(logits)
            grad = [0.0] * len(logits)
            for action, adv in zip(actions, advantages):
                ratio = probs[action] / old[action]
                clipped = (adv >= 0 and ratio > 1 + clip) or (adv < 0 and ratio < 1 - clip)
                if not clipped:
                    for j in range(len(logits)):
                        score = (1.0 if j == action else 0.0) - probs[j]
                        grad[j] -= adv * ratio * score / group_size
            for j in range(len(logits)):
                grad[j] += beta * (probs[j] - reference[j])
                logits[j] -= lr * grad[j]
        history.append(dict(step=step, **policy_stats(logits, mode), logits=list(logits)))
    return history


def policy_coverage(logits: list[float], ks=(1, 2, 4, 8, 16, 32),
                    problems: int = 2000, seed: int = 31) -> dict[int, float]:
    """Coverage@k of a strategy policy: does the reachable set change with RL?

    Samples strategies from the policy, runs each through the solver, and
    reports pass@k over the resulting attempts. Comparing this curve before
    and after training answers the capability question a single pass@1 number
    cannot: whether RL enlarged the set of reachable answers or merely
    sharpened sampling within a set the base policy already covered at high k.

    Args:
        logits: The policy logits to evaluate (e.g. base vs RL-trained).
        ks: Sample budgets at which to report coverage.
        problems: Number of problems to average over.
        seed: RNG seed.

    Returns:
        A ``{k: pass@k}`` map — the coverage curve for this policy, to be read
        against a second policy's curve rather than in isolation.
    """
    rng = random.Random(seed)
    probs = softmax(logits)
    out = {k: 0 for k in ks}
    for _ in range(problems):
        prob = make_problem(STEPS_RLVR, rng)
        hits = [final_answer(sample_trace(prob, STRATEGIES[_choice(probs, rng)][1], 0.5, rng))
                == prob.gold for _ in range(max(ks))]
        for k in ks:
            out[k] += any(hits[:k])
    return {k: out[k] / problems for k in ks}
