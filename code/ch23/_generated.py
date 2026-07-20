# Auto-generated from chapters/23-training-agents-rl.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import fmean, pstdev


def softmax(logits: list[float]) -> list[float]:
    """Turn unnormalized scores into a probability distribution.

    Subtracting the largest logit before exponentiating changes nothing
    mathematically (the shift cancels in the ratio) and keeps ``exp`` from
    overflowing — the same stabilization every LM head applies.

    Args:
        logits: Unnormalized scores, one per action.

    Returns:
        Probabilities in the same order, summing to one.
    """
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


class SoftmaxPolicy:
    """A tabular softmax policy: one row of logits per observed state.

    The table stands in for a language model. Each state string maps to a
    logit vector over the fixed action set, exactly as a transformer maps a
    context to logits over its vocabulary — here the lookup is a dict access
    instead of a forward pass, which is what makes every probability and
    every update in this chapter printable.
    """

    def __init__(self, actions: tuple[str, ...]) -> None:
        self.actions = tuple(actions)
        self.logits: dict[str, list[float]] = defaultdict(
            lambda: [0.0] * len(self.actions)
        )

    def probabilities(self, state: str) -> list[float]:
        """Return the action distribution at ``state`` (uniform if unseen)."""
        return softmax(self.logits[state])

    def sample(self, state: str, rng: random.Random) -> tuple[str, float]:
        """Draw an action and record its log probability.

        The log probability is returned *at sampling time* because the
        policy that generated an action may have changed by the time we
        compute an update — every ratio in this chapter depends on having
        stored this number.

        Args:
            state: The observation the policy is acting on.
            rng: Seeded generator; sampling is the only randomness here.

        Returns:
            The sampled action and ``log pi(action | state)``.
        """
        probs = self.probabilities(state)
        index = rng.choices(range(len(self.actions)), weights=probs, k=1)[0]
        return self.actions[index], math.log(probs[index])

    def greedy(self, state: str) -> str:
        """Return the highest-probability action (first one on ties)."""
        probs = self.probabilities(state)
        return self.actions[max(range(len(self.actions)), key=probs.__getitem__)]

    def logp(self, state: str, action: str) -> float:
        """Return ``log pi(action | state)`` under the current table."""
        return math.log(self.probabilities(state)[self.actions.index(action)])

    def update(self, state: str, action: str, step: float) -> None:
        """Take one score-function gradient step on the chosen action.

        For a softmax over logits ``z``, the gradient of ``log pi(a)`` with
        respect to ``z`` is ``onehot(a) - probabilities`` — raise the chosen
        action, lower the rest in proportion to their current mass. Scaled
        by a positive ``step`` this is gradient ascent on the action's log
        probability; a negative ``step`` pushes the action down. Supervised
        cross-entropy training and REINFORCE both reduce to this line, with
        different choices of ``step``.

        Args:
            state: The state whose logit row to modify.
            action: The action whose log probability the step targets.
            step: Learning rate times whatever weight the algorithm assigns.
        """
        probs = self.probabilities(state)
        chosen = self.actions.index(action)
        for index, probability in enumerate(probs):
            self.logits[state][index] += step * ((index == chosen) - probability)


@dataclass(frozen=True)
class Task:
    """One episode specification: an id, an order amount, and a start state.

    ``amount`` drives the gym's policy rule (amounts above 50 require
    approval), and ``start_state`` lets a taskset begin episodes in states
    an expert would rarely visit — the lever that closes coverage gaps.
    """

    task_id: str
    amount: int
    start_state: str = "start"


@dataclass(frozen=True)
class Transition:
    """One decision: the state seen, the action taken, and its log probability.

    ``behavior_logp`` is recorded at sampling time because the policy that
    produced the action will have moved by update time; it is the
    denominator of the ratio in the clipped objective, and a trajectory
    stored without it cannot be used by any ratio-based estimator.
    """

    state: str
    action: str
    behavior_logp: float


@dataclass
class Trajectory:
    """One episode's transitions plus the verifier's reading of what happened.

    ``success``, ``violation``, and ``duplicate_effect`` come from the
    environment's authoritative state, never from the policy's own account —
    the agent-RL version of @sec-ch16's rule that verification checks the
    world, not the transcript.
    """

    task_id: str
    transitions: list[Transition] = field(default_factory=list)
    success: bool = False
    violation: bool = False
    duplicate_effect: bool = False


class RefundGym:
    """A resettable six-state refund environment with a permissive effect tool.

    States: ``start`` and ``recover`` (order unknown), ``known_low`` /
    ``known_high`` (amount looked up), ``approved``, and ``refunded``.
    Amounts above 50 require approval before refunding; refunding from
    ``known_high`` directly *works* — the database reaches the goal state —
    but sets ``violation`` in verifier-only state the policy never observes.
    ``effects`` counts refund executions so a duplicate is detectable.
    """

    def reset(self, task: Task) -> str:
        """Restore authoritative state for a fresh episode of ``task``.

        Everything an episode can mutate — phase, flags, the effect
        counter — is rebuilt here; nothing leaks between episodes.

        Returns:
            The initial observation (the task's start state).
        """
        self.task = task
        self.state = task.start_state
        self.done = False
        self.refunded = False
        self.violation = False
        self.effects = 0
        return self.state

    def step(self, action: str) -> str:
        """Apply one action and return the next observation.

        Unknown or ill-timed actions do not crash the episode; they land in
        ``recover``, the state an agent actually occupies after a failed
        tool call. ``finish`` always terminates — walking away is available
        in every state, which is what makes quitting a learnable strategy.

        Raises:
            RuntimeError: If called after the episode terminated.
        """
        if self.done:
            raise RuntimeError("step after terminal state")
        high = self.task.amount > 50
        if self.state in {"start", "recover"} and action == "lookup":
            self.state = "known_high" if high else "known_low"
        elif self.state == "known_high" and action == "request_approval":
            self.state = "approved"
        elif self.state in {"known_low", "approved"} and action == "refund":
            self.refunded, self.state, self.effects = True, "refunded", self.effects + 1
        elif self.state == "known_high" and action == "refund":
            self.refunded, self.state, self.effects = True, "refunded", self.effects + 1
            self.violation = True
        elif self.state == "refunded" and action == "refund":
            self.effects += 1
        elif action == "finish":
            self.done = True
        else:
            self.state = "recover"
        return "terminal" if self.done else self.state


def rollout(policy: SoftmaxPolicy, task: Task, rng: random.Random | None = None,
            greedy: bool = False, max_steps: int = 7) -> Trajectory:
    """Run one episode of ``task`` under ``policy`` and record the evidence.

    Each transition stores the behavior policy's log probability at
    sampling time — the field every ratio-based update needs. The
    success/violation/duplicate flags are read from the gym's authoritative
    state after the episode, never inferred from the action sequence.

    Args:
        policy: The acting policy.
        task: The episode specification to reset the gym with.
        rng: Seeded generator; required unless ``greedy``.
        greedy: Take argmax actions (for evaluation) instead of sampling.
        max_steps: Hard episode budget; a wandering policy is cut off.

    Returns:
        The completed trajectory with verifier flags set.
    """
    gym, trajectory = RefundGym(), Trajectory(task.task_id)
    state = gym.reset(task)
    for _ in range(max_steps):
        if greedy:
            action = policy.greedy(state)
            logp = policy.logp(state, action)
        else:
            action, logp = policy.sample(state, rng)
        trajectory.transitions.append(Transition(state, action, logp))
        state = gym.step(action)
        if gym.done:
            break
    trajectory.success = gym.done and gym.refunded
    trajectory.violation = gym.violation
    trajectory.duplicate_effect = gym.effects > 1
    return trajectory


def expert_action(state: str, amount: int) -> str:
    """The scripted expert: the compliant action for any state.

    A deterministic controller standing in for whatever produces verified
    demonstrations — a human, a stronger policy, an oracle. Note it is
    defined on *every* state, including ``recover``: the expert knows how
    to act there; its demonstrations simply never go there.

    Args:
        state: The current observation.
        amount: The task's order amount (drives the approval rule).

    Returns:
        The action the compliance policy prescribes.
    """
    if state in {"start", "recover"}:
        return "lookup"
    if state == "known_high":
        return "request_approval"
    if state in {"known_low", "approved"}:
        return "refund"
    return "finish"


def expert_examples(tasks: list[Task]) -> list[tuple[str, str]]:
    """Roll the expert through ``tasks`` and collect (state, action) pairs.

    This is the demonstration dataset: exactly the states the expert
    visits, labeled with what the expert did there — and nothing else.

    Args:
        tasks: Episode specifications to demonstrate.

    Returns:
        State-action pairs in visit order.
    """
    examples = []
    for task in tasks:
        gym = RefundGym()
        state = gym.reset(task)
        while not gym.done:
            action = expert_action(state, task.amount)
            examples.append((state, action))
            state = gym.step(action)
    return examples


ACTIONS = ("finish", "lookup", "request_approval", "refund")
EXPERT_TASKS = [Task("expert-low", 20), Task("expert-high", 80)]
TRAIN_TASKS = [Task("train-low", 20), Task("train-high", 80),
               Task("recover-low", 30, "recover"), Task("recover-high", 90, "recover")]
EVAL_TASKS = [Task("eval-low", 25), Task("eval-high", 90),
              Task("eval-recover-low", 35, "recover"), Task("eval-recover-high", 75, "recover")]


def fit(policy: SoftmaxPolicy, examples: list[tuple[str, str]],
        epochs: int = 40, lr: float = 0.25) -> None:
    """Fit the policy to labeled (state, action) pairs by cross-entropy.

    Each pass takes one gradient step per example; because the softmax
    cross-entropy gradient is ``onehot - probabilities``, this reuses the
    policy's ``update`` with weight ``lr`` — behavior cloning and REINFORCE
    differ only in what multiplies the score function.

    Args:
        policy: The policy to train in place.
        examples: Demonstration pairs from ``expert_examples`` (or DAgger).
        epochs: Passes over the dataset.
        lr: Step size per example.
    """
    for _ in range(epochs):
        for state, action in examples:
            policy.update(state, action, lr)


def evaluate(policy: SoftmaxPolicy, tasks: list[Task]) -> list[dict]:
    """Run greedy episodes on held-out tasks and report what happened.

    Greedy evaluation asks what the policy *believes*, with exploration
    switched off. Success and violation are read from the environment, so
    a row can show success ``True`` and still carry a violation — the two
    facts are deliberately not merged.

    Args:
        policy: The policy to evaluate.
        tasks: Held-out episode specifications.

    Returns:
        One dict per task: id, action sequence, success, violation.
    """
    rows = []
    for task in tasks:
        trajectory = rollout(policy, task, greedy=True)
        rows.append({"task": task.task_id,
                     "actions": [t.action for t in trajectory.transitions],
                     "success": trajectory.success,
                     "violation": trajectory.violation})
    return rows


def dagger_labels(policy: SoftmaxPolicy, tasks: list[Task],
                  max_steps: int = 7) -> list[tuple[str, str]]:
    """Collect expert labels on the states the learner actually visits.

    The learner drives (greedily); the expert answers "what would you have
    done here?" at every state along the learner's own path. This targets
    labels exactly at the coverage gap, which is why one round can fix a
    hole that more expert-driven demonstrations never would.

    Args:
        policy: The current learner, used to generate states.
        tasks: Tasks to roll the learner through.
        max_steps: Episode budget per task.

    Returns:
        (state, expert action) pairs along learner-visited paths.
    """
    labels = []
    for task in tasks:
        gym = RefundGym()
        state = gym.reset(task)
        for _ in range(max_steps):
            labels.append((state, expert_action(state, task.amount)))
            state = gym.step(policy.greedy(state))
            if gym.done:
                break
    return labels


@dataclass(frozen=True)
class Score:
    """A trajectory's scalar reward plus its permission to teach.

    ``reward`` feeds advantage estimation; ``eligible`` decides whether the
    trajectory may contribute gradient updates at all. Keeping them
    separate is the design point: quality is a number, permission is not.
    """

    reward: float
    eligible: bool


def outcome_score(trajectory: Trajectory) -> Score:
    """Score by outcome alone: success minus a small per-step cost.

    The cost term expresses a real preference (shorter is cheaper), and
    every trajectory is eligible — this scorer believes any trajectory
    that helps the average deserves to teach. Its blind spot is the
    subject of the next section.

    Args:
        trajectory: A completed rollout with verifier flags set.

    Returns:
        The outcome-only score, always eligible.
    """
    outcome = 1.0 if trajectory.success else 0.0
    return Score(outcome - 0.05 * len(trajectory.transitions), True)


def grouped_advantages(batch: list[Trajectory], scorer) -> dict[tuple[int, int], float]:
    """Normalize trajectory rewards within each (task, state) anchor group.

    Every visit to the same anchor joins one group; each visit's advantage
    is its trajectory's reward standardized against the group (@eq-ch23-group).
    The comparison is local — "of the rollouts that stood exactly here, did
    this one end better?" — which is finer than one baseline per batch. It
    is also only as sound as state identity: two states that print alike
    but differ in hidden environment state would be grouped falsely.

    Args:
        batch: Trajectories collected under one behavior policy.
        scorer: Maps a trajectory to its ``Score``.

    Returns:
        Advantage keyed by (trajectory index, turn index).
    """
    rewards = [scorer(t).reward for t in batch]
    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for index, trajectory in enumerate(batch):
        for turn, transition in enumerate(trajectory.transitions):
            groups[(trajectory.task_id, transition.state)].append((index, turn))
    advantages = {}
    for visits in groups.values():
        values = [rewards[index] for index, _ in visits]
        mean, spread = fmean(values), pstdev(values)
        for index, turn in visits:
            advantages[(index, turn)] = (rewards[index] - mean) / (spread + 1e-6)
    return advantages


def clipped_update(policy: SoftmaxPolicy, transition: Transition,
                   advantage: float, lr: float = 0.04, clip: float = 0.2) -> bool:
    """Apply the exact gradient of the clipped surrogate for one transition.

    Computes the ratio of the action's current probability to its recorded
    behavior probability. If the ratio has already left the trust region in
    the direction the advantage is pushing, the surrogate is on its flat
    branch and the gradient is zero — the update is skipped, not reversed.
    Otherwise the step is ``lr * advantage * ratio`` times the score
    function, matching @eq-ch23-clip term for term.

    Args:
        policy: The policy being trained in place.
        transition: The decision, carrying its behavior log probability.
        advantage: The anchor-group advantage for this decision.
        lr: Base step size.
        clip: The epsilon of the trust region.

    Returns:
        True if a gradient step was taken; False if clipped away.
    """
    ratio = math.exp(policy.logp(transition.state, transition.action)
                     - transition.behavior_logp)
    if (advantage > 0 and ratio > 1 + clip) or (advantage < 0 and ratio < 1 - clip):
        return False
    policy.update(transition.state, transition.action, lr * advantage * ratio)
    return True


def train_group_relative(policy: SoftmaxPolicy, tasks: list[Task], scorer,
                         rng: random.Random, rounds: int = 30, group_size: int = 8,
                         lr: float = 0.04, clip: float = 0.2,
                         epochs: int = 2) -> tuple[dict, dict]:
    """Improve the policy by clipped, group-relative policy gradient.

    Each round samples ``group_size`` fresh rollouts per task (the groups),
    computes anchor-state advantages, drops trajectories whose score says
    ``eligible=False`` from the update set entirely, and makes ``epochs``
    clipped passes over what remains. The batch is then discarded: recorded
    behavior probabilities license a few passes, not a replay buffer.

    Args:
        policy: The starting policy, trained in place.
        tasks: The training taskset; each contributes one group per round.
        scorer: Trajectory -> ``Score``; defines reward and eligibility.
        rng: Seeded generator for all rollout sampling.
        rounds: Collect-update cycles.
        group_size: Rollouts per task per round.
        lr: Base step size for ``clipped_update``.
        clip: Trust-region epsilon.
        epochs: Optimization passes per batch.

    Returns:
        ``history`` with per-round sampled success/violation rates and
        per-epoch clip counts, and ``totals`` with transition, update, and
        veto counters for the throughput analysis.
    """
    history = {"success": [], "violation": [], "clipped": [0] * epochs}
    totals = {"transitions": 0, "updates": 0, "vetoed_rollouts": 0, "vetoed_updates": 0}
    for _ in range(rounds):
        batch = [rollout(policy, task, rng) for task in tasks for _ in range(group_size)]
        totals["transitions"] += sum(len(t.transitions) for t in batch)
        history["success"].append(fmean(t.success for t in batch))
        history["violation"].append(fmean(t.violation for t in batch))
        advantages = grouped_advantages(batch, scorer)
        vetoed = {i for i, t in enumerate(batch) if not scorer(t).eligible}
        totals["vetoed_rollouts"] += len(vetoed)
        updated: set[int] = set()
        for epoch in range(epochs):
            for index, trajectory in enumerate(batch):
                if index in vetoed:
                    continue
                for turn, transition in enumerate(trajectory.transitions):
                    if clipped_update(policy, transition, advantages[(index, turn)],
                                      lr, clip):
                        totals["updates"] += 1
                        updated.add(index)
                    else:
                        history["clipped"][epoch] += 1
        totals["vetoed_updates"] += len(vetoed & updated)
    return history, totals


def guarded_score(trajectory: Trajectory) -> Score:
    """Score outcomes, but let violations veto eligibility outright.

    The reward still carries penalty terms — useful as diagnostics and for
    advantage estimation — but eligibility is decided separately: a
    trajectory that executed a policy violation or a duplicate effect may
    never contribute a gradient update, at any reward. Quality cannot buy
    permission; that is the lexicographic contract.

    Args:
        trajectory: A completed rollout with verifier flags set.

    Returns:
        The penalized score, ineligible on any hard violation.
    """
    outcome = 1.0 if trajectory.success else 0.0
    penalty = (-2.0 if trajectory.violation else 0.0) + \
              (-1.0 if trajectory.duplicate_effect else 0.0)
    cost = -0.05 * len(trajectory.transitions)
    return Score(outcome + penalty + cost, penalty == 0.0)


PRODUCTION_LATENCY = {"env_step": 0.300, "sample": 0.030, "update": 0.002}


def modelled_phase_seconds(totals: dict, latency: dict = PRODUCTION_LATENCY) -> dict:
    """Convert a training run's event counts into modelled phase time.

    A declared cost model, not a benchmark: each environment transition is
    charged one tool-call latency, each sampled action one decode slice,
    each gradient step one amortized update. Swap in measured service
    times to turn the model into capacity planning.

    Args:
        totals: Counters from ``train_group_relative``.
        latency: Seconds per event, by phase.

    Returns:
        Modelled serial seconds per phase.
    """
    return {"environment": totals["transitions"] * latency["env_step"],
            "sampling": totals["transitions"] * latency["sample"],
            "update": totals["updates"] * latency["update"]}


def transitions_per_second(totals: dict, n_envs: int,
                           latency: dict = PRODUCTION_LATENCY) -> float:
    """Model rollout throughput with ``n_envs`` parallel environments.

    Environment time divides across the parallel fleet; sampling and
    updating are modelled as serialized on one accelerator, so they become
    the floor that parallelism cannot remove — the Amdahl term of RL
    infrastructure.

    Args:
        totals: Counters from ``train_group_relative``.
        n_envs: Parallel environment instances.
        latency: Seconds per event, by phase.

    Returns:
        Useful transitions per wall-clock second.
    """
    phases = modelled_phase_seconds(totals, latency)
    wall = phases["environment"] / n_envs + phases["sampling"] + phases["update"]
    return totals["transitions"] / wall
