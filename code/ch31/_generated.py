# Auto-generated from chapters/31-world-models-vla-embodied.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import math

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

SIZE, MOVE, AMP = 6.0, 0.7, 0.40  # box side, action gain, current strength


def current(state: np.ndarray) -> np.ndarray:
    """Return the state-dependent drift at each position.

    The current is a smooth, bounded, nonlinear field — the "physics" a
    learned model must capture. Because it depends on *where* the agent is,
    a model that predicts only the average effect of an action (a straight
    nudge) is systematically wrong, and that error is what compounds when
    the model is rolled forward.

    Args:
        state: Positions with shape ``(..., 2)`` in the box ``[0, SIZE]``.

    Returns:
        Drift vectors with the same shape as ``state``.
    """
    x, y = state[..., 0], state[..., 1]
    return np.stack([AMP * np.sin(1.1 * y + 0.5),
                     AMP * np.sin(1.1 * x + 1.3)], axis=-1)


def true_step(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Advance the true dynamics one step: nudge, drift, clip to the box.

    This is ``f`` from @eq-ch31-loop — the ground truth no policy sees
    directly. An action in ``[-1, 1]^2`` is scaled by ``MOVE``; the current
    adds a position-dependent push; the box clips the result.

    Args:
        state: Current positions, shape ``(..., 2)``.
        action: Commanded velocities in ``[-1, 1]``, shape ``(..., 2)``.

    Returns:
        Next positions, clipped into ``[0, SIZE]``.
    """
    action = np.clip(action, -1.0, 1.0)
    return np.clip(state + MOVE * action + current(state), 0.0, SIZE)


def tokenize_action(action: np.ndarray, bins: int = 256) -> np.ndarray:
    """Quantize a normalized action into per-dimension integer tokens.

    Implements @eq-ch31-codec: each coordinate, assumed already normalized
    to ``[-1, 1]``, is mapped to its nearest of ``bins`` bin indices. The
    codec is dimension-agnostic — a 2-D gridworld velocity or a 7-DoF arm
    command tokenizes the same way.

    Args:
        action: Coordinates in ``[-1, 1]``, any length.
        bins: Vocabulary size per coordinate (at least 2).

    Returns:
        Integer tokens in ``[0, bins)``, one per coordinate.

    Raises:
        ValueError: If ``bins < 2`` or any coordinate leaves ``[-1, 1]``.
    """
    a = np.asarray(action, dtype=float)
    if bins < 2 or np.any(np.abs(a) > 1.0 + 1e-9):
        raise ValueError("need bins >= 2 and coordinates normalized to [-1, 1]")
    return np.rint((a + 1.0) * (bins - 1) / 2.0).astype(int)


def detokenize_action(tokens: np.ndarray, bins: int = 256) -> np.ndarray:
    """Map action tokens back to bin-center values in ``[-1, 1]``.

    The inverse of :func:`tokenize_action`, up to the half-bin rounding the
    forward map cannot undo.

    Args:
        tokens: Integer tokens in ``[0, bins)``.
        bins: Vocabulary size per coordinate.

    Returns:
        Decoded coordinates in ``[-1, 1]``.
    """
    return 2.0 * np.asarray(tokens, dtype=float) / (bins - 1) - 1.0


def roundtrip_error(action: np.ndarray, bins: int = 256) -> float:
    """Return the worst-coordinate error from one tokenize/detokenize pass."""
    a = np.asarray(action, dtype=float)
    return float(np.max(np.abs(a - detokenize_action(tokenize_action(a, bins), bins))))


def dct_ii(x: np.ndarray) -> np.ndarray:
    """Return the DCT-II frequency coefficients of a 1-D signal.

    A smooth action chunk concentrates its energy in the first few
    coefficients, which is exactly why frequency-domain action tokenization
    can represent a long chunk with a short token string.

    Args:
        x: The signal (an action chunk), shape ``(n,)``.

    Returns:
        Coefficients ordered low to high frequency, shape ``(n,)``.
    """
    n = len(x)
    basis = np.cos(np.pi * (2 * np.arange(n)[:, None] + 1) * np.arange(n)[None, :] / (2 * n))
    return basis.T @ x


def idct_ii(coef: np.ndarray) -> np.ndarray:
    """Reconstruct a signal from DCT-II coefficients (inverse of :func:`dct_ii`).

    Zeroing the high-frequency tail before calling this is the lossy
    compression step FAST exploits.

    Args:
        coef: Coefficients, shape ``(n,)``; zero the tail to compress.

    Returns:
        The reconstructed signal, shape ``(n,)``.
    """
    n = len(coef)
    basis = np.cos(np.pi * (2 * np.arange(n)[:, None] + 1) * np.arange(n)[None, :] / (2 * n))
    weight = np.full(n, 2.0)
    weight[0] = 1.0
    return (basis * (weight * coef)[None, :]).sum(1) / n


def expert_action(state: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """Return the demonstration controller's action toward ``goal``.

    Because the true current is known here, the expert inverts it: it asks
    for the nudge that, added to the drift, lands on the goal, then clips to
    the actuator range. This produces clean, near-optimal demonstrations for
    the policy to imitate — the toy stand-in for a teleoperator or a scripted
    oracle on a real robot.

    Args:
        state: Current position, shape ``(2,)``.
        goal: Target position, shape ``(2,)``.

    Returns:
        A demonstration action in ``[-1, 1]^2``.
    """
    state, goal = np.asarray(state, float), np.asarray(goal, float)
    return np.clip((goal - state - current(state)) / MOVE, -1.0, 1.0)


def collect_demos(n_pairs: int, seed: int, horizon: int = 30, tol: float = 0.25):
    """Roll the expert through random start/goal pairs and log (obs, action).

    Each observation is ``[x, y, goal_x, goal_y]``; each label is the
    expert's continuous action there. The dataset is exactly the states the
    *expert* visits — a coverage property that matters for imitation, as
    @sec-ch23 showed.

    Args:
        n_pairs: Number of start/goal episodes to demonstrate.
        seed: Seed for the start/goal sampler.
        horizon: Maximum steps per demonstration.
        tol: Distance at which the goal counts as reached.

    Returns:
        ``(obs, acts)`` float arrays of shape ``(N, 4)`` and ``(N, 2)``.
    """
    rng = np.random.default_rng(seed)
    obs, acts = [], []
    made = 0
    while made < n_pairs:
        s = rng.uniform(0.5, SIZE - 0.5, size=2)
        g = rng.uniform(0.5, SIZE - 0.5, size=2)
        if np.linalg.norm(s - g) < 2.0:
            continue
        for _ in range(horizon):
            a = expert_action(s, g)
            obs.append([s[0], s[1], g[0], g[1]])
            acts.append(a)
            s = true_step(s, a)
            if np.linalg.norm(s - g) <= tol:
                break
        made += 1
    return np.asarray(obs, np.float32), np.asarray(acts, np.float32)


ACTION_BINS = 21  # bins per action coordinate for the policy


class ActionTokenPolicy(nn.Module):
    """A tiny autoregressive vision-language-action policy.

    An observation encoder produces a feature vector (the stand-in for a
    VLA's fused image-and-language embedding); action coordinates are then
    decoded one token at a time, each conditioned on the observation and the
    tokens already emitted. This is the actions-as-tokens mechanism of
    RT-2/OpenVLA at teaching scale: a real VLA replaces the encoder with a
    vision-language transformer and predicts more tokens, but the
    autoregressive action factorization is the same.
    """

    def __init__(self, dim: int = 64, bins: int = ACTION_BINS):
        super().__init__()
        self.bins = bins
        self.enc = nn.Sequential(nn.Linear(4, dim), nn.GELU(),
                                 nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = nn.Parameter(torch.zeros(2, dim))
        self.tok = nn.Embedding(bins, dim)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(dim, bins))

    def logits(self, obs: torch.Tensor, prev: torch.Tensor | None, pos: int) -> torch.Tensor:
        """Return next-token logits at action position ``pos`` (0 or 1).

        Args:
            obs: Observations ``[x, y, goal_x, goal_y]``, shape ``(B, 4)``.
            prev: The previous action token, or ``None`` at position 0.
            pos: Which action coordinate is being decoded.

        Returns:
            Logits over the ``bins`` action tokens, shape ``(B, bins)``.
        """
        h = self.enc(obs / SIZE) + self.pos[pos]
        if prev is not None:
            h = h + self.tok(prev)
        return self.head(h)

    @torch.no_grad()
    def act(self, obs_np) -> np.ndarray:
        """Greedily decode both action tokens and return a continuous action."""
        obs = torch.as_tensor(np.atleast_2d(obs_np), dtype=torch.float32)
        t0 = self.logits(obs, None, 0).argmax(-1)
        t1 = self.logits(obs, t0, 1).argmax(-1)
        toks = torch.stack([t0, t1], -1).numpy()[0]
        return detokenize_action(toks, self.bins)


def train_policy(seed: int = 0, steps: int = 1500):
    """Behavior-clone the action-token policy from expert demonstrations.

    The loss is cross-entropy on both action tokens with teacher forcing —
    the second token is trained against the *true* first token, exactly the
    next-token objective from @sec-ch02, now over action bins.

    Args:
        seed: Seeds demonstrations, initialization, and batching.
        steps: Number of gradient steps.

    Returns:
        ``(policy, obs, acts)``: the trained policy and its training data.
    """
    torch.manual_seed(seed)
    obs, acts = collect_demos(260, seed)
    obs_t = torch.as_tensor(obs)
    tokens = torch.as_tensor(tokenize_action(acts, ACTION_BINS), dtype=torch.long)
    policy = ActionTokenPolicy()
    opt = torch.optim.Adam(policy.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, obs_t.shape[0], (256,), generator=gen)
        o, tk = obs_t[idx], tokens[idx]
        opt.zero_grad()
        loss = ce(policy.logits(o, None, 0), tk[:, 0]) + ce(policy.logits(o, tk[:, 0], 1), tk[:, 1])
        loss.backward()
        opt.step()
    return policy, obs, acts


def run_vla_episode(policy, start, goal, horizon: int = 30, tol: float = 0.6):
    """Drive one closed-loop episode with the token policy; grade final state.

    Args:
        policy: An object with ``act(obs) -> action``.
        start: Start position, shape ``(2,)``.
        goal: Goal position, shape ``(2,)``.
        horizon: Step budget.
        tol: Success distance to the goal.

    Returns:
        ``(reached, path)``: whether the goal was reached and the trajectory.
    """
    s = np.asarray(start, float)
    path = [s.copy()]
    for _ in range(horizon):
        s = true_step(s, policy.act([s[0], s[1], goal[0], goal[1]]))
        path.append(s.copy())
        if np.linalg.norm(s - goal) <= tol:
            return True, np.asarray(path)
    return False, np.asarray(path)


def vla_success(policy, n: int = 40, seed: int = 5):
    """Fraction of random start/goal episodes the policy solves closed-loop.

    Args:
        policy: A policy with an ``act(obs) -> action`` method.
        n: Number of random start/goal episodes to evaluate.
        seed: Seed for the start/goal sampler.

    Returns:
        ``(rate, paths)``: the success fraction and each episode's trajectory.
    """
    rng = np.random.default_rng(seed)
    wins, paths = 0, []
    for _ in range(n):
        s = rng.uniform(0.5, SIZE - 0.5, size=2)
        g = rng.uniform(0.5, SIZE - 0.5, size=2)
        if np.linalg.norm(s - g) < 2.0:
            g = np.clip(g + 2.5, 0.5, SIZE - 0.5)
        reached, path = run_vla_episode(policy, s, g)
        wins += reached
        paths.append((path, g, reached))
    return wins / n, paths


def collect_transitions(n: int, seed: int):
    """Sample random (state, action, delta) transitions from the true dynamics.

    Random exploration is the simplest data engine: uniform states, uniform
    actions, and the resulting state change. Predicting the *delta* rather
    than the next state lets the model focus on the effect of the action and
    the current, which is where all the nonlinearity lives.

    Args:
        n: Number of transitions.
        seed: Seed for states and actions.

    Returns:
        ``(states, actions, deltas)`` float32 arrays.
    """
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, SIZE, size=(n, 2))
    a = rng.uniform(-1, 1, size=(n, 2))
    return s.astype(np.float32), a.astype(np.float32), (true_step(s, a) - s).astype(np.float32)


class Dynamics(nn.Module):
    """A tiny action-conditioned dynamics model: (state, action) -> delta.

    This is $p_\\theta$ from @eq-ch31-wm made deterministic and small. It is
    the object we will plan through — and the object whose one-step accuracy
    turns out not to guarantee that plans through it succeed.
    """

    def __init__(self, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 2))

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict the state delta for states ``s`` and actions ``a``."""
        return self.net(torch.cat([s / SIZE, a], dim=-1))

    @torch.no_grad()
    def step(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Advance one step under the model, clipping to the box like reality."""
        st = torch.as_tensor(np.atleast_2d(s), dtype=torch.float32)
        at = torch.as_tensor(np.atleast_2d(a), dtype=torch.float32)
        nxt = np.clip(np.atleast_2d(s) + self.forward(st, at).numpy(), 0.0, SIZE)
        return nxt.reshape(np.shape(s))


def train_dynamics(n_trans: int, seed: int, steps: int) -> Dynamics:
    """Fit a :class:`Dynamics` model by MSE on sampled transitions.

    Args:
        n_trans: Size of the transition dataset (the data budget).
        seed: Seeds data, initialization, and batching.
        steps: Gradient steps.

    Returns:
        The trained model.
    """
    torch.manual_seed(seed)
    s, a, d = collect_transitions(n_trans, seed)
    s, a, d = map(torch.as_tensor, (s, a, d))
    model = Dynamics()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, s.shape[0], (256,), generator=gen)
        opt.zero_grad()
        loss_fn(model(s[idx], a[idx]), d[idx]).backward()
        opt.step()
    return model


def rollout_error(model, horizon: int, n_traj: int, seed: int):
    """Compare teacher-forced vs free-running model error over a horizon.

    For each trajectory, roll the true dynamics for ground truth, then
    measure two model errors at each step: teacher-forced (the model is fed
    the *true* state, isolating one-step error) and free-running (the model
    is fed its *own* prediction, the way a planner uses it). The gap between
    the two curves is compounding error — the subject of @eq-ch31-compound.

    Args:
        model: A dynamics model with a ``step`` method.
        horizon: Number of steps to roll.
        n_traj: Trajectories to average over.
        seed: Seed for starts and action sequences.

    Returns:
        ``(one_step, free_running)`` arrays of mean error per step.
    """
    rng = np.random.default_rng(seed)
    one, free = np.zeros(horizon), np.zeros(horizon)
    for _ in range(n_traj):
        s0 = rng.uniform(1, SIZE - 1, size=2)
        acts = rng.uniform(-1, 1, size=(horizon, 2))
        true = [s0.copy()]
        for t in range(horizon):
            true.append(true_step(true[-1], acts[t]))
        true = np.asarray(true)
        rolled = s0.copy()
        for t in range(horizon):
            rolled = model.step(rolled, acts[t])
            free[t] += np.linalg.norm(rolled - true[t + 1])
            one[t] += np.linalg.norm(model.step(true[t], acts[t]) - true[t + 1])
    return one / n_traj, free / n_traj


def shooting_plan(model, state, goal, horizon: int, n_samples: int, rng) -> np.ndarray:
    """Pick an action sequence by random-shooting model-predictive control.

    Sample ``n_samples`` random action sequences, roll each through the
    model, and score by *running* distance to the goal (summed over the
    horizon, not just the endpoint) so the first action is a genuine step
    toward the goal. This is the sampling-based MPC used across robotics,
    stripped to its core.

    Args:
        model: A dynamics model with a batched ``step`` method.
        state: Current state, shape ``(2,)``.
        goal: Goal state, shape ``(2,)``.
        horizon: Planning lookahead.
        n_samples: Candidate sequences to sample.
        rng: A NumPy generator.

    Returns:
        The best action sequence, shape ``(horizon, 2)``.
    """
    seqs = rng.uniform(-1, 1, size=(n_samples, horizon, 2)).astype(np.float32)
    states = np.tile(np.asarray(state, np.float32), (n_samples, 1))
    cost = np.zeros(n_samples)
    for t in range(horizon):
        states = model.step(states, seqs[:, t, :])
        cost += np.linalg.norm(states - goal, axis=-1)
    return seqs[int(np.argmin(cost))]


def planning_success(model, horizon: int, replan: bool, n_starts: int = 40,
                     seed: int = 3, budget: int = 30, tol: float = 0.25) -> float:
    """Measure goal-reaching success for open-loop vs receding-horizon control.

    With ``replan=False`` the planner commits the whole ``horizon``-step plan
    before looking again (open loop); with ``replan=True`` it executes one
    step and replans (receding horizon). Everything else is held fixed, so
    the two differ only in how long they trust the model between observations.

    Args:
        model: The dynamics model to plan through.
        horizon: Planning lookahead and open-loop commitment length.
        replan: Whether to replan after every executed step.
        n_starts: Random start/goal pairs to average over.
        seed: Seed for the start/goal sampler.
        budget: Total environment steps allowed per episode.
        tol: Success distance to the goal.

    Returns:
        Fraction of start/goal pairs solved.
    """
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0.5, SIZE - 0.5, size=(n_starts, 2))
    goals = rng.uniform(0.5, SIZE - 0.5, size=(n_starts, 2))
    wins = 0
    for start, goal in zip(starts, goals):
        if np.linalg.norm(start - goal) < 2.0:
            goal = np.clip(goal + 2.5, 0.5, SIZE - 0.5)
        s, steps, commit = start.astype(float), 0, (1 if replan else horizon)
        while steps < budget:
            plan = shooting_plan(model, s, goal, horizon, 256, rng)
            for t in range(commit):
                s = true_step(s, plan[t])
                steps += 1
                if np.linalg.norm(s - goal) <= tol:
                    wins += 1
                    break
                if steps >= budget:
                    break
            else:
                continue
            break
    return round(wins / n_starts, 3)


def saycan_select(skills, p_lang, p_aff):
    """Pick a skill by SayCan's language-times-affordance product (@eq-ch31-saycan).

    Args:
        skills: Candidate skill names.
        p_lang: Language relevance of each skill to the instruction.
        p_aff: Grounded probability each skill can succeed right now.

    Returns:
        ``(best_skill, products)``: the winner and the per-skill products.
    """
    products = [l * a for l, a in zip(p_lang, p_aff)]
    return skills[int(np.argmax(products))], products


def reversibility_gate(action, approve=None, max_disp: float = 0.9, max_force: float = 8.0) -> bool:
    """Authorize an action, escalating irreversible ones for confirmation.

    The same propose-gate-execute seam as @sec-ch16, specialized to physical
    consequence: small, low-force actions pass automatically; a large
    displacement or a hard predicted contact force requires an explicit
    approval callback, and with none the action is denied. A real gate reads
    validated force and geometry, not this toy heuristic — but the control
    point is the placement, not the threshold.

    Args:
        action: Proposed action; first coords are displacement, last is grip.
        approve: Callback that returns True to authorize an escalated action.
        max_disp: Displacement magnitude allowed without approval.
        max_force: Predicted contact force (N) allowed without approval.

    Returns:
        True if the action may execute, False if it is held.
    """
    disp = float(np.linalg.norm(action[:2]))
    force = float(max(0.0, -action[-1]) * 12.0)  # a firm grasp implies contact force
    if disp <= max_disp and force <= max_force:
        return True
    return bool(approve and approve({"displacement": disp, "predicted_force_n": force}))
