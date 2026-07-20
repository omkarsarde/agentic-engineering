# Auto-generated from chapters/06-distributed-frontier-training.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import importlib.util
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch

warnings.filterwarnings("ignore", message="CUDA initialization")  # CPU-only chapter
GIB = 1024**3


@dataclass(frozen=True)
class TrainingConfig:
    """The shape of a dense decoder-only Transformer training run.

    These eight numbers determine everything this chapter computes:
    parameter count follows from the widths and depths, activation memory
    from the sequence length and width, and communication volume from how
    the resulting tensors are cut across devices. ``n_kv_heads`` matters
    because grouped-query attention (Chapter 3) shrinks the K/V
    projections and therefore the parameter count.

    Args:
        name: A label for printed tables.
        d_model: Residual-stream width, the ``h`` in every memory formula.
        n_layers: Transformer block count.
        n_heads: Query heads per layer.
        n_kv_heads: Key/value heads per layer (equal to ``n_heads`` for
            classic multi-head attention, smaller under GQA).
        d_ff: Feed-forward hidden width (SwiGLU uses three ``d_model x
            d_ff`` matrices).
        vocab_size: Token vocabulary, counted twice for untied input and
            output embeddings.
        seq_len: Training sequence length in tokens.
    """

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    seq_len: int


LLAMA_7B = TrainingConfig("7B-class", 4096, 32, 32, 32, 11008, 32000, 4096)
LLAMA_70B = TrainingConfig("70B-class", 8192, 80, 64, 8, 28672, 128256, 4096)


def param_count(cfg: TrainingConfig) -> dict[str, int]:
    """Count parameters per component from the configuration's shapes.

    Attention costs one ``d x d`` matrix each for the query and output
    projections plus two ``d x (d_head * n_kv_heads)`` matrices for keys
    and values; a SwiGLU feed-forward block costs three ``d x d_ff``
    matrices; untied embeddings appear once at the input and once at the
    LM head. Norm scales and biases are omitted — they are parts in ten
    thousand at these widths.

    Args:
        cfg: The model configuration to price.

    Returns:
        Parameter counts for ``"attention"``, ``"ffn"``, and
        ``"embeddings"``, plus a ``"total"`` entry summing them.
    """
    d = cfg.d_model
    kv_width = (d // cfg.n_heads) * cfg.n_kv_heads
    attention = cfg.n_layers * (2 * d * d + 2 * d * kv_width)
    ffn = cfg.n_layers * 3 * d * cfg.d_ff
    embeddings = 2 * cfg.vocab_size * d
    total = attention + ffn + embeddings
    return {"attention": attention, "ffn": ffn, "embeddings": embeddings, "total": total}


PRECISION_RECIPES = {
    "fp32": {"weights": 4, "gradients": 4, "master weights": 0, "Adam moments": 8},
    "mixed": {"weights": 2, "gradients": 2, "master weights": 4, "Adam moments": 8},
}


def state_ledger(
    params: int, recipe: str = "mixed", zero_stage: int = 0, shards: int = 1
) -> dict[str, float]:
    """Price the persistent training state a single device must hold.

    The ledger multiplies the parameter count by bytes-per-parameter for
    each state component, then divides the components that the chosen
    ZeRO stage shards across ``shards`` data-parallel devices: stage 1
    shards the fp32 master weights and Adam moments, stage 2 also shards
    gradients, stage 3 also shards the bf16 weights themselves (gathering
    them transiently per block during compute). Activations are priced
    separately — they live on a different schedule.

    Args:
        params: Trainable parameter count.
        recipe: Key into ``PRECISION_RECIPES`` (``"fp32"`` or ``"mixed"``).
        zero_stage: 0 (fully replicated) through 3 (fully sharded).
        shards: Devices in the sharding group.

    Returns:
        Bytes per device for each component plus a ``"total"`` entry.
    """
    sharded_from = {"master weights": 1, "Adam moments": 1, "gradients": 2, "weights": 3}
    ledger = {
        component: params * bytes_per / (shards if zero_stage >= sharded_from[component] else 1)
        for component, bytes_per in PRECISION_RECIPES[recipe].items()
    }
    ledger["total"] = sum(ledger.values())
    return ledger


def activation_bytes_per_layer(cfg: TrainingConfig, microbatch: int, policy: str) -> float:
    """Price the tensors autograd saves per layer under a memory policy.

    ``"materialize scores"`` is naive attention: every linear input plus
    the ``a x s x s`` score and softmax tensors. ``"fused attention"``
    keeps the linear inputs but never stores the score matrix (the
    FlashAttention effect). ``"full recompute"`` checkpoints the block:
    only its input survives the forward pass, and the rest is recomputed
    during backward at the cost of roughly one extra forward pass.

    Args:
        cfg: The model configuration (supplies ``s``, ``h``, and ``a``).
        microbatch: Sequences per device per forward pass (the ``b``).
        policy: One of the three policy names above.

    Returns:
        Saved bytes per layer, assuming two-byte activations.

    Raises:
        ValueError: If ``policy`` is not one of the three names.
    """
    s, b, h = cfg.seq_len, microbatch, cfg.d_model
    if policy == "materialize scores":
        return 34 * s * b * h + 5 * cfg.n_heads * s * s * b
    if policy == "fused attention":
        return 34 * s * b * h
    if policy == "full recompute":
        return 2 * s * b * h
    raise ValueError(f"unknown activation policy: {policy}")


def load_chapter_module(chapter: str, name: str):
    """Import an earlier chapter's tangled teaching module by path.

    Each finished chapter tangles its ``# @save`` cells into
    ``code/chNN/_generated.py``. This helper walks up from the working
    directory to the book root and imports the requested module under a
    fresh name, so later chapters build on earlier artifacts instead of
    re-implementing them.

    Args:
        chapter: Chapter code directory, e.g. ``"ch02"``.
        name: Module name to register, e.g. ``"ch02_generated"``.

    Returns:
        The executed module object.
    """
    root = Path.cwd()
    while not (root / "code" / chapter / "_generated.py").exists():
        if root.parent == root:
            raise FileNotFoundError(f"cannot find code/{chapter}/_generated.py")
        root = root.parent
    spec = importlib.util.spec_from_file_location(name, root / "code" / chapter / "_generated.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def measure_saved_bytes(model, tokens: torch.Tensor, targets: torch.Tensor, checkpoint_blocks: bool) -> int:
    """Measure the bytes autograd saves during one forward/backward pass.

    Registers a ``saved_tensors_hooks`` pair that counts every tensor the
    autograd graph stores for backward, then runs the model's blocks
    either plainly or wrapped in ``torch.utils.checkpoint`` (which saves
    only each block's input and recomputes the interior during backward).
    The count is ground truth for the policy comparison that
    ``activation_bytes_per_layer`` only models.

    Args:
        model: A Chapter 2 ``TinyGPT`` whose blocks we drive manually.
        tokens: Input token ids of shape ``(batch, seq)``.
        targets: Next-token targets of the same shape.
        checkpoint_blocks: Whether to checkpoint each block.

    Returns:
        Total bytes of tensors saved for the backward pass.
    """
    from torch.utils.checkpoint import checkpoint as run_checkpointed

    saved = 0

    def record(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved
        saved += tensor.numel() * tensor.element_size()
        return tensor

    positions = torch.arange(tokens.size(1))
    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        x = model.token_embedding(tokens) + model.position_embedding(positions)
        for block in model.blocks:
            step = lambda value, layer=block: layer(value)[0]
            x = run_checkpointed(step, x, use_reentrant=False) if checkpoint_blocks else step(x)
        logits = model.lm_head(model.final_norm(x))
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
        loss.backward()
    return saved


def tensor_parallel_ffn(
    x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor, shards: int
) -> torch.Tensor:
    """Run a two-matmul FFN split column-then-row across simulated shards.

    The up-projection is split by output columns (each shard computes a
    slice of the hidden features from the full input), the down-projection
    by input rows (each shard consumes exactly its own hidden slice and
    emits a partial sum of the output). Summing the partials — the one
    all-reduce a Megatron-style block performs per matmul pair — must
    reproduce the unsharded result up to float round-off.

    Args:
        x: Input activations of shape ``(batch, d_model)``.
        w_up: Up-projection weight of shape ``(d_model, d_ff)``.
        w_down: Down-projection weight of shape ``(d_ff, d_model)``.
        shards: How many tensor-parallel devices to simulate.

    Returns:
        The combined output, mathematically equal to
        ``relu(x @ w_up) @ w_down``.
    """
    up_shards = w_up.chunk(shards, dim=1)
    down_shards = w_down.chunk(shards, dim=0)
    partials = [torch.relu(x @ up) @ down for up, down in zip(up_shards, down_shards)]
    return torch.stack(partials).sum(dim=0)


def tp_comm_bytes_per_layer(cfg: TrainingConfig, microbatch: int, tp: int, scalar_bytes: int = 2) -> float:
    """Estimate per-device tensor-parallel traffic for one layer's fwd+bwd.

    A Megatron-style layer performs four all-reduces per microbatch (two
    forward, two backward), each over the ``s x b x h`` activation tensor.
    A ring all-reduce moves ``2 (tp - 1) / tp`` times the payload per
    participating device. This is a leading-order planning number: it
    ignores latency terms and any sequence-parallel optimizations.

    Args:
        cfg: The model configuration (supplies ``s`` and ``h``).
        microbatch: Sequences per device per forward pass.
        tp: Tensor-parallel degree (1 means no traffic).
        scalar_bytes: Bytes per activation scalar (two for bf16).

    Returns:
        Bytes per device per layer per microbatch.
    """
    if tp == 1:
        return 0.0
    payload = cfg.seq_len * microbatch * cfg.d_model * scalar_bytes
    return 4 * 2 * (tp - 1) / tp * payload


def one_f1b_order(stage: int, stages: int, microbatches: int) -> list[tuple[str, int]]:
    """Return the (kind, microbatch) sequence a 1F1B stage executes.

    Each stage runs a warmup of forwards — more for earlier stages, so
    later stages have work queued — then strictly alternates backward and
    forward until forwards run out, and drains the remaining backwards.
    The early backwards are what let a stage free each microbatch's
    activations promptly instead of holding all ``m`` at once.

    Args:
        stage: This stage's index, 0-based from the front.
        stages: Total pipeline stages.
        microbatches: Microbatches per optimizer step.

    Returns:
        Ordered ``("F", m)`` / ``("B", m)`` work items for this stage.
    """
    warmup = min(microbatches, stages - stage)
    ops = [("F", micro) for micro in range(1, warmup + 1)]
    forward_next = warmup + 1
    for backward in range(1, microbatches + 1):
        ops.append(("B", backward))
        if forward_next <= microbatches:
            ops.append(("F", forward_next))
            forward_next += 1
    return ops


def simulate_pipeline(
    stages: int, microbatches: int, forward_time: float = 1.0, backward_time: float = 2.0
) -> tuple[list[tuple], float, float]:
    """Simulate a 1F1B pipeline schedule and measure its idle fraction.

    Walks each stage's 1F1B work order under the true dependency rules
    (a forward waits on the upstream forward; a backward waits on the
    downstream backward and the local forward), packing operations
    greedily. The returned idle fraction is measured from the resulting
    timeline, so it can be checked against the closed-form bubble
    formula instead of assumed from it.

    Args:
        stages: Pipeline stages (devices in the pipeline group).
        microbatches: Microbatches per optimizer step.
        forward_time: Time units per microbatch forward on one stage.
        backward_time: Time units per microbatch backward (about twice
            the forward cost for a Transformer block).

    Returns:
        A tuple of the event list ``(stage, kind, microbatch, start,
        duration)``, the makespan, and the measured idle fraction.
    """
    orders = {k: one_f1b_order(k, stages, microbatches) for k in range(stages)}
    finish: dict[tuple[str, int, int], float] = {}
    events, free, pointer = [], [0.0] * stages, [0] * stages
    remaining = sum(len(ops) for ops in orders.values())
    while remaining:
        progressed = False
        for stage in range(stages):
            if pointer[stage] == len(orders[stage]):
                continue
            kind, micro = orders[stage][pointer[stage]]
            if kind == "F":
                needs = ("F", micro, stage - 1) if stage else None
            else:
                needs = ("B", micro, stage + 1) if stage < stages - 1 else ("F", micro, stage)
            if needs is not None and needs not in finish:
                continue
            ready = finish.get(needs, 0.0)
            if kind == "B":
                ready = max(ready, finish[("F", micro, stage)])
            start = max(free[stage], ready)
            duration = forward_time if kind == "F" else backward_time
            finish[(kind, micro, stage)] = start + duration
            events.append((stage, kind, micro, start, duration))
            free[stage] = start + duration
            pointer[stage] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            raise RuntimeError("schedule deadlocked; a dependency can never be met")
    makespan = max(start + duration for *_, start, duration in events)
    busy = microbatches * (forward_time + backward_time)
    return events, makespan, 1.0 - busy / makespan


def evaluate_plan(
    cfg: TrainingConfig, tp: int, pp: int, dp: int, *,
    global_batch: int = 128, zero_stage: int = 1, device_gib: float = 80.0,
) -> dict[str, object]:
    """Price one tensor/pipeline/data-parallel layout for a training run.

    Weights are split ``tp * pp`` ways; ZeRO at ``zero_stage`` shards the
    remaining state across the ``dp`` replicas. Activations assume full
    block recomputation (the large-model default): each stage holds one
    boundary tensor per resident layer for up to ``pp`` in-flight
    microbatches, divided by ``tp``, plus one layer's transient
    recompute working set. Communication columns are leading-order bytes
    per device per optimizer step on each axis. Deliberately ignored:
    latency terms, overlap, expert/context axes, and kernel efficiency —
    this is the memory-and-volume sieve that comes before any benchmark.

    Args:
        cfg: The model configuration to lay out.
        tp: Tensor-parallel degree (keep within one node in practice).
        pp: Pipeline-parallel degree.
        dp: Data-parallel degree.
        global_batch: Global batch in sequences; each replica processes
            ``global_batch / dp`` single-sequence microbatches per step.
        zero_stage: ZeRO stage applied across the data-parallel axis.
        device_gib: Device memory used for the feasibility verdict.

    Returns:
        A dict with the plan label, per-device state/activation/total
        GiB, a feasibility flag, bubble percentage, and per-axis
        communication GiB per device per step.
    """
    params_per_shard = param_count(cfg)["total"] / (tp * pp)
    state = state_ledger(params_per_shard, "mixed", zero_stage, dp)
    microbatches = global_batch // dp
    boundary = 2 * cfg.seq_len * cfg.d_model
    layers_here = cfg.n_layers / pp
    inflight = min(pp, microbatches) if pp > 1 else 1
    saved = boundary * layers_here * inflight / tp
    transient = activation_bytes_per_layer(cfg, 1, "fused attention") / tp
    activations = saved + transient
    bubble = (pp - 1) / (microbatches + pp - 1)
    dp_comm = 2 * (dp - 1) / dp * 2 * params_per_shard
    tp_comm = tp_comm_bytes_per_layer(cfg, 1, tp) * layers_here * microbatches
    pp_comm = 2 * boundary * microbatches if pp > 1 else 0.0
    total = state["total"] + activations
    return {
        "plan": f"tp={tp} pp={pp} dp={dp}",
        "state_gib": state["total"] / GIB, "act_gib": activations / GIB,
        "total_gib": total / GIB, "fits": total <= device_gib * GIB,
        "bubble": bubble, "dp_gib": dp_comm / GIB, "tp_gib": tp_comm / GIB,
        "pp_gib": pp_comm / GIB,
    }


def plan_table(cfg: TrainingConfig, devices: int, global_batch: int, **kwargs) -> list[dict]:
    """Evaluate and print every legal (tp, pp, dp) layout for a cluster.

    Enumerates tensor degrees up to 8 (the within-node bound), pipeline
    degrees that divide the layer count, and the implied data degree,
    skipping layouts whose data degree cannot divide the global batch.
    Rows print in enumeration order so the eye can scan how each axis
    trades memory against bubble and traffic.

    Args:
        cfg: The model configuration to lay out.
        devices: Total accelerators available.
        global_batch: Global batch in sequences.
        **kwargs: Passed through to ``evaluate_plan``.

    Returns:
        The evaluated plan dicts, one per legal layout.
    """
    rows = []
    print("plan             state   act  total fits  bubble  DP-comm  TP-comm  PP-comm")
    for tp in (1, 2, 4, 8):
        for pp in (1, 2, 4, 8):
            if devices % (tp * pp) or cfg.n_layers % pp:
                continue
            dp = devices // (tp * pp)
            if dp > global_batch or global_batch % dp:
                continue
            row = evaluate_plan(cfg, tp, pp, dp, global_batch=global_batch, **kwargs)
            rows.append(row)
            print(f"{row['plan']:15s} {row['state_gib']:6.1f} {row['act_gib']:5.1f} "
                  f"{row['total_gib']:6.1f} {'yes' if row['fits'] else ' NO':>4s} "
                  f"{row['bubble']:6.1%} {row['dp_gib']:7.1f}  {row['tp_gib']:7.1f} {row['pp_gib']:7.1f}")
    return rows


def expert_dispatch_bytes(
    tokens: int, d_model: int, top_k: int, scalar_bytes: int = 2, off_device_fraction: float = 0.875
) -> float:
    """Price the dispatch-plus-combine all-to-all of one MoE layer pass.

    Every routed token ships its hidden vector to each selected expert's
    device and receives the result back, so the traffic is twice
    ``tokens x d_model x top_k`` scaled by the fraction of assignments
    that land off-device (near ``1 - 1/experts_per_device_group`` when
    routing is balanced).

    Args:
        tokens: Tokens routed in the group per layer per step.
        d_model: Hidden width shipped per token.
        top_k: Experts selected per token.
        scalar_bytes: Bytes per activation scalar.
        off_device_fraction: Share of assignments leaving the device.

    Returns:
        Bytes crossing the fabric per expert layer per step.
    """
    return 2 * tokens * d_model * top_k * scalar_bytes * off_device_fraction


def measured_matmul_flops(n: int = 1024, repeats: int = 8) -> float:
    """Measure this machine's achieved matmul FLOP rate.

    Times ``repeats`` square fp32 matmuls of size ``n`` after two warmup
    runs and converts to FLOPs per second using the ``2 n^3`` cost of a
    matmul. This is the honest denominator for a toy MFU: what the
    hardware demonstrably achieves on the operation Transformers spend
    their time in, not a datasheet peak.

    Args:
        n: Square matrix dimension.
        repeats: Timed repetitions to average over.

    Returns:
        Achieved FLOPs per second.
    """
    a, b = torch.randn(n, n), torch.randn(n, n)
    for _ in range(2):
        a @ b
    start = time.perf_counter()
    for _ in range(repeats):
        a @ b
    return 2 * n**3 * repeats / (time.perf_counter() - start)


def wsd_multiplier(
    step: int, total_steps: int, warmup_fraction: float = 0.05, decay_fraction: float = 0.15
) -> float:
    """Return the warmup-stable-decay learning-rate multiplier for a step.

    Linear warmup to 1.0, a long flat plateau, then a linear decay to
    zero over the final ``decay_fraction`` of the run. Because the
    plateau value is constant, any plateau checkpoint can branch into
    its own decay leg — the property that makes WSD budget-agnostic
    where cosine is budget-committed.

    Args:
        step: Current optimizer step, from 0.
        total_steps: Total steps in this run (or this branch).
        warmup_fraction: Share of steps spent warming up.
        decay_fraction: Share of steps spent decaying.

    Returns:
        A multiplier in [0, 1] to apply to the peak learning rate.
    """
    warmup = max(1, round(total_steps * warmup_fraction))
    decay = max(1, round(total_steps * decay_fraction))
    if step < warmup:
        return (step + 1) / warmup
    if step < total_steps - decay:
        return 1.0
    return max(0.0, (total_steps - step - 1) / decay)


def checkpoint_waste(interval_s: float, save_s: float, mtbf_s: float, restart_s: float = 300.0) -> float:
    """Return the fraction of cluster time wasted by a checkpoint policy.

    Implements the classic three-term model: save overhead ``C_s / I``,
    expected lost work ``I / 2M`` (a failure lands midway through an
    interval on average), and restart cost ``R / M``. Valid when
    intervals are short against MTBF and failures arrive independently —
    correlated rack failures and preemption warnings break it, in the
    conservative direction.

    Args:
        interval_s: Seconds between checkpoint starts.
        save_s: Seconds of training stalled per synchronous save.
        mtbf_s: Mean seconds between job-interrupting failures.
        restart_s: Seconds to detect, reschedule, and reload.

    Returns:
        The wasted fraction of wall-clock time.
    """
    return save_s / interval_s + interval_s / (2 * mtbf_s) + restart_s / mtbf_s


def young_interval(save_s: float, mtbf_s: float) -> float:
    """Return Young's optimal checkpoint interval ``sqrt(2 * C_s * M)``."""
    return math.sqrt(2 * save_s * mtbf_s)


def run_economics(
    params: float, tokens: float, devices: int, *,
    peak_tflops: float = 989.0, mfu: float = 0.40, price_per_hour: float = 2.50,
    kw_per_device: float = 1.4, pue: float = 1.2,
) -> dict[str, float]:
    """Convert a training budget into hours, dollars, and megawatts.

    Applies the dense-Transformer estimate ``C = 6 N D``, divides by
    delivered FLOPs (peak times MFU) for device-hours, then prices time
    at an hourly rate and power at per-device draw times PUE. The
    defaults are illustrative 2026 H100-class planning constants — the
    dated landscape callout carries their provenance; the formulas are
    the durable part.

    Args:
        params: Trainable parameters ``N``.
        tokens: Training tokens ``D``.
        devices: Fleet size ``G`` (sets calendar time and megawatts).
        peak_tflops: Per-device peak TFLOP/s in training precision.
        mfu: Delivered fraction of peak (fold goodput in here too).
        price_per_hour: Effective all-in device-hour price in dollars.
        kw_per_device: Average IT draw per device including host share.
        pue: Facility power overhead factor (total / IT).

    Returns:
        FLOPs, device-hours, wall days, cost, megawatts, energy MWh,
        and dollars per million tokens, keyed by name.
    """
    flops = 6 * params * tokens
    device_hours = flops / (peak_tflops * 1e12 * mfu) / 3600
    wall_days = device_hours / devices / 24
    megawatts = devices * kw_per_device * pue / 1000
    return {
        "flops": flops, "device_hours": device_hours, "wall_days": wall_days,
        "cost": device_hours * price_per_hour, "megawatts": megawatts,
        "energy_mwh": megawatts * wall_days * 24,
        "usd_per_m_tokens": device_hours * price_per_hour / tokens * 1e6,
    }


def capital_recovery_factor(rate: float, years: int) -> float:
    """Return the annuity factor that annualizes a capital purchase.

    Multiplying a purchase price by the CRF gives the constant annual
    payment that repays principal plus the required return over the
    asset's life — the standard way to put "buy a cluster" and "rent by
    the hour" in the same units.

    Args:
        rate: Annual discount rate (0.12 means twelve percent).
        years: Useful life in years.

    Returns:
        The annual payment per dollar of capital.
    """
    growth = (1 + rate) ** years
    return rate * growth / (growth - 1)
