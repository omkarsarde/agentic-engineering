# Auto-generated from chapters/10-serving-at-scale.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import numpy as np


def sweep_load(rates, slots=8, mean_service=20.0, ttft_slo=30,
               horizon=4000, seed=0):
    """Measure throughput and goodput of a slot server across arrival rates.

    A fixed pool of ``slots`` decode streams advances one step per unit time.
    Requests arrive by a Poisson process; each waits in a FIFO queue for a free
    slot, and its queue wait is charged as time-to-first-token. Goodput counts
    only completions whose admission delay met ``ttft_slo``, so it can fall even
    as throughput climbs toward the pool's capacity of ``slots / mean_service``
    requests per step (@eq-ch10-goodput).

    Args:
        rates: Arrival rates (requests per step) to sweep.
        slots: Concurrent decode streams the server can run.
        mean_service: Mean decode length; service is uniform on ``[1, 2*mean)``.
        ttft_slo: Admission-delay budget, in steps, that a completion must meet.
        horizon: Simulated steps per rate.
        seed: Seed for the arrival and service draws.

    Returns:
        One dict per rate with ``throughput``, ``goodput`` (both per step), and
        the ``qualified`` fraction of completions that met the SLO.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for rate in rates:
        gaps = rng.exponential(1.0 / rate, size=int(rate * horizon * 1.4) + 100)
        arrivals = np.floor(np.cumsum(gaps)).astype(int)
        arrivals = arrivals[arrivals < horizon]
        service = rng.integers(1, 2 * int(mean_service), size=len(arrivals))
        active, queue, done, ok, cursor = [], [], 0, 0, 0
        for step in range(horizon):
            while cursor < len(arrivals) and arrivals[cursor] == step:
                queue.append((int(arrivals[cursor]), int(service[cursor])))
                cursor += 1
            active = [(fin, arr) for (fin, arr) in active if fin > step]
            while queue and len(active) < slots:
                arrival, length = queue.pop(0)
                ok += (step - arrival) + 1 <= ttft_slo
                active.append((step + length, arrival))
                done += 1
        rows.append({"rate": rate, "throughput": done / horizon,
                     "goodput": ok / horizon, "qualified": ok / max(1, done)})
    return rows


REQUEST_LENGTHS = [2, 12, 4, 10, 3, 9, 5, 8, 6, 7, 2, 11]


def static_schedule(lengths, batch_size):
    """Run fixed batches that each wait for their longest member to finish.

    Requests are grouped in arrival order; a group occupies all ``batch_size``
    slots until its longest member completes, so short members leave idle slots
    (the loss @eq-ch10-ustatic quantifies).

    Args:
        lengths: Decode length of each request, in steps.
        batch_size: Number of slots the batch fills.

    Returns:
        A dict of per-request ``starts`` and ``finishes``, the total
        ``makespan_steps``, and slot ``utilization``.
    """
    starts, finishes, elapsed = [0] * len(lengths), [0] * len(lengths), 0
    for base in range(0, len(lengths), batch_size):
        group = lengths[base:base + batch_size]
        for offset, length in enumerate(group):
            starts[base + offset] = elapsed
            finishes[base + offset] = elapsed + length
        elapsed += max(group)
    return {"starts": starts, "finishes": finishes, "makespan_steps": elapsed,
            "utilization": sum(lengths) / (batch_size * elapsed)}


def continuous_schedule(lengths, batch_size):
    """Refill each decode slot as soon as its current request completes.

    Admission is decided every step: finished requests leave and waiting ones
    enter the freed slots, so slot occupancy stays near ``batch_size`` until the
    queue drains. This is iteration-level batching; its win over
    ``static_schedule`` is the point of @eq-ch10-ustatic.

    Args:
        lengths: Decode length of each request, in steps.
        batch_size: Number of concurrent slots.

    Returns:
        A dict of per-request ``starts`` and ``finishes``, the total
        ``makespan_steps``, and slot ``utilization``.
    """
    starts, finishes = [-1] * len(lengths), [-1] * len(lengths)
    remaining, active, waiting, step = lengths[:], [], list(range(len(lengths))), 0
    while waiting and len(active) < batch_size:
        request = waiting.pop(0); starts[request] = step; active.append(request)
    used = 0
    while active:
        used += len(active)
        for request in active:
            remaining[request] -= 1
        step += 1
        for request in [r for r in active if remaining[r] == 0]:
            finishes[request] = step; active.remove(request)
        while waiting and len(active) < batch_size:
            request = waiting.pop(0); starts[request] = step; active.append(request)
    return {"starts": starts, "finishes": finishes, "makespan_steps": step,
            "utilization": used / (batch_size * step)}


def goodput(schedule, lengths, ttft_slo=16, tpot_slo=1.05):
    """Count requests meeting both latency SLOs, and the per-step goodput.

    A request qualifies when its admission delay is within ``ttft_slo`` and its
    steps-per-token stays within ``tpot_slo``. This is @eq-ch10-goodput applied
    to a finished schedule, and it is what separates the two policies even when
    both eventually complete every request.

    Args:
        schedule: A dict from ``static_schedule`` or ``continuous_schedule``.
        lengths: Decode length of each request, in steps.
        ttft_slo: Admission-delay budget, in steps.
        tpot_slo: Time-per-output-token budget, in steps.

    Returns:
        A dict with the ``qualified`` count and ``per_step`` goodput.
    """
    qualified = 0
    for start, finish, length in zip(schedule["starts"], schedule["finishes"], lengths):
        if start + 1 <= ttft_slo and (finish - start) / length <= tpot_slo:
            qualified += 1
    return {"qualified": qualified, "per_step": qualified / schedule["makespan_steps"]}


import math


def kv_fragmentation(lengths, block_size, max_len):
    """Compare KV memory reserved by contiguous vs paged allocation.

    A contiguous allocator reserves ``max_len`` tokens per sequence regardless
    of true length, stranding the difference; a paged allocator reserves whole
    ``block_size`` blocks, so its only waste is the partial final block of each
    sequence — bounded by less than one block per live sequence.

    Args:
        lengths: True token length of each live sequence.
        block_size: Tokens per KV block.
        max_len: Maximum sequence length the contiguous allocator reserves.

    Returns:
        A dict of tokens ``used``, tokens reserved by each allocator, each
        waste fraction, and the contiguous-over-paged reservation ratio.
    """
    used = sum(lengths)
    naive = len(lengths) * max_len
    paged = sum(math.ceil(length / block_size) * block_size for length in lengths)
    return {"used": used, "naive_reserved": naive, "paged_reserved": paged,
            "naive_waste": 1 - used / naive, "paged_waste": 1 - used / paged,
            "naive_over_paged": naive / paged}


import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class CacheKey:
    """Identity fields that together make a cached prefix safe to reuse.

    Reuse is sound only when every input that could change the cached keys and
    values matches. The tenant field is policy, not mathematics: it keeps a hit
    from becoming a cross-tenant presence oracle even when the token IDs are
    identical.
    """

    tenant: str
    model_revision: str
    tokenizer_revision: str
    template_revision: str
    adapter_revision: str
    prefix_hash: str


def make_cache_key(tenant, token_ids, *, model="model-r1", tokenizer="tok-r1",
                   template="tmpl-r1", adapter="base"):
    """Hash exact token IDs with every state-changing revision into a CacheKey.

    Args:
        tenant: Trust domain the prefix belongs to.
        token_ids: The exact prefix token IDs; any difference is a miss.
        model, tokenizer, template, adapter: Revisions whose change would make
            the cached keys and values invalid for a new request.

    Returns:
        A ``CacheKey`` whose equality decides whether reuse is permitted.
    """
    digest = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
    return CacheKey(tenant, model, tokenizer, template, adapter, digest[:16])


def cache_breakeven(compute, write, read):
    """Return the future reuses a cache write needs to beat recomputation.

    Implements @eq-ch10-breakeven. A read no cheaper than recomputation can
    never pay off on the critical path, so the function reports infinity.

    Args:
        compute: Cost to recompute the prefix.
        write: Cost to compute once and store the entry.
        read: Cost of one later cache read.

    Returns:
        The reuse count above which caching is cheaper, or ``inf``.
    """
    if read >= compute:
        return math.inf
    return max(0.0, (write - compute) / (compute - read))


def effective_cost(hit_rate, compute=1.0, write=1.25, read=0.10):
    """Average per-request prefix cost at a given cache hit rate.

    A hit pays ``read``; a miss pays ``write`` (compute plus store). The result
    is compared against the no-cache baseline of ``compute`` per request.

    Args:
        hit_rate: Fraction of requests served from cache.
        compute: No-cache recompute cost (the baseline).
        write: Miss cost: first compute plus writing the entry.
        read: Hit cost.

    Returns:
        The mean cost per request under caching.
    """
    return (1 - hit_rate) * write + hit_rate * read


import random


def categorical(probs, rng):
    """Draw one index from a categorical distribution with a seeded generator.

    Args:
        probs: A probability vector summing to one.
        rng: A ``random.Random`` supplying the uniform draw.

    Returns:
        The sampled index.
    """
    draw, total = rng.random(), 0.0
    for index, probability in enumerate(probs):
        total += probability
        if draw <= total:
            return index
    return len(probs) - 1


def speculative_draw(target, draft, rng):
    """Sample one exact target token through speculative accept/reject.

    Implements @eq-ch10-accept: a draft proposal is accepted with probability
    ``min(1, target/draft)``; on rejection the token is resampled from the
    positive residual ``(target - draft)+``. The emitted distribution equals
    ``target`` regardless of the draft's quality — the draft changes speed, not
    correctness.

    Args:
        target: The target model's next-token distribution.
        draft: The draft model's proposal distribution.
        rng: A ``random.Random`` for the proposal and accept draws.

    Returns:
        A ``(token, accepted)`` pair; ``accepted`` is False when the residual
        path was taken.
    """
    candidate = categorical(draft, rng)
    if rng.random() <= min(1.0, target[candidate] / draft[candidate]):
        return candidate, True
    residual = [max(p - q, 0.0) for p, q in zip(target, draft)]
    mass = sum(residual)
    return categorical([value / mass for value in residual], rng), False


def expected_accepted_tokens(alpha, block):
    """Expected tokens accepted per verification for a draft block (@eq-ch10-spectokens).

    Args:
        alpha: Per-token acceptance probability (``1 - TV`` of @eq-ch10-accept).
        block: Number of tokens the draft proposes before verification.

    Returns:
        The geometric-sum expectation ``(1 - alpha**(block+1)) / (1 - alpha)``.
    """
    return sum(alpha ** k for k in range(block + 1))


def speculative_speedup(alpha, block, draft_cost):
    """Speedup of speculative decoding relative to plain autoregressive decode.

    Divides expected accepted tokens by the added work of one verification plus
    ``block`` draft steps at ``draft_cost`` each. A cheap draft (idle fleet)
    gives a large speedup; an expensive draft (saturated fleet) can push the
    ratio below one, so an isolated latency win becomes a goodput loss.

    Args:
        alpha: Per-token acceptance probability.
        block: Draft block length.
        draft_cost: Cost of one draft token as a fraction of a target step.

    Returns:
        The speedup factor; below 1.0 means speculation is a net loss.
    """
    return expected_accepted_tokens(alpha, block) / (1 + block * draft_cost)


import torch


def quantize_tensor(weight, bits):
    """Symmetric per-row integer quantization of a 2-D weight (@eq-ch10-symquant).

    Each row gets its own scale from its largest magnitude, so an outlier in one
    row cannot inflate the step size of another. The returned tensor holds the
    dequantized values a real integer kernel would compute against.

    Args:
        weight: A 2-D floating-point weight tensor.
        bits: Integer width; the signed bound is ``2**(bits-1) - 1``.

    Returns:
        The dequantized tensor, same shape and dtype as ``weight``.
    """
    qmax = 2 ** (bits - 1) - 1
    scale = weight.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    return torch.clamp(torch.round(weight / scale), -qmax, qmax) * scale


def model_storage(model, bits):
    """Return (fp32_bytes, quantized_bytes) counting each weight once.

    Quantized 2-D weights cost ``bits/8`` bytes plus a 16-bit per-row scale;
    everything else stays 32-bit. Tied weights are deduped by identity so a
    shared embedding and LM head are not charged twice.

    Args:
        model: The module to size.
        bits: Integer width applied to 2-D float weights.

    Returns:
        A ``(fp32_bytes, quantized_bytes)`` tuple.
    """
    seen, fp32, quant = set(), 0, 0
    for _, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        fp32 += p.numel() * 4
        if p.dim() == 2 and p.dtype.is_floating_point:
            quant += p.numel() * bits // 8 + p.shape[0] * 2
        else:
            quant += p.numel() * 4
    return fp32, quant


def quantize_blockwise(values, bits, block_size):
    """Symmetric integer quantization applied independently to each block.

    A smaller block gives each local range its own scale, so a distant outlier
    can no longer coarsen a whole tensor's step size — at the cost of storing
    one more scale per block (the effective-width overhead ``b_s / g``).

    Args:
        values: The sequence to quantize.
        bits: Integer width.
        block_size: Values sharing one scale.

    Returns:
        A ``(reconstructed, rmse)`` pair.
    """
    qmax, restored = 2 ** (bits - 1) - 1, []
    for base in range(0, len(values), block_size):
        block = values[base:base + block_size]
        scale = max(abs(v) for v in block) / qmax or 1.0
        restored.extend(max(-qmax, min(qmax, round(v / scale))) * scale for v in block)
    rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(values, restored)) / len(values))
    return restored, rmse


def constrained_distribution(probs, allowed):
    """Mask disallowed tokens to zero and renormalize the survivors.

    This is exact for the *constrained* distribution: it is the target
    distribution conditioned on the grammar's allowed set. The reported removed
    mass is how strongly the model wanted a token the grammar forbids.

    Args:
        probs: The model's next-token distribution.
        allowed: Indices the grammar permits at this parser state.

    Returns:
        A dict with the renormalized ``distribution`` and the ``removed_mass``.

    Raises:
        ValueError: If the allowed set holds no probability (unsatisfiable).
    """
    kept = sum(probs[i] for i in allowed)
    if kept == 0:
        raise ValueError("grammar admits no token here")
    return {"distribution": [p / kept if i in allowed else 0.0 for i, p in enumerate(probs)],
            "removed_mass": 1 - kept}


def disaggregation_decision(kv_gib, interference_saved_ms, bandwidth_gibps=200.0,
                            coordination_ms=2.8):
    """Decide whether phase disaggregation beats a co-located worker (@eq-ch10-disagg).

    Charges KV transfer as ``kv_gib / bandwidth`` plus fixed coordination, and
    compares that overhead against the interference latency removing prefill
    from the decode path recovers. Longer prompts save more interference but
    also move more KV, so the decision can flip with prompt size.

    Args:
        kv_gib: KV state to transfer between pools, in GiB.
        interference_saved_ms: Latency recovered by removing co-location.
        bandwidth_gibps: Effective transfer bandwidth, GiB/s.
        coordination_ms: Routing, handshake, and expected-retry overhead.

    Returns:
        A dict with the transfer and total overhead, and the ``disaggregate``
        verdict.
    """
    transfer_ms = kv_gib / bandwidth_gibps * 1000
    overhead_ms = transfer_ms + coordination_ms
    return {"kv_gib": kv_gib, "transfer_ms": transfer_ms, "overhead_ms": overhead_ms,
            "interference_saved_ms": interference_saved_ms,
            "disaggregate": interference_saved_ms > overhead_ms}


def shared_prefix_leak(include_tenant):
    """Show whether a prefix cache leaks cross-tenant presence by hit timing.

    Tenant B warms a prefix; tenant A then probes the same token IDs. A hit is
    cheap and a miss is expensive, so if the cache key omits the tenant, tenant
    A's fast probe reveals that tenant B used the prefix — a presence oracle.
    Binding the tenant into the key (as ``make_cache_key`` does) turns the probe
    into a miss and closes the channel.

    Args:
        include_tenant: Whether the cache key includes the tenant identity.

    Returns:
        A dict with the probe's ``hit`` result and whether presence ``leaked``.
    """
    prefix = list(range(60))
    cache = {}

    def key(tenant):
        return (tenant, tuple(prefix)) if include_tenant else tuple(prefix)

    cache[key("tenant-b")] = "warm"                 # tenant B warms the prefix
    hit = key("tenant-a") in cache                  # tenant A probes it
    return {"probe_hit": hit, "leaked_presence": hit}


def budget_forcing_sweep(budgets, trials=4000, candidates=40, tokens_per_thought=8,
                         derail_rate=0.0016, seed=0):
    """Sweep a toy reasoner's accuracy against a thinking-token budget.

    Each thought samples one candidate; coverage rises as the budget buys more
    samples (the @sec-ch08 coverage law). Past a point an overthinking term
    switches away from a found answer with probability growing in the budget, so
    accuracy saturates and then declines while tokens keep accruing — the shape
    that forces @eq-ch10-budget to stop early. Budget forcing is the truncation
    at each swept budget; the curve is illustrative, not a model measurement.

    Args:
        budgets: Thinking-token budgets to sweep.
        trials: Independent tasks per budget.
        candidates: Answer-space size; per-sample hit probability is ``1/candidates``.
        tokens_per_thought: Tokens each sampled candidate costs.
        derail_rate: Overthinking rate; derail probability is ``1 - exp(-rate*budget)``.
        seed: Seed for the sampling.

    Returns:
        One dict per budget with ``accuracy`` and accuracy ``per_1k`` tokens.
    """
    rng = random.Random(seed)
    rows = []
    for budget in budgets:
        samples = max(1, budget // tokens_per_thought)
        correct = 0
        for _ in range(trials):
            found = any(rng.random() < 1.0 / candidates for _ in range(samples))
            if found and rng.random() < 1 - math.exp(-derail_rate * budget):
                found = False
            correct += found
        accuracy = correct / trials
        rows.append({"budget": budget, "accuracy": accuracy,
                     "per_1k": accuracy * 1000 / budget})
    return rows
