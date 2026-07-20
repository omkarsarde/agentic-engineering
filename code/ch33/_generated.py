# Auto-generated from chapters/33-sd-interview-framework.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from dataclasses import dataclass


@dataclass(frozen=True)
class TokenPrice:
    """Dollar prices for one model, stated per million tokens.

    Interview arithmetic wants prices in the unit vendors quote (dollars per
    million tokens) while formulas want dollars per token. This object stores
    the quoted form and converts exactly once, so a six-order-of-magnitude
    slip cannot hide inside a longer calculation.

    Args:
        input_per_million: Price of one million uncached input tokens.
        output_per_million: Price of one million generated tokens.
        cached_input_per_million: Price of one million input tokens served
            from the provider's prompt cache.
    """

    input_per_million: float
    output_per_million: float
    cached_input_per_million: float = 0.0

    def dollars(self, fresh_in: float, out: float, cached_in: float = 0.0) -> float:
        """Return dollars for one call given its three token counts."""
        return (fresh_in * self.input_per_million
                + cached_in * self.cached_input_per_million
                + out * self.output_per_million) / 1e6


def cost_per_query(input_tokens: int, output_tokens: int, price: TokenPrice,
                   cached_input_tokens: int = 0) -> float:
    """Price one model call from its token counts (@eq-ch33-cost).

    The three terms stay separate because they respond to different knobs:
    fresh input shrinks with context discipline, cached input shrinks the bill
    only if the prompt prefix is stable, and output shrinks only with a length
    cap. The interview move is to say the split aloud — "input dominates at
    this shape, so caching and retrieval width are my cost levers" — before
    naming a total.

    Args:
        input_tokens: All input tokens the call sends, cached or not.
        output_tokens: Generated tokens the call is billed for.
        price: The model's prices per million tokens.
        cached_input_tokens: The part of ``input_tokens`` billed at the
            cached rate.

    Returns:
        Dollars for the call.

    Raises:
        ValueError: If ``cached_input_tokens`` exceeds ``input_tokens``.
    """
    if not 0 <= cached_input_tokens <= input_tokens:
        raise ValueError("cached_input_tokens must lie within input_tokens")
    fresh = input_tokens - cached_input_tokens
    return price.dollars(fresh, output_tokens, cached_input_tokens)


def completion_ms(ttft_ms: float, output_tokens: int, tpot_ms: float) -> float:
    """Return end-to-end latency for one streamed response (@eq-ch33-completion).

    TTFT buys the first token; every later token costs one TPOT. This is why a
    design answer must state two SLOs rather than one: TTFT is set by queueing
    and prefill, completion is set by the output cap, and no amount of
    retrieval tuning moves the second term.

    Args:
        ttft_ms: Time to first token, queueing and prefill included.
        output_tokens: Tokens the response streams.
        tpot_ms: Mean time per output token after the first.

    Returns:
        Milliseconds until the last token arrives; 0.0 for an empty response.
    """
    if output_tokens < 1:
        return 0.0
    return ttft_ms + (output_tokens - 1) * tpot_ms


def kv_concurrency_per_device(hbm_bytes: float, weights_bytes: float,
                              kv_bytes_per_request: float,
                              workspace_fraction: float = 0.1) -> int:
    """Return how many requests' KV caches fit on one accelerator.

    Serving capacity is a memory statement before it is a compute statement:
    weights are resident once, a workspace fraction covers activations and
    allocator slack, and the remainder is divided among per-request KV caches.
    The per-request number comes from Chapter 3's calculator
    (@eq-ch03-kvbytes); this function only does the division, so the classic
    trap stays where it belongs — in the KV arithmetic, not here.

    Args:
        hbm_bytes: Device memory in bytes.
        weights_bytes: Resident model weights, after any quantization.
        kv_bytes_per_request: KV payload of one request at its budgeted
            context length.
        workspace_fraction: Fraction of device memory held back for
            activations, fragmentation, and temporaries.

    Returns:
        Whole concurrent requests per device; never negative.

    Raises:
        ValueError: If ``kv_bytes_per_request`` is not positive.
    """
    if kv_bytes_per_request <= 0:
        raise ValueError("kv_bytes_per_request must be positive")
    free = hbm_bytes * (1 - workspace_fraction) - weights_bytes
    return max(int(free // kv_bytes_per_request), 0)


import math


def concurrent_requests(qps: float, residence_seconds: float) -> float:
    """Return mean requests in flight via Little's law (@eq-ch33-little).

    Arrival rate times residence time is in-flight work, with no assumptions
    beyond stationarity. In an interview it converts an SLO into a capacity
    statement in one step: cutting residence (a tighter output cap, a faster
    model) shrinks the fleet exactly as effectively as cutting traffic.

    Args:
        qps: Mean arrival rate, requests per second.
        residence_seconds: Mean time one request occupies the system.

    Returns:
        Mean concurrent requests.

    Raises:
        ValueError: If either argument is negative.
    """
    if qps < 0 or residence_seconds < 0:
        raise ValueError("qps and residence_seconds must be nonnegative")
    return qps * residence_seconds


def devices_for_load(qps: float, residence_seconds: float,
                     per_device_concurrency: int, headroom: float = 0.3) -> int:
    """Turn an arrival rate into an accelerator count.

    Little's law gives mean in-flight requests; a headroom factor covers
    burst and tail (the mean is not the p99); dividing by per-device
    concurrency — from :func:`kv_concurrency_per_device` or a measured
    batching limit, whichever is smaller — and rounding up gives devices.

    Args:
        qps: Mean arrival rate, requests per second.
        residence_seconds: Mean residence time per request.
        per_device_concurrency: Concurrent requests one device sustains.
        headroom: Fractional margin above the mean (0.3 = 30%).

    Returns:
        Whole devices, at least 1.

    Raises:
        ValueError: If ``per_device_concurrency`` is not positive.
    """
    if per_device_concurrency < 1:
        raise ValueError("per_device_concurrency must be positive")
    in_flight = concurrent_requests(qps, residence_seconds) * (1 + headroom)
    return max(math.ceil(in_flight / per_device_concurrency), 1)


def index_bytes(vectors: int, dim: int, bytes_per_dim: float,
                overhead: float = 1.5, replicas: int = 1) -> float:
    """Size a vector index from first principles (@eq-ch33-index).

    Raw payload is vectors times dimension times element width; a graph or
    codebook structure multiplies it by an overhead factor, and replication
    multiplies again. The result decides a real architecture question — one
    memory-heavy node or a sharded cluster — which is why the number belongs
    in move 2, before any boxes. Source text, metadata, ACL indexes, and
    build-time copies are extra and worth naming aloud.

    Args:
        vectors: Number of stored vectors (chunks, not documents).
        dim: Embedding dimension.
        bytes_per_dim: Element width: 4 for fp32, 2 for fp16, 1 for int8.
        overhead: Index-structure multiplier over raw payload.
        replicas: Full copies serving reads.

    Returns:
        Total bytes across replicas.

    Raises:
        ValueError: If any count is not positive.
    """
    if min(vectors, dim, replicas) < 1 or bytes_per_dim <= 0 or overhead < 1:
        raise ValueError("all sizing factors must be positive (overhead >= 1)")
    return vectors * dim * bytes_per_dim * overhead * replicas


def loop_cost(steps: int, base_context_tokens: int, growth_tokens_per_step: int,
              output_tokens_per_step: int, price: TokenPrice,
              cached_resend: bool = False) -> float:
    """Price an agent loop that resends its growing context (@eq-ch33-loop).

    Step ``i`` sends the whole context so far — base plus ``i - 1`` growth
    increments — so uncached input cost grows quadratically in steps even
    though each single step looks cheap. With ``cached_resend`` the previous
    step's context is billed at the cached rate and only the new tokens are
    fresh, which is the mechanism that makes long loops affordable. Either
    way, the honest design move is a hard step cap with an escalation path,
    priced in advance.

    Args:
        steps: Loop iterations to price.
        base_context_tokens: Context at step one (system, tools, task).
        growth_tokens_per_step: Tokens appended per step (tool results plus
            the previous output).
        output_tokens_per_step: Tokens generated at each step.
        price: The model's prices per million tokens.
        cached_resend: Bill each step's previously-sent prefix at the cached
            rate instead of the fresh rate.

    Returns:
        Dollars for the whole loop.

    Raises:
        ValueError: If ``steps`` is not positive or any count is negative.
    """
    if steps < 1 or min(base_context_tokens, growth_tokens_per_step,
                        output_tokens_per_step) < 0:
        raise ValueError("steps must be positive and token counts nonnegative")
    total = 0.0
    for i in range(1, steps + 1):
        context = base_context_tokens + (i - 1) * growth_tokens_per_step
        cached = context - growth_tokens_per_step if (cached_resend and i > 1) else 0
        total += cost_per_query(context, output_tokens_per_step, price,
                                cached_input_tokens=cached)
    return total


def max_affordable_steps(budget_dollars: float, base_context_tokens: int,
                         growth_tokens_per_step: int, output_tokens_per_step: int,
                         price: TokenPrice, cached_resend: bool = False,
                         step_cap: int = 1_000) -> int:
    """Return the largest step count whose loop cost fits a budget.

    This is the loop cap stated as money instead of a guess: given a per-task
    budget, it walks steps upward until the next one would break the budget.
    A returned 0 means even one step exceeds the budget — a design signal,
    not an edge case.

    Args:
        budget_dollars: The per-task spend ceiling.
        base_context_tokens: Context at step one.
        growth_tokens_per_step: Tokens appended per step.
        output_tokens_per_step: Tokens generated per step.
        price: The model's prices per million tokens.
        cached_resend: Whether resent prefixes bill at the cached rate.
        step_cap: Safety bound on the search.

    Returns:
        The largest affordable whole number of steps, up to ``step_cap``.
    """
    affordable = 0
    for k in range(1, step_cap + 1):
        if loop_cost(k, base_context_tokens, growth_tokens_per_step,
                     output_tokens_per_step, price, cached_resend) > budget_dollars:
            break
        affordable = k
    return affordable


def context_budget(window_tokens: int, *, system: int, history: int,
                   retrieval: int, output_reserve: int) -> dict[str, int]:
    """Balance one call's context window like an account (@eq-ch33-context).

    The window is a hard capacity shared by everything the call sends and
    reserves; this function makes the split explicit and refuses an
    overcommitted plan instead of letting truncation decide silently at
    runtime. Saying the split aloud is the fastest way to expose a design
    error — a 40k-token retrieval plan inside a 32k window is a redesign,
    not a tuning problem. Chapter 13 owns what to do when the budget stops
    fitting: compact deliberately rather than truncate accidentally.

    Args:
        window_tokens: The model's context window.
        system: Tokens for instructions and tool schemas.
        history: Conversation or trajectory tokens kept verbatim.
        retrieval: Retrieved or observed evidence tokens.
        output_reserve: Tokens reserved so generation cannot be cut off.

    Returns:
        A dict of the four components plus ``"free"``, the uncommitted slack.

    Raises:
        ValueError: If the window is not positive, a component is negative,
            or the components exceed the window.
    """
    parts = {"system": system, "history": history,
             "retrieval": retrieval, "output_reserve": output_reserve}
    if window_tokens < 1 or any(v < 0 for v in parts.values()):
        raise ValueError("window must be positive and components nonnegative")
    used = sum(parts.values())
    if used > window_tokens:
        raise ValueError(f"overcommitted by {used - window_tokens:,} tokens")
    return {**parts, "free": window_tokens - used}
