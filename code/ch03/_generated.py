# Auto-generated from chapters/03-attention-position-long-context.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from dataclasses import dataclass


@dataclass(frozen=True)
class KVConfig:
    """Architectural quantities that fix a model's persistent KV state.

    These are exactly the fields @eq-ch03-kvbytes multiplies together. A
    standard attention layer stores ``2 * kv_heads * head_dim`` scalars per
    layer per token; a latent (MLA) layer overrides that with a compressed
    width, which is why the stored width is a method rather than a constant.

    Args:
        name: A label used in tables and plots.
        layers: Number of attention layers, ``L``.
        query_heads: Number of query heads, ``H_q`` (state-neutral; kept for
            the MHA/GQA ratio).
        head_dim: Dimensions per cached head, ``d_h``.
        bytes_per_scalar: Cache dtype width ``s`` (2 for fp16/bf16, 1 for int8).
        kv_heads: Key-value heads ``H_kv`` for standard attention; ``None`` for
            a latent cache.
        latent_rank: Width ``r`` of an MLA latent, when the cache is compressed.
        rope_key_dim: Width ``d_R`` of an MLA decoupled rotary key, cached
            alongside the latent.
    """

    name: str
    layers: int
    query_heads: int
    head_dim: int
    bytes_per_scalar: int
    kv_heads: int | None = None
    latent_rank: int | None = None
    rope_key_dim: int = 0

    def cached_scalars_per_layer_token(self) -> int:
        """Return the stored scalar width per layer per token.

        Standard attention stores ``2 * H_kv * d_h`` (keys and values); a
        latent cache stores ``r + d_R`` (the compressed latent plus its
        decoupled rotary key). This is the only place the two cache designs
        differ in the byte accounting.
        """
        if self.latent_rank is not None:
            return self.latent_rank + self.rope_key_dim
        if self.kv_heads is None:
            raise ValueError("standard attention requires kv_heads")
        return 2 * self.kv_heads * self.head_dim


def kv_bytes(config: KVConfig, tokens: int, batch: int = 1) -> int:
    """Return payload bytes for one append-only, unquantized KV cache.

    This is the book's canonical KV-byte declaration (@eq-ch03-kvbytes). It is
    tensor payload only: it excludes allocator metadata, padding, temporary
    attention workspaces, and tensor-parallel duplication, all of which a
    deployment capacity model (Chapter 10) adds on top.

    Args:
        config: The architecture whose cache we are sizing.
        tokens: Retained tokens per sequence, ``T``.
        batch: Sequences represented at once, ``B``.

    Returns:
        Total cache payload in bytes.

    Raises:
        ValueError: If ``tokens`` is negative or ``batch`` is not positive.
    """
    if tokens < 0 or batch < 1:
        raise ValueError("tokens must be nonnegative and batch positive")
    return batch * tokens * config.layers * config.cached_scalars_per_layer_token() * config.bytes_per_scalar


import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GroupedQueryAttention(nn.Module):
    """Causal attention whose kv-head dial spans MHA, GQA, and MQA.

    Queries get ``query_heads`` full heads; keys and values get only
    ``kv_heads`` heads, which is all that is cached. At read time each KV head
    is repeated to serve its group of query heads, so the stored state shrinks
    by ``kv_heads / query_heads`` while the output shape is unchanged. A
    production kernel repeats the heads implicitly; we materialize them so the
    mechanism is visible.
    """

    def __init__(self, d_model: int, query_heads: int, kv_heads: int) -> None:
        super().__init__()
        if d_model % query_heads or query_heads % kv_heads:
            raise ValueError("query_heads must divide width and be divisible by kv_heads")
        self.query_heads, self.kv_heads = query_heads, kv_heads
        self.head_dim = d_model // query_heads
        self.q_proj = nn.Linear(d_model, query_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cache: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """Run causal attention and return the output and the compact KV cache.

        The returned cache holds only ``kv_heads`` heads; group expansion
        happens after it, so what is stored follows ``kv_heads`` while the
        computation still uses ``query_heads``.
        """
        batch, query_len, _ = x.shape
        q = self.q_proj(x).view(batch, query_len, self.query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, query_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, query_len, self.kv_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)
        present = (k, v)  # compact: only kv_heads heads are stored
        repeats = self.query_heads // self.kv_heads
        scores = q @ k.repeat_interleave(repeats, dim=1).transpose(-2, -1) / self.head_dim**0.5
        past = k.size(2) - query_len
        q_pos = past + torch.arange(query_len)[:, None]
        k_pos = torch.arange(k.size(2))[None, :]
        weights = F.softmax(scores.masked_fill(k_pos > q_pos, float("-inf")), dim=-1)
        out = (weights @ v.repeat_interleave(repeats, dim=1)).transpose(1, 2).reshape(batch, query_len, -1)
        return self.out_proj(out), present


class ToyLatentKVAttention(nn.Module):
    """Didactic MLA-style attention that caches one low-rank latent per token.

    The layer down-projects the residual to a rank-``latent_rank`` latent, caches
    only that, and up-projects it back to per-head keys and values at read time.
    Production MLA also absorbs the up-projections into the query and output
    maps so the per-head tensors are never materialized, and adds a decoupled
    rotary key of width ``rope_key_dim`` (@fig-ch03-mla-flow); we leave the
    boundary explicit and measure the latent compression on its own.
    """

    def __init__(self, d_model: int, query_heads: int, latent_rank: int, rope_key_dim: int = 0) -> None:
        super().__init__()
        if d_model % query_heads:
            raise ValueError("query_heads must divide d_model")
        self.query_heads = query_heads
        self.head_dim = d_model // query_heads
        self.latent_rank, self.rope_key_dim = latent_rank, rope_key_dim
        self.kv_down = nn.Linear(d_model, latent_rank, bias=False)
        self.k_up = nn.Linear(latent_rank, d_model, bias=False)
        self.v_up = nn.Linear(latent_rank, d_model, bias=False)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, cache: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Run causal attention, caching (and returning) only the latent."""
        batch, query_len, _ = x.shape
        latent = self.kv_down(x)
        if cache is not None:
            latent = torch.cat((cache, latent), dim=1)
        key_len = latent.size(1)
        q = self.q_proj(x).view(batch, query_len, self.query_heads, self.head_dim).transpose(1, 2)
        k = self.k_up(latent).view(batch, key_len, self.query_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(batch, key_len, self.query_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        q_pos = (key_len - query_len) + torch.arange(query_len)[:, None]
        k_pos = torch.arange(key_len)[None, :]
        weights = F.softmax(scores.masked_fill(k_pos > q_pos, float("-inf")), dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(batch, query_len, -1)
        return self.out_proj(out), latent

    def latent_width(self) -> int:
        """Return the stored width per token, ``r + d_R`` (@eq-ch03-kvbytes)."""
        return self.latent_rank + self.rope_key_dim


def visibility_mask(seq_len: int, window: int | None = None, sinks: int = 0) -> Tensor:
    """Return the boolean attention-visibility grid for one layer.

    Entry ``[i, j]`` is True when query position ``i`` may attend key position
    ``j``. With ``window=None`` this is the ordinary causal mask; a finite
    ``window`` keeps only the last ``W`` keys; ``sinks`` additionally keeps the
    first few positions always visible, the StreamingLLM pattern.

    Args:
        seq_len: Sequence length of the square grid.
        window: Sliding-window width ``W``; ``None`` for full causal.
        sinks: Number of initial "attention sink" positions kept visible.

    Returns:
        A ``(seq_len, seq_len)`` boolean tensor.
    """
    i = torch.arange(seq_len)[:, None]
    j = torch.arange(seq_len)[None, :]
    causal = j <= i
    if window is None:
        return causal
    return causal & (((i - j) < window) | (j < sinks))


def apply_rope(vectors: Tensor, positions: Tensor, frequencies: Tensor, magnitude: float = 1.0) -> Tensor:
    """Rotate interleaved coordinate pairs by position-dependent angles.

    Implements @eq-ch03-rope: pair ``2j, 2j+1`` at sequence position ``m`` is
    rotated by ``m * frequencies[j]``. Because the rotation is orthogonal it
    preserves each vector's norm, and rotating a query at ``m`` and a key at
    ``n`` makes their dot product depend only on ``n - m``
    (@eq-ch03-ropeinvariant).

    Args:
        vectors: Tensor ``[..., sequence, dim]`` of queries or keys.
        positions: Integer position per sequence element, length ``sequence``.
        frequencies: The ``dim/2`` band frequencies from ``inverse_frequencies``.
        magnitude: Optional scalar applied to the result (YaRN's attention
            temperature; 1.0 leaves the vector's norm unchanged).

    Returns:
        The rotated tensor, same shape as ``vectors``.

    Raises:
        ValueError: If the dimension or position count is inconsistent.
    """
    if vectors.size(-1) != 2 * frequencies.numel():
        raise ValueError("frequency count must be half the vector dimension")
    if vectors.size(-2) != positions.numel():
        raise ValueError("one position is required per sequence element")
    angles = positions.to(frequencies.dtype).unsqueeze(-1) * frequencies
    shape = (1,) * (vectors.ndim - 2) + angles.shape
    cos, sin = angles.cos().reshape(shape).to(vectors), angles.sin().reshape(shape).to(vectors)
    even, odd = vectors[..., 0::2], vectors[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2) * magnitude


import math


def ntk_base(base: float, factor: float, dim: int) -> float:
    """Return the NTK-aware adjusted RoPE base for an extension ``factor``.

    NTK-aware scaling stretches low frequencies more than high ones by raising
    the base, so local resolution survives better than under uniform
    interpolation.

    Args:
        base: The original RoPE base ``b``.
        factor: Context extension factor ``a >= 1``.
        dim: Rotary dimension ``d_R``.

    Returns:
        The adjusted base ``b'``.

    Raises:
        ValueError: If ``dim <= 2`` or ``factor < 1``.
    """
    if dim <= 2 or factor < 1:
        raise ValueError("NTK scaling requires dim > 2 and factor >= 1")
    return base * factor ** (dim / (dim - 2))


def _correction_index(rotations: float, dim: int, base: float, original: int) -> float:
    return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))


def inverse_frequencies(dim: int, *, method: str = "rope", base: float = 10_000.0,
                        factor: float = 1.0, original_context: int = 2_048,
                        beta_fast: float = 32.0, beta_slow: float = 1.0) -> tuple[Tensor, float]:
    """Return RoPE band frequencies under a context-extension method.

    Each method reshapes the ``dim/2`` bands of @eq-ch03-rope differently:
    ``rope`` is unscaled; ``pi`` divides every band by ``factor``; ``ntk``
    raises the base so high bands survive; ``yarn`` interpolates band-by-band
    and also returns a query/key magnitude multiplier (its attention
    temperature). Only YaRN returns a magnitude other than 1.

    Args:
        dim: Rotary dimension ``d_R`` (even, >= 4).
        method: One of ``"rope"``, ``"pi"``, ``"ntk"``, ``"yarn"``.
        base: Original RoPE base ``b``.
        factor: Extension factor ``a``.
        original_context: Trained context ``T_0`` (used by YaRN's ramp).
        beta_fast, beta_slow: YaRN band boundaries in rotations per ``T_0``.

    Returns:
        A tuple ``(frequencies, magnitude)``: the ``dim/2`` band frequencies and
        the scalar applied to rotated queries and keys.

    Raises:
        ValueError: If ``dim`` is invalid or the method is unknown.
    """
    if dim % 2 or dim < 4 or factor < 1:
        raise ValueError("dim must be even and >= 4; factor must be >= 1")
    indices = torch.arange(0, dim, 2, dtype=torch.float64)
    ordinary = base ** (-indices / dim)
    method = method.casefold()
    if method == "rope":
        return ordinary, 1.0
    if method == "pi":
        return ordinary / factor, 1.0
    if method == "ntk":
        return ntk_base(base, factor, dim) ** (-indices / dim), 1.0
    if method != "yarn":
        raise ValueError(f"unknown RoPE method: {method}")
    interpolated = ordinary / factor
    low = max(math.floor(_correction_index(beta_fast, dim, base, original_context)), 0)
    high = min(math.ceil(_correction_index(beta_slow, dim, base, original_context)), dim // 2 - 1)
    high = high + 1e-3 if low == high else high
    ramp = ((torch.arange(dim // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    frequencies = interpolated * ramp + ordinary * (1 - ramp)
    magnitude = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0
    return frequencies, magnitude


def retrieval_probe(method: str, *, seed: int = 1_700, trials: int = 256, context: int = 128,
                    original_context: int = 32, dim: int = 32) -> list[dict[str, float]]:
    """Measure rotary retrieval versus needle depth, without a trained model.

    At each of 21 depths, ``trials`` independent trials place one content needle
    among random distractor keys, rotate keys and the end-position query with
    ``method`` at ``context / original_context`` extension, and record whether
    the needle receives the top dot product. This isolates RoPE phase effects;
    it has no learned weights, instruction hierarchy, or attention sinks, so its
    curve is a mechanism illustration, not a language-model benchmark.

    Args:
        method: A RoPE method understood by ``inverse_frequencies``.
        seed: Base seed; depth ``i`` uses ``seed + i`` for reproducibility.
        trials: Independent trials averaged per depth.
        context: Sequence length swept.
        original_context: Trained length ``T_0``; the extension factor is
            ``context / original_context``.
        dim: Rotary dimension of the probe vectors.

    Returns:
        One dict per depth with ``depth_percent``, ``accuracy``, and
        ``standard_error``.
    """
    factor = context / original_context
    freqs, magnitude = inverse_frequencies(dim, method=method, factor=factor, original_context=original_context)
    rows: list[dict[str, float]] = []
    for i, depth in enumerate(torch.linspace(0, 1, 21)):
        g = torch.Generator().manual_seed(seed + i)
        content = torch.randn(trials, dim, generator=g)
        query = content + 0.15 * torch.randn(trials, dim, generator=g)
        needle = content + 0.15 * torch.randn(trials, dim, generator=g)
        keys = torch.randn(trials, context, dim, generator=g)
        pos = round(float(depth) * (context - 2))
        keys[:, pos] = needle
        q_rot = apply_rope(query[:, None], torch.tensor([context - 1]), freqs, magnitude)[:, 0]
        k_rot = apply_rope(keys, torch.arange(context), freqs, magnitude)
        scores = torch.einsum("btd,bd->bt", k_rot, q_rot) / dim**0.5
        hits = int((scores.argmax(1) == pos).sum())
        p = hits / trials
        rows.append({"depth_percent": round(float(depth) * 100),
                     "accuracy": p, "standard_error": math.sqrt(p * (1 - p) / trials)})
    return rows
