# Auto-generated from chapters/02-transformer-first-principles.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from dataclasses import dataclass


@dataclass(frozen=True)
class Merge:
    """One learned BPE rule: the adjacent pair (left, right) becomes new_id.

    Merges are the entire learned state of a BPE tokenizer. Applied in
    training order they encode text; expanded recursively they decode it.
    """

    left: int
    right: int
    new_id: int


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of pair with new_id."""
    merged: list[int] = []
    index = 0
    while index < len(ids):
        if index + 1 < len(ids) and (ids[index], ids[index + 1]) == pair:
            merged.append(new_id)
            index += 2
        else:
            merged.append(ids[index])
            index += 1
    return merged


from collections import Counter


class BytePairTokenizer:
    """A byte-level BPE tokenizer: 256 byte tokens plus learned merges.

    Because the base vocabulary is every possible byte, any Unicode string
    encodes without an unknown-token path — unfamiliar text simply falls
    back to more, smaller tokens. The vocabulary is fixed the moment
    training stops; a model trained on top of it inherits that freeze.
    """

    def __init__(self, merges: list[Merge] | None = None) -> None:
        self.merges = list(merges or [])
        self.token_bytes = {index: bytes([index]) for index in range(256)}
        for merge in self.merges:
            self.token_bytes[merge.new_id] = (
                self.token_bytes[merge.left] + self.token_bytes[merge.right]
            )

    @property
    def vocab_size(self) -> int:
        """Return the fixed number of token IDs this tokenizer recognizes."""
        return 256 + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BytePairTokenizer":
        """Learn merges from UTF-8 text until the vocabulary reaches vocab_size.

        Args:
            text: Training text; it determines merge order and nothing else.
            vocab_size: Target vocabulary size, at least 256.

        Returns:
            A tokenizer with up to ``vocab_size - 256`` learned merges.

        Raises:
            ValueError: If vocab_size would not retain all 256 byte tokens.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must retain all 256 byte tokens")
        ids = list(text.encode("utf-8"))
        tokenizer = cls()
        for new_id in range(256, vocab_size):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda item: (-counts[item], item))
            ids = _merge_pair(ids, pair, new_id)
            tokenizer.merges.append(Merge(*pair, new_id))
            tokenizer.token_bytes[new_id] = (
                tokenizer.token_bytes[pair[0]] + tokenizer.token_bytes[pair[1]]
            )
        return tokenizer

    def encode(self, text: str) -> list[int]:
        """Encode any Unicode string by replaying the learned merges in order.

        Args:
            text: Text to encode as UTF-8 bytes plus merges.

        Returns:
            Token IDs, every one below ``vocab_size``.
        """
        ids = list(text.encode("utf-8"))
        for merge in self.merges:
            ids = _merge_pair(ids, (merge.left, merge.right), merge.new_id)
        return ids

    def decode(self, ids: list[int], errors: str = "strict") -> str:
        """Expand token IDs to bytes and decode the result as UTF-8.

        Args:
            ids: Token IDs produced by ``encode`` (or by a model).
            errors: UTF-8 error policy; a sampling model can emit byte
                sequences that are not valid UTF-8, so generation uses
                ``"replace"`` while round-trip checks keep ``"strict"``.

        Returns:
            The reconstructed string.
        """
        return b"".join(self.token_bytes[index] for index in ids).decode(
            "utf-8", errors=errors
        )


import torch


def next_token_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw aligned (input, target) windows from one contiguous token stream.

    Targets are the same window shifted one position later, so position t of
    the input is paired with the token that actually followed it. Every
    position in the window is a training example.

    Args:
        data: A 1-D tensor holding the encoded token stream.
        block_size: Window length T; each example predicts T next tokens.
        batch_size: Number of windows to draw.
        generator: Seeded generator choosing the window start offsets.

    Returns:
        A tuple ``(inputs, targets)``, each shaped ``(batch_size, block_size)``.
    """
    starts = torch.randint(
        0, data.numel() - block_size - 1, (batch_size,), generator=generator
    )
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs, targets


import math


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention outputs and weights for one set of projections.

    Scores are query-key dot products scaled by sqrt(d_k) so their spread
    stays inside softmax's responsive range at any head width; each output
    row is the weight-averaged mixture of value rows.

    Args:
        query: Queries shaped ``(..., T_q, d_k)``.
        key: Keys shaped ``(..., T_k, d_k)``.
        value: Values shaped ``(..., T_k, d_v)``.
        mask: Optional boolean tensor broadcastable to ``(..., T_q, T_k)``;
            True marks pairs that must receive zero weight.

    Returns:
        A tuple ``(output, weights)`` shaped ``(..., T_q, d_v)`` and
        ``(..., T_q, T_k)``; each weight row is non-negative and sums to 1.
    """
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ value, weights


from torch import Tensor, nn

LayerCache = tuple[Tensor, Tensor]


@dataclass(frozen=True)
class GPTConfig:
    """Hyperparameters that fix every tensor shape in the model.

    vocab_size ties the model to one frozen tokenizer; block_size caps how
    many positions the model can represent at once; d_model is the width of
    the residual stream; n_heads splits attention into d_model // n_heads
    subspaces; n_layers stacks identical blocks; mlp_ratio sizes the
    feed-forward hidden width relative to d_model.
    """

    vocab_size: int
    block_size: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    mlp_ratio: float = 8 / 3


class CausalSelfAttention(nn.Module):
    """Masked multi-head attention over a token sequence, cache-ready.

    One linear layer projects the input to queries, keys, and values for
    all heads at once; heads are views of that tensor, not separate
    modules. Masking compares absolute key positions with absolute query
    positions, so the same code is correct whether the layer sees a full
    window (training) or one new token extending a cache (generation).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor, past: LayerCache | None = None) -> tuple[Tensor, LayerCache]:
        batch, steps, width = x.shape
        qkv = self.qkv(x).view(batch, steps, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        query, key = self.q_norm(query), self.k_norm(key)
        past_steps = 0 if past is None else past[0].size(-2)
        if past is not None:
            key = torch.cat((past[0], key), dim=-2)
            value = torch.cat((past[1], value), dim=-2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        query_positions = past_steps + torch.arange(steps, device=x.device)[:, None]
        key_positions = torch.arange(key.size(-2), device=x.device)[None, :]
        scores = scores.masked_fill(key_positions > query_positions, float("-inf"))
        mixed = scores.softmax(dim=-1) @ value
        mixed = mixed.transpose(1, 2).contiguous().view(batch, steps, width)
        return self.output(mixed), (key, value)


from torch.nn import functional as F


class SwiGLU(nn.Module):
    """A gated feed-forward sublayer applied to each position independently.

    One fused projection produces both the gate and the value path; the
    SiLU-activated gate multiplicatively filters the value before the down
    projection. This is where the model does its per-position nonlinear
    computation — no information crosses positions here.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        hidden = max(8, round(config.mlp_ratio * config.d_model))
        self.up_gate = nn.Linear(config.d_model, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up_gate(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * value)


class Block(nn.Module):
    """One pre-norm transformer block: attention update, then FFN update.

    The block never replaces the residual stream; each sublayer reads a
    normalized copy and adds its result back, so stacking blocks composes
    updates rather than chaining replacements.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(self, x: Tensor, past: LayerCache | None = None) -> tuple[Tensor, LayerCache]:
        update, present = self.attention(self.attention_norm(x), past)
        x = x + update
        return x + self.mlp(self.mlp_norm(x)), present


class TinyGPT(nn.Module):
    """A decoder-only causal language model with a tied LM head.

    The forward pass is the whole story of this chapter: embed tokens and
    positions into the residual stream, apply the blocks under the causal
    mask, normalize, and read logits out through the tied head. The same
    method serves training (pass targets to get a loss) and cached
    generation (pass the cache from the previous call).
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: Tensor,
        targets: Tensor | None = None,
        cache: list[LayerCache] | None = None,
    ) -> tuple[Tensor, Tensor | None, list[LayerCache]]:
        """Compute causal logits, an optional loss, and the updated cache.

        Args:
            tokens: Token IDs shaped ``(batch, steps)``.
            targets: Optional next-token labels with the same shape; when
                given, the mean cross-entropy of @eq-ch02-loss is returned.
            cache: Optional per-layer past keys and values; when given,
                ``tokens`` is treated as a continuation of that prefix.

        Returns:
            Logits shaped ``(batch, steps, vocab_size)``, the loss or
            ``None``, and one ``(key, value)`` pair per layer.

        Raises:
            ValueError: If cached plus new positions exceed ``block_size``.
        """
        _, steps = tokens.shape
        past_steps = 0 if cache is None else cache[0][0].size(-2)
        if past_steps + steps > self.config.block_size:
            raise ValueError("sequence exceeds configured block_size")
        positions = torch.arange(past_steps, past_steps + steps, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        present: list[LayerCache] = []
        for index, block in enumerate(self.blocks):
            layer_past = None if cache is None else cache[index]
            x, layer_present = block(x, layer_past)
            present.append(layer_present)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss, present


@torch.inference_mode()
def generate(
    model: TinyGPT,
    prompt: Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int = 0,
    use_cache: bool = True,
) -> Tensor:
    """Autoregressively extend a prompt, with or without KV caching.

    Each step feeds the model either the whole sequence so far (uncached)
    or only the newest token plus the cache (cached); the two paths must
    produce the same tokens, which the chapter verifies directly.

    Args:
        model: A ``TinyGPT`` in any mode; evaluation mode is set here.
        prompt: Token IDs shaped ``(batch, prompt_steps)``.
        max_new_tokens: How many tokens to append.
        temperature: Zero selects the argmax; positive values sample from
            the temperature-scaled distribution.
        seed: Generator seed used when sampling.
        use_cache: Reuse past keys and values instead of recomputing them.

    Returns:
        The prompt with ``max_new_tokens`` generated IDs appended.

    Raises:
        ValueError: If the finished sequence would exceed ``block_size``.
    """
    model.eval()
    if prompt.size(1) + max_new_tokens > model.config.block_size:
        raise ValueError("generation would exceed configured block_size")
    output = prompt.clone()
    cache: list[LayerCache] | None = None
    sampler = torch.Generator(device=prompt.device).manual_seed(seed)
    for _ in range(max_new_tokens):
        step_input = output if not use_cache or cache is None else output[:, -1:]
        logits, _, cache = model(step_input, cache=cache if use_cache else None)
        next_logits = logits[:, -1]
        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probabilities = (next_logits / temperature).softmax(dim=-1)
            next_token = torch.multinomial(probabilities, 1, generator=sampler)
        output = torch.cat((output, next_token), dim=1)
    return output


TinyGPT.generate = generate


def param_breakdown(config: GPTConfig) -> dict[str, int]:
    """Predict parameter counts per component from shapes alone.

    Attention costs 4·d² per layer (fused QKV plus output projection);
    the SwiGLU FFN costs 3·d·h per layer (fused gate and value up
    projections plus the down projection); the tied embedding is counted
    once because the LM head shares its storage.

    Args:
        config: The architecture hyperparameters.

    Returns:
        A dict mapping component names to parameter counts, plus a
        ``"total"`` entry summing them.
    """
    d, layers = config.d_model, config.n_layers
    hidden = max(8, round(config.mlp_ratio * d))
    head_dim = d // config.n_heads
    breakdown = {
        "embedding (tied with LM head)": config.vocab_size * d,
        "position table": config.block_size * d,
        "attention projections": layers * 4 * d * d,
        "feed-forward (SwiGLU)": layers * 3 * d * hidden,
        "norm scales": layers * (2 * d + 2 * head_dim) + d,
    }
    breakdown["total"] = sum(breakdown.values())
    return breakdown


def parameter_count(model: nn.Module) -> int:
    """Return the number of unique trainable scalars in a model."""
    return sum(parameter.numel() for parameter in model.parameters())


TinyGPT.parameter_count = parameter_count


def flops_per_token(config: GPTConfig, context_len: int) -> dict[str, int]:
    """Estimate forward-pass FLOPs for one token at a given context length.

    Weight FLOPs follow the 2-per-parameter rule for every matmul weight
    (block projections plus the LM head; embedding lookups are free).
    Attention FLOPs cover the score and value-mixing products, which scale
    with context length rather than parameter count — the term that makes
    long context expensive even when weights are fixed.

    Args:
        config: The architecture hyperparameters.
        context_len: How many positions the token attends over.

    Returns:
        A dict with ``"weights"``, ``"attention"``, and ``"total"`` FLOPs.
    """
    d, layers = config.d_model, config.n_layers
    hidden = max(8, round(config.mlp_ratio * d))
    weight_params = layers * (4 * d * d + 3 * d * hidden) + config.vocab_size * d
    weights = 2 * weight_params
    attention_flops = 4 * layers * context_len * d
    return {
        "weights": weights,
        "attention": attention_flops,
        "total": weights + attention_flops,
    }


def lr_schedule(step: int, total_steps: int, peak: float) -> float:
    """Return the learning rate for one step of warmup-then-cosine decay.

    The first 5% of steps ramp linearly from near zero to ``peak`` while
    optimizer moment estimates fill; the remainder follows a half cosine
    from ``peak`` down to one tenth of it.

    Args:
        step: The current optimizer step, from 0.
        total_steps: Total steps in the run.
        peak: The maximum learning rate.

    Returns:
        The learning rate to apply at this step.
    """
    warmup = max(1, total_steps // 20)
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.inference_mode()
def attention_weights(model: TinyGPT, token_ids: list[int], layer: int = 0) -> Tensor:
    """Recompute one layer's per-head attention pattern for a prompt.

    Runs the embedding and the blocks below ``layer`` to reproduce that
    layer's input, then reapplies its QKV projections and causal softmax.
    Each returned row is a probability distribution over visible positions.

    Args:
        model: A ``TinyGPT`` whose pattern we want to inspect.
        token_ids: The prompt as a list of token IDs.
        layer: Which block's attention to extract, from 0.

    Returns:
        Weights shaped ``(n_heads, T, T)``; rows sum to 1 and the upper
        triangle is exactly zero.
    """
    model.eval()
    tokens = torch.tensor([token_ids])
    positions = torch.arange(tokens.size(1))
    x = model.token_embedding(tokens) + model.position_embedding(positions)
    for block in model.blocks[:layer]:
        x, _ = block(x)
    attn = model.blocks[layer].attention
    normed = model.blocks[layer].attention_norm(x)
    batch, steps, _ = normed.shape
    qkv = attn.qkv(normed).view(batch, steps, 3, attn.n_heads, attn.head_dim)
    query, key, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    query, key = attn.q_norm(query), attn.k_norm(key)
    scores = query @ key.transpose(-2, -1) / math.sqrt(attn.head_dim)
    mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool), diagonal=1)
    return scores.masked_fill(mask, float("-inf")).softmax(dim=-1)[0]


def kv_cache_bytes(cache: list[LayerCache]) -> int:
    """Return the storage bytes occupied by all cached keys and values."""
    return sum(t.numel() * t.element_size() for pair in cache for t in pair)
