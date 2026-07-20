# Auto-generated from chapters/04-moe-efficient-architectures.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class RoutingState:
    """Everything a caller needs to mix experts and to diagnose balance.

    A single forward pass returns the full score distribution, the selected
    experts with their normalized gates, the measured per-expert load, and the
    two balancing losses, so nothing about a routing decision is hidden.

    Args:
        probabilities: Scores over all experts, shape ``(..., E)``.
        weights: Normalized gates for the selected experts, ``(..., k)``.
        indices: Selected expert ids, ``(..., k)``.
        load: Fraction of routed slots each expert received, ``(E,)``.
        balance_loss: Switch-style auxiliary load-balance loss (scalar).
        z_loss: Router z-loss penalizing oversized logits (scalar).
    """

    probabilities: Tensor
    weights: Tensor
    indices: Tensor
    load: Tensor
    balance_loss: Tensor
    z_loss: Tensor


class TopKRouter(nn.Module):
    """Score each token over ``experts`` experts and keep the top ``top_k``.

    The router is one linear map from width ``d`` to ``E`` logits. A separate
    ``selection_bias`` buffer steers *which* experts win (the aux-loss-free
    controller writes it) without touching the output gates, keeping the "who
    computes" and "how much they count" decisions cleanly apart.

    Args:
        width: Token width ``d``.
        experts: Number of experts ``E``.
        top_k: Experts selected per token ``k``.
        scoring: ``"softmax"`` for competing scores, ``"sigmoid"`` for
            independent ones.
    """

    def __init__(self, width: int, experts: int, top_k: int, scoring: str = "softmax") -> None:
        super().__init__()
        if not 1 <= top_k <= experts or scoring not in {"softmax", "sigmoid"}:
            raise ValueError("invalid top_k or scoring function")
        self.experts, self.top_k, self.scoring = experts, top_k, scoring
        self.projection = nn.Linear(width, experts)
        self.register_buffer("selection_bias", torch.zeros(experts))

    def forward(self, tokens: Tensor) -> RoutingState:
        logits = self.projection(tokens)
        probabilities = logits.softmax(-1) if self.scoring == "softmax" else logits.sigmoid()
        indices = (probabilities + self.selection_bias).topk(self.top_k, dim=-1).indices
        weights = probabilities.gather(-1, indices)
        if self.top_k > 1:
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        flat = indices.reshape(-1, self.top_k)
        load = F.one_hot(flat, self.experts).float().sum(1).mean(0) / self.top_k
        mean_probability = probabilities.reshape(-1, self.experts).mean(0)
        balance_loss = self.experts * (load.detach() * mean_probability).sum()
        z_loss = logits.logsumexp(-1).square().mean()
        return RoutingState(probabilities, weights, indices, load, balance_loss, z_loss)

    @torch.no_grad()
    def update_selection_bias(self, load: Tensor, rate: float = 1e-2) -> None:
        """Nudge each expert's bias toward uniform load (aux-loss-free control).

        Overloaded experts get their bias lowered and underloaded ones raised by
        a fixed step; subtracting the mean keeps the biases from drifting as a
        group, since only their differences change the selection.

        Args:
            load: Measured per-expert routed fraction, ``(E,)``.
            rate: Step size for the sign update.
        """
        target = torch.full_like(load, 1 / self.experts)
        self.selection_bias.add_(rate * torch.sign(target - load))
        self.selection_bias.sub_(self.selection_bias.mean())


class ToyMoE(nn.Module):
    """A local top-1 MoE classifier used to demonstrate routing collapse.

    Experts are tiny MLPs and dispatch is a plain Python loop over experts —
    no expert-parallel communication — so the training dynamics of the router,
    not systems throughput, are what this model exposes.

    Args:
        width: Input width.
        experts: Number of routed experts.
        hidden: Per-expert hidden width.
        classes: Output classes.
    """

    def __init__(self, width: int = 8, experts: int = 4, hidden: int = 16, classes: int = 4) -> None:
        super().__init__()
        self.router = TopKRouter(width, experts, top_k=1)
        self.classes = classes
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, classes))
            for _ in range(experts)
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, RoutingState]:
        routing = self.router(tokens)
        flat = tokens.reshape(-1, tokens.size(-1))
        indices = routing.indices.reshape(-1, 1)
        weights = routing.weights.reshape(-1, 1)
        out = flat.new_zeros(flat.size(0), self.classes)
        for slot in range(routing.indices.size(-1)):
            for expert_id, expert in enumerate(self.experts):
                rows = torch.where(indices[:, slot] == expert_id)[0]
                if rows.numel():
                    out[rows] += weights[rows, slot, None] * expert(flat[rows])
        return out.view(*tokens.shape[:-1], self.classes), routing


def train_router(balance_weight: float, steps: int = 300) -> dict[str, object]:
    """Train the toy MoE with a chosen load-balance weight and report the result.

    Expert 0 is biased at initialization so that, with no balancing pressure,
    the router collapses onto it; the balance term is the only thing that can
    reopen the other experts, which is exactly the effect we want to isolate.

    Args:
        balance_weight: Coefficient on the auxiliary load-balance loss.
        steps: Optimizer steps.

    Returns:
        A dict with the held-out per-expert ``load``, task ``accuracy``, and the
        two router losses.
    """
    torch.manual_seed(5)
    torch.set_num_threads(1)
    model = ToyMoE()
    nn.init.zeros_(model.router.projection.weight)
    nn.init.constant_(model.router.projection.bias, -1.0)
    model.router.projection.bias.data[0] = 2.0  # expert 0 starts favored
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)
    generator = torch.Generator().manual_seed(8)
    for _ in range(steps):
        batch = torch.randn(256, 8, generator=generator)
        labels = batch[:, :4].argmax(-1)
        logits, routing = model(batch)
        loss = (F.cross_entropy(logits, labels)
                + balance_weight * routing.balance_loss
                + 1e-4 * routing.z_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        batch = torch.randn(8_192, 8, generator=generator)
        labels = batch[:, :4].argmax(-1)
        logits, routing = model(batch)
    return {"balance_weight": balance_weight, "load": routing.load.tolist(),
            "accuracy": (logits.argmax(-1) == labels).float().mean().item(),
            "balance_loss": routing.balance_loss.item(), "z_loss": routing.z_loss.item()}


class GatedLinearAttention(nn.Module):
    """A linear-attention block whose recurrent state is fixed-size in context.

    Each step folds one token into a running matrix state ``S`` and normalizer
    ``z`` (@eq-ch04-gla), so decoding cost and stored state do not grow with
    sequence length. Passing the returned ``recurrent`` tuple back in resumes
    the scan, so the same weights serve a full-sequence pass and a streaming
    decode.

    Args:
        width: Model width ``d`` (also the value width ``d_v``).
        state_width: Feature width ``m`` of the key/query maps.
    """

    def __init__(self, width: int, state_width: int) -> None:
        super().__init__()
        self.query = nn.Linear(width, state_width, bias=False)
        self.key = nn.Linear(width, state_width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.decay = nn.Linear(width, 1)
        self.output = nn.Linear(width, width, bias=False)

    def forward(self, tokens: Tensor, recurrent: tuple[Tensor, Tensor] | None = None
                ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, steps, width = tokens.shape
        state_width = self.query.out_features
        if recurrent is None:
            state = tokens.new_zeros(batch, state_width, width)
            normalizer = tokens.new_zeros(batch, state_width)
        else:
            state, normalizer = recurrent
        outputs = []
        for step in range(steps):
            token = tokens[:, step]
            query = F.elu(self.query(token)) + 1  # positive feature map
            key = F.elu(self.key(token)) + 1
            value = self.value(token)
            decay = self.decay(token).sigmoid()
            state = decay[:, :, None] * state + torch.einsum("bf,bd->bfd", key, value)
            normalizer = decay * normalizer + key
            numerator = torch.einsum("bf,bfd->bd", query, state)
            denominator = (query * normalizer).sum(-1, keepdim=True).clamp_min(1e-6)
            outputs.append(self.output(numerator / denominator))
        return torch.stack(outputs, dim=1), (state, normalizer)


def associative_recall_curve(pair_counts, feature_dim, value_dim=8, trials=400, seed=3):
    """Measure exact recall of a linear-attention state as associations pile up.

    Writes ``n`` random unit-norm key / value pairs into one outer-product state
    ``S = sum phi(k) v^T`` — the identical update the ``GatedLinearAttention``
    block runs each step, with an identity feature map for clarity — then reads
    one value back by its key and counts a hit when the readout is nearest to the
    correct stored value. Recall stays high while ``n`` is below the feature
    width and decays past it, so widening the state moves the cliff without
    removing it.

    Args:
        pair_counts: Numbers of stored pairs to sweep.
        feature_dim: Feature width ``m`` (the state has ``m * value_dim`` scalars).
        value_dim: Width of each stored value.
        trials: Independent trials averaged per point.
        seed: Base RNG seed.

    Returns:
        One ``{"pairs", "accuracy"}`` dict per entry in ``pair_counts``.
    """
    rows = []
    for pairs in pair_counts:
        generator = torch.Generator().manual_seed(seed + pairs)
        hits = 0
        for _ in range(trials):
            keys = F.normalize(torch.randn(pairs, feature_dim, generator=generator), dim=-1)
            values = torch.randn(pairs, value_dim, generator=generator)
            state = keys.t() @ values                       # (feature_dim, value_dim)
            probe = int(torch.randint(pairs, (1,), generator=generator))
            readout = keys[probe] @ state                   # phi(k_j)^T S
            predicted = (readout - values).pow(2).sum(-1).argmin().item()
            hits += predicted == probe
        rows.append({"pairs": pairs, "accuracy": hits / trials})
    return rows


REAL_CONFIGS = {
    "Llama 3.1 8B": {  # dense GQA -- https://huggingface.co/meta-llama/Llama-3.1-8B
        "model_type": "llama", "hidden_size": 4096, "intermediate_size": 14336,
        "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
        "head_dim": 128, "vocab_size": 128256, "tie_word_embeddings": False,
        "torch_dtype": "bfloat16", "_reported_total": 8_000_000_000, "_verified_on": "2026-07-19"},
    "DeepSeek-V3": {  # MoE + MLA -- https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
        "model_type": "deepseek_v3", "hidden_size": 7168, "intermediate_size": 18432,
        "moe_intermediate_size": 2048, "num_hidden_layers": 61, "num_attention_heads": 128,
        "first_k_dense_replace": 3, "n_routed_experts": 256, "n_shared_experts": 1,
        "num_experts_per_tok": 8, "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "vocab_size": 129280, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "_reported_total": 671_000_000_000, "_verified_on": "2026-07-19"},
    "Qwen3-Next 80B-A3B": {  # 3:1 linear/full hybrid MoE -- https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
        "model_type": "qwen3_next", "hidden_size": 2048, "intermediate_size": 5120,
        "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512,
        "num_hidden_layers": 48, "full_attention_interval": 4, "num_attention_heads": 16,
        "num_key_value_heads": 2, "head_dim": 256, "linear_num_key_heads": 16,
        "linear_num_value_heads": 32, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "num_experts": 512, "num_experts_per_tok": 10,
        "vocab_size": 151936, "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "_reported_total": 80_000_000_000, "_verified_on": "2026-07-19"},
}


import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


def _load_ch03(name="ch03_kv_generated", relative="code/ch03/_generated.py"):
    bases = []
    try:  # inside the tangled module __file__ exists; during render it may not
        bases.append(Path(__file__).resolve().parents[2])
    except NameError:
        pass
    bases.append(Path.cwd())
    for base in bases:
        root = base
        for _ in range(6):
            candidate = root / relative
            if candidate.exists():
                spec = importlib.util.spec_from_file_location(name, candidate)
                module = importlib.util.module_from_spec(spec); sys.modules[name] = module
                spec.loader.exec_module(module); return module
            root = root.parent
    raise FileNotFoundError(relative)


_CH03 = _load_ch03()
KVConfig, kv_bytes = _CH03.KVConfig, _CH03.kv_bytes


def _dtype_bytes(cfg):
    return {"float32": 4, "float16": 2, "bfloat16": 2}[str(cfg.get("torch_dtype", "bfloat16"))]


def _embeddings(cfg):
    copies = 1 if cfg.get("tie_word_embeddings", False) else 2
    return copies * int(cfg["vocab_size"]) * int(cfg["hidden_size"])


def _gqa_params(cfg):
    w = int(cfg["hidden_size"]); qh = int(cfg["num_attention_heads"])
    kvh = int(cfg.get("num_key_value_heads", qh)); hd = int(cfg.get("head_dim", w // qh))
    return w * qh * hd + 2 * w * kvh * hd + qh * hd * w


def _mla_params(cfg):
    w = int(cfg["hidden_size"]); h = int(cfg["num_attention_heads"])
    nope, rope, val = (int(cfg[key]) for key in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
    qr, kvr = int(cfg["q_lora_rank"]), int(cfg["kv_lora_rank"])
    return (w * qr + qr + qr * h * (nope + rope) + w * (kvr + rope)
            + kvr + kvr * h * (nope + val) + h * val * w)


def _linear_mixer(cfg):
    w = int(cfg["hidden_size"])
    key_dim = int(cfg["linear_num_key_heads"]) * int(cfg["linear_key_head_dim"])
    value_heads = int(cfg["linear_num_value_heads"]); value_dim = value_heads * int(cfg["linear_value_head_dim"])
    conv, kernel = 2 * key_dim + value_dim, int(cfg["linear_conv_kernel_dim"])
    params = (w * (2 * key_dim + 2 * value_dim) + w * (2 * value_heads) + conv * kernel
              + 2 * value_heads + int(cfg["linear_value_head_dim"]) + value_dim * w)
    fixed = value_heads * int(cfg["linear_value_head_dim"]) * int(cfg["linear_key_head_dim"]) + conv * (kernel - 1)
    return params, fixed


@dataclass(frozen=True)
class ArchitectureEstimate:
    """A model's reconstructed economics from its config fields alone.

    Bundles the two parameter ledgers of @sec-ch04-ledgers with the per-token
    and 32K-context sequence state, plus the error against the published total,
    so a spec can be sanity-checked before any weights are downloaded.

    Args:
        name: Model label.
        total: Reconstructed total parameters.
        active: Reconstructed active parameters per token.
        reported_total: The publisher's stated total.
        total_error_percent: Percent error of ``total`` against ``reported_total``.
        active_fraction: ``active / total``.
        kv_bytes_per_token: Cache bytes added per token by the attention layers.
        state_gib_32k: Persistent state at 32,768 tokens, batch 1, in GiB.
        fixed_state_bytes: Context-independent recurrent state (hybrids only).
    """

    name: str
    total: int
    active: int
    reported_total: int
    total_error_percent: float
    active_fraction: float
    kv_bytes_per_token: int
    state_gib_32k: float
    fixed_state_bytes: int


def estimate_config(name, cfg, context_tokens=32_768):
    """Reconstruct total and active parameters and sequence state from a config.

    Dispatches on ``model_type`` to the dense, MoE-plus-MLA, or linear/full
    hybrid estimator, counting the common trunk once and the experts by ledger,
    then sizes the KV cache with Chapter 3's ``kv_bytes`` over only the layers
    that actually keep a cache.

    Args:
        name: Model label carried into the result.
        cfg: The config field dict.
        context_tokens: Context length at which to price sequence state.

    Returns:
        An :class:`ArchitectureEstimate`.

    Raises:
        ValueError: If ``model_type`` is unsupported.
    """
    w, layers, mt = int(cfg["hidden_size"]), int(cfg["num_hidden_layers"]), cfg["model_type"]
    fixed = 0
    if mt == "llama":
        mlp = 3 * w * int(cfg["intermediate_size"])
        total = _embeddings(cfg) + layers * (_gqa_params(cfg) + mlp + 2 * w) + w
        active = total
        cache = KVConfig(name, layers, int(cfg["num_attention_heads"]),
                         int(cfg.get("head_dim", w // int(cfg["num_attention_heads"]))),
                         _dtype_bytes(cfg), kv_heads=int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])))
    elif mt == "deepseek_v3":
        dense, moe = int(cfg["first_k_dense_replace"]), layers - int(cfg["first_k_dense_replace"])
        dense_mlp, expert_mlp = 3 * w * int(cfg["intermediate_size"]), 3 * w * int(cfg["moe_intermediate_size"])
        experts, selected, shared = int(cfg["n_routed_experts"]), int(cfg["num_experts_per_tok"]), int(cfg["n_shared_experts"])
        router = w * experts
        common = _embeddings(cfg) + layers * (_mla_params(cfg) + 2 * w) + dense * dense_mlp + w
        total = common + moe * ((experts + shared) * expert_mlp + router)
        active = common + moe * ((selected + shared) * expert_mlp + router)
        cache = KVConfig(name, layers, int(cfg["num_attention_heads"]), int(cfg["v_head_dim"]),
                         _dtype_bytes(cfg), latent_rank=int(cfg["kv_lora_rank"]), rope_key_dim=int(cfg["qk_rope_head_dim"]))
    elif mt == "qwen3_next":
        full = layers // int(cfg["full_attention_interval"]); linear = layers - full
        linear_params, fixed_scalars = _linear_mixer(cfg)
        experts, selected = int(cfg["num_experts"]), int(cfg["num_experts_per_tok"])
        expert_mlp, shared_mlp = 3 * w * int(cfg["moe_intermediate_size"]), 3 * w * int(cfg["shared_expert_intermediate_size"])
        router = w * experts
        common = (_embeddings(cfg) + full * _gqa_params(cfg) + linear * linear_params
                  + layers * (shared_mlp + router + 2 * w) + w)
        total, active = common + layers * experts * expert_mlp, common + layers * selected * expert_mlp
        cache = KVConfig(name, full, int(cfg["num_attention_heads"]), int(cfg["head_dim"]),
                         _dtype_bytes(cfg), kv_heads=int(cfg["num_key_value_heads"]))
        fixed = linear * fixed_scalars * _dtype_bytes(cfg)
    else:
        raise ValueError(f"unsupported model_type: {mt}")
    reported = int(cfg["_reported_total"])
    return ArchitectureEstimate(
        name, total, active, reported, 100 * (total - reported) / reported, active / total,
        kv_bytes(cache, 1), (kv_bytes(cache, context_tokens) + fixed) / 2**30, fixed)
