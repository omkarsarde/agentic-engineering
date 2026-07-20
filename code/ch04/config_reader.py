"""Translate selected public model configs into economic quantities."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


# Chapter 3's KV arithmetic now lives in its tangled teaching module.  Load it
# by path under a unique name so this legacy reader keeps working without a
# sys.path insert that could collide with another chapter's ``_generated``.  The
# module must be registered before ``exec_module`` so its dataclass can resolve.
_KV_SPEC = importlib.util.spec_from_file_location(
    "ch03_kv_generated", Path(__file__).resolve().parents[1] / "ch03" / "_generated.py"
)
assert _KV_SPEC is not None and _KV_SPEC.loader is not None
_KV = importlib.util.module_from_spec(_KV_SPEC)
sys.modules.setdefault("ch03_kv_generated", _KV)
_KV_SPEC.loader.exec_module(_KV)
KVConfig, kv_bytes = _KV.KVConfig, _KV.kv_bytes


@dataclass(frozen=True)
class ArchitectureEstimate:
    name: str
    kind: str
    total_params: int
    active_params: int
    reported_total_params: int
    reported_active_params: int
    total_error_percent: float
    active_fraction: float
    kv_bytes_per_token: int
    fixed_state_bytes: int
    context_state_gib_32k: float
    source: str
    verified_on: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _dtype_bytes(config: dict[str, object]) -> int:
    name = str(config.get("torch_dtype", config.get("dtype", "bfloat16")))
    return {"float32": 4, "float16": 2, "bfloat16": 2}[name]


def _embeddings(config: dict[str, object]) -> int:
    copies = 1 if config.get("tie_word_embeddings", False) else 2
    return copies * int(config["vocab_size"]) * int(config["hidden_size"])


def _gqa_params(config: dict[str, object]) -> int:
    width = int(config["hidden_size"])
    q_heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", q_heads))
    head_dim = int(config.get("head_dim", width // q_heads))
    return width * q_heads * head_dim + 2 * width * kv_heads * head_dim + q_heads * head_dim * width


def _dense_estimate(config: dict[str, object]) -> tuple[int, int, KVConfig, int]:
    width, layers = int(config["hidden_size"]), int(config["num_hidden_layers"])
    attention = _gqa_params(config)
    mlp = 3 * width * int(config["intermediate_size"])
    total = _embeddings(config) + layers * (attention + mlp + 2 * width) + width
    cache = KVConfig(
        str(config["_fixture_name"]), layers, int(config["num_attention_heads"]),
        int(config.get("head_dim", width // int(config["num_attention_heads"]))),
        _dtype_bytes(config), kv_heads=int(config.get("num_key_value_heads", config["num_attention_heads"])),
    )
    return total, total, cache, 0


def _mla_params(config: dict[str, object]) -> int:
    width = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    nope, rope, value = (int(config[key]) for key in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"))
    q_rank, kv_rank = int(config["q_lora_rank"]), int(config["kv_lora_rank"])
    return (
        width * q_rank + q_rank + q_rank * heads * (nope + rope)
        + width * (kv_rank + rope) + kv_rank + kv_rank * heads * (nope + value)
        + heads * value * width
    )


def _deepseek_estimate(config: dict[str, object]) -> tuple[int, int, KVConfig, int]:
    width, layers = int(config["hidden_size"]), int(config["num_hidden_layers"])
    dense_layers = int(config["first_k_dense_replace"])
    moe_layers = layers - dense_layers
    dense_mlp = 3 * width * int(config["intermediate_size"])
    expert_mlp = 3 * width * int(config["moe_intermediate_size"])
    experts, selected = int(config["n_routed_experts"]), int(config["num_experts_per_tok"])
    shared = int(config.get("n_shared_experts", 0))
    router = width * experts
    common = _embeddings(config) + layers * (_mla_params(config) + 2 * width) + dense_layers * dense_mlp + width
    total = common + moe_layers * ((experts + shared) * expert_mlp + router)
    active = common + moe_layers * ((selected + shared) * expert_mlp + router)
    cache = KVConfig(
        str(config["_fixture_name"]), layers, int(config["num_attention_heads"]),
        int(config["v_head_dim"]), _dtype_bytes(config),
        latent_rank=int(config["kv_lora_rank"]), rope_key_dim=int(config["qk_rope_head_dim"]),
    )
    return total, active, cache, 0


def _linear_mixer_params(config: dict[str, object]) -> tuple[int, int]:
    width = int(config["hidden_size"])
    key_dim = int(config["linear_num_key_heads"]) * int(config["linear_key_head_dim"])
    value_heads = int(config["linear_num_value_heads"])
    value_dim = value_heads * int(config["linear_value_head_dim"])
    conv_dim, kernel = 2 * key_dim + value_dim, int(config["linear_conv_kernel_dim"])
    parameters = (
        width * (2 * key_dim + 2 * value_dim) + width * (2 * value_heads)
        + conv_dim * kernel + 2 * value_heads + int(config["linear_value_head_dim"])
        + value_dim * width
    )
    fixed_scalars = value_heads * int(config["linear_value_head_dim"]) * int(config["linear_key_head_dim"])
    fixed_scalars += conv_dim * (kernel - 1)
    return parameters, fixed_scalars


def _qwen_next_estimate(config: dict[str, object]) -> tuple[int, int, KVConfig, int]:
    width, layers = int(config["hidden_size"]), int(config["num_hidden_layers"])
    full_layers = layers // int(config["full_attention_interval"])
    linear_layers = layers - full_layers
    linear_params, fixed_scalars = _linear_mixer_params(config)
    experts, selected = int(config["num_experts"]), int(config["num_experts_per_tok"])
    expert_mlp = 3 * width * int(config["moe_intermediate_size"])
    shared_mlp = 3 * width * int(config["shared_expert_intermediate_size"])
    router = width * experts
    common = (
        _embeddings(config) + full_layers * _gqa_params(config) + linear_layers * linear_params
        + layers * (shared_mlp + router + 2 * width) + width
    )
    total = common + layers * experts * expert_mlp
    active = common + layers * selected * expert_mlp
    cache = KVConfig(
        str(config["_fixture_name"]), full_layers, int(config["num_attention_heads"]),
        int(config["head_dim"]), _dtype_bytes(config), kv_heads=int(config["num_key_value_heads"]),
    )
    return total, active, cache, linear_layers * fixed_scalars * _dtype_bytes(config)


def estimate_config(path: Path, context_tokens: int = 32_768) -> ArchitectureEstimate:
    config = json.loads(path.read_text(encoding="utf-8"))
    model_type = str(config["model_type"])
    if model_type == "llama":
        total, active, cache, fixed = _dense_estimate(config)
    elif model_type == "deepseek_v3":
        total, active, cache, fixed = _deepseek_estimate(config)
    elif model_type == "qwen3_next":
        total, active, cache, fixed = _qwen_next_estimate(config)
    else:
        raise ValueError(f"unsupported model_type: {model_type}")
    reported_total, reported_active = int(config["_reported_total_params"]), int(config["_reported_active_params"])
    kv_per_token = kv_bytes(cache, 1)
    return ArchitectureEstimate(
        name=str(config["_fixture_name"]), kind=str(config["_fixture_kind"]),
        total_params=total, active_params=active,
        reported_total_params=reported_total, reported_active_params=reported_active,
        total_error_percent=100 * (total - reported_total) / reported_total,
        active_fraction=active / total, kv_bytes_per_token=kv_per_token,
        fixed_state_bytes=fixed,
        context_state_gib_32k=(kv_bytes(cache, context_tokens) + fixed) / 2**30,
        source=str(config["_source"]), verified_on=str(config["_verified_on"]),
    )
