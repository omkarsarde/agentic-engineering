"""Deterministic AdamW-versus-Muon-like optimizer probe on Chapter 2 TinyGPT."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn


_GENERATED = Path(__file__).resolve().parent.parent / "ch02" / "_generated.py"
if "ch02_generated" in sys.modules:
    _ch02 = sys.modules["ch02_generated"]
else:
    _spec = importlib.util.spec_from_file_location("ch02_generated", _GENERATED)
    assert _spec is not None and _spec.loader is not None
    _ch02 = importlib.util.module_from_spec(_spec)
    sys.modules["ch02_generated"] = _ch02
    _spec.loader.exec_module(_ch02)

GPTConfig, TinyGPT = _ch02.GPTConfig, _ch02.TinyGPT


def wsd_multiplier(step: int, total_steps: int, *, warmup_fraction: float = 0.1, decay_fraction: float = 0.2) -> float:
    """Return a linear warmup, stable plateau, and linear decay multiplier."""

    if not 0 <= step < total_steps or total_steps < 2:
        raise ValueError("step must be inside a run of at least two steps")
    warmup = max(1, round(total_steps * warmup_fraction))
    decay = max(1, round(total_steps * decay_fraction))
    if step < warmup:
        return (step + 1) / warmup
    if step < total_steps - decay:
        return 1.0
    return max(0.0, (total_steps - step - 1) / decay)


def _zeroth_power(update: torch.Tensor, iterations: int = 4) -> torch.Tensor:
    """Approximate a semi-orthogonal matrix update with Newton--Schulz steps."""

    matrix = update.float()
    transposed = matrix.size(0) > matrix.size(1)
    if transposed:
        matrix = matrix.T
    matrix = matrix / (matrix.norm() + 1e-7)
    for _ in range(iterations):
        gram = matrix @ matrix.T
        matrix = 1.5 * matrix - 0.5 * gram @ matrix
    return matrix.T if transposed else matrix


class MuonAdam:
    """Muon-like matrix updates plus AdamW for embeddings, norms, and vectors."""

    def __init__(self, model: nn.Module, *, matrix_lr: float = 0.02, other_lr: float = 0.003) -> None:
        named = list(model.named_parameters())
        self.matrix = [
            parameter
            for name, parameter in named
            if parameter.ndim == 2 and "embedding" not in name and "lm_head" not in name
        ]
        matrix_ids = {id(parameter) for parameter in self.matrix}
        self.other = [parameter for _, parameter in named if id(parameter) not in matrix_ids]
        self.matrix_lr = matrix_lr
        self.momentum = {id(parameter): torch.zeros_like(parameter) for parameter in self.matrix}
        self.adam = torch.optim.AdamW(self.other, lr=other_lr, weight_decay=0.01)

    def zero_grad(self) -> None:
        """Clear gradients for both optimizer families."""

        self.adam.zero_grad(set_to_none=True)
        for parameter in self.matrix:
            parameter.grad = None

    @torch.no_grad()
    def step(self, multiplier: float) -> None:
        """Apply one scheduled matrix-polar update and one AdamW update."""

        for group in self.adam.param_groups:
            group["lr"] = 0.003 * multiplier
        self.adam.step()
        for parameter in self.matrix:
            if parameter.grad is None:
                continue
            buffer = self.momentum[id(parameter)]
            buffer.mul_(0.95).add_(parameter.grad)
            update = _zeroth_power(buffer)
            scale = math.sqrt(max(1.0, parameter.size(0) / parameter.size(1)))
            parameter.mul_(1 - 0.01 * self.matrix_lr * multiplier)
            parameter.add_(update.to(parameter.dtype), alpha=-self.matrix_lr * multiplier * scale)


def _batch(tokens: torch.Tensor, block: int, batch: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(tokens.numel() - block - 1, (batch,), generator=generator)
    inputs = torch.stack([tokens[start : start + block] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + block + 1] for start in starts])
    return inputs, targets


def compare_optimizers(*, steps: int = 60, seed: int = 61) -> list[dict[str, float | int | str]]:
    """Train identical TinyGPT initializations with AdamW and a Muon-like split."""

    if steps < 10:
        raise ValueError("the optimizer probe needs at least ten steps")
    config = GPTConfig(vocab_size=64, block_size=32, d_model=32, n_heads=4, n_layers=1, mlp_ratio=2)
    torch.manual_seed(seed)
    template_model = TinyGPT(config)
    template = {key: value.clone() for key, value in template_model.state_dict().items()}
    pattern = [(index * 7 + index // 5) % config.vocab_size for index in range(256)]
    tokens = torch.tensor(pattern * 80, dtype=torch.long)
    rows: list[dict[str, float | int | str]] = []
    for name in ("AdamW", "Muon-like"):
        model = TinyGPT(config)
        model.load_state_dict(template)
        adam = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
        muon = MuonAdam(model)
        generator = torch.Generator().manual_seed(seed + 1)
        for step in range(steps):
            inputs, targets = _batch(tokens, config.block_size, 8, generator)
            _, loss, _ = model(inputs, targets)
            assert loss is not None
            multiplier = wsd_multiplier(step, steps)
            (adam if name == "AdamW" else muon).zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if name == "AdamW":
                for group in adam.param_groups:
                    group["lr"] = 0.003 * multiplier
                adam.step()
            else:
                muon.step(multiplier)
            rows.append(
                {
                    "optimizer": name,
                    "step": step + 1,
                    "schedule_multiplier": multiplier,
                    "training_loss": float(loss.detach()),
                }
            )
    return rows
