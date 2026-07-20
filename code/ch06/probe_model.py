"""Chapter 2 TinyGPT adapter and activation-memory measurement for Chapter 6."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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


class TrainingTinyGPT(nn.Module):
    """Chapter 2 TinyGPT with optional block-level activation checkpointing."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.base = TinyGPT(config)
        # A parameter shared by a nested FSDP unit and the root is unsafe.
        self.base.lm_head.weight = nn.Parameter(self.base.lm_head.weight.detach().clone())

    def forward(
        self, tokens: torch.Tensor, targets: torch.Tensor, checkpoint_blocks: bool
    ) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.base.token_embedding(tokens) + self.base.position_embedding(positions)
        for block in self.base.blocks:
            run_block = lambda value, layer=block: layer(value)[0]
            x = checkpoint(run_block, x, use_reentrant=False) if checkpoint_blocks else run_block(x)
        logits = self.base.lm_head(self.base.final_norm(x))
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def saved_tensor_bytes(config: GPTConfig, checkpoint_blocks: bool, seed: int) -> int:
    """Measure tensors autograd saves for one unsharded forward/backward."""

    torch.manual_seed(seed)
    model = TrainingTinyGPT(config)
    tokens = torch.randint(config.vocab_size, (2, config.block_size))
    targets = torch.roll(tokens, -1, dims=1)
    saved = 0

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        nonlocal saved
        saved += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        model(tokens, targets, checkpoint_blocks).backward()
    return saved
