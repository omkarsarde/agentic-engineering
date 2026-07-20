# Auto-generated from chapters/11-customization.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F


def load_chapter_module(chapter: str, name: str):
    """Import a finished chapter's tangled module by path, not by install.

    Earlier chapters tangle their teaching code into ``code/chNN/_generated.py``.
    Rather than copy the Chapter 2 transformer, we walk up from the working
    directory to the book root and load it under a chapter-unique module name.

    Args:
        chapter: The code directory to load, e.g. ``"ch02"``.
        name: The module name to register it under.

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


_ch02 = load_chapter_module("ch02", "ch11_ch02")
TinyGPT, GPTConfig, generate = _ch02.TinyGPT, _ch02.GPTConfig, _ch02.generate


ALPHABET = "0123456789=|EAFS"
STOI = {c: i for i, c in enumerate(ALPHABET)}
VOCAB, NDIG, BLOCK, EQ = len(ALPHABET), 4, 24, STOI["="]

# echo, add-one, flip (nine's complement), and shift-by-position
OPS = {
    "E": lambda p: list(p),
    "A": lambda p: [(d + 1) % 10 for d in p],
    "F": lambda p: [9 - d for d in p],
    "S": lambda p: [(d + i) % 10 for i, d in enumerate(p)],
}


def make_example(rng: np.random.Generator, op: str) -> str:
    """Return one ``op<digits>=<answer>|`` example with random digits."""
    p = list(rng.integers(0, 10, size=NDIG))
    return op + "".join(map(str, p)) + "=" + "".join(map(str, OPS[op](p))) + "|"


def make_stream(ops, n: int, seed: int) -> Tensor:
    """Encode ``n`` examples of one or more operations into a token stream.

    Args:
        ops: A single operation tag or a list cycled across the examples.
        n: Number of examples to draw.
        seed: Seed for the digit generator, so a dataset is reproducible.

    Returns:
        A 1-D long tensor of token ids ready for windowed batching.
    """
    if isinstance(ops, str):
        ops = [ops]
    rng = np.random.default_rng(seed)
    text = "".join(make_example(rng, ops[i % len(ops)]) for i in range(n))
    return torch.tensor([STOI[c] for c in text], dtype=torch.long)


def batch(stream: Tensor, size: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    """Draw aligned (input, next-token) windows from a token stream.

    Args:
        stream: The 1-D token stream to sample from.
        size: Number of windows to draw.
        gen: Seeded generator choosing the window offsets.

    Returns:
        A tuple ``(inputs, targets)``, each shaped ``(size, BLOCK)``, with
        targets the inputs shifted one position later.
    """
    starts = torch.randint(0, stream.numel() - BLOCK - 1, (size,), generator=gen)
    x = torch.stack([stream[s:s + BLOCK] for s in starts])
    y = torch.stack([stream[s + 1:s + BLOCK + 1] for s in starts])
    return x, y


def fit(model, stream, steps, lr, size=64, seed=0, warmup=20):
    """Train a model (or its trainable subset) with warmup-then-flat AdamW.

    Only parameters with ``requires_grad`` receive updates, so the same loop
    trains a full model or just a set of LoRA matrices. Gradients are clipped
    to unit norm to keep the tiny model's steps stable.

    Args:
        model: Any module whose ``forward`` returns ``(logits, loss, cache)``.
        stream: The token stream to sample windows from.
        steps: Number of optimizer steps.
        lr: Peak learning rate after warmup.
        size: Batch size in windows.
        seed: Seed for the window sampler.
        warmup: Steps of linear warmup from zero to ``lr``.

    Returns:
        The list of per-step training losses.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    losses = []
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / warmup)
        x, y = batch(stream, size, gen)
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def accuracy(model, op, n=150, seed=999):
    """Return the per-character answer accuracy of ``model`` on operation ``op``.

    For each of ``n`` fresh prompts the model greedily generates four answer
    characters, which are compared position-by-position against the true
    transform. A value near 0.1 is chance; 1.0 is a solved task.

    Args:
        model: The model to score.
        op: The operation tag to test.
        n: Number of held-out prompts.
        seed: Seed for the evaluation prompts, kept apart from training seeds.

    Returns:
        The fraction of answer characters generated correctly.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    correct = total = 0
    for _ in range(n):
        p = list(rng.integers(0, 10, size=NDIG))
        prompt = op + "".join(map(str, p)) + "="
        ids = torch.tensor([[STOI[c] for c in prompt]])
        out = generate(model, ids, NDIG, temperature=0.0)
        got = [ALPHABET[i] for i in out[0, -NDIG:].tolist()]
        correct += sum(a == b for a, b in zip(got, map(str, OPS[op](p))))
        total += NDIG
    return correct / total


def small_gpt(d_model, n_layers, n_heads):
    """Build a fresh tinygpt over the tagged-transform vocabulary."""
    return TinyGPT(GPTConfig(vocab_size=VOCAB, block_size=BLOCK,
                             d_model=d_model, n_heads=n_heads, n_layers=n_layers))


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update ``(alpha/r)·B·A``.

    The wrapped layer's weights never receive gradients; only ``A`` and ``B``
    do. With ``B`` initialized to zero the layer starts as an exact copy of the
    base, so fine-tuning departs from trusted behavior rather than from noise.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        out_features, in_features = linear.weight.shape
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x) + self.scaling * (x @ self.A.t()) @ self.B.t()

    @property
    def delta(self) -> Tensor:
        """Return the realized dense update ``(alpha/r)·B·A`` for this layer."""
        return self.scaling * (self.B @ self.A)


TARGETS = ("qkv", "output", "up_gate", "down")


def attach_lora(model, rank=4, alpha=8):
    """Replace every targeted linear in each block with a ``LoRALinear``.

    Args:
        model: The model to adapt in place.
        rank: The bottleneck rank ``r`` of each adapter.
        alpha: The update scale; the effective multiplier is ``alpha/rank``.

    Returns:
        The same model, now carrying LoRA adapters on its targeted linears.
    """
    for block in model.blocks:
        for sub in ("attention", "mlp"):
            for name, mod in list(getattr(block, sub).named_children()):
                if name in TARGETS and isinstance(mod, nn.Linear):
                    setattr(getattr(block, sub), name, LoRALinear(mod, rank, alpha))
    return model


def lora_parameters(model):
    """Freeze the base and return only the trainable LoRA matrices.

    Args:
        model: A model already carrying ``LoRALinear`` layers.

    Returns:
        The list of trainable ``A`` and ``B`` parameters, so an optimizer and a
        parameter count can both see exactly what LoRA moves.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
            trainable += [m.A, m.B]
    return trainable


NF4 = torch.tensor(
    [-1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734,
     -0.0910500, 0.0, 0.0795803, 0.1609302, 0.2461123, 0.3379152,
     0.4407098, 0.5626170, 0.7229568, 1.0]
)


def nf4_quantize(weight: Tensor, group_size: int = 32) -> Tensor:
    """Round a weight tensor to the NF4 codebook, per absmax-scaled group.

    Each contiguous group of ``group_size`` values is divided by its largest
    magnitude, snapped to the nearest of the sixteen NF4 code points, and
    rescaled. Smaller groups track local scale better at the cost of more
    stored scales — the quantization-granularity trade-off of @sec-ch10.

    Args:
        weight: The full-precision weight tensor to quantize.
        group_size: Number of consecutive values sharing one absmax scale.

    Returns:
        The dequantized reconstruction, the same shape as ``weight``.
    """
    flat = weight.flatten()
    restored = torch.empty_like(flat)
    for s in range(0, flat.numel(), group_size):
        group = flat[s:s + group_size]
        scale = group.abs().max().clamp_min(1e-12)
        codes = (group / scale).unsqueeze(1).sub(NF4).abs().argmin(1)
        restored[s:s + group_size] = NF4[codes] * scale
    return restored.view_as(weight)


def answer_positions(x: Tensor) -> Tensor:
    """Mark the ``NDIG`` positions after each ``=`` — where the answer is written.

    Distillation targets these positions only; the prompt digits are random and
    carry no transferable signal.

    Args:
        x: A batch of token id windows.

    Returns:
        A boolean mask, True on answer-token positions.
    """
    mask = torch.zeros_like(x, dtype=torch.bool)
    for b in range(x.size(0)):
        row = x[b].tolist()
        for t in range(x.size(1)):
            if row[t] == EQ:
                for k in range(NDIG):
                    if t + k < x.size(1):
                        mask[b, t + k] = True
    return mask


@torch.no_grad()
def teacher_completions(teacher, op, n, seed):
    """Build a training stream from the teacher's own greedy answers.

    This is sequence-level distillation: the student will imitate whatever the
    teacher writes, right or wrong, with no access to its distribution.

    Args:
        teacher: The model whose completions become labels.
        op: The operation tag to prompt with.
        n: Number of prompts to generate answers for.
        seed: Seed for the prompts.

    Returns:
        A token stream of ``op<digits>=<teacher answer>|`` examples.
    """
    rng = np.random.default_rng(seed)
    teacher.eval()
    parts = []
    for _ in range(n):
        p = list(rng.integers(0, 10, size=NDIG))
        prompt = op + "".join(map(str, p)) + "="
        out = generate(teacher, torch.tensor([[STOI[c] for c in prompt]]), NDIG, temperature=0.0)
        parts.append("".join(ALPHABET[i] for i in out[0].tolist()) + "|")
    return torch.tensor([STOI[c] for c in "".join(parts)], dtype=torch.long)


def distill_logits(student, teacher, stream, steps, lr, tau=2.0, alpha=0.7, size=64, seed=7, warmup=20):
    """Train a student on the teacher's softened answer distribution plus labels.

    The loss blends the temperature-scaled KL of @eq-ch11-kd (weight ``alpha``)
    with the ordinary hard-label cross-entropy, both restricted to answer
    positions. The ``tau**2`` factor keeps the soft term's gradient scale
    comparable to the hard term's.

    Args:
        student: The model being trained.
        teacher: The frozen teacher providing target logits.
        stream: The (small) training stream of real examples.
        steps: Number of optimizer steps.
        lr: Peak learning rate.
        tau: Softmax temperature applied to both models.
        alpha: Weight on the soft KL term versus the hard term.
        size: Batch size.
        seed: Seed for the window sampler.
        warmup: Linear-warmup steps.

    Returns:
        The list of per-step blended losses.
    """
    opt = torch.optim.AdamW(student.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    teacher.eval()
    losses = []
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * min(1.0, (step + 1) / warmup)
        x, y = batch(stream, size, gen)
        with torch.no_grad():
            t_logits, _, _ = teacher(x)
        s_logits, _, _ = student(x)
        m = answer_positions(x)
        p_t = (t_logits[m] / tau).softmax(-1)
        logp_s = (s_logits[m] / tau).log_softmax(-1)
        soft = -(p_t * logp_s).sum(-1).mean() * tau * tau
        hard = F.cross_entropy(s_logits[m], y[m])
        loss = alpha * soft + (1 - alpha) * hard
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def collect_deltas(model) -> dict:
    """Collect every LoRA layer's realized dense update as a task vector.

    Args:
        model: A model carrying ``LoRALinear`` adapters.

    Returns:
        A dict from ``(block_index, sublayer, name)`` to the dense delta
        ``(alpha/r)·B·A`` for that layer — the per-layer task vector to merge.
    """
    out = {}
    for i, block in enumerate(model.blocks):
        for sub in ("attention", "mlp"):
            for name, mod in getattr(block, sub).named_children():
                if isinstance(mod, LoRALinear):
                    out[(i, sub, name)] = mod.delta.detach()
    return out


def ties_merge(delta_a: Tensor, delta_b: Tensor, wa: float, wb: float, density: float = 0.6) -> Tensor:
    """Merge two weighted deltas by trim, sign-elect, and disjoint average.

    Each delta keeps only its top ``density`` fraction by magnitude; a sign is
    elected per coordinate from the weighted sum; only the survivors whose sign
    matches the elected sign are averaged. Opposed updates that would cancel in
    a plain average are dropped instead.

    Args:
        delta_a: The first task's dense update.
        delta_b: The second task's dense update.
        wa: Weight on the first update.
        wb: Weight on the second update.
        density: Fraction of each delta's coordinates to keep after trimming.

    Returns:
        The merged dense update, the shape of one input delta.
    """
    trimmed = []
    for d, w in ((delta_a, wa), (delta_b, wb)):
        k = max(1, int((1 - density) * d.numel()))
        cutoff = d.abs().flatten().kthvalue(k).values
        trimmed.append(torch.where(d.abs() >= cutoff, w * d, torch.zeros_like(d)))
    stack = torch.stack(trimmed)
    elected = torch.sign(stack.sum(0))
    keep = (torch.sign(stack) == elected).float()
    count = keep.sum(0).clamp_min(1)
    return (stack * keep).sum(0) / count


def merged_model(base_model, deltas_a, deltas_b, wa, wb, method):
    """Add a linear or TIES combination of two task vectors onto a base copy.

    Args:
        base_model: The shared base to merge onto.
        deltas_a: The first adapter's per-layer deltas from ``collect_deltas``.
        deltas_b: The second adapter's per-layer deltas.
        wa: Weight on the first task vector.
        wb: Weight on the second task vector.
        method: ``"linear"`` for weighted sum, ``"ties"`` for the TIES rule.

    Returns:
        A new model whose targeted weights carry the merged update.
    """
    m = copy.deepcopy(base_model)
    with torch.no_grad():
        for i, block in enumerate(m.blocks):
            for sub in ("attention", "mlp"):
                for name, mod in getattr(block, sub).named_children():
                    key = (i, sub, name)
                    if key in deltas_a:
                        d = (ties_merge(deltas_a[key], deltas_b[key], wa, wb)
                             if method == "ties" else wa * deltas_a[key] + wb * deltas_b[key])
                        mod.weight.data = mod.weight.data + d
    return m


def release_gate(baseline, candidate, task_op, regression_ops, min_gain=0.30, max_drop=0.05):
    """Decide whether a candidate ships, given a task gain and regression budget.

    Implements @eq-ch11-gate: the candidate must improve the task set by at
    least ``min_gain`` and must not drop any regression capability by more than
    ``max_drop``. A large task gain cannot buy an unacceptable regression.

    Args:
        baseline: ``{op: accuracy}`` for the pre-tuning model.
        candidate: ``{op: accuracy}`` for the tuned model.
        task_op: The operation whose gain is required.
        regression_ops: Operations that must be preserved.
        min_gain: Minimum required task-set gain.
        max_drop: Maximum tolerated drop on any regression op.

    Returns:
        A dict with the measured gain, the worst drop, and the ship decision.
    """
    gain = candidate[task_op] - baseline[task_op]
    worst_drop = max(baseline[o] - candidate[o] for o in regression_ops)
    return {"gain": round(gain, 3), "worst_regression_drop": round(worst_drop, 3),
            "ship": bool(gain >= min_gain and worst_drop <= max_drop)}
