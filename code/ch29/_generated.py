# Auto-generated from chapters/29-multimodal-vlm-documents-gui.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def visual_token_count(height: int, width: int, patch: int) -> int:
    """Return the number of ViT patches for one image, from @eq-ch29-patches."""
    if height % patch or width % patch:
        raise ValueError("image side is not divisible by the patch size")
    return (height // patch) * (width // patch)


def patchify(image: np.ndarray, patch: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Cut an image into flattened square patches in raster order.

    This is the whole input tokenizer of a Vision Transformer: a
    ``(H, W, C)`` image becomes an ``(N, P*P*C)`` matrix, one row per patch,
    which a single learned linear layer then projects to the encoder width.

    Args:
        image: Pixel array shaped ``(H, W)`` or ``(H, W, C)``.
        patch: Side length ``P`` of each square patch; must divide both sides.

    Returns:
        A tuple ``(patches, grid)`` where ``patches`` is the ``(N, P*P*C)``
        matrix and ``grid`` is the ``(rows, cols)`` patch layout, so that
        ``rows * cols == N``.
    """
    height, width = image.shape[:2]
    grid_h, grid_w = height // patch, width // patch
    rows = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            tile = image[gy * patch:(gy + 1) * patch, gx * patch:(gx + 1) * patch]
            rows.append(np.asarray(tile, dtype=np.float32).reshape(-1))
    return np.stack(rows), (grid_h, grid_w)


def toy_chart(side: int = 224) -> np.ndarray:
    """Draw a constructed four-bar revenue chart with one tall tinted bar.

    Args:
        side: Height and width of the square RGB image, in pixels.

    Returns:
        A ``(side, side, 3)`` uint8 array with three blue bars, one taller
        red bar, and a baseline — a miniature of the analyst's page-17 figure.
    """
    img = np.full((side, side, 3), 245, dtype=np.uint8)
    heights, base, x = [70, 120, 90, 175], side - 30, 26
    for i, h in enumerate(heights):
        color = (200, 90, 70) if i == 3 else (70, 110, 160)
        img[base - h:base, x:x + 34] = color
        x += 54
    img[base:base + 3, :] = 40
    return img


def info_nce_loss(similarity: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric CLIP loss: each matched pair must win its row and column."""
    targets = torch.arange(similarity.size(0))
    logits = similarity / tau
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


class DualEncoder(nn.Module):
    """Two linear encoders projecting images and captions into one space.

    The stand-in for CLIP's vision and text towers. Each modality arrives in
    its own raw feature space; the encoders learn projections whose cosine
    similarity is high for a matched image-caption pair and low otherwise.

    Args:
        img_dim: Width of the raw image feature vector.
        cap_dim: Width of the raw caption feature vector.
        shared_dim: Width of the joint embedding space both map into.
    """

    def __init__(self, img_dim: int, cap_dim: int, shared_dim: int) -> None:
        super().__init__()
        self.img = nn.Linear(img_dim, shared_dim, bias=False)
        self.cap = nn.Linear(cap_dim, shared_dim, bias=False)

    def forward(self, images: torch.Tensor, captions: torch.Tensor):
        """Return L2-normalized image and caption embeddings."""
        return (F.normalize(self.img(images), dim=-1),
                F.normalize(self.cap(captions), dim=-1))


def make_pairs(count: int, shared_dim: int, img_dim: int, cap_dim: int, seed: int):
    """Build synthetic image-caption pairs sharing a hidden latent.

    Each pair ``i`` is generated from one latent vector ``z_i`` seen through
    two different random linear views plus small noise, so a matched pair is
    only recoverable by a model that learns to invert both views into a
    common space — exactly the job contrastive training does.

    Args:
        count: Number of pairs (the batch size ``B``).
        shared_dim: Dimensionality of the hidden latent.
        img_dim: Raw image feature width.
        cap_dim: Raw caption feature width.
        seed: Seed for the local generator, so pairs are reproducible.

    Returns:
        A tuple ``(images, captions)`` of raw feature matrices shaped
        ``(count, img_dim)`` and ``(count, cap_dim)``.
    """
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(count, shared_dim, generator=g)
    images = z @ torch.randn(shared_dim, img_dim, generator=g) + 0.05 * torch.randn(count, img_dim, generator=g)
    captions = z @ torch.randn(shared_dim, cap_dim, generator=g) + 0.05 * torch.randn(count, cap_dim, generator=g)
    return images, captions


def image_token_bill(tiles: int, tokens_per_tile: int = 256, thumbnail: int = 64) -> int:
    """Return per-page visual tokens under dynamic tiling, from @eq-ch29-bill."""
    if tiles <= 0:
        raise ValueError("tiles must be positive")
    return tiles * tokens_per_tile + thumbnail


def dollars(tokens: int, price_per_mtok: float) -> float:
    """Convert a token count to dollars at a per-million-token input price."""
    return tokens / 1e6 * price_per_mtok


from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

FIELDS = ("invoice_id", "vendor", "total", "due_date", "chart_peak")
CELL_W, CELL_H = 26, 34
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
FONT = ImageFont.load_default(size=28)


@dataclass(frozen=True)
class Box:
    """An axis-aligned evidence rectangle in page-pixel coordinates."""
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class Page:
    """A synthetic invoice with gold field values and their evidence boxes."""
    page_id: str
    width: int
    height: int
    gold: dict
    regions: dict


def _draw_text(draw: "ImageDraw.ImageDraw", text: str, x: int, y: int) -> None:
    for i, ch in enumerate(text):
        bbox = draw.textbbox((0, 0), ch, font=FONT)
        w = bbox[2] - bbox[0]
        draw.text((x + i * CELL_W + (CELL_W - w) // 2 - bbox[0], y), ch, fill=0, font=FONT)


def make_pages(count: int = 12) -> list[Page]:
    """Build source-traceable invoices whose gold values come from the generator.

    Every field's value and evidence box is emitted by construction, so the
    ground truth is exact rather than hand-transcribed. Values vary per page
    (rising totals, distinct vendors, shifting due dates) to give the reader
    real work.

    Args:
        count: Number of pages to generate.

    Returns:
        A list of ``Page`` objects, each with five gold fields and boxes.
    """
    vendors = ("NORTHWIND", "CONTOSO", "TAILSPIN")
    layout = {"invoice_id": (60, 60), "vendor": (60, 130), "total": (60, 210),
              "due_date": (360, 60), "chart_peak": (360, 210)}
    pages = []
    for i in range(count):
        gold = {"invoice_id": f"INV-{1040 + i}", "vendor": vendors[i % 3],
                "total": f"{125 + 17 * i}.00", "due_date": f"2026-08-{10 + i:02d}",
                "chart_peak": ("Q4", "Q3", "Q2")[i % 3]}
        regions = {f: Box(x, y, x + len(gold[f]) * CELL_W, y + CELL_H)
                   for f, (x, y) in layout.items()}
        pages.append(Page(f"page-{i:02d}", 640, 300, gold, regions))
    return pages


def render_page(page: Page) -> "Image.Image":
    """Rasterize a page's gold fields into a grayscale image.

    Args:
        page: The page whose gold values and boxes fix what is drawn where.

    Returns:
        A grayscale ``PIL`` image with each field drawn at its box origin.
    """
    img = Image.new("L", (page.width, page.height), 255)
    draw = ImageDraw.Draw(img)
    for f in FIELDS:
        _draw_text(draw, page.gold[f], page.regions[f].x0, page.regions[f].y0)
    return img


def _templates() -> dict:
    templates = {}
    for ch in ALPHABET:
        cell = Image.new("L", (CELL_W, CELL_H), 255)
        d = ImageDraw.Draw(cell)
        bbox = d.textbbox((0, 0), ch, font=FONT)
        d.text(((CELL_W - (bbox[2] - bbox[0])) // 2 - bbox[0], 0), ch, fill=0, font=FONT)
        templates[ch] = np.asarray(cell, dtype=np.float32) / 255.0
    return templates


TEMPLATES = _templates()
SCALE = {1: 0.38, 4: 0.42, 9: 0.46, 16: 0.60}   # effective resolution per tile budget


def _degrade(img: "Image.Image", scale: float) -> "Image.Image":
    small = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                       Image.BILINEAR)
    return small.resize((img.width, img.height), Image.BILINEAR)


def _read_cell(cell: np.ndarray) -> str:
    return min(TEMPLATES, key=lambda ch: float(((cell - TEMPLATES[ch]) ** 2).sum()))


def read_fields(image: "Image.Image", page: Page, tiles: int) -> dict:
    """Recognize each field from a resolution-degraded page image.

    The page is downscaled to the effective resolution the tile budget buys,
    then upscaled back; each field's box is split into fixed-width character
    cells and every cell is matched to its nearest reference glyph. Low tile
    counts blur thin strokes and produce genuine confusions (T vs I, Q vs O).

    Args:
        image: The rendered (undegraded) page image.
        page: The page whose boxes locate each field's cells.
        tiles: Tile budget; indexes ``SCALE`` for the degradation factor.

    Returns:
        A dict mapping each field to ``{"value": str, "box": {x0,y0,x1,y1}}``.
    """
    degraded = _degrade(image, SCALE[tiles])
    out = {}
    for f in FIELDS:
        box = page.regions[f]
        n_cells = round((box.x1 - box.x0) / CELL_W)
        chars = []
        for c in range(n_cells):
            x0 = box.x0 + c * CELL_W
            crop = degraded.crop((x0, box.y0, x0 + CELL_W, box.y0 + CELL_H))
            chars.append(_read_cell(np.asarray(crop, dtype=np.float32) / 255.0))
        out[f] = {"value": "".join(chars), "box": box.__dict__}
    return out


def validate_extraction(extraction: dict, page: Page) -> None:
    """Reject a result missing fields, mistyped, or citing an off-page box.

    Args:
        extraction: The reader's output, field -> ``{value, box}``.
        page: The page the extraction claims to describe.

    Raises:
        ValueError: If the field set is wrong, a box is malformed, or a box
            escapes the page bounds.
        TypeError: If a value is not a string.
    """
    if set(extraction) != set(FIELDS):
        raise ValueError("extraction does not match the field schema")
    for f, item in extraction.items():
        if not isinstance(item.get("value"), str):
            raise TypeError(f"{f} value must be a string")
        b = item.get("box", {})
        if set(b) != {"x0", "y0", "x1", "y1"}:
            raise ValueError(f"{f} lacks a well-formed evidence box")
        if not (0 <= b["x0"] < b["x1"] <= page.width and 0 <= b["y0"] < b["y1"] <= page.height):
            raise ValueError(f"{f} evidence box escapes the page")


def score_fields(extraction: dict, page: Page) -> tuple[int, int]:
    """Return (exact-match count, field count) after case/space normalization."""
    correct = sum(extraction[f]["value"].strip().casefold() == page.gold[f].strip().casefold()
                  for f in FIELDS)
    return correct, len(FIELDS)


def evaluate(tiles: int, pages: list[Page]) -> dict:
    """Extract and score every page at one tile budget.

    Args:
        tiles: The tile budget passed to the reader and the token bill.
        pages: The evaluation set.

    Returns:
        A row with the tile budget, per-page image tokens, exact-field count,
        total fields, and field accuracy.
    """
    correct = total = 0
    for page in pages:
        extraction = read_fields(render_page(page), page, tiles)
        validate_extraction(extraction, page)
        c, t = score_fields(extraction, page)
        correct, total = correct + c, total + t
    return {"tiles": tiles, "tokens_per_page": image_token_bill(tiles),
            "correct": correct, "total": total, "accuracy": round(correct / total, 4)}


def _normalize(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def pooled_score(query: list, document: list) -> float:
    """Score a query and document by the cosine of their mean-pooled vectors."""
    q = _normalize(np.mean([_normalize(v) for v in query], axis=0))
    d = _normalize(np.mean([_normalize(v) for v in document], axis=0))
    return float(q @ d)


def maxsim(query: list, document: list) -> float:
    """Late-interaction score: each query vector takes its best document match.

    Implements @eq-ch29-maxsim over L2-normalized vectors. Unlike pooling, a
    query term that matches one small region contributes its full similarity,
    so a decisive patch is not averaged away — and a page cannot win on one
    lucky patch while the rest of the query finds no home.

    Args:
        query: Query token vectors.
        document: Document patch vectors.

    Returns:
        The summed best-match similarity ``S(Q, D)``.
    """
    qn, dn = [_normalize(v) for v in query], [_normalize(v) for v in document]
    return float(sum(max(q @ d for d in dn) for q in qn))


def to_pixels(u: int, v: int, grid: int, screen_w: int, screen_h: int,
              pad_x: int = 0, pad_y: int = 0) -> tuple[int, int]:
    """Map a normalized model coordinate to a screen pixel, undoing letterbox pad.

    The model emits ``(u, v)`` on a ``grid``-wide square it was shown. If the
    original screen was letterboxed into that square, the content occupies
    only ``grid - 2*pad`` of it, so the pad must be removed and the remainder
    rescaled to the true screen. Passing ``pad_x=pad_y=0`` is the common bug:
    it treats the padded square as if it were the whole screen.

    Args:
        u: Horizontal model coordinate on the ``0..grid`` scale.
        v: Vertical model coordinate on the ``0..grid`` scale.
        grid: Width of the square grid the model emits on (e.g. 1000).
        screen_w: True screen width in pixels.
        screen_h: True screen height in pixels.
        pad_x: Horizontal letterbox pad, in grid units (0 if none).
        pad_y: Vertical letterbox pad, in grid units (0 if none).

    Returns:
        The ``(x, y)`` screen pixel the coordinate points at.
    """
    fx = (u - pad_x) / (grid - 2 * pad_x)
    fy = (v - pad_y) / (grid - 2 * pad_y)
    return round(fx * screen_w), round(fy * screen_h)


import hashlib
import json
from dataclasses import replace


@dataclass(frozen=True)
class Element:
    """One accessibility-tree control: an id, a label, a box, a risk flag."""
    element_id: str
    label: str
    box: tuple
    destructive: bool = False


@dataclass(frozen=True)
class Screen:
    """A captured screen: a revision, size, and its elements, with a digest."""
    revision: int
    width: int
    height: int
    elements: tuple

    @property
    def digest(self) -> str:
        payload = {"revision": self.revision, "elements": [e.__dict__ for e in self.elements]}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class ClickProposal:
    """A model's proposed click, bound to the screen digest it was made on."""
    screen_digest: str
    element_id: str
    x: int
    y: int
    destructive: bool


def initial_screen() -> Screen:
    """Build the demo screen.

    Returns:
        A revision-1 ``Screen`` with three controls: a safe "Save draft", a
        destructive "Delete account", and an "Email address" text field.
    """
    return Screen(1, 1000, 700, (
        Element("save", "Save draft", (700, 610, 820, 670)),
        Element("delete", "Delete account", (830, 610, 980, 670), True),
        Element("email", "Email address", (120, 180, 620, 230)),
    ))


def ground(screen: Screen, label: str, u: int, v: int) -> ClickProposal:
    """Resolve a label to a click proposal via the coordinate mapping.

    Stands in for a VLM: it finds the uniquely matching element, maps the
    model's normalized ``(u, v)`` to a screen pixel with ``to_pixels``, and
    binds the result to the current screen digest so a later gate can detect
    a stale screen.

    Args:
        screen: The screen the model observed.
        label: The element label the model chose.
        u: Normalized horizontal coordinate on the 0..1000 grid.
        v: Normalized vertical coordinate on the 0..1000 grid.

    Returns:
        A ``ClickProposal`` bound to ``screen.digest``.

    Raises:
        LookupError: If the label matches zero or several elements.
    """
    matches = [e for e in screen.elements if e.label.casefold() == label.casefold()]
    if len(matches) != 1:
        raise LookupError("label is missing or ambiguous")
    element = matches[0]
    x, y = to_pixels(u, v, 1000, screen.width, screen.height)
    return ClickProposal(screen.digest, element.element_id, x, y, element.destructive)


def authorize(screen: Screen, proposal: ClickProposal, confirmed: bool = False) -> str:
    """Re-validate a proposal against the live screen before any click.

    The trust boundary: it rejects a proposal made on a stale screen, a
    coordinate that fell outside its named target, or a destructive action
    without confirmation. Confirmation authorizes one semantic action on one
    screen state — not a coordinate forever.

    Args:
        screen: The current, authoritative screen at click time.
        proposal: The click proposed after grounding.
        confirmed: Whether a human confirmed a destructive action.

    Returns:
        ``"allow"``, ``"review:confirmation_required"``, or a ``"deny:..."``
        reason string.
    """
    if proposal.screen_digest != screen.digest:
        return "deny:stale_screen"
    element = next((e for e in screen.elements if e.element_id == proposal.element_id), None)
    if element is None:
        return "deny:missing_target"
    x0, y0, x1, y1 = element.box
    if not (x0 <= proposal.x <= x1 and y0 <= proposal.y <= y1):
        return "deny:coordinates_outside_target"
    if element.destructive and not confirmed:
        return "review:confirmation_required"
    return "allow"


def mutate(screen: Screen) -> Screen:
    """Move the delete control and bump the revision, staling old proposals."""
    moved = tuple(replace(e, box=(40, 610, 190, 670)) if e.element_id == "delete" else e
                  for e in screen.elements)
    return replace(screen, revision=screen.revision + 1, elements=moved)


def pope_metrics(answers: list, gold: list) -> dict:
    """Score object-presence answers the POPE way: accuracy plus its structure.

    Accuracy alone hides the characteristic VLM failure. A yes-biased model
    keeps high recall (it rarely misses a present object) while precision and
    the yes-rate expose the confabulation: it also says yes to absent ones.

    Args:
        answers: Model answers, each ``"yes"`` or ``"no"``.
        gold: Ground-truth presence, aligned with ``answers``.

    Returns:
        A dict with ``accuracy``, ``precision``, ``recall``, and ``yes_rate``.
    """
    tp = sum(a == "yes" and g == "yes" for a, g in zip(answers, gold))
    fp = sum(a == "yes" and g == "no" for a, g in zip(answers, gold))
    fn = sum(a == "no" and g == "yes" for a, g in zip(answers, gold))
    return {"accuracy": round(sum(a == g for a, g in zip(answers, gold)) / len(gold), 3),
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
            "yes_rate": round(sum(a == "yes" for a in answers) / len(answers), 3)}
