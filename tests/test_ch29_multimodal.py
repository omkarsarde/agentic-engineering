"""Executable invariants for Chapter 29's multimodal spine.

Imports only the tangled module ``code/ch29/_generated.py`` (the chapter's
``# @save`` cells in document order) under a chapter-unique module name, so it
cannot collide with any other chapter's generated module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "ch29_generated", ROOT / "code" / "ch29" / "_generated.py"
)
ch29 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["ch29_generated"] = ch29  # dataclasses resolve annotations via sys.modules
_SPEC.loader.exec_module(ch29)


# --- ViT patchify -----------------------------------------------------------

def test_patch_count_matches_formula() -> None:
    image = ch29.toy_chart(224)
    patches, grid = ch29.patchify(image, 16)
    assert grid == (14, 14)
    assert patches.shape == (196, 16 * 16 * 3)
    assert patches.shape[0] == ch29.visual_token_count(224, 224, 16)


def test_patchify_rejects_indivisible_sides() -> None:
    try:
        ch29.visual_token_count(225, 224, 16)
    except ValueError:
        return
    raise AssertionError("indivisible image side was accepted")


# --- Contrastive alignment --------------------------------------------------

def test_contrastive_training_drives_similarity_to_the_diagonal() -> None:
    torch.manual_seed(0)
    images, captions = ch29.make_pairs(6, shared_dim=6, img_dim=16, cap_dim=12, seed=1)
    model = ch29.DualEncoder(16, 12, 8)
    vi, vt = model(images, captions)
    before = int((torch.as_tensor(vi @ vt.t()).argmax(1) == torch.arange(6)).sum())

    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(300):
        vi, vt = model(images, captions)
        loss = ch29.info_nce_loss(vi @ vt.t(), 0.1)
        opt.zero_grad(); loss.backward(); opt.step()

    vi, vt = model(images, captions)
    after = int(((vi @ vt.t()).argmax(1) == torch.arange(6)).sum())
    assert before < 6 and after == 6
    assert float(ch29.info_nce_loss(vi @ vt.t(), 0.1)) < 0.05


# --- Token bill -------------------------------------------------------------

def test_image_token_bill_is_linear_in_tiles() -> None:
    assert ch29.image_token_bill(1) == 320
    assert ch29.image_token_bill(4) == 1088
    assert ch29.image_token_bill(16) == 4160
    assert ch29.dollars(83200, 3.0) == 0.2496


# --- Document extraction ----------------------------------------------------

def test_extraction_accuracy_rises_then_plateaus() -> None:
    pages = ch29.make_pages(12)
    rows = [ch29.evaluate(n, pages) for n in (1, 4, 9, 16)]
    accs = [r["accuracy"] for r in rows]
    assert accs[0] < accs[1] < accs[2]          # strictly improving up to the elbow
    assert accs[2] == accs[3] == 1.0            # plateau at the ceiling
    assert rows[3]["tokens_per_page"] > rows[2]["tokens_per_page"]  # cost still rising


def test_every_extracted_field_carries_an_in_page_box() -> None:
    page = ch29.make_pages(1)[0]
    extraction = ch29.read_fields(ch29.render_page(page), page, 16)
    ch29.validate_extraction(extraction, page)  # must not raise
    extraction["total"]["box"]["x1"] = page.width + 1
    try:
        ch29.validate_extraction(extraction, page)
    except ValueError:
        return
    raise AssertionError("an off-page evidence box was accepted")


def test_low_resolution_produces_real_misreads() -> None:
    page = ch29.make_pages(1)[0]
    low = ch29.read_fields(ch29.render_page(page), page, 1)
    correct, total = ch29.score_fields(low, page)
    assert 0 < correct < total  # some fields readable, some genuinely wrong under blur


# --- MaxSim vs pooling ------------------------------------------------------

def test_maxsim_keeps_a_region_that_pooling_drowns() -> None:
    query = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    relevant = [query[0], query[1]] + [(0.0, 0.0, 1.0)] * 8
    flooded = [(0.7, 0.7, 0.0)] * 10
    assert ch29.pooled_score(query, flooded) > ch29.pooled_score(query, relevant)
    assert ch29.maxsim(query, relevant) > ch29.maxsim(query, flooded)


# --- Coordinate mapping -----------------------------------------------------

def test_letterbox_pad_moves_the_click() -> None:
    right = ch29.to_pixels(760, 700, 1000, 1920, 1080, pad_y=219)
    wrong = ch29.to_pixels(760, 700, 1000, 1920, 1080)
    assert right[1] == 924 and wrong[1] == 756
    assert wrong[1] - right[1] == -168  # forgetting the pad lands 168px too high


# --- GUI gate ---------------------------------------------------------------

def test_gate_allows_safe_and_reviews_destructive() -> None:
    screen = ch29.initial_screen()
    save = ch29.ground(screen, "Save draft", u=760, v=914)
    dele = ch29.ground(screen, "Delete account", u=905, v=914)
    assert ch29.authorize(screen, save) == "allow"
    assert ch29.authorize(screen, dele) == "review:confirmation_required"
    assert ch29.authorize(screen, dele, confirmed=True) == "allow"


def test_gate_denies_a_stale_confirmed_proposal() -> None:
    screen = ch29.initial_screen()
    dele = ch29.ground(screen, "Delete account", u=905, v=914)
    assert ch29.authorize(ch29.mutate(screen), dele, confirmed=True) == "deny:stale_screen"


def test_gate_denies_a_coordinate_outside_its_named_target() -> None:
    screen = ch29.initial_screen()
    save = ch29.ground(screen, "Save draft", u=760, v=914)
    off = ch29.ClickProposal(save.screen_digest, save.element_id, 0, 0, False)
    assert ch29.authorize(screen, off) == "deny:coordinates_outside_target"


def test_grounding_rejects_an_ambiguous_label() -> None:
    screen = ch29.initial_screen()
    try:
        ch29.ground(screen, "Nonexistent", u=500, v=500)
    except LookupError:
        return
    raise AssertionError("a missing label was grounded")


# --- POPE probe -------------------------------------------------------------

def test_pope_exposes_a_yes_biased_model() -> None:
    gold = ["yes", "no", "yes", "no", "yes", "no", "yes", "no"]
    biased = ["yes", "yes", "yes", "yes", "yes", "no", "yes", "yes"]
    m = ch29.pope_metrics(biased, gold)
    assert m["recall"] == 1.0            # never misses a present object
    assert m["precision"] < 0.75         # but confabulates absent ones
    assert m["yes_rate"] > 0.5           # the tell-tale yes-bias
