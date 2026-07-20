"""Regression tests for the manuscript structure validator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "newbook_structure_validator", ROOT / "scripts" / "validate_book.py"
)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


# A minimal chapter that satisfies every v2 new-contract check.
GOOD_CHAPTER = """\
# Title {#sec-ch99}

Opening prose that names the artifact and its parts.

## Alpha

Prose. ![alpha](a.png)

```{python}
# @save
print(1)
```

More prose interpreting the output.

## Beta

Prose. ![beta](b.png)

## Gamma

Prose. ![gamma](c.png)

## Summary

A short synthesis of what was built, plus one forward pointer to keep going.

## Exercises

1. one
2. two
3. three
4. four
5. five
"""


def test_local_link_validator_checks_fragments(tmp_path: Path) -> None:
    """A present file with a misspelled section anchor is still a broken link."""
    target = tmp_path / "target.qmd"
    target.write_text(
        "# Explicit {#sec-explicit}\n\n## Automatic heading\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.qmd"
    source.write_text(
        "[explicit](target.qmd#sec-explicit)\n"
        "[automatic](target.qmd#automatic-heading)\n"
        "[broken](target.qmd#sec-misspelled)\n",
        encoding="utf-8",
    )

    assert VALIDATOR.validate_local_links(source) == [
        "broken local fragment: target.qmd#sec-misspelled"
    ]


def test_chapter_template_uses_the_v2_shape() -> None:
    """The starter template delegates numbering to Quarto and drops the [F] tag."""
    text = (ROOT / "templates" / "chapter-template.qmd").read_text(encoding="utf-8")
    frontmatter = text.split("---", 2)[1]
    assert "title:" not in frontmatter
    assert "# Chapter NN" not in text
    assert "# Exact Title {#sec-chNN}" in text
    # The deleted apparatus must not reappear in the exemplar.
    assert "route-b" not in text
    assert "## Summary" in text and "## Exercises" in text


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_redone_chapter_passes_new_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A well-formed redone chapter reports no errors from the new-contract tier."""
    monkeypatch.setattr(VALIDATOR, "REDONE", {"99-demo"})
    path = _write(tmp_path, "99-demo.qmd", GOOD_CHAPTER)
    errors, _ = VALIDATOR.validate_file(path)
    assert errors == []


def test_new_contract_checks_are_skipped_for_legacy_chapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy chapter (absent from redone.txt) is spared the new-contract tier."""
    monkeypatch.setattr(VALIDATOR, "REDONE", set())
    # Same chapter but stripped of Summary/Exercises and figures.
    legacy = "# Title {#sec-ch99}\n\nOpening.\n\n## Alpha\n\nProse.\n"
    path = _write(tmp_path, "99-demo.qmd", legacy)
    errors, _ = VALIDATOR.validate_file(path)
    assert errors == []


def test_redone_chapter_requires_summary_and_exercises_ending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping the Summary heading breaks the required closing pair."""
    monkeypatch.setattr(VALIDATOR, "REDONE", {"99-demo"})
    body = GOOD_CHAPTER.replace("## Summary\n\n", "")
    path = _write(tmp_path, "99-demo.qmd", body)
    errors, _ = VALIDATOR.validate_file(path)
    assert any("Summary" in error for error in errors)


def test_redone_chapter_flags_forbidden_apparatus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resurrected apparatus heading is an error under the new contract."""
    monkeypatch.setattr(VALIDATOR, "REDONE", {"99-demo"})
    body = GOOD_CHAPTER.replace("## Alpha", "## Contents\n\n- a\n- b\n\n## Alpha")
    path = _write(tmp_path, "99-demo.qmd", body)
    errors, _ = VALIDATOR.validate_file(path)
    assert any("forbidden deleted apparatus" in error for error in errors)


def test_redone_chapter_flags_elision_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ellipsis comment inside a code fence is flagged as an elision."""
    monkeypatch.setattr(VALIDATOR, "REDONE", {"99-demo"})
    body = GOOD_CHAPTER.replace("print(1)", "# ... rest omitted ...\nprint(1)")
    path = _write(tmp_path, "99-demo.qmd", body)
    errors, _ = VALIDATOR.validate_file(path)
    assert any("elision" in error for error in errors)


def test_redone_chapter_enforces_figure_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fewer than three figures fails the figure floor."""
    monkeypatch.setattr(VALIDATOR, "REDONE", {"99-demo"})
    body = GOOD_CHAPTER.replace("![gamma](c.png)", "no figure here")
    path = _write(tmp_path, "99-demo.qmd", body)
    errors, _ = VALIDATOR.validate_file(path)
    assert any("figures" in error for error in errors)


def test_docstring_check_rejects_bare_public_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long public function without an Args:/Returns: docstring is an error."""
    module = tmp_path / "_generated.py"
    module.write_text(
        "def build_pipeline(a, b, c):\n"
        "    x = a + b\n"
        "    y = x + c\n"
        "    z = y * 2\n"
        "    w = z - 1\n"
        "    return w\n",
        encoding="utf-8",
    )
    errors = VALIDATOR.docstring_errors(module)
    assert any("build_pipeline" in error for error in errors)
