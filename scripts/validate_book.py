"""Validate the structural and pedagogical contracts of the Quarto book.

The book is being ported chapter by chapter to a d2l.ai-style editorial contract
(see EDITORIAL-CONTRACT.md v2).  While the port is under way most chapters still
follow the older apparatus-heavy contract, so the validator runs two tiers of
checks:

* Hygiene checks -- encoding, one-H1/no-title numbering, duplicate H2s, the four
  forbidden hedges, ``$`` currency escaping, dated-fact ``.landscape-2026``
  callouts, and the whole broken-link / broken-fragment checker -- run against
  *every* manuscript file, legacy or redone.
* New-contract checks -- the required ``## Summary`` then ``## Exercises``
  ending, the teaching-H2 and figure floors, at least one executable
  ``{python}`` cell, the anti-elision and forbidden-apparatus checks, the
  exercises band, the prose-word floor, and the teaching-docstring check on the
  tangled module -- run only against chapters whose file stem is listed in
  ``scripts/redone.txt``.

Legacy chapters therefore see only the hygiene checks, which keeps ``--strict``
green while chapters are migrated one at a time.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "_quarto.yml"
REDONE_FILE = ROOT / "scripts" / "redone.txt"

# The two H2 sections a redone chapter must carry, in this order, as its final
# two headings.
CLOSING = ("summary", "exercises")

# Hedges the contract forbids everywhere.  (The old house-voice phrases that
# policed the deleted route / gatekeeping voice have been dropped.)
BANNED = (
    "obviously",
    "of course",
    "clearly",
    "trivially",
)

# Deleted apparatus that must not reappear in a redone chapter, matched against
# heading and callout titles.  "what endures, what changes" is matched by prefix.
FORBIDDEN_APPARATUS = frozenset(
    {
        "what you need going in",
        "contents",
        "what you will build",
        "honesty note",
        "acceptance checks",
        "notes and sources",
    }
)

# Fenced-div classes that carry deleted apparatus.
FORBIDDEN_DIVS = (".route-b", ".artifact-checkpoint")


def load_redone() -> set[str]:
    """Return the chapter file stems already ported to the v2 contract.

    Reads ``scripts/redone.txt`` -- one file stem per line, for example
    ``01-operational-on-ramp`` -- ignoring blank lines and ``#`` comments.  A
    missing file yields an empty set, so a fresh checkout treats every chapter
    as legacy.

    Returns:
        The set of redone chapter file stems (without the ``.qmd`` suffix).
    """
    redone: set[str] = set()
    if REDONE_FILE.exists():
        for line in REDONE_FILE.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry and not entry.startswith("#"):
                redone.add(entry)
    return redone


REDONE = load_redone()


def configured_chapters(text: str) -> list[str]:
    """Return qmd paths from the simple list entries in `_quarto.yml`."""
    return re.findall(r"(?m)^\s*-\s+([A-Za-z0-9_./-]+\.qmd)\s*$", text)


def strip_code(text: str) -> str:
    """Remove fenced code so prose checks do not inspect examples."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def chapter_number(path: Path) -> int | None:
    """Read a two-digit chapter number from a chapter filename."""
    match = re.match(r"(\d{2})-", path.name)
    return int(match.group(1)) if match else None


def h2_titles(text: str) -> list[str]:
    """Return normalized level-two headings."""
    titles: list[str] = []
    for heading in re.findall(r"(?m)^##\s+(.+?)\s*$", strip_code(text)):
        heading = re.sub(r"\s*\{[^}]+\}\s*$", "", heading)
        titles.append(heading.strip())
    return titles


def heading_titles(text: str) -> list[str]:
    """Return the titles of every heading from level two to six (code stripped)."""
    titles: list[str] = []
    for _, heading in re.findall(r"(?m)^(#{2,6})\s+(.+?)\s*$", strip_code(text)):
        heading = re.sub(r"\s*\{[^}]+\}\s*$", "", heading)
        titles.append(heading.strip())
    return titles


def callout_titles(text: str) -> list[str]:
    """Return titles declared on callout / div fences (``title="..."`` etc.)."""
    return re.findall(r'(?i)\btitle\s*[:=]\s*["\']?([^"\'\n}]+)', text)


def visual_count(text: str) -> int:
    """Count source-backed figures without counting decorative icons."""
    diagrams = len(re.findall(r"```\{(?:mermaid|dot)\}", text))
    images = len(re.findall(r"!\[[^\]]*\]\([^)]+\)", strip_code(text)))
    computed = len(re.findall(r"(?m)^#\|\s*label:\s*fig-", text))
    return diagrams + images + computed


def python_cell_count(text: str) -> int:
    """Count executable Quarto ``{python}`` cells."""
    return len(re.findall(r"(?m)^```\s*\{python\b[^}]*\}", text))


def elision_hits(text: str) -> list[str]:
    """Return elision-comment lines (``# ...``) that appear inside code fences.

    Inline teaching code must be a runnable build-up, never an excerpt, so a
    comment whose first content is an ellipsis -- ``# ...`` or
    ``# ... added later ...`` -- is flagged wherever it appears in a fence.

    Args:
        text: The full source of a manuscript file.

    Returns:
        The offending comment lines, stripped, in document order.
    """
    hits: list[str] = []
    for block in re.findall(r"(?ms)^```.*?^```", text):
        for line in block.splitlines():
            if re.match(r"^\s*#\s*\.\.\.", line):
                hits.append(line.strip())
    return hits


def forbidden_apparatus_hits(text: str) -> list[str]:
    """Return names of deleted apparatus that a redone chapter still carries.

    Scans heading titles, callout titles, and fenced-div classes for the
    apparatus the v2 contract removes (route boxes, contents lists,
    artifact-checkpoint tables, and the like).

    Args:
        text: The full source of a manuscript file.

    Returns:
        The offending titles and div classes, in the order encountered.
    """
    hits: list[str] = []
    for title in heading_titles(text) + callout_titles(text):
        collapsed = re.sub(r"\s+", " ", title.strip()).casefold()
        if collapsed.startswith("what endures") or collapsed in FORBIDDEN_APPARATUS:
            hits.append(title.strip())
    for div_class in FORBIDDEN_DIVS:
        if re.search(r"\{[^}]*" + re.escape(div_class), text):
            hits.append(div_class)
    return hits


def docstring_errors(module_path: Path) -> list[str]:
    """Return teaching-docstring failures for a tangled chapter module.

    Public top-level functions longer than five source lines must carry a
    Google-style docstring that mentions ``Args:`` or ``Returns:``; public
    top-level classes longer than five source lines must carry any docstring.
    Names beginning with ``_`` and short helpers are exempt.

    Args:
        module_path: Path to a ``code/chNN/_generated.py`` module.

    Returns:
        One error string per public object that lacks the required docstring; a
        single error if the module does not parse.
    """
    source = module_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{module_path.name} does not parse: {exc}"]

    errors: list[str] = []
    for node in tree.body:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        if node.name.startswith("_"):
            continue
        span = (node.end_lineno or node.lineno) - node.lineno + 1
        if span <= 5:
            continue
        doc = ast.get_docstring(node)
        if isinstance(node, ast.ClassDef):
            if not doc:
                errors.append(
                    f"{module_path.name}: class {node.name} lacks a teaching docstring"
                )
        elif not doc:
            errors.append(
                f"{module_path.name}: function {node.name} lacks a teaching docstring"
            )
        elif "Args:" not in doc and "Returns:" not in doc:
            errors.append(
                f"{module_path.name}: function {node.name} docstring lacks Args:/Returns:"
            )
    return errors


def validate_file(path: Path) -> tuple[list[str], list[str]]:
    """Return errors and warnings for one manuscript source."""
    errors: list[str] = []
    warnings: list[str] = []
    text = path.read_text(encoding="utf-8")
    prose = strip_code(text)
    number = chapter_number(path)
    redone = path.stem in REDONE

    # --- Hygiene checks: every manuscript file ---
    if any(
        marker in text
        for marker in ("�", "â€”", "â†", "Â§")
    ):
        errors.append("contains likely mojibake")

    normalized: list[str] = []
    if number is not None:
        h1 = re.findall(r"(?m)^#\s+(.+?)\s*$", prose)
        if len(h1) != 1:
            errors.append(f"expected exactly one H1, found {len(h1)}")
        elif re.match(rf"Chapter\s+{number}\b", h1[0]):
            errors.append("H1 repeats the chapter number supplied by Quarto")

        yaml_title = re.search(r'(?m)^title:\s*["\']?(.+?)["\']?\s*$', text)
        if yaml_title:
            errors.append("YAML title duplicates the anchored body H1 in a Quarto book")

        headings = h2_titles(text)
        normalized = [h.casefold() for h in headings]
        counts = Counter(normalized)
        duplicates = [name for name, count in counts.items() if count > 1]
        if duplicates:
            errors.append("duplicate H2 headings: " + ", ".join(duplicates))

    lower_prose = prose.casefold()
    for phrase in BANNED:
        if re.search(rf"\b{re.escape(phrase)}\b", lower_prose):
            errors.append(f"banned phrase: {phrase!r}")

    if re.search(r"(?<!\\)\$\d", prose):
        errors.append("unescaped currency marker may be parsed as math")

    for block in re.finditer(r"(?ms)^:::.*?landscape-2026.*?^:::\s*$", text):
        content = block.group(0)
        if "Verify live:" not in content:
            warnings.append("Landscape 2026 block lacks 'Verify live:'")
        if not re.search(r"\b20\d{2}-\d{2}-\d{2}\b", content):
            warnings.append("Landscape 2026 block lacks an ISO verification date")

    # --- New-contract checks: redone numbered chapters only ---
    if number is not None and redone:
        if normalized[-2:] != list(CLOSING):
            errors.append(
                "redone chapter must end with '## Summary' then '## Exercises' "
                f"(final H2s: {normalized[-2:] or ['none']})"
            )

        teaching = [
            h
            for h in normalized
            if h not in CLOSING and not h.startswith("landscape")
        ]
        if len(teaching) < 3:
            errors.append(
                f"expected at least 3 teaching H2s, found {len(teaching)}: "
                + ", ".join(teaching)
            )

        figures = visual_count(text)
        if figures < 3:
            errors.append(f"expected at least 3 figures, found {figures}")

        cells = python_cell_count(text)
        if cells < 1:
            errors.append("expected at least 1 executable ```{python}``` cell, found 0")

        elisions = elision_hits(text)
        if elisions:
            errors.append(
                f"elision comments in code fences ({len(elisions)}): "
                + "; ".join(elisions[:3])
            )

        forbidden = forbidden_apparatus_hits(text)
        if forbidden:
            errors.append(
                "forbidden deleted apparatus: " + ", ".join(dict.fromkeys(forbidden))
            )

        exercise_match = re.search(r"(?ms)^##\s+Exercises.*?(?=^##\s+|\Z)", prose)
        exercise_count = 0
        if exercise_match:
            exercise_count = len(
                re.findall(r"(?m)^\s*\d+\.\s+", exercise_match.group(0))
            )
        if not 5 <= exercise_count <= 8:
            errors.append(f"expected 5-8 exercises, found {exercise_count}")

        word_count = len(re.findall(r"\b[\w'-]+\b", prose))
        if word_count < 3000:
            warnings.append(f"short chapter: {word_count} prose words (floor 3,000)")

        generated = ROOT / "code" / f"ch{number:02d}" / "_generated.py"
        if generated.exists():
            errors.extend(docstring_errors(generated))

    return errors, warnings


def pandoc_heading_id(heading: str) -> str:
    """Approximate Pandoc's automatic identifier for a Markdown heading."""
    heading = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", heading)
    heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", heading)
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = re.sub(r"[`*_~]", "", heading).casefold()
    heading = re.sub(r"[^\w\s.-]", "", heading, flags=re.UNICODE)
    heading = re.sub(r"\s+", "-", heading.strip())
    heading = re.sub(r"^[^\w]*", "", heading, flags=re.UNICODE)
    return heading


def document_anchors(path: Path) -> set[str]:
    """Return explicit and automatic anchors defined by a text document."""
    if path.suffix.casefold() not in {".qmd", ".md", ".markdown", ".html", ".htm"}:
        return set()

    text = path.read_text(encoding="utf-8")
    anchors = set(re.findall(r"\{#[A-Za-z0-9_.:-]+[^}]*\}", text))
    anchors = {item[2:].split()[0].rstrip("}") for item in anchors}
    anchors.update(re.findall(r"\bid=[\"']([^\"']+)[\"']", text))

    automatic_counts: Counter[str] = Counter()
    for raw_heading in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", strip_code(text)):
        if re.search(r"\{#[A-Za-z0-9_.:-]+[^}]*\}\s*$", raw_heading):
            continue
        raw_heading = re.sub(r"\s*\{[^}]+\}\s*$", "", raw_heading)
        base = pandoc_heading_id(raw_heading)
        if not base:
            continue
        count = automatic_counts[base]
        anchors.add(base if count == 0 else f"{base}-{count}")
        automatic_counts[base] += 1
    return anchors


def validate_local_links(path: Path) -> list[str]:
    """Return unresolved relative files and fragments in Markdown links."""
    text = strip_code(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    anchor_cache: dict[Path, set[str]] = {}
    for raw_target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = unquote(raw_target.strip().strip("<>").split(maxsplit=1)[0])
        if "://" in target or target.startswith(("mailto:", "data:")):
            continue

        file_part, separator, fragment = target.partition("#")
        candidate = path if not file_part else (path.parent / file_part).resolve()
        if not candidate.exists():
            errors.append(f"broken local link: {target}")
            continue

        if separator and fragment:
            if candidate not in anchor_cache:
                anchor_cache[candidate] = document_anchors(candidate)
            anchors = anchor_cache[candidate]
            if anchors and fragment not in anchors:
                errors.append(f"broken local fragment: {target}")
    return errors


def main() -> int:
    """Validate all files named by the Quarto book configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as errors"
    )
    args = parser.parse_args()

    relative_paths = configured_chapters(CONFIG.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    for relative in relative_paths:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"{relative}: missing source file")
            continue
        file_errors, file_warnings = validate_file(path)
        errors.extend(f"{relative}: {message}" for message in file_errors)
        warnings.extend(f"{relative}: {message}" for message in file_warnings)
        errors.extend(f"{relative}: {message}" for message in validate_local_links(path))

    for warning in warnings:
        print(f"WARNING {warning}")
    if args.strict:
        errors.extend(warnings)
    for error in errors:
        print(f"ERROR   {error}")

    print(
        f"Checked {len(relative_paths)} manuscript files "
        f"({len(REDONE)} on the v2 contract): "
        f"{len(errors)} errors, {len(warnings)} warnings."
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
