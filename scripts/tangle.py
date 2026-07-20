"""Extract ``# @save`` cells from a chapter into ``code/chNN/_generated.py``.

Teaching code is authored inline in executable ```` ```{python} ```` cells.  A
cell whose first line after any ``#|`` options is ``# @save`` holds definitions
that the consolidated test suite imports; this script concatenates those cells,
in document order, into a single deterministic module beside the chapter's code.

Usage:
    python scripts/tangle.py 16                 # a chapter number
    python scripts/tangle.py 16-agent-anatomy   # a file stem
    python scripts/tangle.py chapters/16-agent-anatomy.qmd
    python scripts/tangle.py a-bridge           # an appendix -> code/appa/
    python scripts/tangle.py --all              # every chapter and appendix
    python scripts/tangle.py --all --check      # fail if any file is stale
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FUTURE = "from __future__ import annotations"
CELL = re.compile(r"(?ms)^```\{python\}[ \t]*\n(.*?)\n```[ \t]*$")


def resolve_source(token: str) -> Path:
    """Resolve a path, file stem, chapter number, or appendix letter to a .qmd."""
    candidate = Path(token)
    if candidate.exists():
        return candidate.resolve()
    for base in (ROOT / "chapters", ROOT / "appendices"):
        exact = base / (token if token.endswith(".qmd") else f"{token}.qmd")
        if exact.exists():
            return exact
        matches = sorted(base.glob(f"{token}-*.qmd")) or sorted(base.glob(f"{token}*.qmd"))
        if matches:
            return matches[0]
    raise SystemExit(f"tangle: cannot resolve chapter {token!r}")


def target_for(source: Path) -> Path | None:
    """Return the ``_generated.py`` path for a chapter or appendix source."""
    chapter = re.match(r"(\d{2})-", source.name)
    if chapter:
        return ROOT / "code" / f"ch{chapter.group(1)}" / "_generated.py"
    appendix = re.match(r"([a-z])-", source.name)
    if appendix:
        return ROOT / "code" / f"app{appendix.group(1)}" / "_generated.py"
    return None


def saved_bodies(text: str) -> list[str]:
    """Return the code of every ``# @save`` cell, marker and options removed."""
    bodies: list[str] = []
    for body in CELL.findall(text):
        lines = body.split("\n")
        index = 0
        while index < len(lines) and lines[index].lstrip().startswith("#|"):
            index += 1
        if index < len(lines) and lines[index].strip() == "# @save":
            code = [ln for ln in lines[index + 1 :] if ln.strip() != FUTURE]
            snippet = "\n".join(code).strip("\n")
            if snippet:
                bodies.append(snippet)
    return bodies


def render(source: Path) -> str | None:
    """Render the deterministic module for one chapter, or None if it has no cells."""
    bodies = saved_bodies(source.read_text(encoding="utf-8"))
    if not bodies:
        return None
    relative = source.relative_to(ROOT).as_posix()
    header = f"# Auto-generated from {relative} by scripts/tangle.py — do not edit."
    return f"{header}\n{FUTURE}\n\n\n" + "\n\n\n".join(bodies) + "\n"


def tangle(source: Path, check: bool) -> int:
    """Write (or verify with ``check``) the generated module for one chapter."""
    target = target_for(source)
    if target is None:
        print(f"tangle: {source.name} is not a numbered chapter or appendix")
        return 1
    content = render(source)
    if content is None:
        print(f"tangle: {source.name} has no # @save cells; nothing to generate")
        return 0
    current = target.read_text(encoding="utf-8") if target.exists() else None
    if check:
        if current != content:
            print(f"tangle: {target.relative_to(ROOT).as_posix()} is stale")
            return 1
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"tangle: wrote {target.relative_to(ROOT).as_posix()}")
    return 0


def main() -> int:
    """Tangle one chapter, or every chapter and appendix with ``--all``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", nargs="?", help="chapter number, file stem, or path")
    parser.add_argument("--all", action="store_true", help="tangle every chapter and appendix")
    parser.add_argument("--check", action="store_true", help="exit nonzero if any output is stale")
    args = parser.parse_args()

    if args.all:
        sources = sorted((ROOT / "chapters").glob("*.qmd"))
        sources += sorted((ROOT / "appendices").glob("*.qmd"))
    elif args.chapter:
        sources = [resolve_source(args.chapter)]
    else:
        parser.error("give a chapter or --all")

    return max(tangle(source, args.check) for source in sources)


if __name__ == "__main__":
    sys.exit(main())
