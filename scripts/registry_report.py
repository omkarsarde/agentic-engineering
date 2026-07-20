"""Report landscape-registry rows whose owner chapter lacks the source anchor.

Run after rewriting chapters: rows owned by a rewritten chapter fail the
appendix-C lint until the chapter's landscape callout links the row's exact
source URL (or the row declares an owner_anchor). Usage:

    python scripts/registry_report.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "appc"))
import landscape_lint  # noqa: E402

data = json.loads((ROOT / "data" / "landscape-2026.json").read_text(encoding="utf-8"))
findings = landscape_lint.lint_owner_grounding(data["entries"], book_root=ROOT)
flagged = {f.entry_id for f in findings}
if not flagged:
    print("registry clean")
for entry in data["entries"]:
    if entry["id"] in flagged:
        print(entry["id"])
        print("   owner :", entry["owner_chapter"])
        print("   source:", entry["source"])
