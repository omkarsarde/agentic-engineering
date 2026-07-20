"""Guard the checked-in Matplotlib SVG build contract."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_every_svg_generator_fixes_ids_and_creation_metadata() -> None:
    """Stable IDs plus an omitted creation date make SVG bytes reproducible."""

    generators = [
        path
        for path in sorted((ROOT / "code").rglob("*.py"))
        if "savefig(" in path.read_text(encoding="utf-8")
    ]
    assert generators
    for path in generators:
        source = path.read_text(encoding="utf-8")
        assert "svg.hashsalt" in source, f"missing stable SVG IDs: {path}"
        assert 'metadata={"Date": None}' in source, f"timestamped SVG output: {path}"


@pytest.mark.parametrize(
    "script",
    (
        ROOT / "code" / "appa" / "render_pareto.py",
    ),
)
def test_repeated_svg_builds_are_byte_identical(script: Path, tmp_path: Path) -> None:
    """Regenerate representative chapter and appendix figures in fresh processes."""

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    digests: list[str] = []
    for run in (1, 2):
        output = tmp_path / f"run-{run}.svg"
        subprocess.run(
            [sys.executable, str(script), "--plot", str(output)],
            cwd=script.parent,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        digests.append(hashlib.sha256(output.read_bytes()).hexdigest())
    assert digests[0] == digests[1]
