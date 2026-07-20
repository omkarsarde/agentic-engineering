# Auto-generated from chapters/21-agent-applications.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


def autonomy_regime(verification: float, reversibility: float) -> str:
    """Map a (verification strength, reversibility) pair to an autonomy regime.

    Autonomy is not a property of the model; it follows from whether completion
    can be proven and whether a mistake can be undone. Strong-and-reversible
    earns free iteration; weak-but-reversible earns staged drafts;
    strong-but-irreversible earns a single gated commit; weak-and-irreversible
    stays advisory.

    Args:
        verification: Strength of the available verifier in [0, 1].
        reversibility: Ease of undoing the effect in [0, 1].

    Returns:
        One of four regime labels.
    """
    strong, reversible = verification >= 0.5, reversibility >= 0.5
    if strong and reversible:
        return "iterate freely"
    if not strong and reversible:
        return "stage for review"
    if strong and not reversible:
        return "verify, then commit"
    return "advisory (human commits)"


import ast
import difflib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class EditRejected(ValueError):
    """Raised when a proposed edit violates an executor-owned invariant."""


@dataclass(frozen=True)
class Edit:
    path: str
    old: str
    new: str
    reason: str


@dataclass
class TaskResult:
    """The countable outcome of one task plus its full event trace.

    Everything an auditor needs lives here: whether the task resolved, how many
    proposals and test runs it cost, how many edits were rejected before
    execution, and an ordered event log. The counts are what we score; the
    events are what we read when a count surprises us.
    """

    task_id: str
    resolved: bool = False
    proposals: int = 0
    test_runs: int = 0
    rejected_edits: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


def safe_path(root: Path, relative: str, *, allow_tests: bool = False) -> Path:
    """Resolve a repository-relative path and refuse anything outside the rules.

    The executor, not the model, owns three invariants here: an edit may not
    escape the workspace, may touch only Python source, and may not rewrite a
    test file (which would let the agent pass by changing the oracle).

    Args:
        root: Absolute path of the isolated workspace.
        relative: Path the proposal named, relative to ``root``.
        allow_tests: Whether ``test_*.py`` targets are writable; only
            materialization sets this, never an edit.

    Returns:
        The resolved absolute path inside the workspace.

    Raises:
        EditRejected: If the path escapes the workspace, is not ``.py``, or
            names a read-only test file.
    """
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise EditRejected(f"path escapes workspace: {relative}")
    if candidate.suffix != ".py":
        raise EditRejected(f"only Python source files are editable: {relative}")
    if not allow_tests and candidate.name.startswith("test_"):
        raise EditRejected(f"tests are read-only: {relative}")
    return candidate


def apply_edit(root: Path, edit: Edit) -> str:
    """Apply one exact single-occurrence replacement and return a diff receipt.

    The admission checks are the whole point: the anchor text must occur exactly
    once (so the edit is unambiguous), and the result must parse as Python (so a
    broken file never reaches the test runner). Syntax validity is only an
    admission gate; the authoritative check is the test run that follows.

    Args:
        root: Absolute workspace path.
        edit: The proposed edit, naming a file, an anchor, and its replacement.

    Returns:
        A unified-diff string documenting exactly what changed.

    Raises:
        EditRejected: If the file is missing, the anchor is absent or repeated,
            or the edited file would not parse.
    """
    target = safe_path(root, edit.path)
    if not target.is_file():
        raise EditRejected(f"file does not exist: {edit.path}")
    before = target.read_text(encoding="utf-8")
    occurrences = before.count(edit.old)
    if occurrences != 1:
        raise EditRejected(f"anchor must occur once, found {occurrences}: {edit.path}")
    after = before.replace(edit.old, edit.new, 1)
    try:
        ast.parse(after)
    except SyntaxError as exc:
        raise EditRejected(f"edit creates invalid Python: {exc.msg}") from exc
    target.write_text(after, encoding="utf-8")
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{edit.path}", tofile=f"b/{edit.path}",
        )
    )


def materialize(task: dict[str, Any], root: Path) -> None:
    """Write a task's files into a fresh workspace before the loop starts.

    Args:
        task: A task record whose ``files`` maps repository-relative paths to
            their contents; test files are written with ``allow_tests`` set.
        root: The empty workspace directory to populate.
    """
    for relative, contents in task["files"].items():
        target = safe_path(root, relative, allow_tests=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")


def repository_map(root: Path) -> list[dict[str, Any]]:
    """Return an orientation layer: paths and top-level symbols, not full source.

    A repository map is what the model reads to decide *where* to look. It lists
    each file with its line count and its top-level function and class names, so
    the loop can request full contents on demand instead of paying for the whole
    tree on every turn.

    Args:
        root: Absolute path of the workspace to summarize.

    Returns:
        One record per ``.py`` file with ``path``, ``lines``, and ``symbols``.
    """
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            symbols = [
                node.name
                for node in ast.parse(source).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
        except SyntaxError:
            symbols = ["<syntax-error>"]
        result.append(
            {"path": path.relative_to(root).as_posix(), "lines": source.count("\n") + 1,
             "symbols": symbols}
        )
    return result


def run_tests(root: Path, timeout_seconds: float = 8.0) -> tuple[bool, str]:
    """Run the repository's own test command as the authoritative verifier.

    Completion is earned from this observation, taken *after* an edit, not from
    the model's narration that the bug is fixed. A ``pass`` requires the process
    to exit zero; a timeout returns ``fail`` with an honest reason rather than
    an unknown collapsed into either verdict.

    Args:
        root: Workspace to run the tests in.
        timeout_seconds: Wall-clock bound on the test process.

    Returns:
        A ``(passed, observation)`` pair; the observation is the tail of the
        combined stdout and stderr, kept short enough to feed back to the model.
    """
    command = [sys.executable, "-m", "unittest", "discover", "-s", ".", "-p", "test_*.py"]
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        done = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                              text=True, timeout=timeout_seconds, check=False)
        return done.returncode == 0, (done.stdout + done.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_seconds:.1f}s"


def solve_task(task: dict[str, Any], max_proposals: int = 3) -> TaskResult:
    """Drive one task through the observe-propose-execute-verify-commit loop.

    Proposals are scripted so every reader sees the same rejection and repair;
    swapping in a real model means replacing only this proposal source. The
    executor owns every invariant, so a wrong or malicious proposal costs a
    rejection or a failing test, never workspace damage or a gamed oracle.

    Args:
        task: One task record with ``id``, ``files``, and ``proposals``.
        max_proposals: How many proposals the loop may spend before giving up;
            this is the repair budget, and it bounds recovery from both failing
            tests and rejected edits.

    Returns:
        A :class:`TaskResult` with the outcome, the counts, and the full event
        trace for audit.
    """
    result = TaskResult(task_id=task["id"])
    with tempfile.TemporaryDirectory(prefix=f"ch21-{task['id']}-") as directory:
        root = Path(directory).resolve()
        materialize(task, root)
        result.events.append({"kind": "observe", "repo_map": repository_map(root)})
        for raw in task["proposals"][:max_proposals]:
            result.proposals += 1
            edit = Edit(**raw)
            result.events.append({"kind": "propose", "path": edit.path, "reason": edit.reason})
            try:
                receipt = apply_edit(root, edit)
            except EditRejected as exc:
                result.rejected_edits += 1
                result.events.append({"kind": "reject", "reason": str(exc)})
                continue
            result.events.append({"kind": "execute", "diff": receipt})
            passed, observation = run_tests(root)
            result.test_runs += 1
            result.events.append({"kind": "verify", "passed": passed, "observation": observation})
            if passed:
                result.resolved = True
                result.events.append({"kind": "commit"})
                break
    return result


def run_suite(tasks: list[dict[str, Any]], max_proposals: int = 3) -> dict[str, Any]:
    """Score every task and aggregate the loop's countable outcomes.

    Args:
        tasks: The task list to score.
        max_proposals: Repair budget passed to each task.

    Returns:
        A report dict with the task count and the summed resolved, proposal,
        test-run, and rejected-edit counts, plus each task's result.
    """
    results = [solve_task(task, max_proposals=max_proposals) for task in tasks]
    return {
        "tasks": len(results),
        "resolved": sum(r.resolved for r in results),
        "proposals": sum(r.proposals for r in results),
        "test_runs": sum(r.test_runs for r in results),
        "rejected_edits": sum(r.rejected_edits for r in results),
        "results": [asdict(r) for r in results],
    }


import hashlib


def digest(text: str) -> str:
    """Return a short content hash used as a freshness token for a file."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def apply_guarded_edit(root: Path, path: str, observed_digest: str,
                       start: int, end: int, new_lines: list[str]) -> str:
    """Apply a line-range edit only if the file still matches what was observed.

    A line-range edit is compact but its coordinates go stale the moment another
    writer touches the file. Carrying the digest the proposer observed turns that
    silent hazard into a typed rejection: if the current file hashes differently,
    the edit refers to a version that no longer exists and is refused.

    Args:
        root: Workspace path.
        path: Repository-relative file to edit.
        observed_digest: The :func:`digest` the proposer saw when it chose the range.
        start: First line index to replace (0-based, inclusive).
        end: Line index to stop at (exclusive).
        new_lines: Replacement lines, without trailing newlines.

    Returns:
        A unified-diff receipt for the applied change.

    Raises:
        EditRejected: If the file's current digest differs from ``observed_digest``.
    """
    target = safe_path(root, path)
    before = target.read_text(encoding="utf-8")
    if digest(before) != observed_digest:
        raise EditRejected(f"stale digest: file changed since it was observed ({path})")
    lines = before.splitlines(keepends=True)
    after = "".join(lines[:start] + [ln + "\n" for ln in new_lines] + lines[end:])
    ast.parse(after)
    target.write_text(after, encoding="utf-8")
    return "".join(difflib.unified_diff(lines, after.splitlines(keepends=True),
                                        fromfile=f"a/{path}", tofile=f"b/{path}"))


@dataclass(frozen=True)
class EvidenceRecord:
    """One row of the evidence ledger: a claim tied to a single source.

    The ledger is the application state deep research must build before prose.
    Each record binds an atomic claim to exactly one source, records the span
    that supports it, and — crucially — its ``stance``, because a source that
    merely mentions a topic is not a source that establishes the claim.
    """

    claim_id: str
    source_id: str
    span: str
    stance: str          # "supports" | "contradicts" | "context"
    source_class: str
    retrieved: str


def citation_audit(claims: dict[str, str], records: list[EvidenceRecord]) -> list[str]:
    """Audit claims against the ledger in both directions and return findings.

    Two failures are distinct and both matter. A load-bearing claim with no
    supporting record is unsupported synthesis. A record whose stance is only
    ``context`` cannot license the claim it is attached to, even though a
    citation is present. Citation presence, source stance, and entailment are
    three different checks; this audit keeps them apart.

    Args:
        claims: Mapping of claim id to the sentence the report will assert.
        records: Every ledger record gathered for those claims.

    Returns:
        One human-readable finding per problem, empty if the ledger is clean.
    """
    findings: list[str] = []
    by_claim: dict[str, list[EvidenceRecord]] = {claim_id: [] for claim_id in claims}
    for record in records:
        by_claim.setdefault(record.claim_id, []).append(record)
    for claim_id in claims:
        present = by_claim.get(claim_id, [])
        supporting = [r for r in present if r.stance == "supports"]
        if not present:
            findings.append(f"{claim_id}: NO CITATION for a load-bearing claim")
        elif not supporting:
            stances = ", ".join(sorted({r.stance for r in present}))
            findings.append(f"{claim_id}: cited but UNSUPPORTED (stance: {stances})")
    return findings


class StaleObservation(RuntimeError):
    """Raised when the UI changed between observation and action."""


@dataclass(frozen=True)
class UINode:
    """One control from an accessibility tree: role, accessible name, and a ref.

    Semantic addressing ("the button named Submit") survives layout changes that
    would break a pixel coordinate, and it audits cleanly. The ``ref`` is a lease
    on one observation, not a permanent address.
    """

    role: str
    name: str
    ref: str


def resolve(tree: list[UINode], role: str, name: str) -> str:
    """Resolve a semantic target to exactly one element reference.

    Grounding fails safe: zero matches means the control is gone and more than
    one means the name is ambiguous. Either way we refuse to act rather than
    click a guess, because acting on a misresolved control is how a browser
    agent reports success while the real state never changed.

    Args:
        tree: The current accessibility-tree snapshot.
        role: The ARIA role to match (for example ``"button"``).
        name: The accessible name to match.

    Returns:
        The single matching element's ``ref``.

    Raises:
        StaleObservation: If zero or more than one node matches.
    """
    matches = [node.ref for node in tree if node.role == role and node.name == name]
    if len(matches) != 1:
        raise StaleObservation(f"{name!r} resolved to {len(matches)} controls, expected 1")
    return matches[0]


def tree_digest(tree: list[UINode]) -> str:
    """Return a freshness token over a tree snapshot's roles, names, and refs."""
    return digest("|".join(f"{n.role}:{n.name}:{n.ref}" for n in tree))


@dataclass
class Account:
    """A tiny customer account: the authoritative state a refund tool must move."""

    customer_id: str
    paid: float
    days_since_purchase: int
    refunded: float = 0.0


def refund(account: Account, amount: float, policy: dict[str, float]) -> dict[str, Any]:
    """Apply a policy-gated refund and verify it against final account state.

    The action language is a policy-constrained tool, and the verifier is the
    resulting account state plus policy compliance. Anything outside policy is
    not an error to retry but a typed escalation carrying exactly what a human
    needs to decide.

    Args:
        account: The mutable account record.
        amount: The refund the dialogue requested.
        policy: Limits with ``max_days`` and ``max_amount`` keys.

    Returns:
        Either a ``committed`` verdict with the verified post-state, or an
        ``escalate`` verdict with a decision packet.
    """
    remaining = account.paid - account.refunded
    if account.days_since_purchase > policy["max_days"]:
        reason = f"{account.days_since_purchase} days exceeds the {int(policy['max_days'])}-day window"
    elif amount > policy["max_amount"]:
        reason = f"amount {amount:.0f} exceeds the tool limit {policy['max_amount']:.0f}"
    elif amount > remaining:
        reason = f"amount {amount:.0f} exceeds the refundable balance {remaining:.0f}"
    else:
        account.refunded += amount                       # commit: move the DB state
        verified = account.refunded == amount            # re-read the final state
        return {"verdict": "committed", "refunded": account.refunded, "verified": verified}
    return {"verdict": "escalate",
            "packet": {"customer": account.customer_id, "requested": amount,
                       "reason": reason, "decision": "approve exception?"}}


class UnitError(ValueError):
    """Raised when quantities with different declared units are combined raw."""


def to_dollars(amount: float, unit: str) -> float:
    """Normalize a money figure to dollars from its declared unit.

    Args:
        amount: The raw figure as reported by its source.
        unit: One of ``"dollars"``, ``"thousands"``, or ``"millions"``.

    Returns:
        The figure in dollars.

    Raises:
        UnitError: If the unit is not recognized.
    """
    factors = {"dollars": 1.0, "thousands": 1_000.0, "millions": 1_000_000.0}
    if unit not in factors:
        raise UnitError(f"unknown money unit: {unit!r}")
    return amount * factors[unit]


def reconcile_join(left_rows: int, joined_rows: int, key_unique: bool) -> None:
    """Fail loudly when a join fanned out and will double-count aggregates.

    A join that multiplies rows silently inflates every SUM taken over it. The
    cheap guard is a row-count reconciliation: if the right-hand key is not
    unique and the join produced more rows than it started with, an aggregate
    over the result is not trustworthy and must be recomputed on deduplicated
    grain.

    Args:
        left_rows: Row count of the left table before the join.
        joined_rows: Row count after the join.
        key_unique: Whether the join key is unique on the right table.

    Raises:
        UnitError: Reused as a rigor-gate failure when a fan-out is detected.
    """
    if not key_unique and joined_rows > left_rows:
        raise UnitError(f"join fan-out: {left_rows} rows -> {joined_rows}; "
                        f"aggregates will double-count")
