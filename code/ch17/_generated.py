# Auto-generated from chapters/17-tool-harness-engineering.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import re
from dataclasses import dataclass

STOP = {
    "the", "a", "an", "to", "and", "or", "for", "of", "in", "on", "is", "it",
    "this", "that", "their", "them", "they", "our", "us", "we", "you", "your",
    "with", "but", "not", "yet", "if", "so", "do", "does", "did", "has", "have",
    "was", "are", "what", "when", "how", "out", "up", "about", "at", "by", "as",
    "customer", "order", "please", "want", "wants", "asking", "says", "find",
    "here", "should", "can", "be", "my", "me",
}


def terms(text: str) -> set[str]:
    """Reduce text to the content words a lexical matcher compares on.

    Lowercases, keeps alphabetic words longer than two characters, drops a small
    stopword list, and stems a few common suffixes so ``refunded`` and ``refund``
    match. It is deliberately crude: a production harness would embed text
    instead, but the crudeness makes every match in this chapter explainable.

    Args:
        text: A task phrase or a tool card to tokenize.

    Returns:
        The set of normalized content words.
    """
    def stem(word: str) -> str:
        for suffix in ("ing", "ed", "es", "s"):
            if len(word) > 4 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    return {stem(w) for w in re.findall(r"[a-z]+", text.casefold())
            if w not in STOP and len(w) > 2}


@dataclass(frozen=True)
class ToolCard:
    """The model-visible summary of a tool: what the router reads to choose it.

    Only ``name`` and ``description`` reach the model at selection time. Keeping
    this small and honest is the whole ACI lever — a card that names its job and
    its exclusions is retrievable; a card called ``get`` is not.
    """

    name: str
    description: str


def route(task: str, cards: list[ToolCard]) -> int:
    """Pick the index of the card whose words best cover the task.

    Ties break toward the earlier card, deterministically, so a surface of
    indistinguishable cards collapses to always choosing the first one — which is
    exactly the failure a vague surface produces.

    Args:
        task: A user-intent phrase, in the user's words, not the tool's.
        cards: The candidate tool cards.

    Returns:
        The index into ``cards`` of the best match.
    """
    task_terms = terms(task)
    return max(
        range(len(cards)),
        key=lambda i: (len(task_terms & terms(f"{cards[i].name} {cards[i].description}")), -i),
    )


from enum import Enum
from typing import Any, Callable


class Risk(str, Enum):
    """Whether a tool observes state or changes it — the discovery-time split.

    READ tools can run freely; WRITE tools must clear an approval before the
    handler fires. The rating is a property of the tool, known before arguments
    are seen, so the harness can assign gates and budgets at selection time
    rather than parsing every call first.
    """

    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class Tool:
    """A model-facing card bound to an application-owned handler and risk class.

    ``target_version`` is not shown to the model. It lets the harness ask the
    application "is this resource still the one a human reviewed?" at execution
    time, which is what makes the approval binding in a later section possible.
    """

    name: str
    summary: str
    schema: dict[str, type]
    risk: Risk
    handler: Callable[..., Any]
    target_version: Callable[[dict[str, Any]], str] | None = None


@dataclass(frozen=True)
class Call:
    """One exact proposed invocation: which tool, with which arguments."""

    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ExecutionContext:
    """Authenticated identity supplied by the application, never by the model."""

    tenant_id: str
    actor_id: str


def validate_call(call: Call, tool: Tool) -> None:
    """Reject a call whose arguments do not match the tool's typed schema.

    Checks the argument names exactly (no missing, no extra) and each value's
    type. Raising a precise ``ValueError`` gives the harness a repairable signal
    to hand back to the model instead of a crash.

    Args:
        call: The proposed invocation to check.
        tool: The contract whose schema the call must satisfy.

    Raises:
        ValueError: If argument names or types do not match the schema.
    """
    if set(call.arguments) != set(tool.schema):
        raise ValueError(f"expected fields {sorted(tool.schema)}")
    for name, expected in tool.schema.items():
        if not isinstance(call.arguments[name], expected):
            raise ValueError(f"{name} must be {expected.__name__}")


def select_tools(query: str, tools: list[Tool], k: int = 3) -> list[Tool]:
    """Retrieve the k tools whose card best matches a task query.

    Ranks by shared content words between the query and each tool's name and
    summary, breaking ties toward earlier tools. Recall-oriented: raising k
    trades exposed context for a better chance the needed tool is present.

    Args:
        query: The task in user-intent language.
        tools: The full catalog to rank.
        k: How many tools to expose to the model.

    Returns:
        The top-k tools, most relevant first.
    """
    query_terms = terms(query)
    ranked = sorted(
        range(len(tools)),
        key=lambda i: (len(query_terms & terms(f"{tools[i].name} {tools[i].summary}")), -i),
        reverse=True,
    )
    return [tools[i] for i in ranked[:k]]


@dataclass(frozen=True)
class LedgerRow:
    """One accounted contribution to a model call.

    Trust class travels with the row because two equally sized components can
    carry different injection risk; priority is the admission order, lowest
    first, so hard requirements are never evicted for stale evidence.
    """

    source: str
    tokens: int
    trust: str
    priority: int


class ContextLedger:
    """An inspectable budget over the components of one model call.

    Records every candidate row and, given a window and an output reserve,
    admits rows by ascending priority until the budget in @eq-ch17-admission is
    spent. The rows it evicts are the testable answer to "why did the model not
    see the policy?" — the question an invisible prompt string cannot answer.
    """

    def __init__(self, window: int, reserve: int) -> None:
        self.budget = window - reserve
        self.rows: list[LedgerRow] = []

    def add(self, row: LedgerRow) -> None:
        """Register a candidate component; admission happens in :meth:`assemble`."""
        self.rows.append(row)

    def assemble(self) -> tuple[list[LedgerRow], list[LedgerRow]]:
        """Admit rows by priority within budget, returning (admitted, evicted).

        Returns:
            Two lists: the rows that fit the budget, lowest priority first, and
            the rows evicted because the running total would exceed it.
        """
        used, admitted, evicted = 0, [], []
        for row in sorted(self.rows, key=lambda r: r.priority):
            if used + row.tokens <= self.budget:
                used += row.tokens
                admitted.append(row)
            else:
                evicted.append(row)
        return admitted, evicted


from pathlib import Path


class Workspace:
    """Name-scope files to one thread root; this is naming, not containment.

    ``resolve`` blocks ``../`` traversal and absolute-path escapes through this
    API. It does not stop a process that opens another path directly — that
    needs the execution sandbox below. Naming containment and execution
    containment are separate guarantees, and a secure harness uses both.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def resolve(self, relative: str) -> Path:
        """Return a path inside the root, or reject an escape.

        Args:
            relative: A path expressed relative to the workspace root.

        Returns:
            The resolved absolute path, guaranteed under the root.

        Raises:
            PermissionError: If the resolved path escapes the workspace root.
        """
        candidate = (self.root / relative).resolve()
        if not candidate.is_relative_to(self.root):
            raise PermissionError(f"path escapes workspace: {relative}")
        return candidate


import subprocess
import sys


def run_restricted(code: str, *, cpu_seconds: int = 1, mem_mb: int = 256,
                   timeout: float = 5.0) -> dict[str, str]:
    """Run untrusted Python in a subprocess bounded by kernel resource limits.

    Sets ``RLIMIT_CPU`` and ``RLIMIT_AS`` in the child before ``exec`` so a busy
    loop is killed by the kernel and an allocation bomb fails with MemoryError,
    with a wall-clock timeout as a final backstop. This is the execution
    containment a workspace path check cannot provide.

    Args:
        code: Python source to execute in isolation.
        cpu_seconds: CPU-time limit; exceeding it kills the child by signal.
        mem_mb: Address-space limit in megabytes.
        timeout: Wall-clock backstop in seconds.

    Returns:
        A dict with ``status`` in {ok, killed, error} and a short ``detail``.
    """
    import resource

    def apply_limits() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        soft = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (soft, soft))

    try:
        proc = subprocess.run([sys.executable, "-I", "-c", code], preexec_fn=apply_limits,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "killed", "detail": "wall-clock timeout"}
    if proc.returncode < 0:
        return {"status": "killed", "detail": "resource limit"}
    if proc.returncode != 0:
        last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "nonzero exit"
        return {"status": "error", "detail": last}
    return {"status": "ok", "detail": proc.stdout.strip()}


class SkillCard:
    """A skill's always-resident metadata plus lazy access to deeper levels.

    Level 1 (name, triggers, exclusions) is parsed at construction and is the
    only text the harness keeps resident. Level 2 (instructions) and level 3
    (resources) are read from disk only when :meth:`instructions` or
    :meth:`resource` is called, which is progressive disclosure in one object.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        front = (root / "SKILL.md").read_text().split("---")[1]
        self.name = re.search(r"name:\s*(.+)", front).group(1).strip()
        self.description = re.search(r"description:\s*(.+)", front).group(1).strip()
        self.exclude = re.search(r"exclude:\s*(.+)", front).group(1).strip()
        self.metadata_chars = len((root / "SKILL.md").read_text())

    def activates_on(self, task: str) -> bool:
        """Decide whether a task should load this skill.

        Fires when the task shares a content word with the description and none
        with the exclusions. The exclusion check is what stops a naive matcher
        from activating on the boundary terms a good description spells out.

        Args:
            task: The user-intent phrase to classify.

        Returns:
            True if the skill's instructions should be disclosed for this task.
        """
        task_terms = terms(task)
        return bool(task_terms & terms(self.description)) and not (task_terms & terms(self.exclude))

    def instructions(self) -> str:
        """Disclose level 2: read the instructions body from disk on demand."""
        return (self.root / "instructions.md").read_text()


import hashlib
import json
import time


def _canonical(value: Any) -> bytes:
    """Serialize a value deterministically so its hash is stable."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: Any) -> str:
    """Hash a canonical value for binding and comparison."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def action_digest(call: Call, context: ExecutionContext) -> str:
    """Bind a proposal to its exact tool, arguments, tenant, and actor.

    Any change to the arguments or the authenticated identity changes this
    digest, which is what lets execution-time revalidation detect a substituted
    action that a human never reviewed.

    Args:
        call: The proposed invocation.
        context: The authenticated tenant and actor.

    Returns:
        A hex digest binding the action to its exact fields.
    """
    return _digest({"tool": call.tool, "arguments": call.arguments,
                    "tenant": context.tenant_id, "actor": context.actor_id})


@dataclass(frozen=True)
class ApprovalRequest:
    """An immutable reviewable proposal, bound to action, target, and expiry."""

    request_id: str
    action_digest: str
    target_digest: str
    expires_at: float


@dataclass(frozen=True)
class Approval:
    """A person's decision over one request, signing its exact digests."""

    request_id: str
    action_digest: str
    target_digest: str
    expires_at: float
    approver_id: str


class ApprovalError(RuntimeError):
    """The supplied approval does not authorize the current action."""


def request_approval(call: Call, context: ExecutionContext, target_version: str,
                     ttl_s: float = 60, now: float | None = None) -> ApprovalRequest:
    """Freeze a proposal's action and target digests with an expiry.

    Args:
        call: The proposed invocation to bind.
        context: The authenticated identity to bind into the action digest.
        target_version: The resource version the reviewer will see.
        ttl_s: Seconds the approval remains valid.
        now: Injectable clock for deterministic tests.

    Returns:
        An immutable request the reviewer approves as-is.
    """
    issued = time.time() if now is None else now
    action = action_digest(call, context)
    return ApprovalRequest(action[:12], action, _digest({"target": target_version}), issued + ttl_s)


def approve(request: ApprovalRequest, approver_id: str) -> Approval:
    """Record an approver over a request without altering its bound payload."""
    return Approval(request.request_id, request.action_digest, request.target_digest,
                    request.expires_at, approver_id)


def dispatch(call: Call, context: ExecutionContext, tool: Tool,
             approval: Approval | None = None, now: float | None = None) -> Any:
    """Validate a call and, for writes, revalidate the approval before running.

    Reads the current target version through the tool, recomputes the action and
    target digests, and checks expiry at the last responsible moment. A read runs
    freely; a write runs only if every bound digest still matches.

    Args:
        call: The proposed invocation.
        context: The authenticated identity.
        tool: The contract to run.
        approval: The signed approval, required for a write.
        now: Injectable clock for deterministic tests.

    Returns:
        The handler's result for an authorized call.

    Raises:
        ApprovalError: If a write lacks an approval or fails any revalidation.
        ValueError: If the call does not match the schema.
    """
    validate_call(call, tool)
    if tool.risk is Risk.WRITE:
        if approval is None:
            raise ApprovalError("write requires approval")
        current = time.time() if now is None else now
        target_now = tool.target_version(call.arguments) if tool.target_version else ""
        checks = {
            "expired": current > approval.expires_at,
            "substituted action": action_digest(call, context) != approval.action_digest,
            "stale target": _digest({"target": target_now}) != approval.target_digest,
        }
        failed = [name for name, bad in checks.items() if bad]
        if failed:
            raise ApprovalError(", ".join(failed))
    return tool.handler(**call.arguments)


class DualControl:
    """Collect distinct approver signatures until a threshold is met.

    Rejects a repeated signature from the same identity, so one operator cannot
    satisfy a two-person rule by signing twice. This is the mechanism a
    cooling-off or large-transfer policy binds to; the policy lives in Ch 24.
    """

    def __init__(self, threshold: int = 2) -> None:
        self.threshold = threshold
        self.signers: set[str] = set()

    def sign(self, approver_id: str) -> bool:
        """Add one distinct approver; return whether the threshold is now met.

        Args:
            approver_id: The signing identity.

        Returns:
            True once at least ``threshold`` distinct approvers have signed.

        Raises:
            ValueError: If this identity has already signed.
        """
        if approver_id in self.signers:
            raise ValueError(f"dual control: {approver_id} already signed")
        self.signers.add(approver_id)
        return len(self.signers) >= self.threshold


import sqlite3


class Journal:
    """Checkpoint thread state and deduplicate audit events across restarts.

    ``record`` is idempotent on the event id, so a replayed ``approved`` event
    after a crash leaves exactly one audit row. This is a guarantee about local
    harness records only — it does not make a remote payment durable, which is
    the boundary Ch 26 owns.
    """

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS events "
                        "(event_id TEXT PRIMARY KEY, thread_id TEXT, payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS threads "
                        "(thread_id TEXT PRIMARY KEY, state TEXT)")

    def record(self, event_id: str, thread_id: str, payload: Any) -> bool:
        """Append one idempotent audit event; return whether it was new.

        Args:
            event_id: Stable identity that makes replays deduplicate.
            thread_id: The thread the event belongs to.
            payload: JSON-serializable event body.

        Returns:
            True if the row was inserted, False if the id already existed.
        """
        cur = self.db.execute("INSERT OR IGNORE INTO events VALUES (?, ?, ?)",
                              (event_id, thread_id, json.dumps(payload, sort_keys=True)))
        self.db.commit()
        return cur.rowcount == 1

    def checkpoint(self, thread_id: str, state: Any) -> None:
        """Replace a thread's harness state; not a durable-effect guarantee."""
        self.db.execute("INSERT INTO threads VALUES (?, ?) ON CONFLICT(thread_id) "
                        "DO UPDATE SET state=excluded.state", (thread_id, json.dumps(state, sort_keys=True)))
        self.db.commit()

    def load(self, thread_id: str) -> Any:
        """Return a thread's checkpointed state, or None."""
        row = self.db.execute("SELECT state FROM threads WHERE thread_id=?", (thread_id,)).fetchone()
        return None if row is None else json.loads(row[0])

    def audit_rows(self, thread_id: str) -> int:
        """Count durable audit rows for a thread."""
        return self.db.execute("SELECT COUNT(*) FROM events WHERE thread_id=?", (thread_id,)).fetchone()[0]
