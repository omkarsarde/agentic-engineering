# Auto-generated from chapters/18-memory-experiential-learning.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import hashlib
import re
from dataclasses import dataclass, replace
from enum import StrEnum


class Kind(StrEnum):
    """What a memory record means, which fixes its write and verification policy.

    SEMANTIC facts supersede on contradiction; EPISODIC events accumulate;
    PROCEDURAL skills need the strictest promotion; WORKING notes belong to one
    run and should not persist.
    """

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class Source(StrEnum):
    """Where a candidate came from, which sets the evidence bar to admit it."""

    USER = "user"
    VERIFIED_TOOL = "verified_tool"
    MODEL_INFERENCE = "model_inference"
    RETRIEVED_DOCUMENT = "retrieved_document"


class Status(StrEnum):
    """Lifecycle state of one immutable record.

    ACTIVE is current truth; SUPERSEDED is retained history; EXPIRED is past its
    time-to-live; DELETED is owned by a removal process. Retrieval treats the
    four very differently, so status is data, not an 'alive' flag.
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    DELETED = "deleted"


class Op(StrEnum):
    """The self-editing decision a memory write reduces to on each turn."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


@dataclass(frozen=True)
class Scope:
    """The authorization boundary checked before any record is ranked.

    A user-owned record is visible only to a read carrying the same tenant and
    user; a tenant-only record (``user_id=None``) is visible to everyone in the
    tenant. Scope is enforced inside retrieval, never as a post-filter on rows.
    """

    tenant_id: str
    user_id: str | None = None


@dataclass(frozen=True)
class Candidate:
    """A proposed write; the policy may reject it, the store may supersede with it.

    Args:
        key: The slot the fact occupies, e.g. ``"home city"``; semantic keys
            hold one active value per (key, scope).
        value: The remembered content.
        kind: Which memory tier this belongs to.
        scope: Who owns and may read it.
        source: Provenance class, which sets the evidence bar.
        evidence_id: A pointer to the turn or tool result that justifies it.
        event_time: When the fact became true in the world (valid time).
        confidence: Calibrated support in [0, 1] for the source and extraction.
        ttl: Optional forgetting horizon; the record expires at
            ``event_time + ttl``.
    """

    key: str
    value: str
    kind: Kind
    scope: Scope
    source: Source
    evidence_id: str
    event_time: int
    confidence: float = 1.0
    ttl: int | None = None


@dataclass(frozen=True)
class Record:
    """An immutable, scoped, bitemporal memory record.

    Two clocks matter. Valid time (``valid_from``/``valid_to``) says when the
    fact held in the world; transaction time (``recorded_at``) says when the
    system learned it. A late correction changes the first without rewriting
    the second. ``parents`` links a derived record to its evidence so deletion
    can traverse to every copy.
    """

    record_id: str
    key: str
    value: str
    kind: Kind
    scope: Scope
    source: Source
    evidence_id: str
    confidence: float
    valid_from: int
    recorded_at: int
    valid_to: int | None = None
    expires_at: int | None = None
    status: Status = Status.ACTIVE
    parents: tuple[str, ...] = ()

    def span(self) -> str:
        """Return a compact ``[valid_from, valid_to)`` interval for printing."""
        end = "inf" if self.valid_to is None else str(self.valid_to)
        return f"[{self.valid_from},{end})"


class MemoryPolicy:
    """The write gate: a candidate becomes durable only if it clears these bars.

    The gate exists because model output and retrieved text can *propose* a
    write but must never *become* durable authority on their own. Each source
    faces a different bar, and instruction-like content is refused outright so
    an injected document cannot rewrite the agent's future behavior.
    """

    forbidden = ("ignore previous", "ignore all previous", "system prompt", "wire money")

    def validate(self, candidate: Candidate) -> tuple[bool, str]:
        """Decide whether one candidate may be written, and say why.

        Args:
            candidate: The proposed write.

        Returns:
            An ``(accepted, reason)`` pair; ``reason`` is a short audit string
            whether or not the candidate was accepted.
        """
        if not candidate.evidence_id:
            return False, "missing provenance"
        if not 0.0 <= candidate.confidence <= 1.0:
            return False, "confidence outside [0, 1]"
        if candidate.source is Source.RETRIEVED_DOCUMENT:
            return False, "retrieved text is data, not a memory command"
        lowered = candidate.value.casefold()
        if any(phrase in lowered for phrase in self.forbidden):
            return False, "instruction-like content requires review"
        if candidate.kind is Kind.PROCEDURAL and candidate.source is Source.MODEL_INFERENCE:
            return False, "the model may not self-author a durable procedure"
        return True, "accepted"


def _tokens(text: str) -> set[str]:
    """Normalize text to a lexical token set for the toy retrieval projection."""
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _record_id(candidate: Candidate) -> str:
    """Derive a stable id from provenance and owned scope, not from content."""
    raw = "|".join((candidate.scope.tenant_id, candidate.scope.user_id or "",
                    candidate.key, candidate.evidence_id, str(candidate.event_time)))
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class MemoryStore:
    """Records as the source of truth, plus an inverted index and read cache.

    The index is a *projection*: a token-to-id map that makes lexical recall
    cheap and that retrieval reads from. Because it is derived, every lifecycle
    event (supersede, expire, delete) must keep it consistent, or a stale
    posting resurrects a record the truth store already retired.
    """

    def __init__(self, policy: MemoryPolicy | None = None) -> None:
        self.policy = policy or MemoryPolicy()
        self.records: dict[str, Record] = {}
        self.index: dict[str, set[str]] = {}
        self.cache: dict[tuple, str] = {}

    def _index_record(self, record: Record) -> None:
        for token in _tokens(f"{record.key} {record.value}"):
            self.index.setdefault(token, set()).add(record.record_id)

    def _deindex(self, record_id: str) -> None:
        for postings in self.index.values():
            postings.discard(record_id)

    def active(self, key: str, scope: Scope) -> Record | None:
        """Return the current active record for a (key, scope), or None."""
        for record in self.records.values():
            if record.status is Status.ACTIVE and record.key == key and record.scope == scope:
                return record
        return None


def add_method(cls):
    """Attach the decorated function to an existing class as a method.

    Returns:
        A decorator that binds a function onto ``cls`` and returns it, letting
        us grow a class one method per cell with prose between the fragments.
    """
    def decorator(function):
        setattr(cls, function.__name__, function)
        return function
    return decorator


@add_method(MemoryStore)
def write(self, candidate: Candidate, now: int | None = None) -> tuple[Record | None, str]:
    """Validate a candidate, supersede any conflicting fact, and index it.

    A semantic write closes the validity interval of the current active record
    for the same (key, scope) before appending the new one, so old truth is
    kept as history rather than overwritten.

    Args:
        candidate: The proposed write.
        now: Transaction time (when the system learns the fact); defaults to the
            candidate's event time.

    Returns:
        A ``(record, reason)`` pair; ``record`` is None when the gate rejected
        the write.
    """
    accepted, reason = self.policy.validate(candidate)
    if not accepted:
        return None, reason
    if candidate.kind is Kind.SEMANTIC:
        current = self.active(candidate.key, candidate.scope)
        if current is not None:
            self.records[current.record_id] = replace(
                current, status=Status.SUPERSEDED, valid_to=candidate.event_time)
    expires_at = None if candidate.ttl is None else candidate.event_time + candidate.ttl
    record = Record(
        _record_id(candidate), candidate.key, candidate.value, candidate.kind,
        candidate.scope, candidate.source, candidate.evidence_id, candidate.confidence,
        valid_from=candidate.event_time,
        recorded_at=candidate.event_time if now is None else now,
        expires_at=expires_at)
    self.records[record.record_id] = record
    self._index_record(record)
    self.cache.clear()
    return record, reason


@add_method(MemoryStore)
def decide_op(self, candidate: Candidate) -> Op:
    """Classify what writing this candidate would do to current memory.

    Args:
        candidate: The proposed semantic write.

    Returns:
        ADD when the slot is empty, NOOP when the value is unchanged, UPDATE
        when it contradicts the current active value.
    """
    existing = self.active(candidate.key, candidate.scope)
    if existing is None:
        return Op.ADD
    if existing.value.casefold() == candidate.value.casefold():
        return Op.NOOP
    return Op.UPDATE


@add_method(MemoryStore)
def _visible(self, record: Record, scope: Scope) -> bool:
    if record.scope.tenant_id != scope.tenant_id:
        return False
    if record.scope.user_id is not None and record.scope.user_id != scope.user_id:
        return False
    return True


@add_method(MemoryStore)
def _temporal(self, record: Record, as_of: int | None, now: int | None) -> bool:
    if as_of is None:
        if record.status is not Status.ACTIVE:
            return False
        return not (now is not None and record.expires_at is not None and now >= record.expires_at)
    if record.status is Status.DELETED:
        return False
    if record.valid_from > as_of:
        return False
    return record.valid_to is None or as_of < record.valid_to


@add_method(MemoryStore)
def retrieve(self, query: str, scope: Scope, as_of: int | None = None,
             now: int | None = None, threshold: float = 1.5) -> Record | None:
    """Return the best authorized, temporally valid record, or abstain.

    The pipeline is ordered on purpose: gather candidate ids from the index,
    drop everything outside ``scope`` *before* scoring, keep only records valid
    for the query time, then rank by @eq-ch18-score. Returning None is a
    first-class answer — the store abstains rather than surface an unrelated row.

    Args:
        query: The natural-language read.
        scope: The caller's authenticated tenant and user.
        as_of: A past instant for a historical read; None reads the present.
        now: The current clock, used to hide expired records on present reads.
        threshold: Minimum score to answer rather than abstain.

    Returns:
        The winning record, or None to abstain.
    """
    query_tokens = _tokens(query)
    candidate_ids: set[str] = set()
    for token in query_tokens:
        candidate_ids |= self.index.get(token, set())
    scored: list[tuple[float, int, Record]] = []
    for record_id in candidate_ids:
        record = self.records[record_id]
        if not self._visible(record, scope) or not self._temporal(record, as_of, now):
            continue
        overlap = len(query_tokens & _tokens(f"{record.key} {record.value}"))
        scored.append((overlap + record.confidence, record.valid_from, record))
    if not scored:
        return None
    score, _, record = max(scored)
    if score < threshold:
        return None
    self.cache[(scope.tenant_id, scope.user_id or "", query.casefold(), as_of)] = record.record_id
    return record


@add_method(MemoryStore)
def expire(self, now: int) -> list[str]:
    """Retire active records whose time-to-live has passed as of ``now``.

    Args:
        now: The current clock reading.

    Returns:
        The ids moved to EXPIRED, whose index postings are removed so retrieval
        can no longer surface them.
    """
    expired: list[str] = []
    for record in list(self.records.values()):
        if (record.status is Status.ACTIVE and record.expires_at is not None
                and now >= record.expires_at):
            self.records[record.record_id] = replace(record, status=Status.EXPIRED)
            self._deindex(record.record_id)
            expired.append(record.record_id)
    return expired


@dataclass
class ToolCall:
    """One memory tool the agent's extractor emitted for a turn."""

    name: str
    op: Op
    key: str
    value: str


class ScriptedExtractor:
    """A deterministic stand-in for the memory-extraction model.

    A production system prompts an LLM to read a turn and emit memory
    operations as JSON; here the same contract is met by visible rules so the
    chapter runs offline and every decision is inspectable. The extractor
    proposes facts and retractions; the agent decides ADD/UPDATE/NOOP/DELETE.
    """

    def __init__(self, scope: Scope) -> None:
        self.scope = scope

    def extract(self, turn: str, event_time: int) -> list[tuple[str, Candidate | str]]:
        """Return proposed writes and retractions for one conversational turn.

        Args:
            turn: The user's message.
            event_time: The clock value for this turn.

        Returns:
            A list of ``("write", Candidate)`` proposals and ``("delete", key)``
            retractions.
        """
        proposals: list[tuple[str, Candidate | str]] = []
        low = turn.casefold()
        move = re.search(r"(?:live in|living in|moved to|now in)\s+([a-z]+)", low)
        if move:
            proposals.append(("write", Candidate(
                "home city", move.group(1).capitalize(), Kind.SEMANTIC, self.scope,
                Source.USER, f"turn-{event_time}", event_time)))
        seat = re.search(r"prefer\s+(\w+)\s+seats?", low)
        if seat:
            proposals.append(("write", Candidate(
                "seat preference", seat.group(1), Kind.SEMANTIC, self.scope,
                Source.USER, f"turn-{event_time}", event_time)))
        if "forget where i live" in low or "forget my home" in low:
            proposals.append(("delete", "home city"))
        return proposals


class MemoryAgent:
    """A self-editing memory agent: it reads turns and rewrites its own store.

    On each turn the extractor proposes operations, the agent classifies each as
    ADD/UPDATE/NOOP/DELETE and applies it through the write gate, and questions
    are answered only from retrieved memory. The tool surface — memory_insert,
    memory_delete, memory_search — is the MemGPT-style contract kept small
    enough for the application to police.
    """

    def __init__(self, store: MemoryStore, extractor: ScriptedExtractor) -> None:
        self.store = store
        self.extractor = extractor

    def observe(self, turn: str, event_time: int) -> list[ToolCall]:
        """Apply the memory operations one turn implies and log the tool calls.

        Args:
            turn: The user's message.
            event_time: The clock value for this turn.

        Returns:
            The tool calls made, each tagged with its ADD/UPDATE/NOOP/DELETE
            classification.
        """
        calls: list[ToolCall] = []
        for kind, payload in self.extractor.extract(turn, event_time):
            if kind == "delete":
                existing = self.store.active(payload, self.extractor.scope)
                if existing is not None:
                    self.store.records[existing.record_id] = replace(existing, status=Status.DELETED)
                    self.store._deindex(existing.record_id)
                    self.store.cache.clear()
                calls.append(ToolCall("memory_delete", Op.DELETE, payload, ""))
            else:
                op = self.store.decide_op(payload)
                if op in (Op.ADD, Op.UPDATE):
                    self.store.write(payload)
                calls.append(ToolCall("memory_insert", op, payload.key, payload.value))
        return calls

    def answer(self, question: str, as_of: int | None = None, now: int | None = None) -> str:
        """Answer a question only from retrieved memory, abstaining when unsure."""
        record = self.store.retrieve(question, self.extractor.scope, as_of=as_of, now=now)
        return record.value if record is not None else "I don't have that in memory."


@dataclass(frozen=True)
class Ticket:
    """One triage case: the keywords the agent sees and the correct queue."""

    ticket_id: str
    keywords: frozenset[str]
    gold: str


def base_policy(ticket: Ticket) -> str:
    """Route a ticket by first matching keyword — the naive starting policy.

    Deliberately incomplete: it sends anything mentioning a refund to billing,
    missing the fraud exception the agent must learn from failure.

    Args:
        ticket: The case to route.

    Returns:
        The queue the base policy chooses.
    """
    for keyword, queue in [("refund", "BILLING"), ("crash", "ENGINEERING"),
                           ("password", "ACCOUNT")]:
        if keyword in ticket.keywords:
            return queue
    return "GENERAL"


@dataclass(frozen=True)
class Lesson:
    """A candidate procedure mined from one failure.

    It fires when ``pattern`` is a subset of a case's keywords and then
    recommends ``action``. A more specific pattern outranks a broader one, and
    a lesson is inert until ``active`` is set by a clean held-out replay.
    """

    pattern: frozenset[str]
    action: str
    active: bool = False


def reflect(ticket: Ticket) -> Lesson:
    """Turn one failed case into a candidate lesson keyed on its full pattern."""
    return Lesson(pattern=ticket.keywords, action=ticket.gold)


class LessonStore:
    """The procedural tier: lessons with a held-out promotion gate and rollback.

    A lesson mined from a failure is a candidate; it is replayed on held-out
    matching cases and negatives, and only a clean replay promotes it to active.
    An over-general lesson that misfires on a negative case is blocked — the
    software-release discipline a standing procedure deserves.
    """

    def __init__(self) -> None:
        self.lessons: list[Lesson] = []

    def decide(self, ticket: Ticket) -> str:
        """Route with the most specific active lesson, else the base policy.

        Args:
            ticket: The case to route.

        Returns:
            The chosen queue.
        """
        firing = [l for l in self.lessons if l.active and l.pattern <= ticket.keywords]
        if firing:
            return max(firing, key=lambda l: len(l.pattern)).action
        return base_policy(ticket)

    def promote(self, lesson: Lesson, held_out: list[Ticket],
                negatives: list[Ticket]) -> tuple[bool, str]:
        """Replay a candidate lesson and promote it only if the replay is clean.

        Args:
            lesson: The candidate mined from a failure.
            held_out: Matching cases the lesson should fix, unseen when mined.
            negatives: Cases the lesson must not disturb.

        Returns:
            A ``(promoted, report)`` pair; the report prints the replay counts.
        """
        matching = [t for t in held_out if lesson.pattern <= t.keywords]
        fixed = sum(1 for t in matching if lesson.action == t.gold)
        misfire = sum(1 for t in negatives
                      if lesson.pattern <= t.keywords and lesson.action != t.gold)
        promoted = bool(matching) and fixed == len(matching) and misfire == 0
        if promoted:
            self.lessons.append(replace(lesson, active=True))
        return promoted, f"held-out {fixed}/{len(matching)} fixed, {misfire} negative misfire"


@add_method(MemoryStore)
def consolidate(self, episodes: list[Record], candidate: Candidate) -> tuple[Record | None, str]:
    """Derive one summary from verified episodes, keeping lineage to them.

    A summary must point back to the episodes that support it; a summary without
    parents launders away their uncertainty and deletion obligations. At least
    two verified episodes are required so one event is not promoted to a fact.

    Args:
        episodes: The supporting episodic records.
        candidate: The proposed summary write.

    Returns:
        A ``(record, reason)`` pair; the record's ``parents`` are the supporting
        episode ids.
    """
    verified = [e for e in episodes if e.kind is Kind.EPISODIC
                and e.source is Source.VERIFIED_TOOL and e.status is Status.ACTIVE]
    if len(verified) < 2:
        return None, "need two verified episodes"
    record, reason = self.write(candidate)
    if record is None:
        return None, reason
    linked = replace(record, parents=tuple(e.record_id for e in verified))
    self.records[record.record_id] = linked
    return linked, "consolidated"


@add_method(MemoryStore)
def _deleted_ids(self) -> set[str]:
    return {rid for rid, r in self.records.items() if r.status is Status.DELETED}


@add_method(MemoryStore)
def delete_subject(self, scope: Scope) -> "DeletionManifest":
    """Delete a subject's primary records and report unfinished derived work.

    This pass removes the subject's *source* records (no parents) from the truth
    store, the index, and the cache. Derived records are left for
    :meth:`invalidate_derived`, mirroring production where projections are
    rebuilt by separate jobs. If any derived record still depends on a deleted
    parent, the manifest reports that target OPEN.

    Args:
        scope: The subject to erase (tenant plus user).

    Returns:
        A manifest naming each target's state; OPEN signals unfinished work.
    """
    deleted: list[str] = []
    owned = [r for r in self.records.values()
             if r.scope.tenant_id == scope.tenant_id and r.scope.user_id == scope.user_id
             and r.status is not Status.DELETED]
    for record in owned:
        if record.parents:
            continue  # derived: handled by invalidate_derived, not here
        self.records[record.record_id] = replace(record, status=Status.DELETED)
        self._deindex(record.record_id)
        deleted.append(record.record_id)
    gone = self._deleted_ids()
    self.cache = {k: rid for k, rid in self.cache.items() if rid not in gone}
    dangling = [r for r in self.records.values() if r.status is not Status.DELETED
                and r.parents and gone.intersection(r.parents)]
    targets = {"primary_store": "deleted", "search_index": "deleted", "cache": "deleted",
               "derived_summaries": "OPEN" if dangling else "deleted"}
    return DeletionManifest(scope.user_id or scope.tenant_id, tuple(deleted), targets)


@add_method(MemoryStore)
def invalidate_derived(self) -> list[str]:
    """Retire derived records whose parents were deleted.

    Returns:
        The ids invalidated because a parent no longer exists — the pass that
        closes the OPEN target left by :meth:`delete_subject`.
    """
    gone = self._deleted_ids()
    invalidated: list[str] = []
    for record in list(self.records.values()):
        if record.status is not Status.DELETED and record.parents and gone.intersection(record.parents):
            self.records[record.record_id] = replace(record, status=Status.DELETED)
            self._deindex(record.record_id)
            invalidated.append(record.record_id)
    return invalidated


@dataclass(frozen=True)
class DeletionManifest:
    """Evidence that a subject was removed from every declared target.

    A target maps a store name to ``"deleted"`` or ``"OPEN"``. An OPEN target is
    unfinished deletion work — usually a derived summary still referencing the
    subject — and must be closed before the request is done.
    """

    subject: str
    record_ids: tuple[str, ...]
    targets: dict[str, str]

    def complete(self) -> bool:
        """Return True only when no declared target is still OPEN."""
        return all(state != "OPEN" for state in self.targets.values())


def build_governed() -> MemoryStore:
    """Build the governed store the ability probes score.

    Returns:
        A store holding Mina's Munich-then-Berlin history and Raj's Paris fact,
        so the probes can exercise update, temporal, isolation, and deletion.
    """
    store = MemoryStore()
    a, b = Scope("acme", user_id="mina"), Scope("globex", user_id="raj")
    store.write(Candidate("home city", "Munich", Kind.SEMANTIC, a, Source.USER, "t1", 10))
    store.write(Candidate("home city", "Berlin", Kind.SEMANTIC, a, Source.USER, "t3", 20))
    store.write(Candidate("home city", "Paris", Kind.SEMANTIC, b, Source.USER, "u1", 10))
    return store


def probe_governed() -> dict[str, int]:
    """Score the governed store on five LongMemEval-style ability slices.

    Returns:
        A per-ability 0/1 map over update, temporal, abstention, isolation, and
        deletion — each measured by querying the store directly, not by asking
        a model whether it remembers.
    """
    store = build_governed()
    a, b = Scope("acme", user_id="mina"), Scope("globex", user_id="raj")
    update = store.retrieve("home city", a)
    temporal = store.retrieve("home city", a, as_of=15)
    isolation = store.retrieve("home city", b)
    abstain = store.retrieve("favorite color", a)
    store.delete_subject(a)
    deleted = store.retrieve("home city", a)
    return {"update": int(update is not None and update.value == "Berlin"),
            "temporal": int(temporal is not None and temporal.value == "Munich"),
            "abstention": int(abstain is None),
            "isolation": int(isolation is not None and isolation.value == "Paris"),
            "deletion": int(deleted is None)}


def probe_transcript_only() -> dict[str, int]:
    """Score a 'stuff the whole transcript into context' baseline.

    It answers from the most recent line mentioning the query terms, with no
    validity intervals, no scope predicate, and no deletion path.

    Returns:
        A per-ability 0/1 map; recency gets the current value, every governance
        slice fails.
    """
    latest = "home city Berlin"  # newest mention in a Munich->Berlin log
    return {"update": int("Berlin" in latest), "temporal": 0, "abstention": 0,
            "isolation": 0, "deletion": 0}


def probe_no_memory() -> dict[str, int]:
    """Score a stateless baseline that always abstains."""
    return {"update": 0, "temporal": 0, "abstention": 1, "isolation": 0, "deletion": 1}
