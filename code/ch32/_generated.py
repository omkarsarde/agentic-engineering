# Auto-generated from chapters/32-capstone.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path


def load_chapter_module(chapter: str, name: str):
    """Load a committed chapter's tangled module by path, not by install.

    Earlier chapters tangle their teaching code into ``code/chNN/_generated.py``.
    The capstone reuses those artifacts instead of re-implementing them, so this
    helper walks up from the working directory to the book root and imports the
    requested module under a capstone-unique name.

    Args:
        chapter: The chapter directory, e.g. ``"ch14"``.
        name: The module name to register, e.g. ``"ch14_rag"``.

    Returns:
        The executed module object.
    """
    root = Path.cwd()
    while not (root / "code" / chapter / "_generated.py").exists():
        if root.parent == root:
            raise FileNotFoundError(f"cannot find code/{chapter}/_generated.py")
        root = root.parent
    spec = importlib.util.spec_from_file_location(name, root / "code" / chapter / "_generated.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Ticket:
    """One task in the fixed evaluation suite, with its ground truth attached.

    The grader — never the model — owns the expectations: the intent the reply
    must carry, the exact fact the draft must contain when the ticket needs one,
    whether the correct behavior is to answer or to escalate, and whether the
    final state must include a durable note on the ticket.

    Args:
        ticket_id: Stable identifier; also the seed for deterministic jitter.
        user: The requester; every ticket in the fixture has a distinct user.
        clearance: Document clearance of the requester, ``"public"`` or
            ``"internal"``; enforced by retrieval before ranking.
        text: What the requester wrote.
        kind: The fixture family (``plain``, ``parse``, ``doc``, ``gap``,
            ``jargon``, ``internal``, ``status``, ``write``), which drives the
            scripted model's behavior.
        intent: The classification the grader expects.
        expected: ``"answer"`` or ``"escalate"`` — the correct outcome.
        needs_fact: Exact substring the draft must contain, or None.
        write_required: Whether success requires exactly one ledger receipt.
    """

    ticket_id: str
    user: str
    clearance: str
    text: str
    kind: str
    intent: str
    expected: str
    needs_fact: str | None = None
    write_required: bool = False


def build_suite() -> list[Ticket]:
    """Build the fixed 50-ticket evaluation suite the whole ladder is scored on.

    The families are chosen so that each capability layer has a measurable job:
    ``parse`` tickets defeat freeform output, ``doc``/``jargon`` tickets need
    retrieval, ``internal`` tickets must be escalated (their evidence is
    clearance-blocked), ``status`` needs the live lookup, ``gap`` tickets ask
    for a fact the corpus never contained, and ``write`` tickets require a
    durable note. The suite is data; nothing in the runner may read the
    ground-truth fields.

    Returns:
        The 50 tickets, in a fixed order.
    """
    tickets: list[Ticket] = []

    def add(kind: str, intent: str, expected: str, text: str,
            needs_fact: str | None = None, write_required: bool = False,
            clearance: str = "public") -> None:
        tid = f"{kind}-{sum(t.kind == kind for t in tickets):02d}"
        tickets.append(Ticket(tid, f"user-{len(tickets):02d}", clearance, text,
                              kind, intent, expected, needs_fact, write_required))

    plain = {
        "billing": ["Please refund the duplicate charge from this morning.",
                    "My invoice shows a plan we cancelled.",
                    "The billing contact on our account is wrong.",
                    "You charged the old card instead of the new one."],
        "access": ["I forgot my password again and the reset mail never lands.",
                   "My login loops back to the sign-in page forever.",
                   "New teammate cannot log in with her invite.",
                   "The password rules reject everything I type."],
        "export": ["My export never finishes, it just spins.",
                   "The export file arrives empty every time.",
                   "Can you re-run the export for our workspace?",
                   "Our export is missing the archived projects."],
        "sync": ["Sync is stuck on one file and will not move on.",
                 "Two laptops keep syncing different versions of the same doc.",
                 "Sync says complete but the folder is stale.",
                 "After the update, sync eats my local edits."],
    }
    for intent, texts in plain.items():
        for text in texts:
            add("plain", intent, "answer", text)

    burying = [
        ("billing", "So here is the thing about the charge that appeared twice on our statement view."),
        ("billing", "Something odd happened with our invoice this cycle and I cannot make sense of it."),
        ("access", "It has been a strange week and now the login will not let me in at all."),
        ("access", "After the reorg my password stopped working on the second workspace."),
        ("export", "Long story, but the export we rely on for audits quietly broke."),
        ("sync", "Ever since Tuesday the sync between my machines has been cursed."),
    ]
    for intent, text in burying:
        add("parse", intent, "answer", text)

    docs_needed = [
        ("export", "How much storage does the Pro plan actually include?", "250 GB"),
        ("billing", "What storage limit do we get before you start billing overage?", "250 GB"),
        ("export", "Is the Pro plan storage limit per workspace or per user?", "250 GB"),
        ("access", "We rotated our identity provider; what do I upload to fix SSO?", "SAML metadata"),
        ("access", "Where in the admin console does the new SSO metadata go?", "SAML metadata"),
        ("export", "What time do the nightly exports actually run?", "02:00 UTC"),
        ("export", "Our nightly export lands late; when is it scheduled?", "02:00 UTC"),
        ("export", "When should I expect the nightly export to be done by?", "02:00 UTC"),
        ("billing", "Which day of the month are invoices issued?", "first business day"),
        ("billing", "When does the invoice for last month get generated?", "first business day"),
    ]
    for intent, text, fact in docs_needed:
        add("doc", intent, "answer", text, needs_fact=fact)

    add("gap", "general", "answer",
        "Where does our data actually live after the residency addendum?",
        needs_fact="residency addendum")
    add("gap", "general", "answer",
        "Which region hosts the analytics replica now?",
        needs_fact="analytics replica region")

    add("jargon", "sync", "answer", "Nimbus keeps dropping my edits after I reconnect.")
    add("jargon", "sync", "answer", "Nimbus swallowed a whole afternoon of work.")
    add("jargon", "billing", "answer", "Basalt flagged us again and support is stumped.")

    add("internal", "access", "escalate",
        "What is the current staging VPN key and when does it rotate?")
    add("internal", "access", "escalate",
        "Can you paste the on-call escalation chain for the payments pod?")
    add("internal", "export", "escalate",
        "I need the internal retention override procedure for a legal hold.")

    add("status", "outage", "answer",
        "Is this afternoon's outage over yet? The status page still shows red for us.",
        needs_fact="14:20 UTC")

    write_texts = [
        ("billing", "Refund posted wrong; please note the correction on ticket."),
        ("billing", "Confirm the credit and note it for finance."),
        ("access", "Access restored for the contractor; please record it."),
        ("access", "Note that the lockout was a false positive."),
        ("export", "Export re-run worked; log the resolution."),
        ("export", "Mark the export incident resolved on our ticket."),
        ("sync", "Sync conflict resolved by support; please note the fix."),
        ("sync", "Record that the stale-folder issue is closed."),
    ]
    for intent, text in write_texts:
        add("write", intent, "answer", text, write_required=True)
    add("write", "sync", "answer",
        "Please write up the full history of this sync saga on the ticket: "
        "it began three weeks ago after the office move, touched four machines, "
        "two operating systems, a stubborn firewall, and at least one power cut, and "
        "every support step so far should be preserved verbatim for the audit "
        "trail we owe our compliance team after the last incident review.",
        write_required=True)

    return tickets


@dataclass(frozen=True)
class Trial:
    """One ticket's measured outcome: verdict, reason, and the cost axes."""

    ticket_id: str
    ok: bool
    reason: str
    latency_ms: float
    cost_usd: float
    escalated: bool = False
    approvals: int = 0
    interventions: int = 0


def grade(ticket: Ticket, parsed: bool, intent: str, draft: str,
          escalated: bool, receipts: int) -> tuple[bool, str]:
    """Score one trial against the ticket's ground truth; every predicate must hold.

    The predicates are conjunctive on purpose: a polite draft with the wrong
    classification fails, a correct draft that should have been an escalation
    fails, and a perfect reply without its required durable write fails. One
    axis is never allowed to launder another.

    Args:
        ticket: The ticket with its expectations.
        parsed: Whether the harness obtained a structured result.
        intent: The classified intent, when parsed.
        draft: The reply draft, when parsed.
        escalated: Whether the system routed this ticket to a human.
        receipts: Durable write receipts recorded for this ticket.

    Returns:
        ``(ok, reason)`` where ``reason`` names the first failed predicate, or
        ``"pass"``.
    """
    if ticket.expected == "escalate":
        return (True, "pass") if escalated else (False, "should-have-escalated")
    if escalated:
        return False, "escalated-supported-task"
    if not parsed:
        return False, "unparseable-reply"
    if intent != ticket.intent:
        return False, "wrong-intent"
    if ticket.needs_fact and ticket.needs_fact not in draft:
        return False, "ungrounded-draft"
    if ticket.write_required and receipts != 1:
        return False, "no-durable-write"
    return True, "pass"


INTENT_KEYWORDS = {
    "billing": ("invoice", "charge", "billing", "refund", "credit", "card",
                "statement", "overage"),
    "access": ("password", "login", "log in", "locked", "lockout", "sso",
               "sign-in", "vpn", "access", "on-call", "retention"),
    "export": ("export", "backup", "storage"),
    "sync": ("sync", "syncing", "stale-folder"),
    "outage": ("outage", "down", "status page"),
}

CODENAMES = {"Nimbus": "sync", "Basalt": "billing"}


def classify(text: str, evidence: list) -> str:
    """Classify a ticket the way the stub model does: keywords, then glossary.

    The script is blunt about what it models: common phrasing is classified
    correctly, and internal codenames are resolved only when the glossary
    passage is present in the provided evidence — the mechanism by which
    retrieval improves classification on this fixture.

    Args:
        text: The ticket text.
        evidence: Retrieved chunks (each with a ``.text`` attribute).

    Returns:
        The intent label the stub will emit.
    """
    lowered = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return intent
    joined = " ".join(chunk.text for chunk in evidence)
    for codename, intent in CODENAMES.items():
        if codename.lower() in lowered and codename in joined:
            return intent
    return "general"


def stub_reply(ticket: Ticket, evidence: list, status_note: str | None,
               structured: bool, strict: bool = False) -> str:
    """The chapter's model: a deterministic, visible script — not a network call.

    The script answers from whatever is in front of it. A fact is included only
    when some evidence chunk or the status note actually contains it; with no
    supporting passage the script fabricates a confident, uncited figure,
    because that is the failure mode the surrounding architecture must catch.
    With ``structured`` it emits clean JSON; without it, some tickets get the
    label buried mid-sentence, unless ``strict`` reformatting is requested,
    which this script honors only for even-numbered tickets.

    Args:
        ticket: The ticket being answered.
        evidence: Retrieved chunks visible to the model.
        status_note: A status-service line, if the harness fetched one.
        structured: Emit schema JSON instead of freeform prose.
        strict: A reformat retry; honored for half the burying tickets.

    Returns:
        The raw model output, freeform or JSON.
    """
    intent = classify(ticket.text, evidence)
    sources: list[str] = []
    supporting = [c for c in evidence if ticket.needs_fact and ticket.needs_fact in c.text]
    if supporting:
        chunk = supporting[0]
        draft = f"Per our documentation, {ticket.needs_fact} applies here [per {chunk.source_id}]."
        sources = [chunk.source_id]
    elif status_note and ticket.needs_fact and ticket.needs_fact in status_note:
        draft = f"Live status: {ticket.needs_fact} [per status:sync]."
        sources = ["status:sync"]
    elif ticket.needs_fact or ticket.kind == "internal":
        draft = "I believe the default is 10 GB and rotation happens on day 30."
    else:
        draft = "I have flagged this on our side; reply with what you see and we will close it out."
        if ticket.write_required:
            draft = f"Done — resolving as described: {ticket.text}"
    if structured:
        return json.dumps({"intent": intent, "urgency": "normal",
                           "draft": draft, "sources": sources})
    if ticket.kind == "parse" and not (strict and int(ticket.ticket_id[-1]) % 2 == 0):
        return f"Happy to dig in. {draft} This smells like a {intent} problem to me."
    return f"intent: {intent}\n{draft}\nsources: {', '.join(sources) or 'none'}"


def parse_reply(raw: str, structured: bool) -> tuple[bool, str, str, list[str]]:
    """Parse the model output into ``(parsed, intent, draft, sources)``.

    The freeform path is the brittle one on purpose: it accepts only replies
    that lead with an ``intent:`` line, which is exactly what the burying
    tickets defeat. The structured path validates JSON and required fields.

    Args:
        raw: The raw model output.
        structured: Whether the harness requested schema JSON.

    Returns:
        A tuple ``(parsed, intent, draft, sources)``; on failure ``parsed`` is
        False and the rest are empty.
    """
    if structured:
        try:
            data = json.loads(raw)
            return True, str(data["intent"]), str(data["draft"]), list(data["sources"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return False, "", "", []
    match = re.match(r"intent:\s*(\w+)\n(.*?)\nsources:\s*(.*)", raw, re.DOTALL)
    if not match:
        return False, "", "", []
    sources = [] if match.group(3).strip() == "none" else [
        s.strip() for s in match.group(3).split(",")]
    return True, match.group(1), match.group(2).strip(), sources


PRICE_IN = 3.00 / 1_000_000    # USD per input token (illustrative constant)
PRICE_OUT = 15.00 / 1_000_000  # USD per output token

LATENCY_MS = {
    "overhead": 30.0, "model_call": 420.0, "per_output_token": 0.35,
    "retrieval": 40.0, "status": 260.0, "memory_op": 25.0, "write": 90.0,
}

TOOL_COST_USD = {"retrieval": 0.00002, "status": 0.0001,
                 "memory_op": 0.00001, "write": 0.0002}


def tokens(text: str) -> int:
    """Estimate tokens at four characters each — crude but monotone."""
    return max(1, len(text) // 4)


def jitter_ms(ticket_id: str) -> float:
    """Deterministic per-ticket latency jitter derived from the ticket id."""
    return int(hashlib.sha256(ticket_id.encode()).hexdigest(), 16) % 90


@dataclass(frozen=True)
class LayerConfig:
    """Which capability layers are wired in. Rungs are configs; so are ablations."""

    schema: bool = False
    retrieval: bool = False
    status_tool: bool = False
    workflow: bool = False
    loop: bool = False
    memory: bool = False
    gated_write: bool = False


RUNGS: list[tuple[str, LayerConfig]] = [
    ("0 one call", LayerConfig()),
    ("1 + schema", LayerConfig(schema=True)),
    ("2 + retrieval", LayerConfig(schema=True, retrieval=True)),
    ("3 + read tool", LayerConfig(schema=True, retrieval=True, status_tool=True)),
    ("4 + workflow", LayerConfig(schema=True, retrieval=True, status_tool=True,
                                 workflow=True)),
    ("5 + loop", LayerConfig(schema=True, retrieval=True, status_tool=True,
                             workflow=True, loop=True)),
    ("6 + memory", LayerConfig(schema=True, retrieval=True, status_tool=True,
                               workflow=True, loop=True, memory=True)),
    ("7 + gated write", LayerConfig(schema=True, retrieval=True, status_tool=True,
                                    workflow=True, loop=True, memory=True,
                                    gated_write=True)),
]


def attack_edges(config: LayerConfig) -> list[str]:
    """Enumerate the named trust edges this configuration exposes.

    A count is a crude summary, but every edge on the list is a sentence a
    threat model must answer, which is why the ledger stores the enumeration
    rather than a security score. Schema validation and the deterministic
    workflow add no edges: they constrain flows that already exist.

    Args:
        config: The capability layers wired in.

    Returns:
        The named edges, two per authority-adding layer.
    """
    edges = ["user ticket text -> model instructions",
             "model draft -> requester"]
    if config.retrieval:
        edges += ["retrieved document text -> model context (indirect injection)",
                  "query -> cross-tenant index boundary"]
    if config.status_tool:
        edges += ["model context -> external status read",
                  "status payload -> model context"]
    if config.loop:
        edges += ["model -> turn and token budget (denial of wallet)",
                  "tool output -> next-action selection"]
    if config.memory:
        edges += ["session content -> persistent store",
                  "persistent store -> future sessions"]
    if config.gated_write:
        edges += ["approved proposal -> external ticket write",
                  "proposal volume -> reviewer attention"]
    return edges


@dataclass(frozen=True)
class RungReport:
    """One comparable row of the ladder: five axes, same units at every rung.

    Args:
        rung: Display name of the configuration measured.
        trials: Suite size.
        successes: Trials passing every grader predicate.
        p95_latency_ms: Modeled 95th-percentile task latency.
        cost_per_task_usd: Modeled mean cost per task (model tokens plus tools).
        attack_edges: Count of named trust edges from :func:`attack_edges`.
        approvals_per_100: Human approval actions per 100 tasks.
        escalations_per_100: Tickets routed to a human per 100 tasks.
        interventions_per_100: Operator reviews of stuck runs per 100 tasks.
    """

    rung: str
    trials: int
    successes: int
    p95_latency_ms: float
    cost_per_task_usd: float
    attack_edges: int
    approvals_per_100: float
    escalations_per_100: float
    interventions_per_100: float

    def __post_init__(self) -> None:
        if not 0 <= self.successes <= self.trials:
            raise ValueError("successes must lie in [0, trials]")
        if min(self.p95_latency_ms, self.cost_per_task_usd, self.attack_edges,
               self.approvals_per_100, self.escalations_per_100,
               self.interventions_per_100) < 0:
            raise ValueError("axes must be non-negative")

    @property
    def task_success(self) -> float:
        """Fraction of trials that passed every predicate."""
        return self.successes / self.trials

    @property
    def operator_burden(self) -> float:
        """Human actions per 100 tasks: approvals + escalations + interventions."""
        return (self.approvals_per_100 + self.escalations_per_100
                + self.interventions_per_100)


def run_ticket(ticket: Ticket, config: LayerConfig, world: dict) -> Trial:
    """Run one ticket through whichever layers the configuration wires in.

    The runner knows the whole ladder's shape; each layer's implementation
    arrives in the section that teaches it, and Python resolves the stage
    functions at call time. Disabled layers cost nothing and add nothing —
    which is exactly what makes rung-to-rung deltas attributable.

    Args:
        ticket: The ticket to run.
        config: Which layers are active.
        world: Per-run shared state (indexes, memory store, effect ledger,
            resource versions), built by :func:`make_world`.

    Returns:
        The graded, costed :class:`Trial`.
    """
    latency = LATENCY_MS["overhead"] + jitter_ms(ticket.ticket_id)
    cost = 0.0
    escalated, approvals, interventions, receipts = False, 0, 0, 0
    evidence: list = []
    status_note: str | None = None

    if config.memory:
        scope = ch18.Scope(tenant_id="acme", user_id=ticket.user)
        record = world["memory"].retrieve(ticket.text, scope)
        world["memory_hits"] += record is not None
        latency += LATENCY_MS["memory_op"]
        cost += TOOL_COST_USD["memory_op"]

    if config.loop:
        parsed, intent, draft, sources, loop_latency, loop_cost, spun = stage_loop(
            ticket, config, world)
        latency += loop_latency
        cost += loop_cost
        if spun:
            interventions += 1
        evidence = acl_search(ticket, world["indexes"]) if config.retrieval else []
        outage = any(k in ticket.text.lower() for k in INTENT_KEYWORDS["outage"])
        status_note = status_service("sync") if config.status_tool and outage else None
    else:
        if config.retrieval:
            evidence = acl_search(ticket, world["indexes"])
            latency += LATENCY_MS["retrieval"]
            cost += TOOL_COST_USD["retrieval"]
        if config.status_tool:
            wanted = (any(k in ticket.text.lower() for k in INTENT_KEYWORDS["outage"])
                      if config.workflow else True)
            if wanted:
                status_note = status_service("sync")
                latency += LATENCY_MS["status"]
                cost += TOOL_COST_USD["status"]
        prompt = ticket.text + " ".join(c.text for c in evidence) + (status_note or "")
        raw = stub_reply(ticket, evidence, status_note, structured=config.schema)
        latency += LATENCY_MS["model_call"] + LATENCY_MS["per_output_token"] * tokens(raw)
        cost += PRICE_IN * (tokens(prompt) + 120) + PRICE_OUT * tokens(raw)
        parsed, intent, draft, sources = parse_reply(raw, config.schema)

    if config.workflow and not parsed:
        raw = stub_reply(ticket, evidence, status_note, structured=config.schema,
                         strict=True)
        latency += LATENCY_MS["model_call"] + LATENCY_MS["per_output_token"] * tokens(raw)
        cost += PRICE_IN * (tokens(ticket.text) + 160) + PRICE_OUT * tokens(raw)
        parsed, intent, draft, sources = parse_reply(raw, config.schema)

    if config.workflow and parsed:
        ok, _ = policy_check(draft, sources, evidence, status_note)
        if not ok:
            escalated = True

    if config.memory and parsed:
        scope = ch18.Scope(tenant_id="acme", user_id=ticket.user)
        candidate = ch18.Candidate(
            key=f"{ticket.user} last-intent", value=intent, kind=ch18.Kind.SEMANTIC,
            scope=scope, source=ch18.Source.USER, evidence_id=ticket.ticket_id,
            event_time=world["clock"])
        world["memory"].write(candidate)
        world["clock"] += 1
        latency += LATENCY_MS["memory_op"]
        cost += TOOL_COST_USD["memory_op"]

    if ticket.write_required and parsed and not escalated and config.gated_write:
        proposal = ActionProposal("acme", ticket.ticket_id, f"[{intent}] {draft}",
                                  "internal_note",
                                  world["versions"][ticket.ticket_id])
        approvals += 1
        approval_hash, _ = scripted_reviewer(proposal)
        latency += LATENCY_MS["write"]
        cost += TOOL_COST_USD["write"]
        if approval_hash is None:
            escalated = True
        else:
            receipt, _ = execute_write(proposal, approval_hash,
                                       world["versions"], world["ledger"])
            receipts = 1 if receipt is not None else 0

    ok, reason = grade(ticket, parsed, intent if parsed else "",
                       draft if parsed else "", escalated, receipts)
    return Trial(ticket.ticket_id, ok, reason, latency, cost,
                 escalated, approvals, interventions)


def make_world(config: LayerConfig) -> dict:
    """Build the per-run shared state each enabled layer needs, fresh every run.

    Args:
        config: The layer configuration about to be measured.

    Returns:
        A dict with retrieval indexes, a fresh memory store, a fresh effect
        ledger, resource versions for every ticket, and a logical clock.
    """
    world: dict = {"clock": 0}
    if config.retrieval or config.loop:
        world["indexes"] = build_indexes()
    if config.memory:
        world["memory"] = make_memory()
        world["memory_hits"] = 0
    if config.gated_write:
        world["ledger"] = EffectLedger()
    world["versions"] = {t.ticket_id: 1 for t in build_suite()}
    return world


def run_suite(name: str, config: LayerConfig,
              tickets: list[Ticket]) -> tuple[RungReport, list[Trial]]:
    """Run the whole suite under one configuration and aggregate the five axes.

    Args:
        name: Display name for the report row.
        config: The layer configuration to measure.
        tickets: The fixed evaluation suite.

    Returns:
        The aggregated :class:`RungReport` and the per-ticket trials.
    """
    world = make_world(config)
    trials = [run_ticket(ticket, config, world) for ticket in tickets]
    latencies = sorted(t.latency_ms for t in trials)
    p95 = latencies[max(0, -(-len(latencies) * 95 // 100) - 1)]
    per100 = 100.0 / len(trials)
    report = RungReport(
        rung=name, trials=len(trials),
        successes=sum(t.ok for t in trials),
        p95_latency_ms=round(p95, 1),
        cost_per_task_usd=round(sum(t.cost_usd for t in trials) / len(trials), 6),
        attack_edges=len(attack_edges(config)),
        approvals_per_100=round(sum(t.approvals for t in trials) * per100, 1),
        escalations_per_100=round(sum(t.escalated for t in trials) * per100, 1),
        interventions_per_100=round(sum(t.interventions for t in trials) * per100, 1),
    )
    return report, trials


ch14 = load_chapter_module("ch14", "ch14_rag")

CORPUS = [
    ch14.Document("kb-plans", "kb", "Plan limits", "v3", True,
                  "Plan limits and storage. The Pro plan includes 250 GB of storage "
                  "per workspace. Storage above the included 250 GB is billed as "
                  "overage at the end of the cycle. Team plans share the same "
                  "workspace storage pool and the same overage rules."),
    ch14.Document("kb-sso", "kb", "Single sign-on", "v2", True,
                  "Single sign-on setup. After an identity provider rotation, "
                  "upload the new SAML metadata file in the admin console under "
                  "Security. Until the SAML metadata is refreshed, sign-in falls "
                  "back to the previous provider and may fail for new users."),
    ch14.Document("kb-export", "kb", "Exports", "v5", True,
                  "Exports and backups. Nightly exports run at 02:00 UTC for every "
                  "workspace. A re-run can be requested once per day. Export files "
                  "include archived projects unless the archive flag is cleared."),
    ch14.Document("kb-billing", "kb", "Invoices", "v4", True,
                  "Invoices and credits. Invoices are issued on the first business "
                  "day of each month. Credits apply to the next invoice. Duplicate "
                  "charges are reversed to the original payment method."),
    ch14.Document("kb-glossary", "kb", "Codenames", "v1", True,
                  "Internal codenames, safe to share. Nimbus is the sync engine "
                  "that reconciles offline edits. Basalt is the billing pipeline "
                  "that posts charges and credits. Cinder is the export scheduler."),
    ch14.Document("kb-vpn", "kb", "Staging VPN", "v7", True,
                  "INTERNAL ONLY. Staging VPN pre-shared keys rotate every 30 days "
                  "via the infra vault. The current key lives in the vault path "
                  "infra/staging-vpn and must never be pasted into tickets."),
    ch14.Document("kb-oncall", "kb", "Escalation chain", "v2", True,
                  "INTERNAL ONLY. The payments pod on-call escalation chain and the "
                  "retention override procedure for legal holds are documented in "
                  "the operations handbook, access-controlled by role."),
]

DOC_CLEARANCE = {"kb-vpn": "internal", "kb-oncall": "internal"}


def build_indexes() -> dict[str, object]:
    """Build one BM25 index per clearance level, filtering before ranking.

    Authorization is applied to the corpus before the index exists, so an
    unauthorized chunk can never appear in a ranked list — the ACL-before-
    similarity rule from @sec-ch14 and @sec-ch24, enforced structurally.

    Returns:
        A dict mapping clearance level to a BM25 index over permitted chunks.
    """
    chunks = ch14.chunk_documents(CORPUS, size=48, overlap=9)
    public = [c for c in chunks if DOC_CLEARANCE.get(c.source_id, "public") == "public"]
    return {"public": ch14.BM25(public), "internal": ch14.BM25(chunks)}


def acl_search(ticket: Ticket, indexes: dict, k: int = 2) -> list:
    """Retrieve the top-k chunks the requester is cleared to see.

    Args:
        ticket: The ticket whose text is the query and whose clearance picks
            the index.
        indexes: Clearance-keyed BM25 indexes from :func:`build_indexes`.
        k: How many chunks to return.

    Returns:
        The permitted chunks, best first.
    """
    index = indexes[ticket.clearance]
    lookup = {c.chunk_id: c for c in index.chunks}
    return [lookup[hit.chunk_id] for hit in index.search(ticket.text, k)]


def status_service(service: str = "sync") -> str:
    """The one read-only tool: a stub status page with a fixed answer."""
    return f"status:{service} operational since 14:20 UTC (incident INC-88 resolved)"


def policy_check(draft: str, sources: list[str], evidence: list,
                 status_note: str | None) -> tuple[bool, str]:
    """The workflow's deterministic output gate: cite it or do not ship it.

    Two rules, both code. A draft that states a figure (any digit) without a
    ``[per ...]`` citation is refused — deliberately crude, and it catches
    exactly the confident fabrications our stub produces. A draft whose
    citation names a source that was not actually in front of the model is
    refused as a citation mismatch. Refusal means escalation to a human, never
    silent delivery.

    Args:
        draft: The reply draft about to be delivered.
        sources: The sources the reply claims to rest on.
        evidence: The chunks the harness actually retrieved.
        status_note: The status line the harness actually fetched, if any.

    Returns:
        ``(ok, reason)``; on refusal the reason names the rule that fired.
    """
    seen = {chunk.source_id for chunk in evidence}
    if status_note:
        seen.add("status:sync")
    for source in sources:
        if source not in seen:
            return False, "citation mismatch"
    if re.search(r"\d", draft) and "[per " not in draft:
        return False, "uncited factual claim"
    return True, "ok"


ch16 = load_chapter_module("ch16", "ch16_loop")

SPIN_TICKETS = frozenset({"plain-03", "plain-11"})  # scripted to spin: never answer


def loop_tools(ticket: Ticket, indexes: dict, use_retrieval: bool) -> dict:
    """Build the two read-only tools the adaptive loop may call.

    Args:
        ticket: The ticket, which fixes the clearance the search enforces.
        indexes: Clearance-keyed BM25 indexes.
        use_retrieval: When False the search tool returns nothing, which is how
            an ablation unplugs retrieval without changing the loop's surface.

    Returns:
        A ch16 ``ToolSpec`` registry with ``search_docs`` and ``service_status``.
    """
    def search_docs(query: str) -> str:
        if not use_retrieval:
            return "[]"
        probe = replace(ticket, text=query)
        return json.dumps([{"source_id": c.source_id, "text": c.text}
                           for c in acl_search(probe, indexes)])

    def service_status_tool(service: str) -> str:
        return status_service(service)

    return {
        "search_docs": ch16.ToolSpec("Search the KB the requester may read.",
                                     {"query": str}, search_docs),
        "service_status": ch16.ToolSpec("Read the public status page.",
                                        {"service": str}, service_status_tool),
    }


def scripted_loop_model(ticket: Ticket, indexes: dict, structured: bool,
                        use_retrieval: bool, use_status: bool) -> object:
    """Script the adaptive-loop model's turns for one ticket, deterministically.

    Because every component is deterministic we can precompute what retrieval
    will return and script the loop model's turns exactly: search, optionally
    check status, then answer with the same reply the workflow path would
    produce. Two tickets are scripted to spin — they re-issue the same search
    until the loop's turn ceiling fires — because a loop that cannot fail to
    terminate teaches nothing about why the ceiling exists.

    Args:
        ticket: The ticket to script.
        indexes: Clearance-keyed BM25 indexes.
        structured: Whether the answer turn emits schema JSON.
        use_retrieval: Whether the search actually returns evidence.
        use_status: Whether the status tool is wired in.

    Returns:
        A ch16 ``ScriptedModel`` ready for ``run_agent``.
    """
    if ticket.ticket_id in SPIN_TICKETS:
        turns = [ch16.tool_call_message(f"c{i}", "search_docs", {"query": ticket.text})
                 for i in range(1, 9)]
        return ch16.ScriptedModel(turns)
    evidence = acl_search(ticket, indexes) if use_retrieval else []
    outage = any(k in ticket.text.lower() for k in INTENT_KEYWORDS["outage"])
    status_note = status_service("sync") if use_status and outage else None
    reply = stub_reply(ticket, evidence, status_note, structured=structured)
    turns = [ch16.tool_call_message("c1", "search_docs", {"query": ticket.text})]
    if status_note:
        turns.append(ch16.tool_call_message("c2", "service_status", {"service": "sync"}))
    turns.append(ch16.answer_message(reply))
    return ch16.ScriptedModel(turns)


def stage_loop(ticket: Ticket, config: LayerConfig, world: dict) -> tuple:
    """Run one ticket under the @sec-ch16 adaptive loop and measure what it spent.

    Token spend is read off the loop's own prompt log, so the cost of transcript
    growth is measured, not assumed. A run that hits the turn ceiling falls back
    to the deterministic path — the answer is preserved, the spend is not.

    Args:
        ticket: The ticket to run.
        config: The active layers; the loop honors the same schema, retrieval,
            and status toggles the direct path does, so ablations stay fair.
        world: Per-run shared state with the retrieval indexes.

    Returns:
        ``(parsed, intent, draft, sources, latency_ms, cost_usd, spun)``.
    """
    tools = loop_tools(ticket, world["indexes"], config.retrieval)
    model = scripted_loop_model(ticket, world["indexes"], config.schema,
                                config.retrieval, config.status_tool)
    result = ch16.run_agent(ticket.text, model, tools, ch16.allow_all,
                            ch16.Limits(max_turns=5))
    tool_calls = len(result.state.observations)
    prompt_tokens = sum(result.state.prompt_token_log)
    output_tokens = 60 * result.state.turns
    latency = (result.state.turns * LATENCY_MS["model_call"]
               + LATENCY_MS["per_output_token"] * output_tokens
               + tool_calls * LATENCY_MS["retrieval"])
    cost = (PRICE_IN * prompt_tokens + PRICE_OUT * output_tokens
            + tool_calls * TOOL_COST_USD["retrieval"])
    if result.stop is not ch16.Stop.ANSWERED:
        evidence = acl_search(ticket, world["indexes"]) if config.retrieval else []
        raw = stub_reply(ticket, evidence, None, structured=config.schema)
        parsed, intent, draft, sources = parse_reply(raw, structured=config.schema)
        latency += LATENCY_MS["model_call"] + LATENCY_MS["retrieval"]
        cost += PRICE_IN * (tokens(ticket.text) + 120) + PRICE_OUT * tokens(raw)
        return parsed, intent, draft, sources, latency, cost, True
    parsed, intent, draft, sources = parse_reply(result.answer, structured=config.schema)
    return parsed, intent, draft, sources, latency, cost, False


ch18 = load_chapter_module("ch18", "ch18_memory")


def make_memory() -> object:
    """A fresh governed memory store (the @sec-ch18 artifact) for one run."""
    return ch18.MemoryStore()


@dataclass(frozen=True)
class ActionProposal:
    """A typed, hashable proposal to write one note to one ticket.

    The reviewer approves a hash of exactly these fields, so approval binds
    the action's full content and its resource version — not a paraphrase of
    it. Any drift between what was approved and what reaches the executor
    changes the hash and is refused at execution time.

    Args:
        tenant: The tenant whose ticket is written.
        ticket_id: The ticket receiving the note.
        note: The note body.
        visibility: Where the note is visible; only ``"internal_note"`` is
            approvable in this fixture.
        resource_version: The ticket version the proposal was built against.
    """

    tenant: str
    ticket_id: str
    note: str
    visibility: str
    resource_version: int

    def action_hash(self) -> str:
        """Hash the normalized action content; this is what a reviewer approves."""
        payload = json.dumps(
            [self.tenant, self.ticket_id, self.note, self.visibility,
             self.resource_version], separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def scripted_reviewer(proposal: ActionProposal) -> tuple[str | None, str]:
    """The fixture's reviewer: a visible script with a scope and a size rule.

    It approves internal notes of bounded size and nothing else — the two
    checks a human reviewer is actually asked to make here. The size bound is
    what rejects the fixture's one oversized note; approval is not a rubber
    stamp even in a deterministic fixture.

    Args:
        proposal: The typed action awaiting review.

    Returns:
        ``(approval_hash, reason)``; the hash is None when the proposal is
        rejected.
    """
    if proposal.visibility != "internal_note":
        return None, "rejected: visibility out of scope"
    if len(proposal.note) > 240:
        return None, "rejected: note exceeds 240 characters, summarize instead"
    return proposal.action_hash(), "approved"


class EffectLedger:
    """An append-only, idempotent record of external effects.

    One idempotency key maps to at most one effect. A duplicate delivery gets
    the original receipt back instead of a second effect — the property that
    makes retries and duplicated queue messages safe (@sec-ch26).
    """

    def __init__(self) -> None:
        self.effects: dict[str, dict] = {}

    def record(self, idempotency_key: str, receipt: dict) -> tuple[dict, bool]:
        """Record an effect once; replay returns the original receipt.

        Args:
            idempotency_key: Stable identity of the intended effect.
            receipt: The receipt to store on first delivery.

        Returns:
            ``(receipt, fresh)`` — the stored receipt and whether this call
            created it.
        """
        if idempotency_key in self.effects:
            return self.effects[idempotency_key], False
        self.effects[idempotency_key] = receipt
        return receipt, True


def execute_write(proposal: ActionProposal, approval_hash: str,
                  versions: dict[str, int], ledger: EffectLedger) -> tuple[dict | None, str]:
    """Revalidate at the point of effect, then record exactly one write.

    Approval happened earlier, against a hash; the world may have moved since.
    So the executor re-derives the hash (did the action drift?), re-reads the
    resource version (did the ticket change since review?), and only then
    records the effect under an idempotency key (was this delivery a
    duplicate?). Chapter 17 taught this as stale-approval revalidation; here
    it is simply how every write runs.

    Args:
        proposal: The typed action to execute.
        approval_hash: The hash the reviewer approved.
        versions: Current resource versions, keyed by ticket id.
        ledger: The append-only effect ledger.

    Returns:
        ``(receipt, reason)``; the receipt is None when a check refused the
        write.
    """
    if proposal.action_hash() != approval_hash:
        return None, "blocked: approval does not match action"
    if versions.get(proposal.ticket_id) != proposal.resource_version:
        return None, "blocked: stale resource version"
    key = f"note:{proposal.ticket_id}:v{proposal.resource_version}"
    receipt, fresh = ledger.record(key, {"ticket_id": proposal.ticket_id,
                                         "note_hash": proposal.action_hash()})
    return receipt, "recorded" if fresh else "duplicate: original receipt returned"


def ablate(full_config: LayerConfig, tickets: list[Ticket]) -> dict[str, dict[str, float]]:
    """Remove one layer at a time from the maximal stack and re-measure live.

    Ablation answers a different question than the ladder: the ladder measures
    what a layer added on the way up; ablation measures what it is carrying in
    the finished system, interactions included. The deltas are full-stack minus
    ablated, so a positive success delta means the layer is load-bearing.

    Args:
        full_config: The maximal configuration (every layer on).
        tickets: The fixed suite.

    Returns:
        Per-layer deltas: success points, cost, p95 latency, attack edges, and
        operator burden.
    """
    full_report, _ = run_suite("full", full_config, tickets)
    deltas: dict[str, dict[str, float]] = {}
    for layer in ("schema", "retrieval", "status_tool", "workflow",
                  "loop", "memory", "gated_write"):
        ablated_config = replace(full_config, **{layer: False})
        ablated, _ = run_suite(f"-{layer}", ablated_config, tickets)
        deltas[layer] = {
            "success_pts": round(100 * (full_report.task_success - ablated.task_success), 1),
            "cost_usd": round(full_report.cost_per_task_usd - ablated.cost_per_task_usd, 6),
            "p95_ms": round(full_report.p95_latency_ms - ablated.p95_latency_ms, 1),
            "edges": full_report.attack_edges - ablated.attack_edges,
            "burden": round(full_report.operator_burden - ablated.operator_burden, 1),
        }
    return deltas


def earn_decisions(deltas: dict[str, dict[str, float]],
                   min_gain_pts: float = 2.5) -> dict[str, str]:
    """Turn ablation deltas into keep/cut decisions against a pre-declared bar.

    The bar exists to stop motivated reasoning: it was fixed before the runs at
    2.5 success points — more than one ticket on a 50-ticket suite — so a layer
    cannot be kept on the strength of a single lucky fixture item. A cut is a
    decision record, not a deletion: the memo names the evidence that would
    reopen it.

    Args:
        deltas: Per-layer ablation deltas from :func:`ablate`.
        min_gain_pts: Success points a layer must carry to stay.

    Returns:
        Per-layer verdict strings, each naming the measured delta it rests on.
    """
    decisions: dict[str, str] = {}
    for layer, delta in deltas.items():
        if delta["success_pts"] >= min_gain_pts:
            decisions[layer] = (f"keep — removing it costs {delta['success_pts']:.1f} "
                                "success points")
        else:
            decisions[layer] = (f"cut — worth only {delta['success_pts']:.1f} points "
                                f"(< {min_gain_pts}); reclaims {delta['edges']} attack "
                                f"edges and {delta['cost_usd']:.6f} USD/task")
    return decisions


def rejection_memo(layer: str, delta: dict[str, float], reopen: str) -> str:
    """Render a justified-rejection memo from measured deltas.

    Args:
        layer: The rejected layer.
        delta: Its ablation deltas.
        reopen: The observation that would reopen the decision.

    Returns:
        A compact memo string suitable for the decision log.
    """
    return (f"REJECTED: {layer}. Evidence: {delta['success_pts']:+.1f} success pts, "
            f"{delta['cost_usd']:+.6f} USD/task, {delta['p95_ms']:+.1f} ms p95, "
            f"{delta['edges']:+d} attack edges, {delta['burden']:+.1f} burden/100. "
            f"Reopen when: {reopen}")
