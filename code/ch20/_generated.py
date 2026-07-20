# Auto-generated from chapters/20-multi-agent-systems.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


def should_split(*, decomposable: bool, useful_boundary: bool, measured_failure: bool) -> bool:
    """Return True only when all three conditions for a team are present.

    A multi-agent design is justified only when the work can actually run in
    parallel, the split creates a real context/permission/model/ownership
    boundary, and the strongest single-agent baseline has a *measured* failure the
    team targets. Any missing condition is a veto: keep one agent.

    Args:
        decomposable: The work has independent or weakly coupled branches.
        useful_boundary: The split creates a real boundary, not a renamed persona.
        measured_failure: The best single-agent baseline has a named, measured gap.

    Returns:
        True if the team is justified; False if any condition fails.
    """
    return decomposable and useful_boundary and measured_failure


def tokens(text: str) -> int:
    """Count whitespace-separated words as a transparent token stand-in."""
    return len(text.split())


def keyword_score(query: str, text: str) -> int:
    """Count how many query terms appear in one document.

    This is the whole "reasoning" of the stub worker: it ranks the documents in
    its shard by shared lowercased words and reads the best match. A hosted model
    would replace the ranking with a real search-and-read loop; keeping it
    deterministic is what lets every number in this chapter reproduce.

    Args:
        query: The task query whose terms we look for.
        text: One document's text.

    Returns:
        The count of distinct query terms present in the document.
    """
    return len(set(query.lower().split()) & set(text.lower().split()))


def _best_doc(query: str, docs: list[dict]) -> dict:
    """Read a shard one document at a time, keeping the best keyword match."""
    best = None
    for doc in docs:
        if best is None or keyword_score(query, doc["text"]) > keyword_score(query, best["text"]):
            best = doc
    return best


QUERY = "on-premises deployment permitted"
OBJECTIVE = (
    "For each region, name the vendor whose data platform permits "
    "on-premises deployment, with the evidence line that supports it."
)
CORPUS = {
    "americas": [
        {"id": "am-1", "vendor": "Aurora", "on_prem": True,
         "text": "Aurora permits on-premises deployment for regulated customer data"},
        {"id": "am-2", "vendor": "Nimbus", "on_prem": False,
         "text": "Nimbus runs as a cloud only managed analytics platform"},
        {"id": "am-3", "vendor": "Delta", "on_prem": False,
         "text": "Delta streams hosted dashboards over the public internet"},
    ],
    "emea": [
        {"id": "em-1", "vendor": "Basalt", "on_prem": True,
         "text": "Basalt supports on-premises deployment inside a private data center"},
        {"id": "em-2", "vendor": "Cirro", "on_prem": False,
         "text": "Cirro is offered only as a multi tenant cloud subscription"},
        {"id": "em-3", "vendor": "Loire", "on_prem": False,
         "text": "Loire provides a hosted reporting service with no local install"},
    ],
    "apac": [
        {"id": "ap-1", "vendor": "Sakura", "on_prem": True,
         "text": "Sakura allows on-premises deployment behind the customer firewall"},
        {"id": "ap-2", "vendor": "Monsoon", "on_prem": False,
         "text": "Monsoon is a cloud native warehouse with no on site option"},
        {"id": "ap-3", "vendor": "Ganges", "on_prem": False,
         "text": "Ganges delivers analytics purely through a hosted portal"},
    ],
}
print(f"{len(CORPUS)} shards, {sum(len(v) for v in CORPUS.values())} documents")


from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TaskBrief:
    """The typed envelope one worker receives: scope, authority, and budget.

    A brief is the whole contract for a delegated subtask. It names the region the
    worker may read, the single tool it may call, the schema its report must fit,
    and the token budget it may spend. A worker can only ever narrow these; the
    brief is why authority flows one way, from parent to child.
    """

    worker_id: str
    region: str
    query: str
    allowed_tools: tuple[str, ...]
    output_schema: str
    token_budget: int


@dataclass(frozen=True)
class WorkerReport:
    """A compressed, typed finding — the only thing that crosses the handoff.

    The worker returns one finding plus the evidence id that supports it and the
    modeled tokens and latency it spent, not its scored candidates or its
    reasoning. A typed report survives a handoff that a prose summary would
    corrupt, because the merge gate can validate every field.
    """

    worker_id: str
    region: str
    finding: str
    evidence_ids: tuple[str, ...]
    tokens: int
    latency: int


class Worker:
    """A stub research worker: retrieve over one shard, report one finding.

    The worker reads only the documents named in its brief and reports the top
    match's vendor together with that document's id as provenance. Its modeled
    token cost is the brief query plus the shard it read plus its finding, and its
    latency is one unit per document scanned — the numbers a real worker would
    incur reading its context.
    """

    def run(self, brief: TaskBrief, docs: list[dict], poison: bool = False) -> WorkerReport:
        """Retrieve over the brief's shard and return one typed report.

        Args:
            brief: The authority and scope envelope for this subtask.
            docs: The documents of the worker's shard.
            poison: If True, omit the evidence id, simulating the
                information-withholding failure the merge gate must catch.

        Returns:
            A :class:`WorkerReport` grounded in the retrieved document.

        Raises:
            PermissionError: If the brief grants any tool other than ``search`` —
                a worker may not widen its own authority.
        """
        if brief.allowed_tools != ("search",):
            raise PermissionError("worker may not use tools outside its brief")
        best = _best_doc(brief.query, docs)
        evidence = () if poison else (best["id"],)
        shard_text = " ".join(doc["text"] for doc in docs)
        used = tokens(brief.query) + tokens(shard_text) + tokens(best["vendor"])
        return WorkerReport(brief.worker_id, brief.region, best["vendor"], evidence,
                            tokens=used, latency=len(docs))


import json
from pathlib import Path


class Workspace:
    """A parent-owned directory of immutable JSON handoff artifacts.

    Each worker writes exactly one report file; the orchestrator reads it back and
    validates it. The write refuses to overwrite an existing report or to escape
    the root, so the filesystem — not the workers' good behavior — enforces
    single-writer ownership and immutability of the handoff.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, report: WorkerReport) -> Path:
        """Persist one report as immutable JSON and return its path.

        Args:
            report: The typed finding to store.

        Returns:
            The path written.

        Raises:
            PermissionError: If the target path escapes the workspace root.
            FileExistsError: If a report for this worker already exists.
        """
        path = (self.root / f"{report.worker_id}.json").resolve()
        if not path.is_relative_to(self.root):
            raise PermissionError("report path escapes workspace")
        if path.exists():
            raise FileExistsError(f"report already exists: {path.name}")
        path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        return path

    def read(self, path: Path) -> WorkerReport:
        """Read one report back, restoring the tuple-typed evidence field."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["evidence_ids"] = tuple(data["evidence_ids"])
        return WorkerReport(**data)


@dataclass(frozen=True)
class Failure:
    """A MAST-style attribution: what category, which agent, which step, and why."""

    category: str
    agent: str
    step: str
    reason: str


def validate_report(report: WorkerReport, docs: list[dict]) -> Failure | None:
    """Check one report at the merge boundary; return a Failure or None.

    The gate confirms the finding names a real vendor in the shard and that the
    report carries the evidence id that grounds it. A missing evidence id is the
    information-withholding failure (MAST 2.4): the claim may be correct, but it is
    unsupported and must not be laundered into the team's answer.

    Args:
        report: The worker's typed finding.
        docs: The shard the worker was asked to read, for grounding.

    Returns:
        A :class:`Failure` for the first problem found, or None if the report
        passes schema, grounding, and provenance checks.
    """
    vendors = {doc["vendor"] for doc in docs}
    if report.finding not in vendors:
        return Failure("inter-agent misalignment", report.worker_id, "handoff", "unsupported finding")
    if not report.evidence_ids:
        return Failure("inter-agent misalignment", report.worker_id, "handoff", "missing provenance")
    return None


@dataclass(frozen=True)
class RunResult:
    """One run's comparable outcome: answer, quality, and modeled budget.

    Both the team and the single agent return this shape, so the A/B is a diff.
    ``status`` is ``"completed"`` on success or ``"contained"`` when a gate stops
    an attributed failure, keeping containment success and task success apart.
    """

    architecture: str
    status: str
    answer: str | None
    score: float
    tokens: int
    latency: int
    failure: Failure | None = None


from concurrent.futures import ThreadPoolExecutor

PLAN_LATENCY = 1
MERGE_LATENCY = 1


class Orchestrator:
    """Decompose the objective, fan out workers, and be the only writer.

    The orchestrator issues one typed brief per region, runs the workers over
    isolated shards (the map), validates each report at the merge gate, and
    synthesizes the single answer (the reduce). Workers never decide the parent
    task is done and never write the final answer, so an unvalidated finding
    cannot reach the user.
    """

    def run(self, objective: str, corpus: dict, workspace: Workspace,
            poison_worker: str | None = None, verify: bool = True) -> RunResult:
        """Run the orchestrator-worker team over the corpus.

        Args:
            objective: The parent task, restated into each worker's brief.
            corpus: Mapping of region -> list of shard documents.
            workspace: Parent-owned store for the handoff artifacts.
            poison_worker: Worker id that should omit its evidence (fault injection).
            verify: If False, skip the merge gate so unsupported findings reach the
                answer — the no-verification failure (MAST 3.2).

        Returns:
            A :class:`RunResult`; ``status`` is ``"completed"`` on success or
            ``"contained"`` when the gate stops an attributed failure.
        """
        briefs = [
            TaskBrief(f"worker-{i}", region, QUERY, ("search",), "WorkerReport/v1", token_budget=200)
            for i, region in enumerate(corpus, start=1)
        ]
        with ThreadPoolExecutor(max_workers=len(briefs)) as pool:
            paths = list(pool.map(
                lambda b: workspace.write(
                    Worker().run(b, corpus[b.region], poison=(b.worker_id == poison_worker))),
                briefs,
            ))
        reports = [workspace.read(path) for path in paths]
        findings = [report.finding for report in reports]
        answer = ", ".join(sorted(findings))

        plan_tokens = tokens(objective) + sum(tokens(brief.query) for brief in briefs)
        worker_tokens = sum(report.tokens for report in reports)
        merge_tokens = tokens(objective) + sum(tokens(f) for f in findings) + tokens(answer)
        total_tokens = plan_tokens + worker_tokens + merge_tokens
        latency = PLAN_LATENCY + max(report.latency for report in reports) + MERGE_LATENCY

        if verify:
            for brief, report in zip(briefs, reports):
                failure = validate_report(report, corpus[brief.region])
                if failure:
                    return RunResult("orchestrator-worker", "contained", None, 0.0,
                                     total_tokens, latency, failure)
        return RunResult("orchestrator-worker", "completed", answer, 1.0, total_tokens, latency)


def cost_breakdown(objective: str, corpus: dict) -> dict:
    """Decompose team and single-agent token cost by phase for one task.

    The team's plan re-reads the objective and writes a brief query per worker;
    each worker re-reads its brief and shard; the merge re-reads the objective and
    every finding. The single agent reads the objective and shards once. Exposing
    the terms shows exactly where the multiplier comes from.

    Args:
        objective: The parent task string.
        corpus: Mapping of region -> shard documents.

    Returns:
        A dict of the team's plan/workers/merge terms and totals and the single
        agent's objective/shards/answer terms and total.
    """
    regions = list(corpus)
    findings = [_best_doc(QUERY, corpus[region])["vendor"] for region in regions]
    answer = ", ".join(sorted(findings))
    plan = tokens(objective) + sum(tokens(QUERY) for _ in regions)
    workers = sum(
        tokens(QUERY) + tokens(" ".join(doc["text"] for doc in corpus[region])) + tokens(_best_doc(QUERY, corpus[region])["vendor"])
        for region in regions
    )
    merge = tokens(objective) + sum(tokens(f) for f in findings) + tokens(answer)
    shards = sum(tokens(" ".join(doc["text"] for doc in corpus[region])) for region in regions)
    single = tokens(objective) + shards + tokens(answer)
    return {"plan": plan, "workers": workers, "merge": merge, "team": plan + workers + merge,
            "obj": tokens(objective), "shards": shards, "answer": tokens(answer), "single": single}


def single_agent(objective: str, corpus: dict, token_budget: int) -> RunResult:
    """Solve the whole task in one context, sequentially, at a fixed budget.

    The single agent reads every shard in one context and writes the answer, with
    no briefs to duplicate and no reports to re-read, so it spends fewer tokens.
    But it scans the shards one after another, so its modeled latency is the sum of
    the per-shard costs rather than the maximum.

    Args:
        objective: The parent task.
        corpus: Mapping of region -> shard documents.
        token_budget: Budget granted for a cost-matched comparison; the run fails
            closed below the tokens it actually needs.

    Returns:
        A :class:`RunResult` for the single-agent architecture.
    """
    findings, scan = [], 0
    for docs in corpus.values():
        findings.append(_best_doc(QUERY, docs)["vendor"])
        scan += len(docs)
    answer = ", ".join(sorted(findings))
    shard_tokens = sum(tokens(" ".join(doc["text"] for doc in docs)) for docs in corpus.values())
    used = tokens(objective) + shard_tokens + tokens(answer)
    if token_budget < used:
        return RunResult("single-agent", "budget_exhausted", None, 0.0, token_budget, 0)
    latency = PLAN_LATENCY + scan + MERGE_LATENCY
    return RunResult("single-agent", "completed", answer, 1.0, used, latency)


from tempfile import TemporaryDirectory


def run_experiment(corpus: dict, poison_worker: str | None = "worker-2") -> dict:
    """Run the cost-matched A/B and a fault injection, returning shown numbers.

    Args:
        corpus: Region -> shard documents.
        poison_worker: Which worker omits its provenance in the fault run.

    Returns:
        A dict with the team run, the cost-matched single-agent run, the measured
        token multiplier and latency speedup, whether the answers matched, and the
        contained fault attribution.
    """
    with TemporaryDirectory(prefix="ch20-") as directory:
        team = Orchestrator().run(OBJECTIVE, corpus, Workspace(Path(directory) / "team"))
        single = single_agent(OBJECTIVE, corpus, token_budget=team.tokens)
        faulted = Orchestrator().run(OBJECTIVE, corpus, Workspace(Path(directory) / "fault"),
                                     poison_worker=poison_worker)
    return {
        "team": asdict(team),
        "single": asdict(single),
        "token_multiplier": round(team.tokens / single.tokens, 2),
        "latency_speedup": round(single.latency / team.latency, 2),
        "same_answer": team.answer == single.answer,
        "fault": asdict(faulted),
    }


def delta_utility(*, d_quality: float, d_cost: float, d_latency: float, d_risk: float,
                  w_quality: float = 1.0, w_cost: float = 1.0,
                  w_latency: float = 1.0, w_risk: float = 1.0) -> float:
    """Score a design change as weighted quality gain minus its costs.

    The incremental-utility test makes "is the team worth it?" a number: quality
    improvement is a benefit; extra spend, deadline-relevant latency, and
    operational risk are charged against it under product-chosen weights. A
    negative result is a well-supported rejection of the team.

    Args:
        d_quality: Change in task quality (team minus baseline).
        d_cost: Change in total tokens or spend (positive = the team spent more).
        d_latency: Change in deadline-relevant latency (negative = the team is faster).
        d_risk: Change in operational risk (positive = more failure surface).
        w_quality: Weight on quality; the remaining weights price each cost term.

    Returns:
        The net utility; positive favors the change, negative rejects it.
    """
    return w_quality * d_quality - w_cost * d_cost - w_latency * d_latency - w_risk * d_risk


import random


def majority_vote_accuracy(n_voters: int, p_correct: float, correlation: float,
                           trials: int = 4000, seed: int = 0) -> float:
    """Estimate the accuracy of an n-voter majority with a shared error component.

    Each trial is either a correlated case (with probability ``correlation``) in
    which all voters answer alike, or an independent case in which each voter is
    right on its own with probability ``p_correct``. Majority vote beats a single
    voter only to the extent errors are independent — correlated blind spots erase
    the gain, which is why an ensemble of near-identical model calls disappoints.

    Args:
        n_voters: Number of voters (odd, to avoid ties).
        p_correct: Per-voter probability of being correct.
        correlation: Fraction of cases in which the voters share one answer.
        trials: Monte-Carlo trials.
        seed: RNG seed for reproducibility.

    Returns:
        The fraction of trials in which the majority was correct.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        if rng.random() < correlation:
            hits += rng.random() < p_correct          # one shared vote decides all
        else:
            votes = sum(rng.random() < p_correct for _ in range(n_voters))
            hits += votes > n_voters // 2
    return hits / trials
