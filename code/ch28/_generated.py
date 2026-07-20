# Auto-generated from chapters/28-products-people-organizations.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


from collections import Counter
from dataclasses import dataclass
from enum import Enum


class EpistemicLabel(str, Enum):
    """The provenance a product attaches to each claim it shows a user.

    Distinguishing a sourced fact from a model inference from an unresolved
    question is what lets a reviewer calibrate trust to evidence instead of to
    fluent phrasing. The labels are stable strings so the same word means the
    same thing on every screen and in every modality.
    """

    SOURCE_FACT = "source fact"
    USER_PROVIDED = "user-provided"
    SYSTEM_STATE = "system state"
    MODEL_INFERENCE = "model inference"
    ESTIMATE = "estimate"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class PlanDiff:
    """The exact change an agent proposes, shown before any effect runs.

    A person should approve this delta, not a summary of it: the affected
    object and field, the current and proposed state, the permission the
    effect would exercise, and the labeled evidence behind it. Approving a
    plan diff is narrow by construction; approving "the plan" is not.
    """

    affected_object: str
    field: str
    from_state: str
    to_state: str
    permission: str
    evidence: tuple[tuple[EpistemicLabel, str], ...]


@dataclass(frozen=True)
class Receipt:
    """The record of an effect after it commits, tied to authoritative state.

    Unlike a generated "done" message, a receipt names the system of record,
    the operation identity a later reconciliation can join on, the committed
    result, when it happened, and whether downstream state agrees. It is the
    product-level analogue of Chapter 16's effect log.
    """

    system: str
    op_id: str
    result: str
    timestamp: str
    reconciliation: str


QUADRANTS = ("correct_accept", "correct_reject", "wrong_accept", "wrong_reject")
STAGES = ("exposed", "invoked", "completed", "accepted_or_edited", "outcome_realized", "retained")

COHORT_LOG = {
    "experienced": {
        "size": 200,
        "quadrants": {"correct_accept": 160, "correct_reject": 10, "wrong_accept": 4, "wrong_reject": 26},
        "confirmed_catches": 24, "review_seconds": 35,
        "funnel": {"exposed": 200, "invoked": 190, "completed": 180,
                   "accepted_or_edited": 170, "outcome_realized": 160, "retained": 145},
    },
    "new_reviewer": {
        "size": 200,
        "quadrants": {"correct_accept": 165, "correct_reject": 5, "wrong_accept": 25, "wrong_reject": 5},
        "confirmed_catches": 5, "review_seconds": 85,
        "funnel": {"exposed": 200, "invoked": 185, "completed": 175,
                   "accepted_or_edited": 155, "outcome_realized": 140, "retained": 115},
    },
    "assistive_technology": {
        "size": 100,
        "quadrants": {"correct_accept": 70, "correct_reject": 10, "wrong_accept": 5, "wrong_reject": 15},
        "confirmed_catches": 14, "review_seconds": 70,
        "funnel": {"exposed": 100, "invoked": 85, "completed": 65,
                   "accepted_or_edited": 55, "outcome_realized": 45, "retained": 35},
    },
}


def expand_logs(cohorts: dict) -> tuple[list[dict], list[dict]]:
    """Expand compact cohort counts into two aligned per-case ledgers.

    The reviewer-decision ledger has one row per reviewed case tagged with its
    reliance quadrant and whether its rejection was a confirmed catch; the
    usage ledger has one row per exposed case flagged with the funnel stages it
    reached. Working from per-case rows means every reported rate is a count
    over a slice, which is what makes cohort breakdowns honest.

    Args:
        cohorts: The cohort log mapping each group to its counts and funnel.

    Returns:
        A tuple ``(decisions, usage)`` of two row lists.
    """
    decisions: list[dict] = []
    usage: list[dict] = []
    case_id = 0
    for name, cohort in cohorts.items():
        for quadrant in QUADRANTS:
            for offset in range(cohort["quadrants"][quadrant]):
                decisions.append({
                    "case_id": case_id, "cohort": name, "quadrant": quadrant,
                    "confirmed_catch": quadrant == "wrong_reject" and offset < cohort["confirmed_catches"],
                    "review_seconds": cohort["review_seconds"],
                })
                case_id += 1
        for index in range(cohort["size"]):
            usage.append({"cohort": name, **{stage: index < cohort["funnel"][stage] for stage in STAGES}})
    return decisions, usage


decisions, usage = expand_logs(COHORT_LOG)
print(f"reviewer decisions: {len(decisions)}   usage rows: {len(usage)}")
print("one decision row:", decisions[0])


def reliance_rates(rows) -> dict:
    """Compute the four-cell reliance rates for one slice of decisions.

    Returns appropriate reliance (correct-accept plus wrong-reject over all),
    overreliance (accepting the agent's wrong recommendations), and
    underreliance (rejecting its good ones), plus a confirmed-catch rate that
    asks whether a rejection prevented a real defect or merely made rework.
    Reporting all of these keeps a harmless correct-accept from masking a
    dangerous wrong-accept.

    Args:
        rows: An iterable of decision rows, each carrying a ``quadrant`` key.

    Returns:
        A dict of counts and the rates from @eq-ch28-reliance.
    """
    rows = list(rows)
    counts = Counter(row["quadrant"] for row in rows)
    correct = counts["correct_accept"] + counts["correct_reject"]
    wrong = counts["wrong_accept"] + counts["wrong_reject"]
    catches = sum(row["confirmed_catch"] for row in rows)
    return {
        "n": len(rows),
        "quadrants": {name: counts[name] for name in QUADRANTS},
        "appropriate_rate": round((counts["correct_accept"] + counts["wrong_reject"]) / len(rows), 4),
        "overreliance_rate": round(counts["wrong_accept"] / wrong, 4),
        "underreliance_rate": round(counts["correct_reject"] / correct, 4),
        "confirmed_catch_rate": round(catches / counts["wrong_reject"], 4),
    }


def reliance_report(decisions: list[dict]) -> dict:
    """Compute overall reliance and one breakdown per cohort.

    Args:
        decisions: The reviewer-decision ledger from ``expand_logs``.

    Returns:
        A dict with ``overall`` rates and a ``by_cohort`` mapping.
    """
    cohorts = sorted({row["cohort"] for row in decisions})
    return {
        "overall": reliance_rates(decisions),
        "by_cohort": {name: reliance_rates(r for r in decisions if r["cohort"] == name) for name in cohorts},
    }


reliance = reliance_report(decisions)
overall = reliance["overall"]
print(f"overall appropriate={overall['appropriate_rate']}  "
      f"over={overall['overreliance_rate']}  under={overall['underreliance_rate']}")


REVIEW_HOURLY_RATE = 36.0
BASELINE_COST_PER_OUTCOME = 1.60

DESIGNS = {
    "deterministic_rules": {"machine": 5.0, "review_seconds": 6000, "remediation": 0.0, "outcomes": 250},
    "single_call_classifier": {"machine": 20.0, "review_seconds": 22500, "remediation": 48.0, "outcomes": 300},
    "bounded_agent": {"machine": 45.0, "review_seconds": 31000, "remediation": 136.0, "outcomes": 345},
}


def cost_per_outcome(machine: float, review_seconds: float, remediation: float,
                     outcomes: int, rate: float = REVIEW_HOURLY_RATE) -> dict:
    """Price one design's realized outcomes, review labor charged in full.

    The denominator is a realized, quality-qualified outcome — not an
    invocation or a token. Review labor (reading, checking, correcting) is the
    hidden cost a product shifts onto humans, so it enters the numerator at the
    reviewer's hourly rate rather than being waved away as "free" human time.

    Args:
        machine: Inference and retrieval cost for the window.
        review_seconds: Total human review time over all cases.
        remediation: Cost of cleaning up wrong recommendations that slipped through.
        outcomes: Count of realized, quality-qualified outcomes.
        rate: Reviewer cost per hour.

    Returns:
        A dict of the cost components, the total, and the cost per outcome.
    """
    review = review_seconds * rate / 3600.0
    total = machine + review + remediation
    return {
        "machine": round(machine, 2), "review_labor": round(review, 2),
        "remediation": round(remediation, 2), "total_cost": round(total, 2),
        "outcomes": outcomes, "cost_per_outcome": round(total / outcomes, 4),
    }


for name, design in DESIGNS.items():
    row = cost_per_outcome(**design)
    print(f"{name:24} outcomes={row['outcomes']:4}  review={row['review_labor']:6}  "
          f"total={row['total_cost']:6}  per_outcome={row['cost_per_outcome']}")


def _funnel_slice(rows) -> dict:
    """Count the nested funnel stages and adjacent conversions for one slice."""
    rows = list(rows)
    counts = {stage: sum(row[stage] for row in rows) for stage in STAGES}
    conversions = {STAGES[0]: 1.0}
    for prior, stage in zip(STAGES, STAGES[1:]):
        conversions[stage] = round(counts[stage] / counts[prior], 4)
    return {"counts": counts, "conversion_from_previous": conversions}


def funnel_report(usage: list[dict]) -> dict:
    """Compute the invocation-to-outcome funnel overall and per cohort.

    Each stage is a subset of the previous one, so counts never increase and
    the end-to-end rate (outcome-realized over exposed) is the product of the
    adjacent conversions. Keeping both the adjacent and end-to-end views is
    what reveals that a better model-completion step can coexist with falling
    realized value if a later stage — review, integration, downstream
    settlement — quietly worsens.

    Args:
        usage: The usage ledger from ``expand_logs``.

    Returns:
        A dict with ``overall`` and ``by_cohort`` funnel slices.
    """
    cohorts = sorted({row["cohort"] for row in usage})
    return {
        "overall": _funnel_slice(usage),
        "by_cohort": {name: _funnel_slice(r for r in usage if r["cohort"] == name) for name in cohorts},
    }


funnel = funnel_report(usage)
counts = funnel["overall"]["counts"]
end_to_end = round(counts["outcome_realized"] / counts["exposed"], 4)
offline_eval = 0.90  # representative held-out benchmark completion quality
print("funnel:", counts)
print(f"offline eval score={offline_eval}   realized end-to-end value={end_to_end}   gap={round(offline_eval-end_to_end,4)}")


def economics_report(cohorts: dict, decisions: list[dict], usage: list[dict]) -> dict:
    """Assemble the bounded agent's unit economics from the ledgers.

    Computes cost per realized outcome (machine, review labor, and remediation
    over realized outcomes) and the break-even average review time — the
    longest average review compatible with the conventional baseline — so a
    team can see how interface changes or harder cohorts move the economics.
    Energy is reported per outcome within one stated boundary, useful for
    regression within that boundary and not comparable across different ones.

    Args:
        cohorts: The cohort log (for the energy per-exposure constant).
        decisions: The reviewer-decision ledger.
        usage: The usage ledger.

    Returns:
        A dict of cost components, cost per outcome, break-even seconds, and energy.
    """
    outcomes = sum(row["outcome_realized"] for row in usage)
    review_seconds = sum(row["review_seconds"] for row in decisions)
    wrong_accepts = sum(row["quadrant"] == "wrong_accept" for row in decisions)
    machine = round(len(usage) * 0.09, 2)
    base = cost_per_outcome(machine, review_seconds, wrong_accepts * 4.0, outcomes)
    non_review = base["machine"] + base["remediation"]
    t_max = (BASELINE_COST_PER_OUTCOME * outcomes - non_review) * 3600.0 / (REVIEW_HOURLY_RATE * len(decisions))
    base.update({
        "baseline_cost_per_outcome": BASELINE_COST_PER_OUTCOME,
        "average_review_seconds": round(review_seconds / len(decisions), 1),
        "break_even_review_seconds": round(t_max, 1),
        "energy_wh_per_outcome": round(len(usage) * 0.18 / outcomes, 4),
        "energy_boundary": "representative allocated serving energy; excludes user devices",
    })
    return base


economics = economics_report(COHORT_LOG, decisions, usage)
print(f"cost/outcome={economics['cost_per_outcome']} (baseline {economics['baseline_cost_per_outcome']})  "
      f"avg review={economics['average_review_seconds']}s  break-even={economics['break_even_review_seconds']}s")
print(f"energy={economics['energy_wh_per_outcome']} Wh/outcome  ({economics['energy_boundary']})")


KILL_CRITERIA = {"minimum_appropriate_rate": 0.90, "maximum_cohort_overreliance": 0.25,
                 "minimum_assistive_completion": 0.85}


def evaluate_gates(reliance: dict, funnel: dict, economics: dict, kill: dict = KILL_CRITERIA) -> list[dict]:
    """Evaluate the four predeclared launch gates against computed evidence.

    The gates are conjunctive: reliance, cohort overreliance, accessibility
    completion, and cost per outcome must all pass, and a passing economic gate
    cannot buy back a failing reliance or accessibility gate. Each check records
    what was observed and the predeclared threshold so the decision is a record,
    not a presentation.

    Args:
        reliance: The reliance report.
        funnel: The funnel report.
        economics: The economics report.
        kill: The predeclared thresholds.

    Returns:
        One dict per gate with observed value, threshold, operator, and pass flag.
    """
    max_cohort = max(c["overreliance_rate"] for c in reliance["by_cohort"].values())
    assistive = funnel["by_cohort"]["assistive_technology"]["conversion_from_previous"]["completed"]
    checks = (
        ("appropriate_reliance", reliance["overall"]["appropriate_rate"], kill["minimum_appropriate_rate"], ">="),
        ("cohort_overreliance", max_cohort, kill["maximum_cohort_overreliance"], "<="),
        ("assistive_completion", assistive, kill["minimum_assistive_completion"], ">="),
        ("cost_per_outcome", economics["cost_per_outcome"], economics["baseline_cost_per_outcome"], "<="),
    )
    return [{"gate": n, "observed": o, "threshold": t, "operator": op,
             "passed": o >= t if op == ">=" else o <= t} for n, o, t, op in checks]


gates = evaluate_gates(reliance, funnel, economics)
for g in gates:
    print(f"{g['gate']:22} {g['observed']:>8}  {g['operator']} {g['threshold']:<6} "
          f"{'PASS' if g['passed'] else 'FAIL'}")


@dataclass(frozen=True)
class RiskRegisterEntry:
    """One hazard tracked from control to evidence to gate to residual owner.

    The entry ties a named hazard to the control that treats it, the computed
    metric that is its evidence, the launch gate that decides it, and the owner
    accountable for the residual. Because ``status`` is derived from the gate
    result rather than typed by hand, the register stays synchronized with the
    measurement instead of becoming a stale spreadsheet.
    """

    risk_id: str
    hazard: str
    control: str
    evidence: str
    gate: str
    residual: str
    owner: str
    status: str


def assemble_risk_register(gates: list[dict]) -> list[RiskRegisterEntry]:
    """Join predeclared hazards to their gate outcomes into register entries.

    Args:
        gates: The gate evaluations from ``evaluate_gates``.

    Returns:
        One ``RiskRegisterEntry`` per hazard, its status closed only if its
        gate passed and its residual marked for treatment otherwise.
    """
    passed = {g["gate"]: g["passed"] for g in gates}
    template = [
        ("R-EXP-017a", "New reviewers accept wrong categories without checking evidence",
         "Evidence-first review: show the proposed write and receipt before the recommendation",
         "cohort reliance matrix (overreliance by cohort)", "cohort_overreliance", "expense-product-director"),
        ("R-EXP-017b", "Streaming interface prevents assistive-technology users from completing review",
         "Non-streaming fallback with announced completion states",
         "invoked→completed conversion by access mode", "assistive_completion", "expense-product-director"),
        ("R-EXP-017c", "Aggregate appropriate reliance below the launch floor",
         "Evidence legibility and predeclared cohort gates",
         "overall appropriate-reliance rate", "appropriate_reliance", "enterprise-ai-risk"),
    ]
    entries = []
    for risk_id, hazard, control, evidence, gate, owner in template:
        ok = passed.get(gate, False)
        entries.append(RiskRegisterEntry(
            risk_id=risk_id, hazard=hazard, control=control, evidence=evidence, gate=gate,
            residual="within tolerance" if ok else "exceeds tolerance — treatment required",
            owner=owner, status="CLOSED" if ok else "OPEN"))
    return entries


for entry in assemble_risk_register(gates):
    print(f"{entry.risk_id} [{entry.status}] gate={entry.gate}")
    print(f"   hazard:   {entry.hazard}")
    print(f"   evidence: {entry.evidence}")
    print(f"   residual: {entry.residual}  (owner: {entry.owner})")


def build_launch_review(cohorts: dict | None = None) -> dict:
    """Assemble the launch-review appendix from the chapter's own metrics.

    Runs the whole pipeline — expand the ledgers, compute reliance, funnel, and
    economics, evaluate the conjunctive gates, and join the risk register — and
    recommends SHIP only if every gate passed. The recommendation is therefore
    a function of computed evidence, not a vote, which is what lets another
    reviewer reproduce it exactly.

    Args:
        cohorts: The cohort log, or the chapter default when omitted.

    Returns:
        A dict holding the row counts, reliance, funnel, economics, gates, risk
        register, the recommendation, and the list of failed conditions.
    """
    cohorts = cohorts or COHORT_LOG
    decisions, usage = expand_logs(cohorts)
    reliance = reliance_report(decisions)
    funnel = funnel_report(usage)
    economics = economics_report(cohorts, decisions, usage)
    gates = evaluate_gates(reliance, funnel, economics)
    failed = [g["gate"] for g in gates if not g["passed"]]
    return {
        "log_rows": {"reviewer_decisions": len(decisions), "usage_ledger": len(usage)},
        "reliance": reliance, "funnel": funnel, "economics": economics, "gates": gates,
        "risk_register": assemble_risk_register(gates),
        "recommendation": "SHIP" if not failed else "HOLD_FULL_LAUNCH",
        "failed_conditions": failed,
    }


def render_appendix(report: dict) -> str:
    """Format the launch-review appendix as the plain text a reviewer reads.

    Args:
        report: The dict from ``build_launch_review``.

    Returns:
        A fixed-width block: row counts, each gate with its verdict, the review
        and energy economics, and the recommendation.
    """
    econ = report["economics"]
    lines = [f"reviewer decision rows       {report['log_rows']['reviewer_decisions']:>8}",
             f"usage-ledger rows            {report['log_rows']['usage_ledger']:>8}"]
    for g in report["gates"]:
        verdict = "PASS" if g["passed"] else "FAIL"
        lines.append(f"{g['gate']:<26} {g['observed']:>8}   {verdict} ({g['operator']} {g['threshold']})")
    lines += [f"average review time          {econ['average_review_seconds']:>8} s",
              f"break-even review time       {econ['break_even_review_seconds']:>8} s",
              f"serving energy per outcome   {econ['energy_wh_per_outcome']:>8} Wh",
              f"recommendation               {report['recommendation']:>8}"]
    return "\n".join(lines)


report = build_launch_review()
print(render_appendix(report))
