# Auto-generated from chapters/22-evaluation.qmd by scripts/tangle.py — do not edit.
from __future__ import annotations


import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist, fmean, stdev
from typing import Any

FIXTURES = Path("code/ch22/fixtures")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load one JSON object per non-blank line of a JSON Lines file.

    Args:
        path: The ``.jsonl`` fixture to read.

    Returns:
        The parsed objects, in file order.

    Raises:
        ValueError: If the file is empty or any line is not a JSON object,
            which we treat as a corrupt eval set rather than a silent skip.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain one JSON object per line")
    return rows


def trajectory_f1(actual: list[str], expected: list[str]) -> float:
    """Multiset F1 between the tools a trial used and the reference tool set.

    This is a *diagnostic*, not a verdict: order-insensitive overlap between
    the tools called and the reference path. A high value does not prove
    success and a low value does not prove failure; it flags trajectories
    worth inspecting.

    Args:
        actual: Tool names the trial called, in order.
        expected: The reference tool names for the task.

    Returns:
        The multiset F1 in ``[0, 1]``; ``0.0`` when both sides are empty of
        overlap, ``1.0`` for an exact multiset match.
    """
    overlap = sum((Counter(actual) & Counter(expected)).values())
    precision = overlap / len(actual) if actual else 0.0
    recall = overlap / len(expected) if expected else 1.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def grade_trial(run: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Score one trial against the task's four component graders.

    Release ``success`` is the conjunction of four independent checks:
    the authoritative final state matches, the structured output is schema
    valid, the recorded action obeyed policy, and the calibrated judge passed.
    Component scores are kept even when the conjunction is false, because they
    are the raw material of error analysis.

    Args:
        run: One replayed trial: final state, schema flag, policy flag, judge
            label, and the tools it called.
        task: The versioned task, supplying the expected state and tool path.

    Returns:
        A dict of the four component booleans, the combined ``success``, and
        the diagnostic ``trajectory_f1``.
    """
    state_pass = run["final_state"] == task["expected_state"]
    schema_pass = bool(run["schema_valid"])
    policy_pass = bool(run["policy_ok"])
    judge_pass = run["judge_label"] == "PASS"
    return {
        "state_pass": state_pass,
        "schema_pass": schema_pass,
        "policy_pass": policy_pass,
        "judge_pass": judge_pass,
        "success": state_pass and schema_pass and policy_pass and judge_pass,
        "trajectory_f1": trajectory_f1(list(run["tools"]), list(task["expected_tools"])),
    }


def grade_traces(path: Path = FIXTURES / "traces.jsonl") -> list[dict[str, Any]]:
    """Expand every task's repeated trials into graded, isolated records.

    Each trial is assigned a unique ``environment_id`` of
    ``snapshot:system:trial``; a repeat raises, because a shared environment
    lets a later trial pass on state an earlier one left behind. Every task
    must supply at least two trials per system so repeated-run reliability is
    estimable.

    Args:
        path: The JSON Lines task fixture.

    Returns:
        One graded record per trial, carrying task identity, slice, the golden
        flag, the four component scores, ``success``, and trajectory diagnostics.

    Raises:
        ValueError: On a duplicate task id, a reused environment identity, or a
            system with fewer than two trials.
    """
    rows: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    environments: set[str] = set()
    for task in load_jsonl(path):
        task_id = str(task["task_id"])
        if task_id in task_ids:
            raise ValueError(f"duplicate task_id: {task_id}")
        task_ids.add(task_id)
        for system in ("baseline", "candidate"):
            runs = task["runs"][system]
            if len(runs) < 2:
                raise ValueError(f"{task_id}/{system} needs repeated trials")
            for trial, run in enumerate(runs):
                environment_id = f"{task['snapshot']}:{system}:{trial}"
                if environment_id in environments:
                    raise ValueError(f"environment reused: {environment_id}")
                environments.add(environment_id)
                rows.append({
                    "task_id": task_id,
                    "slice": task["slice"],
                    "golden": bool(task["golden"]),
                    "system": system,
                    "trial": trial,
                    "environment_id": environment_id,
                    **grade_trial(run, task),
                })
    return rows


def cohen_kappa(first: list[str], second: list[str]) -> float:
    """Chance-corrected agreement between two categorical raters.

    Implements @eq-ch22-kappa: observed agreement discounted by the agreement
    expected from the two raters' marginal label frequencies. A judge that
    always emits the majority label scores near zero however high its raw
    accuracy, which is exactly why kappa, not accuracy, is the calibration
    metric. Kappa can be negative when raters agree less than chance.

    Args:
        first: One rater's labels (for example, adjudicated human labels).
        second: The other rater's labels, index-aligned with ``first``.

    Returns:
        Cohen's kappa in ``[-1, 1]``; ``1.0`` only when both raters are
        perfectly and non-trivially in agreement.

    Raises:
        ValueError: If the label lists are empty or of unequal length.
    """
    if not first or len(first) != len(second):
        raise ValueError("raters need equal, non-empty label lists")
    labels = set(first) | set(second)
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    ca, cb = Counter(first), Counter(second)
    expected = sum((ca[label] / len(first)) * (cb[label] / len(first)) for label in labels)
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1.0 - expected)


def judge_report(path: Path = FIXTURES / "judge_calibration.json") -> dict[str, Any]:
    """Summarize pointwise calibration and pairwise position consistency.

    Expands the compressed contingency cells into aligned human/judge label
    lists, then reports the four numbers a release rule actually consults:
    chance-corrected agreement (kappa), recall on the human-labeled failure
    class, per-slice agreement, and the pairwise position-flip rate. Failure
    recall and slice agreement are reported separately because a healthy global
    kappa can still hide a judge that misses the rare severe class or fails in
    one language.

    Args:
        path: The judge-calibration JSON fixture.

    Returns:
        A report dict with ``n``, ``agreement``, ``kappa``, ``fail_recall``,
        ``position_flip_rate``, and per-slice ``slice_agreement``.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    human: list[str] = []
    judge: list[str] = []
    by_slice: dict[str, list[bool]] = defaultdict(list)
    for cell in payload["pointwise_cells"]:
        count = int(cell["count"])
        human.extend([cell["human"]] * count)
        judge.extend([cell["judge"]] * count)
        by_slice[cell["slice"]].extend([cell["human"] == cell["judge"]] * count)
    fail_total = sum(label == "FAIL" for label in human)
    fail_caught = sum(a == b == "FAIL" for a, b in zip(human, judge))
    pair_total = sum(int(cell["count"]) for cell in payload["pairwise_cells"])
    pair_flips = sum(int(cell["count"]) for cell in payload["pairwise_cells"]
                     if cell["ab"] != cell["ba_normalized"])
    return {
        "n": len(human),
        "agreement": fmean(a == b for a, b in zip(human, judge)),
        "kappa": cohen_kappa(human, judge),
        "fail_recall": fail_caught / fail_total,
        "position_flip_rate": pair_flips / pair_total,
        "slice_agreement": {name: fmean(v) for name, v in sorted(by_slice.items())},
    }


def pass_at_k_estimate(n: int, successes: int, k: int) -> float:
    """Finite-sample pass@k: the chance a size-k draw contains a success.

    Implements the without-replacement estimator of @eq-ch22-passk-estimate.
    It answers "if we drew k of these n observed runs, would at least one
    succeed?" and measures accessible capability, appropriate only when the
    system truly produces k candidates and a trustworthy verifier selects one.
    Average per-task values across tasks; do not pool successes first.

    Args:
        n: Number of observed runs of the task.
        successes: How many of those runs succeeded.
        k: Draw size, ``1 <= k <= n``.

    Returns:
        The estimated pass@k in ``[0, 1]``.

    Raises:
        ValueError: If ``k`` or ``successes`` fall outside their valid range.
    """
    if not 1 <= k <= n or not 0 <= successes <= n:
        raise ValueError("require 1 <= k <= n and 0 <= successes <= n")
    failures = n - successes
    return 1.0 if failures < k else 1.0 - math.comb(failures, k) / math.comb(n, k)


def pass_pow_k_estimate(n: int, successes: int, k: int) -> float:
    """Finite-sample pass^k: the chance all k drawn runs succeed.

    Implements the without-replacement estimator of @eq-ch22-passk-estimate.
    It answers "if we drew k of these n observed runs, would every one
    succeed?" and measures repeated-run reliability, the right metric when a
    user expects the same behavior on every interaction rather than the best of
    several.

    Args:
        n: Number of observed runs of the task.
        successes: How many of those runs succeeded.
        k: Draw size, ``1 <= k <= n``.

    Returns:
        The estimated pass^k in ``[0, 1]``; ``0.0`` when fewer than ``k`` runs
        succeeded.

    Raises:
        ValueError: If ``k`` or ``successes`` fall outside their valid range.
    """
    if not 1 <= k <= n or not 0 <= successes <= n:
        raise ValueError("require 1 <= k <= n and 0 <= successes <= n")
    return 0.0 if successes < k else math.comb(successes, k) / math.comb(n, k)


def task_metrics(rows: list[dict[str, Any]], system: str, k: int = 2) -> dict[str, dict[str, float]]:
    """Aggregate a system's repeated trials within each independent task.

    The task, not the trial, is the unit of independence, so we collapse each
    task's repeats into one cluster carrying its pass rate, the finite-sample
    pass@k and pass^k estimates, and mean trajectory F1. ``k`` is fixed at 2 to
    suit the four-trial fixture; a production suite sizes ``k`` to its own trial
    budget.

    Args:
        rows: Graded trial records from :func:`grade_traces`.
        system: Which system's rows to aggregate (``"baseline"`` or ``"candidate"``).
        k: Draw size for the repeated-run estimators.

    Returns:
        A mapping from task id to its per-task metrics.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["system"] == system:
            grouped[row["task_id"]].append(row)
    result: dict[str, dict[str, float]] = {}
    for task_id, trials in sorted(grouped.items()):
        n = len(trials)
        successes = sum(row["success"] for row in trials)
        result[task_id] = {
            "pass_rate": successes / n,
            "pass_at_k": pass_at_k_estimate(n, successes, k),
            "pass_pow_k": pass_pow_k_estimate(n, successes, k),
            "trajectory_f1": fmean(row["trajectory_f1"] for row in trials),
        }
    return result


def percentile(values: list[float], q: float) -> float:
    """Linearly interpolated ``q``-quantile of a list of numbers.

    Args:
        values: The sample to summarize; need not be sorted.
        q: The quantile in ``[0, 1]`` (``0.025`` for a 95% lower bound).

    Returns:
        The interpolated quantile, matching the percentile convention used by
        the bootstrap interval so the reported bounds are reproducible.
    """
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def paired_cluster_uncertainty(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    *,
    resamples: int = 10_000,
    seed: int = 22,
) -> dict[str, Any]:
    """Bootstrap the paired task delta and report an 80%-power MDE.

    Pairs the two systems on shared task ids, resamples *task ids* with
    replacement (keeping each task's repeats together, so the unit of
    resampling is the unit of independence), and reads a 95% interval off the
    bootstrap distribution of the mean delta. The cluster standard error of
    @eq-ch22-cluster-se gives a scale check and an approximate minimum
    detectable effect at 80% power. The observed tasks must be an exchangeable
    sample from the contract's population for the interval to mean anything; a
    hand-curated threat suite is not, and its failures are reported separately.

    Args:
        baseline: Per-task metrics for the baseline system.
        candidate: Per-task metrics for the candidate, over the same task ids.
        resamples: Bootstrap resample count.
        seed: Seed making the bootstrap reproducible.

    Returns:
        A dict with the point delta, the 95% ``low``/``high`` bounds, the
        cluster SE, the 80%-power ``mde_80``, and the per-task ``task_deltas``.

    Raises:
        ValueError: If the two systems do not cover the same two-or-more tasks.
    """
    if set(baseline) != set(candidate) or len(baseline) < 2:
        raise ValueError("paired systems need the same two or more task ids")
    deltas = [candidate[key]["pass_rate"] - baseline[key]["pass_rate"] for key in sorted(baseline)]
    rng = random.Random(seed)
    boot = [fmean(rng.choice(deltas) for _ in deltas) for _ in range(resamples)]
    cluster_se = stdev(deltas) / math.sqrt(len(deltas))
    z_alpha = NormalDist().inv_cdf(0.975)
    z_power = NormalDist().inv_cdf(0.80)
    return {
        "point": fmean(deltas),
        "low": percentile(boot, 0.025),
        "high": percentile(boot, 0.975),
        "cluster_se": cluster_se,
        "mde_80": (z_alpha + z_power) * cluster_se,
        "task_deltas": dict(zip(sorted(baseline), deltas)),
    }


def slice_rates(rows: list[dict[str, Any]], system: str) -> dict[str, float]:
    """Trial success rate for each predeclared workload slice.

    Args:
        rows: Graded trial records.
        system: Which system's rows to group.

    Returns:
        A mapping from slice name to mean trial success in ``[0, 1]``.
    """
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if row["system"] == system:
            grouped[row["slice"]].append(row["success"])
    return {name: fmean(values) for name, values in sorted(grouped.items())}


FAILURE_ORDER = ("malformed_output", "wrong_final_state", "policy_violation", "poor_explanation")


def primary_failure(row: dict[str, Any]) -> str | None:
    """Assign one failure category at the earliest boundary a trial violated.

    Returns ``None`` for a success. Otherwise reports the first failed check in
    escalating order — malformed output, then wrong final state, then policy
    violation, then poor explanation — so each failure gets exactly one primary
    cause for error analysis and burden accounting, with secondary components
    still available on the row.

    Args:
        row: A graded trial record from :func:`grade_traces`.

    Returns:
        The primary failure category, or ``None`` if the trial succeeded.
    """
    if row["success"]:
        return None
    if not row["schema_pass"]:
        return "malformed_output"
    if not row["state_pass"]:
        return "wrong_final_state"
    if not row["policy_pass"]:
        return "policy_violation"
    return "poor_explanation"


def error_burden(rows: list[dict[str, Any]], system: str) -> dict[str, dict[str, Any]]:
    """Per-slice success rate plus the composition of its failures.

    Two slices with equal success can carry very different harm: the same mean
    hides whether failures are unauthorized actions or harmless extra
    questions. For each slice this returns the success rate and a category
    breakdown of its failing trials, so a fairness review can see composition,
    not just the average.

    Args:
        rows: Graded trial records.
        system: Which system's rows to summarize.

    Returns:
        A mapping from slice name to ``{"success": rate, "failures": {category: count}}``.
    """
    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["system"] == system:
            by_slice[row["slice"]].append(row)
    report: dict[str, dict[str, Any]] = {}
    for name, group in sorted(by_slice.items()):
        failures: Counter[str] = Counter()
        for row in group:
            category = primary_failure(row)
            if category is not None:
                failures[category] += 1
        report[name] = {"success": fmean(r["success"] for r in group), "failures": dict(failures)}
    return report


def release_report(
    traces: Path = FIXTURES / "traces.jsonl",
    calibration: Path = FIXTURES / "judge_calibration.json",
    *,
    margin: float = 0.02,
    slice_floor: float = 0.75,
) -> dict[str, Any]:
    """Assemble the evidence packet and apply the predeclared release rule.

    The rule blocks the candidate if the judge is uncalibrated, the paired
    lower confidence bound falls below the allowed loss ``-margin``, any
    candidate slice sits under ``slice_floor``, or a golden task regresses.
    Every threshold here is a fixture decision, not a universal standard; a
    real contract derives them from the harm of a false promotion versus a
    false rejection.

    Args:
        traces: The task fixture to grade.
        calibration: The judge-calibration fixture.
        margin: The largest tolerated quality loss (as a positive number).
        slice_floor: The minimum acceptable success rate per candidate slice.

    Returns:
        The full evidence packet plus ``verdict`` (``"SHIP"`` or ``"BLOCK"``)
        and the list of blocking ``reasons``.
    """
    rows = grade_traces(traces)
    judge = judge_report(calibration)
    baseline = task_metrics(rows, "baseline")
    candidate = task_metrics(rows, "candidate")
    uncertainty = paired_cluster_uncertainty(baseline, candidate)
    cand_slices = slice_rates(rows, "candidate")
    golden = sorted(
        task_id for task_id in baseline
        if any(r["task_id"] == task_id and r["golden"] for r in rows)
        and candidate[task_id]["pass_rate"] < baseline[task_id]["pass_rate"]
    )
    reasons: list[str] = []
    if judge["kappa"] < 0.60 or judge["fail_recall"] < 0.70 or judge["position_flip_rate"] > 0.15:
        reasons.append("judge calibration does not meet the contract")
    if uncertainty["low"] < -margin:
        reasons.append("paired lower bound exceeds the allowed quality loss")
    weak = [name for name, rate in cand_slices.items() if rate < slice_floor]
    if weak:
        reasons.append("candidate misses slice floor: " + ", ".join(weak))
    if golden:
        reasons.append("must-not-break regression: " + ", ".join(golden))
    return {
        "verdict": "SHIP" if not reasons else "BLOCK",
        "reasons": reasons,
        "judge": judge,
        "uncertainty": uncertainty,
        "slices": {"baseline": slice_rates(rows, "baseline"), "candidate": cand_slices},
        "trajectory_f1": {
            "baseline": fmean(v["trajectory_f1"] for v in baseline.values()),
            "candidate": fmean(v["trajectory_f1"] for v in candidate.values()),
        },
    }
