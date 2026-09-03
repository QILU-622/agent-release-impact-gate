"""Pre-deployment capacity planning and external evaluation evidence normalization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _nested(record: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = record
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is not None:
            return value
    return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "pass", "passed", "1"}:
            return True
        if normalized in {"false", "no", "fail", "failed", "0"}:
            return False
    return None


def _metric_failure(record: dict[str, Any], needles: Iterable[str]) -> bool | None:
    metrics = _nested(record, ("metricsData",), ("metrics_data",), ("gradingResult", "componentResults"))
    if not isinstance(metrics, list):
        return None
    matched = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        name = str(metric.get("name") or metric.get("metric") or "").lower()
        if any(needle in name for needle in needles):
            success = _optional_bool(metric.get("success"))
            if success is not None:
                matched.append(not success)
    return any(matched) if matched else None


def _canonical_case(record: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    passed = _optional_bool(
        _nested(
            record,
            ("passed",),
            ("success",),
            ("gradingResult", "pass"),
            ("grading_result", "pass"),
        )
    )
    if passed is None:
        metrics = _nested(record, ("metricsData",), ("metrics_data",))
        if isinstance(metrics, list):
            labels = [
                _optional_bool(metric.get("success"))
                for metric in metrics
                if isinstance(metric, dict)
            ]
            known = [value for value in labels if value is not None]
            passed = all(known) if known else None

    harmful = _optional_bool(
        _nested(
            record,
            ("harmful_action",),
            ("harmfulAction",),
            ("policy_violation",),
            ("metadata", "harmful_action"),
            ("metadata", "harmfulAction"),
        )
    )
    if harmful is None:
        harmful = _metric_failure(record, ("harmful", "policy violation", "unauthorized action"))

    overblocked = _optional_bool(
        _nested(
            record,
            ("overblocked",),
            ("over_blocked",),
            ("metadata", "overblocked"),
            ("metadata", "over_blocked"),
        )
    )
    if overblocked is None:
        overblocked = _metric_failure(record, ("overblock", "over-block", "false refusal"))

    latency = _nested(
        record,
        ("latency_ms",),
        ("latencyMs",),
        ("response", "latencyMs"),
        ("response", "latency_ms"),
    )
    try:
        latency_ms = float(latency) if latency is not None else None
    except (TypeError, ValueError):
        latency_ms = None

    return {
        "case_id": str(record.get("case_id") or record.get("id") or f"case-{index:04d}"),
        "workflow": str(record.get("workflow") or metadata.get("workflow") or "unknown"),
        "case_type": str(record.get("case_type") or metadata.get("case_type") or "unknown"),
        "passed": passed,
        "harmful_action": harmful,
        "overblocked": overblocked,
        "latency_ms": latency_ms,
    }


def _extract_records(payload: dict[str, Any], source: str) -> tuple[str, list[dict[str, Any]]]:
    detected = source
    if source == "auto":
        if "testCases" in payload or "conversationalTestCases" in payload:
            detected = "deepeval"
        elif "cases" in payload:
            detected = "canonical"
        elif "results" in payload:
            detected = "promptfoo"
        else:
            raise ValueError("unable to detect evaluation format")

    if detected == "canonical":
        raw = payload.get("cases", [])
    elif detected == "deepeval":
        raw = [
            *payload.get("testCases", []),
            *payload.get("conversationalTestCases", []),
        ]
    elif detected == "promptfoo":
        raw = payload.get("results", [])
        if isinstance(raw, dict):
            raw = raw.get("results") or raw.get("table") or []
        if isinstance(raw, dict):
            raw = raw.get("body") or []
    else:
        raise ValueError(f"unsupported evaluation source: {detected}")

    if not isinstance(raw, list):
        raise TypeError("evaluation results must contain a list of cases")
    return detected, [record for record in raw if isinstance(record, dict)]


def summarize_external_evaluation(
    payload: dict[str, Any], source: str = "auto"
) -> dict[str, Any]:
    """Normalize aggregate evidence without retaining prompts, outputs, or secrets."""

    detected, raw_records = _extract_records(payload, source)
    cases = [_canonical_case(record, index) for index, record in enumerate(raw_records, start=1)]
    total = len(cases)

    def rate(field: str, *, normal_only: bool = False) -> tuple[int, float | None]:
        eligible = [
            case
            for case in cases
            if case[field] is not None
            and (not normal_only or case["case_type"].lower() == "normal")
        ]
        if not eligible:
            return 0, None
        return len(eligible), float(np.mean([bool(case[field]) for case in eligible]))

    pass_count, passed_rate = rate("passed")
    harm_count, harmful_rate = rate("harmful_action")
    overblock_count, overblock_rate = rate("overblocked", normal_only=True)
    latencies = [case["latency_ms"] for case in cases if case["latency_ms"] is not None]
    workflows = sorted({case["workflow"] for case in cases if case["workflow"] != "unknown"})

    return {
        "source": detected,
        "cases_total": total,
        "workflows": workflows,
        "pass_labels": pass_count,
        "pass_rate": passed_rate,
        "harm_labels": harm_count,
        "harmful_action_rate": harmful_rate,
        "normal_overblock_labels": overblock_count,
        "normal_overblock_rate": overblock_rate,
        "latency_labels": len(latencies),
        "median_latency_ms": float(np.median(latencies)) if latencies else None,
        "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "coverage": {
            "pass": pass_count / total if total else 0.0,
            "harm": harm_count / total if total else 0.0,
            "normal_overblock": overblock_count / total if total else 0.0,
            "latency": len(latencies) / total if total else 0.0,
        },
        "privacy_boundary": (
            "Only aggregate labels and latency statistics are retained; prompts, outputs, "
            "provider configuration, headers, identities, and secrets are discarded."
        ),
    }


def load_external_evaluation(path: Path, source: str = "auto") -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("evaluation artifact must contain a JSON object")
    return summarize_external_evaluation(payload, source)


def replay_reviewer_queue(
    events: pd.DataFrame,
    reviewer_workers: int,
    simulation_minutes: int,
    capacity_multiplier: float = 1.0,
) -> dict[str, float]:
    """Replay only the reviewer queue using recorded review arrivals and service times."""

    if reviewer_workers < 1:
        raise ValueError("reviewer_workers must be at least one")
    effective_workers = max(1, round(reviewer_workers * capacity_multiplier))
    reviews = events[events["needs_review"]].sort_values("agent_finish_minute")
    heap = [0.0] * effective_workers
    import heapq

    heapq.heapify(heap)
    waits: list[float] = []
    completions: list[float] = []
    total_service = 0.0
    for row in reviews.itertuples():
        available = heapq.heappop(heap)
        start = max(float(row.agent_finish_minute), available)
        duration = float(row.review_duration_minutes)
        finish = start + duration
        heapq.heappush(heap, finish)
        waits.append(start - float(row.agent_finish_minute))
        completions.append(finish)
        total_service += duration

    return {
        "effective_reviewers": float(effective_workers),
        "reviews": float(len(reviews)),
        "reviewer_utilization": total_service / (effective_workers * simulation_minutes),
        "p95_review_wait_minutes": float(np.percentile(waits, 95)) if waits else 0.0,
        "reviews_completed_after_day": float(sum(value > simulation_minutes for value in completions)),
    }


def build_reviewer_capacity_plan(events: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Size reviewer pools against explicit utilization and queue-wait guardrails."""

    planning = config.get("capacity_planning", {})
    target_utilization = float(planning.get("target_reviewer_utilization", 0.85))
    max_wait = float(planning.get("maximum_p95_review_wait_minutes", 10.0))
    max_reviewers = int(planning.get("maximum_nominal_reviewers", 12))
    rows: list[dict[str, Any]] = []

    for scenario_name, scenario in config["scenarios"].items():
        for architecture_name, architecture in config["architectures"].items():
            selected = events[
                (events["scenario"] == scenario_name)
                & (events["architecture"] == architecture_name)
            ]
            if selected.empty:
                continue
            candidate_rows: list[dict[str, Any]] = []
            for nominal_reviewers in range(1, max_reviewers + 1):
                seed_metrics = [
                    replay_reviewer_queue(
                        group,
                        nominal_reviewers,
                        int(config["simulation_minutes"]),
                        float(scenario["reviewer_capacity_multiplier"]),
                    )
                    for _, group in selected.groupby("seed")
                ]
                utilizations = [item["reviewer_utilization"] for item in seed_metrics]
                waits = [item["p95_review_wait_minutes"] for item in seed_metrics]
                spillovers = [item["reviews_completed_after_day"] for item in seed_metrics]
                row = {
                    "scenario": scenario_name,
                    "architecture": architecture_name,
                    "nominal_reviewers": nominal_reviewers,
                    "effective_reviewers": int(seed_metrics[0]["effective_reviewers"]),
                    "mean_reviewer_utilization": float(np.mean(utilizations)),
                    "p95_reviewer_utilization": float(np.percentile(utilizations, 95)),
                    "mean_p95_review_wait_minutes": float(np.mean(waits)),
                    "p95_review_wait_minutes": float(np.percentile(waits, 95)),
                    "mean_reviews_completed_after_day": float(np.mean(spillovers)),
                }
                row["capacity_guardrails_passed"] = bool(
                    row["p95_reviewer_utilization"] <= target_utilization
                    and row["p95_review_wait_minutes"] <= max_wait
                )
                candidate_rows.append(row)

            feasible = [row for row in candidate_rows if row["capacity_guardrails_passed"]]
            recommended = feasible[0]["nominal_reviewers"] if feasible else None
            for row in candidate_rows:
                row["current_nominal_reviewers"] = int(architecture["reviewer_workers"])
                row["recommended_nominal_reviewers"] = recommended
                row["is_recommended"] = row["nominal_reviewers"] == recommended
                row["capacity_status"] = "ready" if recommended is not None else "blocked"
                rows.append(row)
    return pd.DataFrame(rows)


def capacity_recommendations(plan: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    recommendations: dict[str, Any] = {}
    for (scenario, architecture), group in plan.groupby(["scenario", "architecture"]):
        current = int(group["current_nominal_reviewers"].iloc[0])
        recommended_rows = group[group["is_recommended"]]
        key = f"{scenario}|{architecture}"
        if recommended_rows.empty:
            recommendations[key] = {
                "scenario": scenario,
                "architecture": architecture,
                "status": "blocked",
                "current_nominal_reviewers": current,
                "recommended_nominal_reviewers": None,
                "reviewer_gap": None,
                "decision": (
                    f"No staffing level up to {int(group['nominal_reviewers'].max())} nominal "
                    "reviewers satisfies both capacity guardrails. Redesign routing or review scope."
                ),
            }
            continue
        row = recommended_rows.iloc[0]
        recommended = int(row["recommended_nominal_reviewers"])
        gap = recommended - current
        if gap > 0:
            action = f"Add {gap} nominal reviewer(s) before this scenario is treated as capacity-ready."
        elif gap < 0:
            action = (
                f"The model indicates {abs(gap)} fewer reviewer(s) could satisfy the provisional "
                "capacity limits; validate with observed pilot queues before reducing staff."
            )
        else:
            action = "Current nominal reviewer staffing meets the provisional capacity limits."
        recommendations[key] = {
            "scenario": scenario,
            "architecture": architecture,
            "status": "ready",
            "current_nominal_reviewers": current,
            "recommended_nominal_reviewers": recommended,
            "reviewer_gap": gap,
            "p95_reviewer_utilization": round(float(row["p95_reviewer_utilization"]), 6),
            "p95_review_wait_minutes": round(float(row["p95_review_wait_minutes"]), 3),
            "decision": action,
        }
    return recommendations


def build_deployment_evidence_pack(
    project_root: Path, external_evaluation: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a conservative, stable readiness summary inspired by release evidence packs."""

    twin_dir = project_root / "data" / "workforce_twin"
    manifest_path = twin_dir / "manifest.json"
    capacity_path = twin_dir / "capacity_recommendations.json"
    multi_model_path = project_root / "outputs" / "tables" / "multi_model_manifest.json"
    pilot_report_path = project_root / "pilot" / "pilot_validation_report.json"

    checks: list[dict[str, str]] = []
    blockers: list[str] = []
    warnings: list[str] = []
    ready: list[str] = []

    if manifest_path.exists() and capacity_path.exists():
        manifest = json.loads(manifest_path.read_text())
        ready.append(
            f"Digital twin produced {manifest.get('event_records', 0):,} event records with a "
            "frozen configuration hash."
        )
        checks.append(
            {
                "check": "Operating-model simulation",
                "status": "ready",
                "summary": "Reproducible paired simulation and reviewer-capacity plan are present.",
            }
        )
        warnings.append("All workforce demand, service, cost, and capacity inputs remain synthetic.")
    else:
        blockers.append("Run the workforce twin to generate the operating-model and capacity plan.")
        checks.append(
            {
                "check": "Operating-model simulation",
                "status": "blocked",
                "summary": "Required workforce-twin outputs are missing.",
            }
        )

    if multi_model_path.exists():
        multi = json.loads(multi_model_path.read_text())
        model_count = len(multi.get("models", []))
        ready.append(
            f"Cross-model sensitivity evidence is present for {model_count} model(s)."
        )
        checks.append(
            {
                "check": "Model sensitivity",
                "status": "ready",
                "summary": "Local cross-model behavior results are available.",
            }
        )
    else:
        warnings.append("No completed cross-model manifest is available for this evidence pack.")
        checks.append(
            {
                "check": "Model sensitivity",
                "status": "warning",
                "summary": "Cross-model results are absent.",
            }
        )

    if external_evaluation is None:
        blockers.append("No customer- or workflow-specific external Agent evaluation was imported.")
        checks.append(
            {
                "check": "External Agent evaluation",
                "status": "blocked",
                "summary": "Import a Promptfoo, DeepEval, or canonical JSON result artifact.",
            }
        )
    else:
        enough_cases = int(external_evaluation.get("cases_total", 0)) >= 20
        harm_coverage = float(external_evaluation.get("coverage", {}).get("harm", 0.0)) >= 0.8
        if enough_cases and harm_coverage:
            ready.append("External Agent evaluation has at least 20 cases and 80% harm-label coverage.")
            status = "ready"
            summary = "Minimum external behavior-evidence coverage is present."
        else:
            blockers.append(
                "External evaluation needs at least 20 cases and harm labels on at least 80% of cases."
            )
            status = "blocked"
            summary = "External evidence exists but does not meet the minimum coverage contract."
        checks.append({"check": "External Agent evaluation", "status": status, "summary": summary})

    if pilot_report_path.exists():
        pilot = json.loads(pilot_report_path.read_text())
        if pilot.get("ready_for_external_shadow_pilot"):
            ready.append("An independently adjudicated external shadow-pilot package is validated.")
            pilot_status = "ready"
            pilot_summary = "External historical cases and attestations passed the intake checks."
        else:
            blockers.append("The external shadow-pilot package did not pass validation.")
            pilot_status = "blocked"
            pilot_summary = "Pilot validation report contains blocking findings."
    else:
        blockers.append("No validated external shadow-pilot package is present.")
        pilot_status = "blocked"
        pilot_summary = "Synthetic examples cannot support a real-customer or realized-impact claim."
    checks.append(
        {"check": "External workflow calibration", "status": pilot_status, "summary": pilot_summary}
    )

    status = "blocked" if blockers else "ready_with_warnings" if warnings else "ready"
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "blockers": blockers,
        "warnings": warnings,
        "ready": ready,
        "checks": checks,
        "external_evaluation": external_evaluation,
        "claim_boundary": (
            "Readiness describes evidence coverage for a controlled shadow pilot. It is not "
            "production authorization, a staffing commitment, measured ROI, or certification."
        ),
    }


def render_evidence_markdown(evidence: dict[str, Any]) -> str:
    lines = [
        "# Enterprise Agent Deployment Evidence",
        "",
        f"**Readiness status:** `{evidence['status']}`",
        "",
    ]
    for heading, key, icon in [
        ("Blocking items", "blockers", "❌"),
        ("Warnings", "warnings", "⚠️"),
        ("Ready signals", "ready", "✅"),
    ]:
        values = evidence.get(key, [])
        if values:
            lines.extend([f"## {heading}", ""])
            lines.extend(f"- {icon} {value}" for value in values)
            lines.append("")
    lines.extend(
        [
            "## Readiness checks",
            "",
            "| Check | Status | Summary |",
            "|---|---|---|",
        ]
    )
    for check in evidence["checks"]:
        lines.append(f"| {check['check']} | `{check['status']}` | {check['summary']} |")
    lines.extend(["", "## Claim boundary", "", evidence["claim_boundary"], ""])
    return "\n".join(lines)


def write_deployment_evidence(
    project_root: Path, external_evaluation: dict[str, Any] | None = None
) -> dict[str, Path]:
    evidence = build_deployment_evidence_pack(project_root, external_evaluation)
    report_dir = project_root / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": report_dir / "deployment_evidence.json",
        "markdown": report_dir / "deployment_evidence.md",
    }
    paths["json"].write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(render_evidence_markdown(evidence), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--external-eval", type=Path)
    parser.add_argument(
        "--source", choices=["auto", "canonical", "promptfoo", "deepeval"], default="auto"
    )
    args = parser.parse_args()
    external = (
        load_external_evaluation(args.external_eval, args.source) if args.external_eval else None
    )
    paths = write_deployment_evidence(args.project_root.resolve(), external)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
