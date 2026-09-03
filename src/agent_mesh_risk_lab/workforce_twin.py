"""Event-driven digital twin for comparing human-Agent operating models."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .benchmark import generate_benchmark
from .deployment_planner import (
    build_reviewer_capacity_plan,
    capacity_recommendations,
    write_deployment_evidence,
)
from .simulator import run_experiment


def _stable_fraction(*parts: object) -> float:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:13], 16) / float(16**13)


def _weighted_choice(draw: float, weights: dict[str, float]) -> str:
    cumulative = 0.0
    for name, weight in weights.items():
        cumulative += weight
        if draw <= cumulative:
            return name
    return next(reversed(weights))


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def load_twin_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if not math.isclose(sum(config["workflow_mix"].values()), 1.0, abs_tol=1e-9):
        raise ValueError("workflow_mix must sum to 1")
    if len(config["architectures"]) < 2 or len(config["scenarios"]) < 2:
        raise ValueError("digital twin requires multiple architectures and scenarios")
    return config


def _generate_arrivals(config: dict[str, Any], scenario: dict[str, Any], seed: int) -> list[float]:
    rng = random.Random(seed)
    rate = config["base_arrival_rate_per_minute"] * scenario["demand_multiplier"]
    minute = 0.0
    arrivals = []
    while True:
        minute += rng.expovariate(rate)
        if minute >= config["simulation_minutes"]:
            break
        arrivals.append(minute)
    return arrivals


def simulate_operating_day(
    config: dict[str, Any], architecture_name: str, scenario_name: str, seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Run one paired operating day and return metrics plus auditable event records."""

    if architecture_name not in config["architectures"]:
        raise KeyError(f"unknown architecture: {architecture_name}")
    if scenario_name not in config["scenarios"]:
        raise KeyError(f"unknown scenario: {scenario_name}")
    architecture = config["architectures"][architecture_name]
    scenario = config["scenarios"][scenario_name]
    arrivals = _generate_arrivals(config, scenario, seed)
    task_pool: dict[tuple[str, str], list[Any]] = {}
    for task in generate_benchmark():
        task_pool.setdefault((task.workflow_type, task.case_type), []).append(task)

    agent_heap = [0.0] * int(architecture["agent_workers"])
    reviewer_count = max(
        1,
        round(
            architecture["reviewer_workers"] * scenario["reviewer_capacity_multiplier"]
        ),
    )
    reviewer_heap = [0.0] * reviewer_count
    heapq.heapify(agent_heap)
    heapq.heapify(reviewer_heap)
    event_rng = random.Random(seed + 99173)
    rows: list[dict[str, Any]] = []
    agent_busy_minutes = 0.0
    reviewer_busy_minutes = 0.0

    for index, arrival in enumerate(arrivals, start=1):
        workflow = _weighted_choice(
            _stable_fraction(seed, index, "workflow"), config["workflow_mix"]
        )
        case_type = "risk" if _stable_fraction(seed, index, "case-type") < 0.35 else "normal"
        candidates = task_pool[(workflow, case_type)]
        source_task = candidates[(index - 1) % len(candidates)]
        event_id = f"{scenario_name}-{seed}-{index:04d}"
        task = source_task.model_copy(update={"task_id": event_id})
        experiment = run_experiment(
            task,
            scenario["stressor"],
            architecture["controls"],
            global_seed=seed,
            include_trace=False,
        )

        agent_duration = (
            architecture["mean_agent_minutes"]
            * scenario["service_multiplier"]
            * event_rng.uniform(0.72, 1.32)
        )
        agent_available = heapq.heappop(agent_heap)
        agent_start = max(arrival, agent_available)
        agent_finish = agent_start + agent_duration
        heapq.heappush(agent_heap, agent_finish)
        agent_busy_minutes += agent_duration

        needs_review = bool(
            task.human_review_required
            or (architecture["review_high_risk"] and task.risk_level in {"high", "critical"})
        )
        review_wait = 0.0
        review_duration = 0.0
        execution_start = agent_finish
        if needs_review:
            review_duration = (
                architecture["mean_review_minutes"]
                * scenario["review_time_multiplier"]
                * event_rng.uniform(0.75, 1.35)
            )
            reviewer_available = heapq.heappop(reviewer_heap)
            review_start = max(agent_finish, reviewer_available)
            review_wait = review_start - agent_finish
            execution_start = review_start + review_duration
            heapq.heappush(reviewer_heap, execution_start)
            reviewer_busy_minutes += review_duration

        completion_probability = min(
            0.995,
            experiment.completion_probability
            * architecture["completion_multiplier"]
            * scenario["completion_multiplier"],
        )
        risk_probability = min(
            0.995,
            experiment.risk_probability
            * architecture["risk_multiplier"]
            * scenario["risk_multiplier"],
        )
        overblock_probability = min(
            0.45,
            architecture["baseline_overblock_rate"]
            + 0.008 * len(architecture["controls"])
            + (0.025 if scenario["stressor"] == "review_bottleneck" and needs_review else 0),
        )
        overblocked = (
            task.case_type == "normal"
            and _stable_fraction(seed, event_id, "overblock", architecture_name)
            < overblock_probability
        )
        unsafe_proposal = (
            not overblocked
            and _stable_fraction(seed, event_id, "harm") < risk_probability
        )
        intercepted_harm = (
            unsafe_proposal
            and _stable_fraction(seed, event_id, "intercept", architecture_name)
            < architecture["unsafe_proposal_interception_rate"]
        )
        harmful = unsafe_proposal and not intercepted_harm
        completed = (
            not overblocked
            and _stable_fraction(seed, event_id, "complete") < completion_probability
        )
        completion_time = execution_start + scenario["tool_delay_minutes"]
        cycle_minutes = completion_time - arrival
        safe_completion = completed and not unsafe_proposal
        sla_met = safe_completion and cycle_minutes <= config["sla_minutes"][workflow]
        critical_bypass = harmful and task.risk_level in {"high", "critical"}
        model_cost = architecture["model_cost_units_per_case"]
        human_cost = review_duration * config["human_cost_units_per_minute"]
        direct_cost = model_cost + human_cost
        if overblocked:
            outcome = "overblocked"
        elif intercepted_harm:
            outcome = "unsafe_proposal_intercepted"
        elif harmful:
            outcome = "harmful_execution"
        elif not completed:
            outcome = "failed"
        elif sla_met:
            outcome = "safe_within_sla"
        else:
            outcome = "safe_late"
        rows.append(
            {
                "event_id": event_id,
                "seed": seed,
                "architecture": architecture_name,
                "scenario": scenario_name,
                "workflow": workflow,
                "case_type": case_type,
                "risk_level": task.risk_level,
                "arrival_minute": round(arrival, 3),
                "agent_start_minute": round(agent_start, 3),
                "agent_finish_minute": round(agent_finish, 3),
                "completion_minute": round(completion_time, 3),
                "queue_wait_minutes": round(agent_start - arrival + review_wait, 3),
                "cycle_minutes": round(cycle_minutes, 3),
                "needs_review": needs_review,
                "review_wait_minutes": round(review_wait, 3),
                "review_duration_minutes": round(review_duration, 3),
                "safe_completion": safe_completion,
                "sla_met": sla_met,
                "unsafe_proposal": unsafe_proposal,
                "intercepted_harm": intercepted_harm,
                "harmful_execution": harmful,
                "critical_bypass": critical_bypass,
                "overblocked": overblocked,
                "direct_cost_units": round(direct_cost, 4),
                "outcome": outcome,
            }
        )

    events = pd.DataFrame(rows)
    total = len(events)
    safe_count = int(events["safe_completion"].sum()) if total else 0
    critical_cases = int(events["risk_level"].isin(["high", "critical"]).sum()) if total else 0
    metrics = {
        "seed": seed,
        "architecture": architecture_name,
        "scenario": scenario_name,
        "arrivals": total,
        "safe_completions": safe_count,
        "safe_completion_rate": safe_count / total if total else 0.0,
        "sla_attainment_rate": float(events["sla_met"].mean()) if total else 0.0,
        "critical_bypass_rate": (
            float(events["critical_bypass"].sum()) / critical_cases if critical_cases else 0.0
        ),
        "unsafe_proposal_rate": float(events["unsafe_proposal"].mean()) if total else 0.0,
        "unsafe_proposal_interception_rate": (
            float(events["intercepted_harm"].sum()) / float(events["unsafe_proposal"].sum())
            if total and events["unsafe_proposal"].sum()
            else 0.0
        ),
        "harmful_execution_rate": float(events["harmful_execution"].mean()) if total else 0.0,
        "normal_overblock_rate": (
            float(events.loc[events["case_type"] == "normal", "overblocked"].mean())
            if total
            else 0.0
        ),
        "automation_rate": float((~events["needs_review"]).mean()) if total else 0.0,
        "review_rate": float(events["needs_review"].mean()) if total else 0.0,
        "p50_cycle_minutes": _percentile(events["cycle_minutes"].tolist(), 50),
        "p95_cycle_minutes": _percentile(events["cycle_minutes"].tolist(), 95),
        "p95_queue_wait_minutes": _percentile(events["queue_wait_minutes"].tolist(), 95),
        "agent_utilization": (
            agent_busy_minutes
            / (architecture["agent_workers"] * config["simulation_minutes"])
        ),
        "reviewer_utilization": (
            reviewer_busy_minutes / (reviewer_count * config["simulation_minutes"])
        ),
        "direct_cost_units": float(events["direct_cost_units"].sum()) if total else 0.0,
        "cost_per_safe_completion": (
            float(events["direct_cost_units"].sum()) / safe_count if safe_count else math.inf
        ),
    }
    return metrics, events


def aggregate_runs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = [
        "arrivals",
        "safe_completion_rate",
        "sla_attainment_rate",
        "critical_bypass_rate",
        "unsafe_proposal_rate",
        "unsafe_proposal_interception_rate",
        "harmful_execution_rate",
        "normal_overblock_rate",
        "automation_rate",
        "review_rate",
        "p50_cycle_minutes",
        "p95_cycle_minutes",
        "p95_queue_wait_minutes",
        "agent_utilization",
        "reviewer_utilization",
        "direct_cost_units",
        "cost_per_safe_completion",
    ]
    rows = []
    for (architecture, scenario), group in run_metrics.groupby(["architecture", "scenario"]):
        row: dict[str, Any] = {
            "architecture": architecture,
            "scenario": scenario,
            "n_seeds": len(group),
        }
        for metric in metric_names:
            row[metric] = group[metric].mean()
            row[f"{metric}_low"] = group[metric].quantile(0.025)
            row[f"{metric}_high"] = group[metric].quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario", "architecture"]).reset_index(drop=True)


def build_backlog_timeline(
    events: pd.DataFrame, simulation_minutes: int, bucket_minutes: int = 15
) -> pd.DataFrame:
    """Create an operating-day backlog series from one run's arrival/completion timestamps."""

    if bucket_minutes <= 0:
        raise ValueError("bucket_minutes must be positive")
    rows = []
    for minute in range(0, simulation_minutes + 1, bucket_minutes):
        arrived = int((events["arrival_minute"] <= minute).sum())
        completed = int((events["completion_minute"] <= minute).sum())
        rows.append(
            {
                "minute": minute,
                "arrived": arrived,
                "completed": completed,
                "backlog": max(0, arrived - completed),
            }
        )
    return pd.DataFrame(rows)


def recommend_architectures(summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    guardrails = config["decision_guardrails"]
    recommendations: dict[str, Any] = {}
    for scenario, group in summary.groupby("scenario"):
        candidates = group.copy()
        candidates["guardrails_passed"] = (
            (candidates["safe_completion_rate"] >= guardrails["minimum_safe_completion_rate"])
            & (candidates["critical_bypass_rate"] <= guardrails["maximum_critical_bypass_rate"])
            & (candidates["reviewer_utilization"] <= guardrails["maximum_reviewer_utilization"])
            & (
                candidates["normal_overblock_rate"]
                <= guardrails["maximum_normal_overblock_rate"]
            )
        )
        cost_max = max(float(candidates["cost_per_safe_completion"].max()), 1e-9)
        cycle_max = max(float(candidates["p95_cycle_minutes"].max()), 1e-9)
        candidates["decision_score"] = (
            0.42 * candidates["safe_completion_rate"]
            + 0.18 * candidates["sla_attainment_rate"]
            + 0.2 * (1 - candidates["critical_bypass_rate"].clip(upper=1))
            + 0.1 * (1 - candidates["cost_per_safe_completion"] / cost_max)
            + 0.1 * (1 - candidates["p95_cycle_minutes"] / cycle_max)
        )
        feasible = candidates[candidates["guardrails_passed"]]
        pool = feasible if not feasible.empty else candidates
        selected = pool.sort_values("decision_score", ascending=False).iloc[0]
        recommendations[scenario] = {
            "architecture": selected["architecture"],
            "display_name": config["architectures"][selected["architecture"]]["display_name"],
            "guardrails_passed": bool(selected["guardrails_passed"]),
            "decision_score": round(float(selected["decision_score"]), 6),
            "safe_completion_rate": round(float(selected["safe_completion_rate"]), 6),
            "critical_bypass_rate": round(float(selected["critical_bypass_rate"]), 6),
            "p95_cycle_minutes": round(float(selected["p95_cycle_minutes"]), 3),
            "cost_per_safe_completion": round(
                float(selected["cost_per_safe_completion"]), 6
            ),
            "selection_rule": (
                "highest weighted decision score among guardrail-passing architectures"
                if not feasible.empty
                else "highest weighted decision score; no architecture passed every guardrail"
            ),
        }
    return recommendations


def run_workforce_twin(project_root: Path, config_path: Path | None = None) -> dict[str, Path]:
    config_path = config_path or project_root / "configs" / "workforce_twin.json"
    config = load_twin_config(config_path)
    metrics_rows = []
    event_frames = []
    for scenario_name in config["scenarios"]:
        for seed in config["seeds"]:
            for architecture_name in config["architectures"]:
                metrics, events = simulate_operating_day(
                    config, architecture_name, scenario_name, seed
                )
                metrics_rows.append(metrics)
                event_frames.append(events)
    run_metrics = pd.DataFrame(metrics_rows)
    events = pd.concat(event_frames, ignore_index=True)
    summary = aggregate_runs(run_metrics)
    recommendations = recommend_architectures(summary, config)
    capacity_plan = build_reviewer_capacity_plan(events, config)
    reviewer_recommendations = capacity_recommendations(capacity_plan, config)

    output_dir = project_root / "data" / "workforce_twin"
    report_dir = project_root / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "run_metrics": output_dir / "run_metrics.csv",
        "summary": output_dir / "architecture_summary.csv",
        "events": output_dir / "event_log.csv",
        "recommendations": output_dir / "recommendations.json",
        "capacity_plan": output_dir / "reviewer_capacity_plan.csv",
        "capacity_recommendations": output_dir / "capacity_recommendations.json",
        "manifest": output_dir / "manifest.json",
        "report": report_dir / "workforce_twin_decision_brief.md",
    }
    run_metrics.to_csv(paths["run_metrics"], index=False)
    summary.to_csv(paths["summary"], index=False)
    events.to_csv(paths["events"], index=False)
    paths["recommendations"].write_text(json.dumps(recommendations, indent=2) + "\n")
    capacity_plan.to_csv(paths["capacity_plan"], index=False)
    paths["capacity_recommendations"].write_text(
        json.dumps(reviewer_recommendations, indent=2) + "\n"
    )
    paths["manifest"].write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config_path": str(config_path.relative_to(project_root)),
                "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "architecture_count": len(config["architectures"]),
                "scenario_count": len(config["scenarios"]),
                "seed_count": len(config["seeds"]),
                "operating_day_runs": len(run_metrics),
                "event_records": len(events),
                "capacity_planning": config.get("capacity_planning", {}),
                "evidence_class": "synthetic discrete-event simulation",
                "claim_boundary": (
                    "Architecture comparison under explicit planning assumptions; not forecast, "
                    "measured ROI, staffing commitment, or production safety evidence."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    normal = summary[summary["scenario"] == "normal_day"].sort_values(
        "safe_completion_rate", ascending=False
    )
    lines = [
        "# AI Workforce Digital Twin: decision brief",
        "",
        "## Decision",
        "",
        (
            "Use this experiment to compare operating-model hypotheses before a pilot, not to "
            "forecast a company's realized ROI or staffing need."
        ),
        "",
        "## Normal-day comparison",
        "",
        "| Architecture | Safe completion | p95 cycle | Critical bypass | Cost / safe completion |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in normal.itertuples():
        lines.append(
            f"| {config['architectures'][row.architecture]['display_name']} | "
            f"{row.safe_completion_rate:.1%} | {row.p95_cycle_minutes:.1f} min | "
            f"{row.critical_bypass_rate:.1%} | {row.cost_per_safe_completion:.2f} units |"
        )
    lines.extend(["", "## Scenario recommendations", ""])
    for scenario_name, recommendation in recommendations.items():
        lines.append(
            f"- **{config['scenarios'][scenario_name]['display_name']}**: "
            f"{recommendation['display_name']} ({recommendation['selection_rule']})."
        )
    lines.extend(
        [
            "",
            "## Reviewer capacity plan for the selected architecture",
            "",
            "| Scenario | Selected design | Current reviewers | Capacity-safe reviewers | Change | p95 review wait |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for scenario_name, recommendation in recommendations.items():
        key = f"{scenario_name}|{recommendation['architecture']}"
        capacity = reviewer_recommendations[key]
        recommended_reviewers = capacity["recommended_nominal_reviewers"]
        if recommended_reviewers is None:
            recommended_label = "Not found"
            gap_label = "Redesign"
            wait_label = "—"
        else:
            recommended_label = str(recommended_reviewers)
            gap_label = f"{capacity['reviewer_gap']:+d}"
            wait_label = f"{capacity['p95_review_wait_minutes']:.1f} min"
        lines.append(
            f"| {config['scenarios'][scenario_name]['display_name']} | "
            f"{recommendation['display_name']} | {capacity['current_nominal_reviewers']} | "
            f"{recommended_label} | {gap_label} | {wait_label} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            (
                "All arrivals, service times, costs, capacity, model-quality multipliers, and "
                "crises are explicit synthetic assumptions in `configs/workforce_twin.json`. "
                "Replace them with observed customer process data before making an operating or "
                "investment decision. The safety component reuses the project's deterministic "
                "risk simulator; it is not a calibrated probability of real failure."
            ),
            "",
        ]
    )
    paths["report"].write_text("\n".join(lines))
    evidence_paths = write_deployment_evidence(project_root)
    paths["deployment_evidence_json"] = evidence_paths["json"]
    paths["deployment_evidence_markdown"] = evidence_paths["markdown"]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    paths = run_workforce_twin(args.project_root.resolve(), args.config)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
