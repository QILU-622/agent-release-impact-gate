"""Empirical all-subset control experiments, Shapley attribution, and seed sensitivity."""

from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path

import pandas as pd

from .catalog import CONTROLS, STRESSORS
from .schema import WorkflowTask
from .simulator import run_experiment


def all_control_subsets() -> list[tuple[str, ...]]:
    names = list(CONTROLS)
    return [combo for size in range(len(names) + 1) for combo in combinations(names, size)]


def portfolio_id(controls: tuple[str, ...] | list[str]) -> str:
    return "+".join(sorted(controls)) if controls else "none"


def _aggregate_runs(rows: list[dict[str, object]]) -> dict[str, float]:
    frame = pd.DataFrame(rows)
    incidents = int(frame["incident"].sum())
    return {
        "runs": float(len(frame)),
        "incident_rate": float(frame["incident"].mean()),
        "task_success_rate": float(frame["task_success"].mean()),
        "safety_success_rate": float(frame["safety_success"].mean()),
        "cascading_failure_rate": float(frame["cascading_failure"].mean()),
        "policy_violation_rate": float(frame["policy_violation"].mean()),
        "human_review_load": float(frame["human_review"].mean()),
        "review_saturation_rate": float(frame["review_saturated"].mean()),
        "rollback_coverage": float(frame["rollback_success"].sum() / incidents)
        if incidents
        else 0.0,
        "mean_blast_radius": float(frame.loc[frame["incident"], "blast_radius"].mean())
        if incidents
        else 0.0,
    }


def run_control_portfolio_grid(
    tasks: list[WorkflowTask], seed: int = 20260827
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stressors = [name for name in STRESSORS if name != "none"]
    overall_rows: list[dict[str, object]] = []
    workflow_rows: list[dict[str, object]] = []
    for controls in all_control_subsets():
        run_rows: list[dict[str, object]] = []
        for task in tasks:
            for stressor in stressors:
                run = run_experiment(
                    task,
                    stressor=stressor,
                    controls=controls,
                    global_seed=seed,
                    control_config=portfolio_id(controls),
                    include_trace=False,
                )
                run_rows.append(
                    {
                        "workflow": task.workflow_type,
                        "incident": run.incident,
                        "task_success": run.task_success,
                        "safety_success": run.safety_success,
                        "cascading_failure": run.cascading_failure,
                        "policy_violation": run.policy_violation,
                        "human_review": run.human_review,
                        "review_saturated": run.review_saturated,
                        "rollback_success": run.rollback_success,
                        "blast_radius": run.blast_radius,
                    }
                )
        control_name = portfolio_id(controls)
        cost = float(sum(CONTROLS[name]["cost"] for name in controls))
        overall_rows.append(
            {
                "portfolio": control_name,
                "controls": ",".join(controls),
                "control_count": len(controls),
                "cost": cost,
                **_aggregate_runs(run_rows),
            }
        )
        frame = pd.DataFrame(run_rows)
        for workflow, group in frame.groupby("workflow"):
            workflow_rows.append(
                {
                    "portfolio": control_name,
                    "workflow": workflow,
                    "controls": ",".join(controls),
                    "control_count": len(controls),
                    "cost": cost,
                    **_aggregate_runs(group.to_dict("records")),
                }
            )
    overall = pd.DataFrame(overall_rows)
    by_workflow = pd.DataFrame(workflow_rows)
    baseline_risk = float(overall.loc[overall["portfolio"] == "none", "incident_rate"].iloc[0])
    overall["risk_reduction"] = baseline_risk - overall["incident_rate"]
    workflow_baseline = by_workflow[by_workflow["portfolio"] == "none"].set_index("workflow")[
        "incident_rate"
    ]
    by_workflow["risk_reduction"] = by_workflow.apply(
        lambda row: float(workflow_baseline[row["workflow"]] - row["incident_rate"]), axis=1
    )
    overall["worst_workflow_incident_rate"] = overall["portfolio"].map(
        by_workflow.groupby("portfolio")["incident_rate"].max()
    )
    overall["feasible_default"] = (
        (overall["cost"] <= 40)
        & (overall["task_success_rate"] >= 0.85)
        & (overall["human_review_load"] <= 0.30)
    )
    overall["pareto_efficient"] = _pareto_flags(overall)
    return overall, by_workflow


def _pareto_flags(frame: pd.DataFrame) -> list[bool]:
    flags = []
    for row in frame.itertuples():
        dominated = frame[
            (frame["cost"] <= row.cost)
            & (frame["incident_rate"] <= row.incident_rate)
            & (frame["task_success_rate"] >= row.task_success_rate)
            & (
                (frame["cost"] < row.cost)
                | (frame["incident_rate"] < row.incident_rate)
                | (frame["task_success_rate"] > row.task_success_rate)
            )
        ]
        flags.append(dominated.empty)
    return flags


def shapley_control_value(grid: pd.DataFrame) -> pd.DataFrame:
    names = list(CONTROLS)
    n_controls = len(names)
    risk = {row.portfolio: float(row.incident_rate) for row in grid.itertuples()}
    rows = []
    for control in names:
        contribution = 0.0
        for size in range(n_controls):
            for subset in combinations([name for name in names if name != control], size):
                without = portfolio_id(subset)
                with_control = portfolio_id((*subset, control))
                weight = (
                    math.factorial(size)
                    * math.factorial(n_controls - size - 1)
                    / math.factorial(n_controls)
                )
                contribution += weight * (risk[without] - risk[with_control])
        rows.append(
            {
                "control": control,
                "label": CONTROLS[control]["label"],
                "shapley_risk_reduction": contribution,
                "cost": float(CONTROLS[control]["cost"]),
                "shapley_per_cost": contribution / float(CONTROLS[control]["cost"]),
            }
        )
    result = pd.DataFrame(rows).sort_values("shapley_risk_reduction", ascending=False)
    total_grid_reduction = risk["none"] - risk[portfolio_id(tuple(names))]
    result["efficiency_gap"] = float(result["shapley_risk_reduction"].sum()) - total_grid_reduction
    return result


def pairwise_control_interactions(grid: pd.DataFrame) -> pd.DataFrame:
    risk = {row.portfolio: float(row.incident_rate) for row in grid.itertuples()}
    baseline = risk["none"]
    rows = []
    for left, right in combinations(CONTROLS, 2):
        observed_pair = risk[portfolio_id((left, right))]
        expected_independent = risk[left] * risk[right] / baseline if baseline else 0.0
        synergy = expected_independent - observed_pair
        rows.append(
            {
                "control_a": left,
                "control_b": right,
                "label_a": CONTROLS[left]["label"],
                "label_b": CONTROLS[right]["label"],
                "observed_pair_risk": observed_pair,
                "expected_independent_risk": expected_independent,
                "synergy": synergy,
                "interpretation": "complementary"
                if synergy > 0.005
                else ("diminishing_returns" if synergy < -0.005 else "approximately_independent"),
            }
        )
    return pd.DataFrame(rows).sort_values("synergy", ascending=False)


def optimize_empirical_portfolio(
    grid: pd.DataFrame,
    budget: float = 40,
    min_completion: float = 0.85,
    max_review_load: float = 0.30,
    objective: str = "average_risk",
) -> dict[str, object]:
    feasible = grid[
        (grid["cost"] <= budget)
        & (grid["task_success_rate"] >= min_completion)
        & (grid["human_review_load"] <= max_review_load)
    ].copy()
    if feasible.empty:
        return {
            "portfolio": "none",
            "controls": [],
            "cost": 0.0,
            "incident_rate": float(grid.loc[grid["portfolio"] == "none", "incident_rate"].iloc[0]),
            "risk_reduction": 0.0,
            "task_success_rate": float(
                grid.loc[grid["portfolio"] == "none", "task_success_rate"].iloc[0]
            ),
            "human_review_load": float(
                grid.loc[grid["portfolio"] == "none", "human_review_load"].iloc[0]
            ),
            "objective": objective,
            "method": "no feasible empirical portfolio",
        }
    score_column = (
        "worst_workflow_incident_rate" if objective == "worst_workflow_risk" else "incident_rate"
    )
    best = feasible.sort_values(
        [score_column, "cost", "task_success_rate"], ascending=[True, True, False]
    ).iloc[0]
    return {
        "portfolio": best["portfolio"],
        "controls": [item for item in str(best["controls"]).split(",") if item],
        "cost": float(best["cost"]),
        "incident_rate": float(best["incident_rate"]),
        "worst_workflow_incident_rate": float(best["worst_workflow_incident_rate"]),
        "risk_reduction": float(best["risk_reduction"]),
        "task_success_rate": float(best["task_success_rate"]),
        "human_review_load": float(best["human_review_load"]),
        "objective": objective,
        "method": "exhaustive empirical search over all 64 control portfolios",
    }


def seed_sensitivity(
    tasks: list[WorkflowTask], selected_controls: list[str], base_seed: int = 20260827
) -> pd.DataFrame:
    configs = {
        "none": [],
        "recommended_bundle": ["context_envelope", "tool_version_lock", "permission_scope"],
        "empirical_budget_40": selected_controls,
    }
    stressors = [name for name in STRESSORS if name != "none"]
    rows = []
    for seed in range(base_seed, base_seed + 12):
        for config, controls in configs.items():
            runs = []
            for task in tasks:
                for stressor in stressors:
                    run = run_experiment(
                        task,
                        stressor=stressor,
                        controls=controls,
                        global_seed=seed,
                        control_config=config,
                        include_trace=False,
                    )
                    runs.append(
                        {
                            "incident": run.incident,
                            "task_success": run.task_success,
                            "human_review": run.human_review,
                        }
                    )
            frame = pd.DataFrame(runs)
            rows.append(
                {
                    "seed": seed,
                    "configuration": config,
                    "incident_rate": float(frame["incident"].mean()),
                    "task_success_rate": float(frame["task_success"].mean()),
                    "human_review_load": float(frame["human_review"].mean()),
                    "runs": len(frame),
                }
            )
    return pd.DataFrame(rows)


def run_portfolio_experiments(
    project_root: Path, tasks: list[WorkflowTask], seed: int = 20260827
) -> dict[str, Path]:
    output_dir = project_root / "data" / "control_science"
    output_dir.mkdir(parents=True, exist_ok=True)
    grid, by_workflow = run_control_portfolio_grid(tasks, seed=seed)
    shapley = shapley_control_value(grid)
    interactions = pairwise_control_interactions(grid)
    average_best = optimize_empirical_portfolio(grid)
    robust_best = optimize_empirical_portfolio(grid, objective="worst_workflow_risk")
    sensitivity = seed_sensitivity(tasks, list(average_best["controls"]), base_seed=seed)
    budget_rows = []
    for budget in range(0, 101, 5):
        budget_rows.append({"budget": budget, **optimize_empirical_portfolio(grid, budget=budget)})
    empirical_curve = pd.DataFrame(budget_rows)

    grid_path = output_dir / "control_portfolio_grid.csv"
    workflow_path = output_dir / "control_portfolio_by_workflow.csv"
    shapley_path = output_dir / "control_shapley.csv"
    interactions_path = output_dir / "control_interactions.csv"
    sensitivity_path = output_dir / "seed_sensitivity.csv"
    curve_path = output_dir / "empirical_budget_curve.csv"
    grid.to_csv(grid_path, index=False)
    by_workflow.to_csv(workflow_path, index=False)
    shapley.to_csv(shapley_path, index=False)
    interactions.to_csv(interactions_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    empirical_curve.to_csv(curve_path, index=False)
    recommendation_path = output_dir / "empirical_recommendations.json"
    recommendation_path.write_text(
        json.dumps(
            {"average_risk_objective": average_best, "worst_workflow_objective": robust_best},
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "portfolio_grid": grid_path,
        "portfolio_by_workflow": workflow_path,
        "shapley": shapley_path,
        "interactions": interactions_path,
        "seed_sensitivity": sensitivity_path,
        "empirical_budget_curve": curve_path,
        "recommendations": recommendation_path,
    }
