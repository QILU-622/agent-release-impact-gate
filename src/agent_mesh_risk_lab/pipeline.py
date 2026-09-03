"""End-to-end experiment pipeline and command-line entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .benchmark import generate_benchmark, write_benchmark, write_catalogs
from .catalog import CONTROL_CONFIGS, STRESSORS
from .evaluation import (
    calibration_table,
    certification_table,
    governance_value,
    metrics_table,
)
from .modeling import run_model_evaluation
from .multitask import run_multitask_evaluation
from .optimizer import budget_curve, optimize_portfolio
from .portfolio_experiments import run_portfolio_experiments
from .simulator import run_experiment
from .visuals import generate_figures
from .visuals_v2 import generate_deep_figures
from .workforce_twin import run_workforce_twin


def run_pipeline(project_root: Path, seed: int = 20260827) -> dict[str, Path]:
    data_dir = project_root / "data"
    benchmark_dir = data_dir / "benchmark"
    results_dir = data_dir / "results"
    runs_dir = data_dir / "runs"
    tables_dir = project_root / "outputs" / "tables"
    for directory in (benchmark_dir, results_dir, runs_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks = generate_benchmark()
    write_benchmark(tasks, benchmark_dir)
    write_catalogs(project_root / "configs")

    runs = []
    trace_path = runs_dir / "experiment_traces.jsonl"
    with trace_path.open("w", encoding="utf-8") as trace_handle:
        for task in tasks:
            for stressor in STRESSORS:
                for config_name, controls in CONTROL_CONFIGS.items():
                    run = run_experiment(
                        task,
                        stressor=stressor,
                        controls=controls,
                        global_seed=seed,
                        control_config=config_name,
                    )
                    payload = run.model_dump(mode="json")
                    trace_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    payload.pop("trace")
                    payload["controls"] = ",".join(payload["controls"])
                    runs.append(payload)

    results = pd.DataFrame(runs)
    results_path = results_dir / "experiment_results.csv"
    results.to_csv(results_path, index=False)

    metrics = metrics_table(results)
    metrics_path = results_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    roi = governance_value(results)
    roi_path = results_dir / "governance_roi.csv"
    roi.to_csv(roi_path, index=False)

    certification = certification_table(results)
    certification_path = results_dir / "production_certification.csv"
    certification.to_csv(certification_path, index=False)

    calibration = calibration_table(results)
    calibration_path = results_dir / "calibration.csv"
    calibration.to_csv(calibration_path, index=False)

    curve = budget_curve(roi)
    curve_path = results_dir / "budget_curve.csv"
    curve.to_csv(curve_path, index=False)

    portfolio = optimize_portfolio(roi, budget=40)
    portfolio_path = results_dir / "recommended_portfolio.json"
    portfolio_path.write_text(json.dumps(portfolio, indent=2), encoding="utf-8")

    control_outputs = run_portfolio_experiments(project_root, tasks, seed=seed)
    empirical_recommendations = json.loads(control_outputs["recommendations"].read_text())
    portfolio_path.write_text(
        json.dumps(empirical_recommendations["average_risk_objective"], indent=2),
        encoding="utf-8",
    )
    evaluation_outputs = run_model_evaluation(project_root, tasks, results, seed=seed)
    feature_frame = pd.read_csv(data_dir / "evaluation" / "feature_dataset.csv")
    multitask_outputs = run_multitask_evaluation(project_root, feature_frame, results, seed=seed)
    workforce_outputs = run_workforce_twin(project_root)
    workforce_manifest = json.loads(workforce_outputs["manifest"].read_text())

    benchmark_stats = (
        pd.DataFrame([task.model_dump() for task in tasks])
        .groupby(["workflow_type", "case_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"workflow_type": "workflow"})
    )
    benchmark_stats["total"] = benchmark_stats.select_dtypes("number").sum(axis=1)
    benchmark_stats.to_csv(tables_dir / "table_1_benchmark_statistics.csv", index=False)

    model_comparison = pd.read_csv(evaluation_outputs["model_comparison"])
    model_comparison.to_csv(tables_dir / "table_2_model_comparison.csv", index=False)

    stressor_table = (
        metrics[(metrics["control_config"] == "none") & (metrics["stressor"] != "none")]
        .groupby("stressor", as_index=False)
        .agg(
            failure_rate=("incident_rate", "mean"),
            blast_radius=("mean_blast_radius", "mean"),
            recovery=("rollback_coverage", "mean"),
        )
    )
    stressor_table.to_csv(tables_dir / "table_3_stressor_comparison.csv", index=False)
    roi.to_csv(tables_dir / "table_4_governance_value.csv", index=False)
    certification[certification["control_config"] == "recommended_bundle"].to_csv(
        tables_dir / "table_5_production_decision.csv", index=False
    )
    pd.read_csv(evaluation_outputs["ablation"]).to_csv(
        tables_dir / "table_6_ablation_study.csv", index=False
    )
    pd.read_csv(evaluation_outputs["unseen_stressor"]).to_csv(
        tables_dir / "table_7_unseen_stressor_generalization.csv", index=False
    )
    pd.read_csv(control_outputs["shapley"]).to_csv(
        tables_dir / "table_8_control_shapley_value.csv", index=False
    )
    pd.read_csv(multitask_outputs["feature_access"]).to_csv(
        tables_dir / "table_9_feature_access_audit.csv", index=False
    )
    pd.read_csv(multitask_outputs["multitask_comparison"]).to_csv(
        tables_dir / "table_10_multitask_comparison.csv", index=False
    )
    pd.read_csv(multitask_outputs["governance_unseen"]).to_csv(
        tables_dir / "table_11_governance_unseen_stressor.csv", index=False
    )

    real_llm_manifest_path = data_dir / "llm_evaluation" / "manifest.json"
    real_llm_manifest = (
        json.loads(real_llm_manifest_path.read_text()) if real_llm_manifest_path.exists() else None
    )
    manifest = {
        "benchmark_tasks": len(tasks),
        "experiment_runs": len(results),
        "workflows": sorted(results["workflow"].unique().tolist()),
        "stressors": list(STRESSORS),
        "control_configurations": list(CONTROL_CONFIGS),
        "seed": seed,
        "result_type": (
            "synthetic_simulation_offline_models_and_real_local_llm_evaluation"
            if real_llm_manifest
            else "synthetic_probability_simulation_plus_offline_model_evaluation"
        ),
        "offline_model_evaluation": {
            "dataset_rows": len(results),
            "task_group_split": True,
            "real_llm_results": bool(real_llm_manifest),
            "evaluation_tasks": 4,
            "feature_access_audit": True,
        },
        "control_science": {
            "portfolios_evaluated": 64,
            "stressed_runs_per_portfolio": 1400,
            "seed_sensitivity_runs": 12,
        },
        "workforce_twin": {
            "operating_day_runs": workforce_manifest["operating_day_runs"],
            "event_records": workforce_manifest["event_records"],
            "architecture_count": workforce_manifest["architecture_count"],
            "scenario_count": workforce_manifest["scenario_count"],
            "evidence_class": workforce_manifest["evidence_class"],
        },
    }
    if real_llm_manifest:
        manifest["real_llm_evaluation"] = {
            "model": real_llm_manifest["model"],
            "scenarios": real_llm_manifest["scenario_count"],
            "paired_decisions": real_llm_manifest["decision_calls"],
            "prompt_modes": real_llm_manifest["prompt_modes"],
            "temperature": real_llm_manifest["temperature"],
        }
    manifest_path = results_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    generate_figures(project_root)
    generate_deep_figures(project_root)
    return {
        "benchmark": benchmark_dir / "benchmark.jsonl",
        "results": results_path,
        "traces": trace_path,
        "metrics": metrics_path,
        "governance_roi": roi_path,
        "certification": certification_path,
        "model_comparison": evaluation_outputs["model_comparison"],
        "multitask_comparison": multitask_outputs["multitask_comparison"],
        "portfolio_grid": control_outputs["portfolio_grid"],
        "workforce_twin": workforce_outputs["summary"],
        "manifest": manifest_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    outputs = run_pipeline(args.project_root.resolve(), seed=args.seed)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
