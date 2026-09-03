"""Cross-layer validity and few-shot failure-mechanism analysis.

This module deliberately consumes persisted simulator and real-model artifacts. It does not
perform model inference. Its purpose is to test whether simulator risk transfers to observed LLM
behavior and whether aggregate safety/utility trade-offs occur within the same scenarios.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .visuals import BLUE, GOLD, GREY, INK, LIGHT, ORANGE

SEED = 20260827
BOOTSTRAP_SAMPLES = 2_000
MODE_LABELS = {
    "baseline": "Baseline",
    "governed": "Governed",
    "governed_few_shot": "Governed + few-shot",
}
WORKFLOW_LABELS = {
    "refund": "Refund",
    "email": "Email",
    "data_export": "Data export",
    "it_access": "IT access",
}


def _bootstrap_validity(
    frame: pd.DataFrame, samples: int = BOOTSTRAP_SAMPLES, seed: int = SEED
) -> dict[str, tuple[float, float]]:
    """Return scenario-bootstrap intervals for AUROC, AP, and Brier score."""
    rng = np.random.default_rng(seed)
    values: list[tuple[float, float, float]] = []
    n_rows = len(frame)
    for _ in range(samples):
        sample = frame.iloc[rng.integers(0, n_rows, n_rows)]
        if sample["harmful_action"].nunique() < 2:
            continue
        observed = sample["harmful_action"].astype(int)
        predicted = sample["simulator_risk_probability"]
        values.append(
            (
                roc_auc_score(observed, predicted),
                average_precision_score(observed, predicted),
                brier_score_loss(observed, predicted),
            )
        )
    if not values:
        raise ValueError("No valid bootstrap resamples contained both outcome classes")
    array = np.asarray(values)
    bounds = np.quantile(array, [0.025, 0.975], axis=0)
    return {
        "auroc": (float(bounds[0, 0]), float(bounds[1, 0])),
        "average_precision": (float(bounds[0, 1]), float(bounds[1, 1])),
        "brier_score": (float(bounds[0, 2]), float(bounds[1, 2])),
    }


def build_transition_analysis(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Classify scenario-level changes from governed zero-shot to governed few-shot."""
    required_modes = {"governed", "governed_few_shot"}
    if not required_modes.issubset(set(decisions["prompt_mode"])):
        raise ValueError("Transition analysis requires governed and governed_few_shot decisions")
    columns = [
        "scenario_id",
        "task_id",
        "workflow",
        "stressor",
        "case_type",
        "prompt_mode",
        "action",
        "action_correct",
        "harmful_action",
        "over_blocked",
    ]
    paired = decisions[decisions["prompt_mode"].isin(required_modes)][columns].copy()
    if paired.duplicated(["scenario_id", "prompt_mode"]).any():
        raise ValueError("Duplicate scenario/prompt-mode decisions")
    scenario_meta = paired.drop_duplicates("scenario_id").set_index("scenario_id")
    pivot = paired.pivot(index="scenario_id", columns="prompt_mode")
    output = scenario_meta[["task_id", "workflow", "stressor", "case_type"]].copy()
    for field in ("action", "action_correct", "harmful_action", "over_blocked"):
        output[f"governed_{field}"] = pivot[field]["governed"]
        output[f"few_shot_{field}"] = pivot[field]["governed_few_shot"]
    output["accuracy_change"] = output["few_shot_action_correct"].astype(int) - output[
        "governed_action_correct"
    ].astype(int)
    output["harm_change"] = output["few_shot_harmful_action"].astype(int) - output[
        "governed_harmful_action"
    ].astype(int)
    output["overblocking_change"] = output["few_shot_over_blocked"].astype(int) - output[
        "governed_over_blocked"
    ].astype(int)
    conditions = [
        (output["accuracy_change"] == 1) & (output["harm_change"] == 0),
        (output["accuracy_change"] == 0) & (output["harm_change"] == 1),
        (output["accuracy_change"] == -1) & (output["harm_change"] == 1),
        (output["accuracy_change"] == -1) & (output["harm_change"] == 0),
    ]
    labels = [
        "utility_gain_without_safety_loss",
        "safety_loss_without_utility_gain",
        "accuracy_and_safety_regression",
        "accuracy_regression_only",
    ]
    output["transition"] = np.select(conditions, labels, default="no_primary_change")
    output = output.reset_index()

    order = [
        "utility_gain_without_safety_loss",
        "safety_loss_without_utility_gain",
        "accuracy_and_safety_regression",
        "accuracy_regression_only",
        "no_primary_change",
    ]
    summary = (
        output.groupby("transition", as_index=False)
        .agg(
            scenarios=("scenario_id", "size"),
            normal_scenarios=("case_type", lambda values: int((values == "normal").sum())),
            risk_scenarios=("case_type", lambda values: int((values == "risk").sum())),
        )
        .set_index("transition")
        .reindex(order, fill_value=0)
        .reset_index()
    )
    summary["share"] = summary["scenarios"] / len(output)
    return output, summary


def build_simulator_validity(
    decisions: pd.DataFrame,
    scenarios: pd.DataFrame,
    simulator_results: pd.DataFrame,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare simulator no-control risk scores with observed LLM harmful actions."""
    simulator = simulator_results[simulator_results["control_config"] == "none"][
        ["task_id", "stressor", "risk_probability", "incident"]
    ].copy()
    if simulator.duplicated(["task_id", "stressor"]).any():
        raise ValueError("Simulator no-control rows are not unique by task and stressor")
    joined = scenarios.merge(
        simulator,
        on=["task_id", "stressor"],
        how="left",
        validate="one_to_one",
    ).rename(
        columns={
            "risk_probability": "simulator_risk_probability",
            "incident": "simulator_incident",
        }
    )
    if len(joined) != 64 or joined["simulator_risk_probability"].isna().any():
        raise ValueError("Expected complete one-to-one simulator coverage for 64 scenarios")
    scored = joined[
        [
            "scenario_id",
            "task_id",
            "workflow",
            "stressor",
            "case_type",
            "simulator_risk_probability",
            "simulator_incident",
        ]
    ].merge(
        decisions[
            ["scenario_id", "prompt_mode", "harmful_action", "action_correct", "over_blocked"]
        ],
        on="scenario_id",
        how="left",
        validate="one_to_many",
    )
    if len(scored) != len(decisions):
        raise ValueError("Simulator-to-LLM join changed the decision population")

    rows = []
    for mode, frame in scored.groupby("prompt_mode", sort=False):
        observed = frame["harmful_action"].astype(int)
        predicted = frame["simulator_risk_probability"]
        intervals = _bootstrap_validity(frame, samples=bootstrap_samples, seed=SEED + len(rows))
        rows.append(
            {
                "prompt_mode": mode,
                "n_scenarios": len(frame),
                "observed_harm_rate": observed.mean(),
                "mean_simulator_risk": predicted.mean(),
                "auroc": roc_auc_score(observed, predicted),
                "auroc_ci_low": intervals["auroc"][0],
                "auroc_ci_high": intervals["auroc"][1],
                "average_precision": average_precision_score(observed, predicted),
                "average_precision_ci_low": intervals["average_precision"][0],
                "average_precision_ci_high": intervals["average_precision"][1],
                "prevalence_baseline": observed.mean(),
                "brier_score": brier_score_loss(observed, predicted),
                "brier_ci_low": intervals["brier_score"][0],
                "brier_ci_high": intervals["brier_score"][1],
                "calibration_gap_mean_risk_minus_harm": predicted.mean() - observed.mean(),
            }
        )
    validity = pd.DataFrame(rows).set_index("prompt_mode").reindex(MODE_LABELS).reset_index()
    return scored, validity


def build_action_shift(
    decisions: pd.DataFrame, benchmark: pd.DataFrame, manifest: dict
) -> pd.DataFrame:
    """Measure terminal-action shifts and whether the action appeared in development examples."""
    example_ids = {
        task_id
        for task_ids in manifest["few_shot_example_task_ids"].values()
        for task_id in task_ids
    }
    examples = benchmark[benchmark["task_id"].isin(example_ids)].copy()
    demonstrated = examples.groupby("workflow_type")["expected_action"].agg(set).to_dict()
    harm_targets = decisions.groupby("workflow")["harm_target"].first().to_dict()
    rows = []
    for workflow, harm_target in harm_targets.items():
        for mode in ("governed", "governed_few_shot"):
            frame = decisions[
                (decisions["workflow"] == workflow) & (decisions["prompt_mode"] == mode)
            ]
            rows.append(
                {
                    "workflow": workflow,
                    "prompt_mode": mode,
                    "n_scenarios": len(frame),
                    "terminal_action": harm_target,
                    "terminal_action_count": int((frame["action"] == harm_target).sum()),
                    "terminal_action_rate": float((frame["action"] == harm_target).mean()),
                    "terminal_action_demonstrated": harm_target
                    in demonstrated.get(workflow, set()),
                    "demonstrated_actions": " | ".join(sorted(demonstrated.get(workflow, set()))),
                }
            )
    output = pd.DataFrame(rows)
    governed = output[output["prompt_mode"] == "governed"].set_index("workflow")
    few_shot = output[output["prompt_mode"] == "governed_few_shot"].set_index("workflow")
    increases = few_shot["terminal_action_rate"] - governed["terminal_action_rate"]
    output["few_shot_minus_governed_terminal_rate"] = output["workflow"].map(increases)
    return output


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": INK,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "text.color": INK,
            "font.size": 10,
            "axes.titleweight": "bold",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_mechanism_figures(
    figures_dir: Path,
    validity: pd.DataFrame,
    transition_summary: pd.DataFrame,
    action_shift: pd.DataFrame,
) -> list[Path]:
    """Create three static figures with explicit comparison and denominator context."""
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    ordered = validity.set_index("prompt_mode").loc[list(MODE_LABELS)]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(ordered))
    ax.errorbar(
        ordered["auroc"],
        y,
        xerr=[
            ordered["auroc"] - ordered["auroc_ci_low"],
            ordered["auroc_ci_high"] - ordered["auroc"],
        ],
        fmt="o",
        color=BLUE,
        ecolor=GOLD,
        capsize=5,
        linewidth=2,
        markersize=8,
    )
    ax.axvline(0.5, color=INK, linestyle="--", linewidth=1, label="Random ranking")
    ax.set_yticks(y, [MODE_LABELS[mode] for mode in ordered.index])
    ax.set_xlim(0.3, 0.9)
    ax.set_xlabel("AUROC: simulator risk ranking vs observed harmful action")
    ax.set_title("Simulator-to-real-model risk-ranking validity")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    for index, value in enumerate(ordered["auroc"]):
        ax.text(value + 0.015, index, f"{value:.2f}", va="center")
    path = figures_dir / "25_simulator_to_llm_validity.png"
    _save(fig, path)
    paths.append(path)

    labels = {
        "utility_gain_without_safety_loss": "Accuracy improved, no safety loss",
        "safety_loss_without_utility_gain": "Safety regressed, no accuracy gain",
        "accuracy_and_safety_regression": "Accuracy and safety both regressed",
        "accuracy_regression_only": "Accuracy regressed only",
        "no_primary_change": "No accuracy or safety change",
    }
    plot_frame = transition_summary.copy()
    plot_frame["label"] = plot_frame["transition"].map(labels)
    colors = [BLUE, ORANGE, ORANGE, GREY, "#CBD5E1"]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(plot_frame["label"], plot_frame["scenarios"], color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.set_xlabel("Scenarios out of 64")
    ax.set_title("Scenario-level effect of adding few-shot examples")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    for bar, value in zip(bars, plot_frame["scenarios"], strict=True):
        ax.text(value + 0.5, bar.get_y() + bar.get_height() / 2, f"{int(value)}", va="center")
    path = figures_dir / "26_few_shot_transition_decomposition.png"
    _save(fig, path)
    paths.append(path)

    pivot = action_shift.pivot(
        index="workflow", columns="prompt_mode", values="terminal_action_rate"
    ).loc[list(WORKFLOW_LABELS)]
    demonstrated = action_shift.drop_duplicates("workflow").set_index("workflow")[
        "terminal_action_demonstrated"
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(pivot))
    width = 0.34
    first = ax.bar(
        x - width / 2,
        pivot["governed"],
        width,
        label="Governed",
        color=BLUE,
    )
    second = ax.bar(
        x + width / 2,
        pivot["governed_few_shot"],
        width,
        label="Governed + few-shot",
        color=GOLD,
    )
    ax.set_xticks(x, [WORKFLOW_LABELS[name] for name in pivot.index])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Terminal-action selection rate")
    ax.set_title("Terminal-action selection before and after few-shot examples")
    ax.grid(axis="y", color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False)
    for bars in (first, second):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{bar.get_height():.0%}",
                ha="center",
            )
    for index, workflow in enumerate(pivot.index):
        note = "terminal shown" if demonstrated.loc[workflow] else "terminal not shown"
        ax.text(index, -0.12, note, ha="center", transform=ax.get_xaxis_transform(), fontsize=9)
    path = figures_dir / "27_few_shot_terminal_action_shift.png"
    _save(fig, path)
    paths.append(path)
    return paths


def run_mechanism_analysis(project_root: Path) -> dict[str, Path]:
    """Run and persist the complete cross-layer mechanism audit."""
    data_dir = project_root / "data" / "llm_evaluation"
    decisions = pd.read_csv(data_dir / "decisions.csv")
    scenarios = pd.read_json(data_dir / "scenarios.jsonl", lines=True)
    simulator_results = pd.read_csv(project_root / "data" / "results" / "experiment_results.csv")
    benchmark = pd.read_csv(project_root / "data" / "benchmark" / "benchmark.csv")
    manifest = json.loads((data_dir / "manifest.json").read_text())

    transitions, transition_summary = build_transition_analysis(decisions)
    scored, validity = build_simulator_validity(decisions, scenarios, simulator_results)
    action_shift = build_action_shift(decisions, benchmark, manifest)

    paths = {
        "transitions": data_dir / "few_shot_transitions.csv",
        "transition_summary": data_dir / "few_shot_transition_summary.csv",
        "simulator_scored": data_dir / "simulator_to_llm_scenarios.csv",
        "simulator_validity": data_dir / "simulator_to_llm_validity.csv",
        "action_shift": data_dir / "few_shot_action_shift.csv",
        "summary": data_dir / "mechanism_summary.json",
    }
    transitions.to_csv(paths["transitions"], index=False)
    transition_summary.to_csv(paths["transition_summary"], index=False)
    scored.to_csv(paths["simulator_scored"], index=False)
    validity.to_csv(paths["simulator_validity"], index=False)
    action_shift.to_csv(paths["action_shift"], index=False)

    harm_regressions = transitions["harm_change"].eq(1)
    summary = {
        "decision_population": len(decisions),
        "scenario_population": transitions["scenario_id"].nunique(),
        "few_shot_accuracy_improvements": int(transitions["accuracy_change"].eq(1).sum()),
        "few_shot_accuracy_regressions": int(transitions["accuracy_change"].eq(-1).sum()),
        "few_shot_safety_regressions": int(harm_regressions.sum()),
        "safety_regressions_with_accuracy_gain": int(
            (harm_regressions & transitions["accuracy_change"].eq(1)).sum()
        ),
        "overblocking_improvements": int(transitions["overblocking_change"].eq(-1).sum()),
        "non_demonstrated_terminal_workflows": int(
            (~action_shift.drop_duplicates("workflow")["terminal_action_demonstrated"]).sum()
        ),
        "non_demonstrated_terminal_workflows_with_increase": int(
            (
                (~action_shift.drop_duplicates("workflow")["terminal_action_demonstrated"])
                & (
                    action_shift.drop_duplicates("workflow")[
                        "few_shot_minus_governed_terminal_rate"
                    ]
                    > 0
                )
            ).sum()
        ),
        "simulator_validity": validity.set_index("prompt_mode").round(6).to_dict("index"),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")

    run_manifest_path = project_root / "data" / "results" / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text())
        run_manifest["mechanism_analysis"] = {
            "scenario_transitions": len(transitions),
            "simulator_to_llm_join_rows": len(scored),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "few_shot_safety_regressions": summary["few_shot_safety_regressions"],
            "safety_regressions_with_accuracy_gain": summary[
                "safety_regressions_with_accuracy_gain"
            ],
        }
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")

    table_dir = project_root / "outputs" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    validity.to_csv(table_dir / "table_14_simulator_to_llm_validity.csv", index=False)
    transition_summary.to_csv(table_dir / "table_15_few_shot_transitions.csv", index=False)
    action_shift.to_csv(table_dir / "table_16_few_shot_action_shift.csv", index=False)
    generate_mechanism_figures(
        project_root / "outputs" / "figures", validity, transition_summary, action_shift
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit simulator transfer and few-shot failure mechanisms"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    paths = run_mechanism_analysis(args.project_root.resolve())
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
