"""Portfolio-quality figures for offline model evaluation and empirical control science."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .visuals import BLUE, GOLD, GREY, INK, LIGHT, ORANGE, _save, _style


def generate_deep_figures(project_root: Path) -> list[Path]:
    _style()
    evaluation_dir = project_root / "data" / "evaluation"
    control_dir = project_root / "data" / "control_science"
    output_dir = project_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    comparison = pd.read_csv(evaluation_dir / "model_comparison.csv").sort_values("pr_auc")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    ranking_bars = axes[0].barh(comparison["model"], comparison["pr_auc"], color=BLUE)
    axes[0].bar_label(ranking_bars, fmt="%.3f", padding=4, fontsize=9)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("PR-AUC on task-group holdout")
    axes[0].set_title("Risk ranking quality")
    y = np.arange(len(comparison))
    axes[1].barh(
        y - 0.22, comparison["safety_recall"], height=0.22, color=BLUE, label="Safety recall"
    )
    axes[1].barh(y, comparison["f1"], height=0.22, color=GOLD, label="F1")
    axes[1].barh(
        y + 0.22,
        comparison["over_blocking_rate"],
        height=0.22,
        color=ORANGE,
        label="Over-blocking",
    )
    axes[1].set_yticks(y, comparison["model"])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Rate")
    axes[1].set_title("Thresholded operating trade-off")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Simulator-informed offline risk-classifier comparison",
        fontsize=16,
        fontweight="bold",
    )
    path = output_dir / "09_model_comparison.png"
    _save(fig, path)
    paths.append(path)

    calibration = pd.read_csv(evaluation_dir / "model_calibration.csv")
    fig, ax = plt.subplots(figsize=(7, 6.4))
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK, label="Ideal")
    styles = [(BLUE, "o", "-"), (GOLD, "s", "--")]
    for (model, group), (color, marker, line) in zip(
        calibration.groupby("model"), styles, strict=False
    ):
        ax.plot(
            group["mean_predicted_risk"],
            group["observed_incident_rate"],
            color=color,
            marker=marker,
            linestyle=line,
            label=model,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed incident rate")
    ax.set_title("Model calibration on held-out task groups")
    ax.grid(color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False)
    path = output_dir / "10_model_calibration.png"
    _save(fig, path)
    paths.append(path)

    ablation = pd.read_csv(evaluation_dir / "ablation_study.csv")
    ablation = ablation[ablation["configuration"] != "Full input"].sort_values("f1_delta_vs_full")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [ORANGE if value < 0 else BLUE for value in ablation["f1_delta_vs_full"]]
    ax.barh(ablation["configuration"], ablation["f1_delta_vs_full"], color=colors)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("F1 change versus full input")
    ax.set_title("Ablation study on held-out task groups")
    path = output_dir / "11_ablation_study.png"
    _save(fig, path)
    paths.append(path)

    unseen = pd.read_csv(evaluation_dir / "unseen_stressor_generalization.csv")
    cross = pd.read_csv(evaluation_dir / "cross_workflow_generalization.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    x = np.arange(len(unseen))
    axes[0].bar(x - 0.18, unseen["pr_auc"], width=0.36, color=BLUE, label="PR-AUC")
    axes[0].bar(x + 0.18, unseen["safety_recall"], width=0.36, color=GOLD, label="Safety recall")
    axes[0].set_xticks(x, unseen["held_out_stressor"].str.replace("_", " "), rotation=15)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Strict unseen-stressor holdout")
    axes[0].legend(frameon=False)
    x = np.arange(len(cross))
    axes[1].bar(x - 0.18, cross["pr_auc"], width=0.36, color=BLUE, label="PR-AUC")
    axes[1].bar(x + 0.18, cross["safety_recall"], width=0.36, color=GOLD, label="Safety recall")
    axes[1].set_xticks(x, cross["held_out_workflow"].str.replace("_", " "), rotation=15)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Leave-one-workflow-out")
    axes[1].legend(frameon=False)
    fig.suptitle("Generalization stress tests", fontsize=16, fontweight="bold")
    path = output_dir / "12_generalization_stress_tests.png"
    _save(fig, path)
    paths.append(path)

    grid = pd.read_csv(control_dir / "control_portfolio_grid.csv")
    fig, ax = plt.subplots(figsize=(9.5, 6))
    feasible = grid["feasible_default"]
    ax.scatter(
        grid.loc[~feasible, "cost"],
        grid.loc[~feasible, "incident_rate"],
        facecolors="none",
        edgecolors=GREY,
        marker="o",
        label="Infeasible under default guardrails",
    )
    ax.scatter(
        grid.loc[feasible, "cost"],
        grid.loc[feasible, "incident_rate"],
        color=BLUE,
        marker="o",
        label="Feasible",
    )
    frontier = grid[grid["pareto_efficient"]].sort_values("cost")
    ax.scatter(
        frontier["cost"],
        frontier["incident_rate"],
        facecolors="none",
        edgecolors=GOLD,
        linewidths=1.8,
        s=95,
        label="3-objective Pareto set",
    )
    best = grid[feasible].sort_values(["incident_rate", "cost"]).iloc[0]
    ax.annotate(
        "Budget-40 optimum",
        (best["cost"], best["incident_rate"]),
        xytext=(8, -20),
        textcoords="offset points",
        color=INK,
    )
    ax.set_xlabel("Governance cost")
    ax.set_ylabel("Incident rate across stressed runs")
    ax.set_ylim(0, min(1.0, grid["incident_rate"].max() * 1.08))
    ax.set_title("Empirical frontier across all 64 control portfolios")
    ax.grid(color=LIGHT, linewidth=0.8)
    ax.legend(frameon=False)
    path = output_dir / "13_empirical_portfolio_frontier.png"
    _save(fig, path)
    paths.append(path)

    shapley = pd.read_csv(control_dir / "control_shapley.csv").sort_values("shapley_risk_reduction")
    fig, ax = plt.subplots(figsize=(10, 5.3))
    ax.barh(shapley["label"], shapley["shapley_risk_reduction"], color=BLUE)
    ax.set_xlabel("Average marginal incident-rate reduction")
    ax.set_title("Shapley attribution across all control coalitions")
    path = output_dir / "14_control_shapley_value.png"
    _save(fig, path)
    paths.append(path)

    interactions = pd.read_csv(control_dir / "control_interactions.csv")
    labels = sorted(set(interactions["label_a"]) | set(interactions["label_b"]))
    matrix = pd.DataFrame(0.0, index=labels, columns=labels)
    for row in interactions.itertuples():
        matrix.loc[row.label_a, row.label_b] = row.synergy
        matrix.loc[row.label_b, row.label_a] = row.synergy
    limit = max(abs(matrix.to_numpy().min()), abs(matrix.to_numpy().max()))
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    interaction_map = LinearSegmentedColormap.from_list("interaction", [ORANGE, "#FFFFFF", BLUE])
    image = ax.imshow(matrix.to_numpy(), cmap=interaction_map, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                ax.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Synergy: expected independent risk − observed pair risk")
    ax.set_title("Pairwise control interaction")
    path = output_dir / "15_control_interaction_heatmap.png"
    _save(fig, path)
    paths.append(path)

    sensitivity = pd.read_csv(control_dir / "seed_sensitivity.csv")
    summary = sensitivity.groupby("configuration", as_index=False).agg(
        mean_incident=("incident_rate", "mean"),
        sd_incident=("incident_rate", "std"),
        min_incident=("incident_rate", "min"),
        max_incident=("incident_rate", "max"),
    )
    summary = summary.sort_values("mean_incident")
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    lower = summary["mean_incident"] - summary["min_incident"]
    upper = summary["max_incident"] - summary["mean_incident"]
    ax.errorbar(
        summary["mean_incident"],
        summary["configuration"].str.replace("_", " "),
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=BLUE,
        ecolor=GOLD,
        capsize=5,
    )
    ax.set_xlim(0, min(1.0, summary["max_incident"].max() * 1.12))
    ax.set_xlabel("Incident rate; point = 12-seed mean, whisker = observed min–max")
    ax.set_title("Seed sensitivity across stressed runs")
    ax.grid(axis="x", color=LIGHT, linewidth=0.8)
    path = output_dir / "16_seed_sensitivity.png"
    _save(fig, path)
    paths.append(path)

    access = pd.read_csv(evaluation_dir / "feature_access_audit.csv")
    access = access.sort_values("pr_auc")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    access_colors = [
        ORANGE if row.uses_simulator_privileged_features else (GREY if row.label_shuffle else BLUE)
        for row in access.itertuples()
    ]
    bars = axes[0].barh(access["feature_access"], access["pr_auc"], color=access_colors)
    axes[0].bar_label(bars, fmt="%.3f", padding=4)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("PR-AUC on 2,048 held-out rows")
    axes[0].set_title("Risk ranking by feature-access policy")
    bars = axes[1].barh(access["feature_access"], access["safety_recall"], color=access_colors)
    axes[1].bar_label(bars, labels=[f"{value:.1%}" for value in access["safety_recall"]], padding=4)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Safety recall at validation-selected threshold")
    axes[1].set_title("Harmful-action detection")
    fig.suptitle("Feature-access audit for risk classification", fontsize=16, fontweight="bold")
    path = output_dir / "17_feature_access_audit.png"
    _save(fig, path)
    paths.append(path)

    multitask = pd.read_csv(evaluation_dir / "multitask_comparison.csv")
    task_order = ["failure_attribution", "severity_prediction", "governance_recommendation"]
    baseline = multitask[multitask["model"] == "Majority Class"].set_index("task").loc[task_order]
    learned = multitask[multitask["model"] != "Majority Class"].set_index("task").loc[task_order]
    task_labels = ["Failure attribution", "Severity prediction", "Governance recommendation"]
    x = np.arange(len(task_labels))
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    baseline_bars = ax.bar(
        x - 0.19, baseline["macro_f1"], width=0.38, color=GREY, label="Majority baseline"
    )
    learned_bars = ax.bar(
        x + 0.19,
        learned["macro_f1"],
        width=0.38,
        color=BLUE,
        label="Multinomial logistic",
    )
    ax.bar_label(baseline_bars, fmt="%.3f", padding=3)
    ax.bar_label(learned_bars, fmt="%.3f", padding=3)
    ax.set_xticks(x, task_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro-F1 on task-group holdout")
    ax.set_title("Three additional tasks: learned baseline versus majority class")
    ax.legend(frameon=False)
    path = output_dir / "18_multitask_comparison.png"
    _save(fig, path)
    paths.append(path)

    confusion = pd.read_csv(evaluation_dir / "multitask_confusion.csv")
    governance_confusion = confusion[confusion["task"] == "governance_recommendation"]
    matrix = governance_confusion.pivot(index="actual", columns="predicted", values="count").fillna(
        0
    )
    normalized = matrix.div(matrix.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    image = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(normalized.columns)), normalized.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(normalized.index)), normalized.index)
    ax.set_xlabel("Recommended control")
    ax.set_ylabel("Empirical best single control")
    for row_index in range(len(normalized.index)):
        for column_index in range(len(normalized.columns)):
            value = normalized.iloc[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else INK,
            )
    fig.colorbar(image, ax=ax, label="Share within actual class")
    ax.set_title("Governance recommendation confusion matrix")
    path = output_dir / "19_governance_recommendation_confusion.png"
    _save(fig, path)
    paths.append(path)

    per_class = pd.read_csv(evaluation_dir / "multitask_per_class_recall.csv")
    failure_recall = per_class[per_class["task"] == "failure_attribution"].sort_values("recall")
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.barh(failure_recall["class"], failure_recall["recall"], color=BLUE)
    ax.bar_label(bars, labels=[f"{value:.1%}" for value in failure_recall["recall"]], padding=4)
    ax.axvline(0.125, color=ORANGE, linestyle="--", label="8-class chance reference")
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Recall on incident rows")
    ax.set_title("Failure-attribution recall by taxonomy class")
    ax.legend(frameon=False)
    path = output_dir / "20_failure_attribution_recall.png"
    _save(fig, path)
    paths.append(path)
    return paths
