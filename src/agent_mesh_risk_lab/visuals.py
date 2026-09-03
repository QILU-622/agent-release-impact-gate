"""Generate the eight portfolio figures from completed synthetic experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from .catalog import WORKFLOWS
from .graph import build_workflow_graph

BLUE = "#2563EB"
GOLD = "#D4A72C"
ORANGE = "#E87722"
INK = "#172033"
GREY = "#94A3B8"
LIGHT = "#E8EEF8"


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


def generate_figures(project_root: Path) -> list[Path]:
    _style()
    results_dir = project_root / "data" / "results"
    output_dir = project_root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(results_dir / "experiment_results.csv")
    roi = pd.read_csv(results_dir / "governance_roi.csv")
    empirical_curve_path = project_root / "data" / "control_science" / "empirical_budget_curve.csv"
    curve = pd.read_csv(
        empirical_curve_path if empirical_curve_path.exists() else results_dir / "budget_curve.csv"
    )
    calibration = pd.read_csv(results_dir / "calibration.csv")
    certification = pd.read_csv(results_dir / "production_certification.csv")
    paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, workflow in zip(axes.flat, WORKFLOWS, strict=True):
        graph = build_workflow_graph(workflow)
        pos = nx.spring_layout(graph, seed=42)
        colors = {
            "agent": BLUE,
            "tool": ORANGE,
            "policy": GOLD,
        }
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#CBD5E1", arrows=True)
        for node_type, color in colors.items():
            nodes = [n for n, d in graph.nodes(data=True) if d["node_type"] == node_type]
            nx.draw_networkx_nodes(
                graph, pos, nodelist=nodes, node_color=color, node_size=720, ax=ax
            )
        nx.draw_networkx_labels(graph, pos, font_size=7, font_color="white", ax=ax)
        ax.set_title(WORKFLOWS[workflow].display_name)
        ax.axis("off")
    fig.suptitle("Agent mesh graphs", fontsize=16, fontweight="bold")
    path = output_dir / "01_agent_mesh_graph.png"
    _save(fig, path)
    paths.append(path)

    baseline = results[(results["control_config"] == "none") & (results["stressor"] != "none")]
    heat = baseline.pivot_table(
        index="workflow", columns="stressor", values="incident", aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(heat.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(
        range(len(heat.columns)),
        [x.replace("_", " ") for x in heat.columns],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(heat.index)), [WORKFLOWS[x].display_name for x in heat.index])
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j, i, f"{heat.iloc[i, j]:.0%}", ha="center", va="center", color=INK)
    fig.colorbar(image, ax=ax, label="Incident rate")
    ax.set_title("Stressor failure heatmap")
    path = output_dir / "02_stressor_failure_heatmap.png"
    _save(fig, path)
    paths.append(path)

    cascade = baseline.groupby("stressor")["cascading_failure"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([x.replace("_", " ") for x in cascade.index], cascade.values, color=BLUE)
    ax.set_xlim(0, max(0.5, cascade.max() * 1.15))
    ax.set_xlabel("Cascading failure rate")
    ax.set_title("Cascading failure rate by stressor")
    path = output_dir / "03_cascading_failure_by_stressor.png"
    _save(fig, path)
    paths.append(path)

    comparison = results.groupby(["workflow", "control_config"], as_index=False).agg(
        safety_success=("safety_success", "mean"), task_success=("task_success", "mean")
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    for workflow, group in comparison.groupby("workflow"):
        ax.scatter(
            group["task_success"],
            group["safety_success"],
            s=64,
            label=WORKFLOWS[workflow].display_name,
            alpha=0.8,
        )
    ax.set_xlabel("Task success rate")
    ax.set_ylabel("Safety success rate")
    ax.set_xlim(0.55, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False)
    ax.grid(color=LIGHT, linewidth=0.8)
    ax.set_title("Safety versus task success")
    path = output_dir / "04_safety_vs_task_success.png"
    _save(fig, path)
    paths.append(path)

    ranked = roi.sort_values("cgv")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ranked["label"], ranked["cgv"], color=BLUE)
    ax.set_xlabel("Risk reduction per governance cost unit")
    ax.set_title("Governance ROI ranking")
    path = output_dir / "05_governance_roi_ranking.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(curve["budget"], curve["risk_reduction"], marker="o", color=BLUE)
    ax.fill_between(curve["budget"], 0, curve["risk_reduction"], color=LIGHT)
    ax.set_xlabel("Governance budget units")
    ax.set_ylabel("Estimated absolute risk reduction")
    ax.set_title("Budget versus risk reduction")
    ax.grid(color=LIGHT, linewidth=0.8)
    path = output_dir / "06_budget_risk_reduction_curve.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK, label="Ideal calibration")
    ax.plot(
        calibration["predicted_risk"],
        calibration["observed_incident_rate"],
        marker="o",
        color=BLUE,
        label="Simulator",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed incident rate")
    ax.set_title("Risk calibration")
    ax.legend(frameon=False)
    ax.grid(color=LIGHT, linewidth=0.8)
    path = output_dir / "07_calibration_curve.png"
    _save(fig, path)
    paths.append(path)

    cert = certification[certification["control_config"] == "recommended_bundle"].sort_values(
        "score"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [ORANGE if value < 70 else GOLD if value < 85 else BLUE for value in cert["score"]]
    ax.barh(cert["workflow_label"], cert["score"], color=colors)
    ax.axvline(55, color=GREY, linestyle="--", linewidth=1)
    ax.axvline(70, color=GREY, linestyle="--", linewidth=1)
    ax.axvline(85, color=GREY, linestyle="--", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Production Insurability Score (experimental)")
    ax.set_title("Production certification by workflow")
    path = output_dir / "08_production_certification.png"
    _save(fig, path)
    paths.append(path)
    return paths
