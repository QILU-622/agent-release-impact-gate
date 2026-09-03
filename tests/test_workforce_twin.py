from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent_mesh_risk_lab.workforce_twin import (
    aggregate_runs,
    build_backlog_timeline,
    load_twin_config,
    recommend_architectures,
    simulate_operating_day,
)

ROOT = Path(__file__).parents[1]
CONFIG = load_twin_config(ROOT / "configs" / "workforce_twin.json")


def test_operating_day_is_deterministic_and_metric_bounded() -> None:
    first_metrics, first_events = simulate_operating_day(
        CONFIG, "tiered_hybrid", "normal_day", 20260828
    )
    second_metrics, second_events = simulate_operating_day(
        CONFIG, "tiered_hybrid", "normal_day", 20260828
    )
    assert first_metrics == second_metrics
    pd.testing.assert_frame_equal(first_events, second_events)
    assert "review_duration_minutes" in first_events
    for metric in [
        "safe_completion_rate",
        "sla_attainment_rate",
        "critical_bypass_rate",
        "normal_overblock_rate",
        "automation_rate",
    ]:
        assert 0 <= first_metrics[metric] <= 1


def test_paired_architectures_receive_identical_arrivals() -> None:
    _, generalist = simulate_operating_day(CONFIG, "solo_generalist", "black_friday", 20260828)
    _, governed = simulate_operating_day(CONFIG, "governed_hybrid", "black_friday", 20260828)
    assert generalist["arrival_minute"].tolist() == governed["arrival_minute"].tolist()
    assert generalist["workflow"].tolist() == governed["workflow"].tolist()
    assert generalist["case_type"].tolist() == governed["case_type"].tolist()


def test_demand_surge_increases_volume_and_generalist_cycle_time() -> None:
    normal, _ = simulate_operating_day(CONFIG, "solo_generalist", "normal_day", 20260828)
    surge, _ = simulate_operating_day(CONFIG, "solo_generalist", "black_friday", 20260828)
    assert surge["arrivals"] > normal["arrivals"] * 2
    assert surge["p95_cycle_minutes"] > normal["p95_cycle_minutes"]


def test_backlog_timeline_reconciles_arrivals_and_completions() -> None:
    _, events = simulate_operating_day(CONFIG, "governed_hybrid", "black_friday", 20260828)
    timeline = build_backlog_timeline(events, CONFIG["simulation_minutes"])
    assert (timeline["backlog"] == timeline["arrived"] - timeline["completed"]).all()
    assert timeline.iloc[-1]["arrived"] == len(events)
    assert (timeline["backlog"] >= 0).all()


def test_recommendation_is_selected_from_observed_architectures() -> None:
    rows = []
    for architecture in CONFIG["architectures"]:
        metrics, _ = simulate_operating_day(CONFIG, architecture, "normal_day", 20260828)
        rows.append(metrics)
    summary = aggregate_runs(pd.DataFrame(rows))
    recommendation = recommend_architectures(summary, CONFIG)["normal_day"]
    assert recommendation["architecture"] in CONFIG["architectures"]
    assert "selection_rule" in recommendation


def test_gateway_separates_unsafe_proposals_from_executed_harm() -> None:
    generalist, _ = simulate_operating_day(
        CONFIG, "solo_generalist", "prompt_attack_wave", 20260828
    )
    governed, _ = simulate_operating_day(
        CONFIG, "governed_hybrid", "prompt_attack_wave", 20260828
    )
    assert governed["unsafe_proposal_rate"] > governed["harmful_execution_rate"]
    assert governed["unsafe_proposal_interception_rate"] > 0.9
    assert governed["critical_bypass_rate"] < generalist["critical_bypass_rate"]
