from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_mesh_risk_lab.deployment_planner import (
    build_deployment_evidence_pack,
    build_reviewer_capacity_plan,
    capacity_recommendations,
    replay_reviewer_queue,
    summarize_external_evaluation,
)
from agent_mesh_risk_lab.workforce_twin import load_twin_config, simulate_operating_day

ROOT = Path(__file__).parents[1]
CONFIG = load_twin_config(ROOT / "configs" / "workforce_twin.json")


def test_external_evaluation_summary_discards_content_and_measures_coverage() -> None:
    payload = {
        "cases": [
            {
                "case_id": f"case-{index:02d}",
                "workflow": "refund",
                "case_type": "normal" if index % 2 == 0 else "risk",
                "passed": index % 4 != 0,
                "harmful_action": index % 5 == 0,
                "overblocked": index % 6 == 0,
                "latency_ms": 100 + index,
                "prompt": "sensitive prompt that must not be retained",
                "output": "sensitive output that must not be retained",
                "api_key": "secret",
            }
            for index in range(20)
        ]
    }
    summary = summarize_external_evaluation(payload)
    serialized = json.dumps(summary)

    assert summary["source"] == "canonical"
    assert summary["cases_total"] == 20
    assert summary["coverage"]["harm"] == 1
    assert summary["pass_rate"] == 0.75
    assert "sensitive prompt" not in serialized
    assert "api_key" not in serialized


def test_promptfoo_and_deepeval_shapes_are_detected() -> None:
    promptfoo = summarize_external_evaluation(
        {
            "results": [
                {
                    "success": True,
                    "metadata": {"workflow": "refund", "case_type": "normal"},
                    "response": {"latencyMs": 120},
                }
            ]
        }
    )
    deepeval = summarize_external_evaluation(
        {
            "testCases": [
                {
                    "success": False,
                    "metricsData": [
                        {"name": "Harmful action", "success": False, "score": 0}
                    ],
                }
            ]
        }
    )

    assert promptfoo["source"] == "promptfoo"
    assert promptfoo["pass_rate"] == 1
    assert promptfoo["median_latency_ms"] == 120
    assert deepeval["source"] == "deepeval"
    assert deepeval["harmful_action_rate"] == 1


def test_reviewer_capacity_plan_finds_smallest_guardrail_passing_pool() -> None:
    frames = []
    for seed in CONFIG["seeds"]:
        _, events = simulate_operating_day(CONFIG, "governed_hybrid", "black_friday", seed)
        frames.append(events)
    events = pd.concat(frames, ignore_index=True)

    two = replay_reviewer_queue(events[events["seed"] == CONFIG["seeds"][0]], 2, 480)
    six = replay_reviewer_queue(events[events["seed"] == CONFIG["seeds"][0]], 6, 480)
    assert six["p95_review_wait_minutes"] < two["p95_review_wait_minutes"]

    plan = build_reviewer_capacity_plan(events, CONFIG)
    recommendation = capacity_recommendations(plan, CONFIG)["black_friday|governed_hybrid"]
    assert recommendation["status"] == "ready"
    assert recommendation["recommended_nominal_reviewers"] >= 2
    recommended_row = plan[plan["is_recommended"]].iloc[0]
    assert recommended_row["capacity_guardrails_passed"]


def test_evidence_pack_blocks_unverified_customer_claim(tmp_path: Path) -> None:
    evidence = build_deployment_evidence_pack(tmp_path)
    assert evidence["status"] == "blocked"
    assert any("external Agent evaluation" in item for item in evidence["blockers"])
    assert any("shadow-pilot" in item for item in evidence["blockers"])
