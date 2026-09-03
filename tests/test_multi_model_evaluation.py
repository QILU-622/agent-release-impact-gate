from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from agent_mesh_risk_lab.multi_model_evaluation import compare_runs, model_slug


def _fake_run(path: Path, model: str, actions: list[str]) -> None:
    path.mkdir(parents=True)
    rows = []
    for scenario_index in range(64):
        for mode_index, mode in enumerate(("baseline", "governed", "governed_few_shot")):
            action = actions[(scenario_index + mode_index) % len(actions)]
            rows.append(
                {
                    "scenario_id": f"scenario-{scenario_index:02d}",
                    "prompt_mode": mode,
                    "action": action,
                    "harmful_action": action == "refund_order",
                    "action_correct": action == "request_human_approval",
                }
            )
    pd.DataFrame(rows).to_csv(path / "decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "prompt_mode": mode,
                "harmful_action_rate": harm,
                "action_accuracy": accuracy,
                "normal_case_overblocking_rate": overblock,
                "policy_compliance_rate": 1 - harm,
            }
            for mode, harm, accuracy, overblock in [
                ("baseline", 0.5, 0.2, 0.1),
                ("governed", 0.1, 0.2, 0.6),
                ("governed_few_shot", 0.3, 0.3, 0.3),
            ]
        ]
    ).to_csv(path / "aggregate.csv", index=False)
    pd.DataFrame(
        [
            {
                "comparison": "governed_vs_baseline",
                "metric": "harmful_action_rate",
                "raw_delta_treatment_minus_reference": -0.4,
                "ci_low": -0.5,
                "ci_high": -0.3,
            }
        ]
    ).to_csv(path / "paired_effects.csv", index=False)
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "model": model,
                "valid_schema_responses": 192,
                "failed_responses": 0,
                "model_metadata": {"digest": f"digest-{model}"},
            }
        )
    )


def test_model_slug_is_path_safe() -> None:
    assert model_slug("qwen2.5:3b-instruct") == "qwen2.5_3b-instruct"


def test_compare_runs_builds_cross_model_evidence(tmp_path: Path) -> None:
    qwen = tmp_path / "qwen"
    llama = tmp_path / "llama"
    _fake_run(qwen, "qwen-test", ["request_human_approval", "refund_order"])
    _fake_run(llama, "llama-test", ["request_human_approval", "refuse"])

    paths = compare_runs([qwen, llama], tmp_path / "tables", tmp_path / "report.md")

    aggregate = pd.read_csv(paths["aggregate"])
    agreement = pd.read_csv(paths["scenario_agreement"])
    manifest = json.loads(paths["manifest"].read_text())
    assert set(aggregate["model"]) == {"qwen-test", "llama-test"}
    assert len(agreement) == 3
    assert manifest["decision_calls_per_model"] == 192
    assert "safety certificate" in paths["report"].read_text()
