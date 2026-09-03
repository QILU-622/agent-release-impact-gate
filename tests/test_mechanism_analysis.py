import pandas as pd

from agent_mesh_risk_lab.mechanism_analysis import (
    build_action_shift,
    build_simulator_validity,
    build_transition_analysis,
)


def _decision(
    scenario: int,
    mode: str,
    *,
    correct: bool,
    harmful: bool,
    action: str,
) -> dict:
    return {
        "scenario_id": f"scenario_{scenario:02d}",
        "task_id": f"task_{scenario:02d}",
        "workflow": "refund",
        "stressor": "none",
        "case_type": "normal" if scenario % 2 == 0 else "risk",
        "prompt_mode": mode,
        "action": action,
        "action_correct": correct,
        "harmful_action": harmful,
        "over_blocked": not correct and not harmful,
        "harm_target": "refund_order",
    }


def test_transition_analysis_separates_utility_gain_from_safety_loss():
    decisions = pd.DataFrame(
        [
            _decision(0, "governed", correct=False, harmful=False, action="refuse"),
            _decision(0, "governed_few_shot", correct=True, harmful=False, action="refund_order"),
            _decision(1, "governed", correct=False, harmful=False, action="refuse"),
            _decision(1, "governed_few_shot", correct=False, harmful=True, action="refund_order"),
            _decision(
                2, "governed", correct=True, harmful=False, action="check_refund_eligibility"
            ),
            _decision(2, "governed_few_shot", correct=False, harmful=True, action="refund_order"),
            _decision(3, "governed", correct=False, harmful=False, action="refuse"),
            _decision(3, "governed_few_shot", correct=False, harmful=False, action="refuse"),
        ]
    )
    transitions, summary = build_transition_analysis(decisions)
    counts = summary.set_index("transition")["scenarios"]
    assert counts["utility_gain_without_safety_loss"] == 1
    assert counts["safety_loss_without_utility_gain"] == 1
    assert counts["accuracy_and_safety_regression"] == 1
    assert (transitions["harm_change"].eq(1) & transitions["accuracy_change"].eq(1)).sum() == 0


def test_simulator_transfer_and_action_shift_preserve_expected_grain():
    scenarios = pd.DataFrame(
        [
            {
                "scenario_id": f"scenario_{index:02d}",
                "task_id": f"task_{index:02d}",
                "workflow": "refund",
                "stressor": "none",
                "case_type": "normal" if index % 2 == 0 else "risk",
            }
            for index in range(64)
        ]
    )
    simulator = pd.DataFrame(
        [
            {
                "task_id": f"task_{index:02d}",
                "stressor": "none",
                "control_config": "none",
                "risk_probability": 0.2 + 0.01 * index,
                "incident": index > 31,
            }
            for index in range(64)
        ]
    )
    decisions = pd.DataFrame(
        [
            _decision(
                index,
                mode,
                correct=index % 3 == 0,
                harmful=index >= 32,
                action="refund_order" if index >= 32 else "refuse",
            )
            for index in range(64)
            for mode in ("baseline", "governed", "governed_few_shot")
        ]
    )
    scored, validity = build_simulator_validity(
        decisions, scenarios, simulator, bootstrap_samples=50
    )
    assert len(scored) == 192
    assert list(validity["prompt_mode"]) == ["baseline", "governed", "governed_few_shot"]
    assert (validity["auroc"] == 1).all()

    benchmark = pd.DataFrame(
        [
            {
                "task_id": "example_1",
                "workflow_type": "refund",
                "expected_action": "check_refund_eligibility",
            }
        ]
    )
    manifest = {"few_shot_example_task_ids": {"refund": ["example_1"]}}
    shift = build_action_shift(decisions, benchmark, manifest)
    assert len(shift) == 2
    assert not shift["terminal_action_demonstrated"].any()
