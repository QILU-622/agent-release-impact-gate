import pandas as pd

from agent_mesh_risk_lab.optimizer import optimize_portfolio


def test_optimizer_respects_budget_and_operating_constraints():
    controls = [
        ("context_envelope", 12, 0.13, 0.91, 0.20),
        ("tool_version_lock", 5, 0.06, 0.92, 0.20),
        ("permission_scope", 10, 0.09, 0.90, 0.20),
        ("human_review_gate", 45, 0.25, 0.86, 0.55),
        ("rollback_hook", 18, 0.05, 0.91, 0.20),
        ("external_isolation", 14, 0.10, 0.90, 0.20),
    ]
    frame = pd.DataFrame(
        [
            {
                "control": name,
                "risk_before": 0.60,
                "risk_reduction": reduction,
                "completion_before": 0.93,
                "completion_after": completion,
                "baseline_review_load": 0.20,
                "human_review_load": review,
            }
            for name, _cost, reduction, completion, review in controls
        ]
    )
    result = optimize_portfolio(frame, budget=40, min_completion=0.85, max_review_load=0.30)
    assert result["cost"] <= 40
    assert result["estimated_completion"] >= 0.85
    assert result["estimated_review_load"] <= 0.30
    assert result["risk_reduction"] > 0
