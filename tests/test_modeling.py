import numpy as np

from agent_mesh_risk_lab.modeling import classification_metrics, tune_threshold


def test_threshold_tuning_respects_over_blocking_guardrail_when_feasible():
    y_true = np.array([0] * 12 + [1] * 8)
    y_prob = np.array(
        [0.05, 0.08, 0.12, 0.18, 0.22, 0.27, 0.31, 0.36, 0.42, 0.48, 0.52, 0.58]
        + [0.35, 0.44, 0.53, 0.61, 0.69, 0.76, 0.84, 0.93]
    )
    threshold, objective = tune_threshold(y_true, y_prob, max_over_blocking=0.35)
    metrics = classification_metrics(y_true, y_prob, threshold)
    assert objective > 0
    assert metrics["over_blocking_rate"] <= 0.35
    assert 0.15 <= threshold <= 0.85
