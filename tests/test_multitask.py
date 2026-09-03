from pathlib import Path

import joblib
import pandas as pd
import pytest

from agent_mesh_risk_lab.multitask import (
    DEPLOYABLE_CATEGORICAL_FEATURES,
    DEPLOYABLE_NUMERIC_FEATURES,
    PRIVILEGED_SIMULATOR_FEATURES,
    TRACE_CATEGORICAL_FEATURES,
    TRACE_NUMERIC_FEATURES,
)

ROOT = Path(__file__).parents[1]


def test_deployable_features_exclude_simulator_privileged_fields():
    deployable = set(DEPLOYABLE_NUMERIC_FEATURES + DEPLOYABLE_CATEGORICAL_FEATURES)
    assert not deployable & PRIVILEGED_SIMULATOR_FEATURES
    assert "case_type" not in deployable
    assert "risk_level" not in deployable


def test_failure_attribution_does_not_receive_stressor_identity_or_raw_trace_text():
    trace_features = set(TRACE_NUMERIC_FEATURES + TRACE_CATEGORICAL_FEATURES)
    assert "stressor" not in trace_features
    assert "failure_label" not in trace_features
    assert "trace" not in trace_features
    assert "blast_radius" not in trace_features


def test_feature_access_audit_has_expected_negative_control_and_optimism_gap():
    audit = pd.read_csv(ROOT / "data" / "evaluation" / "feature_access_audit.csv")
    simulator = audit[audit["feature_access"] == "Simulator-informed structured"].iloc[0]
    deployable = audit[audit["feature_access"] == "Deployable structured"].iloc[0]
    shuffled = audit[audit["feature_access"] == "Label-shuffled negative control"].iloc[0]
    assert simulator["pr_auc"] > deployable["pr_auc"] > shuffled["pr_auc"]
    assert (
        abs(audit["pr_auc_optimism_gap"].iloc[0] - (simulator["pr_auc"] - deployable["pr_auc"]))
        < 1e-12
    )


def test_governance_model_reduces_decision_regret_against_majority():
    comparison = pd.read_csv(ROOT / "data" / "evaluation" / "multitask_comparison.csv")
    governance = comparison[comparison["task"] == "governance_recommendation"]
    majority = governance[governance["model"] == "Majority Class"].iloc[0]
    learned = governance[governance["model"] != "Majority Class"].iloc[0]
    assert learned["mean_decision_regret"] < majority["mean_decision_regret"]
    assert learned["top_3_accuracy"] >= learned["accuracy"]


@pytest.mark.filterwarnings("ignore:Setting the shape on a NumPy array:DeprecationWarning")
def test_persisted_deployable_scorer_uses_only_approved_columns():
    bundle = joblib.load(ROOT / "outputs" / "models" / "deployable_risk_model.joblib")
    assert set(bundle["columns"]) == set(
        DEPLOYABLE_NUMERIC_FEATURES + DEPLOYABLE_CATEGORICAL_FEATURES
    )
    assert not set(bundle["columns"]) & PRIVILEGED_SIMULATOR_FEATURES
    assert 0 < bundle["threshold"] < 1
