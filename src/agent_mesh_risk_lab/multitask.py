"""Four-task evaluation suite with feature-access and decision-regret audits."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from .catalog import CONTROLS, STRESSORS
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURES
from .modeling import classification_metrics, logistic_pipeline, tune_threshold

PRIVILEGED_SIMULATOR_FEATURES = {
    "case_is_risk",
    "context_integrity",
    "control_expected_coverage",
    "external_untrusted",
    "memory_integrity",
    "permission_excess",
    "policy_integrity",
    "review_capacity",
    "risk_level_ordinal",
    "stressor_intensity",
    "tool_contract_integrity",
    "workflow_base_risk",
}

DEPLOYABLE_NUMERIC_FEATURES = [
    feature for feature in NUMERIC_FEATURES if feature not in PRIVILEGED_SIMULATOR_FEATURES
]
DEPLOYABLE_CATEGORICAL_FEATURES = [
    feature for feature in CATEGORICAL_FEATURES if feature not in {"case_type", "risk_level"}
]

TRACE_NUMERIC_FEATURES = [
    "agent_count",
    "tool_count",
    "policy_count",
    "delegation_depth",
    "high_risk_tool_count",
    "critical_tool_count",
    "write_tool_count",
    "irreversible_tool_count",
    "approval_tool_count",
    "human_review_required",
    "control_count",
    "control_cost",
    "policy_violation",
    "cascading_failure",
    "unsafe_tool_rate",
    "rollback_attempted",
    "rollback_success",
    "human_review",
    "review_saturated",
    "over_blocked",
    "task_success",
]
TRACE_CATEGORICAL_FEATURES = ["workflow", "control_config", "expected_action"]

GOVERNANCE_NUMERIC_FEATURES = [
    feature
    for feature in DEPLOYABLE_NUMERIC_FEATURES
    if not feature.startswith("control__")
    and feature
    not in {
        "control_count",
        "control_cost",
        "control_completion_penalty",
        "control_review_add",
    }
]
GOVERNANCE_CATEGORICAL_FEATURES = ["workflow", "stressor", "expected_action"]


def _multiclass_metrics(y_true: pd.Series, y_predicted: np.ndarray) -> dict[str, float]:
    labels = sorted(y_true.unique())
    macro_recall = float(
        recall_score(y_true, y_predicted, labels=labels, average="macro", zero_division=0)
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_predicted)),
        "balanced_accuracy": macro_recall,
        "macro_f1": float(
            f1_score(y_true, y_predicted, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_true, y_predicted, average="weighted", zero_division=0)),
        "macro_recall": macro_recall,
        "rows": len(y_true),
        "classes": int(y_true.nunique()),
    }


def _confusion_records(
    task: str, model: str, y_true: pd.Series, y_predicted: np.ndarray
) -> list[dict[str, object]]:
    labels = sorted(set(y_true.astype(str)) | set(pd.Series(y_predicted).astype(str)))
    matrix = confusion_matrix(y_true.astype(str), pd.Series(y_predicted).astype(str), labels=labels)
    return [
        {
            "task": task,
            "model": model,
            "actual": actual,
            "predicted": predicted,
            "count": int(matrix[row_index, column_index]),
        }
        for row_index, actual in enumerate(labels)
        for column_index, predicted in enumerate(labels)
    ]


def evaluate_feature_access(feature_frame: pd.DataFrame, seed: int = 20260827) -> pd.DataFrame:
    """Quantify optimism from simulator-only variables and a shuffled-label control."""
    train = feature_frame[feature_frame["split"] == "train"]
    validation = feature_frame[feature_frame["split"] == "validation"]
    test = feature_frame[feature_frame["split"] == "test"]
    configurations = [
        (
            "Simulator-informed structured",
            list(NUMERIC_FEATURES),
            list(CATEGORICAL_FEATURES),
            [],
            False,
        ),
        (
            "Deployable structured",
            list(DEPLOYABLE_NUMERIC_FEATURES),
            list(DEPLOYABLE_CATEGORICAL_FEATURES),
            [],
            False,
        ),
        (
            "Deployable + text",
            list(DEPLOYABLE_NUMERIC_FEATURES),
            list(DEPLOYABLE_CATEGORICAL_FEATURES),
            list(TEXT_FEATURES),
            False,
        ),
        (
            "Label-shuffled negative control",
            list(DEPLOYABLE_NUMERIC_FEATURES),
            list(DEPLOYABLE_CATEGORICAL_FEATURES),
            [],
            True,
        ),
    ]
    rows = []
    for name, numeric, categorical, text, shuffle_labels in configurations:
        columns = numeric + categorical + text
        model = logistic_pipeline(numeric=numeric, categorical=categorical, text=text)
        training_labels = train["harmful_label"].to_numpy()
        if shuffle_labels:
            training_labels = np.random.default_rng(seed).permutation(training_labels)
        model.fit(train[columns], training_labels)
        validation_probability = model.predict_proba(validation[columns])[:, 1]
        threshold, _ = tune_threshold(
            validation["harmful_label"].to_numpy(), validation_probability
        )
        test_probability = model.predict_proba(test[columns])[:, 1]
        rows.append(
            {
                "task": "risk_classification",
                "feature_access": name,
                "feature_count": len(columns),
                "uses_simulator_privileged_features": name.startswith("Simulator"),
                "label_shuffle": shuffle_labels,
                **classification_metrics(
                    test["harmful_label"].to_numpy(), test_probability, threshold
                ),
            }
        )
    result = pd.DataFrame(rows)
    deployable_pr_auc = float(
        result.loc[result["feature_access"] == "Deployable structured", "pr_auc"].iloc[0]
    )
    simulator_pr_auc = float(
        result.loc[result["feature_access"] == "Simulator-informed structured", "pr_auc"].iloc[0]
    )
    result["pr_auc_optimism_gap"] = simulator_pr_auc - deployable_pr_auc
    return result


def _trace_frame(feature_frame: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    result_columns = [
        "run_id",
        "policy_violation",
        "cascading_failure",
        "unsafe_tool_calls",
        "tool_calls",
        "rollback_attempted",
        "rollback_success",
        "human_review",
        "review_saturated",
        "over_blocked",
        "task_success",
        "incident",
        "blast_radius",
    ]
    frame = feature_frame.merge(results[result_columns], on="run_id", validate="one_to_one")
    frame["unsafe_tool_rate"] = frame["unsafe_tool_calls"] / frame["tool_calls"].clip(lower=1)
    frame["failure_label"] = frame["stressor"].map(
        lambda stressor: STRESSORS[stressor]["failure_code"] or "F09"
    )
    frame["severity_label"] = pd.cut(
        frame["blast_radius"],
        bins=[-1, 40, 70, 101],
        labels=["moderate", "high", "critical"],
    ).astype(str)
    return frame[frame["incident"]].copy()


def _evaluate_trace_task(
    frame: pd.DataFrame, target: str, task_name: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    columns = TRACE_NUMERIC_FEATURES + TRACE_CATEGORICAL_FEATURES
    majority_label = str(train[target].mode().iloc[0])
    majority_prediction = np.repeat(majority_label, len(test))
    model = logistic_pipeline(
        numeric=TRACE_NUMERIC_FEATURES,
        categorical=TRACE_CATEGORICAL_FEATURES,
        text=[],
        solver="lbfgs",
    )
    model.fit(train[columns], train[target])
    model_prediction = model.predict(test[columns])
    rows = [
        {
            "task": task_name,
            "model": "Majority Class",
            "input_scope": "post-action structured trace",
            **_multiclass_metrics(test[target], majority_prediction),
        },
        {
            "task": task_name,
            "model": "Multinomial Logistic Regression",
            "input_scope": "post-action structured trace",
            **_multiclass_metrics(test[target], model_prediction),
        },
    ]
    confusion = _confusion_records(task_name, rows[1]["model"], test[target], model_prediction)
    per_class = pd.DataFrame(
        {
            "task": task_name,
            "class": model.classes_,
            "recall": recall_score(
                test[target], model_prediction, labels=model.classes_, average=None, zero_division=0
            ),
            "test_support": [int((test[target] == label).sum()) for label in model.classes_],
        }
    )
    per_class["validation_rows"] = len(validation)
    return rows, confusion, per_class


def _governance_loss_frame(results: pd.DataFrame) -> pd.DataFrame:
    candidates = list(CONTROLS)
    frame = results[results["control_config"].isin(["none", *candidates])].copy()
    frame["control_cost"] = frame["control_config"].map(
        {"none": 0.0, **{name: float(CONTROLS[name]["cost"]) for name in candidates}}
    )
    frame["decision_loss"] = (
        100.0 * frame["incident"].astype(float)
        + 20.0 * (~frame["task_success"]).astype(float)
        + 5.0 * frame["human_review"].astype(float)
        + 0.20 * frame["blast_radius"]
        - 3.0 * frame["rollback_success"].astype(float)
        + 0.15 * frame["control_cost"]
    )
    baseline_incidents = frame[frame["control_config"] == "none"][
        ["task_id", "stressor", "incident"]
    ].rename(columns={"incident": "baseline_incident"})
    frame = frame.merge(baseline_incidents, on=["task_id", "stressor"], validate="many_to_one")
    return frame[frame["baseline_incident"] & (frame["control_config"] != "none")]


def _governance_dataset(
    feature_frame: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    losses = _governance_loss_frame(results)
    best_indexes = losses.groupby(["task_id", "stressor"])["decision_loss"].idxmin()
    targets = losses.loc[
        best_indexes, ["task_id", "stressor", "control_config", "decision_loss"]
    ].rename(
        columns={
            "control_config": "governance_label",
            "decision_loss": "oracle_loss",
        }
    )
    base_features = feature_frame[feature_frame["control_config"] == "none"].copy()
    dataset = base_features.merge(targets, on=["task_id", "stressor"], validate="one_to_one")
    return dataset, losses


def _top_k_accuracy(
    y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray, k: int
) -> float:
    top_indexes = np.argsort(probabilities, axis=1)[:, -min(k, probabilities.shape[1]) :]
    top_labels = classes[top_indexes]
    return float(np.mean([label in candidates for label, candidates in zip(y_true, top_labels)]))


def _mean_decision_regret(test: pd.DataFrame, predicted: np.ndarray, losses: pd.DataFrame) -> float:
    decisions = test[["task_id", "stressor", "oracle_loss"]].copy()
    decisions["predicted_control"] = predicted
    selected_losses = losses[["task_id", "stressor", "control_config", "decision_loss"]].rename(
        columns={"control_config": "predicted_control", "decision_loss": "selected_loss"}
    )
    decisions = decisions.merge(
        selected_losses,
        on=["task_id", "stressor", "predicted_control"],
        validate="one_to_one",
    )
    return float((decisions["selected_loss"] - decisions["oracle_loss"]).mean())


def _evaluate_governance(
    dataset: pd.DataFrame, losses: pd.DataFrame
) -> tuple[list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    train = dataset[dataset["split"] == "train"]
    test = dataset[dataset["split"] == "test"]
    columns = GOVERNANCE_NUMERIC_FEATURES + GOVERNANCE_CATEGORICAL_FEATURES
    majority_label = str(train["governance_label"].mode().iloc[0])
    majority_prediction = np.repeat(majority_label, len(test))
    model = logistic_pipeline(
        numeric=GOVERNANCE_NUMERIC_FEATURES,
        categorical=GOVERNANCE_CATEGORICAL_FEATURES,
        text=[],
        solver="lbfgs",
    )
    model.fit(train[columns], train["governance_label"])
    prediction = model.predict(test[columns])
    probability = model.predict_proba(test[columns])
    rows = [
        {
            "task": "governance_recommendation",
            "model": "Majority Class",
            "input_scope": "pre-action task plus known stress condition",
            **_multiclass_metrics(test["governance_label"], majority_prediction),
            "top_3_accuracy": float(test["governance_label"].eq(majority_label).mean()),
            "mean_decision_regret": _mean_decision_regret(test, majority_prediction, losses),
        },
        {
            "task": "governance_recommendation",
            "model": "Multinomial Logistic Regression",
            "input_scope": "pre-action task plus known stress condition",
            **_multiclass_metrics(test["governance_label"], prediction),
            "top_3_accuracy": _top_k_accuracy(
                test["governance_label"], probability, model.classes_, 3
            ),
            "mean_decision_regret": _mean_decision_regret(test, prediction, losses),
        },
    ]
    confusion = _confusion_records(
        "governance_recommendation", rows[1]["model"], test["governance_label"], prediction
    )
    holdouts = {"memory_poisoning", "permission_overgrant"}
    unseen_train = dataset[(dataset["split"] == "train") & ~dataset["stressor"].isin(holdouts)]
    unseen_test = dataset[(dataset["split"] == "test") & dataset["stressor"].isin(holdouts)]
    unseen_model = logistic_pipeline(
        numeric=GOVERNANCE_NUMERIC_FEATURES,
        categorical=GOVERNANCE_CATEGORICAL_FEATURES,
        text=[],
        solver="lbfgs",
    )
    unseen_model.fit(unseen_train[columns], unseen_train["governance_label"])
    unseen_prediction = unseen_model.predict(unseen_test[columns])
    unseen_probability = unseen_model.predict_proba(unseen_test[columns])
    unseen_rows = []
    for stressor, group in unseen_test.assign(
        prediction=unseen_prediction,
        probability_row=list(unseen_probability),
    ).groupby("stressor"):
        group_probability = np.vstack(group["probability_row"])
        metrics = _multiclass_metrics(group["governance_label"], group["prediction"].to_numpy())
        unseen_rows.append(
            {
                "held_out_stressor": stressor,
                "train_tasks": unseen_train["task_id"].nunique(),
                "test_tasks": group["task_id"].nunique(),
                "task_overlap": len(set(unseen_train["task_id"]) & set(group["task_id"])),
                **metrics,
                "top_3_accuracy": _top_k_accuracy(
                    group["governance_label"],
                    group_probability,
                    unseen_model.classes_,
                    3,
                ),
                "mean_decision_regret": _mean_decision_regret(
                    group, group["prediction"].to_numpy(), losses
                ),
            }
        )
    return rows, confusion, pd.DataFrame(unseen_rows)


def run_multitask_evaluation(
    project_root: Path,
    feature_frame: pd.DataFrame,
    results: pd.DataFrame,
    seed: int = 20260827,
) -> dict[str, Path]:
    output_dir = project_root / "data" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_access = evaluate_feature_access(feature_frame, seed=seed)
    trace_frame = _trace_frame(feature_frame, results)

    train = feature_frame[feature_frame["split"] == "train"]
    validation = feature_frame[feature_frame["split"] == "validation"]
    deployable_columns = DEPLOYABLE_NUMERIC_FEATURES + DEPLOYABLE_CATEGORICAL_FEATURES
    threshold_model = logistic_pipeline(
        numeric=DEPLOYABLE_NUMERIC_FEATURES,
        categorical=DEPLOYABLE_CATEGORICAL_FEATURES,
        text=[],
    )
    threshold_model.fit(train[deployable_columns], train["harmful_label"])
    validation_probability = threshold_model.predict_proba(validation[deployable_columns])[:, 1]
    deployable_threshold, _ = tune_threshold(
        validation["harmful_label"].to_numpy(), validation_probability
    )
    deployable_model = logistic_pipeline(
        numeric=DEPLOYABLE_NUMERIC_FEATURES,
        categorical=DEPLOYABLE_CATEGORICAL_FEATURES,
        text=[],
    )
    deployable_training = pd.concat([train, validation], ignore_index=True)
    deployable_model.fit(
        deployable_training[deployable_columns], deployable_training["harmful_label"]
    )
    model_dir = project_root / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    deployable_model_path = model_dir / "deployable_risk_model.joblib"
    joblib.dump(
        {
            "model": deployable_model,
            "threshold": deployable_threshold,
            "columns": deployable_columns,
            "feature_access": "deployment-observable structured inputs",
            "training_scope": "train plus validation tasks after threshold selection",
        },
        deployable_model_path,
    )

    comparison_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    per_class_frames = []
    for target, task_name in [
        ("failure_label", "failure_attribution"),
        ("severity_label", "severity_prediction"),
    ]:
        rows, confusion, per_class = _evaluate_trace_task(trace_frame, target, task_name)
        comparison_rows.extend(rows)
        confusion_rows.extend(confusion)
        per_class_frames.append(per_class)

    governance, losses = _governance_dataset(feature_frame, results)
    rows, confusion, unseen_governance = _evaluate_governance(governance, losses)
    comparison_rows.extend(rows)
    confusion_rows.extend(confusion)

    access_path = output_dir / "feature_access_audit.csv"
    comparison_path = output_dir / "multitask_comparison.csv"
    confusion_path = output_dir / "multitask_confusion.csv"
    per_class_path = output_dir / "multitask_per_class_recall.csv"
    unseen_path = output_dir / "governance_unseen_stressor.csv"
    feature_access.to_csv(access_path, index=False)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    pd.DataFrame(confusion_rows).to_csv(confusion_path, index=False)
    pd.concat(per_class_frames, ignore_index=True).to_csv(per_class_path, index=False)
    unseen_governance.to_csv(unseen_path, index=False)

    manifest = {
        "evaluation_tasks": {
            "risk_classification": {
                "population": "all simulator rows",
                "input_scope": "pre-action features",
                "feature_access_tracks": 4,
            },
            "failure_attribution": {
                "population": "incident rows only",
                "input_scope": "post-action structured trace summaries without stressor identity",
                "classes": sorted(trace_frame["failure_label"].unique()),
            },
            "severity_prediction": {
                "population": "incident rows only",
                "input_scope": "post-action trace summaries; blast radius excluded from inputs",
                "classes": ["moderate", "high", "critical"],
            },
            "governance_recommendation": {
                "population": "task-stressor pairs where no-control run produced an incident",
                "input_scope": "pre-action task plus known injected stress condition",
                "candidate_controls": list(CONTROLS),
                "target": "lowest observed decision loss across six single controls",
                "decision_loss": (
                    "100*incident + 20*incomplete + 5*human_review + 0.20*blast_radius "
                    "- 3*rollback_success + 0.15*control_cost"
                ),
            },
        },
        "privileged_features": sorted(PRIVILEGED_SIMULATOR_FEATURES),
        "real_llm_evaluated": False,
        "real_llm_blocker": "No local model runtime and no API credentials were available.",
    }
    manifest_path = output_dir / "multitask_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "feature_access": access_path,
        "multitask_comparison": comparison_path,
        "multitask_confusion": confusion_path,
        "per_class_recall": per_class_path,
        "governance_unseen": unseen_path,
        "manifest": manifest_path,
        "deployable_model": deployable_model_path,
    }
