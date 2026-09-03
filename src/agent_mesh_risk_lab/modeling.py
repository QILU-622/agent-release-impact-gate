"""Offline model baselines, calibration, ablation, generalization, and error analysis."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .catalog import FAILURE_TAXONOMY, STRESSORS
from .features import (
    CATEGORICAL_FEATURES,
    FEATURE_GROUPS,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    build_feature_frame,
    feature_manifest,
)
from .schema import WorkflowTask

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES
ModelFactory = Callable[[], Pipeline]


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (y_prob >= lower) & (y_prob < upper if index < bins - 1 else y_prob <= upper)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return error if total else 0.0


def tune_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, max_over_blocking: float = 0.35
) -> tuple[float, float]:
    """Maximize safety-weighted F2 subject to an explicit over-blocking guardrail."""
    candidates: list[tuple[float, float, float]] = []
    for threshold in np.linspace(0.15, 0.85, 71):
        predicted = y_prob >= threshold
        negatives = y_true == 0
        false_positive_rate = float(predicted[negatives].mean()) if negatives.any() else 0.0
        objective = float(fbeta_score(y_true, predicted, beta=2, zero_division=0))
        candidates.append((float(threshold), objective, false_positive_rate))
    feasible = [item for item in candidates if item[2] <= max_over_blocking]
    pool = feasible or candidates
    threshold, objective, _ = max(pool, key=lambda item: (item[1], -item[2]))
    return threshold, objective


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    predicted = y_prob >= threshold
    negative = y_true == 0
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "safety_recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(expected_calibration_error(y_true, y_prob)),
        "over_blocking_rate": float(predicted[negative].mean()) if negative.any() else 0.0,
        "threshold": float(threshold),
        "rows": len(y_true),
        "positive_rate": float(y_true.mean()),
    }


def _preprocessor(
    numeric: list[str], categorical: list[str], text: list[str], dense: bool = False
) -> ColumnTransformer:
    transformers: list[tuple[str, object, object]] = []
    if numeric:
        numeric_transformer = "passthrough" if dense else StandardScaler(with_mean=False)
        transformers.append(("numeric", numeric_transformer, numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=not dense),
                categorical,
            )
        )
    for field in text:
        transformers.append(
            (
                f"text_{field}",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=700,
                    sublinear_tf=True,
                ),
                field,
            )
        )
    return ColumnTransformer(transformers, sparse_threshold=0.0 if dense else 0.3)


def logistic_pipeline(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    text: list[str] | None = None,
    solver: str = "liblinear",
) -> Pipeline:
    numeric = list(NUMERIC_FEATURES if numeric is None else numeric)
    categorical = list(CATEGORICAL_FEATURES if categorical is None else categorical)
    text = list(TEXT_FEATURES if text is None else text)
    return Pipeline(
        [
            ("features", _preprocessor(numeric, categorical, text, dense=False)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=700,
                    class_weight="balanced",
                    solver=solver,
                    random_state=20260827,
                ),
            ),
        ]
    )


def model_factories(seed: int = 20260827) -> dict[str, ModelFactory]:
    structured_dense = lambda: _preprocessor(
        list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES), [], dense=True
    )
    return {
        "Logistic Regression": lambda: logistic_pipeline(text=[]),
        "Logistic + TF-IDF": lambda: logistic_pipeline(),
        "Random Forest": lambda: Pipeline(
            [
                ("features", structured_dense()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=260,
                        max_depth=13,
                        min_samples_leaf=7,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
        "Extra Trees": lambda: Pipeline(
            [
                ("features", structured_dense()),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=260,
                        max_depth=14,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
        "Histogram Gradient Boosting": lambda: Pipeline(
            [
                ("features", structured_dense()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.055,
                        max_leaf_nodes=31,
                        min_samples_leaf=24,
                        l2_regularization=0.8,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "XGBoost": lambda: Pipeline(
            [
                ("features", structured_dense()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=260,
                        max_depth=6,
                        learning_rate=0.045,
                        min_child_weight=8,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=0.15,
                        reg_lambda=1.2,
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=seed,
                        n_jobs=2,
                    ),
                ),
            ]
        ),
    }


def rule_based_scores(frame: pd.DataFrame) -> np.ndarray:
    score = (
        0.08
        + 0.12 * frame["case_is_risk"].to_numpy()
        + 0.14 * frame["critical_tool_count"].clip(0, 1).to_numpy()
        + 0.10 * frame["high_risk_tool_count"].clip(0, 1).to_numpy()
        + 0.18 * frame["stressor_intensity"].to_numpy()
        + 0.12 * frame["external_untrusted"].to_numpy()
        + 0.11 * frame["permission_excess"].to_numpy()
        + 0.09 * (1 - frame["policy_integrity"].to_numpy())
        + 0.08 * (1 - frame["context_integrity"].to_numpy())
        + 0.08 * (1 - frame["memory_integrity"].to_numpy())
        + 0.06 * frame["human_review_required"].to_numpy()
        - 0.46 * frame["control_expected_coverage"].to_numpy()
    )
    return np.clip(score, 0.01, 0.99)


def _calibration_points(
    y_true: np.ndarray, y_prob: np.ndarray, model: str, bins: int = 10
) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    frame["bin"] = pd.qcut(frame["y_prob"], q=bins, duplicates="drop")
    points = (
        frame.groupby("bin", observed=True)
        .agg(
            mean_predicted_risk=("y_prob", "mean"),
            observed_incident_rate=("y_true", "mean"),
            rows=("y_true", "size"),
        )
        .reset_index(drop=True)
    )
    points.insert(0, "model", model)
    return points


def _cluster_bootstrap(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    iterations: int = 300,
    seed: int = 20260827,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    task_ids = test["task_id"].drop_duplicates().to_numpy()
    values: dict[str, list[float]] = {
        "f1": [],
        "safety_recall": [],
        "auroc": [],
        "pr_auc": [],
        "over_blocking_rate": [],
    }
    y_all = test["harmful_label"].to_numpy(dtype=int)
    task_array = test["task_id"].to_numpy()
    for _ in range(iterations):
        sampled_tasks = rng.choice(task_ids, size=len(task_ids), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(task_array == task_id) for task_id in sampled_tasks]
        )
        y_true = y_all[indices]
        y_prob = probabilities[indices]
        if len(np.unique(y_true)) < 2:
            continue
        metrics = classification_metrics(y_true, y_prob, threshold)
        for metric, samples in values.items():
            samples.append(metrics[metric])
    rows = []
    for metric, samples in values.items():
        rows.append(
            {
                "metric": metric,
                "estimate": classification_metrics(y_all, probabilities, threshold)[metric],
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_iterations": len(samples),
                "resampling_unit": "task_id",
            }
        )
    return pd.DataFrame(rows)


def _ablation_experiments(frame: pd.DataFrame) -> pd.DataFrame:
    configurations = {
        "Full input": set(),
        "No policy signal": set(FEATURE_GROUPS["policy"]),
        "No tool contract": set(FEATURE_GROUPS["tool"]),
        "No graph": set(FEATURE_GROUPS["graph"]),
        "No handoff context": set(FEATURE_GROUPS["context"]),
        "No text": set(FEATURE_GROUPS["text"]),
    }
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    rows = []
    for name, removed in configurations.items():
        numeric = [field for field in NUMERIC_FEATURES if field not in removed]
        categorical = [field for field in CATEGORICAL_FEATURES if field not in removed]
        text = [field for field in TEXT_FEATURES if field not in removed]
        model = logistic_pipeline(numeric=numeric, categorical=categorical, text=text)
        model.fit(train[numeric + categorical + text], train["harmful_label"])
        validation_prob = model.predict_proba(validation[numeric + categorical + text])[:, 1]
        threshold, _ = tune_threshold(validation["harmful_label"].to_numpy(), validation_prob)
        test_prob = model.predict_proba(test[numeric + categorical + text])[:, 1]
        metrics = classification_metrics(test["harmful_label"].to_numpy(), test_prob, threshold)
        rows.append(
            {
                "configuration": name,
                "removed_features": ",".join(sorted(removed)) or "none",
                **metrics,
            }
        )
    result = pd.DataFrame(rows)
    full_f1 = float(result.loc[result["configuration"] == "Full input", "f1"].iloc[0])
    full_recall = float(
        result.loc[result["configuration"] == "Full input", "safety_recall"].iloc[0]
    )
    result["f1_delta_vs_full"] = result["f1"] - full_f1
    result["recall_delta_vs_full"] = result["safety_recall"] - full_recall
    return result


def _strict_unseen_stressor(
    frame: pd.DataFrame, factory: ModelFactory, holdouts: tuple[str, ...]
) -> pd.DataFrame:
    train = frame[(frame["split"] == "train") & ~frame["stressor"].isin(holdouts)]
    validation = frame[(frame["split"] == "validation") & ~frame["stressor"].isin(holdouts)]
    test = frame[(frame["split"] == "test") & frame["stressor"].isin(holdouts)]
    model = factory()
    model.fit(train[ALL_FEATURES], train["harmful_label"])
    validation_prob = model.predict_proba(validation[ALL_FEATURES])[:, 1]
    threshold, _ = tune_threshold(validation["harmful_label"].to_numpy(), validation_prob)
    probabilities = model.predict_proba(test[ALL_FEATURES])[:, 1]
    rows = []
    for stressor, group in test.assign(probability=probabilities).groupby("stressor"):
        metrics = classification_metrics(
            group["harmful_label"].to_numpy(), group["probability"].to_numpy(), threshold
        )
        rows.append(
            {
                "held_out_stressor": stressor,
                "train_tasks": train["task_id"].nunique(),
                "test_tasks": group["task_id"].nunique(),
                "task_overlap": len(set(train["task_id"]) & set(group["task_id"])),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _cross_workflow_generalization(frame: pd.DataFrame, factory: ModelFactory) -> pd.DataFrame:
    rows = []
    for workflow in sorted(frame["workflow"].unique()):
        train = frame[(frame["split"] == "train") & (frame["workflow"] != workflow)]
        validation = frame[(frame["split"] == "validation") & (frame["workflow"] != workflow)]
        test = frame[(frame["split"] == "test") & (frame["workflow"] == workflow)]
        model = factory()
        model.fit(train[ALL_FEATURES], train["harmful_label"])
        validation_prob = model.predict_proba(validation[ALL_FEATURES])[:, 1]
        threshold, _ = tune_threshold(validation["harmful_label"].to_numpy(), validation_prob)
        probabilities = model.predict_proba(test[ALL_FEATURES])[:, 1]
        rows.append(
            {
                "held_out_workflow": workflow,
                "train_workflows": ",".join(sorted(train["workflow"].unique())),
                "train_tasks": train["task_id"].nunique(),
                "test_tasks": test["task_id"].nunique(),
                "task_overlap": len(set(train["task_id"]) & set(test["task_id"])),
                **classification_metrics(
                    test["harmful_label"].to_numpy(), probabilities, threshold
                ),
            }
        )
    return pd.DataFrame(rows)


def run_model_evaluation(
    project_root: Path,
    tasks: list[WorkflowTask],
    results: pd.DataFrame,
    seed: int = 20260827,
) -> dict[str, Path]:
    output_dir = project_root / "data" / "evaluation"
    model_dir = project_root / "outputs" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = build_feature_frame(tasks, results, seed=seed)
    frame.to_csv(output_dir / "feature_dataset.csv", index=False)
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(feature_manifest(), indent=2), encoding="utf-8"
    )
    split_manifest = (
        frame[["task_id", "workflow", "case_type", "split"]]
        .drop_duplicates()
        .sort_values(["split", "workflow", "case_type", "task_id"])
    )
    split_manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    train = frame[frame["split"] == "train"].copy()
    validation = frame[frame["split"] == "validation"].copy()
    test = frame[frame["split"] == "test"].copy()
    y_train = train["harmful_label"].to_numpy(dtype=int)
    y_validation = validation["harmful_label"].to_numpy(dtype=int)
    y_test = test["harmful_label"].to_numpy(dtype=int)

    comparison_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    fitted_models: dict[str, Pipeline] = {}

    rule_validation = rule_based_scores(validation)
    rule_threshold, _ = tune_threshold(y_validation, rule_validation)
    rule_test = rule_based_scores(test)
    comparison_rows.append(
        {
            "model": "Rule Based",
            "model_family": "heuristic",
            "validation_pr_auc": average_precision_score(y_validation, rule_validation),
            **classification_metrics(y_test, rule_test, rule_threshold),
        }
    )
    prediction_frames.append(
        test[["run_id", "task_id", "workflow", "stressor", "control_config"]].assign(
            model="Rule Based",
            y_true=y_test,
            y_probability=rule_test,
            y_predicted=(rule_test >= rule_threshold).astype(int),
        )
    )

    factories = model_factories(seed)
    validation_scores: dict[str, float] = {}
    for name, factory in factories.items():
        model = factory()
        model.fit(train[ALL_FEATURES], y_train)
        validation_prob = model.predict_proba(validation[ALL_FEATURES])[:, 1]
        threshold, _ = tune_threshold(y_validation, validation_prob)
        test_prob = model.predict_proba(test[ALL_FEATURES])[:, 1]
        validation_pr_auc = float(average_precision_score(y_validation, validation_prob))
        validation_scores[name] = validation_pr_auc
        comparison_rows.append(
            {
                "model": name,
                "model_family": "offline_ml",
                "validation_pr_auc": validation_pr_auc,
                **classification_metrics(y_test, test_prob, threshold),
            }
        )
        prediction_frames.append(
            test[["run_id", "task_id", "workflow", "stressor", "control_config"]].assign(
                model=name,
                y_true=y_test,
                y_probability=test_prob,
                y_predicted=(test_prob >= threshold).astype(int),
            )
        )
        fitted_models[name] = model

    best_name = max(validation_scores, key=validation_scores.get)
    best_model = fitted_models[best_name]
    validation_prob = best_model.predict_proba(validation[ALL_FEATURES])[:, 1]
    validation_task_ids = sorted(validation["task_id"].unique())
    calibration_tasks = set(validation_task_ids[::2])
    calibration_mask = validation["task_id"].isin(calibration_tasks).to_numpy()
    tune_mask = ~calibration_mask
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(validation_prob[calibration_mask], y_validation[calibration_mask])
    platt = LogisticRegression(random_state=seed)
    platt.fit(validation_prob[calibration_mask].reshape(-1, 1), y_validation[calibration_mask])
    calibration_candidates = {
        "Isotonic": isotonic.transform(validation_prob[tune_mask]),
        "Platt": platt.predict_proba(validation_prob[tune_mask].reshape(-1, 1))[:, 1],
    }
    calibration_method = min(
        calibration_candidates,
        key=lambda name: expected_calibration_error(
            y_validation[tune_mask], calibration_candidates[name]
        ),
    )
    calibrator = isotonic if calibration_method == "Isotonic" else platt
    tuned_validation = calibration_candidates[calibration_method]
    calibrated_threshold, _ = tune_threshold(y_validation[tune_mask], tuned_validation)
    raw_test_prob = best_model.predict_proba(test[ALL_FEATURES])[:, 1]
    calibrated_test_prob = (
        calibrator.transform(raw_test_prob)
        if calibration_method == "Isotonic"
        else calibrator.predict_proba(raw_test_prob.reshape(-1, 1))[:, 1]
    )
    calibrated_name = f"{best_name} + {calibration_method}"
    comparison_rows.append(
        {
            "model": calibrated_name,
            "model_family": "calibrated_offline_ml",
            "validation_pr_auc": average_precision_score(y_validation[tune_mask], tuned_validation),
            **classification_metrics(y_test, calibrated_test_prob, calibrated_threshold),
        }
    )
    calibrated_predictions = test[
        ["run_id", "task_id", "workflow", "stressor", "control_config", "risk_level"]
    ].assign(
        model=calibrated_name,
        y_true=y_test,
        y_probability=calibrated_test_prob,
        y_predicted=(calibrated_test_prob >= calibrated_threshold).astype(int),
    )
    prediction_frames.append(calibrated_predictions.drop(columns=["risk_level"]))

    comparison = pd.DataFrame(comparison_rows).sort_values("pr_auc", ascending=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    calibration = pd.concat(
        [
            _calibration_points(y_test, raw_test_prob, best_name),
            _calibration_points(y_test, calibrated_test_prob, calibrated_name),
        ],
        ignore_index=True,
    )
    calibration.to_csv(output_dir / "model_calibration.csv", index=False)

    bootstrap = _cluster_bootstrap(
        test, calibrated_test_prob, calibrated_threshold, iterations=300, seed=seed
    )
    bootstrap.insert(0, "model", calibrated_name)
    bootstrap.to_csv(output_dir / "bootstrap_confidence_intervals.csv", index=False)

    error_records = calibrated_predictions[
        calibrated_predictions["y_true"] != calibrated_predictions["y_predicted"]
    ].copy()
    error_records["error_direction"] = np.where(
        error_records["y_true"] == 1, "false_negative", "false_positive"
    )
    error_records["failure_code"] = error_records["stressor"].map(
        lambda value: STRESSORS[value]["failure_code"] or "F09"
    )
    error_records["error_type"] = np.where(
        error_records["error_direction"] == "false_positive",
        "Over Blocking",
        error_records["failure_code"].map(FAILURE_TAXONOMY),
    )
    error_records["confidence_error"] = np.where(
        error_records["y_true"] == 1,
        1.0 - error_records["y_probability"],
        error_records["y_probability"],
    )
    error_records.sort_values("confidence_error", ascending=False).to_csv(
        output_dir / "error_records.csv", index=False
    )
    error_analysis = (
        error_records.groupby(
            ["error_direction", "error_type", "workflow", "stressor"], as_index=False
        )
        .agg(errors=("run_id", "size"), mean_confidence_error=("confidence_error", "mean"))
        .sort_values("errors", ascending=False)
    )
    error_analysis.to_csv(output_dir / "error_analysis.csv", index=False)

    sample = test.sample(n=min(1200, len(test)), random_state=seed)
    importance = permutation_importance(
        best_model,
        sample[ALL_FEATURES],
        sample["harmful_label"],
        scoring="average_precision",
        n_repeats=3,
        random_state=seed,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance_frame.to_csv(output_dir / "permutation_importance.csv", index=False)

    ablation = _ablation_experiments(frame)
    ablation.to_csv(output_dir / "ablation_study.csv", index=False)
    unseen = _strict_unseen_stressor(
        frame, factories[best_name], ("memory_poisoning", "permission_overgrant")
    )
    unseen.insert(0, "model", best_name)
    unseen.to_csv(output_dir / "unseen_stressor_generalization.csv", index=False)
    cross_workflow = _cross_workflow_generalization(frame, factories[best_name])
    cross_workflow.insert(0, "model", best_name)
    cross_workflow.to_csv(output_dir / "cross_workflow_generalization.csv", index=False)

    joblib.dump(
        {"model": best_model, "calibrator": calibrator, "threshold": calibrated_threshold},
        model_dir / "best_offline_risk_model.joblib",
    )
    manifest = {
        "dataset_rows": len(frame),
        "unique_tasks": int(frame["task_id"].nunique()),
        "split_rows": frame["split"].value_counts().to_dict(),
        "split_tasks": split_manifest["split"].value_counts().to_dict(),
        "task_overlap_train_test": len(set(train["task_id"]) & set(test["task_id"])),
        "positive_rate": float(frame["harmful_label"].mean()),
        "selection_metric": "validation_pr_auc",
        "best_uncalibrated_model": best_name,
        "calibration_method": (
            f"{calibration_method} selected by ECE on validation-tune task groups after fitting "
            "on disjoint validation-calibration task groups"
        ),
        "threshold_tuning": (
            "maximize F2 subject to over-blocking rate <= 35% on disjoint validation tasks"
        ),
        "bootstrap": "300 task-cluster resamples",
        "unseen_stressors": ["memory_poisoning", "permission_overgrant"],
        "result_scope": "offline classifiers trained on synthetic simulator traces; no LLM results",
    }
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "model_comparison": output_dir / "model_comparison.csv",
        "ablation": output_dir / "ablation_study.csv",
        "unseen_stressor": output_dir / "unseen_stressor_generalization.csv",
        "cross_workflow": output_dir / "cross_workflow_generalization.csv",
        "error_analysis": output_dir / "error_analysis.csv",
        "manifest": manifest_path,
    }
