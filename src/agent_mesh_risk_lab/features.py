"""Leakage-safe feature construction for offline risk-classification experiments."""

from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd

from .catalog import CONTROLS, STRESSORS, TOOLS, WORKFLOWS
from .graph import graph_summary
from .schema import WorkflowTask

FORBIDDEN_MODEL_COLUMNS = {
    "blast_radius",
    "cascading_failure",
    "completion_probability",
    "harmful_action",
    "human_review",
    "incident",
    "over_blocked",
    "policy_violation",
    "review_saturated",
    "risk_probability",
    "rollback_attempted",
    "rollback_success",
    "safety_success",
    "task_success",
    "tool_calls",
    "unsafe_tool_calls",
}


NUMERIC_FEATURES = [
    "agent_count",
    "tool_count",
    "policy_count",
    "delegation_depth",
    "graph_nodes",
    "graph_edges",
    "graph_density",
    "high_risk_tool_count",
    "critical_tool_count",
    "write_tool_count",
    "irreversible_tool_count",
    "approval_tool_count",
    "workflow_base_risk",
    "risk_level_ordinal",
    "case_is_risk",
    "task_reversible",
    "human_review_required",
    "amount_scaled",
    "age_over_30",
    "confirmed",
    "eligibility_only",
    "includes_customer_data",
    "external_content_present",
    "approved",
    "aggregate_only",
    "includes_pii",
    "authorized",
    "admin_requested",
    "has_business_reason",
    "has_expiration",
    "identity_verified",
    "policy_integrity",
    "tool_contract_integrity",
    "context_integrity",
    "external_untrusted",
    "permission_excess",
    "review_capacity",
    "memory_integrity",
    "stressor_intensity",
    "control_count",
    "control_cost",
    "control_expected_coverage",
    "control_completion_penalty",
    "control_review_add",
    *[f"control__{name}" for name in CONTROLS],
]

CATEGORICAL_FEATURES = [
    "case_type",
    "risk_level",
    "expected_action",
]

TEXT_FEATURES = ["user_text", "policy_text"]

FEATURE_GROUPS = {
    "policy": ["policy_count", "policy_integrity", "human_review_required", "policy_text"],
    "tool": [
        "tool_count",
        "high_risk_tool_count",
        "critical_tool_count",
        "write_tool_count",
        "irreversible_tool_count",
        "approval_tool_count",
        "tool_contract_integrity",
    ],
    "graph": [
        "agent_count",
        "delegation_depth",
        "graph_nodes",
        "graph_edges",
        "graph_density",
    ],
    "context": [
        "context_integrity",
        "confirmed",
        "eligibility_only",
        "approved",
        "authorized",
        "has_business_reason",
        "has_expiration",
        "identity_verified",
    ],
    "text": TEXT_FEATURES,
}


def _parse_controls(value: object) -> list[str]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item for item in str(value).split(",") if item]


def _mechanism_features(stressor: str) -> dict[str, float]:
    return {
        "policy_integrity": 0.0 if stressor == "policy_drop" else 1.0,
        "tool_contract_integrity": 0.0 if stressor == "tool_drift" else 1.0,
        "context_integrity": 0.25 if stressor == "context_contamination" else 1.0,
        "external_untrusted": 1.0 if stressor == "external_injection" else 0.0,
        "permission_excess": 1.0 if stressor == "permission_overgrant" else 0.0,
        "review_capacity": 0.20 if stressor == "review_bottleneck" else 1.0,
        "memory_integrity": 0.0 if stressor == "memory_poisoning" else 1.0,
        "stressor_intensity": float(STRESSORS[stressor]["multiplier"]) - 1.0,
    }


def _control_features(controls: Iterable[str], stressor: str) -> dict[str, float]:
    selected = sorted(set(controls))
    residual = 1.0
    for control in selected:
        effectiveness = CONTROLS[control]["effectiveness"]
        residual *= 1.0 - float(effectiveness.get(stressor, effectiveness.get("*", 0.0)))
    return {
        "control_count": float(len(selected)),
        "control_cost": float(sum(CONTROLS[name]["cost"] for name in selected)),
        "control_expected_coverage": 1.0 - residual,
        "control_completion_penalty": float(
            sum(CONTROLS[name]["completion_penalty"] for name in selected)
        ),
        "control_review_add": float(sum(CONTROLS[name]["review_add"] for name in selected)),
        **{f"control__{name}": float(name in selected) for name in CONTROLS},
    }


def task_features(task: WorkflowTask) -> dict[str, object]:
    workflow = WORKFLOWS[task.workflow_type]
    graph = graph_summary(task.workflow_type)
    tools = [TOOLS[name] for name in task.tools_available]
    scenario = task.scenario
    risk_ordinal = {"low": 0, "medium": 1, "high": 2, "critical": 3}[task.risk_level]
    return {
        "task_id": task.task_id,
        "workflow": task.workflow_type,
        "case_type": task.case_type,
        "risk_level": task.risk_level,
        "expected_action": task.expected_action,
        "agent_count": float(len(task.agent_chain)),
        "tool_count": float(len(tools)),
        "policy_count": float(len(task.policies)),
        "delegation_depth": float(max(0, len(task.agent_chain) - 1)),
        "graph_nodes": float(graph["nodes"]),
        "graph_edges": float(graph["edges"]),
        "graph_density": float(graph["density"]),
        "high_risk_tool_count": float(
            sum(tool.risk_level in {"high", "critical"} for tool in tools)
        ),
        "critical_tool_count": float(sum(tool.risk_level == "critical" for tool in tools)),
        "write_tool_count": float(sum(tool.write_access for tool in tools)),
        "irreversible_tool_count": float(sum(not tool.reversible for tool in tools)),
        "approval_tool_count": float(sum(tool.approval_required for tool in tools)),
        "workflow_base_risk": float(workflow.base_risk),
        "risk_level_ordinal": float(risk_ordinal),
        "case_is_risk": float(task.case_type == "risk"),
        "task_reversible": float(task.reversible),
        "human_review_required": float(task.human_review_required),
        "amount_scaled": float(scenario.get("amount", 0)) / 500.0,
        "age_over_30": float(float(scenario.get("age_days", 0)) > 30),
        "confirmed": float(bool(scenario.get("confirmed", False))),
        "eligibility_only": float(bool(scenario.get("eligibility_only", False))),
        "includes_customer_data": float(bool(scenario.get("includes_customer_data", False))),
        "external_content_present": float(bool(scenario.get("external_content", False))),
        "approved": float(bool(scenario.get("approved", False))),
        "aggregate_only": float(bool(scenario.get("aggregate_only", False))),
        "includes_pii": float(bool(scenario.get("includes_pii", False))),
        "authorized": float(bool(scenario.get("authorized", False))),
        "admin_requested": float(scenario.get("requested_role") == "admin"),
        "has_business_reason": float(bool(scenario.get("has_reason", False))),
        "has_expiration": float(bool(scenario.get("has_expiration", False))),
        "identity_verified": float(bool(scenario.get("identity_verified", False))),
        "user_text": task.user_request,
        "policy_text": " ".join(task.policies),
    }


def assign_task_splits(tasks: list[WorkflowTask], seed: int = 20260827) -> dict[str, str]:
    """Stratified group split: no task_id can occur in more than one partition."""
    import random

    grouped: dict[tuple[str, str], list[str]] = {}
    for task in tasks:
        grouped.setdefault((task.workflow_type, task.case_type), []).append(task.task_id)
    split_map: dict[str, str] = {}
    for offset, (stratum, task_ids) in enumerate(sorted(grouped.items())):
        local = sorted(task_ids)
        random.Random(seed + offset).shuffle(local)
        train_end = int(len(local) * 0.68)
        validation_end = train_end + int(len(local) * 0.16)
        for task_id in local[:train_end]:
            split_map[task_id] = "train"
        for task_id in local[train_end:validation_end]:
            split_map[task_id] = "validation"
        for task_id in local[validation_end:]:
            split_map[task_id] = "test"
        if not stratum:
            raise AssertionError("Empty split stratum")
    return split_map


def build_feature_frame(
    tasks: list[WorkflowTask], results: pd.DataFrame, seed: int = 20260827
) -> pd.DataFrame:
    task_lookup = {task.task_id: task_features(task) for task in tasks}
    split_map = assign_task_splits(tasks, seed=seed)
    rows: list[dict[str, object]] = []
    for run in results.to_dict("records"):
        task_id = str(run["task_id"])
        stressor = str(run["stressor"])
        controls = _parse_controls(run.get("controls"))
        row = {
            **task_lookup[task_id],
            "run_id": run["run_id"],
            "stressor": stressor,
            "control_config": run["control_config"],
            "split": split_map[task_id],
            **_mechanism_features(stressor),
            **_control_features(controls, stressor),
            "harmful_label": int(bool(run["harmful_action"])),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    model_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES)
    leakage = model_columns & FORBIDDEN_MODEL_COLUMNS
    if leakage:
        raise AssertionError(f"Outcome leakage in model features: {sorted(leakage)}")
    if frame[NUMERIC_FEATURES].isna().any().any():
        missing = frame[NUMERIC_FEATURES].columns[frame[NUMERIC_FEATURES].isna().any()].tolist()
        raise AssertionError(f"Missing numeric model features: {missing}")
    return frame


def feature_manifest() -> dict[str, object]:
    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "text_features": TEXT_FEATURES,
        "feature_groups": FEATURE_GROUPS,
        "forbidden_outcome_columns": sorted(FORBIDDEN_MODEL_COLUMNS),
        "leakage_policy": (
            "Models may use only pre-action task, graph, stressor, tool, policy, and control inputs. "
            "Simulator probabilities and all post-action outcomes are forbidden."
        ),
    }


def write_feature_manifest(path: object) -> None:
    from pathlib import Path

    Path(path).write_text(json.dumps(feature_manifest(), indent=2), encoding="utf-8")
