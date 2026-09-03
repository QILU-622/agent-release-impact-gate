"""Deterministic probability simulator with auditable agent traces."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable

from .catalog import CONTROLS, STRESSORS, WORKFLOWS
from .schema import ExperimentRun, TraceStep, WorkflowTask


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _scenario_risk(task: WorkflowTask) -> float:
    scenario = task.scenario
    bump = 0.10 if task.case_type == "risk" else 0.0
    bump += 0.07 if task.human_review_required else 0.0
    bump += 0.08 if scenario.get("includes_pii") else 0.0
    bump += 0.07 if scenario.get("requested_role") == "admin" else 0.0
    bump += 0.05 if scenario.get("confirmed") is False else 0.0
    bump += 0.05 if scenario.get("external_content") else 0.0
    return bump


def _control_reduction(control: str, stressor: str, review_saturated: bool) -> float:
    mapping = CONTROLS[control]["effectiveness"]
    reduction = mapping.get(stressor, mapping.get("*", 0.0))
    if control == "human_review_gate" and review_saturated:
        reduction *= 0.35
    return reduction


def _build_trace(
    task: WorkflowTask,
    stressor: str,
    controls: list[str],
    harmful: bool,
    over_blocked: bool,
    rollback_success: bool,
    review_saturated: bool,
) -> list[TraceStep]:
    steps: list[TraceStep] = []
    stress_label = STRESSORS[stressor]["label"]
    for index, actor in enumerate(task.agent_chain, start=1):
        actor_type = "tool" if actor.endswith("Tool") else "agent"
        action = (
            "tool_call"
            if actor_type == "tool"
            else ("receive_request" if index == 1 else "handoff_and_decide")
        )
        status = "ok"
        detail = "Policy and context checks passed."
        if stressor != "none" and index == max(2, len(task.agent_chain) // 2):
            status = "warning"
            detail = f"Injected stressor observed: {stress_label}."
        if index == len(task.agent_chain):
            if over_blocked:
                status = "blocked"
                detail = "A safe action was blocked by governance friction."
            elif harmful:
                status = "unsafe"
                detail = "Unsafe terminal action executed after upstream safeguards failed."
            else:
                detail = "Terminal action completed within the active policy envelope."
        steps.append(
            TraceStep(
                sequence=index,
                actor=actor,
                actor_type=actor_type,
                action=action,
                status=status,
                detail=detail,
            )
        )

    if "human_review_gate" in controls:
        steps.insert(
            -1,
            TraceStep(
                sequence=len(steps),
                actor="HumanReviewer",
                actor_type="human",
                action="review_high_risk_action",
                status="warning" if review_saturated else "ok",
                detail=(
                    "Review queue saturated; service-level objective missed."
                    if review_saturated
                    else "Selective review completed."
                ),
            ),
        )
    if rollback_success:
        steps.append(
            TraceStep(
                sequence=len(steps) + 1,
                actor="RollbackController",
                actor_type="system",
                action="restore_previous_state",
                status="recovered",
                detail="The previous recoverable state was restored and logged.",
            )
        )
    return [step.model_copy(update={"sequence": i}) for i, step in enumerate(steps, start=1)]


def run_experiment(
    task: WorkflowTask,
    stressor: str = "none",
    controls: Iterable[str] = (),
    global_seed: int = 20260827,
    control_config: str | None = None,
    include_trace: bool = True,
) -> ExperimentRun:
    if stressor not in STRESSORS:
        raise KeyError(f"Unknown stressor: {stressor}")
    controls = sorted(set(controls))
    unknown = set(controls) - set(CONTROLS)
    if unknown:
        raise KeyError(f"Unknown controls: {sorted(unknown)}")

    # Paired random draws are intentionally independent of the control configuration.
    seed = _stable_seed(global_seed, task.task_id, stressor)
    rng = random.Random(seed)
    draws = [rng.random() for _ in range(12)]

    workflow = WORKFLOWS[task.workflow_type]
    base_risk = workflow.base_risk + _scenario_risk(task)
    risk_probability = _clamp(base_risk * STRESSORS[stressor]["multiplier"], 0.01, 0.94)

    review_requested = task.human_review_required or "human_review_gate" in controls
    review_probability = 0.20 + sum(CONTROLS[c]["review_add"] for c in controls)
    if task.human_review_required:
        review_probability += 0.28
    human_review = review_requested and draws[5] < _clamp(review_probability)
    saturation_probability = 0.68 if stressor == "review_bottleneck" else 0.06
    review_saturated = human_review and draws[6] < saturation_probability

    for control in controls:
        risk_probability *= 1 - _control_reduction(control, stressor, review_saturated)
    risk_probability = _clamp(risk_probability, 0.005, 0.98)

    normal_case = task.case_type == "normal" and stressor == "none"
    friction = sum(CONTROLS[c]["completion_penalty"] for c in controls)
    over_block_probability = (0.015 + friction * 1.8) if normal_case else (0.005 + friction * 0.35)
    over_blocked = draws[1] < _clamp(over_block_probability, 0.0, 0.35)
    harmful = draws[0] < risk_probability and not over_blocked
    policy_violation = harmful and draws[2] < 0.88
    incident = harmful

    cascade_probability = 0.16 + 0.055 * max(0, len(task.agent_chain) - 3)
    if stressor in {"context_contamination", "memory_poisoning", "external_injection"}:
        cascade_probability += 0.18
    cascading = harmful and draws[3] < _clamp(cascade_probability)

    rollback_attempted = (
        incident and task.reversible and ("rollback_hook" in controls or draws[4] < 0.16)
    )
    rollback_probability = 0.84 if "rollback_hook" in controls else 0.28
    if cascading:
        rollback_probability -= 0.16
    rollback_success = rollback_attempted and draws[7] < _clamp(rollback_probability)

    # Functional completion is kept distinct from safety: an unsafe action can still
    # complete the user's requested workflow, which is the core governance trade-off.
    completion_probability = 0.95 - (STRESSORS[stressor]["multiplier"] - 1) * 0.04 - friction
    completion_probability -= 0.08 if review_saturated else 0.0
    completion_probability -= 0.03 if harmful else 0.0
    completion_probability = _clamp(completion_probability, 0.20, 0.98)
    task_success = (not over_blocked) and draws[8] < completion_probability

    tool_calls = max(1, len(task.tools_available))
    unsafe_tool_calls = 1 + int(cascading) if harmful else 0
    impact_base = {"refund": 38, "email": 48, "data_export": 72, "it_access": 76}[
        task.workflow_type
    ]
    blast_radius = 0.0
    if incident:
        blast_radius = impact_base * (1.18 if cascading else 0.82)
        blast_radius *= 1.15 if stressor in {"external_injection", "permission_overgrant"} else 1.0
        blast_radius *= 0.42 if rollback_success else 1.0
        blast_radius = round(max(1.0, min(100.0, blast_radius)), 2)

    config_name = control_config or ("+".join(controls) if controls else "none")
    trace = (
        _build_trace(
            task, stressor, controls, harmful, over_blocked, rollback_success, review_saturated
        )
        if include_trace
        else []
    )
    run_id = hashlib.sha1(
        f"{task.task_id}|{stressor}|{config_name}|{global_seed}".encode()
    ).hexdigest()[:14]
    return ExperimentRun(
        run_id=run_id,
        task_id=task.task_id,
        workflow=task.workflow_type,
        case_type=task.case_type,
        stressor=stressor,
        controls=controls,
        control_config=config_name,
        seed=seed,
        task_success=task_success,
        safety_success=not incident,
        policy_violation=policy_violation,
        harmful_action=harmful,
        cascading_failure=cascading,
        unsafe_tool_calls=unsafe_tool_calls,
        tool_calls=tool_calls,
        incident=incident,
        rollback_attempted=rollback_attempted,
        rollback_success=rollback_success,
        human_review=human_review,
        review_saturated=review_saturated,
        over_blocked=over_blocked,
        blast_radius=blast_radius,
        risk_probability=round(risk_probability, 6),
        completion_probability=round(completion_probability, 6),
        trace=trace,
    )
