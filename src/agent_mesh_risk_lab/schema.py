"""Validated records shared by benchmark generation, simulation, and evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high", "critical"]


class ToolSpec(BaseModel):
    tool_name: str
    risk_level: RiskLevel
    write_access: bool
    reversible: bool
    approval_required: bool
    version: str = "1.0"
    capabilities: list[str] = Field(default_factory=list)


class WorkflowTask(BaseModel):
    task_id: str
    workflow_type: str
    case_type: Literal["normal", "risk"]
    user_request: str
    agent_chain: list[str]
    tools_available: list[str]
    policies: list[str]
    expected_action: str
    risk_level: RiskLevel
    reversible: bool
    human_review_required: bool
    risk_label: str
    failure_type: str | None = None
    root_cause: str | None = None
    scenario: dict[str, Any] = Field(default_factory=dict)


class TraceStep(BaseModel):
    sequence: int
    actor: str
    actor_type: Literal["agent", "tool", "human", "system"]
    action: str
    status: Literal["ok", "warning", "blocked", "unsafe", "recovered"]
    detail: str


class ExperimentRun(BaseModel):
    run_id: str
    task_id: str
    workflow: str
    case_type: str
    stressor: str
    controls: list[str]
    control_config: str
    seed: int
    task_success: bool
    safety_success: bool
    policy_violation: bool
    harmful_action: bool
    cascading_failure: bool
    unsafe_tool_calls: int
    tool_calls: int
    incident: bool
    rollback_attempted: bool
    rollback_success: bool
    human_review: bool
    review_saturated: bool
    over_blocked: bool
    blast_radius: float = Field(ge=0, le=100)
    risk_probability: float = Field(ge=0, le=1)
    completion_probability: float = Field(ge=0, le=1)
    trace: list[TraceStep]
