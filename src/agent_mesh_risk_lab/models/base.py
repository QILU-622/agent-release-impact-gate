"""Provider-neutral contracts for observable real-LLM decision experiments."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from ..schema import WorkflowTask

ActionName = Literal[
    "check_refund_eligibility",
    "request_human_approval",
    "refund_order",
    "create_draft",
    "create_redacted_draft",
    "send_email",
    "export_aggregate",
    "compliance_review",
    "export_approved_rows",
    "export_customer_data",
    "security_review",
    "grant_scoped_access",
    "grant_permission",
    "request_clarification",
    "refuse",
]


class AgentDecision(BaseModel):
    """Auditable decision output; never stores hidden chain of thought."""

    action: ActionName
    tool_name: str | None = None
    handoff_to: str | None = None
    policy_checks: list[str] = Field(default_factory=list, max_length=5)
    risk_flags: list[str] = Field(default_factory=list, max_length=5)
    structured_rationale: str = Field(max_length=240)
    confidence: float = Field(
        ge=0,
        le=100,
        description="Confidence as either a 0-1 proportion or a 0-100 percentage.",
    )

    @field_validator("confidence")
    @classmethod
    def normalize_confidence(cls, value: float) -> float:
        """Normalize the two common model confidence scales to a stored 0-1 value."""
        return value / 100 if value > 1 else value


class AgentObservation(BaseModel):
    """One model decision plus timing and token telemetry returned by the provider."""

    decision: AgentDecision
    latency_ms: float = Field(ge=0)
    total_duration_ms: float | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    prompt_sha256: str
    raw_response_sha256: str


class AgentModel(Protocol):
    model_name: str

    def decide(
        self,
        task: WorkflowTask,
        active_policies: list[str],
        available_tools: list[str],
        handoff_context: dict,
    ) -> AgentObservation:
        """Return an auditable observation; hidden reasoning is never requested or stored."""
