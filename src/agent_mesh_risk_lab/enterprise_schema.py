"""Contracts for the enterprise action-authorization control plane."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class Principal(BaseModel):
    """Authenticated machine or human identity supplied by the API layer."""

    tenant_id: str = Field(min_length=1)
    subject_id: str = Field(min_length=1)
    roles: set[str] = Field(default_factory=set)
    scopes: set[str] = Field(default_factory=set)


class ActionContext(BaseModel):
    """Deployment-observable context used by deterministic authorization rules."""

    user_confirmed: bool = False
    source_trust: Literal["trusted", "mixed", "untrusted"] = "trusted"
    data_classification: Literal["public", "internal", "confidential", "restricted"] = "internal"
    user_intent: Literal["read", "draft", "execute", "export", "grant"] = "read"
    purpose: str = Field(default="", max_length=500)
    correlation_id: str = Field(min_length=1, max_length=128)


class ActionRequest(BaseModel):
    """A proposed tool call. The gateway decides; the model never self-authorizes."""

    request_id: str = Field(min_length=8, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    workflow: Literal["refund", "email", "data_export", "it_access"]
    agent_id: str = Field(min_length=1, max_length=128)
    tool_name: str = Field(min_length=1, max_length=128)
    tool_version: str = Field(min_length=1, max_length=32)
    arguments: dict[str, Any] = Field(default_factory=dict)
    context: ActionContext
    requested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def require_timezone(self) -> ActionRequest:
        if self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")
        return self


class PolicyDecision(BaseModel):
    decision_id: str
    request_id: str
    tenant_id: str
    outcome: Literal["allow", "deny", "review"]
    risk_tier: Literal["low", "medium", "high", "critical"]
    reason_codes: list[str]
    obligations: list[str] = Field(default_factory=list)
    policy_version: str
    approval_id: str | None = None
    grant_token: str | None = None
    expires_at: datetime | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    idempotent_replay: bool = False


class ApprovalResolution(BaseModel):
    approved: bool
    reason: str = Field(min_length=3, max_length=500)


class ApprovalRecord(BaseModel):
    approval_id: str
    request_id: str
    tenant_id: str
    status: Literal["pending", "approved", "rejected"]
    required_role: str
    requester_subject: str
    decided_by: str | None = None
    reason: str | None = None
    created_at: datetime
    decided_at: datetime | None = None
    grant_token: str | None = None


class GrantConsumption(BaseModel):
    token: str = Field(min_length=20)
    request_id: str
    tenant_id: str
    tool_name: str
    tool_version: str
    arguments: dict[str, Any]


class ExecutionAuthorization(BaseModel):
    authorized: bool
    request_id: str
    grant_id: str
    tool_name: str
    tenant_id: str
    consumed_at: datetime


class ExecutionResultReport(BaseModel):
    tenant_id: str
    request_id: str
    tool_name: str
    status: Literal["succeeded", "failed", "rolled_back"]
    external_reference: str | None = Field(default=None, max_length=256)
    detail: str = Field(default="", max_length=500)


class ExecutionResultRecord(ExecutionResultReport):
    recorded_by: str
    recorded_at: datetime
    idempotent_replay: bool = False


class AuditIntegrity(BaseModel):
    tenant_id: str
    valid: bool
    events_checked: int
    first_invalid_event_id: int | None = None
