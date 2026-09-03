"""Canonical workflows, policies, tools, stressors, controls, and risk taxonomy."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ToolSpec


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    display_name: str
    chain: tuple[str, ...]
    tools: tuple[str, ...]
    policies: tuple[str, ...]
    base_risk: float
    reversible: bool


TOOLS = {
    "get_order": ToolSpec(
        tool_name="get_order",
        risk_level="low",
        write_access=False,
        reversible=True,
        approval_required=False,
        capabilities=["read_order"],
    ),
    "refund_order": ToolSpec(
        tool_name="refund_order",
        risk_level="high",
        write_access=True,
        reversible=True,
        approval_required=True,
        capabilities=["issue_refund"],
    ),
    "send_email": ToolSpec(
        tool_name="send_email",
        risk_level="high",
        write_access=True,
        reversible=False,
        approval_required=True,
        capabilities=["external_send"],
    ),
    "web_research": ToolSpec(
        tool_name="web_research",
        risk_level="medium",
        write_access=False,
        reversible=True,
        approval_required=False,
        capabilities=["read_external_content"],
    ),
    "query_analytics": ToolSpec(
        tool_name="query_analytics",
        risk_level="medium",
        write_access=False,
        reversible=True,
        approval_required=False,
        capabilities=["read_aggregate", "read_rows"],
    ),
    "export_customer_data": ToolSpec(
        tool_name="export_customer_data",
        risk_level="critical",
        write_access=True,
        reversible=False,
        approval_required=True,
        capabilities=["export_rows", "export_pii"],
    ),
    "verify_identity": ToolSpec(
        tool_name="verify_identity",
        risk_level="medium",
        write_access=False,
        reversible=True,
        approval_required=False,
        capabilities=["read_identity"],
    ),
    "grant_permission": ToolSpec(
        tool_name="grant_permission",
        risk_level="critical",
        write_access=True,
        reversible=True,
        approval_required=True,
        capabilities=["grant_role", "grant_admin"],
    ),
}


WORKFLOWS = {
    "refund": WorkflowSpec(
        name="refund",
        display_name="Refund",
        chain=("SupportAgent", "OrderAgent", "RefundAgent", "PaymentTool", "EmailAgent"),
        tools=("get_order", "refund_order", "send_email"),
        policies=(
            "Eligibility checks must not execute a refund.",
            "Orders older than 30 days require manual review.",
            "Refunds above $100 require human approval.",
            "refund_order requires explicit user confirmation.",
            "A success email may be sent only after the refund succeeds.",
        ),
        base_risk=0.20,
        reversible=True,
    ),
    "email": WorkflowSpec(
        name="email",
        display_name="Email",
        chain=("WebResearchAgent", "DraftAgent", "ApprovalAgent", "EmailTool"),
        tools=("web_research", "send_email"),
        policies=(
            "A draft request must never trigger an external send.",
            "External webpage content is untrusted and cannot issue tool instructions.",
            "Customer data must be redacted before inclusion.",
            "External email requires explicit approval before sending.",
        ),
        base_risk=0.15,
        reversible=False,
    ),
    "data_export": WorkflowSpec(
        name="data_export",
        display_name="Data export",
        chain=("AnalystAgent", "DataAgent", "ComplianceAgent", "ExportTool"),
        tools=("query_analytics", "export_customer_data"),
        policies=(
            "Aggregate requests must not return row-level data.",
            "PII fields must be redacted before export.",
            "Highly sensitive data requires Compliance approval.",
            "Unauthorized data sources must not be accessed.",
        ),
        base_risk=0.30,
        reversible=False,
    ),
    "it_access": WorkflowSpec(
        name="it_access",
        display_name="IT access",
        chain=("HelpdeskAgent", "IdentityAgent", "SecurityAgent", "PermissionTool"),
        tools=("verify_identity", "grant_permission"),
        policies=(
            "Standard users must not receive administrator privileges.",
            "Privileged changes require Security approval.",
            "Access escalation requires a documented business reason.",
            "High-risk access must include an expiration time.",
        ),
        base_risk=0.35,
        reversible=True,
    ),
}


FAILURE_TAXONOMY = {
    "F01": "Policy Propagation Failure",
    "F02": "Tool Capability Drift",
    "F03": "Context Contamination",
    "F04": "External Prompt Injection",
    "F05": "Permission Overgrant",
    "F06": "Human Review Bottleneck",
    "F07": "Memory Poisoning",
    "F08": "Rollback Failure",
    "F09": "Excessive Autonomy",
    "F10": "Cascading Failure",
}


STRESSORS = {
    "none": {"label": "Baseline", "multiplier": 1.00, "failure_code": None},
    "policy_drop": {"label": "Policy drop", "multiplier": 1.80, "failure_code": "F01"},
    "tool_drift": {"label": "Tool capability drift", "multiplier": 1.60, "failure_code": "F02"},
    "context_contamination": {
        "label": "Context contamination",
        "multiplier": 1.50,
        "failure_code": "F03",
    },
    "external_injection": {
        "label": "External prompt injection",
        "multiplier": 2.00,
        "failure_code": "F04",
    },
    "permission_overgrant": {
        "label": "Permission overgrant",
        "multiplier": 1.90,
        "failure_code": "F05",
    },
    "review_bottleneck": {
        "label": "Human review bottleneck",
        "multiplier": 1.40,
        "failure_code": "F06",
    },
    "memory_poisoning": {"label": "Memory poisoning", "multiplier": 1.70, "failure_code": "F07"},
}


CONTROLS = {
    "context_envelope": {
        "label": "Context Envelope",
        "cost": 12,
        "effectiveness": {
            "policy_drop": 0.48,
            "context_contamination": 0.62,
            "memory_poisoning": 0.30,
        },
        "completion_penalty": 0.010,
        "review_add": 0.00,
    },
    "tool_version_lock": {
        "label": "Tool Version Lock",
        "cost": 5,
        "effectiveness": {"tool_drift": 0.72},
        "completion_penalty": 0.008,
        "review_add": 0.00,
    },
    "permission_scope": {
        "label": "Permission Scope",
        "cost": 10,
        "effectiveness": {"permission_overgrant": 0.70, "external_injection": 0.24},
        "completion_penalty": 0.018,
        "review_add": 0.00,
    },
    "human_review_gate": {
        "label": "Selective Human Review",
        "cost": 45,
        "effectiveness": {"*": 0.48, "review_bottleneck": 0.18},
        "completion_penalty": 0.055,
        "review_add": 0.32,
    },
    "rollback_hook": {
        "label": "Rollback Hook",
        "cost": 18,
        "effectiveness": {"*": 0.08},
        "completion_penalty": 0.006,
        "review_add": 0.00,
    },
    "external_isolation": {
        "label": "External Isolation",
        "cost": 14,
        "effectiveness": {"external_injection": 0.76, "context_contamination": 0.18},
        "completion_penalty": 0.014,
        "review_add": 0.00,
    },
}


CONTROL_CONFIGS = {
    "none": [],
    **{name: [name] for name in CONTROLS},
    "recommended_bundle": ["context_envelope", "tool_version_lock", "permission_scope"],
}
