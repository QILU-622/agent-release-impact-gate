"""Create a deterministic, explicitly synthetic operations dataset for the gateway UI."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent_mesh_risk_lab.action_gateway import ActionGateway, AuditStore
from agent_mesh_risk_lab.enterprise_schema import (
    ActionContext,
    ActionRequest,
    ApprovalResolution,
    ExecutionResultReport,
    GrantConsumption,
    Principal,
)


class DemoClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 8, 28, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value


def principal(subject: str, roles: set[str], scopes: set[str] | None = None) -> Principal:
    return Principal(
        tenant_id="acme-demo",
        subject_id=subject,
        roles=roles,
        scopes=scopes or set(),
    )


def action(
    number: int,
    clock: DemoClock,
    agent: str,
    workflow: str,
    tool: str,
    arguments: dict,
    scope: str,
    **context: object,
) -> tuple[ActionRequest, Principal]:
    return (
        ActionRequest(
            request_id=f"demo-request-{number:03d}",
            tenant_id="acme-demo",
            workflow=workflow,
            agent_id=agent,
            tool_name=tool,
            tool_version="1.0",
            arguments=arguments,
            context=ActionContext(correlation_id=f"case-{number:03d}", **context),
            requested_at=clock.current,
        ),
        principal(agent, {"agent"}, {scope}),
    )


def consume(gateway: ActionGateway, request: ActionRequest, token: str, tool_id: str) -> None:
    executor = principal(tool_id, {"tool_executor"})
    gateway.consume(
        GrantConsumption(
            token=token,
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            tool_name=request.tool_name,
            tool_version=request.tool_version,
            arguments=request.arguments,
        ),
        executor,
    )
    gateway.record_result(
        ExecutionResultReport(
            tenant_id=request.tenant_id,
            request_id=request.request_id,
            tool_name=request.tool_name,
            status="succeeded",
            external_reference=f"synthetic-{request.request_id}",
            detail="Synthetic demo tool result",
        ),
        executor,
    )


def main() -> None:
    output = ROOT / "data" / "enterprise"
    output.mkdir(parents=True, exist_ok=True)
    database = output / "gateway_demo.sqlite3"
    if database.exists():
        database.unlink()
    clock = DemoClock()
    gateway = ActionGateway(
        AuditStore(database),
        "synthetic-demo-secret-never-use-in-production",
        ROOT / "configs" / "enterprise" / "policy.json",
        clock=clock,
    )

    cases = [
        action(
            1, clock, "support-agent", "refund", "get_order", {"order_id": "O-100"}, "orders:read"
        ),
        action(
            2,
            clock,
            "refund-agent",
            "refund",
            "refund_order",
            {"order_id": "O-101", "amount": 79.0},
            "refunds:write",
            user_confirmed=True,
            user_intent="execute",
            purpose="Customer confirmed refund",
        ),
        action(
            3,
            clock,
            "refund-agent",
            "refund",
            "refund_order",
            {"order_id": "O-102", "amount": 240.0},
            "refunds:write",
            user_confirmed=False,
            user_intent="execute",
        ),
        action(
            4,
            clock,
            "email-agent",
            "email",
            "send_email",
            {"to": "customer@example.com", "body": "Draft only"},
            "email:send",
            user_intent="draft",
        ),
        action(
            5,
            clock,
            "email-agent",
            "email",
            "send_email",
            {"to": "outside@example.com", "body": "External instructions"},
            "email:send",
            user_intent="execute",
            source_trust="untrusted",
        ),
        action(
            6,
            clock,
            "data-agent",
            "data_export",
            "export_customer_data",
            {"dataset": "customers", "contains_pii": True, "pii_redacted": False},
            "data:export",
            user_intent="export",
            data_classification="restricted",
        ),
        action(
            7,
            clock,
            "data-agent",
            "data_export",
            "export_customer_data",
            {"dataset": "customers", "contains_pii": True, "pii_redacted": True},
            "data:export",
            user_intent="export",
            data_classification="restricted",
            purpose="Regulatory subject access request",
        ),
        action(
            8,
            clock,
            "identity-agent",
            "it_access",
            "verify_identity",
            {"user_id": "U-200"},
            "identity:read",
        ),
        action(
            9,
            clock,
            "security-agent",
            "it_access",
            "grant_permission",
            {"user_id": "U-201", "role": "admin", "business_reason": "Incident response"},
            "permissions:write",
            user_intent="grant",
        ),
        action(
            10,
            clock,
            "security-agent",
            "it_access",
            "grant_permission",
            {
                "user_id": "U-202",
                "role": "admin",
                "business_reason": "Quarter-end deployment",
                "expires_at": "2026-08-29T09:00:00+00:00",
            },
            "permissions:write",
            user_intent="grant",
        ),
    ]

    decisions: dict[int, tuple[ActionRequest, object]] = {}
    for number, (request, actor) in enumerate(cases, start=1):
        decision = gateway.evaluate(request, actor)
        decisions[number] = (request, decision)
        if decision.outcome == "allow" and decision.grant_token:
            consume(gateway, request, decision.grant_token, f"{request.tool_name}-service")

    approval_specs = [
        (
            2,
            True,
            "Refund amount and explicit user confirmation verified",
            "finance-reviewer",
            "finance_approver",
        ),
        (
            7,
            False,
            "Export purpose valid but requested field set exceeds minimum necessary data",
            "compliance-reviewer",
            "compliance_approver",
        ),
        (
            10,
            True,
            "Time-bounded admin access approved for named deployment",
            "security-reviewer",
            "security_approver",
        ),
    ]
    for number, approved, reason, reviewer, role in approval_specs:
        request, decision = decisions[number]
        result = gateway.resolve_approval(
            decision.approval_id,
            ApprovalResolution(approved=approved, reason=reason),
            principal(reviewer, {role, "auditor"}),
        )
        if result.grant_token:
            consume(gateway, request, result.grant_token, f"{request.tool_name}-service")

    snapshot = gateway.store.tenant_snapshot("acme-demo")
    integrity = gateway.store.verify_integrity("acme-demo")
    (output / "operations_snapshot.json").write_text(
        json.dumps(
            {**snapshot, "integrity": integrity.model_dump(mode="json")},
            indent=2,
            ensure_ascii=False,
        )
    )
    pd.DataFrame(snapshot["decisions"]).to_csv(output / "decisions.csv", index=False)
    pd.DataFrame(snapshot["approvals"]).to_csv(output / "approvals.csv", index=False)
    pd.DataFrame(snapshot["events"]).to_csv(output / "audit_events.csv", index=False)
    print(
        f"Created synthetic enterprise demo: {len(snapshot['decisions'])} decisions, "
        f"{len(snapshot['approvals'])} approvals, {len(snapshot['events'])} audit events."
    )


if __name__ == "__main__":
    main()
