from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_mesh_risk_lab.action_gateway import (
    ActionGateway,
    AuditStore,
    AuthorizationError,
    ConflictError,
    InvalidGrantError,
)
from agent_mesh_risk_lab.enterprise_schema import (
    ActionContext,
    ActionRequest,
    ApprovalResolution,
    ExecutionResultReport,
    GrantConsumption,
    Principal,
)

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


@pytest.fixture
def gateway(tmp_path: Path) -> ActionGateway:
    root = Path(__file__).parents[1]
    return ActionGateway(
        AuditStore(tmp_path / "gateway.sqlite3"),
        "test-signing-secret-with-safe-length",
        root / "configs" / "enterprise" / "policy.json",
        clock=lambda: NOW,
    )


def principal(
    subject: str, roles: set[str], scopes: set[str] | None = None, tenant: str = "tenant-a"
) -> Principal:
    return Principal(
        tenant_id=tenant,
        subject_id=subject,
        roles=roles,
        scopes=scopes or set(),
    )


def request(
    request_id: str = "req-00000001",
    tool: str = "get_order",
    workflow: str = "refund",
    arguments: dict | None = None,
    **context: object,
) -> ActionRequest:
    return ActionRequest(
        request_id=request_id,
        tenant_id="tenant-a",
        workflow=workflow,
        agent_id="refund-agent",
        tool_name=tool,
        tool_version="1.0",
        arguments=arguments or {"order_id": "O-100"},
        context=ActionContext(correlation_id=f"corr-{request_id}", **context),
        requested_at=NOW,
    )


def test_low_risk_read_is_allowed_and_grant_is_single_use(gateway: ActionGateway) -> None:
    action = request()
    actor = principal("refund-agent", {"agent"}, {"orders:read"})
    decision = gateway.evaluate(action, actor)
    assert decision.outcome == "allow"
    assert decision.grant_token

    attempt = GrantConsumption(
        token=decision.grant_token,
        request_id=action.request_id,
        tenant_id=action.tenant_id,
        tool_name=action.tool_name,
        tool_version=action.tool_version,
        arguments=action.arguments,
    )
    executor = principal("order-tool", {"tool_executor"})
    assert gateway.consume(attempt, executor).authorized
    with pytest.raises(InvalidGrantError, match="already been consumed"):
        gateway.consume(attempt, executor)


def test_high_risk_refund_requires_separate_human_approval(gateway: ActionGateway) -> None:
    action = request(
        request_id="req-00000002",
        tool="refund_order",
        arguments={"order_id": "O-100", "amount": 150.0},
        user_confirmed=True,
        user_intent="execute",
    )
    requester = principal("refund-agent", {"agent"}, {"refunds:write"})
    decision = gateway.evaluate(action, requester)
    assert decision.outcome == "review"
    assert decision.approval_id
    assert decision.grant_token is None

    with pytest.raises(AuthorizationError, match="own action"):
        gateway.resolve_approval(
            decision.approval_id,
            ApprovalResolution(approved=True, reason="Approved within refund policy"),
            principal("refund-agent", {"finance_approver"}),
        )

    result = gateway.resolve_approval(
        decision.approval_id,
        ApprovalResolution(approved=True, reason="Order and amount verified"),
        principal("finance-reviewer", {"finance_approver"}),
    )
    assert result.status == "approved"
    assert result.grant_token


def test_untrusted_write_and_unredacted_pii_fail_closed(gateway: ActionGateway) -> None:
    actor = principal("refund-agent", {"agent"}, {"email:send", "data:export"})
    untrusted = request(
        request_id="req-00000003",
        tool="send_email",
        workflow="email",
        arguments={"to": "customer@example.com", "body": "Hello"},
        user_intent="execute",
        source_trust="untrusted",
    )
    assert gateway.evaluate(untrusted, actor).reason_codes == ["UNTRUSTED_CONTEXT_WRITE"]

    export = request(
        request_id="req-00000004",
        tool="export_customer_data",
        workflow="data_export",
        arguments={"dataset": "customers", "contains_pii": True, "pii_redacted": False},
        user_intent="export",
        data_classification="restricted",
    )
    decision = gateway.evaluate(export, actor)
    assert decision.outcome == "deny"
    assert "PII_REDACTION_REQUIRED" in decision.reason_codes


def test_machine_scope_tenant_and_version_are_enforced(gateway: ActionGateway) -> None:
    action = request(request_id="req-00000005")
    decision = gateway.evaluate(action, principal("refund-agent", {"agent"}))
    assert decision.reason_codes == ["MISSING_MACHINE_SCOPE"]

    with pytest.raises(AuthorizationError, match="cross-tenant"):
        gateway.evaluate(
            request(request_id="req-00000006"),
            principal("refund-agent", {"agent"}, {"orders:read"}, tenant="tenant-b"),
        )

    versioned = request(request_id="req-00000007").model_copy(update={"tool_version": "2.0"})
    assert gateway.evaluate(
        versioned, principal("refund-agent", {"agent"}, {"orders:read"})
    ).reason_codes == ["TOOL_VERSION_MISMATCH"]


def test_idempotency_replays_same_decision_and_rejects_key_reuse(gateway: ActionGateway) -> None:
    actor = principal("refund-agent", {"agent"}, {"orders:read"})
    original = request(request_id="req-00000008")
    first = gateway.evaluate(original, actor)
    second = gateway.evaluate(original, actor)
    assert second.decision_id == first.decision_id
    assert second.idempotent_replay

    changed = original.model_copy(update={"arguments": {"order_id": "O-CHANGED"}})
    with pytest.raises(ConflictError, match="different action payload"):
        gateway.evaluate(changed, actor)


def test_authorized_arguments_cannot_change_at_execution(gateway: ActionGateway) -> None:
    actor = principal("refund-agent", {"agent"}, {"orders:read"})
    action = request(request_id="req-00000009")
    decision = gateway.evaluate(action, actor)
    attempt = GrantConsumption(
        token=decision.grant_token,
        request_id=action.request_id,
        tenant_id=action.tenant_id,
        tool_name=action.tool_name,
        tool_version=action.tool_version,
        arguments={"order_id": "O-ATTACKER-CHANGED"},
    )
    with pytest.raises(InvalidGrantError, match="arguments changed"):
        gateway.consume(attempt, principal("order-tool", {"tool_executor"}))


def test_stale_requests_are_denied(gateway: ActionGateway) -> None:
    stale = request(request_id="req-00000010").model_copy(
        update={"requested_at": NOW - timedelta(minutes=10)}
    )
    decision = gateway.evaluate(stale, principal("refund-agent", {"agent"}, {"orders:read"}))
    assert decision.outcome == "deny"
    assert decision.reason_codes == ["REQUEST_EXPIRED"]


def test_hash_chain_detects_tampering(gateway: ActionGateway) -> None:
    gateway.evaluate(
        request(request_id="req-00000011"),
        principal("refund-agent", {"agent"}, {"orders:read"}),
    )
    assert gateway.store.verify_integrity("tenant-a").valid
    with sqlite3.connect(gateway.store.path) as connection:
        connection.execute(
            "UPDATE audit_events SET payload_json = ? WHERE tenant_id = ?",
            ('{"outcome":"allow"}', "tenant-a"),
        )
    result = gateway.store.verify_integrity("tenant-a")
    assert not result.valid
    assert result.first_invalid_event_id is not None


def test_tool_records_idempotent_execution_result(gateway: ActionGateway) -> None:
    actor = principal("refund-agent", {"agent"}, {"orders:read"})
    action = request(request_id="req-00000012")
    decision = gateway.evaluate(action, actor)
    executor = principal("order-tool", {"tool_executor"})
    gateway.consume(
        GrantConsumption(
            token=decision.grant_token,
            request_id=action.request_id,
            tenant_id=action.tenant_id,
            tool_name=action.tool_name,
            tool_version=action.tool_version,
            arguments=action.arguments,
        ),
        executor,
    )
    report = ExecutionResultReport(
        tenant_id=action.tenant_id,
        request_id=action.request_id,
        tool_name=action.tool_name,
        status="succeeded",
        external_reference="order-provider-123",
    )
    first = gateway.record_result(report, executor)
    second = gateway.record_result(report, executor)
    assert first.status == "succeeded"
    assert second.idempotent_replay

    changed = report.model_copy(update={"status": "failed"})
    with pytest.raises(ConflictError, match="already recorded differently"):
        gateway.record_result(changed, executor)
