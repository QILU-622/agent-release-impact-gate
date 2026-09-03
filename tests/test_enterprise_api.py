from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from agent_mesh_risk_lab.action_gateway import ActionGateway, AuditStore
from agent_mesh_risk_lab.api import create_app
from agent_mesh_risk_lab.api_auth import PrincipalRegistry

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


def build_client(tmp_path: Path) -> TestClient:
    root = Path(__file__).parents[1]
    registry_path = tmp_path / "principals.json"
    principals = []
    for key, subject, roles, scopes in [
        ("agent-key", "refund-agent", ["agent"], ["orders:read", "refunds:write"]),
        ("approver-key", "reviewer", ["finance_approver", "auditor"], []),
        ("tool-key", "order-tool", ["tool_executor"], []),
    ]:
        principals.append(
            {
                "key_sha256": hashlib.sha256(key.encode()).hexdigest(),
                "tenant_id": "tenant-a",
                "subject_id": subject,
                "roles": roles,
                "scopes": scopes,
            }
        )
    registry_path.write_text(json.dumps({"principals": principals}))
    gateway = ActionGateway(
        AuditStore(tmp_path / "api.sqlite3"),
        "test-api-signing-secret-long-enough",
        root / "configs" / "enterprise" / "policy.json",
        clock=lambda: NOW,
    )
    return TestClient(create_app(gateway, PrincipalRegistry(registry_path)))


def action_payload(request_id: str, tool: str = "get_order") -> dict:
    arguments = {"order_id": "O-100"}
    context = {
        "correlation_id": f"corr-{request_id}",
        "user_confirmed": False,
        "source_trust": "trusted",
        "data_classification": "internal",
        "user_intent": "read",
        "purpose": "Customer support",
    }
    if tool == "refund_order":
        arguments["amount"] = 80
        context.update({"user_confirmed": True, "user_intent": "execute"})
    return {
        "request_id": request_id,
        "tenant_id": "tenant-a",
        "workflow": "refund",
        "agent_id": "refund-agent",
        "tool_name": tool,
        "tool_version": "1.0",
        "arguments": arguments,
        "context": context,
        "requested_at": NOW.isoformat(),
    }


def test_api_requires_auth_and_exposes_health(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/v1/actions/evaluate", json=action_payload("req-api-0001"))
    assert response.status_code == 401


def test_api_end_to_end_approval_and_audit(tmp_path: Path) -> None:
    client = build_client(tmp_path)
    agent_headers = {"X-API-Key": "agent-key"}
    decision = client.post(
        "/v1/actions/evaluate",
        json=action_payload("req-api-0002", "refund_order"),
        headers=agent_headers,
    )
    assert decision.status_code == 200
    body = decision.json()
    assert body["outcome"] == "review"

    queue = client.get("/v1/approvals", headers={"X-API-Key": "approver-key"})
    assert queue.status_code == 200
    assert queue.json()[0]["approval_id"] == body["approval_id"]

    approval = client.post(
        f"/v1/approvals/{body['approval_id']}/decision",
        json={"approved": True, "reason": "Order and user confirmation verified"},
        headers={"X-API-Key": "approver-key"},
    )
    assert approval.status_code == 200
    grant = approval.json()["grant_token"]

    consume = client.post(
        "/v1/grants/consume",
        headers={"X-API-Key": "tool-key"},
        json={
            "token": grant,
            "request_id": "req-api-0002",
            "tenant_id": "tenant-a",
            "tool_name": "refund_order",
            "tool_version": "1.0",
            "arguments": {"order_id": "O-100", "amount": 80},
        },
    )
    assert consume.status_code == 200
    assert consume.json()["authorized"]

    result = client.post(
        "/v1/executions/result",
        headers={"X-API-Key": "tool-key"},
        json={
            "tenant_id": "tenant-a",
            "request_id": "req-api-0002",
            "tool_name": "refund_order",
            "status": "succeeded",
            "external_reference": "payment-123",
            "detail": "Provider accepted refund",
        },
    )
    assert result.status_code == 200
    assert result.json()["status"] == "succeeded"

    replay = client.post(
        "/v1/grants/consume",
        headers={"X-API-Key": "tool-key"},
        json={
            "token": grant,
            "request_id": "req-api-0002",
            "tenant_id": "tenant-a",
            "tool_name": "refund_order",
            "tool_version": "1.0",
            "arguments": {"order_id": "O-100", "amount": 80},
        },
    )
    assert replay.status_code == 422

    integrity = client.get("/v1/audit/integrity", headers={"X-API-Key": "approver-key"})
    assert integrity.json()["valid"]
    assert integrity.json()["events_checked"] == 4
