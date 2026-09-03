from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agent_mesh_risk_lab.action_gateway import ActionGateway, InvalidGrantError
from agent_mesh_risk_lab.enterprise_schema import (
    ActionContext,
    ActionRequest,
    ExecutionResultReport,
    GrantConsumption,
    Principal,
)
from agent_mesh_risk_lab.postgres_store import POSTGRES_SCHEMA_SQL, PostgresAuditStore


def test_postgres_schema_has_release_safety_constraints() -> None:
    assert "PRIMARY KEY (tenant_id, request_id)" in POSTGRES_SCHEMA_SQL
    assert "token_digest TEXT NOT NULL UNIQUE" in POSTGRES_SCHEMA_SQL
    assert "CHECK (status IN ('pending', 'approved', 'rejected'))" in POSTGRES_SCHEMA_SQL


@pytest.mark.skipif(
    not os.getenv("AGENT_MESH_TEST_POSTGRES_URL"),
    reason="set AGENT_MESH_TEST_POSTGRES_URL to run the PostgreSQL integration test",
)
def test_postgres_store_runs_gateway_transaction_and_audit_chain() -> None:
    import psycopg

    dsn = os.environ["AGENT_MESH_TEST_POSTGRES_URL"]
    schema = f"agent_mesh_test_{uuid.uuid4().hex}"
    now = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)
    try:
        store = PostgresAuditStore(dsn, schema=schema)
        gateway = ActionGateway(
            store,
            "postgres-integration-signing-secret",
            Path(__file__).parents[1] / "configs" / "enterprise" / "policy.json",
            clock=lambda: now,
        )
        request = ActionRequest(
            request_id="req-postgres-integration-0001",
            tenant_id="tenant-postgres",
            workflow="refund",
            agent_id="refund-agent",
            tool_name="get_order",
            tool_version="1.0",
            arguments={"order_id": "O-100"},
            context=ActionContext(
                correlation_id="postgres-integration",
                user_intent="read",
            ),
            requested_at=now,
        )
        principal = Principal(
            tenant_id="tenant-postgres",
            subject_id="refund-agent",
            roles={"agent"},
            scopes={"orders:read"},
        )

        decision = gateway.evaluate(request, principal)

        assert decision.outcome == "allow"
        replay = gateway.evaluate(request, principal)
        assert replay.idempotent_replay
        executor = Principal(
            tenant_id="tenant-postgres",
            subject_id="order-tool",
            roles={"tool_executor"},
            scopes=set(),
        )
        attempt = GrantConsumption(
            token=decision.grant_token,
            request_id=request.request_id,
            tenant_id=request.tenant_id,
            tool_name=request.tool_name,
            tool_version=request.tool_version,
            arguments=request.arguments,
        )
        assert gateway.consume(attempt, executor).authorized
        with pytest.raises(InvalidGrantError, match="already been consumed"):
            gateway.consume(attempt, executor)
        result = gateway.record_result(
            ExecutionResultReport(
                tenant_id=request.tenant_id,
                request_id=request.request_id,
                tool_name=request.tool_name,
                status="succeeded",
                external_reference="postgres-integration-result",
            ),
            executor,
        )
        assert result.status == "succeeded"
        assert store.verify_integrity("tenant-postgres").valid
        assert store.tenant_snapshot("tenant-postgres")["decisions"][0]["tool_name"] == "get_order"
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
