"""PostgreSQL persistence for the Action Gateway.

The module imports psycopg lazily so the local SQLite research workflow remains dependency-light.
PostgreSQL transactions, row locks, and tenant-scoped advisory locks replace process-local locks.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any

from .action_gateway import ConflictError, InvalidGrantError, _digest, _iso, _parse_time
from .enterprise_schema import (
    ActionRequest,
    ApprovalResolution,
    AuditIntegrity,
    ExecutionResultRecord,
    ExecutionResultReport,
    PolicyDecision,
)

POSTGRES_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS action_requests (
    request_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    request_json TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    requester_subject TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, request_id)
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected')),
    required_role TEXT NOT NULL,
    requester_subject TEXT NOT NULL,
    decided_by TEXT,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    decided_at TIMESTAMPTZ,
    FOREIGN KEY (tenant_id, request_id)
        REFERENCES action_requests (tenant_id, request_id)
);
CREATE TABLE IF NOT EXISTS execution_grants (
    grant_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    token_digest TEXT NOT NULL UNIQUE,
    action_fingerprint TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    used_at TIMESTAMPTZ,
    FOREIGN KEY (tenant_id, request_id)
        REFERENCES action_requests (tenant_id, request_id)
);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE TABLE IF NOT EXISTS execution_results (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    result_json TEXT NOT NULL,
    recorded_by TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (tenant_id, request_id),
    FOREIGN KEY (tenant_id, request_id)
        REFERENCES action_requests (tenant_id, request_id)
);
CREATE INDEX IF NOT EXISTS idx_audit_tenant_event
    ON audit_events (tenant_id, event_id);
CREATE INDEX IF NOT EXISTS idx_approval_tenant_status
    ON approvals (tenant_id, status);
"""


def _load_driver() -> tuple[Any, Any]:
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PostgreSQL support requires: pip install 'agent-mesh-risk-lab[postgres]'"
        ) from exc
    return psycopg, dict_row


class PostgresAuditStore:
    """Transactional, multi-process-safe Action Gateway store."""

    backend = "postgresql"

    def __init__(self, dsn: str, schema: str = "public") -> None:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
            raise ValueError("PostgreSQL schema must be a simple SQL identifier")
        self.dsn = dsn
        self.schema = schema
        self._initialize()

    def _connect(self) -> Any:
        psycopg, dict_row = _load_driver()
        connection = psycopg.connect(self.dsn, row_factory=dict_row)
        connection.execute(f'SET search_path TO "{self.schema}"')
        return connection

    def _initialize(self) -> None:
        psycopg, dict_row = _load_driver()
        with psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True) as connection:
            connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
            connection.execute(f'SET search_path TO "{self.schema}"')
            # Execute one DDL statement at a time. This works with psycopg's extended
            # query protocol as well as managed PostgreSQL proxies that reject
            # multi-statement requests.
            for statement in POSTGRES_SCHEMA_SQL.split(";"):
                if statement.strip():
                    connection.execute(statement)

    @staticmethod
    @contextmanager
    def _translate_unique_conflict() -> Any:
        psycopg, _dict_row = _load_driver()
        try:
            yield
        except psycopg.errors.UniqueViolation as exc:
            raise ConflictError("concurrent idempotency conflict") from exc

    def get_request(self, tenant_id: str, request_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM action_requests WHERE tenant_id = %s AND request_id = %s",
                (tenant_id, request_id),
            ).fetchone()

    def record_evaluation(
        self,
        request: ActionRequest,
        fingerprint: str,
        decision: PolicyDecision,
        requester_subject: str,
        approval_role: str | None,
        grant: dict[str, str] | None,
    ) -> None:
        with self._translate_unique_conflict(), self._connect() as connection:
            connection.execute(
                """INSERT INTO action_requests
                (request_id, tenant_id, fingerprint, request_json, decision_json,
                 requester_subject, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (
                    request.request_id,
                    request.tenant_id,
                    fingerprint,
                    request.model_dump_json(),
                    decision.model_dump_json(),
                    requester_subject,
                    decision.decided_at,
                ),
            )
            if approval_role and decision.approval_id:
                connection.execute(
                    """INSERT INTO approvals
                    (approval_id, request_id, tenant_id, status, required_role,
                     requester_subject, created_at) VALUES (%s, %s, %s, 'pending', %s, %s, %s)""",
                    (
                        decision.approval_id,
                        request.request_id,
                        request.tenant_id,
                        approval_role,
                        requester_subject,
                        decision.decided_at,
                    ),
                )
            if grant:
                self._insert_grant(connection, grant)
            self._append_event(
                connection,
                request.tenant_id,
                "action_evaluated",
                request.request_id,
                requester_subject,
                {
                    "outcome": decision.outcome,
                    "reason_codes": decision.reason_codes,
                    "policy_version": decision.policy_version,
                    "tool_name": request.tool_name,
                    "workflow": request.workflow,
                },
                decision.decided_at,
            )

    @staticmethod
    def _insert_grant(connection: Any, grant: dict[str, str]) -> None:
        connection.execute(
            """INSERT INTO execution_grants
            (grant_id, request_id, tenant_id, token_digest, action_fingerprint,
             tool_name, expires_at) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                grant["grant_id"],
                grant["request_id"],
                grant["tenant_id"],
                grant["token_digest"],
                grant["action_fingerprint"],
                grant["tool_name"],
                _parse_time(grant["expires_at"]),
            ),
        )

    def get_approval(self, tenant_id: str, approval_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM approvals WHERE tenant_id = %s AND approval_id = %s",
                (tenant_id, approval_id),
            ).fetchone()

    def list_approvals(self, tenant_id: str, status: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if status:
                return connection.execute(
                    """SELECT * FROM approvals WHERE tenant_id = %s AND status = %s
                    ORDER BY created_at""",
                    (tenant_id, status),
                ).fetchall()
            return connection.execute(
                "SELECT * FROM approvals WHERE tenant_id = %s ORDER BY created_at",
                (tenant_id,),
            ).fetchall()

    def resolve_approval(
        self,
        approval_id: str,
        tenant_id: str,
        resolution: ApprovalResolution,
        approver: str,
        grant: dict[str, str] | None,
        decided_at: Any,
    ) -> None:
        status = "approved" if resolution.approved else "rejected"
        with self._connect() as connection:
            changed = connection.execute(
                """UPDATE approvals SET status = %s, decided_by = %s, reason = %s, decided_at = %s
                WHERE tenant_id = %s AND approval_id = %s AND status = 'pending'""",
                (
                    status,
                    approver,
                    resolution.reason,
                    decided_at,
                    tenant_id,
                    approval_id,
                ),
            ).rowcount
            if changed != 1:
                raise ConflictError("approval is no longer pending")
            if grant:
                self._insert_grant(connection, grant)
            self._append_event(
                connection,
                tenant_id,
                f"approval_{status}",
                approval_id,
                approver,
                {"reason": resolution.reason, "grant_issued": grant is not None},
                decided_at,
            )

    def consume_grant(
        self, tenant_id: str, token_digest: str, consumer: str, consumed_at: Any
    ) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT * FROM execution_grants
                WHERE tenant_id = %s AND token_digest = %s FOR UPDATE""",
                (tenant_id, token_digest),
            ).fetchone()
            if row is None:
                raise InvalidGrantError("execution grant is unknown")
            if row["used_at"] is not None:
                raise InvalidGrantError("execution grant has already been consumed")
            if row["expires_at"] < consumed_at:
                raise InvalidGrantError("execution grant has expired")
            changed = connection.execute(
                """UPDATE execution_grants SET used_at = %s
                WHERE grant_id = %s AND used_at IS NULL""",
                (consumed_at, row["grant_id"]),
            ).rowcount
            if changed != 1:
                raise InvalidGrantError("execution grant replay detected")
            self._append_event(
                connection,
                tenant_id,
                "grant_consumed",
                row["grant_id"],
                consumer,
                {"request_id": row["request_id"], "tool_name": row["tool_name"]},
                consumed_at,
            )
            return row

    def get_execution_result(self, tenant_id: str, request_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            return connection.execute(
                """SELECT * FROM execution_results
                WHERE tenant_id = %s AND request_id = %s""",
                (tenant_id, request_id),
            ).fetchone()

    def record_execution_result(
        self,
        report: ExecutionResultReport,
        fingerprint: str,
        record: ExecutionResultRecord,
    ) -> None:
        with self._translate_unique_conflict(), self._connect() as connection:
            connection.execute(
                """INSERT INTO execution_results
                (tenant_id, request_id, fingerprint, result_json, recorded_by, recorded_at)
                VALUES (%s, %s, %s, %s, %s, %s)""",
                (
                    report.tenant_id,
                    report.request_id,
                    fingerprint,
                    record.model_dump_json(),
                    record.recorded_by,
                    record.recorded_at,
                ),
            )
            self._append_event(
                connection,
                report.tenant_id,
                "execution_result_recorded",
                report.request_id,
                record.recorded_by,
                {
                    "tool_name": report.tool_name,
                    "status": report.status,
                    "external_reference": report.external_reference,
                },
                record.recorded_at,
            )

    @staticmethod
    def _append_event(
        connection: Any,
        tenant_id: str,
        event_type: str,
        entity_id: str,
        actor_id: str,
        payload: dict[str, Any],
        created_at: Any,
    ) -> None:
        connection.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (tenant_id,))
        previous = connection.execute(
            """SELECT event_hash FROM audit_events
            WHERE tenant_id = %s ORDER BY event_id DESC LIMIT 1""",
            (tenant_id,),
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else "0" * 64
        payload_json = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        event_material = {
            "tenant_id": tenant_id,
            "event_type": event_type,
            "entity_id": entity_id,
            "actor_id": actor_id,
            "payload_json": payload_json,
            "previous_hash": previous_hash,
            "created_at": _iso(created_at),
        }
        connection.execute(
            """INSERT INTO audit_events
            (tenant_id, event_type, entity_id, actor_id, payload_json, previous_hash,
             event_hash, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                tenant_id,
                event_type,
                entity_id,
                actor_id,
                payload_json,
                previous_hash,
                _digest(event_material),
                created_at,
            ),
        )

    def verify_integrity(self, tenant_id: str) -> AuditIntegrity:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE tenant_id = %s ORDER BY event_id",
                (tenant_id,),
            ).fetchall()
        previous_hash = "0" * 64
        for index, row in enumerate(rows, start=1):
            material = {
                "tenant_id": row["tenant_id"],
                "event_type": row["event_type"],
                "entity_id": row["entity_id"],
                "actor_id": row["actor_id"],
                "payload_json": row["payload_json"],
                "previous_hash": row["previous_hash"],
                "created_at": _iso(row["created_at"]),
            }
            if row["previous_hash"] != previous_hash or row["event_hash"] != _digest(material):
                return AuditIntegrity(
                    tenant_id=tenant_id,
                    valid=False,
                    events_checked=index,
                    first_invalid_event_id=row["event_id"],
                )
            previous_hash = row["event_hash"]
        return AuditIntegrity(tenant_id=tenant_id, valid=True, events_checked=len(rows))

    def tenant_snapshot(self, tenant_id: str) -> dict[str, list[dict[str, Any]]]:
        with self._connect() as connection:
            decisions = connection.execute(
                """SELECT request_id, request_json, decision_json, requester_subject, created_at
                FROM action_requests WHERE tenant_id = %s ORDER BY created_at DESC""",
                (tenant_id,),
            ).fetchall()
            approvals = connection.execute(
                "SELECT * FROM approvals WHERE tenant_id = %s ORDER BY created_at DESC",
                (tenant_id,),
            ).fetchall()
            events = connection.execute(
                "SELECT * FROM audit_events WHERE tenant_id = %s ORDER BY event_id DESC",
                (tenant_id,),
            ).fetchall()
            execution_results = connection.execute(
                """SELECT result_json FROM execution_results
                WHERE tenant_id = %s ORDER BY recorded_at DESC""",
                (tenant_id,),
            ).fetchall()
        decision_rows = []
        for row in decisions:
            request = json.loads(row["request_json"])
            decision = json.loads(row["decision_json"])
            decision_rows.append(
                {
                    "request_id": row["request_id"],
                    "workflow": request["workflow"],
                    "tool_name": request["tool_name"],
                    "agent_id": request["agent_id"],
                    "outcome": decision["outcome"],
                    "risk_tier": decision["risk_tier"],
                    "reason_codes": decision["reason_codes"],
                    "created_at": _iso(row["created_at"]),
                }
            )
        return {
            "decisions": decision_rows,
            "approvals": [dict(row) for row in approvals],
            "events": [dict(row) for row in events],
            "execution_results": [json.loads(row["result_json"]) for row in execution_results],
        }


def postgres_store_from_env(database_url: str, schema: str = "public") -> PostgresAuditStore:
    if not database_url.startswith(("postgresql://", "postgres://")):
        raise ValueError("AGENT_MESH_DATABASE_URL must be a PostgreSQL URL")
    return PostgresAuditStore(database_url, schema=schema)
