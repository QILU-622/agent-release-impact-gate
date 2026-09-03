"""Non-bypassable authorization gateway for high-impact agent tool calls.

The model proposes an action. This module makes the authorization decision, creates a
single-use execution grant, and records a tenant-scoped tamper-evident audit trail.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .catalog import TOOLS, WORKFLOWS
from .enterprise_schema import (
    ActionRequest,
    ApprovalRecord,
    ApprovalResolution,
    AuditIntegrity,
    ExecutionAuthorization,
    ExecutionResultRecord,
    ExecutionResultReport,
    GrantConsumption,
    PolicyDecision,
    Principal,
)


class GatewayError(RuntimeError):
    """Base class for errors that API adapters may map to stable status codes."""


class AuthorizationError(GatewayError):
    pass


class ConflictError(GatewayError):
    pass


class NotFoundError(GatewayError):
    pass


class InvalidGrantError(GatewayError):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _canonical(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(value).astimezone(UTC)


class AuditStore:
    """SQLite state store with per-tenant hash-chained audit events."""

    backend = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS action_requests (
                    request_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    requester_subject TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    required_role TEXT NOT NULL,
                    requester_subject TEXT NOT NULL,
                    decided_by TEXT,
                    reason TEXT,
                    created_at TEXT NOT NULL,
                    decided_at TEXT,
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
                    expires_at TEXT NOT NULL,
                    used_at TEXT,
                    FOREIGN KEY (tenant_id, request_id)
                        REFERENCES action_requests (tenant_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS execution_results (
                    tenant_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    recorded_by TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, request_id),
                    FOREIGN KEY (tenant_id, request_id)
                        REFERENCES action_requests (tenant_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_audit_tenant_event
                    ON audit_events (tenant_id, event_id);
                CREATE INDEX IF NOT EXISTS idx_approval_tenant_status
                    ON approvals (tenant_id, status);
                """
            )

    def get_request(self, tenant_id: str, request_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM action_requests WHERE tenant_id = ? AND request_id = ?",
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
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO action_requests
                (request_id, tenant_id, fingerprint, request_json, decision_json,
                 requester_subject, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    request.request_id,
                    request.tenant_id,
                    fingerprint,
                    request.model_dump_json(),
                    decision.model_dump_json(),
                    requester_subject,
                    _iso(decision.decided_at),
                ),
            )
            if approval_role and decision.approval_id:
                connection.execute(
                    """INSERT INTO approvals
                    (approval_id, request_id, tenant_id, status, required_role,
                     requester_subject, created_at) VALUES (?, ?, ?, 'pending', ?, ?, ?)""",
                    (
                        decision.approval_id,
                        request.request_id,
                        request.tenant_id,
                        approval_role,
                        requester_subject,
                        _iso(decision.decided_at),
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
    def _insert_grant(connection: sqlite3.Connection, grant: dict[str, str]) -> None:
        connection.execute(
            """INSERT INTO execution_grants
            (grant_id, request_id, tenant_id, token_digest, action_fingerprint,
             tool_name, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                grant["grant_id"],
                grant["request_id"],
                grant["tenant_id"],
                grant["token_digest"],
                grant["action_fingerprint"],
                grant["tool_name"],
                grant["expires_at"],
            ),
        )

    def get_approval(self, tenant_id: str, approval_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM approvals WHERE tenant_id = ? AND approval_id = ?",
                (tenant_id, approval_id),
            ).fetchone()

    def list_approvals(self, tenant_id: str, status: str | None = None) -> list[sqlite3.Row]:
        with self._connect() as connection:
            if status:
                return connection.execute(
                    """SELECT * FROM approvals WHERE tenant_id = ? AND status = ?
                    ORDER BY created_at""",
                    (tenant_id, status),
                ).fetchall()
            return connection.execute(
                "SELECT * FROM approvals WHERE tenant_id = ? ORDER BY created_at",
                (tenant_id,),
            ).fetchall()

    def resolve_approval(
        self,
        approval_id: str,
        tenant_id: str,
        resolution: ApprovalResolution,
        approver: str,
        grant: dict[str, str] | None,
        decided_at: datetime,
    ) -> None:
        status = "approved" if resolution.approved else "rejected"
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """UPDATE approvals SET status = ?, decided_by = ?, reason = ?, decided_at = ?
                WHERE tenant_id = ? AND approval_id = ? AND status = 'pending'""",
                (
                    status,
                    approver,
                    resolution.reason,
                    _iso(decided_at),
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
        self,
        tenant_id: str,
        token_digest: str,
        consumer: str,
        consumed_at: datetime,
    ) -> sqlite3.Row:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM execution_grants
                WHERE tenant_id = ? AND token_digest = ?""",
                (tenant_id, token_digest),
            ).fetchone()
            if row is None:
                raise InvalidGrantError("execution grant is unknown")
            if row["used_at"] is not None:
                raise InvalidGrantError("execution grant has already been consumed")
            if _parse_time(row["expires_at"]) < consumed_at:
                raise InvalidGrantError("execution grant has expired")
            changed = connection.execute(
                """UPDATE execution_grants SET used_at = ?
                WHERE grant_id = ? AND used_at IS NULL""",
                (_iso(consumed_at), row["grant_id"]),
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

    def get_execution_result(self, tenant_id: str, request_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """SELECT * FROM execution_results
                WHERE tenant_id = ? AND request_id = ?""",
                (tenant_id, request_id),
            ).fetchone()

    def record_execution_result(
        self,
        report: ExecutionResultReport,
        fingerprint: str,
        record: ExecutionResultRecord,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO execution_results
                (tenant_id, request_id, fingerprint, result_json, recorded_by, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    report.tenant_id,
                    report.request_id,
                    fingerprint,
                    record.model_dump_json(),
                    record.recorded_by,
                    _iso(record.recorded_at),
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

    def _append_event(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        event_type: str,
        entity_id: str,
        actor_id: str,
        payload: dict[str, Any],
        created_at: datetime,
    ) -> None:
        previous = connection.execute(
            """SELECT event_hash FROM audit_events
            WHERE tenant_id = ? ORDER BY event_id DESC LIMIT 1""",
            (tenant_id,),
        ).fetchone()
        previous_hash = previous["event_hash"] if previous else "0" * 64
        payload_json = _canonical(payload)
        event_material = {
            "tenant_id": tenant_id,
            "event_type": event_type,
            "entity_id": entity_id,
            "actor_id": actor_id,
            "payload_json": payload_json,
            "previous_hash": previous_hash,
            "created_at": _iso(created_at),
        }
        event_hash = _digest(event_material)
        connection.execute(
            """INSERT INTO audit_events
            (tenant_id, event_type, entity_id, actor_id, payload_json, previous_hash,
             event_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                tenant_id,
                event_type,
                entity_id,
                actor_id,
                payload_json,
                previous_hash,
                event_hash,
                _iso(created_at),
            ),
        )

    def verify_integrity(self, tenant_id: str) -> AuditIntegrity:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events WHERE tenant_id = ? ORDER BY event_id",
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
                "created_at": row["created_at"],
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
                FROM action_requests WHERE tenant_id = ? ORDER BY created_at DESC""",
                (tenant_id,),
            ).fetchall()
            approvals = connection.execute(
                "SELECT * FROM approvals WHERE tenant_id = ? ORDER BY created_at DESC",
                (tenant_id,),
            ).fetchall()
            events = connection.execute(
                "SELECT * FROM audit_events WHERE tenant_id = ? ORDER BY event_id DESC",
                (tenant_id,),
            ).fetchall()
            execution_results = connection.execute(
                """SELECT result_json FROM execution_results
                WHERE tenant_id = ? ORDER BY recorded_at DESC""",
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
                    "created_at": row["created_at"],
                }
            )
        return {
            "decisions": decision_rows,
            "approvals": [dict(row) for row in approvals],
            "events": [dict(row) for row in events],
            "execution_results": [json.loads(row["result_json"]) for row in execution_results],
        }


class ActionGateway:
    """Policy engine, approval workflow, token issuer, and audit coordinator."""

    def __init__(
        self,
        store: AuditStore,
        signing_secret: str,
        policy_path: str | Path,
        clock: Any = _utc_now,
    ) -> None:
        if len(signing_secret) < 24:
            raise ValueError("signing_secret must be at least 24 characters")
        self.store = store
        self.secret = signing_secret.encode()
        self.policy = json.loads(Path(policy_path).read_text())
        self.clock = clock
        self._lock = threading.RLock()

    @property
    def policy_version(self) -> str:
        return str(self.policy["policy_version"])

    @staticmethod
    def _request_payload(request: ActionRequest) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        payload.pop("requested_at", None)
        return payload

    @classmethod
    def request_fingerprint(cls, request: ActionRequest) -> str:
        return _digest(cls._request_payload(request))

    @staticmethod
    def action_fingerprint(
        tenant_id: str, request_id: str, tool_name: str, version: str, arguments: dict[str, Any]
    ) -> str:
        return _digest(
            {
                "tenant_id": tenant_id,
                "request_id": request_id,
                "tool_name": tool_name,
                "tool_version": version,
                "arguments": arguments,
            }
        )

    def evaluate(self, request: ActionRequest, principal: Principal) -> PolicyDecision:
        self._authorize_agent(request, principal)
        fingerprint = self.request_fingerprint(request)
        with self._lock:
            existing = self.store.get_request(request.tenant_id, request.request_id)
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise ConflictError("request_id was reused with a different action payload")
                decision = PolicyDecision.model_validate_json(existing["decision_json"])
                return decision.model_copy(update={"idempotent_replay": True})

            now = self.clock()
            age = (now - request.requested_at.astimezone(UTC)).total_seconds()
            if age > int(self.policy["max_request_age_seconds"]):
                outcome, risk, reasons, obligations, approval_role = (
                    "deny",
                    "high",
                    ["REQUEST_EXPIRED"],
                    [],
                    None,
                )
            elif age < -30:
                outcome, risk, reasons, obligations, approval_role = (
                    "deny",
                    "high",
                    ["REQUEST_FROM_FUTURE"],
                    [],
                    None,
                )
            else:
                outcome, risk, reasons, obligations, approval_role = self._apply_policy(
                    request, principal
                )

            decision_id = f"dec_{uuid.uuid4().hex}"
            approval_id = f"apr_{uuid.uuid4().hex}" if outcome == "review" else None
            grant_token = None
            grant_record = None
            expires_at = None
            if outcome == "allow":
                grant_token, grant_record, expires_at = self._issue_grant(request, now)
            decision = PolicyDecision(
                decision_id=decision_id,
                request_id=request.request_id,
                tenant_id=request.tenant_id,
                outcome=outcome,
                risk_tier=risk,
                reason_codes=reasons,
                obligations=obligations,
                policy_version=self.policy_version,
                approval_id=approval_id,
                grant_token=grant_token,
                expires_at=expires_at,
                decided_at=now,
            )
            try:
                self.store.record_evaluation(
                    request,
                    fingerprint,
                    decision,
                    principal.subject_id,
                    approval_role,
                    grant_record,
                )
            except ConflictError:
                # Another gateway process may have committed the same idempotency key first.
                concurrent = self.store.get_request(request.tenant_id, request.request_id)
                if concurrent and concurrent["fingerprint"] == fingerprint:
                    persisted = PolicyDecision.model_validate_json(concurrent["decision_json"])
                    return persisted.model_copy(update={"idempotent_replay": True})
                raise
            return decision

    def _authorize_agent(self, request: ActionRequest, principal: Principal) -> None:
        if principal.tenant_id != request.tenant_id:
            raise AuthorizationError("cross-tenant action request denied")
        if "agent" not in principal.roles and "agent_proxy" not in principal.roles:
            raise AuthorizationError("principal cannot submit agent actions")
        if principal.subject_id != request.agent_id and "agent_proxy" not in principal.roles:
            raise AuthorizationError("principal cannot impersonate another agent")

    def _apply_policy(
        self, request: ActionRequest, principal: Principal
    ) -> tuple[str, str, list[str], list[str], str | None]:
        tool = TOOLS.get(request.tool_name)
        workflow = WORKFLOWS.get(request.workflow)
        if tool is None:
            return "deny", "critical", ["UNKNOWN_TOOL"], [], None
        if workflow is None or request.tool_name not in workflow.tools:
            return "deny", "critical", ["TOOL_OUTSIDE_WORKFLOW"], [], None
        if request.tool_version != tool.version:
            return "deny", "high", ["TOOL_VERSION_MISMATCH"], [], None

        required_scope = self.policy["tool_scopes"].get(request.tool_name)
        if required_scope and required_scope not in principal.scopes:
            return "deny", tool.risk_level, ["MISSING_MACHINE_SCOPE"], [], None
        if (
            self.policy["deny_untrusted_write_context"]
            and tool.write_access
            and request.context.source_trust != "trusted"
        ):
            return "deny", tool.risk_level, ["UNTRUSTED_CONTEXT_WRITE"], [], None

        reasons = self._argument_denials(request)
        if reasons:
            return "deny", tool.risk_level, reasons, [], None

        obligations = self._obligations(request)
        approval_role = self.policy["approval_roles"].get(request.tool_name)
        if tool.approval_required or approval_role:
            return (
                "review",
                tool.risk_level,
                ["HUMAN_APPROVAL_REQUIRED"],
                obligations,
                approval_role,
            )
        return "allow", tool.risk_level, ["POLICY_CHECKS_PASSED"], obligations, None

    @staticmethod
    def _argument_denials(request: ActionRequest) -> list[str]:
        args = request.arguments
        reasons: list[str] = []
        if request.tool_name == "get_order" and not args.get("order_id"):
            reasons.append("ORDER_ID_REQUIRED")
        elif request.tool_name == "refund_order":
            if not args.get("order_id"):
                reasons.append("ORDER_ID_REQUIRED")
            if not isinstance(args.get("amount"), (int, float)) or args.get("amount", 0) <= 0:
                reasons.append("VALID_REFUND_AMOUNT_REQUIRED")
            if not request.context.user_confirmed:
                reasons.append("EXPLICIT_USER_CONFIRMATION_REQUIRED")
        elif request.tool_name == "send_email":
            if request.context.user_intent == "draft":
                reasons.append("DRAFT_INTENT_CANNOT_SEND")
            if not args.get("to") or not args.get("body"):
                reasons.append("EMAIL_RECIPIENT_AND_BODY_REQUIRED")
            if args.get("contains_unredacted_pii"):
                reasons.append("UNREDACTED_PII_IN_EMAIL")
        elif request.tool_name == "query_analytics":
            if args.get("row_level") and request.context.data_classification == "restricted":
                reasons.append("RESTRICTED_ROW_QUERY_FORBIDDEN")
        elif request.tool_name == "export_customer_data":
            if not args.get("dataset"):
                reasons.append("DATASET_REQUIRED")
            if request.context.user_intent != "export":
                reasons.append("EXPORT_NOT_REQUESTED")
            if args.get("contains_pii") and not args.get("pii_redacted"):
                reasons.append("PII_REDACTION_REQUIRED")
            if args.get("aggregate_only"):
                reasons.append("AGGREGATE_REQUEST_CANNOT_EXPORT_ROWS")
        elif request.tool_name == "verify_identity" and not args.get("user_id"):
            reasons.append("USER_ID_REQUIRED")
        elif request.tool_name == "grant_permission":
            if not args.get("user_id") or not args.get("role"):
                reasons.append("ACCESS_TARGET_AND_ROLE_REQUIRED")
            if not args.get("business_reason"):
                reasons.append("BUSINESS_REASON_REQUIRED")
            expires_at = args.get("expires_at")
            if not expires_at:
                reasons.append("ACCESS_EXPIRATION_REQUIRED")
            else:
                try:
                    if _parse_time(str(expires_at)) <= request.requested_at.astimezone(UTC):
                        reasons.append("ACCESS_EXPIRATION_INVALID")
                except ValueError:
                    reasons.append("ACCESS_EXPIRATION_INVALID")
        return reasons

    @staticmethod
    def _obligations(request: ActionRequest) -> list[str]:
        obligations = ["LOG_TOOL_RESULT", "PRESERVE_CORRELATION_ID"]
        if request.tool_name == "refund_order":
            obligations.append("RECORD_REFUND_TRANSACTION_ID")
            if float(request.arguments.get("amount", 0)) > 100:
                obligations.append("FINANCE_RECONCILIATION")
        elif request.tool_name == "send_email":
            obligations.append("CAPTURE_PROVIDER_MESSAGE_ID")
        elif request.tool_name == "export_customer_data":
            obligations.extend(["ENCRYPT_EXPORT", "DELETE_EXPORT_AFTER_TTL"])
        elif request.tool_name == "grant_permission":
            obligations.extend(["SCHEDULE_ACCESS_REVOCATION", "VERIFY_POST_CHANGE_STATE"])
        return obligations

    def _issue_grant(
        self, request: ActionRequest, now: datetime
    ) -> tuple[str, dict[str, str], datetime]:
        grant_id = f"grt_{uuid.uuid4().hex}"
        expires_at = now + timedelta(seconds=int(self.policy["grant_ttl_seconds"]))
        action_fingerprint = self.action_fingerprint(
            request.tenant_id,
            request.request_id,
            request.tool_name,
            request.tool_version,
            request.arguments,
        )
        claims = {
            "grant_id": grant_id,
            "tenant_id": request.tenant_id,
            "request_id": request.request_id,
            "tool_name": request.tool_name,
            "tool_version": request.tool_version,
            "action_fingerprint": action_fingerprint,
            "exp": int(expires_at.timestamp()),
            "policy_version": self.policy_version,
        }
        encoded = base64.urlsafe_b64encode(_canonical(claims).encode()).decode().rstrip("=")
        signature = hmac.new(self.secret, encoded.encode(), hashlib.sha256).hexdigest()
        token = f"amg1.{encoded}.{signature}"
        record = {
            "grant_id": grant_id,
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "token_digest": hashlib.sha256(token.encode()).hexdigest(),
            "action_fingerprint": action_fingerprint,
            "tool_name": request.tool_name,
            "expires_at": _iso(expires_at),
        }
        return token, record, expires_at

    def resolve_approval(
        self,
        approval_id: str,
        resolution: ApprovalResolution,
        principal: Principal,
    ) -> ApprovalRecord:
        with self._lock:
            row = self.store.get_approval(principal.tenant_id, approval_id)
            if row is None:
                raise NotFoundError("approval not found")
            if row["status"] != "pending":
                raise ConflictError("approval is no longer pending")
            if row["required_role"] not in principal.roles:
                raise AuthorizationError("principal lacks the required approval role")
            if row["requester_subject"] == principal.subject_id:
                raise AuthorizationError("requester cannot approve their own action")
            request_row = self.store.get_request(principal.tenant_id, row["request_id"])
            if request_row is None:
                raise NotFoundError("action request not found")
            request = ActionRequest.model_validate_json(request_row["request_json"])
            now = self.clock()
            token = None
            grant = None
            if resolution.approved:
                token, grant, _ = self._issue_grant(request, now)
            self.store.resolve_approval(
                approval_id,
                principal.tenant_id,
                resolution,
                principal.subject_id,
                grant,
                now,
            )
            return ApprovalRecord(
                approval_id=approval_id,
                request_id=row["request_id"],
                tenant_id=principal.tenant_id,
                status="approved" if resolution.approved else "rejected",
                required_role=row["required_role"],
                requester_subject=row["requester_subject"],
                decided_by=principal.subject_id,
                reason=resolution.reason,
                created_at=_parse_time(row["created_at"]),
                decided_at=now,
                grant_token=token,
            )

    def consume(self, attempt: GrantConsumption, principal: Principal) -> ExecutionAuthorization:
        if principal.tenant_id != attempt.tenant_id:
            raise AuthorizationError("cross-tenant grant consumption denied")
        if "tool_executor" not in principal.roles:
            raise AuthorizationError("principal cannot consume execution grants")
        claims = self._verify_token(attempt.token)
        expected = {
            "tenant_id": attempt.tenant_id,
            "request_id": attempt.request_id,
            "tool_name": attempt.tool_name,
            "tool_version": attempt.tool_version,
        }
        if any(claims.get(key) != value for key, value in expected.items()):
            raise InvalidGrantError("grant claims do not match the attempted tool call")
        action_fingerprint = self.action_fingerprint(
            attempt.tenant_id,
            attempt.request_id,
            attempt.tool_name,
            attempt.tool_version,
            attempt.arguments,
        )
        if not hmac.compare_digest(claims["action_fingerprint"], action_fingerprint):
            raise InvalidGrantError("tool arguments changed after authorization")
        now = self.clock()
        if int(claims["exp"]) < int(now.timestamp()):
            raise InvalidGrantError("execution grant has expired")
        row = self.store.consume_grant(
            attempt.tenant_id,
            hashlib.sha256(attempt.token.encode()).hexdigest(),
            principal.subject_id,
            now,
        )
        return ExecutionAuthorization(
            authorized=True,
            request_id=attempt.request_id,
            grant_id=row["grant_id"],
            tool_name=attempt.tool_name,
            tenant_id=attempt.tenant_id,
            consumed_at=now,
        )

    def record_result(
        self, report: ExecutionResultReport, principal: Principal
    ) -> ExecutionResultRecord:
        if principal.tenant_id != report.tenant_id:
            raise AuthorizationError("cross-tenant execution result denied")
        if "tool_executor" not in principal.roles:
            raise AuthorizationError("principal cannot record execution results")
        request_row = self.store.get_request(report.tenant_id, report.request_id)
        if request_row is None:
            raise NotFoundError("action request not found")
        request = ActionRequest.model_validate_json(request_row["request_json"])
        if request.tool_name != report.tool_name:
            raise AuthorizationError("tool cannot report a result for another tool")
        fingerprint = _digest(report.model_dump(mode="json"))
        with self._lock:
            existing = self.store.get_execution_result(report.tenant_id, report.request_id)
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise ConflictError("execution result was already recorded differently")
                record = ExecutionResultRecord.model_validate_json(existing["result_json"])
                return record.model_copy(update={"idempotent_replay": True})
            record = ExecutionResultRecord(
                **report.model_dump(),
                recorded_by=principal.subject_id,
                recorded_at=self.clock(),
            )
            try:
                self.store.record_execution_result(report, fingerprint, record)
                return record
            except ConflictError:
                concurrent = self.store.get_execution_result(
                    report.tenant_id, report.request_id
                )
                if concurrent and concurrent["fingerprint"] == fingerprint:
                    persisted = ExecutionResultRecord.model_validate_json(
                        concurrent["result_json"]
                    )
                    return persisted.model_copy(update={"idempotent_replay": True})
                raise

    def _verify_token(self, token: str) -> dict[str, Any]:
        try:
            prefix, encoded, signature = token.split(".")
            if prefix != "amg1":
                raise ValueError
            expected = hmac.new(self.secret, encoded.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                raise InvalidGrantError("execution grant signature is invalid")
            padded = encoded + "=" * (-len(encoded) % 4)
            return json.loads(base64.urlsafe_b64decode(padded).decode())
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise InvalidGrantError("execution grant is malformed") from error
