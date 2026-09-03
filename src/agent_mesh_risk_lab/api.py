"""FastAPI adapter for the Agent Mesh Action Gateway."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse

from .action_gateway import (
    ActionGateway,
    AuditStore,
    AuthorizationError,
    ConflictError,
    GatewayError,
    InvalidGrantError,
    NotFoundError,
)
from .api_auth import PrincipalRegistry
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app(
    gateway: ActionGateway | None = None,
    registry: PrincipalRegistry | None = None,
) -> FastAPI:
    demo_mode = gateway is None or registry is None
    if gateway is None:
        secret = os.getenv(
            "AGENT_MESH_SIGNING_SECRET",
            "local-demo-signing-secret-change-before-production",
        )
        database_url = os.getenv("AGENT_MESH_DATABASE_URL")
        if database_url:
            from .postgres_store import postgres_store_from_env

            store = postgres_store_from_env(
                database_url,
                schema=os.getenv("AGENT_MESH_POSTGRES_SCHEMA", "public"),
            )
        else:
            database = Path(
                os.getenv(
                    "AGENT_MESH_DB",
                    str(PROJECT_ROOT / "data" / "enterprise" / "gateway.sqlite3"),
                )
            )
            store = AuditStore(database)
        gateway = ActionGateway(
            store,
            secret,
            Path(
                os.getenv(
                    "AGENT_MESH_POLICY",
                    str(PROJECT_ROOT / "configs" / "enterprise" / "policy.json"),
                )
            ),
        )
    if registry is None:
        registry = PrincipalRegistry(
            os.getenv(
                "AGENT_MESH_PRINCIPALS",
                str(PROJECT_ROOT / "configs" / "enterprise" / "principals.example.json"),
            )
        )

    app = FastAPI(
        title="Agent Mesh Action Gateway",
        version="1.0.0",
        description=(
            "Deterministic authorization, approval, single-use execution grants, and "
            "tamper-evident audit for agent tool calls."
        ),
    )
    app.state.gateway = gateway
    app.state.registry = registry
    app.state.demo_mode = demo_mode

    def principal(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> Principal:
        try:
            return app.state.registry.authenticate(x_api_key or "")
        except AuthorizationError as error:
            raise HTTPException(status_code=401, detail=str(error)) from error

    principal_dependency = Depends(principal)
    snapshot_limit = Query(default=200, ge=1, le=1000)

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(_request: object, error: GatewayError) -> JSONResponse:
        if isinstance(error, AuthorizationError):
            status = 403
        elif isinstance(error, NotFoundError):
            status = 404
        elif isinstance(error, ConflictError):
            status = 409
        elif isinstance(error, InvalidGrantError):
            status = 422
        else:
            status = 400
        return JSONResponse(status_code=status, content={"detail": str(error)})

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "policy_version": app.state.gateway.policy_version,
            "storage_backend": app.state.gateway.store.backend,
            "demo_mode": app.state.demo_mode,
        }

    @app.post("/v1/actions/evaluate", response_model=PolicyDecision)
    def evaluate_action(
        request: ActionRequest,
        actor: Principal = principal_dependency,
    ) -> PolicyDecision:
        return app.state.gateway.evaluate(request, actor)

    @app.get("/v1/actions/{request_id}", response_model=PolicyDecision)
    def get_action(
        request_id: str,
        actor: Principal = principal_dependency,
    ) -> PolicyDecision:
        row = app.state.gateway.store.get_request(actor.tenant_id, request_id)
        if row is None:
            raise NotFoundError("action request not found")
        return PolicyDecision.model_validate_json(row["decision_json"])

    @app.post("/v1/approvals/{approval_id}/decision", response_model=ApprovalRecord)
    def resolve_approval(
        approval_id: str,
        resolution: ApprovalResolution,
        actor: Principal = principal_dependency,
    ) -> ApprovalRecord:
        return app.state.gateway.resolve_approval(approval_id, resolution, actor)

    @app.get("/v1/approvals", response_model=list[ApprovalRecord])
    def list_approvals(
        actor: Principal = principal_dependency,
        status: str = Query(default="pending", pattern="^(pending|approved|rejected)$"),
    ) -> list[ApprovalRecord]:
        rows = app.state.gateway.store.list_approvals(actor.tenant_id, status)
        visible = []
        for row in rows:
            if row["required_role"] not in actor.roles and "auditor" not in actor.roles:
                continue
            visible.append(
                ApprovalRecord(
                    approval_id=row["approval_id"],
                    request_id=row["request_id"],
                    tenant_id=row["tenant_id"],
                    status=row["status"],
                    required_role=row["required_role"],
                    requester_subject=row["requester_subject"],
                    decided_by=row["decided_by"],
                    reason=row["reason"],
                    created_at=row["created_at"],
                    decided_at=row["decided_at"],
                )
            )
        return visible

    @app.post("/v1/grants/consume", response_model=ExecutionAuthorization)
    def consume_grant(
        attempt: GrantConsumption,
        actor: Principal = principal_dependency,
    ) -> ExecutionAuthorization:
        return app.state.gateway.consume(attempt, actor)

    @app.post("/v1/executions/result", response_model=ExecutionResultRecord)
    def record_execution_result(
        report: ExecutionResultReport,
        actor: Principal = principal_dependency,
    ) -> ExecutionResultRecord:
        return app.state.gateway.record_result(report, actor)

    @app.get("/v1/audit/integrity", response_model=AuditIntegrity)
    def audit_integrity(
        actor: Principal = principal_dependency,
    ) -> AuditIntegrity:
        if "auditor" not in actor.roles:
            raise AuthorizationError("principal lacks auditor role")
        return app.state.gateway.store.verify_integrity(actor.tenant_id)

    @app.get("/v1/operations/snapshot")
    def operations_snapshot(
        actor: Principal = principal_dependency,
        limit: int = snapshot_limit,
    ) -> dict:
        if not ({"auditor", "operations"} & actor.roles):
            raise AuthorizationError("principal lacks operations or auditor role")
        snapshot = app.state.gateway.store.tenant_snapshot(actor.tenant_id)
        return {key: rows[:limit] for key, rows in snapshot.items()}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("agent_mesh_risk_lab.api:app", host="127.0.0.1", port=8080, reload=False)
