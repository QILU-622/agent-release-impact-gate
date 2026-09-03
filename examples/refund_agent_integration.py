"""Minimal production-shaped integration: model proposes, gateway authorizes, tool consumes."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

from agent_mesh_risk_lab.enterprise_schema import (
    ActionContext,
    ActionRequest,
    ExecutionResultReport,
    GrantConsumption,
)
from agent_mesh_risk_lab.gateway_client import GatewayClient


def propose_refund(order_id: str, amount: float, user_confirmed: bool) -> ActionRequest:
    """In a real agent, the LLM supplies the proposal fields but never an authorization token."""
    request_id = f"req_{uuid.uuid4().hex}"
    return ActionRequest(
        request_id=request_id,
        tenant_id="acme-demo",
        workflow="refund",
        agent_id="refund-agent",
        tool_name="refund_order",
        tool_version="1.0",
        arguments={"order_id": order_id, "amount": amount},
        context=ActionContext(
            user_confirmed=user_confirmed,
            source_trust="trusted",
            data_classification="confidential",
            user_intent="execute",
            purpose="Customer-requested refund",
            correlation_id=f"support-{request_id}",
        ),
        requested_at=datetime.now(UTC),
    )


def request_authorization(proposal: ActionRequest) -> None:
    api_key = os.environ["AGENT_MESH_AGENT_API_KEY"]
    with GatewayClient("http://127.0.0.1:8080", api_key) as gateway:
        decision = gateway.evaluate(proposal)
    if decision.outcome == "deny":
        print(f"Blocked by policy: {', '.join(decision.reason_codes)}")
    elif decision.outcome == "review":
        print(f"Queued for independent approval: {decision.approval_id}")
    else:
        print("Low-risk action authorized; pass the grant only to the named tool service.")


def tool_side_consumption(proposal: ActionRequest, grant_token: str) -> None:
    """The business tool verifies the exact action before touching production state."""
    tool_key = os.environ["AGENT_MESH_TOOL_API_KEY"]
    attempt = GrantConsumption(
        token=grant_token,
        request_id=proposal.request_id,
        tenant_id=proposal.tenant_id,
        tool_name=proposal.tool_name,
        tool_version=proposal.tool_version,
        arguments=proposal.arguments,
    )
    with GatewayClient("http://127.0.0.1:8080", tool_key) as gateway:
        gateway.consume(attempt)
        # Only after consume succeeds should the service call its payment provider.
        payment_result = {"status": "succeeded", "transaction_id": "provider-reference"}
        gateway.record_result(
            ExecutionResultReport(
                tenant_id=proposal.tenant_id,
                request_id=proposal.request_id,
                tool_name=proposal.tool_name,
                status=payment_result["status"],
                external_reference=payment_result["transaction_id"],
            )
        )


if __name__ == "__main__":
    request_authorization(propose_refund("O-123", 79.0, user_confirmed=True))
