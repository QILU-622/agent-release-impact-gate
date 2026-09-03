"""Small Python integration client for agent runtimes and business tools."""

from __future__ import annotations

from typing import Any, Self

import httpx

from .enterprise_schema import (
    ActionRequest,
    ExecutionResultReport,
    GrantConsumption,
    PolicyDecision,
)


class GatewayClient:
    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 5.0) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key},
            timeout=timeout_seconds,
        )

    def evaluate(self, request: ActionRequest) -> PolicyDecision:
        response = self._client.post("/v1/actions/evaluate", json=request.model_dump(mode="json"))
        response.raise_for_status()
        return PolicyDecision.model_validate(response.json())

    def approve(self, approval_id: str, approved: bool, reason: str) -> dict[str, Any]:
        response = self._client.post(
            f"/v1/approvals/{approval_id}/decision",
            json={"approved": approved, "reason": reason},
        )
        response.raise_for_status()
        return response.json()

    def consume(self, attempt: GrantConsumption) -> dict[str, Any]:
        response = self._client.post("/v1/grants/consume", json=attempt.model_dump(mode="json"))
        response.raise_for_status()
        return response.json()

    def record_result(self, report: ExecutionResultReport) -> dict[str, Any]:
        response = self._client.post("/v1/executions/result", json=report.model_dump(mode="json"))
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
