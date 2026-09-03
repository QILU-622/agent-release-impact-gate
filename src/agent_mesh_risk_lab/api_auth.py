"""API-key authentication with hashed-at-rest principal records."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path

from .action_gateway import AuthorizationError
from .enterprise_schema import Principal


class PrincipalRegistry:
    def __init__(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        self._records = payload.get("principals", [])

    def authenticate(self, api_key: str) -> Principal:
        if not api_key:
            raise AuthorizationError("API key is required")
        supplied = hashlib.sha256(api_key.encode()).hexdigest()
        for record in self._records:
            if hmac.compare_digest(supplied, record["key_sha256"]):
                return Principal(
                    tenant_id=record["tenant_id"],
                    subject_id=record["subject_id"],
                    roles=set(record.get("roles", [])),
                    scopes=set(record.get("scopes", [])),
                )
        raise AuthorizationError("API key is invalid")
