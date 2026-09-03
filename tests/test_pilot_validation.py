from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_pilot_data.py"
SPEC = importlib.util.spec_from_file_location("validate_pilot_data", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_valid_package(tmp_path: Path) -> tuple[Path, Path]:
    cases = tmp_path / "cases.csv"
    with cases.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(MODULE.REQUIRED_COLUMNS))
        writer.writeheader()
        for index in range(20):
            writer.writerow(
                {
                    "case_id": f"case-{index:02d}",
                    "workflow": "refund",
                    "date_bucket": "2026-Q3",
                    "case_type": "normal" if index % 2 == 0 else "risk",
                    "risk_tier": "low" if index % 2 == 0 else "high",
                    "redacted_input": f"Check synthetic order ORDER-{index:03d}",
                    "expected_tool": "get_order",
                    "expected_outcome": "allow",
                    "reviewer_a": "operations-reviewer",
                    "reviewer_b": "security-reviewer",
                    "adjudication_status": "agreed",
                }
            )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "external_participant": True,
                "real_agent_build_id": "customer-build-hash",
                "customer_attestations": {
                    "real_historical_cases": True,
                    "real_agent_proposals": True,
                    "direct_identifiers_removed": True,
                    "secrets_removed": True,
                    "authorized_for_shadow_evaluation": True,
                },
            }
        )
    )
    return cases, manifest


def test_valid_external_package_is_ready(tmp_path: Path) -> None:
    cases, manifest = _write_valid_package(tmp_path)
    report = MODULE.validate(cases, manifest)
    assert report["ready_for_external_shadow_pilot"]
    assert report["case_count"] == 20


def test_pii_and_false_attestation_block_pilot_claim(tmp_path: Path) -> None:
    cases, manifest = _write_valid_package(tmp_path)
    rows = list(csv.DictReader(cases.open()))
    rows[0]["redacted_input"] = "Email customer@example.com"
    with cases.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    payload = json.loads(manifest.read_text())
    payload["external_participant"] = False
    manifest.write_text(json.dumps(payload))

    report = MODULE.validate(cases, manifest)

    assert not report["ready_for_external_shadow_pilot"]
    assert any("possible email" in error for error in report["errors"])
    assert any("external participant" in error for error in report["errors"])
