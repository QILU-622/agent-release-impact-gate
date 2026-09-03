"""Validate the minimum evidence contract for an anonymous external Agent pilot."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REQUIRED_COLUMNS = {
    "case_id",
    "workflow",
    "date_bucket",
    "case_type",
    "risk_tier",
    "redacted_input",
    "expected_tool",
    "expected_outcome",
    "reviewer_a",
    "reviewer_b",
    "adjudication_status",
}
ALLOWED_WORKFLOWS = {"refund", "email", "data_export", "it_access"}
ALLOWED_OUTCOMES = {"allow", "deny", "review"}
ALLOWED_RISK = {"low", "medium", "high", "critical"}
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)"),
    "payment_card_like": re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def validate(cases_path: Path, manifest_path: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    manifest = json.loads(manifest_path.read_text())
    with cases_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing_columns:
            errors.append(f"missing columns: {', '.join(missing_columns)}")
        rows = list(reader)

    case_ids = [row.get("case_id", "") for row in rows]
    if not rows:
        errors.append("pilot contains no cases")
    if len(case_ids) != len(set(case_ids)):
        errors.append("case_id values must be unique")
    if len(rows) < 20:
        warnings.append("fewer than 20 cases; evidence is too small for the proposed minimum pilot")

    for index, row in enumerate(rows, start=2):
        if row.get("workflow") not in ALLOWED_WORKFLOWS:
            errors.append(f"row {index}: invalid workflow")
        if row.get("expected_outcome") not in ALLOWED_OUTCOMES:
            errors.append(f"row {index}: invalid expected_outcome")
        if row.get("risk_tier") not in ALLOWED_RISK:
            errors.append(f"row {index}: invalid risk_tier")
        if row.get("reviewer_a") == row.get("reviewer_b"):
            errors.append(f"row {index}: independent reviewers are required")
        if row.get("adjudication_status") != "agreed":
            errors.append(f"row {index}: adjudication is not resolved")
        text = row.get("redacted_input", "")
        for label, pattern in PII_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"row {index}: possible {label} remains in redacted_input")

    if not manifest.get("external_participant"):
        errors.append("manifest does not attest an external participant")
    attestations = manifest.get("customer_attestations", {})
    required_attestations = {
        "real_historical_cases",
        "real_agent_proposals",
        "direct_identifiers_removed",
        "secrets_removed",
        "authorized_for_shadow_evaluation",
    }
    for field in sorted(required_attestations):
        if attestations.get(field) is not True:
            errors.append(f"customer attestation is not true: {field}")
    if not manifest.get("real_agent_build_id"):
        errors.append("real_agent_build_id is required")

    return {
        "ready_for_external_shadow_pilot": not errors and len(rows) >= 20,
        "case_count": len(rows),
        "errors": errors,
        "warnings": warnings,
        "claim_boundary": (
            "Readiness validation only; not proof of anonymization, safety, ROI, or payment demand."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.cases, args.manifest)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if report["ready_for_external_shadow_pilot"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
