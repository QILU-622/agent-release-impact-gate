"""Evidence-bound release gate for comparing two Agent regression builds.

The gate deliberately authorizes at most a canary.  It converts case-level
contract results into an operational impact for a declared 1,000-transaction
profile; it does not infer production safety from an offline test suite.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ALLOWED_STAGES = ("BLOCK", "OFFLINE_ONLY", "SHADOW", "CANARY")
_OUTCOMES = {"allow", "review", "deny"}
_CRITICALITIES = {"standard", "high", "critical"}
_EVIDENCE_STAGE_CEILINGS = {
    "synthetic_demo": "OFFLINE_ONLY",
    "external_replay": "SHADOW",
    "validated_shadow_pilot": "CANARY",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUILD_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_BUILD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,127}$")


class GateConfigurationError(ValueError):
    """The evidence cannot be compared without making an unsafe assumption."""


def _expect_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateConfigurationError(f"{location} must be a JSON object")
    return value


def _expect_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateConfigurationError(f"{location} must be a non-empty string")
    return value


def _validate_digest(value: Any, location: str) -> str:
    digest = _expect_string(value, location)
    if not _SHA256.fullmatch(digest):
        raise GateConfigurationError(f"{location} must be a lowercase SHA-256 digest")
    return digest


def _validate_build_id(value: Any, location: str) -> str:
    build_id = _expect_string(value, location)
    if not _BUILD_ID.fullmatch(build_id):
        raise GateConfigurationError(
            f"{location} must be 1-128 safe build-identity characters"
        )
    return build_id


def _validate_build_digest(value: Any, location: str) -> str:
    digest = _expect_string(value, location)
    if not _BUILD_DIGEST.fullmatch(digest):
        raise GateConfigurationError(
            f"{location} must match sha256:<64 lowercase hexadecimal characters>"
        )
    return digest


def _validate_report(report: Any, label: str) -> dict[str, Any]:
    payload = _expect_mapping(report, label)
    if payload.get("schema_version") != "1.0":
        raise GateConfigurationError(f"{label}.schema_version must equal 1.0")

    _expect_string(payload.get("suite"), f"{label}.suite")
    _expect_string(payload.get("policy_version"), f"{label}.policy_version")
    _validate_digest(payload.get("suite_sha256"), f"{label}.suite_sha256")
    _validate_digest(payload.get("policy_sha256"), f"{label}.policy_sha256")
    _validate_build_id(payload.get("agent_build_id"), f"{label}.agent_build_id")
    _validate_build_digest(
        payload.get("agent_build_digest"), f"{label}.agent_build_digest"
    )
    if payload.get("trusted_context_enforced") is not True:
        raise GateConfigurationError(f"{label}.trusted_context_enforced must be true")

    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or not raw_results:
        raise GateConfigurationError(f"{label}.results must be a non-empty array")

    seen: set[str] = set()
    calculated_passed = 0
    for index, raw_result in enumerate(raw_results):
        location = f"{label}.results[{index}]"
        result = _expect_mapping(raw_result, location)
        case_id = _expect_string(result.get("id"), f"{location}.id")
        if case_id in seen:
            raise GateConfigurationError(f"{label} contains duplicate case id {case_id}")
        seen.add(case_id)
        if not isinstance(result.get("passed"), bool):
            raise GateConfigurationError(f"{location}.passed must be boolean")
        calculated_passed += int(result["passed"])
        mismatches = result.get("mismatches")
        if not isinstance(mismatches, list) or not all(
            isinstance(item, str) and item for item in mismatches
        ):
            raise GateConfigurationError(f"{location}.mismatches must be an array of strings")
        if result["passed"] != (len(mismatches) == 0):
            raise GateConfigurationError(
                f"{location}.passed is inconsistent with its mismatches"
            )

        expected = _expect_mapping(result.get("expected"), f"{location}.expected")
        if expected.get("outcome") not in _OUTCOMES:
            raise GateConfigurationError(
                f"{location}.expected.outcome must be allow, review, or deny"
            )

        proposal = result.get("proposal")
        if proposal is not None:
            proposal = _expect_mapping(proposal, f"{location}.proposal")
            _validate_digest(
                proposal.get("behavior_fingerprint"),
                f"{location}.proposal.behavior_fingerprint",
            )

        actual = result.get("actual")
        if actual is not None:
            actual = _expect_mapping(actual, f"{location}.actual")
            if actual.get("outcome") not in _OUTCOMES:
                raise GateConfigurationError(
                    f"{location}.actual.outcome must be allow, review, or deny"
                )
            if not isinstance(actual.get("grant_issued"), bool):
                raise GateConfigurationError(f"{location}.actual.grant_issued must be boolean")
            if actual.get("policy_version") != payload["policy_version"]:
                raise GateConfigurationError(
                    f"{location}.actual.policy_version differs from report policy_version"
                )
            grant_expected = actual["outcome"] == "allow"
            if actual["grant_issued"] != grant_expected:
                raise GateConfigurationError(
                    f"{location}.actual grant state is inconsistent with its outcome"
                )
        elif result["passed"]:
            raise GateConfigurationError(f"{location} cannot pass without an actual decision")

    summary = _expect_mapping(payload.get("summary"), f"{label}.summary")
    total = len(raw_results)
    calculated_failed = total - calculated_passed
    expected_summary = {
        "total": total,
        "passed": calculated_passed,
        "failed": calculated_failed,
        "release_gate": "pass" if calculated_failed == 0 else "fail",
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise GateConfigurationError(
                f"{label}.summary.{field} is inconsistent with case results"
            )
    expected_pass_rate = round(calculated_passed / total, 4)
    if summary.get("pass_rate") != expected_pass_rate:
        raise GateConfigurationError(
            f"{label}.summary.pass_rate is inconsistent with case results"
        )
    return payload


def _read_limit(value: Any, location: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateConfigurationError(f"{location} must be numeric")
    number = float(value)
    if not 0 <= number <= maximum:
        raise GateConfigurationError(f"{location} must be between 0 and {maximum:g}")
    return number


def _validate_config(config: Any, case_ids: set[str]) -> dict[str, Any]:
    payload = _expect_mapping(config, "config")
    if payload.get("schema_version") != "1.0":
        raise GateConfigurationError("config.schema_version must equal 1.0")
    if payload.get("release_evidence_stage") not in _EVIDENCE_STAGE_CEILINGS:
        raise GateConfigurationError(
            "config.release_evidence_stage must be synthetic_demo, external_replay, "
            "or validated_shadow_pilot"
        )

    evidence = _expect_mapping(payload.get("evidence_contract"), "config.evidence_contract")
    _expect_string(evidence.get("suite"), "config.evidence_contract.suite")
    _validate_digest(
        evidence.get("suite_sha256"), "config.evidence_contract.suite_sha256"
    )
    _validate_digest(
        evidence.get("policy_sha256"), "config.evidence_contract.policy_sha256"
    )
    _validate_build_id(
        evidence.get("baseline_build_id"), "config.evidence_contract.baseline_build_id"
    )
    _validate_build_id(
        evidence.get("candidate_build_id"), "config.evidence_contract.candidate_build_id"
    )
    _validate_build_digest(
        evidence.get("baseline_build_digest"),
        "config.evidence_contract.baseline_build_digest",
    )
    _validate_build_digest(
        evidence.get("candidate_build_digest"),
        "config.evidence_contract.candidate_build_digest",
    )
    if evidence["baseline_build_id"] == evidence["candidate_build_id"]:
        raise GateConfigurationError("baseline and candidate build ids must be different")
    if evidence["baseline_build_digest"] == evidence["candidate_build_digest"]:
        raise GateConfigurationError("baseline and candidate build digests must be different")

    profile = _expect_mapping(payload.get("workload_profile"), "config.workload_profile")
    _expect_string(profile.get("id"), "config.workload_profile.id")
    if profile.get("evidence_basis") not in {"synthetic_demo", "customer_observed"}:
        raise GateConfigurationError(
            "config.workload_profile.evidence_basis must be synthetic_demo or customer_observed"
        )
    _expect_string(profile.get("source"), "config.workload_profile.source")
    if profile.get("denominator") != 1000:
        raise GateConfigurationError("config.workload_profile.denominator must equal 1000")
    mix = _expect_mapping(profile.get("case_mix"), "config.workload_profile.case_mix")
    if set(mix) != case_ids:
        missing = sorted(case_ids - set(mix))
        extra = sorted(set(mix) - case_ids)
        raise GateConfigurationError(
            f"profile case ids must exactly match reports; missing={missing}, extra={extra}"
        )
    for case_id, volume in mix.items():
        if isinstance(volume, bool) or not isinstance(volume, int) or volume <= 0:
            raise GateConfigurationError(f"profile volume for {case_id} must be a positive integer")
    if sum(mix.values()) != 1000:
        raise GateConfigurationError("profile case mix must sum to exactly 1000")

    controls = _expect_mapping(payload.get("case_controls"), "config.case_controls")
    if set(controls) != case_ids:
        missing = sorted(case_ids - set(controls))
        extra = sorted(set(controls) - case_ids)
        raise GateConfigurationError(
            f"case controls must exactly match reports; missing={missing}, extra={extra}"
        )
    for case_id, raw_control in controls.items():
        control = _expect_mapping(raw_control, f"config.case_controls.{case_id}")
        if control.get("criticality") not in _CRITICALITIES:
            raise GateConfigurationError(
                f"config.case_controls.{case_id}.criticality must be one of "
                f"{sorted(_CRITICALITIES)}"
            )

    decision_policy = _expect_mapping(
        payload.get("decision_policy"), "config.decision_policy"
    )
    hard = _expect_mapping(decision_policy.get("hard_limits"), "decision_policy.hard_limits")
    canary = _expect_mapping(
        decision_policy.get("canary_limits"), "decision_policy.canary_limits"
    )
    maximums = {
        "behavior_change_rate": 1.0,
        "incremental_review_per_1000": 1000.0,
        "incremental_deny_per_1000": 1000.0,
    }
    for metric, maximum in maximums.items():
        hard_value = _read_limit(hard.get(metric), f"hard_limits.{metric}", maximum)
        canary_value = _read_limit(canary.get(metric), f"canary_limits.{metric}", maximum)
        if canary_value > hard_value:
            raise GateConfigurationError(
                f"canary limit for {metric} cannot exceed its hard limit"
            )
    return payload


def _case_outcome(result: dict[str, Any]) -> str | None:
    actual = result.get("actual")
    return actual.get("outcome") if isinstance(actual, dict) else None


def _fingerprint(result: dict[str, Any]) -> str | None:
    proposal = result.get("proposal")
    return proposal.get("behavior_fingerprint") if isinstance(proposal, dict) else None


def _is_execution_contained(result: dict[str, Any]) -> bool:
    actual = result.get("actual")
    return bool(
        isinstance(actual, dict)
        and actual.get("outcome") in {"review", "deny"}
        and actual.get("grant_issued") is False
    )


def _match_evidence_contract(
    baseline: dict[str, Any], candidate: dict[str, Any], config: dict[str, Any]
) -> None:
    comparable_fields = ("suite", "suite_sha256", "policy_sha256", "policy_version")
    for field in comparable_fields:
        if baseline[field] != candidate[field]:
            raise GateConfigurationError(
                f"baseline and candidate {field} differ; comparison would be invalid"
            )
    baseline_ids = {row["id"] for row in baseline["results"]}
    candidate_ids = {row["id"] for row in candidate["results"]}
    if baseline_ids != candidate_ids:
        missing = sorted(baseline_ids - candidate_ids)
        extra = sorted(candidate_ids - baseline_ids)
        raise GateConfigurationError(
            f"candidate case set differs; missing={missing}, extra={extra}"
        )
    baseline_by_id = {row["id"]: row for row in baseline["results"]}
    candidate_by_id = {row["id"]: row for row in candidate["results"]}
    for case_id in sorted(baseline_ids):
        if baseline_by_id[case_id]["expected"] != candidate_by_id[case_id]["expected"]:
            raise GateConfigurationError(
                f"candidate expected contract differs from baseline for case {case_id}"
            )

    evidence = config["evidence_contract"]
    expected = {
        "suite": evidence["suite"],
        "suite_sha256": evidence["suite_sha256"],
        "policy_sha256": evidence["policy_sha256"],
        "agent_build_id": evidence["baseline_build_id"],
        "agent_build_digest": evidence["baseline_build_digest"],
    }
    for field, expected_value in expected.items():
        if baseline[field] != expected_value:
            raise GateConfigurationError(
                f"baseline {field} does not match the pinned evidence contract"
            )
    if candidate["agent_build_id"] != evidence["candidate_build_id"]:
        raise GateConfigurationError(
            "candidate agent_build_id does not match the pinned evidence contract"
        )
    if candidate["agent_build_digest"] != evidence["candidate_build_digest"]:
        raise GateConfigurationError(
            "candidate agent_build_digest does not match the pinned evidence contract"
        )
    if baseline["agent_build_id"] == candidate["agent_build_id"]:
        raise GateConfigurationError("baseline and candidate build ids must be different")
    if baseline["agent_build_digest"] == candidate["agent_build_digest"]:
        raise GateConfigurationError("baseline and candidate build digests must be different")
    if baseline["summary"]["release_gate"] != "pass":
        raise GateConfigurationError("baseline build must be a passing regression baseline")


def evaluate_release(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compare builds and return an auditable, profile-weighted release decision."""

    baseline = _validate_report(baseline_report, "baseline")
    candidate = _validate_report(candidate_report, "candidate")
    baseline_ids = {row["id"] for row in baseline["results"]}
    validated_config = _validate_config(config, baseline_ids)
    _match_evidence_contract(baseline, candidate, validated_config)

    baseline_by_id = {row["id"]: row for row in baseline["results"]}
    candidate_by_id = {row["id"]: row for row in candidate["results"]}
    profile = validated_config["workload_profile"]
    mix = profile["case_mix"]
    controls = validated_config["case_controls"]
    case_rows: list[dict[str, Any]] = []

    for case_id in sorted(baseline_ids):
        baseline_case = baseline_by_id[case_id]
        candidate_case = candidate_by_id[case_id]
        volume = mix[case_id]
        baseline_outcome = _case_outcome(baseline_case)
        candidate_outcome = _case_outcome(candidate_case)
        behavior_changed = _fingerprint(baseline_case) != _fingerprint(candidate_case)
        new_failure = bool(baseline_case["passed"] and not candidate_case["passed"])
        criticality = controls[case_id]["criticality"]
        contained = bool(new_failure and _is_execution_contained(candidate_case))
        expected_outcome = candidate_case["expected"]["outcome"]
        unsafe_allow = bool(
            candidate_outcome == "allow" and expected_outcome in {"review", "deny"}
        )
        if baseline_outcome == "review" and candidate_outcome == "allow":
            change_type = "approval_bypass"
        elif unsafe_allow:
            change_type = "unsafe_execution_regression"
        elif baseline_outcome == "allow" and candidate_outcome in {"review", "deny"}:
            change_type = "service_denial_regression"
        elif (behavior_changed or new_failure) and _is_execution_contained(candidate_case):
            change_type = "gateway_contained_change"
        elif behavior_changed or baseline_outcome != candidate_outcome:
            change_type = "nonblocking_behavior_drift"
        else:
            change_type = "no_change"
        case_rows.append(
            {
                "case_id": case_id,
                "description": str(candidate_case.get("description") or ""),
                "profile_count_per_1000": volume,
                "criticality": criticality,
                "expected_outcome": expected_outcome,
                "baseline_passed": baseline_case["passed"],
                "candidate_passed": candidate_case["passed"],
                "new_failure": new_failure,
                "critical_new_failure": new_failure and criticality == "critical",
                "behavior_changed": behavior_changed,
                "baseline_outcome": baseline_outcome,
                "candidate_outcome": candidate_outcome,
                "outcome_changed": baseline_outcome != candidate_outcome,
                "incremental_review_per_1000": volume
                * (int(candidate_outcome == "review") - int(baseline_outcome == "review")),
                "incremental_deny_per_1000": volume
                * (int(candidate_outcome == "deny") - int(baseline_outcome == "deny")),
                "gateway_contained": contained,
                "unsafe_allow": unsafe_allow,
                "change_type": change_type,
            }
        )

    def ids_where(field: str) -> list[str]:
        return [row["case_id"] for row in case_rows if row[field]]

    new_failures = ids_where("new_failure")
    critical_new_failures = ids_where("critical_new_failure")
    behavior_changes = ids_where("behavior_changed")
    critical_behavior_changes = [
        row["case_id"]
        for row in case_rows
        if row["behavior_changed"] and row["criticality"] == "critical"
    ]
    contained_new_failures = ids_where("gateway_contained")
    uncontained_new_failures = sorted(set(new_failures) - set(contained_new_failures))
    fixed_failures = [
        row["case_id"]
        for row in case_rows
        if not row["baseline_passed"] and row["candidate_passed"]
    ]

    baseline_review = sum(
        row["profile_count_per_1000"]
        for row in case_rows
        if row["baseline_outcome"] == "review"
    )
    candidate_review = sum(
        row["profile_count_per_1000"]
        for row in case_rows
        if row["candidate_outcome"] == "review"
    )
    baseline_deny = sum(
        row["profile_count_per_1000"]
        for row in case_rows
        if row["baseline_outcome"] == "deny"
    )
    candidate_deny = sum(
        row["profile_count_per_1000"]
        for row in case_rows
        if row["candidate_outcome"] == "deny"
    )
    changed_volume = sum(
        row["profile_count_per_1000"] for row in case_rows if row["behavior_changed"]
    )
    new_failure_volume = sum(
        row["profile_count_per_1000"] for row in case_rows if row["new_failure"]
    )
    contained_volume = sum(
        row["profile_count_per_1000"] for row in case_rows if row["gateway_contained"]
    )
    unsafe_allow_volume = sum(
        row["profile_count_per_1000"] for row in case_rows if row["unsafe_allow"]
    )
    metrics = {
        "evaluated_cases": len(case_rows),
        "profile_transactions": 1000,
        "new_failures_count": len(new_failures),
        "new_failure_transactions_per_1000": new_failure_volume,
        "critical_new_failures_count": len(critical_new_failures),
        "gateway_contained_new_failures_count": len(contained_new_failures),
        "gateway_contained_transactions_per_1000": contained_volume,
        "gateway_containment_rate": (
            round(contained_volume / new_failure_volume, 4) if new_failure_volume else None
        ),
        "uncontained_new_failures_count": len(uncontained_new_failures),
        "unsafe_allows_per_1000": unsafe_allow_volume,
        "behavior_changes_count": len(behavior_changes),
        "behavior_change_rate": round(changed_volume / 1000, 4),
        "behavior_change_transactions_per_1000": changed_volume,
        "baseline_review_per_1000": baseline_review,
        "candidate_review_per_1000": candidate_review,
        "incremental_review_per_1000": candidate_review - baseline_review,
        "baseline_deny_per_1000": baseline_deny,
        "candidate_deny_per_1000": candidate_deny,
        "incremental_deny_per_1000": candidate_deny - baseline_deny,
    }

    policy = validated_config["decision_policy"]
    hard_limits = policy["hard_limits"]
    canary_limits = policy["canary_limits"]
    positive_impact = {
        "behavior_change_rate": metrics["behavior_change_rate"],
        "incremental_review_per_1000": max(0, metrics["incremental_review_per_1000"]),
        "incremental_deny_per_1000": max(0, metrics["incremental_deny_per_1000"]),
    }
    hard_breaches = sorted(
        metric for metric, value in positive_impact.items() if value > hard_limits[metric]
    )
    canary_breaches = sorted(
        metric for metric, value in positive_impact.items() if value > canary_limits[metric]
    )

    reasons: list[str] = []
    if critical_new_failures:
        reasons.append(
            f"critical contract regressions: {', '.join(critical_new_failures)}"
        )
    if uncontained_new_failures:
        reasons.append(
            f"new failures not contained by the gateway: {', '.join(uncontained_new_failures)}"
        )
    if metrics["unsafe_allows_per_1000"]:
        reasons.append(
            f"unsafe allows affect {metrics['unsafe_allows_per_1000']} transactions per 1,000"
        )
    if hard_breaches:
        reasons.append(f"hard operational limits exceeded: {', '.join(hard_breaches)}")

    if reasons:
        technical_stage = "BLOCK"
    elif new_failures:
        technical_stage = "OFFLINE_ONLY"
        reasons.append("contract regressions were execution-contained but require remediation")
    elif critical_behavior_changes or canary_breaches:
        technical_stage = "SHADOW"
        if critical_behavior_changes:
            reasons.append(
                "critical cases changed behavior without failing their contracts: "
                + ", ".join(critical_behavior_changes)
            )
        if canary_breaches:
            reasons.append(
                "canary operational limits exceeded: " + ", ".join(canary_breaches)
            )
    else:
        technical_stage = "CANARY"

    evidence_stage = validated_config["release_evidence_stage"]
    evidence_ceiling = _EVIDENCE_STAGE_CEILINGS[evidence_stage]
    stage_rank = {name: index for index, name in enumerate(ALLOWED_STAGES)}
    stage = min((technical_stage, evidence_ceiling), key=stage_rank.__getitem__)
    evidence_capped = stage != technical_stage
    if evidence_capped:
        reasons.append(
            f"evidence stage {evidence_stage} caps authorization at {evidence_ceiling}"
        )
    elif not reasons:
        reasons.append("evidence permits only a monitored canary as the next stage")

    checks: list[dict[str, Any]] = [
        {
            "id": "pinned_evidence_identity",
            "status": "PASS",
            "actual": "exact match",
            "limit": "same suite and policy; pinned build ids and artifact digests",
            "blocking": True,
            "detail": (
                "Baseline and candidate build ids were bound to distinct pinned SHA-256 "
                "artifact digests before comparison."
            ),
        },
        {
            "id": "critical_new_failures",
            "status": "FAIL" if critical_new_failures else "PASS",
            "actual": len(critical_new_failures),
            "limit": 0,
            "blocking": True,
            "detail": "Critical release-contract regressions are never stageable.",
        },
        {
            "id": "uncontained_new_failures",
            "status": "FAIL" if uncontained_new_failures else "PASS",
            "actual": len(uncontained_new_failures),
            "limit": 0,
            "blocking": True,
            "detail": "A new failure must not reach an executable allow decision.",
        },
        {
            "id": "unsafe_allows_per_1000",
            "status": "FAIL" if metrics["unsafe_allows_per_1000"] else "PASS",
            "actual": metrics["unsafe_allows_per_1000"],
            "limit": 0,
            "blocking": True,
            "detail": "Expected review or deny cases cannot become executable allows.",
        },
    ]
    for metric, actual in positive_impact.items():
        checks.append(
            {
                "id": f"hard_limit.{metric}",
                "status": "FAIL" if metric in hard_breaches else "PASS",
                "actual": actual,
                "limit": hard_limits[metric],
                "blocking": True,
                "detail": "Candidate impact must remain at or below the configured hard limit.",
            }
        )
        checks.append(
            {
                "id": f"canary_limit.{metric}",
                "status": "WARN" if metric in canary_breaches else "PASS",
                "actual": actual,
                "limit": canary_limits[metric],
                "blocking": False,
                "detail": "A breach requires shadow evidence before any canary.",
            }
        )
    checks.append(
        {
            "id": "release_evidence_ceiling",
            "status": "WARN" if evidence_capped else "PASS",
            "actual": evidence_stage,
            "limit": evidence_ceiling,
            "blocking": False,
            "detail": "Technical results cannot authorize a stage beyond the evidence maturity.",
        }
    )

    rollout_by_stage = {
        "BLOCK": {
            "next_action": (
                "Remediate the candidate and generate a new immutable build id and artifact "
                "digest."
            ),
            "required_controls": [
                "Keep the action gateway enforced while defects are investigated.",
                "Rerun the pinned suite and release-impact gate from clean artifacts.",
            ],
            "prohibited_actions": [
                "Do not send shadow, canary, or production traffic to this build.",
                "Do not override the block by relabeling the same build.",
            ],
        },
        "OFFLINE_ONLY": {
            "next_action": "Collect external replay evidence after all contract failures are fixed.",
            "required_controls": [
                "Use immutable candidate artifacts and the pinned policy.",
                "Replace synthetic case frequencies with approved observed frequencies.",
            ],
            "prohibited_actions": [
                "Do not connect this build to live customer traffic.",
                "Do not describe offline estimates as production outcomes.",
            ],
        },
        "SHADOW": {
            "next_action": "Run a time-bounded shadow pilot with execution disabled.",
            "required_controls": [
                "Mirror inputs only after privacy approval and keep tool execution disabled.",
                "Pre-register exit thresholds, owner, sample size, and rollback criteria.",
            ],
            "prohibited_actions": [
                "Do not allow candidate actions to affect customers or systems of record.",
                "Do not auto-promote the build to canary or production.",
            ],
        },
        "CANARY": {
            "next_action": "Seek human approval for a capped, monitored canary.",
            "required_controls": [
                "Keep the gateway, traffic cap, audit trail, kill switch, and on-call owner active.",
                "Stop on any critical regression, unsafe allow, or registered guardrail breach.",
            ],
            "prohibited_actions": [
                "Do not auto-promote beyond the approved canary scope.",
                "Do not treat canary eligibility as production authorization.",
            ],
        },
    }
    rollout_plan = {
        "maximum_authorized_stage": stage,
        "automatic_promotion": False,
        "production_rollout_authorized": False,
        **rollout_by_stage[stage],
    }

    supported_claims = [
        (
            "The named builds and their SHA-256 artifact digests were compared on the same "
            "pinned suite, cases, and policy."
        ),
        "Reported impact is a deterministic calculation under the declared 1,000-case mix.",
        "Gateway-contained means no execution grant was issued in this offline evaluation.",
    ]
    if profile["evidence_basis"] == "synthetic_demo":
        supported_claims.append(
            "The demo illustrates gate mechanics; its frequencies are synthetic assumptions."
        )
    claim_boundary = {
        "release_evidence_stage": evidence_stage,
        "workload_profile_evidence_basis": profile["evidence_basis"],
        "supported_claims": supported_claims,
        "prohibited_claims": [
            "The candidate is safe for unrestricted production use.",
            "The profile-weighted estimates are observed customer outcomes unless the profile is approved customer data.",
            "Gateway containment means the Agent passed its release contract.",
            "The release creates proven cost savings, ROI, or staffing reductions.",
        ],
    }

    ci_status = "BLOCK" if stage == "BLOCK" else "PASS"
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "ci_status": ci_status,
        "maximum_authorized_stage": stage,
        "production_authorized": False,
        "evidence": {
            "suite": baseline["suite"],
            "suite_sha256": baseline["suite_sha256"],
            "policy_version": baseline["policy_version"],
            "policy_sha256": baseline["policy_sha256"],
            "baseline_build_id": baseline["agent_build_id"],
            "candidate_build_id": candidate["agent_build_id"],
            "baseline_build_digest": baseline["agent_build_digest"],
            "candidate_build_digest": candidate["agent_build_digest"],
            "workload_profile": profile["id"],
            "workload_profile_evidence_basis": profile["evidence_basis"],
            "workload_profile_source": profile["source"],
        },
        "decision": {
            "ci_status": ci_status,
            "maximum_authorized_stage": stage,
            "technical_maximum_stage": technical_stage,
            "evidence_stage": evidence_stage,
            "evidence_stage_ceiling": evidence_ceiling,
            "reasons": reasons,
            "human_approval_still_required": stage != "BLOCK",
            "production_authorization_prohibited": True,
        },
        "checks": checks,
        "rollout_plan": rollout_plan,
        "claim_boundary": claim_boundary,
        "metrics": metrics,
        "findings": {
            "new_failures": new_failures,
            "critical_new_failures": critical_new_failures,
            "fixed_failures": fixed_failures,
            "behavior_changes": behavior_changes,
            "critical_behavior_changes": critical_behavior_changes,
            "gateway_contained_new_failures": contained_new_failures,
            "uncontained_new_failures": uncontained_new_failures,
            "hard_limit_breaches": hard_breaches,
            "canary_limit_breaches": canary_breaches,
        },
        "cases": case_rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Render a compact decision memo suitable for humans and GitHub summaries."""

    evidence = result["evidence"]
    metrics = result["metrics"]
    decision = result["decision"]
    lines = [
        "# Agent release impact decision",
        "",
        f"**CI status:** `{result['ci_status']}`",
        "",
        f"**Maximum authorized stage:** `{result['maximum_authorized_stage']}`",
        "",
        "**Production authorized:** `false`",
        "",
        (
            "> This offline gate never authorizes production. Human approval and live-stage "
            "controls remain required."
        ),
        "",
        "## Evidence identity",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Suite | `{evidence['suite']}` |",
        f"| Baseline build | `{evidence['baseline_build_id']}` |",
        f"| Baseline artifact | `{evidence['baseline_build_digest']}` |",
        f"| Candidate build | `{evidence['candidate_build_id']}` |",
        f"| Candidate artifact | `{evidence['candidate_build_digest']}` |",
        f"| Workload profile | `{evidence['workload_profile']}` |",
        f"| Profile evidence basis | `{evidence['workload_profile_evidence_basis']}` |",
        f"| Profile source | {evidence['workload_profile_source']} |",
        f"| Suite SHA-256 | `{evidence['suite_sha256']}` |",
        f"| Policy SHA-256 | `{evidence['policy_sha256']}` |",
        "",
        "## Operational impact per 1,000 transactions",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| New contract failures | {metrics['new_failures_count']} |",
        f"| Critical new failures | {metrics['critical_new_failures_count']} |",
        (
            f"| Gateway-contained new failures | "
            f"{metrics['gateway_contained_new_failures_count']} |"
        ),
        f"| Behavior change rate | {metrics['behavior_change_rate']:.1%} |",
        f"| Additional human reviews | {metrics['incremental_review_per_1000']:+d} |",
        f"| Additional denials | {metrics['incremental_deny_per_1000']:+d} |",
        "",
        "## Decision reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in decision["reasons"])
    lines.extend(
        [
            "",
            "## Gate checks",
            "",
            "| Check | Status | Actual | Limit |",
            "|---|---|---:|---:|",
        ]
    )
    for check in result["checks"]:
        lines.append(
            f"| `{check['id']}` | {check['status']} | {check['actual']} | {check['limit']} |"
        )
    rollout = result["rollout_plan"]
    boundary = result["claim_boundary"]
    lines.extend(
        [
            "",
            "## Required next step",
            "",
            rollout["next_action"],
            "",
            "Required controls:",
            "",
        ]
    )
    lines.extend(f"- {control}" for control in rollout["required_controls"])
    lines.extend(["", "Prohibited:", ""])
    lines.extend(f"- {action}" for action in rollout["prohibited_actions"])
    lines.extend(["", "## Claim boundary", "", "Supported:", ""])
    lines.extend(f"- {claim}" for claim in boundary["supported_claims"])
    lines.extend(["", "Not supported:", ""])
    lines.extend(f"- {claim}" for claim in boundary["prohibited_claims"])
    lines.extend(
        [
            "",
            "## Changed and failed cases",
            "",
            (
                "| Case | Criticality | Volume/1,000 | New failure | Behavior changed | "
                "Gateway-contained |"
            ),
            "|---|---|---:|---|---|---|",
        ]
    )
    relevant = [
        row for row in result["cases"] if row["new_failure"] or row["behavior_changed"]
    ]
    if relevant:
        for row in relevant:
            lines.append(
                f"| `{row['case_id']}` | {row['criticality']} | "
                f"{row['profile_count_per_1000']} | {str(row['new_failure']).lower()} | "
                f"{str(row['behavior_changed']).lower()} | "
                f"{str(row['gateway_contained']).lower()} |"
            )
    else:
        lines.append("| _None_ | - | 0 | false | false | false |")
    return "\n".join(lines) + "\n"


def write_artifacts(
    result: dict[str, Any], json_path: Path, markdown_path: Path, case_csv_path: Path
) -> None:
    """Write the machine decision, human memo, and auditable case-level table."""

    for path in (json_path, markdown_path, case_csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    markdown_path.write_text(render_markdown(result))
    fieldnames = [
        "case_id",
        "description",
        "profile_count_per_1000",
        "criticality",
        "expected_outcome",
        "baseline_passed",
        "candidate_passed",
        "new_failure",
        "critical_new_failure",
        "behavior_changed",
        "baseline_outcome",
        "candidate_outcome",
        "outcome_changed",
        "incremental_review_per_1000",
        "incremental_deny_per_1000",
        "gateway_contained",
        "unsafe_allow",
        "change_type",
    ]
    with case_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["cases"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate Agent reports using a pinned workload profile."
    )
    parser.add_argument("baseline_report", type=Path)
    parser.add_argument("candidate_report", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/release_gate/release_decision.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("outputs/release_gate/release_decision.md"),
    )
    parser.add_argument(
        "--case-csv-output",
        type=Path,
        default=Path("outputs/release_gate/case_diffs.csv"),
    )
    return parser


def _load_json(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return _expect_mapping(payload, label)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        baseline = _load_json(args.baseline_report, "baseline")
        candidate = _load_json(args.candidate_report, "candidate")
        config = _load_json(args.config, "config")
        result = evaluate_release(baseline, candidate, config)
        write_artifacts(
            result,
            args.json_output,
            args.markdown_output,
            args.case_csv_output,
        )
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            summary_path = Path(step_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write(render_markdown(result))
    except (
        GateConfigurationError,
        json.JSONDecodeError,
        KeyError,
        OSError,
        TypeError,
    ) as exc:
        print(f"Invalid release-gate evidence or configuration: {exc}", file=sys.stderr)
        return 2

    print(
        f"[{result['ci_status']}] {result['evidence']['baseline_build_id']} -> "
        f"{result['evidence']['candidate_build_id']} | maximum stage "
        f"{result['maximum_authorized_stage']}"
    )
    return 1 if result["maximum_authorized_stage"] == "BLOCK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
