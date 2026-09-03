from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from agent_mesh_risk_lab.regression import canonical_json_sha256, run_suite
from agent_mesh_risk_lab.release_impact_gate import (
    GateConfigurationError,
    evaluate_release,
    main,
)

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "configs" / "regression" / "refund_action_contracts.json"
POLICY = ROOT / "configs" / "enterprise" / "policy.json"
CONFIG = ROOT / "configs" / "release" / "refund_release_gate.json"
BUILD_DEMO = runpy.run_path(str(ROOT / "scripts" / "build_release_demo.py"))["build_demo"]


def _reports() -> tuple[dict, dict, dict]:
    config = json.loads(CONFIG.read_text())
    baseline = run_suite(SUITE, POLICY)
    baseline["agent_build_id"] = config["evidence_contract"]["baseline_build_id"]
    baseline["agent_build_digest"] = config["evidence_contract"][
        "baseline_build_digest"
    ]
    candidate = json.loads(json.dumps(baseline))
    candidate["agent_build_id"] = config["evidence_contract"]["candidate_build_id"]
    candidate["agent_build_digest"] = config["evidence_contract"][
        "candidate_build_digest"
    ]
    return baseline, candidate, config


def _fail_case(
    report: dict,
    case_id: str,
    *,
    outcome: str,
    fingerprint: str = "1" * 64,
) -> None:
    result = next(row for row in report["results"] if row["id"] == case_id)
    result["passed"] = False
    result["proposal"]["behavior_fingerprint"] = fingerprint
    result["actual"]["outcome"] = outcome
    result["actual"]["grant_issued"] = False
    result["mismatches"] = ["candidate behavior violated the release contract"]
    passed = sum(row["passed"] for row in report["results"])
    report["summary"].update(
        {
            "passed": passed,
            "failed": len(report["results"]) - passed,
            "release_gate": "fail",
            "pass_rate": round(passed / len(report["results"]), 4),
        }
    )


def test_profile_weighted_gate_blocks_critical_regression() -> None:
    baseline, candidate, config = _reports()
    _fail_case(candidate, "refund-without-confirmation-is-blocked", outcome="review")

    result = evaluate_release(baseline, candidate, config)

    assert result["ci_status"] == "BLOCK"
    assert result["maximum_authorized_stage"] == "BLOCK"
    assert result["production_authorized"] is False
    assert result["findings"]["critical_new_failures"] == [
        "refund-without-confirmation-is-blocked"
    ]
    assert result["findings"]["gateway_contained_new_failures"] == [
        "refund-without-confirmation-is-blocked"
    ]
    assert result["metrics"]["behavior_change_rate"] == 0.1
    assert result["metrics"]["incremental_review_per_1000"] == 100
    assert result["metrics"]["incremental_deny_per_1000"] == -100


def test_approval_bypass_is_counted_as_an_unsafe_allow() -> None:
    baseline, candidate, config = _reports()
    _fail_case(candidate, "confirmed-refund-needs-independent-review", outcome="allow")
    target = next(
        row
        for row in candidate["results"]
        if row["id"] == "confirmed-refund-needs-independent-review"
    )
    target["actual"]["grant_issued"] = True

    result = evaluate_release(baseline, candidate, config)
    case = next(
        row
        for row in result["cases"]
        if row["case_id"] == "confirmed-refund-needs-independent-review"
    )

    assert result["metrics"]["unsafe_allows_per_1000"] == 180
    assert case["change_type"] == "approval_bypass"
    assert result["maximum_authorized_stage"] == "BLOCK"


def test_synthetic_evidence_caps_unchanged_candidate_at_offline_only() -> None:
    baseline, candidate, config = _reports()

    result = evaluate_release(baseline, candidate, config)

    assert result["ci_status"] == "PASS"
    assert result["decision"]["technical_maximum_stage"] == "CANARY"
    assert result["maximum_authorized_stage"] == "OFFLINE_ONLY"
    assert result["decision"]["human_approval_still_required"] is True
    assert result["decision"]["production_authorization_prohibited"] is True


def test_validated_shadow_evidence_can_reach_but_not_exceed_canary() -> None:
    baseline, candidate, config = _reports()
    config["release_evidence_stage"] = "validated_shadow_pilot"

    result = evaluate_release(baseline, candidate, config)

    assert result["maximum_authorized_stage"] == "CANARY"
    assert result["production_authorized"] is False


def test_contained_noncritical_contract_failure_stays_offline() -> None:
    baseline, candidate, config = _reports()
    _fail_case(candidate, "tool-contract-drift-fails-closed", outcome="deny")

    result = evaluate_release(baseline, candidate, config)

    assert result["ci_status"] == "PASS"
    assert result["maximum_authorized_stage"] == "OFFLINE_ONLY"
    assert result["metrics"]["gateway_containment_rate"] == 1.0


def test_passing_critical_behavior_change_requires_shadow() -> None:
    baseline, candidate, config = _reports()
    config["release_evidence_stage"] = "external_replay"
    target = next(
        row
        for row in candidate["results"]
        if row["id"] == "untrusted-content-cannot-trigger-refund"
    )
    target["proposal"]["behavior_fingerprint"] = "2" * 64

    result = evaluate_release(baseline, candidate, config)

    assert result["maximum_authorized_stage"] == "SHADOW"
    assert result["findings"]["critical_behavior_changes"] == [
        "untrusted-content-cannot-trigger-refund"
    ]


def test_evidence_identity_and_profile_are_fail_closed() -> None:
    baseline, candidate, config = _reports()
    candidate["policy_sha256"] = "f" * 64
    with pytest.raises(GateConfigurationError, match="policy_sha256 differ"):
        evaluate_release(baseline, candidate, config)

    baseline, candidate, config = _reports()
    candidate["suite"] = "another-suite"
    with pytest.raises(GateConfigurationError, match="suite differ"):
        evaluate_release(baseline, candidate, config)

    baseline, candidate, config = _reports()
    candidate["suite_sha256"] = "e" * 64
    with pytest.raises(GateConfigurationError, match="suite_sha256 differ"):
        evaluate_release(baseline, candidate, config)

    baseline, candidate, config = _reports()
    candidate["results"].pop()
    candidate["summary"].update(
        {"total": 5, "passed": 5, "failed": 0, "pass_rate": 1.0, "release_gate": "pass"}
    )
    with pytest.raises(GateConfigurationError, match="candidate case set differs"):
        evaluate_release(baseline, candidate, config)

    baseline, candidate, config = _reports()
    candidate.pop("agent_build_id")
    with pytest.raises(GateConfigurationError, match="candidate.agent_build_id"):
        evaluate_release(baseline, candidate, config)

    baseline, candidate, config = _reports()
    candidate.pop("agent_build_digest")
    with pytest.raises(GateConfigurationError, match="candidate.agent_build_digest"):
        evaluate_release(baseline, candidate, config)

    baseline, candidate, config = _reports()
    candidate["agent_build_digest"] = "sha256:" + "f" * 64
    with pytest.raises(GateConfigurationError, match="pinned evidence contract"):
        evaluate_release(baseline, candidate, config)

    baseline, candidate, config = _reports()
    config["evidence_contract"]["candidate_build_digest"] = "f" * 64
    with pytest.raises(GateConfigurationError, match="sha256:<64 lowercase"):
        evaluate_release(baseline, candidate, config)

    baseline, candidate, config = _reports()
    config["workload_profile"]["case_mix"]["read-order-with-least-privilege"] -= 1
    with pytest.raises(GateConfigurationError, match="sum to exactly 1000"):
        evaluate_release(baseline, candidate, config)


def test_cli_writes_all_artifacts_and_github_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline, candidate, config = _reports()
    _fail_case(candidate, "refund-without-confirmation-is-blocked", outcome="review")
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    config_path = tmp_path / "config.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    config_path.write_text(json.dumps(config))
    summary_path = tmp_path / "github-summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_path))

    exit_code = main(
        [
            str(baseline_path),
            str(candidate_path),
            "--config",
            str(config_path),
            "--json-output",
            str(tmp_path / "decision.json"),
            "--markdown-output",
            str(tmp_path / "decision.md"),
            "--case-csv-output",
            str(tmp_path / "cases.csv"),
        ]
    )

    assert exit_code == 1
    assert json.loads((tmp_path / "decision.json").read_text())["ci_status"] == "BLOCK"
    assert "Maximum authorized stage" in (tmp_path / "decision.md").read_text()
    assert "profile_count_per_1000" in (tmp_path / "cases.csv").read_text()
    assert "Agent release impact decision" in summary_path.read_text()


def test_cli_returns_two_for_invalid_configuration(tmp_path: Path) -> None:
    baseline, candidate, config = _reports()
    config["evidence_contract"]["candidate_build_id"] = config["evidence_contract"][
        "baseline_build_id"
    ]
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    config_path = tmp_path / "config.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    config_path.write_text(json.dumps(config))

    assert main([str(baseline_path), str(candidate_path), "--config", str(config_path)]) == 2


def test_release_contract_requires_distinct_build_artifact_digests() -> None:
    baseline, candidate, config = _reports()
    config["evidence_contract"]["candidate_build_digest"] = config[
        "evidence_contract"
    ]["baseline_build_digest"]

    with pytest.raises(GateConfigurationError, match="build digests must be different"):
        evaluate_release(baseline, candidate, config)


def test_cli_returns_zero_for_nonblocking_stage(tmp_path: Path) -> None:
    baseline, candidate, config = _reports()
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    config_path = tmp_path / "config.json"
    baseline_path.write_text(json.dumps(baseline))
    candidate_path.write_text(json.dumps(candidate))
    config_path.write_text(json.dumps(config))

    assert (
        main(
            [
                str(baseline_path),
                str(candidate_path),
                "--config",
                str(config_path),
                "--json-output",
                str(tmp_path / "decision.json"),
                "--markdown-output",
                str(tmp_path / "decision.md"),
                "--case-csv-output",
                str(tmp_path / "cases.csv"),
            ]
        )
        == 0
    )


def test_demo_is_executed_from_suite_and_produces_a_block(tmp_path: Path) -> None:
    decision = BUILD_DEMO(tmp_path)
    baseline = json.loads((tmp_path / "baseline_report.json").read_text())
    candidate = json.loads((tmp_path / "candidate_report.json").read_text())
    baseline_capture = json.loads((tmp_path / "baseline_proposals.json").read_text())
    candidate_capture = json.loads((tmp_path / "candidate_proposals.json").read_text())

    assert baseline["proposal_source"] == "refund-agent-v1.4.2"
    assert candidate["proposal_source"] == "refund-agent-v1.5.0-risky"
    assert baseline["agent_build_digest"] == canonical_json_sha256(baseline_capture)
    assert candidate["agent_build_digest"] == canonical_json_sha256(candidate_capture)
    assert baseline["agent_build_digest"] != candidate["agent_build_digest"]
    assert decision["evidence"]["baseline_build_digest"] == baseline[
        "agent_build_digest"
    ]
    assert decision["evidence"]["candidate_build_digest"] == candidate[
        "agent_build_digest"
    ]
    assert baseline["summary"]["failed"] == 0
    assert candidate["summary"]["failed"] == 3
    assert decision["maximum_authorized_stage"] == "BLOCK"
    assert decision["metrics"]["incremental_deny_per_1000"] == 550
    read_case = next(
        row
        for row in decision["cases"]
        if row["case_id"] == "read-order-with-least-privilege"
    )
    assert read_case["change_type"] == "service_denial_regression"
    assert (tmp_path / "release_decision.md").exists()
    assert (tmp_path / "case_diffs.csv").exists()
