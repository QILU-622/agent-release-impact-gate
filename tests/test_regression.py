from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from agent_mesh_risk_lab.regression import (
    CapturedProposalProvider,
    HttpProposalProvider,
    canonical_json_sha256,
    compare_with_baseline,
    main,
    run_suite,
    write_junit,
)

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "configs" / "regression" / "refund_action_contracts.json"
POLICY = ROOT / "configs" / "enterprise" / "policy.json"


def test_refund_release_contracts_pass() -> None:
    report = run_suite(SUITE, POLICY)
    summary = report["summary"]
    assert summary["total"] == 6
    assert summary["passed"] == 6
    assert summary["failed"] == 0
    assert summary["pass_rate"] == 1.0
    assert summary["release_gate"] == "pass"
    assert all("grant_token" not in str(result) for result in report["results"])
    assert "O-100" not in json.dumps(report)
    assert len(report["suite_sha256"]) == 64
    assert len(report["policy_sha256"]) == 64
    assert report["trusted_context_enforced"] is True
    assert "agent_build_digest" not in report
    assert all(result["context_source"] == "release-contract" for result in report["results"])


def test_failure_returns_ci_exit_code_and_reports_diff(tmp_path: Path) -> None:
    suite = json.loads(SUITE.read_text())
    suite["cases"][0]["expect"]["outcome"] = "deny"
    failing_suite = tmp_path / "failing.json"
    failing_suite.write_text(json.dumps(suite))
    json_report = tmp_path / "report.json"
    junit_report = tmp_path / "junit.xml"

    exit_code = main(
        [
            str(failing_suite),
            "--policy",
            str(POLICY),
            "--json-report",
            str(json_report),
            "--junit-report",
            str(junit_report),
        ]
    )

    assert exit_code == 1
    report = json.loads(json_report.read_text())
    assert report["summary"]["failed"] == 1
    assert "outcome expected deny but received allow" in report["results"][0]["mismatches"]
    assert 'failures="1"' in junit_report.read_text()


def test_write_junit_emits_successful_suite(tmp_path: Path) -> None:
    report = run_suite(SUITE, POLICY)
    target = tmp_path / "results.xml"
    write_junit(report, target)
    xml = target.read_text()
    assert 'tests="6"' in xml
    assert 'failures="0"' in xml


def test_captured_agent_proposal_detects_tool_selection_regression(tmp_path: Path) -> None:
    suite = json.loads(SUITE.read_text())
    proposals = {case["id"]: case["request"] for case in suite["cases"]}
    proposals["read-order-with-least-privilege"] = {
        **proposals["read-order-with-least-privilege"],
        "tool_name": "refund_order",
        "arguments": {"order_id": "O-100", "amount": 50.0},
    }
    capture_path = tmp_path / "captured.json"
    capture_path.write_text(
        json.dumps({"schema_version": "1.0", "source": "refund-agent-v2", "proposals": proposals})
    )

    report = run_suite(SUITE, POLICY, CapturedProposalProvider(capture_path))

    assert report["proposal_source"] == "refund-agent-v2"
    assert report["agent_build_digest"] == canonical_json_sha256(
        json.loads(capture_path.read_text())
    )
    assert report["summary"]["failed"] == 1
    first = report["results"][0]
    assert "tool expected get_order but Agent proposed refund_order" in first["mismatches"]


def test_captured_artifact_digest_is_canonical_and_cannot_be_overridden(
    tmp_path: Path,
) -> None:
    suite = json.loads(SUITE.read_text())
    proposals = {case["id"]: case["request"] for case in suite["cases"]}
    payload = {
        "schema_version": "1.0",
        "source": "refund-agent-v2",
        "proposals": proposals,
    }
    compact_path = tmp_path / "compact.json"
    pretty_path = tmp_path / "pretty.json"
    compact_path.write_text(json.dumps(payload, separators=(",", ":")))
    pretty_path.write_text(json.dumps(dict(reversed(payload.items())), indent=4))

    compact = CapturedProposalProvider(compact_path)
    pretty = CapturedProposalProvider(pretty_path)
    assert compact.build_digest == pretty.build_digest
    assert compact.build_digest.startswith("sha256:")

    with pytest.raises(ValueError, match="does not match the canonical captured-proposal"):
        run_suite(
            SUITE,
            POLICY,
            compact,
            build_id="refund-agent-v2",
            build_digest="sha256:" + "0" * 64,
        )
    with pytest.raises(ValueError, match="build id does not match the identity inside"):
        run_suite(
            SUITE,
            POLICY,
            CapturedProposalProvider(compact_path),
            build_id="relabelled-build",
            build_digest=compact.build_digest,
        )


def test_optional_build_digest_uses_strict_prefixed_sha256() -> None:
    digest = "sha256:" + "a" * 64
    report = run_suite(SUITE, POLICY, build_digest=digest)
    assert report["agent_build_digest"] == digest

    with pytest.raises(ValueError, match="sha256:<64 lowercase"):
        run_suite(SUITE, POLICY, build_digest="A" * 64)


def test_http_adapter_runs_real_agent_contract_boundary() -> None:
    suite = json.loads(SUITE.read_text())
    proposals = {case["id"]: case["request"] for case in suite["cases"]}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stimulus"]
        return httpx.Response(200, json={"request": proposals[payload["case_id"]]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = HttpProposalProvider("https://agent.example/propose", client=client)
    report = run_suite(SUITE, POLICY, provider)
    client.close()

    assert report["proposal_source"] == "live-agent-http-adapter"
    assert report["summary"]["release_gate"] == "pass"


def test_external_agent_cannot_override_trusted_release_context(tmp_path: Path) -> None:
    suite = json.loads(SUITE.read_text())
    proposals = {case["id"]: case["request"] for case in suite["cases"]}
    case_id = "untrusted-content-cannot-trigger-refund"
    proposals[case_id]["context"] = {
        **proposals[case_id]["context"],
        "source_trust": "trusted",
        "user_confirmed": True,
    }
    capture_path = tmp_path / "captured.json"
    capture_path.write_text(json.dumps({"source": "candidate-build", "proposals": proposals}))

    report = run_suite(SUITE, POLICY, CapturedProposalProvider(capture_path))
    result = next(row for row in report["results"] if row["id"] == case_id)

    assert result["passed"] is True
    assert result["proposal"]["source_trust"] == "untrusted"
    assert result["context_source"] == "release-contract"


def test_external_agent_requires_contract_owned_context(tmp_path: Path) -> None:
    suite = json.loads(SUITE.read_text())
    suite["cases"][0].pop("trusted_context")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(suite))
    proposals = {case["id"]: case["request"] for case in suite["cases"]}
    capture_path = tmp_path / "captured.json"
    capture_path.write_text(json.dumps({"source": "candidate-build", "proposals": proposals}))

    report = run_suite(suite_path, POLICY, CapturedProposalProvider(capture_path))
    first = report["results"][0]

    assert first["passed"] is False
    assert "requires trusted_context" in first["mismatches"][0]


def test_baseline_comparison_can_block_unreviewed_behavior_change() -> None:
    current = run_suite(SUITE, POLICY)
    baseline = json.loads(json.dumps(current))
    baseline["results"][0]["proposal"]["behavior_fingerprint"] = "0" * 64

    compared = compare_with_baseline(current, baseline, fail_on_behavior_change=True)

    assert compared["comparison"]["behavior_changes"] == ["read-order-with-least-privilege"]
    assert compared["summary"]["release_gate"] == "fail"
