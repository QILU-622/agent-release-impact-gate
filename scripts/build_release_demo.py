"""Build a release-gate demo by executing the real refund regression suite.

The risky candidate is produced by changing captured Agent proposals and running
them through the existing ActionGateway.  No release result is hand-authored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_mesh_risk_lab.regression import CapturedProposalProvider, run_suite
from agent_mesh_risk_lab.release_impact_gate import evaluate_release, write_artifacts

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "configs" / "regression" / "refund_action_contracts.json"
POLICY = ROOT / "configs" / "enterprise" / "policy.json"
GATE_CONFIG = ROOT / "configs" / "release" / "refund_release_gate.json"


def _suite_proposals() -> dict[str, dict[str, Any]]:
    suite = json.loads(SUITE.read_text())
    return {case["id"]: case["request"] for case in suite["cases"]}


def build_demo(output_dir: Path) -> dict[str, Any]:
    """Execute baseline/candidate builds and materialize their gate evidence."""

    config = json.loads(GATE_CONFIG.read_text())
    evidence = config["evidence_contract"]
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_proposals = _suite_proposals()
    baseline_capture = {
        "schema_version": "1.0",
        "source": evidence["baseline_build_id"],
        "proposals": baseline_proposals,
    }
    baseline_capture_path = output_dir / "baseline_proposals.json"
    baseline_capture_path.write_text(
        json.dumps(
            baseline_capture,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )
    baseline = run_suite(
        SUITE,
        POLICY,
        CapturedProposalProvider(baseline_capture_path),
        build_id=evidence["baseline_build_id"],
        build_digest=evidence["baseline_build_digest"],
    )

    # The candidate proposes high-value refunds for two requests that must not execute.
    # Trusted context still comes from the release contract; the real gateway evaluates
    # and contains the unsafe proposals.
    candidate_proposals = json.loads(json.dumps(baseline_proposals))
    candidate_proposals["read-order-with-least-privilege"]["tool_version"] = "2.0"
    candidate_proposals["refund-without-confirmation-is-blocked"]["arguments"][
        "amount"
    ] = 5000.0
    candidate_proposals["untrusted-content-cannot-trigger-refund"]["arguments"][
        "amount"
    ] = 5000.0
    candidate_capture = {
        "schema_version": "1.0",
        "source": evidence["candidate_build_id"],
        "proposals": candidate_proposals,
    }
    candidate_capture_path = output_dir / "candidate_proposals.json"
    candidate_capture_path.write_text(
        json.dumps(
            candidate_capture,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    )
    candidate = run_suite(
        SUITE,
        POLICY,
        CapturedProposalProvider(candidate_capture_path),
        build_id=evidence["candidate_build_id"],
        build_digest=evidence["candidate_build_digest"],
    )

    decision = evaluate_release(baseline, candidate, config)
    (output_dir / "baseline_report.json").write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False) + "\n"
    )
    (output_dir / "candidate_report.json").write_text(
        json.dumps(candidate, indent=2, ensure_ascii=False) + "\n"
    )
    write_artifacts(
        decision,
        output_dir / "release_decision.json",
        output_dir / "release_decision.md",
        output_dir / "case_diffs.csv",
    )
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate baseline, risky candidate, and release-decision demo artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "release_gate" / "demo",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    decision = build_demo(args.output_dir)
    print(
        f"Generated executed demo at {args.output_dir} | "
        f"decision={decision['maximum_authorized_stage']}"
    )
    # A deliberately blocked candidate is the successful output of this demo builder.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
