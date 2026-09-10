from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

from agent_mesh_risk_lab.regression import CapturedProposalProvider, run_suite
from agent_mesh_risk_lab.release_impact_gate import evaluate_release

ROOT = Path(__file__).parents[1]
BUILD = runpy.run_path(str(ROOT / "scripts" / "build_project_site.py"))["build_site"]


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    output = tmp_path_factory.mktemp("interactive-site")
    BUILD(output)
    return output, json.loads((output / "demo" / "scenarios.json").read_text())


@pytest.mark.parametrize("scenario_id,stage,failures,critical,denials,behavior", [
    ("risky", "BLOCK", 3, 2, 550, 0.7),
    ("partial", "BLOCK", 1, 0, 550, 0.55),
    ("fixed", "OFFLINE_ONLY", 0, 0, 0, 0.0),
])
def test_demo_decisions_are_engine_executed(
    site, scenario_id, stage, failures, critical, denials, behavior,
):
    output, bundle = site
    demo = output / "demo"
    scenario = next(item for item in bundle["scenarios"] if item["id"] == scenario_id)
    packet = scenario["packet"]
    assert packet["maximum_authorized_stage"] == stage
    assert packet["metrics"]["new_failures_count"] == failures
    assert packet["metrics"]["critical_new_failures_count"] == critical
    assert packet["metrics"]["incremental_deny_per_1000"] == denials
    assert packet["metrics"]["behavior_change_rate"] == behavior
    assert packet["metrics"]["unsafe_allows_per_1000"] == 0
    # Independently execute the published captures/config, not just the bundled report.
    reports = []
    for side in ("baseline", "candidate"):
        contract = scenario["config"]["evidence_contract"]
        reports.append(run_suite(
            demo / "evidence" / "suite.json", demo / "evidence" / "policy.json",
            CapturedProposalProvider(demo / scenario["artifacts"][f"{side}_proposals.json"]),
            build_id=contract[f"{side}_build_id"],
            build_digest=contract[f"{side}_build_digest"],
        ))
    rerun = evaluate_release(*reports, scenario["config"])
    for field in ("decision", "metrics", "checks", "evidence", "cases"):
        assert rerun[field] == packet[field]


def test_demo_preserves_synthetic_evidence_ceiling(site):
    _, bundle = site
    assert bundle["execution_mode"] == "build_time_engine_replay"
    assert bundle["evidence_class"] == "synthetic_demo"
    assert bundle["production_authorized"] is False
    identities = set()
    for scenario in bundle["scenarios"]:
        packet = scenario["packet"]
        assert packet["production_authorized"] is False
        assert packet["decision"]["evidence_stage_ceiling"] == "OFFLINE_ONLY"
        assert packet["decision"]["evidence_stage"] == "synthetic_demo"
        identities.add(packet["evidence"]["candidate_build_digest"])
    assert len(identities) == 3
    assert bundle["scenarios"][-1]["packet"]["decision"]["technical_maximum_stage"] == "CANARY"


def test_demo_changes_proposals_not_rules(site):
    _, bundle = site
    scenarios = bundle["scenarios"]
    for scenario in scenarios:
        for field in ("workload_profile", "decision_policy", "case_controls"):
            assert scenario["config"][field] == scenarios[0]["config"][field]
        assert scenario["baseline_proposals"] == scenarios[0]["baseline_proposals"]
        assert scenario["packet"]["metrics"]["evaluated_cases"] == 6
    assert scenarios[-1]["baseline_proposals"] == scenarios[-1]["candidate_proposals"]
    # Preserve the originally published risky demo's claims.
    original = json.loads((ROOT / "outputs/release_gate/demo/release_decision.json").read_text())
    assert scenarios[0]["packet"]["metrics"] == original["metrics"]


def test_demo_downloads_and_integrity_manifest(site):
    output, bundle = site
    demo = output / "demo"
    manifest = json.loads((demo / "manifest.json").read_text())
    assert len(manifest["sha256"]) == 30  # 24 scenario artifacts + 2 inputs + 4 app files
    for name, digest in manifest["sha256"].items():
        assert hashlib.sha256((demo / name).read_bytes()).hexdigest() == digest
    for scenario in bundle["scenarios"]:
        for name, path in scenario["artifacts"].items():
            assert (demo / path).is_file()
            assert path.startswith(f'evidence/{scenario["id"]}/')
            if name == "release_decision.json":
                assert json.loads((demo / path).read_text()) == scenario["packet"]


def test_demo_has_entrypoints_and_no_javascript_fallback(site):
    output, _ = site
    assert 'href="demo/"' in (output / "index.html").read_text()
    html = (output / "demo/index.html").read_text()
    assert '<noscript>' in html
    assert 'id="run" type="button" disabled' in html
    assert 'id="result" hidden' in html
    assert 'aria-live="polite"' in html
    for name in ("demo.js", "demo.css"):
        assert (output / "demo" / name).read_text() == (ROOT / "site/demo" / name).read_text()
