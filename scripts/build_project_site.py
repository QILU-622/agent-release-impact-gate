"""Build the project page and engine-executed interactive demo (core install only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import shutil
from html import escape
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/QILU-622/agent-release-impact-gate"
EVIDENCE_FILES = (
    "release_decision.json", "release_decision.md", "case_diffs.csv",
    "baseline_report.json", "candidate_report.json",
    "baseline_proposals.json", "candidate_proposals.json",
)


def build_interactive_demo(output: Path) -> dict:
    """Replay three captured builds through the real gateway and release engine.

    Only the proposals change. The suite, policy, synthetic mix, thresholds and
    evidence ceiling are held constant. Browser interactions inspect these
    build-time results; they never execute a live Agent or business tool.
    """
    from agent_mesh_risk_lab.regression import (
        CapturedProposalProvider,
        canonical_json_sha256,
        run_suite,
    )
    from agent_mesh_risk_lab.release_impact_gate import evaluate_release, write_artifacts

    source = runpy.run_path(str(ROOT / "scripts" / "build_release_demo.py"))
    suite, policy = source["SUITE"], source["POLICY"]
    base_config = json.loads(source["GATE_CONFIG"].read_text())
    risky_dir = output / "evidence" / "risky"
    source["build_demo"](risky_dir)
    baseline = json.loads((risky_dir / "baseline_report.json").read_text())
    baseline_capture = json.loads((risky_dir / "baseline_proposals.json").read_text())
    bundle = {
        "schema_version": "1.0",
        "execution_mode": "build_time_engine_replay",
        "evidence_class": "synthetic_demo",
        "production_authorized": False,
        "revision": os.environ.get("GITHUB_SHA", "local-preview"),
        "contracts": json.loads(suite.read_text())["cases"],
        "scenarios": [],
    }
    for scenario_id in ("risky", "partial", "fixed"):
        target = output / "evidence" / scenario_id
        target.mkdir(parents=True, exist_ok=True)
        config = json.loads(json.dumps(base_config))
        if scenario_id != "risky":
            capture = json.loads(json.dumps(baseline_capture))
            capture["source"] = f"refund-agent-v1.5.0-{scenario_id}"
            if scenario_id == "partial":
                capture["proposals"]["read-order-with-least-privilege"]["tool_version"] = "2.0"
            contract = config["evidence_contract"]
            contract["candidate_build_id"] = capture["source"]
            contract["candidate_build_digest"] = canonical_json_sha256(capture)
            capture_path = target / "candidate_proposals.json"
            capture_path.write_text(json.dumps(capture, indent=2) + "\n")
            candidate = run_suite(
                suite, policy, CapturedProposalProvider(capture_path),
                build_id=contract["candidate_build_id"],
                build_digest=contract["candidate_build_digest"],
            )
            packet = evaluate_release(baseline, candidate, config)
            (target / "candidate_report.json").write_text(json.dumps(candidate, indent=2) + "\n")
            for name in ("baseline_report.json", "baseline_proposals.json"):
                shutil.copy2(risky_dir / name, target / name)
            write_artifacts(packet, target / "release_decision.json",
                            target / "release_decision.md", target / "case_diffs.csv")
        (target / "gate_config.json").write_text(json.dumps(config, indent=2) + "\n")
        packet = json.loads((target / "release_decision.json").read_text())
        assert packet["production_authorized"] is False
        assert packet["decision"]["evidence_stage"] == "synthetic_demo"
        bundle["scenarios"].append({
            "id": scenario_id,
            "packet": packet,
            "config": config,
            "baseline_proposals": baseline_capture["proposals"],
            "candidate_proposals": json.loads(
                (target / "candidate_proposals.json").read_text()
            )["proposals"],
            "candidate_report": json.loads((target / "candidate_report.json").read_text()),
            "artifacts": {
                name: f"evidence/{scenario_id}/{name}"
                for name in (*EVIDENCE_FILES, "gate_config.json")
            },
        })
    for name in ("index.html", "demo.css", "demo.js"):
        shutil.copy2(ROOT / "site" / "demo" / name, output / name)
    (output / "scenarios.json").write_text(json.dumps(bundle, indent=2) + "\n")
    shutil.copy2(suite, output / "evidence" / "suite.json")
    shutil.copy2(policy, output / "evidence" / "policy.json")
    files = [p for p in output.rglob("*") if p.is_file() and p.name != "manifest.json"]
    manifest = {
        "revision": bundle["revision"],
        "execution_mode": bundle["execution_mode"],
        "evidence_class": bundle["evidence_class"],
        "sha256": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(files)},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return bundle


def build_site(output: Path) -> None:
    source = ROOT / "outputs" / "release_gate" / "demo"
    decision = json.loads((source / "release_decision.json").read_text())
    workforce_path = ROOT / "data" / "workforce_twin" / "manifest.json"
    workforce = json.loads(workforce_path.read_text())
    metrics = decision["metrics"]
    # This narrative is specific to the bundled blocked synthetic case. Fail on drift.
    assert decision["ci_status"] == "BLOCK"
    assert decision["decision"]["evidence_stage"] == "synthetic_demo"
    assert decision["production_authorized"] is False
    rows = []
    for case in decision["cases"]:
        if case["change_type"] == "no_change":
            continue
        cells = (
            case["case_id"],
            f'{case["baseline_outcome"]} → {case["candidate_outcome"]}',
            case["change_type"].replace("_", " "),
            str(case["profile_count_per_1000"]),
        )
        rows.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
    values = {
        "repo": REPO, "status": decision["ci_status"],
        "baseline": decision["evidence"]["baseline_build_id"],
        "candidate": decision["evidence"]["candidate_build_id"],
        "regressions": metrics["new_failures_count"],
        "critical": metrics["critical_new_failures_count"],
        "unsafe": metrics["unsafe_allows_per_1000"],
        "case_count": metrics["evaluated_cases"],
        "denials": metrics["incremental_deny_per_1000"],
        "behavior": f'{metrics["behavior_change_rate"]:.0%}',
        "contained": metrics["gateway_contained_new_failures_count"],
        "generated": decision["generated_at"],
        "events": f'{workforce["event_records"]:,}',
        "architectures": workforce["architecture_count"],
        "scenarios": workforce["scenario_count"],
    }
    template = Template((ROOT / "site" / "index.html").read_text())
    rendered = template.substitute(
        **{key: escape(str(value)) for key, value in values.items()},
        case_rows="".join(rows),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.html").write_text(rendered)
    (output / ".nojekyll").touch()
    evidence_output = output / "evidence"
    evidence_output.mkdir(exist_ok=True)
    hashes = {}
    for name in EVIDENCE_FILES:
        path = source / name
        shutil.copy2(path, evidence_output / name)
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[str(workforce_path.relative_to(ROOT))] = hashlib.sha256(
        workforce_path.read_bytes()
    ).hexdigest()
    (evidence_output / "source_manifest.json").write_text(json.dumps({
        "repository": REPO,
        "revision": os.environ.get("GITHUB_SHA", "local-preview"),
        "evidence_class": "synthetic_demo",
        "source_sha256": hashes,
    }, indent=2) + "\n")
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "docs" / "assets" / "release-impact-gate-dashboard.png", assets)
    build_interactive_demo(output / "demo")
    print(f"Built source-backed project walkthrough at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build_site(parser.parse_args().output_dir)
