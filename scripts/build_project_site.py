"""Build a static GitHub Pages walkthrough from versioned demo evidence (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    print(f"Built source-backed project walkthrough at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    build_site(parser.parse_args().output_dir)
