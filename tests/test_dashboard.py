import json
import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).parents[1]
RELEASE_WORKFLOW_PAGES = {
    "Release Impact Gate",
    "Enterprise Action Gateway",
    "Enterprise Deployment Planner",
}


def _isolated_dashboard(tmp_path: Path) -> tuple[Path, Path]:
    app = tmp_path / "dashboard" / "app.py"
    app.parent.mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "dashboard" / "app.py", app)
    output = tmp_path / "outputs" / "release_gate" / "demo"
    output.mkdir(parents=True)
    return app, output


def _navigate_to(result: AppTest, page: str) -> AppTest:
    if page in RELEASE_WORKFLOW_PAGES:
        result.sidebar.radio[0].set_value("Release workflow").run()
        return result.sidebar.radio[1].set_value(page).run()
    result.sidebar.radio[0].set_value("Supporting research").run()
    return result.sidebar.selectbox[0].set_value(page).run()


def test_dashboard_default_page_renders_without_exception():
    app = Path(__file__).parents[1] / "dashboard" / "app.py"
    result = AppTest.from_file(str(app), default_timeout=20).run()
    assert not result.exception
    assert not result.error
    assert any("Release Impact Gate" in title.value for title in result.title)


def test_release_impact_gate_renders_source_backed_decision() -> None:
    app = Path(__file__).parents[1] / "dashboard" / "app.py"
    result = AppTest.from_file(str(app), default_timeout=20).run()
    assert not result.exception
    assert not result.error
    decision_path = (
        Path(__file__).parents[1]
        / "outputs"
        / "release_gate"
        / "demo"
        / "release_decision.json"
    )
    decision = json.loads(decision_path.read_text())
    rendered_markdown = "\n".join(element.value for element in result.markdown)
    assert decision["evidence"]["baseline_build_id"] in rendered_markdown
    assert decision["evidence"]["candidate_build_id"] in rendered_markdown
    assert f"CI {decision['ci_status']}" in rendered_markdown
    assert "NOT AUTHORIZED" in rendered_markdown
    assert decision["decision"]["evidence_stage"] in rendered_markdown
    metrics = {metric.label: metric.value for metric in result.metric}
    assert metrics["New contract failures"] == str(decision["metrics"]["new_failures_count"])
    assert metrics["New critical failures"] == str(
        decision["metrics"]["critical_new_failures_count"]
    )
    assert len(result.dataframe) == 2
    changed_cases = result.dataframe[1].value
    assert len(changed_cases) == len(decision["findings"]["behavior_changes"])
    assert set(changed_cases["case_id"]) == set(decision["findings"]["behavior_changes"])
    assert any(
        "Modeled impact under the declared synthetic profile" in heading.value
        for heading in result.subheader
    )
    assert any("Release threshold checks" in heading.value for heading in result.subheader)
    assert any("Evidence ladder and rollout plan" in heading.value for heading in result.subheader)


def test_release_impact_gate_fails_closed_when_artifacts_are_missing(tmp_path: Path) -> None:
    app, _ = _isolated_dashboard(tmp_path)
    result = AppTest.from_file(str(app), default_timeout=20).run()
    assert not result.exception
    assert result.error
    assert any("no decision has been inferred" in error.value for error in result.error)
    assert not result.metric


def test_release_impact_gate_fails_closed_on_conflicting_decision(tmp_path: Path) -> None:
    app, output = _isolated_dashboard(tmp_path)
    decision = json.loads(
        (PROJECT_ROOT / "outputs/release_gate/demo/release_decision.json").read_text()
    )
    decision["decision"]["ci_status"] = "PASS"
    (output / "release_decision.json").write_text(json.dumps(decision))
    shutil.copyfile(
        PROJECT_ROOT / "outputs/release_gate/demo/case_diffs.csv",
        output / "case_diffs.csv",
    )
    result = AppTest.from_file(str(app), default_timeout=20).run()
    assert not result.exception
    assert any("failed validation" in error.value for error in result.error)
    assert not result.metric


def test_release_impact_gate_fails_closed_on_invalid_case_flags(tmp_path: Path) -> None:
    app, output = _isolated_dashboard(tmp_path)
    shutil.copyfile(
        PROJECT_ROOT / "outputs/release_gate/demo/release_decision.json",
        output / "release_decision.json",
    )
    case_csv = (PROJECT_ROOT / "outputs/release_gate/demo/case_diffs.csv").read_text()
    (output / "case_diffs.csv").write_text(case_csv.replace("True", "unknown", 1))
    result = AppTest.from_file(str(app), default_timeout=20).run()
    assert not result.exception
    assert any("failed validation" in error.value for error in result.error)
    assert not result.metric


def test_release_impact_gate_rejects_json_csv_content_mismatch(tmp_path: Path) -> None:
    app, output = _isolated_dashboard(tmp_path)
    shutil.copyfile(
        PROJECT_ROOT / "outputs/release_gate/demo/release_decision.json",
        output / "release_decision.json",
    )
    case_csv = (PROJECT_ROOT / "outputs/release_gate/demo/case_diffs.csv").read_text()
    tampered = case_csv.replace(
        "Routine order lookup remains automated.",
        "Tampered description from a different artifact.",
        1,
    )
    (output / "case_diffs.csv").write_text(tampered)
    result = AppTest.from_file(str(app), default_timeout=20).run()
    assert not result.exception
    assert any("failed validation" in error.value for error in result.error)
    assert not result.metric


def test_release_impact_gate_rejects_missing_malformed_or_reused_build_digest(
    tmp_path: Path,
) -> None:
    app, output = _isolated_dashboard(tmp_path)
    source = json.loads(
        (PROJECT_ROOT / "outputs/release_gate/demo/release_decision.json").read_text()
    )
    shutil.copyfile(
        PROJECT_ROOT / "outputs/release_gate/demo/case_diffs.csv",
        output / "case_diffs.csv",
    )
    mutations = ("missing", "malformed", "reused")
    for mutation in mutations:
        decision = json.loads(json.dumps(source))
        evidence = decision["evidence"]
        if mutation == "missing":
            evidence.pop("candidate_build_digest")
        elif mutation == "malformed":
            evidence["candidate_build_digest"] = "D7A2700F"
        else:
            evidence["candidate_build_digest"] = evidence["baseline_build_digest"]
        (output / "release_decision.json").write_text(json.dumps(decision))

        result = AppTest.from_file(str(app), default_timeout=20).run()

        assert not result.exception
        assert any("failed validation" in error.value for error in result.error)
        assert not result.metric


def test_workforce_war_room_filters_and_playback_render() -> None:
    app = Path(__file__).parents[1] / "dashboard" / "app.py"
    result = AppTest.from_file(str(app), default_timeout=30).run()
    result = _navigate_to(result, "AI Workforce War Room")
    result.selectbox[0].set_value("normal_day").run()
    result.selectbox[1].set_value("solo_generalist").run()
    result.slider[0].set_value(240).run()
    assert not result.exception
    assert any(metric.label == "Safe completion" for metric in result.metric)
    assert any("Operating exceptions" in heading.value for heading in result.markdown)


def test_deployment_planner_renders_capacity_and_evidence_gates() -> None:
    app = Path(__file__).parents[1] / "dashboard" / "app.py"
    result = AppTest.from_file(str(app), default_timeout=30).run()
    result = _navigate_to(result, "Enterprise Deployment Planner")
    assert not result.exception
    assert any("Enterprise Deployment Planner" in title.value for title in result.title)
    assert any(metric.label == "Capacity-safe reviewers" for metric in result.metric)
    assert any("Import real Agent evaluation evidence" in heading.value for heading in result.subheader)


def test_dashboard_deep_evaluation_pages_render_without_exception():
    app = Path(__file__).parents[1] / "dashboard" / "app.py"
    for page in [
        "Release Impact Gate",
        "AI Workforce War Room",
        "Enterprise Deployment Planner",
        "Risk Dashboard",
        "Agent Mesh Explorer",
        "Stress Test",
        "Failure Trace",
        "Offline Model Evaluation",
        "Evaluation Task Suite",
        "Real LLM Evaluation",
        "Cross-model Evaluation",
        "Control Science",
        "Governance ROI",
        "Production Certification",
    ]:
        result = AppTest.from_file(str(app), default_timeout=30).run()
        result = _navigate_to(result, page)
        assert not result.exception, page
        assert any(page in title.value for title in result.title)
