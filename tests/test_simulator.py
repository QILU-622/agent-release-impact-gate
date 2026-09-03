import pandas as pd

from agent_mesh_risk_lab.benchmark import generate_benchmark
from agent_mesh_risk_lab.evaluation import compute_metrics
from agent_mesh_risk_lab.simulator import run_experiment


def test_runs_are_deterministic_and_auditable():
    task = generate_benchmark()[0]
    first = run_experiment(task, "policy_drop", ["context_envelope"])
    second = run_experiment(task, "policy_drop", ["context_envelope"])
    assert first == second
    assert len(first.trace) >= len(task.agent_chain)
    assert first.risk_probability <= run_experiment(task, "policy_drop", []).risk_probability


def test_targeted_control_reduces_paired_harm():
    tasks = generate_benchmark()
    untreated = [run_experiment(task, "tool_drift", []) for task in tasks]
    treated = [run_experiment(task, "tool_drift", ["tool_version_lock"]) for task in tasks]
    assert sum(run.harmful_action for run in treated) < sum(run.harmful_action for run in untreated)


def test_core_metrics_stay_in_valid_ranges():
    tasks = generate_benchmark()[:20]
    rows = []
    for task in tasks:
        payload = run_experiment(task, "external_injection", []).model_dump()
        payload.pop("trace")
        rows.append(payload)
    metrics = compute_metrics(pd.DataFrame(rows))
    bounded = [key for key in metrics if key.endswith("rate") or key == "rollback_coverage"]
    assert all(0 <= metrics[key] <= 1 for key in bounded)
    assert 0 <= metrics["mean_blast_radius"] <= 100
