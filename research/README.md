# Supporting research track

This track is the research foundation of **Agent Release Impact Gate**. It studies how
tool failures, policy controls, reviewer capacity, and model choice affect Agent operations.
The deterministic release gate itself does not train or load a machine-learning model.

## Installation and reproduction

From the repository root:

```bash
python -m pip install -e ".[research]"
agent-release-research-workforce-twin --help
agent-release-research-run --help
```

Add `[dashboard,research]` to explore the **Supporting research** workspace in Streamlit,
or `[research,notebook]` to run the notebooks. The deployment planner also needs `[research]`.
Model weights and the full workforce event log are generated locally and are not committed.

## Evidence map

| Study | Implementation | Reproducible evidence |
|---|---|---|
| Human–Agent workforce and review queues | [configuration](../configs/workforce_twin.json) · [`workforce_twin.py`](../src/agent_mesh_risk_lab/workforce_twin.py) | [manifest](../data/workforce_twin/manifest.json) · [decision brief](../outputs/reports/workforce_twin_decision_brief.md) · [reviewer-capacity plan](../data/workforce_twin/reviewer_capacity_plan.csv) |
| Offline risk classification | [`modeling.py`](../src/agent_mesh_risk_lab/modeling.py) | [evaluation manifest](../data/evaluation/evaluation_manifest.json) · [model comparison](../data/evaluation/model_comparison.csv) · [bootstrap intervals](../data/evaluation/bootstrap_confidence_intervals.csv) · [notebook](../notebooks/01_offline_model_evaluation.ipynb) |
| Control combinations and interactions | [`portfolio_experiments.py`](../src/agent_mesh_risk_lab/portfolio_experiments.py) | [empirical recommendations](../data/control_science/empirical_recommendations.json) · [Shapley effects](../data/control_science/control_shapley.csv) · [notebook](../notebooks/02_control_portfolio_science.ipynb) |
| Model sensitivity | [`multi_model_evaluation.py`](../src/agent_mesh_risk_lab/multi_model_evaluation.py) | [Llama manifest](../data/multi_model/llama3.2_3b/manifest.json) · [Qwen manifest](../data/multi_model/qwen2.5_3b-instruct/manifest.json) · [paired effects](../data/multi_model/qwen2.5_3b-instruct/paired_effects.csv) |
| Deployment evidence and reviewer capacity | [`deployment_planner.py`](../src/agent_mesh_risk_lab/deployment_planner.py) | [readiness report](../outputs/reports/deployment_evidence.md) · [pilot protocol](../pilot/README.md) · [external-evaluation contract](../docs/external_evaluation_import.md) |

### Audit the 37,135-event result

The workforce claim now has a direct source-to-evidence trail:

1. [`configs/workforce_twin.json`](../configs/workforce_twin.json) defines the five architectures,
   six scenarios, workload assumptions, costs, and capacity guardrails.
2. [`workforce_twin.py`](../src/agent_mesh_risk_lab/workforce_twin.py) executes the paired,
   six-seed simulation.
3. [`manifest.json`](../data/workforce_twin/manifest.json) records 180 operating-day runs,
   37,135 synthetic events, the configuration SHA-256, and the claim boundary.
4. The [decision brief](../outputs/reports/workforce_twin_decision_brief.md) and
   [reviewer-capacity plan](../data/workforce_twin/reviewer_capacity_plan.csv) expose the
   decision-facing summaries without presenting them as observed customer outcomes.

The full event log is intentionally regenerated locally rather than committed. The checked-in
manifest and summaries make the design, scale, provenance, and interpretation boundary reviewable.

The two local model families are documented in the linked model-sensitivity manifests. They are
distinct from the statistical classifiers used in offline risk classification.

XGBoost and scikit-learn support the offline classification experiments; NetworkX supports
workflow graphs. None is required by the release-gate or contract-runner installation.

## Package compatibility

`agent_mesh_risk_lab` is the historical Python namespace, retained so notebooks, integrations,
and saved model references remain importable. The public product and distribution are named
`agent-release-impact-gate`. Current commands start with `agent-release-`; research commands
add `research-`. The original `agent-mesh-*` commands remain compatibility aliases.

Research implementations stay in the installed Python package so existing imports continue to
work; this directory is their documented track entry point. They load only when selected.

## Interpretation

Synthetic scenario results support comparisons under explicit assumptions. They do not establish
customer demand, measured savings, staffing commitments, or production safety. Counts describe
the checked-in experiment manifest, not a customer deployment.
