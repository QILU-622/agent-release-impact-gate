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
| Human–Agent workforce and review queues | [`workforce_twin.py`](../src/agent_mesh_risk_lab/workforce_twin.py) | [`manifest.json`](../data/workforce_twin/manifest.json): 37,135 synthetic events, five architectures, six scenarios, six seeds |
| Offline risk classification | [`modeling.py`](../src/agent_mesh_risk_lab/modeling.py) | [`data/evaluation/`](../data/evaluation/), [`notebooks/`](../notebooks/) |
| Control combinations and interactions | [`portfolio_experiments.py`](../src/agent_mesh_risk_lab/portfolio_experiments.py) | [`data/control_science/`](../data/control_science/) |
| Model sensitivity | [`multi_model_evaluation.py`](../src/agent_mesh_risk_lab/multi_model_evaluation.py) | [`data/multi_model/`](../data/multi_model/) |
| Deployment evidence and reviewer capacity | [`deployment_planner.py`](../src/agent_mesh_risk_lab/deployment_planner.py) | [`pilot/`](../pilot/), [`external evaluation contract`](../docs/external_evaluation_import.md) |

The two local model families are documented in the model-sensitivity results. They are distinct
from the statistical classifiers used in offline risk classification.

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
