# Executed Audit Notebooks

- `01_offline_model_evaluation.ipynb` reconciles split isolation, model metrics, calibration,
  feature-access policies, negative controls, four evaluation tasks, ablations, unseen stressors,
  decision regret, and cross-workflow transfer.
- `02_control_portfolio_science.ipynb` reconciles all 64 portfolios, the constrained optimum,
  Shapley efficiency, pair interactions, and 12-seed sensitivity.
- `03_real_llm_evaluation.ipynb` verifies the 64 scenario triplets, zero few-shot leakage,
  aggregate metrics, both paired comparisons, attack-surface errors, and local inference cost.
- `04_mechanism_and_transfer_audit.ipynb` decomposes scenario-level few-shot transitions,
  falsifies simple terminal-action copying, and tests whether simulator risk ranks observed LLM
  harmful actions.

All four notebooks read persisted pipeline artifacts and have been executed top-to-bottom. To
rebuild their structure, run `python3 scripts/build_notebooks.py`; to refresh outputs, execute
them from the repository with Jupyter.
