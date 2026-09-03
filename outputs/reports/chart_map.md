# Chart Map

| Figure | Decision question | Source | Encodings | Validation note |
|---|---|---|---|---|
| W1 Operating-day Playback | Does work arrive faster than the selected organization can finish it? | One paired digital-twin operating day | line = cumulative arrivals/completions and open backlog | Arrivals and completions reconcile at each 15-minute bucket |
| W2 Case Outcomes | What happened to work by the selected playback minute? | Event log filtered to one design, scenario, seed, and minute | horizontal bars = mutually exclusive outcomes | Outcome counts reconcile to visible arrivals |
| W3 Architecture Tournament | Which design balances safe completion and direct operating cost? | Six-seed architecture summary | x = cost/safe completion, y = safe completion, size = automation, color = critical bypass | Same paired arrivals per scenario and seed; guardrail shown |
| W4 Capacity Pressure | Is the limiting resource the Agent pool or reviewer pool? | Busy minutes / scheduled capacity | horizontal bars with 100% reference | Values above 100% mean work spills beyond the operating day |
| 01 Agent Mesh Graph | Where can risk propagate? | Workflow catalog | node type, edge, call path | Reconciled to graph summary |
| 02 Failure Heatmap | Which workflow-stressor cells fail most? | Core results | color = incident rate | Same denominator per cell |
| 03 Cascading Failure | Which stressors propagate? | Core results | bar = cascade rate | Rate bounded 0-1 |
| 04 Safety vs Utility | Which controls trade safety for completion? | Core results | x = success, y = safety | One point per config |
| 05 Governance ROI | Which isolated control is efficient? | Paired ROI | bar = reduction/cost | Descriptive, not portfolio optimizer |
| 06 Budget Curve | How much measured risk can budget buy? | Empirical 64-grid | x = budget, y = reduction | Measured feasible portfolio |
| 07 Simulator Calibration | Does configured probability match incidents? | Core results | predicted vs observed | Synthetic diagnostic |
| 08 Certification | Which workflows clear thresholds? | Certification table | bar = score | Experimental thresholds |
| 09 Model Comparison | Which classifier ranks risk best? | Frozen test | PR-AUC and operating metrics | Task-group holdout |
| 10 Model Calibration | Do model probabilities match labels? | Frozen test | predicted vs observed | Raw and selected calibrator |
| 11 Ablation | Which feature families matter? | Frozen test | F1 delta | Same split and model family |
| 12 Generalization | Does performance transfer? | Strict holdouts | PR-AUC and recall | Zero task overlap |
| 13 Portfolio Frontier | Which joint controls are feasible? | All 64 portfolios | cost vs incidents | Pareto grid |
| 14 Shapley Value | What is each control's marginal value? | All coalitions | mean risk reduction | Reconciles to full coalition |
| 15 Interaction Heatmap | Which pairs complement or overlap? | Pair portfolios | divergence around zero | Orange to blue |
| 16 Seed Sensitivity | Is the result seed-fragile? | 12-seed runs | mean and range | 1,400 runs per seed/config |
| 17 Feature Access | How much does privileged information inflate risk scores? | Frozen test | PR-AUC and recall bars | Includes shuffled-label control |
| 18 Multi-task Comparison | Which evaluation tasks are learnable? | Frozen task groups | Macro-F1 grouped bars | Same majority comparator |
| 19 Governance Confusion | Which controls are confused? | Incident recommendation test | row-normalized matrix | Six empirical targets |
| 20 Failure Recall | Which root causes remain undetected? | Incident attribution test | recall bars | Stressor identity excluded |
| 21 Real LLM Prompt Comparison | How do zero-shot governance and few-shot examples change observed decisions? | 64 scenario triplets | grouped rates | Same model and scenario in all three arms |
| 22 Real LLM Stressor Heatmap | Under which attacks does the model choose unauthorized terminal actions? | 192 local-model decisions | color = harmful-action rate | Eight observations per condition-stressor cell |
| 23 Real LLM Paired Effects | What is the estimated governance-prompt effect and uncertainty? | scenario-paired decisions | point + 95% bootstrap interval | 2,000 scenario-cluster resamples |
| 24 LLM Reliability and Latency | Are outputs parseable and fast enough to operate? | local provider telemetry | schema-valid rate plus P50/P95 latency | Parse failures are visible, not discarded |
| 25 Simulator-to-LLM Validity | Does simulator risk rank observed real-model harm? | exact task-stressor join | AUROC point + 95% scenario-bootstrap interval | Random-ranking reference at 0.5 |
| 26 Few-shot Transition Decomposition | Are accuracy gains and safety losses the same scenarios? | 64 governed/few-shot pairs | horizontal scenario counts | Mutually exclusive categories reconcile to 64 |
| 27 Few-shot Terminal-action Shift | Is increased harm explained by direct action copying? | exact development examples + decisions | grouped terminal-action rates | Demonstration presence shown per workflow |
