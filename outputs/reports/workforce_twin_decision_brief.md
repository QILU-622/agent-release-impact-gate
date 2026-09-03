# AI Workforce Digital Twin: decision brief

## Decision

Use this experiment to compare operating-model hypotheses before a pilot, not to forecast a company's realized ROI or staffing need.

## Normal-day comparison

| Architecture | Safe completion | p95 cycle | Critical bypass | Cost / safe completion |
|---|---:|---:|---:|---:|
| Tiered human-AI | 76.9% | 25.3 min | 4.2% | 0.64 units |
| Manager + specialists | 76.3% | 32.6 min | 6.5% | 0.74 units |
| Gateway-governed hybrid | 70.1% | 25.6 min | 0.6% | 0.72 units |
| Specialist mesh | 67.9% | 183.4 min | 28.1% | 0.63 units |
| Solo generalist | 56.0% | 182.4 min | 42.2% | 0.72 units |

## Scenario recommendations

- **Black Friday surge**: Gateway-governed hybrid (highest weighted decision score; no architecture passed every guardrail).
- **Model migration drift**: Tiered human-AI (highest weighted decision score; no architecture passed every guardrail).
- **Normal day**: Gateway-governed hybrid (highest weighted decision score among guardrail-passing architectures).
- **Prompt attack wave**: Gateway-governed hybrid (highest weighted decision score; no architecture passed every guardrail).
- **Reviewer shortage**: Gateway-governed hybrid (highest weighted decision score; no architecture passed every guardrail).
- **Tool outage**: Tiered human-AI (highest weighted decision score; no architecture passed every guardrail).

## Reviewer capacity plan for the selected architecture

| Scenario | Selected design | Current reviewers | Capacity-safe reviewers | Change | p95 review wait |
|---|---|---:|---:|---:|---:|
| Black Friday surge | Gateway-governed hybrid | 2 | 8 | +6 | 0.3 min |
| Model migration drift | Tiered human-AI | 2 | 3 | +1 | 5.3 min |
| Normal day | Gateway-governed hybrid | 2 | 3 | +1 | 5.2 min |
| Prompt attack wave | Gateway-governed hybrid | 2 | 3 | +1 | 7.2 min |
| Reviewer shortage | Gateway-governed hybrid | 2 | 7 | +5 | 4.5 min |
| Tool outage | Tiered human-AI | 2 | 3 | +1 | 5.1 min |

## Evidence boundary

All arrivals, service times, costs, capacity, model-quality multipliers, and crises are explicit synthetic assumptions in `configs/workforce_twin.json`. Replace them with observed customer process data before making an operating or investment decision. The safety component reuses the project's deterministic risk simulator; it is not a calibrated probability of real failure.
