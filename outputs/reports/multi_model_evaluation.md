# Multi-model Agent evaluation

## Scope

Compared qwen2.5:3b-instruct, llama3.2:3b on the same 64 synthetic scenarios under three prompt modes.
Each model contributed 192 schema-constrained, decision-only observations.

## Governed prompt effect versus baseline

| Model | Harm change (paired 95% CI) | Accuracy change | Over-blocking change |
|---|---:|---:|---:|
| qwen2.5:3b-instruct | -39.06% [-51.56%, -26.56%] | -1.56% | +43.75% |
| llama3.2:3b | -67.19% [-78.12%, -56.25%] | -4.69% | +81.25% |

## Cross-model interpretation

Under the governed prompt, harmful-action rates ranged from 0.00% to 9.38%, while normal-case over-blocking ranged from 62.50% to 93.75%.
Pairwise exact-action agreement under governance averaged 18.75%; agreement on whether the action was harmful averaged 90.62%. The models can therefore look similarly safe in an aggregate while choosing materially different actions.
Few-shot harmful-action rates spanned 0.00% to 37.50%. Prompt examples are not a portable authorization mechanism across these model families.

## Evidence boundary

The comparison isolates model-family sensitivity better than the earlier single-model run, but it still uses synthetic English tasks, one deterministic sample per condition, local quantized models, and no real business-tool execution. It must not be presented as a leaderboard, safety certificate, or measured enterprise impact.
