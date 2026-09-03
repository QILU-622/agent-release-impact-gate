# Competitive design notes

Version 2.0 reviewed public GitHub projects for product patterns, not for source-code reuse.

| Public project | Pattern carried forward | Boundary retained here |
|---|---|---|
| Namanau9/ai-workforce-simulator | deterministic operations simulation, explicit assumptions, scenario comparison | this project models LLM Agent behavior, human review, and execution governance together |
| Azure/agentops | stable readiness checks separated into ready, warning, and blocked evidence | readiness is conservative and remains blocked without external workflow evidence |
| promptfoo/promptfoo | CI quality gates, JSON/JUnit artifacts, and secret-aware evidence handling | the release gate consumes bounded results and adds paired business-action impact rather than reimplementing general LLM scoring |
| confident-ai/deepeval | test cases as CI-blocking regression assertions | contract failures are translated into approval bypass, service denial, review-load, or contained-change consequences |
| Agenta-AI/agenta | compare variants, then promote, stop, and feed production traces back into tests | evidence maturity caps the next stage; a passing synthetic replay cannot authorize production |
| NVIDIA-NeMo/Guardrails and Giskard-AI/giskard-oss | layered controls and adversarial evaluation | controls are inputs to an operating decision, not substitutes for capacity and workflow design |

The differentiated product decision is deliberately narrow: **should this specific Agent build move
past the approved build, and what business-action changes explain that decision?** Workforce
simulation, model sensitivity, and runtime authorization are supporting evidence rather than three
separate products.

No third-party code was copied into the implementation. Repository links and current behavior
should be rechecked before using this note as a market statement.
