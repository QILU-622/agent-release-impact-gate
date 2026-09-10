# Release Lab demo guide

Open the [English demo](https://qilu-622.github.io/agent-release-impact-gate/demo/)
or [中文演示](https://qilu-622.github.io/agent-release-impact-gate/demo/?lang=zh).
No account, API key or software installation is required.

## Two-minute walkthrough / 两分钟讲解

1. **Risky upgrade / 有风险的升级.** Click the decision button. The result is BLOCK.
   Open “Refund without confirmation” to see the amount change from 50 to 5,000.
   The gateway still denies execution, but the proposed action breaks the contract.
   中文：没有真的转出钱，不代表 AI 的行为没有变坏。
2. **Partial repair / 只修复安全问题.** The old result clears when you change versions.
   Inspect the new decision: zero critical regressions, but still BLOCK. The ordinary lookup
   uses an unreviewed tool version, so it becomes a denial. Under the illustrative mix, that
   adds 550 denials per 1,000 tasks and exceeds the configured limit of 100.
   中文：退款问题修好了，但客户连订单都查不了，这种升级也不能发。
3. **Complete repair / 完整修复版.** All six proposals now match the baseline. The gate
   permits only OFFLINE_ONLY because this is synthetic evidence. Download the decision JSON,
   differences CSV and pinned config. 中文：测试通过也不等于能上线，系统要明确说出证据的边界。

## What actually runs

`scripts/build_project_site.py` runs the real `CapturedProposalProvider`, `run_suite`,
`ActionGateway` and `evaluate_release` implementations during the site build. Each candidate
gets a distinct source identity and canonical SHA-256 digest, pinned before evaluation. No
decision is hand-authored, and the browser does not calculate scores or change authorization.
The original risky artifact is reproduced by `scripts/build_release_demo.py`.

The browser loads `demo/scenarios.json` and displays the selected build-time result. It deliberately
does not show fake live-execution progress. Changing the candidate clears stale evidence.
Each scenario has its own reports, captured proposals, gate config, decision memo and differences
CSV under `demo/evidence/`. Shared suite/policy inputs and `demo/manifest.json` make the site auditable.
Execution timestamps and timings can differ between builds; decisions and business metrics should not.

## Limits and the next real integration

- Six synthetic contract cases are not representative customer traffic, a model benchmark or a
  production-safety certificate. Per-1,000 figures use a declared illustrative case mix.
- “Repair” means restoring captured tool proposals, not proving that an independently running
  model or prompt has been repaired. No external LLM is called in this demo.
- To evaluate a real Agent, replace captures with an authorized export from that Agent and
  independently approve the baseline, contracts, policy, build identities and workload profile.
  Preserve unknowns and remove customer identifiers before publication.
- Do not relabel evidence to unlock SHADOW or CANARY. Those stages require the corresponding
  external replay or validated shadow-pilot evidence, controls and human approval.

## Reproduce locally

From the repository root with Python 3.12+:

```bash
python -m pip install -e ".[dev]"
python scripts/build_project_site.py --output-dir _site
python -m pytest -q tests/test_project_site.py
python -m http.server 8000 --bind 127.0.0.1 --directory _site
```

Open `http://127.0.0.1:8000/demo/?lang=zh`. To publish, the existing Project page workflow
installs only the lightweight core, executes all three scenarios and deploys the generated site.
