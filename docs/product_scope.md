# Product scope: Agent Release Impact Gate

## The decision this product owns

Every time a team changes an Agent's model, prompt, tool contract, policy, or permission, the gate
answers one question:

> Compared with the approved build, what business actions changed, what operational burden did the
> change create, and how far may the candidate build proceed?

The output is a release decision, not a general safety score and not a certification.

## First customer and first workflow

The initial user is the engineering or product owner of a customer-support Agent that can propose
refund actions. The first release contract covers order lookup, confirmed refunds, unconfirmed
requests, untrusted retrieved content, machine permissions, and tool-version drift.

This narrow starting point is intentional. The project does not claim to be a universal governance
platform, workforce-management product, or independent auditor.

## Why another layer is useful

Public evaluation tools already provide important parts of the workflow:

- [Promptfoo CI/CD integration](https://github.com/promptfoo/promptfoo/blob/main/site/docs/integrations/ci-cd.md)
  runs evaluations in CI, applies thresholds, and exports JSON, HTML, and JUnit evidence.
- [DeepEval regression testing](https://github.com/confident-ai/deepeval/blob/main/docs/content/guides/guides-regression-testing-in-cicd.mdx)
  treats evaluation cases as tests that can fail a build.
- [Agenta's evaluation loop](https://github.com/Agenta-AI/agenta/blob/main/web/_reference/agenta-sdk/src/auto-agenta/00-overview.md)
  compares variants and promotes or stops them, while noting that high-quality test cases and
  annotations are the hard part.

Agent Release Impact Gate does not replace those tools. It borrows their CI and regression patterns,
but the paired gate currently accepts only its strict contract-runner report schema. Third-party
results must first be converted and enriched with the required action, policy, context, and build
identity fields. The gate then adds the business-action layer:

- pairs the approved and candidate builds on identical cases;
- distinguishes an approval bypass from a service denial or contained behavior drift;
- converts the configured case mix into impact per 1,000 cases;
- records whether the deterministic gateway contained a changed proposal;
- caps the next release stage according to the available evidence;
- returns a machine-readable CI result and a human-readable decision packet.

No third-party source code is copied into this repository.

## Required inputs

The minimum comparison requires:

1. an immutable baseline build identifier and independently pinned `sha256:` artifact digest;
2. an immutable candidate build identifier and a different pinned artifact digest;
3. the same versioned release-contract suite for both builds;
4. the same gateway policy, unless a policy migration is reviewed separately;
5. one result for every case in both builds;
6. a release profile that assigns criticality and a case-mix weight totaling 1,000;
7. trusted test context owned by the release contract, not asserted by the Agent.

In captured-proposal mode, the runner derives each digest from stable canonical JSON and refuses a
caller-supplied mismatch. With an HTTP adapter, CI must supply an artifact or provenance digest
from the immutable deployment being called; the local gate can check that it matches the pinned
release contract but cannot independently attest the external registry. A build label alone is not
accepted by the paired gate.

The bundled case mix is synthetic and exists only to demonstrate the calculation. A customer must
replace it with a reviewed distribution from its own workflow before interpreting the per-1,000
figures as an operational forecast.

## Decision semantics

| Result | Meaning | Permitted next action |
|---|---|---|
| `BLOCK` | A release threshold or comparison-integrity check failed | Do not merge or deploy |
| `OFFLINE_ONLY` | Tests passed, but evidence remains internal or synthetic | Expand and review the offline replay |
| `SHADOW` | External replay evidence passed | Run without executing real business actions |
| `CANARY` | A validated shadow pilot passed | Expose a bounded cohort with rollback controls |

The current implementation never grants an automatic production-wide approval. Production rollout
requires enterprise-owned telemetry, incident response, authority, and change-management controls
outside this repository.

## Real pilot that would validate value

The first credible pilot should compare two real builds of one refund Agent against 100–500
authorized, redacted historical cases. Two business reviewers should adjudicate expected actions
before either build is scored. The Agent runs in replay or shadow mode and never calls a live refund
tool during the evaluation.

The pilot succeeds only if the output changes or confirms a real release decision and the team says
it would reuse the gate for the next change. Record:

- newly detected release regressions;
- critical regressions missed by the existing manual process;
- manual preparation and review time before and during the pilot;
- false-block and approval-escalation rates;
- whether the candidate was released, revised, or stopped;
- the signed business owner's decision and evidence limitations.

Until that pilot exists, the repository demonstrates a testable product mechanism and a bounded
commercial hypothesis. It does not demonstrate customer demand, realized savings, or production
safety.
