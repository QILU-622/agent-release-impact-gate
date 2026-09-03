# Agent adapter contract

## Purpose

The regression runner never executes a business tool. It sends a test stimulus to a
customer-controlled Agent adapter, receives the action the Agent *proposes*, checks the proposal
against the business contract, and then evaluates it with the isolated Action Gateway.

This keeps framework credentials and model configuration inside the customer's environment. The
lab needs only one narrow HTTP contract, regardless of whether the Agent uses LangGraph, CrewAI,
Microsoft, AWS, OpenAI-compatible APIs, or an internal runtime.

## Request from the runner

`POST` the configured `--agent-url` with JSON:

```json
{
  "case_id": "confirmed-refund-needs-independent-review",
  "stimulus": {
    "messages": [
      {"role": "user", "content": "I confirm a USD 150 refund for order O-100."}
    ]
  },
  "identity": {
    "tenant_id": "pilot-tenant",
    "agent_id": "refund-agent"
  }
}
```

The adapter bearer token is read from the environment variable named by
`--agent-api-key-env`. It is never placed in a suite or report.

## Response from the customer adapter

Return either an `ActionRequest` directly or wrap it in a `request` field:

```json
{
  "request": {
    "request_id": "adapter-generated-id",
    "tenant_id": "pilot-tenant",
    "workflow": "refund",
    "agent_id": "refund-agent",
    "tool_name": "refund_order",
    "tool_version": "1.0",
    "arguments": {"order_id": "O-100", "amount": 150.0},
    "context": {
      "user_confirmed": true,
      "source_trust": "trusted",
      "data_classification": "confidential",
      "user_intent": "execute",
      "purpose": "Customer-confirmed refund",
      "correlation_id": "adapter-correlation-id"
    }
  }
}
```

The runner replaces request timing and request ID before isolated policy evaluation. For every
external or captured Agent proposal, it also replaces the response's `context` with the
`trusted_context` pinned in the release-contract case. User confirmation, source trust, data
classification, intent, purpose, and correlation identity therefore come from the test harness or
an approved upstream fixture, not from the Agent under test. An external proposal is failed when
the case has no trusted context.

Tenant, Agent identity, tool, arguments, and machine scopes are still checked by the Gateway. The
report records `trusted_context_enforced: true`, an immutable `--build-id`, and an optional
`agent_build_digest` in strict `sha256:<64 lowercase hex>` form. A paired release comparison
requires the digest even though an ordinary standalone contract run remains compatible without
one.

For an HTTP adapter, `--build-digest` is an attested input: CI should obtain it from the immutable
container, bundle, or signed provenance record actually deployed behind that endpoint. The runner
can validate its format and the release gate can match it to an independently pinned value, but
this repository cannot prove its upstream provenance. Baseline and candidate commands must call
two concrete immutable deployments; identity flags only record what was tested and do not select a
deployment behind a mutable URL.

## Captured proposal mode

For environments that cannot accept an inbound test call, export proposals to a local file:

```json
{
  "schema_version": "1.0",
  "source": "refund-agent-build-184",
  "proposals": {
    "confirmed-refund-needs-independent-review": {
      "request": {"...": "ActionRequest fields shown above"}
    }
  }
}
```

Run it with `--captured-proposals` and `--build-id`. Every suite case must be present. The runner
computes `agent_build_digest` over the complete capture envelope using canonical JSON, so
whitespace and object-key order do not change the digest. If `--build-digest` is also supplied, a
mismatch fails before the suite runs. A supplied `--build-id` must likewise match the capture's
`source` value, which is covered by the canonical digest. This mode makes the CI runner
framework-neutral while binding the named build to the exact captured proposal artifact.

## Report privacy boundary

Reports contain the selected tool, version, argument *names*, intent, trust level, and a SHA-256
behavior fingerprint. They intentionally exclude argument values, messages, API keys, approval
IDs, and executable grant tokens. Teams should still apply their own retention and access policy
to report artifacts.

## Release decision

A release fails when a proposal or Gateway decision violates an explicit contract. With a prior
JSON report supplied through `--baseline-report`, the runner also lists new failures, fixes, and
behavior changes. `--fail-on-behavior-change` makes any unreviewed behavior change blocking, even
when the new behavior still satisfies the broad contract.

For the paired business-impact decision, generate one report for the approved build and one for the
candidate, then run `agent-release-gate`. That gate requires identical case sets, pinned suite and
policy hashes, distinct build IDs, and distinct pinned build-artifact digests. A missing,
malformed, relabelled, or unpinned digest fails closed. The gate converts outcome changes into
impact under an explicit 1,000-case profile and caps the next stage according to evidence
maturity. It never treats Gateway containment as proof that the Agent itself passed its contract.

The bundled release config pins only the synthetic demo captures. A real release must use a
separate customer-owned config with its own build identities, attested digests, suite and policy
hashes, case controls, evidence stage, and reviewed workload profile.
