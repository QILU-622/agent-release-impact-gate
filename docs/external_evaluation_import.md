# External evaluation import contract

The Enterprise Deployment Planner can summarize JSON artifacts from Promptfoo, DeepEval, or a
small canonical interchange format. The import exists to connect observed Agent behavior to an
operating-model decision; it does not silently convert a generic test failure into a safety event.

## Privacy boundary

The normalizer retains only:

- case count and workflow labels;
- pass, harmful-action, and normal-overblock coverage and rates;
- median and p95 latency;
- the detected source format.

Prompts, outputs, provider configuration, request/response headers, identities, API keys, tokens,
and arbitrary metadata are discarded. Dashboard uploads are processed in memory and are not
written to disk. The command-line path writes only the aggregate summary inside the deployment
evidence pack.

## Canonical JSON

\`\`\`json
{
  "cases": [
    {
      "case_id": "refund-001",
      "workflow": "refund",
      "case_type": "normal",
      "passed": true,
      "harmful_action": false,
      "overblocked": false,
      "latency_ms": 820
    }
  ]
}
\`\`\`

\`harmful_action\` must mean that an unauthorized high-impact terminal action was proposed or
executed under the customer's adjudication contract. \`overblocked\` should be labelled only for
normal cases where the expected business action was safe and authorized.

The provisional external-evidence gate requires at least 20 cases and harm labels on at least 80%
of them. This is a minimum coverage contract for a controlled shadow pilot, not a statistical
guarantee or production certification.

## Promptfoo and DeepEval

The auto-detector recognizes Promptfoo artifacts with a top-level \`results\` field and DeepEval
test runs with \`testCases\` or \`conversationalTestCases\`. Generic pass/fail and latency fields can
be summarized directly. For safety coverage, attach explicit \`harmful_action\`, \`overblocked\`,
\`workflow\`, and \`case_type\` labels in the case or case metadata; otherwise the readiness report
shows those labels as missing instead of guessing.

## Command line

\`\`\`bash
PYTHONPATH=src .venv/bin/python -m agent_mesh_risk_lab.deployment_planner \
  --project-root . \
  --external-eval /authorized/redacted/evaluation.json \
  --source auto
\`\`\`

The result is written to:

- \`outputs/reports/deployment_evidence.json\`
- \`outputs/reports/deployment_evidence.md\`

Do not import raw customer artifacts unless the customer has authorized the analysis and direct
identifiers and secrets have already been removed.
