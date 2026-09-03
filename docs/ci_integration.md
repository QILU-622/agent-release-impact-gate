# Production CI integration

The repository's own workflow is deliberately a **self-test**: it rebuilds a known risky demo and
passes when the mechanism correctly returns `BLOCK`. A customer release workflow has the opposite
purpose. It compares evidence generated for the exact approved and candidate artifacts, and it
fails the candidate job when the result is `BLOCK`.

## Trust boundary

Keep these inputs outside the candidate developer's unilateral control:

1. The approved-build report should come from a protected branch, artifact registry, or release
   record—not from a file rewritten in the candidate pull request.
2. Both reports must record immutable artifact digests, not just human-readable build labels.
3. The gate configuration, case weights, criticality, and thresholds should require release-owner
   review through CODEOWNERS or an equivalent policy.
4. Pin this action to a reviewed commit SHA. Do not use a floating branch name for a production
   release decision.
5. Retain the decision JSON, Markdown memo, CSV differences, input reports, and source artifact
   attestations under the same release identifier.

## Consumer workflow pattern

The Agent test jobs are customer-specific. They must deploy or call the two exact artifacts and
produce the two regression reports before invoking this action.

```yaml
jobs:
  compare-agent-builds:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      # Customer-owned steps generate these from immutable image/model/prompt digests.
      - name: Produce approved and candidate evidence
        run: ./customer-ci/run-paired-agent-replay.sh

      - name: Decide maximum rollout stage
        id: release-gate
        uses: QILU-622/agent-release-impact-gate/.github/actions/release-impact-gate@<reviewed-commit-sha>
        with:
          baseline-report: evidence/approved.json
          candidate-report: evidence/candidate.json
          gate-config: release-policy/refund-agent.json
          output-directory: release-decision

      - name: Retain the decision packet
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: agent-release-decision
          path: release-decision/
          if-no-files-found: error
```

The candidate must not be able to make the check green by changing its own baseline, thresholds,
case mix, digest, or gate implementation. Branch protection and artifact provenance remain
customer responsibilities; this project verifies declared evidence but cannot create that trust
boundary on its own.
