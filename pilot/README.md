# Anonymous Agent pilot kit

This folder prepares a real company-scoped shadow pilot without pretending that synthetic records
are customer evidence. The pilot is complete only when an external participant supplies
anonymized cases and Agent proposals, two reviewers adjudicate the expected behavior, and the
manifest attestations are set truthfully.

## Minimum pilot

1. Select one workflow and one named business owner.
2. Export 20-50 recent cases after the company removes direct identifiers and secrets.
3. Assign stable synthetic case IDs; retain the re-identification key only inside the company.
4. Have two reviewers independently label expected tool, expected outcome, and risk tier.
5. Resolve disagreements before running the Agent.
6. Run the current Agent in shadow mode. Never invoke the real business tool.
7. Compare the Agent proposal, Gateway decision, latency, and manual testing time with the existing
   release process.
8. Report every case and every disagreement; do not select only favorable examples.

## Files

- `pilot_manifest.example.json`: ownership, source, and anonymization attestations.
- `pilot_cases.example.csv`: anonymized cases and adjudicated acceptance criteria.
- `pilot_kpi_baseline.example.csv`: pre-pilot process measurements; blanks remain unknown.
- `scripts/validate_pilot_data.py`: structural and basic PII readiness check.

## Evidence rule

Passing the validator means the data package meets the project's minimum intake contract. It does
not prove legal anonymization, production safety, causal impact, or customer willingness to pay.
Those conclusions require the company's review and observed pilot results.

After the company has supplied an authorized, redacted pilot package, validate it and write the
aggregate report to the location consumed by the deployment evidence pack:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_pilot_data.py \
  --cases /authorized/redacted/pilot_cases.csv \
  --manifest /authorized/redacted/pilot_manifest.json \
  --output pilot/pilot_validation_report.json
```

Do not use the bundled `*.example.*` files as evidence of a real pilot. Keep the source cases in
the company's approved storage; this repository needs only the validator's aggregate report.

## Proposed decision metrics

- Agent tool-selection agreement with adjudicated expected tool.
- Gateway outcome agreement with adjudicated `allow`, `deny`, or `review` decision.
- Critical missed-violation count.
- Legitimate-action false-block rate.
- Review share and reviewer turnaround.
- Test preparation and execution hours before versus during the pilot.
- Version regressions caught before release.

Go/no-go thresholds must be agreed before looking at results. A reasonable starting requirement is
zero known approval bypasses and zero cross-tenant authorization, but acceptable false-block,
latency, and review-load thresholds depend on the customer's workflow.
