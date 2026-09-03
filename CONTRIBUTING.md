# Contributing

Thank you for helping improve Agent Release Impact Gate. Contributions should keep
the project reproducible, evidence-led, and safe to publish.

## Set up a development environment

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,postgres]"
```

Copy `.env.example` to `.env` only when local services require it. Use newly
generated local secrets; never commit the resulting `.env` file.

## Before opening a pull request

Run the same core checks as the automated release gate:

```bash
ruff check src tests scripts dashboard
pytest -q
agent-mesh-regression configs/regression/refund_action_contracts.json \
  --policy configs/enterprise/policy.json \
  --json-report outputs/regression/refund.json \
  --junit-report outputs/regression/refund.junit.xml
PYTHONPATH=src python scripts/build_release_demo.py
```

Then confirm that the pull request:

- explains the business or safety problem and the expected behaviour;
- adds or updates tests for material behaviour changes;
- distinguishes simulated findings from external or production evidence;
- documents new assumptions, metrics, and decision thresholds;
- avoids unrelated generated-file churn; and
- contains no secrets, real pilot data, personal data, or customer content.

## Data and evidence contributions

Use synthetic, anonymised fixtures that are small enough to review. Pilot files
must follow the examples in `pilot/`, but real pilot exports stay outside this
repository. Large row-level experiment outputs and fitted model files should be
regenerated locally and are intentionally ignored.

Compact aggregate tables, manifests, reports, and the release-gate demo may be
committed when they make a result independently inspectable. State the command,
configuration, seed, and source category needed to reproduce them.

## Code style

- Keep public interfaces typed and focused.
- Prefer deterministic tests and seeded simulations.
- Do not weaken a policy or release threshold merely to make a test pass.
- Preserve fail-closed behaviour for authorization and evidence checks.
- Keep error messages useful without exposing secrets or sensitive payloads.

## Pull-request scope

Keep each pull request focused on one change. A proposed change to a metric,
policy, schema, or release decision should describe compatibility implications
and include a migration note when existing users could be affected.

Security issues should follow `SECURITY.md` and must not be reported through a
public pull request.
