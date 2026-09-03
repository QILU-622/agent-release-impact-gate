# Security Policy

## Supported versions

Security fixes are applied to the latest version on the default branch. Older
commits and generated demonstration artifacts are not maintained separately.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through the repository's
**Security** tab by opening a private vulnerability report. Do not include
secrets, customer records, exploit details, or other sensitive information in
a public issue.

Include, when possible:

- the affected component and version or commit;
- the conditions required to reproduce the issue;
- the potential effect on authorization, audit integrity, or data exposure;
- a minimal reproduction that uses synthetic data; and
- any mitigation already tested.

If private vulnerability reporting is not enabled, open a public issue that
only asks the maintainer to enable a private reporting channel. Do not disclose
technical details in that issue.

## Sensitive-data boundary

This repository contains synthetic examples and generated summaries. Do not
submit real customer prompts, Agent outputs, identifiers, approval records,
credentials, API keys, production policies, database files, or unredacted
pilot exports in an issue or pull request.

The project's simulations, release decisions, and example policies are not a
security certification. Anyone deploying the software remains responsible for
independent threat modelling, access control, secret management, monitoring,
and validation against their own operating environment.
