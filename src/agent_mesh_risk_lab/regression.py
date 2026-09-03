"""Framework-neutral release contracts for agent-proposed business actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel, Field, ValidationError, model_validator

from .action_gateway import ActionGateway, AuditStore, GatewayError
from .enterprise_schema import ActionContext, ActionRequest, Principal

_BUILD_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ExpectedProposal(BaseModel):
    """Business-level assertions about the action selected by the Agent."""

    tool_name: str | None = None
    tool_version: str | None = None
    arguments_include: dict[str, Any] = Field(default_factory=dict)


class ExpectedDecision(BaseModel):
    outcome: Literal["allow", "deny", "review"]
    reason_codes: list[str] = Field(default_factory=list)
    obligations: list[str] = Field(default_factory=list)
    proposal: ExpectedProposal = Field(default_factory=ExpectedProposal)


class RegressionCase(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=500)
    stimulus: dict[str, Any] = Field(default_factory=dict)
    principal: Principal
    trusted_context: ActionContext | None = None
    request: ActionRequest | None = None
    expect: ExpectedDecision

    @model_validator(mode="after")
    def identities_must_match(self) -> RegressionCase:
        if self.request and self.principal.tenant_id != self.request.tenant_id:
            raise ValueError("principal and request tenant_id must match")
        return self


class RegressionSuite(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=500)
    cases: list[RegressionCase] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_must_be_unique(self) -> RegressionSuite:
        case_ids = [case.id for case in self.cases]
        request_ids = [case.request.request_id for case in self.cases if case.request]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("regression case ids must be unique")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request_ids must be unique within a regression suite")
        return self


class ProposalProvider(Protocol):
    source: str

    def propose(self, case: RegressionCase) -> ActionRequest: ...

    def close(self) -> None: ...


class EmbeddedProposalProvider:
    source = "embedded-suite-fixtures"

    def propose(self, case: RegressionCase) -> ActionRequest:
        if case.request is None:
            raise ValueError(f"case {case.id} has no embedded request")
        return case.request

    def close(self) -> None:
        return None


class CapturedProposalProvider:
    """Replay tool proposals exported by any Agent framework."""

    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text())
        raw_proposals = payload.get("proposals")
        if not isinstance(raw_proposals, dict):
            raise TypeError("captured proposal file requires a proposals object keyed by case id")
        self._proposals = raw_proposals
        declared_source = payload.get("source")
        self.declared_build_id = (
            declared_source.strip()
            if isinstance(declared_source, str) and declared_source.strip()
            else None
        )
        self.source = self.declared_build_id or path.name
        self.sha256 = _file_digest(path)
        # Bind the report to the semantic JSON artifact rather than its whitespace or
        # object-key ordering.  The complete capture envelope is included, so its source
        # identity and proposals cannot be relabelled without changing this digest.
        self.build_digest = canonical_json_sha256(payload)

    def propose(self, case: RegressionCase) -> ActionRequest:
        if case.id not in self._proposals:
            raise ValueError(f"captured proposal missing case {case.id}")
        payload = self._proposals[case.id]
        if isinstance(payload, dict) and "request" in payload:
            payload = payload["request"]
        return ActionRequest.model_validate(payload)

    def close(self) -> None:
        return None


class HttpProposalProvider:
    """Call a customer-controlled adapter that returns an ActionRequest JSON object."""

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(headers=headers, timeout=timeout_seconds)
        self._owns_client = client is None
        self.url = url
        self.source = "live-agent-http-adapter"

    def propose(self, case: RegressionCase) -> ActionRequest:
        response = self._client.post(
            self.url,
            json={
                "case_id": case.id,
                "stimulus": case.stimulus,
                "identity": {
                    "tenant_id": case.principal.tenant_id,
                    "agent_id": case.principal.subject_id,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "request" in payload:
            payload = payload["request"]
        return ActionRequest.model_validate(payload)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Return a stable, prefixed SHA-256 digest for a JSON-compatible artifact."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _validate_build_digest(value: str) -> str:
    if not _BUILD_DIGEST.fullmatch(value):
        raise ValueError("build digest must match sha256:<64 lowercase hexadecimal characters>")
    return value


def _proposal_fingerprint(request: ActionRequest) -> str:
    """Hash behavior while excluding request, tenant, correlation, and timing metadata."""

    context = request.context.model_dump(mode="json")
    context.pop("correlation_id", None)
    material = {
        "workflow": request.workflow,
        "tool_name": request.tool_name,
        "tool_version": request.tool_version,
        "arguments": request.arguments,
        "context": context,
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _proposal_summary(request: ActionRequest) -> dict[str, Any]:
    """Return comparison fields without copying argument values into CI artifacts."""

    return {
        "workflow": request.workflow,
        "tool_name": request.tool_name,
        "tool_version": request.tool_version,
        "argument_keys": sorted(request.arguments),
        "user_intent": request.context.user_intent,
        "source_trust": request.context.source_trust,
        "behavior_fingerprint": _proposal_fingerprint(request),
    }


def _case_result(
    case: RegressionCase,
    request: ActionRequest | None,
    actual: dict[str, Any] | None,
    error: str | None,
    elapsed_ms: float,
    context_source: str,
) -> dict[str, Any]:
    expected = case.expect.model_dump(mode="json")
    expected_for_report = {
        **expected,
        "proposal": {
            "tool_name": case.expect.proposal.tool_name,
            "tool_version": case.expect.proposal.tool_version,
            "argument_keys": sorted(case.expect.proposal.arguments_include),
        },
    }
    mismatches: list[str] = []
    if error:
        mismatches.append(error)
    elif request is not None and actual is not None:
        proposal = case.expect.proposal
        if proposal.tool_name and request.tool_name != proposal.tool_name:
            mismatches.append(
                f"tool expected {proposal.tool_name} but Agent proposed {request.tool_name}"
            )
        if proposal.tool_version and request.tool_version != proposal.tool_version:
            mismatches.append(
                f"tool version expected {proposal.tool_version} but received {request.tool_version}"
            )
        for key, expected_value in proposal.arguments_include.items():
            if key not in request.arguments:
                mismatches.append(f"Agent proposal missing required argument {key}")
            elif request.arguments[key] != expected_value:
                mismatches.append(f"Agent proposal changed required argument {key}")
        if actual["outcome"] != expected["outcome"]:
            mismatches.append(
                f"outcome expected {expected['outcome']} but received {actual['outcome']}"
            )
        for field in ("reason_codes", "obligations"):
            missing = sorted(set(expected[field]) - set(actual[field]))
            if missing:
                mismatches.append(f"missing expected {field}: {', '.join(missing)}")
    return {
        "id": case.id,
        "description": case.description,
        "passed": not mismatches,
        "expected": expected_for_report,
        "proposal": _proposal_summary(request) if request else None,
        "actual": actual,
        "mismatches": mismatches,
        "elapsed_ms": round(elapsed_ms, 3),
        "context_source": context_source,
    }


def run_suite(
    suite_path: Path,
    policy_path: Path,
    provider: ProposalProvider | None = None,
    build_id: str | None = None,
    build_digest: str | None = None,
) -> dict[str, Any]:
    """Run a suite in an isolated store without touching a real business tool."""

    suite = RegressionSuite.model_validate_json(suite_path.read_text())
    anchor = datetime.now(UTC)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    selected_provider = provider or EmbeddedProposalProvider()
    supplied_build_digest = (
        _validate_build_digest(build_digest) if build_digest is not None else None
    )
    try:
        declared_build_id = getattr(selected_provider, "declared_build_id", None)
        if build_id and declared_build_id and build_id != declared_build_id:
            raise ValueError(
                "supplied build id does not match the identity inside the canonical "
                "captured-proposal artifact"
            )
        provider_build_digest = getattr(selected_provider, "build_digest", None)
        if provider_build_digest is not None:
            provider_build_digest = _validate_build_digest(provider_build_digest)
            if supplied_build_digest and supplied_build_digest != provider_build_digest:
                raise ValueError(
                    "supplied build digest does not match the canonical captured-proposal "
                    "artifact"
                )
    except ValueError:
        selected_provider.close()
        raise
    resolved_build_digest = provider_build_digest or supplied_build_digest
    try:
        with tempfile.TemporaryDirectory(prefix="agent-mesh-regression-") as temp_dir:
            gateway = ActionGateway(
                AuditStore(Path(temp_dir) / "regression.sqlite3"),
                "regression-only-signing-secret",
                policy_path,
                clock=lambda: anchor,
            )
            for index, case in enumerate(suite.cases):
                case_started = time.perf_counter()
                actual = None
                error = None
                request = None
                context_source = "agent-proposal"
                try:
                    proposed = selected_provider.propose(case)
                    if not isinstance(selected_provider, EmbeddedProposalProvider):
                        if case.trusted_context is None:
                            raise ValueError(
                                f"case {case.id} requires trusted_context when testing an "
                                "external Agent proposal"
                            )
                        context_source = "release-contract"
                    elif case.trusted_context is not None:
                        context_source = "release-contract"

                    # Evaluation metadata and trusted business context are generated by the
                    # release contract, not accepted from the Agent under test.
                    context = case.trusted_context or proposed.context
                    request = proposed.model_copy(
                        update={
                            "request_id": f"reg-live-{index:04d}-{hashlib.sha256(case.id.encode()).hexdigest()[:12]}",
                            "requested_at": anchor,
                            "context": context,
                        }
                    )
                    decision = gateway.evaluate(request, case.principal)
                    # Executable grant tokens must never leak into a CI artifact.
                    actual = {
                        "outcome": decision.outcome,
                        "risk_tier": decision.risk_tier,
                        "reason_codes": decision.reason_codes,
                        "obligations": decision.obligations,
                        "policy_version": decision.policy_version,
                        "approval_required": decision.approval_id is not None,
                        "grant_issued": decision.grant_token is not None,
                    }
                except ValidationError:
                    error = "Agent proposal did not match the ActionRequest schema"
                except (GatewayError, httpx.HTTPError, ValueError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                results.append(
                    _case_result(
                        case,
                        request,
                        actual,
                        error,
                        (time.perf_counter() - case_started) * 1000,
                        context_source,
                    )
                )
    finally:
        selected_provider.close()

    passed = sum(result["passed"] for result in results)
    report = {
        "schema_version": "1.0",
        "suite": suite.name,
        "suite_description": suite.description,
        "proposal_source": selected_provider.source,
        "agent_build_id": build_id or selected_provider.source,
        "trusted_context_enforced": all(
            result["context_source"] == "release-contract" for result in results
        ),
        "policy_version": json.loads(policy_path.read_text())["policy_version"],
        "suite_sha256": _file_digest(suite_path),
        "policy_sha256": _file_digest(policy_path),
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 4),
            "release_gate": "pass" if passed == len(results) else "fail",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "results": results,
    }
    # Non-release contract runs remain valid without an artifact digest.  The paired
    # release gate separately requires and pins this field before it will compare builds.
    if resolved_build_digest is not None:
        report["agent_build_digest"] = resolved_build_digest
    return report


def write_junit(report: dict[str, Any], path: Path) -> None:
    summary = report["summary"]
    root = ElementTree.Element(
        "testsuite",
        name=str(report["suite"]),
        tests=str(summary["total"]),
        failures=str(summary["failed"]),
        time=f"{summary['elapsed_ms'] / 1000:.6f}",
        policy_version=str(report["policy_version"]),
    )
    for result in report["results"]:
        case = ElementTree.SubElement(
            root,
            "testcase",
            name=result["id"],
            classname="agent_action_contract",
            time=f"{result['elapsed_ms'] / 1000:.6f}",
        )
        if not result["passed"]:
            failure = ElementTree.SubElement(
                case, "failure", message="; ".join(result["mismatches"])
            )
            failure.text = json.dumps(result, indent=2, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def compare_with_baseline(
    report: dict[str, Any],
    baseline: dict[str, Any],
    fail_on_behavior_change: bool = False,
) -> dict[str, Any]:
    """Attach version-to-version changes and optionally make any behavior change blocking."""

    if report["suite"] != baseline.get("suite"):
        raise ValueError("baseline and current reports must use the same suite")
    baseline_results = {row["id"]: row for row in baseline.get("results", [])}
    current_results = {row["id"]: row for row in report["results"]}
    if set(baseline_results) != set(current_results):
        raise ValueError("baseline and current reports contain different case ids")

    new_failures = sorted(
        case_id
        for case_id, current in current_results.items()
        if not current["passed"] and baseline_results[case_id]["passed"]
    )
    fixed = sorted(
        case_id
        for case_id, current in current_results.items()
        if current["passed"] and not baseline_results[case_id]["passed"]
    )
    behavior_changes = sorted(
        case_id
        for case_id, current in current_results.items()
        if (current.get("proposal") or {}).get("behavior_fingerprint")
        != (baseline_results[case_id].get("proposal") or {}).get("behavior_fingerprint")
    )
    comparison = {
        "new_failures": new_failures,
        "fixed_failures": fixed,
        "behavior_changes": behavior_changes,
        "fail_on_behavior_change": fail_on_behavior_change,
    }
    report["comparison"] = comparison
    if new_failures or (fail_on_behavior_change and behavior_changes):
        report["summary"]["release_gate"] = "fail"
    return report


def _print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    status = "PASS" if summary["release_gate"] == "pass" else "FAIL"
    print(
        f"[{status}] {report['suite']} | policy {report['policy_version']} | "
        f"{summary['passed']}/{summary['total']} contracts passed"
    )
    for result in report["results"]:
        marker = "PASS" if result["passed"] else "FAIL"
        print(f"  [{marker}] {result['id']}")
        for mismatch in result["mismatches"]:
            print(f"         {mismatch}")
    comparison = report.get("comparison")
    if comparison:
        print(
            "  [DIFF] "
            f"{len(comparison['new_failures'])} new failures, "
            f"{len(comparison['fixed_failures'])} fixed, "
            f"{len(comparison['behavior_changes'])} behavior changes"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run framework-neutral Agent Action Contract regression tests."
    )
    parser.add_argument("suite", type=Path, help="JSON action contract suite")
    parser.add_argument("--policy", type=Path, required=True, help="gateway policy JSON")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--captured-proposals",
        type=Path,
        help="replay ActionRequest objects captured from an Agent runtime",
    )
    source.add_argument(
        "--agent-url",
        help="customer-controlled HTTP adapter returning an ActionRequest",
    )
    parser.add_argument(
        "--agent-api-key-env",
        help="environment variable containing the adapter bearer token",
    )
    parser.add_argument("--baseline-report", type=Path, help="compare with a prior JSON report")
    parser.add_argument(
        "--build-id",
        help="immutable Agent build identifier recorded in the release evidence",
    )
    parser.add_argument(
        "--build-digest",
        help=(
            "attested Agent artifact digest in sha256:<64 lowercase hex> form; captured "
            "proposal files are hashed canonically and any supplied value must match"
        ),
    )
    parser.add_argument(
        "--fail-on-behavior-change",
        action="store_true",
        help="fail even when changed Agent behavior still satisfies the contract",
    )
    parser.add_argument("--json-report", type=Path, help="write a machine-readable JSON report")
    parser.add_argument("--junit-report", type=Path, help="write a CI-compatible JUnit report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        provider: ProposalProvider | None = None
        if args.captured_proposals:
            provider = CapturedProposalProvider(args.captured_proposals)
        elif args.agent_url:
            api_key = None
            if args.agent_api_key_env:
                api_key = os.environ.get(args.agent_api_key_env)
                if not api_key:
                    raise ValueError(
                        f"environment variable {args.agent_api_key_env} is missing or empty"
                    )
            provider = HttpProposalProvider(args.agent_url, api_key=api_key)
        elif args.agent_api_key_env:
            raise ValueError("--agent-api-key-env requires --agent-url")

        report = run_suite(
            args.suite,
            args.policy,
            provider=provider,
            build_id=args.build_id,
            build_digest=args.build_digest,
        )
        if args.baseline_report:
            baseline = json.loads(args.baseline_report.read_text())
            report = compare_with_baseline(
                report,
                baseline,
                fail_on_behavior_change=args.fail_on_behavior_change,
            )
    except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as exc:
        print(f"Invalid regression configuration: {exc}", file=sys.stderr)
        return 2
    _print_summary(report)
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if args.junit_report:
        write_junit(report, args.junit_report)
    return 0 if report["summary"]["release_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
