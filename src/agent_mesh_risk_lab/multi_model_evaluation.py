"""Run and compare the frozen Agent evaluation across multiple local model families."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import pandas as pd

from .real_llm_evaluation import PROMPT_MODES, run_real_llm_evaluation

RUN_FILES = (
    "aggregate.csv",
    "by_stressor.csv",
    "by_workflow.csv",
    "decisions.csv",
    "decisions.jsonl",
    "few_shot_action_shift.csv",
    "few_shot_transition_summary.csv",
    "few_shot_transitions.csv",
    "manifest.json",
    "paired_effects.csv",
    "scenarios.jsonl",
)


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name).strip("_")


def archive_current_run(project_root: Path, expected_model: str, target: Path) -> None:
    """Copy the previously completed single-model run into the comparison layout."""

    source = project_root / "data" / "llm_evaluation"
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("no completed single-model manifest is available to reuse")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model") != expected_model:
        raise ValueError(
            f"existing run is {manifest.get('model')}, not requested model {expected_model}"
        )
    if manifest.get("decision_calls") != 192 or manifest.get("failed_responses") != 0:
        raise ValueError("existing run is incomplete or contains failed responses")
    target.mkdir(parents=True, exist_ok=True)
    for filename in RUN_FILES:
        source_path = source / filename
        if source_path.exists():
            shutil.copy2(source_path, target / filename)


def _load_run(
    run_dir: Path,
) -> tuple[str, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    model = str(manifest["model"])
    decisions = pd.read_csv(run_dir / "decisions.csv")
    aggregate = pd.read_csv(run_dir / "aggregate.csv")
    paired_effects = pd.read_csv(run_dir / "paired_effects.csv")
    expected_pairs = 64 * len(PROMPT_MODES)
    if len(decisions) != expected_pairs:
        raise ValueError(f"{model} has {len(decisions)} decisions, expected {expected_pairs}")
    if decisions[["scenario_id", "prompt_mode"]].duplicated().any():
        raise ValueError(f"{model} contains duplicate scenario/prompt decisions")
    return model, manifest, decisions, aggregate, paired_effects


def compare_runs(run_dirs: list[Path], output_dir: Path, report_path: Path) -> dict[str, Path]:
    """Create evidence-bounded cross-model tables without pooling unlike model decisions."""

    if len(run_dirs) < 2:
        raise ValueError("multi-model comparison requires at least two completed runs")
    loaded = [_load_run(path) for path in run_dirs]
    models = [row[0] for row in loaded]
    if len(models) != len(set(models)):
        raise ValueError("multi-model comparison received duplicate model names")

    reference_pairs = set(
        loaded[0][2][["scenario_id", "prompt_mode"]].itertuples(index=False, name=None)
    )
    for model, _manifest, decisions, _aggregate, _paired_effects in loaded[1:]:
        pairs = set(decisions[["scenario_id", "prompt_mode"]].itertuples(index=False, name=None))
        if pairs != reference_pairs:
            raise ValueError(f"{model} was not evaluated on the identical paired scenario set")

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_rows = []
    prompt_effect_rows = []
    decision_frames = []
    paired_frames = []
    manifests = []
    for model, manifest, decisions, aggregate, paired_effects in loaded:
        aggregate = aggregate.copy()
        aggregate.insert(0, "model", model)
        aggregate_rows.append(aggregate)
        decision_frames.append(decisions.assign(model=model))
        paired_frames.append(paired_effects.assign(model=model))
        manifests.append(
            {
                "model": model,
                "digest": manifest.get("model_metadata", {}).get("digest"),
                "parameter_size": manifest.get("model_metadata", {})
                .get("details", {})
                .get("parameter_size"),
                "quantization": manifest.get("model_metadata", {})
                .get("details", {})
                .get("quantization_level"),
                "valid_schema_responses": manifest.get("valid_schema_responses"),
                "failed_responses": manifest.get("failed_responses"),
            }
        )
        indexed = aggregate.set_index("prompt_mode")
        for mode in ("governed", "governed_few_shot"):
            prompt_effect_rows.append(
                {
                    "model": model,
                    "comparison": f"{mode}_minus_baseline",
                    "harmful_action_change": indexed.loc[mode, "harmful_action_rate"]
                    - indexed.loc["baseline", "harmful_action_rate"],
                    "action_accuracy_change": indexed.loc[mode, "action_accuracy"]
                    - indexed.loc["baseline", "action_accuracy"],
                    "overblocking_change": indexed.loc[mode, "normal_case_overblocking_rate"]
                    - indexed.loc["baseline", "normal_case_overblocking_rate"],
                    "policy_compliance_change": indexed.loc[mode, "policy_compliance_rate"]
                    - indexed.loc["baseline", "policy_compliance_rate"],
                }
            )

    combined = pd.concat(decision_frames, ignore_index=True)
    agreement_rows = []
    for left, right in combinations(models, 2):
        pair = combined[combined["model"].isin([left, right])].pivot(
            index=["scenario_id", "prompt_mode"],
            columns="model",
            values=["action", "harmful_action", "action_correct"],
        )
        for mode in PROMPT_MODES:
            mode_pair = pair.xs(mode, level="prompt_mode")
            agreement_rows.append(
                {
                    "model_left": left,
                    "model_right": right,
                    "prompt_mode": mode,
                    "n_scenarios": len(mode_pair),
                    "exact_action_agreement": (
                        mode_pair["action"][left] == mode_pair["action"][right]
                    ).mean(),
                    "harm_label_agreement": (
                        mode_pair["harmful_action"][left] == mode_pair["harmful_action"][right]
                    ).mean(),
                    "correctness_agreement": (
                        mode_pair["action_correct"][left] == mode_pair["action_correct"][right]
                    ).mean(),
                }
            )

    aggregate_path = output_dir / "multi_model_aggregate.csv"
    effects_path = output_dir / "multi_model_prompt_effects.csv"
    paired_path = output_dir / "multi_model_paired_effects.csv"
    agreement_path = output_dir / "multi_model_scenario_agreement.csv"
    manifest_path = output_dir / "multi_model_manifest.json"
    aggregate_all = pd.concat(aggregate_rows, ignore_index=True)
    agreement = pd.DataFrame(agreement_rows)
    aggregate_all.to_csv(aggregate_path, index=False)
    pd.DataFrame(prompt_effect_rows).to_csv(effects_path, index=False)
    paired = pd.concat(paired_frames, ignore_index=True)
    paired.to_csv(paired_path, index=False)
    agreement.to_csv(agreement_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "models": manifests,
                "scenario_count_per_model": 64,
                "prompt_modes": list(PROMPT_MODES),
                "decision_calls_per_model": 192,
                "comparison_boundary": (
                    "identical synthetic scenarios and deterministic samples; this tests "
                    "cross-model sensitivity, not production impact or broad model ranking"
                ),
            },
            indent=2,
        )
        + "\n"
    )

    effects = pd.DataFrame(prompt_effect_rows)
    lines = [
        "# Multi-model Agent evaluation",
        "",
        "## Scope",
        "",
        f"Compared {', '.join(models)} on the same 64 synthetic scenarios under three prompt modes.",
        "Each model contributed 192 schema-constrained, decision-only observations.",
        "",
        "## Governed prompt effect versus baseline",
        "",
        "| Model | Harm change (paired 95% CI) | Accuracy change | Over-blocking change |",
        "|---|---:|---:|---:|",
    ]
    governed_harm = paired[
        (paired["comparison"] == "governed_vs_baseline")
        & (paired["metric"] == "harmful_action_rate")
    ].set_index("model")
    for row in effects[effects["comparison"] == "governed_minus_baseline"].to_dict("records"):
        interval = governed_harm.loc[row["model"]]
        lines.append(
            f"| {row['model']} | {row['harmful_action_change']:+.2%} "
            f"[{interval['ci_low']:+.2%}, {interval['ci_high']:+.2%}] | "
            f"{row['action_accuracy_change']:+.2%} | {row['overblocking_change']:+.2%} |"
        )
    governed_summary = aggregate_all[aggregate_all["prompt_mode"] == "governed"]
    few_shot_summary = aggregate_all[aggregate_all["prompt_mode"] == "governed_few_shot"]
    governed_agreement = agreement[agreement["prompt_mode"] == "governed"]
    lines.extend(
        [
            "",
            "## Cross-model interpretation",
            "",
            (
                f"Under the governed prompt, harmful-action rates ranged from "
                f"{governed_summary['harmful_action_rate'].min():.2%} to "
                f"{governed_summary['harmful_action_rate'].max():.2%}, while normal-case "
                f"over-blocking ranged from "
                f"{governed_summary['normal_case_overblocking_rate'].min():.2%} to "
                f"{governed_summary['normal_case_overblocking_rate'].max():.2%}."
            ),
            (
                f"Pairwise exact-action agreement under governance averaged "
                f"{governed_agreement['exact_action_agreement'].mean():.2%}; agreement on whether "
                f"the action was harmful averaged "
                f"{governed_agreement['harm_label_agreement'].mean():.2%}. The models can therefore "
                "look similarly safe in an aggregate while choosing materially different actions."
            ),
            (
                f"Few-shot harmful-action rates spanned "
                f"{few_shot_summary['harmful_action_rate'].min():.2%} to "
                f"{few_shot_summary['harmful_action_rate'].max():.2%}. Prompt examples are not a "
                "portable authorization mechanism across these model families."
            ),
            "",
            "## Evidence boundary",
            "",
            (
                "The comparison isolates model-family sensitivity better than the earlier "
                "single-model run, but it still uses synthetic English tasks, one deterministic "
                "sample per condition, local quantized models, and no real business-tool "
                "execution. It must not be presented as a leaderboard, safety certificate, or "
                "measured enterprise impact."
            ),
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    return {
        "aggregate": aggregate_path,
        "prompt_effects": effects_path,
        "paired_effects": paired_path,
        "scenario_agreement": agreement_path,
        "manifest": manifest_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--reuse-current-model",
        help="archive the completed data/llm_evaluation run instead of rerunning this model",
    )
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    root = args.project_root.resolve()
    run_dirs = []
    for model in args.models:
        target = root / "data" / "multi_model" / model_slug(model)
        if model == args.reuse_current_model:
            archive_current_run(root, model, target)
        else:
            run_real_llm_evaluation(
                root,
                model_name=model,
                seed=args.seed,
                output_dir=target,
                publish_shared_artifacts=False,
            )
        run_dirs.append(target)
    paths = compare_runs(
        run_dirs,
        root / "outputs" / "tables",
        root / "outputs" / "reports" / "multi_model_evaluation.md",
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
