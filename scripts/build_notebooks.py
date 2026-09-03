"""Build the reader-facing, reproducible analysis notebooks."""

import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def cell_markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def cell_code(code: str):
    return nbf.v4.new_code_cell(dedent(code).strip())


def model_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        cell_markdown(
            """
            # Offline Model Evaluation Audit

            ## tl;dr

            Structured Logistic Regression reaches **0.773 PR-AUC** with simulator-informed
            features but only **0.709** with deployment-observable structured inputs, exposing a
            **0.064 information-access gap**. Failure attribution remains weak at **0.171
            Macro-F1**, while severity prediction reaches **0.906** and governance recommendation
            reaches **0.498** with **95.4% Top-3 accuracy**. All rows are synthetic; no LLM was
            evaluated.
            """
        ),
        cell_markdown(
            """
            ## Context & Methods

            This companion notebook audits persisted evaluation outputs rather than retraining
            models. The source of truth is `data/evaluation/`, generated with seed `20260827`.

            ### Key Assumptions

            - The harmful-action label is synthetic and deterministic for a fixed seed.
            - Split membership is assigned at `task_id` level, not row level.
            - Model selection uses validation PR-AUC; the test split is reserved for reporting.
            - Threshold selection maximizes F2 subject to validation over-blocking <= 35%.
            - Simulator-informed and deployment-observable features are reported separately.
            - Failure attribution excludes injected-stressor identity and raw trace text.
            """
        ),
        cell_markdown("## Data\n\n### 1. Load persisted artifacts"),
        cell_code(
            """
            from pathlib import Path
            import json
            import pandas as pd
            import matplotlib.pyplot as plt

            ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
            assert (ROOT / "pyproject.toml").exists(), "Run from the repository root"
            evaluation_dir = ROOT / "data" / "evaluation"
            comparison = pd.read_csv(evaluation_dir / "model_comparison.csv")
            calibration = pd.read_csv(evaluation_dir / "model_calibration.csv")
            bootstrap = pd.read_csv(evaluation_dir / "bootstrap_confidence_intervals.csv")
            ablation = pd.read_csv(evaluation_dir / "ablation_study.csv")
            unseen = pd.read_csv(evaluation_dir / "unseen_stressor_generalization.csv")
            cross_workflow = pd.read_csv(evaluation_dir / "cross_workflow_generalization.csv")
            feature_access = pd.read_csv(evaluation_dir / "feature_access_audit.csv")
            multitask = pd.read_csv(evaluation_dir / "multitask_comparison.csv")
            per_class = pd.read_csv(evaluation_dir / "multitask_per_class_recall.csv")
            governance_unseen = pd.read_csv(evaluation_dir / "governance_unseen_stressor.csv")
            manifest = json.loads((evaluation_dir / "evaluation_manifest.json").read_text())
            manifest
            """
        ),
        cell_markdown("### 2. Verify leakage boundaries and output ranges"),
        cell_code(
            """
            split_manifest = pd.read_csv(evaluation_dir / "split_manifest.csv")
            split_sets = {
                name: set(split_manifest.loc[split_manifest["split"] == name, "task_id"])
                for name in ["train", "validation", "test"]
            }
            assert not split_sets["train"] & split_sets["validation"]
            assert not split_sets["train"] & split_sets["test"]
            assert not split_sets["validation"] & split_sets["test"]
            assert manifest["task_overlap_train_test"] == 0
            metric_columns = ["accuracy", "precision", "recall", "f1", "auroc", "pr_auc", "brier", "ece", "over_blocking_rate"]
            assert comparison[metric_columns].apply(lambda column: column.between(0, 1).all()).all()
            {name: len(values) for name, values in split_sets.items()}
            """
        ),
        cell_markdown("## Results\n\n### 3. Compare model ranking and operating trade-offs"),
        cell_code(
            """
            columns = ["model", "pr_auc", "auroc", "f1", "safety_recall", "over_blocking_rate", "brier", "ece"]
            comparison[columns].sort_values("pr_auc", ascending=False).round(3)
            """
        ),
        cell_code(
            """
            ordered = comparison.sort_values("pr_auc")
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].barh(ordered["model"], ordered["pr_auc"], color="#2563EB")
            axes[0].set(xlim=(0, 1), xlabel="PR-AUC", title="Task-group holdout ranking")
            axes[1].scatter(comparison["over_blocking_rate"], comparison["safety_recall"], color="#D4A72C", s=65)
            for row in comparison.itertuples():
                axes[1].annotate(row.model, (row.over_blocking_rate, row.safety_recall), fontsize=8, xytext=(4, 4), textcoords="offset points")
            axes[1].set(xlim=(0, .4), ylim=(0, 1), xlabel="Over-blocking rate", ylabel="Safety recall", title="Operating trade-off")
            plt.tight_layout()
            """
        ),
        cell_markdown("### 4. Audit calibration and uncertainty"),
        cell_code(
            """
            raw = comparison.loc[comparison["model"] == manifest["best_uncalibrated_model"]].iloc[0]
            calibrated = comparison.loc[comparison["model_family"] == "calibrated_offline_ml"].iloc[0]
            calibration_effect = pd.DataFrame({
                "raw": raw[["pr_auc", "brier", "ece", "f1"]],
                "calibrated": calibrated[["pr_auc", "brier", "ece", "f1"]],
            })
            calibration_effect["delta"] = calibration_effect["calibrated"] - calibration_effect["raw"]
            calibration_effect.round(4)
            """
        ),
        cell_code(
            """
            bootstrap[["metric", "estimate", "ci_low", "ci_high", "resampling_unit"]].round(3)
            """
        ),
        cell_markdown("### 5. Test feature dependence and transfer"),
        cell_code(
            """
            display_ablation = ablation[["configuration", "pr_auc", "f1", "safety_recall", "f1_delta_vs_full", "recall_delta_vs_full"]]
            display_ablation.round(3)
            """
        ),
        cell_code(
            """
            assert (unseen["task_overlap"] == 0).all()
            assert (cross_workflow["task_overlap"] == 0).all()
            print("Strict unseen-stressor tests")
            display(unseen[["held_out_stressor", "pr_auc", "safety_recall", "f1", "over_blocking_rate"]].round(3))
            print("Leave-one-workflow-out tests")
            display(cross_workflow[["held_out_workflow", "pr_auc", "safety_recall", "f1", "over_blocking_rate"]].round(3))
            """
        ),
        cell_markdown("### 6. Measure the simulator-information advantage"),
        cell_code(
            """
            assert feature_access.loc[feature_access["label_shuffle"], "pr_auc"].iloc[0] < feature_access.loc[feature_access["feature_access"] == "Deployable structured", "pr_auc"].iloc[0]
            feature_access[["feature_access", "feature_count", "pr_auc", "f1", "safety_recall", "over_blocking_rate", "pr_auc_optimism_gap"]].round(3)
            """
        ),
        cell_code(
            """
            ordered_access = feature_access.sort_values("pr_auc")
            colors = ["#E87722" if value else ("#94A3B8" if shuffled else "#2563EB") for value, shuffled in zip(ordered_access["uses_simulator_privileged_features"], ordered_access["label_shuffle"])]
            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.barh(ordered_access["feature_access"], ordered_access["pr_auc"], color=colors)
            ax.set(xlim=(0, 1), xlabel="PR-AUC on task-group holdout", title="Risk classification by feature-access policy")
            plt.tight_layout()
            """
        ),
        cell_markdown(
            "### 7. Evaluate failure attribution, severity, and governance recommendation"
        ),
        cell_code(
            """
            multitask[["task", "model", "accuracy", "macro_f1", "macro_recall", "top_3_accuracy", "mean_decision_regret", "rows"]].round(3)
            """
        ),
        cell_code(
            """
            failure_recall = per_class[per_class["task"] == "failure_attribution"]
            assert failure_recall["test_support"].sum() == 939
            display(failure_recall.sort_values("recall", ascending=False).round(3))
            print("Governance recommendation on strictly unseen stressors")
            display(governance_unseen.round(3))
            """
        ),
        cell_markdown(
            """
            ## Takeaways

            1. The task-group split is clean: 136/32/32 tasks and zero overlap.
            2. The deployable structured model falls from 0.773 to 0.709 PR-AUC when simulator-only
               variables are removed. The higher number must not be presented as deployable quality.
            3. The selected isotonic calibration is not stable on the test set: PR-AUC and Brier
               both worsen. Keep the raw scorer until a larger independent calibration set exists.
            4. Transfer is uneven. Email is the weakest held-out workflow (PR-AUC about 0.380),
               while unseen-stressor PR-AUC remains above 0.74. Workflow shift is therefore the
               more serious current generalization risk.
            5. Failure attribution is the main capability gap (Macro-F1 0.171). The structured
               trace lacks enough diagnostic evidence to distinguish most taxonomy classes.
            6. Governance recommendation is useful as a ranked shortlist (95.4% Top-3) but not as
               an autonomous decision (53.3% Top-1; substantial unseen-stressor regret).
            7. These conclusions describe a synthetic trace distribution, not real agent behavior.
            """
        ),
    ]
    return notebook


def control_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        cell_markdown(
            """
            # Empirical Control-Portfolio Audit

            ## tl;dr

            Exhaustive evaluation of all **64 control portfolios** corrects the earlier additive
            recommendation. Under budget 40, completion >= 85%, and review load <= 30%, the
            optimum is **Context Envelope + Tool Version Lock + Permission Scope** (cost 27),
            reducing stressed incidents from **62.21% to 33.71%**. The result remains between
            32.00% and 35.86% across 12 seeds.
            """
        ),
        cell_markdown(
            """
            ## Context & Methods

            Each portfolio is evaluated on 200 tasks x 7 non-null stressors = 1,400 runs. The
            grid contains every subset of six controls. Shapley values use all coalitions, pair
            interactions compare observed risk with an independent-risk expectation, and seed
            sensitivity repeats the stressed evaluation for 12 consecutive seeds.

            ### Key Assumptions

            - Governance cost and control mechanics are synthetic project parameters.
            - Portfolio comparisons are empirical within the simulator, not causal claims about production.
            - Default feasibility requires cost <= 40, completion >= 85%, and review load <= 30%.
            """
        ),
        cell_markdown("## Data\n\n### 1. Load and reconcile the full portfolio grid"),
        cell_code(
            """
            from pathlib import Path
            import pandas as pd
            import matplotlib.pyplot as plt

            ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
            assert (ROOT / "pyproject.toml").exists(), "Run from the repository root"
            control_dir = ROOT / "data" / "control_science"
            grid = pd.read_csv(control_dir / "control_portfolio_grid.csv")
            by_workflow = pd.read_csv(control_dir / "control_portfolio_by_workflow.csv")
            shapley = pd.read_csv(control_dir / "control_shapley.csv")
            interactions = pd.read_csv(control_dir / "control_interactions.csv")
            sensitivity = pd.read_csv(control_dir / "seed_sensitivity.csv")
            assert len(grid) == 64 and grid["portfolio"].nunique() == 64
            assert len(by_workflow) == 64 * 4
            assert (grid["runs"] == 1400).all()
            grid.shape, by_workflow.shape
            """
        ),
        cell_markdown("## Results\n\n### 2. Solve the constrained empirical optimization"),
        cell_code(
            """
            feasible = grid[(grid["cost"] <= 40) & (grid["task_success_rate"] >= .85) & (grid["human_review_load"] <= .30)]
            optimum = feasible.sort_values(["incident_rate", "cost", "task_success_rate"], ascending=[True, True, False]).iloc[0]
            optimum[["portfolio", "cost", "incident_rate", "risk_reduction", "task_success_rate", "human_review_load", "worst_workflow_incident_rate"]]
            """
        ),
        cell_code(
            """
            frontier = grid[grid["pareto_efficient"]].sort_values("cost")
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.scatter(grid.loc[~grid["feasible_default"], "cost"], grid.loc[~grid["feasible_default"], "incident_rate"], facecolors="none", edgecolors="#94A3B8", label="Infeasible")
            ax.scatter(grid.loc[grid["feasible_default"], "cost"], grid.loc[grid["feasible_default"], "incident_rate"], color="#2563EB", label="Feasible")
            ax.scatter(frontier["cost"], frontier["incident_rate"], facecolors="none", edgecolors="#D4A72C", s=90, label="3-objective Pareto set")
            ax.scatter([optimum["cost"]], [optimum["incident_rate"]], color="#E87722", s=100, marker="*", label="Budget-40 optimum")
            ax.set(xlabel="Governance cost", ylabel="Stressed incident rate", title="All 64 empirical portfolios")
            ax.legend(frameon=False)
            plt.tight_layout()
            """
        ),
        cell_markdown("### 3. Reconcile marginal attribution"),
        cell_code(
            """
            baseline_risk = float(grid.loc[grid["portfolio"] == "none", "incident_rate"].iloc[0])
            full_risk = float(grid.loc[grid["control_count"] == 6, "incident_rate"].iloc[0])
            assert abs(shapley["shapley_risk_reduction"].sum() - (baseline_risk - full_risk)) < 1e-10
            shapley[["label", "shapley_risk_reduction", "cost", "shapley_per_cost"]].round(4)
            """
        ),
        cell_markdown("### 4. Identify complementarity and diminishing returns"),
        cell_code(
            """
            strongest = interactions.nlargest(3, "synergy")[["label_a", "label_b", "synergy", "interpretation"]]
            weakest = interactions.nsmallest(3, "synergy")[["label_a", "label_b", "synergy", "interpretation"]]
            print("Strongest complementarity")
            display(strongest.round(4))
            print("Strongest diminishing returns")
            display(weakest.round(4))
            """
        ),
        cell_markdown("### 5. Quantify seed sensitivity"),
        cell_code(
            """
            seed_summary = sensitivity.groupby("configuration").agg(
                mean=("incident_rate", "mean"),
                standard_deviation=("incident_rate", "std"),
                minimum=("incident_rate", "min"),
                maximum=("incident_rate", "max"),
            )
            seed_summary.round(4)
            """
        ),
        cell_markdown(
            """
            ## Takeaways

            1. Exact joint testing changes the recommendation: Tool Version Lock replaces External
               Isolation, lowering cost from 36 to 27 while retaining the measured 28.5-point reduction.
            2. Selective Human Review has the largest average marginal Shapley value but costs 45,
               so it is infeasible under the default budget. Tool Version Lock leads per cost.
            3. Context Envelope + Permission Scope is the strongest complementary pair (+0.0161),
               while Context Envelope + Selective Human Review has the strongest diminishing return.
            4. The chosen bundle is stable in this simulator across 12 seeds, but the assumptions—not
               only random variation—must be challenged before production use.
            """
        ),
    ]
    return notebook


def real_llm_notebook():
    data_dir = ROOT / "data" / "llm_evaluation"
    summary = pd.read_csv(data_dir / "aggregate.csv").set_index("prompt_mode")
    effect_rows = pd.read_csv(data_dir / "paired_effects.csv")
    effects = effect_rows[effect_rows["comparison"] == "governed_vs_baseline"].set_index("metric")
    few_shot_effects = effect_rows[effect_rows["comparison"] == "few_shot_vs_governed"].set_index(
        "metric"
    )
    governed = summary.loc["governed"]
    few_shot = summary.loc["governed_few_shot"]
    safety_delta = effects.loc["safety_success_rate", "raw_delta_treatment_minus_reference"]
    harm_delta = effects.loc["harmful_action_rate", "raw_delta_treatment_minus_reference"]
    accuracy_delta = effects.loc["action_accuracy", "raw_delta_treatment_minus_reference"]
    few_harm_delta = few_shot_effects.loc[
        "harmful_action_rate", "raw_delta_treatment_minus_reference"
    ]
    few_accuracy_delta = few_shot_effects.loc[
        "action_accuracy", "raw_delta_treatment_minus_reference"
    ]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        cell_markdown(
            f"""
            # Real LLM Paired Evaluation Audit

            ## tl;dr

            On 64 frozen scenarios, the governed prompt changes safety success by
            **{safety_delta:+.1%}**, harmful actions by **{harm_delta:+.1%}**, and exact action
            accuracy by **{accuracy_delta:+.1%}** relative to the same local model under the
            baseline prompt. Governed safety is **{governed["safety_success_rate"]:.1%}**, with
            **{governed["normal_case_overblocking_rate"]:.1%}** normal-case over-blocking. These
            Few-shot then changes harm by **{few_harm_delta:+.1%}** and accuracy by
            **{few_accuracy_delta:+.1%}** versus governed zero-shot, with
            **{few_shot["normal_case_overblocking_rate"]:.1%}** over-blocking. These are paired
            results for Qwen2.5 3B on synthetic tasks, not external production claims.
            """
        ),
        cell_markdown(
            """
            ## Context & Methods

            This audit independently recomputes the persisted real-model results in
            `data/llm_evaluation/`. The frozen design is 4 workflows x 8 stressors x 2 case types,
            evaluated under baseline zero-shot, governed zero-shot, and leakage-safe governed
            few-shot prompts with temperature zero.

            ### Key Assumptions

            - Each `scenario_id` is the paired unit; individual decisions are not independent.
            - Expected actions and labels are scoring data and were not supplied to the model.
            - Safety means no unauthorized high-impact terminal action; exact action accuracy is
              stricter and captures utility.
            - Schema-invalid outputs fail closed but remain action and compliance failures.
            - Intervals describe this frozen set, not other models, workflows, or organizations.
            """
        ),
        cell_markdown("## Data\n\n### 1. Load decisions, summaries, and run manifest"),
        cell_code(
            """
            from pathlib import Path
            import json
            import pandas as pd
            import matplotlib.pyplot as plt

            ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
            assert (ROOT / "pyproject.toml").exists(), "Run from the repository root"
            data_dir = ROOT / "data" / "llm_evaluation"
            decisions = pd.read_csv(data_dir / "decisions.csv")
            aggregate = pd.read_csv(data_dir / "aggregate.csv")
            stressors = pd.read_csv(data_dir / "by_stressor.csv")
            effects = pd.read_csv(data_dir / "paired_effects.csv")
            manifest = json.loads((data_dir / "manifest.json").read_text())
            manifest
            """
        ),
        cell_markdown("### 2. Verify the factorial and paired unit"),
        cell_code(
            """
            assert len(decisions) == 192
            assert decisions["scenario_id"].nunique() == 64
            assert (decisions.groupby("scenario_id").size() == 3).all()
            assert set(decisions["prompt_mode"]) == {"baseline", "governed", "governed_few_shot"}
            assert decisions.groupby(["workflow", "stressor", "case_type"]).size().eq(3).all()
            assert manifest["few_shot_evaluation_task_overlap"] == 0
            assert decisions["valid_schema"].sum() == manifest["valid_schema_responses"]
            decisions.groupby(["prompt_mode", "case_type"]).size().unstack()
            """
        ),
        cell_markdown("## Results\n\n### 3. Independently recompute headline rates"),
        cell_code(
            """
            recalculated = decisions.groupby("prompt_mode").agg(
                n=("scenario_id", "size"),
                valid_schema_rate=("valid_schema", "mean"),
                action_accuracy=("action_correct", "mean"),
                safety_success_rate=("safety_success", "mean"),
                harmful_action_rate=("harmful_action", "mean"),
                policy_compliance_rate=("policy_compliant", "mean"),
            ).reset_index()
            persisted = aggregate[recalculated.columns]
            pd.testing.assert_frame_equal(
                recalculated.sort_values("prompt_mode").reset_index(drop=True).sort_index(axis=1),
                persisted.sort_values("prompt_mode").reset_index(drop=True).sort_index(axis=1),
                check_dtype=False,
            )
            recalculated.round(3)
            """
        ),
        cell_markdown("### 4. Recompute scenario-paired changes"),
        cell_code(
            """
            paired = decisions.pivot(index="scenario_id", columns="prompt_mode")
            comparisons = {
                "governed_vs_baseline": ("baseline", "governed"),
                "few_shot_vs_governed": ("governed", "governed_few_shot"),
            }
            for comparison, (reference, treatment) in comparisons.items():
                stored = effects[effects["comparison"] == comparison].set_index("metric")
                for metric, source in {
                    "action_accuracy": "action_correct",
                    "safety_success_rate": "safety_success",
                    "harmful_action_rate": "harmful_action",
                    "policy_compliance_rate": "policy_compliant",
                }.items():
                    values = paired[source].astype(float)
                    recomputed = (values[treatment] - values[reference]).mean()
                    assert abs(recomputed - stored.loc[metric, "raw_delta_treatment_minus_reference"]) < 1e-12
            effects[["comparison", "metric", "reference_rate", "treatment_rate", "raw_delta_treatment_minus_reference", "ci_low", "ci_high", "improved_scenarios", "regressed_scenarios"]].round(3)
            """
        ),
        cell_markdown("### 5. Separate safety gains from utility cost"),
        cell_code(
            """
            rate_columns = ["action_accuracy", "safety_success_rate", "policy_compliance_rate", "normal_case_overblocking_rate"]
            plot_frame = aggregate.set_index("prompt_mode")[rate_columns].T
            plot_frame.columns = [column.title() for column in plot_frame.columns]
            ax = plot_frame.plot(kind="bar", figsize=(10, 5), color=["#94A3B8", "#2563EB", "#D4A72C"])
            ax.set(ylim=(0, 1), ylabel="Rate", title="Decision quality and safety by prompt condition")
            ax.legend(frameon=False)
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            """
        ),
        cell_markdown("### 6. Locate residual harmful behavior"),
        cell_code(
            """
            attack_surface = stressors.pivot(index="stressor", columns="prompt_mode", values="harmful_action_rate")
            attack_surface["governed_change"] = attack_surface["governed"] - attack_surface["baseline"]
            attack_surface["few_shot_change"] = attack_surface["governed_few_shot"] - attack_surface["governed"]
            attack_surface.sort_values("governed_few_shot", ascending=False).round(3)
            """
        ),
        cell_code(
            """
            errors = decisions[(~decisions["action_correct"]) | decisions["harmful_action"]]
            errors.groupby(["prompt_mode", "workflow", "stressor"]).agg(
                errors=("scenario_id", "size"),
                harmful=("harmful_action", "sum"),
                safe_abstentions=("safe_abstention", "sum"),
            ).sort_values(["harmful", "errors"], ascending=False).head(20)
            """
        ),
        cell_markdown("### 7. Check reliability and local inference cost"),
        cell_code(
            """
            aggregate[["prompt_mode", "valid_schema_rate", "permitted_action_rate", "tool_consistency_rate", "latency_p50_ms", "latency_p95_ms", "mean_prompt_tokens", "mean_completion_tokens"]].round(2)
            """
        ),
        cell_markdown(
            f"""
            ## Takeaways

            1. The audit contains exactly 64 scenario triplets and 192 decisions; all reported
               treatment effects are calculated within scenario.
            2. The governance intervention changes harmful-action rate by **{harm_delta:+.1%}**
               and safety success by **{safety_delta:+.1%}** on the frozen set.
            3. Exact action accuracy changes by **{accuracy_delta:+.1%}**. This must be read beside
               the **{governed["normal_case_overblocking_rate"]:.1%}** governed normal-case
               over-blocking rate; safety is not free if the model refuses legitimate work.
            4. Leakage-safe few-shot changes harm by **{few_harm_delta:+.1%}** and accuracy by
               **{few_accuracy_delta:+.1%}** versus governed zero-shot. This tests whether examples
               recover utility without undoing the safety gain.
            5. Schema validity and latency are operational guardrails, not side notes. A safe
               action policy is unusable if the response cannot be parsed or exceeds the workflow
               latency budget.
            6. The intervention tests prompt-layer governance only. Production controls must still
               enforce permissions and approvals outside the model.
            """
        ),
    ]
    return notebook


def mechanism_audit_notebook():
    data_dir = ROOT / "data" / "llm_evaluation"
    mechanism = json.loads((data_dir / "mechanism_summary.json").read_text())
    validity = pd.read_csv(data_dir / "simulator_to_llm_validity.csv").set_index("prompt_mode")
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        cell_markdown(
            f"""
            # Mechanism and Simulator-to-LLM Transfer Audit

            ## tl;dr

            Adding few-shot examples improves exact accuracy in
            **{mechanism["few_shot_accuracy_improvements"]}** scenarios without a safety loss,
            but creates **{mechanism["few_shot_safety_regressions"]}** safety regressions in
            different scenarios. There are **{mechanism["safety_regressions_with_accuracy_gain"]}**
            scenarios where a safety loss accompanies an accuracy gain. The aggregate trade-off
            is therefore a mixture of separable gains and failures, not one unavoidable per-case
            frontier.

            The simulator risk score has baseline AUROC
            **{validity.loc["baseline", "auroc"]:.2f}** against observed model harm, with a 95%
            scenario-bootstrap interval of
            **{validity.loc["baseline", "auroc_ci_low"]:.2f}–{validity.loc["baseline", "auroc_ci_high"]:.2f}**.
            It is useful for stress prioritization but not validated as a calibrated probability
            of real-model failure.
            """
        ),
        cell_markdown(
            """
            ## Context & Methods

            This diagnostic joins the frozen 64-scenario real-model evaluation to the simulator's
            matching no-control task-stressor rows. The join key is `task_id + stressor`, and the
            expected grain is one simulator row per scenario plus three model decisions per
            scenario.

            ### Key Assumptions

            - The simulator's configured `risk_probability` is tested as a score; it was not
              trained or recalibrated on the LLM decisions.
            - AUROC measures ranking only. Brier score and the mean risk-minus-harm gap diagnose
              probability transport, not ranking.
            - Scenario transitions compare governed few-shot with governed zero-shot and therefore
              hold model, task, and stressor fixed.
            - The action-shift analysis can falsify simple direct copying, but cannot by itself
              identify attention dilution or another causal mechanism.
            """
        ),
        cell_markdown("## Data\n\n### 1. Load the joined evidence and persisted diagnostics"),
        cell_code(
            """
            from pathlib import Path
            import json
            import pandas as pd
            import matplotlib.pyplot as plt
            from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

            ROOT = Path.cwd() if (Path.cwd() / "pyproject.toml").exists() else Path.cwd().parent
            data_dir = ROOT / "data" / "llm_evaluation"
            decisions = pd.read_csv(data_dir / "decisions.csv")
            transitions = pd.read_csv(data_dir / "few_shot_transitions.csv")
            transition_summary = pd.read_csv(data_dir / "few_shot_transition_summary.csv")
            scored = pd.read_csv(data_dir / "simulator_to_llm_scenarios.csv")
            validity = pd.read_csv(data_dir / "simulator_to_llm_validity.csv")
            action_shift = pd.read_csv(data_dir / "few_shot_action_shift.csv")
            mechanism = json.loads((data_dir / "mechanism_summary.json").read_text())
            mechanism
            """
        ),
        cell_markdown("### 2. Verify grain, completeness, and join coverage"),
        cell_code(
            """
            assert len(decisions) == 192
            assert decisions["scenario_id"].nunique() == 64
            assert decisions.groupby("scenario_id").size().eq(3).all()
            assert len(scored) == 192
            assert scored["simulator_risk_probability"].notna().all()
            assert scored.groupby("scenario_id")["simulator_risk_probability"].nunique().eq(1).all()
            assert len(transitions) == 64
            assert transition_summary["scenarios"].sum() == 64
            assert action_shift.groupby("workflow").size().eq(2).all()
            "All grain and coverage checks passed"
            """
        ),
        cell_markdown("## Results\n\n### 3. Recompute the scenario transition decomposition"),
        cell_code(
            """
            recomputed = transitions.groupby("transition").size().rename("recomputed")
            persisted = transition_summary.set_index("transition")["scenarios"].rename("persisted")
            comparison = persisted.to_frame().join(recomputed, how="left").fillna(0).astype(int)
            assert comparison["persisted"].equals(comparison["recomputed"])
            assert ((transitions["harm_change"] == 1) & (transitions["accuracy_change"] == 1)).sum() == 0
            comparison
            """
        ),
        cell_markdown("### 4. Independently recompute simulator-to-LLM validity"),
        cell_code(
            """
            rows = []
            for mode, frame in scored.groupby("prompt_mode"):
                observed = frame["harmful_action"].astype(int)
                predicted = frame["simulator_risk_probability"]
                rows.append({
                    "prompt_mode": mode,
                    "auroc": roc_auc_score(observed, predicted),
                    "average_precision": average_precision_score(observed, predicted),
                    "brier_score": brier_score_loss(observed, predicted),
                    "observed_harm_rate": observed.mean(),
                    "mean_simulator_risk": predicted.mean(),
                })
            recomputed_validity = pd.DataFrame(rows).set_index("prompt_mode")
            persisted_validity = validity.set_index("prompt_mode")
            for column in recomputed_validity.columns:
                pd.testing.assert_series_equal(
                    recomputed_validity[column].sort_index(),
                    persisted_validity[column].sort_index(),
                    check_names=False,
                )
            recomputed_validity.round(3)
            """
        ),
        cell_markdown("### 5. Test the direct action-copying hypothesis"),
        cell_code(
            """
            workflow_shift = action_shift.drop_duplicates("workflow").set_index("workflow")
            not_demonstrated = workflow_shift[~workflow_shift["terminal_action_demonstrated"]]
            assert len(not_demonstrated) == 3
            assert (not_demonstrated["few_shot_minus_governed_terminal_rate"] > 0).all()
            workflow_shift[[
                "terminal_action",
                "terminal_action_demonstrated",
                "demonstrated_actions",
                "few_shot_minus_governed_terminal_rate",
            ]].sort_values("few_shot_minus_governed_terminal_rate", ascending=False)
            """
        ),
        cell_markdown("### 6. Visualize terminal-action shifts"),
        cell_code(
            """
            pivot = action_shift.pivot(index="workflow", columns="prompt_mode", values="terminal_action_rate")
            pivot = pivot.loc[["refund", "email", "data_export", "it_access"]]
            ax = pivot.plot(kind="bar", figsize=(10, 5), color=["#2563EB", "#D4A72C"])
            ax.set(
                ylim=(0, 1.05),
                ylabel="Terminal-action selection rate",
                xlabel="Workflow",
                title="Terminal-action selection before and after few-shot examples",
            )
            ax.legend(["Governed", "Governed + few-shot"], frameon=False)
            plt.xticks(rotation=0)
            plt.tight_layout()
            """
        ),
        cell_markdown(
            f"""
            ## Takeaways

            1. The observed safety–utility trade-off is compositional. Few-shot creates
               **{mechanism["few_shot_accuracy_improvements"]}** clean accuracy gains and
               **{mechanism["few_shot_safety_regressions"]}** safety losses elsewhere. A
               conditional policy could in principle retain the former without accepting the
               latter, if those scenarios can be identified prospectively.
            2. Direct action copying is not sufficient to explain the failure. Terminal actions
               were absent from the email, data-export, and IT-access demonstrations, but their
               selection rate increased in all three workflows.
            3. The simulator is not a portable probability model. Its mean score remains 59.2%
               across prompt arms while observed harm ranges from 9.4% to 48.4%.
            4. The next discriminating experiment should compare boundary-focused examples,
               length-matched placebo context, and policy repetition. That design can separate
               example semantics from context-length or attention-dilution effects.
            """
        ),
    ]
    return notebook


def main():
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    nbf.write(model_notebook(), NOTEBOOKS / "01_offline_model_evaluation.ipynb")
    nbf.write(control_notebook(), NOTEBOOKS / "02_control_portfolio_science.ipynb")
    llm_data = ROOT / "data" / "llm_evaluation" / "aggregate.csv"
    if llm_data.exists():
        nbf.write(real_llm_notebook(), NOTEBOOKS / "03_real_llm_evaluation.ipynb")
    mechanism_data = ROOT / "data" / "llm_evaluation" / "mechanism_summary.json"
    if mechanism_data.exists():
        nbf.write(
            mechanism_audit_notebook(),
            NOTEBOOKS / "04_mechanism_and_transfer_audit.ipynb",
        )


if __name__ == "__main__":
    main()
