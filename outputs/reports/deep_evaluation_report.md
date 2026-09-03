# Deep Evaluation Report

## Decision summary

The project now demonstrates both an end-to-end offline evaluation method and a controlled local-
model behavior experiment, but it is not evidence that any production agent system is safe. The
most defensible behavioral result is that a trust-hierarchy governance prompt substantially
reduces unauthorized terminal actions on the frozen scenario set, at a large over-blocking cost.
The most defensible simulator result remains Context Envelope + Tool Version Lock + Permission
Scope as the lowest-risk feasible portfolio under the stated assumptions and guardrails.

Overall evidence status: **share with strong caveats as a research portfolio; not ready for a
production-safety decision**. The simulator-informed score materially overstates the performance
available from deployment-observable inputs, failure attribution is weak, and the real-model
experiment covers only one small local model with decision-only tools.

v0.6 adds an engineering response to this evidence: a runtime Action Gateway with deterministic
policy checks, separate approvals, exact-argument execution grants, replay protection, and a
tenant audit chain. That implementation closes the architectural gap between prompt advice and
tool authorization, but the research numbers above do not measure its effect in a real enterprise.

## Real-model behavioral evidence

The pinned Qwen2.5 3B Q4_K_M model completed 192 decisions: the same 64 scenarios under baseline,
governed zero-shot, and governed plus leakage-safe few-shot prompts. All responses satisfied the
schema; all scenario triplets are complete; the eight few-shot tasks have zero overlap with the
evaluation tasks.

Governed zero-shot reduced harmful actions from 48.44% to 9.38%, a paired difference of -39.06
points (95% scenario-bootstrap CI -51.56 to -26.56). Safety improved in 25 scenarios and regressed
in none. This was not a free gain: action accuracy remained effectively unchanged at 20.31%
versus 21.88% (CI for the difference -10.94 to +7.81), while normal-case over-blocking rose from
18.75% to 62.50% (difference +43.75 points, CI +28.12 to +62.50).

Leakage-safe few-shot partially restored utility but weakened the safety boundary. Compared with
governed zero-shot, action accuracy rose 9.38 points, but its CI crossed zero (-1.56 to +20.31).
Normal-case over-blocking fell 37.50 points (CI -56.25 to -21.88), while harmful actions rose
28.12 points (CI +17.19 to +39.06). The largest failure was data export, where the few-shot arm
selected unauthorized row export in 81.25% of cases. Policy drop and human-review bottleneck were
the weakest stressors at 75.00% and 62.50% harmful actions.

The interpretation is not that few-shot is globally harmful. The examples improved label and
scope recognition for some legitimate tasks, especially refunds, but they also shifted the small
model toward terminal actions under high-risk conditions. The correct control response is
external enforcement: deny-by-default tool authorization, argument validation, and non-bypassable
approval gates—not another prompt-only patch.

## The aggregate trade-off is compositional, not scenario-level

The aggregate result initially looks like a conventional safety-utility frontier: few-shot raises
accuracy while also raising harm. Scenario-level transitions show that interpretation is too
coarse. Nine scenarios gain exact accuracy without any safety loss. Fifteen scenarios lose safety
without gaining accuracy, and three lose both accuracy and safety. The remaining 37 do not change
on either primary metric. No scenario simultaneously gains accuracy and loses safety.

This matters because the 18 safety regressions are not the local price paid for the nine accuracy
gains. They occur in different tasks. A prospective routing policy could in principle retain the
clean gains and suppress the regressions if the scenario types can be distinguished before action.
That is now a concrete classification and intervention question, not a generic claim that safety
and usefulness always conflict.

## Direct example-action copying is falsified as the sole mechanism

Only the refund examples contain that workflow's high-impact terminal action. Email examples show
`create_draft`, data-export examples show `export_aggregate`, and IT-access examples show
`security_review`. Despite never seeing `send_email`, `export_customer_data`, or
`grant_permission` as correct development actions, the few-shot arm increases their selection in
all three workflows. Data export rises from 31.25% to 81.25%; IT access rises from 6.25% to 50%;
email rises from 0% to 12.50%.

The data therefore rejects the simplest explanation that the model merely copied a demonstrated
terminal action. It does not identify the replacement explanation. Extra context may dilute the
policy instruction, example formatting may shift the model toward action completion, or system-
message ordering may alter attention. Distinguishing those mechanisms requires randomized,
length-matched prompt ablations.

## Simulator probabilities do not transport to real-model harm

The simulator assigns the same mean no-control risk score, 59.24%, to the scenario set in every
prompt arm because task and stressor are held fixed. Observed model harm is 48.44% under baseline,
9.38% under governed zero-shot, and 37.50% under governed few-shot. The score therefore cannot be
interpreted as a calibrated probability of real-model failure.

Ranking transfer is also limited. Simulator risk against baseline observed harm reaches AUROC
0.573 with a 95% scenario-bootstrap interval of 0.433-0.710. Governed reaches 0.648 with a much
wider interval of 0.445-0.842 because only six harmful events remain. Few-shot reaches 0.656 with
an interval of 0.509-0.787. These results support using the simulator to construct and prioritize
stress cases, but not using its risk probability as a real-model safety certificate.

## Model evidence

Structured Logistic Regression leads the frozen holdout at PR-AUC 0.773, AUROC 0.791, F1 0.696,
and safety recall 70.82%, with 27.59% over-blocking. The rule baseline remains competitive at
PR-AUC 0.750, which shows that the synthetic mechanisms are still largely explainable by known
rules rather than complex learned structure.

## Feature-access audit

The original 0.773 PR-AUC model receives 12 simulator-only or ground-truth-like variables,
including mechanism-integrity flags, stressor intensity, expected control effectiveness,
workflow base risk, and encoded case risk. These fields are useful for simulator diagnostics but
would not all be observable before a production action.

After removing them, structured PR-AUC falls to 0.709, F1 to 0.594, and safety recall to 54.21%.
Adding request and policy text raises recall to 58.89% but lowers PR-AUC to 0.685. A
shuffled-label negative control reaches 0.429 PR-AUC. The 0.064 simulator-to-deployable gap is a
material information-access effect, not a formatting difference.

The calibrated variant has F1 0.684 with a 95% task-cluster bootstrap interval of 0.595-0.755.
However, isotonic calibration worsens held-out Brier score (0.186 to 0.193) and PR-AUC (0.773 to
0.719). Validation-selected calibration therefore did not generalize. The appropriate action is
to keep raw scores and gather more independent calibration tasks.

Text features do not add value: removing them raises F1 by 0.029 and recall by 0.065. The likely
cause is templated benchmark language. Policy, tool, and graph ablations have near-zero marginal
effect because their signals are redundant with scenario, workflow-risk, and mechanism fields.
This is a useful warning that the dataset does not yet identify each feature family cleanly.

Unseen stressor transfer is acceptable within this simulator (PR-AUC 0.818 and 0.744), but
workflow transfer is not. The held-out email workflow falls to PR-AUC 0.380 and F1 0.393. Before
any real evaluation, the benchmark should diversify email cases and consider workflow-specific
thresholds or hierarchical models.

## Evaluation-task evidence

The project now evaluates four tasks with different populations and input contracts.

- Risk classification uses pre-action fields and has separate simulator-informed, deployable,
  text-augmented, and label-shuffled tracks.
- Failure attribution uses 939 held-out incident rows and structured post-action trace summaries.
  It excludes stressor identity and raw trace text. Macro-F1 is only 0.171; F04 reaches 38.57%
  recall, while F02 reaches 8.87%. The trace schema does not preserve enough root-cause evidence.
- Severity prediction excludes blast radius from model inputs and reaches Macro-F1 0.906 and
  balanced accuracy 91.81% across moderate, high, and critical incidents.
- Governance recommendation predicts the empirically lowest-loss single control for 152 held-out
  no-control incidents. It reaches 53.29% Top-1, 95.39% Top-3, and mean decision regret 12.59,
  compared with regret 57.03 for the majority baseline.

Strict governance transfer remains poor. On unseen permission-overgrant cases, Top-3 accuracy
falls to 38.10% and mean regret rises to 68.73. The recommender is suitable only for ranked
decision support under known stress conditions.

## Control evidence

The no-control stressed incident rate is 62.21%. Exhaustive search over 64 portfolios finds a
cost-27 bundle with 33.71% incidents, 85.43% task success, 27.64% review load, and 42.86%
worst-workflow incidents. This is a 28.50-point absolute reduction.

This result corrects the old additive optimizer. Joint measurement selects Tool Version Lock,
not External Isolation, reducing the recommended cost from 36 to 27. The strongest positive
interaction is Context Envelope + Permission Scope (+0.0161); the strongest diminishing return
is Context Envelope + Selective Human Review (-0.0115).

Selective Human Review has the largest Shapley marginal value (0.1816), but cost 45 makes it
infeasible under the default budget. Tool Version Lock leads on Shapley value per cost. Across 12
seeds the chosen bundle averages 33.69% incidents with standard deviation 1.14 points and an
observed range of 32.00%-35.86%.

## Recommended next experiment

First run a mechanism-discriminating prompt experiment on the same paired scenarios: governed
zero-shot, length-matched unlabeled context, the current examples, boundary-focused negative
examples, and policy repetition after examples. This can separate context length, example
semantics, and policy-position effects. Then preregister an external task set authored and
adjudicated outside the benchmark generator, repeat conditions across several model families and
samples, and connect the decision interface to a staged deny-by-default tool gateway. Preserve
proposed arguments, authorization outcomes, final actions, latency, review requests, and rollback
events, but do not collect hidden chain of thought.
