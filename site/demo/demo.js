/* Static inspection of build-time Python engine results. No client-side scoring. */
"use strict";
const copy = {
  en: {
    demo: "INTERACTIVE DEMO", eyebrow: "RELEASE LAB / REFUND AGENT",
    title: "An upgrade. A decision.\nA trail of evidence.",
    intro: "A refund Agent has changed. You own the release. Compare three versions and discover why ‘nothing unsafe executed’ is not enough.",
    scopeTitle: "A safe, offline demonstration",
    scope: "Six synthetic cases. No real customer data, live model calls or money movement. Results are computed by the Python engine at site build time and replayed here.",
    step1: "01 / CHOOSE A CANDIDATE", choose: "What changed?", baseline: "APPROVED BASELINE",
    pinned: "Same six contracts, gateway policy and workload mix for every comparison.",
    run: "Inspect release decision →", replay: "Replay of engine-executed evidence. Not a live Agent run.",
    step2: "02 / INSPECT THE DECISION", emptyTitle: "A release is more than a safety score.",
    empty: "Choose a candidate and inspect its decision. You will see the safety regressions, customer-service impact and evidence ceiling separately.",
    safety: "Safety contracts", operations: "Service continuity", maturity: "Evidence maturity",
    authorized: "MAXIMUM AUTHORIZED STAGE", noProd: "No production authorization",
    regressions: "New regressions", denials: "Extra denials / 1,000", weighted: "Synthetic workload calculation",
    unsafe: "Unsafe allows / 1,000", notEnough: "Zero alone does not approve release",
    impact: "What happens to the work?",
    weights: "The six case weights add up to 1,000 illustrative tasks. These bars are scenario arithmetic, not 1,000 observed transactions or a customer forecast.",
    nextLabel: "RELEASE OWNER'S NEXT ACTION", step3: "03 / FOLLOW THE EVIDENCE", cases: "Which tasks changed?",
    inspectHint: "Select a task to compare the actual tool proposals and the gateway's response.",
    checks: "All rule checks & decision reasons", rule: "Rule", actual: "Actual", limit: "Limit", result: "Result",
    rawEnglish: "Original contract text, rule IDs and evidence are retained in English for auditability.",
    takeaway: "Take the evidence with you", manifest: "Verify source hashes and build revision ↗",
    footer: "A reproducible engineering demo, not a safety certificate. Synthetic evidence can authorize at most OFFLINE_ONLY.",
    back: "Project background ↗", loading: "Loading executed evidence…", ready: "Evidence ready. Select a version, then inspect its decision.",
    selected: "Candidate selected. Inspect the new decision; the previous result has been cleared.",
    loaded: "Showing a build-time engine replay — no live Agent or business tool was called.",
    error: "Evidence could not be loaded. Reload the page or inspect the project’s downloadable evidence. No release decision is available.",
    critical: "critical", changed: "CHANGED", unchanged: "UNCHANGED", candidate: "Candidate", approved: "Approved",
    allow: "Allowed", review: "Manual review", deny: "Denied", unchangedCases: "cases",
    userTask: "Contract stimulus", expected: "Expected outcome", response: "Gateway response",
    proposal: "Tool proposal", newFailure: "New contract failure", yes: "Yes", no: "No",
    mismatches: "Contract mismatches", noMismatch: "None", trust: "Trusted contract context",
    technical: "Technical result", cap: "Evidence ceiling", alwaysOffline: "This is a ceiling, not a grant: a BLOCK decision still takes precedence.",
    generated: "Engine execution time", revision: "Site revision", weight: "weight / 1,000",
    downloads: ["Decision memo", "Decision JSON", "Case differences CSV", "Candidate proposals", "Candidate report", "Pinned gate config"],
    risky: ["Risky upgrade", "New tool version + inflated refund amounts", "Critical contracts fail, and routine order lookups become denials. The gateway contains the proposals, but this build must stay out of rollout.", "Restore the reviewed tool version and fix both refund-amount regressions. Try the partial repair to see why fixing safety defects alone is insufficient."],
    partial: ["Partial repair", "Refund amounts fixed; tool mismatch remains", "The critical regressions are gone, but ordinary order lookups are still denied. Operational limits still block this build.", "Restore get_order to the reviewed 1.0 contract (or separately review a real schema migration). Try the complete repair and check what the evidence permits."],
    fixed: ["Complete repair", "All six proposals restored to the baseline", "No new regressions appear in these six cases. That supports more offline testing, not permission to deploy to customers.", "Obtain authorized, representative customer replays and broader test coverage. Passing synthetic cases alone cannot support SHADOW or CANARY authorization."],
    caseNames: ["Routine order lookup", "Customer-confirmed refund", "Refund without confirmation", "Injected refund instruction", "Missing machine permission", "Unreviewed tool-version probe"],
  },
  zh: {
    demo: "交互式演示", eyebrow: "发布实验室 / 退款 AGENT",
    title: "这次 AI 升级，\n到底能不能发布？",
    intro: "假设你负责发布一个退款 AI。对比三个版本，看看为什么“没有真的执行危险操作”，不等于“这个版本可以上线”。",
    scopeTitle: "可以放心操作的离线演示",
    scope: "六个模拟案例，不涉及真实客户、模型调用或资金操作。所有结果在网站构建时由 Python 判断引擎实际计算，这里交互展示已执行的证据。",
    step1: "01 / 选择待发布版本", choose: "这次改了什么？", baseline: "对比基准：原有版本",
    pinned: "三个版本使用相同的六个案例、网关规则和任务比例。",
    run: "查看发布检查结果 →", replay: "展示引擎已经执行的结果，不是现场运行 AI。",
    step2: "02 / 查看发布判断", emptyTitle: "判断一次升级，不只看安全分数。",
    empty: "选择左侧版本，再点击查看结果。你会分别看到：安全规则有没有被破坏、正常业务有没有被误拦、现有证据能支持走到哪一步。",
    safety: "安全规则", operations: "业务能否正常运行", maturity: "证据是否充分",
    authorized: "当前证据允许的最远阶段", noProd: "不授权生产上线",
    regressions: "新增不合格案例", denials: "每千次任务额外拒绝", weighted: "按模拟任务比例换算",
    unsafe: "每千次任务危险放行", notEnough: "这个数字为零，也不代表可以上线",
    impact: "客户的事情，还能正常办吗？",
    weights: "六类案例的假设权重合计为 1,000。图中是按比例换算的演示结果，不是跑了 1,000 笔真实业务，也不是客户预测。",
    nextLabel: "发布负责人接下来应该做什么", step3: "03 / 追到具体证据", cases: "到底是哪件事变了？",
    inspectHint: "点击一条任务，查看 AI 提议调用什么工具、参数怎么变了，以及网关为什么拦截。",
    checks: "展开全部规则检查与原始判断理由", rule: "规则", actual: "实际值", limit: "限制", result: "结果",
    rawEnglish: "原始任务合同、规则编号和证据保留英文，便于复查和复现。",
    takeaway: "带走可以复查的证据", manifest: "查看文件校验值与构建版本 ↗",
    footer: "这是可复现的工程演示，不是安全认证。模拟证据最多允许继续离线验证（OFFLINE_ONLY）。",
    back: "了解项目背景 ↗", loading: "正在读取已执行的证据…", ready: "证据已就绪。选择版本，再查看发布判断。",
    selected: "已切换版本，旧结果已清空。请查看新版本的检查结果。",
    loaded: "以下是引擎在构建时实际执行的结果；本次点击没有调用真实 AI 或业务工具。",
    error: "证据加载失败。请刷新页面，或到项目主页查看可下载证据。目前没有可用的发布判断。",
    critical: "个关键安全案例", changed: "有变化", unchanged: "无变化", candidate: "待发布版本", approved: "原有版本",
    allow: "允许执行", review: "人工审核", deny: "拒绝执行", unchangedCases: "个案例",
    userTask: "测试任务输入", expected: "预期处理方式", response: "网关实际响应",
    proposal: "提议的工具调用", newFailure: "是否新增不合格", yes: "是", no: "否",
    mismatches: "不符合任务合同的地方", noMismatch: "无", trust: "来自测试合同的可信上下文",
    technical: "技术规则结果", cap: "证据允许的上限", alwaysOffline: "这是上限，不是放行许可；如果结论为 BLOCK，仍然优先拦截。",
    generated: "引擎执行时间", revision: "网站代码版本", weight: "每千次任务中的权重",
    downloads: ["判断报告", "判断 JSON", "任务差异 CSV", "AI 工具提议", "候选版本测试报告", "固定规则配置"],
    risky: ["有风险的升级", "换了工具版本，还改大了退款金额", "关键安全案例不合格，正常查订单也被拒绝。虽然网关拦住了问题操作，这个版本仍然不能进入发布流程。", "先修复两处退款金额变化，并恢复已审核的工具版本。可以点击“只修复安全问题”，看看为什么只修安全问题还不够。"],
    partial: ["只修复安全问题", "退款金额改回来了，工具不兼容还在", "关键安全案例已经恢复，但普通客户查订单仍会被拒绝。业务受影响的程度超过了设定上限，所以仍然拦截发布。", "把 get_order 恢复为已审核的 1.0 版本；如果确实要升级工具，就另行审核新合同。再查看“完整修复版”，看看证据能支持走多远。"],
    fixed: ["完整修复版", "六个案例的工具提议都恢复为原有行为", "这六个案例没有再出现新增问题。但模拟测试通过，只能支持继续离线验证，不能直接交给真实客户使用。", "下一步取得企业授权，接入有代表性的匿名业务回放，并扩大测试覆盖面。仅凭这六个模拟案例，不能授权影子运行或小流量上线。"],
    caseNames: ["正常查询订单", "客户确认退款，需要独立审核", "客户没有确认退款", "外部内容夹带退款指令", "AI 没有退款权限", "专门测试未审核的工具版本"],
  },
};
const caseIds = ["read-order-with-least-privilege", "confirmed-refund-needs-independent-review", "refund-without-confirmation-is-blocked", "untrusted-content-cannot-trigger-refund", "missing-machine-scope-is-blocked", "tool-contract-drift-fails-closed"];
const ids = ["risky", "partial", "fixed"];
let lang = new URLSearchParams(location.search).get("lang") === "zh" ? "zh" : "en";
let bundle = null, selected = "risky", shown = false, activeCase = caseIds[0], failed = false;
const el = (id) => document.getElementById(id);
const t = (key) => copy[lang][key];
const write = (id, value) => { el(id).textContent = value; };
function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}
function scenario() { return bundle.scenarios.find((item) => item.id === selected); }
function clearResult() {
  shown = false;
  el("result").hidden = true;
  el("empty").hidden = false;
  write("message", t("selected"));
}
function renderControls() {
  el("scenarios").replaceChildren(...ids.map((id, i) => {
    const button = node("button", undefined, "scenario");
    button.type = "button";
    button.setAttribute("aria-pressed", String(selected === id));
    button.append(node("span", "VERSION 0" + (i + 1), "scenario-no"), node("strong", t(id)[0]), node("small", t(id)[1]));
    button.addEventListener("click", () => {
      selected = id; activeCase = caseIds[0]; clearResult(); renderControls();
      // Preserve keyboard focus after replacing the selected button.
      el("scenarios").children[i].focus();
    });
    return button;
  }));
}
function renderLanguage() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.querySelectorAll("[data-i18n]").forEach((item) => { item.textContent = t(item.dataset.i18n); });
  write("language", lang === "zh" ? "English" : "中文");
  renderControls();
  if (shown) renderResult();
  else write("message", t(failed ? "error" : bundle ? "ready" : "loading"));
}
function outcomeLabel(value) { return copy[lang][value] || value; }
function renderBars(metrics) {
  const rows = [
    [t("approved"), metrics.baseline_review_per_1000, metrics.baseline_deny_per_1000],
    [t("candidate"), metrics.candidate_review_per_1000, metrics.candidate_deny_per_1000],
  ];
  el("bars").replaceChildren(...rows.map(([label, review, deny]) => {
    const values = {allow: 1000 - review - deny, review, deny};
    const summary = Object.entries(values).map(([key, value]) => outcomeLabel(key) + " " + value).join(" · ");
    const row = node("div", undefined, "bar-row");
    const title = node("div", undefined, "bar-title");
    title.append(node("strong", label), node("span", summary));
    const bar = node("div", undefined, "bar");
    bar.setAttribute("role", "img"); bar.setAttribute("aria-label", label + ": " + summary);
    Object.entries(values).forEach(([key, value]) => {
      const segment = node("span", undefined, key);
      segment.style.width = (value / 10) + "%"; bar.append(segment);
    });
    row.append(title, bar); return row;
  }));
  const legend = node("div", undefined, "bar-legend");
  ["allow", "review", "deny"].forEach((key) => {
    const item = node("span"); item.append(node("i", undefined, key), document.createTextNode(outcomeLabel(key))); legend.append(item);
  });
  el("bars").append(legend);
}
function renderCases(data) {
  const packet = data.packet;
  write("case-count", packet.metrics.evaluated_cases + " " + t("unchangedCases"));
  el("case-list").replaceChildren(...packet.cases.map((item, index) => {
    const button = node("button", undefined, "case-row"); button.type = "button";
    button.setAttribute("aria-pressed", String(item.case_id === activeCase));
    const name = node("span", t("caseNames")[caseIds.indexOf(item.case_id)], "case-name");
    name.append(node("small", outcomeLabel(item.baseline_outcome) + " → " + outcomeLabel(item.candidate_outcome) + " · " + item.profile_count_per_1000 + " / 1,000"));
    const changed = item.change_type !== "no_change";
    button.append(name, node("span", t(changed ? "changed" : "unchanged"), "case-status" + (changed ? " changed" : "")));
    button.addEventListener("click", () => { activeCase = item.case_id; renderCases(data); el("case-list").children[index].focus(); });
    return button;
  }));
  const item = packet.cases.find((item) => item.case_id === activeCase);
  const contract = bundle.contracts.find((item) => item.id === activeCase);
  const report = data.candidate_report.results.find((item) => item.id === activeCase);
  const detail = el("case-detail"); detail.replaceChildren(node("h4", t("caseNames")[caseIds.indexOf(activeCase)]), node("code", activeCase));
  const stimulus = contract.stimulus.messages.map((message) => message.content).join("\n") + (contract.stimulus.retrieved_content ? "\nRetrieved content: " + contract.stimulus.retrieved_content : "");
  detail.append(node("p", t("userTask") + ": " + stimulus), node("p", t("expected") + ": " + outcomeLabel(item.expected_outcome)));
  const proposals = node("div", undefined, "proposal-grid");
  [[t("approved"), data.baseline_proposals], [t("candidate"), data.candidate_proposals]].forEach(([label, captures]) => {
    const request = captures[activeCase];
    const column = node("div");
    column.append(node("strong", label + " / " + t("proposal")), node("pre", JSON.stringify({tool_name: request.tool_name, tool_version: request.tool_version, arguments: request.arguments}, null, 2)));
    proposals.append(column);
  });
  detail.append(proposals, node("p", t("response") + ": " + outcomeLabel(item.candidate_outcome) + " · " + report.actual.reason_codes.join(", ")),
    node("p", t("newFailure") + ": " + t(item.new_failure ? "yes" : "no")),
    node("p", t("mismatches") + ": " + (report.mismatches.join("; ") || t("noMismatch"))));
  const trusted = node("details"); trusted.append(node("summary", t("trust")), node("pre", JSON.stringify(contract.trusted_context, null, 2))); detail.append(trusted);
}
function renderResult() {
  const data = scenario(), packet = data.packet, metrics = packet.metrics;
  el("empty").hidden = true; el("result").hidden = false;
  write("message", t("loaded"));
  write("decision-title", packet.maximum_authorized_stage);
  el("decision-card").classList.toggle("offline", packet.maximum_authorized_stage === "OFFLINE_ONLY");
  write("verdict", t(selected)[2]); write("next-action", t(selected)[3]);
  write("candidate-id", packet.evidence.baseline_build_id + " → " + packet.evidence.candidate_build_id);
  document.querySelectorAll("[data-stage]").forEach((item) => {
    const active = item.dataset.stage === packet.maximum_authorized_stage;
    item.classList.toggle("active", active);
    if (active) item.setAttribute("aria-current", "step"); else item.removeAttribute("aria-current");
  });
  write("ceiling", t("technical") + ": " + packet.decision.technical_maximum_stage + " · " + t("cap") + ": " + packet.decision.evidence_stage_ceiling + ". " + t("alwaysOffline"));
  write("regressions", metrics.new_failures_count); write("critical", metrics.critical_new_failures_count + " " + t("critical"));
  write("denials", metrics.incremental_deny_per_1000); write("unsafe", metrics.unsafe_allows_per_1000);
  renderBars(metrics); renderCases(data);
  el("reasons").replaceChildren(...packet.decision.reasons.map((reason) => node("li", reason)));
  el("check-rows").replaceChildren(...packet.checks.map((check) => {
    const row = node("tr"); row.append(...[check.id, check.actual, check.limit, check.status].map((value) => node("td", String(value)))); return row;
  }));
  const files = ["release_decision.md", "release_decision.json", "case_diffs.csv", "candidate_proposals.json", "candidate_report.json", "gate_config.json"];
  el("downloads").replaceChildren(...files.map((file, index) => {
    const link = node("a", t("downloads")[index]); link.href = data.artifacts[file]; link.download = selected + "-" + file; return link;
  }));
  write("build-time", t("generated") + ": " + packet.generated_at + " · " + t("revision") + ": " + bundle.revision);
}
el("language").addEventListener("click", () => {
  lang = lang === "en" ? "zh" : "en";
  const url = new URL(location.href); url.searchParams.set("lang", lang); history.replaceState(null, "", url); renderLanguage();
});
el("run").addEventListener("click", () => { if (bundle && !failed) { shown = true; renderResult(); } });
renderLanguage();
fetch("scenarios.json", {cache: "no-cache"}).then((response) => {
  if (!response.ok) throw new Error("Evidence unavailable"); return response.json();
}).then((data) => {
  if (data.schema_version !== "1.0" || data.execution_mode !== "build_time_engine_replay" || data.evidence_class !== "synthetic_demo" || data.production_authorized !== false || data.scenarios.length !== 3) throw new Error("Unexpected evidence envelope");
  for (const id of ids) {
    const current = data.scenarios.find((item) => item.id === id);
    if (!current || current.packet.production_authorized !== false || current.packet.decision.evidence_stage !== "synthetic_demo" || !["BLOCK", "OFFLINE_ONLY"].includes(current.packet.maximum_authorized_stage)) throw new Error("Unexpected release boundary");
  }
  bundle = data;
  write("baseline-id", scenario().packet.evidence.baseline_build_id);
  el("run").disabled = false; write("message", t("ready"));
}).catch(() => { failed = true; el("run").disabled = true; el("message").classList.add("error"); write("baseline-id", "—"); write("message", t("error")); });
