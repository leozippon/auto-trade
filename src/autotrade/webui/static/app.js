/* ADM-Cube research console SPA — no build step, no dependencies. */

const $main = document.getElementById("main");
const $topbarRight = document.getElementById("topbar-right");
const $modalRoot = document.getElementById("modal-root");
const $toastRoot = document.getElementById("toast-root");

const STATE_LABELS = {
  launching: "启动中",
  initializing: "初始化中",
  running_session: "运行中",
  paused: "已暂停",
  completed: "已完成",
  stopped: "已停止",
  failed: "失败",
  interrupted: "已中断",
  terminated: "已强制终止",
  created: "未启动",
  unreadable: "不可解析",
  unknown: "未知",
};
// How a research session ended (pipelines/config.py SESSION_OUTCOMES).
const OUTCOME_LABELS = {
  continue: "继续",
  freeze: "提名冻结",
  no_edge: "无边际",
  deadline: "到时",
};
const VERDICT_LABELS = {
  graduated: "graduated",
  discarded: "discarded",
  no_deliverable: "无交付",
};
// Failed conditions of the freeze gate and of the forward and Held-out verdict
// (pipelines/verdict.py). A reason not listed renders as written.
const REASON_LABELS = {
  freeze_needs_full_span_validation: "提名节点不是全区间验证",
  freeze_too_few_full_span_validations: "全区间验证不足 2 次",
  freeze_deflated_sharpe_unavailable: "去偏 Sharpe 概率算不出",
  freeze_deflated_sharpe_below_threshold: "去偏 Sharpe 概率低于 0.5",
  freeze_unmeasurable: "研究期统计无法计算",
  forward_strategy_error: "前推期策略报错",
  heldout_strategy_error: "Held-out 期策略报错",
  forward_lower_bound_not_positive: "前推中性化超额 80% 下界不为正",
  forward_recency_negative: "前推最近 6 个月中性化超额为负",
  forward_max_drawdown_exceeded: "前推回撤超限",
  forward_not_positive_at_cost_stress: "加倍滑点后前推超额不为正",
  forward_too_few_round_trips: "前推平仓次数不足",
  forward_exposure_below_floor: "前推平均仓位不足 0.5",
  heldout_excess_below_tolerance: "Held-out 中性化超额低于容忍线",
  heldout_max_drawdown_exceeded: "Held-out 回撤超限",
  heldout_exposure_below_floor: "Held-out 平均仓位不足 0.5",
};

function reasonLabel(reason) {
  return REASON_LABELS[reason] || String(reason);
}

const ENVIRONMENT_STAGE_LABELS = {
  preparing_session: "准备会话",
  pit_snapshot: "准备 PIT 快照",
  sandbox_layout: "准备 Sandbox 工作区",
  pit_view: "装载 PIT 可见视图",
  sandbox_start: "启动 Sandbox",
  llm_call: "Agent 推理",
  tool_call: "执行工具",
  subagent_wait: "等待子代理",
  backtest: "执行验证回测",
  agent_complete: "Agent 推理完成",
  freezing: "冻结策略",
  forward_replay: "前推与 Held-out 连续回放",
  verdict: "判定毕业",
  publishing: "结果落盘",
  session_retry: "会话失败重试",
};
// Stages with no Agent session to watch: the session panel shows the stage
// instead of a live Trace. The forward replay runs with no Agent at all.
const PREP_ENVIRONMENT_STAGES = new Set([
  "preparing_session",
  "pit_snapshot",
  "sandbox_layout",
  "pit_view",
  "sandbox_start",
  "forward_replay",
  "verdict",
  "session_retry",
]);
// Dead-worker states the backend can relaunch from a ledger resume; mirrors
// manager.py _TERMINAL_RESUMABLE_STATES. Keep in sync or the resume button
// silently disappears for a resumable experiment (e.g. "terminated").
const RESUMABLE_STATES = [
  "stopped",
  "failed",
  "interrupted",
  "terminated",
  "created",
];
// Worker states that carry a live Agent session; mirrors
// hitl_state.LIVE_RUN_STATES. Keep in sync or the console offers message
// injection on a session the backend will refuse.
const LIVE_RUN_STATES = new Set(["running_session"]);
const ACTIVE_SESSION_STATES = LIVE_RUN_STATES;
const INJECT_MESSAGE_MAX_CHARS = 8192;
const INJECT_MESSAGE_QUEUED_NOTE = "已排队，将在 Agent 下一安全点生效";
const TERMINAL_INJECT_STATES = new Set([
  "completed",
  "stopped",
  "failed",
  "interrupted",
  "terminated",
]);

let pollTimer = null;
let liveTimers = [];
let liveSources = [];
const injectDrafts = new Map();

/* ---------------- theme ---------------- */

function currentTheme() {
  return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try {
    localStorage.setItem("ch_theme", theme);
  } catch {
    /* private mode */
  }
  const button = document.getElementById("theme-toggle");
  if (button) button.textContent = theme === "dark" ? "☀️" : "🌙";
}

/* Theme switches repaint charts in place without rebuilding the page. */
function refreshCharts() {
  document.querySelectorAll(".svg-chart").forEach((node) => {
    if (typeof node.__rerender === "function")
      node.replaceWith(node.__rerender());
  });
}

(function initTheme() {
  let stored = null;
  try {
    stored = localStorage.getItem("ch_theme");
  } catch {
    /* private mode */
  }
  const preferred =
    window.matchMedia &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  applyTheme(stored === "dark" || stored === "light" ? stored : preferred);
  const button = document.getElementById("theme-toggle");
  if (button)
    button.addEventListener("click", () => {
      applyTheme(currentTheme() === "dark" ? "light" : "dark");
      refreshCharts();
    });
})();

/* Per-device UI scale: port-forwarded browsers and embedded webviews disagree
   wildly about effective size; the choice persists per browser profile. */
(function initZoom() {
  const select = document.getElementById("ui-zoom");
  if (!select) return;
  let stored = null;
  try {
    stored = localStorage.getItem("ch_zoom");
  } catch {
    /* private mode */
  }
  const apply = (value) => {
    document.documentElement.style.setProperty("--ui-zoom", value);
    document.body.style.zoom = "";
    try {
      localStorage.setItem("ch_zoom", value);
    } catch {
      /* private mode */
    }
  };
  if (stored && [...select.options].some((option) => option.value === stored)) {
    select.value = stored;
    apply(stored);
  }
  select.addEventListener("change", () => apply(select.value));
})();

(function pinTopbarHeight() {
  const bar = document.querySelector(".topbar");
  if (!bar || typeof ResizeObserver !== "function") return;
  const sync = () => {
    const height = Math.ceil(bar.getBoundingClientRect().height);
    if (height > 0)
      document.documentElement.style.setProperty("--topbar-h", `${height}px`);
  };
  sync();
  new ResizeObserver(sync).observe(bar);
})();

/* Session keys (s1, s2, …, forward) travel in the hash as they are. */
function sessionKeyToUrl(key) {
  return encodeURIComponent(String(key));
}

function sessionKeyFromUrl(segment) {
  return decodeURIComponent(segment);
}

function sessionLabel(key) {
  return key === "forward" ? "前推回放" : `研究 ${key}`;
}

/* Ledger period ranges are serialized as "YYYYMMDD..YYYYMMDD"; render them
   as human dates without touching the stored format. */
function fmtPeriodRange(value) {
  const match = /^(\d{4})(\d{2})(\d{2})\.\.(\d{4})(\d{2})(\d{2})$/.exec(
    String(value || ""),
  );
  if (!match) return value || "—";
  return `${match[1]}-${match[2]}-${match[3]} ～ ${match[4]}-${match[5]}-${match[6]}`;
}

function fmtDate(value) {
  const match = /^(\d{4})(\d{2})(\d{2})$/.exec(String(value || ""));
  return match ? `${match[1]}-${match[2]}-${match[3]}` : String(value || "—");
}

/* All backend timestamps are ISO-UTC; the console displays UTC+8 (Asia/Shanghai)
   regardless of the browser's locale. */
const TS_FMT = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  hour12: false,
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});
const TS_TIME_FMT = new Intl.DateTimeFormat("zh-CN", {
  timeZone: "Asia/Shanghai",
  hour12: false,
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
});

function fmtTs(iso) {
  const ms = Date.parse(iso || "");
  if (Number.isNaN(ms)) return "—";
  return TS_FMT.format(ms).replaceAll("/", "-");
}

function fmtTsTime(iso) {
  const ms = Date.parse(iso || "");
  if (Number.isNaN(ms)) return "";
  return TS_TIME_FMT.format(ms).replaceAll("/", "-");
}

function fmtDuration(totalSeconds) {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(seconds / 3600),
    m = Math.floor((seconds % 3600) / 60),
    s = seconds % 60;
  const mm = String(m).padStart(2, "0"),
    ss = String(s).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${m}:${ss}`;
}

function sessionDurationNode(detail, session, prefix = "", className = "") {
  const node = el("span", { class: className });
  const fixedValue = (session.record || {}).run_wall_seconds;
  const fixed = Number(fixedValue);
  const isFixed =
    fixedValue !== null &&
    fixedValue !== undefined &&
    Number.isFinite(fixed) &&
    fixed >= 0;
  const status = detail.status || {};
  const startedAt =
    status.session_key === session.key
      ? Date.parse(status.session_started_at || "")
      : NaN;
  const isLive =
    !isFixed &&
    detail.worker_alive &&
    ACTIVE_SESSION_STATES.has(status.state) &&
    Number.isFinite(startedAt);
  const update = () => {
    const seconds = isFixed
      ? fixed
      : isLive
        ? Math.max(0, (Date.now() - startedAt) / 1000)
        : null;
    node.textContent = [prefix, seconds === null ? "" : fmtDuration(seconds)]
      .filter(Boolean)
      .join(" · ");
  };
  update();
  if (isLive)
    liveTimers.push(
      setInterval(() => {
        if (node.isConnected) update();
      }, 1000),
    );
  return node;
}

/* ---------------- utilities ---------------- */

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      detail = (await response.json()).detail || detail;
    } catch {
      /* keep status */
    }
    throw new Error(detail);
  }
  return response.json();
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    // Through the CSSOM, never as an attribute: the public site is served
    // under a CSP whose style-src has no 'unsafe-inline', which refuses every
    // inline style attribute. A CSSOM write is outside that check, so the
    // same declaration applies locally and on the deployed site.
    else if (key === "style") node.style.cssText = value;
    else if (key.startsWith("on") && typeof value === "function")
      node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined)
      node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(
      child.nodeType ? child : document.createTextNode(String(child)),
    );
  }
  return node;
}

function toast(message, isError = false) {
  const node = el("div", { class: `toast${isError ? " error" : ""}` }, message);
  $toastRoot.append(node);
  setTimeout(() => node.remove(), isError ? 7000 : 3500);
}

function fmtPct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function fmtSharpe(value) {
  return value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : Number(value).toFixed(2);
}

function formatStageLine(status, { elapsed = true } = {}) {
  const stage = status && status.environment_stage;
  if (!stage) return "";
  const label = ENVIRONMENT_STAGE_LABELS[stage] || stage;
  const progress = (status && status.environment_progress) || {};
  const done = Number(progress.completed ?? progress.day_index);
  const total = Number(progress.total ?? progress.total_days);
  const measured =
    Number.isFinite(done) && Number.isFinite(total) && total > 0
      ? ` ${done}/${total}`
      : "";
  const action = progress.tool
    ? ` · ${progress.tool}`
    : progress.call_index
      ? ` · 第 ${progress.call_index} 次调用`
      : "";
  if (!elapsed) return `${label}${measured}${action}`;
  const started = Date.parse(
    status.environment_stage_started_at || status.session_started_at || "",
  );
  const wait = Number.isFinite(started)
    ? ` · ${fmtDuration((Date.now() - started) / 1000)}`
    : "";
  return `${label}${measured}${action}${wait}`;
}

function isPrepEnvironment(status, state) {
  const stage = (status && status.environment_stage) || "";
  if (PREP_ENVIRONMENT_STAGES.has(stage)) return true;
  return (
    !stage &&
    (state === "running_session" ||
      state === "initializing" ||
      state === "launching")
  );
}

function signCls(value) {
  if (value === null || value === undefined) return "";
  return value >= 0 ? "pos" : "neg";
}

function stateBadge(state) {
  return el(
    "span",
    { class: `badge state-${state}` },
    STATE_LABELS[state] || state,
  );
}

function escapeHtml(text) {
  return String(text).replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
}

/* ---------------- charts ----------------
   Specs: thin marks (bars ≤24px, 2px surface gap, 4px rounded data-end,
   square baseline; 2px lines; ≥8px markers with 2px surface ring), hairline
   solid gridlines, muted-ink labels, legend, hover tooltips. One categorical
   slot (blue) against a neutral dashed 沪深300, validated on both panels. */
function themeInk() {
  if (currentTheme() === "dark") {
    return {
      strategyColor: "#3987e5",
      grid: "#2b303c",
      baseline: "#4a5163",
      muted: "#98a0af",
      ring: "#1b1f28",
    };
  }
  return {
    strategyColor: "#2a78d6",
    grid: "#e9ebf1",
    baseline: "#c2c7d2",
    muted: "#68717f",
    ring: "#ffffff",
  };
}

let $chartTip = null;
function chartTipNode() {
  if (!$chartTip) {
    $chartTip = el("div", { class: "chart-tip" });
    document.body.append($chartTip);
  }
  return $chartTip;
}

function bindChartTips(wrap) {
  const tip = chartTipNode();
  wrap.addEventListener("mousemove", (event) => {
    const target = event.target.closest("[data-tip]");
    if (!target) {
      tip.style.display = "none";
      return;
    }
    tip.textContent = target.getAttribute("data-tip");
    tip.style.display = "block";
    const pad = 14;
    const rect = tip.getBoundingClientRect();
    let x = event.clientX + pad,
      y = event.clientY + pad;
    if (x + rect.width > window.innerWidth - 8)
      x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8)
      y = event.clientY - rect.height - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  });
  wrap.addEventListener("mouseleave", () => {
    tip.style.display = "none";
  });
  return wrap;
}

function chartLegend(seriesList) {
  return el(
    "div",
    { class: "chart-legend" },
    ...seriesList.map((series) =>
      el(
        "span",
        { class: "legend-item" },
        el("span", {
          class: "legend-swatch",
          style: `background:${series.color}`,
        }),
        series.label,
      ),
    ),
  );
}

/* ---- daily equity line + drawdown and position panes (vs 沪深300) ----
   One ledger-named replay result per chart (equity.result_equity_payload),
   compounded on the server; 沪深300 is served on the strategy's own days, so
   both lines start at 0 together. A marker is a dashed vertical line, e.g.
   the forward/Held-out boundary of the continuous replay. */
const RESULT_EQUITY_CACHE = new Map(); // `${experiment_id}/${result}` -> promise

function resultEquityHost(expId, result, opts = {}) {
  const key = `${expId}/${result}`;
  if (!RESULT_EQUITY_CACHE.has(key))
    RESULT_EQUITY_CACHE.set(
      key,
      api(
        `/api/experiments/${encodeURIComponent(expId)}/results/${encodeURIComponent(result)}/equity`,
      ),
    );
  const host = el("div", {}, el("div", { class: "hint" }, "收益曲线加载中…"));
  RESULT_EQUITY_CACHE.get(key)
    .then((payload) => host.replaceChildren(equityChart(payload, opts)))
    .catch((error) => {
      RESULT_EQUITY_CACHE.delete(key);
      host.replaceChildren(
        el("div", { class: "hint" }, `收益曲线加载失败：${error.message}`),
      );
    });
  return host;
}

function fmtDateTick(date, withYear) {
  return withYear
    ? `${date.slice(2, 4)}/${date.slice(4, 6)}-${date.slice(6, 8)}`
    : `${date.slice(4, 6)}-${date.slice(6, 8)}`;
}

/* A chart draws into a fixed viewBox that scales to its box, so a viewBox far
   wider than a phone shrinks its text below legibility; on a narrow viewport
   the chart is drawn about as wide as the screen instead. */
function fitChartWidth(width) {
  return Math.min(width, Math.max(360, window.innerWidth));
}

function equityChart(payload, opts = {}) {
  const { height = 240, mini = false, markers = [] } = opts;
  const width = fitChartWidth(opts.width || 680);
  let { ddH = 90 } = opts;
  const INK = themeInk();
  const colorOf = { strategy: INK.strategyColor, benchmark: INK.muted };
  const shown = (payload.series || []).filter((s) => (s.dates || []).length);
  if (!shown.length) return el("div", { class: "hint" }, "暂无日度收益数据");
  const bench = payload.benchmark;
  if (bench && (bench.dates || []).length) shown.push(bench);
  const seriesList = shown.map((s) => ({
    key: s.key,
    label: s.label,
    final: s.final,
    dates: s.dates,
    cum: new Map(s.dates.map((d, i) => [d, s.cum[i]])),
    dd: new Map(s.dates.map((d, i) => [d, s.drawdown[i]])),
    color: colorOf[s.key] || INK.strategyColor,
    dash: s.key === "benchmark" ? "6 4" : null,
  }));
  const dates = [...new Set(seriesList.flatMap((s) => s.dates))].sort();
  // Position-weight pane (EOD gross market value / equity), keyed like the
  // return series so identity carries across the linked panes.
  const exposureBy = payload.exposure || {};
  const expList = mini
    ? []
    : seriesList
        .filter(
          (s) =>
            s.key !== "benchmark" &&
            exposureBy[s.key] &&
            (exposureBy[s.key].dates || []).length,
        )
        .map((s) => {
          const e = exposureBy[s.key];
          return {
            key: s.key,
            color: s.color,
            long: new Map(e.dates.map((d, i) => [d, e.long[i]])),
          };
        });
  if (mini) ddH = 0;
  const showDD = ddH > 0;
  const showExp = expList.length > 0;
  const expH = showExp ? 64 : 0;
  // Account pane (a Paper book): end-of-day equity and cash from one journal
  // row per day, drawn on this chart's x-scale so a day's cash bar and equity
  // point sit directly under its return.
  const account = mini ? null : payload.account;
  const equityBy = new Map();
  const cashBy = new Map();
  (account ? account.dates : []).forEach((d, i) => {
    if (Number.isFinite(account.equity[i])) equityBy.set(d, account.equity[i]);
    if (Number.isFinite(account.cash[i])) cashBy.set(d, account.cash[i]);
  });
  const showAccount = equityBy.size > 0 || cashBy.size > 0;
  const accountH = showAccount ? 80 : 0;
  const hasPanes = showDD || showExp || showAccount;
  const padL = mini ? 44 : 52,
    padR = 12,
    padT = 8,
    gap = hasPanes ? 16 : 0;
  // With subplots the shared date labels sit BELOW them, so the main plot
  // needs only a slim bottom pad; standalone charts keep the label band.
  const padB = hasPanes ? 12 : mini ? 26 : 32;
  const labelBand = hasPanes ? 24 : 0;
  const panesBottom =
    height +
    (showDD ? gap + ddH : 0) +
    (showExp ? gap + expH : 0) +
    (showAccount ? gap + accountH : 0);
  const totalH = panesBottom + labelBand;
  const plotW = width - padL - padR,
    mainH = height - padT - padB;
  const xOf = (i) =>
    padL + (dates.length === 1 ? plotW / 2 : (i / (dates.length - 1)) * plotW);
  const cums = seriesList.flatMap((s) => [...s.cum.values()]);
  let lo = Math.min(0, ...cums),
    hi = Math.max(0, ...cums);
  const pad = Math.max((hi - lo) * 0.08, 0.002);
  lo -= pad;
  hi += pad;
  const yOf = (v) => padT + ((hi - v) / (hi - lo)) * mainH;
  const svg = [];
  // main gridlines: 4 evenly spaced levels + emphasized zero line
  for (let t = 0; t <= 4; t += 1) {
    const v = lo + ((hi - lo) * t) / 4;
    const y = yOf(v);
    svg.push(
      `<line x1="${padL}" y1="${y}" x2="${width - padR}" y2="${y}" stroke="${INK.grid}" stroke-width="1"/>`,
    );
    svg.push(
      `<text x="${padL - 6}" y="${y + 3.5}" text-anchor="end" font-size="${mini ? 10 : 11}" fill="${INK.muted}">${(v * 100).toFixed(1)}%</text>`,
    );
  }
  if (lo < 0 && hi > 0) {
    svg.push(
      `<line x1="${padL}" y1="${yOf(0)}" x2="${width - padR}" y2="${yOf(0)}" stroke="${INK.baseline}" stroke-width="1"/>`,
    );
  }
  // x ticks: at most 7 (4 when mini) and no more than fit at about 80 units a
  // label; year shown on the first tick and on year changes
  const tickSlots = Math.max(2, Math.min(mini ? 4 : 7, Math.floor(plotW / 80)));
  const tickEvery = Math.max(1, Math.ceil(dates.length / tickSlots));
  let prevYear = null;
  // Date labels: below the lowest subplot when present (shared axis at the
  // figure bottom), otherwise a clear step below the main axis line.
  const tickY = hasPanes ? panesBottom + 10 : padT + mainH + (mini ? 17 : 20);
  const lastTick = dates.length - 1;
  dates.forEach((d, i) => {
    // Render modulo ticks plus the final date; drop a modulo tick that would
    // overlap the end-anchored final label.
    if (
      i !== lastTick &&
      (i % tickEvery !== 0 || xOf(lastTick) - xOf(i) < 80)
    )
      return;
    const withYear = prevYear !== d.slice(0, 4);
    prevYear = d.slice(0, 4);
    // The final tick sits at the plot's right edge (padR is slim): end-anchor
    // it so the label stays inside the SVG instead of overflowing the border.
    const anchor = i === lastTick ? "end" : "middle";
    svg.push(
      `<text x="${xOf(i)}" y="${tickY}" text-anchor="${anchor}" font-size="${mini ? 10 : 11}" fill="${INK.muted}">${fmtDateTick(d, withYear)}</text>`,
    );
  });
  // markers: a dashed vertical line through every pane at the first day on or
  // after the marker date
  for (const marker of markers) {
    const index = dates.findIndex((d) => d >= String(marker.date));
    if (index < 0) continue;
    const x = xOf(index).toFixed(1);
    svg.push(
      `<line x1="${x}" y1="${padT}" x2="${x}" y2="${hasPanes ? panesBottom : padT + mainH}" stroke="${INK.baseline}" stroke-width="1" stroke-dasharray="3 3"/>`,
    );
    svg.push(
      `<text x="${Number(x) + 4}" y="${padT + 11}" font-size="11" fill="${INK.muted}">${escapeHtml(marker.label)}</text>`,
    );
  }
  // drawdown subplot
  if (showDD) {
    const ddTop = height + gap;
    const ddLo = Math.min(
      -0.001,
      ...seriesList.flatMap((s) => [...s.dd.values()]),
    );
    const ddY = (v) => ddTop + (v / ddLo) * (ddH - 14);
    svg.push(
      `<line x1="${padL}" y1="${ddY(0)}" x2="${width - padR}" y2="${ddY(0)}" stroke="${INK.baseline}" stroke-width="1"/>`,
    );
    svg.push(
      `<line x1="${padL}" y1="${ddY(ddLo)}" x2="${width - padR}" y2="${ddY(ddLo)}" stroke="${INK.grid}" stroke-width="1"/>`,
    );
    svg.push(
      `<text x="${padL - 6}" y="${ddY(ddLo) + 3.5}" text-anchor="end" font-size="10" fill="${INK.muted}">${(ddLo * 100).toFixed(1)}%</text>`,
    );
    svg.push(
      `<text x="${padL - 6}" y="${ddY(0) + 3.5}" text-anchor="end" font-size="10" fill="${INK.muted}">回撤</text>`,
    );
    for (const s of seriesList) {
      const pts = dates.filter((d) => s.dd.has(d));
      if (!pts.length) continue;
      const line = pts
        .map(
          (d, j) =>
            `${j ? "L" : "M"}${xOf(dates.indexOf(d)).toFixed(1)},${ddY(s.dd.get(d)).toFixed(1)}`,
        )
        .join(" ");
      if (s.key !== "benchmark") {
        const first = xOf(dates.indexOf(pts[0])).toFixed(1);
        const last = xOf(dates.indexOf(pts[pts.length - 1])).toFixed(1);
        svg.push(
          `<path d="M${first},${ddY(0).toFixed(1)} ${line.slice(1)} L${last},${ddY(0).toFixed(1)} Z" fill="${s.color}" fill-opacity="0.16" stroke="none"/>`,
        );
      }
      svg.push(
        `<path d="${line}" fill="none" stroke="${s.color}" stroke-width="1.5"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>`,
      );
    }
  }
  // position-weight subplot: 0..max(100%, observed) with 100% as reference line
  if (showExp) {
    const expTop = height + (showDD ? gap + ddH : 0) + gap;
    const expMax = Math.max(1, ...expList.flatMap((s) => [...s.long.values()]));
    const yExp = (v) => expTop + (1 - v / expMax) * (expH - 14);
    svg.push(
      `<line x1="${padL}" y1="${yExp(0)}" x2="${width - padR}" y2="${yExp(0)}" stroke="${INK.baseline}" stroke-width="1"/>`,
    );
    svg.push(
      `<line x1="${padL}" y1="${yExp(1)}" x2="${width - padR}" y2="${yExp(1)}" stroke="${INK.grid}" stroke-width="1"/>`,
    );
    svg.push(
      `<text x="${padL - 6}" y="${yExp(1) + 3.5}" text-anchor="end" font-size="10" fill="${INK.muted}">100%</text>`,
    );
    svg.push(
      `<text x="${padL - 6}" y="${yExp(0) + 3.5}" text-anchor="end" font-size="10" fill="${INK.muted}">仓位</text>`,
    );
    for (const s of expList) {
      const pts = dates.filter((d) => s.long.has(d));
      if (!pts.length) continue;
      const line = pts
        .map(
          (d, j) =>
            `${j ? "L" : "M"}${xOf(dates.indexOf(d)).toFixed(1)},${yExp(s.long.get(d)).toFixed(1)}`,
        )
        .join(" ");
      svg.push(
        `<path d="M${xOf(dates.indexOf(pts[0])).toFixed(1)},${yExp(0).toFixed(1)} ${line.slice(1)} L${xOf(dates.indexOf(pts[pts.length - 1])).toFixed(1)},${yExp(0).toFixed(1)} Z" fill="${s.color}" fill-opacity="0.16" stroke="none"/>`,
      );
      svg.push(
        `<path d="${line}" fill="none" stroke="${s.color}" stroke-width="1.5"/>`,
      );
    }
  }
  // One date column's width on the shared x-scale.
  const step = dates.length > 1 ? plotW / (dates.length - 1) : plotW;
  // account subplot: one ¥ scale from zero, cash as a bar per day and equity
  // as a line above it, so the gap between them is the invested value
  if (showAccount) {
    const top = panesBottom - accountH;
    const amountMax = niceCeil(Math.max(1, ...equityBy.values(), ...cashBy.values()));
    // a 12-unit band above the ceiling carries the pane's own label
    const yAmount = (v) => top + 12 + (1 - Math.max(v, 0) / amountMax) * (accountH - 26);
    svg.push(
      `<line x1="${padL}" y1="${yAmount(0)}" x2="${width - padR}" y2="${yAmount(0)}" stroke="${INK.baseline}" stroke-width="1"/>`,
    );
    svg.push(
      `<line x1="${padL}" y1="${yAmount(amountMax)}" x2="${width - padR}" y2="${yAmount(amountMax)}" stroke="${INK.grid}" stroke-width="1"/>`,
    );
    svg.push(
      `<text x="${padL - 6}" y="${yAmount(amountMax) + 3.5}" text-anchor="end" font-size="10" fill="${INK.muted}">${escapeHtml(fmtAmount(amountMax))}</text>`,
    );
    svg.push(
      `<text x="${padL - 6}" y="${yAmount(0) + 3.5}" text-anchor="end" font-size="10" fill="${INK.muted}">资金</text>`,
    );
    svg.push(
      `<text x="${padL + 4}" y="${top + 9}" font-size="10" fill="${INK.muted}">权益（线）· 现金（柱）</text>`,
    );
    const barW = Math.max(2, Math.min(16, step * 0.6));
    dates.forEach((d, i) => {
      if (!cashBy.has(d)) return;
      svg.push(
        `<path d="${barPath(xOf(i) - barW / 2, yAmount(0), yAmount(cashBy.get(d)), barW)}" fill="${INK.strategyColor}" fill-opacity="0.35"/>`,
      );
    });
    const pts = dates.filter((d) => equityBy.has(d));
    if (pts.length) {
      svg.push(
        `<path d="${pts.map((d, j) => `${j ? "L" : "M"}${xOf(dates.indexOf(d)).toFixed(1)},${yAmount(equityBy.get(d)).toFixed(1)}`).join(" ")}" fill="none" stroke="${INK.strategyColor}" stroke-width="1.5"/>`,
      );
      const last = pts[pts.length - 1];
      svg.push(
        `<circle cx="${xOf(dates.indexOf(last)).toFixed(1)}" cy="${yAmount(equityBy.get(last)).toFixed(1)}" r="3" fill="${INK.strategyColor}" stroke="${INK.ring}" stroke-width="1.5"/>`,
      );
    }
  }
  // main lines (benchmark first so the strategy line sits on top) + endpoint dot
  for (const s of [...seriesList].reverse()) {
    const pts = dates.filter((d) => s.cum.has(d));
    if (!pts.length) continue;
    const line = pts
      .map(
        (d, j) =>
          `${j ? "L" : "M"}${xOf(dates.indexOf(d)).toFixed(1)},${yOf(s.cum.get(d)).toFixed(1)}`,
      )
      .join(" ");
    svg.push(
      `<path d="${line}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>`,
    );
    if (s.key !== "benchmark") {
      const lastDate = pts[pts.length - 1];
      svg.push(
        `<circle cx="${xOf(dates.indexOf(lastDate)).toFixed(1)}" cy="${yOf(s.cum.get(lastDate)).toFixed(1)}" r="3.5" fill="${s.color}" stroke="${INK.ring}" stroke-width="2"/>`,
      );
    }
  }
  // hover columns: one hit target per date spanning every pane, rich tooltip
  dates.forEach((d, i) => {
    const lines = [fmtDate(d)];
    for (const s of seriesList) {
      if (!s.cum.has(d)) continue;
      const exposure = expList.find((entry) => entry.key === s.key);
      const expText =
        exposure && exposure.long.has(d)
          ? ` ｜ 仓位 ${(exposure.long.get(d) * 100).toFixed(1)}%`
          : "";
      lines.push(
        `${s.label} 累计 ${(s.cum.get(d) * 100).toFixed(2)}% ｜ 回撤 ${(s.dd.get(d) * 100).toFixed(2)}%${expText}`,
      );
    }
    if (equityBy.has(d) || cashBy.has(d))
      lines.push(
        `权益 ${equityBy.has(d) ? fmtAmount(equityBy.get(d)) : "—"} ｜ 现金 ${cashBy.has(d) ? fmtAmount(cashBy.get(d)) : "—"}`,
      );
    const x = i === 0 ? padL : xOf(i) - step / 2;
    const w =
      dates.length === 1
        ? plotW
        : i === 0 || i === dates.length - 1
          ? step / 2
          : step;
    svg.push(
      `<rect class="xcol" x="${x.toFixed(1)}" y="${padT}" width="${Math.max(w, 1).toFixed(1)}" height="${totalH - padT - 4}" data-tip="${escapeHtml(lines.join("\n"))}"/>`,
    );
  });
  const wrap = el(
    "div",
    { class: "svg-chart" },
    chartLegend(
      seriesList.map((s) => ({
        color: s.color,
        label: `${s.label} ${fmtPct(s.final)}`,
      })),
    ),
  );
  const svgHost = el("div", {});
  svgHost.innerHTML = `<svg viewBox="0 0 ${width} ${totalH}" xmlns="http://www.w3.org/2000/svg">${svg.join("")}</svg>`;
  wrap.append(svgHost);
  wrap.__rerender = () => equityChart(payload, opts);
  return bindChartTips(wrap);
}

function niceCeil(value) {
  const mag = 10 ** Math.floor(Math.log10(value));
  for (const mult of [1, 2, 2.5, 5, 10]) {
    if (mult * mag >= value) return mult * mag;
  }
  return 10 * mag;
}

/* Rounded data-end bar: square at the baseline, 4px radius at the value end. */
function barPath(x, zeroY, valueY, w) {
  const up = valueY < zeroY;
  const h = Math.max(Math.abs(zeroY - valueY), 1);
  const r = Math.min(4, w / 2, h);
  if (up) {
    const y = zeroY - h;
    return `M${x},${zeroY} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${zeroY} Z`;
  }
  const y = zeroY + h;
  return `M${x},${zeroY} L${x + w},${zeroY} L${x + w},${y - r} Q${x + w},${y} ${x + w - r},${y} L${x + r},${y} Q${x},${y} ${x},${y - r} Z`;
}

/* Two decimals with thousands separators: the cent resolution the Paper orders
   sheet prints (paper/orders.py `_money`, `_price`). */
const CENTS_FMT = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

/* A money amount: 万 and 亿 abbreviate the large ones, anything below ¥10,000
   keeps its cents. Never used for a per-share price, which is fmtPrice. */
function fmtAmount(value) {
  const n = Number(value) || 0;
  if (Math.abs(n) >= 1e8) return `¥${(n / 1e8).toFixed(2)}亿`;
  if (Math.abs(n) >= 1e4) return `¥${(n / 1e4).toFixed(1)}万`;
  return `¥${CENTS_FMT.format(n)}`;
}

/* A per-share price, whatever its size; "—" when the value is missing. */
function fmtPrice(value) {
  const number = Number(value);
  return value === null || value === undefined || !Number.isFinite(number)
    ? "—"
    : CENTS_FMT.format(number);
}

/* Single-series bar chart (no legend needed for one series); direct value
   labels when the set is small, tooltips always. */
function singleSeriesBarChart(
  rows,
  { width: requestedWidth = 640, height = 200, fmt = fmtPct } = {},
) {
  const width = fitChartWidth(requestedWidth);
  const INK = themeInk();
  const color = INK.strategyColor; // categorical slot 1
  const values = rows
    .map((row) => row.value)
    .filter((v) => v !== null && v !== undefined);
  if (!rows.length || !values.length)
    return el("div", { class: "hint" }, "暂无数据");
  const signed = values.some((v) => v < 0);
  const maxAbs = niceCeil(Math.max(1e-9, ...values.map(Math.abs)));
  const gridFracs = signed ? [-1, -0.5, 0.5, 1] : [0.5, 1];
  // The left pad fits the widest axis label (11px text, about 6.5px a glyph).
  const padL = Math.max(
      56,
      12 + 6.5 * Math.max(...gridFracs.map((frac) => fmt(frac * maxAbs).length)),
    ),
    padR = 10,
    padB = 30,
    padT = signed ? 8 : 18;
  const plotW = width - padL - padR,
    plotH = height - padT - padB;
  const zeroY = signed ? padT + plotH / 2 : padT + plotH;
  const scale = signed ? plotH / 2 : plotH;
  const yOf = (v) => zeroY - (v / maxAbs) * scale;
  const svg = [];
  for (const frac of gridFracs) {
    const y = yOf(frac * maxAbs);
    svg.push(
      `<line x1="${padL}" y1="${y}" x2="${width - padR}" y2="${y}" stroke="${INK.grid}" stroke-width="1"/>`,
    );
    svg.push(
      `<text x="${padL - 6}" y="${y + 3.5}" text-anchor="end" font-size="11" fill="${INK.muted}">${escapeHtml(fmt(frac * maxAbs))}</text>`,
    );
  }
  svg.push(
    `<line x1="${padL}" y1="${zeroY}" x2="${width - padR}" y2="${zeroY}" stroke="${INK.baseline}" stroke-width="1"/>`,
  );
  const groupW = plotW / rows.length;
  const barW = Math.max(4, Math.min(24, groupW - 6));
  const showTipLabels = rows.length <= 8;
  // No more x labels than fit side by side (11px text, about 6.5px a glyph).
  const labelChars = Math.max(...rows.map((row) => String(row.label).length));
  const labelSlots = Math.max(1, Math.floor(plotW / (6.5 * labelChars + 12)));
  const labelEvery = Math.max(
    1,
    Math.ceil(rows.length / Math.min(12, labelSlots)),
  );
  rows.forEach((row, index) => {
    const cx = padL + groupW * index + groupW / 2;
    const value = row.value;
    if (value === null || value === undefined) return;
    const tip = `${row.label} ${fmt(value)}`;
    svg.push(
      `<path d="${barPath(cx - barW / 2, zeroY, yOf(value), barW)}" fill="${color}" data-tip="${escapeHtml(tip)}"/>`,
    );
    if (showTipLabels) {
      const labelY = value >= 0 ? yOf(value) - 5 : yOf(value) + 13;
      svg.push(
        `<text x="${cx}" y="${labelY}" text-anchor="middle" font-size="11" fill="${INK.muted}">${escapeHtml(fmt(value))}</text>`,
      );
    }
    if (index % labelEvery === 0) {
      svg.push(
        `<text x="${cx}" y="${height - 8}" text-anchor="middle" font-size="11" fill="${INK.muted}">${escapeHtml(String(row.label))}</text>`,
      );
    }
  });
  const wrap = el("div", { class: "svg-chart" });
  const svgHost = el("div", {});
  svgHost.innerHTML = `<svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg">${svg.join("")}</svg>`;
  wrap.append(svgHost);
  wrap.__rerender = () =>
    singleSeriesBarChart(rows, { width: requestedWidth, height, fmt });
  return bindChartTips(wrap);
}

/* Stat tiles: label + semibold value (proportional figures). */
function statTilesRow(tiles) {
  return el(
    "div",
    { class: "tiles" },
    ...tiles.map((tile) =>
      el(
        "div",
        { class: "tile", title: tile.title || null },
        el("div", { class: "tile-label" }, tile.label),
        el("div", { class: `tile-value ${tile.cls || ""}` }, tile.value),
      ),
    ),
  );
}

/* A panel's title row: the title, then its badges, notes and actions. */
function panelHead(title, ...extras) {
  return el("div", { class: "panel-head" }, el("h4", {}, title), ...extras);
}

/* Every data table. A column marked `num` aligns its header and cells right in
   tabular digits and every other column reads left, so alignment is decided
   once per column. The box scrolls sideways instead of the page; `box` adds
   classes to it ("limit" caps a long list's height). A cell is a value, a node
   or an array of them, or {value, cls, title} to add a class or a tooltip. */
function dataTable(columns, rows, { fit = false, box = "" } = {}) {
  const align = (column) => (column.num ? "num" : "");
  return el(
    "div",
    { class: `table-box ${box}`.trim() },
    el(
      "table",
      { class: fit ? "data fit" : "data" },
      el(
        "tr",
        {},
        ...columns.map((column) =>
          el(
            "th",
            { class: align(column), title: column.title || null },
            column.label,
          ),
        ),
      ),
      ...rows.map((cells) =>
        el(
          "tr",
          {},
          ...cells.map((cell, index) => {
            const spec =
              cell !== null &&
              typeof cell === "object" &&
              !cell.nodeType &&
              !Array.isArray(cell)
                ? cell
                : { value: cell };
            return el(
              "td",
              {
                class: `${align(columns[index])} ${spec.cls || ""}`.trim(),
                title: spec.title || null,
              },
              spec.value ?? "—",
            );
          }),
        ),
      ),
    ),
  );
}

/* ---------------- router ---------------- */

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", route);

function setActiveNav(tab) {
  document.querySelectorAll("#topnav a").forEach((node) => {
    node.classList.toggle("active", node.dataset.nav === tab);
  });
}

function route(forceRefresh = false) {
  const force = forceRefresh === true; // hashchange passes an Event, not a force flag.
  const hash = location.hash || "#/";
  // Trading console pages are matched BEFORE the #/exp/... regex.
  const tradingMatch = hash.match(/^#\/trading\/(paper)(?:\/([^/]+))?$/);
  const qmtMatch = hash === "#/qmt";
  const memoryMatch = hash === "#/memory";
  const expMatch =
    tradingMatch || qmtMatch || memoryMatch
      ? null
      : hash.match(/^#\/exp\/([^/]+)(?:\/(.*))?$/);
  const expId = expMatch ? decodeURIComponent(expMatch[1]) : null;
  const key = expMatch && expMatch[2] ? sessionKeyFromUrl(expMatch[2]) : null;
  // Session switch within an already-rendered experiment swaps only the right
  // panel: no page rebuild, no scroll jump, live stream and timers untouched.
  if (
    !force &&
    expMatch &&
    key &&
    detailView &&
    detailView.experimentId === expId &&
    document.body.contains(detailView.listHost)
  ) {
    selectSession(key);
    return;
  }
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  for (const timer of liveTimers) clearInterval(timer);
  liveTimers = [];
  for (const source of liveSources) source.close();
  liveSources = [];
  document.querySelectorAll(".modal-mask").forEach((node) => node.remove());
  setActiveNav(
    qmtMatch
      ? "qmt"
      : memoryMatch
        ? "memory"
        : tradingMatch
          ? tradingMatch[1]
          : "research",
  );
  if (!tradingMatch) tradingView = null;
  if (tradingMatch) {
    detailView = null;
    renderTradingPage(
      tradingMatch[1],
      tradingMatch[2] ? decodeURIComponent(tradingMatch[2]) : null,
    );
  } else if (qmtMatch) {
    detailView = null;
    renderQmtPage();
  } else if (memoryMatch) {
    detailView = null;
    renderMemoryPage();
  } else if (expMatch) renderDetailPage(expId, key);
  else {
    detailView = null;
    renderHomePage();
  }
}

function selectSession(key) {
  // In-experiment switches bypass route(): stop the previous session's live
  // timers (GPU refresh, analysis re-polls) or they accumulate per visit.
  for (const timer of liveTimers) clearInterval(timer);
  liveTimers = [];
  for (const source of liveSources) source.close();
  liveSources = [];
  detailView.selectedKey = key;
  const fresh = sessionDetailPanel(detailView.detail, key);
  detailView.rightHost.replaceWith(fresh);
  detailView.rightHost = fresh;
  detailView.listHost.querySelectorAll(".session-item").forEach((node) => {
    node.classList.toggle("selected", node.dataset.key === key);
  });
}

/* ---------------- home page ---------------- */

async function renderHomePage() {
  $main.innerHTML = '<div class="loading">加载中…</div>';
  $topbarRight.innerHTML = "";
  let payload;
  try {
    payload = await api("/api/experiments");
  } catch (error) {
    $main.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
    return;
  }
  $topbarRight.append(
    el(
      "span",
      { class: "mode-note" },
      `并行运行 ${payload.running.length}/${payload.max_running_experiments}`,
    ),
    el(
      "button",
      { class: "btn primary", onclick: openCreateModal },
      "＋ 新建实验",
    ),
  );
  $main.replaceChildren(homeView(payload));
  pollTimer = setInterval(async () => {
    if (location.hash && location.hash !== "#/" && location.hash !== "#")
      return;
    try {
      await refreshHomePage();
    } catch {
      /* keep last view */
    }
  }, 5000);
}

function bestRow(payload) {
  const best = payload.best;
  const item =
    best &&
    (payload.experiments || []).find(
      (row) => row.experiment_id === best.experiment_id,
    );
  return item ? { item, basis: best.basis } : null;
}

function homeView(payload) {
  const container = el("div", { id: "home" });
  const best = bestRow(payload);
  if (best)
    container.append(
      heroPanel(best.item, best.basis),
      el("div", { class: "section-gap" }),
    );
  const rows = payload.experiments || [];
  container.append(
    el("div", { class: "page-head" }, el("h2", {}, "实验列表")),
    rows.length
      ? experimentGrid(rows)
      : el("div", { class: "empty" }, "还没有实验 —— 点右上角「新建实验」开始。"),
  );
  return container;
}

/* The grid is rebuilt on every poll; the hero only when what it shows
   changed, so its curve is not re-fetched and redrawn every five seconds. */
async function refreshHomePage() {
  const payload = await api("/api/experiments");
  const home = document.getElementById("home");
  const grid = home && home.querySelector(".grid");
  const hero = document.getElementById("hero-panel");
  const best = bestRow(payload);
  const signature = best ? heroSignature(best.item, best.basis) : "";
  if (!home || !grid || (hero ? hero.__signature : "") !== signature) {
    if (home) home.replaceWith(homeView(payload));
    return;
  }
  grid.replaceWith(experimentGrid(payload.experiments || []));
}

function heroSignature(item, basis) {
  return [
    item.experiment_id,
    basis,
    item.state,
    item.stage,
    item.research_recorded,
    (item.verdict || {}).status,
    (item.forward || {}).result,
  ].join("|");
}

function experimentGrid(rows) {
  return el("div", { class: "grid" }, ...rows.map(experimentCard));
}

function verdictBadge(verdict) {
  if (!verdict || !verdict.status) return null;
  const graduated = verdict.status === "graduated";
  return el(
    "span",
    {
      class: `badge state-${graduated ? "completed" : "failed"}`,
      title: (verdict.reasons || []).map(reasonLabel).join("；") || null,
    },
    VERDICT_LABELS[verdict.status] || verdict.status,
  );
}

/* A long experiment id must not reflow the heading: the name takes one
   elastic column and truncates, the badges keep their own column, so every
   card's badges line up on the same edge. The full id stays in the tooltip. */
function experimentName(experimentId, { link = true } = {}) {
  const attrs = { class: "exp-name", title: experimentId };
  return link
    ? el("a", { ...attrs, href: `#/exp/${encodeURIComponent(experimentId)}` }, experimentId)
    : el("span", attrs, experimentId);
}

function experimentBadges(...badges) {
  return el("span", { class: "exp-badges" }, ...badges.filter(Boolean));
}

/* Where the arm is, from the ledger: research sessions recorded, the sealed
   forward replay, or the verdict. Never a research-period number. */
function stageText(item) {
  const research = `研究 ${item.research_recorded ?? 0}/${item.research_total ?? "?"}`;
  if (item.stage === "verdict")
    return (item.verdict || {}).status === "no_deliverable"
      ? `${research} · 未冻结`
      : `${research} · 已判定`;
  if (item.stage === "forward") return `${research} · 已冻结 · 前推回放封存中`;
  return item.worker_alive && item.current_session
    ? `${research} · 当前 ${sessionLabel(item.current_session)}`
    : research;
}

/* The forward and Held-out slices the verdict read; absent until it exists,
   and absent for a replay the strategy's own error stopped. */
function forwardTiles(item) {
  const slices = (item.forward || {}).slices || {};
  const forward = slices.forward;
  const heldout = slices.heldout;
  if (!forward && !heldout) return null;
  const f = forward || {};
  const h = heldout || {};
  return statTilesRow([
    {
      label: "前推超额 80% 下界",
      value: fmtPct(f.lower_bound),
      cls: signCls(f.lower_bound),
    },
    {
      label: "前推中性化超额",
      value: fmtPct(f.neutralized_excess),
      cls: signCls(f.neutralized_excess),
    },
    {
      label: "Held-out 中性化超额",
      value: fmtPct(h.neutralized_excess),
      cls: signCls(h.neutralized_excess),
    },
    { label: "前推回撤", value: fmtPct(f.max_drawdown) },
  ]);
}

function forwardMarkers(forward) {
  const start = ((forward || {}).replay || {}).heldout_start;
  return start ? [{ date: start, label: "Held-out" }] : [];
}

function experimentCard(item) {
  const card = el("div", {
    class: "card clickable",
    onclick: () => {
      location.hash = `#/exp/${encodeURIComponent(item.experiment_id)}`;
    },
  });
  card.append(
    el(
      "h3",
      {},
      experimentName(item.experiment_id),
      experimentBadges(stateBadge(item.state), verdictBadge(item.verdict)),
    ),
    el(
      "div",
      { class: "meta-line" },
      `创建 ${fmtTs(item.created_at)}`,
      item.error ? ` ｜ ${item.error}` : "",
    ),
  );
  if (item.state !== "unreadable")
    card.append(el("div", { class: "meta-line" }, stageText(item)));
  if (item.worker_alive && item.environment_stage) {
    card.append(
      el(
        "div",
        { class: "meta-line" },
        formatStageLine({
          environment_stage: item.environment_stage,
          environment_stage_started_at: item.environment_stage_started_at,
          environment_progress: item.environment_progress,
          session_started_at: item.session_started_at,
        }),
      ),
    );
  }
  const tiles = forwardTiles(item);
  if (tiles) card.append(tiles);
  const actions = el("div", { class: "actions" });
  if (item.kind === "hitl" && RESUMABLE_STATES.includes(item.state)) {
    actions.append(
      el(
        "button",
        {
          class: "btn small primary",
          onclick: async (event) => {
            event.stopPropagation();
            try {
              await api(
                `/api/experiments/${encodeURIComponent(item.experiment_id)}/control`,
                { method: "POST", body: JSON.stringify({ action: "resume" }) },
              );
              toast("已请求恢复运行");
              refreshHomePage();
            } catch (error) {
              toast(`恢复失败：${error.message}`, true);
            }
          },
        },
        "恢复运行",
      ),
    );
  }
  if (!item.worker_alive) {
    actions.append(
      el(
        "button",
        {
          class: "btn small danger",
          onclick: (event) => {
            event.stopPropagation();
            confirmDeleteExperiment(item.experiment_id);
          },
        },
        "删除",
      ),
    );
  }
  if (actions.children.length) card.append(actions);
  return card;
}

const BEST_BASIS_LABELS = {
  forward_lower_bound: "按前推超额 80% 下界",
  stage: "按运行阶段",
};

function heroPanel(item, basis) {
  const panel = el("div", { class: "panel hero", id: "hero-panel" });
  panel.__signature = heroSignature(item, basis);
  panel.append(
    el(
      "div",
      { class: "panel-head" },
      el(
        "h3",
        { class: "hero-title" },
        el("span", { "aria-hidden": "true" }, "🏆"),
        experimentName(item.experiment_id),
      ),
      stateBadge(item.state),
      verdictBadge(item.verdict),
      el(
        "span",
        { class: "mode-note" },
        `最佳实验 · ${BEST_BASIS_LABELS[basis] || basis}`,
      ),
    ),
  );
  panel.append(
    forwardTiles(item) || el("div", { class: "meta-line" }, stageText(item)),
  );
  const result = (item.forward || {}).result;
  if (result)
    panel.append(
      el(
        "div",
        { class: "section-gap" },
        el("h4", { class: "subsection-title" }, "前推与 Held-out 连续回放 vs 沪深300"),
        resultEquityHost(item.experiment_id, result, {
          width: 980,
          height: 240,
          ddH: 90,
          markers: forwardMarkers(item.forward),
        }),
      ),
    );
  return panel;
}

function confirmDeleteExperiment(experimentId) {
  const input = el("input", { type: "text", placeholder: experimentId });
  showModal(
    "删除实验",
    el(
      "div",
      {},
      el(
        "p",
        {},
        `此操作会永久删除 experiments/${experimentId}/ 目录（含账本、冻结策略与全部运行产物），不可恢复。`,
      ),
      el("p", {}, "输入实验名以确认："),
      el("div", { class: "field" }, input),
    ),
    [
      el("button", { class: "btn", onclick: closeModal }, "取消"),
      el(
        "button",
        {
          class: "btn danger",
          onclick: async () => {
            if (input.value !== experimentId) {
              toast("实验名不匹配", true);
              return;
            }
            try {
              await api(
                `/api/experiments/${encodeURIComponent(experimentId)}?confirm=${encodeURIComponent(experimentId)}`,
                { method: "DELETE" },
              );
              toast("已删除");
              closeModal();
              if (location.hash === "#/") renderHomePage();
              else location.hash = "#/";
            } catch (error) {
              toast(`删除失败：${error.message}`, true);
            }
          },
        },
        "确认删除",
      ),
    ],
  );
}

/* ---------------- create modal ---------------- */

async function openCreateModal() {
  let schema;
  try {
    schema = await api("/api/parameter-schema");
  } catch (error) {
    toast(error.message, true);
    return;
  }
  const inputs = new Map();
  const body = el("div", {});
  // Validation errors surface at the TOP of the (scrollable) modal body.
  const errorBox = el("div", {});
  body.append(errorBox);
  for (const group of schema.groups) {
    const basic = group.fields.filter((field) => !field.advanced);
    const advanced = group.fields.filter((field) => field.advanced);
    if (!basic.length && !advanced.length) continue;
    const section = el(
      "div",
      { class: "form-group" },
      el("h4", {}, group.name),
    );
    if (basic.length) section.append(fieldGrid(basic, inputs));
    if (advanced.length) {
      section.append(
        el(
          "details",
          { class: "advanced" },
          el("summary", {}, `高级参数（${advanced.length}）`),
          fieldGrid(advanced, inputs),
        ),
      );
    }
    body.append(section);
  }
  showModal("新建实验", body, [
    el("button", { class: "btn", onclick: closeModal }, "取消"),
    el(
      "button",
      {
        class: "btn primary",
        onclick: async (event) => {
          const params = collectParams(inputs);
          event.target.disabled = true;
          try {
            const created = await api("/api/experiments", {
              method: "POST",
              body: JSON.stringify({ params }),
            });
            toast(`实验 ${created.experiment_id} 已创建并启动`);
            closeModal();
            location.hash = `#/exp/${encodeURIComponent(created.experiment_id)}`;
          } catch (error) {
            errorBox.innerHTML = "";
            errorBox.append(
              el("div", { class: "form-error" }, `创建失败：${error.message}`),
            );
            const scroller = errorBox.closest(".body");
            if (scroller) scroller.scrollTop = 0;
          } finally {
            event.target.disabled = false;
          }
        },
      },
      "创建并启动",
    ),
  ]);
}

function fieldGrid(fields, inputs) {
  const grid = el("div", { class: "form-grid" });
  for (const field of fields) grid.append(fieldNode(field, inputs));
  return grid;
}

function fieldNode(field, inputs) {
  const wrap = el("div", { class: "field" });
  if (field.key === "gpu_count") wrap.classList.add("field-wide", "gpu-field");
  if (field.wide === true) wrap.classList.add("field-wide");
  // multi defaults to a full row (long chip lists); "wide": false opts a short
  // chip group into a normal grid cell so it can share a row (e.g. 板块范围).
  if (field.type === "multi" && field.wide !== false)
    wrap.classList.add("field-wide");
  const labelText = field.required ? `${field.label} *` : field.label;
  if (field.type === "bool") {
    const input = el("input", { type: "checkbox" });
    input.checked = Boolean(field.default);
    inputs.set(field.key, { field, input });
    wrap.className = "field checkbox";
    wrap.append(
      input,
      el(
        "div",
        {},
        el("label", {}, labelText),
        el("div", { class: "help" }, field.help || ""),
      ),
    );
    return wrap;
  }
  wrap.append(el("label", {}, labelText));
  let input;
  if (field.type === "choice") {
    input = el(
      "select",
      {},
      ...field.choices.map((choice) => {
        const option = el(
          "option",
          { value: choice },
          (field.choice_labels || {})[choice] || choice,
        );
        if (choice === field.default) option.selected = true;
        return option;
      }),
    );
  } else if (field.type === "period") {
    // Options are cadence-dependent; repopulatePeriodSelects fills them.
    input = el("select", { class: "period-select" });
  } else if (field.type === "multi") {
    // Checkbox group: multi-selects require ctrl-click and mis-toggle easily.
    const boxes = field.choices.map((choice) => {
      const box = el("input", { type: "checkbox", value: choice });
      box.checked = (field.default || []).includes(choice);
      return box;
    });
    // Chip text prefers the Chinese display label; the raw API name stays on
    // the tooltip for cross-referencing docs/data contracts.
    const groupNode = el(
      "div",
      { class: "check-group" },
      ...boxes.map((box, index) => {
        const choice = field.choices[index];
        const label = (field.choice_labels || {})[choice];
        return el(
          "label",
          { class: "check-item", title: label ? choice : "" },
          box,
          label || choice,
        );
      }),
    );
    inputs.set(field.key, {
      field,
      getValue: () =>
        boxes.filter((box) => box.checked).map((box) => box.value),
    });
    wrap.append(groupNode, el("div", { class: "help" }, field.help || ""));
    return wrap;
  } else if (field.type === "text") {
    input = el("textarea", { rows: "3" });
    input.value = field.default ?? "";
  } else {
    input = el("input", {
      type:
        field.type === "int" || field.type === "float"
          ? "number"
          : field.type === "time"
            ? "time"
            : "text",
    });
    if (field.type === "float") input.setAttribute("step", "any");
    if (field.min !== undefined) input.setAttribute("min", String(field.min));
    if (field.max !== undefined) input.setAttribute("max", String(field.max));
    input.value = field.default ?? "";
    if (field.optional) input.placeholder = "留空使用默认";
    if (field.type === "int" || field.type === "float") {
      // Focused number inputs change value on mouse wheel (browser default) —
      // an easy silent mis-edit while scrolling the form. Block the spin but
      // keep page scrolling (unfocused inputs ignore wheel anyway).
      input.addEventListener(
        "wheel",
        (event) => {
          if (document.activeElement === input) event.preventDefault();
        },
        { passive: false },
      );
    }
    if (field.type === "int") {
      // Native WebKit spinners are hidden (unstylable); draw our own steppers.
      const step = (direction) => {
        if (direction > 0) input.stepUp();
        else input.stepDown();
        input.dispatchEvent(new Event("change", { bubbles: true }));
      };
      const host = el(
        "div",
        { class: "number-input" },
        input,
        el(
          "div",
          { class: "spin-col" },
          el(
            "button",
            {
              type: "button",
              class: "spin",
              tabindex: "-1",
              onclick: () => step(1),
            },
            "▲",
          ),
          el(
            "button",
            {
              type: "button",
              class: "spin",
              tabindex: "-1",
              onclick: () => step(-1),
            },
            "▼",
          ),
        ),
      );
      inputs.set(field.key, { field, input });
      wrap.append(host, el("div", { class: "help" }, field.help || ""));
      if (field.key === "gpu_count") {
        const gpuStatus = el(
          "div",
          { class: "gpu-status" },
          el("span", { class: "help" }, "正在读取当前 GPU 状态…"),
        );
        wrap.append(gpuStatus);
        api("/api/gpus")
          .then((payload) => {
            const gpus = payload.gpus || [];
            if (!gpus.length) {
              gpuStatus.replaceChildren(
                el(
                  "span",
                  { class: "help" },
                  `当前无可用 GPU 信息${payload.error ? `：${payload.error}` : ""}`,
                ),
              );
              return;
            }
            gpuStatus.replaceChildren(
              ...gpus.map((gpu) =>
                el(
                  "div",
                  { class: "gpu-status-item" },
                  el("strong", {}, `GPU ${gpu.index}`),
                  el(
                    "span",
                    {},
                    `空闲 ${(gpu.memory_free_mib / 1024).toFixed(1)} / ${(gpu.memory_total_mib / 1024).toFixed(1)} GiB`,
                  ),
                ),
              ),
            );
            input.max = String(Math.min(Number(field.max || 4), gpus.length));
          })
          .catch((error) => {
            gpuStatus.replaceChildren(
              el(
                "span",
                { class: "help" },
                `GPU 状态读取失败：${error.message}`,
              ),
            );
          });
      }
      return wrap;
    }
  }
  inputs.set(field.key, { field, input });
  wrap.append(input, el("div", { class: "help" }, field.help || ""));
  return wrap;
}

function collectParams(inputs) {
  const params = {};
  for (const [key, entry] of inputs.entries()) {
    const { field, input } = entry;
    let value;
    if (entry.getValue) value = entry.getValue();
    else if (field.type === "bool") value = input.checked;
    else value = input.value;
    if (field.type === "int" || field.type === "float") {
      if (value === "" || value === null) {
        if (field.optional) continue;
        value = field.default;
      } else
        value = field.type === "int" ? parseInt(value, 10) : parseFloat(value);
      if (Number.isNaN(value)) continue;
    }
    if (typeof value === "string") value = value.trim();
    if (value === "" && !field.required) {
      if (field.default === null || field.default === undefined) continue;
      value = field.default;
    }
    if (
      JSON.stringify(value) === JSON.stringify(field.default) &&
      !field.required
    )
      continue;
    params[key] = value;
  }
  return params;
}

/* ---------------- modal helpers ---------------- */

function showModal(title, body, footerButtons, modalClass = "") {
  closeModal();
  const mask = el("div", {
    class: "modal-mask",
    onclick: (event) => {
      if (event.target === mask) closeModal();
    },
  });
  mask.append(
    el(
      "div",
      { class: `modal${modalClass ? ` ${modalClass}` : ""}` },
      el(
        "header",
        {},
        el("h3", {}, title),
        el("button", { class: "btn small", onclick: closeModal }, "✕"),
      ),
      el("div", { class: "body" }, body),
      el("footer", {}, ...footerButtons),
    ),
  );
  $modalRoot.append(mask);
}

function closeModal() {
  $modalRoot.innerHTML = "";
}

/* ---------------- detail page ---------------- */

let detailView = null; // {experimentId, detail, listHost, rightHost, barHost, selectedKey}

function isSessionDone(detail, session) {
  return session.kind === "forward"
    ? Boolean(detail.forward)
    : Boolean(session.record);
}

async function renderDetailPage(experimentId, selectedKey) {
  $main.innerHTML = '<div class="loading">加载中…</div>';
  $topbarRight.innerHTML = "";
  let detail;
  try {
    detail = await api(`/api/experiments/${encodeURIComponent(experimentId)}`);
  } catch (error) {
    $main.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
    return;
  }
  const status = detail.status || {};
  const sessions = detail.sessions || [];
  // Default to where the arm is: the running session, the next research
  // session, the sealed replay, or the last session that ran.
  if (!selectedKey) {
    const recorded = sessions.filter((session) => isSessionDone(detail, session));
    selectedKey =
      status.session_key ||
      (detail.stage === "research"
        ? (sessions.find((session) => !isSessionDone(detail, session)) || {}).key
        : detail.stage === "forward"
          ? "forward"
          : (recorded[recorded.length - 1] || sessions[0] || {}).key);
  }
  const head = el(
    "div",
    { class: "page-head" },
    el(
      "h2",
      {},
      el("a", { class: "exp-back", href: "#/" }, "← 实验"),
      experimentName(detail.experiment_id, { link: false }),
      experimentBadges(stateBadge(detail.state), verdictBadge(detail.verdict)),
    ),
  );
  // Progress and the current stage ride on the control row; the head keeps
  // only errors.
  const errors = [
    detail.state === "unreadable" && detail.error ? detail.error : null,
    status.error ? `错误：${status.error}` : null,
  ].filter(Boolean);
  if (detail.params && Object.keys(detail.params).length)
    head.append(
      el(
        "button",
        { class: "btn small", onclick: () => openParamsModal(detail) },
        "创建参数",
      ),
    );
  if (errors.length) head.append(el("div", { class: "sub" }, errors.join(" ｜ ")));
  const container = el("div", {}, head);
  let barHost = null;
  if (detail.kind === "hitl") {
    // The control row and the stage strip are one panel: where the arm is, in
    // words and buttons above, stage by stage below.
    barHost = controlBar(detail);
    container.append(
      el("div", { class: "panel section-gap" }, barHost, stageStrip(detail)),
    );
  }
  const verdict = verdictPanel(detail);
  if (verdict) container.append(verdict);
  const frozen = frozenPanel(detail);
  if (frozen) container.append(frozen);
  const layout = el("div", { class: "detail section-gap" });
  detailView = {
    experimentId,
    detail,
    listHost: null,
    rightHost: null,
    barHost,
    selectedKey,
  };
  const listHost = sessionListPanel(detail, selectedKey);
  const rightHost = sessionDetailPanel(detail, selectedKey);
  detailView.listHost = listHost;
  detailView.rightHost = rightHost;
  layout.append(listHost, rightHost);
  container.append(layout, stepTreePanel(detail), mountedMemoryPanel(detail));
  $main.replaceChildren(container);
  pollTimer = setInterval(async () => {
    try {
      const fresh = await api(
        `/api/experiments/${encodeURIComponent(experimentId)}/status`,
      );
      const raw = fresh.status || {};
      // A state change, a new session or a new run rebuilds the page; stage
      // flips inside one run do not (the live panels poll status themselves).
      if (
        fresh.state !== detail.state ||
        String(raw.session_key || "") !== String(status.session_key || "") ||
        String(raw.run_ref || "") !== String(status.run_ref || "")
      )
        route(true);
    } catch {
      /* transient */
    }
  }, 4000);
}

/* Research s1..sN → 冻结 → 前推 → Held-out → 裁决, read off the ledger
   projection. One continuous replay covers 前推 and Held-out; both stay
   封存中 until the verdict is recorded. */
function stageStrip(detail) {
  const status = detail.status || {};
  const running = (key) => detail.worker_alive && status.session_key === key;
  const researchOver = detail.stage !== "research";
  const research = (detail.sessions || []).filter((s) => s.kind === "research");
  const chips = research.map((session) => {
    const record = session.record;
    if (record)
      return stageChip(
        session.key,
        OUTCOME_LABELS[record.outcome] || record.outcome,
        "done",
      );
    if (running(session.key)) return stageChip(session.key, "运行中", "running");
    return researchOver
      ? stageChip(session.key, "未运行", "skipped")
      : stageChip(session.key, "待运行", "pending");
  });
  if (!research.length) chips.push(stageChip("研究", "未启动", "pending"));
  const verdict = detail.verdict || {};
  const replay = detail.forward
    ? ["已回放", "done"]
    : detail.frozen
      ? ["封存中", running("forward") ? "running" : "sealed"]
      : researchOver
        ? ["不回放", "skipped"]
        : ["待冻结", "pending"];
  chips.push(
    detail.frozen
      ? stageChip("冻结", detail.frozen.session_key, "done")
      : stageChip("冻结", researchOver ? "未冻结" : "待定", researchOver ? "skipped" : "pending"),
    stageChip("前推", replay[0], replay[1]),
    stageChip("Held-out", replay[0], replay[1]),
    verdict.status
      ? stageChip(
          "裁决",
          VERDICT_LABELS[verdict.status] || verdict.status,
          verdict.status === "graduated" ? "pass" : "fail",
        )
      : stageChip("裁决", "待定", "pending"),
  );
  return el("div", { class: "stage-strip section-gap" }, ...chips);
}

function stageChip(name, note, state) {
  return el(
    "span",
    { class: `stage-chip ${state}` },
    el("span", { class: "stage-name" }, name),
    el("span", { class: "stage-note" }, note),
  );
}

function fmtProb(value) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(2);
}

// One row per statistic of the forward and Held-out slices (verdict.py).
const SLICE_ROWS = [
  ["days", "交易日", String],
  ["neutralized_excess", "中性化超额（年化）", fmtPct, true],
  ["lower_bound", "80% 下界", fmtPct, true],
  ["recency_neutralized_excess", "最近 6 个月中性化超额", fmtPct, true],
  ["tolerance", "容忍线", fmtPct],
  ["tracking_error", "残差跟踪误差", fmtPct],
  ["information_ratio", "IR", fmtSharpe, true],
  ["max_drawdown", "最大回撤", fmtPct],
  ["excess_at_cost_stress", "加倍滑点后超额", fmtPct, true],
  ["round_trips", "平仓次数", String],
  ["mean_gross", "平均仓位", fmtPct],
];

function sliceTable(forward) {
  const slices = forward.slices || {};
  const columns = [
    ["forward", "前推"],
    ["heldout", "Held-out"],
  ].filter(([key]) => slices[key]);
  if (!columns.length) return null;
  const present = (value) => value !== null && value !== undefined;
  const rows = SLICE_ROWS.filter(([field]) =>
    columns.some(([key]) => present(slices[key][field])),
  );
  return dataTable(
    [{ label: "" }, ...columns.map(([, label]) => ({ label, num: true }))],
    rows.map(([field, label, fmt, signed]) => [
      label,
      ...columns.map(([key]) => {
        const value = slices[key][field];
        return {
          value: present(value) ? fmt(value) : "—",
          cls: signed ? signCls(value) : "",
        };
      }),
    ]),
    { fit: true, box: "section-gap" },
  );
}

/* The forward and Held-out replay of the frozen artifact: 封存中 until the
   forward record exists, then the verdict with every failed condition, the
   slice statistics, the one continuous curve with the Held-out boundary
   marked, and the Paper command for a graduate. A research that froze nothing
   has only its verdict. */
function verdictPanel(detail) {
  const verdict = detail.verdict || {};
  const forward = detail.forward;
  if (!detail.frozen && !verdict.status) return null;
  const reasons = (verdict.reasons || []).map(reasonLabel);
  const panel = el(
    "div",
    { class: "panel section-gap" },
    panelHead("前推与 Held-out", verdictBadge(detail.verdict)),
  );
  if (!forward) {
    if (verdict.status) {
      panel.append(el("div", {}, reasons.join("；") || "—"));
      return panel;
    }
    const status = detail.status || {};
    const replaying = detail.worker_alive && status.session_key === "forward";
    panel.append(
      el(
        "div",
        { class: "prep-indicator" },
        replaying ? el("span", { class: "spinner" }) : null,
        el(
          "span",
          {},
          replaying
            ? `封存中 · ${formatStageLine(status) || "回放中"}`
            : "封存中 · 回放尚未运行",
        ),
      ),
    );
    return panel;
  }
  const replay = forward.replay || {};
  panel.append(
    el(
      "div",
      {},
      reasons.length ? `未通过：${reasons.join("；")}` : "全部条件通过",
    ),
    el(
      "div",
      { class: "meta-line" },
      `前推 ${fmtDate(replay.start)} ～ ${fmtDate(replay.forward_end)} · Held-out ${fmtDate(replay.heldout_start)} ～ ${fmtDate(replay.replay_end)}`,
      replay.truncation_reason ? `（请求至 ${fmtDate(replay.requested_end)}，截至发布末日）` : "",
    ),
  );
  if (forward.error)
    panel.append(el("div", { class: "hint warn" }, `策略报错：${forward.error}`));
  const table = sliceTable(forward);
  if (table) panel.append(table);
  const refits = forward.refits_executed || {};
  if (forward.result)
    panel.append(
      el(
        "div",
        { class: "meta-line section-gap" },
        `重训 前推 ${refits.forward ?? "—"} 次 · Held-out ${refits.heldout ?? "—"} 次 · 前推 null 分位 ${fmtProb(forward.null_percentile)}`,
      ),
      el(
        "div",
        { class: "section-gap" },
        el("h4", { class: "subsection-title" }, "日度累计收益 vs 沪深300"),
        resultEquityHost(detail.experiment_id, forward.result, {
          width: 980,
          height: 240,
          ddH: 90,
          markers: forwardMarkers(forward),
        }),
      ),
      styleCard(detail.experiment_id, forward.result),
      Object.assign(
        lazyDetails("交易明细", () =>
          ordersNode(detail.experiment_id, forward.result),
        ),
        { className: "fold section-gap" },
      ),
    );
  if (detail.paper_candidate)
    panel.append(
      el("h4", { class: "subsection-title section-gap" }, "Paper 建簿"),
      el("pre", { class: "code-view" }, detail.paper_candidate.command),
      el("div", { class: "hint" }, "在仓库根目录运行；Paper 不会自动启动。"),
    );
  return panel;
}

/* The frozen artifact and the research statistics it was frozen on. */
function frozenPanel(detail) {
  const frozen = detail.frozen;
  if (!frozen) return null;
  const panel = el(
    "div",
    { class: "panel section-gap" },
    panelHead(
      `冻结产物 · ${sessionLabel(frozen.session_key)}`,
      el("span", { class: "spacer" }),
      el(
        "a",
        {
          class: "btn small",
          href: `/api/experiments/${encodeURIComponent(detail.experiment_id)}/frozen/strategy.zip`,
        },
        "⬇ 下载 ZIP",
      ),
    ),
    statTilesRow([
      {
        label: "研究期中性化超额（年化）",
        value: fmtPct(frozen.neutralized_excess),
        cls: signCls(frozen.neutralized_excess),
      },
      { label: "残差跟踪误差", value: fmtPct(frozen.tracking_error) },
      {
        label: "IR",
        value: fmtSharpe(frozen.information_ratio),
        cls: signCls(frozen.information_ratio),
      },
      {
        label: "去偏 Sharpe 概率",
        value: fmtProb(frozen.deflated_sharpe_probability),
        title: `试验 ${frozen.trials ?? "—"} 个 · SR* ${fmtSharpe(frozen.sharpe_star)}`,
      },
      {
        label: "前推可检出超额",
        value: fmtPct(frozen.forward_mde),
        title: "前推检验以 80% 功效能检出的最小年化中性化超额",
      },
    ]),
    el(
      "div",
      { class: "meta-line section-gap" },
      [
        `全区间验证 ${frozen.full_span_validations ?? "—"} 次`,
        `null 分位 ${fmtProb(frozen.null_percentile)}`,
        frozen.fit ? `fit，重训周期 ${frozen.refit_period || "—"}` : "无 fit",
        `来源节点 ${frozen.source_step_id || "—"}`,
      ].join(" · "),
    ),
  );
  const blocks = subWindowSection("研究期分年度表现", frozen.blocks);
  if (blocks) panel.append(blocks);
  if (frozen.result)
    panel.append(
      el(
        "div",
        { class: "section-gap" },
        el("h4", { class: "subsection-title" }, "研究期日度累计收益 vs 沪深300"),
        resultEquityHost(detail.experiment_id, frozen.result, {
          width: 860,
          height: 210,
          ddH: 76,
        }),
      ),
      styleCard(detail.experiment_id, frozen.result),
    );
  return panel;
}

/* Full creation-parameter record (params.json), grouped: explicit settings
   first, metadata last; values rendered verbatim so the researcher sees the
   exact configuration this experiment was built from. */
async function openParamsModal(detail) {
  const params = detail.params || {};
  // The create form only persists values that differ from the defaults, so the
  // full effective configuration = schema defaults overlaid with params.json.
  let schemaFields = [];
  try {
    const schema = await api("/api/parameter-schema");
    schemaFields = (schema.groups || []).flatMap((group) => group.fields || []);
  } catch {
    /* fall back to explicit params only */
  }
  const render = (value) =>
    el(
      "code",
      {},
      typeof value === "object" ? JSON.stringify(value) : String(value),
    );
  const explicitRows = [];
  const defaultRows = [];
  const covered = new Set();
  for (const field of schemaFields) {
    covered.add(field.key);
    if (Object.hasOwn(params, field.key)) {
      explicitRows.push(
        kvRow(
          el(
            "span",
            {},
            field.label,
            el("div", { class: "hint flush" }, field.key),
          ),
          render(params[field.key]),
        ),
      );
    } else {
      defaultRows.push(
        kvRow(
          el(
            "span",
            {},
            field.label,
            el("div", { class: "hint flush" }, field.key),
          ),
          el(
            "span",
            { class: "hint" },
            render(field.default ?? "—").textContent,
          ),
        ),
      );
    }
  }
  // Anything persisted outside the schema (metadata, inherited artifact, …).
  const extraRows = Object.keys(params)
    .filter((key) => !covered.has(key))
    .sort()
    .map((key) => kvRow(key, render(params[key])));
  const body = el(
    "div",
    { class: "params-modal-body" },
    el("p", { class: "hint" }, "实际生效值以 run manifest 为准。"),
    explicitRows.length
      ? el("h4", {}, `显式设置（${explicitRows.length}）`)
      : null,
    explicitRows.length ? el("table", { class: "kv" }, ...explicitRows) : null,
    el("h4", {}, `默认值（${defaultRows.length}）`),
    el("table", { class: "kv" }, ...defaultRows),
    extraRows.length ? el("h4", {}, "元数据 / 其他") : null,
    extraRows.length ? el("table", { class: "kv" }, ...extraRows) : null,
  );
  showModal(`创建参数 · ${detail.experiment_id}`, body, [
    el("button", { class: "btn", onclick: closeModal }, "关闭"),
  ]);
}

/* The control row's left side: where the arm is, the current stage while a
   worker runs, and the mounted skills count. */
function runStatusLine(detail) {
  const status = detail.status || {};
  return el(
    "span",
    { class: "mode-note" },
    [
      stageText(detail),
      detail.worker_alive ? formatStageLine(status, { elapsed: false }) : null,
      `Skills ${Number(detail.skills && detail.skills.count) || 0} 项`,
    ]
      .filter(Boolean)
      .join(" · "),
  );
}

function controlBar(detail) {
  const id = detail.experiment_id;
  const control = detail.control || { request: null };
  const state = detail.state;
  const alive = detail.worker_alive;
  const send = (payload, note) => sendControlAction(id, payload, note);
  const actions = el("div", { class: "control-actions" });
  const bar = el("div", { class: "control-bar" }, runStatusLine(detail));
  if (control.request === "pause")
    bar.append(el("span", { class: "badge state-paused" }, "已请求暂停"));
  if (control.request === "stop")
    bar.append(el("span", { class: "badge state-stopped" }, "已请求停止"));
  if (control.restart_pending)
    bar.append(
      el("span", { class: "badge state-waiting_user" }, "已请求会话边界重启"),
    );
  if (alive) {
    if (control.request === "pause") {
      actions.append(
        el(
          "button",
          {
            class: "btn primary",
            onclick: () => send({ action: "resume" }, "已继续"),
          },
          "继续",
        ),
      );
    } else {
      actions.append(
        el(
          "button",
          {
            class: "btn",
            onclick: () =>
              send({ action: "pause" }, "将在当前会话结束后暂停"),
          },
          "暂停",
        ),
      );
    }
    actions.append(
      el(
        "button",
        {
          class: "btn",
          onclick: () => send({ action: "stop" }, "将在当前会话结束后停止"),
        },
        "停止",
      ),
    );
    actions.append(
      el(
        "button",
        {
          class: "btn danger",
          onclick: () => {
            showModal(
              "强制终止",
              el(
                "p",
                {},
                "立即终止 worker（SIGTERM，10 秒后 SIGKILL）；未落账的当前会话在恢复后整体重跑。",
              ),
              [
                el("button", { class: "btn", onclick: closeModal }, "取消"),
                el(
                  "button",
                  {
                    class: "btn danger",
                    onclick: () => {
                      closeModal();
                      // The request blocks through the 10s SIGTERM grace; say so up
                      // front, then report the actual outcome from the response.
                      toast("正在终止 worker（优雅退出宽限最长约 10 秒）…");
                      send(
                        { action: "terminate" },
                        (result) =>
                          `${
                            result.escalated
                              ? `已强制终止（SIGKILL，pid ${result.terminated_pid}）`
                              : `worker 已优雅退出（pid ${result.terminated_pid}）`
                          }`,
                      );
                    },
                  },
                  "强制终止",
                ),
              ],
            );
          },
        },
        "强制终止",
      ),
    );
    actions.append(
      el(
        "button",
        {
          class: "btn",
          onclick: () => {
            showModal(
              "重启 worker",
              el(
                "div",
                {},
                el(
                  "p",
                  {},
                  "立即重启：终止 worker 并按账本恢复，被中断的会话整体重跑。",
                ),
                el(
                  "p",
                  {},
                  "会话边界重启：当前会话落账后再换用新代码继续。",
                ),
              ),
              [
                el("button", { class: "btn", onclick: closeModal }, "取消"),
                el(
                  "button",
                  {
                    class: "btn",
                    onclick: () => {
                      closeModal();
                      send(
                        { action: "restart", at: "session_boundary" },
                        "已请求会话边界重启：当前会话结束后自动换代码重启",
                      );
                    },
                  },
                  "会话边界重启",
                ),
                el(
                  "button",
                  {
                    class: "btn primary",
                    onclick: () => {
                      closeModal();
                      // The request blocks through the 30s SIGTERM grace; say so
                      // up front, then report the actual outcome from the response.
                      toast("正在重启 worker（优雅退出宽限最长约 30 秒）…");
                      send(
                        { action: "restart" },
                        (result) =>
                          `已重启 worker（pid ${result.spawned_pid}${
                            result.escalated ? "；旧 worker 被强制终止" : ""
                          }）`,
                      );
                    },
                  },
                  "立即重启",
                ),
              ],
            );
          },
        },
        "重启",
      ),
    );
  } else if (RESUMABLE_STATES.includes(state)) {
    actions.append(
      el(
        "button",
        {
          class: "btn primary",
          onclick: () => send({ action: "resume" }, "已请求恢复运行"),
        },
        "恢复运行",
      ),
    );
  }
  if (actions.children.length) bar.append(actions);
  return bar;
}

/* One line per planned session: a research session's outcome and its best
   full-span candidate, or where the forward replay stands. */
function sessionListLine(detail, session, pending) {
  if (session.kind === "forward") {
    const text = detail.forward
      ? "已判定"
      : detail.frozen
        ? "封存中"
        : detail.stage === "research"
          ? pending
          : "不回放";
    return { text, cls: "", note: null };
  }
  const record = session.record;
  if (!record) return { text: pending, cls: "", note: null };
  const best = record.best;
  return {
    text: OUTCOME_LABELS[record.outcome] || record.outcome,
    cls: record.froze ? "pos" : "",
    note: best
      ? `最佳候选 中性化 ${fmtPct(best.neutralized_excess)} · DSR ${fmtProb(best.deflated_sharpe_probability)}`
      : `验证 ${record.validations.length} 次，无全区间`,
    noteTitle: best
      ? "本会话 IR 最高的全区间验证：研究期中性化超额与去偏 Sharpe 概率"
      : null,
  };
}

function sessionListPanel(detail, selectedKey) {
  const panel = el("div", { class: "panel" }, el("h4", {}, "会话"));
  const list = el("div", { class: "session-list" });
  const status = detail.status || {};
  for (const session of detail.sessions || []) {
    const isDone = isSessionDone(detail, session);
    const isCurrent = status.session_key === session.key && detail.worker_alive;
    const dotClass = isDone ? "done" : isCurrent ? "running" : "pending";
    const stateText =
      isCurrent && detail.state === "paused"
        ? "已暂停"
        : isCurrent
          ? formatStageLine(status, { elapsed: false }) || "运行中"
          : "";
    const line = sessionListLine(
      detail,
      session,
      stateText || (isDone ? "" : "未运行"),
    );
    const ret =
      session.kind === "research"
        ? sessionDurationNode(detail, session, line.text, line.cls)
        : el("span", { class: line.cls }, line.text);
    ret.classList.add("ret");
    list.append(
      el(
        "div",
        {
          class: `session-item${session.key === selectedKey ? " selected" : ""}`,
          "data-key": session.key,
          onclick: () => {
            location.hash = `#/exp/${encodeURIComponent(detail.experiment_id)}/${sessionKeyToUrl(session.key)}`;
          },
        },
        el("span", { class: `dot ${dotClass}` }),
        el("span", { class: "label" }, sessionLabel(session.key)),
        ret,
        line.note
          ? el(
              "span",
              { class: "session-note", title: line.noteTitle || null },
              line.note,
            )
          : null,
      ),
    );
  }
  panel.append(list);
  return panel;
}

function sessionDetailPanel(detail, selectedKey) {
  const session = (detail.sessions || []).find(
    (entry) => entry.key === selectedKey,
  );
  // Flex column with a uniform card gap: whichever cards are present, the
  // first one's top aligns with the session list in the left grid column.
  const panel = el("div", { class: "session-detail" });
  if (!session) {
    panel.append(
      el(
        "div",
        { class: "panel" },
        el("div", { class: "empty" }, "请选择左侧的会话"),
      ),
    );
    return panel;
  }
  const status = detail.status || {};
  const isCurrent = status.session_key === session.key && detail.worker_alive;
  const running = isCurrent && ACTIVE_SESSION_STATES.has(detail.state);
  const preparing = isPrepEnvironment(status, detail.state);
  const done = isSessionDone(detail, session);
  if (session.kind === "forward") {
    if (isCurrent && !done) panel.append(environmentStagePanel(detail));
    panel.append(forwardSessionPanel(detail, session));
    return panel;
  }
  if (done) {
    panel.append(researchSessionPanel(detail, session));
    return panel;
  }
  if (isCurrent) {
    if (preparing) panel.append(environmentStagePanel(detail));
    else if (running) panel.append(liveTracePanel(detail, session));
    panel.append(injectMessagePanel(detail, session));
  } else {
    if (detail.kind === "hitl" && detail.stage === "research")
      panel.append(directivePanel(detail, session));
    panel.append(
      el(
        "div",
        { class: "panel section-gap" },
        el(
          "div",
          { class: "empty" },
          detail.stage === "research" ? "该会话尚未开始。" : "研究已结束，该会话不再运行。",
        ),
      ),
    );
  }
  return panel;
}

function forwardSessionPanel(detail, session) {
  const replay = session.replay || {};
  const note = detail.forward
    ? "已判定，结果见上方。"
    : detail.frozen
      ? "封存中：回放结束并判定后才显示结果。"
      : detail.stage === "research"
        ? "研究冻结产物后运行。"
        : "研究未冻结产物，不回放。";
  return el(
    "div",
    { class: "panel section-gap" },
    el("h4", {}, "前推回放"),
    el(
      "table",
      { class: "kv" },
      kvRow(
        "前推",
        replay.start ? `${fmtDate(replay.start)} ～ ${fmtDate(replay.forward_end)}` : "—",
      ),
      kvRow(
        "Held-out",
        replay.heldout_start
          ? `${fmtDate(replay.heldout_start)} ～ ${fmtDate(replay.replay_end)}`
          : "—",
      ),
    ),
    el("div", { class: "meta-line" }, note),
  );
}

/* One recorded research session: how it ended, its best full-span candidate
   with the deflated Sharpe the freeze gate would give it, the freeze gate it
   met when it nominated, every Validation it ran, and its Trace. */
function researchSessionPanel(detail, session) {
  const record = session.record;
  const best = record.best || {};
  const gate = record.freeze_gate;
  const panel = el(
    "div",
    { class: "panel" },
    panelHead(
      sessionLabel(session.key),
      el(
        "span",
        { class: `badge ${record.froze ? "state-completed" : "state-stopped"}` },
        OUTCOME_LABELS[record.outcome] || record.outcome,
      ),
      record.finish_reason
        ? el("span", { class: "mode-note" }, `结束原因 ${record.finish_reason}`)
        : null,
    ),
    statTilesRow([
      {
        label: "最佳候选中性化超额",
        value: fmtPct(best.neutralized_excess),
        cls: signCls(best.neutralized_excess),
        title: "本会话 IR 最高的全区间验证，研究期年化",
      },
      {
        label: "IR",
        value: fmtSharpe(best.information_ratio),
        cls: signCls(best.information_ratio),
      },
      {
        label: "去偏 Sharpe 概率",
        value: fmtProb(best.deflated_sharpe_probability),
      },
      {
        label: "验证 / 累计试验",
        value: `${record.validations.length} / ${record.trials_to_date ?? "—"}`,
      },
    ]),
    el(
      "table",
      { class: "kv section-gap" },
      record.reason ? kvRow("理由", record.reason) : null,
      gate
        ? kvRow(
            "冻结门",
            gate.passed
              ? `通过（去偏 Sharpe 概率 ${fmtProb(gate.deflated_sharpe_probability)}）`
              : `未通过：${gate.reasons.map(reasonLabel).join("；")}`,
          )
        : null,
      record.arm_end ? kvRow("结束实验", record.arm_end.reason || "—") : null,
      record.run_wall_seconds
        ? kvRow("耗时", fmtDuration(record.run_wall_seconds))
        : null,
      record.next_start_node_id
        ? kvRow("下一会话起点", record.next_start_node_id)
        : null,
    ),
  );
  if (record.validations.length)
    panel.append(
      dataTable(
        [
          { label: "节点" },
          { label: "区间" },
          { label: "收益", num: true },
          { label: "Sharpe", num: true },
          { label: "回撤", num: true },
          { label: "中性化超额", num: true },
          { label: "IR", num: true },
        ],
        record.validations.map((row) => [
          {
            value: [
              String(row.step_id || "").split("__").pop(),
              row.step_id === record.nominated_step_id
                ? el("span", { class: "badge kind" }, "提名")
                : null,
            ],
            title: row.step_id,
          },
          row.span || "—",
          { value: fmtPct(row.total_return), cls: signCls(row.total_return) },
          { value: fmtSharpe(row.sharpe), cls: signCls(row.sharpe) },
          fmtPct(row.max_drawdown),
          {
            value: fmtPct(row.neutralized_excess),
            cls: signCls(row.neutralized_excess),
          },
          {
            value: fmtSharpe(row.information_ratio),
            cls: signCls(row.information_ratio),
          },
        ]),
        { box: "section-gap" },
      ),
    );
  if (record.prior_published && record.prior)
    panel.append(
      el(
        "details",
        { class: "fold section-gap" },
        el("summary", {}, "本会话发布的 PRIOR"),
        el("pre", { class: "code-view" }, record.prior),
      ),
    );
  const statsHost = el("div", {});
  panel.append(
    el(
      "div",
      { class: "section-gap" },
      el(
        "button",
        { class: "btn", onclick: () => openInitialPrompt(detail, session) },
        "查看初始 Prompt（实际运行）",
      ),
    ),
    statsHost,
    traceReplayNode(detail.experiment_id, record.run_ref, detail),
  );
  api(
    `/api/experiments/${encodeURIComponent(detail.experiment_id)}/trace/stats?run_id=${encodeURIComponent(record.run_ref)}`,
  )
    .then((stats) => statsHost.append(statsChipsRow(stats)))
    .catch(() => {
      /* a session that crashed before its trace has none */
    });
  return panel;
}

function environmentStagePanel(detail) {
  const value = el(
    "div",
    { class: "prep-indicator" },
    el("span", { class: "spinner" }),
    el("span", {}),
  );
  const panel = el(
    "div",
    { class: "panel" },
    el("h4", {}, "Environment 运行状态"),
    value,
  );
  let status = detail.status || {};
  const update = () => {
    const stage = status.environment_stage;
    const started = Date.parse(
      status.environment_stage_started_at || status.session_started_at || "",
    );
    const elapsed = Number.isFinite(started)
      ? ` · ${fmtDuration((Date.now() - started) / 1000)}`
      : "";
    value.lastChild.textContent = `${ENVIRONMENT_STAGE_LABELS[stage] || stage || "处理中"}${elapsed}`;
  };
  update();
  const timer = setInterval(async () => {
    try {
      const fresh = await api(
        `/api/experiments/${encodeURIComponent(detail.experiment_id)}/status`,
      );
      status = fresh.status || status;
      if (value.isConnected) update();
    } catch {
      /* preserve last confirmed phase */
    }
  }, 2500);
  liveTimers.push(timer);
  return panel;
}

function directivePanel(detail, session) {
  const control = detail.control || { directives: {} };
  const experimentDirective = String(
    (detail.params || {}).fold_exploration_directive || "",
  ).trim();
  const textarea = el("textarea", {
    class: "directive",
    placeholder: "可选：为本会话追加研究方向……",
  });
  textarea.value = (control.directives || {})[session.key] ?? "";
  const panel = el(
    "div",
    { class: "panel" },
    el("h4", {}, `${sessionLabel(session.key)} 研究者指令`),
  );
  if (experimentDirective) {
    panel.append(
      el(
        "details",
        { class: "fold" },
        el("summary", {}, "实验级探索方向（已自动注入）"),
        el("div", { class: "markdown pre-wrap" }, experimentDirective),
      ),
    );
  }
  panel.append(
    textarea,
    el("div", { class: "hint warn" }, "须在会话启动前保存；不要写入日历日期。"),
  );
  const send = (payload, note) =>
    sendControlAction(detail.experiment_id, payload, note);
  panel.append(
    el(
      "div",
      { class: "control-bar section-gap" },
      el(
        "button",
        {
          class: "btn",
          onclick: () => openPromptPreview(detail, session, textarea.value),
        },
        "预览完整系统提示词",
      ),
      el(
        "button",
        {
          class: "btn primary",
          onclick: () =>
            send(
              {
                action: "set_directive",
                session_key: session.key,
                directive: textarea.value,
              },
              textarea.value.trim() ? "已保存本会话指令" : "已清除本会话指令",
            ),
        },
        "保存指令",
      ),
    ),
    gpuAllocationRow(detail, session, send),
  );
  return panel;
}

/* GPU status + per-session allocation picker, shown before the session starts.
   The chosen count rides in control.gpu_counts[session_key]; the sandbox's
   "auto" selector then picks that many GPUs by free memory at start, so rows
   are ranked by free memory, the top N are marked as the likely allocation,
   and each bar tracks FREE memory (longer = more headroom). */
function gpuAllocationRow(detail, session, send) {
  const current = ((detail.control || {}).gpu_counts || {})[session.key];
  const experimentDefault = Number((detail.params || {}).gpu_count || 1);
  const wrap = el(
    "div",
    { class: "section-gap" },
    el("h4", { class: "subsection-title" }, "本会话 GPU 分配"),
    el("div", { class: "hint" }, "设备按空闲显存自动挑选；条越长剩余显存越多。"),
  );
  const statusHost = el(
    "div",
    {},
    el("div", { class: "hint" }, "GPU 状态加载中…"),
  );
  const stamp = el("span", { class: "hint flush push-right" });
  const select = el("select", {
    class: "input",
    // Re-mark the likely allocation instantly on count change; the cached
    // inventory avoids a refetch between the 60s polls.
    onchange: () => {
      if (gpuCache) renderGpus(gpuCache);
    },
  });
  select.append(
    el("option", { value: "" }, `实验默认（${experimentDefault} 块）`),
  );
  for (let n = 0; n <= 4; n += 1)
    select.append(
      el("option", { value: String(n) }, n === 0 ? "0 块（CPU）" : `${n} 块`),
    );
  if (current) select.value = String(current);
  const row = el(
    "div",
    { class: "control-bar section-gap" },
    el("span", { class: "mode-note" }, "分配数量："),
    select,
    el(
      "button",
      {
        class: "btn small",
        onclick: () =>
          send(
            {
              action: "set_gpu_count",
              session_key: session.key,
              directive: select.value,
            },
            select.value
              ? `本会话将分配 ${select.value} 块 GPU`
              : "已恢复默认 GPU 分配",
          ),
      },
      "保存",
    ),
    current
      ? el("span", { class: "badge state-waiting_user" }, `已设 ${current} 块`)
      : null,
    stamp,
  );
  wrap.append(statusHost, row);
  // Render the cached inventory: rows mirror the sandbox "auto" selector
  // (free-memory ranking) so the first N rows match the picker's current
  // count; bars track FREE memory (longer = more free), not machine-wide use.
  let gpuCache = null;
  const renderGpus = (gpus) => {
    gpuCache = gpus;
    const count =
      select.value === "" ? experimentDefault : Number(select.value);
    const grid = el("div", { class: "gpu-grid" });
    [...gpus]
      .sort(
        (a, b) => b.memory_free_mib - a.memory_free_mib || a.index - b.index,
      )
      .forEach((gpu, i) => {
        const picked = i < count;
        const freeGib = (gpu.memory_free_mib / 1024).toFixed(1);
        const totalGib = (gpu.memory_total_mib / 1024).toFixed(1);
        const freePct = gpu.memory_total_mib
          ? Math.round((100 * gpu.memory_free_mib) / gpu.memory_total_mib)
          : 0;
        const util =
          gpu.utilization_pct === null || gpu.utilization_pct === undefined
            ? "—"
            : `${gpu.utilization_pct}%`;
        const temp =
          gpu.temperature_c === null || gpu.temperature_c === undefined
            ? "—"
            : `${gpu.temperature_c}°C`;
        grid.append(
          el(
            "div",
            { class: `gpu-row${picked ? " gpu-pick" : " gpu-dim"}` },
            el(
              "span",
              { class: "gpu-name" },
              `GPU ${gpu.index} · ${gpu.name.replace(/^NVIDIA\s+/, "")}`,
            ),
            el("progress", {
              class: "progress gpu-bar",
              value: gpu.memory_free_mib,
              max: gpu.memory_total_mib,
              title: `显存剩余 ${freePct}%（${freeGib}G / ${totalGib}G）`,
            }),
            // An empty slot on unpicked rows keeps every bar the same length.
            picked ? el("span", { class: "gpu-pick-badge" }, "将分配") : el("span"),
            el(
              "span",
              { class: "gpu-meta" },
              `空闲 ${freeGib}G / ${totalGib}G ｜ 算力 ${util} ｜ ${temp}`,
            ),
          ),
        );
      });
    statusHost.innerHTML = "";
    statusHost.append(grid);
  };
  const refresh = async () => {
    let payload;
    try {
      payload = await api("/api/gpus");
    } catch (error) {
      statusHost.innerHTML = "";
      statusHost.append(
        el("div", { class: "hint" }, `GPU 状态加载失败：${error.message}`),
      );
      return;
    }
    const gpus = payload.gpus || [];
    if (!gpus.length) {
      statusHost.innerHTML = "";
      statusHost.append(
        el(
          "div",
          { class: "hint" },
          `无可用 GPU 信息${payload.error ? `（${payload.error}）` : ""}；将按默认配置运行`,
        ),
      );
      return;
    }
    renderGpus(gpus);
    stamp.textContent = `实时检测 · ${new Date().toLocaleTimeString("zh-CN", { hour12: false, timeZone: "Asia/Shanghai" })}`;
  };
  refresh();
  // Live re-detection while the gate is open; dies with navigation (liveTimers).
  liveTimers.push(
    setInterval(() => {
      if (wrap.isConnected) refresh();
    }, 60_000),
  );
  return wrap;
}

/* POST one control action, then refresh the detail page in place (a full
   route() rebuild flashes the page). Shared by every control-sending panel. */
async function sendControlAction(experimentId, payload, note) {
  try {
    const result = await api(
      `/api/experiments/${encodeURIComponent(experimentId)}/control`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    );
    if (note) toast(typeof note === "function" ? note(result) : note);
    refreshDetail();
    return true;
  } catch (error) {
    toast(error.message, true);
    return false;
  }
}

/* Re-fetch the experiment payload and swap both detail panels in place —
   control-state changes update without a page rebuild or scroll jump. */
async function refreshDetail() {
  if (!detailView) {
    route();
    return;
  }
  try {
    const detail = await api(
      `/api/experiments/${encodeURIComponent(detailView.experimentId)}`,
    );
    detailView.detail = detail;
    if (detailView.barHost) {
      const bar = controlBar(detail);
      detailView.barHost.replaceWith(bar);
      detailView.barHost = bar;
    }
    const list = sessionListPanel(detail, detailView.selectedKey);
    detailView.listHost.replaceWith(list);
    detailView.listHost = list;
    if (detailView.selectedKey) selectSession(detailView.selectedKey);
  } catch {
    route();
  }
}

/* Assemble the session's system prompt (with the draft directive embedded)
   for inspection before the session starts. */
async function openPromptPreview(detail, session, directive) {
  let data;
  try {
    data = await api(
      `/api/experiments/${encodeURIComponent(detail.experiment_id)}/prompt-preview`,
      {
        method: "POST",
        body: JSON.stringify({ session_key: session.key, directive }),
      },
    );
  } catch (error) {
    toast(`预览失败：${error.message}`, true);
    return;
  }
  const footer = [el("button", { class: "btn", onclick: closeModal }, "关闭")];
  showModal(
    `系统提示词预览 — ${sessionLabel(session.key)}`,
    el(
      "div",
      {},
      el("div", { class: "hint" }, data.note),
      el(
        "pre",
        { class: "code-view pre-wrap section-gap" },
        data.prompt,
      ),
      el("div", { class: "hint" }, `共 ${data.prompt.length} 字符`),
    ),
    footer,
  );
}

/* The prompt a recorded session ACTUALLY started with: the session_start
   event of its trace (ground truth, unlike the pre-session preview). */
async function openInitialPrompt(detail, session) {
  let data;
  try {
    data = await api(
      `/api/experiments/${encodeURIComponent(detail.experiment_id)}/trace/initial-prompt?run_id=${encodeURIComponent(session.record.run_ref)}`,
    );
  } catch (error) {
    toast(`加载失败：${error.message}`, true);
    return;
  }
  const roleLabel = { system: "系统提示词", user: "初始用户消息" };
  const blocks = (data.messages || []).map((message) =>
    el(
      "div",
      { class: "section-gap" },
      el("h4", {}, roleLabel[message.role] || message.role),
      el(
        "pre",
        { class: "code-view pre-wrap" },
        message.content || "",
      ),
    ),
  );
  showModal(
    `初始 Prompt（实际运行）— ${sessionLabel(session.key)}`,
    el("div", {}, ...blocks),
    [el("button", { class: "btn", onclick: closeModal }, "关闭")],
    "prompt-modal",
  );
}

/* Icon labels for the operations chips. Event-type keys (llm_call)
   and tool-call names share one map: both arrive as counters in
   trace/stats. Compact uses compact_ops, not the event-type chip map. */
const STAT_CHIPS = [
  ["llm_call", "🤖 LLM"],
  ["batch_validate", "📊 验证"],
  ["shell", "🖥 Shell"],
  ["read_file", "📄 读取"],
];

const MAIN_AGENT_COUNT_TITLE = "主 Agent 本次会话的调用次数，不含子代理";

function fmtTokens(count) {
  const n = Number(count) || 0;
  if (n >= 1_000_000)
    return `${(n / 1_000_000).toFixed(n >= 10_000_000 ? 0 : 1)} M tokens`;
  if (n >= 1000) return `${Math.round(n / 1000)} k tokens`;
  return `${n} tokens`;
}

function statsChipsRow(stats) {
  const counts = { ...(stats.counts || {}), ...(stats.tool_counts || {}) };
  const chips = el("div", { class: "stats-chips" });
  const labelled = new Set();
  const subagentTasks = Number(stats.subagent_tasks) || 0;
  const subagentRunning = Number(stats.subagent_running) || 0;
  for (const [key, label] of STAT_CHIPS) {
    labelled.add(key);
    if (counts[key])
      chips.append(
        el(
          "span",
          { class: "stat-chip", title: MAIN_AGENT_COUNT_TITLE },
          `${label} ${counts[key]}`,
        ),
      );
    if (key === "llm_call" && subagentTasks)
      chips.append(
        el(
          "span",
          {
            class: subagentRunning ? "stat-chip run" : "stat-chip",
            title: "仍在运行的子代理 / 本次会话累计启动的子代理",
          },
          `🧩 子代理 ${subagentRunning} 运行 / ${subagentTasks} 累计`,
        ),
      );
  }
  for (const [tool, count] of Object.entries(stats.tool_counts || {})) {
    if (!labelled.has(tool) && tool !== "agent")
      chips.append(
        el(
          "span",
          { class: "stat-chip", title: MAIN_AGENT_COUNT_TITLE },
          `${tool} ${count}`,
        ),
      );
  }
  chips.append(
    el(
      "span",
      { class: "stat-chip", title: "语义压缩次数" },
      `Compact ${Number(stats.compact_ops) || 0}`,
    ),
  );
  if (stats.llm_prompt_tokens || stats.llm_completion_tokens) {
    chips.append(
      el(
        "span",
        { class: "stat-chip" },
        `主 Agent 累计输入 ${fmtTokens(stats.llm_prompt_tokens)}`,
      ),
      el(
        "span",
        { class: "stat-chip" },
        `主 Agent 累计输出 ${fmtTokens(stats.llm_completion_tokens)}`,
      ),
    );
  } else if (stats.llm_total_tokens) {
    chips.append(
      el(
        "span",
        { class: "stat-chip" },
        `主 Agent Σ ${fmtTokens(stats.llm_total_tokens)}`,
      ),
    );
  }
  // Child LLM calls are a separate event stream: shown beside the parent's
  // totals, never folded into them.
  const subagentTokens = Number(stats.subagent_total_tokens) || 0;
  if (subagentTokens) {
    chips.append(
      el(
        "span",
        {
          class: "stat-chip",
          title: `${fmtTokens(stats.subagent_prompt_tokens)} 输入 · ${fmtTokens(stats.subagent_completion_tokens)} 输出，不计入主 Agent 累计`,
        },
        `🧩 子代理 Σ ${fmtTokens(subagentTokens)}`,
      ),
    );
  }
  const used = Number(stats.last_llm_prompt_tokens) || 0;
  const window = Number(stats.context_window_tokens) || 0;
  if (used > 0 && window > 0) {
    const pct = Math.min(100, Math.round((100 * used) / window));
    chips.append(
      el(
        "span",
        {
          class: "stat-chip",
          title: `${fmtTokens(used)} / ${fmtTokens(window)}`,
        },
        `主 Agent 上下文 ${pct}%`,
      ),
    );
  }
  return chips;
}

function injectDraftKey(experimentId, sessionKey) {
  return `${experimentId}\0${sessionKey}`;
}

function injectMessageEnabled(detail, session) {
  const status = (detail && detail.status) || {};
  const kind = session && session.kind;
  const sessionKey = session && session.key;
  return Boolean(
    kind === "research" &&
      LIVE_RUN_STATES.has(detail && detail.state) &&
      sessionKey &&
      status.session_key === sessionKey &&
      detail &&
      detail.worker_alive,
  );
}

function injectMessageDisableReason(detail, session) {
  if (injectMessageEnabled(detail, session)) return "";
  const state = (detail && detail.state) || "";
  if (state === "paused") return "实验已暂停。请先恢复运行后再发送。";
  if (TERMINAL_INJECT_STATES.has(state))
    return "会话已结束，无法发送。";
  return "当前没有可接收消息的 Agent 会话。";
}

function buildInjectMessagePayload(sessionKey, text, interrupt) {
  return {
    action: "inject_message",
    session_key: sessionKey,
    text,
    interrupt: Boolean(interrupt),
  };
}

function validateInjectMessageText(text) {
  const value = String(text ?? "");
  if (!value.trim()) return { ok: false, error: "消息不能为空" };
  if ([...value].length > INJECT_MESSAGE_MAX_CHARS)
    return {
      ok: false,
      error: `消息不能超过 ${INJECT_MESSAGE_MAX_CHARS} 个字符`,
    };
  return { ok: true, text: value };
}

function inboxQueueSummary(inbox) {
  const pending = Number((inbox && inbox.pending_count) || 0);
  const ids = Array.isArray(inbox && inbox.queued_ids)
    ? inbox.queued_ids.map(String)
    : [];
  return { pending_count: pending, queued_ids: ids };
}

function injectMessagePanel(detail, session) {
  const enabled = injectMessageEnabled(detail, session);
  const reason = injectMessageDisableReason(detail, session);
  const draftKey = injectDraftKey(detail.experiment_id, session.key);
  const queue = inboxQueueSummary(detail.inbox);
  const textarea = el("textarea", {
    class: "directive inject-input",
    maxlength: String(INJECT_MESSAGE_MAX_CHARS),
    placeholder: "写入给当前 Agent 的消息……",
  });
  if (injectDrafts.has(draftKey)) textarea.value = injectDrafts.get(draftKey);
  const count = el("span", { class: "inject-count" });
  const updateCount = () => {
    count.textContent = `${[...textarea.value].length} / ${INJECT_MESSAGE_MAX_CHARS}`;
    injectDrafts.set(draftKey, textarea.value);
  };
  textarea.addEventListener("input", updateCount);
  updateCount();
  const sendBtn = el(
    "button",
    { type: "button", class: "btn primary" },
    "发送",
  );
  const interruptBtn = el(
    "button",
    { type: "button", class: "btn" },
    "发送并打断",
  );
  const setBusy = (busy) => {
    const locked = busy || !enabled;
    textarea.disabled = locked;
    sendBtn.disabled = locked;
    interruptBtn.disabled = locked;
  };
  setBusy(false);
  const submit = async (interrupt) => {
    const checked = validateInjectMessageText(textarea.value);
    if (!checked.ok) {
      toast(checked.error, true);
      return;
    }
    const previous = textarea.value;
    setBusy(true);
    injectDrafts.delete(draftKey);
    const ok = await sendControlAction(
      detail.experiment_id,
      buildInjectMessagePayload(session.key, checked.text, interrupt),
      INJECT_MESSAGE_QUEUED_NOTE,
    );
    if (!ok) {
      injectDrafts.set(draftKey, previous);
      if (textarea.isConnected) {
        textarea.value = previous;
        updateCount();
        setBusy(false);
      }
    }
  };
  sendBtn.addEventListener("click", () => submit(false));
  interruptBtn.addEventListener("click", () => submit(true));
  const queueLine =
    queue.pending_count > 0
      ? `排队 ${queue.pending_count} 条${
          queue.queued_ids.length ? `：${queue.queued_ids.join(", ")}` : ""
        }`
      : "当前没有排队消息";
  return el(
    "div",
    { class: "panel inject-message section-gap" },
    el("h4", {}, "发给当前 Agent"),
    el(
      "div",
      { class: enabled ? "hint" : "hint warn" },
      enabled
        ? "「发送并打断」只跳过尚未开始的工具，不会取消已在途的模型调用或已开始的工具。"
        : reason,
    ),
    el("div", { class: "inject-queue" }, queueLine),
    textarea,
    el("div", { class: "inject-meta" }, count),
    el("div", { class: "control-bar" }, sendBtn, interruptBtn),
  );
}

function liveTracePanel(detail, session) {
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", {}, `实时 Agent Trace — ${sessionLabel(session.key)}`),
  );
  const statusLine = el(
    "div",
    { class: "prep-indicator" },
    el("span", { class: "spinner" }),
    el("span", {}, "准备运行状态…"),
  );
  const statsHost = el("div", {});
  const box = el("div", { class: "trace-box" });
  const auto = el("input", { type: "checkbox", checked: "checked" });
  panel.append(
    el(
      "div",
      { class: "trace-tools" },
      el("span", { class: "badge state-running_session" }, "实时"),
      el("label", {}, auto, " 自动滚动"),
    ),
    statusLine,
    statsHost,
    box,
  );

  const experimentId = encodeURIComponent(detail.experiment_id);
  const runId = String((detail.status || {}).run_ref || "");
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  // Claim before the first await so the initial refreshBlocks and
  // pollStats→refreshBlocks cannot both construct an EventSource.
  let streamOpening = false;
  let streamDone = false;
  let lastBlocks = "";
  let refreshTimer = 0;
  const refreshBlocks = async () => {
    const claimStream = !streamOpening;
    if (claimStream) streamOpening = true;
    try {
      const page = await api(
        `/api/experiments/${experimentId}/trace/blocks${query}`,
      );
      lastBlocks = renderTraceBlocks(box, page.blocks || [], {
        truncated: Boolean(page.history_truncated),
        eof: streamDone,
        previous: lastBlocks,
        detail,
        runRef: runId,
      });
      if (auto.checked) {
        const scroller = box.querySelector(".trace-box-scroll") || box;
        scroller.scrollTop = scroller.scrollHeight;
      }
      if (claimStream) openStream(Number(page.next_offset) || 0);
    } catch {
      if (claimStream) openStream(0);
    }
  };
  const scheduleRefresh = () => {
    if (refreshTimer) return;
    refreshTimer = window.setTimeout(() => {
      refreshTimer = 0;
      refreshBlocks();
    }, 400);
  };
  const openStream = (offset) => {
    const separator = query ? "&" : "?";
    const source = new EventSource(
      `/api/experiments/${experimentId}/trace/stream${query}${separator}offset=${offset}`,
    );
    liveSources.push(source);
    source.onmessage = () => scheduleRefresh();
    source.addEventListener("eof", () => {
      streamDone = true;
      refreshBlocks();
      source.close();
    });
  };
  refreshBlocks();

  let currentStatus = detail.status || {};
  const update = () => {
    const stage = currentStatus.environment_stage;
    const started = Date.parse(
      currentStatus.environment_stage_started_at ||
        currentStatus.session_started_at ||
        "",
    );
    const elapsed = Number.isFinite(started)
      ? ` · ${fmtDuration((Date.now() - started) / 1000)}`
      : "";
    const progress = currentStatus.environment_progress || {};
    const done = Number(progress.completed ?? progress.day_index);
    const total = Number(progress.total ?? progress.total_days);
    const measured =
      Number.isFinite(done) && Number.isFinite(total) && total > 0
        ? ` · ${done}/${total}`
        : "";
    const action = progress.tool
      ? ` · ${progress.tool}`
      : progress.call_index
        ? ` · 第 ${progress.call_index} 次调用`
        : "";
    statusLine.lastChild.textContent = `${ENVIRONMENT_STAGE_LABELS[stage] || stage || "准备 AgentTrace"}${measured}${action}${elapsed}`;
    tickElapsedClocks(box);
  };
  update();
  const pollStats = async () => {
    try {
      const fresh = await api(`/api/experiments/${experimentId}/status`);
      currentStatus = fresh.status || currentStatus;
      update();
    } catch {
      /* preserve the last truthful state */
    }
    try {
      const stats = await api(
        `/api/experiments/${experimentId}/trace/stats${query}`,
      );
      statsHost.replaceChildren(statsChipsRow(stats));
    } catch {
      /* trace may not exist during PIT/Sandbox preparation */
    }
    await refreshBlocks();
  };
  // Two cadences, as the console has always had: the elapsed readout ticks
  // every second, the network polls stay at five.
  liveTimers.push(setInterval(update, 1000), setInterval(pollStats, 5000));
  pollStats();
  return panel;
}

/* Mirrors traces.MAX_BLOCK_READ_BYTES: a larger window is refused outright. */
const MAX_TRACE_BLOCK_BYTES = 32 * 1024 * 1024;

/* Replay loader: one backend projection, plus raw .jsonl download. */
function traceReplayNode(experimentId, runId, detail) {
  const box = el("div", { class: "trace-box" });
  const info = el("span", { class: "hint flush" }, "");
  const moreButton = el(
    "button",
    { type: "button", class: "btn small", hidden: "" },
    "继续加载",
  );
  let loadedBlocks = 0,
    eof = false,
    loading = false,
    windowBytes = 0;
  function syncMore() {
    moreButton.disabled = loading;
    moreButton.hidden = eof || !loadedBlocks;
  }
  async function loadBatch() {
    if (loading || eof) return;
    loading = true;
    syncMore();
    try {
      const extra = windowBytes
        ? `&offset=0&max_bytes=${windowBytes}`
        : "";
      const data = await api(
        `/api/experiments/${encodeURIComponent(experimentId)}/trace/blocks?run_id=${encodeURIComponent(runId)}${extra}`,
      );
      const blocks = data.blocks || [];
      renderTraceBlocks(box, blocks, {
        truncated: Boolean(data.history_truncated),
        eof: Boolean(data.eof),
        detail,
        runRef: runId,
      });
      loadedBlocks = blocks.length;
      // D2: the server rejects a window above its own cap with a 422.
      windowBytes = Math.min(
        (Number(data.next_offset) || 0) + 512 * 1024,
        MAX_TRACE_BLOCK_BYTES,
      );
      eof = Boolean(data.eof);
      info.textContent = `已加载 ${loadedBlocks} 个展示块${eof ? "（全部）" : ""}`;
    } catch (error) {
      info.textContent = `加载失败：${error.message}`;
    } finally {
      loading = false;
      syncMore();
    }
  }
  moreButton.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    loadBatch();
  });
  const download = el(
    "a",
    {
      class: "btn small",
      href: `/api/experiments/${encodeURIComponent(experimentId)}/trace/download?run_id=${encodeURIComponent(runId)}`,
    },
    "⬇ 下载完整 .jsonl",
  );
  const body = el(
    "div",
    { class: "trace-replay-body", hidden: "" },
    el("div", { class: "control-bar" }, moreButton, download, info),
    box,
  );
  const toggle = el(
    "button",
    { type: "button", class: "trace-replay-toggle" },
    "Agent Trace（回放）",
  );
  const wrap = el("div", { class: "trace-replay" }, toggle, body);
  toggle.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    const open = wrap.classList.toggle("open");
    body.hidden = !open;
    if (open && !loadedBlocks && !loading) loadBatch();
  });
  return wrap;
}

function lazyDetails(summaryText, build, key) {
  const details = el("details", {}, el("summary", {}, summaryText));
  if (key) details.dataset.key = key;
  details.addEventListener("toggle", () => {
    if (details.open && !details.__filled) {
      details.__filled = true;
      details.append(build());
    }
  });
  return details;
}

const SUBAGENT_STATUS_LABELS = new Map([
  ["started", "已启动"],
  ["running", "进行中"],
  ["completed", "已完成"],
  ["exhausted", "轮次用尽"],
  ["timeout", "超时"],
  ["error", "失败"],
  ["cancelled", "已取消"],
]);
const TERMINAL_SUBAGENT_STATUS = new Set([
  "completed",
  "exhausted",
  "timeout",
  "error",
  "cancelled",
]);

function isRunningSubagent(block) {
  if (!block || block.kind !== "subagent") return false;
  if (String(block.phase || "") === "ended") return false;
  return !TERMINAL_SUBAGENT_STATUS.has(String(block.status || ""));
}

/* Elapsed readouts carry their own bounds in dataset, so one panel-level
   ticker updates every clock in the box — the trace re-renders on each poll,
   and a per-node interval would leak one timer per rebuild. */
function elapsedClockNode(startedAt, endedAt, className = "hint") {
  const from = Date.parse(startedAt || "");
  if (!Number.isFinite(from)) return null;
  const node = el("span", { class: className });
  node.dataset.elapsedFrom = String(from);
  const to = Date.parse(endedAt || "");
  if (Number.isFinite(to)) node.dataset.elapsedTo = String(to);
  tickElapsedClocks(node);
  return node;
}

function tickElapsedClocks(root) {
  if (!root) return;
  const nodes = root.dataset && root.dataset.elapsedFrom
    ? [root]
    : root.querySelectorAll("[data-elapsed-from]");
  for (const node of nodes) {
    const from = Number(node.dataset.elapsedFrom);
    const to = node.dataset.elapsedTo
      ? Number(node.dataset.elapsedTo)
      : Date.now();
    node.textContent = `⏱ ${fmtDuration((to - from) / 1000)}`;
  }
}

function subagentClockNode(block, className = "hint") {
  return elapsedClockNode(
    block.started_at || block.ts,
    isRunningSubagent(block) ? "" : block.ended_at,
    className,
  );
}

/* Rounds and LLM calls differ when a sub-agent is forced to write a closing
   summary after its last round, so both are shown. */
function subagentProgressParts(block) {
  const parts = [];
  const rounds = Number(block.rounds) || 0;
  const llmCalls = Number(block.llm_calls) || 0;
  const toolCalls = Number(block.tool_calls) || 0;
  if (rounds || llmCalls) parts.push(`${rounds} 轮 · 模型 ${llmCalls} 次`);
  if (toolCalls) parts.push(`工具 ${toolCalls} 次`);
  const total = Number((block.usage || {}).total_tokens) || 0;
  if (total) parts.push(`Σ ${fmtTokens(total)}`);
  return parts;
}

function subagentUsageTitle(block) {
  const usage = block.usage || {};
  const prompt = Number(usage.prompt_tokens) || 0;
  const completion = Number(usage.completion_tokens) || 0;
  if (!prompt && !completion) return "";
  return `${fmtTokens(prompt)} 输入 · ${fmtTokens(completion)} 输出，不计入主 Agent 累计`;
}

const TOOL_STATUS_LABELS = new Map([
  ["running", "进行中"],
  ["failed", "失败"],
  ["ok", "已完成"],
]);

function subagentLastToolLabel(block) {
  const last = block.last_tool || null;
  const name = last && last.name ? String(last.name) : "";
  if (!name) return "";
  const status = TOOL_STATUS_LABELS.get(String(last.status || "")) || "";
  return `最近工具 ${name}${status ? ` · ${status}` : ""}`;
}

function runningSubagentChip(block, detail, runRef) {
  const role = String(block.role || "子代理");
  const status = String(block.status || "running");
  const statusLabel = SUBAGENT_STATUS_LABELS.get(status) || "进行中";
  const task = String(block.description || "");
  const progress = subagentProgressParts(block);
  const lastTool = subagentLastToolLabel(block);
  const chip = el("button", {
    type: "button",
    class: "trace-subagent-chip",
    title: "查看该子代理的详细 Trace",
    onclick: () => openSubagentTrace(detail, runRef, block),
  });
  chip.append(
    el(
      "span",
      { class: "trace-subagent-chip-title" },
      `🧩 ${role} · ${statusLabel}`,
    ),
  );
  if (task) chip.append(el("span", { class: "trace-subagent-chip-task" }, task));
  // The same launch/elapsed line the inline card and the drawer head render.
  chip.append(subagentHeadMetaNode(block, detail));
  if (progress.length)
    chip.append(
      el(
        "span",
        { class: "hint", title: subagentUsageTitle(block) || null },
        progress.join(" · "),
      ),
    );
  if (lastTool) chip.append(el("span", { class: "hint" }, lastTool));
  return chip;
}

/* The child's own Trace, opened from its card or from the running dock chip.
   It is an overlay: the parent trace, its scroll position and its open folds
   stay exactly as they were, and closing returns to them. */
async function openSubagentTrace(detail, runRef, block) {
  const taskId = String((block && block.task_id) || "");
  if (!taskId || !detail || !detail.experiment_id) return;
  const query = runRef ? `?run_id=${encodeURIComponent(runRef)}` : "";
  const head = el("div", {}, el("div", { class: "loading" }, "加载子代理 Trace…"));
  // One box for the lifetime of the drawer, so a refresh keeps the folds the
  // reader opened instead of collapsing them every five seconds.
  const box = el("div", { class: "trace-box subagent-trace-box" });
  const body = el("div", { class: "subagent-trace" }, head, box);
  showModal(
    `🧩 子代理 Trace · ${String(block.role || "子代理")}`,
    body,
    [el("button", { class: "btn", onclick: closeModal }, "关闭")],
    "subagent-modal",
  );
  let previousBlocks = "";
  const load = async () => {
    if (!body.isConnected) return false;
    let payload;
    try {
      payload = await api(
        `/api/experiments/${encodeURIComponent(detail.experiment_id)}/trace/subagents/${encodeURIComponent(taskId)}${query}`,
      );
    } catch (error) {
      head.replaceChildren(
        el("div", { class: "empty" }, `加载失败：${error.message}`),
      );
      return false;
    }
    head.replaceChildren(subagentTraceHead(payload, detail));
    previousBlocks = renderTraceBlocks(box, payload.blocks || [], {
      detail,
      previous: previousBlocks,
    });
    if (!(payload.blocks || []).length)
      box.replaceChildren(
        el("div", { class: "empty" }, "该子代理尚未产生可展示的轮次。"),
      );
    return isRunningSubagent(payload.header || block);
  };
  if (!(await load())) return;
  // Follow a child that is still working: the clock ticks every second, the
  // records refresh every five, and both stop when it ends or the drawer goes.
  const clock = setInterval(() => {
    if (body.isConnected) tickElapsedClocks(body);
    else clearInterval(clock);
  }, 1000);
  const poll = setInterval(async () => {
    if (!body.isConnected) {
      clearInterval(poll);
      return;
    }
    if (!(await load())) {
      clearInterval(poll);
      clearInterval(clock);
    }
  }, 5000);
  liveTimers.push(clock, poll);
}

function subagentTraceHead(payload, detail) {
  const header = payload.header || {};
  const status = String(header.status || "started");
  const statusLabel = SUBAGENT_STATUS_LABELS.get(status) || status;
  const progress = subagentProgressParts(header);
  const wrap = el("div", {});
  wrap.append(
    el(
      "div",
      { class: "subagent-trace-head" },
      el(
        "span",
        { class: `type subagent ${status}` },
        `🧩 ${String(header.role || "子代理")} · ${statusLabel}`,
      ),
      header.description ? el("span", {}, String(header.description)) : null,
      subagentHeadMetaNode(header, detail),
    ),
  );
  if (progress.length)
    wrap.append(
      el(
        "div",
        { class: "hint", title: subagentUsageTitle(header) || null },
        progress.join(" · "),
      ),
    );
  if (header.error)
    wrap.append(el("div", { class: "hint warn" }, `错误：${header.error}`));
  if (payload.truncated_window)
    wrap.append(el("div", { class: "hint" }, "仅显示当前读取窗口内的记录。"));
  return wrap;
}

function runningSubagentBlocks(blocks) {
  return (blocks || []).filter(isRunningSubagent);
}

function renderTraceBlocks(box, blocks, { truncated, eof, previous, detail, runRef } = {}) {
  const serialized = JSON.stringify({
    blocks: blocks || [],
    truncated: Boolean(truncated),
    eof: Boolean(eof),
  });
  if (previous && serialized === previous) return previous;
  const open = new Set(
    [...box.querySelectorAll("details[open]")]
      .map((node) => node.dataset.key)
      .filter(Boolean),
  );
  const fragment = document.createDocumentFragment();
  if (truncated) {
    fragment.append(
      el(
        "div",
        { class: "hint" },
        "仅显示当前窗口的展示投影；完整记录请下载原始 JSONL。",
      ),
    );
  }
  const scroll = el("div", { class: "trace-box-scroll" });
  const appendNode = (host, block, index) => {
    const node = traceBlockNode(block, index, detail, runRef);
    for (const details of node.querySelectorAll("details[data-key]")) {
      if (open.has(details.dataset.key)) details.open = true;
    }
    host.append(node);
  };
  (blocks || []).forEach((block, index) => appendNode(scroll, block, index));
  if (eof) scroll.append(el("div", { class: "hint" }, "—— trace 结束 ——"));
  fragment.append(scroll);
  const running = runningSubagentBlocks(blocks);
  if (running.length) {
    const dock = el("div", { class: "trace-subagent-dock" });
    running.forEach((block) =>
      dock.append(runningSubagentChip(block, detail, runRef)),
    );
    fragment.append(dock);
  }
  box.replaceChildren(fragment);
  tickElapsedClocks(box);
  return serialized;
}

function traceBlockNode(block, index, detail, runRef) {
  const kind = String((block && block.kind) || "");
  const node = el("div", { class: `trace-block ${kind}` });
  if (kind === "subagent" && block && block.task_id)
    node.dataset.taskId = String(block.task_id);
  try {
    if (kind === "agent_output") renderAgentOutputBlock(node, block, detail);
    else if (kind === "tool_group") renderToolGroupBlock(node, block, index);
    else if (kind === "subagent")
      renderSubagentBlock(node, block, detail, runRef);
    else if (kind === "user") renderUserBlock(node, block);
    else if (kind === "raw") renderRawBlock(node, block);
    else if (kind === "marker") renderMarkerBlock(node, block);
    else if (kind === "summary") renderSummaryBlock(node, block);
    else node.append(el("div", { class: "hint" }, "未知展示块"));
  } catch {
    node.append(el("div", { class: "hint" }, "该展示块无法渲染"));
  }
  return node;
}

function parentReasoningLabel(detail) {
  const params = (detail && detail.params) || {};
  if (params.no_thinking) return "off";
  return params.reasoning_effort || "";
}

function renderAgentOutputBlock(node, block, detail) {
  const effort = parentReasoningLabel(detail);
  const model = String((block && block.model) || "").trim();
  const round = Number(block.round) || 0;
  const title = round
    ? [`第 ${round} 轮`, model].filter(Boolean).join(" · ")
    : ["Agent", model, effort ? `推理 ${effort}` : ""].filter(Boolean).join(" · ");
  node.append(
    el(
      "div",
      { class: "head" },
      el("span", { class: "type agent_output" }, title),
      block.ts ? el("span", {}, fmtTsTime(block.ts)) : null,
    ),
  );
  const text = String(block.text || "");
  if (text) node.append(el("div", { class: "llm-content" }, text));
  const contentChars = Number(block.content_chars) || 0;
  if (!text && contentChars)
    node.append(
      el("div", { class: "hint" }, `模型正文未写入 Trace（${contentChars} 字符）`),
    );
  const reasoningChars = Number(block.reasoning_chars) || 0;
  if (reasoningChars) {
    node.append(
      el(
        "div",
        { class: "hint" },
        `推理过程已折叠（${(reasoningChars / 1000).toFixed(1)}k 字符）`,
      ),
    );
  }
}

function renderToolGroupBlock(node, block, index) {
  const key = `tools:${index}`;
  const details = lazyDetails(
    toolGroupTitle(block),
    () =>
      Array.isArray(block.calls) && block.calls.length
        ? toolCallsNode(block.calls)
        : toolRowsNode(block.tools),
    key,
  );
  node.append(
    el(
      "div",
      { class: "head" },
      el("span", { class: "type tool_group" }, "工具"),
      block.ts ? el("span", {}, fmtTsTime(block.ts)) : null,
    ),
    details,
  );
}

/* `thinking: "inherit"` means the child ran at the parent's effort: show that
   effective level and say it was inherited, never a bare "inherit". */
function subagentThinkingLabel(block, detail) {
  const own = String((block && block.thinking) || "").trim();
  if (own && own !== "inherit") return own;
  const parent = parentReasoningLabel(detail);
  return parent ? `${parent}（继承）` : "继承父会话";
}

function subagentContextLabel(block) {
  if (block && block.resumed_from) return `续用 ${block.resumed_from}`;
  if (block && block.inherit_context === true) return "继承上下文";
  if (block && block.inherit_context === false) return "独立上下文";
  return "";
}

/* The one place sub-agent launch metadata is spelled out:
   `model · 推理 xhigh · 上限 48 轮 · 独立上下文`. */
function subagentMetaLine(block, detail) {
  const thinking = subagentThinkingLabel(block, detail);
  const roundsLimit = Number(block.rounds_limit) || 0;
  return [
    block.model ? String(block.model) : "",
    thinking ? `推理 ${thinking}` : "",
    roundsLimit ? `上限 ${roundsLimit} 轮` : "",
    subagentContextLabel(block),
  ]
    .filter(Boolean)
    .join(" · ");
}

/* `model · 推理 x · 独立上下文 ⏱ 5:21 · 08-28 15:13:20` — the clock is a live
   node, so the separators around it are explicit text nodes rather than a
   flex gap that a wrapped line drops. */
function subagentHeadMetaNode(block, detail) {
  const line = el("span", { class: "hint subagent-meta" });
  const meta = subagentMetaLine(block, detail);
  if (meta) line.append(meta);
  const clock = subagentClockNode(block, "subagent-clock");
  if (clock) {
    if (line.childNodes.length) line.append(" ");
    line.append(clock);
  }
  const launched = block.ts ? fmtTsTime(block.ts) : "";
  if (launched) line.append(line.childNodes.length ? ` · ${launched}` : launched);
  return line.childNodes.length ? line : null;
}

function renderSubagentBlock(node, block, detail, runRef) {
  const status = String(block.status || block.phase || "started");
  const phase = String(block.phase || "");
  const statusLabel =
    phase === "ended" || TERMINAL_SUBAGENT_STATUS.has(status)
      ? SUBAGENT_STATUS_LABELS.get(status) || status
      : SUBAGENT_STATUS_LABELS.get(status) || "进行中";
  const role = String(block.role || "子代理");
  const key = `sub:${block.task_id || ""}`;
  const progress = subagentProgressParts(block);
  const lastTool = subagentLastToolLabel(block);
  node.append(
    el(
      "div",
      {
        class: "head subagent-open",
        title: "查看该子代理的详细 Trace",
        onclick: () => openSubagentTrace(detail, runRef, block),
      },
      el(
        "span",
        { class: `type subagent ${status}` },
        `🧩 ${role} · ${statusLabel}`,
      ),
      block.description ? el("span", {}, String(block.description)) : null,
      subagentHeadMetaNode(block, detail),
      el("span", { class: "subagent-open-hint" }, "详细 Trace ↗"),
    ),
  );
  if (progress.length || lastTool)
    node.append(
      el(
        "div",
        { class: "hint", title: subagentUsageTitle(block) || null },
        [...progress, lastTool].filter(Boolean).join(" · "),
      ),
    );
  node.append(lazyDetails("详情", () => subagentDetailNode(block), key));
}

/* A line the trace writer could not encode, or one that exceeded the
   per-event cap: shown as recorded rather than dropped from the projection. */
function renderRawBlock(node, block) {
  node.append(
    el(
      "div",
      { class: "head" },
      el("span", { class: "type raw" }, "无法解析的记录"),
      block.ts ? el("span", {}, fmtTsTime(block.ts)) : null,
    ),
  );
  const text = String(block.text || "");
  if (text) node.append(el("div", { class: "llm-content" }, text));
}

/* A wrap-up prompt or an output-truncation notice recorded for the child. */
function renderMarkerBlock(node, block) {
  node.append(
    el(
      "div",
      { class: "head" },
      el("span", { class: "type marker" }, String(block.label || "标记")),
      block.ts ? el("span", {}, fmtTsTime(block.ts)) : null,
    ),
  );
  const text = String(block.text || "");
  if (text) node.append(el("div", { class: "hint" }, text));
}

function renderSummaryBlock(node, block) {
  const status = String(block.status || "completed");
  node.append(
    el(
      "div",
      { class: "head" },
      el("span", { class: `type subagent ${status}` }, "最终汇报"),
      block.ts ? el("span", {}, fmtTsTime(block.ts)) : null,
    ),
  );
  const text = String(block.text || "");
  const chars = Number(block.text_chars) || 0;
  node.append(
    text
      ? el("div", { class: "llm-content" }, text)
      : el("div", { class: "hint" }, `汇报正文未写入 Trace（${chars} 字符）`),
  );
}

function toolCallsNode(calls) {
  const list = el("div", { class: "trace-tool-list" });
  for (const call of calls || []) {
    const status = TOOL_STATUS_LABELS.get(String(call.status || "")) || "";
    const round = Number(call.round) || 0;
    const row = el(
      "div",
      { class: "trace-tool-call" },
      el(
        "div",
        { class: "tool-brief" },
        [String(call.name || "工具"), round ? `第 ${round} 轮` : "", status]
          .filter(Boolean)
          .join(" · "),
      ),
    );
    for (const [key, value] of Object.entries(call.arguments || {}))
      row.append(el("div", { class: "tool-arg" }, `${key}: ${value}`));
    if (call.error)
      row.append(el("div", { class: "hint warn" }, String(call.error)));
    if (call.result)
      row.append(el("div", { class: "tool-result" }, String(call.result)));
    list.append(row);
  }
  if (!list.childNodes.length) list.append(el("div", { class: "hint" }, "无工具"));
  return list;
}

function renderUserBlock(node, block) {
  node.append(
    el(
      "div",
      { class: "head" },
      el("span", { class: "type user" }, "用户"),
      block.ts ? el("span", {}, fmtTsTime(block.ts)) : null,
    ),
  );
  const text = String(block.text || "");
  if (text) node.append(el("div", { class: "llm-content" }, text));
}

function toolGroupTitle(block) {
  const tools = Array.isArray(block.tools) ? block.tools : [];
  const names = tools
    .map((row) => `${row.name} ×${Number(row.count) || 0}`)
    .join(", ");
  const failed = Number(block.failed) || 0;
  const running = Number(block.running) || 0;
  const bits = [names || "工具"];
  if (failed) bits.push(`${failed} 失败`);
  if (running) bits.push(`${running} 进行中`);
  return bits.join(" · ");
}

function toolRowsNode(tools) {
  const list = el("div", { class: "trace-tool-list" });
  for (const row of tools || []) {
    const parts = [`${row.name} ×${Number(row.count) || 0}`];
    if (row.ok) parts.push(`成功 ${row.ok}`);
    if (row.failed) parts.push(`失败 ${row.failed}`);
    if (row.running) parts.push(`进行中 ${row.running}`);
    const line = el("div", { class: "tool-brief" }, parts.join("  "));
    if (row.summary) line.append(el("span", {}, `  ${row.summary}`));
    list.append(line);
  }
  if (!list.childNodes.length) list.append(el("div", { class: "hint" }, "无工具"));
  return list;
}

function subagentDetailNode(block) {
  const body = el("div", { class: "trace-subagent-detail" });
  if (block.summary) body.append(el("div", {}, `摘要：${block.summary}`));
  if (block.error)
    body.append(el("div", { class: "hint warn" }, `错误：${block.error}`));
  if (Array.isArray(block.tools) && block.tools.length)
    body.append(toolRowsNode(block.tools));
  if (!body.childNodes.length) body.append(el("div", { class: "hint" }, "无更多详情"));
  return body;
}

/* ---------------- Step 产物树 ---------------- */

/* Lineage of validated step artifacts across the arm's research sessions.
   Branches appear when the Agent used step_rollback or batch_validate, or a
   session started from the node its predecessor handed on. Built for large trees: collapsible subtrees, text filter,
   one shared viewport-clamped tooltip (never clipped by the scroll box),
   inline download on every node with a snapshot. */
function stepTreePanel(detail) {
  const host = el("div", {});
  api(`/api/experiments/${encodeURIComponent(detail.experiment_id)}/steps`)
    .then((payload) => {
      if ((payload.nodes || []).length)
        host.append(stepTreeSection(detail, payload));
    })
    .catch(() => {
      /* no tree for this experiment */
    });
  return host;
}

function stepTreeSection(detail, payload) {
  const nodes = payload.nodes;
  const ids = new Set(nodes.map((node) => node.node_id));
  const byParent = new Map();
  for (const node of nodes) {
    const key =
      node.parent_node_id && ids.has(node.parent_node_id)
        ? node.parent_node_id
        : "";
    if (!byParent.has(key)) byParent.set(key, []);
    byParent.get(key).push(node);
  }
  const state = { collapsed: new Set(), filter: "" };
  const rows = el("div", { class: "step-tree", onscroll: hideStepTip });
  const summary = el("span", { class: "hint flush push-right" });
  const validated = nodes.filter((node) => node.complete_validation).length;

  const haystack = (node) =>
    [
      node.node_id,
      node.session_key,
      node.result_name,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();

  const render = () => {
    hideStepTip();
    rows.innerHTML = "";
    const query = state.filter.trim().toLowerCase();
    let visible = new Set(nodes.map((node) => node.node_id));
    if (query) {
      // Matches plus their ancestors, so hits keep their lineage context.
      const parentOf = new Map(
        nodes.map((node) => [node.node_id, node.parent_node_id]),
      );
      visible = new Set();
      for (const node of nodes) {
        if (!haystack(node).includes(query)) continue;
        let cursor = node.node_id;
        while (cursor && ids.has(cursor) && !visible.has(cursor)) {
          visible.add(cursor);
          cursor = parentOf.get(cursor);
        }
      }
    }
    // Connector-rail layout. Lineage chains dominate this tree (each Step
    // parents on the previous one; measured trees run 16+ ancestors deep with
    // <=6 forks), so columns advance ONLY at forks — a chain renders as a
    // straight vertical rail, and every parent-child edge is drawn explicitly:
    // "tee"/"last" elbows attach fork children, "chain" rails attach an only
    // child to the row above, and open sibling rails run past nested subtrees.
    const walk = (parentKey, guides, forked, inheritedOpen) => {
      const siblings = (byParent.get(parentKey) || []).filter((node) =>
        visible.has(node.node_id),
      );
      siblings.forEach((node, index) => {
        const isLast = index === siblings.length - 1;
        const children = (byParent.get(node.node_id) || []).filter((child) =>
          visible.has(child.node_id),
        );
        // A filter overrides manual collapse: hits must never be hidden.
        const collapsed = !query && state.collapsed.has(node.node_id);
        // Does this node's rail column stay live below its own row? Either a
        // later sibling branch still hangs below (fork siblings), the ancestor
        // fork's rail passes through (inherited along a chain), or the chain
        // itself continues with an only child.
        const open = forked ? !isLast : inheritedOpen;
        const continues = !collapsed && children.length === 1;
        rows.append(
          stepTreeRow(detail, payload, node, {
            guides,
            connector: parentKey
              ? forked
                ? open || continues
                  ? "tee"
                  : "last"
                : open || continues
                  ? "chain"
                  : "chain end"
              : "",
            childCount: children.length,
            collapsed,
            toggle: () => {
              if (state.collapsed.has(node.node_id))
                state.collapsed.delete(node.node_id);
              else state.collapsed.add(node.node_id);
              render();
            },
          }),
        );
        if (!collapsed) {
          const childForked = children.length > 1;
          // A fork opens a new column; this node's column keeps its rail
          // through the nested subtree while `open` (sibling/ancestor rail).
          walk(
            node.node_id,
            childForked && parentKey ? [...guides, open] : guides,
            childForked,
            open,
          );
        }
      });
    };
    walk("", [], false, false);
    if (!rows.children.length)
      rows.append(el("div", { class: "empty" }, "没有命中的节点"));
    const failed = nodes.length - validated;
    summary.textContent = query
      ? `命中 ${visible.size} / ${nodes.length} 节点`
      : `${validated} 已验证${failed ? ` · ${failed} 失败` : ""} · 共 ${nodes.length} 节点`;
  };

  const filterInput = el("input", {
    class: "input step-filter",
    type: "search",
    placeholder: "筛选会话 / 节点 / 结果名…",
    oninput: (event) => {
      state.filter = event.target.value;
      render();
    },
  });
  const toolbar = el(
    "div",
    { class: "step-toolbar" },
    filterInput,
    el(
      "button",
      {
        class: "btn small",
        onclick: () => {
          state.collapsed = new Set();
          render();
        },
      },
      "全部展开",
    ),
    el(
      "button",
      {
        class: "btn small",
        onclick: () => {
          state.collapsed = new Set(
            nodes
              .filter((node) => (byParent.get(node.node_id) || []).length)
              .map((node) => node.node_id),
          );
          render();
        },
      },
      "全部折叠",
    ),
    summary,
  );
  render();
  return el(
    "div",
    { class: "panel section-gap" },
    el("h4", {}, "Step 产物树"),
    toolbar,
    rows,
  );
}

function stepTreeRow(
  detail,
  payload,
  node,
  { guides, connector, childCount, collapsed, toggle },
) {
  const metrics = node.metrics || {};
  const failed = node.status === "failed";
  const zipUrl = `/api/experiments/${encodeURIComponent(detail.experiment_id)}/steps/${encodeURIComponent(node.node_id)}/source.zip`;
  const badges = [];
  if (node.is_current)
    badges.push(
      el("span", { class: "badge state-running_session" }, "当前位置"),
    );
  if (node.frozen)
    badges.push(el("span", { class: "badge state-completed" }, "已冻结"));
  if (failed) badges.push(el("span", { class: "badge state-failed" }, "失败"));
  const actions = el("span", { class: "step-actions" });
  if (node.has_snapshot) {
    actions.append(
      el(
        "a",
        {
          class: "btn small",
          href: zipUrl,
          title: "下载该版本完整源代码与验证明细",
          onclick: (event) => event.stopPropagation(),
        },
        "下载",
      ),
    );
  }
  const row = el(
    "div",
    {
      class: `step-node${failed ? " failed" : ""}`,
      onclick: () => {
        hideStepTip();
        openStepNodeModal(detail, payload, node);
      },
      onmouseenter: (event) => showStepTip(detail, node, event.currentTarget),
      onmouseleave: hideStepTip,
    },
    // Lineage rails: ancestor columns (open = a later sibling still hangs
    // below), then this row's own connector to its parent.
    ...guides.map((open) =>
      el("span", { class: `step-rail${open ? " open" : ""}` }),
    ),
    connector ? el("span", { class: `step-rail ${connector}` }) : null,
    childCount
      ? el(
          "button",
          {
            class: "step-toggle",
            title: collapsed ? `展开 ${childCount} 个子节点` : "折叠子树",
            onclick: (event) => {
              event.stopPropagation();
              hideStepTip();
              toggle();
            },
          },
          collapsed ? "▸" : "▾",
        )
      : el("span", { class: "step-toggle leaf" }, "·"),
    el(
      "span",
      { class: "step-label" },
      `${node.session_key || "—"} · ${node.result_name || node.node_id}`,
    ),
    collapsed ? el("span", { class: "step-chip" }, `+${childCount}`) : null,
    Number.isFinite(metrics.total_return)
      ? el(
          "span",
          { class: `step-chip ${signCls(metrics.total_return)}` },
          fmtPct(metrics.total_return),
        )
      : null,
    Number.isFinite(metrics.sharpe)
      ? el(
          "span",
          { class: "step-chip" },
          `S ${Number(metrics.sharpe).toFixed(2)}`,
        )
      : null,
    ...badges,
    el("span", { class: "step-time" }, fmtTs(node.created_at)),
    actions,
  );
  return row;
}

/* One shared fixed-position tooltip: immune to the tree's overflow clipping
   and cheaper than a hidden card per row on large trees. pointer-events:none
   so it never steals hover from the rows underneath. */
function showStepTip(detail, node, row) {
  let tip = document.getElementById("step-tip");
  if (!tip) {
    tip = el("div", { id: "step-tip" });
    document.body.append(tip);
  }
  const m = node.metrics || {};
  const line = (k, v) =>
    el(
      "div",
      { class: "step-tip-line" },
      el("span", { class: "k" }, `${k}：`),
      String(v),
    );
  tip.innerHTML = "";
  tip.append(
    el("div", { class: "step-tip-title" }, node.node_id),
    line("会话", node.session_key || "—"),
    line("验证收益", fmtPct(m.total_return)),
    line("多头收益", fmtPct(m.long_return)),
    line(
      "Sharpe",
      m.sharpe === undefined || m.sharpe === null
        ? "—"
        : Number(m.sharpe).toFixed(2),
    ),
    line("最大回撤", fmtPct(m.max_drawdown)),
    line("记录于", fmtTs(node.created_at)),
    node.frozen ? line("冻结", "本节点是冻结产物") : null,
    node.status === "failed" ? line("失败原因", node.error || "—") : null,
    node.has_snapshot ? null : line("快照", "无（失败尝试不留产物）"),
  );
  tip.style.display = "block";
  const rect = row.getBoundingClientRect();
  const margin = 8;
  tip.style.left = `${Math.max(margin, Math.min(rect.left + 24, window.innerWidth - tip.offsetWidth - margin))}px`;
  let top = rect.bottom + 4;
  if (top + tip.offsetHeight > window.innerHeight - margin)
    top = rect.top - tip.offsetHeight - 4;
  tip.style.top = `${Math.max(margin, top)}px`;
}

function hideStepTip() {
  const tip = document.getElementById("step-tip");
  if (tip) tip.style.display = "none";
}

function openStepNodeModal(detail, payload, node) {
  const m = node.metrics || {};
  const zipUrl = `/api/experiments/${encodeURIComponent(detail.experiment_id)}/steps/${encodeURIComponent(node.node_id)}/source.zip`;
  const body = el(
    "div",
    {},
    el(
      "table",
      { class: "kv" },
      kvRow("节点", node.node_id),
      kvRow("会话", node.session_key || "—"),
      kvRow(
        "验证收益",
        el("span", { class: signCls(m.total_return) }, fmtPct(m.total_return)),
      ),
      kvRow("多头收益", fmtPct(m.long_return)),
      kvRow(
        "Sharpe",
        m.sharpe === undefined || m.sharpe === null
          ? "—"
          : Number(m.sharpe).toFixed(2),
      ),
      kvRow("最大回撤", fmtPct(m.max_drawdown)),
      kvRow("记录时间", fmtTs(node.created_at)),
      node.frozen ? kvRow("冻结", "本节点是冻结产物") : null,
      node.status === "failed" ? kvRow("失败原因", node.error || "—") : null,
      kvRow("附件", (node.attachments || []).join("、") || "—"),
      kvRow("revision", el("code", {}, String(node.strategy_ref || "—"))),
    ),
    node.has_snapshot
      ? null
      : el("p", { class: "hint" }, "失败尝试不保存产物快照。"),
  );
  const buttons = [el("button", { class: "btn", onclick: closeModal }, "关闭")];
  if (node.has_snapshot)
    buttons.push(el("a", { class: "btn", href: zipUrl }, "下载源码 + 结果"));
  showModal("Step 节点详情", body, buttons);
}

function kvRow(key, value) {
  return el("tr", {}, el("td", {}, key), el("td", {}, value));
}

/* Per July-June year breakdown of one replay (stats.sub_windows): the same
   figures as the whole window, one row per year. "部分" marks a year the
   window does not span end to end; 超额 is against 沪深300. */
function subWindowSection(title, rows) {
  if (!Array.isArray(rows) || !rows.length) return null;
  return el(
    "div",
    { class: "section-gap" },
    el("h4", { class: "subsection-title" }, title),
    dataTable(
      [
        { label: "年度" },
        { label: "收益", num: true, title: "年度开盘权益起算的区间收益" },
        { label: "超额", num: true, title: "相对沪深300的超额收益" },
        { label: "Sharpe", num: true, title: "年度内日收益的年化 Sharpe" },
        { label: "回撤", num: true, title: "年度内峰谷回撤" },
        { label: "换手", num: true, title: "成交名义额 / 初始资金" },
        { label: "笔数", num: true, title: "已实现平仓笔数" },
        { label: "交易日", num: true },
      ],
      rows.map((row) => [
        [
          `${row.label || "—"}${row.partial ? " ·部分" : ""}`,
          el(
            "span",
            { class: "mode-note" },
            ` ${fmtPeriodRange(`${row.start}..${row.end}`)}`,
          ),
        ],
        { value: fmtPct(row.return), cls: signCls(row.return) },
        { value: fmtPct(row.excess_return), cls: signCls(row.excess_return) },
        { value: fmtSharpe(row.sharpe), cls: signCls(row.sharpe) },
        fmtPct(row.max_drawdown),
        fmtSharpe(row.turnover),
        row.trade_count,
        row.trade_days,
      ]),
    ),
  );
}

/* Barra-lite style validation card: CSI300 alpha/beta regression + holdings
   style tilts (signed percentile deviation, [-1,1]) + SW industry weights. */
function styleCard(expId, result) {
  const host = el(
    "div",
    { class: "section-gap" },
    el("h4", { class: "subsection-title" }, "风格暴露与基准归因（Barra-lite）"),
    el("div", { class: "hint" }, "加载中…"),
  );
  api(
    `/api/experiments/${encodeURIComponent(expId)}/results/${encodeURIComponent(result)}/style`,
  )
    .then((payload) => {
      host.querySelector(".hint").remove();
      const reg = payload.benchmark_regression || {};
      const style = payload.style || {};
      host.append(
        statTilesRow([
          {
            label: "β（vs 沪深300）",
            value:
              reg.beta === null || reg.beta === undefined
                ? "—"
                : Number(reg.beta).toFixed(2),
          },
          {
            label: "年化 α",
            value: fmtPct(reg.alpha_annualized),
            cls: signCls(reg.alpha_annualized),
          },
          {
            label: "R²",
            value:
              reg.r2 === null || reg.r2 === undefined
                ? "—"
                : Number(reg.r2).toFixed(2),
          },
          { label: "样本天数", value: String(reg.n_days ?? "—") },
        ]),
      );
      const regressionReasons = {
        benchmark_unavailable:
          "回放槽中没有可用的沪深300同窗数据，基准回归为空。",
        insufficient_overlapping_days:
          "与沪深300重叠的交易日不足 8 天，β、α 与 R² 不计算。",
        benchmark_variance_zero:
          "同窗沪深300收益没有可回归的波动，β、α 与 R² 不计算。",
      };
      if (!reg.available && regressionReasons[reg.reason]) {
        host.append(
          el("div", { class: "hint" }, regressionReasons[reg.reason]),
        );
      }
      const tilts = style.tilts;
      if (style.available && tilts) {
        const rows = [
          { label: "市值（+大盘 / −小盘）", value: tilts.size },
          { label: "PB（+高估值 / −低估值）", value: tilts.pb },
          { label: "换手（+高换手 / −低换手）", value: tilts.turnover },
        ];
        const list = el("div", { class: "tilts section-gap" });
        for (const row of rows) {
          const pct = Math.min(Math.abs(row.value), 1) * 50;
          const side = row.value >= 0 ? "left:50%" : `left:${50 - pct}%`;
          list.append(
            el(
              "div",
              { class: "tilt-row" },
              el("span", { class: "tilt-label" }, row.label),
              el(
                "span",
                { class: "tilt-bar" },
                el("span", {
                  class: "tilt-fill",
                  style: `${side};width:${pct}%`,
                }),
              ),
              el(
                "span",
                { class: "tilt-value" },
                (row.value >= 0 ? "+" : "") + Number(row.value).toFixed(2),
              ),
            ),
          );
        }
        host.append(list);
        host.append(
          el(
            "div",
            { class: "hint" },
            `持仓覆盖 ${style.days} 个交易日 ｜ 日均 ${style.avg_names} 只 ｜ 日均多头 ${fmtAmount(style.avg_long_gross)}`,
          ),
        );
        if ((style.industries || []).length) {
          host.append(
            el(
              "div",
              { class: "hint" },
              "行业净权重（申万一级）：" +
                style.industries
                  .map((i) => `${i.name} ${(i.weight * 100).toFixed(0)}%`)
                  .join(" ｜ "),
            ),
          );
        }
      } else {
        const styleReasons = {
          style_columns_unavailable:
            "回放槽缺少市值、PB 或换手截面，风格暴露为空。",
          no_holdings: "该回放没有持仓，风格暴露为空。",
          no_valued_holdings:
            "该回放的持仓没有可用收盘价，风格暴露为空。",
        };
        host.append(
          el(
            "div",
            { class: "hint" },
            styleReasons[style.reason] || "该回放没有可计算的风格暴露。",
          ),
        );
      }
    })
    .catch((error) => {
      const missing = /没有已落盘|404/.test(error.message);
      host.append(
        el(
          "div",
          { class: "hint" },
          missing
            ? "该运行未落盘风格归因数据，无风格分析可展示。"
            : `风格分析加载失败：${error.message}`,
        ),
      );
    });
  return host;
}

/* Render scheduled and matched ISO timestamps in Asia/Shanghai. */
function fmtOrderCell(key, value) {
  if (
    ["decision_time", "execute_at", "matched_at"].includes(key) &&
    typeof value === "string" &&
    value.includes("T")
  ) {
    return fmtTsTime(value) || value;
  }
  return value;
}

// [field, header, numeric column]
const ORDER_TABLE_COLUMNS = [
  ["matched_at", "成交时间"],
  ["execute_at", "计划时间"],
  ["symbol", "代码"],
  ["action", "动作"],
  ["quantity", "数量", true],
  ["price", "价格", true],
  ["status", "状态"],
  ["reason", "拒单原因"],
];

/* Transaction details of one ledger-named replay result: stats tiles, per-day
   amount chart, order table and CSV export. */
function ordersNode(experimentId, result) {
  const base = `/api/experiments/${encodeURIComponent(experimentId)}/results/${encodeURIComponent(result)}`;
  const body = el("div", {}, el("div", { class: "loading" }, "加载交易明细…"));
  api(`${base}/orders`)
    .then((data) => {
      const rows = data.rows || [];
      const stats = data.stats || {};
      const byAction = stats.by_action || {};
      body.replaceChildren(
        el(
          "div",
          { class: "control-bar" },
          el("span", { class: "mode-note" }, data.result),
          el("span", { class: "spacer" }),
          el("a", { class: "btn small", href: `${base}/orders.csv` }, "⬇ 导出 CSV"),
        ),
        el(
          "div",
          { class: "section-gap" },
          statTilesRow([
            {
              label: "订单 / 成交 / 拒单",
              value: `${stats.orders} / ${stats.filled} / ${stats.rejected}`,
            },
            { label: "成交额", value: fmtAmount(stats.turnover) },
            {
              label: "买 / 卖",
              value: `${byAction.buy || 0} / ${byAction.sell || 0}`,
            },
          ]),
        ),
      );
      const daily = (stats.daily || []).map((d) => ({
        label: String(d.trade_date).slice(4),
        value: d.amount,
      }));
      if (daily.length)
        body.append(
          el("h4", { class: "subsection-title section-gap" }, "逐日成交金额"),
          singleSeriesBarChart(daily, { fmt: fmtAmount, width: 980, height: 200 }),
        );
      if (Object.keys(stats.reject_reasons || {}).length)
        body.append(
          el(
            "div",
            { class: "stats-chips section-gap" },
            ...Object.entries(stats.reject_reasons).map(([reason, count]) =>
              el("span", { class: "stat-chip" }, `拒单 ${reason} ×${count}`),
            ),
          ),
        );
      if (!rows.length) return;
      const shown = rows.slice(0, 80);
      body.append(
        dataTable(
          ORDER_TABLE_COLUMNS.map(([, label, num]) => ({ label, num })),
          shown.map((row) =>
            ORDER_TABLE_COLUMNS.map(([key]) =>
              key === "price" ? fmtPrice(row[key]) : fmtOrderCell(key, row[key]),
            ),
          ),
          { box: "limit section-gap" },
        ),
      );
      if (data.row_count > shown.length)
        body.append(
          el(
            "div",
            { class: "hint" },
            `表格显示前 ${shown.length} 条，共 ${data.row_count} 条 —— 完整明细请导出 CSV。`,
          ),
        );
    })
    .catch((error) => {
      body.replaceChildren(
        el("div", { class: "hint" }, `无交易明细：${error.message}`),
      );
    });
  return body;
}

/* ---------------- 运行记忆 ----------------
   One stable two-pane layout. The left pane is the whole catalogue — the
   curated 精选库 above, the 毕业层候选 the tier admits below — and the right
   pane is a single viewer/editor surface with a fixed head, toolbar and body,
   so selecting anything swaps only what those three hosts contain: the page
   header, the notice and both pane widths never move.

   The library is a tracked repository directory that an experiment snapshots
   when it is created, so a write here reaches experiments created afterwards,
   and the researcher still commits it. Who may write is settled by how the console is
   reached (loopback bind, or the edge's login gate), not by this page.

   Inside an experiment, 已挂载记忆 stays a projection of THAT experiment's run
   manifests: what it mounted then, not what the tier admits now. */

let memoryView = null;

function fmtBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes)) return "—";
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
}

function memoryLibraryPath() {
  const curated = (memoryView && memoryView.payload.curated) || {};
  return curated.library || "configs/operating_memory";
}

function sameSelection(left, right) {
  if (!left || !right || left.kind !== right.kind) return false;
  return left.kind === "curated"
    ? left.name === right.name
    : left.experiment_id === right.experiment_id && left.skill === right.skill;
}

async function renderMemoryPage() {
  memoryView = null;
  $main.innerHTML = '<div class="loading">加载运行记忆…</div>';
  $topbarRight.replaceChildren();
  let payload;
  try {
    payload = await api("/api/memory");
  } catch (error) {
    $main.replaceChildren(
      el("div", { class: "empty" }, `加载失败：${error.message}`),
    );
    return;
  }
  memoryView = {
    payload,
    filter: "",
    selection: null,
    entry: null,
    loading: false,
    mode: "view", // view | edit | create | promote
    draft: null,
    draftName: "",
    source: null,
    dirty: false,
    issueExperiment: "",
    issueResolved: false,
    countHost: el("span", {}, ""),
    listHost: el("div", { class: "session-list" }),
    candidateHost: el("div", { class: "session-list" }),
    headHost: el("div", { class: "memory-pane-head" }),
    toolbarHost: el("div", { class: "control-bar memory-toolbar" }),
    bodyHost: el("div", { class: "memory-pane-body" }),
    issueCountHost: el("span", { class: "issue-count" }, ""),
    issuesHost: el("div", { class: "issue-list" }),
  };
  $main.replaceChildren(
    el(
      "div",
      { id: "memory-page" },
      el(
        "div",
        { class: "page-head" },
        el("h2", {}, "运行记忆"),
        el(
          "div",
          { class: "sub" },
          `默认挂载模式 ${payload.default_mode || "—"} ｜ 改动作用于此后创建的实验，由研究者提交`,
        ),
      ),
      memorySection(
        null,
        "记忆条目",
        null,
        el(
          "div",
          { class: "detail" },
          memoryNavPanel(),
          el(
            "div",
            { class: "panel memory-pane" },
            memoryView.headHost,
            memoryView.toolbarHost,
            memoryView.bodyHost,
          ),
        ),
      ),
      memorySection(
        "memory-issues",
        "问题反馈",
        "会话报告的缺陷；处置用 scripts/experiments/resolve_issue.py 记录",
        el("div", { class: "panel" }, issueFilterBar(), memoryView.issuesHost),
      ),
    ),
  );
  renderMemoryList();
  renderMemoryCandidates();
  renderMemoryPane();
  renderIssueReports();
}

/* One pattern for every section on this page: a heading with an optional
   one-line note, then the panels. */
function memorySection(id, title, note, ...panels) {
  return el(
    "section",
    { class: "memory-section", id },
    el(
      "div",
      { class: "memory-section-head" },
      el("h3", {}, title),
      note ? el("div", { class: "sub" }, note) : null,
    ),
    ...panels,
  );
}

/* Operators' inbox for defects the sessions themselves noticed: `report_issue`
   lines from every experiment's ledgers, listed read-only, evidence folded. */
const ISSUE_CATEGORY_LABELS = {
  tool_output: "工具输出",
  environment: "环境",
  data: "数据",
  docs: "文档",
  other: "其他",
};

/* The researcher's verdict, recorded from the shell into the same log. */
const ISSUE_OUTCOME_LABELS = {
  fixed: "已修复",
  not_a_defect: "非缺陷",
  accepted_limitation: "接受的限制",
};

/* The list is one bounded newest-first page across every experiment, so
   narrowing to a single experiment is how an older report is reached at all.
   The options are the experiment directories the memory bundle already listed
   — the same set the reports themselves are read from. */
function issueFilterBar() {
  const rows = (memoryView.payload.graduated || {}).experiments || [];
  const select = el(
    "select",
    {
      onchange: (event) => {
        memoryView.issueExperiment = event.target.value;
        renderIssueReports();
      },
    },
    el("option", { value: "" }, "全部实验"),
    ...rows.map((row) =>
      el("option", { value: row.experiment_id }, row.experiment_id),
    ),
  );
  const resolved = el("input", { type: "checkbox" });
  if (memoryView.issueResolved) resolved.checked = true;
  resolved.addEventListener("change", (event) => {
    memoryView.issueResolved = event.target.checked;
    renderIssueReports();
  });
  return el(
    "div",
    { class: "control-bar" },
    el("label", { class: "issue-filter" }, el("span", {}, "实验"), select),
    el("label", { class: "issue-filter" }, resolved, el("span", {}, "显示已处置")),
    el("span", { class: "spacer" }),
    memoryView.issueCountHost,
  );
}

/* One card per report, read top to bottom: the category badge opens the
   summary line so every card's prose keeps the same left edge, the attribution
   follows as one quiet line, and the raw evidence stays folded behind its own
   small disclosure instead of making the whole card a click target. A resolved
   report keeps its own text untouched and gains the outcome badge plus the
   researcher's note — the two are read together or the card lies. */
function issueReportRow(report) {
  const resolved = Boolean(report.outcome);
  return el(
    "article",
    { class: resolved ? "issue-report resolved" : "issue-report" },
    el(
      "p",
      { class: "issue-line" },
      resolved
        ? el(
            "span",
            { class: "badge state-completed" },
            ISSUE_OUTCOME_LABELS[report.outcome] || report.outcome,
          )
        : null,
      el(
        "span",
        { class: "badge kind" },
        ISSUE_CATEGORY_LABELS[report.category] || report.category || "—",
      ),
      report.summary || "—",
    ),
    el(
      "div",
      { class: "issue-meta" },
      issueMetaItem("时间", fmtTs(report.recorded_at)),
      issueMetaItem("实验", report.experiment_id || "—"),
      issueMetaItem("会话", report.session_label || "—"),
    ),
    resolved
      ? el(
          "p",
          { class: "issue-resolution" },
          el("span", { class: "k" }, `处置于 ${fmtTs(report.resolved_at)}`),
          report.resolution || "—",
        )
      : null,
    el(
      "details",
      { class: "issue-evidence" },
      el("summary", {}, "证据"),
      el("pre", { class: "issue-evidence-body" }, report.evidence || "—"),
    ),
  );
}

/* Label and value are one unbreakable run: the metadata line wraps between
   pairs, never inside one, and a pathological value ends in an ellipsis with
   the full text still on the element. */
function issueMetaItem(label, value) {
  return el(
    "span",
    { class: "issue-meta-item", title: `${label} ${value}` },
    el("span", { class: "k" }, label),
    el("span", { class: "v" }, value),
  );
}

async function renderIssueReports() {
  const host = memoryView.issuesHost;
  const experiment = memoryView.issueExperiment;
  const withResolved = memoryView.issueResolved;
  // A filter change and a page leave both land here while a fetch is open.
  const stale = () =>
    !memoryView ||
    memoryView.issuesHost !== host ||
    memoryView.issueExperiment !== experiment ||
    memoryView.issueResolved !== withResolved;
  host.replaceChildren(el("div", { class: "loading" }, "加载问题反馈…"));
  memoryView.issueCountHost.textContent = "";
  let payload;
  try {
    const params = new URLSearchParams();
    if (experiment) params.set("experiment_id", experiment);
    if (withResolved) params.set("include_resolved", "true");
    const query = params.toString() ? `?${params}` : "";
    payload = await api(`/api/issue-reports${query}`);
  } catch (error) {
    if (stale()) return;
    host.replaceChildren(
      el("div", { class: "hint warn" }, `加载失败：${error.message}`),
    );
    return;
  }
  if (stale()) return;
  const reports = payload.reports || [];
  const total = payload.total || 0;
  const resolved = payload.resolved || 0;
  // The resolved count rides along in both modes: without it, a page whose
  // every report has been answered reads exactly like one that never had any.
  const listed =
    total > reports.length
      ? `最近 ${reports.length} 条，共 ${total} 条`
      : `${total} 条`;
  memoryView.issueCountHost.textContent = withResolved
    ? `共 ${listed}，其中已处置 ${resolved} 条`
    : `未处置 ${listed}，已处置 ${resolved} 条`;
  const nodes = (payload.unreadable || []).map((item) =>
    el("div", { class: "hint warn" }, `${item.experiment_id}：${item.error}`),
  );
  if (reports.length) nodes.push(...reports.map(issueReportRow));
  else
    nodes.push(
      el(
        "div",
        { class: "empty compact" },
        resolved && !withResolved
          ? `没有未处置的报告（已处置 ${resolved} 条，勾选上方可显示）`
          : "会话没有报告过问题",
      ),
    );
  host.replaceChildren(...nodes);
}

function memoryNavPanel() {
  const curated = memoryView.payload.curated || {};
  memoryView.countHost.textContent = String((curated.entries || []).length);
  const filter = el("input", {
    type: "text",
    placeholder: "过滤条目…",
    oninput: (event) => {
      memoryView.filter = event.target.value;
      renderMemoryList();
      renderMemoryCandidates();
    },
  });
  return el(
    "div",
    { class: "panel memory-nav" },
    el("h4", {}, "精选库（", memoryView.countHost, " 条）"),
    el(
      "div",
      { class: "control-bar" },
      el(
        "button",
        { class: "btn small primary", onclick: () => startCuratedCreate() },
        "新建",
      ),
      el("div", { class: "field memory-filter" }, filter),
    ),
    memoryView.listHost,
    el("h4", { class: "section-gap" }, "毕业层候选"),
    memoryView.candidateHost,
  );
}

function memoryFilterHit(...parts) {
  const needle = memoryView.filter.trim().toLowerCase();
  return !needle || parts.join(" ").toLowerCase().includes(needle);
}

function memoryNavItem(label, note, selection) {
  return el(
    "div",
    {
      class: `session-item${sameSelection(memoryView.selection, selection) ? " selected" : ""}`,
      title: label,
      onclick: () => selectMemoryItem(selection),
    },
    el("span", { class: "label" }, label),
    el("span", { class: "ret" }, note),
  );
}

function renderMemoryList() {
  const curated = memoryView.payload.curated || {};
  const entries = curated.entries || [];
  const shown = entries.filter((entry) =>
    memoryFilterHit(entry.name, entry.title || "", entry.summary || ""),
  );
  const nodes = shown.map((entry) =>
    memoryNavItem(
      entry.name,
      fmtBytes(entry.bytes),
      { kind: "curated", name: entry.name },
    ),
  );
  if (curated.error)
    nodes.unshift(
      el("div", { class: "hint warn" }, `精选库不可读：${curated.error}`),
    );
  if (!shown.length)
    nodes.push(
      el(
        "div",
        { class: "empty compact" },
        entries.length ? "没有匹配的条目" : "精选库为空",
      ),
    );
  memoryView.listHost.replaceChildren(...nodes);
}

/* Admitted candidates are selectable, and every other experiment stays visible
   in one collapsed muted block — because "not offered" and "not there" are
   different answers. */
function renderMemoryCandidates() {
  const tier = memoryView.payload.graduated || {};
  const rows = tier.experiments || [];
  const listed = rows.filter(
    (row) => row.admitted === true && (row.entries || []).length,
  );
  const aside = rows.filter((row) => !listed.includes(row));
  const nodes = [];
  if (tier.error)
    nodes.push(
      el("div", { class: "hint warn" }, `毕业层不可解析：${tier.error}`),
    );
  let shown = 0;
  for (const row of listed) {
    const skills = (row.entries || []).filter((skill) =>
      memoryFilterHit(skill, row.experiment_id),
    );
    if (!skills.length) continue;
    shown += skills.length;
    nodes.push(el("div", { class: "epoch-head" }, row.experiment_id));
    for (const skill of skills)
      nodes.push(
        memoryNavItem(
          skill,
          "候选",
          { kind: "candidate", experiment_id: row.experiment_id, skill },
        ),
      );
  }
  if (!shown)
    nodes.push(
      el(
        "div",
        { class: "empty compact" },
        listed.length ? "没有匹配的候选" : "没有准入的候选",
      ),
    );
  if (aside.length)
    nodes.push(
      el(
        "details",
        { class: "memory-aside" },
        el("summary", {}, `其他实验（${aside.length}）`),
        ...aside.map((row) =>
          el(
            "div",
            { class: "memory-aside-row" },
            el("span", { class: "label" }, row.experiment_id),
            candidateAsideReason(row),
          ),
        ),
      ),
    );
  memoryView.candidateHost.replaceChildren(...nodes);
}

/* `admitted === null` means the tier itself could not be resolved, which is
   not the same answer as "contributes nothing". */
function candidateAsideReason(row) {
  if (row.error) return el("span", { class: "hint warn" }, row.error);
  if (row.admitted === null || row.admitted === undefined)
    return el("span", { class: "hint warn" }, "无法解析");
  if (!row.verdict) return el("span", { class: "hint" }, "无裁决");
  if (row.verdict !== "graduated") return verdictBadge({ status: row.verdict });
  return el("span", { class: "hint" }, "无已发布 skill 条目");
}

/* In-page moves are guarded; leaving the 运行记忆 route entirely drops the
   draft, as every other unsubmitted editor in the console does. */
function guardUnsavedMemory(proceed) {
  if (!memoryView || !memoryView.dirty) {
    proceed();
    return;
  }
  showModal(
    "放弃未保存的修改？",
    el("div", {}, el("p", {}, "当前编辑还没有保存，继续会丢弃这些修改。")),
    [
      el("button", { class: "btn", onclick: closeModal }, "继续编辑"),
      el(
        "button",
        {
          class: "btn danger",
          onclick: () => {
            closeModal();
            proceed();
          },
        },
        "放弃修改",
      ),
    ],
  );
}

function resetCuratedDraft() {
  memoryView.mode = "view";
  memoryView.draft = null;
  memoryView.draftName = "";
  memoryView.source = null;
  memoryView.dirty = false;
}

function selectMemoryItem(selection) {
  guardUnsavedMemory(() => {
    resetCuratedDraft();
    memoryView.selection = selection;
    memoryView.entry = null;
    memoryView.loading = true;
    renderMemoryList();
    renderMemoryCandidates();
    renderMemoryPane();
    loadMemorySelection(selection);
  });
}

async function loadMemorySelection(selection) {
  let entry;
  try {
    entry =
      selection.kind === "curated"
        ? await api(`/api/memory/curated/${encodeURIComponent(selection.name)}`)
        : await api(
            `/api/memory/graduated/${encodeURIComponent(selection.experiment_id)}/${encodeURIComponent(selection.skill)}`,
          );
  } catch (error) {
    entry = { error: error.message };
  }
  if (!memoryView || !sameSelection(memoryView.selection, selection)) return;
  memoryView.entry = entry;
  memoryView.loading = false;
  renderMemoryPane();
}

/* Three fixed hosts, always all three: only their contents change, so the
   toolbar row never moves and the body never collapses under a load. */
function renderMemoryPane() {
  const view = memoryPaneView();
  memoryView.headHost.replaceChildren(
    el("h4", { class: "memory-pane-title" }, view.title),
    el("div", { class: "hint memory-pane-meta" }, view.meta || ""),
  );
  memoryView.toolbarHost.replaceChildren(...view.buttons);
  memoryView.bodyHost.replaceChildren(...view.body);
}

function memoryPaneView() {
  if (memoryView.mode === "create" || memoryView.mode === "promote")
    return curatedFormView();
  const selection = memoryView.selection;
  if (!selection)
    return {
      title: "运行记忆条目",
      meta: "",
      buttons: [],
      body: [
        el(
          "div",
          { class: "empty" },
          "从左侧选择精选库条目或毕业层候选，查看 SKILL.md 原文",
        ),
      ],
    };
  const label = selection.kind === "curated" ? selection.name : selection.skill;
  if (memoryView.loading || !memoryView.entry)
    return {
      title: label,
      meta: "",
      buttons: [],
      body: [el("div", { class: "memory-skeleton" }, "加载条目…")],
    };
  const entry = memoryView.entry;
  if (entry.error)
    return {
      title: label,
      meta: "",
      buttons: [],
      body: [el("div", { class: "hint warn" }, `无法读取：${entry.error}`)],
    };
  return selection.kind === "curated"
    ? curatedEntryView(entry)
    : candidateEntryView(selection, entry);
}

function skillMeta(entry, tail) {
  return `${entry.name} ｜ ${fmtBytes(entry.bytes)} ｜ ${entry.files ?? 1} 个文件 ｜ ${tail}`;
}

function curatedEntryView(entry) {
  const mounted = (memoryView.payload.curated || {}).source || "curated";
  const meta = skillMeta(entry, `挂载为 memory/${mounted}/${entry.name}/`);
  if (memoryView.mode === "edit")
    return {
      title: entry.title || entry.name,
      meta,
      buttons: [
        el(
          "button",
          { class: "btn small primary", onclick: () => saveCuratedEntry() },
          "保存",
        ),
        el(
          "button",
          { class: "btn small", onclick: () => cancelCuratedDraft() },
          "取消",
        ),
      ],
      body: curatedEditorBody(entry),
    };
  return {
    title: entry.title || entry.name,
    meta,
    buttons: [
      el(
        "button",
        { class: "btn small", onclick: () => startCuratedEdit(entry) },
        "编辑",
      ),
      el("span", { class: "spacer" }),
      el(
        "button",
        {
          class: "btn small danger",
          onclick: () => confirmDeleteCuratedEntry(entry.name),
        },
        "删除",
      ),
    ],
    body: [
      entry.summary ? el("div", { class: "hint" }, entry.summary) : null,
      el("pre", { class: "code-view skill-body" }, entry.content || ""),
    ].filter(Boolean),
  };
}

function candidateEntryView(selection, entry) {
  return {
    title: entry.title || entry.name,
    meta: skillMeta(entry, `毕业层 ${selection.experiment_id} 的 skill，只读`),
    buttons: [
      el(
        "button",
        {
          class: "btn small primary",
          onclick: () =>
            startCuratedPromotion(selection.experiment_id, selection.skill),
        },
        "晋升到精选库",
      ),
    ],
    body: [
      entry.summary ? el("div", { class: "hint" }, entry.summary) : null,
      el("pre", { class: "code-view skill-body" }, entry.content || ""),
    ].filter(Boolean),
  };
}

function curatedEditorBody(entry) {
  const editor = el("textarea", {
    class: "directive skill-editor",
    oninput: () => {
      memoryView.draft = editor.value;
      memoryView.dirty = editor.value !== entry.content;
    },
  });
  editor.value = memoryView.draft ?? entry.content ?? "";
  return [
    el(
      "div",
      { class: "hint" },
      `保存到 ${memoryLibraryPath()}/${entry.name}/SKILL.md`,
    ),
    editor,
  ];
}

function curatedFormView() {
  const promoting = memoryView.mode === "promote";
  const nameInput = el("input", {
    type: "text",
    placeholder: "kebab-case 条目名",
    oninput: () => {
      memoryView.draftName = nameInput.value;
      memoryView.dirty = true;
    },
  });
  nameInput.value = memoryView.draftName || "";
  const body = [
    promoting
      ? el(
          "div",
          { class: "hint" },
          `整项复制实验 ${memoryView.source.experiment_id} 的 skill ${memoryView.source.skill}（含 scripts/ 与 references/）`,
        )
      : null,
    el("div", { class: "field" }, el("label", {}, "条目名"), nameInput),
  ].filter(Boolean);
  if (!promoting) {
    const editor = el("textarea", {
      class: "directive skill-editor",
      oninput: () => {
        memoryView.draft = editor.value;
        memoryView.dirty = true;
      },
    });
    editor.value = memoryView.draft || "";
    body.push(el("div", { class: "field" }, el("label", {}, "SKILL.md"), editor));
  }
  return {
    title: promoting ? "晋升到精选库" : "新建精选条目",
    meta: promoting ? "" : `写入 ${memoryLibraryPath()}/`,
    buttons: [
      el(
        "button",
        { class: "btn small primary", onclick: () => submitCuratedCreate() },
        promoting ? "晋升" : "创建",
      ),
      el("button", { class: "btn small", onclick: () => cancelCuratedDraft() }, "取消"),
    ],
    body,
  };
}

function cancelCuratedDraft() {
  guardUnsavedMemory(() => {
    resetCuratedDraft();
    renderMemoryPane();
  });
}

function startCuratedEdit(entry) {
  memoryView.mode = "edit";
  memoryView.draft = entry.content || "";
  memoryView.dirty = false;
  renderMemoryPane();
}

function startCuratedCreate() {
  guardUnsavedMemory(() => {
    resetCuratedDraft();
    memoryView.mode = "create";
    memoryView.selection = null;
    memoryView.entry = null;
    renderMemoryList();
    renderMemoryCandidates();
    renderMemoryPane();
  });
}

/* Prefill from an admitted candidate: the body is copied server-side
   (scripts/ and references/ included), so the form only names the entry. */
function startCuratedPromotion(experimentId, skill) {
  guardUnsavedMemory(() => {
    resetCuratedDraft();
    memoryView.mode = "promote";
    memoryView.source = { experiment_id: experimentId, skill };
    memoryView.draftName = skill;
    renderMemoryPane();
  });
}

async function submitCuratedCreate() {
  const name = String(memoryView.draftName || "").trim();
  if (!name) {
    toast("请填写条目名", true);
    return;
  }
  const promoting = memoryView.mode === "promote";
  try {
    // Both calls stay written out: the client/server route contract is checked
    // by reading these literals out of app.js.
    applyCuratedResult(
      promoting
        ? await api(`/api/memory/curated/${encodeURIComponent(name)}/promote`, {
            method: "POST",
            body: JSON.stringify({
              experiment_id: memoryView.source.experiment_id,
              skill: memoryView.source.skill,
            }),
          })
        : await api("/api/memory/curated", {
            method: "POST",
            body: JSON.stringify({ name, content: memoryView.draft || "" }),
          }),
    );
  } catch (error) {
    toast(`${promoting ? "晋升" : "新建"}失败：${error.message}`, true);
  }
}

async function saveCuratedEntry() {
  const name = memoryView.selection.name;
  try {
    applyCuratedResult(
      await api(`/api/memory/curated/${encodeURIComponent(name)}`, {
        method: "PUT",
        body: JSON.stringify({ content: memoryView.draft || "" }),
      }),
    );
  } catch (error) {
    toast(`保存失败：${error.message}`, true);
  }
}

function confirmDeleteCuratedEntry(name) {
  showModal(
    "删除精选条目",
    el(
      "div",
      {},
      el("p", {}, `将从 ${memoryLibraryPath()}/ 删除 ${name}/ 整项。`),
      el("p", { class: "hint" }, "已创建的实验不受影响；仓库改动由研究者提交。"),
    ),
    [
      el("button", { class: "btn", onclick: closeModal }, "取消"),
      el(
        "button",
        {
          class: "btn danger",
          onclick: async () => {
            try {
              const result = await api(
                `/api/memory/curated/${encodeURIComponent(name)}`,
                { method: "DELETE" },
              );
              closeModal();
              applyCuratedResult(result);
            } catch (error) {
              toast(`删除失败：${error.message}`, true);
            }
          },
        },
        "确认删除",
      ),
    ],
  );
}

const MEMORY_ACTION_LABELS = {
  created: "已新建",
  updated: "已保存",
  deleted: "已删除",
  promoted: "已晋升",
};

/* Every write answers with the refreshed listing, so the page never guesses
   what the library now holds. The response also carries the mount-timing note;
   the page states that once, persistently, instead of in every toast. */
function applyCuratedResult(result) {
  if (!memoryView) return; // the page was left while the write was in flight
  memoryView.payload.curated = result.curated || memoryView.payload.curated;
  resetCuratedDraft();
  memoryView.countHost.textContent = String(
    ((result.curated || {}).entries || []).length,
  );
  toast(`${MEMORY_ACTION_LABELS[result.action] || "已更新"} ${result.name}`);
  memoryView.entry = null;
  if (result.action === "deleted") {
    memoryView.selection = null;
    memoryView.loading = false;
  } else {
    memoryView.selection = { kind: "curated", name: result.name };
    memoryView.loading = true;
    loadMemorySelection(memoryView.selection);
  }
  renderMemoryList();
  renderMemoryCandidates();
  renderMemoryPane();
}

/* ---------------- 已挂载记忆 ----------------
   One list, not a per-session projection: an experiment resolves the curated
   library and the graduated tier once, when it is created, and every session it
   runs mounts that same read-only snapshot. So there is nothing to compare
   between sessions here — only what this experiment holds, and when it froze. */

function mountedMemoryPanel(detail) {
  const host = el("div", {});
  api(`/api/experiments/${encodeURIComponent(detail.experiment_id)}/memory`)
    .then((payload) => host.append(mountedMemorySection(detail, payload)))
    .catch(() => {
      /* no readable memory state for this experiment */
    });
  return host;
}

function mountedSourceLabel(source) {
  return source.origin === "curated"
    ? "精选库"
    : `毕业实验 ${source.source || "—"}`;
}

/* The snapshot's own copy, not the library's current text: the library may have
   moved since, and this block is what the experiment actually read. */
async function openMountedSkill(experimentId, source, name) {
  let entry;
  try {
    entry = await api(
      `/api/experiments/${encodeURIComponent(experimentId)}/memory/${encodeURIComponent(source.source)}/${encodeURIComponent(name)}`,
    );
  } catch (error) {
    toast(`读不到这条快照条目：${error.message}`, true);
    return;
  }
  showModal(
    entry.title || name,
    el(
      "div",
      {},
      el(
        "div",
        { class: "hint" },
        `${name} ｜ ${fmtBytes(entry.bytes)} ｜ ${mountedSourceLabel(source)} ｜ 快照副本`,
      ),
      el("pre", { class: "code-view skill-body section-gap" }, entry.content || ""),
    ),
    [el("button", { class: "btn", onclick: closeModal }, "关闭")],
  );
}

function mountedEntriesList(experimentId, sources) {
  const host = el("div", { class: "mounted-entries" });
  for (const source of sources) {
    host.append(
      el("div", { class: "mounted-group-title" }, mountedSourceLabel(source)),
      el(
        "div",
        { class: "file-list" },
        ...(source.entries || []).map((name) =>
          el(
            "button",
            {
              class: "file-chip",
              type: "button",
              title: "查看本实验快照里的 SKILL.md 原文",
              onclick: () => openMountedSkill(experimentId, source, name),
            },
            name,
          ),
        ),
      ),
    );
  }
  return host;
}

function mountedMemorySection(detail, payload) {
  const snapshot = payload.snapshot;
  const sources = (snapshot && snapshot.sources) || [];
  const entryCount = sources.reduce(
    (total, source) => total + (source.entries || []).length,
    0,
  );
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", {}, "已挂载记忆"),
    el(
      "div",
      { class: "hint" },
      "本实验创建时快照；库的后续改动作用于之后创建的实验。",
    ),
  );
  if (payload.error) {
    panel.append(
      el("div", { class: "hint warn" }, `快照不可读：${payload.error}`),
    );
    return panel;
  }
  if (!snapshot) {
    panel.append(
      el(
        "div",
        { class: "empty" },
        `还没有运行记忆快照（挂载模式 ${payload.mode || "—"}），下一次会话启动时补建`,
      ),
    );
    return panel;
  }
  panel.append(
    el(
      "table",
      { class: "kv section-gap" },
      kvRow("快照时间", fmtTs(snapshot.created_at)),
      kvRow("挂载模式", snapshot.mode || payload.mode || "—"),
      kvRow("挂载条目", `${entryCount} 条`),
      kvRow("已运行会话", `${payload.sessions_seen ?? 0} 个`),
    ),
  );
  if (snapshot.created_from === "first_session")
    panel.append(
      el("div", { class: "hint" }, "快照由首个会话补建。"),
    );
  if (!entryCount) {
    panel.append(
      el(
        "div",
        { class: "empty" },
        `本实验没有挂载运行记忆（挂载模式 ${snapshot.mode || payload.mode || "—"}）`,
      ),
    );
    return panel;
  }
  panel.append(mountedEntriesList(detail.experiment_id, sources));
  return panel;
}

/* ---------------- ADM-Cube trading console ----------------
   Paper reads only the local daily JSON projection. QMT deliberately keeps
   the original visual surface without a backend, transport, or live data. */

let tradingView = null;

/* Reuse the experiment badge palette: ok -> green, stale/no_snapshot -> warn,
   export_error/unreadable -> bad, absent -> muted. */
const TRADING_STATE = {
  ok: ["completed", "正常"],
  stale: ["paused", "数据陈旧"],
  no_snapshot: ["paused", "等待首次运行"],
  export_error: ["failed", "写入错误"],
  unreadable: ["failed", "数据不可读"],
  absent: ["stopped", "等待数据"],
};

/* Banner zone (only when degraded). Data below still renders from the last
   written files — stale-but-visible, never blank. */
function paperBanners(status) {
  if (status.state === "stale" && Number.isFinite(status.age_seconds))
    return [
      el(
        "div",
        { class: "banner warn" },
        `快照 ${Math.round(status.age_seconds / 3600)} 小时未更新（阈值 ${Math.round(status.stale_threshold_seconds / 3600)} 小时）`,
      ),
    ];
  if (status.state === "export_error")
    return [el("div", { class: "banner bad" }, `写入错误：${status.error || "ok=false"}`)];
  if (status.state === "unreadable")
    return [el("div", { class: "banner bad" }, `数据无法解析：${status.error || "—"}`)];
  return [];
}

function tradingBadge(state) {
  const [badgeState, label] = TRADING_STATE[state] || [
    "unknown",
    state || "未知",
  ];
  return el("span", { class: `badge state-${badgeState}` }, label);
}

function fmtAmountOpt(value) {
  const number = Number(value);
  return value === null || value === undefined || !Number.isFinite(number)
    ? "—"
    : fmtAmount(number);
}

const SHARES_FMT = new Intl.NumberFormat("en-US");

/* Share counts with separators, as the Paper orders sheet prints them. */
function fmtShares(value) {
  return Number.isInteger(value) ? SHARES_FMT.format(value) : "—";
}

/* HH:MM (UTC+8) of an order or fill stamp inside a panel already dated. */
function fmtClock(iso) {
  const text = fmtTs(iso);
  return text === "—" ? text : text.slice(-5);
}

/* An order direction as a dataTable cell, in the P&L colors. */
function actionCell(action) {
  const normalized = String(action || "").toLowerCase();
  if (normalized === "buy") return { value: "买入", cls: "pos" };
  if (normalized === "sell") return { value: "卖出", cls: "neg" };
  return action || "—";
}

/* A damaged journal line is counted by the reader, not fatal; surface the count
   so a truncated journal is visible rather than a silently shorter table. */
function skippedChip(skipped) {
  if (!skipped) return null;
  return el(
    "div",
    { class: "stats-chips" },
    el("span", { class: "stat-chip warn" }, `${skipped} 行无法解析`),
  );
}

function bookHash(env, book) {
  return `#/trading/${env}/${encodeURIComponent(book)}`;
}

/* #/trading/paper is the books overview; #/trading/paper/<book> one book. */
async function renderTradingPage(env, book) {
  $main.innerHTML = '<div class="loading">加载模拟交易…</div>';
  $topbarRight.replaceChildren();
  tradingView = { env, book, signature: "", openDays: new Set() };
  const hash = book ? bookHash(env, book) : `#/trading/${env}`;
  const load = book ? () => fetchBookBundle(env, book) : () => api(`/api/trading/${env}/books`);
  const render = book ? renderBookBundle : renderBooksOverview;
  try {
    render(await load());
  } catch (error) {
    $main.replaceChildren(el("div", { class: "empty" }, `加载失败：${error.message}`));
    return;
  }
  pollTimer = setInterval(async () => {
    if (!tradingView || location.hash !== hash) return;
    try {
      render(await load());
    } catch {
      /* keep last view */
    }
  }, 15000);
}

/* Rebuilt only when what it shows changed, so a poll never closes the history
   day being read or moves the page. */
function redrawTrading(payload, build) {
  const signature = JSON.stringify(payload);
  if (!tradingView || signature === tradingView.signature) return;
  tradingView.signature = signature;
  $main.replaceChildren(build());
}

function renderBooksOverview(payload) {
  const books = payload.books || [];
  redrawTrading(payload, () =>
    el(
      "div",
      { id: "trading-page" },
      el(
        "div",
        { class: "page-head" },
        el("h2", {}, "Paper 模拟交易", el("span", { class: "mode-note" }, `${books.length} 本账簿`)),
      ),
      payload.state === "unreadable"
        ? el("div", { class: "banner bad" }, payload.error)
        : null,
      el(
        "div",
        { class: "panel" },
        books.length
          ? dataTable(
              [
                { label: "账簿" },
                { label: "起始" },
                { label: "初始资金", num: true },
                { label: "总资产", num: true },
                { label: "累计收益", num: true },
                { label: "超额", num: true },
                { label: "今日订单", num: true },
                { label: "状态" },
              ],
              books.map((row) => [
                {
                  value: el("a", { href: bookHash(tradingView.env, row.book_id) }, row.book_id),
                  title: `${row.experiment_id || "—"} / ${row.artifact_id || "—"}`,
                },
                row.start_date ? fmtDate(row.start_date) : "—",
                fmtAmountOpt(row.initial_cash),
                fmtAmountOpt(row.equity),
                { value: fmtPct(row.total_return), cls: signCls(row.total_return) },
                { value: fmtPct(row.excess_return), cls: signCls(row.excess_return) },
                row.order_count ?? "—",
                { value: tradingBadge(row.state), title: row.error || null },
              ]),
            )
          : el("div", { class: "empty" }, "暂无账簿"),
      ),
    ),
  );
}

/* One book's reads, one per panel, issued together: api() throws inside
   Promise.all, so one missing route blanks the page. */
async function fetchBookBundle(env, book) {
  const base = `/api/trading/${env}/books/${encodeURIComponent(book)}`;
  const [status, identity, signal, history, performance, snapshot] = await Promise.all([
    api(`${base}/status`),
    api(`${base}/book`),
    api(`${base}/signal`),
    api(`${base}/history`),
    api(`${base}/performance`),
    api(`${base}/snapshot`),
  ]);
  return { status, identity, signal, history, performance, snapshot };
}

/* The snapshot age enters by the hour its banner quotes. */
function renderBookBundle(bundle) {
  const { status } = bundle;
  redrawTrading(
    { ...bundle, status: { ...status, generated_at: null, age_seconds: Math.round((status.age_seconds || 0) / 3600) } },
    () =>
      el(
        "div",
        { id: "trading-page" },
        paperHead(status, bundle.identity),
        ...paperBanners(status),
        paperSignalPanel(bundle.signal, bundle.identity),
        paperHistoryPanel(bundle.history),
        paperPerformancePanel(bundle.performance),
        paperPositionsPanel(bundle.snapshot),
      ),
  );
}

function paperHead(status, payload) {
  const book = payload.book || {};
  const head = el(
    "div",
    { class: "page-head" },
    el(
      "h2",
      {},
      el("a", { class: "exp-back", href: `#/trading/${tradingView.env}` }, "← 账簿"),
      el("span", { class: "exp-name" }, status.book_id),
      tradingBadge(status.state),
    ),
    el(
      "div",
      { class: "sub" },
      [
        // The book is named after its experiment unless init said otherwise.
        book.experiment_id === status.book_id
          ? book.artifact_id || "—"
          : `${book.experiment_id || "—"} / ${book.artifact_id || "—"}`,
        payload.start_date ? `${fmtDate(payload.start_date)} 起` : null,
        `初始资金 ${fmtAmountOpt(book.initial_cash)}`,
        payload.settled_through ? `结算至 ${fmtDate(payload.settled_through)}` : null,
      ]
        .filter(Boolean)
        .join(" · "),
    ),
  );
  // The observation note on one line; the full text stays in book.json.
  if (book.note) head.append(el("div", { class: "sub one-line", title: book.note }, book.note));
  return head;
}

function paperOrdersTable(orders) {
  return dataTable(
    [
      { label: "时间" },
      { label: "代码" },
      { label: "名称" },
      { label: "方向" },
      { label: "股数", num: true },
      { label: "参考价", num: true, title: "前一交易日收盘价" },
      { label: "金额", num: true, title: "参考价 × 股数，未计费用" },
    ],
    orders.map((row) => [
      fmtClock(row.execute_at),
      row.symbol,
      row.name,
      actionCell(row.action),
      fmtShares(row.quantity),
      fmtPrice(row.reference_price),
      fmtAmountOpt(row.notional),
    ]),
  );
}

/* Today's signal: the latest decision's orders and the holdings, weights and
   cash once they fill, all from the same order sheet the book prints. */
function paperSignalPanel(payload, identity) {
  const signal = payload.signal;
  const panel = el("div", { class: "panel section-gap" });
  if (!signal) {
    panel.append(
      panelHead("今日信号"),
      payload.state === "unreadable"
        ? el("div", { class: "hint warn" }, payload.error)
        : el("div", { class: "empty" }, "暂无决策"),
    );
    return panel;
  }
  const lastFit = identity.last_fit_date;
  const body = [
    panelHead(
      `今日信号 · ${fmtDate(signal.trade_date)}`,
      signal.fitted ? el("span", { class: "badge state-waiting_user" }, "重新拟合") : null,
    ),
    el(
      "div",
      { class: "meta-line" },
      [
        `决策 ${fmtTs(signal.inference_at)}`,
        `数据截至 ${fmtDate(signal.data_through)}`,
        signal.generation_id ? `发布 ${signal.generation_id.slice(0, 8)}` : null,
        !signal.fitted && lastFit ? `拟合于 ${fmtDate(lastFit)}` : null,
        `重放 ${signal.replayed_matching_journal ?? "—"}/${signal.replayed_calls ?? "—"}`,
      ]
        .filter(Boolean)
        .join(" · "),
    ),
    el("h4", { class: "subsection-title section-gap" }, `订单 ${signal.orders.length}`),
    signal.orders.length
      ? paperOrdersTable(signal.orders)
      : el("div", { class: "meta-line" }, "无订单"),
    skippedChip(signal.skipped_lines),
    el("h4", { class: "subsection-title section-gap" }, `成交后持仓 ${signal.target.length}`),
    signal.target.length
      ? dataTable(
          [
            { label: "代码" },
            { label: "名称" },
            { label: "股数", num: true },
            { label: "参考价", num: true },
            { label: "市值", num: true },
            { label: "权重", num: true },
          ],
          [
            ...signal.target.map((row) => [
              row.symbol,
              row.name,
              fmtShares(row.quantity),
              fmtPrice(row.reference_price),
              fmtAmountOpt(row.value),
              fmtPct(row.weight, 1),
            ]),
            ["现金", "", "", "", fmtAmountOpt(signal.cash_after), fmtPct(signal.cash_weight, 1)],
          ],
        )
      : el("div", { class: "meta-line" }, `空仓 · 现金 ${fmtAmountOpt(signal.cash_after)}`),
  ];
  panel.append(...body.filter(Boolean));
  return panel;
}

const FILL_STATUS_LABELS = { filled: "成交", rejected: "拒单" };

function paperHistoryDay(day) {
  return el(
    "div",
    { class: "history-day-body" },
    el("h4", { class: "subsection-title" }, "订单"),
    day.orders.length
      ? paperOrdersTable(day.orders)
      : el("div", { class: "meta-line" }, "无"),
    el("h4", { class: "subsection-title section-gap" }, "成交"),
    day.fills.length
      ? dataTable(
          [
            { label: "时间" },
            { label: "代码" },
            { label: "名称" },
            { label: "方向" },
            { label: "股数", num: true },
            { label: "成交价", num: true },
            { label: "费用", num: true },
            { label: "状态" },
            { label: "说明" },
          ],
          day.fills.map((row) => [
            fmtClock(row.matched_at),
            row.symbol,
            row.name,
            actionCell(row.action),
            fmtShares(row.quantity),
            fmtPrice(row.price),
            fmtAmountOpt(row.cost),
            {
              value: FILL_STATUS_LABELS[row.status] || row.status,
              cls: row.status === "rejected" ? "neg" : "",
            },
            row.reason,
          ]),
        )
      : el("div", { class: "meta-line" }, "无"),
    skippedChip(day.skipped_lines),
  );
}

/* Every earlier trading day, newest first and folded: what the book decided
   that morning and what filled. A poll keeps the days the reader opened. */
function paperHistoryPanel(payload) {
  const days = payload.days || [];
  const panel = el(
    "div",
    { class: "panel section-gap" },
    panelHead(
      "历史信号",
      days.length ? el("span", { class: "mode-note" }, `${days.length} 日`) : null,
    ),
  );
  if (payload.state === "unreadable") {
    panel.append(el("div", { class: "hint warn" }, payload.error));
    return panel;
  }
  if (!days.length) {
    panel.append(el("div", { class: "empty" }, "暂无"));
    return panel;
  }
  const list = el("div", { class: "history-list" });
  for (const day of days) {
    const filled = day.fills.filter((row) => row.status === "filled").length;
    const rejected = day.fills.filter((row) => row.status === "rejected").length;
    const details = lazyDetails(
      [
        fmtDate(day.trade_date),
        `订单 ${day.orders.length}`,
        `成交 ${filled}`,
        rejected ? `拒单 ${rejected}` : null,
      ]
        .filter(Boolean)
        .join(" · "),
      () => paperHistoryDay(day),
      day.trade_date,
    );
    details.className =
      day.orders.length || day.fills.length ? "fold history-day" : "fold history-day idle";
    details.addEventListener("toggle", () => {
      if (details.open) tradingView.openDays.add(day.trade_date);
      else tradingView.openDays.delete(day.trade_date);
    });
    if (tradingView.openDays.has(day.trade_date)) details.open = true;
    list.append(details);
  }
  panel.append(list);
  return panel;
}

/* The research return chart, fed the book's own days: cumulative return
   against CSI 300, drawdown, and end-of-day equity and cash on the same axis. */
function paperPerformancePanel(payload) {
  const stats = payload.statistics;
  const panel = el(
    "div",
    { class: "panel section-gap" },
    panelHead(
      "收益表现",
      stats ? el("span", { class: "mode-note" }, `${stats.days} 个交易日`) : null,
    ),
  );
  if (payload.state !== "ok") {
    panel.append(
      payload.state === "unreadable"
        ? el("div", { class: "hint warn" }, payload.error)
        : el("div", { class: "empty" }, "暂无"),
    );
    return panel;
  }
  const early = stats.days < payload.min_days;
  const waiting = { value: "—", title: `满 ${payload.min_days} 个交易日后计算` };
  const chart = payload.chart;
  const benchmarkNote = payload.benchmark_error
    ? `沪深300 读取失败：${payload.benchmark_error}`
    : !chart.benchmark
      ? "无沪深300 数据"
      : payload.benchmark_days < stats.days
        ? `沪深300 覆盖 ${payload.benchmark_days}/${stats.days} 日`
        : null;
  panel.append(
    ...[
      statTilesRow([
        { label: "累计收益", value: fmtPct(stats.total_return), cls: signCls(stats.total_return) },
        { label: "沪深300", value: fmtPct(stats.benchmark_return), cls: signCls(stats.benchmark_return) },
        { label: "超额", value: fmtPct(stats.excess_return), cls: signCls(stats.excess_return) },
        early
          ? { label: "年化", ...waiting }
          : { label: "年化", value: fmtPct(stats.annualized_return), cls: signCls(stats.annualized_return) },
        early ? { label: "Sharpe", ...waiting } : { label: "Sharpe", value: fmtSharpe(stats.sharpe) },
        early ? { label: "最大回撤", ...waiting } : { label: "最大回撤", value: fmtPct(stats.max_drawdown) },
        { label: "换手", value: fmtSharpe(stats.turnover), title: "成交名义额 / 初始资金" },
        { label: "佣金", value: fmtAmountOpt(stats.fees) },
        { label: "印花税", value: fmtAmountOpt(stats.stamp_duty) },
        { label: "成交笔数", value: String(stats.fills) },
      ]),
      benchmarkNote ? el("div", { class: "meta-line section-gap" }, benchmarkNote) : null,
      el("div", { class: "section-gap" }, equityChart(chart, { width: 980, height: 240, ddH: 80 })),
    ].filter(Boolean),
  );
  return panel;
}

function paperPositionsPanel(payload) {
  const account = payload.snapshot;
  const panel = el(
    "div",
    { class: "panel section-gap" },
    panelHead(
      "当前持仓",
      account && account.settled_through
        ? el("span", { class: "mode-note" }, `${fmtDate(account.settled_through)} 收盘`)
        : null,
    ),
  );
  if (!account) {
    panel.append(el("div", { class: "empty" }, "暂无快照"));
    return panel;
  }
  const positions = account.positions || [];
  const unmapped = positions.filter((row) => row.unmapped).length;
  panel.append(
    ...[
      el(
        "div",
        { class: "meta-line" },
        [
          `总资产 ${fmtAmountOpt(account.equity)}`,
          `现金 ${fmtAmountOpt(account.cash)}`,
          `持仓市值 ${fmtAmountOpt(account.market_value)}`,
          account.pending_order_count ? `待成交 ${account.pending_order_count} 笔` : null,
        ]
          .filter(Boolean)
          .join(" · "),
      ),
      unmapped ? el("div", { class: "hint warn" }, `${unmapped} 行无法映射`) : null,
      positions.length
        ? dataTable(
            [
              { label: "代码" },
              { label: "股数", num: true },
              { label: "可用", num: true },
              { label: "成本", num: true },
              { label: "最新价", num: true },
              { label: "市值", num: true },
              { label: "浮动盈亏", num: true },
              { label: "权重", num: true },
            ],
            positions.map((row) => [
              row.symbol,
              fmtShares(row.quantity),
              fmtShares(row.available_quantity),
              fmtPrice(row.average_cost),
              fmtPrice(row.last_price),
              fmtAmountOpt(row.market_value),
              { value: fmtAmountOpt(row.pnl), cls: signCls(row.pnl) },
              fmtPct(row.weight, 1),
            ]),
            { box: "limit section-gap" },
          )
        : el("div", { class: "empty" }, "无持仓"),
    ].filter(Boolean),
  );
  return panel;
}

function renderQmtPage() {
  tradingView = null;
  $topbarRight.replaceChildren();
  const unavailable = el(
    "span",
    { class: "badge state-stopped" },
    "后端未连接",
  );
  const page = el(
    "div",
    { id: "trading-page" },
    el("div", { class: "page-head" }, el("h2", {}, "实盘交易", unavailable)),
    statTilesRow([
      { label: "总资产", value: "—" },
      { label: "可用资金", value: "—" },
      { label: "持仓市值", value: "—" },
      { label: "持仓数", value: "—" },
      { label: "今日成交笔数", value: "—" },
      { label: "今日成交额", value: "—" },
    ]),
    qmtChartsPanel(),
    qmtEmptyPanel("持仓"),
    qmtEmptyPanel("成交"),
    qmtOrdersPanel(),
  );
  $main.replaceChildren(page);
}

function qmtChartsPanel() {
  return el(
    "div",
    { class: "panel section-gap" },
    el(
      "div",
      { class: "charts-row" },
      el(
        "div",
        { class: "chart-cell" },
        el("h4", { class: "subsection-title" }, "账户权益曲线"),
        el("div", { class: "empty" }, "后端未连接"),
      ),
      el(
        "div",
        { class: "chart-cell" },
        el("h4", { class: "subsection-title" }, "日成交额"),
        el("div", { class: "empty" }, "后端未连接"),
      ),
    ),
  );
}

function qmtEmptyPanel(title) {
  return el(
    "div",
    { class: "panel section-gap" },
    el("h4", {}, title),
    el("div", { class: "empty" }, "后端未连接"),
  );
}

function qmtOrdersPanel() {
  return el(
    "div",
    { class: "panel section-gap" },
    el(
      "details",
      { class: "fold" },
      el("summary", { class: "panel-summary" }, "委托"),
      el("div", { class: "empty" }, "后端未连接"),
    ),
  );
}
