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
// How the research session ended (pipelines/config.py SESSION_OUTCOMES).
const OUTCOME_LABELS = {
  freeze: "提名冻结",
  no_edge: "无边际",
  deadline: "到时",
};
const VERDICT_LABELS = {
  graduated: "graduated",
  discarded: "discarded",
  no_deliverable: "无交付",
};
// The conditions of the freeze gate and of the forward and Held-out verdict
// (pipelines/verdict.py), keyed by the token the pipeline records when one
// fails, worded as the criterion so a checklist reads them with a pass or fail
// mark. A token not listed renders as written.
const REASON_LABELS = {
  freeze_needs_full_span_validation: "提名节点为全区间验证",
  freeze_too_few_full_span_validations: "全区间验证次数",
  freeze_deflated_sharpe_unavailable: "去偏 Sharpe 概率可算",
  freeze_deflated_sharpe_below_threshold: "去偏 Sharpe 概率",
  freeze_unmeasurable: "研究期统计可算",
  forward_strategy_error: "前推期策略无报错",
  heldout_strategy_error: "Held-out 期策略无报错",
  forward_lower_bound_not_positive: "前推超额 80% 下界",
  forward_recency_negative: "前推最近 6 个月超额",
  forward_max_drawdown_exceeded: "前推回撤",
  forward_not_positive_at_cost_stress: "前推加倍滑点后超额",
  forward_too_few_round_trips: "前推平仓次数",
  forward_exposure_below_floor: "前推平均仓位",
  heldout_excess_below_tolerance: "Held-out 超额",
  heldout_max_drawdown_exceeded: "Held-out 回撤",
  heldout_exposure_below_floor: "Held-out 平均仓位",
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
// One glyph per stage family, so a card's activity line reads at a glance.
const ENVIRONMENT_STAGE_ICONS = {
  llm_call: "🤖",
  tool_call: "🛠",
  subagent_wait: "🧩",
  backtest: "📊",
  agent_complete: "🤖",
  freezing: "❄",
  forward_replay: "▶",
  verdict: "⚖",
  publishing: "💾",
};
// Stages with no Agent session to watch: no live Trace is offered for them.
// The forward replay runs with no Agent at all.
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

/* Pipeline step keys (research, frozen, forward, heldout, verdict) travel in
   the hash as they are. */
function stepKeyToUrl(key) {
  return encodeURIComponent(String(key));
}

function stepKeyFromUrl(segment) {
  return decodeURIComponent(segment);
}

/* Display names of the pipeline's steps. The session keys themselves, the
   ledgers, the trace files and every Agent-facing label stay as written; this
   map is the console's vocabulary only. */
const STEP_LABELS = {
  research: "研究",
  frozen: "冻结",
  forward: "前推回放",
  heldout: "Held-out",
  verdict: "裁决",
};

function sessionLabel(key) {
  return STEP_LABELS[key] || String(key);
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
    LIVE_RUN_STATES.has(status.state) &&
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

/* Any two-decimal figure: a Sharpe, an IR, a probability, a beta. */
function fmtSharpe(value) {
  return value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : Number(value).toFixed(2);
}

/* What the worker is doing right now: a stage glyph, the stage, its progress
   and the tool or call it is on, then a clock since the stage began. The clock
   is an elapsed node, so whoever holds the line ticks it with
   tickElapsedClocks; a caller that has its own duration passes elapsed: false.
   Null when the status carries no stage. */
function activityNode(status, { elapsed = true, className = "activity" } = {}) {
  const stage = status && status.environment_stage;
  if (!stage) return null;
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
  const node = el(
    "span",
    { class: className },
    el("span", { class: "activity-icon", "aria-hidden": "true" }, ENVIRONMENT_STAGE_ICONS[stage] || "⏳"),
    `${ENVIRONMENT_STAGE_LABELS[stage] || stage}${measured}${action}`,
  );
  if (elapsed) {
    const clock = elapsedClockNode(
      status.environment_stage_started_at || status.session_started_at,
      "",
      "activity-clock",
    );
    if (clock) node.append(" ", clock);
  }
  return node;
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

/* One row of {color, label}. An item without a color is a note about how a
   pane is drawn rather than a series, so it carries no swatch. */
function chartLegend(items) {
  return el(
    "div",
    { class: "chart-legend" },
    ...items.map((item) =>
      el(
        "span",
        { class: "legend-item" },
        item.color
          ? el("span", {
              class: "legend-swatch",
              style: `background:${item.color}`,
            })
          : null,
        item.label,
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
    const yAmount = (v) => top + (1 - Math.max(v, 0) / amountMax) * (accountH - 14);
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
  // The account pane's own encoding belongs in the legend row: drawn inside
  // the plot it collided with the pane's own ¥ ceiling tick.
  const legend = seriesList.map((s) => ({
    color: s.color,
    label: `${s.label} ${fmtPct(s.final)}`,
  }));
  if (showAccount) legend.push({ color: null, label: "资金：权益（线）· 现金（柱）" });
  const wrap = el("div", { class: "svg-chart" }, chartLegend(legend));
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

/* Tiles for the figures that exist. A figure the ledger does not carry has no
   tile at all, so a card never draws a dash where no measurement was made. */
function presentTiles(specs) {
  return specs
    .filter((spec) => spec.value !== null && spec.value !== undefined)
    .map((spec) => ({
      label: spec.label,
      value: spec.fmt(spec.value),
      cls: spec.signed ? signCls(spec.value) : "",
      title: spec.title || null,
    }));
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

/* ---------------- graphical primitives ----------------
   Small, shared drawings the pages use instead of sentences: a ratio as a
   thin bar, a ratio as a ring, a date span as a bar, criteria as a checklist,
   a weight as a bar in its table cell. Each takes the numbers a page already
   has and answers nothing when it has none, so el() drops it. */

/* A ratio as a thin bar with its percentage: attention past three quarters,
   alarm past nine tenths. */
function ratioClass(ratio) {
  return ratio >= 0.9 ? "bad" : ratio >= 0.75 ? "warn" : "";
}

function ratioBar(ratio, { cls = "", tone = true } = {}) {
  const pct = Math.max(0, Math.min(100, Math.round(ratio * 100)));
  return el(
    "span",
    { class: `bar ${cls}`.trim() },
    el("span", {
      class: `bar-fill ${tone ? ratioClass(ratio) : ""}`.trim(),
      style: `width:${pct}%`,
    }),
  );
}

// The research budgets, keyed as the trace's budget_used block and the
// listing's budget totals carry them.
const BUDGET_ROWS = [
  ["inference_seconds", "时间", fmtDuration],
  ["llm_calls", "模型调用", String],
  ["replay_years", "回放年", String],
  ["null_controls", "空对照", String],
];

/* The research budget as one labelled bar per limit; `mini` keeps only the
   most consumed one. Null while nothing was spent or no limit is known. */
function budgetBars(used, total, { mini = false } = {}) {
  if (!used || !total) return null;
  const rows = BUDGET_ROWS.map(([key, label, fmt]) => {
    const limit = Number(total[key]);
    const spent = Number(used[key]);
    if (!(limit > 0) || !Number.isFinite(spent)) return null;
    return { key, label, ratio: spent / limit, text: `${fmt(spent)} / ${fmt(limit)}` };
  }).filter(Boolean);
  if (!rows.length) return null;
  const shown = mini ? [rows.reduce((top, row) => (row.ratio > top.ratio ? row : top))] : rows;
  return el(
    "div",
    { class: `budget-bars${mini ? " mini" : ""}` },
    ...shown.map((row) =>
      el(
        "div",
        { class: "budget-row", title: `${row.label} ${row.text}` },
        el("span", { class: "budget-label" }, row.label),
        ratioBar(row.ratio),
        el("span", { class: `budget-pct ${ratioClass(row.ratio)}`.trim() }, `${Math.round(row.ratio * 100)}%`),
      ),
    ),
  );
}

/* A ratio as a ring, the percentage beside it. */
function ringGauge(ratio, label, title) {
  const r = 8,
    c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(1, ratio));
  const svg = el("span", { class: `ring ${ratioClass(ratio)}`.trim(), "aria-hidden": "true" });
  svg.innerHTML =
    `<svg viewBox="0 0 22 22"><circle class="ring-track" cx="11" cy="11" r="${r}"/>` +
    `<circle class="ring-value" cx="11" cy="11" r="${r}" stroke-dasharray="${c.toFixed(2)}" stroke-dashoffset="${(c * (1 - clamped)).toFixed(2)}"/></svg>`;
  return el(
    "span",
    { class: "gauge", title: title || null },
    svg,
    el("span", { class: "gauge-value" }, `${Math.round(ratio * 100)}%`),
    label ? el("span", { class: "gauge-label" }, label) : null,
  );
}

function dayIndex(yyyymmdd) {
  const match = /^(\d{4})(\d{2})(\d{2})$/.exec(String(yyyymmdd || ""));
  return match ? Date.UTC(Number(match[1]), Number(match[2]) - 1, Number(match[3])) / 86_400_000 : NaN;
}

function fmtMonth(yyyymmdd) {
  const text = String(yyyymmdd || "");
  return text.length >= 6 ? `${text.slice(0, 4)}-${text.slice(4, 6)}` : text;
}

/* A calendar range as a bar, with segments filled inside it and tick labels
   under it. Null when the range has no two dates. */
function spanBar(start, end, segments, ticks, { mini = false, pending = false } = {}) {
  const from = dayIndex(start),
    to = dayIndex(end);
  if (!Number.isFinite(from) || !Number.isFinite(to) || to <= from) return null;
  const at = (date) => Math.max(0, Math.min(100, ((dayIndex(date) - from) / (to - from)) * 100));
  return el(
    "span",
    { class: `span-bar${mini ? " mini" : ""}${pending ? " pending" : ""}` },
    el(
      "span",
      { class: "span-track" },
      ...segments.map((segment) =>
        el("span", {
          class: `span-fill ${segment.cls || ""}`.trim(),
          style: `left:${at(segment.from).toFixed(1)}%;width:${(at(segment.to) - at(segment.from)).toFixed(1)}%`,
          title: segment.title || null,
        }),
      ),
    ),
    el(
      "span",
      { class: "span-ticks" },
      ...ticks.map((tick) =>
        el(
          "span",
          { class: `span-tick${tick.mid ? " mid" : ""}`, style: `left:${at(tick.at).toFixed(1)}%` },
          tick.label,
        ),
      ),
    ),
  );
}

/* The one continuous replay: the forward slice, then Held-out. `focus` fills
   only that slice, for the process row that stands for it; `pending` draws
   the slices as outlines until the replay has actually run. */
function replaySpanBar(replay, focus, opts) {
  if (!replay || !replay.start || !replay.replay_end) return null;
  const segments = [
    { from: replay.start, to: replay.forward_end, cls: focus === "heldout" ? "dim" : "", title: `前推 ${fmtDate(replay.start)} ～ ${fmtDate(replay.forward_end)}` },
    { from: replay.heldout_start, to: replay.replay_end, cls: focus === "forward" ? "dim" : "alt", title: `Held-out ${fmtDate(replay.heldout_start)} ～ ${fmtDate(replay.replay_end)}` },
  ].filter((segment) => segment.from && segment.to);
  const ticks = focus
    ? []
    : [
        { at: replay.start, label: fmtMonth(replay.start) },
        { at: replay.heldout_start, label: `Held-out ${fmtMonth(replay.heldout_start)}`, mid: true },
        { at: replay.replay_end, label: fmtMonth(replay.replay_end) },
      ].filter((tick) => tick.at);
  return spanBar(replay.start, replay.replay_end, segments, ticks, opts);
}

/* A labelled fact as a chip, and a row of the chips that exist: a figure the
   record does not carry is not a chip at all. */
function chip(text, title) {
  return el("span", { class: "stat-chip", title: title || null }, text);
}

function chipsRow(chips) {
  const present = chips.filter(Boolean);
  return present.length ? el("div", { class: "stats-chips section-gap" }, ...present) : null;
}

/* Criteria as one line each: a pass or fail mark, the criterion, the measured
   value and the threshold it is held to. `ok: null` is an unmeasured one. */
function checklist(items) {
  return el(
    "div",
    { class: "checklist" },
    ...items.map((item) =>
      el(
        "div",
        { class: `check-row ${item.ok === null ? "na" : item.ok ? "ok" : "fail"}` },
        el("span", { class: "check-mark", "aria-hidden": "true" }, item.ok === null ? "–" : item.ok ? "✓" : "✕"),
        el("span", { class: "check-label" }, item.label),
        el("span", { class: "check-value" }, item.value ?? "—"),
        item.threshold ? el("span", { class: "check-threshold" }, item.threshold) : null,
      ),
    ),
  );
}

/* A portfolio weight as a bar behind its figure, scaled to the table's
   largest weight so the holdings compare at a glance. */
function weightCell(weight, largest) {
  if (weight === null || weight === undefined || !Number.isFinite(Number(weight))) return "—";
  return {
    value: el(
      "span",
      { class: "wcell" },
      ratioBar(largest > 0 ? Number(weight) / largest : 0, { cls: "wbar", tone: false }),
      fmtPct(weight, 1),
    ),
  };
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
  const key = expMatch && expMatch[2] ? stepKeyFromUrl(expMatch[2]) : null;
  // A step switch within an already-rendered experiment swaps the two panes of
  // the process grid only: no page rebuild, no scroll jump, no refetch.
  if (
    !force &&
    expMatch &&
    key &&
    detailView &&
    detailView.experimentId === expId &&
    document.body.contains(detailView.listHost)
  ) {
    selectStep(key);
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

function selectStep(key) {
  // In-experiment switches bypass route(): stop the previous step's live timers
  // (GPU refresh, elapsed clocks) or they accumulate per visit. The process
  // list is rebuilt with them, so its own clocks keep running.
  for (const timer of liveTimers) clearInterval(timer);
  liveTimers = [];
  for (const source of liveSources) source.close();
  liveSources = [];
  detailView.selectedKey = key;
  const list = processListPanel(detailView.detail, key);
  detailView.listHost.replaceWith(list);
  detailView.listHost = list;
  const fresh = sessionDetailPanel(detailView.detail, key);
  detailView.rightHost.replaceWith(fresh);
  detailView.rightHost = fresh;
  // The control panel is not rebuilt here, so its activity clock is re-armed.
  if (detailView.barHost)
    liveTimers.push(setInterval(() => tickElapsedClocks(detailView.barHost), 1000));
}

/* ---------------- the arm's pipeline, as one list of steps ----------------

   The research session, the freeze, the one continuous forward/Held-out
   replay and the verdict, in the order the pipeline runs them. The listing
   summary and the experiment detail both carry the fields read here, so the
   experiment card's miniature stepper and the experiment page's process list
   are one list at two densities, and the console answers "where are we" in
   one vocabulary: `state` draws the node, `status` says it in words. */
const STEP_STATUS_LABELS = {
  pending: "待运行",
  running: "运行中",
  paused: "已暂停",
  skipped: "未运行",
  // The freeze has not been decided yet; the replay waits on that decision.
  awaiting_nomination: "待定",
  awaiting_freeze: "待冻结",
  frozen: "已冻结",
  not_frozen: "未冻结",
  sealed: "封存中",
  replayed: "已回放",
  not_replayed: "不回放",
  undecided: "待判定",
};

/* The research step: its recorded outcome once the session ended — a freeze
   is the step done, anything else ended the arm — else where the worker is. */
function researchStep(item) {
  const status = item.status || {};
  const step = { key: "research", label: STEP_LABELS.research };
  if (item.research_outcome)
    return {
      ...step,
      state: item.frozen_session ? "done" : "failed",
      status: OUTCOME_LABELS[item.research_outcome] || item.research_outcome,
    };
  if (item.worker_alive && status.session_key === "research")
    return item.state === "paused"
      ? { ...step, state: "waiting", status: STEP_STATUS_LABELS.paused }
      : { ...step, state: "running", status: STEP_STATUS_LABELS.running };
  return { ...step, state: "pending", status: STEP_STATUS_LABELS.pending };
}

/* The freeze, the continuous replay and the verdict. `frozen_session` is the
   listing's own fact, so neither page infers a freeze from stage and verdict. */
function pipelineTailSteps(item) {
  const status = item.status || {};
  const researchOver = item.stage !== "research";
  const frozen = item.frozen_session;
  const verdict = item.verdict || {};
  const freeze = frozen
    ? { state: "done", status: STEP_STATUS_LABELS.frozen }
    : researchOver
      ? { state: "skipped", status: STEP_STATUS_LABELS.not_frozen }
      : { state: "pending", status: STEP_STATUS_LABELS.awaiting_nomination };
  const replay = item.forward
    ? { state: "done", status: STEP_STATUS_LABELS.replayed }
    : frozen
      ? {
          state:
            item.worker_alive && status.session_key === "forward"
              ? "running"
              : "pending",
          status: STEP_STATUS_LABELS.sealed,
        }
      : researchOver
        ? { state: "skipped", status: STEP_STATUS_LABELS.not_replayed }
        : { state: "pending", status: STEP_STATUS_LABELS.awaiting_freeze };
  const decided = verdict.status
    ? {
        state: verdict.status === "graduated" ? "done" : "failed",
        status: VERDICT_LABELS[verdict.status] || verdict.status,
      }
    : { state: "pending", status: STEP_STATUS_LABELS.undecided };
  return [
    { key: "frozen", label: STEP_LABELS.frozen, ...freeze },
    { key: "forward", label: STEP_LABELS.forward, ...replay },
    { key: "heldout", label: STEP_LABELS.heldout, ...replay },
    { key: "verdict", label: STEP_LABELS.verdict, ...decided },
  ];
}

function pipelineSteps(item) {
  return [researchStep(item), ...pipelineTailSteps(item)];
}

/* One stepper node: the dot whose drawing is the state (a tick when done, a
   cross when failed, a dash when skipped, a pulse while running) and its
   label. The word for the state rides in the tooltip. */
function stepNode(step, ...children) {
  return el(
    "span",
    { class: `stepper-node ${step.state}`, title: `${step.label} · ${step.status}` },
    el("span", { class: "stepper-dot", "aria-hidden": "true" }),
    el("span", { class: "stepper-label" }, step.label),
    ...children,
  );
}

/* The experiment card's miniature stepper: five nodes on one rail, the step
   the arm is on in full weight. */
function pipelineStepper(item) {
  return el(
    "div",
    { class: "stepper mini" },
    ...pipelineSteps(item).map((step) => stepNode(step)),
  );
}

/* ---------------- home page ---------------- */

/* A page render that outlives the reader's stay on that page must not
   write $main or install its poll: by then the newer page owns both. Every
   render captures the hash it was started for and stops after its fetch when
   the hash has moved. */
function navigatedAway(hash) {
  return location.hash !== hash;
}

async function renderHomePage() {
  const hash = location.hash;
  $main.innerHTML = '<div class="loading">加载中…</div>';
  $topbarRight.innerHTML = "";
  let payload;
  try {
    payload = await api("/api/experiments");
  } catch (error) {
    if (navigatedAway(hash)) return;
    $main.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
    return;
  }
  if (navigatedAway(hash)) return;
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

/* The server names a best experiment only when one has out-of-sample
   evidence, so the hero is absent while every arm is still researching. */
function bestRow(payload) {
  const best = payload.best;
  if (!best) return null;
  return (
    (payload.experiments || []).find(
      (row) => row.experiment_id === best.experiment_id,
    ) || null
  );
}

function homeView(payload) {
  const container = el("div", { id: "home" });
  const best = bestRow(payload);
  if (best)
    container.append(heroPanel(best), el("div", { class: "section-gap" }));
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
  const signature = best ? heroSignature(best) : "";
  if (!home || !grid || (hero ? hero.__signature : "") !== signature) {
    if (home) home.replaceWith(homeView(payload));
    return;
  }
  grid.replaceWith(experimentGrid(payload.experiments || []));
}

function heroSignature(item) {
  return [
    item.experiment_id,
    item.state,
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

/* The forward and Held-out slices the verdict read; absent until it exists,
   and absent for a replay the strategy's own error stopped. */
function forwardTiles(item) {
  const slices = (item.forward || {}).slices || {};
  const f = slices.forward || {};
  const h = slices.heldout || {};
  const tiles = presentTiles([
    { label: "前推超额 80% 下界", value: f.lower_bound, fmt: fmtPct, signed: true },
    { label: "前推中性化超额", value: f.neutralized_excess, fmt: fmtPct, signed: true },
    {
      label: "Held-out 中性化超额",
      value: h.neutralized_excess,
      fmt: fmtPct,
      signed: true,
    },
    { label: "前推回撤", value: f.max_drawdown, fmt: fmtPct },
  ]);
  return tiles.length ? statTilesRow(tiles) : null;
}

/* Whatever evidence the arm already has: the judged forward slices once they
   exist, else the best full-span candidate research has measured so far. */
function evidenceTiles(item) {
  const forward = forwardTiles(item);
  if (forward) return forward;
  const best = item.research_best;
  if (!best) return null;
  const tiles = presentTiles([
    {
      label: "最佳候选中性化超额",
      value: best.neutralized_excess,
      fmt: fmtPct,
      signed: true,
      title: `${sessionLabel(best.session_key)} 中 IR 最高的全区间验证，研究期年化`,
    },
    { label: "IR", value: best.information_ratio, fmt: fmtSharpe, signed: true },
    {
      label: "冻结门 去偏 Sharpe 概率",
      value: best.deflated_sharpe_probability,
      fmt: fmtSharpe,
      title: `试验 ${best.trials ?? "—"} 个`,
    },
  ]);
  return tiles.length ? statTilesRow(tiles) : null;
}

/* The continuous forward/Held-out curve, drawn only once the ledger names the
   replay's result. The node is kept by id across the five-second grid rebuild,
   so the curve is neither refetched nor redrawn while the page sits open. */
function cardEquityNode(item) {
  const result = (item.forward || {}).result;
  if (!result) return null;
  const id = `equity-card-${item.experiment_id}`;
  const existing = document.getElementById(id);
  if (existing && existing.dataset.result === result) return existing;
  const host = resultEquityHost(item.experiment_id, result, {
    width: 420,
    height: 130,
    mini: true,
    markers: forwardMarkers(item.forward),
  });
  host.id = id;
  host.dataset.result = result;
  return host;
}

function forwardMarkers(forward) {
  const start = ((forward || {}).replay || {}).heldout_start;
  return start ? [{ date: start, label: "Held-out" }] : [];
}

/* Name and badges, then the stepper, the live activity, the budget, the
   evidence and the curve — each only when the arm has it. The grid is rebuilt
   every poll, so the activity clock needs no ticker. */
function experimentCard(item) {
  const readable = item.state !== "unreadable";
  const card = el(
    "div",
    {
      class: "card clickable",
      onclick: () => {
        location.hash = `#/exp/${encodeURIComponent(item.experiment_id)}`;
      },
    },
    el(
      "h3",
      { title: `创建 ${fmtTs(item.created_at)}` },
      experimentName(item.experiment_id),
      experimentBadges(stateBadge(item.state), verdictBadge(item.verdict)),
    ),
    item.error ? el("div", { class: "meta-line" }, item.error) : null,
    readable ? pipelineStepper(item) : null,
    readable && item.worker_alive
      ? activityNode(item.status, { className: "activity meta-line" })
      : null,
    readable ? budgetBars(item.budget_used, item.budget, { mini: true }) : null,
    readable ? evidenceTiles(item) : null,
    readable ? cardEquityNode(item) : null,
  );
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

/* Only an arm the server ranked on out-of-sample evidence reaches here, so the
   tiles exist; they still go through el(), which drops an absent child instead
   of printing it. */
function heroPanel(item) {
  const panel = el(
    "div",
    { class: "panel hero", id: "hero-panel" },
    el(
      "div",
      { class: "panel-head" },
      el(
        "h3",
        { class: "hero-title" },
        el("span", { "aria-hidden": "true", title: "最佳实验：按前推超额 80% 下界" }, "🏆"),
        experimentName(item.experiment_id),
      ),
      stateBadge(item.state),
      verdictBadge(item.verdict),
    ),
    forwardTiles(item),
  );
  panel.__signature = heroSignature(item);
  const result = (item.forward || {}).result;
  if (result)
    panel.append(
      el(
        "div",
        { class: "section-gap" },
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

/* The experiment the hash names, if it names one: a step switch inside the
   experiment keeps it, so a render or poll of that experiment goes on. */
function hashExperimentId() {
  const match = location.hash.match(/^#\/exp\/([^/]+)/);
  return match ? decodeURIComponent(match[1]) : null;
}

async function renderDetailPage(experimentId, selectedKey) {
  $main.innerHTML = '<div class="loading">加载中…</div>';
  $topbarRight.innerHTML = "";
  let detail;
  try {
    detail = await api(`/api/experiments/${encodeURIComponent(experimentId)}`);
  } catch (error) {
    if (hashExperimentId() !== experimentId) return;
    $main.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
    return;
  }
  if (hashExperimentId() !== experimentId) return;
  const status = detail.status || {};
  const rows = processRows(detail);
  if (!selectedKey || !rows.some((row) => row.key === selectedKey))
    selectedKey = defaultStepKey(detail, rows);
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
    // The state, what the worker is doing, the budget and the controls that
    // act on it; where the arm is, step by step, is the process list's job.
    barHost = controlPanel(detail);
    container.append(barHost);
  }
  // Creation-time context, one line high: it frames what follows without
  // pushing the live content down.
  container.append(mountedMemoryPanel(detail));
  const layout = el("div", { class: "detail section-gap" });
  detailView = {
    experimentId,
    detail,
    listHost: null,
    rightHost: null,
    barHost,
    selectedKey,
  };
  const listHost = processListPanel(detail, selectedKey);
  const rightHost = sessionDetailPanel(detail, selectedKey);
  detailView.listHost = listHost;
  detailView.rightHost = rightHost;
  layout.append(listHost, rightHost);
  container.append(layout);
  $main.replaceChildren(container);
  if (barHost)
    liveTimers.push(setInterval(() => tickElapsedClocks(barHost), 1000));
  pollTimer = setInterval(async () => {
    if (hashExperimentId() !== experimentId) return;
    try {
      const fresh = await api(
        `/api/experiments/${encodeURIComponent(experimentId)}/status`,
      );
      if (hashExperimentId() !== experimentId) return;
      const raw = fresh.status || {};
      // A state change, a new session or a new run rebuilds the page; stage
      // flips inside one run update the control panel in place (the live
      // panels poll status themselves).
      if (
        fresh.state !== detail.state ||
        String(raw.session_key || "") !== String(status.session_key || "") ||
        String(raw.run_ref || "") !== String(status.run_ref || "")
      )
        route(true);
      else if (detailView && detailView.barHost)
        detailView.barHost.__follow(fresh);
    } catch {
      /* transient */
    }
  }, 4000);
}

/* The experiment page's process list: the research session, then the freeze,
   the continuous replay and the verdict, each carrying what it left behind —
   a figure, a date span or a live stage. `filled` says whether the step has a
   right-pane detail yet. */
function processRows(detail) {
  const session = (detail.sessions || []).find((entry) => entry.kind === "research");
  const record = (session || {}).record;
  const research = researchStep(detail);
  const live = research.state === "running" || research.state === "waiting";
  const best = record && record.best;
  const replay =
    ((detail.sessions || []).find((entry) => entry.kind === "forward") || {})
      .replay || {};
  const frozen = detail.frozen;
  const note = {
    research: record
      ? best
        ? el(
            "span",
            { title: "IR 最高的全区间验证：研究期中性化超额与去偏 Sharpe 概率" },
            `最佳候选 ${fmtPct(best.neutralized_excess)} · DSR ${fmtSharpe(best.deflated_sharpe_probability)}`,
          )
        : `验证 ${record.validations.length} 次 · 无全区间`
      : null,
    frozen: frozen
      ? `${fmtPct(frozen.neutralized_excess)} · DSR ${fmtSharpe(frozen.deflated_sharpe_probability)}`
      : null,
    forward: replaySpanBar(replay, "forward", { mini: true, pending: !detail.forward }),
    heldout: replaySpanBar(replay, "heldout", { mini: true, pending: !detail.forward }),
    verdict: null,
  };
  const filled = {
    research: Boolean(record) || live,
    frozen: Boolean(frozen),
    forward: Boolean(detail.frozen || detail.forward),
    heldout: Boolean(detail.forward),
    verdict: Boolean((detail.verdict || {}).status),
  };
  return pipelineSteps(detail).map((step) => ({
    ...step,
    session: step.key === "research" ? session : null,
    note: step.state === "skipped" ? null : note[step.key],
    filled: filled[step.key],
  }));
}

/* Where the reader lands: the session running right now, else the last step of
   the pipeline that has something to show. */
function defaultStepKey(detail, rows) {
  const status = detail.status || {};
  if (detail.worker_alive && status.session_key) {
    const live = rows.find((row) => row.key === status.session_key);
    if (live) return live.key;
  }
  const filled = rows.filter((row) => row.filled);
  return (filled[filled.length - 1] || rows[0] || {}).key;
}

/* The stepper at full size: one row per step on a vertical rail, the state
   drawn on the node, the status word beside the label and the step's own
   note under them. A row opens its step in the right pane. */
function processListPanel(detail, selectedKey) {
  const rows = processRows(detail).map((row) => {
    const status = row.session
      ? sessionDurationNode(detail, row.session, row.status, "stepper-status")
      : el("span", { class: "stepper-status" }, row.status);
    return el(
      "div",
      {
        class: `stepper-row ${row.state}${row.key === selectedKey ? " selected" : ""}`,
        "data-key": row.key,
        onclick: () => {
          location.hash = `#/exp/${encodeURIComponent(detail.experiment_id)}/${stepKeyToUrl(row.key)}`;
        },
      },
      el("span", { class: "stepper-rail", "aria-hidden": "true" }),
      el("span", { class: "stepper-dot", "aria-hidden": "true" }),
      el("span", { class: "stepper-label" }, row.label),
      status,
      row.note ? el("span", { class: "stepper-note" }, row.note) : null,
    );
  });
  return el(
    "div",
    { class: "panel" },
    el("h4", {}, "研究流程"),
    el("div", { class: "stepper vertical" }, ...rows),
  );
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

/* The graduation criteria (pipelines/verdict.py F1–F6, H1–H4) as a
   checklist: the measured figure of each slice against the threshold the
   record carries. A slice the strategy's error left unmeasured shows its
   criteria unmarked; the error itself is the failed one. */
function verdictChecklist(forward, verdict) {
  const failed = new Set(verdict.reasons || []);
  const t = ((forward.verdict || {}).thresholds) || {};
  const slices = forward.slices || {};
  const f = slices.forward,
    h = slices.heldout;
  const item = (token, slice, value, threshold) => ({
    ok: slice ? !failed.has(token) : null,
    label: reasonLabel(token),
    value: slice ? value : "—",
    threshold,
  });
  const drawdown = t.max_drawdown === undefined ? "" : `≤ ${fmtPct(t.max_drawdown)}`;
  const exposure = t.min_mean_gross === undefined ? "" : `≥ ${fmtPct(t.min_mean_gross)}`;
  return checklist([
    ...["forward", "heldout"]
      .filter((where) => failed.has(`${where}_strategy_error`))
      .map((where) => ({ ok: false, label: reasonLabel(`${where}_strategy_error`), value: forward.error })),
    item("forward_lower_bound_not_positive", f, fmtPct(f && f.lower_bound), "> 0"),
    item("forward_recency_negative", f, fmtPct(f && f.recency_neutralized_excess), "≥ 0"),
    item("forward_max_drawdown_exceeded", f, fmtPct(f && f.max_drawdown), drawdown),
    item(
      "forward_not_positive_at_cost_stress",
      f,
      fmtPct(f && f.excess_at_cost_stress),
      t.cost_stress_multiplier ? `> 0（滑点 ×${t.cost_stress_multiplier}）` : "> 0",
    ),
    item("forward_too_few_round_trips", f, f && f.round_trips, t.min_round_trips === undefined ? "" : `≥ ${t.min_round_trips}`),
    item("forward_exposure_below_floor", f, fmtPct(f && f.mean_gross), exposure),
    item("heldout_excess_below_tolerance", h, fmtPct(h && h.neutralized_excess), h ? `≥ ${fmtPct(h.tolerance)}` : ""),
    item("heldout_max_drawdown_exceeded", h, fmtPct(h && h.max_drawdown), drawdown),
    item("heldout_exposure_below_floor", h, fmtPct(h && h.mean_gross), exposure),
  ]);
}

/* The forward and Held-out replay of the frozen artifact: 封存中 until the
   forward record exists, then the verdict as a checklist, the replay span,
   the slice statistics, the one continuous curve with the Held-out boundary
   marked, and the Paper command for a graduate. A research that froze nothing
   has only its verdict. */
function verdictPanel(detail) {
  const verdict = detail.verdict || {};
  const forward = detail.forward;
  if (!detail.frozen && !verdict.status) return null;
  const head = panelHead("前推与 Held-out", verdictBadge(detail.verdict));
  if (!forward) {
    if (verdict.status)
      return el(
        "div",
        { class: "panel section-gap" },
        head,
        el("div", { class: "meta-line" }, (verdict.reasons || []).map(reasonLabel).join("；") || "—"),
      );
    const status = detail.status || {};
    const replaying = detail.worker_alive && status.session_key === "forward";
    return el(
      "div",
      { class: "panel section-gap" },
      head,
      // The replay's own stage is the control panel's line.
      el(
        "div",
        { class: "prep-indicator" },
        replaying ? el("span", { class: "spinner" }) : null,
        el("span", {}, STEP_STATUS_LABELS.sealed),
      ),
    );
  }
  const replay = forward.replay || {};
  const refits = forward.refits_executed || {};
  return el(
    "div",
    { class: "panel section-gap" },
    head,
    verdictChecklist(forward, verdict),
    el(
      "div",
      { class: "section-gap" },
      replaySpanBar(replay),
      replay.truncation_reason
        ? el("div", { class: "meta-line" }, `请求至 ${fmtDate(replay.requested_end)} · 截至发布末日`)
        : null,
    ),
    forward.error ? el("div", { class: "hint warn" }, `策略报错：${forward.error}`) : null,
    sliceTable(forward),
    chipsRow([
      Number.isFinite(refits.forward) && Number.isFinite(refits.heldout)
        ? chip(`重训 前推 ${refits.forward} · Held-out ${refits.heldout}`)
        : null,
      forward.null_percentile === null || forward.null_percentile === undefined
        ? null
        : chip(`前推 null 分位 ${fmtSharpe(forward.null_percentile)}`, "前推期超额在随机名单回放中的分位"),
    ]),
    forward.result
      ? el(
          "div",
          { class: "section-gap" },
          resultEquityHost(detail.experiment_id, forward.result, {
            width: 980,
            height: 240,
            ddH: 90,
            markers: forwardMarkers(forward),
          }),
        )
      : null,
    forward.result ? styleCard(detail.experiment_id, forward.result) : null,
    forward.result
      ? Object.assign(
          lazyDetails("交易明细", () => ordersNode(detail.experiment_id, forward.result)),
          { className: "fold section-gap" },
        )
      : null,
    detail.paper_candidate
      ? el(
          "h4",
          { class: "subsection-title section-gap", title: "在仓库根目录运行；Paper 不会自动启动" },
          "Paper 建簿",
        )
      : null,
    detail.paper_candidate ? el("pre", { class: "code-view" }, detail.paper_candidate.command) : null,
  );
}

/* The frozen artifact and the research statistics it was frozen on. */
function frozenPanel(detail) {
  const frozen = detail.frozen;
  if (!frozen) return null;
  const panel = el(
    "div",
    { class: "panel section-gap" },
    panelHead(
      "冻结产物",
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
    statTilesRow(
      presentTiles([
        { label: "研究期中性化超额（年化）", value: frozen.neutralized_excess, fmt: fmtPct, signed: true },
        { label: "残差跟踪误差", value: frozen.tracking_error, fmt: fmtPct },
        { label: "IR", value: frozen.information_ratio, fmt: fmtSharpe, signed: true },
        {
          label: "去偏 Sharpe 概率",
          value: frozen.deflated_sharpe_probability,
          fmt: fmtSharpe,
          title:
            [
              frozen.trials === null || frozen.trials === undefined ? null : `试验 ${frozen.trials} 个`,
              frozen.sharpe_star === null || frozen.sharpe_star === undefined
                ? null
                : `SR* ${fmtSharpe(frozen.sharpe_star)}`,
            ]
              .filter(Boolean)
              .join(" · ") || null,
        },
        {
          label: "前推可检出超额",
          value: frozen.forward_mde,
          fmt: fmtPct,
          title: "前推检验以 80% 功效能检出的最小年化中性化超额",
        },
      ]),
    ),
    chipsRow([
      frozen.full_span_validations === null || frozen.full_span_validations === undefined
        ? null
        : chip(`全区间验证 ${frozen.full_span_validations}`),
      frozen.null_percentile === null || frozen.null_percentile === undefined
        ? null
        : chip(`null 分位 ${fmtSharpe(frozen.null_percentile)}`, "前推期超额在随机名单回放中的分位"),
      chip(frozen.fit ? ["fit", frozen.refit_period ? `重训 ${frozen.refit_period}` : null].filter(Boolean).join(" · ") : "无 fit"),
      frozen.source_step_id
        ? chip(`节点 ${String(frozen.source_step_id).split("__").pop()}`, frozen.source_step_id)
        : null,
    ]),
    subWindowSection("研究期分年度表现", frozen.blocks),
    frozen.result
      ? el(
          "div",
          { class: "section-gap" },
          resultEquityHost(detail.experiment_id, frozen.result, {
            width: 860,
            height: 210,
            ddH: 76,
          }),
        )
      : null,
    frozen.result ? styleCard(detail.experiment_id, frozen.result) : null,
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

/* The control panel: the state badge, what the worker is doing, the skills it
   published (only once there are any), any pending request, the controls,
   and the research budget as bars. `__follow(status payload)` redraws the
   activity and the budget in place between page rebuilds. */
function controlPanel(detail) {
  const activityHost = el("span", { class: "control-activity" });
  const budgetHost = el("div", {});
  const skills = Number(detail.skills && detail.skills.count) || 0;
  const panel = el(
    "div",
    { class: "panel section-gap" },
    controlBar(
      detail,
      stateBadge(detail.state),
      activityHost,
      skills ? el("span", { class: "stat-chip", title: "本实验发布的 skills" }, `📚 Skills ${skills}`) : null,
    ),
    budgetHost,
  );
  const follow = (fresh) => {
    const activity = fresh.worker_alive ? activityNode(fresh.status) : null;
    activityHost.replaceChildren(...(activity ? [activity] : []));
    const bars = budgetBars(fresh.budget_used, detail.budget);
    budgetHost.replaceChildren(...(bars ? [bars] : []));
  };
  follow(detail);
  panel.__follow = follow;
  return panel;
}

function controlBar(detail, ...lead) {
  const id = detail.experiment_id;
  const control = detail.control || { request: null };
  const state = detail.state;
  const alive = detail.worker_alive;
  const send = (payload, note) => sendControlAction(id, payload, note);
  const actions = el("div", { class: "control-actions" });
  const bar = el("div", { class: "control-bar" }, ...lead);
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

/* A step with nothing recorded yet says only where the pipeline stands on it,
   in the process list's own words. */
function stepPlaceholder(detail, key, title) {
  const step = pipelineSteps(detail).find((row) => row.key === key);
  return el(
    "div",
    { class: "panel" },
    el("h4", {}, title),
    el("div", { class: "empty" }, step.status),
  );
}

/* The right pane of the process grid: whichever step the reader selected. The
   research sessions and the replay keep their own panels; the freeze opens the
   frozen artifact, Held-out and 裁决 the judgement that reads both slices. */
function sessionDetailPanel(detail, selectedKey) {
  // Flex column with a uniform card gap: whichever cards are present, the
  // first one's top aligns with the process list in the left grid column.
  const panel = el("div", { class: "session-detail" });
  if (selectedKey === "frozen") {
    panel.append(frozenPanel(detail) || stepPlaceholder(detail, "frozen", "冻结产物"));
    return panel;
  }
  if (selectedKey === "heldout" || selectedKey === "verdict") {
    panel.append(
      verdictPanel(detail) ||
        stepPlaceholder(detail, selectedKey, "前推与 Held-out"),
    );
    return panel;
  }
  const session = (detail.sessions || []).find(
    (entry) => entry.key === selectedKey,
  );
  // An arm with no plan yet (never started) serves no sessions at all.
  if (!session) {
    panel.append(stepPlaceholder(detail, selectedKey, sessionLabel(selectedKey)));
    return panel;
  }
  const status = detail.status || {};
  const isCurrent = status.session_key === session.key && detail.worker_alive;
  const running = isCurrent && LIVE_RUN_STATES.has(detail.state);
  const done = isSessionDone(detail, session);
  if (session.kind === "forward") {
    panel.append(forwardSessionPanel(detail, session));
    return panel;
  }
  if (done) panel.append(researchSessionPanel(detail, session));
  else if (isCurrent) {
    // Before the Agent speaks (PIT, Sandbox) there is no trace to follow;
    // the control panel already says which preparation stage the worker is in.
    if (running && !isPrepEnvironment(status, detail.state))
      panel.append(liveTracePanel(detail, session));
    panel.append(injectMessagePanel(detail, session));
  } else {
    if (detail.kind === "hitl" && detail.stage === "research")
      panel.append(directivePanel(detail, session));
    panel.append(
      el(
        "div",
        { class: "panel section-gap" },
        el("div", { class: "empty" }, researchStep(detail).status),
      ),
    );
  }
  // The Step tree is what the research sessions built, so it reads under them
  // rather than as a loose panel at the foot of the page.
  panel.append(stepTreePanel(detail));
  return panel;
}

/* The continuous replay's step: where the pipeline stands on it and the span
   it covers, the forward slice then Held-out. */
function forwardSessionPanel(detail, session) {
  const step = pipelineTailSteps(detail).find((row) => row.key === "forward");
  return el(
    "div",
    { class: "panel section-gap" },
    panelHead(STEP_LABELS.forward, el("span", { class: "badge kind" }, step.status)),
    replaySpanBar(session.replay, null, { pending: !detail.forward }),
  );
}

/* The freeze gate as the pipeline judged the nomination: the two measured
   criteria always, a failed precondition only when it failed. */
function freezeGateChecklist(gate) {
  const failed = new Set(gate.reasons || []);
  const preconditions = [
    "freeze_needs_full_span_validation",
    "freeze_unmeasurable",
    "freeze_deflated_sharpe_unavailable",
  ].filter((token) => failed.has(token));
  return checklist([
    ...preconditions.map((token) => ({ ok: false, label: reasonLabel(token) })),
    {
      ok: !failed.has("freeze_too_few_full_span_validations"),
      label: reasonLabel("freeze_too_few_full_span_validations"),
      value: gate.full_span_validations ?? "—",
      threshold: "≥ 2",
    },
    {
      ok: failed.has("freeze_deflated_sharpe_unavailable")
        ? null
        : !failed.has("freeze_deflated_sharpe_below_threshold"),
      label: reasonLabel("freeze_deflated_sharpe_below_threshold"),
      value: fmtSharpe(gate.deflated_sharpe_probability),
      threshold: "≥ 0.5",
    },
  ]);
}

/* The recorded research session: how it ended, its best full-span candidate
   with the deflated Sharpe the freeze gate would give it, the gate it met
   when it nominated, the budget it spent, every Validation it ran, and its
   Trace. */
function researchSessionPanel(detail, session) {
  const record = session.record;
  const best = record.best || {};
  const gate = record.freeze_gate;
  const attempts = Number(record.attempts) || 0;
  const budget = budgetBars(record.budget_used, detail.budget);
  const panel = el(
    "div",
    { class: "panel" },
    panelHead(
      STEP_LABELS.research,
      el(
        "span",
        {
          class: `badge ${record.froze ? "state-completed" : "state-stopped"}`,
          title: record.finish_reason || null,
        },
        OUTCOME_LABELS[record.outcome] || record.outcome,
      ),
      attempts > 1
        ? el("span", { class: "badge kind", title: "失败后原地续跑的尝试数" }, `${attempts} 次尝试`)
        : null,
    ),
    statTilesRow([
      ...presentTiles([
        {
          label: "最佳候选中性化超额",
          value: best.neutralized_excess,
          fmt: fmtPct,
          signed: true,
          title: "本会话 IR 最高的全区间验证，研究期年化",
        },
        { label: "IR", value: best.information_ratio, fmt: fmtSharpe, signed: true },
        { label: "去偏 Sharpe 概率", value: best.deflated_sharpe_probability, fmt: fmtSharpe },
      ]),
      {
        label: "验证 / 累计试验",
        value: `${record.validations.length} / ${record.trials_to_date}`,
      },
    ]),
    gate
      ? el(
          "div",
          { class: "section-gap" },
          el("h4", { class: "subsection-title" }, "冻结门"),
          freezeGateChecklist(gate),
        )
      : null,
    budget
      ? el(
          "div",
          { class: "section-gap" },
          el("h4", { class: "subsection-title" }, "预算用量"),
          budget,
        )
      : null,
    record.reason ? el("blockquote", { class: "quote section-gap" }, record.reason) : null,
    record.arm_end
      ? el("div", { class: "meta-line" }, `结束实验 · ${record.arm_end.reason || "—"}`)
      : null,
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
  );
  // A session whose trace file is gone has neither a replay nor counters.
  if (!record.trace) return panel;
  panel.append(traceReplayNode(detail.experiment_id, record.run_ref, detail));
  api(
    `/api/experiments/${encodeURIComponent(detail.experiment_id)}/trace/stats?run_id=${encodeURIComponent(record.run_ref)}`,
  )
    .then((stats) => statsHost.append(statsChipsRow(stats)))
    .catch(() => {
      /* the trace vanished between the listing and this read */
    });
  return panel;
}

function directivePanel(detail, session) {
  const control = detail.control || { directives: {} };
  const experimentDirective = String(
    (detail.params || {}).research_directive || "",
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
  panel.append(textarea);
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
          title: "须在会话启动前保存；不要写入日历日期",
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
    el(
      "h4",
      { class: "subsection-title", title: "设备按空闲显存自动挑选；条越长剩余显存越多" },
      "本会话 GPU 分配",
    ),
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
      const bar = controlPanel(detail);
      detailView.barHost.replaceWith(bar);
      detailView.barHost = bar;
    }
    selectStep(detailView.selectedKey);
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

/* The main Agent's context as a ring against its model's window, the
   compaction count beside it, then the operation counters as chips. The
   context figure is the last request's prompt tokens: what the window held
   when the Agent last spoke. */
function statsChipsRow(stats) {
  const counts = { ...(stats.counts || {}), ...(stats.tool_counts || {}) };
  const chips = el("div", { class: "stats-chips" });
  const labelled = new Set();
  const subagentTasks = Number(stats.subagent_tasks) || 0;
  const subagentRunning = Number(stats.subagent_running) || 0;
  const used = Number(stats.last_llm_prompt_tokens) || 0;
  const window = Number(stats.context_window_tokens) || 0;
  if (window > 0)
    chips.append(
      ringGauge(
        used / window,
        "上下文",
        `主 Agent 上下文 ${fmtTokens(used)} / ${fmtTokens(window)}`,
      ),
    );
  chips.append(
    el(
      "span",
      { class: "stat-chip", title: "上下文压缩次数（Agent 自行压缩与宿主兜底压缩）" },
      `⟲ 压缩 ${Number(stats.compact_ops) || 0}`,
    ),
  );
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
  if (stats.llm_prompt_tokens || stats.llm_completion_tokens) {
    chips.append(
      el(
        "span",
        { class: "stat-chip", title: "主 Agent 累计输入 tokens" },
        `↑ ${fmtTokens(stats.llm_prompt_tokens)}`,
      ),
      el(
        "span",
        { class: "stat-chip", title: "主 Agent 累计输出 tokens" },
        `↓ ${fmtTokens(stats.llm_completion_tokens)}`,
      ),
    );
  } else if (stats.llm_total_tokens) {
    chips.append(
      el(
        "span",
        { class: "stat-chip", title: "主 Agent 累计 tokens" },
        `Σ ${fmtTokens(stats.llm_total_tokens)}`,
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
    {
      type: "button",
      class: "btn",
      title:
        "只跳过尚未开始的工具，不会取消已在途的模型调用或已开始的工具。",
    },
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
  return el(
    "div",
    { class: "panel inject-message section-gap" },
    el("h4", {}, "发给当前 Agent"),
    enabled ? null : el("div", { class: "hint warn" }, reason),
    queue.pending_count > 0
      ? el(
          "div",
          { class: "inject-queue" },
          `排队 ${queue.pending_count} 条${
            queue.queued_ids.length ? `：${queue.queued_ids.join(", ")}` : ""
          }`,
        )
      : null,
    textarea,
    el("div", { class: "inject-meta" }, count),
    el("div", { class: "control-bar" }, sendBtn, interruptBtn),
  );
}

function liveTracePanel(detail, session) {
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", {}, `实时 Agent Trace · ${sessionLabel(session.key)}`),
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
      // The panel may have been replaced while the request was in flight;
      // a stream opened now would outlive the page that owns liveSources.
      if (!box.isConnected) return;
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
      if (claimStream && box.isConnected) openStream(0);
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

  // What the worker is doing is the control panel's line; the trace keeps
  // its own clocks ticking and its counters fresh.
  const pollStats = async () => {
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
  liveTimers.push(
    setInterval(() => tickElapsedClocks(box), 1000),
    setInterval(pollStats, 5000),
  );
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
  if (truncated)
    fragment.append(
      el("div", { class: "hint", title: "完整记录请下载原始 JSONL" }, "仅显示当前窗口"),
    );
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
    else if (kind === "compaction") renderCompactionBlock(node, block, index);
    else if (kind === "notice") renderNoticeBlock(node, block);
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

/* A context compaction, drawn as a rule across the trace: who triggered it
   and what it replaced on the line, the summary that now stands for those
   calls behind a fold. A compaction that did not go through says why. */
function renderCompactionBlock(node, block, index) {
  const range = block.replaced_call_range;
  const trigger = block.trigger === "agent" ? "Agent" : "宿主";
  const ok = String(block.status || "") === "ok";
  const parts = [`上下文压缩 · ${trigger}`];
  if (ok && range) parts.push(`替换第 ${range[0]}–${range[1]} 次调用 · ${block.dropped_messages} 条`);
  if (!ok) parts.push(block.error || String(block.status || "未完成"));
  const summary = String(block.summary || "");
  const details = lazyDetails(
    parts.join(" · "),
    () => el("div", { class: "llm-content" }, summary),
    `compact:${index}`,
  );
  details.className = "compaction-fold";
  if (!summary) details.querySelector("summary").classList.add("bare");
  node.classList.toggle("failed", !ok);
  node.append(details);
  if (block.ts) node.append(el("span", { class: "hint flush" }, fmtTsTime(block.ts)));
}

/* The runtime's advisory at three quarters of the compaction threshold. */
function renderNoticeBlock(node, block) {
  const estimated = Number(block.estimated_tokens) || 0;
  const threshold = Number(block.token_threshold) || 0;
  node.append(
    el("span", { class: "notice-mark", "aria-hidden": "true" }, "◔"),
    el(
      "span",
      {},
      threshold
        ? `上下文 ${Math.round((100 * estimated) / threshold)}% · ${fmtTokens(estimated)} / ${fmtTokens(threshold)}`
        : "上下文压缩提示",
    ),
  );
  if (block.ts) node.append(el("span", { class: "hint flush" }, fmtTsTime(block.ts)));
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
      [node.session_key ? sessionLabel(node.session_key) : null, node.result_name || node.node_id]
        .filter(Boolean)
        .join(" · "),
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
    node.session_key ? line("会话", sessionLabel(node.session_key)) : null,
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
      node.session_key ? kvRow("会话", sessionLabel(node.session_key)) : null,
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
      const tiles = presentTiles([
        { label: "β（vs 沪深300）", value: reg.beta, fmt: fmtSharpe },
        { label: "年化 α", value: reg.alpha_annualized, fmt: fmtPct, signed: true },
        { label: "R²", value: reg.r2, fmt: fmtSharpe },
        { label: "样本天数", value: reg.n_days, fmt: String },
      ]);
      if (tiles.length) host.append(statTilesRow(tiles));
      if (!reg.available)
        host.append(
          chipsRow([
            chip(
              `基准回归 · ${STYLE_REASON_LABELS[reg.reason] || "不可算"}`,
              STYLE_REASON_TITLES[reg.reason] || null,
            ),
          ]),
        );
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
        host.append(
          list,
          chipsRow([
            chip(`持仓 ${style.days} 日`, "有持仓的交易日数"),
            chip(`日均 ${style.avg_names} 只`),
            chip(`日均多头 ${fmtAmount(style.avg_long_gross)}`),
            ...(style.industries || []).map((row) =>
              chip(`${row.name} ${(row.weight * 100).toFixed(0)}%`, "行业净权重（申万一级）"),
            ),
          ]),
        );
      } else {
        host.append(
          chipsRow([
            chip(
              `风格暴露 · ${STYLE_REASON_LABELS[style.reason] || "不可算"}`,
              STYLE_REASON_TITLES[style.reason] || null,
            ),
          ]),
        );
      }
    })
    .catch((error) => {
      const missing = /没有已落盘|404/.test(error.message);
      host.append(
        el("div", { class: "hint" }, missing ? "无风格归因数据" : `加载失败：${error.message}`),
      );
    });
  return host;
}

/* Why a style figure is absent, as a label on the chip and the full reason
   in its tooltip (environment/replay/style.py). */
const STYLE_REASON_LABELS = {
  benchmark_unavailable: "无同窗沪深300",
  insufficient_overlapping_days: "重叠交易日不足 8 天",
  benchmark_variance_zero: "沪深300 无波动",
  style_columns_unavailable: "缺市值 / PB / 换手截面",
  no_holdings: "无持仓",
  no_valued_holdings: "持仓无收盘价",
};
const STYLE_REASON_TITLES = {
  benchmark_unavailable: "回放槽中没有可用的沪深300同窗数据，基准回归为空",
  insufficient_overlapping_days: "与沪深300重叠的交易日不足 8 天，β、α 与 R² 不计算",
  benchmark_variance_zero: "同窗沪深300收益没有可回归的波动，β、α 与 R² 不计算",
  style_columns_unavailable: "回放槽缺少市值、PB 或换手截面，风格暴露为空",
  no_holdings: "该回放没有持仓，风格暴露为空",
  no_valued_holdings: "该回放的持仓没有可用收盘价，风格暴露为空",
};

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
      const shown = rows.slice(0, 80);
      body.replaceChildren(
        el(
          "div",
          { class: "control-bar" },
          el("span", { class: "mode-note" }, data.result),
          el("span", { class: "spacer" }),
          data.row_count > shown.length
            ? el(
                "span",
                { class: "mode-note", title: "表格只列出前几条，完整明细请导出 CSV" },
                `前 ${shown.length} / ${data.row_count} 条`,
              )
            : null,
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
  const hash = location.hash;
  memoryView = null;
  $main.innerHTML = '<div class="loading">加载运行记忆…</div>';
  $topbarRight.replaceChildren();
  let payload;
  try {
    payload = await api("/api/memory");
  } catch (error) {
    if (navigatedAway(hash)) return;
    $main.replaceChildren(
      el("div", { class: "empty" }, `加载失败：${error.message}`),
    );
    return;
  }
  if (navigatedAway(hash)) return;
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
          { class: "sub", title: "改动作用于此后创建的实验" },
          `默认挂载 ${payload.default_mode || "—"}`,
        ),
      ),
      memorySection(
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
        "问题反馈",
        "处置：scripts/experiments/resolve_issue.py",
        el("div", { class: "panel" }, issueFilterBar(), memoryView.issuesHost),
      ),
    ),
  );
  renderMemoryList();
  renderMemoryCandidates();
  renderMemoryPane();
  renderIssueReports();
}

/* One pattern for every section on this page: a heading, a tooltip for its
   one operational note, then the panels. */
function memorySection(title, note, ...panels) {
  return el(
    "section",
    { class: "memory-section" },
    el("div", { class: "memory-section-head" }, el("h3", { title: note || null }, title)),
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
      issueMetaItem(
        "会话",
        report.session_label ? sessionLabel(report.session_label) : "—",
      ),
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
          ? `无未处置报告 · 已处置 ${resolved} 条`
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
        el("div", { class: "empty compact" }, "从左侧选择一个条目"),
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
          { class: "hint", title: "整项复制，含 scripts/ 与 references/" },
          `来源 ${memoryView.source.experiment_id} / ${memoryView.source.skill}`,
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
  const host = el(
    "div",
    { class: "panel section-gap mounted-memory" },
    el("div", { class: "hint" }, "读取已挂载记忆…"),
  );
  api(`/api/experiments/${encodeURIComponent(detail.experiment_id)}/memory`)
    .then((payload) => host.replaceChildren(mountedMemorySection(detail, payload)))
    .catch((error) =>
      host.replaceChildren(
        el("div", { class: "hint warn" }, `读不到已挂载记忆：${error.message}`),
      ),
    );
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

/* One line of creation-time context, opened only when the reader wants the
   snapshot itself. An unreadable or missing snapshot says so on that line
   rather than leaving the strip looking empty. */
function mountedMemorySection(detail, payload) {
  if (payload.error)
    return el(
      "div",
      { class: "hint warn" },
      `运行记忆快照不可读：${payload.error}`,
    );
  const snapshot = payload.snapshot;
  if (!snapshot)
    return el(
      "div",
      { class: "hint" },
      `已挂载记忆 · ${payload.mode || "—"} · 快照待会话启动时补建`,
    );
  const sources = snapshot.sources || [];
  const count = (origin) =>
    sources
      .filter((source) =>
        origin === "curated"
          ? source.origin === "curated"
          : source.origin !== "curated",
      )
      .reduce((total, source) => total + (source.entries || []).length, 0);
  const curated = count("curated");
  const graduated = count("graduated");
  const summary = [
    "已挂载记忆",
    snapshot.mode || payload.mode || "—",
    `精选 ${curated} · 毕业 ${graduated}`,
  ].join(" · ");
  const fold = el(
    "details",
    { class: "fold" },
    el("summary", {}, summary),
    el(
      "table",
      { class: "kv section-gap" },
      kvRow("快照时间", fmtTs(snapshot.created_at)),
      kvRow("已运行会话", `${payload.sessions_seen ?? 0} 个`),
      snapshot.created_from === "first_session"
        ? kvRow("快照来源", "由首个会话补建")
        : null,
    ),
    curated + graduated
      ? mountedEntriesList(detail.experiment_id, sources)
      : el("div", { class: "empty compact" }, "本实验没有挂载运行记忆"),
  );
  return fold;
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
  let bundle;
  try {
    bundle = await load();
  } catch (error) {
    if (navigatedAway(hash)) return;
    $main.replaceChildren(el("div", { class: "empty" }, `加载失败：${error.message}`));
    return;
  }
  if (navigatedAway(hash)) return;
  render(bundle);
  pollTimer = setInterval(async () => {
    if (!tradingView || navigatedAway(hash)) return;
    try {
      const fresh = await load();
      if (!tradingView || navigatedAway(hash)) return;
      render(fresh);
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

/* One card per book, in the research home's language: what the book trades,
   the figures its own panels measured and its curve once two days settled.
   A figure the book has not measured has no tile. */
function bookCard(row) {
  const href = bookHash(tradingView.env, row.book_id);
  const card = el("div", {
    class: "card clickable",
    onclick: () => {
      location.hash = href;
    },
  });
  const tiles = presentTiles([
    { label: "总资产", value: row.equity, fmt: fmtAmount },
    { label: "累计收益", value: row.total_return, fmt: fmtPct, signed: true },
    {
      label: "持仓",
      value: row.position_count,
      fmt: (count) => `${count} 只`,
      title: "最近一次结算收盘时的持仓只数",
    },
    { label: "超额 vs 沪深300", value: row.excess_return, fmt: fmtPct, signed: true },
    { label: "最大回撤", value: row.max_drawdown, fmt: fmtPct },
    {
      label: "今日订单",
      value: row.order_count,
      fmt: String,
      title: row.signal_date ? `${fmtDate(row.signal_date)} 的决策` : null,
    },
  ]);
  card.append(
    el(
      "h3",
      {},
      el("a", { class: "exp-name", href, title: row.book_id }, row.book_id),
      experimentBadges(tradingBadge(row.state)),
    ),
    el(
      "div",
      { class: "meta-line" },
      [
        row.candidate_source,
        row.artifact_id,
        row.start_date ? `${fmtDate(row.start_date)} 起` : null,
        row.initial_cash === null || row.initial_cash === undefined
          ? null
          : `初始资金 ${fmtAmount(row.initial_cash)}`,
      ]
        .filter(Boolean)
        .join(" · "),
      row.error ? ` ｜ ${row.error}` : "",
    ),
  );
  if (tiles.length) card.append(statTilesRow(tiles));
  if (row.curve)
    card.append(equityChart(row.curve, { width: 420, height: 130, mini: true }));
  return card;
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
        el("h2", {}, "模拟交易", el("span", { class: "mode-note" }, `${books.length} 本账簿`)),
      ),
      payload.state === "unreadable"
        ? el("div", { class: "banner bad" }, payload.error)
        : null,
      books.length
        ? el("div", { class: "grid" }, ...books.map(bookCard))
        : el("div", { class: "empty" }, "暂无账簿"),
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

/* The book's frozen identity, one value per row: what it trades, where the
   candidate came from — with the whole 建簿 note, which a one-line subtitle
   used to cut off mid-sentence — and where its calendar stands. */
function paperHead(status, payload) {
  const book = payload.book || {};
  const rows = [
    // The book is named after its experiment unless init said otherwise.
    book.artifact_id
      ? kvRow(
          "产物",
          book.experiment_id && book.experiment_id !== status.book_id
            ? `${book.experiment_id} / ${book.artifact_id}`
            : book.artifact_id,
        )
      : null,
    book.candidate_source || book.note
      ? kvRow(
          "候选来源",
          el(
            "div",
            {},
            book.candidate_source,
            book.note ? el("div", { class: "hint" }, book.note) : null,
          ),
        )
      : null,
    payload.start_date ? kvRow("起始", fmtDate(payload.start_date)) : null,
    book.initial_cash === null || book.initial_cash === undefined
      ? null
      : kvRow("初始资金", fmtAmount(book.initial_cash)),
    payload.settled_through ? kvRow("结算至", fmtDate(payload.settled_through)) : null,
  ].filter(Boolean);
  return el(
    "div",
    { class: "page-head" },
    el(
      "h2",
      {},
      el("a", { class: "exp-back", href: `#/trading/${tradingView.env}` }, "← 账簿"),
      el("span", { class: "exp-name" }, status.book_id),
      tradingBadge(status.state),
    ),
    rows.length
      ? el("div", { class: "sub book-facts" }, el("table", { class: "kv" }, ...rows))
      : null,
  );
}

/* The scale of a holdings table's weight bars: its largest weight. */
function largestWeight(rows) {
  return Math.max(0, ...rows.map((row) => Number(row.weight) || 0));
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

/* One decision's order sheet, for today's panel and for any past day: the
   orders it placed and, when it placed any, the holdings and cash they fill
   into. No orders is no change, so the holdings stay in 当前持仓 alone. */
function paperSheetBody(sheet) {
  if (!sheet.orders.length) return [el("div", { class: "meta-line" }, "无订单 · 持仓不变")];
  return [
    el("h4", { class: "subsection-title" }, `订单 ${sheet.orders.length}`),
    paperOrdersTable(sheet.orders),
    el("h4", { class: "subsection-title section-gap" }, `成交后持仓 ${sheet.target.length}`),
    sheet.target.length
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
            ...sheet.target.map((row) => [
              row.symbol,
              row.name,
              fmtShares(row.quantity),
              fmtPrice(row.reference_price),
              fmtAmountOpt(row.value),
              weightCell(row.weight, largestWeight(sheet.target)),
            ]),
            [
              "现金",
              "",
              "",
              "",
              fmtAmountOpt(sheet.cash_after),
              weightCell(sheet.cash_weight, largestWeight(sheet.target)),
            ],
          ],
        )
      : el("div", { class: "meta-line" }, `空仓 · 现金 ${fmtAmountOpt(sheet.cash_after)}`),
  ];
}

/* Today's signal: the decision that produced it in one caption, then its
   order sheet — the same calculation the printed 订单单 reads. */
function paperSignalPanel(payload, identity) {
  const signal = payload.signal;
  if (!signal)
    return el(
      "div",
      { class: "panel section-gap" },
      panelHead("今日信号"),
      payload.state === "unreadable"
        ? el("div", { class: "hint warn" }, payload.error)
        : el("div", { class: "empty" }, "暂无决策"),
    );
  const lastFit = identity.last_fit_date;
  return el(
    "div",
    { class: "panel section-gap" },
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
        Number.isFinite(signal.replayed_calls) && Number.isFinite(signal.replayed_matching_journal)
          ? `重放 ${signal.replayed_matching_journal}/${signal.replayed_calls}`
          : null,
      ]
        .filter(Boolean)
        .join(" · "),
    ),
    el("div", { class: "section-gap" }, ...paperSheetBody(signal)),
    skippedChip(signal.skipped_lines),
  );
}

const FILL_STATUS_LABELS = { filled: "成交", rejected: "拒单" };

/* A past day reads like today's panel: that morning's order sheet — its
   orders and the holdings they filled into — and then what actually filled.
   A day the book only settled has no sheet at all, and shows only its fills. */
function paperHistoryDay(day) {
  const sheet = day.target ? paperSheetBody(day) : [];
  // Only a rejected order carries a reason: a day where everything filled has
  // no 说明 to show, and a column of dashes is not one.
  const explained = day.fills.some((row) => row.reason);
  return el(
    "div",
    { class: "history-day-body" },
    ...sheet,
    el(
      "h4",
      { class: `subsection-title${sheet.length ? " section-gap" : ""}` },
      "成交",
    ),
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
            ...(explained ? [{ label: "说明" }] : []),
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
            ...(explained ? [row.reason] : []),
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
   against CSI 300, drawdown, and end-of-day equity and cash on the same axis.
   Tiles only for the figures the book has measured; the statistics the day
   count still gates, and a CSI 300 that does not cover the book, say so in
   one caption instead of leaving a dash behind. */
function paperPerformancePanel(payload) {
  const stats = payload.statistics;
  const head = panelHead(
    "收益表现",
    stats ? el("span", { class: "mode-note" }, `${stats.days} 个交易日`) : null,
  );
  if (payload.state !== "ok")
    return el(
      "div",
      { class: "panel section-gap" },
      head,
      payload.state === "unreadable"
        ? el("div", { class: "hint warn" }, payload.error)
        : el("div", { class: "empty" }, "暂无"),
    );
  if (stats.days < payload.min_days)
    head.querySelector(".mode-note").title =
      `年化、Sharpe 与最大回撤自第 ${payload.min_days} 个交易日起给出`;
  const notes = [
    payload.benchmark_error
      ? `沪深300 读取失败：${payload.benchmark_error}`
      : !payload.benchmark_days
        ? "无沪深300 数据"
        : payload.benchmark_days < stats.days
          ? `沪深300 覆盖 ${payload.benchmark_days}/${stats.days} 日`
          : null,
  ].filter(Boolean);
  const cost = [
    stats.fees === null ? null : `佣金 ${fmtAmount(stats.fees)}`,
    stats.stamp_duty === null ? null : `印花税 ${fmtAmount(stats.stamp_duty)}`,
    `成交 ${stats.fills} 笔`,
  ].filter(Boolean);
  const tiles = presentTiles([
    { label: "累计收益", value: stats.total_return, fmt: fmtPct, signed: true },
    { label: "沪深300", value: stats.benchmark_return, fmt: fmtPct, signed: true },
    { label: "超额", value: stats.excess_return, fmt: fmtPct, signed: true },
    { label: "年化", value: stats.annualized_return, fmt: fmtPct, signed: true },
    { label: "Sharpe", value: stats.sharpe, fmt: fmtSharpe, signed: true },
    { label: "最大回撤", value: stats.max_drawdown, fmt: fmtPct },
    {
      label: "换手",
      value: stats.turnover,
      fmt: fmtSharpe,
      title: "成交名义额 / 初始资金",
    },
  ]);
  return el(
    "div",
    { class: "panel section-gap" },
    head,
    tiles.length ? statTilesRow(tiles) : null,
    notes.length ? el("div", { class: "meta-line section-gap" }, notes.join(" · ")) : null,
    payload.chart
      ? el(
          "div",
          { class: "section-gap" },
          equityChart(payload.chart, { width: 980, height: 240, ddH: 80 }),
        )
      : null,
    el("div", { class: "meta-line section-gap" }, cost.join(" · ")),
  );
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
              weightCell(row.weight, largestWeight(positions)),
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
