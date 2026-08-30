/* ADM-Cube HITL console SPA — no build step, no dependencies. */

const $main = document.getElementById("main");
const $topbarRight = document.getElementById("topbar-right");
const $modalRoot = document.getElementById("modal-root");
const $toastRoot = document.getElementById("toast-root");

const STATE_LABELS = {
  launching: "启动中",
  starting: "初始化",
  preparing: "准备数据",
  running_session: "运行中",
  running_heldout: "Held-out 运行中",
  waiting_user: "等待批准",
  waiting_step_user: "等待 Step 批准",
  waiting_user_reply: "等待答复提问",
  paused: "已暂停",
  completed: "已完成",
  stopped: "已停止",
  failed: "失败",
  interrupted: "已中断",
  terminated: "已强制终止",
  created: "未启动",
  development_complete: "开发完成",
  unreadable: "不可解析",
  unknown: "未知",
};
const KIND_LABELS = {
  fold: "Fold",
  meta_learning: "元学习",
  heldout: "Held-out",
};

function sessionDisplayKey(session) {
  if (!session) return "";
  if (session.kind === "fold") {
    const period = String(session.label || "").trim();
    const epoch = String(session.epoch_id || "");
    if (period && epoch) return `${epoch}/${period}`;
    if (period) return period;
    const display = String(session.display_key || "");
    if (display && !display.includes("fold_ref_")) return display;
    return "Fold";
  }
  if (session.kind === "meta_learning") {
    const epoch = String(session.epoch_id || "");
    return epoch ? `${epoch}/元学习` : "元学习";
  }
  if (session.kind === "heldout") return "Held-out";
  const display = String(session.display_key || session.label || "");
  return display.includes("fold_ref_") || display.includes("meta_ref_")
    ? ""
    : display;
}

function sessionListLabel(session) {
  if (session.kind === "fold")
    return String(session.label || "Fold");
  if (
    session.kind === "meta_learning" &&
    Number(session.trigger_after_folds || 0) > 0
  )
    return `元学习（${session.trigger_after_folds} Fold 后）`;
  return KIND_LABELS[session.kind] || session.kind;
}

function foldPeriodLabel(detail, foldRef) {
  const hit = ((detail && detail.sessions) || []).find(
    (session) => session.fold_ref === foldRef,
  );
  return (hit && hit.label) || "—";
}
const ENVIRONMENT_STAGE_LABELS = {
  preparing_session: "准备会话",
  pit_snapshot: "准备 PIT 快照",
  sandbox_layout: "准备 Sandbox 工作区",
  pit_view: "装载 PIT 可见视图",
  sandbox_start: "启动 Sandbox",
  parent_control: "父本对照回测",
  llm_call: "Agent 推理",
  tool_call: "执行工具",
  subagent_wait: "等待子代理",
  backtest: "执行验证回测",
  agent_complete: "Agent 推理完成",
  frozen_test: "执行冻结测试",
  publishing: "结果落盘",
  meta_finalize: "元学习结果校验",
  environment_update: "Sandbox 环境更新",
  analysis: "Fold 策略分析",
  heldout: "执行 Held-out",
  session_retry: "会话失败重试",
};
// How a walk-forward transition was measured: without a Test stage it is the
// host's parent control of every Fold after the Epoch's first, with one it is
// each Fold's frozen Test (pipelines/ledger.py::walk_forward_transitions).
const WALK_FORWARD_SOURCES = {
  parent_control: "父本对照",
  frozen_test: "冻结测试",
};
const PREP_ENVIRONMENT_STAGES = new Set([
  "preparing_session",
  "pit_snapshot",
  "sandbox_layout",
  "pit_view",
  "sandbox_start",
  "parent_control",
  "heldout",
  "session_retry",
  "environment_update",
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
const LIVE_RUN_STATES = new Set([
  "running_session",
  "waiting_step_user",
  "waiting_user_reply",
]);
const ACTIVE_SESSION_STATES = LIVE_RUN_STATES;
const RESEARCHER_WAIT_STATES = new Set([
  "waiting_step_user",
  "waiting_user_reply",
]);
const INJECT_MESSAGE_MAX_CHARS = 8192;
const INJECT_MESSAGE_QUEUED_NOTE = "已排队，将在 Agent 下一安全点生效";
const TERMINAL_INJECT_STATES = new Set([
  "completed",
  "stopped",
  "failed",
  "interrupted",
  "terminated",
  "development_complete",
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

/* Session keys contain "/" (epoch_001/fold_2022Q1); in the hash they travel as
   "~" so URLs stay readable (no %2F). Old encoded links still parse. */
function sessionKeyToUrl(key) {
  return encodeURIComponent(String(key).replaceAll("/", "~"));
}

function sessionKeyFromUrl(segment) {
  return decodeURIComponent(segment).replaceAll("~", "/");
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

/* Acceptance warnings are durable ledger text. Keep old records readable while
   new records already arrive pre-formatted from the pipeline. */
function fmtAcceptanceWarning(value) {
  const text = String(value || "");
  let match = /^validation return\s+([-+0-9.eE]+)\s+<\s+([-+0-9.eE]+)$/.exec(
    text,
  );
  if (match)
    return `validation return ${fmtPct(Number(match[1]))} < ${fmtPct(Number(match[2]))}`;
  match = /^sharpe\s+([-+0-9.eE]+)\s+<\s+([-+0-9.eE]+)$/.exec(text);
  if (match)
    return `sharpe ${Number(match[1]).toFixed(2)} < ${Number(match[2]).toFixed(2)}`;
  return text;
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

function foldDurationNode(detail, session, prefix = "", className = "") {
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
    const completedWait = Number(status.researcher_wait_seconds) || 0;
    const waitStartedAt = RESEARCHER_WAIT_STATES.has(status.state)
      ? Date.parse(status.wait_started_at || "")
      : NaN;
    const activeWait = Number.isFinite(waitStartedAt)
      ? Math.max(0, (Date.now() - waitStartedAt) / 1000)
      : 0;
    const seconds = isFixed
      ? fixed
      : isLive
        ? Math.max(
            0,
            (Date.now() - startedAt) / 1000 - completedWait - activeWait,
          )
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

function sealedMetricTile(revealed, label, value, format = fmtPct) {
  if (!revealed) return { label, value: "未揭示", cls: "" };
  return { label, value: format(value), cls: signCls(value) };
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
  if (state === "running_heldout") return true;
  const stage = (status && status.environment_stage) || "";
  if (PREP_ENVIRONMENT_STAGES.has(stage)) return true;
  return (
    !stage &&
    (state === "running_session" ||
      state === "starting" ||
      state === "preparing" ||
      state === "launching")
  );
}

function numClass(value) {
  if (value === null || value === undefined) return "num";
  return value >= 0 ? "num pos" : "num neg";
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

/* Minimal markdown renderer for the analysis panel (headings, lists, code,
   bold, inline code). Input is escaped first, so no raw HTML passes through. */
function renderMarkdown(text) {
  const lines = escapeHtml(text).split("\n");
  const out = [];
  let inCode = false,
    inList = false;
  for (const line of lines) {
    if (line.startsWith("```")) {
      if (inList) {
        out.push("</ul>");
        inList = false;
      }
      out.push(inCode ? "</pre>" : "<pre>");
      inCode = !inCode;
      continue;
    }
    if (inCode) {
      out.push(line);
      continue;
    }
    const html = line
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>");
    const heading = html.match(/^(#{1,4})\s+(.*)$/);
    const listItem = html.match(/^\s*[-*]\s+(.*)$/);
    if (listItem && !heading) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${listItem[1]}</li>`);
      continue;
    }
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
    if (heading)
      out.push(
        `<h${heading[1].length + 1}>${heading[2]}</h${heading[1].length + 1}>`,
      );
    else if (html.trim() === "") out.push("");
    else out.push(`<p>${html}</p>`);
  }
  if (inList) out.push("</ul>");
  if (inCode) out.push("</pre>");
  const div = el("div", { class: "markdown" });
  div.innerHTML = out.join("\n");
  return div;
}

/* ---------------- charts ----------------
   Specs: thin marks (bars ≤24px, 2px surface gap, 4px rounded data-end,
   square baseline; 2px lines; ≥8px markers with 2px surface ring), hairline
   solid gridlines, muted-ink labels, legend for 2 series, hover tooltips.
   Palette: categorical slots 1-2 (blue/aqua), CVD+contrast validated on white;
   aqua's sub-3:1 relief is carried by the result tables and tooltips. */

/* Both palettes validated (dataviz validator): light pair on white, dark pair
   (#3987e5/#199e70, the palette's dark steps) on the dark panel — all checks pass. */
function themeInk() {
  if (currentTheme() === "dark") {
    return {
      // Categorical slots 1-3 (dark steps; validated with the dataviz checker
      // on the dark panel #1b1f28 — all pass incl. contrast).
      validColor: "#3987e5",
      testColor: "#199e70",
      heldoutColor: "#c98500",
      validLight: "#7fb2ef",
      testLight: "#5ec49a",
      grid: "#2b303c",
      baseline: "#4a5163",
      muted: "#98a0af",
      faint: "#6f7787",
      ring: "#1b1f28",
    };
  }
  return {
    // Light slots 1-3 validated on white; aqua/yellow sit in the sub-3:1
    // relief band — carried by the result tables and rich tooltips.
    validColor: "#2a78d6",
    testColor: "#1baf7a",
    heldoutColor: "#eda100",
    validLight: "#86b6ef",
    testLight: "#66cfa4",
    grid: "#e9ebf1",
    baseline: "#c2c7d2",
    muted: "#68717f",
    faint: "#a5abb8",
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

/* ---- daily equity lines + drawdown subplot (vs 沪深300 benchmark) ----
   Series = daily simple returns [[YYYYMMDD, r], ...]; the client compounds
   into cumulative curves and running drawdowns. Benchmark is drawn dashed in
   neutral ink (a reference, not a categorical slot). */
const EQUITY_CACHE = new Map(); // `${experiment_id}::${epoch_id}` -> { fp, payload }

function fetchExperimentEquity(expId, fp, epochId = null) {
  const cacheKey = `${expId}::${epochId || ""}`;
  const hit = EQUITY_CACHE.get(cacheKey);
  if (hit && hit.fp === fp) return hit.ready;
  const query = epochId ? `?epoch_id=${encodeURIComponent(epochId)}` : "";
  const ready = api(
    `/api/experiments/${encodeURIComponent(expId)}/equity${query}`,
  );
  EQUITY_CACHE.set(cacheKey, { fp, ready });
  return ready;
}

function epochShort(epochId) {
  const m = /^epoch_0*(\d+)$/.exec(String(epochId || ""));
  return m ? `E${m[1]}` : String(epochId || "");
}

/* Full-cycle statistics (server-computed in equity.py::_cycle_stats), rendered
   as ONE compact row per chained series (metrics as columns) so the block adds
   a few text lines under the chart instead of a screen-tall table. Cumulative
   return is omitted — the chart legend already shows each series' final. */
const CYCLE_STAT_COLUMNS = [
  ["annualized_return", "年化", "年化收益", (v) => fmtPct(v), true],
  ["annualized_vol", "波动", "年化波动", (v) => fmtPct(v), false],
  ["sharpe", "Sharpe", "年化 Sharpe", (v) => Number(v).toFixed(2), true],
  ["max_drawdown", "回撤", "最大回撤", (v) => fmtPct(v), false],
  ["daily_win_rate", "日胜率", "日度胜率", (v) => fmtPct(v), false],
  [
    "benchmark_return",
    "基准",
    "沪深300 同期收益（按日期配对）",
    (v) => fmtPct(v),
    false,
  ],
  ["excess_return", "超额", "相对沪深300 的超额收益", (v) => fmtPct(v), true],
  ["beta", "β", "对沪深300 的日收益 β", (v) => Number(v).toFixed(2), false],
  ["tracking_error", "跟踪误差", "年化跟踪误差", (v) => fmtPct(v), false],
  [
    "information_ratio",
    "IR",
    "信息比率（年化）",
    (v) => Number(v).toFixed(2),
    true,
  ],
  ["n_days", "天数", "交易日数", (v) => String(v), false],
];
const CYCLE_SERIES_SHORT = { valid: "验证", test: "测试", heldout: "Held-out" };

function cycleStatsTable(payload) {
  const stats = payload.stats || {};
  const keys = ["valid", "test", "heldout"].filter((k) => stats[k]);
  if (!keys.length) return null;
  const INK = themeInk();
  const seriesColor = {
    valid: INK.validColor,
    test: INK.testColor,
    heldout: INK.heldoutColor,
  };
  const columns = CYCLE_STAT_COLUMNS.filter(([field]) =>
    keys.some((k) => stats[k][field] !== null && stats[k][field] !== undefined),
  );
  const head = el(
    "tr",
    {},
    el("th", {}, `全周期（${epochShort(payload.epoch_id)}）`),
    ...columns.map(([, label, full]) => el("th", { title: full }, label)),
  );
  // Identity rides a colored swatch matching the chart line, never colored text.
  const rows = keys.map((k) =>
    el(
      "tr",
      {},
      el(
        "td",
        {},
        el("span", {
          class: "legend-swatch",
          style: `background:${seriesColor[k]}`,
        }),
        CYCLE_SERIES_SHORT[k] || k,
      ),
      ...columns.map(([field, , , fmt, signed]) => {
        const v = stats[k][field];
        return el(
          "td",
          { class: signed ? signCls(v) : "" },
          v === null || v === undefined ? "—" : fmt(v),
        );
      }),
    ),
  );
  return el("table", { class: "data cycle-stats" }, head, ...rows);
}

/* Async host: renders the chart (plus, on full-size charts, the epoch switcher
   and the full-cycle stats table) when the series payload arrives. Each epoch
   is charted alone — epochs re-run the same fold calendar and must not blend. */
function equityHost(expId, fp, opts) {
  const host = el("div", {}, el("div", { class: "hint" }, "收益曲线加载中…"));
  const render = (epochId) =>
    fetchExperimentEquity(expId, fp, epochId)
      .then((payload) => {
        host.innerHTML = "";
        if (!opts?.mini && (payload.epochs || []).length > 1) {
          host.append(
            el(
              "div",
              { class: "epoch-switch" },
              ...payload.epochs.map((e) =>
                el(
                  "button",
                  {
                    class: `btn small${e === payload.epoch_id ? " primary" : ""}`,
                    onclick: () => {
                      host.innerHTML = "";
                      host.append(
                        el("div", { class: "hint" }, "收益曲线加载中…"),
                      );
                      render(e);
                    },
                  },
                  epochShort(e),
                ),
              ),
            ),
          );
        }
        host.append(equityChart(payload, opts));
        if (!opts?.mini) {
          const statsTable = cycleStatsTable(payload);
          if (statsTable) host.append(statsTable);
        }
      })
      .catch((error) => {
        host.innerHTML = "";
        host.append(
          el("div", { class: "hint" }, `收益曲线加载失败：${error.message}`),
        );
      });
  render(null);
  return host;
}

function fmtDateTick(date, withYear) {
  return withYear
    ? `${date.slice(2, 4)}/${date.slice(4, 6)}-${date.slice(6, 8)}`
    : `${date.slice(4, 6)}-${date.slice(6, 8)}`;
}

function rebaseBenchmarkToStrategyWindows(seriesList) {
  const bench = seriesList.find((s) => s.key === "benchmark");
  const strategies = seriesList.filter((s) => s.key !== "benchmark");
  if (!bench || !strategies.length) return seriesList;
  const benchDates = [...bench.cum.keys()].sort();
  const segments = [];
  for (const strategy of strategies) {
    const pts = strategy.dates.filter((day) => bench.cum.has(day));
    if (!pts.length) continue;
    const first = pts[0];
    const prior = benchDates.indexOf(first);
    const origin = prior > 0 ? 1 + bench.cum.get(benchDates[prior - 1]) : 1;
    const cum = new Map();
    const dd = new Map();
    let peak = 0;
    for (const day of pts) {
      const value = (1 + bench.cum.get(day)) / origin - 1;
      cum.set(day, value);
      peak = Math.max(peak, 1 + value);
      dd.set(day, (1 + value) / peak - 1);
    }
    const tag = CYCLE_SERIES_SHORT[strategy.key] || strategy.label;
    segments.push({
      ...bench,
      key: `benchmark:${strategy.key}`,
      label: strategies.length > 1 ? `沪深300（${tag}）` : bench.label,
      dates: pts,
      cum,
      dd,
      final: cum.get(pts[pts.length - 1]),
      dash: "6 4",
    });
  }
  return segments.length ? [...segments, ...strategies] : seriesList;
}

/* Server series are already compounded. The CSI 300 overlay is rebased to each
   strategy window so a Held-out line that starts at 0 is compared with a 300
   that also starts at 0 in that window. */
function equityChart(
  payload,
  { width = 680, height = 240, ddH = 90, mini = false, keys = null } = {},
) {
  const INK = themeInk();
  const colorOf = {
    valid: INK.validColor,
    test: INK.testColor,
    heldout: INK.heldoutColor,
    benchmark: INK.muted,
  };
  const wanted = (payload.series || []).filter(
    (s) => (s.dates || []).length && (!keys || keys.includes(s.key)),
  );
  if (!wanted.length) return el("div", { class: "hint" }, "暂无日度收益数据");
  const wantedDates = new Set(wanted.flatMap((s) => s.dates));
  const shown = [...wanted];
  const bench = payload.benchmark;
  if (
    bench &&
    (bench.dates || []).length &&
    bench.dates.some((d) => wantedDates.has(d))
  ) {
    shown.push(bench);
  }
  const mapped = shown.map((s) => ({
    key: s.key,
    label: s.label,
    final: s.final,
    dates: s.dates,
    cum: new Map(s.dates.map((d, i) => [d, s.cum[i]])),
    dd: new Map(s.dates.map((d, i) => [d, s.drawdown[i]])),
    color: colorOf[s.key] || INK.validColor,
    dash: s.key === "benchmark" ? "6 4" : null,
  }));
  const seriesList = rebaseBenchmarkToStrategyWindows(mapped);
  const dates = [...new Set(seriesList.flatMap((s) => s.dates))].sort();
  // Position-weight pane (EOD gross market value / equity), keyed like the
  // return series so identity carries across the linked panes.
  const exposureBy = payload.exposure || {};
  const expList = mini
    ? []
    : seriesList
        .filter(
          (s) =>
            !String(s.key).startsWith("benchmark") &&
            exposureBy[s.key] &&
            (exposureBy[s.key].dates || []).length,
        )
        .map((s) => {
          const e = exposureBy[s.key];
          return {
            key: s.key,
            color: s.color,
            long: new Map(e.dates.map((d, i) => [d, e.long[i]])),
            short: new Map(e.dates.map((d, i) => [d, (e.short || [])[i] || 0])),
            hasShort: (e.short || []).some((v) => v > 0.005),
          };
        });
  if (mini) ddH = 0;
  const showDD = ddH > 0;
  const showExp = expList.length > 0;
  const expH = showExp ? 64 : 0;
  const hasPanes = showDD || showExp;
  const padL = mini ? 44 : 52,
    padR = 12,
    padT = 8,
    gap = hasPanes ? 16 : 0;
  // With subplots the shared date labels sit BELOW them, so the main plot
  // needs only a slim bottom pad; standalone charts keep the label band.
  const padB = hasPanes ? 12 : mini ? 26 : 32;
  const labelBand = hasPanes ? 24 : 0;
  const totalH =
    height + (showDD ? ddH + gap : 0) + (showExp ? expH + gap : 0) + labelBand;
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
  // x ticks (≤7), year shown on the first tick and on year changes
  const tickEvery = Math.max(1, Math.ceil(dates.length / (mini ? 4 : 7)));
  let prevYear = null;
  // Date labels: below the lowest subplot when present (shared axis at the
  // figure bottom), otherwise a clear step below the main axis line.
  const panesBottom =
    height + (showDD ? gap + ddH : 0) + (showExp ? gap + expH : 0);
  const tickY = hasPanes ? panesBottom + 10 : padT + mainH + (mini ? 17 : 20);
  const lastTick = dates.length - 1;
  dates.forEach((d, i) => {
    // Render modulo ticks plus the final date; drop a modulo tick that would
    // crowd the end-anchored final label.
    if (i !== lastTick && (i % tickEvery !== 0 || lastTick - i < tickEvery / 2))
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
  // drawdown subplot
  let ddY = null;
  if (showDD) {
    const ddTop = height + gap;
    const ddLo = Math.min(
      -0.001,
      ...seriesList.flatMap((s) => [...s.dd.values()]),
    );
    ddY = (v) => ddTop + (v / ddLo) * (ddH - 14);
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
      if (!String(s.key).startsWith("benchmark")) {
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
    const expMax = Math.max(
      1,
      ...expList.flatMap((s) => [...s.long.values(), ...s.short.values()]),
    );
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
      if (s.hasShort) {
        const shortLine = pts
          .filter((d) => s.short.has(d))
          .map(
            (d, j) =>
              `${j ? "L" : "M"}${xOf(dates.indexOf(d)).toFixed(1)},${yExp(s.short.get(d)).toFixed(1)}`,
          )
          .join(" ");
        svg.push(
          `<path d="${shortLine}" fill="none" stroke="${s.color}" stroke-width="1.5" stroke-dasharray="4 3"/>`,
        );
      }
    }
  }
  // main lines (benchmark first so strategy lines sit on top) + endpoint dots
  for (const s of [...seriesList].sort(
    (a, b) =>
      (a.key === "benchmark" ? -1 : 0) - (b.key === "benchmark" ? -1 : 0),
  )) {
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
    if (!String(s.key).startsWith("benchmark")) {
      const lastDate = pts[pts.length - 1];
      svg.push(
        `<circle cx="${xOf(dates.indexOf(lastDate)).toFixed(1)}" cy="${yOf(s.cum.get(lastDate)).toFixed(1)}" r="3.5" fill="${s.color}" stroke="${INK.ring}" stroke-width="2"/>`,
      );
    }
  }
  // hover columns: one hit target per date spanning both plots, rich tooltip
  const step = dates.length > 1 ? plotW / (dates.length - 1) : plotW;
  dates.forEach((d, i) => {
    const lines = [fmtDate(d)];
    for (const s of seriesList) {
      if (!s.cum.has(d)) continue;
      const exposure = expList.find((entry) => entry.key === s.key);
      const expText =
        exposure && exposure.long.has(d)
          ? ` ｜ 仓位 ${(exposure.long.get(d) * 100).toFixed(1)}%${exposure.hasShort ? `（空 ${(exposure.short.get(d) * 100).toFixed(1)}%）` : ""}`
          : "";
      lines.push(
        `${s.label} 累计 ${(s.cum.get(d) * 100).toFixed(2)}% ｜ 回撤 ${(s.dd.get(d) * 100).toFixed(2)}%${expText}`,
      );
    }
    const x = i === 0 ? padL : xOf(i) - step / 2;
    const w = i === 0 || i === dates.length - 1 ? step / 2 : step;
    svg.push(
      `<rect class="xcol" x="${x.toFixed(1)}" y="${padT}" width="${Math.max(w, 1).toFixed(1)}" height="${totalH - padT - 4}" data-tip="${escapeHtml(lines.join("\n"))}"/>`,
    );
  });
  const wrap = el(
    "div",
    { class: "svg-chart" },
    chartLegend(
      (mini
        ? seriesList.filter((s) => !String(s.key).startsWith("benchmark"))
        : seriesList
      ).map((s) => ({
        color: s.color,
        label: mini
          ? `${CYCLE_SERIES_SHORT[s.key] || s.label} ${fmtPct(s.final)}`
          : `${s.label}: ${fmtPct(s.final)}`,
      })),
    ),
  );
  const svgHost = el("div", {});
  svgHost.innerHTML = `<svg viewBox="0 0 ${width} ${totalH}" xmlns="http://www.w3.org/2000/svg">${svg.join("")}</svg>`;
  wrap.append(svgHost);
  wrap.__rerender = () =>
    equityChart(payload, { width, height, ddH, mini, keys });
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

function fmtAmount(value) {
  const n = Number(value) || 0;
  if (Math.abs(n) >= 1e8) return `¥${(n / 1e8).toFixed(2)}亿`;
  if (Math.abs(n) >= 1e4) return `¥${(n / 1e4).toFixed(1)}万`;
  return `¥${n.toFixed(0)}`;
}

/* Single-series bar chart (no legend needed for one series); direct value
   labels when the set is small, tooltips always. */
function singleSeriesBarChart(
  rows,
  { width = 640, height = 200, fmt = fmtPct } = {},
) {
  const INK = themeInk();
  const color = INK.validColor; // categorical slot 1
  const values = rows
    .map((row) => row.value)
    .filter((v) => v !== null && v !== undefined);
  if (!rows.length || !values.length)
    return el("div", { class: "hint" }, "暂无数据");
  const signed = values.some((v) => v < 0);
  const maxAbs = niceCeil(Math.max(1e-9, ...values.map(Math.abs)));
  const padL = 56,
    padR = 10,
    padB = 30,
    padT = signed ? 8 : 18;
  const plotW = width - padL - padR,
    plotH = height - padT - padB;
  const zeroY = signed ? padT + plotH / 2 : padT + plotH;
  const scale = signed ? plotH / 2 : plotH;
  const yOf = (v) => zeroY - (v / maxAbs) * scale;
  const svg = [];
  for (const frac of signed ? [-1, -0.5, 0.5, 1] : [0.5, 1]) {
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
  const labelEvery = Math.max(1, Math.ceil(rows.length / 12));
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
  wrap.__rerender = () => singleSeriesBarChart(rows, { width, height, fmt });
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
        { class: "tile" },
        el("div", { class: "tile-label" }, tile.label),
        el("div", { class: `tile-value ${tile.cls || ""}` }, tile.value),
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
  const tradingMatch = hash.match(/^#\/trading\/(paper)$/);
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
    renderTradingPage(tradingMatch[1]);
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
    ...[
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
    ].filter(Boolean),
  );
  const container = el("div", {});
  const best = pickBestExperiment(payload.experiments);
  if (best)
    container.append(heroPanel(best), el("div", { class: "section-gap" }));
  container.append(
    el(
      "div",
      { class: "page-head" },
      el("h2", {}, "实验列表"),
      el(
        "span",
        { class: "sub" },
        "点击实验卡片查看 Epoch/Fold 结果、运行状态与 Agent Trace",
      ),
    ),
  );
  if (payload.experiments.length) {
    const grid = el("div", { class: "grid" });
    for (const item of payload.experiments) grid.append(experimentCard(item));
    container.append(grid);
  } else {
    container.append(
      el("div", { class: "empty" }, "还没有实验 —— 点右上角「新建实验」开始。"),
    );
  }
  $main.innerHTML = "";
  $main.append(container);
  pollTimer = setInterval(async () => {
    if (location.hash && location.hash !== "#/" && location.hash !== "#")
      return;
    try {
      await renderHomePageSilent();
    } catch {
      /* keep last view */
    }
  }, 5000);
}

async function renderHomePageSilent() {
  const payload = await api("/api/experiments");
  const grid = document.querySelector(".grid");
  if (!grid) return;
  const fresh = el("div", { class: "grid" });
  for (const item of payload.experiments) fresh.append(experimentCard(item));
  grid.replaceWith(fresh);
  const hero = document.getElementById("hero-panel");
  const best = pickBestExperiment(payload.experiments);
  // Only rebuild the hero when its content actually changed. Replacing it every
  // poll re-creates the equity host, whose epoch switcher would reset to the
  // default (latest) epoch and clobber the user's E1/E2 selection.
  if (hero && best && hero.__heroSig !== heroSignature(best))
    hero.replaceWith(heroPanel(best));
}

/* Everything heroPanel renders that can change between polls; when unchanged the
   panel (and its selected epoch) is left in place. */
function heroSignature(item) {
  const m = item.metrics || {};
  return [
    item.experiment_id,
    item.state,
    item.worker_alive,
    item.test_revealed,
    equityFingerprint(item),
    m.mean_test_sharpe,
    m.epoch_id,
  ].join("|");
}

/* Held-out graduation verdict (sealed until the reveal). */
function verdictBadge(verdict) {
  if (!verdict || !verdict.status) return null;
  const graduated = verdict.status === "graduated";
  const reasons = (verdict.reasons || []).join("、");
  return el(
    "span",
    {
      class: `badge state-${graduated ? "completed" : "failed"}`,
      title: graduated
        ? "Held-out 通过：超额收益 > 0、Sharpe > 0、回撤在限内"
        : `Held-out 未通过：${reasons || "见账本"}`,
    },
    graduated ? "graduated" : "discarded",
  );
}

/* Graduation term (b) beside the verdict badge: how many of the final Epoch's
   walk-forward transitions kept a positive excess return, and the count they
   had to reach. The failing reason itself rides in verdict.reasons. */
function walkForwardTerm(verdict) {
  const term = (verdict || {}).walk_forward;
  if (!term || !term.status) return null;
  if (term.status === "not_applicable")
    return el(
      "span",
      { class: "mode-note" },
      "走向前：本排程没有转移，裁决只看 Held-out",
    );
  const consistent = term.status === "consistent";
  return el(
    "span",
    {
      class: `badge state-${consistent ? "completed" : "failed"}`,
      title: `走向前一致性（${WALK_FORWARD_SOURCES[term.source] || term.source || "—"}）：${term.positive_excess}/${term.transitions} 个转移超额为正，需 ≥ ${term.required}`,
    },
    `走向前 ${term.positive_excess}/${term.transitions}`,
  );
}

/* Walk-forward per Epoch: without a Test stage every Fold after the Epoch's
   first opens with the previous Fold's frozen strategy replayed on this Fold's
   Validation window, and graduation term (b) counts how many of those kept a
   positive excess return. The counts come from the ledger, the rows from the
   same parent-control metrics the fold panel reads. */
function walkForwardPanel(detail) {
  const epochs = (detail.metrics_by_epoch || []).filter(
    (epoch) => (epoch.walk_forward || {}).transitions > 0,
  );
  if (!epochs.length) return null;
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", { class: "subsection-title" }, "走向前转移（父本对照）"),
  );
  for (const epoch of epochs) {
    const term = epoch.walk_forward || {};
    panel.append(
      el(
        "div",
        { class: "meta-line" },
        `${epochShort(epoch.epoch_id)} ｜ 来源 ${WALK_FORWARD_SOURCES[term.source] || term.source || "—"} ｜ 转移 ${term.transitions} ｜ 超额为正 ${term.positive_excess}/${term.transitions}`,
      ),
    );
    if (term.source !== "parent_control") continue;
    const rows = (detail.fold_returns || []).filter(
      (row) => row.epoch_id === epoch.epoch_id && row.parent_control,
    );
    if (!rows.length) continue;
    panel.append(
      el(
        "table",
        { class: "data" },
        el(
          "tr",
          {},
          el("th", {}, "Fold"),
          el("th", { title: "对照未完成时没有数字" }, "状态"),
          el("th", {}, "收益"),
          el("th", { title: "相对沪深300的超额收益" }, "超额"),
          el("th", {}, "Sharpe"),
          el("th", {}, "回撤"),
        ),
        ...rows.map((row) => {
          const control = row.parent_control || {};
          return el(
            "tr",
            {},
            el("td", {}, foldPeriodLabel(detail, row.fold_ref)),
            el("td", {}, control.status === "ok" ? "完成" : "失败"),
            el("td", { class: signCls(control.return) }, fmtPct(control.return)),
            el(
              "td",
              { class: signCls(control.excess_return) },
              fmtPct(control.excess_return),
            ),
            el("td", { class: signCls(control.sharpe) }, fmtSharpe(control.sharpe)),
            el("td", {}, fmtPct(control.max_drawdown)),
          );
        }),
      ),
    );
  }
  return panel;
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

function experimentCard(item) {
  const metrics = item.metrics || {};
  const total = item.total_sessions,
    done = item.completed_sessions ?? 0;
  const numericTotal = Number(total),
    numericDone = Number(done);
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
      item.current_session_label || item.current_session
        ? ` ｜ 当前 ${item.current_session_label || item.current_session}`
        : "",
      item.error ? ` ｜ ${item.error}` : "",
    ),
  );
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
  if (Number.isFinite(numericTotal) && numericTotal > 0) {
    const progressValue = Number.isFinite(numericDone)
      ? Math.min(numericTotal, Math.max(0, numericDone))
      : 0;
    const progressText = `${done}/${total}`;
    const complete =
      Number.isFinite(numericDone) && numericDone >= numericTotal;
    card.append(
      el(
        "progress",
        {
          class: `progress${complete ? " done" : ""}`,
          value: progressValue,
          max: numericTotal,
          "aria-label": `${item.experiment_id} 会话进度`,
          "aria-valuetext": `${progressText} 个会话`,
        },
        progressText,
      ),
      el("div", { class: "meta-line" }, `进度 ${done}/${total} 个会话`),
    );
  } else if (item.folds_recorded) {
    card.append(
      el(
        "div",
        { class: "meta-line" },
        `账本记录：${item.folds_recorded} 个 Fold ｜ ${item.heldout_recorded || 0} 个 held-out`,
      ),
    );
  }
  // Same component + order as the hero and detail pages (Held-out first).
  // Cumulative valid/test metrics are per-epoch (latest) — never mixed across
  // epochs, which re-run the same fold calendar.
  const epochTag = metrics.epoch_id
    ? `（${epochShort(metrics.epoch_id)}）`
    : "";
  card.append(
    statTilesRow([
      sealedMetricTile(
        item.test_revealed,
        "Held-out 收益",
        metrics.cum_heldout_return,
      ),
      sealedMetricTile(
        item.test_revealed,
        `累计测试收益${epochTag}`,
        metrics.cum_test_return,
      ),
      {
        label: `累计验证收益${epochTag}`,
        value: fmtPct(metrics.cum_valid_return),
        cls: signCls(metrics.cum_valid_return),
      },
    ]),
  );
  if ((item.fold_returns || []).length) {
    const fingerprint = equityFingerprint(item);
    const keepId = `equity-card-${item.experiment_id}`;
    const existing = document.getElementById(keepId);
    if (existing && existing.dataset.fp === fingerprint) {
      card.append(existing);
    } else {
      const host = equityHost(item.experiment_id, fingerprint, {
        width: 400,
        height: 130,
        mini: true,
      });
      host.id = keepId;
      host.dataset.fp = fingerprint;
      card.append(host);
    }
  }
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
              renderHomePageSilent();
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

/* Best-performing experiment hero: revealed test Sharpe when present,
   otherwise latest-epoch validation return. */
function pickBestExperiment(list) {
  const scored = list
    .filter((item) => (item.fold_returns || []).length)
    .map((item) => ({
      item,
      sharpe: item.metrics?.mean_test_sharpe ?? null,
      ret:
        item.metrics?.cum_test_return ?? item.metrics?.cum_valid_return ?? null,
    }))
    .filter((entry) => entry.sharpe !== null || entry.ret !== null);
  if (!scored.length) return null;
  scored.sort((a, b) => {
    if (a.sharpe !== null && b.sharpe !== null) return b.sharpe - a.sharpe;
    if (a.sharpe !== null) return -1;
    if (b.sharpe !== null) return 1;
    return b.ret - a.ret;
  });
  return scored[0].item;
}

/* Cache key: equity only changes when new records land (or a rerun replaces
   results — caught by the cumulative-return components). */
function equityFingerprint(item) {
  const metrics = item.metrics || {};
  return `${item.folds_recorded}|${item.heldout_recorded}|${metrics.cum_test_return}|${metrics.cum_valid_return}|${metrics.cum_heldout_return}`;
}

function heroPanel(item) {
  const metrics = item.metrics || {};
  const panel = el("div", { class: "panel hero", id: "hero-panel" });
  panel.__heroSig = heroSignature(item);
  const charts = el(
    "div",
    { class: "section-gap" },
    el("h4", {}, "日度累计收益 vs 沪深300（含回撤）"),
    equityHost(item.experiment_id, equityFingerprint(item), {
      width: 980,
      height: 240,
      ddH: 90,
    }),
  );
  panel.append(
    el(
      "div",
      { class: "control-bar" },
      el("span", { class: "hero-crown" }, "🏆"),
      el("h3", { style: "margin:0" }, experimentName(item.experiment_id)),
      stateBadge(item.state),
      el(
        "span",
        { class: "mode-note" },
        item.test_revealed
          ? "当前最佳实验（按测试期平均 Sharpe）"
          : "当前最佳实验（按验证期收益）",
      ),
    ),
    el(
      "div",
      { class: "section-gap" },
      statTilesRow([
        {
          ...sealedMetricTile(
            item.test_revealed,
            "Held-out 收益（最终样本外）",
            metrics.cum_heldout_return,
          ),
          cls: item.test_revealed
            ? `hero-key ${signCls(metrics.cum_heldout_return)}`
            : "",
        },
        sealedMetricTile(
          item.test_revealed,
          `累计测试收益${metrics.epoch_id ? `（${epochShort(metrics.epoch_id)}）` : ""}`,
          metrics.cum_test_return,
        ),
        {
          label: `累计验证收益${metrics.epoch_id ? `（${epochShort(metrics.epoch_id)}）` : ""}`,
          value: fmtPct(metrics.cum_valid_return),
          cls: signCls(metrics.cum_valid_return),
        },
        sealedMetricTile(
          item.test_revealed,
          "平均测试 Sharpe",
          metrics.mean_test_sharpe,
          fmtSharpe,
        ),
        { label: "已完成 Fold", value: String(item.folds_recorded ?? 0) },
      ]),
    ),
    charts,
  );
  return panel;
}

function confirmRevealTests(experimentId) {
  showModal(
    "揭示测试结果",
    el(
      "div",
      {},
      el(
        "p",
        {},
        "Fold Test 在各 Fold 冻结时是样本外评估，之后仅通过受控的 compact 指标投影成为 Meta-development 反馈；Held-out 才是唯一最终未触碰评估。",
      ),
      el(
        "p",
        {},
        "人工揭示会打开不受控的明细反馈通道，因此揭示后本实验封存：不能再批准会话、重跑、回滚、逐 Step 放行或注入任何指令。",
      ),
      el("p", {}, "查看/停止/删除仍然可用。此操作不可撤销。"),
    ),
    [
      el("button", { class: "btn", onclick: closeModal }, "取消"),
      el(
        "button",
        {
          class: "btn danger",
          onclick: () =>
            sendControlAction(
              experimentId,
              { action: "reveal_test_results" },
              "测试结果已揭示，实验已封存",
              { modal: true, reload: true },
            ),
        },
        "揭示并封存",
      ),
    ],
  );
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

let createSchema = null;

async function openCreateModal() {
  let schema;
  try {
    schema = await api("/api/parameter-schema");
  } catch (error) {
    toast(error.message, true);
    return;
  }
  createSchema = schema;
  const hasPeriodOptions = Object.keys(schema.period_options || {}).length > 0;
  const inputs = new Map();
  const body = el("div", {});
  // Validation errors surface at the TOP of the (scrollable) modal body.
  const errorBox = el("div", {});
  body.append(errorBox);
  body.append(
    el(
      "p",
      { class: "hint" },
      hasPeriodOptions
        ? "所有参数均有默认值。周期从交易日历自动生成，仅列出数据完整、可回测的周期；切换 Fold 周期后选项与推荐值随之更新。任一周期字段也接受显式区间 20220101..20251231。"
        : "所有参数均有默认值；仅实验名与周期标签必填。周期标签格式随 Fold 周期而定：quarter → 2024Q1，month → 202401，" +
            "week → 周一日期 20240108，year → 2024；任一周期字段也接受显式区间 20260101..20260630。" +
            "策略按固定周期在固定推理时间运行；推理时间使用 Asia/Shanghai 24 小时制，并始终遵守 PIT 可见性。",
    ),
  );
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
  // Period selects depend on the fold cadence: repopulate options + suggested
  // defaults whenever fold_period changes.
  const cadenceEntry = inputs.get("fold_period");
  if (cadenceEntry && hasPeriodOptions) {
    repopulatePeriodSelects(inputs);
    cadenceEntry.input.addEventListener("change", () =>
      repopulatePeriodSelects(inputs),
    );
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

const PERIOD_FIELD_KEYS = [
  "development_first_period",
  "development_last_period",
  "heldout_first_period",
  "heldout_last_period",
];

function repopulatePeriodSelects(inputs) {
  const cadence = inputs.get("fold_period").input.value;
  const options = (createSchema.period_options || {})[cadence] || [];
  const defaults = (createSchema.period_defaults || {})[cadence] || {};
  for (const key of PERIOD_FIELD_KEYS) {
    const entry = inputs.get(key);
    if (!entry || entry.field.type !== "period" || !entry.input) continue;
    const previous = entry.input.value;
    entry.input.innerHTML = "";
    for (const label of options) {
      // Cadence labels (2024Q1 / 2024 / 202401) render as-is; an explicit
      // YYYYMMDD..YYYYMMDD range renders as dates, value stays the raw label.
      const option = el("option", { value: label }, fmtPeriodRange(label));
      entry.input.append(option);
    }
    const wanted = options.includes(previous) ? previous : defaults[key];
    if (wanted && options.includes(wanted)) entry.input.value = wanted;
  }
  updateValidationHint(inputs);
}

/* Spell out what the development window becomes: one regular Fold per period
   of the window with a 元学习 before each of them by default, or rolling
   Fold → Test pairs once the Test stage is switched on. */
function updateValidationHint(inputs) {
  const first = inputs.get("development_first_period");
  const last = inputs.get("development_last_period");
  if (!first || first.field.type !== "period" || !first.input) return;
  const cadence = inputs.get("fold_period").input.value;
  const options = (createSchema.period_options || {})[cadence] || [];
  const stage = inputs.get("test_stage");
  if (!first.__hint) {
    first.__hint = el("div", { class: "help derived-hint" });
    first.input.parentElement.append(first.__hint);
    const refresh = () => updateValidationHint(inputs);
    first.input.addEventListener("change", refresh);
    if (last && last.input) last.input.addEventListener("change", refresh);
    if (stage && stage.input) stage.input.addEventListener("change", refresh);
  }
  const start = first.input.value;
  const end = last && last.input ? last.input.value : "";
  const rolling = Boolean(stage && stage.input && stage.input.checked);
  if (!start || !end) {
    first.__hint.textContent = "";
    return;
  }
  const index = options.indexOf(start);
  if (!rolling) {
    const last = options.indexOf(end);
    const folds = index >= 0 && last >= index ? last - index + 1 : 0;
    const count = folds ? `${folds} 个常规 Fold` : "每个周期一个常规 Fold";
    first.__hint.textContent = `↳ ${fmtPeriodRange(start)} ～ ${fmtPeriodRange(end)} 按周期切成 ${count}：每个 Fold 只验证本周期、没有测试区间，每个 Fold 前先做一次元学习；末个 Fold 冻结后进入 Held-out 裁决`;
    return;
  }
  first.__hint.textContent =
    index >= 0 && index + 1 < options.length
      ? `↳ 首个 Fold：验证区间 ${fmtPeriodRange(options[index])} → 测试区间 ${fmtPeriodRange(options[index + 1])}（首个周期只做验证，之后逐周期滚动）`
      : "↳ Test 阶段需要至少两个 Development 周期";
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

let detailView = null; // {experimentId, detail, listHost, rightHost, selectedKey}
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
  if (!selectedKey) {
    selectedKey =
      status.session_key ||
      (
        sessions.find((session) => !session.record && !session.records) ||
        sessions[sessions.length - 1] ||
        {}
      ).key;
  }
  const head = el(
    "div",
    { class: "page-head" },
    el(
      "h2",
      {},
      el("a", { class: "exp-back", href: "#/" }, "← 实验"),
      experimentName(detail.experiment_id, { link: false }),
      experimentBadges(
        stateBadge(detail.state),
        detail.kind === "hitl" && detail.test_revealed
          ? el(
              "span",
              {
                class: "badge state-waiting_user",
                title:
                  "测试/Held-out 结果已揭示：实验已封存，不能再批准、重跑、回滚或注入指令",
              },
              "已揭示测试（封存）",
            )
          : null,
      ),
    ),
    el(
      "div",
      { class: "sub" },
      `进度 ${detail.completed_sessions ?? 0}/${detail.total_sessions ?? "?"}`,
      ` ｜ Skills ${Number(detail.skills && detail.skills.count) || 0} 项`,
      detail.state === "unreadable" && detail.error
        ? ` ｜ ${detail.error}`
        : "",
      status.error ? ` ｜ 错误：${status.error}` : "",
      detail.worker_alive && status.environment_stage
        ? ` ｜ ${formatStageLine(status, { elapsed: false })}`
        : "",
      // A worker-recorded analysis error is only current while that worker
      // lives; stale failures are visible per fold in the analysis section.
      detail.worker_alive && status.analysis_error
        ? ` ｜ 分析：${status.analysis_error}`
        : "",
    ),
  );
  const container = el("div", {});
  let barHost = null;
  if (detail.params && Object.keys(detail.params).length) {
    head.querySelector("h2").append(
      el(
        "button",
        {
          class: "btn small",
          style: "margin-left:0.4rem",
          onclick: () => openParamsModal(detail),
        },
        "创建参数",
      ),
    );
  }
  if (detail.kind === "hitl" && detail.control && !detail.test_revealed) {
    head.querySelector("h2").append(
      el(
        "button",
        {
          class: "btn small",
          style: "margin-left:0.4rem",
          onclick: () => confirmRevealTests(detail.experiment_id),
        },
        "揭示测试结果",
      ),
    );
  }
  container.append(head);
  if (detail.kind === "hitl") {
    barHost = controlBar(detail);
    container.append(barHost);
  }
  if ((detail.fold_returns || []).length) {
    const metrics = detail.metrics || {};
    const sharpe = metrics.mean_test_sharpe;
    const charts = el(
      "div",
      { class: "section-gap" },
      el("h4", {}, "日度累计收益 vs 沪深300（含回撤）"),
      equityHost(detail.experiment_id, equityFingerprint(detail), {
        width: 980,
        height: 240,
        ddH: 90,
      }),
    );
    // Tile order standardized with the homepage hero: Held-out → test → valid.
    const epochTag = metrics.epoch_id
      ? `（${epochShort(metrics.epoch_id)}）`
      : "";
    container.append(
      el(
        "div",
        { class: "panel section-gap" },
        statTilesRow([
          sealedMetricTile(
            detail.test_revealed,
            "Held-out 收益（最终样本外）",
            metrics.cum_heldout_return,
          ),
          sealedMetricTile(
            detail.test_revealed,
            `累计测试收益${epochTag}`,
            metrics.cum_test_return,
          ),
          {
            label: `累计验证收益${epochTag}`,
            value: fmtPct(metrics.cum_valid_return),
            cls: signCls(metrics.cum_valid_return),
          },
          sealedMetricTile(
            detail.test_revealed,
            "平均测试 Sharpe",
            sharpe,
            fmtSharpe,
          ),
          {
            label: "会话进度",
            value: `${detail.completed_sessions ?? 0} / ${detail.total_sessions ?? "?"}`,
          },
        ]),
        charts,
      ),
    );
  }
  const walkForward = walkForwardPanel(detail);
  if (walkForward) container.append(walkForward);
  container.append(stepTreePanel(detail));
  container.append(mountedMemoryPanel(detail));
  const layout = el("div", { class: "detail section-gap" });
  // Panels read the global detailView (held-out equity/style card): set it
  // BEFORE building them, or the first render of a detail page silently
  // skips those blocks (and could chart the previous experiment's data).
  detailView = {
    experimentId,
    detail,
    listHost: null,
    rightHost: null,
    barHost: null,
    selectedKey,
  };
  const listHost = sessionListPanel(detail, selectedKey);
  const rightHost = sessionDetailPanel(detail, selectedKey);
  detailView.listHost = listHost;
  detailView.rightHost = rightHost;
  detailView.barHost = barHost;
  layout.append(listHost, rightHost);
  container.append(layout);
  $main.innerHTML = "";
  $main.append(container);
  pollTimer = setInterval(async () => {
    try {
      const fresh = await api(
        `/api/experiments/${encodeURIComponent(experimentId)}/status`,
      );
      const freshState = fresh.state;
      const raw = fresh.status || {};
      const badge = document.querySelector(".page-head .badge");
      if (badge && !badge.className.includes(`state-${freshState}`))
        route(true); // fetch fresh detail on state change
      else if (
        raw.session_key &&
        raw.session_key !== (status.session_key || null)
      )
        route(true);
      else if (String(raw.run_ref || "") !== String(status.run_ref || ""))
        route(true);
      else if (
        String(raw.question_key || "") !== String(status.question_key || "")
      )
        route(true);
      else if (String(raw.step_index || "") !== String(status.step_index || ""))
        route(true);
      // llm_call ↔ tool_call stage flips must not rebuild the page: the live
      // Trace panel already polls status, and route(true) flashes "加载中…".
    } catch {
      /* transient */
    }
  }, 4000);
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
            el("div", { class: "hint", style: "margin:0" }, field.key),
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
            el("div", { class: "hint", style: "margin:0" }, field.key),
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
    el(
      "p",
      { class: "hint" },
      "创建时显式设置的参数在前；其余按创建表单默认生效（灰色）。运行期实际生效值以 run manifest / snapshot manifest 为准。",
    ),
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

function controlBar(detail) {
  const id = detail.experiment_id;
  const control = detail.control || { mode: "manual", request: null };
  const state = detail.state;
  const alive = detail.worker_alive;
  const send = (payload, note) => sendControlAction(id, payload, note);
  const bar = el("div", { class: "panel control-bar section-gap" });
  bar.append(el("span", { class: "mode-note" }, "运行模式："));
  const modeSelect = el(
    "select",
    {
      onchange: () =>
        send(
          { action: "set_mode", mode: modeSelect.value },
          `模式已切换为 ${modeSelect.value}`,
        ),
    },
    el("option", { value: "manual" }, "逐会话批准"),
    el("option", { value: "step" }, "逐 Step 批准（最细）"),
    el("option", { value: "auto" }, "自动运行（连续执行）"),
  );
  modeSelect.value = control.mode;
  bar.append(modeSelect);
  if (control.request === "pause")
    bar.append(el("span", { class: "badge state-paused" }, "已请求暂停"));
  if (control.request === "stop")
    bar.append(el("span", { class: "badge state-stopped" }, "已请求停止"));
  if (control.skip_to_heldout)
    bar.append(
      el("span", { class: "badge state-waiting_user" }, "已请求提前收官"),
    );
  if (control.restart_pending)
    bar.append(
      el("span", { class: "badge state-waiting_user" }, "已请求会话边界重启"),
    );
  bar.append(el("span", { class: "spacer" }));
  // Early finish: skip the remaining folds and jump straight to Held-out with
  // the latest frozen artifact (needs at least one recorded fold).
  if (detail.state !== "completed") {
    if (!control.skip_to_heldout && (detail.folds_recorded || 0) > 0) {
      bar.append(
        el(
          "button",
          {
            class: "btn",
            onclick: () => {
              showModal(
                "提前进入 Held-out",
                el(
                  "p",
                  {},
                  "跳过剩余全部 Fold（及后续元学习），直接以最新冻结策略进入 Held-out 冻结测试。已完成的 Fold 不受影响；人工控制模式下 Held-out 会话仍需批准。确定？",
                ),
                [
                  el("button", { class: "btn", onclick: closeModal }, "取消"),
                  el(
                    "button",
                    {
                      class: "btn primary",
                      onclick: () => {
                        closeModal();
                        send(
                          { action: "skip_to_heldout" },
                          "已请求提前进入 Held-out",
                        );
                      },
                    },
                    "确认提前收官",
                  ),
                ],
              );
            },
          },
          "提前收官 → Held-out",
        ),
      );
    } else if (control.skip_to_heldout) {
      bar.append(
        el(
          "button",
          {
            class: "btn",
            onclick: () =>
              send({ action: "cancel_skip_to_heldout" }, "已取消提前收官"),
          },
          "取消提前收官",
        ),
      );
    }
  }
  if (alive) {
    if (control.request === "pause") {
      bar.append(
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
      bar.append(
        el(
          "button",
          {
            class: "btn",
            onclick: () =>
              send({ action: "pause" }, "将在当前 Fold 结束后暂停"),
          },
          "暂停",
        ),
      );
    }
    bar.append(
      el(
        "button",
        {
          class: "btn",
          onclick: () => send({ action: "stop" }, "将在当前会话结束后停止"),
        },
        "停止",
      ),
    );
    bar.append(
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
                "立即向 worker 发送 SIGTERM。未落账的当前会话会被中断并撤销批准；恢复后先回到待批准状态，可重新编辑、预览并批准 Prompt。确定？",
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
                          }${
                            result.approval_revoked_session
                              ? "；未完成会话已退回待批准"
                              : ""
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
    bar.append(
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
                  "立即重启：终止当前 worker 并按账本恢复运行，已完成会话保留，被中断的会话整体重跑。宽限内没有退出的 worker 会被强制终止。",
                ),
                el(
                  "p",
                  {},
                  "会话边界重启：worker 先把当前会话跑完记账，再在原地换用新代码继续，不丢进行中的 Fold。",
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
    bar.append(
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
  return bar;
}

function sessionListPanel(detail, selectedKey) {
  const panel = el(
    "div",
    { class: "panel" },
    el("h4", {}, "会话（元学习 / Fold / Held-out）"),
  );
  const list = el("div", { class: "session-list" });
  const status = detail.status || {};
  let currentEpoch = null;
  for (const session of detail.sessions || []) {
    if (session.epoch_id !== currentEpoch && session.kind !== "heldout") {
      currentEpoch = session.epoch_id;
      list.append(
        el(
          "div",
          { class: "epoch-head" },
          `Epoch ${String(currentEpoch).replace("epoch_", "")}`,
        ),
      );
    }
    const isDone = Boolean(session.record || (session.records || []).length);
    const isCurrent = status.session_key === session.key && detail.worker_alive;
    const isWaiting = isCurrent && detail.state === "waiting_user";
    const dotClass = isDone
      ? "done"
      : isWaiting
        ? "waiting"
        : isCurrent
          ? "running"
          : "pending";
    const validReturn =
      session.record && session.record.validation_result
        ? session.record.validation_result.total_return
        : null;
    const stateText =
      isCurrent && status.state === "waiting_step_user"
        ? `Step ${status.step_index ?? "?"} 待批准`
        : isCurrent && status.state === "waiting_user_reply"
          ? "提问待答复"
          : isWaiting
            ? "待批准"
            : isCurrent && detail.state === "paused"
              ? "已暂停"
              : isCurrent &&
                  (detail.state === "failed" ||
                    detail.state === "interrupted" ||
                    detail.state === "terminated" ||
                    detail.state === "stopped")
                ? STATE_LABELS[detail.state] || detail.state
                : isCurrent
                  ? formatStageLine(status, { elapsed: false }) || "运行中"
                  : "";
    const ret =
      session.kind === "fold" || session.kind === "meta_learning"
        ? foldDurationNode(
            detail,
            session,
            session.kind === "fold" &&
              validReturn !== null &&
              validReturn !== undefined
              ? fmtPct(validReturn)
              : stateText,
            session.kind === "fold" &&
              validReturn !== null &&
              validReturn !== undefined
              ? numClass(validReturn)
              : "",
          )
        : validReturn !== null && validReturn !== undefined
          ? el("span", { class: numClass(validReturn) }, fmtPct(validReturn))
          : el("span", {}, stateText);
    ret.classList.add("ret");
    const item = el(
      "div",
      {
        class: `session-item${session.kind === "heldout" ? " phase-head" : ""}${session.key === selectedKey ? " selected" : ""}`,
        "data-key": session.key,
        onclick: () => {
          location.hash = `#/exp/${encodeURIComponent(detail.experiment_id)}/${sessionKeyToUrl(session.key)}`;
        },
      },
      el("span", { class: `dot ${dotClass}` }),
      el(
        "span",
        { class: "label" },
        sessionListLabel(session),
      ),
      ret,
    );
    list.append(item);
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
  const isCurrent = status.session_key === session.key;
  const running =
    isCurrent && detail.worker_alive && ACTIVE_SESSION_STATES.has(detail.state);
  const runningEnvironment =
    isCurrent && detail.worker_alive && detail.state === "running_heldout";
  const waiting = isCurrent && detail.state === "waiting_user";
  const done = Boolean(session.record || (session.records || []).length);

  // Directive editor for sessions that have not run yet or await approval.
  if (detail.kind === "hitl" && (!done || waiting) && !running) {
    panel.append(directivePanel(detail, session, waiting));
  }
  const preparing = isPrepEnvironment(status, detail.state);
  if (
    (running && preparing) ||
    runningEnvironment ||
    (isCurrent &&
      detail.worker_alive &&
      preparing &&
      !waiting &&
      !done)
  )
    panel.append(environmentStagePanel(detail));
  if (running && !preparing)
    panel.append(
      askUserPanel(detail, session),
      stepGatePanel(detail, session),
      liveTracePanel(detail, session),
    );
  if (
    session.kind === "fold" ||
    session.kind === "meta_learning" ||
    session.kind === "heldout"
  )
    panel.append(injectMessagePanel(detail, session));
  if (session.kind === "fold" && done) {
    const resultPanel = foldResultPanel(detail, session);
    // The ledger can appear while post-Fold analysis is still running, briefly
    // leaving the live Trace card above the result card. Space that transition
    // exactly like the settled layout.
    if (panel.children.length) resultPanel.classList.add("section-gap");
    panel.append(resultPanel);
    // The LLM strategy review gets its own card, peer to the fold result.
    panel.append(
      analysisPanel(
        detail.experiment_id,
        session.epoch_id,
        session.fold_ref || (session.record || {}).fold_ref,
      ),
    );
    const recordedFolds = (detail.sessions || []).filter(
      (s) => s.kind === "fold" && s.record,
    );
    if (
      detail.kind === "hitl" &&
      recordedFolds.length &&
      recordedFolds[recordedFolds.length - 1].key === session.key
    ) {
      panel.append(rerunPanel(detail, session));
    } else if (
      detail.kind === "hitl" &&
      recordedFolds.some((s) => s.key === session.key)
    ) {
      // Any earlier recorded fold can become the frontier again via rollback.
      panel.append(rollbackPanel(detail, session));
    }
  }
  if (session.kind === "meta_learning" && done)
    panel.append(metaResultPanel(detail, session));
  if (session.kind === "heldout" && done)
    panel.append(heldoutPanel(detail, session));
  if (done && session.record && session.record.run_ref) {
    const statsHost = el("div", {});
    panel.append(
      el(
        "div",
        { class: "panel section-gap" },
        session.kind === "fold"
          ? el(
              "div",
              { class: "section-gap" },
              el(
                "button",
                {
                  class: "btn",
                  onclick: () => openInitialPrompt(detail, session),
                },
                "查看初始 Prompt（实际运行）",
              ),
            )
          : null,
        statsHost,
        traceReplayNode(detail.experiment_id, session.record.run_ref, detail),
      ),
    );
    (async () => {
      try {
        const stats = await api(
          `/api/experiments/${encodeURIComponent(detail.experiment_id)}/trace/stats?run_id=${encodeURIComponent(session.record.run_ref)}`,
        );
        statsHost.append(statsChipsRow(stats));
      } catch {
        /* trace may be absent for legacy runs */
      }
    })();
  }
  if (!done && !running && !runningEnvironment && !waiting) {
    const idleLabel =
      isCurrent && detail.worker_alive
        ? STATE_LABELS[detail.state] || detail.state
        : isCurrent
          ? STATE_LABELS[detail.state] || "该会话已中断。"
          : "该会话尚未开始。";
    if (!preparing || !isCurrent || !detail.worker_alive) {
      panel.append(
        el(
          "div",
          { class: "panel section-gap" },
          el("div", { class: "empty" }, idleLabel),
        ),
      );
    }
  }
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

function directivePanel(detail, session, waiting) {
  const control = detail.control || { directives: {}, approved_sessions: [] };
  const isMeta = session.kind === "meta_learning";
  // A meta session with no per-session override inherits the experiment-level
  // directive from creation; prefill it so it never needs retyping.
  const inherited = isMeta
    ? String((detail.params || {}).meta_learning_directive || "")
    : "";
  const foldDefault =
    session.kind === "fold" || isMeta
      ? String((detail.params || {}).fold_exploration_directive || "").trim()
      : "";
  const existing = (control.directives || {})[session.key] ?? "";
  const approved = (control.approved_sessions || []).includes(session.key);
  const textarea = el("textarea", {
    class: "directive",
    placeholder: foldDefault
      ? "可选：仅为本 Fold 追加更具体的局部假设……"
      : "可选：为该会话注入研究方向 / 优化假设……",
  });
  textarea.value = existing || inherited;
  const panel = el(
    "div",
    { class: "panel" },
    el(
      "h4",
      {},
      isMeta
        ? "元学习指令（当前阶段）"
        : session.kind === "heldout"
          ? "Held-out 启动"
          : "本 Fold 研究者指令",
    ),
  );
  if (session.kind !== "heldout") {
    if (isMeta && inherited && !existing) {
      panel.append(
        el(
          "div",
          { class: "hint" },
          "已预填实验级元学习探索方向；不修改则按原方向执行，可编辑覆盖当前元学习阶段。",
        ),
      );
    }
    if (foldDefault) {
      panel.append(
        el(
          "details",
          { class: "section-gap" },
          el(
            "summary",
            { class: "hint" },
            "已自动注入实验级默认 Fold 探索方向（Meta 与 Fold 共用）",
          ),
          el(
            "div",
            { class: "markdown section-gap", style: "white-space:pre-wrap" },
            foldDefault,
          ),
        ),
      );
    }
    panel.append(
      textarea,
      el(
        "div",
        { class: "hint warn" },
        "指令会注入系统提示词并记入账本。已完成 Fold 的 Test 只由系统向 Meta 投影 compact 指标；请勿人工写入 Test/Held-out 明细或具体日历日期，以免绕过受控反馈边界。",
      ),
    );
    if ((detail.control || {}).mode === "auto") {
      panel.append(
        el(
          "div",
          { class: "hint" },
          "自动模式不会等待批准。点「保存指令」才会写入本会话；须在该会话启动前保存。",
        ),
      );
    }
  }
  const buttons = el("div", { class: "control-bar section-gap" });
  const send = (payload, note) =>
    sendControlAction(detail.experiment_id, payload, note);
  if (session.kind !== "heldout") {
    if (session.kind === "fold" || session.kind === "meta_learning") {
      buttons.append(
        el(
          "button",
          {
            class: "btn",
            onclick: () => openPromptEditor(detail, session),
          },
          "编辑额外用户指令",
        ),
      );
    }
    buttons.append(
      el(
        "button",
        {
          class: "btn",
          onclick: () =>
            openPromptPreview(detail, session, textarea.value, {
              approved,
              waiting,
              send,
            }),
        },
        "预览完整系统提示词",
      ),
    );
    buttons.append(
      el(
        "button",
        {
          class:
            (detail.control || {}).mode === "auto" && !approved
              ? "btn primary"
              : "btn",
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
    );
    if ((detail.control?.prompt_overrides || {})[session.key]) {
      buttons.append(
        el("span", { class: "badge state-waiting_user" }, "已设置额外用户指令"),
      );
    }
  }
  if ((detail.control || {}).mode !== "auto" && !approved) {
    buttons.append(
      el(
        "button",
        {
          class: "btn primary",
          onclick: () =>
            send(
              {
                action: "approve",
                session_key: session.key,
                directive: textarea.value,
              },
              "已批准，会话即将启动",
            ),
        },
        waiting ? "批准并启动" : "预先批准",
      ),
    );
  } else if (approved) {
    buttons.append(el("span", { class: "badge state-completed" }, "已批准"));
  }
  panel.append(buttons);
  // Pre-fold GPU allocation: live nvidia-smi inventory + per-session count.
  if (session.kind === "fold" && !session.record)
    panel.append(gpuAllocationRow(detail, session, send));
  if (waiting)
    panel.append(
      el(
        "div",
        { class: "hint" },
        "worker 正在等待此会话的批准。建议先预览完整系统提示词，确认注入内容无误后再批准。",
      ),
    );
  return panel;
}

/* GPU status + per-fold allocation picker, shown at the fold approval gate.
   The chosen count rides in control.gpu_counts[session_key]; the sandbox's
   "auto" selector then picks that many GPUs by free memory at start, so rows
   are ranked by free memory, the top N are marked as the likely allocation,
   and each bar tracks FREE memory (longer = more headroom). */
function gpuAllocationRow(detail, session, send) {
  const current = ((detail.control || {}).gpu_counts || {})[session.key];
  const experimentDefault = Number((detail.params || {}).gpu_count || 1);
  const wrap = el(
    "div",
    { class: "panel section-gap" },
    el("h4", { class: "subsection-title" }, "本 Fold GPU 分配"),
    el(
      "div",
      { class: "hint" },
      "批准前可查看实时资源并为本 Fold 沙箱设定 GPU 数；具体设备仍按空闲显存自动挑选，蓝条越长表示剩余显存越多。",
    ),
  );
  const statusHost = el(
    "div",
    {},
    el("div", { class: "hint" }, "GPU 状态加载中…"),
  );
  const stamp = el("span", { class: "hint", style: "margin-left:auto" });
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
              ? `本 Fold 将分配 ${select.value} 块 GPU`
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
            picked ? el("span", { class: "gpu-pick-badge" }, "将分配") : null,
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
async function sendControlAction(
  experimentId,
  payload,
  note,
  { modal = false, reload = false } = {},
) {
  try {
    const result = await api(
      `/api/experiments/${encodeURIComponent(experimentId)}/control`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      },
    );
    if (note) toast(typeof note === "function" ? note(result) : note);
    if (modal) closeModal();
    // Actions that change the whole page structure (e.g. sealing) need a full
    // re-render: refreshDetail() only swaps the control bar and session list,
    // leaving the page head (seal badge) and other panels stale.
    if (reload) route();
    else refreshDetail();
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

/* Extra user-instruction editor: does not replace the runtime system prompt. */
function openPromptEditor(detail, session) {
  const existing = (detail.control?.prompt_overrides || {})[session.key] || "";
  const editor = el("textarea", {
    class: "directive prompt-editor",
    spellcheck: "false",
  });
  editor.value = existing;
  const send = (payload, note) =>
    sendControlAction(detail.experiment_id, payload, note, { modal: true });
  const footer = [el("button", { class: "btn", onclick: closeModal }, "取消")];
  if (existing) {
    footer.push(
      el(
        "button",
        {
          class: "btn danger",
          onclick: () =>
            send(
              {
                action: "set_prompt_override",
                session_key: session.key,
                directive: "",
              },
              "已清除额外用户指令",
            ),
        },
        "清除额外指令",
      ),
    );
  }
  footer.push(
    el(
      "button",
      {
        class: "btn primary",
        onclick: () =>
          send(
            {
              action: "set_prompt_override",
              session_key: session.key,
              directive: editor.value,
            },
            "已保存额外用户指令",
          ),
      },
      "保存额外用户指令",
    ),
  );
  showModal(
    `编辑额外用户指令 — ${sessionDisplayKey(session)}`,
    el(
      "div",
      {},
      el(
        "div",
        { class: "hint warn" },
        "这段文字是额外用户指令，不会替换运行时系统提示词；build_system_prompt 仍会装配稳定合同与动态事实。请勿写入 Test/Held-out 明细或焊接的日历日期。",
      ),
      editor,
    ),
    footer,
    "prompt-modal",
  );
}

/* Review-then-approve: assemble the session's system prompt (with the draft
   directive embedded) for inspection before the session is allowed to start. */
async function openPromptPreview(
  detail,
  session,
  directive,
  { approved, waiting, send },
) {
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
  if ((detail.control || {}).mode !== "auto" && !approved) {
    footer.push(
      el(
        "button",
        {
          class: "btn primary",
          onclick: () => {
            closeModal();
            send(
              { action: "approve", session_key: session.key, directive },
              "已批准，会话即将启动",
            );
          },
        },
        waiting ? "确认无误，批准并启动" : "确认无误，预先批准",
      ),
    );
  }
  showModal(
    `系统提示词预览 — ${sessionDisplayKey(session)}`,
    el(
      "div",
      {},
      el("div", { class: "hint" }, data.note),
      el(
        "pre",
        {
          class: "code-view section-gap",
          style: "white-space:pre-wrap; max-height:58vh",
        },
        data.prompt,
      ),
      el(
        "div",
        { class: "hint" },
        `共 ${data.prompt.length} 字符。修改指令请关闭后在指令框编辑，再重新预览。`,
      ),
    ),
    footer,
  );
}

/* The prompt a completed fold session ACTUALLY started with: the session_start
   event recorded in its trace (ground truth, unlike the pre-session assembled
   preview, and unaffected by later code changes). */
async function openInitialPrompt(detail, session) {
  let data;
  try {
    data = await api(
      `/api/experiments/${encodeURIComponent(detail.experiment_id)}` +
        `/folds/${encodeURIComponent(session.epoch_id)}/${encodeURIComponent(session.fold_ref)}/initial-prompt`,
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
        { class: "code-view", style: "white-space:pre-wrap; max-height:46vh" },
        message.content || "",
      ),
    ),
  );
  showModal(
    `初始 Prompt（实际运行）— ${sessionDisplayKey(session)}`,
    el(
      "div",
      {},
      el(
        "div",
        { class: "hint" },
        `来自本 Fold 运行 trace 的会话起始事件 ｜ run ${data.run_ref || "?"}`,
      ),
      ...blocks,
    ),
    [el("button", { class: "btn", onclick: closeModal }, "关闭")],
    "prompt-modal",
  );
}

/* ask_user tool: when the Agent pauses on a question (state=waiting_user_reply),
   show it and send the researcher's reply (empty reply = proceed, Agent decides).
   The wait is excluded from the Agent's reasoning budget. */
function currentStrategyDownloadNode(detail) {
  const host = el(
    "span",
    {},
    el(
      "button",
      {
        class: "btn",
        disabled: "",
        title: "正在确认是否已有正式验证 Step 快照",
      },
      "策略快照检查中…",
    ),
  );
  api(
    `/api/experiments/${encodeURIComponent(detail.experiment_id)}/current-step`,
  )
    .then((payload) => {
      host.innerHTML = "";
      if (payload.available) {
        host.append(
          el(
            "a",
            {
              class: "btn",
              href: `/api/experiments/${encodeURIComponent(detail.experiment_id)}/current-step/source.zip`,
              title: "下载最近一次正式验证 Step 保存的只读策略快照",
            },
            "下载当前策略",
          ),
        );
      } else {
        host.append(
          el(
            "button",
            {
              class: "btn",
              disabled: "",
              title: payload.reason || "尚无正式验证 Step 快照",
            },
            "暂无策略快照",
          ),
        );
      }
    })
    .catch(() => {
      host.innerHTML = "";
      host.append(
        el("button", { class: "btn", disabled: "" }, "策略快照不可用"),
      );
    });
  return host;
}

function askUserPanel(detail, session) {
  const status = detail.status || {};
  const question = String(status.question || "");
  const questionKey = String(status.question_key || "");
  if (
    detail.kind !== "hitl" ||
    status.state !== "waiting_user_reply" ||
    status.session_key !== session.key ||
    !question ||
    !questionKey
  )
    return el("span", {});
  const textarea = el("textarea", {
    class: "directive",
    placeholder:
      "方向性指引（作为研究者答复注入对话；留空=让 Agent 自行决策）……",
  });
  const send = (reply, message) =>
    sendControlAction(
      detail.experiment_id,
      { action: "reply_question", session_key: questionKey, directive: reply },
      message,
    );
  return el(
    "div",
    { class: "panel section-gap" },
    el(
      "h4",
      { class: "subsection-title" },
      `Agent 提问 ${questionKey}（等待不消耗推理预算）`,
    ),
    el("div", { class: "ask-user-question" }, question),
    status.question_summary
      ? el("div", { class: "hint" }, String(status.question_summary))
      : null,
    textarea,
    el(
      "div",
      { class: "control-bar" },
      currentStrategyDownloadNode(detail),
      el(
        "button",
        {
          class: "btn primary",
          onclick: () => send(textarea.value, "已答复，Agent 继续"),
        },
        "答复并继续",
      ),
      el(
        "button",
        { class: "btn", onclick: () => send("", "已放行（无指引）") },
        "不给指引，继续",
      ),
    ),
  );
}

/* Step-level HITL: toggle per-session gating and, when the worker is holding
   at a step (state=waiting_step_user), show the step result and release it
   with an optional per-step directive (injected into the tool observation). */
function stepGatePanel(detail, session) {
  if (session.kind !== "fold" || detail.kind !== "hitl") return el("span", {});
  const control = detail.control || {};
  const status = detail.status || {};
  const override = (control.step_gate || {})[session.key];
  const enabled =
    override === undefined ? control.mode === "step" : Boolean(override);
  const send = (payload, message) =>
    sendControlAction(detail.experiment_id, payload, message);
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", { class: "subsection-title" }, "逐 Step 门控"),
    el(
      "div",
      { class: "hint" },
      (detail.control || {}).mode === "step"
        ? "运行模式为「逐 Step 批准」：所有 Fold 默认开启门控，此处仅用于为本 Fold 单独例外（关闭/恢复默认）。"
        : "为本 Fold 单独开启：每次正式验证回测完成即暂停等待批准，可在放行时注入 Step 级指令（等待不消耗推理预算）。全局默认请用运行模式「逐 Step 批准」。",
    ),
    el(
      "div",
      { class: "control-bar" },
      el(
        "button",
        {
          class: enabled ? "btn small" : "btn small primary",
          onclick: () =>
            send(
              {
                action: "set_step_gate",
                session_key: session.key,
                directive: enabled ? "0" : "1",
              },
              enabled
                ? "已关闭本 Fold 逐 Step 门控"
                : "已开启本 Fold 逐 Step 门控",
            ),
        },
        enabled ? "关闭门控" : "开启门控",
      ),
      override !== undefined && control.mode === "step"
        ? el(
            "button",
            {
              class: "btn small",
              onclick: () =>
                send(
                  {
                    action: "set_step_gate",
                    session_key: session.key,
                    directive: "",
                  },
                  "已恢复模式默认",
                ),
            },
            "恢复模式默认",
          )
        : null,
      enabled
        ? el(
            "span",
            { class: "badge state-waiting_user" },
            override === undefined ? "门控开启（逐 Step 模式）" : "门控已开启",
          )
        : null,
    ),
  );
  if (
    status.state === "waiting_step_user" &&
    status.session_key === session.key
  ) {
    const stepIndex = status.step_index;
    const summary = status.step_summary || {};
    const stats = summary.stats || {};
    const textarea = el("textarea", {
      class: "directive section-gap",
      placeholder:
        "可选：本 Step 结果的针对性指令（作为待检验假设注入下一轮对话）……",
    });
    panel.append(
      el(
        "div",
        { class: "section-gap" },
        statTilesRow([
          {
            label: `Step ${stepIndex ?? "?"} 验证收益`,
            value: fmtPct(stats.total_return),
            cls: signCls(stats.total_return),
          },
          {
            label: "Sharpe",
            value:
              stats.sharpe === null || stats.sharpe === undefined
                ? "—"
                : Number(stats.sharpe).toFixed(2),
          },
          { label: "最大回撤", value: fmtPct(stats.max_drawdown) },
          { label: "本 Fold 已耗时", value: foldDurationNode(detail, session) },
        ]),
      ),
      analysisNode(
        `/api/experiments/${encodeURIComponent(detail.experiment_id)}/current-step/analysis`,
        "当前 Step 策略分析（可选，仅基于验证期证据）",
        { standalone: false },
      ),
      textarea,
      el(
        "div",
        { class: "control-bar" },
        currentStrategyDownloadNode(detail),
        el(
          "button",
          {
            class: "btn primary",
            onclick: () =>
              send(
                {
                  action: "approve_step",
                  session_key: session.key,
                  step_index: stepIndex,
                  directive: textarea.value,
                },
                "已放行该 Step",
              ),
          },
          "批准并继续",
        ),
      ),
    );
  }
  return panel;
}

/* Icon labels for the operations chips. Event-type keys (llm_call)
   and tool-call names share one map: both arrive as counters in
   trace/stats. Compact uses compact_ops, not the event-type chip map. */
const STAT_CHIPS = [
  ["llm_call", "🤖 LLM"],
  ["daily_backtest", "📊 回测"],
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
    (kind === "fold" || kind === "meta_learning") &&
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
        ? "消息在 Agent 下一安全点生效。「发送并打断」只请求跳过尚未开跑的工具，不会取消已在途的模型调用或已开始的工具。"
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
    el("h4", {}, `实时 Agent Trace — ${sessionDisplayKey(session)}`),
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
  const info = el("span", { class: "hint", style: "margin:0" }, "");
  const moreButton = el(
    "button",
    { type: "button", class: "btn small", style: "display:none" },
    "继续加载",
  );
  let loadedBlocks = 0,
    eof = false,
    loading = false,
    windowBytes = 0;
  function syncMore() {
    moreButton.disabled = loading;
    moreButton.style.display = eof || !loadedBlocks ? "none" : "";
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
  ["timeout", "超时"],
  ["error", "失败"],
  ["cancelled", "已取消"],
]);
const TERMINAL_SUBAGENT_STATUS = new Set([
  "completed",
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
  if (payload.reduced)
    wrap.append(
      el(
        "div",
        { class: "hint warn" },
        "Meta 会话的子代理记录按设计只保留形状：轮次、工具、用量与字符数可见，模型正文与工具结果正文不写入 Trace。",
      ),
    );
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

function analysisPanel(experimentId, epochId, foldId) {
  const base = `/api/experiments/${encodeURIComponent(experimentId)}/analysis/${encodeURIComponent(epochId)}/${encodeURIComponent(foldId)}`;
  return analysisNode(base, "Fold 策略分析（可选，仅基于验证期证据）");
}

function analysisNode(base, title, { standalone = true } = {}) {
  const panel = el("div", {
    class: standalone ? "panel section-gap" : "section-gap",
  });
  const regenButton = el("button", { class: "btn small" }, "生成分析");
  const head = el(
    "div",
    { class: "control-bar" },
    el("h4", { class: "subsection-title" }, title),
    el("span", { class: "spacer" }),
    regenButton,
  );
  const body = el(
    "div",
    { class: "section-gap" },
    el("div", { class: "loading" }, "加载分析…"),
  );
  panel.append(head, body);
  regenButton.addEventListener("click", async () => {
    try {
      await api(base, { method: "POST" });
      toast("分析已开始生成，稍后自动刷新");
      regenButton.disabled = true;
      liveTimers.push(setTimeout(load, 20_000));
    } catch (error) {
      toast(error.message, true);
    }
  });
  async function load() {
    let payload;
    try {
      payload = await api(base);
    } catch (error) {
      body.innerHTML = "";
      body.append(
        el("div", { class: "hint" }, `分析加载失败：${error.message}`),
      );
      return;
    }
    body.innerHTML = "";
    regenButton.disabled = Boolean(payload.pending);
    regenButton.textContent = payload.available ? "重新生成" : "生成分析";
    if (payload.pending) {
      body.append(
        el(
          "div",
          { class: "prep-indicator" },
          el("span", { class: "spinner" }),
          el("span", {}, "分析生成中…"),
        ),
      );
      liveTimers.push(setTimeout(load, 8000));
    } else if (payload.content) {
      const meta = payload.meta || {};
      if (meta.model) {
        body.append(
          el(
            "div",
            { class: "hint", style: "margin-top:0" },
            `模型 ${meta.model} ｜ 生成于 ${fmtTs(meta.created_at)}`,
          ),
        );
      }
      body.append(renderMarkdown(payload.content));
    } else {
      body.append(
        el("div", { class: "hint" }, "尚未生成分析——点击右上角「生成分析」。"),
      );
    }
  }
  load();
  return panel;
}

function rerunPanel(detail, session) {
  const alive = detail.worker_alive || detail.state === "launching";
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", { class: "subsection-title" }, "重跑本 Fold（最新完成）"),
    el(
      "div",
      { class: "hint" },
      "追加一次全新的 Fold 会话：账本新增记录（旧记录保留供审计），冻结产物以重跑标签另存，已有 Held-out 结果将在重跑后自动重放。启动后在本会话的指令面板修改指令或额外用户指令，再批准运行。",
    ),
  );
  const bar = el("div", { class: "control-bar section-gap" });
  if (alive) {
    bar.append(
      el(
        "span",
        { class: "hint warn", style: "margin:0" },
        "worker 运行中——先「停止」或「强制终止」后方可重跑。",
      ),
    );
  } else {
    bar.append(
      el(
        "button",
        {
          class: "btn primary",
          onclick: () => {
            showModal(
              "确认重跑该 Fold？",
              el(
                "div",
                {},
                el(
                  "p",
                  {},
                  `将重跑 ${sessionDisplayKey(session)}，并使现有 Held-out 结果过期（重跑完成后自动重放 Held-out）。`,
                ),
                el(
                  "p",
                  { class: "hint" },
                  "重跑会话默认等待批准：批准前可修改本 Fold 指令或额外用户指令。",
                ),
              ),
              [
                el("button", { class: "btn", onclick: closeModal }, "取消"),
                el(
                  "button",
                  {
                    class: "btn primary",
                    onclick: async () => {
                      closeModal();
                      try {
                        await api(
                          `/api/experiments/${encodeURIComponent(detail.experiment_id)}/control`,
                          {
                            method: "POST",
                            body: JSON.stringify({
                              action: "rerun_fold",
                              session_key: session.key,
                            }),
                          },
                        );
                        toast("重跑已启动，等待批准");
                        route(true);
                      } catch (error) {
                        toast(error.message, true);
                      }
                    },
                  },
                  "确认重跑",
                ),
              ],
            );
          },
        },
        "修改提示词并重跑",
      ),
    );
  }
  panel.append(bar);
  return panel;
}

/* Roll the experiment back so this (earlier) fold becomes the frontier:
   every later ledger record is dropped (frozen dirs archived, ledger backed
   up server-side) and the run resumes from the next fold. */
function rollbackPanel(detail, session) {
  const alive = detail.worker_alive || detail.state === "launching";
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", { class: "subsection-title" }, "回滚到此 Fold"),
    el(
      "div",
      { class: "hint" },
      "把实验进度回退到本 Fold 刚完成时：其后所有 Fold、元学习会话与全部 Held-out 账本记录将被移除（原账本自动备份、冻结产物归档到 _archive，可人工找回），随后从下一个会话继续（人工控制模式下等待批准，可先修改指令/提示词）。",
    ),
  );
  const bar = el("div", { class: "control-bar section-gap" });
  if (alive) {
    bar.append(
      el(
        "span",
        { class: "hint warn", style: "margin:0" },
        "worker 运行中——先「停止」或「强制终止」后方可回滚。",
      ),
    );
  } else {
    bar.append(
      el(
        "button",
        {
          class: "btn danger",
          onclick: () => {
            showModal(
              "确认回滚？",
              el(
                "div",
                {},
                el(
                  "p",
                  {},
                  `将把实验回退到 ${sessionDisplayKey(session)} 完成时点，丢弃其后全部账本记录（含 Held-out）。`,
                ),
                el(
                  "p",
                  { class: "hint" },
                  "账本会先备份（experiment_ledger.rollback_*.jsonl），被丢弃的冻结产物移入 artifacts/strategy/_archive/。此操作不可从界面撤销。",
                ),
              ),
              [
                el("button", { class: "btn", onclick: closeModal }, "取消"),
                el(
                  "button",
                  {
                    class: "btn danger",
                    onclick: async () => {
                      closeModal();
                      try {
                        await api(
                          `/api/experiments/${encodeURIComponent(detail.experiment_id)}/control`,
                          {
                            method: "POST",
                            body: JSON.stringify({
                              action: "rollback_fold",
                              session_key: session.key,
                            }),
                          },
                        );
                        toast("已回滚并重启 worker");
                        route(true);
                      } catch (error) {
                        toast(error.message, true);
                      }
                    },
                  },
                  "确认回滚",
                ),
              ],
            );
          },
        },
        "回滚到此 Fold",
      ),
    );
  }
  panel.append(bar);
  return panel;
}

/* ---------------- Step 产物树 ---------------- */

/* Cross-fold lineage of validated step artifacts. Branches appear when the
   Agent used step_rollback, or a fold session restarted from a user-set
   parent override. Built for large trees: collapsible subtrees, text filter,
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
  const summary = el("span", { class: "hint", style: "margin-left:auto" });
  const validated = nodes.filter((node) => node.complete_validation).length;

  const haystack = (node) =>
    [
      node.node_id,
      node.fold_ref,
      node.result_name,
      node.epoch_id,
      ...(node.frozen_for || []),
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
    placeholder: "筛选 Fold / 节点 / 结果名…",
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
    el(
      "div",
      { class: "hint" },
      "跨 Fold 的已验证策略谱系：每个节点保存该版本完整源代码与验证明细。悬停看指标详情，点行看完整信息，行内直接下载；HITL 实验可从节点回滚。",
    ),
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
  for (const key of node.frozen_for || [])
    badges.push(el("span", { class: "badge state-completed" }, `冻结 ${key}`));
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
  if (node.has_snapshot && detail.kind === "hitl") {
    actions.append(
      el(
        "button",
        {
          class: "btn small",
          onclick: (event) => {
            event.stopPropagation();
            hideStepTip();
            openStepParentOverrideModal(detail, payload, node);
          },
        },
        "回滚…",
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
      `${foldPeriodLabel(detail, node.fold_ref)} · ${node.result_name || node.node_id}`,
    ),
    collapsed ? el("span", { class: "step-chip" }, `+${childCount}`) : null,
    Number.isFinite(metrics.total_return)
      ? el(
          "span",
          { class: `step-chip ${numClass(metrics.total_return)}` },
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
    line(
      "Fold",
      `${node.epoch_id || "—"} / ${foldPeriodLabel(detail, node.fold_ref)}`,
    ),
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
    (node.frozen_for || []).length
      ? line("冻结用于", node.frozen_for.join("、"))
      : null,
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
      kvRow(
        "Fold",
        `${node.epoch_id || "—"} / ${foldPeriodLabel(detail, node.fold_ref)}`,
      ),
      kvRow(
        "验证收益",
        el("span", { class: numClass(m.total_return) }, fmtPct(m.total_return)),
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
      (node.frozen_for || []).length
        ? kvRow("冻结用于", node.frozen_for.join("、"))
        : null,
      node.status === "failed" ? kvRow("失败原因", node.error || "—") : null,
      kvRow("附件", (node.attachments || []).join("、") || "—"),
      kvRow("revision", el("code", {}, String(node.strategy_ref || "—"))),
    ),
    node.has_snapshot
      ? el(
          "p",
          { class: "hint" },
          "「下载源码 + 结果」包含该版本完整 output/ 源代码、models/ 参数与该次验证的明细结果文件。",
        )
      : el(
          "p",
          { class: "hint" },
          "失败尝试不保存产物快照，仅记录失败原因供避坑。",
        ),
  );
  const buttons = [el("button", { class: "btn", onclick: closeModal }, "关闭")];
  if (node.has_snapshot)
    buttons.push(el("a", { class: "btn", href: zipUrl }, "下载源码 + 结果"));
  if (node.has_snapshot && detail.kind === "hitl") {
    buttons.push(
      el(
        "button",
        {
          class: "btn primary",
          onclick: () => openStepParentOverrideModal(detail, payload, node),
        },
        "从此节点回滚…",
      ),
    );
  }
  showModal("Step 节点详情", body, buttons);
}

/* User-side step rollback: make this node the parent of a fold session
   (pending fold: takes effect at its next start; completed fold: combine with
   rerun, which stays restricted to the latest completed fold). */
function openStepParentOverrideModal(detail, payload, node) {
  const sessions = payload.fold_sessions || [];
  if (!sessions.length) {
    toast("该实验没有 Fold 会话", true);
    return;
  }
  const select = el(
    "select",
    { class: "input" },
    ...sessions.map((session) =>
      el("option", { value: session.key }, sessionDisplayKey(session)),
    ),
  );
  const own = sessions.find(
    (session) =>
      session.epoch_id === node.epoch_id && session.fold_ref === node.fold_ref,
  );
  if (own) select.value = own.key;
  const send = (action, sessionKey, directive) =>
    api(
      `/api/experiments/${encodeURIComponent(detail.experiment_id)}/control`,
      {
        method: "POST",
        body: JSON.stringify({ action, session_key: sessionKey, directive }),
      },
    );
  const body = el(
    "div",
    {},
    el(
      "p",
      {},
      `把 ${node.node_id} 设为所选 Fold 会话的父产物起点（替代默认的冻结继承链）。`,
    ),
    el(
      "p",
      { class: "hint" },
      "只能选不晚于该节点所属会话的目标（更晚节点携带未来验证信息会被拒绝）。尚未运行的 Fold：下次启动该会话时生效（人工控制模式下批准后）。已完成的 Fold：用「设置并重跑」" +
        "（仅允许重跑最新完成的 Fold，且需先停止 worker）。设置持续有效：重新设置即覆盖，「清除」即恢复默认继承链。",
    ),
    el("label", { class: "hint" }, "目标 Fold 会话"),
    select,
  );
  showModal("从此节点回滚 / 设为起点", body, [
    el("button", { class: "btn", onclick: closeModal }, "取消"),
    el(
      "button",
      {
        class: "btn",
        onclick: async () => {
          try {
            await send("set_parent_override", select.value, "");
            toast(`已清除 ${select.value} 的起点覆盖`);
            closeModal();
          } catch (error) {
            toast(error.message, true);
          }
        },
      },
      "清除该会话覆盖",
    ),
    el(
      "button",
      {
        class: "btn primary",
        onclick: async () => {
          try {
            await send("set_parent_override", select.value, node.node_id);
            toast(`已把 ${select.value} 的起点设为该节点`);
            closeModal();
          } catch (error) {
            toast(error.message, true);
          }
        },
      },
      "仅设置起点",
    ),
    el(
      "button",
      {
        class: "btn danger",
        onclick: async () => {
          try {
            await send("set_parent_override", select.value, node.node_id);
            await send("rerun_fold", select.value, null);
            toast("已设置起点并启动重跑（等待批准）");
            closeModal();
            route(true);
          } catch (error) {
            toast(error.message, true);
          }
        },
      },
      "设置并重跑",
    ),
  ]);
}

function foldResultPanel(detail, session) {
  const record = session.record || {};
  const validation = record.validation_result || {};
  const statusLabels = {
    frozen: "已冻结新产物",
    no_update: "沿用父产物（有验证未获接受）",
    no_valid_backtest: "沿用父产物（无完整验证）",
  };
  const panel = el(
    "div",
    { class: "panel" },
    el(
      "div",
      { class: "control-bar" },
      el(
        "h4",
        { style: "margin:0" },
        `Fold 结果 — ${sessionDisplayKey(session)}`,
      ),
      el(
        "span",
        {
          class: `badge state-${
            record.fold_status === "frozen"
              ? (record.accept_warnings || []).length
                ? "waiting_user"
                : "completed"
              : "stopped"
          }`,
        },
        record.fold_status === "frozen" && (record.accept_warnings || []).length
          ? "已冻结（有验收警告）"
          : statusLabels[record.fold_status] || record.fold_status || "—",
      ),
      record.finish_reason
        ? el("span", { class: "mode-note" }, `结束原因 ${record.finish_reason}`)
        : null,
    ),
  );
  // Headline validation metrics as tiles, metadata as a compact kv block.
  panel.append(
    el(
      "div",
      { class: "section-gap" },
      statTilesRow([
        {
          label: "验证收益",
          value: fmtPct(validation.total_return),
          cls: signCls(validation.total_return),
        },
        {
          label: "验证 Sharpe",
          value:
            validation.sharpe === undefined || validation.sharpe === null
              ? "—"
              : Number(validation.sharpe).toFixed(2),
          cls: signCls(validation.sharpe),
        },
        { label: "验证回撤", value: fmtPct(validation.max_drawdown) },
        {
          label: "多头收益",
          value: fmtPct(validation.long_return),
          cls: signCls(validation.long_return),
        },
      ]),
    ),
  );
  panel.append(parentControlSection(detail, session, validation));
  const benchmark = validation.benchmark || {};
  panel.append(
    el(
      "table",
      { class: "kv section-gap" },
      kvRow(
        "验证区间",
        fmtPeriodRange(record.validation_period || session.validation_period),
      ),
      // The raw excess cannot separate an edge from a small-cap or high-beta
      // tilt, so the neutralized figure is read beside it, never alone.
      kvRow(
        "超额收益（vs 沪深300）",
        el(
          "span",
          { class: numClass(benchmark.excess_return) },
          fmtPct(benchmark.excess_return),
        ),
      ),
      kvRow(
        el(
          "span",
          { title: benchmark.neutralized_excess_method || "" },
          "规模/β 中性化超额（年化）",
        ),
        el(
          "span",
          { class: numClass(benchmark.neutralized_excess_return) },
          fmtPct(benchmark.neutralized_excess_return),
        ),
      ),
      record.run_wall_seconds
        ? kvRow("总耗时", fmtDuration(record.run_wall_seconds))
        : null,
      kvRow("冻结产物", record.frozen_strategy_artifact_ref || "—"),
      (record.accept_reasons || []).length
        ? kvRow("未接受原因", (record.accept_reasons || []).join("；"))
        : null,
      (record.accept_warnings || []).length
        ? kvRow(
            "验收警告",
            el(
              "span",
              { class: "num neg" },
              record.accept_warnings.map(fmtAcceptanceWarning).join("；"),
            ),
          )
        : null,
    ),
  );
  const validationSubWindows = subWindowSection(
    "验证期分季度表现",
    validation.sub_windows,
  );
  if (validationSubWindows) panel.append(validationSubWindows);
  if (record.run_ref) {
    panel.append(
      el(
        "div",
        { class: "section-gap" },
        el(
          "h4",
          { class: "subsection-title" },
          "验证期日度累计收益 vs 沪深300（含回撤）",
        ),
        foldEquityHost(
          detail.experiment_id,
          session.epoch_id,
          session.fold_ref || record.fold_ref,
          record.run_ref,
          "valid",
          { width: 860, height: 210, ddH: 76 },
        ),
      ),
    );
    panel.append(styleCard(detail.experiment_id, record.run_ref, "valid"));
  }
  // Guarded test audit block (collapsed, clearly labelled).
  panel.append(
    loadFoldExtras(
      detail.experiment_id,
      session.epoch_id,
      session.fold_ref || record.fold_ref,
    ),
  );
  return panel;
}

/* This Fold's baseline: before the session starts the host replays the
   inherited parent unchanged over this Fold's Validation window, so the Fold's
   own Validation is read against it rather than on its own. The first Fold of
   the first Epoch inherits nothing and has no control. Metrics come from the
   same fold_returns row the walk-forward table uses; the failure text only
   exists on the ledger record. */
function parentControlSection(detail, session, validation) {
  const record = session.record || {};
  const control = record.parent_control;
  if (!control)
    return el(
      "div",
      { class: "meta-line" },
      "父本对照：本 Fold 没有继承父产物，因此没有基线",
    );
  const row = (detail.fold_returns || []).find(
    (item) =>
      item.epoch_id === session.epoch_id &&
      item.fold_ref === (session.fold_ref || record.fold_ref),
  );
  const metrics = (row || {}).parent_control || {};
  const failed = control.status !== "ok";
  const metricRow = (label, values) =>
    el(
      "tr",
      {},
      el("td", {}, label),
      el("td", { class: signCls(values.total) }, fmtPct(values.total)),
      el("td", { class: signCls(values.excess) }, fmtPct(values.excess)),
      el("td", { class: signCls(values.sharpe) }, fmtSharpe(values.sharpe)),
      el("td", {}, fmtPct(values.drawdown)),
    );
  const section = el(
    "div",
    { class: "section-gap" },
    el(
      "h4",
      { class: "subsection-title" },
      "父本对照（本 Fold 基线）",
      failed ? el("span", { class: "badge state-failed" }, "对照失败") : null,
    ),
    el(
      "table",
      { class: "data" },
      el(
        "tr",
        {},
        el("th", {}, "对照"),
        el("th", { title: "整个验证区间的区间收益" }, "收益"),
        el("th", { title: "相对沪深300的超额收益" }, "超额"),
        el("th", { title: "验证区间日收益的年化 Sharpe" }, "Sharpe"),
        el("th", { title: "验证区间峰谷回撤" }, "回撤"),
      ),
      metricRow("本 Fold 验证", {
        total: validation.total_return,
        excess: (validation.benchmark || {}).excess_return,
        sharpe: validation.sharpe,
        drawdown: validation.max_drawdown,
      }),
      metricRow(
        el(
          "span",
          {},
          "父本原样重跑",
          el(
            "span",
            { class: "mode-note" },
            ` ${control.parent_strategy_artifact_ref || "—"}`,
          ),
        ),
        {
          total: metrics.return,
          excess: metrics.excess_return,
          sharpe: metrics.sharpe,
          drawdown: metrics.max_drawdown,
        },
      ),
    ),
  );
  if (failed)
    section.append(
      el(
        "div",
        { class: "meta-line" },
        `对照未完成：${control.error || "见账本"}`,
      ),
    );
  return section;
}

function kvRow(key, value) {
  return el("tr", {}, el("td", {}, key), el("td", {}, value));
}

/* Per-calendar-quarter breakdown of one replay window (stats.sub_windows):
   the same figures as the headline tiles, one row per quarter, so a whole-
   window number can be read against its sub-periods instead of on its own.
   "部分" marks a quarter the window does not span end to end. 超额 is against
   沪深300 and stays blank when the slot had no usable benchmark. */
function subWindowSection(title, rows) {
  if (!Array.isArray(rows) || !rows.length) return null;
  const head = el(
    "tr",
    {},
    el("th", {}, "分季度"),
    el("th", { title: "季度开盘权益起算的区间收益" }, "收益"),
    el("th", { title: "相对沪深300的超额收益" }, "超额"),
    el("th", { title: "季度内日收益的年化 Sharpe" }, "Sharpe"),
    el("th", { title: "季度内峰谷回撤" }, "回撤"),
    el("th", { title: "成交名义额 / 初始资金" }, "换手"),
    el("th", { title: "已实现平仓笔数" }, "笔数"),
    el("th", {}, "交易日"),
  );
  const body = rows.map((row) =>
    el(
      "tr",
      {},
      el(
        "td",
        {},
        `${row.label || "—"}${row.partial ? " ·部分" : ""}`,
        el(
          "span",
          { class: "mode-note" },
          ` ${fmtPeriodRange(`${row.start}..${row.end}`)}`,
        ),
      ),
      el("td", { class: signCls(row.return) }, fmtPct(row.return)),
      el(
        "td",
        { class: signCls(row.excess_return) },
        fmtPct(row.excess_return),
      ),
      el("td", { class: signCls(row.sharpe) }, fmtSharpe(row.sharpe)),
      el("td", {}, fmtPct(row.max_drawdown)),
      el("td", {}, fmtSharpe(row.turnover)),
      el(
        "td",
        {},
        row.trade_count === null || row.trade_count === undefined
          ? "—"
          : String(row.trade_count),
      ),
      el(
        "td",
        {},
        row.trade_days === null || row.trade_days === undefined
          ? "—"
          : String(row.trade_days),
      ),
    ),
  );
  return el(
    "div",
    { class: "section-gap" },
    el("h4", { class: "subsection-title" }, title),
    el("table", { class: "data" }, head, ...body),
  );
}

/* Barra-lite style validation card: CSI300 alpha/beta regression + holdings
   style tilts (signed percentile deviation, [-1,1]) + SW industry weights. */
function styleCard(expId, runId, prefix) {
  const host = el(
    "div",
    { class: "section-gap" },
    el("h4", { class: "subsection-title" }, "风格暴露与基准归因（Barra-lite）"),
    el("div", { class: "hint" }, "加载中…"),
  );
  api(
    `/api/experiments/${encodeURIComponent(expId)}/style?run_id=${encodeURIComponent(runId)}&prefix=${encodeURIComponent(prefix)}`,
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
          "冻结回放槽中没有可用的沪深300同窗数据，基准回归为空。",
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
            "冻结回放槽缺少市值、PB 或换手截面，风格暴露为空。",
          no_holdings: "该 Validation 窗口没有持仓，风格暴露为空。",
          no_valued_holdings:
            "该 Validation 窗口的持仓没有可用收盘价，风格暴露为空。",
        };
        host.append(
          el(
            "div",
            { class: "hint" },
            styleReasons[style.reason] || "该窗口没有可计算的风格暴露。",
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

/* Per-fold daily equity (validation and guarded test parts share one fetch). */
const FOLD_EQUITY_CACHE = new Map(); // `${exp}/${epoch}/${fold}/${run}` -> promise
function foldEquityHost(expId, epochId, foldId, runId, part, opts) {
  const key = `${expId}/${epochId}/${foldId}/${runId || ""}`;
  if (!FOLD_EQUITY_CACHE.has(key)) {
    FOLD_EQUITY_CACHE.set(
      key,
      api(
        `/api/experiments/${encodeURIComponent(expId)}/folds/${encodeURIComponent(epochId)}/${encodeURIComponent(foldId)}/equity`,
      ),
    );
  }
  const host = el("div", {}, el("div", { class: "hint" }, "收益曲线加载中…"));
  FOLD_EQUITY_CACHE.get(key)
    .then((payload) => {
      host.innerHTML = "";
      const selected = (payload.series || []).filter(
        (series) => series.key === part,
      );
      host.append(equityChart({ ...payload, series: selected }, opts));
    })
    .catch((error) => {
      FOLD_EQUITY_CACHE.delete(key);
      host.innerHTML = "";
      host.append(
        el("div", { class: "hint" }, `收益曲线加载失败：${error.message}`),
      );
    });
  return host;
}

function loadFoldExtras(experimentId, epochId, foldId) {
  const wrap = el(
    "div",
    { class: "section-gap" },
    el("div", { class: "loading" }, "加载策略与分析…"),
  );
  (async () => {
    let fold;
    try {
      fold = await api(
        `/api/experiments/${encodeURIComponent(experimentId)}/folds/${encodeURIComponent(epochId)}/${encodeURIComponent(foldId)}`,
      );
    } catch (error) {
      wrap.innerHTML = "";
      wrap.append(
        el(
          "div",
          { class: "hint" },
          `无法加载 Fold 附加信息：${error.message}`,
        ),
      );
      return;
    }
    wrap.innerHTML = "";
    // Frozen strategy: one ZIP package (output + models), no per-file listing.
    wrap.append(
      el(
        "div",
        { class: "control-bar" },
        el("h4", { class: "subsection-title" }, "冻结策略产物"),
        el("span", { class: "mode-note" }, "打包 output 与 models 全部文件"),
        el("span", { class: "spacer" }),
        el(
          "a",
          {
            class: "btn small",
            href: `/api/experiments/${encodeURIComponent(experimentId)}/folds/${encodeURIComponent(epochId)}/${encodeURIComponent(foldId)}/strategy.zip`,
          },
          "⬇ 下载 ZIP 包",
        ),
      ),
    );
    // Validation-backtest order stream: stats, charts, table, CSV export.
    wrap.append(ordersNode(experimentId, epochId, foldId));
    // Test audit, collapsed with warning.
    const audit = fold.test_audit || {};
    if (audit.test_result) {
      const test = audit.test_result;
      wrap.append(
        el(
          "details",
          { class: "test-audit section-gap" },
          el("summary", {}, "测试期结果（Meta-development 审计 — 谨慎查看）"),
          el(
            "div",
            { class: "hint warn" },
            "本结果在该 Fold 冻结时是样本外评估，之后其 compact 指标可由系统提供给 Meta，因而属于自适应开发证据而非最终未触碰估计。人工明细只在实验封存后揭示；Held-out 才是最终未触碰评估。",
          ),
          el(
            "table",
            { class: "kv" },
            kvRow(
              "测试收益",
              el(
                "span",
                { class: numClass(test.total_return) },
                fmtPct(test.total_return),
              ),
            ),
            kvRow(
              "测试 Sharpe",
              test.sharpe !== undefined && test.sharpe !== null
                ? Number(test.sharpe).toFixed(2)
                : "—",
            ),
            kvRow("测试回撤", fmtPct(test.max_drawdown)),
          ),
          // Sealed with the rest of this block: fold_detail returns
          // {hidden:true} until the reveal, so no sub-window row exists here
          // before then.
          subWindowSection("测试期分季度表现", test.sub_windows),
          el(
            "div",
            { class: "section-gap" },
            el(
              "h4",
              { class: "subsection-title" },
              "测试期日度累计收益 vs 沪深300（含回撤）",
            ),
            foldEquityHost(
              experimentId,
              epochId,
              foldId,
              (fold.record || {}).run_ref,
              "test",
              { width: 760, height: 190, ddH: 64 },
            ),
          ),
          (fold.record || {}).run_ref
            ? styleCard(experimentId, fold.record.run_ref, "test")
            : null,
          // The result id comes from the read-model: result directories are
          // named frozen_test_<uuid>, so a guessed name would only ever 404.
          audit.result
            ? el(
                "div",
                { class: "section-gap" },
                el(
                  "a",
                  {
                    class: "btn small",
                    href: `/api/experiments/${encodeURIComponent(experimentId)}/folds/${encodeURIComponent(epochId)}/${encodeURIComponent(foldId)}/orders.csv?result=${encodeURIComponent(audit.result)}`,
                  },
                  "⬇ 测试期交易明细 CSV",
                ),
              )
            : el(
                "div",
                { class: "hint section-gap" },
                "该测试评估没有可导出的交易明细。",
              ),
        ),
      );
    }
  })();
  return wrap;
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

const ORDER_TABLE_COLUMNS = [
  ["matched_at", "成交时间"],
  ["execute_at", "计划时间"],
  ["symbol", "代码"],
  ["action", "动作"],
  ["quantity", "数量"],
  ["price", "价格"],
  ["status", "状态"],
  ["reason", "拒单原因"],
];

/* Validation-backtest transaction details: stats tiles, per-day amount chart,
   order table, CSV export. Result switcher covers the fold's valid_* runs. */
function ordersNode(experimentId, epochId, foldId) {
  const base = `/api/experiments/${encodeURIComponent(experimentId)}/folds/${encodeURIComponent(epochId)}/${encodeURIComponent(foldId)}`;
  const body = el("div", {});
  const wrap = el(
    "div",
    { class: "section-gap" },
    el(
      "h4",
      { class: "subsection-title", style: "margin-bottom:0.4rem" },
      "交易明细（验证回测）",
    ),
    body,
  );
  let loading = false;
  async function load(result) {
    // No flash: keep the current content (dimmed) until the new data arrives.
    if (loading) return;
    loading = true;
    body.style.opacity = body.children.length ? "0.55" : "";
    if (!body.children.length)
      body.append(el("div", { class: "loading" }, "加载交易明细…"));
    let data;
    try {
      data = await api(
        `${base}/orders${result ? `?result=${encodeURIComponent(result)}` : ""}`,
      );
    } catch (error) {
      body.innerHTML = "";
      body.style.opacity = "";
      loading = false;
      body.append(el("div", { class: "hint" }, `无交易明细：${error.message}`));
      return;
    }
    body.innerHTML = "";
    body.style.opacity = "";
    loading = false;
    const bar = el("div", { class: "control-bar" });
    const available = data.available || [];
    if (available.length > 1) {
      for (const name of available) {
        bar.append(
          el(
            "span",
            {
              class: `file-chip${name === data.result ? " active" : ""}`,
              onclick: () => load(name),
            },
            name,
          ),
        );
      }
    } else {
      bar.append(el("span", { class: "mode-note" }, data.result));
    }
    bar.append(
      el("span", { class: "spacer" }),
      el(
        "a",
        {
          class: "btn small",
          href: `${base}/orders.csv?result=${encodeURIComponent(data.result)}`,
        },
        "⬇ 导出 CSV",
      ),
    );
    const rows = data.rows || [];
    const stats = data.stats || {};
    const byAction = stats.by_action || {};
    body.append(
      bar,
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
    if (daily.length) {
      body.append(
        el("h4", { class: "section-gap" }, "逐日成交金额"),
        singleSeriesBarChart(daily, { fmt: fmtAmount, height: 180 }),
      );
    }
    if (Object.keys(stats.reject_reasons || {}).length) {
      body.append(
        el(
          "div",
          { class: "stats-chips section-gap" },
          ...Object.entries(stats.reject_reasons).map(([reason, count]) =>
            el("span", { class: "stat-chip" }, `拒单 ${reason} ×${count}`),
          ),
        ),
      );
    }
    if (rows.length) {
      const table = el(
        "table",
        { class: "data section-gap" },
        el(
          "tr",
          {},
          ...ORDER_TABLE_COLUMNS.map(([, label]) => el("th", {}, label)),
        ),
        ...rows.slice(0, 80).map((row) =>
          el(
            "tr",
            {},
            ...ORDER_TABLE_COLUMNS.map(([key]) => {
              let value = fmtOrderCell(key, row[key]);
              if (key === "price" && value !== null && value !== undefined)
                value = Number(value).toFixed(3);
              return el(
                "td",
                {},
                value === null || value === undefined ? "—" : String(value),
              );
            }),
          ),
        ),
      );
      const box = el("div", { class: "orders-table-box" }, table);
      body.append(box);
      if (data.row_count > Math.min(rows.length, 80)) {
        body.append(
          el(
            "div",
            { class: "hint" },
            `表格显示前 ${Math.min(rows.length, 80)} 条，共 ${data.row_count} 条 —— 完整明细请导出 CSV。`,
          ),
        );
      }
    }
  }
  load(null);
  return wrap;
}

/* Summarize the bounded derived-image build result without retaining process output. */
function sandboxImageNode(update) {
  const status = String(update.status || "unknown");
  if (status === "ok") {
    const secs =
      (Date.parse(update.finished_at || "") -
        Date.parse(update.started_at || "")) /
      1000;
    const pruned = Array.isArray(update.pruned_image_refs)
      ? update.pruned_image_refs.length
      : 0;
    return [
      "构建成功",
      update.image_ref,
      Number.isFinite(secs) ? `${secs.toFixed(1)}s` : null,
      pruned ? `清理旧镜像 ${pruned} 个` : null,
    ]
      .filter(Boolean)
      .join(" ｜ ");
  }
  const notes = {
    skipped_empty: "请求为空，未构建（沿用基础镜像）",
    skipped_local_dev: "local_dev 运行，未构建（沿用基础镜像）",
    disabled: "派生镜像构建已禁用（沿用基础镜像）",
  };
  if (notes[status]) return notes[status];
  const tail = String(update.reason || "").trim();
  return el(
    "div",
    {},
    el(
      "div",
      { class: "form-error" },
      `构建未成功（${status}）${update.image_ref ? `：${update.image_ref}` : ""}`,
    ),
    tail ? el("pre", { class: "code-view" }, tail.slice(-2000)) : null,
  );
}

function metaResultPanel(detail, session) {
  const record = session.record || {};
  const trigger = Number(
    session.trigger_after_folds || record.trigger_after_folds || 0,
  );
  const label =
    trigger > 0 ? `${session.epoch_id} / ${trigger} Fold 后` : session.epoch_id;
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", {}, `元学习结果 — ${label}`),
  );
  panel.append(
    el(
      "table",
      { class: "kv" },
      kvRow("状态", record.status || "—"),
      kvRow("总耗时", foldDurationNode(detail, session)),
      record.meta_learning_directive
        ? kvRow("注入指令", record.meta_learning_directive)
        : null,
      record.fold_exploration_directive
        ? kvRow("实验探索主线", record.fold_exploration_directive)
        : null,
      record.sandbox_image_update
        ? kvRow("沙箱镜像", sandboxImageNode(record.sandbox_image_update))
        : null,
    ),
  );
  if (record.prior) {
    panel.append(
      el("h4", { class: "section-gap" }, "PRIOR（后续 Fold 的方向与经验）"),
      renderMarkdown(record.prior),
    );
  }
  return panel;
}

function heldoutPanel(detail, session) {
  const records = session.records || [];
  const plannedPeriods = new Map(
    (session.periods || [])
      .filter((item) => item && item.label)
      .map((item) => [String(item.label), item]),
  );
  const hidden =
    !detail.test_revealed || records.some((record) => record.hidden);
  if (hidden) {
    return el(
      "div",
      { class: "panel section-gap" },
      el("h4", { class: "subsection-title" }, "Held-out 冻结测试（最终样本外）"),
      el(
        "div",
        { class: "empty" },
        detail.test_revealed
          ? "Held-out 结果尚未写入。"
          : "测试与 Held-out 尚未揭示。揭示后才会显示样本外数字。",
      ),
    );
  }
  const results = records.map((record) => record.result || {});
  const returns = results
    .map((result) => result.total_return)
    .filter((value) => value !== null && value !== undefined);
  const sharpes = results
    .map((result) => result.sharpe)
    .filter((value) => value !== null && value !== undefined);
  const drawdowns = results
    .map((result) => result.max_drawdown)
    .filter((value) => value !== null && value !== undefined);
  const longs = results
    .map((result) => result.long_return)
    .filter((value) => value !== null && value !== undefined);
  const cumulative = returns.length
    ? returns.reduce((total, value) => total * (1 + value), 1) - 1
    : null;
  const wins = returns.filter((value) => value > 0).length;
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el(
      "h4",
      { class: "subsection-title" },
      "Held-out 冻结测试（最终样本外）",
      verdictBadge(detail.verdict),
      walkForwardTerm(detail.verdict),
    ),
  );
  if (detail.verdict && (detail.verdict.reasons || []).length)
    panel.append(
      el(
        "div",
        { class: "meta-line" },
        `未达标：${detail.verdict.reasons.join("、")}`,
      ),
    );
  panel.append(
    el(
      "div",
      { class: "section-gap" },
      statTilesRow([
        {
          label: "累计收益",
          value: fmtPct(cumulative),
          cls: signCls(cumulative),
        },
        {
          label: "平均 Sharpe",
          value: sharpes.length
            ? (sharpes.reduce((a, b) => a + b, 0) / sharpes.length).toFixed(2)
            : "—",
        },
        {
          label: "最差单期回撤",
          value: drawdowns.length ? fmtPct(Math.max(...drawdowns)) : "—",
        },
        {
          label: "正收益期数",
          value: returns.length ? `${wins} / ${returns.length}` : "—",
        },
        {
          label: "多头贡献（累计）",
          value: longs.length ? fmtPct(longs.reduce((a, b) => a + b, 0)) : "—",
        },
      ]),
    ),
  );
  if (returns.length && detailView) {
    panel.append(
      el(
        "div",
        { class: "section-gap" },
        el("h4", {}, "日度累计收益 vs 沪深300（含回撤）"),
        equityHost(
          detailView.experimentId,
          equityFingerprint(detailView.detail),
          { width: 860, height: 220, ddH: 80, keys: ["heldout"] },
        ),
      ),
    );
    const lastRun =
      records[records.length - 1] && records[records.length - 1].run_ref;
    if (lastRun)
      panel.append(
        styleCard(detailView.experimentId, String(lastRun), "heldout"),
      );
  }
  panel.append(
    el(
      "table",
      { class: "data section-gap" },
      el(
        "tr",
        {},
        el("th", {}, "区间"),
        el("th", {}, "起止"),
        el("th", {}, "收益"),
        el("th", {}, "多头"),
        el("th", {}, "Sharpe"),
        el("th", {}, "回撤"),
        el("th", {}, "订单"),
      ),
      ...records.map((record) => {
        const result = record.result || {};
        // The ledger carries the period LABEL; its calendar bounds live on the
        // planned held-out session (revealed alongside these records).
        const label = String(record.period || "");
        const period = plannedPeriods.get(label) || {};
        return el(
          "tr",
          {},
          el("td", {}, label ? fmtPeriodRange(label) : "—"),
          el(
            "td",
            {},
            period.start && period.end
              ? `${fmtDate(period.start)} ～ ${fmtDate(period.end)}`
              : "—",
          ),
          el(
            "td",
            { class: numClass(result.total_return) },
            fmtPct(result.total_return),
          ),
          el(
            "td",
            { class: numClass(result.long_return) },
            fmtPct(result.long_return),
          ),
          el(
            "td",
            {},
            result.sharpe !== undefined && result.sharpe !== null
              ? Number(result.sharpe).toFixed(2)
              : "—",
          ),
          el("td", {}, fmtPct(result.max_drawdown)),
          el("td", {}, result.order_count ?? "—"),
        );
      }),
    ),
  );
  // Held-out is the final untouched estimate; its per-quarter rows are what
  // say whether one stretch of market carried the whole span.
  for (const record of records) {
    const result = record.result || {};
    const label = String(record.period || "");
    const section = subWindowSection(
      label ? `Held-out ${label} 分季度表现` : "Held-out 分季度表现",
      result.sub_windows,
    );
    if (section) panel.append(section);
  }
  return panel;
}

/* ---------------- 运行记忆 ----------------
   One stable two-pane layout. The left pane is the whole catalogue — the
   curated 精选库 above, the 毕业层候选 the tier admits below — and the right
   pane is a single viewer/editor surface with a fixed head, toolbar and body,
   so selecting anything swaps only what those three hosts contain: the page
   header, the notice and both pane widths never move.

   The library is a tracked repository directory that every session copies
   read-only into its workspace at session start, so a write here reaches
   sessions started afterwards and never the ones already running, and the
   researcher still commits it. Who may write is settled by how the console is
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
  $topbarRight.replaceChildren(
    el("span", { class: "mode-note" }, "跨实验知识"),
  );
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
    countHost: el("span", {}, ""),
    listHost: el("div", { class: "session-list" }),
    candidateHost: el("div", { class: "session-list" }),
    headHost: el("div", { class: "memory-pane-head" }),
    toolbarHost: el("div", { class: "control-bar memory-toolbar" }),
    bodyHost: el("div", { class: "memory-pane-body" }),
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
          `每个 Fold 与元学习会话按来源只读挂载到 memory/<来源>/ ｜ 默认模式 ${payload.default_mode || "—"}`,
        ),
      ),
      memoryNotice(),
      el(
        "div",
        { class: "detail section-gap" },
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
  );
  renderMemoryList();
  renderMemoryCandidates();
  renderMemoryPane();
}

/* Persistent, not a toast: when a change takes effect and where it lives are
   the two facts the entry list itself cannot show. */
function memoryNotice() {
  return el(
    "div",
    { class: "panel memory-notice" },
    el(
      "div",
      { class: "hint" },
      `精选库是仓库目录 ${memoryLibraryPath()}/，纳入版本控制，改动由研究者自行提交。挂载发生在会话启动时，因此新增、修改和删除只对此后启动的会话生效；运行中的会话保留启动时挂载的只读副本。会话只能引用和质疑挂载内容、用 memory_feedback 记录判断，不能改写它；条目上的徽标就是这些判断的汇总。`,
    ),
  );
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
    el(
      "div",
      { class: "hint" },
      "Held-out 判定全部 graduated 且已发布 skills 世代的历史实验自动准入，本实验自己始终排除在外。点选候选先看原文，再晋升到精选库。",
    ),
    memoryView.candidateHost,
  );
}

/* Sessions may doubt, ignore and report mounted memory; they never rewrite it.
   `memory_feedback` verdicts reach the console through the run manifests, so an
   entry carries what other experiments concluded about it. */
const MEMORY_VERDICTS = [
  ["confirmed", "确认", "completed"],
  ["outdated", "过时", "paused"],
  ["wrong", "有误", "failed"],
];

function memoryFeedbackFor(key) {
  const feedback = (memoryView && memoryView.payload.feedback) || {};
  return (feedback.entries || {})[key] || null;
}

/* Small enough for a list row: one badge per verdict that was actually used,
   plus the dispute badge when two experiments called the entry wrong. */
function feedbackBadges(record) {
  if (!record) return [];
  const counts = record.counts || {};
  const badges = MEMORY_VERDICTS.filter(([verdict]) => counts[verdict]).map(
    ([verdict, label, state]) =>
      el(
        "span",
        { class: `badge mini state-${state}`, title: `${label} ${counts[verdict]} 次` },
        `${label}${counts[verdict]}`,
      ),
  );
  if (record.disputed)
    badges.push(
      el(
        "span",
        {
          class: "badge mini state-failed",
          title: "至少两个实验判定这条有误",
        },
        "有争议",
      ),
    );
  return badges;
}

/* The right pane's half: who said what, and what they saw. */
function feedbackSection(record) {
  if (!record || !(record.reports || []).length) return null;
  const head = el(
    "div",
    { class: "control-bar" },
    el(
      "span",
      { class: "hint" },
      `会话反馈 ${(record.reports || []).length} 条，来自 ${record.experiments || 0} 个实验`,
    ),
    ...feedbackBadges(record),
  );
  return el(
    "div",
    { class: "feedback-block" },
    head,
    el(
      "table",
      { class: "data text" },
      el(
        "tr",
        {},
        el("th", { class: "nowrap" }, "实验"),
        el("th", { class: "nowrap" }, "会话"),
        el("th", { class: "nowrap" }, "判断"),
        el("th", {}, "说明"),
      ),
      ...(record.reports || []).map((report) =>
        el(
          "tr",
          {},
          el("td", { class: "nowrap" }, report.experiment_id || "—"),
          el("td", { class: "nowrap" }, report.session_label || "—"),
          el(
            "td",
            { class: "nowrap" },
            (MEMORY_VERDICTS.find(([verdict]) => verdict === report.verdict) || [
              "",
              report.verdict || "—",
            ])[1],
          ),
          el("td", {}, report.note || "—"),
        ),
      ),
    ),
  );
}

function memoryFilterHit(...parts) {
  const needle = memoryView.filter.trim().toLowerCase();
  return !needle || parts.join(" ").toLowerCase().includes(needle);
}

function memoryNavItem(label, note, selection, feedbackKey) {
  return el(
    "div",
    {
      class: `session-item${sameSelection(memoryView.selection, selection) ? " selected" : ""}`,
      title: label,
      onclick: () => selectMemoryItem(selection),
    },
    el("span", { class: "label" }, label),
    ...feedbackBadges(memoryFeedbackFor(feedbackKey)),
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
      `curated/${entry.name}`,
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

/* Admitted candidates are selectable, withdrawn ones stay listed under their
   experiment so the withdrawal can be undone, and every other experiment stays
   visible in one collapsed muted block — because "not offered", "taken out" and
   "not there" are three different answers. */
function renderMemoryCandidates() {
  const tier = memoryView.payload.graduated || {};
  const rows = tier.experiments || [];
  const listed = rows.filter(
    (row) =>
      (row.admitted === true && (row.entries || []).length) ||
      (row.excluded || []).length,
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
    const withdrawn = (row.excluded || []).filter((item) =>
      memoryFilterHit(item.skill, row.experiment_id),
    );
    if (!skills.length && !withdrawn.length) continue;
    shown += skills.length + withdrawn.length;
    nodes.push(el("div", { class: "epoch-head" }, row.experiment_id));
    for (const skill of skills)
      nodes.push(
        memoryNavItem(
          skill,
          "候选",
          { kind: "candidate", experiment_id: row.experiment_id, skill },
          `${row.experiment_id}/${skill}`,
        ),
      );
    for (const item of withdrawn)
      nodes.push(withdrawnCandidateRow(row.experiment_id, item));
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

/* Withdrawn: still named, never mounted, one click from coming back. */
function withdrawnCandidateRow(experimentId, item) {
  const reason = String(item.reason || "");
  return el(
    "div",
    {
      class: "session-item withdrawn",
      title: reason ? `已排除：${reason}` : "已排除，不再进入新会话",
    },
    el("span", { class: "label" }, item.skill),
    el(
      "button",
      {
        class: "btn small",
        onclick: (event) => {
          event.stopPropagation();
          restoreGraduatedSkill(experimentId, item.skill);
        },
      },
      "恢复",
    ),
  );
}

/* `admitted === null` means the tier itself could not be resolved, which is
   not the same answer as "contributes nothing". */
function candidateAsideReason(row) {
  if (row.error) return el("span", { class: "hint warn" }, row.error);
  if (row.admitted === null || row.admitted === undefined)
    return el("span", { class: "hint warn" }, "无法解析");
  if (!row.revealed) return el("span", { class: "hint" }, "未揭示");
  if (!row.verdict) return el("span", { class: "hint" }, "无 Held-out 记录");
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
    el("div", { class: "memory-pane-title" }, view.title),
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
      feedbackSection(entry.feedback),
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
      el("span", { class: "spacer" }),
      el(
        "button",
        {
          class: "btn small danger",
          title: "不再让新会话挂载这条 skill",
          onclick: () =>
            confirmExcludeGraduated(selection.experiment_id, selection.skill),
        },
        "排除",
      ),
    ],
    body: [
      entry.summary ? el("div", { class: "hint" }, entry.summary) : null,
      feedbackSection(entry.feedback),
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
      `保存后写回 ${memoryLibraryPath()}/${entry.name}/SKILL.md，此后启动的会话挂载新正文。`,
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
    el(
      "div",
      { class: "hint" },
      promoting
        ? `来源：实验 ${memoryView.source.experiment_id} 的 skill ${memoryView.source.skill}，整项原样复制（含 scripts/ 与 references/），复制后可在此就地删改。`
        : `新建 ${memoryLibraryPath()}/<条目名>/SKILL.md。条目名为小写 kebab-case，正文按共享 skill 格式校验后才写入。`,
    ),
    el("div", { class: "field" }, el("label", {}, "条目名"), nameInput),
  ];
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
      el("p", {}, `将从仓库目录 ${memoryLibraryPath()}/ 删除 ${name}/ 整项。`),
      el(
        "p",
        { class: "hint" },
        "运行中的会话持有启动时挂载的只读副本，不受影响；此后启动的会话不再挂载它。这是一次仓库改动，由研究者提交。",
      ),
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

/* A graduated skill is another experiment's immutable artifact: the console
   never rewrites one, it only records that sessions stop mounting it. */
function confirmExcludeGraduated(experimentId, skill) {
  const reason = el("input", {
    type: "text",
    placeholder: "可选，例如：已被更好的做法取代",
  });
  showModal(
    "从毕业层排除",
    el(
      "div",
      {},
      el("p", {}, `实验 ${experimentId} 的 skill ${skill} 将不再进入此后启动的会话。`),
      el(
        "p",
        { class: "hint" },
        "毕业实验的 skill 是它自己的不可变产物，这里不会改动它，只是记下不再挂载；已经跑过的会话不受影响。排除记录写在仓库文件里，由研究者提交，随时可以恢复。",
      ),
      el("div", { class: "field" }, el("label", {}, "原因"), reason),
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
                `/api/memory/graduated/${encodeURIComponent(experimentId)}/${encodeURIComponent(skill)}/exclude`,
                {
                  method: "POST",
                  body: JSON.stringify({ reason: reason.value }),
                },
              );
              closeModal();
              applyTierResult(result);
            } catch (error) {
              toast(`排除失败：${error.message}`, true);
            }
          },
        },
        "确认排除",
      ),
    ],
  );
}

async function restoreGraduatedSkill(experimentId, skill) {
  try {
    applyTierResult(
      await api(
        `/api/memory/graduated/${encodeURIComponent(experimentId)}/${encodeURIComponent(skill)}/exclude`,
        { method: "DELETE" },
      ),
    );
  } catch (error) {
    toast(`恢复失败：${error.message}`, true);
  }
}

/* The refreshed tier comes back with the write, so the pane never guesses. A
   withdrawn skill also stops being readable, so a selection on it is dropped. */
function applyTierResult(result) {
  if (!memoryView) return;
  memoryView.payload.graduated = result.graduated || memoryView.payload.graduated;
  toast(`${result.action === "excluded" ? "已排除" : "已恢复"} ${result.skill}`);
  const selection = memoryView.selection;
  if (
    result.action === "excluded" &&
    selection &&
    selection.kind === "candidate" &&
    selection.experiment_id === result.experiment_id &&
    selection.skill === result.skill
  ) {
    memoryView.selection = null;
    memoryView.entry = null;
    renderMemoryPane();
  }
  renderMemoryCandidates();
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
        `${name} ｜ ${fmtBytes(entry.bytes)} ｜ ${mountedSourceLabel(source)} ｜ 本实验创建时的快照副本，只读`,
      ),
      el("pre", { class: "code-view skill-body section-gap" }, entry.content || ""),
    ),
    [el("button", { class: "btn", onclick: closeModal }, "关闭")],
  );
}

function mountedEntriesList(experimentId, sources) {
  const host = el("div", {});
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
      "本实验创建时快照；库的后续改动作用于之后创建的实验。每个 Fold 与元学习会话都挂载这同一份只读副本，因此本实验各次会话看到的运行记忆完全一致。",
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
      el(
        "div",
        { class: "hint" },
        "本实验创建于快照机制之前，快照由它的第一个会话补建。",
      ),
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
function paperBanners(summary) {
  const banners = [];
  if (
    summary.state === "stale" &&
    summary.age_seconds !== null &&
    summary.age_seconds !== undefined
  ) {
    banners.push(
      el(
        "div",
        { class: "banner warn" },
        `快照已 ${Math.round(summary.age_seconds)} 秒未更新（阈值 ${Math.round(summary.stale_threshold_seconds)} 秒），以下显示最后一次写入的数据。`,
      ),
    );
  }
  if (summary.state === "export_error") {
    banners.push(
      el(
        "div",
        { class: "banner bad" },
        `写入错误：${summary.error || "writer reported ok=false"}`,
      ),
    );
  }
  if (summary.state === "unreadable") {
    banners.push(
      el(
        "div",
        { class: "banner bad" },
        `快照无法解析：${summary.error || "未知错误"}（成交/委托面板仍尝试读取各自文件）`,
      ),
    );
  }
  return banners;
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

function fmtPaperTime(value) {
  if (typeof value !== "string" || !value) return "—";
  return value.includes("T") ? fmtTs(value) : value;
}

function paperCurve(points) {
  const clean = (points || []).filter(
    (point) =>
      Number.isFinite(Number(point.equity)) && Number(point.equity) > 0,
  );
  if (!clean.length) return { series: [] };
  const initial = Number(clean[0].equity);
  let peak = initial;
  const cumulative = [],
    drawdown = [];
  for (const point of clean) {
    const value = Number(point.equity);
    peak = Math.max(peak, value);
    cumulative.push(value / initial - 1);
    drawdown.push(value / peak - 1);
  }
  return {
    series: [
      {
        key: "paper",
        label: "Paper 权益",
        dates: clean.map((point) => String(point.trade_date)),
        cum: cumulative,
        drawdown,
        final: cumulative.at(-1),
      },
    ],
  };
}

function actionCell(action) {
  const normalized = String(action || "").toLowerCase();
  const label =
    normalized === "buy"
      ? "买入"
      : normalized === "sell"
        ? "卖出"
        : action || "—";
  return el(
    "td",
    {
      class:
        normalized === "buy"
          ? "num pos"
          : normalized === "sell"
            ? "num neg"
            : "",
    },
    label,
  );
}

async function fetchTradingBundle(env, date) {
  const query = date ? `?date=${encodeURIComponent(date)}` : "";
  // /health repeats the roster entry plus an `ok` flag for external monitors;
  // the page reads the roster, so polling it too would only add a request.
  const [roster, snapshot, orders, deals, series] = await Promise.all([
    api("/api/trading/environments"),
    api(`/api/trading/${env}/snapshot`),
    api(`/api/trading/${env}/orders${query}`),
    api(`/api/trading/${env}/deals${query}`),
    api(`/api/trading/${env}/series`),
  ]);
  return { roster, snapshot, orders, deals, series };
}

async function renderTradingPage(env) {
  $main.innerHTML = '<div class="loading">加载模拟交易…</div>';
  $topbarRight.replaceChildren(
    el("span", { class: "mode-note" }, "本地日级 Paper 账户"),
  );
  tradingView = { env, date: null };
  let bundle;
  try {
    bundle = await fetchTradingBundle(env, null);
  } catch (error) {
    $main.replaceChildren(
      el("div", { class: "empty" }, `加载失败：${error.message}`),
    );
    return;
  }
  const selected =
    [
      ...new Set([
        ...(bundle.orders.available_dates || []),
        ...(bundle.deals.available_dates || []),
      ]),
    ]
      .sort()
      .at(-1) || null;
  tradingView.date = selected;
  if (
    selected &&
    selected !== bundle.orders.trade_date &&
    selected !== bundle.deals.trade_date
  ) {
    bundle = await fetchTradingBundle(env, selected);
  }
  renderTradingBundle(env, bundle);
  pollTimer = setInterval(async () => {
    if (!tradingView || location.hash !== `#/trading/${env}`) return;
    try {
      renderTradingBundle(env, await fetchTradingBundle(env, tradingView.date));
    } catch {
      /* keep last view */
    }
  }, 15000);
}

function renderTradingBundle(env, bundle) {
  if (!tradingView || tradingView.env !== env) return;
  const summary = (bundle.roster.environments || []).find(
    (item) => item.env === env,
  ) || { state: "absent" };
  const account = bundle.snapshot.snapshot || {};
  const dates = [
    ...new Set([
      ...(bundle.orders.available_dates || []),
      ...(bundle.deals.available_dates || []),
    ]),
  ]
    .sort()
    .reverse();
  const dateSelect = el(
    "select",
    { title: "交易日" },
    ...dates.map((date) => el("option", { value: date }, fmtDate(date))),
  );
  dateSelect.value = tradingView.date || summary.trade_date || "";
  dateSelect.addEventListener("change", async () => {
    tradingView.date = dateSelect.value;
    try {
      renderTradingBundle(env, await fetchTradingBundle(env, tradingView.date));
    } catch (error) {
      toast(error.message, true);
    }
  });
  const page = el(
    "div",
    { id: "trading-page" },
    paperPageHead(summary, dates.length ? dateSelect : null),
    ...paperBanners(summary),
    paperSchedulePanel(account),
    el("div", { class: "section-gap" }, paperTiles(summary, account)),
    el(
      "div",
      { class: "panel section-gap" },
      paperChartsRow(bundle.series.series || []),
    ),
    paperPositionsPanel(account),
    paperDealsPanel(bundle.deals),
    paperOrdersPanel(bundle.orders),
  );
  $main.replaceChildren(page);
}

function paperPageHead(summary, dateSelect) {
  return el(
    "div",
    { class: "page-head" },
    el("h2", {}, "Paper 模拟交易", tradingBadge(summary.state)),
    el(
      "div",
      { class: "control-bar" },
      dateSelect,
      el("span", { class: "sub" }, "日级策略订单 · 本地撮合"),
    ),
  );
}

function paperSchedulePanel(account) {
  const status =
    account.day_complete === true
      ? "当日处理完成"
      : account.phase
        ? `当前阶段：${account.phase}`
        : "等待首次日级运行";
  return el(
    "div",
    { class: "panel" },
    el(
      "div",
      { class: "control-bar" },
      el("span", { class: "mode-note" }, "策略调度"),
      el("strong", {}, "固定周期 / 固定推理时间"),
      el("span", { class: "spacer" }),
      el("span", { class: "mode-note" }, status),
    ),
    el(
      "div",
      { class: "hint" },
      "Agent 产出股票代码、操作时间、操作与数量等 JSON 订单；环境在每笔 execute_at 到达时读取对应价格并撮合。",
    ),
  );
}

function paperTiles(summary, account) {
  const positions = Array.isArray(account.positions) ? account.positions : [];
  return statTilesRow([
    { label: "总资产", value: fmtAmountOpt(account.equity) },
    { label: "可用资金", value: fmtAmountOpt(account.cash) },
    {
      label: "持仓市值",
      value: fmtAmountOpt(
        positions.reduce(
          (total, row) =>
            total + (Number(row.quantity) || 0) * (Number(row.last_price) || 0),
          0,
        ),
      ),
    },
    { label: "持仓数", value: String(positions.length) },
    { label: "当日订单", value: String(summary.order_count ?? 0) },
    { label: "当日成交", value: String(summary.deal_count ?? 0) },
  ]);
}

function paperChartsRow(points) {
  const equityCell = el(
    "div",
    { class: "chart-cell" },
    el("h4", {}, "账户权益曲线"),
  );
  const curve = paperCurve(points);
  equityCell.append(
    curve.series.length
      ? equityChart(curve, { width: 640, height: 220, ddH: 70 })
      : el("div", { class: "hint" }, "暂无权益数据"),
  );
  const cashRows = (points || [])
    .filter((point) => Number.isFinite(Number(point.cash)))
    .map((point) => ({
      label: fmtDate(point.trade_date),
      value: Number(point.cash),
    }));
  const cashCell = el(
    "div",
    { class: "chart-cell" },
    el("h4", {}, "每日现金余额"),
    cashRows.length
      ? singleSeriesBarChart(cashRows, {
          width: 640,
          height: 220,
          fmt: fmtAmount,
        })
      : el("div", { class: "hint" }, "暂无现金记录"),
  );
  return el("div", { class: "charts-row" }, equityCell, cashCell);
}

function paperPositionsPanel(account) {
  const positions = Array.isArray(account.positions) ? account.positions : [];
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el("h4", { class: "subsection-title" }, "持仓"),
  );
  if (!positions.length) {
    panel.append(el("div", { class: "empty" }, "当前无持仓"));
    return panel;
  }
  const unmapped = positions.filter((row) => row.unmapped).length;
  if (unmapped) {
    panel.append(
      el(
        "div",
        { class: "hint warn" },
        `有 ${unmapped} 行持仓无法映射，已按原样计入行数。`,
      ),
    );
  }
  panel.append(
    el(
      "div",
      { class: "orders-table-box" },
      el(
        "table",
        { class: "data" },
        el(
          "tr",
          {},
          el("th", {}, "代码"),
          el("th", {}, "数量"),
          el("th", {}, "可用"),
          el("th", {}, "成本"),
          el("th", {}, "最新价"),
          el("th", {}, "市值"),
          el("th", {}, "浮动盈亏"),
        ),
        ...positions.map((row) => {
          const quantity = Number(row.quantity) || 0;
          const marketValue = quantity * (Number(row.last_price) || 0);
          const pnl =
            quantity *
            ((Number(row.last_price) || 0) - (Number(row.average_cost) || 0));
          return el(
            "tr",
            {},
            el("td", {}, row.symbol || "—"),
            el("td", {}, String(row.quantity ?? "—")),
            el("td", {}, String(row.available_quantity ?? "—")),
            el("td", {}, fmtAmountOpt(row.average_cost)),
            el("td", {}, fmtAmountOpt(row.last_price)),
            el("td", {}, fmtAmountOpt(marketValue)),
            el("td", { class: numClass(pnl) }, fmtAmountOpt(pnl)),
          );
        }),
      ),
    ),
  );
  return panel;
}

/* A damaged JSONL line is counted by the reader, not fatal; surface the count
   so a truncated journal is visible rather than a silently shorter table. */
function skippedChip(kind, tradeDate, skipped) {
  if (!skipped) return null;
  return el(
    "span",
    { class: "stat-chip warn" },
    `${kind}_${tradeDate}.jsonl 有 ${skipped} 行无法解析`,
  );
}

function paperDealsPanel(payload) {
  const rows = payload.deals || [];
  const filled = rows.filter((row) => row.status === "filled");
  const rejected = rows.filter((row) => row.status === "rejected");
  const costs = filled.reduce(
    (total, row) =>
      total + (Number(row.commission) || 0) + (Number(row.stamp_duty) || 0),
    0,
  );
  const panel = el(
    "div",
    { class: "panel section-gap" },
    el(
      "h4",
      { class: "subsection-title" },
      `成交${payload.trade_date ? ` — ${fmtDate(payload.trade_date)}` : ""}`,
    ),
  );
  const chips = el(
    "div",
    { class: "stats-chips section-gap" },
    el("span", { class: "stat-chip" }, `成交 ${filled.length} 笔`),
    el("span", { class: "stat-chip" }, `拒单 ${rejected.length} 笔`),
    el("span", { class: "stat-chip" }, `交易费用 ${fmtAmountOpt(costs)}`),
    skippedChip("executions", payload.trade_date, payload.skipped_lines),
  );
  panel.append(chips);
  if (!rows.length) {
    panel.append(
      el(
        "div",
        { class: "empty" },
        payload.trade_date ? "该日无成交" : "暂无成交记录",
      ),
    );
    return panel;
  }
  panel.append(
    el(
      "div",
      { class: "orders-table-box" },
      el(
        "table",
        { class: "data" },
        el(
          "tr",
          {},
          el("th", {}, "时间"),
          el("th", {}, "代码"),
          el("th", {}, "方向"),
          el("th", {}, "数量"),
          el("th", {}, "价格"),
          el("th", {}, "状态"),
          el("th", {}, "说明"),
        ),
        ...rows.map((row) =>
          el(
            "tr",
            {},
            el("td", {}, fmtPaperTime(row.matched_at)),
            el("td", {}, row.symbol || "—"),
            actionCell(row.action),
            el("td", {}, String(row.quantity ?? "—")),
            el("td", {}, fmtAmountOpt(row.price)),
            el("td", {}, row.status || "—"),
            el("td", {}, row.reason || "—"),
          ),
        ),
      ),
    ),
  );
  return panel;
}

function paperOrdersPanel(payload) {
  const rows = payload.orders || [];
  const panel = el("div", { class: "panel section-gap" });
  const details = el(
    "details",
    {},
    el(
      "summary",
      { style: "cursor:pointer;font-weight:700;font-size:1.02rem" },
      `委托（${payload.trade_date ? fmtDate(payload.trade_date) : "未选择交易日"}，共 ${rows.length} 条）`,
    ),
  );
  const skipped = skippedChip(
    "orders",
    payload.trade_date,
    payload.skipped_lines,
  );
  if (skipped)
    details.append(el("div", { class: "stats-chips section-gap" }, skipped));
  if (rows.length) {
    details.append(
      el(
        "div",
        { class: "orders-table-box section-gap" },
        el(
          "table",
          { class: "data" },
          el(
            "tr",
            {},
            el("th", {}, "计划时间"),
            el("th", {}, "代码"),
            el("th", {}, "方向"),
            el("th", {}, "数量"),
          ),
          ...rows.map((row) =>
            el(
              "tr",
              {},
              el("td", {}, fmtPaperTime(row.execute_at)),
              el("td", {}, row.symbol || "—"),
              actionCell(row.action),
              el("td", {}, String(row.quantity ?? "—")),
            ),
          ),
        ),
      ),
    );
  } else {
    details.append(
      el(
        "div",
        { class: "empty" },
        payload.trade_date ? "该日无委托" : "暂无委托记录",
      ),
    );
  }
  panel.append(details);
  return panel;
}

function renderQmtPage() {
  tradingView = null;
  $topbarRight.replaceChildren(
    el("span", { class: "mode-note" }, "后端未连接"),
  );
  const unavailable = el(
    "span",
    { class: "badge state-stopped" },
    "后端未连接",
  );
  const page = el(
    "div",
    { id: "trading-page" },
    el(
      "div",
      { class: "page-head" },
      el("h2", {}, "实盘交易", unavailable),
      el("span", { class: "sub" }, "后端未连接"),
    ),
    el("div", { class: "banner warn" }, "后端未连接"),
    el(
      "div",
      { class: "panel" },
      el(
        "div",
        { class: "control-bar" },
        el("span", { class: "mode-note" }, "连接状态"),
        el("strong", {}, "后端未连接"),
        el("span", { class: "spacer" }),
        el("button", { class: "btn small", disabled: "" }, "后端未连接"),
        el("button", { class: "btn small", disabled: "" }, "后端未连接"),
      ),
      el("div", { class: "hint" }, "后端未连接"),
    ),
    el(
      "div",
      { class: "section-gap" },
      statTilesRow([
        { label: "总资产", value: "—" },
        { label: "可用资金", value: "—" },
        { label: "持仓市值", value: "—" },
        { label: "持仓数", value: "—" },
        { label: "今日成交笔数", value: "—" },
        { label: "今日成交额", value: "—" },
      ]),
    ),
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
        el("h4", {}, "账户权益曲线"),
        el("div", { class: "empty" }, "后端未连接"),
      ),
      el(
        "div",
        { class: "chart-cell" },
        el("h4", {}, "日成交额"),
        el("div", { class: "empty" }, "后端未连接"),
      ),
    ),
  );
}

function qmtEmptyPanel(title) {
  return el(
    "div",
    { class: "panel section-gap" },
    el("h4", { class: "subsection-title" }, title),
    el("div", { class: "empty" }, "后端未连接"),
  );
}

function qmtOrdersPanel() {
  return el(
    "div",
    { class: "panel section-gap" },
    el(
      "details",
      {},
      el(
        "summary",
        { style: "cursor:pointer;font-weight:700;font-size:1.02rem" },
        "委托",
      ),
      el("div", { class: "empty" }, "后端未连接"),
    ),
  );
}
