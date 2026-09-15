"""Creation-form parameter schema for the HITL console.

Field keys mirror ``autotrade.pipelines.hitl_state.WEB_CREATE_DEFAULTS`` (which
in turn mirror the run_experiment.py CLI dests); defaults are read from there so
this schema can never drift from the worker. Descriptions follow
docs/parameters-reference.md.

Deliberately NOT exposed in the form: ``experiments_root``/``work_root`` are
force-overwritten with manager-owned values on creation (ExperimentManager);
``workspace_reference`` is persisted and accepted by create/worker but has no
form field, so it is set in ``params.json``; and ``WEB_INTERNAL_PARAMS``
describe the only supported research environment — the console API rejects them
outright, so they can only be set in a worker-side ``params.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from autotrade.environment.data.snapshot import DEFAULT_DATASETS, SELECTABLE_DATASETS
from autotrade.environment.llm.model_profiles import MODEL_CHOICES
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from autotrade.pipelines.skills import (
    OPERATING_MEMORY_LIBRARY,
    OPERATING_MEMORY_MODES,
    build_skills_index,
)

_SCHEDULE_REGISTRY = Path(__file__).resolve().parents[3] / "configs" / "tushare_update_schedule.json"
_OPERATING_MEMORY_LIBRARY = Path(__file__).resolve().parents[3] / OPERATING_MEMORY_LIBRARY


def _operating_memory_entries() -> list[dict[str, str]]:
    """Curated operating-memory entries, used only to describe the mode field.

    A missing or malformed library degrades to an empty list (display-only
    concern); what actually mounts is decided by the session at run time."""

    try:
        index = build_skills_index(_OPERATING_MEMORY_LIBRARY)
    except (OSError, ValueError):
        return []
    return [
        {"name": str(entry["name"]), "title": str(entry["title"])}
        for entry in index["skills"]  # type: ignore[union-attr]
    ]


def _dataset_labels() -> dict[str, str]:
    """Chinese display names for dataset chips, derived from the schedule
    registry's per-interface descriptor (its leading clause) — the registry is
    already the operational fact source for datasets, so there is no second
    mapping to maintain. Missing/unreadable registry -> chips fall back to the
    raw API names (display-only concern)."""

    try:
        interfaces = json.loads(_SCHEDULE_REGISTRY.read_text(encoding="utf-8"))["interfaces"]
    except (OSError, ValueError, KeyError):
        return {}
    labels: dict[str, str] = {}
    for row in interfaces:
        text = str(row.get("official_update", "")).split("；")[0].split("。")[0]
        if len(text) > 12:
            text = text.split("（")[0]
        if text:
            labels[str(row.get("dataset", ""))] = text
    return labels


_DATASET_LABELS = _dataset_labels()
_OPERATING_MEMORY_ENTRIES = _operating_memory_entries()

_FIELDS: list[dict[str, object]] = [
    # 基本与排程
    {
        "key": "experiment_id",
        "group": "基本与排程",
        "label": "实验名称（ID）",
        "type": "string",
        "required": True,
        "help": "唯一实验标识，仅限字母、数字、下划线和连字符；对应 experiments/<id>/ 目录。",
    },
    {
        "key": "research_start",
        "group": "基本与排程",
        "label": "研究期起点",
        "type": "string",
        "required": True,
        "help": "YYYYMMDD，必须是 7 月 1 日。研究期按整年（7 月至次年 6 月）切块，研究会话只看研究期末的决策视图、只回放研究期。",
    },
    {
        "key": "research_end",
        "group": "基本与排程",
        "label": "研究期终点",
        "type": "string",
        "required": True,
        "help": "YYYYMMDD，必须是某个 6 月 30 日，且晚于起点。",
    },
    {
        "key": "forward_end",
        "group": "基本与排程",
        "label": "前推期终点",
        "type": "string",
        "required": True,
        "help": "YYYYMMDD，必须是研究期终点之后第 12 个月的 6 月 30 日。冻结产物从研究期终点次日起连续回放到这里，再接着回放 Held-out。",
    },
    {
        "key": "heldout_end",
        "group": "基本与排程",
        "label": "Held-out 终点",
        "type": "string",
        "required": True,
        "help": "YYYYMMDD，晚于前推期终点；回放截到已固定发布的最后一个交易日，截断会记在前推记录里。",
    },
    {
        "key": "research_sessions",
        "group": "基本与排程",
        "label": "研究会话数",
        "type": "int",
        "help": "背靠背运行的研究会话数；每个会话以继续、冻结或无边际结束，任一会话可冻结，至多冻结一次。",
    },
    {"key": "research_directive", "group": "基本与排程", "label": "默认探索方向", "type": "text",
     "optional": True, "wide": True,
     "help": "可选。作为实验级待检验主线注入每个研究会话，详情页仍可追加单会话假设。"},
    {
        "key": "strategy_period",
        "group": "基本与排程",
        "label": "策略调用周期",
        "type": "choice",
        "choices": ["day", "month", "quarter", "year"],
        "choice_labels": {"day": "日", "month": "月", "quarter": "季度", "year": "年"},
        "help": "日级 JSON ABI 的策略推理排程，不启用分钟级策略循环。",
    },
    {
        "key": "inference_time",
        "group": "基本与排程",
        "label": "固定推理时间",
        "type": "time",
        "required": True,
        "help": "Asia/Shanghai 24 小时制 HH:MM。",
    },
    # 数据窗口
    {
        "key": "window_months",
        "group": "数据窗口",
        "label": "基础历史窗口（月）",
        "type": "int",
        "help": "决策输入快照的默认历史月数；各数据域未单独覆盖时回退此值。",
    },
    {
        "key": "daily_window_months",
        "group": "数据窗口",
        "label": "daily 域窗口（月）",
        "type": "int",
        "optional": True,
        "advanced": True,
        "help": "日线域单独窗口；留空回退基础窗口。",
    },
    {
        "key": "fundamentals_window_months",
        "group": "数据窗口",
        "label": "fundamentals 域窗口（月）",
        "type": "int",
        "optional": True,
        "advanced": True,
        "help": "基本面域单独窗口；留空回退基础窗口。",
    },
    {
        "key": "macro_window_months",
        "group": "数据窗口",
        "label": "macro 域窗口（月）",
        "type": "int",
        "optional": True,
        "advanced": True,
        "help": "宏观域单独窗口；留空回退基础窗口。",
    },
    {
        "key": "intraday_trade_days",
        "group": "数据窗口",
        "label": "历史分钟线交易日窗口",
        "type": "int",
        "help": "决策输入快照包含的最近可见历史分钟线交易日数。",
    },
    {
        "key": "events_window_months",
        "group": "数据窗口",
        "label": "events 域窗口（月）",
        "type": "int",
        "optional": True,
        "advanced": True,
        "help": "事件域单独窗口；留空回退基础窗口。",
    },
    {
        "key": "text_window_months",
        "group": "数据窗口",
        "label": "text 域窗口（月）",
        "type": "int",
        "optional": True,
        "advanced": True,
        "help": "文本域单独窗口；留空回退基础窗口。",
    },
    # 数据域
    {
        "key": "include_fundamentals",
        "group": "数据域",
        "label": "财务/基本面域",
        "type": "bool",
        "help": "财报、业绩预告/快报、分红等 PIT 财务事件；关闭后不加载。",
    },
    {
        "key": "include_macro",
        "group": "数据域",
        "label": "宏观/指数域",
        "type": "bool",
        "help": "宏观指标、利率、宽基指数、行业指数等市场背景；关闭后不加载。",
    },
    {
        "key": "include_intraday",
        "group": "数据域",
        "label": "历史分钟线域",
        "type": "bool",
        "help": "仅把推断时点前可见的历史分钟线作为 PIT 输入；不恢复实时分钟采集或分钟级策略回放。竞价 PIT 数据始终保留。",
    },
    {
        "key": "include_events",
        "group": "数据域",
        "label": "事件/资金域",
        "type": "bool",
        "help": "两融、资金流、股东、龙虎榜、打板情绪等事件面板；关闭后决策快照与回放均不加载。",
    },
    {
        "key": "include_text",
        "group": "数据域",
        "label": "文本域",
        "type": "bool",
        "help": "公告、新闻、研报、互动问答等文本证据；关闭后不加载（ctx.nl 检索也无文本可用）。",
    },
    {
        "key": "fundamental_datasets",
        "group": "数据域",
        "label": "财务数据集子集",
        "type": "multi",
        "optional": True,
        "default": [],
        "advanced": True,
        "choices": list(SELECTABLE_DATASETS["fundamentals"]),
        "choice_labels": {
            name: _DATASET_LABELS[name]
            for name in SELECTABLE_DATASETS["fundamentals"]
            if name in _DATASET_LABELS
        },
        "help": (
            "只加载所选财务事件数据集（可选中默认范围之外的数据集）；全不选 = 默认数据集集合："
            + "、".join(DEFAULT_DATASETS["fundamentals"])
            + "。"
        ),
    },
    {
        "key": "macro_datasets",
        "group": "数据域",
        "label": "宏观数据集子集",
        "type": "multi",
        "optional": True,
        "default": [],
        "advanced": True,
        "choices": list(SELECTABLE_DATASETS["macro"]),
        "choice_labels": {
            name: _DATASET_LABELS[name]
            for name in SELECTABLE_DATASETS["macro"]
            if name in _DATASET_LABELS
        },
        "help": (
            "只加载所选宏观数据集（可选中默认范围之外的数据集）；全不选 = 默认数据集集合："
            + "、".join(DEFAULT_DATASETS["macro"])
            + "。"
        ),
    },
    {
        "key": "events_datasets",
        "group": "数据域",
        "label": "事件数据集子集",
        "type": "multi",
        "optional": True,
        "default": [],
        "advanced": True,
        "choices": list(SELECTABLE_DATASETS["events"]),
        "choice_labels": {
            name: _DATASET_LABELS[name]
            for name in SELECTABLE_DATASETS["events"]
            if name in _DATASET_LABELS
        },
        "help": (
            "只加载所选事件数据集（可选中默认范围之外的数据集）；全不选 = 默认数据集集合："
            + "、".join(DEFAULT_DATASETS["events"])
            + "。"
        ),
    },
    {
        "key": "text_datasets",
        "group": "数据域",
        "label": "文本数据集子集",
        "type": "multi",
        "optional": True,
        "default": [],
        "advanced": True,
        "choices": list(SELECTABLE_DATASETS["text"]),
        "choice_labels": {
            name: _DATASET_LABELS[name]
            for name in SELECTABLE_DATASETS["text"]
            if name in _DATASET_LABELS
        },
        "help": (
            "只加载所选文本数据集（可选中默认范围之外的数据集）；全不选 = 默认数据集集合："
            + "、".join(DEFAULT_DATASETS["text"])
            + "。"
        ),
    },
    {
        "key": "pit_views_seed",
        "group": "数据域",
        "label": "PIT 视图种子目录",
        "type": "string",
        "advanced": True,
        "help": (
            "仓库相对路径，指向本实验硬链接决策/回放视图的预构建种子。默认种子只按默认数据集集合构建："
            "改动上面任一数据集子集、数据域开关或窗口后，必须先用 scripts/data/prebuild_pit_views_seed.py "
            "以同一套参数预构建一棵新种子并填在这里，否则本实验会逐个区间冷构建（数小时）。"
            "填写的目录必须存在，且其记录的快照配置与本实验逐字一致，否则创建直接失败。"
            "填写后本实验固定该种子构建时的 research release（而非最新数据批次），"
            "该 release 未发布或在 Held-out 内不足两个交易日时同样创建失败。"
        ),
    },
    # 股票筛选
    {
        "key": "screen_exclude_st",
        "group": "股票筛选",
        "label": "剔除 ST 股",
        "type": "bool",
        "help": "按锚点在市名称剔除含 ST 的股票（含 *ST）；默认不剔除，由策略自行筛选。",
    },
    {
        "key": "screen_boards",
        "group": "股票筛选",
        "label": "板块范围",
        "type": "multi",
        "optional": True,
        "default": [],
        "wide": False,
        "choices": ["main", "gem", "star", "bj"],
        "choice_labels": {"main": "主板", "gem": "创业板", "star": "科创板", "bj": "北交所"},
        "help": "只保留所选板块（main=主板 gem=创业板 star=科创板 bj=北交所）；全不选（默认）= 全部板块，由策略自行筛选。",
    },
    {
        "key": "screen_exclude_new_listed_days",
        "group": "股票筛选",
        "label": "剔除新股（上市天数 <）",
        "type": "int",
        "help": "剔除锚点前 N 天内上市的新股；0（默认）= 不剔除，由策略自行筛选。",
    },
    {
        "key": "screen_min_circ_mv_yi",
        "group": "股票筛选",
        "label": "流通市值下限（亿元）",
        "type": "float",
        "optional": True,
        "help": "只保留锚点流通市值不低于该值的股票（如填 100 = 只做大盘股）；留空不限制。",
    },
    {
        "key": "screen_max_circ_mv_yi",
        "group": "股票筛选",
        "label": "流通市值上限（亿元）",
        "type": "float",
        "optional": True,
        "advanced": True,
        "help": "只保留锚点流通市值不高于该值的股票（小盘研究）；留空不限制。",
    },
    {
        "key": "screen_min_price",
        "group": "股票筛选",
        "label": "股价下限（元）",
        "type": "float",
        "optional": True,
        "advanced": True,
        "help": "剔除锚点收盘价低于该值的股票（低价股/仙股）；留空不限制。",
    },
    {
        "key": "screen_max_price",
        "group": "股票筛选",
        "label": "股价上限（元）",
        "type": "float",
        "optional": True,
        "advanced": True,
        "help": "剔除锚点收盘价高于该值的股票；留空不限制。",
    },
    # 预算与验收
    {"key": "max_session_minutes", "group": "预算与验收", "label": "单会话推理时长（分钟）", "type": "int",
     "help": "每个研究会话的推理墙钟上限；回测耗时独立计算并回补。"},
    {"key": "min_return", "group": "预算与验收", "label": "验收目标验证收益", "type": "float",
     "help": "验证总收益目标值：低于只记警告，不阻止冻结（AcceptanceRules.min_return；冻结的硬校验只剩非有限指标与完整验证）。"},
    {"key": "min_sharpe", "group": "预算与验收", "label": "验收目标 Sharpe", "type": "float",
     "help": "验证 Sharpe 目标值：低于只记警告，不阻止冻结。"},
    {
        "key": "max_drawdown",
        "group": "预算与验收",
        "label": "验收最大回撤",
        "type": "float",
        "help": "回撤上限（0.25 = 25%）：研究期超限只记警告；毕业裁决要求前推与 Held-out 两段的回撤都不超过它。",
    },
    {
        "key": "cost_stress_multiplier",
        "group": "预算与验收",
        "label": "毕业成本压力倍数",
        "type": "float",
        "help": "毕业裁决的成本压力：前推段中性化超额在滑点放大到该倍数后仍须为正（按该段换手定价）。",
    },
    {"key": "max_replay_years_per_session", "group": "预算与验收", "label": "单会话回放预算（replay-year）", "type": "int",
     "help": "一个候选在其验证区间覆盖的每个研究年份计 1：完整研究期计研究年数，一批按候选数乘年数预留；回测独立计时（墙钟回补推理 deadline）。"},
    {"key": "max_null_controls_per_session", "group": "预算与验收", "label": "单会话按需空对照次数上限", "type": "int",
     "help": "研究会话内 run_null_control 工具的调用上限（冻结节点复用其结果）；0 表示不注册该工具。"},
    {"key": "max_llm_calls", "group": "预算与验收", "label": "单会话模型调用上限", "type": "int",
     "help": "每个研究会话的模型调用总次数上限；主循环、子代理与上下文压缩共享同一计数。"},
    {"key": "nl_failure_policy", "group": "预算与验收", "label": "NL 失败策略", "type": "choice",
     "choice_labels": {"return_error_with_audit": "返回可审计错误，策略自行降级（推荐）", "fail": "任一 NL 调用失败即终止回测"},
     "choices": ["return_error_with_audit", "fail"],
     "help": "策略内 ctx.nl() 调用失败时：返回带审计的错误结果（默认）或使回测失败。"},
    {"key": "finalize_before_deadline_seconds", "group": "预算与验收", "label": "硬收尾保留窗口（秒）", "type": "int", "advanced": True,
     "help": "距推理 deadline 该秒数且已有完整 Validation 时，只保留已有节点的回滚与显式结束；尚无完整节点时继续现有流程。"},
    {"key": "per_call_timeout_seconds", "group": "预算与验收", "label": "单次 LLM 调用超时（秒）", "type": "int", "advanced": True,
     "help": "Agent 主对话单次模型 API 调用的硬超时；默认 3600 秒，与本机网关非流式读超时一致。"},
    {"key": "strategy_fit_timeout_seconds", "group": "预算与验收", "label": "单次策略 fit 超时（秒）", "type": "int", "advanced": True,
     "help": "正式策略可选 fit(context) 单次调用的墙钟上限：回放开始前先训练一次模型，之后按 REFIT_PERIOD 在新周期首个决策日重训；超时或异常即整场回测失败。generate_orders 的单日推断上限（`strategy_inference_timeout_seconds`，代码默认 360 秒）不受影响。"},
    {"key": "disable_step_tree", "group": "预算与验收", "label": "禁用 Step 产物树", "type": "bool", "advanced": True,
     "help": "关闭跨会话的 Step 谱系树（仅用于消融实验）。"},
    {"key": "record_failed_attempts", "group": "预算与验收", "label": "记录失败尝试节点", "type": "bool", "advanced": True,
     "help": "Step 树中记录未通过验证的轻量 [failed] 节点，提示后续会话避开死路。"},
    # Broker 账户
    {"key": "initial_cash", "group": "Broker 账户", "label": "初始资金（元）", "type": "float",
     "help": "long-only 现金账户初始资金，也是组合的初始权益。"},
    {"key": "max_total_holdings", "group": "Broker 账户", "label": "最大持仓数（可选）", "type": "int", "optional": True,
     "help": "最大同时持仓代码数；留空交给 Agent 自控。"},
    {"key": "max_single_name_weight", "group": "Broker 账户", "label": "单票权重上限（可选）", "type": "float", "optional": True,
     "help": "单只股票占组合权益的名义上限（0.2 = 20%）；留空交给 Agent 自控。"},
    {"key": "commission_bps", "group": "Broker 账户", "label": "佣金（bp）", "type": "float", "advanced": True,
     "help": "万一 = 1.0；受最低佣金 5 元/笔约束。"},
    {"key": "slippage_bps", "group": "Broker 账户", "label": "市价滑点（bp）", "type": "float", "advanced": True,
     "help": "市价 taker 成交滑点；限价/竞价成交不计滑点。"},
    # 运行控制
    {
        "key": "operating_memory",
        "group": "运行控制",
        "label": "跨实验运行记忆",
        "type": "choice",
        "choices": list(OPERATING_MEMORY_MODES),
        "choice_labels": {
            "none": "不挂载",
            "curated": "只挂载策展条目",
            "curated+graduated": "策展条目 + 毕业实验的 skills",
        },
        "help": (
            "把跨实验知识只读挂载进每个研究会话工作区："
            f"策展层是仓库里人工维护的 {len(_OPERATING_MEMORY_ENTRIES)} 条运行经验；"
            "毕业层是前推与 Held-out 判定为 graduated 的实验自己写下的 skills，"
            "带来源实验与判定标记，由 Agent 自行取舍。会话不能改写或删除挂载内容。"
        ),
    },
    {"key": "gpu_count", "group": "运行控制", "label": "默认 GPU 数量", "type": "int", "min": 0, "max": 4,
     "help": "每个研究会话与正式回放 Sandbox 默认分配的 GPU 数量（0–4）；0 表示 CPU-only，不占用 L20。大于 0 时按空闲显存自动选择，逐会话设置可覆盖此默认值。"},
    # 模型与上下文
    {
        "key": "model",
        "group": "模型与上下文",
        "label": "研究 Agent 主模型",
        "type": "choice",
        "choices": list(MODEL_CHOICES),
        "help": "研究会话 Agent 主对话模型。",
    },
    {
        "key": "subagent_model",
        "group": "模型与上下文",
        "label": "子代理模型",
        "type": "choice",
        "choices": list(MODEL_CHOICES),
        "help": "研究会话用 agent 工具启动的子代理所用模型；共享父会话的调用配额与时间预算，压缩阈值按该模型的上下文窗口推导。",
    },
    {
        "key": "nl_model",
        "group": "模型与上下文",
        "label": "NL 子代理模型",
        "type": "choice",
        "choices": list(MODEL_CHOICES),
        "help": "策略内 ctx.nl() 文本分析子代理模型。",
    },
    {
        "key": "compact_model",
        "group": "模型与上下文",
        "label": "上下文压缩模型",
        "type": "choice",
        "choices": list(MODEL_CHOICES),
        "help": "语义压缩长会话所用的低成本模型（不启用推理模式）。",
    },
    {
        "key": "reasoning_effort",
        "group": "模型与上下文",
        "label": "推理强度",
        "type": "choice",
        "choices": ["xhigh", "medium", "low"],
        "choice_labels": {"xhigh": "极高", "medium": "中", "low": "低"},
        "help": "启用推理模式时 Agent 主对话与子代理的推理强度；三档即本机 Qwen 模板在线上真正区分的档位（旧参数中的 high/max 等同 xhigh）。NL 固定 medium。",
    },
    {
        "key": "no_thinking",
        "group": "模型与上下文",
        "label": "禁用推理模式",
        "type": "bool",
        "advanced": True,
        "help": "关闭模型推理模式（Agent 与 NL 调用）。",
    },
    {
        "key": "disable_context_compact",
        "group": "模型与上下文",
        "label": "禁用语义压缩",
        "type": "bool",
        "advanced": True,
        "help": "关闭长会话语义上下文压缩。",
    },
    {
        "key": "compact_token_threshold",
        "group": "模型与上下文",
        "label": "压缩触发 token 阈值（可选）",
        "type": "int",
        "optional": True,
        "advanced": True,
        "help": "估算上下文 token 超过该值时触发语义压缩。留空按模型推导：上下文窗口 − 输出上限 − 8,192；填入的值也夹到该上限。",
    },
    {
        "key": "compact_keep_recent_messages",
        "group": "模型与上下文",
        "label": "压缩保留最近消息数",
        "type": "int",
        "advanced": True,
        "help": "语义压缩后保留的最近原始消息条数。",
    },
    {
        "key": "compact_max_tokens",
        "group": "模型与上下文",
        "label": "单次压缩输出 token 上限",
        "type": "int",
        "advanced": True,
        "help": "一次压缩摘要的最大输出 token。",
    },
    {
        "key": "compact_max_calls",
        "group": "模型与上下文",
        "label": "单会话压缩调用上限",
        "type": "int",
        "advanced": True,
        "help": "单个 Agent 会话的语义压缩调用次数上限。",
    },
]

_GROUP_ORDER = (
    "基本与排程",
    "数据窗口",
    "数据域",
    "股票筛选",
    "预算与验收",
    "Broker 账户",
    "运行控制",
    "模型与上下文",
)


def parameter_schema() -> dict[str, object]:
    """Grouped field schema with live defaults for the creation modal."""

    groups: dict[str, list[dict[str, object]]] = {name: [] for name in _GROUP_ORDER}
    for field in _FIELDS:
        entry = dict(field)
        key = str(entry["key"])
        # The worker's accepted-parameter table is the single source of truth for
        # what the form may offer: ExperimentManager rejects anything outside it,
        # so rendering a field it does not accept would be a control that 400s on
        # submit. A key that is not (yet) accepted is simply not shown.
        if key not in WEB_CREATE_DEFAULTS:
            continue
        default = WEB_CREATE_DEFAULTS[key]
        if isinstance(default, tuple):
            default = list(default)
        entry["default"] = default
        groups[str(entry.pop("group"))].append(entry)
    return {
        "schema_version": 3,
        "groups": [{"name": name, "fields": entries} for name, entries in groups.items() if entries],
    }
