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

Period labels are error-prone to type, so the four period fields render as
dropdowns whenever the server can enumerate valid labels from the SSE trading
calendar (``build_period_options``); without a calendar they degrade to plain
text inputs.
"""

from __future__ import annotations

import bisect
import json
from pathlib import Path

import pandas as pd

from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.llm.model_profiles import MODEL_CHOICES
from autotrade.pipelines.folds import MIN_REGION_TRADE_DAYS, period_bounds
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS

# Suggested development length (test periods) per cadence: 8 spans two years at
# the default quarterly cadence — a one-year dev window is a single regime
# sample (lzp-test21 review: dev-fit strategies inverted out-of-sample).
DEV_DEFAULT_PERIODS = 8
_SCHEDULE_REGISTRY = Path(__file__).resolve().parents[3] / "configs" / "tushare_update_schedule.json"


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
_SNAPSHOT_DEFAULTS = SnapshotConfig()

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
        "key": "fold_period",
        "group": "基本与排程",
        "label": "Fold 周期",
        "type": "choice",
        "choices": ["week", "month", "quarter", "year"],
        "choice_labels": {"week": "周", "month": "月", "quarter": "季度", "year": "年"},
        "help": "每个 Fold 的验证/测试周期粒度。验证区间取测试区间的前一个同频周期；切换后下方周期选项随之变化。",
    },
    {
        "key": "first_test_period",
        "group": "基本与排程",
        "label": "首个测试周期（Fold 以测试周期命名）",
        "type": "period",
        "required": True,
        "help": "development 首个 Fold 的测试（样本外）周期；其前一个同频周期自动作为该 Fold 的验证区间，无需单独配置。",
    },
    {
        "key": "last_test_period",
        "group": "基本与排程",
        "label": "末个测试周期",
        "type": "period",
        "required": True,
        "help": "development 末个 Fold 的测试周期；与首个周期共同决定 Fold 数。每个 Fold 的验证区间 = 其测试周期的前一同频周期。",
    },
    {
        "key": "heldout_first_period",
        "group": "基本与排程",
        "label": "Held-out 起始周期",
        "type": "period",
        "required": True,
        "help": "最终冻结测试的起始周期；实验开始前冻结，必须晚于末个测试周期、不得重叠。",
    },
    {
        "key": "heldout_last_period",
        "group": "基本与排程",
        "label": "Held-out 结束周期",
        "type": "period",
        "required": True,
        "help": "最终冻结测试的结束周期。",
    },
    {"key": "epochs", "group": "基本与排程", "label": "Epoch 数", "type": "int",
     "help": "从首个 Fold 到末个 Fold 完整滚动的轮数；每个 Epoch 开始前固定运行一次元学习。"},
    {
        "key": "meta_learning_fold_interval",
        "group": "基本与排程",
        "label": "元学习 Fold 间隔",
        "type": "int",
        "min": 0,
        "help": "0=仅每个 Epoch 开始运行一次；N>0=每完成 N 个 Fold 且仍有下一 Fold 时，再运行一次元学习并更新后续 PRIOR。探索默认 2（约每两个季度一次）。",
    },
    {"key": "inherit_from", "group": "基本与排程", "label": "继承已有实验的 Agent Output", "type": "choice",
     "optional": True,
     "choices": [],  # filled at request time with experiments that have >=1 recorded fold
     "help": "留空=从空白模板开始。选择后，新实验的首个 Fold 以该实验最新冻结的策略产物（output+models）为父产物起步；创建时拷贝为只读快照，源实验之后删除也不受影响。"},
    {"key": "meta_memory_max_epochs", "group": "基本与排程", "label": "元学习原始记忆 Epoch 数", "type": "int",
     "advanced": True,
     "help": "拼接给下一次元学习的最近 Epoch 完整对话数（0 关闭原始记忆）。"},
    {"key": "fold_exploration_directive", "group": "基本与排程", "label": "默认 Fold 探索方向", "type": "text",
     "optional": True, "wide": True,
     "help": "可选。作为实验级待检验主线注入 Meta 与每个普通 Fold；Meta 据此维护 PRIOR，详情页仍可追加单会话假设。"},
    # meta_learning_directive 有意不进创建表单：进入实验详情页后在元学习会话
    # 的指令面板填写（逐 Epoch 可覆盖），避免创建时与详情页两处重复输入。
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
        "help": "决策输入快照与 Fold 输入窗口的默认历史月数；各数据域未单独覆盖时回退此值。",
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
        "choices": list(_SNAPSHOT_DEFAULTS.fundamental_datasets),
        "choice_labels": {
            name: _DATASET_LABELS[name]
            for name in _SNAPSHOT_DEFAULTS.fundamental_datasets
            if name in _DATASET_LABELS
        },
        "help": "只加载所选财务事件数据集；全不选 = 全部默认数据集。",
    },
    {
        "key": "macro_datasets",
        "group": "数据域",
        "label": "宏观数据集子集",
        "type": "multi",
        "optional": True,
        "default": [],
        "advanced": True,
        "choices": list(_SNAPSHOT_DEFAULTS.macro_datasets),
        "choice_labels": {
            name: _DATASET_LABELS[name]
            for name in _SNAPSHOT_DEFAULTS.macro_datasets
            if name in _DATASET_LABELS
        },
        "help": "只加载所选宏观数据集；全不选 = 全部默认数据集。",
    },
    {
        "key": "events_datasets",
        "group": "数据域",
        "label": "事件数据集子集",
        "type": "multi",
        "optional": True,
        "default": [],
        "advanced": True,
        "choices": list(_SNAPSHOT_DEFAULTS.events_datasets),
        "choice_labels": {
            name: _DATASET_LABELS[name]
            for name in _SNAPSHOT_DEFAULTS.events_datasets
            if name in _DATASET_LABELS
        },
        "help": "只加载所选事件数据集；全不选 = 全部默认数据集。",
    },
    {
        "key": "text_datasets",
        "group": "数据域",
        "label": "文本数据集子集",
        "type": "multi",
        "optional": True,
        "default": [],
        "advanced": True,
        "choices": list(_SNAPSHOT_DEFAULTS.text_datasets),
        "choice_labels": {
            name: _DATASET_LABELS[name]
            for name in _SNAPSHOT_DEFAULTS.text_datasets
            if name in _DATASET_LABELS
        },
        "help": "只加载所选文本数据集；全不选 = 全部默认数据集。",
    },
    # 股票筛选
    {
        "key": "screen_exclude_st",
        "group": "股票筛选",
        "label": "剔除 ST 股",
        "type": "bool",
        "help": "按锚点在市名称剔除含 ST 的股票（含 *ST）。",
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
        "help": "只保留所选板块（main=主板 gem=创业板 star=科创板 bj=北交所）；全不选 = 全部板块。",
    },
    {
        "key": "screen_exclude_new_listed_days",
        "group": "股票筛选",
        "label": "剔除新股（上市天数 <）",
        "type": "int",
        "help": "剔除锚点前 N 天内上市的新股；0 = 不剔除。",
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
    {"key": "max_fold_minutes", "group": "预算与验收", "label": "单 Fold 推理时长（分钟）", "type": "int",
     "help": "每个 Fold 和元学习会话的推理墙钟上限；回测耗时独立计算并回补。"},
    {"key": "convergence_start_epoch", "group": "预算与验收", "label": "收敛起始 Epoch", "type": "int",
     "help": "从该 Epoch（1 起）开始 Fold 提示词进入收敛阶段：优先更小更稳的策略。"},
    {"key": "min_return", "group": "预算与验收", "label": "验收目标验证收益", "type": "float",
     "help": "验证总收益目标值：低于只记警告，不阻止冻结（AcceptanceRules.min_return；硬校验为回撤/非有限/完整验证）。"},
    {"key": "min_sharpe", "group": "预算与验收", "label": "验收目标 Sharpe", "type": "float",
     "help": "验证 Sharpe 目标值：低于只记警告，不阻止冻结。"},
    {
        "key": "max_drawdown",
        "group": "预算与验收",
        "label": "验收最大回撤",
        "type": "float",
        "help": "冻结策略允许的最大验证回撤（0.25 = 25%）。",
    },
    {"key": "max_steps_per_fold", "group": "预算与验收", "label": "单 Fold Step 数上限", "type": "int",
     "help": "单 Fold 完整验证回测驱动的 Step 数上限。"},
    {"key": "max_backtests_per_fold", "group": "预算与验收", "label": "单 Fold 回测次数上限", "type": "int",
     "help": "回测独立计时（墙钟回补推理 deadline），该值限制其总次数。"},
    {"key": "max_llm_calls", "group": "预算与验收", "label": "单 Fold 模型调用上限", "type": "int",
     "help": "每个 Fold 和元学习会话的模型调用总次数上限；主循环、子代理与上下文压缩共享同一计数。"},
    {"key": "nl_failure_policy", "group": "预算与验收", "label": "NL 失败策略", "type": "choice",
     "choice_labels": {"return_error_with_audit": "返回可审计错误，策略自行降级（推荐）", "fail": "任一 NL 调用失败即终止回测"},
     "choices": ["return_error_with_audit", "fail"],
     "help": "策略内 ctx.nl() 调用失败时：返回带审计的错误结果（默认）或使回测失败。"},
    {"key": "finalize_before_deadline_seconds", "group": "预算与验收", "label": "硬收尾保留窗口（秒）", "type": "int", "advanced": True,
     "help": "距推理 deadline 该秒数且已有完整 Validation 时，只保留已有节点的回滚与显式结束；尚无完整节点时继续现有流程。"},
    {"key": "per_call_timeout_seconds", "group": "预算与验收", "label": "单次 LLM 调用超时（秒）", "type": "int", "advanced": True,
     "help": "Agent 主对话单次模型 API 调用的硬超时；默认 3600 秒，与本机网关非流式读超时一致。"},
    {"key": "disable_step_tree", "group": "预算与验收", "label": "禁用 Step 产物树", "type": "bool", "advanced": True,
     "help": "关闭跨 Fold 的 Step 谱系树（仅用于消融实验）。"},
    {"key": "record_failed_attempts", "group": "预算与验收", "label": "记录失败尝试节点", "type": "bool", "advanced": True,
     "help": "Step 树中记录未通过验证的轻量 [failed] 节点，提示后续 Fold 避开死路。"},
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
        "key": "initial_control_mode",
        "group": "运行控制",
        "label": "初始运行模式",
        "type": "choice",
        "choices": ["manual", "step", "auto"],
        "choice_labels": {"manual": "逐会话批准", "step": "逐 Step 批准（最细）", "auto": "自动运行"},
        "help": "manual：每个会话（元学习/Fold/Held-out）开始前等待批准并可注入指令；step：在 manual 基础上，每次正式验证回测后再挂起等待批准，可注入 Step 级指令（逐 Fold 可单独覆盖开关）；auto：全自动连续执行，可随时暂停。",
    },
    {"key": "analysis_model", "group": "运行控制", "label": "策略分析模型", "type": "choice",
     "choices": list(MODEL_CHOICES),
     "help": "生成 Fold 与 Step 策略分析所用的模型。"},
    {"key": "analysis_max_tokens", "group": "运行控制", "label": "策略分析输出 token 上限", "type": "int",
     "help": "单次分析调用的输出 token 配额（推理 token 计入）。"},
    {"key": "analysis_enabled", "group": "运行控制", "label": "Fold 完成后自动生成策略分析", "type": "bool",
     "help": "每个 Fold 结束后用预定义模板调用 LLM 生成自然语言策略分析（仅基于验证期证据）。"},
    {"key": "gpu_count", "group": "运行控制", "label": "默认 GPU 数量", "type": "int", "min": 0, "max": 4,
     "help": "每个元学习、Fold 和 Held-out Sandbox 默认分配的 GPU 数量（0–4）；0 表示 CPU-only，不占用 L20。大于 0 时按空闲显存自动选择，逐 Fold 设置可覆盖此默认值。"},
    {"key": "disable_meta_sandbox_rebuild", "group": "运行控制", "label": "禁用派生镜像构建", "type": "bool", "advanced": True,
     "help": "忽略元学习写出的 sandbox_environment.json，不构建派生 Docker 镜像。"},
    {"key": "meta_sandbox_rebuild_timeout_seconds", "group": "运行控制", "label": "派生镜像构建超时（秒）", "type": "int", "advanced": True,
     "help": "元学习请求新依赖时 docker build 的超时上限。"},
    {"key": "meta_sandbox_image_keep", "group": "运行控制", "label": "派生镜像保留数", "type": "int", "advanced": True,
     "help": "本实验保留的派生沙箱镜像数，更旧的尽力 GC。"},
    # 模型与上下文
    {
        "key": "model",
        "group": "模型与上下文",
        "label": "Fold Agent 主模型",
        "type": "choice",
        "choices": list(MODEL_CHOICES),
        "help": "普通 Fold Agent 主对话模型。",
    },
    {
        "key": "meta_model",
        "group": "模型与上下文",
        "label": "Meta Agent 主模型",
        "type": "choice",
        "choices": list(MODEL_CHOICES),
        "help": "元学习阶段 Agent 主对话模型；可与普通 Fold 不同。",
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
        "choices": ["max", "xhigh", "high", "medium", "low"],
        "choice_labels": {"max": "最高", "xhigh": "极高", "high": "高", "medium": "中", "low": "低"},
        "help": "启用推理模式时 Agent 与 NL 调用的推理强度；可用档位由所选模型服务决定。",
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
        "label": "压缩触发 token 阈值",
        "type": "int",
        "advanced": True,
        "help": "估算上下文 token 超过该值时触发语义压缩。",
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


def build_period_options(trading_days: list[str]) -> dict[str, list[str]]:
    """Enumerate complete, backtestable period labels per cadence.

    A label qualifies when its calendar bounds are fully covered by the trading
    calendar and it holds at least MIN_REGION_TRADE_DAYS trading days (the
    replay reserves the final day for forced liquidation). Oldest -> newest.
    """

    days = sorted({str(day) for day in trading_days})
    if not days:
        return {}
    first, last = days[0], days[-1]

    def qualified(label: str, cadence: str) -> bool:
        start, end = period_bounds(label, period=cadence)
        if end > last or end < first:
            return False
        count = bisect.bisect_right(days, end) - bisect.bisect_left(days, start)
        return count >= MIN_REGION_TRADE_DAYS

    first_ts, last_ts = pd.Timestamp(first), pd.Timestamp(last)
    candidates = {
        "week": [stamp.strftime("%Y%m%d") for stamp in pd.date_range(first_ts, last_ts, freq="W-MON")],
        "month": [period.strftime("%Y%m") for period in pd.period_range(first_ts, last_ts, freq="M")],
        "quarter": [f"{period.year}Q{period.quarter}" for period in pd.period_range(first_ts, last_ts, freq="Q")],
        "year": [period.strftime("%Y") for period in pd.period_range(first_ts, last_ts, freq="Y")],
    }
    return {
        cadence: [label for label in labels if qualified(label, cadence)]
        for cadence, labels in candidates.items()
        if any(qualified(label, cadence) for label in labels)
    }


def suggest_period_defaults(options: dict[str, list[str]]) -> dict[str, dict[str, str]]:
    """Safe defaults per cadence: recent development window + the latest complete
    period as held-out (held-out must follow development without overlap)."""

    defaults: dict[str, dict[str, str]] = {}
    preferred = {
        "first_test_period": str(WEB_CREATE_DEFAULTS["first_test_period"]),
        "last_test_period": str(WEB_CREATE_DEFAULTS["last_test_period"]),
        "heldout_first_period": str(WEB_CREATE_DEFAULTS["heldout_first_period"]),
        "heldout_last_period": str(WEB_CREATE_DEFAULTS["heldout_last_period"]),
    }
    for cadence, labels in options.items():
        if len(labels) < 3:
            continue
        if cadence == "quarter" and set(preferred.values()).issubset(labels):
            defaults[cadence] = dict(preferred)
            continue
        defaults[cadence] = {
            "first_test_period": labels[max(1, len(labels) - 1 - DEV_DEFAULT_PERIODS)],
            "last_test_period": labels[-2],
            "heldout_first_period": labels[-1],
            "heldout_last_period": labels[-1],
        }
    return defaults


def parameter_schema(
    trading_days: list[str] | None = None, inherit_sources: list[str] | None = None
) -> dict[str, object]:
    """Grouped field schema with live defaults for the creation modal.

    With a trading calendar the four period fields become dependent dropdowns
    (``type: period`` + top-level ``period_options``/``period_defaults``);
    without one they degrade to required text inputs. ``inherit_sources``
    fills the inherit_from dropdown (experiments with >=1 recorded fold).
    """

    period_options = build_period_options(trading_days or [])
    period_defaults = suggest_period_defaults(period_options)
    default_cadence = str(WEB_CREATE_DEFAULTS["fold_period"])
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
        if entry["type"] == "period":
            if period_options:
                default = period_defaults.get(default_cadence, {}).get(key)
            else:
                entry["type"] = "string"
        if key == "inherit_from":
            entry["choices"] = ["", *(inherit_sources or [])]
        entry["default"] = default
        groups[str(entry.pop("group"))].append(entry)
    return {
        "schema_version": 2,
        "groups": [{"name": name, "fields": entries} for name, entries in groups.items() if entries],
        "period_options": period_options,
        "period_defaults": period_defaults,
    }
