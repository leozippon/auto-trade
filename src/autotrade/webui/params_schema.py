"""Creation-form parameter schema for the HITL console.

Every field is a console parameter of ``pipelines.experiment_params.ExperimentParams`` and
renders its default from ``hitl_state.WEB_CREATE_DEFAULTS`` and whether a
create must state it from the record's ``required`` (both read off the record),
so the form can never drift from what a create stamps. Each console
parameter has a form field here or is named in ``_FORMLESS``; the check below
refuses either list drifting from the record. Descriptions follow
docs/PARAMETERS_REFERENCE.md.

Deliberately NOT exposed in the form: the managed and host parameters, which
the console API rejects outright, so they can only be set in a worker-side
``params.json``; and the console parameters in ``_FORMLESS``.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.common.records import read_json_object
from autotrade.environment.data.contracts import BENCHMARK_INDEXES
from autotrade.environment.data.snapshot import DEFAULT_DATASETS, SELECTABLE_DATASETS
from autotrade.environment.llm.model_profiles import LLM_KEYS, MODEL_CHOICES
from autotrade.environment.llm.openai_compatible import REASONING_EFFORTS
from autotrade.pipelines.config import (
    BATCH_VALIDATE_MAX_CANDIDATES,
    MANDATED_DEFAULTS,
    acceptance_default,
)
from autotrade.pipelines.experiment_params import PARAMETERS, Surface, parameters
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from autotrade.pipelines.skills import (
    SKILL_LIBRARY_MODES,
    SKILL_LIBRARY_PATH,
    build_skills_index,
)

_SCHEDULE_REGISTRY = Path(__file__).resolve().parents[3] / "configs" / "tushare_update_schedule.json"
_SKILL_LIBRARY_PATH = Path(__file__).resolve().parents[3] / SKILL_LIBRARY_PATH


def _skill_library_entries() -> list[dict[str, str]]:
    """Curated skill-library entries, used only to describe the mode field.

    A missing or malformed library degrades to an empty list (display-only
    concern); what actually mounts is decided by the session at run time."""

    try:
        index = build_skills_index(_SKILL_LIBRARY_PATH)
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
        interfaces = read_json_object(_SCHEDULE_REGISTRY)["interfaces"]
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
_SKILL_LIBRARY_ENTRIES = _skill_library_entries()

_FIELDS: list[dict[str, object]] = [
    # 基本与排程
    {
        "key": "experiment_id",
        "group": "基本与排程",
        "label": "实验名称（ID）",
        "type": "string",
        "help": "唯一实验标识，仅限字母、数字、下划线和连字符；对应 experiments/<id>/ 目录。",
    },
    {
        "key": "research_start",
        "group": "基本与排程",
        "label": "研究期起点",
        "type": "string",
        "help": "YYYYMMDD，必须是某月 1 日。研究期从这一天起按完整十二个月切块，研究会话只看研究期末的决策视图、只回测研究期。",
    },
    {
        "key": "research_end",
        "group": "基本与排程",
        "label": "研究期终点",
        "type": "string",
        "help": "YYYYMMDD，必须是起点同一月份至少一个完整周年后的前一天。",
    },
    {
        "key": "forward_end",
        "group": "基本与排程",
        "label": "前推期终点",
        "type": "string",
        "help": "YYYYMMDD，必须是研究期终点之后恰好十二个月的那一天。定版产物从研究期终点次日起连续回测到这里，再接着回测 Held-out。",
    },
    {
        "key": "heldout_end",
        "group": "基本与排程",
        "label": "Held-out 终点",
        "type": "string",
        "help": "YYYYMMDD，晚于前推期终点；回测截到已固定发布的最后一个交易日，截断会记在前推记录里。",
    },
    {"key": "research_directive", "group": "基本与排程", "label": "研究方向", "type": "text",
     "optional": True, "wide": True,
     "help": "可选。作为实验级待检验主线注入研究会话的系统提示，详情页仍可在会话启动前追加假设。"},
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
        "help": "Asia/Shanghai 24 小时制 HH:MM。",
    },
    {
        "key": "benchmark_index",
        "group": "基本与排程",
        "type": "choice",
        "label": "基准指数",
        "choices": list(BENCHMARK_INDEXES),
        "choice_labels": {
            code: f"{label}（{code}）" for code, label in BENCHMARK_INDEXES.items()
        },
        "help": (
            "本实验被度量的基准：风格归因与载荷回归的基准腿，以及随机对照组合"
            "按成分股替换选名的成分表。创建时校验该指数在本实验固定的数据发布中"
            "确实有 index_daily 与 index_weight 分区；成分股本身仍需在 macro "
            "数据集里选上 index_weight 才能被策略读到。"
        ),
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
        "help": "两融、资金流、股东、龙虎榜、打板情绪等事件面板；关闭后决策快照与回测均不加载。",
    },
    {
        "key": "include_text",
        "group": "数据域",
        "label": "文本域",
        "type": "bool",
        "help": "公告、新闻、研报、互动问答等文本证据；关闭后不加载。",
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
        "label": "预构建 PIT 数据视图目录",
        "type": "string",
        "advanced": True,
        "help": (
            "仓库相对路径，指向本实验通过硬链接复用的预构建决策/回测视图；默认目录只含默认研究几何、"
            "窗口与数据集集合的视图。改动数据集、数据域开关、窗口或研究几何后，需用 "
            "scripts/data/prebuild_pit_views_seed.py 按相同参数生成新的视图目录，或留空不复用"
            "预构建视图（逐区间构建，可能耗时数小时，用最新数据批次）。"
            "目录必须存在，记录的快照配置须与本实验一致，否则创建失败。"
            "本实验使用这些视图构建时固定的 research release，而非最新数据批次；"
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
    {"key": "max_research_minutes", "group": "预算与验收", "label": "研究会话推理时长（分钟）", "type": "int",
     "help": "研究会话的有效推理时间上限，跨中断后的续跑累计；回测耗时独立计算并回补。"},
    # The tracking mandate's switch. Filling the cap turns the mandate on and
    # fills a blank beta band; drawdowns stay 0.45 / 0.30 unless named.
    {
        "key": "tracking_error_cap",
        "group": "预算与验收",
        "label": "跟踪误差上限（对基准指数）",
        "type": "float",
        "optional": True,
        "help": "跟踪授权的开关：填入即为本实验设定跟踪授权——策略自身对本实验基准指数的残差跟踪误差上限（0.08 = 8%/年），定版登记时候选须满足，前推期只报告。留空即不设授权，跟踪误差与 β 只报告、不评级。没有上限却填 β 区间会被拒绝。",
    },
    {
        "key": "max_drawdown",
        "group": "预算与验收",
        "label": "权益最大回撤",
        "type": "float",
        "optional": True,
        "placeholder": "留空：0.45",
        "help": "权益回撤上限（0.45 = 45%）：定版登记时候选不得超过，Paper 账户也按它止损；前推与 Held-out 只报告。留空取 0.45，与有没有跟踪授权无关。",
    },
    {
        "key": "active_max_drawdown",
        "group": "预算与验收",
        "label": "主动序列最大回撤",
        "type": "float",
        "optional": True,
        "placeholder": "留空：0.30",
        "help": "主动序列（策略日收益减随机对照组合合成收益）累计净值的回撤上限，Paper 账户按它止损；前推与 Held-out 只报告。留空取 0.30，与有没有跟踪授权无关。",
    },
    {
        "key": "beta_min",
        "group": "预算与验收",
        "label": "市场 β 下限",
        "type": "float",
        "optional": True,
        "advanced": True,
        "mandate_paired": True,
        "help": "跟踪授权的 β 区间下限，只在设有跟踪误差上限时生效（没有上限却填区间会被拒绝）；有上限而留空取 0.85。无上限时 β 只报告、不评级。",
    },
    {
        "key": "beta_max",
        "group": "预算与验收",
        "label": "市场 β 上限",
        "type": "float",
        "optional": True,
        "advanced": True,
        "mandate_paired": True,
        "help": "跟踪授权的 β 区间上限，只在设有跟踪误差上限时生效（没有上限却填区间会被拒绝）；有上限而留空取 1.15。",
    },
    {
        "key": "cost_stress_multiplier",
        "group": "预算与验收",
        "label": "成本压力倍数",
        "type": "float",
        "help": "成本压力读数的滑点倍数：研究期每次验证与前推段报告滑点放大到该倍数后的主动超额（按各段自己的换手定价），只报告，不作判定。",
    },
    {
        "key": "research_years",
        "group": "预算与验收",
        "label": "研究年数",
        "type": "int",
        "min": 1,
        "help": "研究几何的前这么多个研究年对会话可见，用于研究与验证；其余年份是验证年，由会话的每次 test 逐步揭示并形成验证记录。十年包取 5、6 或 7。至少 1，且须留至少一个验证年；必填。",
    },
    {
        "key": "test_step_months",
        "group": "预算与验收",
        "label": "测试步长（月）",
        "type": "int",
        "min": 1,
        "help": "每次 test 把测试终点推进的月数：默认 3（一个季度），允许 1；验证年总月数须能被它整除。",
    },
    {
        "key": "validates_per_step",
        "group": "预算与验收",
        "label": "每步验证调用上限",
        "type": "int",
        "min": 1,
        "help": "首次 test 之前以及相邻两次 test 之间允许的计数 validate 调用次数（至少一个候选完成回放才计数）；用完即拒绝，直到下一次 test。默认 3。",
    },
    {
        "key": "alpha",
        "group": "预算与验收",
        "label": "残差统计量显著性水平",
        "type": "float",
        "help": "残差信息比率的单侧显著性水平，须在 (0, 1)；门槛为 z(1−α)/√年数，只陈述在每个读数旁，不作判定。全项目共用一个值，各实验的读数才能放在一起比。",
    },
    {
        "key": "forward_confidence",
        "group": "预算与验收",
        "label": "前推下界置信度",
        "type": "float",
        "help": "前推期年化主动超额（有种子复现时取逐日的种子平均）单侧块自助下界的置信度，须在 (0, 1)。下界只报告，不作判定。",
    },
    {"key": "max_null_controls", "group": "预算与验收", "label": "按需随机对照次数上限", "type": "int",
     "help": "研究会话内 run_null_control 工具的调用上限（定版节点复用其结果）；0 表示不注册该工具。"},
    {"key": "max_llm_calls", "group": "预算与验收", "label": "研究会话模型调用上限", "type": "int",
     "help": "研究会话的模型调用总次数上限，跨续跑累计；主循环、子代理与上下文压缩共享同一计数。"},
    {"key": "max_api_spend_usd", "group": "预算与验收", "label": "研究会话 API 花费上限（美元）", "type": "float",
     "help": "研究会话调用托管模型的累计花费上限，按目录标价计、跨续跑累计；主循环、子代理与上下文压缩共用，本地模型不计。用尽即结束研究会话（deadline）：有已测或可登记的候选时由宿主补完验证记录并定版，否则本实验无交付。"},
    {"key": "per_call_timeout_seconds", "group": "预算与验收", "label": "单次 LLM 调用超时（秒）", "type": "int", "advanced": True,
     "help": "Agent 主对话单次模型 API 调用的硬超时；默认 3600 秒，与本机网关非流式读超时一致。"},
    {"key": "strategy_fit_timeout_seconds", "group": "预算与验收", "label": "单次策略 fit 超时（秒）", "type": "int", "advanced": True,
     "help": "正式策略可选 fit(context) 单次调用的墙钟上限：回测开始前先训练一次模型，之后按 REFIT_PERIOD 在新周期首个决策日重训；超时或异常即整场回测失败。generate_orders 的单日推断上限是另一项 `strategy_inference_timeout_seconds`。"},
    {"key": "strategy_inference_timeout_seconds", "group": "预算与验收", "label": "单次策略推断超时（秒）", "type": "int", "advanced": True,
     "help": "正式策略 generate_orders(context) 单次调用的墙钟上限；超时即整场回测失败。批量验证并发时与 fit 上限一起按并发路数放大。"},
    {"key": "strategy_cpus", "group": "预算与验收", "label": "策略容器 CPU 配额", "type": "float", "advanced": True,
     "help": "每个策略容器（fit 与推断各一个）的 CPU 配额，线程变量随之设定；研究会话容器取同一个值。并发路数乘以它要放得下宿主的 CPU。"},
    {"key": "strategy_memory", "group": "预算与验收", "label": "策略容器内存上限", "type": "string", "advanced": True,
     "help": "每个策略容器的内存上限（Docker 写法，如 32g），不带交换；研究会话容器取同一个值。并发路数乘以它要放得下宿主的内存。"},
    {"key": "strategy_state_disk", "group": "预算与验收", "label": "策略状态目录磁盘上限", "type": "string", "advanced": True,
     "help": "一场回放的状态目录（context.state_dir）最多占用的磁盘（Docker 写法，如 64g）：每次 fit 与 generate_orders 调用之后在宿主上测量，超过即整场回测失败，记为策略自身的失败。并发路数乘以它要放得下宿主的磁盘；Paper 账户每次运行另有一份暂存副本。"},
    {"key": "record_failed_attempts", "group": "预算与验收", "label": "记录失败尝试节点", "type": "bool", "advanced": True,
     "help": "Step 树中记录未通过验证的轻量 [failed] 节点，提示会话避开死路。"},
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
        "key": "skill_library",
        "group": "运行控制",
        "label": "跨实验技能库",
        "type": "choice",
        "choices": list(SKILL_LIBRARY_MODES),
        "choice_labels": {
            "none": "不挂载",
            "curated": "挂载精选条目",
        },
        "help": (
            "把精选库只读挂载进每个研究会话工作区："
            f"精选库是仓库里人工维护的 {len(_SKILL_LIBRARY_ENTRIES)} 条运行经验，"
            "在实验创建时冻结为本实验的快照。会话不能改写或删除挂载内容。"
        ),
    },
    {"key": "gpu_count", "group": "运行控制", "label": "默认 GPU 数量", "type": "int", "min": 0, "max": 4,
     "help": "每个研究会话与正式回测 Sandbox 默认分配的 GPU 数量（0–4）；0 表示 CPU-only，不占用 L20。大于 0 时控制台在启动 Worker 前为本实验认领同样多张卡：显存余量不低于准入门槛（`AUTOTRADE_GPU_MIN_FREE_GIB`，未设时取默认门槛），且没有被别的在跑实验认领；卡上可以有其他进程在用，门槛只是准入余量，不独占也不预留显存。本实验的会话与回测只用这些卡；逐会话设置可覆盖此默认值。"},
    {"key": "batch_validate_max_concurrency", "group": "运行控制", "label": "批验证并发回测数", "type": "int",
     "min": 1, "max": BATCH_VALIDATE_MAX_CANDIDATES, "advanced": True,
     "help": f"一次 validate 同时回测的候选数（1–{BATCH_VALIDATE_MAX_CANDIDATES}）。每路各起自己的策略容器（声明 fit 的候选两个），GPU 实验的这些容器共用本实验认领的卡；单次 fit 与单日推断的超时按本批实际路数成倍放大。"},
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
        "key": "compact_model",
        "group": "模型与上下文",
        "label": "上下文压缩模型",
        "type": "choice",
        "choices": list(MODEL_CHOICES),
        "help": "语义压缩长会话所用的低成本模型（不启用推理模式）。",
    },
    {
        "key": "llm_key",
        "group": "模型与上下文",
        "label": "中转站 key",
        "type": "choice",
        "choices": list(LLM_KEYS),
        "choice_labels": {"primary": "主 key", "alternate": "备用 key"},
        "help": "本实验调用中转站模型时先用哪个 key：主 key 是 LIGHTING_API_KEY，备用 key 是 LIGHTING_API_KEY_ALT。另一个照旧作后备：先用的 key 被拒（401/402/403）或额度用尽时换它再发，此后一直用它。选备用 key 时每个模型角色都必须是中转站模型。",
    },
    {
        "key": "reasoning_effort",
        "group": "模型与上下文",
        "label": "推理强度",
        "type": "choice",
        "choices": list(reversed(REASONING_EFFORTS)),
        "choice_labels": {"xhigh": "极高", "medium": "中", "low": "低"},
        "help": "启用推理模式时 Agent 主对话与子代理的推理强度；三档即本机 Qwen 模板在线上真正区分的档位。推理只有开关、没有强度档的模型（mimo-v2.6-flash）不使用此项。",
    },
    {
        "key": "no_thinking",
        "group": "模型与上下文",
        "label": "禁用推理模式",
        "type": "bool",
        "advanced": True,
        "help": "关闭模型推理模式（Agent 主对话与子代理）。",
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
        "help": "估算上下文 token 超过该值时触发语义压缩。留空（默认）时主对话与子代理各按自己的上限压缩：该角色模型的上下文窗口 − 8,192，输出与窗口共用时再减输出上限，且不超过压缩模型按同一算式的上限（百万窗口的模型约 96 万，窗口只算输入的 Gemini 约 98 万）。填入的值是更低的阈值，超过上限即拒绝创建。",
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

# Console parameters a create request (a round file) may name but the form does
# not offer: the reference workspace, the retired approval mode, the attempt count,
# the boards (stamped from the initial cash) and a snapshot-builder knob.
_FORMLESS = frozenset(
    {
        "workspace_reference",
        "initial_control_mode",
        "session_max_attempts",
        "permitted_boards",
        "max_intraday_row_group_rows",
    }
)


def _require_form_covers_the_console_parameters() -> None:
    formed = [str(field["key"]) for field in _FIELDS]
    console = set(parameters(Surface.CONSOLE))
    stray = sorted(set(formed) - console)
    missing = sorted(console - set(formed) - _FORMLESS)
    if stray or missing or len(formed) != len(set(formed)) or _FORMLESS & set(formed):
        raise RuntimeError(
            f"creation form drifted from the experiment parameters: fields {stray} are no "
            f"console parameter and {missing} have no field"
        )


_require_form_covers_the_console_parameters()

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
        default = WEB_CREATE_DEFAULTS[str(entry["key"])]
        if isinstance(default, tuple):
            default = list(default)
        entry["default"] = default
        entry["required"] = PARAMETERS[str(entry["key"])].required
        groups[str(entry.pop("group"))].append(entry)
    return {
        "schema_version": 6,
        # What each ``mandate_paired`` field defaults to on either side of the
        # tracking mandate's switch. Only the beta band travels with the cap;
        # drawdowns keep a static placeholder and are not in this map.
        "acceptance_defaults": {
            "mandated": dict(MANDATED_DEFAULTS),
            "unmandated": {key: acceptance_default(key) for key in MANDATED_DEFAULTS},
        },
        "groups": [{"name": name, "fields": entries} for name, entries in groups.items() if entries],
    }
