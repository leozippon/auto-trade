#!/usr/bin/env python
"""The 2026-09-17 round: five arms on one extended seed, three of them continued.

Three arms re-create the 2026-09-16 directions -- `analyst_revision`,
`margin_flow`, `alt_events_ranker` -- on the extended seed and calendar below,
each starting from its predecessor's PRIOR and skills through
`inherit_memory_from` rather than from an empty memory. The sources are still
on the console when this round is created and are retired afterwards, so their
ids are deliberately not in `_round.RETIRED_IDS` yet; the launcher refuses an
arm whose inheritance source is not there.

`github_confirm` is the round's other half of a controlled comparison. Its
source, `explore_github_strategies_20260910`, is the one lineage in this
repository with forward positive evidence, and it keeps searching the feature
space; this arm inherits that experiment's frozen artifact and its memory,
freezes the mechanism, and spends every Fold on turnover and execution cost
instead. That gives the shipped artifact its own forward transitions -- the
graduation term a search arm's last-Fold artifact keeps failing -- and the two
arms together are "keep searching" against "freeze the mechanism".

Because the point is to continue rather than restart, `github_confirm` cannot
state its Development start as a literal: it depends on where the source has
actually got to. The arm therefore computes it when the round is created, from
the source's own ledger, so the first Fold is the quarter after the source's
latest freeze and the two arms never validate the same new quarter twice; the
round refuses to create the arm when too few Folds would remain.

`order_flow_ranker` starts from nothing. It is the first arm to read
`intraday_flow`, the derived daily dataset built from minute bars, and its one
hard gate is the attribution control against the vendor's own daily money-flow
face: same basket, same label, same scorer, features swapped. Losing to it
closes the mechanism, because that is what "the minute bars only recomputed
what the vendor already published" looks like.

The round's own decisions are the data and the calendar. All five arms select
the same extended macro and events datasets and share one prebuilt view tree; a
seed's identity IS the whole snapshot configuration, so every arm must ask for
a byte-identical selection or the seed stops matching and the arm cold-builds
every view. Development runs one quarter further than the previous round and
Held-out moves to the range below -- the pipeline clips a Held-out end past the
release's last trading day and records the truncation, so a partial final
quarter is labelled partial rather than silently scored whole. This is also the
first round to set `deployment_adjustment_start`, so an arm that graduates gets
the post-verdict refit session before Paper.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260917.py <port> [--dry-run] [experiment_id ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Appended, not prepended: the repository root carries directories named `data`,
# `logs` and `results`, which must never shadow an installed package. The
# launcher pins this repo's `src/` itself.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.experiments._round import (
    PARENT_CONTROL_LINE,
    ROBUSTNESS_LINE,
    Round,
    last_completed_fold,
    quarter_shift,
    quarters_between,
)

# The one prebuilt view tree every arm of this round hardlinks from. Built by
# scripts/data/prebuild_pit_views_seed.py with exactly the selection below and
# the calendar this file states; gitignored, so it is an operator precondition,
# not an artifact of this repository.
PIT_VIEWS_SEED = "data/pit_views_seed_ext_20260917"

# One quarter further than the previous round, and a Held-out range that runs
# past the release end on purpose: `folds.heldout_periods` clips the replay to
# the last fixed trading day and records `requested_end`/`truncation_reason`,
# so the window grows with the lake instead of being retyped every month.
DEVELOPMENT_LAST_PERIOD = "2026Q1"
HELDOUT_PERIOD = "20260601..20260930"

# The post-verdict mechanism-frozen refit, run only for an experiment that
# graduates. The window starts here and ends at the release's last trading day,
# and it contains the whole Held-out, which is why nothing that could change
# what the mechanism IS may be changed in it.
DEPLOYMENT_ADJUSTMENT_START = "20250901"

# github_confirm continues this experiment: it inherits the frozen artifact and
# the memory, and its Development start is read from that experiment's ledger.
GITHUB_SOURCE = "explore_github_strategies_20260910"
# A Fold validates the trailing four quarters ending at its own quarter, so the
# first Fold of a Development window starting at Q is Q + 3.
VALIDATION_PERIODS = 4
# Fewer than this many Folds left is not a confirmation arm: the shipped
# artifact would carry too few forward transitions of its own to mean anything.
MIN_REMAINING_FOLDS = 4

MACRO_DATASETS = [
    "cn_gdp",
    "cn_cpi",
    "cn_ppi",
    "cn_pmi",
    "cn_m",
    "sf_month",
    "shibor",
    "shibor_lpr",
    "index_daily",
    "index_dailybasic",
    "sw_daily",
    "fut_basic",
    "fut_mapping",
    "fut_daily",
    "opt_basic",
    "opt_daily",
    "cb_basic",
    "cb_daily",
    "cb_call",
]
EVENTS_DATASETS = [
    "margin",
    "margin_detail",
    "margin_secs",
    "moneyflow",
    "cyq_perf",
    "bak_daily",
    "block_trade",
    "stk_holdernumber",
    "stk_holdertrade",
    "new_share",
    "share_float_complete",
    "top_list",
    "top_inst",
    "limit_list_d",
    "kpl_list",
    "stk_surv",
    "top10_floatholders",
    "report_rc",
    # The round's own addition: the derived daily face of the minute bars,
    # which order_flow_ranker exists to read. The minute domain itself stays
    # off -- the seed carries no intraday views.
    "intraday_flow",
]


def github_confirm_development_start(experiments_root: Path | None = None) -> str:
    """Where `github_confirm` starts Development, read from its source.

    The arm exists to record a forward transition for the inherited artifact on
    every quarter the source has not already frozen, so its first Fold must be
    the quarter after the source's latest freeze. A Development window starting
    at Q first validates at Q + 3, so the start is the source's last completed
    Fold minus two quarters. Refuses rather than creating a confirmation arm
    with too little room left.
    """
    last_fold = last_completed_fold(GITHUB_SOURCE, experiments_root)
    start = quarter_shift(last_fold, -2)
    first_fold = quarter_shift(start, VALIDATION_PERIODS - 1)
    remaining = quarters_between(first_fold, DEVELOPMENT_LAST_PERIOD)
    if remaining < MIN_REMAINING_FOLDS:
        raise ValueError(
            f"{GITHUB_SOURCE} last completed {last_fold}, which leaves {remaining} Fold(s) "
            f"in {first_fold}..{DEVELOPMENT_LAST_PERIOD}; a confirmation arm needs at least "
            f"{MIN_REMAINING_FOLDS}, so the source has to be caught up or the round re-decided"
        )
    return start


ARMS: dict[str, dict[str, object]] = {
    "analyst_revision_20260917": {
        # Re-created on the extended seed; it starts from the earlier arm's
        # PRIOR and skills rather than from an empty memory.
        "inherit_memory_from": "analyst_revision_20260916",
        "workspace_reference": "configs/workspace_refs/analyst_revision_20260916",
        "fold_exploration_directive": "\n".join(
            [
                "方向：在全 A 上把 report_rc 数值列（逐券商逐财年 EPS 预测）做成 08:30 决策的截面信号，"
                "只测 refs 登记的四个修正家族，一次推进一个，产物写在 output/ 包内。"
                "先读 README/exploration-plan/families，按 pit-field-map 核对列名、财年语义与盖章规则；"
                "数值切片不在或列名对不上就立即 no_edge 弃权，不得用标题文本做退化版。"
                "quarter 是财年不是季度，判可见只看 available_at。"
                "第一轮必须同批带家族 1 与家族 2：两者反号即关闭家族 1。"
                "每个候选同批带 EP 腿或覆盖度腿；被吸收即判价值/注意力换皮关闭。"
                "覆盖度与评级属已关卖方族邻域，只能作对照。"
                "淡季可选股票数会掉到 180–280 只，不得靠降门槛凑数，如实弃权。"
                "通用价量打分不得作候选。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
            ]
        ),
    },
    "margin_flow_20260917": {
        # Re-created on the extended seed; it starts from the earlier arm's
        # PRIOR and skills rather than from an empty memory.
        "inherit_memory_from": "margin_flow_20260916",
        "workspace_reference": "configs/workspace_refs/margin_flow_20260916",
        "fold_exploration_directive": "\n".join(
            [
                # The vendor-truncation window this line excludes is named by
                # its cause, not by its dates: resolve_worker_options refuses a
                # fold_exploration_directive carrying a literal calendar date
                # (prior_policy.calendar_policy_violation), so writing the two
                # months out would fail the arm at worker start. The exact range
                # is declared once in the pack the previous sentence orders the
                # Agent to read.
                "方向：以两融管道为信息源只做多——margin_secs 名册的老股纳入事件与 margin_detail 的融资余额拥挤度，"
                "一次只推进一个可分离家族，正式产物写在 output/ 包内。"
                "先读 refs 的 README、exploration-plan 与 families.md，按 pit-field-map.md 核对："
                "名册 T-1 可见、融资明细只到 T-2，写成 T-1 即前视。"
                "任何回测前先做不占预算的普查：事件必须过 universe、60 日往返、上市满 120 天、"
                "margin_detail 三日确认四道过滤并剔除 refs 声明的供应商截断区间，逐季报事件数与空仓比例，"
                "不过门就如实记「不可测」。"
                "每个候选同批带同日匹配对照与名册内随机入场对照，并做规模/动量/换手/波动四项中性化。"
                "拥挤度先算成本：5 日持有年拖累 6.5–15%，若只有 5 日为正即判不可交易并弃权。"
                "通用价量打分不得作候选，只能以 families.md 命名的对照身份出现。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
            ]
        ),
    },
    "alt_events_ranker_20260917": {
        # Re-created on the extended seed; it starts from the earlier arm's
        # PRIOR and skills rather than from an empty memory.
        "inherit_memory_from": "alt_events_ranker_20260916",
        # No GPU: the pack forbids torch and pins every fit to the CPU, so the
        # LightGBM ranker trains inside the strategy container as it stands.
        # gpu_count=0 is round 20260910's value and is inherited, not restated.
        "workspace_reference": "configs/workspace_refs/alt_events_ranker_20260916",
        "fold_exploration_directive": "\n".join(
            [
                "方向：把事件与披露面（report_rc 一致预期、margin_detail 两融、block_trade 折溢价与席位、股东人数/增减持/十大流通股东、cyq_perf 筹码、share_float_complete 解禁）做成一个特征集，训 walk-forward LightGBM 截面排序器，只测 refs 登记的五个特征族与其消融，一次只改一个可分离组件。"
                "每个候选必须同批带两个对照：同学习器同超参同标签、只用日频价量 14 列的归因对照，以及同特征不学习的等权 rank 合成。"
                "归因对照没有在中性化超额与空对照分位上同时被超过，即判事件特征无边际，按 refs 的终止规则关闭本机制，而不是换变体重试。"
                "标签 20 日与再平衡 20 日必须一致；季度重训，训练与验证之间留 20 个交易日禁运，拟合只在 fit(context) 里做。"
                "先按 pit-field-map.md 逐表核对可见时点、单位与三条去重规则：margin_detail、stk_holdernumber、top10_floatholders、share_float_complete 在 08:30 只到 T-2。"
                "手写通用价量打分不得作候选，只能作对照。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
            ]
        ),
    },
    "github_confirm_20260917": {
        # Both halves of continuing: the frozen output/ and models/ of the
        # source's latest Fold copied in read-only as this arm's first parent,
        # and that experiment's PRIOR and skills as its first memory. The
        # mechanism is what is being confirmed, so it arrives rather than being
        # rebuilt.
        "inherit_from": GITHUB_SOURCE,
        "inherit_memory_from": GITHUB_SOURCE,
        # Resolved when the round is created, not now: see the function above.
        "development_first_period": github_confirm_development_start,
        "workspace_reference": "configs/workspace_refs/github_confirm_20260917",
        "fold_exploration_directive": "\n".join(
            [
                "方向：本臂不找新 alpha。继承来的机制原样冻住——特征、标准化、标签、估计器、网格、切分、禁运、TOPK "
                "与再平衡节奏一列都不许动——只测 refs 登记的两个换手/执行变体（V1 名次滞后带、V2 收盘执行），"
                "一次一个，正式产物写在 output/ 包内。"
                "每折先做第 0 步再谈变体：按 pit-field-map.md 逐表核对列合同，缺任何一列就显式失败并 report_issue；"
                "对未改动的父产物跑一次 smoke_backtest，确认订单 reason 是 a158_lgbm_rebal 而不是退化路径；"
                "再读宿主已跑好的 parent_control——全窗中性化超额、2 倍滑点后超额、换手与成交笔数，"
                "以及 sub_windows 最后一行，即本折新季度这份产物真正的样本外前向记录。"
                "这一行是本臂每折的头条结论，也是账本给交付产物记下的那次自有过渡。"
                "父本对照节点不得被提名；保留父本走 finish_fold(outcome=\"no_edge\")。"
                "绝大多数折的正确结局就是 no_edge，这是排程设计而不是弃权文化。"
                "变体在 hypothesis 里先写后跑：改哪一个常量、改成什么、为什么不在关闭清单的轴上、预期体现在哪个指标、"
                "以及本包列出的全部否证条件。"
                "不得改任何 .py 的结构、不得新增特征或数据源、不得把 refs 拷进 output、不得写死路径与股票代码；"
                "每一行输入都必须满足 available_at <= 推断时点。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
            ]
        ),
    },
    "order_flow_ranker_20260917": {
        "workspace_reference": "configs/workspace_refs/order_flow_ranker_20260917",
        "fold_exploration_directive": "\n".join(
            [
                "方向：把 1 分钟 Bar 上的 tick-rule 签名成交量失衡收成一个特征面——events 域的派生表 intraday_flow，"
                "逐分钟涨跌方向给该分钟成交量定号、按日汇总，再取 21 个交易日均值——交给只做多的截面排序器，"
                "只测 refs 登记的 ofi 家族，正式产物写在 output/ 包内。"
                "硬门只有一条：每个候选必须同批带 moneyflow 面的归因对照，同一篮子、同一标签、同一打分器，"
                "特征换成供应商日频资金流分类；比不过它就立即关闭本机制，那说明分钟 Bar 只是把供应商已经算过的东西重算了一遍。"
                "先读 README、exploration-plan.md 与 families.md，按 pit-field-map.md 核对可见时点、单位与去重规则："
                "本臂不挂分钟域，A 面只从 events 的 intraday_flow 读；events.datasets 缺 intraday_flow 或 moneyflow，"
                "对应特征面整块不可构造，立刻 report_issue 并以 no_edge 如实弃权，"
                "不得用日频量价拼一个近似 OFI 顶替。"
                "任何回测前先做不占预算的普查：intraday_flow 在快照视图与验证视图各自的行数与日期覆盖必须分别读、"
                "不得由前者推断后者；再报它在本折可见区间里的最后一个 trade_date 及其与 daily 最后一个交易日的差——"
                "这张表跟随分钟按日层，终点落后于日线，本折区间尾部没有行时 ofi_21 会整段失效，必须先说出来再谈收益；"
                "再报形状不变量与取值域、两条过滤的剔除率（超过四成即关闭本机制）、单位核对，"
                "以及两面在月末决策日的可选池与非空率必须可比——差到几个百分点就是在比覆盖而不是比信号，先统一池。"
                "逐列规模中性 rank IC 的符号与 families.md 声明相反的列，从等权合成里去掉并写进 hypothesis，不得事后翻符号。"
                "只做多的那条腿很薄，顶档超额与往返成本同量级：成本后不为正就如实判不可交易并按 no_edge 弃权，"
                "不得靠调滑点假设或放宽门槛把它救活。"
                "families.md 里已关闭的九个兄弟构造不得重开，通用价量打分不得作候选，只能以本包命名的对照身份出现。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
            ]
        ),
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    overrides={
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        # Part of the seed's snapshot configuration rather than a preference:
        # the seed carries no intraday views, so an arm that asked for them
        # would not match it. `intraday_flow` is an events dataset, not the
        # minute domain.
        "include_intraday": False,
        "development_last_period": DEVELOPMENT_LAST_PERIOD,
        "heldout_first_period": HELDOUT_PERIOD,
        "heldout_last_period": HELDOUT_PERIOD,
        "deployment_adjustment_start": DEPLOYMENT_ADJUSTMENT_START,
    },
    # The deployment adjustment's own budget, left at the console default and
    # pinned so a drift is caught: the round's arms are the first to reach that
    # session, and the prompt promises the Agent this many replays.
    expected_defaults={"deployment_max_backtests": 6},
    # What this round adds to the dry-run report: the dataset selection and the
    # two domain switches that decide whether it applies at all.
    report_keys=("include_macro", "macro_datasets", "include_events", "events_datasets"),
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
