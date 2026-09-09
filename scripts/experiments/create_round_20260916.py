#!/usr/bin/env python
"""The 2026-09-16 round: three new information sources on one extended PIT seed.

All three arms move the information source off the daily price/volume table
every earlier arm kept transforming. `analyst_revision` moves it onto the sell
side: the `report_rc` numeric slice, per-broker and per-fiscal-year EPS
forecasts, which no arm had ever read. `margin_flow` moves it onto the
margin-trading plumbing: the `margin_secs` eligibility roster and the
`margin_detail` financing balance -- a regulatory admission list and a leverage
stock. `alt_events_ranker` keeps the estimator familiar and changes the feature
set instead: a walk-forward LightGBM cross-sectional ranker over the disclosure
and flow tables, judged against a same-learner attribution control that sees
only the 14 daily price/volume columns, so the claim under test is that those
tables carry something the daily bar cannot learn. It requests no GPU -- its
pack forbids torch and fits on the CPU.

What this round decides for itself is the data. All three arms select the same
extended macro and events datasets and therefore share one prebuilt view tree.
A PIT view seed's identity IS the whole snapshot configuration -- the dataset
selection, the domain switches, the window months and the universe screen -- so
every arm must ask for a byte-identical selection or the seed stops matching
and the arm cold-builds every view for hours. That is why the selection is
stated once below and why `include_intraday`, a console default today, is
stated rather than inherited: it is part of what the seed was built for. The
calendar the seed was planned over is the shared one in `_round.BASE_OVERRIDES`
and no arm moves it -- every pack verified its source tables cover the whole
2022Q1--2025Q4 input window, so unlike `site_visits` none of them starts later.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \
      scripts/experiments/create_round_20260916.py <port> [--dry-run] [experiment_id ...]
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

from scripts.experiments._round import PARENT_CONTROL_LINE, ROBUSTNESS_LINE, Round

# The one prebuilt view tree every arm of this round hardlinks from. Built by
# scripts/data/prebuild_pit_views_seed.py with exactly the selection below and
# the shared calendar; gitignored, so it is an operator precondition, not an
# artifact of this repository.
PIT_VIEWS_SEED = "data/pit_views_seed_ext_20260916"

# The extended selection this round's seed was built for. The macro side is
# round 20260914's (futures, options and the three convertible-bond tables
# beyond the console default scope); the events side is that round's plus
# `margin_secs` for margin_flow and `report_rc` for analyst_revision, both of
# which alt_events_ranker also reads. include_macro/include_events must stay
# True for a selection to apply at all -- worker._snapshot_config drops the
# whole domain when its switch is off -- and both are pinned in the shared
# expected defaults.
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
]

ARMS: dict[str, dict[str, object]] = {
    "analyst_revision_20260916": {
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
    "margin_flow_20260916": {
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
    "alt_events_ranker_20260916": {
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
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    overrides={
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        # Part of the seed's snapshot configuration rather than a preference:
        # the seed carries no intraday views, so an arm that asked for them
        # would not match it.
        "include_intraday": False,
    },
    # What this round adds to the dry-run report: the dataset selection and the
    # two domain switches that decide whether it applies at all.
    report_keys=("include_macro", "macro_datasets", "include_events", "events_datasets"),
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
