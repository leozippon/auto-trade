#!/usr/bin/env python
"""Create the 2026-09-14 round: three new research directions on one extended PIT seed.

This round does not replace the console's whole slate. Three arms of round
20260910 (`corner_cases`, `explore_platform_strategies`, `factor_cs`) are
retired; the other three (`open_mechanism_20260910`,
`explore_github_strategies_20260910`, `ml_ranker_20260910`) keep running
untouched, so `scripts/experiments/create_round_20260910.py` stays in the tree
as their definition and this script only has to fill the three freed slots.
Together the two rounds fill `webui.manager.MAX_RUNNING_EXPERIMENTS` exactly.

The three new arms all move the information source off the single daily
price/volume table every previous arm kept transforming: convertible bonds as a
second security of the same issuer (`cb_linkage`), institutional site visits as
buy-side information production (`site_visits`), and valuation/payout levels
with the index-futures basis as a style state (`value_regime`). Each points at
its own reference pack under `configs/workspace_refs/`, and every model role of
every arm stays on the local qwen-3.8-27b-fp8 -- which is the console creation
default for all six roles, so this round overrides none of them and pins them in
EXPECTED_DEFAULTS instead.

What is new at the parameter level is the data. All three arms select the same
extended macro and events datasets (futures, options and the three convertible
bond tables on the macro side; `stk_surv` and `top10_floatholders` on the events
side) and therefore share one prebuilt view tree,
`data/pit_views_seed_ext_20260914`. A PIT view seed's identity IS the whole
snapshot configuration: the dataset selection, the domain switches, the window
months and the universe screen. So the three arms must ask for a byte-identical
selection or the seed stops matching and the arm cold-builds every view for
hours. That is why the selection lives in COMMON_OVERRIDES once, and why
`include_intraday` -- a console default today -- is stated there rather than
inherited: it is part of what the seed was built for.
`worker.resolve_worker_options` checks the seed's recorded snapshot
configuration against the request at create time, so a drift in any of these
fails the create instead of silently cold-building. What that check cannot see
is the calendar the seed was planned over: `fold_period`, the two Development
bounds, the Held-out range, `validation_periods` and `min_region_trade_days`
decide which Folds exist but never enter the snapshot configuration, so a drift
there would not fail anything -- it would just leave views missing. Every one of
them is therefore either stated in COMMON_OVERRIDES or pinned in
EXPECTED_DEFAULTS.

`site_visits` is the single calendar exception. Its `stk_surv` source starts two
years into the lake, and the pack's 250-trading-day baseline plus the 24-month
input window put the first fully covered validation window at 2024Q1, so that
arm starts its Development there. Its Folds are a subset of the seed's plan, so
the shared seed still applies to every one of them.

The rest of the design is round 20260910's and is not re-decided here. The CNY
100k account, the graduation gates and the per-Fold budgets are read from that
round's own COMMON_OVERRIDES rather than retyped, so two rounds sharing one
console cannot drift apart on cost; the quarterly walk-forward geometry agrees
with it but is restated, because this round's Folds have to stay the ones its
seed was built for. The offline validation, the report and the POST path are
that round's functions, imported unchanged.

Every parameter set is validated offline with every request-level check the
console applies on POST /api/experiments, and additionally refuses a directive
the PRIOR calendar policy would reject. Two things --dry-run cannot answer stay
at POST time: an experiment directory that already exists and a free running
slot. The seed is a third, and only half answerable: because it is named
explicitly it is required, so the pre-flight does check that the tree is there
and that its recorded snapshot configuration is this one -- and a run started
before the prebuild created that tree is reported as a named parameter
rejection rather than a traceback. What no check can see is whether the prebuild
has finished: `provider.json` is written when it starts, so a matching contract
means the arm will hardlink whatever views exist by then and build the rest
itself.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \
      scripts/experiments/create_round_20260914.py <port> [--dry-run] [experiment_id ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

# Round 20260910's three surviving arms run beside these three, so that module
# is still the live definition of half the console and stays in the tree.
# Importing it costs nothing (constants and pure functions only) and keeps one
# source for the create contract: the same offline validation, the same report
# keys, the same POST path, the same budgets and the same model-role pins.
# Appended, not prepended: the repository root carries directories named `data`,
# `logs` and `results`, which must never shadow an installed package.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from scripts.experiments.create_round_20260910 import (
    COMMON_OVERRIDES as ROUND_20260910_OVERRIDES,
)
from scripts.experiments.create_round_20260910 import (
    EXPECTED_DEFAULTS as ROUND_20260910_DEFAULTS,
)
from scripts.experiments.create_round_20260910 import (
    PARENT_CONTROL_LINE,
    ROBUSTNESS_LINE,
    normalize,
    post,
)
from scripts.experiments.create_round_20260910 import (
    REPORT_KEYS as ROUND_20260910_REPORT_KEYS,
)

# The one prebuilt view tree all three arms hardlink from. Built by
# scripts/data/prebuild_pit_views_seed.py with exactly the selection and
# calendar below; gitignored, so it is an operator precondition, not an artifact
# of this repository.
PIT_VIEWS_SEED = "data/pit_views_seed_ext_20260914"

# The extended selection this round's seed was built for. Not three lists that
# happen to agree: the seed's identity is the whole snapshot configuration, so
# every arm has to send this exact selection (same names, same order is
# irrelevant to the check but kept for readability) or it finds no matching
# contract and cold-builds every view. Beyond the console default scope this
# adds futures and options (fut_*/opt_*, for the index-basis state) and the
# three convertible-bond tables on the macro side, and stk_surv plus
# top10_floatholders on the events side. include_macro/include_events must stay
# True for a selection to apply at all -- worker._snapshot_config drops the
# whole domain when its switch is off -- which is why both stay pinned in
# EXPECTED_DEFAULTS.
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
]

# What this round decides for every arm.
COMMON_OVERRIDES: dict[str, object] = {
    # No GPU, one Epoch, the CNY 100k account, the graduation gates and the
    # per-Fold budgets (600 min / 24 Steps / 24 backtests / 1600 calls): round
    # 20260910's decisions, read from that module instead of retyped. Its three
    # surviving arms run beside these three under the same cost regime, and a
    # walk-forward comparison across the two halves of the console only means
    # anything while that stays true.
    **ROUND_20260910_OVERRIDES,
    # The shared seed and the selection it was built for.
    "pit_views_seed": PIT_VIEWS_SEED,
    "macro_datasets": MACRO_DATASETS,
    "events_datasets": EVENTS_DATASETS,
    # Part of the seed's snapshot configuration rather than a preference: the
    # seed carries no intraday views, so an arm that asked for them would not
    # match it.
    "include_intraday": False,
    # The calendar the seed was planned over. These six agree with round
    # 20260910 today and are restated anyway, not inherited: none of them
    # reaches the seed contract check, so a later edit to that round's schedule
    # must not be able to move this round's Folds away from the tree that was
    # built for them. site_visits overrides the Development start; see below.
    "fold_period": "quarter",
    "validation_periods": 4,
    "development_first_period": "2022Q1",
    "development_last_period": "2025Q4",
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
}

# Console defaults this round still relies on: round 20260910's pins minus every
# key this round now decides for itself. Values, not commentary -- a drift stops
# the script instead of silently re-scoping three experiments. The six model
# roles are the load-bearing ones: no arm overrides a role, so the console
# default is what actually decides them, and a rename of the local model is
# exactly the drift this has to catch.
EXPECTED_DEFAULTS: dict[str, object] = {
    key: value
    for key, value in ROUND_20260910_DEFAULTS.items()
    if key not in COMMON_OVERRIDES
} | {
    # The last of pit_views_seed.PLAN_PARAMETERS this round does not state
    # itself. Every other one is either in COMMON_OVERRIDES above or already
    # pinned by round 20260910 (test_stage, window_months); this one is only a
    # console default, and it decides which Folds the schedule keeps, so a drift
    # would silently move the plan away from the tree the seed was built over.
    "min_region_trade_days": 2,
}

ROUND: dict[str, dict[str, object]] = {
    "cb_linkage_20260914": {
        "workspace_reference": "configs/workspace_refs/cb_linkage_20260914",
        "fold_exploration_directive": "\n".join(
            [
                "方向：以可转债市场为信息源，在「T-1 有存续可转债的正股」这个池子上只做正股多头，测 refs 登记的家族"
                "——转债残差动量、转股溢价率状态、强赎条款前后的正股行为、转债相对正股的成交额异动、纯债溢价率的债底缓冲"
                "——一次只推进一个可分离的家族，正式产物写在 output/ 包内。",
                "先读 refs/README.md、exploration-plan.md 与 families.md，再按 pit-field-map.md 核对每个字段的域、单位与"
                "可见时间；任何回测之前先在输入窗上做不占回测预算的普查，建发行人—转债面板，分别报池子效应、信号效应与"
                "事件家族的逐季事件数，普查不合格的家族直接淘汰并写明理由。",
                "每个候选都同批带两个对照：同池同过滤的等权篮子，以及按同日、同规模十分位、同申万一级行业、成交额最接近"
                "抽取且每次调仓重抽的无转债匹配篮子。等权池相对匹配篮子的超额只是池子效应，家族相对等权池的超额才是信号，"
                "两者分开汇报；裁决以本折新季度为准，候选在新季度相对父本对照与等权池对照都为正才算通过，对照节点不得提名。",
                "通用价量截面打分（价值、反转、动量、低波、规模）不得作为本轮候选；家族落败后的剩余预算只能用于本包登记的"
                "其他家族或其门控与持有期变体，不得改成通用因子回退，它们只能以 families.md 命名的形式作为某个家族的"
                "机制无关对照出现。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止克隆父策略、禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足"
                " available_at <= 推断时点，转债与公告用 available_at，存续与历史转股价只能从 cb_daily 重算，"
                "cb_basic 的当前状态字段不得当历史用。可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "site_visits_20260914": {
        # The only calendar override in this round. stk_surv starts two years
        # into the lake; with the pack's 250-trading-day baseline and the
        # 24-month input window, the first Fold whose input window is fully
        # covered validates from 2024Q1. Earlier Folds could only be run on a
        # truncated baseline, which the pack forbids, so the arm does not start
        # them at all. Its Folds are a subset of the shared seed's plan.
        "development_first_period": "2024Q1",
        "workspace_reference": "configs/workspace_refs/site_visits_20260914",
        "fold_exploration_directive": "\n".join(
            [
                "方向：在未筛选的全 A 股票池上，把机构实地调研（stk_surv）这条买方主动信息生产的痕迹做成能在固定推理时点"
                "决策的截面信号，只测 refs 登记的调研家族——异常调研强度、长期沉寂后的首次调研、到访机构类型构成、"
                "调研后未消化、调研×机构持股变化——一次只推进一个家族，正式产物写在 output/ 包内。",
                "先读 refs/README.md、exploration-plan.md 与 families.md，按 pit-field-map.md 核对字段、单位与可见时间；"
                "调研的可见性只看 available_at，不看 surv_date，两者相差至少五个自然日，因此五日以内的事件窗在本环境"
                "根本不存在。",
                "任何回测之前先做普查（不占回测预算，只用输入窗）：逐月行数与零行交易日清单、逐季事件数与可选股票数、"
                "交易所构成、每次事件的到访机构家数分布。落库缺口造成的整月零行分区（清单见 refs 的失效模式一节）必须"
                "显式剔除并按有效日归一化，不得当成调研冷却读；普查不过门槛就用 finish_fold 以 no_edge 如实弃权，"
                "不得降级基准窗，也不得用价量因子把这一折填满。",
                "每个候选都同批带匹配对照——同日、同申万一级行业、同规模与换手分位、近期零可见调研的无调研篮子——并且"
                "深市腿与沪市腿分别读数：两腿符号不一致直接淘汰，那是披露制度而不是信息。裁决以本折新季度为准，"
                "每个子窗的读数都要连同事件数与可选股票数一起报。",
                "通用价量截面打分不得作为本轮候选，只能以 families.md 机制无关对照家族的形式出现；调研家族落败后的"
                "剩余预算只能用于本包登记的其他调研家族或其条件与持有期变体，不得改成价量打分。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止克隆父策略、禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足"
                " available_at <= 推断时点。可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "value_regime_20260914": {
        "workspace_reference": "configs/workspace_refs/value_regime_20260914",
        "fold_exploration_directive": "\n".join(
            [
                "方向：在剔除最小 30% 市值后的全 A 上，检验 refs 登记的估值与派现慢腿——EP、股息率（含派现一致性过滤）、"
                "BP 叠加质量，以及向等权收缩的复合分数——默认 30 只等权、持有 60 个交易日，一次只推进一条可分离的腿"
                "或一次合成，正式产物写在 output/ 包内。",
                "先读 refs/README.md、exploration-plan.md 与 families.md，按 pit-field-map.md 核对字段、单位与可见时间；"
                "任何回测之前先在输入窗上做不占回测预算的普查与离线预筛，报各腿的规模中性 rank IC、逐月符号一致性、"
                "覆盖与换手，不达标的腿直接淘汰并写明理由。",
                "第二层是条件层而不是选股信号：用 IC/IM 当季合约的年化基差（先按输入窗做月度去均值再取分位）读出杠杆与"
                "对冲需求状态，深贴水时把同一本账倾向大盘价值腿，其余时间保持全宇宙，始终满仓多头，不得做成清仓闸门。"
                "状态类候选必须同批带伪门控对照：把状态序列整体循环平移一个预先声明的固定偏移后驱动同样的切换，"
                "真门与伪门无法区分即淘汰；验证窗触发日数或独立触发段数不达门槛时判「不可测」，不作为候选结果汇报。",
                "整窗汇总不作论据，四个季度子窗的符号一致性才是主要读数，裁决以本折新季度为准。复合权重只在 fit(context)"
                " 里拟合并强制向等权收缩，训练窗与验证窗之间留出持有期长度的禁运，不在 generate_orders 里重新拟合任何权重。",
                "通用快因子（反转、动量、低波、换手、成长）不得作为本轮候选，只能以 families.md 命名的形式作为某条腿的"
                "机制无关对照出现；四条腿全部落败时切到条件层，条件层也判不可测就如实弃权，不得用截尾分位、持有期或"
                "篮子大小的邻域刷候选数。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止克隆父策略、禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足"
                " available_at <= 推断时点，财务与派现用 available_at。可执行指纹必须不同于父策略。",
            ]
        ),
    },
}

# Round 20260910's report keys plus what this round adds: the dataset selection
# and the two domain switches that decide whether it applies at all.
REPORT_KEYS = (
    *ROUND_20260910_REPORT_KEYS,
    "include_macro",
    "macro_datasets",
    "include_events",
    "events_datasets",
)


def check_console_defaults() -> None:
    drift = {
        key: (value, WEB_CREATE_DEFAULTS[key])
        for key, value in EXPECTED_DEFAULTS.items()
        if WEB_CREATE_DEFAULTS[key] != value
    }
    if drift:
        raise SystemExit(
            "console creation defaults drifted from what this round assumes; "
            "re-decide the round before creating: "
            + json.dumps(
                {k: {"round": str(v[0]), "console": str(v[1])} for k, v in drift.items()},
                ensure_ascii=False,
            )
        )


def request_params(experiment_id: str) -> dict[str, object]:
    """The create request body: console defaults, the round's overrides, the id."""
    base = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in WEB_CREATE_DEFAULTS.items()
    }
    return {
        **base,
        **COMMON_OVERRIDES,
        **ROUND[experiment_id],
        "experiment_id": experiment_id,
    }


def validated(experiment_id: str) -> dict[str, object]:
    """params.json for one arm, or an operator-readable refusal.

    ``normalize`` runs the console's own create-time checks and the worker
    pre-flight; the pre-flight requires the shared PIT view seed, which is a
    separate and slow operator step, so a round dry-run started before the
    prebuild finished has to say that in one line instead of a traceback.
    """
    try:
        return normalize(request_params(experiment_id))
    except ValueError as exc:
        message = f"{experiment_id}: parameters rejected, nothing was sent: {exc}"
        if "pit_views_seed" in str(exc):
            message += (
                f"\n  the three arms share one prebuilt view tree ({PIT_VIEWS_SEED});"
                " build it with scripts/data/prebuild_pit_views_seed.py for exactly"
                " this dataset selection and calendar, and wait for it to finish,"
                " before creating or dry-running the round"
            )
        raise SystemExit(message) from exc


def main() -> int:
    # A mistyped flag must never fall through to the real POST path: without
    # this, --dryrun is read as an experiment-id filter and creates the round.
    mistyped = [
        arg for arg in sys.argv[1:] if arg.startswith("--") and arg != "--dry-run"
    ]
    if mistyped:
        print("unknown option: " + ", ".join(mistyped), file=sys.stderr)
        return 2
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        raise SystemExit(__doc__)
    port = int(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    wanted = {arg for arg in sys.argv[2:] if not arg.startswith("--")}
    unknown_ids = sorted(wanted - set(ROUND))
    if unknown_ids:
        raise SystemExit("not in this round: " + ", ".join(unknown_ids))
    check_console_defaults()
    failed: list[str] = []
    for experiment_id in ROUND:
        if wanted and experiment_id not in wanted:
            continue
        merged = validated(experiment_id)  # exits before anything is sent
        if dry_run:
            directive = str(merged["fold_exploration_directive"])
            print(json.dumps({key: merged[key] for key in REPORT_KEYS}, ensure_ascii=False))
            print(f"  directive: {len(directive.splitlines())} lines, {len(directive)} chars")
            for line in directive.splitlines():
                print("   |", line)
            continue
        if not post(port, request_params(experiment_id)):
            failed.append(str(experiment_id))
    if failed:
        print("not created: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
