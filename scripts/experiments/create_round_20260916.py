#!/usr/bin/env python
"""Create the 2026-09-16 round: new information sources on one extended PIT seed.

This round refills console slots freed by the 2026-09-09 retirement batch
(`cb_linkage_20260914`, `site_visits_20260914`, `corner_cases_20260907`,
`factor_cs_allflash_20260910`) and by `open_mechanism_20260910` reaching its
Held-out verdict. `explore_github_strategies_20260910` and `ml_ranker_20260910`
keep running untouched, so round 20260910 stays in the tree as their
definition; `value_regime_20260914` has been asked to stop at a session
boundary, so round 20260914 stays in the tree as the record of an arm that ran,
not as a live definition. Three new arms therefore sit inside
`webui.manager.MAX_RUNNING_EXPERIMENTS` with room to spare rather than filling
it exactly -- what must hold is the cap, and the console refuses a create past
it whatever else is running at the time.

All three arms move the information source off the daily price/volume table
every earlier arm kept transforming.
`analyst_revision` moves it onto the sell side: the
`report_rc` numeric slice (per-broker, per-fiscal-year EPS forecasts), which
commit 65edfae exposed as an events dataset and which no arm has ever read.
`margin_flow` moves it onto the margin-trading plumbing: the `margin_secs`
eligibility roster and the `margin_detail` financing balance -- a regulatory
admission list and a leverage stock. `alt_events_ranker` keeps the estimator
familiar and changes the feature set instead: a walk-forward LightGBM
cross-sectional ranker over the disclosure and flow tables, judged against a
same-learner attribution control that sees only the 14 daily price/volume
columns, so the claim under test is that those tables carry something the daily
bar cannot learn. It requests no GPU -- its pack forbids torch and fits on the
CPU. Each arm points at its own reference pack under `configs/workspace_refs/`, and
every model role of every arm stays on the local qwen-3.8-27b-fp8, which is the
console creation default for all six roles, so this round overrides none of
them and pins them in EXPECTED_DEFAULTS instead.

What is new at the parameter level is the data. Both arms select the same
extended macro and events datasets and therefore share one prebuilt view tree,
`data/pit_views_seed_ext_20260916`. Against round 20260914's seed the macro side
is unchanged and the events side gains exactly two tables, `margin_secs` and
`report_rc` -- one per arm. A PIT view seed's identity IS the whole snapshot
configuration: the dataset selection, the domain switches, the window months
and the universe screen. So every arm must ask for a byte-identical selection
or the seed stops matching and the arm cold-builds every view for hours. That
is why the selection lives in COMMON_OVERRIDES once, why the two lists are
restated here rather than imported from round 20260914 (they describe a
different tree, and that round's arm must stay creatable exactly as it was),
and why `include_intraday` -- a console default today -- is stated rather than
inherited: it is part of what the seed was built for.
`worker.resolve_worker_options` checks the seed's recorded snapshot
configuration and its cache format against the request at create time, so a
drift in any of these fails the create instead of silently cold-building. What
that check cannot see is the calendar the seed was planned over: `fold_period`,
the two Development bounds, the Held-out range, `validation_periods` and
`min_region_trade_days` decide which Folds exist but never enter the snapshot
configuration, so a drift there would leave views missing rather than fail.
Every one of them is therefore either stated in COMMON_OVERRIDES or pinned in
EXPECTED_DEFAULTS. Both packs verified their source tables cover the whole
2022Q1--2025Q4 input window, so unlike `site_visits` neither arm moves its
Development start.

`heldout_min_final_transitions` is stated at its default value of 1. It is the
Held-out gate commit f6443ae added -- the artifact that ships must carry that
many forward transitions of its own -- and both packs' end-game rules are
written against it, so the round manifest names it instead of leaving the
reader to look up a console default.

The rest of the design is round 20260910's and is not re-decided here. The CNY
100k account, the graduation gates and the per-Fold budgets are read from that
round's own COMMON_OVERRIDES rather than retyped, so three rounds sharing one
console cannot drift apart on cost; the quarterly walk-forward geometry agrees
with it but is restated, because this round's Folds have to stay the ones its
seed was built for. The offline validation, the report and the POST path are
that round's functions, imported unchanged.

Every parameter set is validated offline with every request-level check the
console applies on POST /api/experiments, and additionally refuses a directive
the PRIOR calendar policy would reject -- which `resolve_worker_options` itself
enforces on `fold_exploration_directive`, so a directive carrying a literal
calendar date would not merely be impolite, it would fail the worker at start.
Two things --dry-run cannot answer stay at POST time: an experiment directory
that already exists and a free running slot. The seed is a third, and only half
answerable: because it is named explicitly it is required, so the pre-flight
checks that the tree is there, that its recorded contract is this one, and that
no staged slot is left in it -- a prebuild stages each view beside its
destination and renames it in only when it is complete, so a staged slot is
on-disk proof that the build is unfinished. An arm refused for that reason is
waiting for the prebuild, not misconfigured, and `seed_status` says so in one
line before the arms are read.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \
      scripts/experiments/create_round_20260916.py <port> [--dry-run] [experiment_id ...]
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

# Rounds 20260910 and 20260914 are still the live definitions of the three arms
# running beside this one and stay in the tree. Importing 20260910 costs nothing
# (constants and pure functions only) and keeps one source for the create
# contract: the same offline validation, the same report keys, the same POST
# path, the same budgets and the same model-role pins.
# Appended, not prepended: the repository root carries directories named `data`,
# `logs` and `results`, which must never shadow an installed package.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from autotrade.pipelines.config import SNAPSHOT_CACHE_FORMAT_VERSION
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

# The one prebuilt view tree every arm of this round hardlinks from. Built by
# scripts/data/prebuild_pit_views_seed.py with exactly the selection and
# calendar below; gitignored, so it is an operator precondition, not an artifact
# of this repository.
PIT_VIEWS_SEED = "data/pit_views_seed_ext_20260916"

# The extended selection this round's seed was built for. Not lists that happen
# to agree: the seed's identity is the whole snapshot configuration, so every
# arm has to send this exact selection or it finds no matching contract and
# cold-builds every view. The macro side is round 20260914's (futures, options
# and the three convertible-bond tables beyond the console default scope); the
# events side is that round's plus `margin_secs` for margin_flow and `report_rc`
# for analyst_revision. Both are written out rather than imported from round
# 20260914: they describe a different tree, and that round's surviving arm has
# to stay creatable exactly as it was. include_macro/include_events must stay
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

# What this round decides for every arm.
COMMON_OVERRIDES: dict[str, object] = {
    # No GPU, one Epoch, the CNY 100k account, the graduation gates and the
    # per-Fold budgets (600 min / 24 Steps / 24 backtests / 1600 calls): round
    # 20260910's decisions, read from that module instead of retyped. Its
    # surviving arms run beside these under the same cost regime, and a
    # walk-forward comparison across the console only means anything while that
    # stays true.
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
    # built for them. No arm of this round overrides them -- both packs verified
    # their source tables cover the whole input window.
    "fold_period": "quarter",
    "validation_periods": 4,
    "development_first_period": "2022Q1",
    "development_last_period": "2025Q4",
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
    # The Held-out gate added in f6443ae, stated at its default: the shipped
    # artifact must carry one development transition of its own, judged by the
    # same ceil(2/3)-positive rule. Both packs' end-game sections are written
    # against it, so the round manifest names it rather than leaving it to a
    # console default a reader would have to look up.
    "heldout_min_final_transitions": 1,
}

# Console defaults this round still relies on: round 20260910's pins minus every
# key this round now decides for itself. Values, not commentary -- a drift stops
# the script instead of silently re-scoping the arms. The six model roles are
# the load-bearing ones: no arm overrides a role, so the console default is what
# actually decides them, and a rename of the local model is exactly the drift
# this has to catch.
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
    # Both packs promise the Agent this many host null controls per Fold and
    # write their "no edge" readings around that budget, so it is pinned rather
    # than assumed.
    "max_null_controls_per_fold": 3,
}

ROUND: dict[str, dict[str, object]] = {
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

# Round 20260910's report keys plus what this round adds: the dataset selection,
# the two domain switches that decide whether it applies at all, and the
# Held-out own-transition gate this round states explicitly.
REPORT_KEYS = (
    *ROUND_20260910_REPORT_KEYS,
    "include_macro",
    "macro_datasets",
    "include_events",
    "events_datasets",
    "heldout_min_final_transitions",
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


def seed_status() -> str:
    """One line on the shared seed, printed before the arms.

    What the per-arm pre-flight decides has three parts, and this line says
    which of them can already be read here: the tree exists, `provider.json`
    records this round's snapshot configuration under the current cache
    format, and the build left no staged slot behind. The first two are read
    below; the third is the pre-flight's own
    (`pit_views_seed.assert_seed_snapshot_config` refuses a tree a prebuild is
    still writing into), so an arm's refusal names it rather than this line.
    """
    provider = REPO_ROOT / PIT_VIEWS_SEED / "provider.json"
    if not provider.is_file():
        return (
            f"seed {PIT_VIEWS_SEED}: provider.json absent -- the prebuild has not written"
            " its contract yet, so the snapshot-configuration pre-flight cannot run and"
            " every arm below is refused"
        )
    try:
        record = json.loads(provider.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"seed {PIT_VIEWS_SEED}: provider.json unreadable ({exc}); every arm below is refused"
    version = record.get("schema_version")
    if version != SNAPSHOT_CACHE_FORMAT_VERSION:
        return (
            f"seed {PIT_VIEWS_SEED}: built under snapshot cache format {version!r} but this"
            f" code writes {SNAPSHOT_CACHE_FORMAT_VERSION} -- the pre-flight refuses it;"
            " rebuild the seed under a new directory"
        )
    return (
        f"seed {PIT_VIEWS_SEED}: contract present (snapshot cache format {version}), so the"
        " per-arm pre-flight below compares this round's selection against it and, on top"
        " of that, refuses the tree while a prebuild is still staging views into it --"
        " an arm refused for an unfinished build is a wait, not a wrong parameter."
    )


def validated(experiment_id: str) -> tuple[dict[str, object] | None, str]:
    """params.json for one arm, or ``(None, reason)`` with an operator-readable refusal.

    ``normalize`` runs the console's own create-time checks and the worker
    pre-flight; the pre-flight requires the shared PIT view seed, which is a
    separate and slow operator step, so a round dry-run started before the
    prebuild wrote its contract has to say that in one line instead of a
    traceback. The refusal is returned rather than raised because a dry-run
    should read every arm before it stops -- the seed is shared, so the first
    arm's refusal says nothing about whether the others are well-formed.
    """
    try:
        return normalize(request_params(experiment_id)), ""
    except ValueError as exc:
        reason = f"{experiment_id}: parameters rejected, nothing was sent: {exc}"
        if "unfinished build" in str(exc):
            reason += (
                "\n  the tree is there and its contract is this round's; the prebuild"
                " is still staging views into it. Wait for"
                " scripts/data/prebuild_pit_views_seed.py to report status ok, then"
                " re-run -- no parameter needs changing"
            )
        elif "pit_views_seed" in str(exc) or "view seed" in str(exc):
            reason += (
                f"\n  every arm shares one prebuilt view tree ({PIT_VIEWS_SEED});"
                " build it with scripts/data/prebuild_pit_views_seed.py for exactly"
                " this dataset selection and calendar before creating or dry-running"
                " the round"
            )
        return None, reason


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
    print(seed_status())
    failed: list[str] = []
    for experiment_id in ROUND:
        if wanted and experiment_id not in wanted:
            continue
        merged, reason = validated(experiment_id)
        if merged is None:
            # On the POST path nothing may be sent for a rejected arm; on a
            # dry-run the refusal is a reading, so the remaining arms are read
            # too and the exit code still reports it.
            if not dry_run:
                raise SystemExit(reason)
            print(reason, file=sys.stderr)
            failed.append(experiment_id)
            continue
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
        label = "refused" if dry_run else "not created"
        print(f"{label}: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
