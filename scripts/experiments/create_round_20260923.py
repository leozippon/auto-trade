#!/usr/bin/env python
"""The 2026-09-23 round: one confirmed carrier, three levers, two indices.

The dataset selection and the prebuilt seed are the 2026-09-19 round's -- the
2026-09-20 selection plus `index_weight` -- imported rather than restated so
the rounds that mount it cannot drift apart by a typo. Every arm needs it:
each universe is a decision day's visible index section, and the host's
zero-skill panel matches its replacements on constituent membership, so an arm
without the section would be graded against a different floor than it trades.
One arm, `flow_block`, also selects the events dataset `intraday_flow`, so it
names its own seed, prebuilt from the same release for exactly that selection.
Minute data stays unmounted; `include_intraday` keeps the console default
`False`, which `check_console_defaults` pins for every round.

What this round is. The 20260922 round changed the model four ways -- relation
graph, cross-name attention, persisted GRU, meta-labeling -- and all four arms
closed `no_edge`, while the one leg that read real skill in every one of them
was the required Alpha158 + LightGBM control (active +4.30 %/yr, active IR
0.4642, 3/4 years; +1.59 %/yr and 0.18 on another arm). Both readings are below
the 0.75 freeze bar. The carrier is real and the edge is short, so this round
holds the carrier fixed and moves one lever per arm: the INFORMATION it reads
(`board_block`, `flow_block`), the LABEL it is asked to predict (`label_axis`),
and the INDEX whose section it is drawn from (`csi500_shape`). Nothing else
differs, which is what makes each arm's own control a control.

`benchmark_index` is a create-time parameter as of 6708efb (console default
`000300.SH`), so it is stated on every arm rather than inherited: the arm that
moves it is the point of this round, and an arm that does not move it should
say so in its own request. Moving it moves the verdict's benchmark, the panel's
membership side and the neutralisation leg together, which is precisely the
blocker that kept a wider index shape off the board until now.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
        "initial_cash": <the account this arm runs>,
        "benchmark_index": <the index the host grades this arm against>,
        "gpu_count": <stated; every arm this round is CPU-only>,
        "max_drawdown" / "active_max_drawdown":
            stated so `--dry-run` prints the gates this arm is judged by,
    }

`_round.Round.request_params` merges the arm last. No arm carries a tracking
mandate this round, so every one states the rules' own drawdowns 0.45 / 0.30
and names no `tracking_error_cap`. Every arm also names the statistical bars
(IR, DSR, positive-year share, full-span validations, forward confidence,
recency, activity, Held-out z) so the create request records a choice rather
than inheriting a hidden pair of packages.

Each pack carries its own offline gate (`G-INC2` in `families.md`), which must
pass BEFORE any full-span batch. It is offline: it spends no replay-year and
counts no trial, which matters because each completed validation, whatever its
span, raises this arm's own deflated-Sharpe bar.

Contamination, per arm. Operator-side only.

- Round-level. The decision to hold the carrier fixed and move one lever per
  arm was taken after all four 20260922 arms closed `no_edge`; those closes are
  research-period readings of their own controls, not forward information. The
  operator has read the shared-forward discards that preceded them.
- `board_block_1m_20260923`. The four board tables have been mounted in every
  arm's snapshot and read by none, so no forward reading exists for them. The
  carrier's lineage is the one whose calibration arm's forward verdict the
  operator has read; the carrier runs here as `c_base`, the required first leg
  of every batch, so the contamination reaches the comparison rather than the
  increment being claimed.
- `label_axis_100k_20260923`. Same carrier, same statement. No arm has ever
  closed on its label and none has moved this lever, so the six variants carry
  no prior forward verdict of their own.
- `csi500_shape_1m_20260923`. The CSI 500 section has never carried a book
  here; there is no forward reading on this shape at all. The CSI 300 readings
  that motivate the move (0.4642 and 0.18) are research-period and are
  evidence, never an in-batch control -- an arm has one `benchmark_index`, and
  a CSI 300 leg inside this arm would be graded against the CSI 500 benchmark
  and panel, which is the mismatch the packs name as forbidden.
- `flow_block_1m_20260923`. `intraday_flow` has never been selected into any
  arm's snapshot, so no forward reading exists for it. Minute OFI as a
  standalone score and vendor money flow as the main signal are closed families
  the pack forbids reopening; the block is priced only as the increment over
  `c_base`, and the carrier statement is `board_block`'s.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260923.py <port> [--dry-run] [--fill] [experiment_id ...]
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

from scripts.experiments._round import Round

# The selection and the prebuilt view tree are the benchmark round's, imported
# rather than restated: `index_weight` is what both the in-index universes and
# the panel's membership matching depend on, and the four board tables the
# first arm reads are already in that selection and in the pinned seed.
from scripts.experiments.create_round_20260919 import (
    EVENTS_DATASETS,
    FUNDAMENTAL_DATASETS,
    MACRO_DATASETS,
    PIT_VIEWS_SEED,
    TEXT_DATASETS,
)

# The console default, stated so an arm that keeps it records a choice.
CSI300 = "000300.SH"
CSI500 = "000905.SH"

# Create-time graduation bars, named on every arm so the request records a
# choice. These are today's defaults; an arm that wants other bars overrides
# them the same way it overrides drawdowns.
GATES: dict[str, object] = {
    "min_active_ir": 0.75,
    "min_dsr_probability": 0.90,
    "min_positive_year_share": 0.75,
    "min_full_span_validations": 2,
    "forward_confidence": 0.80,
    "recency_months": 6,
    "min_mean_gross": 0.50,
    "min_round_trips_per_month": 1,
    "heldout_tolerance_z": 1.28,
}

ARMS: dict[str, dict[str, object]] = {
    # Lever: information. One ablation on the confirmed carrier.
    "board_block_1m_20260923": {
        "workspace_reference": "configs/workspace_refs/board_block_20260923",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是消融臂：载体 Alpha158 + LightGBM 在沪深 300 成分上一个字不改，"
            "只在特征矩阵末尾加 17 列涨跌停 / 龙虎榜 / 打板池特征，c_base 永远是同批第一条腿、永不可提名。"
            "先读 families.md 的逐年非零占比表，再过离线 G-INC2 消融门——Δ ≤ 0 或这 17 列的真实拟合增益占比"
            "低于 5 % 就不开整期批次。起步包在第四研究年两次真实 fit 上实测增益占比 4.2–4.5 %，已经低于门槛，"
            "第 0 轮必须先把这一读数复核一遍，命中后走轴 a 把窗口拉长，不要改载体或改池。"
        ),
    },
    # Lever: label. The one part of the carrier no arm has ever moved.
    "label_axis_100k_20260923": {
        "workspace_reference": "configs/workspace_refs/label_axis_20260923",
        "initial_cash": 100_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只换标签：载体、池、席位、成本与持久化节拍一个字不改。预登记的 c_base 是在位的 10 日"
            "开盘到开盘 β 残差秩标签（必跑对照，永不可提名），另有 6 个单成分变体——20 / 40 日 horizon、"
            "收到开的执行口径、收到收、行业去均值、波动缩放，每条只移动一个部件。"
            "先用离线秩 IC 与面板比较筛选，最多 2 个变体进整期批次，并把试验计数与 DSR 算术写进简报。"
            "10 万元 12 席、月频复核、MAX_SWAPS = 3；第 0 轮必须实测复算年换手与每月完成回合数，两头都是提名条件。"
        ),
    },
    # Lever: index. The first arm the new create parameter makes possible.
    "csi500_shape_1m_20260923": {
        "workspace_reference": "configs/workspace_refs/csi500_shape_20260923",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂把同一条载体原样搬到中证 500 成分上：基准、零技能面板的换名侧与中性化腿全部按 000905.SH，"
            "50 席等额现金、月频复核。开工第一件事是核对运行事实的 benchmark_index 与 starter/lib/index.py 的"
            " INDEX_CODE 一致，不一致立即停，不改代码去迁就参数、也不改参数。"
            "批次里不得放一条沪深 300 的腿——一条臂只有一个基准；沪深 300 上同载体的 0.46 / 0.18 是证据不是对照。"
            "离线登记同一载体在 300 与 500 成分上的秩 IC 对比，并在量出中证 500 的零技能地板之前不解释任何绝对读数。"
        ),
    },
    # Lever: information, the second block. The one arm with its own snapshot
    # selection: `intraday_flow` joins the events domain, so the arm names the
    # seed prebuilt from the same release for exactly that selection, and the
    # three arms above keep the 2026-09-19 tree they pin.
    "flow_block_1m_20260923": {
        "workspace_reference": "configs/workspace_refs/flow_block_20260923",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        "events_datasets": [*EVENTS_DATASETS, "intraday_flow"],
        "pit_views_seed": "data/pit_views_seed_research_20260923_flow",
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是消融臂：载体 Alpha158 + LightGBM 在沪深 300 成分上一个字不改，只在特征矩阵末尾加 14 列带符号订单流——"
            "`intraday_flow` 逐股日 OFI 的当日值、5 日与 20 日均值、z 值与金额加权均值，两列通道质量，五列 `moneyflow` 分档资金，"
            "以及两列跨方法分歧（tick 规则定符号 vs `moneyflow` 按单量定符号）。c_base 永远是同批第一条腿、永不可提名，f_only 只作诊断。"
            "先过离线 G-INC2 消融门——Δ ≤ 0 或这 14 列的真实拟合增益占比低于 5 % 就不开整期批次；起步包实测增益占比 15 %、覆盖 99.9 %，"
            "但占比高只说明树用了它，不是收益读数。100 万元 50 席、月频复核、每次最多换 8 只，第 0 轮必须实测年换手；"
            "复核节奏改回周度是违规不是结果，也不得把本块或其中任何一列单独做成排序分数。"
        ),
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    overrides={
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
