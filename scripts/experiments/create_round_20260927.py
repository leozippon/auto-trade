#!/usr/bin/env python
"""The 2026-09-27 round: the first eight-year research-period arms.

What this round is. After the 2026-09-26 breadth round closed seven arms, the
register (docs/research-lessons.md §3) holds one signal whose host readings are
positive on every pool it has run on: the untrained low-range score `range20`
(minus the mean of a name's own last 20 daily (high - low) / pre_close). Over
the four research years 2021-07..2025-06 its active IR read 1.25 on CSI 300
(`d1_range20cap`, the window it was selected on as the best of 18 price-volume
signals), 0.834 on CSI 500 and 0.564 on CSI 1000 (the same 50-seat book,
`range20_*_1m_20260926`). None reached the four-year freeze bar (IR about 0.98
at one trial): what they lacked was T, not the sign (register §5 item 9).

Eight research years answer both halves of that. 2017-07..2021-06 (Y1-Y4) is a
genuine out-of-sample half in time for a score selected on 2021-2025, for every
pool; and the bar falls: sqrt(244 / 1941) puts the DSR bar at IR 0.695 at one
effective trial, so the arm's own `min_active_ir` 0.75 binds, with the
positive-year gate at ceil(0.75 x 8) = 6 of 8. The pack author's faithful
offline reading of that half -- the registered construction replayed daily
against a host-like zero-skill panel, benchmark- and size-neutralized
(logs/notes/round_20260927/PACK_range20_8y.md §3), read after the
registration and changing nothing in it -- is negative on every pool: CSI 500
-0.84, CSI 300 -0.43, CSI 1000 -0.56, one positive year of four each. So these
arms are pre-registered falsifications: each runs its one registered batch
(`c_shuf` control first, then `r1`; 16 replay-years) and closes `no_edge` if
the Y1-Y4 mean active excess is <= 0, the full-span active IR is below the
zero-skill 90th percentile 1.2816 x sqrt(244 / 1941) = 0.4544, fewer than 6
of 8 years are positive, or the active drawdown exceeds 0.30. They are also
the first real run of the eight-year pipeline.

The eight-year geometry and seed (logs/notes/round_20260927/E8_eight_year_build.md).
Research 20170701..20250630 on a 108-month window -- 96 research months plus
the twelve a rule score reads -- over release `41f74471...`, which carries the
2014-07 backfill of `index_weight`, `index_daily`, `index_dailybasic` and
`sw_daily`. Fundamental, event and text history starts inside that window, so
the selection mounts none of them and only the four index and Shenwan macro
series: the starter reads `daily`, the universe and `index_weight`, the host's
panel and benchmark leg read `index_weight` and `index_daily`. Forward and
Held-out are unchanged. Every arm states the geometry, the selection and the
seed as its pack author's pre-flight checked them.

Budgets. A full validation now costs 8 replay-years; the default 96 holds the
registered path (16, or 24 with the registered `r2`) with room for refunds,
and a larger budget would only widen unregistered search, which the gate
charges as trials. The round states it so the choice is recorded. The others
are the shared ones.

The arms, in queue order:

- `range20_csi500_8y_20260927` (primary). CSI 500 constituents, 50 seats,
  monthly, keep band 100, at most 14 names per SW L1; `offline_trials` 0 --
  the 17 siblings of the CSI 300 selection are disclosed, not declared, as in
  2026-09-26.
- `range20_csi300_8y_20260927`. The pool the score was selected on, so only
  Y1-Y4 is out of sample, and the 17 other signals screened there declare
  `offline_trials` 17: N_eff 18 and an IR bar of 1.352, which the pack also
  makes a nomination condition, because a correlated `r2` would otherwise
  dilute those declared trials.
- `dvy_csi300_8y_20260927`. Trailing dividend yield on CSI 300, queued only
  after the dividend-event store and a new eight-year seed
  (`data/pit_views_seed_research_8y_div_20260924`, release `55fc7fd8...`)
  cover ex-dates from Y1. It overrides `pit_views_seed`; the range20 arms
  stay on `data/pit_views_seed_research_8y_20260927`. `offline_trials` 0.

Held, not queued: `range20_csi1000_8y_20260927`. Its pack is written and
checked in and its entry is HELD below. It is appended to ARMS only if either
running arm's out-of-sample half reads a non-negative active IR on the host
(the mean of its Y1-Y4 active neutralized excess >= 0). Otherwise the two
readings, with the offline -0.56 beside them, answer item 9 for the whole pool
curve and the third batch buys nothing.

Drawdown caps clear each benchmark's own maximum drawdown over the eight
research years: CSI 500 -41.8 % -> 0.50, CSI 300 -45.6 % -> 0.50 (not the
rules' 0.45), CSI 1000 -48.1 % -> 0.55; `active_max_drawdown` 0.30 everywhere.
Every arm names the statistical bars (GATES) so the create request records a
choice. No arm carries a tracking mandate.

Contamination, per arm. Operator-side only.

- Round-level. The round was designed after the 2026-09-26 closures and their
  research-period readings; the operator has read the shared-forward discards.
  The eight-year selection mounts nothing a four-year arm did not, and the
  packs quote no forward or Held-out figure (checked by
  `test_no_new_pack_table_quotes_the_forward_period`).
- `range20_csi300_8y`. The score was selected on this pool over Y5-Y8; the
  operator has read the forward discard of `index_relative_1m_20260919`,
  another rule score on CSI 300. Declared, not only disclosed: 17 trials.
- `range20_csi500_8y`. Y5-Y8 repeats the 2026-09-26 host reading (0.834);
  the book is carried over from Y4 rather than built flat on 2021-07-01, so
  that half is not order-identical to it.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260927.py <port> [--dry-run] [--fill] [experiment_id ...]
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

# Built for exactly EIGHT_YEAR over release 41f74471aa754d5e80e52a6360711c5a.
PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_20260927"
# Same geometry, release 55fc7fd82c3c4c77b752920ec4507609: dividend events from
# 2016-01, so Y1 corporate actions are non-empty. Only the dividend-yield arm
# pins this tree; the range20 arms stay on the seed they already ran.
DIVIDEND_PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_div_20260924"

# The research geometry and dataset selection the seed was planned over; part
# of its contract, so no arm may change one of them alone.
EIGHT_YEAR: dict[str, object] = {
    "research_start": "20170701",
    "research_end": "20250630",
    "forward_end": "20260630",
    "heldout_end": "20260930",
    "window_months": 108,
    "include_fundamentals": False,
    "include_events": False,
    "include_text": False,
    "macro_datasets": ["index_daily", "index_dailybasic", "sw_daily", "index_weight"],
}

CSI300 = "000300.SH"
CSI500 = "000905.SH"
CSI1000 = "000852.SH"

# Create-time graduation bars, named on every arm so the request records a
# choice. These are today's defaults.
GATES: dict[str, object] = {
    "min_active_ir": 0.75,
    "min_dsr_probability": 0.975,
    "min_positive_year_share": 0.75,
    "min_full_span_validations": 2,
    "forward_confidence": 0.80,
    "recency_months": 6,
    "min_mean_gross": 0.50,
    "min_round_trips_per_month": 1,
    "heldout_tolerance_z": 1.28,
}

ARMS: dict[str, dict[str, object]] = {
    # Primary. The untrained range20 score on CSI 500 constituents over eight research years:
    # the first four are out of sample in time for a score selected on CSI 300 over the last four.
    "range20_csi500_8y_20260927": {
        "workspace_reference": "configs/workspace_refs/range20_csi500_8y_20260927",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂，"
            "八年研究期三臂里的主臂：range20（每只股票自己最近 20 根日线的 (最高−最低)/昨收 均值取负）在决策日在册的中证 500 成分里建 50 席等额、月度复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决；前四个研究年对这个分数的挑选是样本外。"
            "benchmark_index 必须是 000905.SH、initial_cash 为 100 万，"
            "与 knobs 的 INDEX / CAPITAL 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门 G-R20、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ r1（control: false），"
            "offline_trials = 0；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "都没命中才跑登记的 r2（同号只是读数），按 refs/families.md 提名。"
        ),
    },
    # The pool the score was selected on: only the first four research years are out of sample, and
    # the seventeen other signals screened there are declared as offline trials (N_eff 18, bar 1.352).
    "range20_csi300_8y_20260927": {
        "workspace_reference": "configs/workspace_refs/range20_csi300_8y_20260927",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：range20（每只股票自己最近 20 根日线的 (最高−最低)/昨收 均值取负）在决策日在册的沪深 300 成分里建 50 席等额、月度复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决；分数就是在这个池的后四个研究年上 18 选 1 挑出来的，前四年是样本外。"
            "benchmark_index 必须是 000300.SH、initial_cash 为 100 万，"
            "与 knobs 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门 G-R20、分两半报；"
            "唯一一批 c_shuf（control: true，第一条）+ r1（control: false），"
            "offline_trials = 17（那 17 个兄弟信号，刻意申报，门槛 1.352）；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "提名还要主动 IR ≥ 1.352，按 refs/families.md 走。"
        ),
    },
    # Trailing cash dividend yield on CSI 300. The last four research years are
    # the selection risk; the first four are the test. This arm alone uses the
    # dividend-complete seed.
    "dvy_csi300_8y_20260927": {
        "workspace_reference": "configs/workspace_refs/dvy_csi300_8y_20260927",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；两次年度派息之间滚动窗断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的沪深 300 成分里建 50 席等额、月度复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决；后四个研究年的红利与国企行情是挑选风险，前四年是检验。"
            "benchmark_index 必须是 000300.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 0；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "银行权重与每月完成回合逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
}

# Written, not queued: appended to ARMS only if either range20 arm above reads a non-negative
# out-of-sample half (mean Y1-Y4 active neutralized excess >= 0) on the host. The small-cap end of
# the pool curve on the same eight years; CSI 1000 drew down 48 % over them, hence the 0.55 cap.
HELD: dict[str, dict[str, object]] = {
    "range20_csi1000_8y_20260927": {
        "workspace_reference": "configs/workspace_refs/range20_csi1000_8y_20260927",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：range20（每只股票自己最近 20 根日线的 (最高−最低)/昨收 均值取负，"
            "越安静越前）在决策日在册的中证 1000 成分里建 50 席等额、月度复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决；前四个研究年对这个分数的挑选是样本外。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，"
            "与 knobs 的 INDEX / CAPITAL 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门 G-R20、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ r1（control: false），"
            "offline_trials = 0；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "都没命中才跑登记的 r2（同号只是读数），按 refs/families.md 提名。"
        ),
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=PIT_VIEWS_SEED,
    overrides={**EIGHT_YEAR, "max_replay_years": 96},
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
