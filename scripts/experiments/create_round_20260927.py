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

The three founding arms, first in the queue:

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

The queue then took every eight-year arm opened through 2026-09-27: 181 arms
in all, each with its pack at configs/workspace_refs/<id>, all on the
geometry, gates and drawdown caps below, and every one already created, so
`--fill` now finds nothing to do. After the founders come 15 dividend-yield
variants on CSI 500 and CSI 1000 (`dvy_*_20260924` / `_20260925`: capital,
seats, keep band, review clock, ChiNext excluded); 15 trained-model arms on
CSI 1000 or the whole pool (`gnn`, `gru`, `xsattn`, `lgbm`, `tcn`, `mlp`,
`lstm`, `tattn`, `lagpool`, `shortmlp`, `patchmlp`, `decay`, `spec`; 14 of
them on one GPU); and 148 arms on the whole pool, 100k and 12 seats each
(`*_alla_100k_20260927`). The whole pool is every A-share outside
the Beijing exchange and the STAR market (688 / 689), measured against CSI
1000. Every arm except the two range20 ones pins the dividend-complete seed.

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
  The eight-year selection mounts nothing a four-year arm did not, and no
  pack's closure or register table quotes a forward or Held-out figure
  (`test_no_new_pack_table_quotes_the_forward_period` reads those tables, not
  the prose). The later arms were designed from the research-period readings
  of the arms of this round that had closed before them.
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

# The eight-year geometry and selection, CSI 1000 and the graduation bars this
# round first stated are shared with the rounds after it.
from scripts.experiments._profiles import CSI1000, EIGHT_YEAR, GATES
from scripts.experiments._round import Round

# Built for exactly EIGHT_YEAR over release 41f74471aa754d5e80e52a6360711c5a.
PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_20260927"
# Same geometry, release 55fc7fd82c3c4c77b752920ec4507609: dividend events from
# 2016-01, so Y1 corporate actions are non-empty. Only the dividend-yield arm
# pins this tree; the range20 arms stay on the seed they already ran.
DIVIDEND_PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_div_20260924"

CSI300 = "000300.SH"
CSI500 = "000905.SH"

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
    # Same yield score off the bank-heavy pool. 1m / 50 seats.
    "dvy_csi500_1m_20260924": {
        "workspace_reference": "configs/workspace_refs/dvy_csi500_1m_20260924",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；两次年度派息之间滚动窗断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 500 成分里建 50 席等额、月度复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决。换的是股票池：沪深 300 上的银行集中不要带到本池，行业权重逐次报。"
            "benchmark_index 必须是 000905.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 0；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # 100k cannot fund 50 CSI 500 seats. 12 seats, same low-turnover yield book.
    "dvy_csi500_100k_20260924": {
        "workspace_reference": "configs/workspace_refs/dvy_csi500_100k_20260924",
        "initial_cash": 100_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 500 成分里建 12 席等额、月度复核、保留带前 24 名、每个申万一级至多 3 只的书，"
            "不训练，由宿主零技能面板裁决。10 万账户有 5 元最低佣金，保持这本低换手月度书，不要提高换手。"
            "benchmark_index 必须是 000905.SH、initial_cash 为 10 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 0；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "买不起一手的比例与行业权重逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # Dividend yield on CSI 1000. The CSI 300 and CSI 500 monthly books already finished; this pool has not.
    "dvy_csi1000_1m_20260924": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_1m_20260924",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 50 席等额、月度复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决。小盘股息可能是价值陷阱，行业权重逐次报，不要改成分数。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 0；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    "dvy_csi1000_100k_20260924": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_100k_20260924",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 12 席等额、月度复核、保留带前 24 名、每个申万一级至多 3 只的书，"
            "不训练，由宿主零技能面板裁决。10 万账户有 5 元最低佣金，保持月度低换手，不要提高换手。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 0；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "买不起一手的比例逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # Quarterly review of the CSI 500 yield book. Monthly on this pool already finished below the bar.
    "dvy_csi500_1m_q_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi500_1m_q_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 500 成分里建 50 席等额、每季首个决策日复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决。复核节奏是季度，不要改回月度。"
            "benchmark_index 必须是 000905.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    "dvy_csi1000_100k_q_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_100k_q_20260925",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 12 席等额、每季首个决策日复核、保留带前 24 名、每个申万一级至多 3 只的书，"
            "不训练，由宿主零技能面板裁决。10 万账户有 5 元最低佣金，复核节奏是季度，不要改回月度，也不要提高换手。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "买不起一手的比例逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # 12-seat monthly CSI 1000 yield had IR 0.72 and died on active drawdown. More names.
    "dvy_csi1000_100k_s20_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_100k_s20_20260925",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 20 席等额、月度复核、保留带前 40 名、每个申万一级至多 5 只的书，"
            "不训练，由宿主零技能面板裁决。席位是 20，不要改回 12。10 万账户有 5 元最低佣金，保持月度低换手。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "买不起一手的比例与主动回撤逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # Same 12-seat book, wider keep band, so fewer swaps after the 12-seat drawdown.
    "dvy_csi1000_100k_b3_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_100k_b3_20260925",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 12 席等额、月度复核、保留带前 36 名、每个申万一级至多 3 只的书，"
            "不训练，由宿主零技能面板裁决。保留带是 3.0，不要改回 2.0。10 万账户有 5 元最低佣金，不要提高换手。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "每月完成回合与主动回撤逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # The 100k 12-seat CSI 1000 book had the real IR. Same concentration on a 1m account.
    "dvy_csi1000_1m_s12_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_1m_s12_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 12 席等额、月度复核、保留带前 24 名、每个申万一级至多 3 只的书，"
            "不训练，由宿主零技能面板裁决。席位是 12，不要改成 50。保持月度复核。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "主动回撤与行业权重逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # 50 seats cleared the drawdown gate at IR 0.57. 30 seats sits between that and the 12-seat book.
    "dvy_csi1000_1m_s30_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_1m_s30_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 30 席等额、月度复核、保留带前 60 名、每个申万一级至多 8 只的书，"
            "不训练，由宿主零技能面板裁决。席位是 30，不要改成 12 或 50。保持月度复核。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "主动回撤与每月完成回合逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # Monthly 50-seat CSI 1000 cleared K4 at IR 0.57 with 3 round-trips a month. Quarterly cuts that churn.
    "dvy_csi1000_1m_q_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_1m_q_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 50 席等额、每季首个决策日复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决。复核节奏是季度，不要改回月度。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "每月完成回合逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # 50-seat monthly cleared drawdown at IR 0.57 with about 3 round-trips a month. A wider band keeps the monthly clock and sells less.
    "dvy_csi1000_1m_b25_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_1m_b25_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 50 席等额、月度复核、保留带前 125 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决。保留带是 2.5，不要改回 2.0，也不要改成季度。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "每月完成回合逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # 100k drawdowns came from a panel that can only buy cheap names. Eight seats raise the price a lot can clear.
    "dvy_csi1000_100k_s8_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_100k_s8_20260925",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里建 8 席等额、月度复核、保留带前 16 名、每个申万一级至多 2 只的书，"
            "不训练，由宿主零技能面板裁决。席位是 8，不要改回 12 或 20。保持月度复核。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "买不起一手的比例与主动回撤逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # ChiNext is about a fifth of CSI 1000 and the yield book barely holds it. Drop it so the panel matches the book.
    "dvy_csi1000_1m_excy_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_1m_excy_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里去掉创业板（300 / 301）后建 50 席等额、月度复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练，由宿主零技能面板裁决。创业板过滤不要去掉。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL / SCORE 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "创业板占比与主动回撤逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    "dvy_csi1000_100k_excy_20260925": {
        "workspace_reference": "configs/workspace_refs/dvy_csi1000_100k_excy_20260925",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：股息率 dvy（dv_ttm；断档时取 dv_ratio；没有现金分红排最后），"
            "在决策日在册的中证 1000 成分里去掉创业板（300 / 301）后建 12 席等额、月度复核、保留带前 24 名、每个申万一级至多 3 只的书，"
            "不训练，由宿主零技能面板裁决。创业板过滤不要去掉。10 万账户有 5 元最低佣金，保持月度低换手。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX / CAPITAL / SCORE / SEATS 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门、分两半报（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ dvy（control: false），offline_trials = 1；"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge；"
            "买不起一手的比例与主动回撤逐次报；都没命中才跑登记的邻居 dv_ratio，按 refs/families.md 提名。"
        ),
    },
    # Relational net and a persistent GRU, on CSI 1000 over eight years.
    # The closed arms of these families were CSI 300 over four years.
    "gnn_csi1000_8y_20260925": {
        "workspace_reference": "configs/workspace_refs/gnn_csi1000_8y_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂把已有的关系网络分数换到中证 1000、八年研究期、100 万、50 席。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。不要改回沪深 300。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "gru_csi1000_8y_20260925": {
        "workspace_reference": "configs/workspace_refs/gru_csi1000_8y_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂把已有的持久 GRU 分数换到中证 1000、八年研究期、100 万。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。不要改回沪深 300。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },

    "xsattn_csi1000_8y_20260925": {
        "workspace_reference": "configs/workspace_refs/xsattn_csi1000_8y_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂把已有的截面注意力分数换到中证 1000、八年研究期、100 万。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。不要改回沪深 300。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "lgbm_csi1000_8y_20260925": {
        "workspace_reference": "configs/workspace_refs/lgbm_csi1000_8y_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂把已有的 LightGBM 截面排序换到中证 1000、八年研究期、100 万，用 CPU。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 0。"
            "不要改回沪深 300，也不要去占 GPU。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },

    "tcn_alla_8y_20260925": {
        "workspace_reference": "configs/workspace_refs/tcn_alla_8y_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数是时间卷积，不是 GRU。benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "mlp_alla_8y_20260925": {
        "workspace_reference": "configs/workspace_refs/mlp_alla_8y_20260925",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数是序列末步与均值拼起来的前馈网络，不是 GRU，也不是卷积。benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },

    "lstm_alla_8y_20260926": {
        "workspace_reference": "configs/workspace_refs/lstm_alla_8y_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数是一层 LSTM，不是 GRU，也不是时间卷积或前馈网络。benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },

    "tattn_alla_8y_20260926": {
        "workspace_reference": "configs/workspace_refs/tattn_alla_8y_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数是沿每只股票自己的 60 日序列做的自注意力，读最后一个时间步。不是截面注意力，也不是 GRU、时间卷积或前馈网络。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。没有 CUDA 就停，不要改成 CPU。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },

    "lagpool_alla_8y_20260926": {
        "workspace_reference": "configs/workspace_refs/lagpool_alla_8y_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数是每个滞后日一个共享权重，再对 60 日做加权，权重不看当天的内容。不是时间卷积、LSTM、自注意力或前馈网络。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。没有 CUDA 就停，不要改成 CPU。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },

    "shortmlp_alla_8y_20260926": {
        "workspace_reference": "configs/workspace_refs/shortmlp_alla_8y_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数是最近 5 根日线的特征直接摊平后送进前馈网络，不做 60 日均值池化。不是 LSTM、时间卷积、自注意力或滞后权重。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 100 万，gpu_count 为 1。没有 CUDA 就停，不要改成 CPU。"
            "按包内 families 的腿走：对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "patchmlp_alla_100k_20260926": {
        "workspace_reference": "configs/workspace_refs/patchmlp_alla_100k_20260926",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数把 60 日序列切成不重叠的 10 日小段，每段摊平后用同一层线性映射，再对小段取平均。"
            "不是 GRU、LSTM、时间卷积、自注意力、滞后权重、末五根摊平，也不是末步加全窗口均值的前馈网络。"
            "残差标签的基准指数必须是 000852.SH。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。"
            "对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "decay_alla_1m_20260926": {
        "workspace_reference": "configs/workspace_refs/decay_alla_1m_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数是每个输入通道一个与内容无关的衰减：h_t = sigmoid(a) * h_{t-1} + x_t，用最后一步打分。"
            "不是 GRU、LSTM、时间卷积、自注意力、滞后权重、末五根摊平，也不是末步加全窗口均值或 10 日小段均值的前馈网络。"
            "残差标签的基准指数必须是 000852.SH。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 100 万，席位 30，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。"
            "对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "spec_alla_100k_20260926": {
        "workspace_reference": "configs/workspace_refs/spec_alla_100k_20260926",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的股票池是全部 A 股里去掉北交所和科创板（688 / 689）的名字，不是指数成分，不要改回指数成分。"
            "分数是每个输入通道实数傅里叶变换幅度的最低 4 个频率，再送进一层小前馈网络。"
            "不是 GRU、LSTM、时间卷积、自注意力、滞后权重、末五根摊平、末步加全窗口均值、10 日小段均值，也不是按通道衰减的递推。"
            "残差标签的基准指数必须是 000852.SH。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。"
            "对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "mlp_alla_100k_20260926": {
        "workspace_reference": "configs/workspace_refs/mlp_alla_100k_20260926",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "这是已毕业的末步加均值前馈网络，原样放到 10 万、12 席上再跑一轮。"
            "不要改网络、标签、14 个通道、基准指数或每周复核。"
            "股票池仍是全部 A 股去掉北交所和科创板（688 / 689），不是指数成分。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。"
            "对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "lstm_alla_100k_20260926": {
        "workspace_reference": "configs/workspace_refs/lstm_alla_100k_20260926",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 1,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "这是已毕业的一层 LSTM，原样放到 10 万、12 席上再跑一轮。"
            "不要改网络、标签、14 个通道、基准指数或每周复核。"
            "股票池仍是全部 A 股去掉北交所和科创板（688 / 689），不是指数成分。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，gpu_count 为 1。"
            "没有 CUDA 就停，不要改成 CPU。"
            "对照在先，主候选一条，offline_trials = 1。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "entry_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/entry_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂离开 60 日价量序列网络。分数是不训练的沪深 300 纳入名次："
            "过去一年日均成交额居前的非北交所股票，按日均总市值排名，名次越靠前分数越高。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689），不是中证 500 成分，不要改回指数成分。"
            "账户 10 万、12 席、月度复核。买不起一手的大市值名字会自然落到后面，要在结果里报买不起的比例。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选 m1 一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
            "优势如果只是便宜大市值的规模暴露，同样 no_edge。"
        ),
    },
    "indrev_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indrev_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂不是序列网络，也不是纳入名次。分数是申万一级行业过去 21 个交易日收益取负，"
            "同一行业的股票分数相同，每个行业最多持有 2 只。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689），不要改回指数成分，不要改成个股 20 日反转。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "indcalm_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indcalm_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂不是序列网络，也不是行业反转。分数是申万一级行业过去 21 个交易日收益的绝对值取负，"
            "行业越平静分数越高，同一行业的股票分数相同，每个行业最多持有 2 只。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689），不要改回指数成分，不要改成个股低波。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "inddisp_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/inddisp_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂不是序列网络，也不是行业涨跌方向或行业涨跌绝对值。"
            "分数是申万一级内部个股 21 个交易日收益的截面标准差取负：同行股票越挤在一起分数越高，"
            "每个行业最多持有 2 只。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689），不要改回指数成分，不要改成个股低波或 20 日反转。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "boundary_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/boundary_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "上一笔纳入名次把书放在了名次 1–14 的老成员上，碰不到纳入边界。"
            "本臂分数在纳入名次 270 附近最高，不要改回偏好最大市值。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689），不要改回中证 500。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "peerlag_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/peerlag_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂不是序列网络，也不是行业指数本身的涨跌。"
            "分数是个股 21 日收益减去它所在申万一级的同期收益，落后于同行的更高，每个行业最多 2 只。"
            "不要改成个股自身的 20 日反转，也不要改成行业反转。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "indflow_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indflow_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂不是序列网络，也不是行业价格指数的涨跌。"
            "分数是申万一级成交额过去 21 个交易日的增长，同一行业的股票分数相同，每个行业最多 2 只。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689），不要改回指数成分。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "leader_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/leader_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂不是序列网络，也不是行业收益、行业成交额增长或行业内收益离散。"
            "分数是个股最近一个交易日的成交额占自己申万一级成交额的比例，每个行业最多 1 只。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689），不要改回指数成分。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "peerlead_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/peerlead_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "peerlag 把落后于本行业的股票做多，整期主动为负且差过随机对照。"
            "本臂取反号：个股 21 日收益减去申万一级同期收益，领先于同行的更高，每个行业最多 2 只。"
            "这个符号是看过 peerlag 之后选的，offline_trials = 1。不要改回落后端，不要改成序列网络。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "decouple_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/decouple_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂不是序列网络，也不是个股相对行业的收益差。"
            "分数是个股日收益与申万一级日收益过去 21 个交易日的相关系数取负，越不跟行业走分数越高，每个行业最多 2 只。"
            "股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "wdelta_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/wdelta_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读指数权重表。分数是中证 1000 最近两张可见成分截面的权重差，不是市值名次，也不是序列网络。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689），不要改回只做中证 500。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "swpe_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/swpe_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读申万行业指数的市盈率。分数是 sw_daily.pe 取负，是行业指数自己的估值，不是个股盈利收益率，也不是行业涨跌。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "cashmix_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/cashmix_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读公司行为。分数是最近一次分配里现金占总分配的比例，送股按收盘价计入总分配。不是股息率排名，没有过去一年分配的不打分。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "paylag_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/paylag_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读公司行为的日期。分数是最近一次现金分红从除息日到派息日的间隔取负，到账越快越高。金额不进分数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "freefloat_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/freefloat_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线的市值。分数是流通市值除以总市值，自由流通占比越高越高。不是换手，也不是收益。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "limdn_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/limdn_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线的跌停价。分数是过去 60 个交易日最低价触及跌停的比例取负。不是一段收益的反转。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "freeshare_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/freeshare_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线股本。分数是 free_share / float_share，自由流通占流通股本越高越高。"
            "不是 circ_mv / total_mv，那一条已经测完。缺数或非正的不打分。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要改去读公司行为表。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "indrise_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indrise_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读申万一级指数的成交额。分数是该行业成交额相对 21 个交易日前的涨幅，越放量越高。"
            "同一行业分数相同。不是 sw_daily.pe，也不是个股占行业成交额的比例。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "amtcv_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/amtcv_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线成交额。分数是过去 20 个交易日成交额变异系数取负，越稳越高。"
            "不是换手率，不是一段收益，也不是跌停次数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ps_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ps_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线市销率。分数是 ps_ttm 取负，越便宜越高。不是市盈率，也不是市净率，不要改去读 pe 或 pb。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "listed_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/listed_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读股票池的上市日。分数是决策日距离 list_date 的天数，上市越久越高。"
            "不是一段收益，不要改成新股衰减或价格动量。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "freegrow_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/freegrow_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线自由流通股本的变化。分数是 free_share 相对 60 个交易日前的涨幅取负，扩张越少越高。"
            "不是 free_share / float_share 的水平，那一条已经测完。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "indpb_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indpb_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读申万一级指数的市净率变化。分数是 pb 相对 21 个交易日前的变化取负，下降越多越高。"
            "同一行业分数相同。不是 sw_daily.pe 的水平，也不是个股 pb。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pb_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pb_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线市净率。分数是 pb 取负，越便宜越高。不是市销率，也不是市盈率，不要改去读 ps 或 pe。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pe_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pe_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线市盈率。分数是 pe_ttm 取负，越便宜越高。不是市销率，也不是市净率，不要改去读 ps 或 pb。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "room_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/room_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线涨停价。分数是 (up_limit - close) / close，离涨停越远越高。"
            "不是一段收益的反转，也不是涨停或跌停的次数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "peersize_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/peersize_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读流通市值和申万一级行业。分数是 circ_mv 除以本行业中位数再取负，比同行越小越高。"
            "不是收益，也不是自由流通占比。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psly_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psly_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线静态市销率。分数是 ps 取负，越便宜越高。不是 ps_ttm，也不是市盈率或市净率，不要改列。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "body_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/body_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线开盘、最高、最低、收盘。分数是过去 20 个交易日 |收盘−开盘|/(最高−最低) 的均值，实体占比越大越高。"
            "不是振幅大小，也不是一段收益的反转或动量。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "peerliq_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/peerliq_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读成交额和申万一级行业。分数是过去 20 个交易日成交额中位数除以本行业中位数，比同行成交越多越高。"
            "不是流通市值，也不是换手率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pvcorr_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pvcorr_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读涨跌幅和成交额。分数是过去 20 个交易日两者的相关系数，成交额越随上涨放大越高。"
            "不是收益本身，也不是换手率，不要改成一段动量或反转。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "indto_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indto_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读申万一级的成交额和流通市值。分数是最新 amount / float_mv 取负，行业成交越清淡越高。"
            "同一行业分数相同。不是成交额相对 21 日前的涨幅，也不是个股换手。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "indmv_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indmv_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读申万一级的流通市值。分数是最新 float_mv 取负，行业越小越高。"
            "同一行业分数相同。不是个股流通市值，也不是行业市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "amtac_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/amtac_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线成交额。分数是过去 20 个交易日成交额与前一交易日的相关系数，越连贯越高。"
            "不是收益，也不是换手率，不要改成成交额和涨跌幅的相关。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "amtlag_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/amtlag_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读成交额和涨跌幅。分数是当日成交额与前一交易日涨跌幅的相关系数，前一天上涨后成交额放得越大越高。"
            "不是同一天的量价相关，也不是成交额自己的自相关，不要改成一段动量或反转。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "wtstab_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/wtstab_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 1000 的指数权重。分数是过去可见一年里各月末权重的标准差取负，权重越稳越高。"
            "不满四个截面的不打分。不是最近两期权重差，不要改成按权重大小排序。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "indpbl_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indpbl_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读申万一级的市净率水平。分数是最新 pb 取负，行业越便宜越高。"
            "同一行业分数相同。不是 pb 相对 21 日前的变化，也不是个股 pb，不要改去读 pe。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "amtind_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/amtind_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读个股成交额和申万一级成交额。分数是过去 20 个交易日两者的相关系数，越一起放量越高。"
            "不是收益的相关，也不是个股占行业成交额的比例，不要改成成交额和涨跌幅的相关。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "newmem_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/newmem_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 1000 的成分截面。分数是当前成分在可见一年里出现的月末次数取负，进得越晚越高。"
            "不在最新截面里的不打分。不是权重的增减，也不要去预测还没进入指数的名字。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "new500_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/new500_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 500 的成分截面。分数是当前成分在可见一年里出现的月末次数取负，进得越晚越高。"
            "不在最新截面里的不打分。不是中证 1000，不是权重的增减，也不要去预测还没进入指数的名字。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "new300_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/new300_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读沪深 300 的成分截面。分数是当前成分在可见一年里出现的月末次数取负，进得越晚越高。"
            "不在最新截面里的不打分。不是中证 500，也不是中证 1000。不是权重的增减，也不要去预测还没进入指数的名字。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psind_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psind_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线市销率。分数是行业中位数 ps_ttm 减个股 ps_ttm，比本行业更便宜越高。"
            "不是原始市销率，不要改去读 pe 或 pb，也不要改成全市场直接按 ps_ttm 排序。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pssize_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pssize_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读日线市销率和流通市值。分数是 ps_ttm 对对数流通市值的截面残差取负，比同等市值更便宜越高。"
            "不是原始市销率，不要改去读 pe 或 pb。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "newbig_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/newbig_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 500 的成分截面和流通市值。先按进得晚排序，同一批进去的再让流通市值更大的靠前。"
            "不要改成只按市值，也不要改去中证 1000 或预测还没进入指数的名字。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psnew_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psnew_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 500 的成分截面和日线市销率。先按进得晚排序，同一批进去的再让 ps_ttm 更低的靠前。"
            "不要改成全市场直接按市销率排序，也不要改去读 pe 或 pb。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "newcy_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/newcy_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读创业板指 399006.SZ 的成分截面。先按进得晚排序，同一批进去的再让流通市值更大的靠前。"
            "不是中证 500，也不是中证 1000，不要改指数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "newbmk_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/newbmk_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数仍是中证 500 进得越晚越高，同一批进去的流通市值更大的靠前。不要改分数。"
            "benchmark_index 必须是 000905.SH，不是 000852.SH。记账基准就是这只指数本身。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "exit500_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/exit500_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 500 的成分截面。分数是刚被移出指数的股票，越近移出越高，同一批里流通市值更大的靠前。"
            "仍在指数里的不打分。不要改成去买新进成分。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "exit1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/exit1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 1000 的成分截面。分数是刚被移出指数的股票，越近移出越高，同一批里流通市值更大的靠前。"
            "仍在指数里的不打分。不要改成去买新进成分，也不要改去中证 500。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "exit300_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/exit300_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读沪深 300 的成分截面。分数是刚被移出指数的股票，越近移出越高，同一批里流通市值更大的靠前。"
            "仍在指数里的不打分。不要改成去买新进成分，也不要改去中证 500。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "exitcy_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/exitcy_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读创业板指 399006.SZ 的成分截面。分数是刚被移出指数的股票，越近移出越高，同一批里流通市值更大的靠前。"
            "仍在指数里的不打分。不要改成去买新进成分，也不要改指数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "exitbmk_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/exitbmk_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数仍是刚被移出中证 500 的股票，越近移出越高，同一批里流通市值更大的靠前。不要改分数。"
            "仍在指数里的不打分。benchmark_index 必须是 000905.SH，不是 000852.SH。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "exitwide_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/exitwide_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 500 的成分截面。分数是过去四个截面内被移出的股票，越近移出越高，同一批里流通市值更大的靠前。"
            "不是只看最近两期。仍在指数里的不打分。不要改成去买新进成分。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "midnew_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/midnew_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 500 的成分截面。分数是当前成分里出现次数最接近四个月末截面的股票，同样接近的流通市值更大的靠前。"
            "不要改成只买最新进去的一批，也不要改成买被移出的股票。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "wtheavy_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/wtheavy_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 500 的最新权重。分数就是权重，越大越高。不是权重的变化，也不是被移出的股票。"
            "benchmark_index 必须是 000905.SH。不要改去 000852.SH。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "exit50_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/exit50_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读上证 50 的成分截面。分数是刚被移出指数的股票，越近移出越高，同一批里流通市值更大的靠前。"
            "仍在指数里的不打分。不要改成去买新进成分，也不要改去中证 500。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "midbmk_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/midbmk_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数仍是中证 500 当前成分里最接近四个月末截面的股票，同样接近的流通市值更大的靠前。不要改分数。"
            "benchmark_index 必须是 000905.SH，不是 000852.SH。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "midflat_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/midflat_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只按中证 500 当前成分出现了几个月末截面打分，越接近四期越高。市值不进分数，不要改成按市值排序。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "mid1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/mid1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 1000 的成分截面。分数是当前成分里出现次数最接近四个月末截面的股票，同样接近的流通市值更大的靠前。"
            "不是中证 500，不要改指数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "mid50_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/mid50_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读上证 50 的成分截面。分数是当前成分里出现次数最接近四个月末截面的股票，同样接近的流通市值更大的靠前。"
            "买的是仍在指数里的，不要改成买刚被移出的，也不要改成中证 500 或中证 1000。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "big500_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/big500_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 500 成分打分，分数是流通市值的对数，越大越前。"
            "不要把出现了几个截面加进分数，也不要改成越接近四期越高。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "stay500_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/stay500_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 500 当前成分。分数是近几个月末截面里出现的次数，出现越多越高，同样次数的流通市值更大的靠前。"
            "不要改成越接近四期越高，也不要改成只买刚进指数的。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "midcy_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/midcy_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读创业板指的成分截面。分数是当前成分里出现次数最接近四个月末截面的股票，同样接近的流通市值更大的靠前。"
            "不是中证 500，不要改成只买刚进指数的，也不要改指数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "mid300_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/mid300_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读沪深 300 的成分截面。分数是当前成分里出现次数最接近四个月末截面的股票，同样接近的流通市值更大的靠前。"
            "不是中证 500，不要改成只买刚进指数的，也不要改指数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "earnac_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/earnac_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是同一天的 log(pe) 减去 log(pe_ttm)，只给两者都为正的股票打分。"
            "比的是静态盈利和滚动盈利，不要改成市盈率的水平，也不要改用市净率或市销率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psacc_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psacc_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是同一天的 log(ps) 减去 log(ps_ttm)，只给两者都为正的股票打分。"
            "比的是静态销售和滚动销售，不要改成市销率的水平，也不要改用市盈率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "champ_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/champ_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是当前中证 500 成分的权重，除以同行业、同在中证 500 里的权重合计。行业里不足 4 只的不打分。"
            "不要改成直接比权重大小，也不要改成成交额占比。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "champbmk_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/champbmk_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数仍是当前中证 500 成分的权重，除以同行业、同在中证 500 里的权重合计。行业里不足 4 只的不打分。不要改分数。"
            "benchmark_index 必须是 000905.SH，不是 000852.SH。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "champ1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/champ1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读中证 1000 的成分截面。分数是当前成分的权重，除以同行业、同在中证 1000 里的权重合计。行业里不足 4 只的不打分。"
            "不是中证 500，不要改指数，也不要改成直接比权重大小。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "champ300_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/champ300_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂读沪深 300 的成分截面。分数是当前成分的权重，除以同行业、同在沪深 300 里的权重合计。行业里不足 4 只的不打分。"
            "不是中证 500，不要改指数，也不要改成直接比权重大小。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "mid3bmk_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/mid3bmk_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数仍是沪深 300 当前成分里最接近四个月末截面的股票，同样接近的流通市值更大的靠前。不要改分数。"
            "benchmark_index 必须是 000300.SH，不是 000852.SH。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ch3bmk_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ch3bmk_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数仍是沪深 300 当前成分的权重，除以同行业、同在沪深 300 里的权重合计。行业里不足 4 只的不打分。不要改分数。"
            "benchmark_index 必须是 000300.SH，不是 000852.SH。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "chdelta_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/chdelta_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是中证 500 当前成分的行业内权重占比，减去上一截面的占比。上一截面没有的按 0。行业里不足 4 只的不打分。"
            "不要改成用占比的水平，也不要改成直接比权重大小。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "bigind_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/bigind_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 500 成分打分。分数是流通市值的对数减去本行业的中位数，比行业里更大的靠前。"
            "不要用指数权重，不要改成全市场，也不要改成比行业更小。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "wresid_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/wresid_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是当前中证 500 成分的指数权重，减去本行业里用流通市值对数拟合出来的部分。残差更大的靠前。行业里不足 4 只的不打分。"
            "不要改成权重本身，也不要改成市值本身。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "runner_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/runner_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂不给每个行业里权重占比最高的那只中证 500 成分打分。分数是剩下的占比，占比更高的靠前。行业里不足 4 只的整组不打分。"
            "不要把第一名加回来，也不要改成直接比权重大小。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "nlead_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/nlead_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是近六个中证 500 月末截面里，这只股票有几次是本行业权重占比最高的。次数越多越高，同样次数的当前占比更大的靠前。"
            "行业里不足 4 只的那一期不计。不要改成只用当前这一期的占比。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pond_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pond_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 500 成分打分。分数是所在申万一级里的中证 500 成分个数取负，个数更少的靠前，同样个数的流通市值更大的靠前。"
            "不要改成行业内权重占比，也不要丢掉个数少的行业。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ps500_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ps500_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 500 成分打分。分数是 ps_ttm 取负，越低越前。"
            "不要改成对行业中位数取残差，也不要改成对市值取残差，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ps1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ps1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是 ps_ttm 取负，越低越前。"
            "不要改成对行业中位数取残差，也不要改成对市值取残差，也不要改去中证 500，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "lock1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/lock1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是流通市值除以总市值，越低越前。"
            "不要改成越高越前。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psout_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psout_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给最新截面不在沪深 300、中证 500、中证 1000 里的股票打分。分数是 ps_ttm 取负，越低越前。"
            "不要改成指数里的股票，不要对行业或市值做残差，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ageout_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ageout_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给最新截面不在沪深 300、中证 500、中证 1000 里的股票打分。分数是上市越早越高。"
            "不要把市值加进分数，也不要改成越晚上市越高。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ps1ind_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ps1ind_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是本行业 ps_ttm 中位数减去自己的 ps_ttm，比行业更便宜的靠前。"
            "不要改回原始市销率，不要改去中证 500，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ps1sz_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ps1sz_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是 ps_ttm 对流通市值对数回归后的残差取负，比市值所对应的市销率更便宜的靠前。"
            "不要改回原始市销率，也不要改成对行业中位数。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ps300_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ps300_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前沪深 300 成分打分。分数是 ps_ttm 取负，越低越前。"
            "不要改成中证 1000，不要对行业或市值做残差，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psly1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psly1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是静态 ps 取负，越低越前。"
            "不要改成 ps_ttm，不要对行业或市值做残差，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pb1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pb1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是 pb 取负，越低越前。"
            "不要改成市销率，不要对行业或市值做残差，也不要改用市盈率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pe1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pe1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是 pe_ttm 取负，越低越前。"
            "不要改成市销率，不要对行业或市值做残差，也不要改用市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psly500_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psly500_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 500 成分打分。分数是静态 ps 取负，越低越前。"
            "不要改成 ps_ttm，不要改去中证 1000，不要对行业或市值做残差，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "sale1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/sale1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是流通市值除以 ps_ttm，越大越前。"
            "这是销售规模，不要改成市销率越低越前。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psgap_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psgap_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是 log(ps) 减去 log(ps_ttm)，滚动销售相对静态更强的靠前。"
            "不要改成市销率的水平。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "ssch1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/ssch1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是流通市值除以 ps_ttm 之后，占同行业这个数合计的比例，占比更高的靠前。行业里不足 4 只的不打分。"
            "不要改成市销率越低越前，也不要改成指数权重占比。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psgem_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psgem_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码以 300 或 301 开头的股票打分。分数是 ps_ttm 取负，越低越前。"
            "不要把主板加进来，不要改去创业板指，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psmain_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psmain_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数是 ps_ttm 取负，越低越前。"
            "不要把创业板加进来，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pscy_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pscy_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前创业板指成分打分。分数是 ps_ttm 取负，越低越前。"
            "不是中证 1000，不要改指数，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "sale500_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/sale500_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 500 成分打分。打分读的指数代码必须是 000905.SH，不要改成 000852.SH。"
            "分数是流通市值除以 ps_ttm，越大越前。不要改成市销率越低越前。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制，也不要再试保留带。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslag_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslag_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数用大约 21 个交易日之前的 ps_ttm，越低越前。"
            "不要改成最新一根，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psstay_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psstay_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是近六个月末截面里，有几次落在当日市销率最低的五分之一。次数越多越高，同样次数的最新 ps_ttm 更低的靠前。"
            "不要改成只用最新一期。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psxfin_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psxfin_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分里、申万一级不是银行也不是非银金融的股票打分。分数是 ps_ttm 取负，越低越前。"
            "不要把这两个行业加回来，也不要改用市盈率或市净率。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psmly_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psmly_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数是静态 ps 取负，越低越前。"
            "不要改成 ps_ttm，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslagm_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslagm_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数用大约 21 个交易日之前的 ps_ttm，越低越前。"
            "不要改成最新一根，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "salem_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/salem_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数是流通市值除以 ps_ttm，越大越前。"
            "不要改成市销率越低越前，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psl10_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psl10_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数用大约 10 个交易日之前的 ps_ttm，越低越前。"
            "不要改成 21 日，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psl42_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psl42_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数用大约 42 个交易日之前的 ps_ttm，越低越前。"
            "不要改成 21 日，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslly_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslly_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数用大约 21 个交易日之前的静态 ps，越低越前。"
            "不要改成 ps_ttm，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslage_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslage_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头、而且上市已满三年的股票打分。"
            "分数用大约 21 个交易日之前的 ps_ttm，越低越前。"
            "不要把上市不满三年的加回来，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslthen_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslthen_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂的成分用大约 21 个交易日之前那一天、或更早的最近一张中证 1000 截面，不是最新截面。"
            "代码不以 300 或 301 开头。分数是那一天的 ps_ttm，越低越前。"
            "不要改成用最新成分，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslbig_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslbig_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头、而且流通市值不低于这批中位数的股票打分。"
            "分数用大约 21 个交易日之前的 ps_ttm，越低越前。"
            "不要把较小的一半加回来，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslsml_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslsml_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头、而且流通市值低于这批中位数的股票打分。"
            "分数用大约 21 个交易日之前的 ps_ttm，越低越前。"
            "不要把较大的一半加回来，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pblagm_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pblagm_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数用大约 21 个交易日之前的 pb，越低越前。"
            "不要改成市销率，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslboth_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslboth_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给同时出现在最新中证 1000 截面、以及大约 21 个交易日之前那张截面里的股票打分。"
            "代码不以 300 或 301 开头。分数是那一天的 ps_ttm，越低越前。"
            "不要改成只用其中一张截面，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pelagm_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pelagm_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。分数用大约 21 个交易日之前的 pe_ttm，越低越前。"
            "不要改成市销率，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psltail_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psltail_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码不以 300 或 301 开头的股票打分。大约 21 个交易日之前的 ps_ttm 里，最便宜的一成不打分，其余越低越前。"
            "不要把最便宜的一成加回来，不要把创业板加进来，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psl500_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psl500_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 500 成分打分。打分读的指数代码必须是 000905.SH，不要改成 000852.SH。"
            "分数用大约 21 个交易日之前的 ps_ttm，越低越前。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psl300_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psl300_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前沪深 300 成分打分。打分读的指数代码必须是 000300.SH，不要改成 000852.SH。"
            "分数用大约 21 个交易日之前的 ps_ttm，越低越前。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "pslcy_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/pslcy_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前创业板指成分打分。打分读的指数代码必须是 399006.SZ，不要改成 000852.SH。"
            "分数用大约 21 个交易日之前的 ps_ttm，越低越前。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "psloutm_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/psloutm_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给不在最新中证 1000 截面里、而且代码不以 300 或 301 开头的股票打分。"
            "分数用大约 21 个交易日之前的 ps_ttm，越低越前。不要改成中证 1000 成分，也不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 1。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "dn20_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/dn20_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是最近大约二十个交易日里，下跌日成交额占全部成交额的比例，越高越前。"
            "不要改成上涨日。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "dn60_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/dn60_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是最近大约六十个交易日里，下跌日成交额占全部成交额的比例，越高越前。"
            "不要改成二十日，也不要改成上涨日。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "reldn_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/reldn_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是最近大约二十个交易日里，个股收益低于当天中证 1000 的那些天的成交额占比，越高越前。"
            "不要改成跑赢指数的日子。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "dn1000_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/dn1000_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是最近大约二十个交易日里，下跌日成交额占比，越高越前。"
            "不要改成全市场，也不要改成上涨日。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "beta60_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/beta60_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是最近大约六十个交易日相对中证 1000 的贝塔取负，越低越前。"
            "指数日收益是百分数，个股日收益是小数，计算时把指数收益除以 100。不要改成贝塔越高越前，也不要改这个单位。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "beta1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/beta1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是最近大约六十个交易日相对中证 1000 的贝塔取负，越低越前。"
            "指数日收益是百分数，个股日收益是小数，计算时把指数收益除以 100。不要改成全市场，不要改成贝塔越高越前，也不要改这个单位。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "age1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/age1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是上市越早越高。"
            "不要把市值加进分数，也不要改成越晚上市越高。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "mgnlag_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/mgnlag_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分里、代码不以 300 或 301 开头的股票打分。"
            "分数是大约 21 个交易日前那根 K 线上的 ps_ttm 除以 pe_ttm，利润率越高越前。两者都要为正。"
            "不要改成市销率或市盈率的水平，不要改滞后，也不要改成越低越前。打分读的指数代码必须是 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "aturn_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/aturn_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是同一根 K 线上的 pb 除以 ps，销售相对净资产越高越前。两者都要为正。"
            "不要改成市净率或市销率的水平，也不要改成越低越前。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "iroe_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/iroe_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是申万一级指数自己的 pb 除以 pe，越高越前。同一行业分数相同。"
            "不要改成市净率或市盈率的水平，也不要改成越低越前。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "delay60_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/delay60_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是今天的涨跌幅对昨天中证 1000 涨跌幅的贝塔，越高越前。"
            "不是当天的贝塔，也不是个股自己过去的涨跌。指数日收益是百分数，个股日收益是小数，计算时把指数收益除以 100。"
            "不要改成越低越前，也不要改这个单位。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "delay1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/delay1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是今天的涨跌幅对昨天中证 1000 涨跌幅的贝塔，越高越前。"
            "不是当天的贝塔，也不是个股自己过去的涨跌。指数日收益是百分数，个股日收益是小数，计算时把指数收益除以 100。"
            "不要改成全市场，不要改成越低越前，也不要改这个单位。打分读的指数代码必须是 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "nearvw_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/nearvw_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是大约二十个交易日里，收盘价相对当日成交额除以成交量的绝对偏离，越小越前。"
            "成交额是元，成交量是股，不要再换单位。不要改成有符号的偏离，也不要改成偏离越大越前。"
            "打分读的指数代码必须是 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "volag_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/volag_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是今天的成交额变化对昨天中证 1000 成交额变化的贝塔，越高越前。"
            "不是收益的贝塔，也不是和行业成交额的同期相关。不要改成越低越前。打分读的指数代码必须是 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "liqco_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/liqco_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是个股日收益与中证 1000 换手率日变化的相关系数，越高越前。"
            "用相关系数，不要改成回归贝塔。不是对指数收益的贝塔，也不是个股自己的换手率。不要改成越低越前。"
            "打分读的指数代码必须是 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "indact_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/indact_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是个股日收益与所在申万一级成交额日变化的相关系数，越高越前。"
            "用相关系数，不要改成回归贝塔。不是行业涨跌，也不是个股自己的成交额。不要改成越低越前。"
            "打分读的指数代码必须是 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "next1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/next1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前不在沪深 300、中证 500 和中证 1000 里的股票打分。分数是流通市值，越大越前。"
            "不要改成越小越前，也不要把这三个指数的成分加回去。不要把分数改成去拟合纳入公式。"
            "打分读的指数必须是 000300.SH、000905.SH 和 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "iwadd_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/iwadd_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是该股票所在申万一级在中证 1000 里的权重合计，相对大约六个月前那次截面的变化，增加越多越前。"
            "同一行业分数相同。不是个股权重，也不是成分个数。不要改成增加越少越前。"
            "打分读的指数必须是 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "nextmb_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/nextmb_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前不在沪深 300、中证 500 和中证 1000 里、且代码不以 300 或 301 开头的股票打分。"
            "分数是流通市值，越大越前。不要改成越小越前，不要把指数成分加回去，也不要加回撤控制。"
            "打分读的指数必须是 000300.SH、000905.SH 和 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "nextlo_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/nextlo_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前不在沪深 300、中证 500 和中证 1000 里、且流通市值低于当前中证 1000 最小成分的股票打分。"
            "分数仍是流通市值，越大越前。不要改成越小越前，不要把达到或超过这条下沿的名字加回来，也不要加回撤控制。"
            "打分读的指数必须是 000300.SH、000905.SH 和 000852.SH。不要再试保留带。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "roe1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/roe1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分里、代码不以 300 或 301 开头的股票打分。"
            "分数是同一根 K 线上的 pb 除以 pe，越高越前。两者都要为正。不要改成市净率或市盈率的水平，也不要改成越低越前。"
            "打分读的指数必须是 000852.SH。不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "sz1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/sz1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是流通市值，越大越前。"
            "不要改成越小越前，也不要把指数外的股票加进来。买不起一手就跳过，不要为了买进大票去改席位或加现金。"
            "打分读的指数必须是 000852.SH。不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "roeout_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/roeout_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前不在沪深 300、中证 500 和中证 1000 里的股票打分。"
            "分数是同一根 K 线上的 pb 除以 pe，越高越前。两者都要为正。不要改成市净率或市盈率的水平，也不要改成越低越前。"
            "不要把这三个指数的成分加回来。打分读的指数必须是 000300.SH、000905.SH 和 000852.SH。不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "tom1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/tom1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是过去大约一年里，月初三个交易日和月末三个交易日的日收益均值，越高越前。"
            "书仍然按月持有，不要改成只在月初月末持仓。不要改成整月收益，也不要改成越低越前。"
            "打分读的指数必须是 000852.SH。不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "at1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/at1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分里、代码不以 300 或 301 开头的股票打分。"
            "分数是同一根 K 线上的 pb 除以 ps，销售相对净资产越高越前。两者都要为正。"
            "不要改成市净率或市销率的水平，也不要改成越低越前。打分读的指数必须是 000852.SH。"
            "不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "broad_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/broad_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂分数是该股票所在申万一级里、当天可交易名字的个数，越多越前。同一行业分数相同。"
            "不是指数成分个数，也不是行业市值。不要改成越少越前。"
            "不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "atlag_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/atlag_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分里、代码不以 300 或 301 开头的股票打分。"
            "分数是大约 21 个交易日前那根 K 线上的 pb 除以 ps，销售相对净资产越高越前。两者都要为正。"
            "不要改成最新一根，不要改成市净率或市销率的水平，也不要改成越低越前。打分读的指数必须是 000852.SH。"
            "不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "iat1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/iat1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是该股票所在申万一级里、当前成分的 pb 除以 ps 的中位数，越高越前。"
            "同一行业分数相同。不是个股自己的比值，也不是市净率或市销率的水平。行业里正的比值不足 8 只就不打分。不要改成越低越前。"
            "打分读的指数必须是 000852.SH。不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "atcy_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/atcy_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 里、代码以 300 或 301 开头的股票打分。"
            "分数是同一根 K 线上的 pb 除以 ps，越高越前。两者都要为正。"
            "不要改成主板，不要改成市净率或市销率的水平，也不要改成越低越前。打分读的指数必须是 000852.SH。"
            "不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
        ),
    },
    "csk1k_alla_100k_20260927": {
        "workspace_reference": "configs/workspace_refs/csk1k_alla_100k_20260927",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        **EIGHT_YEAR,
        "pit_views_seed": DIVIDEND_PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂只给当前中证 1000 成分打分。分数是个股日收益对中证 1000 日收益平方项的回归系数取负，指数波动大时更差的排更前。"
            "不是当天的贝塔，也不是贝塔取负。指数日收益是百分数，计算时除以 100。不要改成系数越高越前，也不要改这个单位。"
            "打分读的指数必须是 000852.SH。不要再试保留带，也不要加回撤控制。"
            "每个行业最多 2 只。股票池是全部 A 股去掉北交所和科创板（688 / 689）。"
            "benchmark_index 必须是 000852.SH，initial_cash 为 10 万，席位 12，月度复核，gpu_count 为 0。"
            "对照 c_shuf 在先，主候选一条，offline_trials = 0。不要改成序列网络，也不要加回撤控制。"
            "Y1–Y4 主动均值 ≤ 0、整期主动 IR 低于 0.4544、正年少于 6/8 或主动回撤超过 0.30 即 no_edge。"
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
    closed=True,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
