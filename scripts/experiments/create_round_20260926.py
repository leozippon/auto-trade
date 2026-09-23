#!/usr/bin/env python
"""The 2026-09-26 round: breadth -- eight rule-score and measurement arms.

The dataset selection and the prebuilt seed are the 2026-09-19 round's -- the
2026-09-20 selection plus `index_weight` -- imported rather than restated, so
every arm reads exactly what the 2026-09-25 fresh-book arms read and nothing
new is mounted. Each arm also names the four lists and the seed itself, as its
pack author's pre-flight checked it. Every arm needs `index_weight`: each
universe is a decision day's visible section of the arm's own index, and the
host's zero-skill panel matches its replacements on that index's membership.
Minute data stays unmounted; `include_intraday` keeps the console default
`False`, which `check_console_defaults` pins for every round.

What this round is. After 56 closed arms the register
(docs/research-lessons.md) has closed the Alpha158 + LightGBM carrier as a
source of active return (its rank IC is real, its monthly book reads inside
the zero-skill band), the feature blocks stacked on it (an increment below
about 4 pp/yr is resolved neither offline nor in one host batch), the
construction lever (the 2026-09-25 books that followed the score added tilt,
not selection: pulling their exposures back to the pool erased the increment
on CSI 300 and CSI 500 alike) and the northbound row of the top-ten holders
union (inside the zero-skill band on CSI 300). The same round measured a third
wall: moving only the training seed moved `f1`'s active excess by 3.21 pp/yr,
about six times the 0.55 pp/yr the panel redraw moves the same book. So this
round stops moving one lever on one carrier and covers distinct families
instead (logs/notes/round_20260926/W5_breadth_menu.md): a low-range defensive
score on CSI 500, on CSI 1000 and on a 10万 CSI 1000 book; one-month abnormal
turnover on ChiNext; the seed-bagged carrier as a measurement arm;
same-calendar-month seasonality; amplitude-split momentum; and CSI 300
index-migration anticipation. Each rule-score arm runs one registered batch --
its untrained candidate plus a same-score shuffled control `c_shuf` -- against
bars written before any return-based reading: the candidate's full-span active
IR must reach 1.2816 x sqrt(244 / T), about 0.64 over four research years, with
at least 3 of 4 positive years (range20 also caps the active drawdown at 0.30),
or the arm closes `no_edge` with its readings. The seed bag runs two
single-seed books as controls next to the five- and ten-seed bags and is judged
by the freeze gate alone.

The pack authors read the offline screens on the research-end view before any
arm was created; each pack registers its verdict and the session recomputes it
in round 0 at no replay-year cost:

- `abturn_chinext`: no kill hit -- monthly 20-seat book +7.5 %/yr over the
  pool, IR 0.88, 4 of 4 years (offline levels are not trusted, register §4).
- `range20_csi500`: no kill hit -- book IR 0.71, 4 of 4 years, next to the
  random-book p90 of 0.73.
- `seed_bag`: no gate; a return-free seed census (single-seed pairs rank-
  correlate 0.893, top-50 overlap 0.743; the five-seed bag 0.955 to each seed).
- `range20_csi1000_1m`: no kill hit, but the book reads IR 0.32 and 2 of 4
  years, inside the zero-skill band.
- `range20_csi1000_100k`: its S3 kill hits -- the 12-name top slice loses to a
  same-pool random slice in two of the four years.
- `seasonality_csi500`: three of its five screens hit.
- `amp_momentum_csi500`: killed offline -- rank IC t 1.88 < 2.0 and the book
  -0.34 %/yr, 2 of 4 years.
- `index_migration_csi500`: its return-free precision check fails
  structurally. CSI 500 drops the top 300 names by cap, which is where CSI 300
  additions come from, so most additions are outside the pool when the rule
  predicts them and the top 30 covers 0.40 of the actual additions at only 1 of
  8 reviews; all four return screens hit too.

`--fill` creates the arms in file order. The four whose screens pass, or that
measure, start first. The 10万 twin follows, then the last three, whose
registered path is one batch and then `no_edge` -- unless the host bar is
cleared, and for the migration arm whatever the host reads. `no_edge` needs
one full-span validation, so each still spends one batch; queued last, their
ledger-backed closures cost the fewest slot-hours.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
        "initial_cash": <the account this arm runs>,
        "benchmark_index": <the index the host grades this arm against>,
        "gpu_count": <stated; every arm this round is CPU-only>,
        the four dataset lists and "pit_views_seed": the round's, stated,
        "max_drawdown" / "active_max_drawdown":
            stated so `--dry-run` prints the gates this arm is judged by,
    }

`_round.Round.request_params` merges the arm last. No arm carries a tracking
mandate, so none names a `tracking_error_cap`. An absolute drawdown cap below
the benchmark's own would fail a book with beta near one on drawdown alone, so
each arm's `max_drawdown` clears its benchmark's research-window maximum
drawdown (`_w5_probe_index_dd.out`): ChiNext -57.0 % -> 0.65, CSI 1000 -46.7 %
-> 0.55, CSI 500 -41.8 % -> 0.50, CSI 300 -39.6 % -> the rules' own 0.45;
`active_max_drawdown` is 0.30 everywhere. Every arm also names the statistical
bars (IR, DSR 0.975, positive-year share, full-span validations, forward
confidence, recency, activity, Held-out z) so the create request records a
choice rather than inheriting a hidden pair of packages.

Contamination, per arm. Operator-side only.

- Round-level. The breadth menu was compiled after all 56 arms had closed;
  those closes and the menu's probes are research-period readings, and the
  menu kept forward-window market knowledge out of its priors. The operator
  has read the six shared-forward discards. The menu itself is a
  program-level search on the research window that no arm's trial count
  carries.
- `range20_*` (three arms). The score was selected as the best of 18
  price-volume signals on CSI 300 constituents over the research window
  (`index_relative_1m_b_20260919`), rejected by the old DSR gate and never
  forward-tested; the operator has read the forward discard of its sibling
  `index_relative_1m_20260919`, another rule score on CSI 300. Neither pool
  here has carried the score; the 17 siblings are disclosed, not declared, and
  the CSI 300 reading enters the registered magnitude shrunk toward half.
- `abturn_chinext_1m`, `amp_momentum_csi500_1m`. Neither score has entered a
  book here and ChiNext has never been a benchmark; the CSI 500 books so far
  were carrier books. The sources' windows and constants are their own
  in-sample choices, taken as they are.
- `seed_bag_1m`. The carrier's lineage is the one whose calibration arm's
  forward verdict the operator has read; the seed-variance reading that
  motivates the arm is a research-period reading of `f1`'s second seed.
- `seasonality_csi500_1m`, `index_migration_csi500_1m`. Neither score has
  entered a book here. The migration rule is the index's published entry rule,
  and its precision check read only past index sections, not returns.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260926.py <port> [--dry-run] [--fill] [experiment_id ...]
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
# the panel's membership matching depend on.
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
CSI1000 = "000852.SH"
CHINEXT = "399006.SZ"

# Create-time graduation bars, named on every arm so the request records a
# choice. These are today's defaults; an arm that wants other bars overrides
# them the same way it overrides drawdowns.
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
    # Turnover family: one-month abnormal turnover on ChiNext constituents, 20 seats. ChiNext drew
    # down 57 % over the research years, so the arm states a 0.65 equity drawdown cap.
    "abturn_chinext_1m_20260926": {
        "workspace_reference": "configs/workspace_refs/abturn_chinext_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CHINEXT,
        "gpu_count": 0,
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.65,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：异常换手 t1——负的 20 日平均换手率除以 250 日平均换手率（Liu、Stambaugh、Yuan "
            "的一个月异常换手）——在决策日在册的创业板指成分（剔 ST、停牌、买不起）里建 20 席等额、月度复核、保留带 2.0、"
            "每个申万一级行业至多 5 只的书，由宿主零技能面板裁决，不训练。benchmark_index 必须是 399006.SZ 且与 INDEX_CODE "
            "一致；回撤上限 0.65 是因为创业板指研究期最大回撤约 57%。先按 refs/references/offline-screen.md 重算离线门 G-AT"
            "（包作者读数三条都没命中，但离线水平不可信）；唯一一批 c_shuf（control: true）+ t1（control: false），"
            "offline_trials = 0；t1 主动 IR 低于 0.64 或正年少于 3/4 即以 no_edge 携读数收尾，越过才按 refs/families.md 的"
            "正常路径走。"
        ),
    },
    # Defensive family on the middle pool of the CSI 300 -> 500 -> 1000 curve: the untrained
    # range20 score, 50 seats, monthly, keep band 100, at most 14 names per SW L1.
    "range20_csi500_1m_20260926": {
        "workspace_reference": "configs/workspace_refs/range20_csi500_1m_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂，也是换池曲线沪深 300 → 中证 500 → 中证 1000 的中间一点：range20"
            "（每只股票自己最近 20 根日线的 (最高−最低)/昨收 均值取负）在决策日在册的中证 500 成分里建 50 席等额、"
            "月度复核、保留带前 100 名、每个申万一级至多 14 只的书，不训练，由宿主零技能面板裁决。"
            "benchmark_index 必须是 000905.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL 一致，否则停；"
            "批次里不许有别的指数的腿。按 refs/references/offline-screen.md 重算离线门 G-R20（只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ r1（control: false），offline_trials = 0"
            "（分数是沪深 300 上 18 选 1，兄弟不申报；另筛的变体各加 1）；r1 主动 IR 低于 0.64、正年少于 3 "
            "或主动回撤超过 0.30 即 no_edge；越过才跑登记的邻点 r2，按 refs/families.md 提名。"
        ),
    },
    # Measurement: the carrier's monthly CSI 300 book with only the training seed moved -- two
    # single-seed controls and the five- and ten-seed rank-averaged bags, judged by the freeze gate.
    "seed_bag_1m_20260926": {
        "workspace_reference": "configs/workspace_refs/seed_bag_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI300,
        "gpu_count": 0,
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
        "pit_views_seed": PIT_VIEWS_SEED,  # data/pit_views_seed_research_20260919
        "max_drawdown": 0.45,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是量测臂，只动训练种子：载体 Alpha158 + LightGBM 的特征、标签、参数点、季度续训与月度 50 席书"
            "（每次最多换 8 只、保留带 2.0）一字不改。第 0 轮先冒烟量 fit 秒数，再按 refs/references/seed-census.md "
            "做种子普查（不读收益、不申报）。第 1 批 c_s7（种子 7，即模板书，control: true，第一条）+ c_s8（种子 8，"
            "control: true）+ b5（种子 7–11 五个 booster 的等权秩平均，control: false）+ b10（十个种子，冷启动 fit "
            "≤ 600 秒才进，control: false），offline_trials 0。H-bag：平均种子不改主动收益的期望、只收窄散布；"
            "b5 与 b10 都没过冻结门（主动 IR 到 selection_statistics 的门槛、DSR ≥ 0.975、≥ 3/4 年为正）就 no_edge，"
            "携 R1–R4 收尾；过了先跑不相交种子孪生，孪生也过才提名。不得挑种子或给成员加权。"
            "全臂 host trial ≤ 3、回放 ≤ 20 年。"
        ),
    },
    # Defensive family on the widest pool: the same range20 book on CSI 1000 constituents.
    "range20_csi1000_1m_20260926": {
        "workspace_reference": "configs/workspace_refs/range20_csi1000_1m_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：range20（每只股票自己最近 20 根日线的 (最高−最低)/昨收 均值取负，越安静越前）"
            "在决策日在册的中证 1000 成分里建 50 席等额、月度复核、保留带前 100 名、每个申万一级至多 14 只的书，"
            "不训练、不叠载体，由宿主零技能面板裁决。先对齐运行事实：benchmark_index 必须是 000852.SH、"
            "initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL 一致，否则停。"
            "按 refs/references/offline-screen.md 重算离线门 G-R20（S1–S5，只报不改登记）；"
            "唯一一批 c_shuf（control: true，第一条）+ r1（control: false），offline_trials = 0"
            "（分数是沪深 300 上 18 选 1，那 17 个兄弟不申报；会话另筛的变体各加 1）；"
            "r1 主动 IR 低于 0.64、正年少于 3 或主动回撤超过 0.30 即 no_edge；"
            "越过才跑登记的邻点 r2（60 根），按 refs/families.md 提名。"
        ),
    },
    # The CSI 1000 arm's account twin in the Paper shape: 10万, 12 seats, keep band 48, at most
    # 3 names per SW L1; its registered screen already hits S3.
    "range20_csi1000_100k_20260926": {
        "workspace_reference": "configs/workspace_refs/range20_csi1000_100k_20260926",
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "gpu_count": 0,
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是 100 万中证 1000 那一条的账户孪生：同一个不训练的 range20 分数、同一个池（决策日在册的中证 1000 成分），"
            "只换成 Paper 账户的形状——10 万、12 席等额、月度复核、保留带前 48 名、每个申万一级至多 3 只，"
            "由宿主零技能面板裁决。先对齐运行事实：benchmark_index 必须是 000852.SH、initial_cash 为 10 万，"
            "与 knobs 的 INDEX / CAPITAL 一致，否则停。按 refs/references/offline-screen.md 重算离线门 G-R20"
            "（包作者的读数已命中 S3，只报不改登记）；唯一一批 c_shuf（control: true，第一条）+ r1（control: false），"
            "offline_trials = 0（会话另筛的变体各加 1）；r1 主动 IR 低于 0.64、正年少于 3 或主动回撤超过 0.30 即 no_edge；"
            "年换手（完整期 turnover ÷ 4）超过 12 倍不可提名；越过才跑登记的邻点 r2，按 refs/families.md 提名。"
        ),
    },
    # Calendar: same-calendar-month return seasonality (Heston-Sadka, lags 1-4) as an untrained
    # score inside CSI 500, re-ranked monthly; one batch, then no_edge unless the host bar is cleared.
    "seasonality_csi500_1m_20260926": {
        "workspace_reference": "configs/workspace_refs/seasonality_csi500_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的臂：同月收益季节性（中证 500 成分过去第 1–4 年同一日历月相对池等权超额的均值，至少 3 个滞后）"
            "作不训练的分数 s1，持有前 50 只，等额、每月整本重排、空座次日补；c_shuf（同分数日内打乱）登记 control: true、"
            "永不可提名。benchmark_index 必须是 000905.SH 且与 knobs.INDEX 一致。先按 refs/references/offline-screen.md "
            "重算会话能复核的 23 个月初（/mnt/snapshot 只有这一段有 ≥ 3 个滞后）；包作者的离线筛 Z1、Z2、Z4 已命中。"
            "第一批 c_shuf + s1、span full、offline_trials = 0；s1 主动 IR < 0.50、正年少于 3/4 或主动回撤 > 0.30 即 no_edge；"
            "过了也因筛命中不开变体批，只有冻结门能让它继续。年换手约 20 倍、F5 多扣约 1 pp/年，须实测并报。不改登记去追读数。"
        ),
    },
    # Momentum family: the amplitude-split momentum rule score on CSI 500 constituents, 50 seats,
    # with plain 160-session momentum as a second control. No learner; the 2026-09-25 selection.
    "amp_momentum_csi500_1m_20260926": {
        "workspace_reference": "configs/workspace_refs/amp_momentum_csi500_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的规则分数臂：振幅切割动量 a1——过去 160 个交易日里振幅（最高−最低）/昨收最低的七成交易日的"
            "日收益均值，一份卖方报告的样本内读数——在决策日在册的中证 500 成分（剔科创板、ST、停牌、买不起）里建 50 席等额、"
            "月度复核、保留带 2.0、按行业配额补座位的书，由宿主零技能面板裁决，不训练。benchmark_index 必须是 000905.SH "
            "且与 INDEX_CODE 一致。先按 refs/references/offline-screen.md 用 /mnt/tools/screen.py 重算离线门 G-AM"
            "（包作者读数命中 K2 与 K3）；不论读数，唯一一批 c_shuf（control: true）+ a1（control: false）+ c_mom"
            "（普通 160 日动量，control: true），offline_trials = 0；a1 主动 IR 低于 0.64 或正年少于 3/4 即以 no_edge "
            "携读数收尾，越过才按 refs/families.md 的正常路径走，不得改登记去追读数。"
        ),
    },
    # Events: the public CSI 300 entry rule as an untrained score inside CSI 500. Its registered
    # precision check (K0) already failed at 7 of 8 reviews, structurally, so the arm runs the one
    # full-span batch no_edge needs and closes.
    "index_migration_csi500_1m_20260926": {
        "workspace_reference": "configs/workspace_refs/index_migration_csi500_20260926",
        "initial_cash": 1_000_000,
        "benchmark_index": CSI500,
        "gpu_count": 0,
        "fundamental_datasets": FUNDAMENTAL_DATASETS,
        "macro_datasets": MACRO_DATASETS,
        "events_datasets": EVENTS_DATASETS,
        "text_datasets": TEXT_DATASETS,
        "pit_views_seed": PIT_VIEWS_SEED,
        "max_drawdown": 0.50,
        "active_max_drawdown": 0.30,
        **GATES,
        "research_directive": (
            "本臂是固定方向的臂：沪深 300 的公开纳入规则（过去一年日均成交额前五成、在位者前六成，按日均总市值排名）"
            "作中证 500 内不训练的分数 m1，持有最接近纳入的 30 只，等额、月度复核、保留带 60 名、空座次日补；"
            "c_shuf（同分数日内打乱）登记 control: true、永不可提名。benchmark_index 必须是 000905.SH 且与 knobs.INDEX 一致。"
            "先按 refs/references/offline-screen.md 重算精度检查 K0（前 30 每次覆盖实际纳入 ≥ 0.40）与离线筛 S1–S4；"
            "包作者的 K0 已失败（七次低于 0.40，纳入者大多预判日不在中证 500），重算一致就按 refs/families.md："
            "唯一一批 c_shuf + m1、span full、offline_trials = 0，读数写全后以 no_edge 收尾，不论宿主读数。"
            "绝不读日期不早于决策日的成分截面；不开登记外的批次，不改登记去追读数。"
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
