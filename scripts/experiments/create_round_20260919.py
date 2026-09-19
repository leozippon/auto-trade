#!/usr/bin/env python
"""The 2026-09-19 round: books built inside a benchmark, at two account sizes.

This round exists for its dataset selection. It is the 2026-09-20 selection
plus `index_weight` — the month-end constituents and weights of the seven core
indices, which the raw lake has carried and audited since 2020 and which no
snapshot could select until now.

Why the arms need it, in one reading (`logs/notes/review_20260920/A7_data_vs_diversity.md`):
a random zero-skill 15-name book drawn from the packs' own pool reads 15.3 %/yr
tracking error against CSI 300 plus size neutralization and does not fall below
11.3 %/yr at any basket size, so the graduation bar demands 12.7–14.6 %/yr of
neutralized excess against a measured ceiling near 8.5. The same random book
drawn from the actual CSI 300 constituents reads 8.9 / 6.9 / 5.8 % at 15 / 30 /
50 names, which puts the bar inside the measured alpha band for the first time.
The lever is the universe the book is built in, never the benchmark constant.

Both arms mount one pack, `configs/workspace_refs/index_relative_20260919`, and
differ in the two things the operator decided: the account and the basket range.
At CNY 100k a constituent book is cost-feasible to about 30 names (28–43 % of
the index is unbuyable at 30 seats, minimum commission 15.5 bp a side); at CNY
1M it holds 50 at 2.6 bp a side with 3 % unbuyable, and its tracking error falls
to 5.8 %. The account is therefore not a detail of the book, it is what decides
which book exists — which is why the round runs both and compares them.

An arm is one entry of ARMS:

    "<direction>_<yyyymmdd>": {
        "workspace_reference": "configs/workspace_refs/<pack>",
        "research_directive": "<the direction, with no calendar date>",
        "initial_cash": <the account this arm runs>,
    }

`initial_cash` is an ordinary create parameter, so an arm entry overrides the
round's value for that arm alone (`_round.Round.request_params` merges the arm
last). Both arms state it rather than one inheriting BASE_OVERRIDES: the pair
exists to compare two accounts, and a silent move of the shared default would
turn the comparison into a coincidence.

Contamination, shared by both arms. Operator-side only: the pack itself carries
no reading from after research end, so none of this reaches the Agent, but it
decides how much each forward verdict is worth.

- The direction was chosen AFTER three frozen strategies failed the forward test
  that follows the research period, and after a structural tracking-error
  analysis run on research-period data. The analysis itself reads no row after
  research end — its tracking-error floor, its constituent-book readings and its
  cost arithmetic are all measured inside 2021-07..2025-06 — but the decision to
  spend two arms on it was taken with the forward failures in hand. So the
  forward verdicts of these two arms are partly a pipeline calibration of a
  direction the operator already had reason to try, not independent evidence
  that a benchmark-relative book generalises.
- The account-size comparison is an operator decision, not a finding. Nothing in
  the research-period evidence says which account is the right one to research
  on; the round runs both because the arithmetic says they are different
  problems.
- The pack's closed-family table is carried over from the open-research packs.
  Some of those closures were drawn from Fold-era probes that read rows after
  research end; the pack keeps them as priors with their research-period
  readings only, and explicitly permits reopening any of them inside the new
  universe with a pre-registration.

Exposure-budget amendment, 2026-09-19, this pack only. The absolute
`|size_tilt| <= 0.70` was calibrated on the all-market pool. Inside the CSI 300
it measures the universe rather than the book, because `size_tilt` is the
holding-weighted whole-market cap rank and the constituents ARE the largest
names in that market: on the research-year decision views an equal-weight
all-constituent book reads 0.906 / 0.922 / 0.929, the nine sandbox smoke legs
read 0.893-0.954 — both zero-skill controls included — and only a book of the
fifteen smallest constituents falls under (0.628-0.673), which is the very bet
the gate exists to stop. So it could not discriminate and would have made
nomination impossible.

The operator therefore replaced it, for this pack alone, with a same-batch
relative form: a candidate's `size_tilt` must lie within +/-0.10 of the same
batch's `c_rand` control, `benchmark.size_beta` (the loading the neutralization
divides out, Agent-visible since `eb0b19a`) is reported beside it for the
candidate and BOTH controls, and `top_industry_weight <= 0.30` and `beta <= 1.2`
are unchanged. Decided from the smoke evidence BEFORE either arm had run a
full-span batch, so no candidate reading influenced it. Every other pack keeps
the absolute gate: their universe is the whole market, where `size_tilt` really
does measure the book. The two arms already running were told by operator
message; `refs/` carries the amended text on their next resume.

The third arm exists because the operator held idle slots until this structural
change landed. It runs the same account as the second, and differs only in the
family its directive opens on, so the three seeds diverge instead of re-deriving
one starting point three times.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \\
      scripts/experiments/create_round_20260919.py <port> [--dry-run] [--fill] [experiment_id ...]
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

# Three of the four domains are the 2026-09-20 round's, imported rather than
# restated so the two selections cannot drift apart by a typo.
from scripts.experiments.create_round_20260920 import (
    EVENTS_DATASETS,
    FUNDAMENTAL_DATASETS,
    TEXT_DATASETS,
)
from scripts.experiments.create_round_20260920 import (
    MACRO_DATASETS as _MACRO_20260920,
)

# The one prebuilt view tree this round's arms hardlink from, built by
# scripts/data/prebuild_pit_views_seed.py for exactly the selection below and
# the default research geometry; gitignored, so an operator precondition.
PIT_VIEWS_SEED = "data/pit_views_seed_research_20260919"

# `index_weight` is opt-in and lands in the macro domain: one row per (index,
# constituent, month-end trade date), stamped 17:30 of that trade date, so a
# decision day reads the latest cross-section dated at or before it and a
# review inside a replay rolls in at the next pre-open.
MACRO_DATASETS = [*_MACRO_20260920, "index_weight"]

PACK = "configs/workspace_refs/index_relative_20260919"

ARMS: dict[str, dict[str, object]] = {
    "index_relative_100k_20260919": {
        "workspace_reference": PACK,
        "initial_cash": 100_000,
        "research_directive": (
            "本臂是指数内相对基准的开放方向臂：股票池就是决策日当天可见的沪深 300 成分，按时点从宏观域读出来，"
            "不得回退全市场池。账户 10 万元，篮子只数在 15 到 30 之间由你选定并写进预登记。先读 refs/README.md 与 "
            "refs/families.md——候选形态、提名硬门、已关闭家族表的用法与终止规则只以 families.md 为准；"
            "本臂不指定机制家族，已关闭表是旧股票池上的先验而不是禁令。从 refs/starter 起步（它只是能跑的基线），"
            "先 smoke_backtest，再按 refs/exploration-plan.md 做完整研究期验证。每一轮都必须同批跑两个对照："
            "同只数的等权随机成分书与本账户最大只数的等权成分书；候选要在信息比率上按预登记裕度赢过前者，"
            "否则没有技能可言。这个账户上最低佣金与买不起的成分是硬约束，第一轮之前先把这笔账算清楚。"
            "一个方向被证伪而预算还有余量时，换下一个预登记方向，不要就此收尾。"
        ),
    },
    "index_relative_1m_20260919": {
        "workspace_reference": PACK,
        "initial_cash": 1_000_000,
        "research_directive": (
            "本臂是指数内相对基准的开放方向臂：股票池就是决策日当天可见的沪深 300 成分，按时点从宏观域读出来，"
            "不得回退全市场池。账户 100 万元，篮子只数在 30 到 50 之间由你选定并写进预登记。先读 refs/README.md 与 "
            "refs/families.md——候选形态、提名硬门、已关闭家族表的用法与终止规则只以 families.md 为准；"
            "本臂不指定机制家族，已关闭表是旧股票池上的先验而不是禁令。从 refs/starter 起步（它只是能跑的基线），"
            "先 smoke_backtest，再按 refs/exploration-plan.md 做完整研究期验证。每一轮都必须同批跑两个对照："
            "同只数的等权随机成分书与本账户最大只数的等权成分书；候选要在信息比率上按预登记裕度赢过前者，"
            "否则没有技能可言。这个账户几乎买得起整个指数，所以门在分子上：更多只数会把对照这条地板一起抬高。"
            "一个方向被证伪而预算还有余量时，换下一个预登记方向，不要就此收尾。"
        ),
    },
    # Same account as the arm above, a different opening family: the two seeds
    # would otherwise start from the same composite baseline.
    "index_relative_1m_b_20260919": {
        "workspace_reference": PACK,
        "initial_cash": 1_000_000,
        "research_directive": (
            "本臂是指数内相对基准的开放方向臂，开局家族与同包另外两条臂不同：从**日频量价截面排序器**做起，"
            "在沪深 300 成分内部打分。股票池就是决策日当天可见的成分，按时点从宏观域读出来，不得回退全市场池。"
            "账户 100 万元，篮子只数在 30 到 50 之间由你选定并写进预登记。先读 refs/README.md 与 refs/families.md——"
            "提名硬门（规模界按同批对照的相对口径）、两个必跑对照与终止规则只以 families.md 为准；"
            "学习型候选的 fit 成本纪律与可照搬的参考实现见 refs/references/fit-cost.md，"
            "整期批次之前必须先在研究期靠后的槽上量一次 fit。先 smoke_backtest，再按 refs/exploration-plan.md "
            "做完整研究期验证。这一族被证伪而预算还有余量时，换下一个预登记方向，不要就此收尾。"
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
