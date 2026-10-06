#!/usr/bin/env python
"""The 2026-10-12 round: the drift after an equity-incentive plan draft, at 100k and at 500k.

A restricted-stock or option plan draft ties management's pay to public
multi-year targets, and its grant price is set off the price just before the
announcement, so insiders launch when they think the price is low. On the
research window the first such draft after 90 quiet days drifts +1.7 % over
40 trading days against names of the same float-cap quintile (2020-07..2025-06
titles, monthly t 3.4), +2.8 % on 2017-07..2019-12 titles the screen never
saw, and nothing when the same events are entered 120 or 250 trading days
late. Crude seat-limited eight-year books read an active IR of 0.75-0.96 at
12-20 seats and 1.21 / 1.27 at 30 / 40, 7 of 8 years each, with a market
loading near 1.08 (logs/notes/round_20261012/DESIGN.md §4,
DESIGN_2_expanded_data.md §3, §7). It is the one stock family with offline
support in some fifty long-side screens, and no arm has read announcement
titles before: the eight-year seeds carried an empty text domain.

One pack (configs/workspace_refs/incdraft_alla_8y_20261012, and its 500k
sibling below) runs the memo's book unchanged as the baseline: the drafts
that became visible since the previous weekly review fill empty equal-cash
seats oldest first, with no ranking among them, and each is sold at the
first review HOLD trading days after it was bought. The registered axes are seats and hold (40 or 60 trading
days). The falsifying control is the memo's late-entry placebo `c_late`, the
same events entered 120 trading days late on the same seats and hold; the
placebo in which as many names enter on the same days from a matched pool is
the host's zero-skill panel itself, which every row is graded against, so it
is not run again. The claim is paired with the same-shape `c_late`: plain
selection higher on both halves of the research window and on its last two
years, and an active IR at least 0.25 higher. The pack points to the arm's
facts for every gate condition and waives none. No `fit`, no seed
replicates, CPU only.

The 100k pair is the primary focus: baseline `b16`, seats 16 / 20 / 24 / 30,
sell 09:30 and buy 15:00; batch 1 runs the four seat legs and `c_late` at 16
and 30 seats (48 replay-years), batch 2 the finalist's 60-day hold with its
controls (16-24). The 500k arm, which the owner allowed beside it because
breadth is the lever, runs the same book with `ACCOUNT` and `SEATS` set at
round 0: baseline `b40`, seats 40 / 30, both legs in the closing auction and
a CNY 40m median-daily-amount floor, the second memo's capacity advice;
batch 1 is the two seat legs and their `c_late` (32), batch 2 the 60-day
hold with its control (16). The owner confirmed that the 500k account holds
the STAR permission, so its pool includes STAR, fixed by the account and not
an axis; the 100k account has none. The pool was hard-coded in the starter
and the running 100k pair mounts that pack, so the 500k arm mounts a sibling
(configs/workspace_refs/incdraft_alla_500k_8y_20261012) that differs only in
the pool: STAR names are bought at 500k at the exchange's 200-share minimum,
then in whole shares. STAR adds 744 events to the 3,694 (Y3..Y8), about 250
of them affordable at a 40-seat ticket and above the floor. The floor stays
one rule for every board: at the same daily amount a STAR closing auction is
about a quarter thinner, and a 12-16k ticket is still at most 5 % of the
thinnest tenth's. Each arm declares the designer's screens that shaped what
it submits and that it never submits in its first batch's `offline_trials`:
17 at 100k, 20 at 500k, where the STAR-draft screen now shapes the pool (the
packs' README §6 lists them and why the rest are left out).

No lineage: a new family on a surface no arm has read. A later arm that
continues it on the same research period -- the drafts on another account
or construction, other commitment categories from titles, a title ranker
trained in `fit`, a book with an event sleeve -- names these three as
`lineage_arms`. Until then the three run side by side and none enters the
others' gate; a freeze from any one is re-read against the others' pooled
trials before it is taken as a result.

The seed is `TITLE_PIT_VIEWS_SEED`: the vendor's announcement titles from
2016-01 (official service to 2019-12, the relay copy after), daily, universe
and the four macro tables. Text-evidence scoring stays on the local model;
the book reads titles itself and makes no evidence call.

Fill order (ids end in `_20261012`): `incdraft_100k_8y_qwen`,
`incdraft_100k_8y_mimo`, `incdraft_500k_8y_qwen`.

Usage: create_round_20261012.py <port> [--dry-run] [--fill] [experiment_id ...]
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

from scripts.experiments._profiles import (
    MIMO,
    TITLE_EIGHT_YEAR_100K,
    TITLE_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

INCDRAFT_PACK = "configs/workspace_refs/incdraft_alla_8y_20261012"
# The same pack with STAR in the 500k pool.
INCDRAFT_500K_PACK = "configs/workspace_refs/incdraft_alla_500k_8y_20261012"

# Said once for both accounts; each directive adds its account's part.
COMMON = (
    "本臂走股权激励计划草案之后的漂移，refs/README.md 是合同，refs/families.md 是已关的近邻。"
    "事件是公告标题里一家公司相隔九十天以上第一次出现的限制性股票或股票期权激励计划草案；"
    "书在每个 ISO 周的第一个决策日把上次复核以来可见的新事件按先后填进空席，等额、一手可买，"
    "持满登记的交易日数后在复核日卖出；不在事件之间排序，不训练、没有种子、只用 CPU。"
    "对照是晚进场安慰剂 c_late：同一批事件晚 120 个交易日进场、同席位同持有，登记为对照；"
    "零技能面板本身就是同日、同数、同额的随机名安慰剂，不另跑。"
    "主张与同形 c_late 配对：朴素选名（每个研究年书的收益减同年零技能面板的收益）"
    "在 Y1..Y4、Y5..Y8 与 Y7..Y8 的平均都更高，主动 IR 至少高 0.25。"
    "批次计划、决赛者规则与杀死线照 refs/README.md；门槛读每行的 information_ratio_bar，不自己算；"
    "冻结门与毕业条件只看运行事实。每一行都报主动 IR、正年、主动回撤、逐年朴素选名、"
    "市场载荷、平均总仓位与权益口径的扣费年化收益。"
)

DIRECTIVE_100K = (
    COMMON
    + "本臂是 10 万账户：第 0 轮核对 broker_replay.initial_cash 为 10 万、benchmark_index 为 000852.SH、"
    "文本域可用，否则停；knobs 的 ACCOUNT 保持不动。登记的轴只有席位（16、20、24、30，基线 b16）"
    "与持有期（40、60 个交易日）；09:30 卖、15:00 买。第一批 offline_trials 报 17，之后报 0。"
)

DIRECTIVE_500K = (
    COMMON
    + "本臂是 50 万账户：第 0 轮核对 broker_replay.initial_cash 为 50 万、benchmark_index 为 000852.SH、"
    "文本域可用，否则停；把 knobs 的 ACCOUNT 改成 500_000、SEATS 改成 40（基线 b40）。"
    "登记的轴只有席位（40、30）与持有期（40、60 个交易日）；两条腿都在 15:00 收盘竞价，"
    "只买近 20 个交易日成交额中位数不低于 4000 万元的名字，这两条随账户固定、不是轴；"
    "账户有科创板权限，池含科创板（200 股起），也随账户固定、不是轴。"
    "成本压力另报 breakeven_extra_slippage_bps 够不够 10 bp。"
    "第一批 offline_trials 报 20，之后报 0。"
)

ARMS: dict[str, dict[str, object]] = {
    "incdraft_100k_8y_qwen_20261012": {
        "workspace_reference": INCDRAFT_PACK,
        "research_directive": DIRECTIVE_100K,
    },
    "incdraft_100k_8y_mimo_20261012": {
        **MIMO,
        "workspace_reference": INCDRAFT_PACK,
        "research_directive": DIRECTIVE_100K,
    },
    "incdraft_500k_8y_qwen_20261012": {
        "workspace_reference": INCDRAFT_500K_PACK,
        "research_directive": DIRECTIVE_500K,
        "initial_cash": 500_000,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=TITLE_PIT_VIEWS_SEED,
    overrides=TITLE_EIGHT_YEAR_100K,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
