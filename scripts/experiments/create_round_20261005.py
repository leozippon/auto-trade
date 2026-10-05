#!/usr/bin/env python
"""The 2026-10-05 round: the fundamentals domain on the eight-year 100k surface.

Nine open sessions on the eight-year 100k pack ended without a deliverable and
kept returning to the same three ideas -- sales scale, index-removal sets and
negative-tail exclusion screens -- because every eight-year arm could read
only daily bars, valuation columns and index series
(logs/notes/round_20261005/DIGEST_finished_arms.md). This round mounts the
fundamentals domain -- statements, indicators, earnings guidance and flash
reports, disclosure dates, dividend events, audit opinions and main-business
segments -- over the same eight research years, on a seed built from the
release the dividend seed pins (logs/notes/round_20261005/DESIGN.md).

One pack serves three lanes and the directive picks the lane:

- `fund_open`: families of the session's choosing on the widened surface, at
  least one of them built on the fundamentals domain.
- `fund_event`: event-time books only, opened after a disclosure becomes
  visible and held for a registered window, against the same book on shifted
  or shuffled event dates.
- `fund_learn`: a linear or tree cross-sectional ranker refitted walk-forward
  inside `fit`, against the same learner on shuffled labels and on daily-only
  features; it inherits the search of the earlier learned composites on the
  same period and account.

Each lane is run at once by the local model and by `mimo-v2.6-flash`, in
alternating fill order so a pair starts together; text-evidence scoring stays
local. The open and event lanes carry no lineage: a freeze from any of them is
read against the pooled trials of the sibling arms before it is treated as a
result.

Usage: create_round_20261005.py <port> [--dry-run] [--fill] [experiment_id ...]
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

# The pack, the fundamentals geometry and the round-level parameters this round
# first stated are shared with the rounds after it.
from scripts.experiments._profiles import FUND_EIGHT_YEAR_100K, FUND_PACK, MIMO
from scripts.experiments._round import Round

# Prebuilt for exactly the fundamentals geometry and selection (FUND_EIGHT_YEAR).
FUND_PIT_VIEWS_SEED = "data/pit_views_seed_research_8y_fund_20261005"

# Every arm that recorded a non-control trial of a learned or sign-fitted
# composite on cross-sectional features at 100k on CSI 1000 over these years.
LEARN_LINEAGE = [
    "census_gbdt_csi1000_100k_8y_20261001",
    "census_lin_csi1000_100k_8y_qwen_20261002",
    "census_lin_csi1000_100k_8y_mimo_20261002",
    "census_lin_csi1000_100k_8y_deepseek_20261004",
]

OPEN_DIRECTIVE = (
    "本臂走开放线：在 10 万元账户上、从挂上财务域的八年数据面里自己提出家族，成批预登记并检验，"
    "以冻结或 no_edge 收尾；只用 CPU，不做逐股序列网络。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停。"
    "至少两个结构不同的家族，其中至少一个建在财务域上；上一版九次会话收敛过的销售规模、指数移出与负端剔除筛不要重走。"
    "每个分数带自己的 c_shuf，主张落在构造、筛子或事件时点上再带随机化的安慰剂，事件书与学习型排序器带 refs/README.md 规定的对照；"
    "先 Y1..Y4 探针、再全期；全期胜过自己对照的家族至少再细化一步。"
    "杀死线只结束候选或家族，不结束本臂；至多四批，登记过的轴没测完就收尾要在 reason 里说明。"
    "重开 refs/families.md 里的家族照 README「什么算重复」申报 offline_trials。"
)

EVENT_DIRECTIVE = (
    "本臂走事件线：只做事件时间的书——财务域里一条披露事件（定期报告、业绩预告、业绩快报、分红预案）可见之后才开仓，"
    "持有登记的窗口后平仓；以冻结或 no_edge 收尾，只用 CPU。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停。"
    "事件、窗口与惊喜的定义在 hypothesis 里先登记，惊喜对股票自己的历史量；"
    "每本书带 c_shuf 与 c_date（同一本书换到平移或打乱的事件日上），事件日照 lib/fund.py 定。"
    "先数清每个复核日窗口内有几只，再 Y1..Y4 探针、再全期；平均总仓位不到 0.5 的书毕业不了。"
    "杀死线只结束候选，至多四批；重开 refs/families.md 里的家族照 README 申报 offline_trials。"
)

LEARN_DIRECTIVE = (
    "本臂走学习线：一个在 fit 里按时点向前滚动重拟合的截面排序器（线性或树，只用 CPU），"
    "特征由你从日线与财务域自造；以冻结或 no_edge 收尾。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停。"
    "每个候选同书必跑 c_lab（同一学习器、训练标签打乱）与 c_daily（同一学习器只用日线特征），"
    "主张是财务特征带来的增量，增量线先登记。"
    "本臂继承了此前四条学习型合成臂的试验，不再申报它们；门槛读每行的 information_ratio_bar，不自己算。"
    "整期批次之前量一次 fit 的秒数与内存；提名前换第二个训练种子复现；杀死线只结束候选，至多四批。"
)

OPEN_ARM: dict[str, object] = {"workspace_reference": FUND_PACK, "research_directive": OPEN_DIRECTIVE}
EVENT_ARM: dict[str, object] = {"workspace_reference": FUND_PACK, "research_directive": EVENT_DIRECTIVE}
LEARN_ARM: dict[str, object] = {
    "workspace_reference": FUND_PACK,
    "lineage_arms": LEARN_LINEAGE,
    "research_directive": LEARN_DIRECTIVE,
}

ARMS: dict[str, dict[str, object]] = {
    "fund_open_100k_8y_qwen_20261005": dict(OPEN_ARM),
    "fund_open_100k_8y_mimo_20261005": {**MIMO, **OPEN_ARM},
    "fund_event_100k_8y_qwen_20261005": dict(EVENT_ARM),
    "fund_event_100k_8y_mimo_20261005": {**MIMO, **EVENT_ARM},
    "fund_learn_100k_8y_qwen_20261005": dict(LEARN_ARM),
    "fund_learn_100k_8y_mimo_20261005": {**MIMO, **LEARN_ARM},
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=FUND_PIT_VIEWS_SEED,
    overrides=FUND_EIGHT_YEAR_100K,
    closed=True,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
