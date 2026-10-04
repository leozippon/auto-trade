#!/usr/bin/env python
"""The 2026-10-07 round: second sessions on the two fundamentals lanes that showed something.

Round 20261005 ran the fundamentals pack in three lanes, each by the local
model and by `mimo-v2.6-flash`. Two lanes left something to build on
(docs/research-lessons.md §2; the arms' closing reasons):

- The event lane froze one book: buy when a dividend implementation
  announcement becomes visible, ranked by cash dividend ratio, sell on the
  ex-date (`fund_event_100k_8y_qwen_20261005`, active IR 1.706 at 9 trials).
  Standardised earnings surprise, guidance, accrual and disclosure-timing
  books were falsified in the two sessions. The rest of the dividend calendar
  and the other disclosure events were never tested.
- The learn lane read a full-span row above its bar on one training seed
  (1.528) and 0.975 on a second, so the session did not nominate it; the same
  learner on daily-only features read 0.756. The seed swing, not the signal,
  is what failed.

This round reuses the pack unchanged and gives each lane a second pair of
sessions whose gates already carry the first pair's search:

- `fund_event2_100k_8y_qwen` / `_mimo`: event-time books on what the first
  sessions left untested; they inherit the two event arms and the two open
  arms of round 20261005, which also ran event books.
- `fund_learn2_100k_8y_qwen` / `_mimo`: seed-robust rankers (a bag over
  training seeds, a strongly regularised linear model); they inherit the four
  earlier learned composites and the two learn arms of round 20261005.

Text-evidence scoring stays on the local model in every arm.

Usage: create_round_20261007.py <port> [--dry-run] [--fill] [experiment_id ...]
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
from scripts.experiments.create_round_20260927 import CSI1000, GATES
from scripts.experiments.create_round_20261002 import MIMO
from scripts.experiments.create_round_20261005 import (
    FUND_EIGHT_YEAR,
    FUND_PACK,
    FUND_PIT_VIEWS_SEED,
    LEARN_LINEAGE,
)

# Every arm of round 20261005 that validated an event-time book on this surface.
EVENT2_LINEAGE = [
    "fund_event_100k_8y_qwen_20261005",
    "fund_event_100k_8y_mimo_20261005",
    "fund_open_100k_8y_qwen_20261005",
    "fund_open_100k_8y_mimo_20261005",
]

LEARN2_LINEAGE = [
    *LEARN_LINEAGE,
    "fund_learn_100k_8y_qwen_20261005",
    "fund_learn_100k_8y_mimo_20261005",
]

EVENT2_DIRECTIVE = (
    "本臂走事件线的第二场：只做事件时间的书，以冻结或 no_edge 收尾，只用 CPU。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停。"
    "上一场测过的不要重走：分红实施公告可见当日买、除权日卖、按现金分红比例取前列的书已经冻结；"
    "标准化盈余惊喜、业绩预告、应计与披露时点四种事件书被证伪。它们的试验已由宿主计入本臂的门槛，不再申报。"
    "本臂做还没测过的事件：分红日历上其余的阶段（预案公告、股东大会通过、除权之后的窗口）、送转、"
    "业绩快报、审计意见，或把两种事件接成一本资金不闲置的书。"
    "事件、窗口与惊喜的定义在 hypothesis 里先登记；每本书带 c_shuf 与 c_date，事件日照 lib/fund.py 定。"
    "先数清每个复核日窗口内有几只，再 Y1..Y4 探针、再全期；平均总仓位不到 0.5 的书毕业不了。"
    "reason 里除主动读数外，报权益口径对基准的超额、年换手与费用。"
    "杀死线只结束候选，至多四批；门槛读每行的 information_ratio_bar，不自己算。"
)

LEARN2_DIRECTIVE = (
    "本臂走学习线的第二场：在 fit 里按时点向前滚动重拟合的截面排序器，特征由你从日线与财务域自造，只用 CPU；"
    "以冻结或 no_edge 收尾。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停。"
    "上一场的结论：日线加财务特征的树排序器在一个训练种子上整期主动 IR 读到 1.528，换第二个种子只有 0.975，没有提名；"
    "同一学习器只用日线特征读 0.756。失败的是种子摆幅，不是信号。"
    "所以本臂的候选必须对种子稳健：同一配置多个训练种子的分数按秩平均的袋、强正则的线性模型，或两者按秩混合。"
    "每个候选同书必跑 c_lab（训练标签打乱）与 c_daily（只用日线特征），对照用与候选同样的袋法。"
    "提名前把袋里的种子整体换一组复现：两次读数都要过该行门槛，差值不超过事先登记的容差。"
    "本臂继承了此前六条学习型臂的试验，不再申报；门槛读每行的 information_ratio_bar，不自己算。"
    "整期批次之前量一次 fit 的秒数与内存，袋的成员数乘上去仍要在限内；杀死线只结束候选，至多四批。"
)

EVENT2_ARM: dict[str, object] = {
    "workspace_reference": FUND_PACK,
    "lineage_arms": EVENT2_LINEAGE,
    "research_directive": EVENT2_DIRECTIVE,
}
LEARN2_ARM: dict[str, object] = {
    "workspace_reference": FUND_PACK,
    "lineage_arms": LEARN2_LINEAGE,
    "research_directive": LEARN2_DIRECTIVE,
}

ARMS: dict[str, dict[str, object]] = {
    "fund_event2_100k_8y_qwen_20261007": dict(EVENT2_ARM),
    "fund_event2_100k_8y_mimo_20261007": {**MIMO, **EVENT2_ARM},
    "fund_learn2_100k_8y_qwen_20261007": dict(LEARN2_ARM),
    "fund_learn2_100k_8y_mimo_20261007": {**MIMO, **LEARN2_ARM},
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=FUND_PIT_VIEWS_SEED,
    overrides={
        **FUND_EIGHT_YEAR,
        **GATES,
        "max_replay_years": 96,
        "initial_cash": 100_000,
        "benchmark_index": CSI1000,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
