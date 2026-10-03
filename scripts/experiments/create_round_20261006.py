#!/usr/bin/env python
"""The 2026-10-06 round: the two untested axes of the 100k sequence bag -- its label and its inputs.

The rank-average bag of four sequence heads on the whole-pool recipe is the one
learned recipe whose research-period edge the register attributes to its
inputs, label and book rather than to its head. Rebuilt as a 100k book, it
froze on 12 seats, at most 4 swaps a weekly review and no residualisation
(`seqbag_alla_100k_8y_qwen_20261003`, logs/notes/round_20261006/DESIGN.md).
Two of the three attributed axes have never been varied: the label is the
10-day beta residual every recipe arm used, and the inputs are daily bars
alone. This round holds the model and that book fixed as a baseline,
`c_base`, and changes one of them. The same book read 0.4 active IR apart on
two seed bases, so an increment counts only when it holds against `c_base`
on both.

One pack serves two lanes and the directive picks the lane:

- `seqlabel`: the training label only -- the pure benchmark residual, the
  residual demeaned within size quintiles, and the horizon aligned to the
  book's holding period -- each against `c_base` on the same seeds and book.
- `seqfund`: point-in-time fundamentals added to the baseline in two
  registered forms, a late rank blend with a fundamentals tree fitted inside
  `fit` and an early join to every head's last layer, each against `c_base`
  and against the same model on fundamentals permuted across stocks within
  each date (`c_perm`).

Both lanes run on the fundamentals seed, so the label lane's `c_base` is the
fundamentals lane's byte for byte; the seed's daily, macro, universe and
corporate-action tables are the dividend seed's, so the baseline also reads
as the 100k bag arms' book does. Each lane is run by the local model and by
`mimo-v2.6-flash`, on one GPU, inheriting the recipe arms, the 1m bag and the
three 100k bag arms as lineage. The queue alternates by lane, in fill order
(ids end in `_20261006`): `seqlabel_alla_100k_8y_qwen`, `seqfund_alla_100k_8y_qwen`,
`seqlabel_alla_100k_8y_mimo`, `seqfund_alla_100k_8y_mimo`.

Usage: create_round_20261006.py <port> [--dry-run] [--fill] [experiment_id ...]
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
from scripts.experiments.create_round_20261003 import SEQBAG_LINEAGE
from scripts.experiments.create_round_20261005 import FUND_EIGHT_YEAR, FUND_PIT_VIEWS_SEED

SEQAXES_PACK = "configs/workspace_refs/seqaxes_alla_100k_8y_20261006"

# The recipe arms and the 1m bag, plus every arm that rebuilt the bag as a
# 100k book: their trials are this search's own history on these eight years.
SEQAXES_LINEAGE = [
    *SEQBAG_LINEAGE,
    "seqbag_alla_100k_8y_qwen_20261003",
    "seqbag_alla_100k_8y_mimo_20261003",
    "seqbag_alla_100k_8y_deepseek_20261004",
]

LABEL_DIRECTIVE = (
    "本臂走标签线：模型、输入与书都是 10 万序列袋的基线 c_base（12 席、每次至多换 4 只、每周复核、不残差化），"
    "只开训练标签一根轴——纯基准残差 bench、按规模五分组去均值 size、与书的持有期对齐的 horizon hold，每个变体是一个试验。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；"
    "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。"
    "每一批都带同 span、同种子基数、同书的 c_base；增量主张要在两个种子基数上都成立："
    "每个种子基数上候选减同种子的 c_base 整期主动 IR 都为正，两个差平均至少 0.25；"
    "回放年够时，提名前再给决赛者与 c_base 换第三个种子基数。"
    "主张不成立的行即使过了门也不提名、以 no_edge 收尾——c_base 自己已经过门并冻结。"
    "门槛读每行的 information_ratio_bar，不自己算；第一批 offline_trials = 0；杀死线只结束候选，至多四批；"
    "批次计划、杀死线与收尾 reason 要写的读数照 refs/README.md。"
)

FUND_DIRECTIVE = (
    "本臂走财务线：模型与书是 10 万序列袋的基线 c_base（12 席、每次至多换 4 只、每周复核、不残差化），"
    "只加按时点的财务信息，两种登记形式各是一个试验——"
    "晚融合 fund_late（袋的分数与 fit 里向前滚动拟合的财务树按秩混合）与早融合 fund_early（财务特征接到每个头的最后一层）。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；"
    "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。"
    "每一批都带同 span、同种子基数的 c_base 与同一形式的 c_perm（财务特征在每个日期内跨股票打乱）；"
    "增量主张要在两个种子基数上都成立：每个种子基数上候选减同种子的 c_base、减同种子的 c_perm 整期主动 IR 都为正，"
    "两个种子基数的差平均都至少 0.25；回放年够时，提名前再给决赛者与 c_base 换第三个种子基数。"
    "主张不成立的行即使过了门也不提名、以 no_edge 收尾——c_base 自己已经过门并冻结。"
    "财务数据的事实读运行记忆 research-8y-data-surface 的财务域一节，不要再派子代理重推；"
    "门槛读每行的 information_ratio_bar，不自己算；第一批 offline_trials = 0；杀死线只结束候选，至多四批；其余照 refs/README.md。"
)

LABEL_ARM: dict[str, object] = {
    "workspace_reference": SEQAXES_PACK,
    "gpu_count": 1,
    "lineage_arms": SEQAXES_LINEAGE,
    "research_directive": LABEL_DIRECTIVE,
}
FUND_ARM: dict[str, object] = {**LABEL_ARM, "research_directive": FUND_DIRECTIVE}

ARMS: dict[str, dict[str, object]] = {
    "seqlabel_alla_100k_8y_qwen_20261006": dict(LABEL_ARM),
    "seqfund_alla_100k_8y_qwen_20261006": dict(FUND_ARM),
    "seqlabel_alla_100k_8y_mimo_20261006": {**MIMO, **LABEL_ARM},
    "seqfund_alla_100k_8y_mimo_20261006": {**MIMO, **FUND_ARM},
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
