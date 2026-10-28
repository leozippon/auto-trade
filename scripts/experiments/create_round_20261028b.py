#!/usr/bin/env python
"""The 2026-10-28 b round: round 20261028's gemini-4-argon arm again, after the first one failed on the relay.

`ttt_100k_8y_argon_zgca_20261028` failed at 05:54 UTC on 10-28 after eight
relay answers of `HTTP 403: This API method requires billing to be enabled`
for gemini-4-argon with both keys, an outage on the relay's side. This arm
relaunches it when the relay answers again: everything it sends is that arm's
-- the pack `ttt_100k_8y_20261028`, the directive word for word, the relay's
alternate key, DeepSeek-V4.1-Flash sub-agents and compaction with the
threshold pinned at 221,184, the 10-24 seed and geometry, one A100, four
replays per batch, three times round 20261027's budgets and a strategy
container of 16 CPUs and 64 GiB with 10,800 s per `fit` and 1,080 s per
`generate_orders` -- except its id. A round file never imports another
(tests/unit/test_round_arms.py), so this definition is a copy, held equal to
round 20261028's argon arm by tests/unit/test_ttt_pack.py.

| Arm | Main session | Sub-agents / compaction |
| --- | --- | --- |
| `ttt_100k_8y_argon_zgca_20261028b` | gemini-4-argon | DeepSeek-V4.1-Flash |

The console refuses to create a GPU arm while too few cards are free
(`ExperimentManager._claimable_gpus`), so queue it with `--fill`, which creates
it once a card is free:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261028b.py --fill

`--dry-run` reads the seed contract and the template's smoke record and
creates nothing.
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

from scripts.experiments._profiles import FUND_TITLE_EIGHT_YEAR_100K, FUND_TITLE_PIT_VIEWS_SEED
from scripts.experiments._round import Round

PACK = "configs/workspace_refs/ttt_100k_8y_20261028"
ARGON = "gemini-4-argon"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

# The pack README's 「方向」 paragraph word for word, round 20261027's call to
# explore with its retrieval-specific objects replaced by this direction's,
# and one sentence on the record of closed families
# (round 20261028's, which tests/unit/test_ttt_pack.py holds equal; a round
# file never imports another).
DIRECTIVE = (
    "在八年研究期上研究测试时训练：在历史上拟合的模型，面对的是分布不断漂移的市场；"
    "测试时训练指在推断时，用此刻可见的信息更新模型的一部分，形式可以是当前窗口上的自监督目标、"
    "快权重的记忆层、在近邻样本上微调、熵或一致性目标、上下文里的适应，或会话自己想出的别的做法。"
    "更新只能用 `available_at <= inference_at` 的记录；最近的样本要等预测期限走完，标签才可见，"
    "所以已可见日子上的自监督目标是自然的选择。"
    "更新可以放在每个决策日的 `generate_orders` 里、跨重拟合的 `fit` 里，或模型自己的前向计算里，"
    "都要守住运行事实给的单次调用时限。"
    "之后评判的主要问题是：测试时训练能不能让模型在分布外的时期学得更好、适应得更好，"
    "胜过不做测试时训练的同一个模型。"
    "读法有两层：一是分布发生了漂移的时期，比如风格在大小盘之间切换、波动与成交骤变、交易规则改变，"
    "会话可以在可见的历史里自己找出这样的时期；二是会话自己登记的配对，"
    "最自然的是同一架构在相同种子上开与关测试时更新，再加一个滚动重拟合的基线，换手相同的配对读得最干净。"
    "怎样做、是否以及怎样用到任何预训练模型或语言模型，都由会话决定。"
    "在这个方向之内要多探索：试几种不同的测试时目标、更新的位置与更新的方式，"
    "不要在第一个出现的信号上过早收敛；去膨胀门槛决定一个节点能不能冻结，不决定什么时候停止探索。"
    "`refs/families.md` 列出上一个方向（检索相似的历史案例）与相邻的持续学习线已经关闭的家族；"
    "重做其中一个，需要一个真正新的机制。"
)

ARMS: dict[str, dict[str, object]] = {
    "ttt_100k_8y_argon_zgca_20261028b": {
        "model": ARGON,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        # The relay's alternate key (LIGHTING_API_KEY_ALT) first; the primary
        # stays the fallback.
        "llm_key": "alternate",
        "workspace_reference": PACK,
        "research_directive": DIRECTIVE,
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=FUND_TITLE_PIT_VIEWS_SEED,
    overrides={
        **FUND_TITLE_EIGHT_YEAR_100K,
        "permitted_boards": ["main", "gem"],
        # One A100, claimed by the console when it starts the worker; every
        # leg of a batch shares it.
        "gpu_count": 1,
        # Four legs at once, all on that card.
        "batch_validate_max_concurrency": 4,
        "reasoning_effort": "xhigh",
        # Three times round 20261027's budgets (the console defaults, and
        # BASE_OVERRIDES for the session's minutes and model calls).
        "max_replay_years": 288,
        "max_null_controls": 36,
        "max_api_spend_usd": 150.0,
        "max_llm_calls": 19_200,
        "max_research_minutes": 7_200,
        "strategy_fit_timeout_seconds": 10_800,
        "strategy_inference_timeout_seconds": 1_080,
        # The strategy (and session) container: twice the default CPUs and
        # memory, so a per-day test-time update has the cores and room.
        "strategy_cpus": 16,
        "strategy_memory": "64g",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
