#!/usr/bin/env python
"""The 2026-10-27 round: round 20261026's retrieval direction with gemini-4-argon on the A100 instance, six replays per batch, and the record of closed families named in the directive.

The arm runs round 20261026's experiment as it stands -- the pack
`retrieval_100k_8y_20261025b`, the eight-year fundamentals-plus-titles
geometry on a 100k account graded against CSI 1000
(`FUND_TITLE_EIGHT_YEAR_100K`), main board and ChiNext, the 10-24 seed,
sub-agents and compaction on DeepSeek-V4.1-Flash with the compaction threshold
pinned at 221,184, xhigh, the console's default budgets (the spend cap
included), no lineage -- with these differences: the relay's primary key, one
A100 (`gpu_count` 1), six concurrent replays per batch
(`batch_validate_max_concurrency` 6: the instance has 128 CPUs and each
strategy container takes 8), and a directive that adds one sentence to round
20261026's: `refs/families.md` lists the families earlier arms closed, and
repeating one needs a genuinely new mechanism.

| Arm | Main session | Sub-agents / compaction |
| --- | --- | --- |
| `retrieval_100k_8y_argon_zgca_20261027` | gemini-4-argon | DeepSeek-V4.1-Flash |

Its gpt-6.1-sol sibling is round 20261027b. The console refuses to create a
GPU arm while too few cards are free -- its create pre-flight
(`ExperimentManager._claimable_gpus`) answers HTTP 400 -- and nothing on the
instance retries, so that arm waits in its own file until the running astra
arm releases its card.

Usage, through `ops/zgca/zgca.py run`, which sets `AUTOTRADE_GPU_NAME=A100`
for the instance's console and this launcher:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261027.py

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

PACK = "configs/workspace_refs/retrieval_100k_8y_20261025b"
ARGON = "gemini-4-argon"
FLASH = "DeepSeek-V4.1-Flash"
COMPACT_TOKEN_THRESHOLD = 221_184

# Round 20261026's directive -- the pack README's 「方向」 paragraph word for
# word, then its call to explore -- and one sentence on the record of closed
# families (tests/unit/test_retrieval_pack.py holds round 20261027b's equal).
DIRECTIVE = (
    "在八年研究期上，检索与当前情形相似的历史价量形态与自然语言信息（种子里的公告标题等文本），"
    "用检索到的历史案例选股，选出的书要相对零技能面板有真实的边。"
    "什么算「相似」、怎样使用检索到的案例、是否以及怎样用到任何模型（语言模型或其他），都由会话决定。"
    "在这个方向之内要多探索：试几种不同的「相似」、不同的检索对象与使用检索案例的不同方式，"
    "不要在第一个出现的信号上过早收敛；去膨胀门槛决定一个节点能不能冻结，不决定什么时候停止探索。"
    "`refs/families.md` 列出此前各臂已经关闭的家族；重做其中一个，需要一个真正新的机制。"
)

ARMS: dict[str, dict[str, object]] = {
    "retrieval_100k_8y_argon_zgca_20261027": {
        "model": ARGON,
        "subagent_model": FLASH,
        "compact_model": FLASH,
        "compact_token_threshold": COMPACT_TOKEN_THRESHOLD,
        # The relay's primary key first (the console default, stated because
        # round 20261026 sent the alternate).
        "llm_key": "primary",
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
        # Six legs at once: 8 CPUs per strategy container on a 128-CPU host.
        "batch_validate_max_concurrency": 6,
        "reasoning_effort": "xhigh",
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
