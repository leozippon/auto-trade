#!/usr/bin/env python
"""The 2026-10-27 c round: round 20261027's gemini-4-argon arm again, on the instance's other card, under the reorganized session prompt and the pack's 「建议（可以不采纳）」 section.

It is a same-model comparison with `retrieval_100k_8y_argon_zgca_20261027`,
which ran on the session prompt before 9f49764 and 30990a8 moved the method
advice into the pack and reorganized what was left. Everything the arm sends
is round 20261027's -- gemini-4-argon, the pack `retrieval_100k_8y_20261025b`,
the eight-year fundamentals-plus-titles geometry on a 100k account graded
against CSI 1000, the 10-24 seed, DeepSeek-V4.1-Flash sub-agents and
compaction with the threshold pinned at 221,184, xhigh, the console's default
budgets, the primary key, one A100, six replays per batch and the directive
word for word -- except its id. A round file never imports another
(tests/unit/test_round_arms.py), so this definition is a copy, held equal to
round 20261027's by tests/unit/test_retrieval_pack.py.

| Arm | Main session | Sub-agents / compaction |
| --- | --- | --- |
| `retrieval_100k_8y_argon_zgca_20261027c` | gemini-4-argon | DeepSeek-V4.1-Flash |

Usage, through `ops/zgca/zgca.py run`, which sets `AUTOTRADE_GPU_NAME=A100`
for the instance's console and this launcher:

    ~/miniconda3/envs/quant/bin/python ops/zgca/zgca.py run scripts/experiments/create_round_20261027c.py

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
# families; round 20261027's, which tests/unit/test_retrieval_pack.py holds
# equal (a round file never imports another).
DIRECTIVE = (
    "在八年研究期上，检索与当前情形相似的历史价量形态与自然语言信息（种子里的公告标题等文本），"
    "用检索到的历史案例选股，选出的书要相对零技能面板有真实的边。"
    "什么算「相似」、怎样使用检索到的案例、是否以及怎样用到任何模型（语言模型或其他），都由会话决定。"
    "在这个方向之内要多探索：试几种不同的「相似」、不同的检索对象与使用检索案例的不同方式，"
    "不要在第一个出现的信号上过早收敛；去膨胀门槛决定一个节点能不能冻结，不决定什么时候停止探索。"
    "`refs/families.md` 列出此前各臂已经关闭的家族；重做其中一个，需要一个真正新的机制。"
)

ARMS: dict[str, dict[str, object]] = {
    "retrieval_100k_8y_argon_zgca_20261027c": {
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
