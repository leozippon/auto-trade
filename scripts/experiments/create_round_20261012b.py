#!/usr/bin/env python
"""The 2026-10-12 GPU round: two novel methods on the four-seed sequence bag, one arm each.

The owner encourages innovative exploration -- state-space models, continual
learning, graph models -- without spending most of the effort there. Three
cards are idle while the training-recipe lane (round 20261011) runs, so this
round puts two small lanes on them. Their honest expectation is no edge: the
research designer prices a state-space head on the same 60 daily bars at one
or two per cent, since nine heads on those bars moved the reading less than a
change of seed, and would test continual updating as one registered variant
of the recipe, which is what this round does (logs/notes/round_20261012/
DESIGN_2_expanded_data.md section 6). What either arm can tell us is whether
the architecture of a head, or updating the heads between refits, matters at
all on this book: a no-edge close with the paired numbers is the expected
and useful result. A graph lane is not run: its edges would come from
information the price history already holds, and its lineage is the
heaviest on the platform.

Both lanes share one pack (configs/workspace_refs/
seqnovel_alla_100k_8y_20261012); the directive picks the lane. The baseline
is round 20261011's `c_bag4`, the 100k bag with four seeds per head: at its
defaults the starter resolves to the same members, fit schedule, training
caches on the Y1, Y5 and Y8 views and orders from the same scores as the
seqfresh starter and the recorded four-seed nodes
(logs/notes/round_20261012b/compare_bag4.out).

- `ssm`: a selective state-space head (Mamba-2 form, plain torch) joins the
  bag as a fifth head, or replaces the MLP head, chosen as the weakest before
  any reading (the only head ever read alone on this book). The placebo puts
  four re-seeded LSTM members in the same seed slot: more of a recurrence the
  bag already has, against the new architecture.
- `cl`: between the quarterly refits, which stay c_bag4's (cold every
  fourth), a monthly warm-start update of every member on the dates labelled
  since the last fit, two passes and no early stopping, alone or with as
  many dates replayed from the trailing window (replay rather than
  distillation: it needs no second loss and no weight on it). The placebo
  keeps the schedule, steps and replay and permutes the new dates' labels, so
  it separates new information from touching the weights more often; a no-op
  refit would be c_bag4 byte for byte and separate nothing.

The claim is round 20261011's, paired against the same-batch, same-seed
`c_bag4` on seed bases 1000, 2000 and 3000 (plain selection higher on each
half of the research window and not lower in its last two years, a mean
active-IR difference of at least +0.20), plus the same +0.20 over the lane's
placebo. A finalist whose claim holds is frozen on its seed-1000 node with
its 2000 and 3000 rows registered as seed replicates; the arm's facts decide
the rest.

Measured through the real pipeline on one free L20 (eight trading days of
Y8 from its last May week, seed base 1000; logs/notes/round_20261012b/
_runs/): a cold refit of `c_bag4` took 1,109 s and one of the bag with the
state-space head added 2,001 s, inside the 3,600 s fit limit (the four new
members about 220 s each, three times the average baseline member; the card
peak stayed at 9.5 GiB); a monthly update took 13.4 s on five new and five
replayed dates, about 25 s for a full month, so about 2 % of a cold fit and
half an hour a full-span leg. Both lanes plan the same three batches: the
two candidates and `c_bag4` on the second half of the window on a fourth
seed base, 4000 (12 replay-years); the finalist, its placebo and `c_bag4`
full span on 1000 and 2000 (48); the same three on 3000 if the finalist
survived those (24). That is 84 of the 96 replay-years, so the default budget stands.
Wall time at the batch's three-way concurrency on the arm's one card: about
13-16, 38-41 and 24-26 hours for the state-space lane, whose legs run up to
1.8 times a baseline leg, and about 7-9.5, 29-31 and 15 hours for the update
lane. On the GPU, the eight two-seed checkpoints the state-space smoke
trained before its new members equal, tensor for tensor, those round
20261011's smoke trained on the same decision day.

`seqnovel_ssm_100k_8y_mimo` runs the state-space lane on `mimo-v2.6-flash`
and `seqnovel_cl_100k_8y_qwen` the update lane on the local model: one arm per
lane, because earlier pairs on packs that prescribe their batches produced
byte-identical rows on both models. Geometry, seed and every round-level
parameter are round 20261011's. The lineage is the sequence bag's last closed
stage (`SEQBAG4_LINEAGE`, 83 trials, an IR bar from about 1.52 before the
arm's own trials); the two seqfresh arms are running siblings on the same
baseline and cannot be inherited until they close, so neither family counts
the other's trials, which is the accepted cost of running them side by side.
Text-evidence scoring stays on the local model in both arms.

Fill order (ids end in `_20261012`): `seqnovel_ssm_100k_8y_mimo`,
`seqnovel_cl_100k_8y_qwen`.

Usage: create_round_20261012b.py <port> [--dry-run] [--fill] [experiment_id ...]
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
    FUND_EIGHT_YEAR_100K,
    MIMO,
    SEQBAG4_LINEAGE,
    TAXED_FUND_PIT_VIEWS_SEED,
)
from scripts.experiments._round import Round

SEQNOVEL_PACK = "configs/workspace_refs/seqnovel_alla_100k_8y_20261012"

# What both directives say alike: the baseline, the arm's fixed facts, the
# claim, the reporting and where the rules are. Each lane's sentence goes first.
SEQNOVEL_COMMON = (
    "基线 c_bag4 是每头四个种子的 10 万序列袋（四个头各训四个种子，12 席、每次至多换 4 只、每周复核、不残差化、按季重训、每第 4 次冷启动），"
    "开关都在第一个值时起步包就是它；输入、标签、训练窗、按季重训与书在每条腿里都不动。"
    "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX 一致，否则停；"
    "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。每一批都带同 span、同种子基数的 c_bag4，安慰剂与 c_bag4 都登记为对照。"
    "主张按种子基数配对：在 refs/README.md 写定的三个种子基数上各与同一批、同种子的 c_bag4 比，"
    "候选的朴素选名（每个研究年书的收益减同一年零技能面板的收益，不回归，从行里逐年取）"
    "在 Y1..Y4 与 Y5..Y8 两半的平均都更高、在 Y7..Y8 的平均不更低，三个种子基数上主动 IR 之差的平均至少 +0.20，"
    "对同批、同种子的安慰剂主动 IR 之差的平均也至少 +0.20。"
    "这条线的先验很低，最可能的结局是没有可检出的差别：那是要带着每个种子基数上两半、Y7..Y8 与 IR 之差写清的结果，不是失败。"
    "主张不成立的决赛者即使过了门也不提名；提名哪一行、登记哪几行种子复现照 refs/README.md。"
    "每一行都报主动 IR、正年、主动回撤、逐年朴素选名、书对基准的市场载荷与权益口径的扣费年化收益。"
    "门槛读每行的 information_ratio_bar，不自己算；第一批 offline_trials = 0；至多四批；"
    "批次计划、决赛者规则与杀死线照 refs/README.md。"
)

SSM_DIRECTIVE = (
    "本臂走 ssm 线：只动袋里有哪些头。knobs.SSM 打开一个选择性状态空间头（Mamba-2 式，纯 torch，写在 lib/model.py），"
    "作为第五个头加进袋（add，腿 ssm_add）或替换前馈头 mlp（replace，腿 ssm_rep）；"
    "安慰剂把同一个种子位换成四个重新播种的 LSTM 成员（PLACEBO = True），问多出来的是「多一个头」还是「这个结构」。"
    "UPDATE 保持 off，main.py 的 REFIT_PERIOD 保持 quarter。"
    "先在第四个种子基数上用 Y5..Y8 探两个候选，再按 refs/README.md 的规则选一个决赛者，与它的安慰剂一起上整期；批 1 之前先单独冒烟 ssm_add 一次。"
    + SEQNOVEL_COMMON
)

CL_DIRECTIVE = (
    "本臂走 cl 线：只动两次按季重训之间做什么。knobs.UPDATE 打开每月一次的热启动小更新："
    "每个成员从自己的检查点出发，在上次拟合之后新有标签的日期上走两遍、不早停（warm，腿 cl_warm），"
    "或再加同样多个从尾部窗口回放的旧日期（replay，腿 cl_replay）；按季重训与它的冷启动和 c_bag4 相同。"
    "开 UPDATE 时 main.py 的 REFIT_PERIOD 必须同时改成 month，两行一起改（knobs 会核对，不一致就在第一次 fit 停下）。"
    "安慰剂用同样的日程、步数与回放，只把新日期的标签在名字之间打乱（PLACEBO = True），问起作用的是「新信息」还是「更常动权重」。"
    "SSM 保持 off。"
    "先在第四个种子基数上用 Y5..Y8 探两个候选，再按 refs/README.md 的规则选一个决赛者，与它的安慰剂一起上整期；"
    "批 1 之前先单独冒烟 cl_replay 一次，开始日放在 Y8 某个月的最后三四个交易日里，让冒烟跨过下一个月初、跑到一次更新。"
    + SEQNOVEL_COMMON
)

SEQNOVEL_ARM: dict[str, object] = {
    "workspace_reference": SEQNOVEL_PACK,
    "gpu_count": 1,
    "lineage_arms": SEQBAG4_LINEAGE,
}

ARMS: dict[str, dict[str, object]] = {
    "seqnovel_ssm_100k_8y_mimo_20261012": {**MIMO, **SEQNOVEL_ARM, "research_directive": SSM_DIRECTIVE},
    "seqnovel_cl_100k_8y_qwen_20261012": {**SEQNOVEL_ARM, "research_directive": CL_DIRECTIVE},
}

# The seed, the geometry and every round-level parameter are round 20261011's.
ROUND = Round(
    arms=ARMS,
    pit_views_seed=TAXED_FUND_PIT_VIEWS_SEED,
    overrides=FUND_EIGHT_YEAR_100K,
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
