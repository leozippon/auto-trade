#!/usr/bin/env python
"""The 2026-10-01 round: one mechanism family per arm, with open axes.

The 2026-09-27 census ran 148 whole-pool arms (100k, 12 seats, monthly, eight
research years), each testing one rule score with one candidate. Its main
candidates' active IRs spread far wider than their shuffled controls (sd 0.67
against 0.38) and the excess sits on the negative side: the scores carry
information, mostly about which names to avoid. But each arm counted only its
own one to three trials, so the search ran outside any freeze-gate trial
family, and 105 of the 142 twelve-seat books failed the active-drawdown line
(logs/notes/round_20261001/DESIGN.md; docs/research-lessons.md §4).

So each arm is now one mechanism family whose pack pre-registers open axes,
the construction levers the gate fails on among them (seats, keep band,
industry cap, review cadence). The session iterates inside the arm for up to
three batches, a Y1..Y4 probe first, and its variants are that arm's trials.
An arm that continues a family earlier arms explored on the same eight years
names them in `lineage_arms`, and the console adds their trials to its gate.

The queue, in fill order (ids end in `_20261001`); four slots take the first four:

- `census_gbdt_alla_8y`: all 44 census features (19 families, chosen by
  coverage whatever sign each read) in a LightGBM ranker refitted in `fit()`,
  against a shuffled score and an equal-weight signed composite. No lineage:
  no census reading selected an input.
- `flip_alla_8y`: the other end of the strongly negative census scores, one
  rule-defined composite of 21 of them. Picking the most negative of 148 scores
  to flip is the same selection as picking the most positive, so the arm
  inherits all 148 census arms.
- `open_research_8y`: no family fixed; the session proposes its own from the
  eight-year data surface, no sequence networks, no lineage.
- `census_gbdt_csi1000_100k_8y`: the first arm's 100k twin, CSI 1000 members.
- `seqbag_alla_8y`: a rank-average bag of four structurally different sequence
  heads on the whole-pool recipe, on one GPU; a measurement arm that inherits
  the ten arms that explored that recipe.

All arms share the 2026-09-27 eight-year geometry, dividend-fixed seed and
gates, 96 replay-years, CSI 1000 as benchmark and the 0.55 / 0.30 drawdown caps
of the earlier eight-year CSI 1000 arms at 1m and 100k alike.

Usage: create_round_20261001.py <port> [--dry-run] [--fill] [experiment_id ...]
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
from scripts.experiments.create_round_20260927 import (
    CSI1000,
    DIVIDEND_PIT_VIEWS_SEED,
    EIGHT_YEAR,
    GATES,
)

# The flip arm's lineage: every whole-pool census arm, each of which researched
# these eight years and recorded a non-control trial.
CENSUS_ARMS: tuple[str, ...] = (
    "age1k_alla_100k_20260927", "ageout_alla_100k_20260927", "amtac_alla_100k_20260927",
    "amtcv_alla_100k_20260927", "amtind_alla_100k_20260927", "amtlag_alla_100k_20260927",
    "at1k_alla_100k_20260927", "atcy_alla_100k_20260927", "atlag_alla_100k_20260927",
    "aturn_alla_100k_20260927", "beta1k_alla_100k_20260927", "beta60_alla_100k_20260927",
    "big500_alla_100k_20260927", "bigind_alla_100k_20260927", "body_alla_100k_20260927",
    "boundary_alla_100k_20260927", "broad_alla_100k_20260927", "cashmix_alla_100k_20260927",
    "ch3bmk_alla_100k_20260927", "champ1000_alla_100k_20260927", "champ300_alla_100k_20260927",
    "champ_alla_100k_20260927", "champbmk_alla_100k_20260927", "chdelta_alla_100k_20260927",
    "csk1k_alla_100k_20260927", "decouple_alla_100k_20260927", "delay1k_alla_100k_20260927",
    "delay60_alla_100k_20260927", "dn1000_alla_100k_20260927", "dn20_alla_100k_20260927",
    "dn60_alla_100k_20260927", "earnac_alla_100k_20260927", "entry_alla_100k_20260927",
    "exit1000_alla_100k_20260927", "exit300_alla_100k_20260927", "exit500_alla_100k_20260927",
    "exit50_alla_100k_20260927", "exitbmk_alla_100k_20260927", "exitcy_alla_100k_20260927",
    "exitwide_alla_100k_20260927", "freefloat_alla_100k_20260927", "freegrow_alla_100k_20260927",
    "freeshare_alla_100k_20260927", "iat1k_alla_100k_20260927", "indact_alla_100k_20260927",
    "indcalm_alla_100k_20260927", "inddisp_alla_100k_20260927", "indflow_alla_100k_20260927",
    "indmv_alla_100k_20260927", "indpb_alla_100k_20260927", "indpbl_alla_100k_20260927",
    "indrev_alla_100k_20260927", "indrise_alla_100k_20260927", "indto_alla_100k_20260927",
    "iroe_alla_100k_20260927", "iwadd_alla_100k_20260927", "leader_alla_100k_20260927",
    "limdn_alla_100k_20260927", "liqco_alla_100k_20260927", "listed_alla_100k_20260927",
    "lock1000_alla_100k_20260927", "mgnlag_alla_100k_20260927", "mid1000_alla_100k_20260927",
    "mid300_alla_100k_20260927", "mid3bmk_alla_100k_20260927", "mid50_alla_100k_20260927",
    "midbmk_alla_100k_20260927", "midcy_alla_100k_20260927", "midflat_alla_100k_20260927",
    "midnew_alla_100k_20260927", "nearvw_alla_100k_20260927", "new300_alla_100k_20260927",
    "new500_alla_100k_20260927", "newbig_alla_100k_20260927", "newbmk_alla_100k_20260927",
    "newcy_alla_100k_20260927", "newmem_alla_100k_20260927", "next1k_alla_100k_20260927",
    "nextlo_alla_100k_20260927", "nextmb_alla_100k_20260927", "nlead_alla_100k_20260927",
    "paylag_alla_100k_20260927", "pb1000_alla_100k_20260927", "pb_alla_100k_20260927",
    "pblagm_alla_100k_20260927", "pe1000_alla_100k_20260927", "pe_alla_100k_20260927",
    "peerlag_alla_100k_20260927", "peerlead_alla_100k_20260927", "peerliq_alla_100k_20260927",
    "peersize_alla_100k_20260927", "pelagm_alla_100k_20260927", "pond_alla_100k_20260927",
    "ps1000_alla_100k_20260927", "ps1ind_alla_100k_20260927", "ps1sz_alla_100k_20260927",
    "ps300_alla_100k_20260927", "ps500_alla_100k_20260927", "ps_alla_100k_20260927",
    "psacc_alla_100k_20260927", "pscy_alla_100k_20260927", "psgap_alla_100k_20260927",
    "psgem_alla_100k_20260927", "psind_alla_100k_20260927", "psl10_alla_100k_20260927",
    "psl300_alla_100k_20260927", "psl42_alla_100k_20260927", "psl500_alla_100k_20260927",
    "pslag_alla_100k_20260927", "pslage_alla_100k_20260927", "pslagm_alla_100k_20260927",
    "pslbig_alla_100k_20260927", "pslboth_alla_100k_20260927", "pslcy_alla_100k_20260927",
    "pslly_alla_100k_20260927", "psloutm_alla_100k_20260927", "pslsml_alla_100k_20260927",
    "psltail_alla_100k_20260927", "pslthen_alla_100k_20260927", "psly1k_alla_100k_20260927",
    "psly500_alla_100k_20260927", "psly_alla_100k_20260927", "psmain_alla_100k_20260927",
    "psmly_alla_100k_20260927", "psnew_alla_100k_20260927", "psout_alla_100k_20260927",
    "pssize_alla_100k_20260927", "psstay_alla_100k_20260927", "psxfin_alla_100k_20260927",
    "pvcorr_alla_100k_20260927", "reldn_alla_100k_20260927", "roe1k_alla_100k_20260927",
    "roeout_alla_100k_20260927", "room_alla_100k_20260927", "runner_alla_100k_20260927",
    "sale1000_alla_100k_20260927", "sale500_alla_100k_20260927", "salem_alla_100k_20260927",
    "ssch1000_alla_100k_20260927", "stay500_alla_100k_20260927", "swpe_alla_100k_20260927",
    "sz1k_alla_100k_20260927", "tom1k_alla_100k_20260927", "volag_alla_100k_20260927",
    "wdelta_alla_100k_20260927", "wresid_alla_100k_20260927", "wtheavy_alla_100k_20260927",
    "wtstab_alla_100k_20260927",
)

# The seqbag arm's lineage: every arm that recorded a non-control trial on the
# whole-pool sequence recipe over these eight years (its pack README §8). The
# 100k MLP and the CSI 1000 GRU recorded controls only and cannot be lineage.
RECIPE_ARMS: tuple[str, ...] = (
    "mlp_alla_8y_20260925",
    "tcn_alla_8y_20260925",
    "lstm_alla_8y_20260926",
    "tattn_alla_8y_20260926",
    "shortmlp_alla_8y_20260926",
    "lagpool_alla_8y_20260926",
    "decay_alla_1m_20260926",
    "patchmlp_alla_100k_20260926",
    "spec_alla_100k_20260926",
    "lstm_alla_100k_20260926",
)

ARMS: dict[str, dict[str, object]] = {
    "census_gbdt_alla_8y_20261001": {
        "research_directive": (
            "本臂是一个机制家族：普查算过的 44 个截面特征喂给在 fit 里按季从头重拟合的 LightGBM 排序器，"
            "全 A 去北交所、科创板与 ST，只用 CPU。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX / CAPITAL 一致，否则停。"
            "不加列，也不把任何一列拿出来做独立分数。"
            "开放轴只在 refs/README.md 的表里取值，探哪些值、谁进决赛由你定，每批在 hypothesis 里先登记；"
            "每个配置与同一本书上的 c_shuf、c_lin（control: true）同批。"
            "第一批 Y1..Y4，offline_trials = 0；批次、杀死线与 SEED = 8 的复现照 README，至多三批。"
        ),
    },
    "flip_alla_8y_20261001": {
        "lineage_arms": list(CENSUS_ARMS),
        "research_directive": (
            "本臂测普查强负分数的另一端：refs/README.md 按规则定下的 21 个普查分数各自取反、"
            "按日取池内百分位秩、等权合成 f_all，全 A 去北交所、科创板与 ST，不训练、只用 CPU。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX 一致，否则停。"
            "翻转集合、分数定义与复核节奏不是变量；子集不是轴，不要逐个筛分数。"
            "本臂以整个普查为 lineage 建臂：门槛只读全期行的 selection_statistics.information_ratio_bar 与 arm.lineage，不自己算。"
            "第一批 Y1..Y4：c_shuf、c_orig（都 control: true）与 f_all，offline_trials = 0；"
            "探针线、第二批全期与五条杀死线照 README；第三批至多一批、每条腿只动一根轴，"
            "提名只在 f_all 与它的单轴变体里选。"
        ),
    },
    "open_research_8y_20261001": {
        "research_directive": (
            "本臂不定机制家族：你从八年数据面上自己提出家族、成批预登记并检验，以冻结或 no_edge 收尾；"
            "全 A 去北交所、科创板与 ST，只用 CPU，不做逐股序列网络。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX 一致，否则停。"
            "起步包的 p0 按构造不含信息，不是候选。"
            "至少两个结构不同的家族；每个分数带自己的 c_shuf，主张落在构造或筛子上再带随机化的安慰剂；"
            "先 Y1..Y4 探针、再全期；拟合出来的东西提名前换第二个训练种子复现。"
            "重开 refs/families.md 里的家族照 refs/README.md「什么算重复」三条，因普查读数而选的把该家族的普查臂数计进 offline_trials。"
            "杀死线按家族判，至多三批。"
        ),
    },
    "census_gbdt_csi1000_100k_8y_20261001": {
        "initial_cash": 100_000,
        "research_directive": (
            "本臂是同一普查特征 LightGBM 排序器在 10 万账户上的孪生：fit 里按季从头重拟合、只用 CPU，"
            "池是决策日在册的中证 1000 成分（去科创板与 ST）。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 10 万，与 knobs 的 INDEX / CAPITAL 一致，否则停。"
            "开放轴只有席位与复核节奏，取值照 refs/README.md，每批在 hypothesis 里先登记；其余照 README 固定。"
            "每个配置与同一本书上的 c_shuf、c_lin（control: true）同批；现金不足被拒的买单比例与费用照实报。"
            "第一批 Y1..Y4，offline_trials = 0；批次、杀死线与 SEED = 8 的复现照 README，至多三批。"
        ),
    },
    "seqbag_alla_8y_20261001": {
        "gpu_count": 1,
        "lineage_arms": list(RECIPE_ARMS),
        "research_directive": (
            "本臂是量测臂：全 A（去北交所、科创板与 ST）[FEAT-14] 配方上四个结构不同的网络头 × 种子的秩平均袋，"
            "不是单头，也不问哪个头最好。"
            "benchmark_index 必须是 000852.SH、initial_cash 为 100 万，与 knobs 的 INDEX 一致，否则停；"
            "gpu_count 为 1，没有 CUDA 就停，不要改成 CPU。"
            "头、标签、窗口、重训节奏、通道与股票池是配方，不改；袋只从 lib/model.py 的 BAGS 里选。"
            "本臂以 [FEAT-14] 各臂为 lineage 建臂：门槛只读 selection_statistics.information_ratio_bar，不自己算。"
            "第一批 Y1..Y4，offline_trials = 0；探针腿与可选腿、决赛袋的选法、第二批整期五条腿、"
            "F1–F3 与 K1–K5 照 refs/README.md §5–§6。"
        ),
    },
}

ROUND = Round(
    arms=ARMS,
    pit_views_seed=DIVIDEND_PIT_VIEWS_SEED,
    overrides={
        **EIGHT_YEAR,
        **GATES,
        "max_replay_years": 96,
        "initial_cash": 1_000_000,
        "benchmark_index": CSI1000,
        "max_drawdown": 0.55,
        "active_max_drawdown": 0.30,
    },
)


if __name__ == "__main__":
    raise SystemExit(ROUND.main(sys.argv, __doc__))
