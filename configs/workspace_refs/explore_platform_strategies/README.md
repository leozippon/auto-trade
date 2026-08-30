# Stock-discussion-platform strategy exploration pack

This pack turns common 雪球、东财股吧、同花顺、财联社 and short-term-forum ideas into point-in-time research hypotheses that can be rebuilt from the repository's frozen data. It is a reference pack, not a strategy artifact and not evidence of expected returns.

## Hard boundary

- The Fold Agent must write the formal strategy only under `output/` (entry `output/main.py`; sibling modules are imported absolutely).
- Do not copy this reference tree into `output`.
- No `/mnt/agent/workspace` hard-code. Read formal inputs only through `context.asof_dir` or `context.snapshot_dir` after confirming the current run's manifest, schema, and unit reference.
- The formal replay is the `output/` package (entry `main.py`, optional sibling modules) using only the runtime contract's supported libraries and `generate_orders(context)`.
- The sandbox has no network. Never fetch 雪球、股吧、同花顺、财联社 or any other website at inference time.
- Every used row must satisfy `available_at <= context.inference_at`. At the default 08:30 decision, today's daily bar, auction, opening gap, intraday comment stream, and intraday hot-list move do not exist yet.
- This pack deliberately excludes the previous naive MA20 risk switch. Each retained playbook has an event, attention, limit-state, chip, seat, or money-flow mechanism with an explicit falsification test.

## Files

| File | Use |
| --- | --- |
| [`pit-field-map.md`](pit-field-map.md) | Exact local datasets, fields, units, timing traps, and coverage boundaries |
| [`research-screen.md`](research-screen.md) | Web-research provenance and hard rejects |
| [`playbooks.md`](playbooks.md) | Nine rewriteable hypotheses with entry, exit, falsification, and one-file sketches |

## Fold exploration order

Development is cut into one regular Fold per year (each Fold's Validation is that year, no Test stage, a Meta session between adjacent Folds) and the input window covers the 24 months before each Validation. Screen all nine mechanisms offline against the input window first — event counts, coverage, and the sub-window sign of each cohort's forward return — then pre-register the three or more strongest survivors and run their complete Validations side by side with `batch_validate`. That is the first round, not the Fold: mechanisms that hold go into further pre-registered rounds — gates and ablations, a within-cohort ranking fitted in `fit` (logistic or a small tree model over the T-1 fields the playbook already uses) against the equal-weight basket, holding-period and sizing rules — until the budget or the hypotheses run out. Keep one mechanism per revision, but do not stop after one round, and do not begin by stacking several weak ideas into an opaque score.

The list below is a priority order for that screening, not a fixed running order.

1. **情绪冰点修复篮子** — longest relevant limit-list history; first establish whether market-level sentiment state adds anything after open execution costs.
2. **资金流—价格背离低吸** — broad 2020+ stock flow coverage; use the base `moneyflow` feed before late-start vendor confirmations.
3. **炸板修复低吸** — direct, distinct event cohort with a short holding period and a clear matched control.
4. **龙头首次强分歧续强** — higher-tail risk; test only after the emotion gate and execution assumptions are understood.
5. **筹码成本带承接低吸** — 2018+ chip data; tests a different inventory/holder-cost mechanism.
6. **龙虎榜机构净买延续** — sparse event study; compare with all-list and non-institution controls.
7. **跨平台热榜升温** — attention rather than sentiment; coverage is broad locally but rank-time semantics require careful deduplication.
8. **行业资金扩散轮动** — first use 2020+ stock flows aggregated by PIT 申万一级行业; vendor board-flow confirmation is only a later-window check.
9. **本地文本催化 + 关注确认** — last because text parsing, negation, deduplication, and large-table reads create the most implementation risk.

For each validated mechanism, report empty-signal rates and rejected orders and compare against a same-date liquidity/size-matched cohort, broken out by the two halves and the four sub-windows of the Validation year. An edge that appears in only one sub-window, or whose halves disagree in sign, is not evidence even when the full-year total looks good. Drop the idea when its stated falsifier fires; do not rescue it with a new MA filter or unrelated factor stack. Keeping the parent is legitimate only after at least two genuinely different mechanisms have completed Validations and failed.

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
