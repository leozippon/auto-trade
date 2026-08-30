# GitHub A-share strategy exploration pack

This pack is a small, screened set of strategy ideas for a Fold Agent to **rewrite** under ADMCubeQuant's strategy-package ABI. It is not a Python package and is never imported by the formal strategy.

The previous `github_strategies` pack exposed many unrelated seeds and large platform-shaped files. The `ref_github_strats` experiment eventually spent folds replaying a parent or committing execution-identical variants. This pack instead presents four distinct, data-mapped hypotheses and requires executable rewrites before tuning.

Development is cut into one regular Fold per year — each Fold's Validation is that year, there is no Test stage, and a Meta session runs between adjacent Folds — with a 24-month input window before each Validation, and the budget allows well more than one complete Validation. Pre-register at least three of the four ideas before implementing, rewrite them, run them side by side with `batch_validate`, and judge each on sub-windows of the development window rather than on a single aggregate number. That first round opens the Fold rather than closing it: the survivors go into further pre-registered rounds — a ranking fitted in `fit` (logistic or a small tree model over the same PIT features) against the hand-signed score, neutralized variants, holding-period and sizing rules — until the budget or the hypotheses run out.

## Non-negotiable contract

- Read references with workspace-relative paths such as `refs/README.md`; write the official strategy only under `output/` (entry `output/main.py`; sibling modules are imported absolutely).
- `output/main.py` must define the single synchronous entry `generate_orders(context)` and return strict-JSON orders.
- Formal code may use only the libraries allowed by the runtime contract; the formulas here require only NumPy and pandas.
- Every source row must satisfy `available_at <= context.inference_at`. At the usual `08:30` inference, daily features end at T-1.
- Read formal inputs only below `context.asof_dir` or `context.snapshot_dir`. Never import `refs`, and never hardcode `/mnt/agent/workspace` or another host/runtime path.
- The sandbox has no network and no package installation. Delete every source repository's database, downloader, scheduler, broker, logger, and framework adapter.
- Broker constraints, T+1, fees, price limits, and exact execution prices belong to the Environment. The strategy emits order sketches; it does not simulate fills.
- Do not copy reported returns from a source README into strategy expectations. Reproduce behavior only through this project's Validation.
- A missing required domain, column, unit rule, or PIT timestamp is a reason to return `[]` or choose another idea, not to guess.

## What to read

1. [`playbook.md`](playbook.md): mandatory Fold workflow and rewrite order.
2. [`screening.md`](screening.md): formulas, project-data mappings, PIT checks, and rejected material.
3. `vendor/*/SOURCE.md` before the adjacent formula excerpt.

The files under `vendor/` are short formula references, not runnable strategies. They intentionally contain no `main.py`, order API, data downloader, framework, or network code.

## Screened ideas

| Priority | Idea | Required domain | Distinct hypothesis |
| ---: | --- | --- | --- |
| 1 | Earnings-forecast announcement drift | fundamentals + daily | Post-announcement underreaction, not price-only ranking |
| 2 | Holder accumulation minus announced unlock pressure | events + daily | Announced demand/supply imbalance |
| 3 | Volume-weighted five-day reversal | daily | Short-horizon overreaction reversal |
| 4 | EP plus abnormal-turnover value/sentiment | daily | Slow value with a speculation/turnover penalty |

The WorldQuant formulas in `vendor/alpha101_control/` are retained only as operator tests and negative controls. The previous experiment already used the #101/#12/#6 family; selecting it first would repeat the old lineage rather than test a new idea.

## Explicitly out of scope

Do not vendor, import, or recommend vn.py, Qlib, RQAlpha, Backtrader, JoinQuant APIs, broker SDKs, database layers, factor platforms, full repositories, notebooks, PDFs, pretrained models, or package environments. Framework code is not strategy evidence.

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
