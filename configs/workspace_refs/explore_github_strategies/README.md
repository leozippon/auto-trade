# GitHub A-share strategy exploration pack

This pack is a small, screened set of strategy ideas for a Fold Agent to **rewrite** under ADMCubeQuant's single-file ABI. It is not a Python package and is never imported by the formal strategy.

The previous `github_strategies` pack exposed many unrelated seeds and large platform-shaped files. The `ref_github_strats` experiment eventually spent folds replaying a parent or committing execution-identical variants. This pack instead presents four distinct, data-mapped hypotheses and requires executable rewrites before tuning.

## Non-negotiable contract

- Read references with workspace-relative paths such as `refs/README.md`; write the official strategy only to `output/main.py`.
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
