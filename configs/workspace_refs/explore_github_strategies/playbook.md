# Fold rewrite playbook

The goal is not to preserve a GitHub implementation. The goal is to turn three or more independent hypotheses into legal `output/main.py` revisions inside one fold and let this project's complete Validation falsify them.

## 1. Establish the local contract

Before editing the strategy:

1. Read this pack, the current parent `output/main.py`, the Fold facts, `data_summary.json`, and `unit_reference.json`.
2. Inspect Parquet metadata for only the domains and columns needed by the chosen idea.
3. Confirm the configured inference time and strategy period. At `08:30`, reject any daily row whose `trade_date` is today or later.
4. Confirm each non-daily row has a parseable `available_at` no later than `context.inference_at`.
5. Record whether `fundamentals` and `events` exist. Do not catch a missing required domain and silently call the idea successful.

Use `context.asof_dir + "/daily"`, `... + "/fundamentals"`, or `... + "/events"` directly in `pandas.read_parquet`. Read bounded dates and selected columns. Do not read or import anything from `refs` at runtime.

## 2. Build one minimal vertical slice

Each candidate revision should contain:

- one stateless rebalance gate derived from visible calendar rows;
- one bounded data load;
- one explicit PIT cutoff;
- one tradability/liquidity funnel using T-1 daily fields;
- one score and deterministic sort;
- target selection, sells first, then buys with a locally decremented cash budget;
- strict-JSON orders with a timezone-aware `execute_at` no earlier than inference.

Read positions only from `context.account.positions`, whose contract is `{symbol: quantity}`. Use `sorted(...)` for deterministic order construction. Quantities are positive integer shares; A-share buys should normally be rounded down to board lots. Leave a cash buffer for costs and price movement. The Broker, not the strategy, decides whether an order fills.

Do not add a generic optimizer, custom backtester, persistent state machine, framework compatibility layer, or copied source order manager.

## 3. Required rewrite order

A Fold with a parent must complete a Validation of executable logic different from that parent before it may retain the parent. Comment-only edits, a parent replay, and an inactive parameter change do not satisfy this pack's purpose.

Before spending a Validation on an idea, screen it offline over the 21-month input window — visible event counts, coverage stability, and the quarter-by-quarter sign of the cohort's forward return — and drop the ones that cannot clear that screen.

Use this order, skipping an item only when its required domain is absent or its schema/PIT contract cannot be proved:

1. **Earnings-forecast drift.** Implement the smallest announcement-visible surprise ranking. Validate it as a standalone candidate.
2. **Holder accumulation minus unlock pressure.** Implement only if `events` contains both required datasets with unit rules. Validate it independently, not as an overlay on idea 1.
3. **Volume-weighted reversal.** Daily-only, so it is always available; validate it standalone in the same fold instead of holding it back as a fallback.
4. **EP plus abnormal turnover.** The slow monthly daily-only alternative; validate it standalone as well.

Complete at least three genuinely different idea Validations inside the fold before tuning thresholds; the fold budget allows more. If only daily data is available, use ideas 3 and 4 and then an independent variant of one of them rather than stopping early. A hybrid of the two best completed candidates is allowed only after both standalone mechanisms have been measured.

For each revision, change one hypothesis family, state its failure condition before Validation, and compare exposure, turnover, rejected orders, costs, concentration, drawdown, and return against a same-window run of the parent. Report those figures separately for the two halves and the four sub-windows of the Validation year: an edge confined to one sub-window, or with halves of opposite sign, is a failure even when the full-year total looks good. Roll back a failed node rather than carrying dead branches into the official file.

## 4. Idea implementation cards

### A. Earnings-forecast drift

- Read `fundamentals` rows where `dataset == "forecast_vip"`.
- Filter `available_at <= inference_at` before any deduplication or latest-version choice.
- Treat each revision's own `ann_date`/`available_at` as its visibility time. Never backdate a revised row to `first_ann_date`.
- Use a robust surprise such as the mean of `p_change_min` and `p_change_max`, with `type` only as a coarse tie-break or sign check.
- Begin eligibility no earlier than the first trading day after the event is visible. Use a fixed, short active window; do not fit it to Validation.
- Exclude negative forecasts from buys; optionally make them an exit/avoidance condition. Long-only Broker means a negative signal is not a short.
- Failure conditions: insufficient visible events, unstable coverage, concentration in a few names, no meaningful exposure, or returns consumed by event churn.

### B. Holder accumulation minus unlock pressure

- Read `events` rows for `stk_holdertrade` and `share_float_complete` only.
- Filter every row by `available_at` first.
- Accumulation leg: signed `change_ratio` or `change_vol`, weighted by a simple 0–30/31–60/61–90 day decay. Prefer within-dataset ranks over uncertain cross-unit arithmetic.
- Supply leg: sum or max `float_ratio` for already-announced unlocks whose `float_date` lies in the next 30 calendar days.
- Score accumulation positively and unlock pressure negatively; require both legs to be explicitly missing-safe.
- Do not infer a future unlock from `float_date` unless its row is already announcement-visible.
- Failure conditions: event domain absent, unit status unresolved, too few names, score driven by one extreme event, or no independent effect after liquidity filters.

### C. Volume-weighted five-day reversal

- Use adjusted closes and normalized daily volume through T-1.
- Compute daily return, rolling five-day volume shares, then negate the weighted return sum.
- Rebalance at most weekly; derive the first visible trading day of the week from the as-of daily calendar rather than module state.
- Add only low-cost eligibility guards (history, T-1 liquidity, suspension, price limits). Do not bury the signal under the old small-cap or Alpha101 composite.
- Failure conditions: churn/cost dominance, same-direction continuation instead of reversal, or fragile exposure.

### D. EP plus abnormal turnover

- Use positive T-1 `pe_ttm`: `EP = 1 / pe_ttm`.
- Define abnormal turnover as recent 20-day mean `turnover_rate` divided by a long trailing mean (about 252 bars); prefer lower abnormal turnover.
- Remove the smallest market-cap tail or rank within coarse T-1 size buckets so this is not another pure-small-cap replay.
- Rebalance monthly. Combine cross-sectional ranks, not incomparable raw units.
- Failure conditions: insufficient long history, value-sector concentration, hidden size dominance, or no improvement after costs.

## 5. Order sketch, not source-platform emulation

At a rebalance:

1. Rank visible eligible names.
2. Keep a small top set and derive exits from actual positions.
3. Emit full-quantity sells for names outside the target.
4. Estimate only a conservative cash budget from the current cash snapshot and optional haircut on sale proceeds.
5. Allocate deterministic equal-weight buy lots, decrementing the budget after each order.
6. Attach short JSON metadata such as idea name and score rank.

Do not use `order_target_value`, `run_daily`, `get_fundamentals`, `attribute_history`, `datetime.now()`, source DB paths, or a GitHub framework object. Those belong to the source environments and are not part of this ABI.

## 6. Finish criteria

Before `finish_fold`, verify that the chosen node:

- differs in executable logic from the parent, or keeps the parent only after at least two distinct-hypothesis Validations completed and were falsified;
- contains a self-contained `output/main.py` and no imported helper source;
- uses only visible rows and T-1 daily data at `08:30`;
- returns strict JSON and uses deterministic positive integer quantities;
- has passed strategy validation, modification checks, and a complete Validation;
- leaves no logs, cache, downloaded source, notebook, or runtime path in `output` or `models`.
