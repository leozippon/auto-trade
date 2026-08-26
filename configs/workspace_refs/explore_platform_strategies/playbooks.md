# Screened playbooks

These are rewrite sketches, not a complete `main.py`. Each should be tested alone before any combination.

## Common one-file scaffold

- Use `period=day`, default `inference_time=08:30`, and the latest **visible trading date** in each dataset rather than `calendar_day - 1`.
- Read bounded columns from `context.asof_dir + "/daily"`, `"/events"`, `"/universe"`, or `"/text_index"`; immediately filter event unions by `dataset` and recheck `available_at`.
- Keep only valid A-share codes and the configured screened universe. Use the PIT `universe` name/industry fields; do not import current classifications.
- Make every score deterministic: stable sort by score then `ts_code`, explicit missing-value rejection, and no unordered-set priority.
- At 08:30, submit entry/exit orders for same-day 09:30. Size from the latest visible close only as a conservative estimate, round buys down to 100 shares, and leave a price/cost buffer.
- Initial experiments should cap gross exposure so new buys fit `context.account.cash` without assuming proceeds from same-call sells. Put sells before buys, but do not size as though the immutable account snapshot has already received sale proceeds.
- Reconstruct state from visible rows and `context.account.positions`. Do not require a mutable “yesterday's candidates” global.
- Report candidate count, exposure, turnover, rejected orders, and empty-signal days. A signal that appears only in one late feed is not a full-history result.

## 1. 情绪冰点修复篮子

**Mechanism.** The retail “emotion cycle” claim is reduced to a market breadth transition: failed boards and limit-down breadth stop worsening, while closed-board success and low-height participation recover. This is not an index moving-average switch.

**PIT signal.** From the last 5–10 visible `limit_list_d` dates compute:

- `up = count(limit == 'U')`, `broken = count(limit == 'Z')`, `down = count(limit == 'D')`;
- close success `up / (up + broken)` when the denominator is nonzero;
- maximum closed-board `limit_times`, first-board count, and industry breadth;
- a repair state when success and first-board breadth rise from a recent trough while down/broken breadth does not make a new high.

On the latest date, select liquid closed first boards (`U`, `limit_times == 1`) from industries with at least 2–3 closed boards. Prefer fewer reopenings and stronger `fd_amount / amount` ranks; reject missing or nonpositive denominators.

**Entry.** Next 09:30, equal-risk or equal-cash basket of 3–8 names, with gross exposure controlled by the repair score. No trade in an undefined/declining state.

**Exit.** Remove a position at the next 09:30 when the repair state fails, its industry breadth disappears, or the stock is no longer in the fresh/continued closed-board cohort. Keep cash rather than inventing a trend fallback.

**Falsifies it.** Net open-to-open performance after costs does not beat a same-date, same-liquidity first-board basket **without** the repair gate; improvement is confined to one fold; or most apparent close-to-close alpha vanishes at next-open execution.

**One-file sketch.** Load a 10-date limit-list tail, aggregate one row per market date, shift every market-state feature by construction to the next execution date, rank the latest eligible cohort, then map stale positions to sells and top-ranked new candidates to buffered 100-share buys.

## 2. 资金流—价格背离低吸

**Mechanism.** Positive signed stock flow during a weak/flat price day may indicate absorption rather than visible momentum. Vendor “main flow” is a classified trade-size proxy, not proven institutional money.

**PIT signal.** Use 2020+ `moneyflow` as the base signal:

```text
flow_share = 10,000 * net_mf_amount / normalized_daily_amount_CNY
```

Rank `flow_share` cross-sectionally on T-1. Require a positive/top-quantile flow score while the stock's T-1 or short-horizon return is in the weak half but not at a down-limit state. Require adequate turnover and no suspension. In 2023-12+ and 2025+ **separate robustness samples**, check sign/rank agreement with `moneyflow_dc.net_amount_rate` and `moneyflow_ths.net_amount` rather than changing the base rule silently.

**Entry.** Next 09:30, diversified top-ranked divergence names; avoid names already closed at limit-up because that is a different mechanism.

**Exit.** Sell when the latest visible flow score turns nonpositive/bottom-half, price weakness becomes a down-limit/failure state, or a small fixed rebalance horizon ends. A first implementation can retain only positions that still satisfy the current signal.

**Falsifies it.** The divergence cohort does not beat a return/liquidity/size-matched weak-price cohort; a single-source sign is unstable across folds; or later vendor agreement merely selects the 2025 regime rather than improving within common coverage.

**One-file sketch.** Read only T-1 plus the short return window, convert the 10k-CNY flow field before forming the dimensionless ratio, merge by `(trade_date, ts_code)`, apply cross-sectional percentiles, and keep the vendor-confirmed test in a separately declared sample flag.

## 3. 炸板修复低吸

**Mechanism.** A failed limit-up is a direct supply/demand event. The hypothesis is that a liquid, still-positive broken board inside a healthy peer group can repair at the next session; this is not a generic Bollinger or MA pullback.

**PIT signal.** Latest visible `limit_list_d` rows with `limit == 'Z'`. Keep names with valid/high turnover or amount, positive close return/close location, and controlled `open_times`; require either same-industry closed-board breadth or positive `moneyflow` absorption. Exclude limit-down, suspension, ST, and extremely illiquid tails.

**Entry.** Next 09:30, small equal-weight cohort. Treat a large next-open gap as unobservable at 08:30; the realized open is part of the test, not a filter.

**Exit.** One-session event lifecycle: at the following 09:30 remove stale cohort members unless a fresh independent signal is present. Do not keep losers by switching them to a trend strategy.

**Falsifies it.** After-cost next-open-to-next-open return is nonpositive or no better than stocks matched on T-1 return, turnover, size, and industry that did not receive `Z`; profits exist only at the unavailable T-1 close; or Broker rejections dominate.

**One-file sketch.** Build the T-1 `Z` cohort, merge one T-1 flow row and liquidity fields, rank within industry/date, and rotate only the bounded top names. Track matched-control returns outside the formal strategy during exploration.

## 4. 龙头首次强分歧续强

**Mechanism.** “龙头分歧转一致” becomes a prior-day recovered-disagreement event: a high-streak stock opened its board but still closed limit-up with meaningful seal/turnover support and peer breadth.

**PIT signal.** On T-1 require:

- `limit == 'U'` and `limit_times` at the market maximum or one below it;
- `open_times` in a small nonzero band, so it is neither an untouched one-word board nor a repeatedly failing board;
- strong cross-sectional `fd_amount / amount`, adequate amount, and a non-late seal/reseal time;
- at least one same-industry closed board and a non-deteriorating market success ratio from playbook 1.

Use ranks rather than invented absolute seal-amount thresholds.

**Entry.** Next 09:30, at most 1–3 leaders and lower gross exposure than a diversified basket. A buy rejected at the limit is a real outcome; never infer a fill from `fd_amount`.

**Exit.** At the next available 09:30 after the stock fails to remain a closed board/height leader, market broken-board breadth jumps, or the position no longer meets the fresh leader rule. No averaging down.

**Falsifies it.** It fails to beat all high-streak closed boards matched on height/liquidity; `open_times` and seal-ratio filters have no stable ordering; returns are concentrated in unfillable limit opens; or tail loss dominates the portfolio despite sizing.

**One-file sketch.** Parse count/time fields defensively, calculate the daily height from visible `limit_list_d` rather than `limit_step`, merge the market-state table, sort by height, seal ratio, amount, and code, and cap both names and exposure.

## 5. 筹码成本带承接低吸

**Mechanism.** A pullback toward the holder cost center with a compact cost distribution and renewed flow may have less overhead supply than an arbitrary technical pullback.

**PIT signal.** Merge T-1 `cyq_perf` with normalized daily close and base money flow. Calculate dimensionless distances such as `(close - cost_50pct) / cost_50pct` and spread `(cost_85pct - cost_15pct) / cost_50pct`. Select stocks near the cost center, with a relatively compact spread, non-extreme `winner_rate`, positive flow rank, and adequate liquidity. Rank within industry/date; do not treat `cost_15pct` as “15%”.

**Entry.** Next 09:30, diversified basket of the strongest flow-confirmed cost-support names.

**Exit.** Sell after visible close breaks materially below the lower cost band, reaches the upper band without continued flow, or the flow confirmation turns negative.

**Falsifies it.** The cost-band cohort does not beat a flow/return/industry-matched cohort without chip fields; exact percentile bands are unstable; or missing/late `cyq_perf` rows cause systematic survivorship by date.

**One-file sketch.** Project five chip columns only, reject nonpositive costs, form ratios, winsorize/rank by trade date, merge T-1 flow and universe industry, and generate targets from the intersection rather than filling missing chip data.

## 6. 龙虎榜机构净买延续

**Mechanism.** Institutional-seat net buying may carry different information from a generic attention-triggering Dragon-Tiger listing. It must be tested separately from hot-money following.

**PIT signal.** For T-1 `top_inst`, aggregate `net_buy` by stock/day after inspecting `side`, `reason`, and `exalter`; require positive net buying and preferably multiple distinct institutional rows. Confirm the sign with one stock-day `top_list.net_amount` and normalize CNY flows by T-1 daily amount. If repeated `top_list` reasons carry the same market aggregate, use one consolidated value rather than summing duplicates.

**Entry.** Next 09:30, sparse small basket, excluding extreme closed high boards unless that condition is deliberately tested as a separate interaction.

**Exit.** Event horizon of one to two sessions; retain only if a fresh visible institution/flow confirmation appears. Do not wait for a long technical stop to redefine the event.

**Falsifies it.** It does not beat (a) all positive-net `top_list` stocks and (b) same-date matched non-list stocks; open-price slippage consumes the effect; or results depend on duplicate-row handling.

**One-file sketch.** Filter the union twice, aggregate `top_inst` and consolidate `top_list` independently, merge only after each table has one stock-day row, attach daily turnover, and produce a low-capacity event basket. Keep `hm_list` and delayed `hm_detail` out.

## 7. 跨平台热榜升温

**Mechanism.** Rising attention can amplify short-lived demand, but attention is not necessarily positive sentiment and may reverse. Requiring independent THS/DC rank confirmation reduces one-platform noise.

**PIT signal.** Within the latest fully visible T-1 date and each exact `data_type`, calculate earliest-to-latest rank improvement from `rank_time`; lower rank is better. Candidate forms:

- final top rank plus positive rank improvement within one platform;
- intersection of THS hot-stock and DC popularity/rise cohorts at their latest visible snapshots;
- moderate, non-limit T-1 return so the test is not merely “buy yesterday's limit-up”.

Rank `hot` only within the same THS list/time; never compare opaque heat levels across platforms.

**Entry.** Next 09:30, 5–10 liquid names, with an explicit attention-only baseline and an optional flow-confirmed variant tested later.

**Exit.** Sell when latest visible rank leaves the retained band, improvement reverses, or after a short declared horizon. Do not call a missing row negative sentiment without checking list coverage.

**Falsifies it.** Rank improvement adds nothing over final rank; cross-platform confirmation adds nothing within common coverage; a same-return/turnover/size attention-matched control performs equally; or direction flips by fold.

**One-file sketch.** Deduplicate exact rank snapshots, sort by source/list/code/time, use group `first`/`last`, outer-report source coverage but inner-join only for the declared confirmation test, and merge T-1 daily state before ranking.

## 8. 行业资金扩散轮动

**Mechanism.** Sustainable retail themes should show broad stock-level flow diffusion inside an industry, not only one headline leader or a vendor board-index jump.

**PIT signal.** Join T-1 2020+ stock `moneyflow` to T-1 daily amount and the PIT `universe.l1_code`. For each SW level-1 industry compute:

- `sum(10,000 * net_mf_amount) / sum(daily_amount_CNY)`;
- fraction of eligible members with positive flow;
- fraction of members with positive T-1 return and limit-up/broken-board breadth.

Select top industries with both positive aggregate flow and broad diffusion. Inside them, choose liquid stocks with positive own flow but moderate prior return. In a separate 2023-12+/2025+ sample, compare with `moneyflow_ind_dc` or THS board-flow ranks; do not name-join vendor concepts without proving the mapping.

**Entry.** Next 09:30, cap names per industry and total industry weight to avoid disguising a single-industry bet as diversification.

**Exit.** Rebalance when the industry's aggregate flow/diffusion falls below the market median or a short weekly-like horizon expires; daily invocation can recompute, but unnecessary daily churn is a cost finding.

**Falsifies it.** Industry aggregation does not beat stock flow alone, diffusion adds no value over aggregate flow, returns disappear after industry-neutral matched controls, or vendor confirmation only reflects late-sample coverage.

**One-file sketch.** Read one T-1 cross-section plus a small flow-history tail, merge the immutable PIT industry map, aggregate numerator and denominator separately, rank industries, then rank stocks only inside selected industries with per-industry caps.

## 9. 本地文本催化 + 关注确认

**Mechanism.** A newly published, stock-linked corporate catalyst may become tradable only when local attention/flow confirms it. This replaces live forum scraping with frozen announcements/news/IR evidence; it does not claim to reconstruct forum mood.

**PIT signal.** Read a bounded recent `text_index` tail and keep rows with nonempty `ts_codes` from code-linked datasets such as announcements, IR Q&A, research reports, or earnings forecasts. Start with a tiny auditable event dictionary, for example positive candidates (`回购`, `增持`, `中标`, `获批`, `订单`) and hard negative vetoes (`减持`, `立案`, `终止`, `风险提示`). Inspect selected bodies for negation and context; a title keyword alone is only a candidate. Require T-1 hot-rank improvement or positive flow confirmation.

**Entry.** Next 09:30 after `available_at`, only when the event remains within a declared 1–3 visible-trading-day age and the stock code is unambiguous.

**Exit.** At event-age expiry, on negative follow-up text, or when attention/flow confirmation disappears. Never keep a position because a model invents uncited background.

**Falsifies it.** Each event class fails its same-date/industry/size/flow-matched control; keyword precision is poor on a reviewed sample; results depend on duplicated reposts; or large-table reads violate the inference budget.

**One-file sketch.** Read projected index columns with a date predicate, split/explode confirmed `ts_codes` formatting, deduplicate by `text_id` and earliest publication, inspect only selected body shards, assign explicit event classes with veto precedence, merge visible hot/flow confirmation, and calculate age from visible trading dates rather than wall-clock days.

## Minimal comparison matrix

Every playbook should report its own mechanism ablation:

| Playbook | Required ablation/control |
| --- | --- |
| 情绪修复 | same first-board cohort without market-state gate |
| 流价背离 | weak-price matched cohort; flow-only and divergence variants |
| 炸板修复 | matched non-`Z` cohort and close-to-close versus open-to-open |
| 龙头分歧 | same board height without reopen/seal filters |
| 筹码承接 | same flow/return cohort without chip fields |
| 机构净买 | all `top_list`, positive-net non-institution, and non-list matches |
| 热榜升温 | final rank only, rank change only, and cross-platform intersection |
| 行业扩散 | stock flow only, aggregate sector flow only, and diffusion variant |
| 文本催化 | each event class separately; no pooled keyword score |

If an ablation performs the same or better, keep the simpler mechanism. A broad MA overlay is not a valid rescue.
