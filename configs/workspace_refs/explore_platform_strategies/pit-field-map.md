# PIT field map

Confirm the current run's `data_summary.json`, manifest, Parquet metadata, and unit reference before using any item below. The inventory is sampled and a configured dataset can still be absent or empty in one fold.

## Runtime read shape

The rolling view exposes Parquet part directories. A one-file strategy should project only needed columns, bound the date window, and filter the union by `dataset` immediately:

```python
# Sketch only; not a complete strategy.
events = pd.read_parquet(
    context.asof_dir + "/events",
    columns=["dataset", "available_at", "trade_date", "ts_code", "net_mf_amount"],
)
events = events[
    (events["dataset"] == "moneyflow")
    & (pd.to_datetime(events["available_at"]) <= pd.Timestamp(context.inference_at))
]
```

Other rolling paths are `context.asof_dir + "/daily"`, `"/universe"`, `"/text_index"`, and `"/text_library"`. The text index has `text_id`, `dataset`, `ts_codes`, `title`, `available_at`, and `library_file`; body shards have `text_id` and `body`. Do not read an arbitrary host path.

A module cache may be keyed by `context.asof_version`, but results must be correct after a cold restart. Account-dependent targets are recomputed on every call.

## Datasets that directly support the retained playbooks

| Dataset | Useful fields | Earliest local/configured coverage | PIT at an 08:30 decision | Main use |
| --- | --- | ---: | --- | --- |
| normalized `daily` | `trade_date`, `ts_code`, OHLC, `amount`, `pct_chg`, turnover/market-value fields if confirmed | run-dependent | same-day bar is not visible; use T-1 and earlier | returns, liquidity, execution-size reference, matched controls |
| `universe` | `ts_code`, `name`, `l1_code`, `l1_name` | decision snapshot | fixed as of the decision date | PIT stock identity and SW level-1 industry; do not backfill today's industry into history |
| `limit_list_d` | `limit` (`U/D/Z`), `limit_times`, `open_times`, `first_time`, `last_time`, `fd_amount`, `amount`, `industry`, `pct_chg`, `turnover_ratio`, `float_mv` | 2020-01 | trade-date 16:00, therefore T-1 is normally visible after the evening node | closed boards, broken boards, breadth, streak height, seal quality |
| `moneyflow` | `net_mf_amount`, buy/sell amount and volume buckets | 2020-01 | trade-date 19:00, therefore T-1 is normally visible | broad-history stock flow and flow/turnover ratios |
| `moneyflow_dc` | `net_amount`, `net_amount_rate`, bucket rates | 2023-12 locally (`docs` note source begins 2023-09-11) | trade-date 19:00 | later-window vendor confirmation |
| `moneyflow_ths` | `net_amount`, `net_d5_amount`, bucket rates | 2025-01 | trade-date 19:00 | later-window vendor confirmation only |
| `cyq_perf` | `cost_15pct`, `cost_50pct`, `cost_85pct`, `weight_avg`, `winner_rate` | 2018-01 | other event panel, normally after the evening refresh | chip-cost support and crowding |
| `top_list` | `net_amount`, `l_buy`, `l_sell`, `amount`, `net_rate`, `turnover_rate`, `reason` | 2020-01 | trade-date 20:00 | Dragon-Tiger aggregate event and matched controls |
| `top_inst` | `exalter`, `buy`, `sell`, `net_buy`, `side`, `reason` | 2020-01 | trade-date 20:00 | institutional-seat event aggregation |
| `ths_hot` | `rank`, `rank_time`, `hot`, `data_type`, `concept`, `rank_reason` | 2020-01 locally | source `rank_time`; intraday/after-close snapshots already published by T-1 may be visible | attention rank and within-day rank change |
| `dc_hot` | `rank`, `rank_time`, `data_type`, `concept` | 2020-01 locally | source `rank_time` | independent attention rank and rise-list confirmation |
| `text_index` + `text_library` | index fields above; body text in shards | 2020-01 locally | dataset-specific publication time plus text refresh node | local announcements/news/IR evidence; not forum comments |

## Important timing traps

1. **08:30 is before today's auction.** `stk_auction` is stamped 09:29. A default 08:30 strategy cannot condition on today's opening gap, bid amount, weak-to-strong auction, or one-word limit state.
2. **KPL/ladder rows are not safe at exactly 08:30.** `kpl_list` and `kpl_concept_cons` are stamped next-day 08:30; `limit_step` and `limit_cpt_list` use a conservative date-EOD stamp. Their real pre-open landing node is ready about 08:55. At an exact 08:30 invocation, derive breadth and height from visible `limit_list_d` history instead of assuming these rows exist.
3. **Hot lists are attention snapshots, not live sentiment.** `rank_time` may provide several T-1 observations. Deduplicate by `(dataset, trade_date, data_type, ts_code, rank_time)` and compare within the same platform/list type. Do not use T-1 `pct_change` as if it were today's price.
4. **`hm_detail` is too late for a clean next-day follow at 08:30.** It starts in 2022-08 and has no trusted publication clock, so it receives conservative trade-date EOD timing; under the evening node, T-1 is not reliably available at the next 08:30. `hm_list` has no historical PIT timestamp and is excluded from the default snapshot. Do not rebuild a current-name hot-money backfill.
5. **Vendor board maps start late.** `moneyflow_ind_dc` starts 2023-12; `moneyflow_ind_ths`, `moneyflow_cnt_ths`, `dc_index`, and `dc_member` start in 2025 locally/configurationally. For long folds, aggregate 2020+ stock `moneyflow` by the PIT `universe.l1_code`; use vendor board flows only as a later-window robustness check.
6. **`limit_list_d.limit_amount` is forbidden.** Its history is unstable and the snapshot masks it. `fd_amount` remains available, but only as a T-1 day-end descriptive field, never as executable queue depth.
7. **Lists can have legitimate duplicate keys.** Aggregate `top_list`/`top_inst` by the intended event identity or by stock/day after inspecting `reason` and `side`; never use `drop_duplicates(ts_code)` blindly.
8. **Text is large and fallible.** First read a bounded index tail with projected columns. Use body shards only for selected `text_id` values. Handle duplicates, negation, multiple stock codes, and conservative publication timestamps.

## Unit traps

| Tuple | Unit/semantics |
|---|---|
| normalized `daily.amount` | CNY |
| `moneyflow.*_amount`, `moneyflow_dc.*_amount`, `moneyflow_ths.*_amount` | 10k CNY; multiply by 10,000 before dividing by normalized daily CNY amount |
| `moneyflow_ind_dc.net_amount` | CNY |
| THS industry/concept `net_amount` | 100m CNY |
| `limit_list_d.amount`, `fd_amount`, `float_mv`, `total_mv` | CNY |
| `limit_list_d.pct_chg`, `turnover_ratio`; vendor flow rates; `winner_rate` | percent-number, not the normalized daily decimal convention |
| `cyq_perf.cost_*`, `weight_avg` | CNY/share; the suffix is a cost-distribution percentile, not a percentage |
| `top_list` and `top_inst` money fields | CNY |
| hot-list `rank` | ordinal rank; `hot` is an opaque vendor score and should be ranked only within source/list/time |

Prefer cross-sectional ranks and dimensionless ratios. Never compare same-named columns across union datasets without the full `(file, dataset, column)` unit identity.

## What the repository does not expose for these ideas

- No live forum posts, comment counts, likes, influencer graph, or scraped 雪球/股吧 sentiment.
- No paid Level-2 order book, queue position, cancellation stream, or executable seal-order depth.
- No dedicated Northbound individual holding/change series in the default event, macro, or text snapshot. `moneyflow`, `moneyflow_dc`, and `moneyflow_ths` are not Northbound proxies.
- No same-call Broker feedback, partial fills, or market-impact model. A limit-up buy can be rejected; `fd_amount` cannot make it fill.
