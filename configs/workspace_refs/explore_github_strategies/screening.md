# Retrieval and screening notes

This screen favors compact formulas with a direct mapping to the project's PIT files. A public repository's backtest result is not treated as evidence of profitability.

## Why the old pack did not force exploration

The old pack mixed 12 seeds, framework adapters, platform APIs, large factor collections, no-license material, and promotional strategy documents. Its README also pointed to a workspace-nested output path rather than the official sibling `output/main.py`. In `ref_github_strats`, later folds repeatedly ran parent-identical small-cap code or reverted inactive variants; the final line was still a small-cap rotation descended from the old references.

This pack therefore:

- narrows the primary set to four mechanisms;
- maps every mechanism to actual project datasets and units;
- makes optional-domain absence explicit;
- demotes already-used Alpha101 and small-cap families;
- vendors only small MIT-licensed formula excerpts;
- requires standalone Validations before overlays or threshold scans.

## Kept ideas

### 1. Earnings-forecast announcement drift

**Public code/research.** `fkchaos/a-share-quant-sim` contains an MIT-licensed earnings-preview signal with positive/negative event decay. `zyukyunman/vortex_quant` publicly describes a forecast-surprise drift rule using `ann_date`, `p_change_min`, `p_change_max`, and `type`, delayed to the next trading day. Vortex has no license visible in the inspected tree, so no Vortex code is copied and none of its reported returns are accepted.

**Local mapping.** `fundamentals`, filtered to `dataset == "forecast_vip"`; relevant fields include `ts_code`, `available_at`, `ann_date`, `first_ann_date`, `end_date`, `type`, `update_flag`, `p_change_min`, and `p_change_max`. Forecast percentages are source percent units.

**PIT screen.** Keep. The project explicitly makes every forecast version visible at its own `ann_date`; `first_ann_date` is not an availability floor. Filter on `available_at` before selecting a latest event. At `08:30`, an event released after the prior decision enters only after its recorded visibility and no earlier than the following tradable session.

**Dropped source behavior.** DB access, `datetime.now()`, data downloads, platform scheduling, performance claims, open-price assumptions, and live handoff code.

### 2. Holder accumulation minus announced unlock pressure

**Public code.** `fkchaos/a-share-quant-sim/core/event_factors.py` implements time-decayed insider net buying and forward unlock pressure. Its database and field names are discarded; only the event arithmetic is retained under MIT terms.

**Local mapping.** `events` rows for:

- `stk_holdertrade`: `available_at`, `ann_date`, `in_de`, `change_vol`, `change_ratio`, `avg_price`;
- `share_float_complete`: `available_at`, `ann_date`, `float_date`, `float_share`, `float_ratio`.

The local unit contract verifies holder share counts and unlock share counts; ratios are source percent. `total_share` in holder trades is the holder's post-trade position, not company capital, so it must not be used as a company-size denominator.

**PIT screen.** Keep conditionally. Every event is filtered by `available_at`. A future `float_date` is legitimate only when the schedule was already announced and visible. A row whose missing announcement date was conservatively delayed to its event date creates no advance signal.

**Dropped source behavior.** SQLite, implicit wall-clock time, global max normalization, unannounced forward schedules, real-time dividend downloads, and any claim that insider activity is proven alpha.

### 3. Volume-weighted five-day reversal

**Public code.** `crisq-star/alpha-lab-cn/src/factors/momentum.py` provides MIT-licensed classic and information-weighted reversal formulas for A-shares.

**Local mapping.** `daily`: adjusted close from `close * adj_factor`, normalized `vol`, `trade_date`, `amount`, `is_suspended`, `up_limit`, and `down_limit`. The core formula is:

```text
r_t = adjusted_close_t / adjusted_close_(t-1) - 1
w_t = volume_t / sum(volume, 5)
score = -sum(w_t * r_t, 5)
```

**PIT screen.** Keep. All inputs are daily and end at T-1 for an `08:30` decision. Use a weekly or slower stateless gate because a daily five-day cross-section can turn its gross signal into costs.

**Dropped source behavior.** Factor framework classes, AkShare ingestion, claims based on the repository's sample, forward-return analysis, and package dependencies.

### 4. EP plus abnormal-turnover value/sentiment

**Public code/research.** Alpha Lab CN implements MIT-licensed positive EP and negative 20-day turnover. `yaoqi1018/Reproduction-of-the-Four-Factor-Model-in-China/build_factors.py` reproduces a China four-factor construction with annual EP, abnormal turnover, and a smallest-30%-market-cap exclusion. No license was found for the reproduction repository, so its code is not vendored.

**Local mapping.** `daily`: `pe_ttm`, `turnover_rate`, `circ_mv`, `trade_date`, and normal tradability fields. Use positive `pe_ttm` only. The long-horizon variant is:

```text
EP = 1 / pe_ttm
abnormal_turnover = mean(turnover_rate, 20) / mean(turnover_rate, about 252)
score = rank(EP) + rank(-abnormal_turnover)
```

**PIT screen.** Keep. The decision uses only T-1 daily rows. Monthly rebalancing matches the slow signal. Remove the smallest market-cap tail or neutralize within coarse size buckets to prevent another pure-size strategy.

**Dropped source behavior.** Full factor-model portfolio construction, future monthly returns, unlicensed source code, and any assumption that a year-end statement is available before announcement.

## Retained only as controls

`lvlh2/alpha101` is MIT-licensed and directly implements WorldQuant Alpha #6, #12, #33, and #101. The formulas are easy to map to daily OHLCV, but #6/#12/#101 already appeared in the old pack and the previous experiment's lineage. They are useful for checking rolling/groupby operators or as a deliberately weak comparison, not as the first new Fold hypothesis.

## Rejected material

| Material | Decision | Reason |
| --- | --- | --- |
| CTBZ/JoinQuant small-cap variants | Drop | Already dominated the previous lineage; some source logic filters with next-day/open information unavailable at `08:30` |
| CST EP + low-vol + momentum | Drop from this pack | Already present in the old pack and accompanied by promotional backtest numbers and platform code |
| RSRS, ETF momentum, Alligator, VMACD | Drop from primary set | Old-pack ideas; ETF/index availability is configuration-dependent and these do not address parent-clone drift |
| Full Alpha101/Alpha191 collections | Drop | Too broad for one per-decision inference budget; encourages formula shopping and repeated composite rewrites |
| `a-share-quant-sim` simulator/strategy adapters | Drop | Framework and execution boilerplate, despite an MIT license |
| Alpha Lab CN factor platform | Drop | Only the formulas are useful; database, analysis, ML, LLM, dashboard, and orchestration are out of scope |
| Vortex strategy stack | Do not vendor | No license found in the inspected tree; reported results are not independently verified |
| Four-factor reproduction code | Do not vendor | No license found; retain only the public formula description and conservative lag idea |
| vn.py, Qlib, RQAlpha, Backtrader and similar | Forbidden | Full platforms, not strategy logic for this ABI |
| Same-day minute/limit-board entries | Drop | The strategy does not receive an intraday clock; `08:30` cannot use today's minute or daily close information |
| Any `datetime.now()` event query | Drop | Wall-clock leakage; use `context.inference_at` and `available_at` |

## Pinned public sources

- Alpha Lab CN, commit `44f599819d1735170f3730b8bc98623b8d22036b`, MIT: <https://github.com/crisq-star/alpha-lab-cn/tree/44f599819d1735170f3730b8bc98623b8d22036b>
- A-share Quant Sim, commit `47d789c4f3e8977f755fcb9e492d452f49d7c8b8`, MIT: <https://github.com/fkchaos/a-share-quant-sim/tree/47d789c4f3e8977f755fcb9e492d452f49d7c8b8>
- Alpha101, commit `b6bd3f4fb53fdf62a9f8454c477561651f764952`, MIT: <https://github.com/lvlh2/alpha101/tree/b6bd3f4fb53fdf62a9f8454c477561651f764952>
- China four-factor reproduction, commit `fc90168b6bac41eacc709c859b45069c154e8ef5`, no license found: <https://github.com/yaoqi1018/Reproduction-of-the-Four-Factor-Model-in-China/tree/fc90168b6bac41eacc709c859b45069c154e8ef5>
- Vortex Quant forecast-drift archive, no license found: <https://github.com/zyukyunman/vortex_quant/blob/main/docs/%E5%9B%A0%E5%AD%90%E7%A0%94%E7%A9%B6/%E5%AE%9E%E9%AA%8C%E6%A1%A3%E6%A1%88/%E4%B8%9A%E7%BB%A9%E9%A2%84%E5%91%8A%E6%BC%82%E7%A7%BB%E5%9B%A0%E5%AD%90.md>

The superseded `github_strategies` pack has been retired; no file from it is copied into this pack.
