# 42 个可重算候选

## 约定

`C/O/H/L/V/A` 是截至 T-1 的冻结 qfq 价、成交量和成交额，`r=C/C.shift(1)-1`，`Turn` 是归一化换手率，`MV/FMV` 是总市值/流通市值，`r_m` 是同日沪深 300 收益。
`方向` 是合成前乘号，使高分倾向做多；它只是先验，单家族 Validation 可以证伪或翻转。
滚动窗口必须有足量真实观测；缺值、零分母和无穷值保持为空，不向前补未来。

## 量价候选

| 家族 | 名称 | 紧凑定义 | 方向 |
| --- | --- | --- | ---: |
| Momentum | `return_21d` | `C/C.shift(21)-1` | + |
| Momentum | `return_63d` | `C/C.shift(63)-1` | + |
| Momentum | `return_126d` | `C/C.shift(126)-1` | + |
| Momentum | `return_252d_skip21d` | `C.shift(21)/C.shift(252)-1` | + |
| Momentum | `close_to_ma_20d` | `C/mean(C,20)-1` | + |
| Momentum | `macd_hist` | `EMA(C,12)-EMA(C,26)-EMA(DIF,9)` | + |
| Momentum | `price_position_ir_60d` | `x=(C-O)/(H-L)`；`mean(x,60)/std(x,60)` | + |
| Reversal | `reversal_5d` | `-(C/C.shift(5)-1)` | + |
| Reversal | `reversal_21d` | `-(C/C.shift(21)-1)` | + |
| Reversal | `rsi14_reversal` | Wilder RSI(14) 的负值 | + |
| Risk | `return_std_21d` | `std(r,21)` | - |
| Risk | `return_std_63d` | `std(r,63)` | - |
| Risk | `downside_std_63d` | `std(min(r,0),63)` | - |
| Risk | `beta_60d_000300` | `cov(r,r_m,60)/var(r_m,60)` | - |
| Risk | `sigma_126d_000300` | 126 日 `r=a+b*r_m+e` 的 `std(e)` | - |
| Risk | `high_low_63d` | `max(C,63)/min(C,63)` | - |
| Risk | `sharpe_60d` | `mean(r,60)/std(r,60)`，无风险利率取 0 | + |
| Liquidity | `avg_turnover_20d` | `mean(Turn,20)` | - |
| Liquidity | `avg_turnover_63d` | `mean(Turn,63)` | - |
| Liquidity | `bias_turn_21d_252d` | `mean(Turn,21)/mean(Turn,252)-1` | - |
| Liquidity | `std_turnover_21d` | `std(Turn,21)` | - |
| Liquidity | `sum_abs_rtn_amount_20d` | `sum(abs(r),20)/sum(A,20)` | - |
| Alpha101 | `alpha101_6` | `-corr(O,V,10)` | + |
| Alpha101 | `alpha101_12` | `sign(diff(V))*(-diff(C))` | + |
| Alpha101 | `alpha101_101` | `(C-O)/(H-L+1e-3)` | + |
| Size | `size` | `-log(MV)` | + |
| Size | `float_size` | `-log(FMV)` | + |

`return_21d` 与 `reversal_21d` 是相反假说；`size` 与 `float_size` 高度同义。每组先二选一，不能用重复投票制造虚假稳定性。
`sum_abs_rtn_amount_20d` 在这里作为交易成本惩罚，方向与“非流动性溢价”研究可能相反；必须让含成本 Validation 决定。

## 估值、财务和成长候选

这些值来自 T-1 `daily` 已合入的 `daily_basic` 字段，或 `fundamentals` 中最新已公告的 `fina_indicator_vip` 行；不是 TuShare 因子库日值，也不是供应商截面排名。

| 家族 | 名称 | PIT parquet 定义 | 方向 |
| --- | --- | --- | ---: |
| Value | `earnings_to_price` | 正且有限时 `1/pe_ttm` | + |
| Value | `book_to_market` | 正且有限时 `1/pb` | + |
| Value | `sales_to_market` | 正且有限时 `1/ps_ttm` | + |
| Value | `dividend_yield_ttm` | T-1 `dv_ttm` | + |
| Quality | `roe` | 最新已公告 `roe` | + |
| Quality | `roa` | 最新已公告 `roa` | + |
| Quality | `grossprofit_margin` | 最新已公告 `grossprofit_margin` | + |
| Quality | `netprofit_margin` | 最新已公告 `netprofit_margin` | + |
| Quality | `assets_turn` | 最新已公告 `assets_turn` | + |
| Quality | `debt_to_assets` | 最新已公告 `debt_to_assets` | - |
| Growth | `or_yoy` | 最新已公告营业收入同比字段 | + |
| Growth | `netprofit_yoy` | 最新已公告净利润同比字段 | + |
| Growth | `ocf_yoy` | 最新已公告经营现金流同比字段 | + |
| Growth | `roe_yoy` | 最新已公告 ROE 同比字段 | + |
| Growth | `assets_yoy` | 最新已公告总资产同比字段 | + |

财务字段保留源单位：比率通常是百分数，而日频比率已归一为小数。截面排名不受固定倍数影响，但跨字段加总前必须各自标准化。
不要把利润表/现金流量表的四个 YTD 累计值直接相加成 TTM；若不用 `fina_indicator_vip` 的派生字段，必须先还原单季值，再由四个已公告单季滚动求和。

## 筛掉的方向

- 不纳入 `factor_value` 日值、供应商 `CrossSectionalRank`、当日 `stk_factor*` 行或其 qfq 技术指标。
- 不纳入 500–1320 日长窗、五年 PEG/ETP、未来实施日才能知道的分红/公司行动字段；当前回看窗和 Validation 难以稳定覆盖。
- 不纳入当前公司简介、当前行业快照、日终榜单盘中字段或当日未复权快照。
- 不堆叠 202 个名称；42 个候选先按家族去冗余。组合规模没有硬上限，但每个因子都要挣到位置：有可测量的边际贡献且不与已选项高度同向。
