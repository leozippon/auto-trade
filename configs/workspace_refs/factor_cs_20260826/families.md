# 五个可分离家族

不要展开成 202 或 42 因子菜单。每个正式候选通常只留 4–6 个代表项。
`C/O/H/L/V` 是截至 T-1 的冻结前复权价与成交量；`r = C/C.shift(1)-1`。
方向是合成前乘号，使高分倾向做多；单家族 Validation 可以证伪或翻转。
缺值、零分母、无穷值保持为空，不填成 0。

## 1. Value+Quality

| 名称 | 紧凑定义 | 方向 |
| --- | --- | ---: |
| `earnings_to_price` | T-1 `pe_ttm` 正且有限时 `1/pe_ttm` | + |
| `book_to_market` | T-1 `pb` 正且有限时 `1/pb` | + |
| `roe` | 最新已公告 `fina_indicator_vip.roe` | + |
| `roa` | 最新已公告 `roa` | + |
| `grossprofit_margin` | 最新已公告毛利率 | + |
| `debt_to_assets` | 最新已公告资产负债率 | - |

估值优先用 T-1 `daily_basic`，减少报表拼接。先规模中性，另试排除最小市值尾部，避免壳价值主导。
`pe_ttm` 与财务比率量纲不同，必须各自标准化后再合成。

## 2. Reversal+Low-Risk

| 名称 | 紧凑定义 | 方向 |
| --- | --- | ---: |
| `reversal_5d` 或 `reversal_21d` | `-(C/C.shift(n)-1)`，二者只留一个 | + |
| `return_std_21d` 或 `return_std_63d` | `std(r, n)` | - |
| `beta_60d_000300` | `cov(r, r_m, 60)/var(r_m, 60)`，`r_m` 为可见沪深 300 收益 | - |
| `high_low_63d` | `max(C,63)/min(C,63)` | - |

低流动性只作资格惩罚，不把它解释成独立溢价。不要与同周期趋势项对冲后自称稳健。

## 3. Growth

| 名称 | 紧凑定义 | 方向 |
| --- | --- | ---: |
| `or_yoy` | 最新已公告营业收入同比 | + |
| `netprofit_yoy` | 最新已公告净利润同比 | + |
| `ocf_yoy` | 最新已公告经营现金流同比 | + |
| `roe_yoy` | 最新已公告 ROE 同比 | + |

只留覆盖稳定者。亏损跨零时同比会爆炸，先去极值。字段按公告 `available_at` 取最后可见版本，不用 `end_date`。

## 4. Medium-term Trend

| 名称 | 紧凑定义 | 方向 |
| --- | --- | ---: |
| `return_63d` | `C/C.shift(63)-1` | + |
| `return_126d` | `C/C.shift(126)-1` | + |
| `return_252d_skip21d` | `C.shift(21)/C.shift(252)-1` | + |
| `macd_hist` | `EMA(C,12)-EMA(C,26)-EMA(DIF,9)`，由冻结 qfq 重算 | + |

去冗余后选 2–3 个。这是中期价格趋势假说，必须单独跑；不要与 `reversal_21d` 同时加分。

## 5. 稀疏 Alpha101 算子（可选控制）

只转述公开算子，不粘贴第三方完整实现。出处见 `sources.md`。

| 控制 | 机制转述 | 方向 |
| --- | --- | ---: |
| `#1` close-vs-delay | 近端收益为负时用约 20 日收益波动，否则用 `close`；对该序列做平方后在约 5 日窗上取 `Ts_ArgMax`，再截面 rank 减 0.5。测的是 close 相对短 delay 窗的位置，不是又一个累计收益。 | 论文为中性 rank；本地先按高分做多，允许 Validation 翻转 |
| `#6` 量价相关 | `-corr(open, volume, 10)` | + |
| `#12` 量价变动 | `sign(Δvolume) * (-Δclose)` | + |
| `#101` 日内位置 | `(close-open)/(high-low+ε)`，ε 取约 `1e-3` 且与价格单位一致 | + |

这些是算子控制，不是 101 条工厂。若与反转/趋势高度同向，只保留更简单的那一项。
窗口必须落在 T-1 冻结 qfq 上；不要读供应商最新日 `*_qfq` 快照。

## 明确不做

- 不读 `factor_value`、供应商 `CrossSectionalRank`、当日 `stk_factor*`。
- 不把 500–1320 日超长窗、五年 PEG、未公告分红实施日写进信号。
- 不在今天的股票/行业全集上重做历史截面。
- 不把本目录或任何 vendor 树拷进 `output`。
