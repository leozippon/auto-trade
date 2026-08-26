# 计算与 PIT

先读本轮 `data_summary.json` 与单位表。下表是常见边界，不是数据合同。

| 域 | 本包所需 | 08:30 边界 |
| --- | --- | --- |
| `daily` | OHLCVA、`adj_factor`、`daily_basic` 估值/换手/市值、涨跌停与停牌 | 最晚 T-1；`daily` 17:30、`daily_basic` 18:00、`adj_factor` 09:30 |
| `fundamentals` | `fina_indicator_vip` 等公告版本 | 按 `available_at`，不用 `end_date` |
| `macro` | `index_daily` 沪深 300 | 只连已可见指数收益 |
| `universe` | 当时上市/退市与可见行业 | 当时截面；不用当前名单回填 |

没有因子库日值。左连接缺值不能填 0。

## 总闸与复权

`record.available_at <= context.inference_at`。
08:30 不得使用当日日线、当日估值或当日复权因子。

推断时冻结前复权锚：

```text
anchor = adj_factor(T-1，推断时最后可见)
qfq_price(t) = raw_price(t) * adj_factor(t) / anchor
return(t) = qfq_close(t) / qfq_close(t-1) - 1
```

不要用供应商当前 `*_qfq` 历史快照。成交额不复权。价格、量、额、市值先核对本轮单位。

## 财务版本

1. 先筛 `dataset == "fina_indicator_vip"`，再解释 `roe` 等列；union 同名空列不是 0。
2. 再滤 `available_at <= inference_at`，按 `(ts_code, end_date)` 取最后可见公告/修订；同一时点优先有效 `update_flag`。
3. 窗口不足、字段缺失或单位未知则留空。
4. `roe/roa/*margin/*_yoy` 通常是百分数，`assets_turn` 是倍数；日频换手/股息率多为小数。各自标准化后再合成。
5. `end_date` 只表示报告期。不要跨版本补字段，不要把四个 YTD 累计报表直接相加。

## 截面

推荐：有限值 → 方向统一 → 1%/99% 去极值 → 可选规模/行业中性 → rank 或 z-score → 家族内等权。

- rank：`2 * (rank(pct=True) - 0.5)`
- z-score：`(x-mean)/std`，标准差为 0 则整列为空
- 规模中性：对同截面 `log(total_mv)` 回归取残差
- 行业中性：只用本次 PIT universe；2021-12-13 前后申万口径切换，缺失必须显式处理
- 相关去重：家族内 Spearman 绝对值长期高于约 0.85 时留更简单、覆盖更高的一项

截面只含当时已上市、未退市、值有限且有可见源行的股票。

## 执行

- 只读已确认列；parts 目录可直接 `pandas.read_parquet`。
- 最长约 253 个真实交易观测加少量 EMA 预热。先按股票滚动量，再留最新截面。
- 缓存只键控 `context.asof_version`。冷启动与热缓存必须得到相同订单。
- 订单排序稳定；预算留费用/滑点；用可见涨跌停、停牌和最低流动性过滤。
- Broker 决定 T+1 与成交。策略按当时现金快照本地递减预算，卖先于买，但不把未成交卖出当成已到账。

## 常见失败

- 08:30 使用当日日线、估值、复权或日终技术指标。
- 用财报 `end_date` 代替公告可见时间。
- 读取 `factor_value` 或供应商截面 rank。
- 把百分数与小数、手与股、千元与元混算。
- 把缺失填成中性 0。
- 硬编码 workspace/refs 或宿主路径。
- 把上一轮父策略或 202 因子菜单整段搬进本折。
