# 计算与 PIT 卡片

## 本仓库实际可用输入

| PIT 域 | 本包所需内容 | 使用边界 |
| --- | --- | --- |
| `daily` | `daily` OHLCVA、`adj_factor`、`daily_basic` 的换手/估值/市值、涨跌停和停牌状态 | 默认 08:30 最晚 T-1；各源覆盖不完全，左连接缺值不能填成 0 |
| `fundamentals` | `income_vip`、`balancesheet_vip`、`cashflow_vip`、`fina_indicator_vip` 等公告版本 | 保留源单位和多版本；必须先按 `dataset` 过滤 |
| `macro` | `index_daily` 的沪深 300 日线 | 与股票日期内连接，只用已可见指数收益 |
| `universe` | 当时上市/退市状态及可用历史行业 | 只用于当时截面；不能用当前股票列表回填 |

仓库没有因子库 `factor_value` 日值，也没有把 `stk_factor`/`stk_factor_pro` 作为正式沙箱输入。
日线、复权因子、每日指标、财务报表和财务指标已经落为 PIT 数据；Fold 应从这些原料重算分数。

## 盘前日线与复权

总闸始终是 `record.available_at <= context.inference_at`。
本项目保守时间为：`daily` 17:30、`daily_basic` 18:00、`adj_factor` 09:30；因此默认 08:30 三者都只能用 T-1。

在推断时冻结前复权锚：

```text
anchor = adj_factor(T-1，推断时最后可见)
qfq_price(t) = raw_price(t) * adj_factor(t) / anchor
return(t) = qfq_close(t) / qfq_close(t-1) - 1
```

不要读取供应商当前 `*_qfq` 历史快照：它以供应商“最新交易日”锚定，不能证明与本次 T-1 PIT 冻结一致。
仓库 `daily.pct_chg` 已归一为小数且基于除权昨收，可作为收益交叉检查；多日价格位置、均线和 Alpha101 仍建议统一由冻结 qfq OHLC 重算。
成交额不复权；价格、量、额和市值的绝对运算先核对本次单位清单。

## 财务版本

1. 从 `fundamentals` 先筛 `dataset == "fina_indicator_vip"`，再解释 `roe` 等列；union 表中的同名空列不是 0。
2. 再次保证 `available_at <= inference_at`，按 `(ts_code,end_date)` 选择最后可见公告/修订，同一时点优先有效 `update_flag`，不要只按文件或报告期取最后一行。
3. 每只股票从已去版本重复的记录中取最新可见报告期；窗口不足、字段缺失或单位未知则留空。
4. `fina_indicator_vip` 的 `roe/roa/*margin/*_yoy` 通常以百分数保存，`assets_turn` 是倍数；daily 的换手/股息率已经归一为小数。每个因子分别标准化后再合成。
5. 若改用三张报表，利润表和现金流量表通常是年内累计口径。TTM 必须先把累计值还原为单季，再滚动四个已公告季度；资产负债表取时点值，不能四季相加。
6. `end_date` 只表示报告期，不能决定可见性；同一公告时点也不能跨版本补字段，避免拼出从未发布过的报表。

估值四项优先从 T-1 daily 中的 `pe_ttm/pb/ps_ttm/dv_ttm` 构造，既减少报表拼接错误，也仍受 T-1 PIT 约束。

## 截面处理

推荐顺序：有限值过滤 → 方向统一 → 1%/99% 去极值 → 可选规模/行业中性化 → rank 或 z-score → 家族内等权。

- rank：`2 * (rank(pct=True) - 0.5)`，对重尾、百分数/小数差异更稳健。
- z-score：`(x - mean(x)) / std(x)`，只在去极值后使用，标准差为 0 时整列为空。
- 规模中性：以同截面的 `log(total_mv)` 回归取残差。
- 行业中性：只用本次 PIT universe 暴露；2021-12-13 前后申万口径切换，2021-12-10 附近和新股行业覆盖偏薄，缺失必须显式处理。
- 相关去重：同一家族 Spearman 绝对相关长期高于约 0.85 时保留覆盖更高、计算更简单的一项。

不要在全历史股票全集上排名；当时未上市、已退市、无有限因子值或无可见源行的股票都不进入截面。

## 执行与性能

- 只读 `context.asof_dir + "/daily"`、`"/fundamentals"`、`"/macro"` 中已确认列；parts 目录可由 pandas 直接读取。
- 最大价格窗口约 253 个真实交易观测，再留少量 EMA 预热；不要每次加载全历史、全列或对每个日期重复全截面排序。
- 一次调用先按股票做滚动量，再只保留最新截面；家族分数共用同一特征表。
- 缓存只键控 `context.asof_version`；依赖 inference time、账户和 bars 的值仍逐次重算。冷启动和热缓存必须得到相同订单。
- 订单排序显式稳定，预算留手续费/滑点缓冲，并用可见涨跌停、停牌与最低流动性条件过滤。

## 常见 PIT 失败

- 08:30 使用当日日线、daily_basic、adj_factor 或当日日终技术指标。
- 用供应商最新日 qfq 回写整段历史，或用未复权 close 计算跨除权窗口收益。
- 用财报 `end_date` 代替公告可见时间，或把四个 YTD 累计报表直接相加。
- 读取 `factor_value`、供应商截面 rank，或在今天的股票/行业全集上重做历史截面。
- 把不同数据集同名列、百分数与小数、手与股、千元与元直接混算。
- 把缺失、零分母或无穷值填成看似中性的 0，导致覆盖漂移被掩盖。
- 模块缓存没有 `asof_version`，让上一推断时点数据泄露到下一次。
- 策略硬编码 workspace/refs 或宿主绝对路径，而不是使用 `context.asof_dir`。
