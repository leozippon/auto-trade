# 08:30 字段图

先核对本轮 `data_summary.json`、manifest 和单位表；下表是仓库常见合同，某一折仍可能缺表或为空。

## 读法

```text
daily  = pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])
events = pd.read_parquet(context.asof_dir + "/events", columns=[...]); 先按 dataset 过滤，再按 available_at 过滤
fund   = pd.read_parquet(context.asof_dir + "/fundamentals", columns=[...]); 同上
macro  = pd.read_parquet(context.asof_dir + "/macro", columns=[...]); dataset == "index_daily"
universe = pd.read_parquet(context.asof_dir + "/universe")   # 决策日冻结：ts_code、name、list_date、l1_code/l1_name
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；不要把 `asof_dir` 读失败回退到 `snapshot_dir`。

## 数据集

| 数据 | 本包所需字段 | 08:30 可见边界与单位 |
| --- | --- | --- |
| 合并日线 `daily` | `open/high/low/close`、`vol`（股）、`amount`（元）、`pct_chg`（小数）、`adj_factor`、`turnover_rate`（小数）、`circ_mv`/`total_mv`（元）、`pe_ttm`、`up_limit`/`down_limit`、`is_suspended` | 当日行 17:30 才可见，08:30 只有 T-1 及更早；`adj_factor` 当日 09:30 盖章，同样只能用 T-1 |
| `events.moneyflow` | `buy_lg_amount`、`sell_lg_amount`、`buy_elg_amount`、`sell_elg_amount`、`net_mf_amount` | 交易日 19:00 可见，T-1 可用；金额单位万元，除以元计的 `amount` 前先乘 1e4；结构性缺北交所 |
| `events.margin_detail` | `rzye`（融资余额，元）、`rzmre`、`rzche`、`rqye` | 行级 `available_at` 为下一日 09:00，08:30 只能用 **T-2** 及更早；只覆盖两融标的 |
| `events.cyq_perf` | `weight_avg`、`cost_15pct`/`cost_50pct`/`cost_85pct`（元/股）、`winner_rate`（百分数） | 2018 年起，晚间节点后 T-1 可见；覆盖约 91%，缺行不能填 0 |
| `fundamentals.fina_indicator_vip` | `netprofit_yoy`（无单季 `q_netprofit_yoy`）、`q_sales_yoy`、`roe`（百分数） | 按公告日 18:00 可见；同一 `(ts_code, end_date)` 取 `available_at` 最新版本 |
| `fundamentals.express_vip` | `n_income`（本期净利润，元）、`yoy_net_profit`（上年同期净利润水平，元——列名有误导，不是百分数）、`ann_date` | 每个版本按自身 `ann_date` 可见；同比要自己算 `n_income / yoy_net_profit - 1` |
| `fundamentals.forecast_vip` | `type`、`p_change_min`、`p_change_max`、`ann_date` | 同上；`first_ann_date` 不是可见时间 |
| `macro.index_daily` | `000300.SH` 的 `pct_chg`（百分数）、`close` | T-1 可见；用于 β 与 `ivol` |
| `universe` | `ts_code`、`name`、`list_date`、`l1_code`/`l1_name` | 决策日冻结；行业中性只用它，不回填今天的行业 |

## 复权与截面

推断时冻结前复权锚：`anchor = adj_factor(T-1)`，`qfq(t) = raw(t) * adj_factor(t) / anchor`，收益用 qfq 收盘；成交额不复权。

截面处理顺序：有限值 → 方向统一 → 1%/99% 截尾 → 规模中性（对 `log(circ_mv)` 回归取残差）与 β 中性（对 60 日 β 回归取残差）→ rank（`2*(rank(pct=True)-0.5)`）或 z-score → 家族内等权或 `fit` 合成。截面只含当时已上市、未退市、T-1 未停牌、值有限且有可见源行的股票；行业中性只用本次 PIT `universe`。

## 执行

- 最长回看约 253 个真实交易观测（`abn_turnover` 的 250 日均值）；先按股票取尾窗，再留最新截面；模块级缓存只键控 `context.asof_version`，冷启动必须得到相同订单。
- 卖先于买，按 `context.account.cash` 快照本地递减预算，留 3% 费用缓冲，买入向下取整到 100 股，不把未成交卖出当成已到账。
- 涨跌停：买单成交价 ≥ 当日 `up_limit` 拒单、卖单 ≤ `down_limit` 拒单，当日限价 08:30 不可见，只能用 T-1 代理估计；拒单是结果，汇报它。
- 停牌：`is_suspended=True` 的 Bar 一律拒单，且该标志含复牌日；「真停牌」按日线缺行判断。

## 常见失败

- 用当日日线、估值、复权或财报 `end_date`。
- 把 `moneyflow` 的万元与 `daily.amount` 的元混算；把百分数与小数混算。
- 把缺失填成中性 0，使缺数据的股票进篮子。
- 在 `generate_orders` 里每天全量重读全历史并重算滚动量。
- 把前几轮的价值/反转/成长打分或 202 因子菜单搬进本折。
