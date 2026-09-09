# 08:30 字段图

先核对本轮 `data_summary.json`、manifest 和单位表；下表是仓库常见合同，某一折仍可能缺表或为空。本包比前几轮多用了三个 opt-in 的 macro 数据集（`fut_daily`/`fut_mapping`/`fut_basic`）与两个期权表（`opt_daily`/`opt_basic`），先在普查里确认它们在本折确实非空。

## 读法

```text
daily    = pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])
fund     = pd.read_parquet(context.asof_dir + "/fundamentals", columns=[...]); 先按 dataset 过滤，再按 available_at 过滤
macro    = pd.read_parquet(context.asof_dir + "/macro", columns=[...]);        同上，用 dataset 列区分 index_daily / fut_* / opt_*
universe = pd.read_parquet(context.asof_dir + "/universe")   # 决策日冻结：ts_code、name、list_date、exchange、market、l1_code/l1_name
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；不要把 `asof_dir` 读失败回退到 `snapshot_dir`。`macro` 是多来源并集，同一列名在不同 `dataset` 下含义不同（`close` 在 `index_daily` 是指数点、在 `fut_daily` 是合约报价、在 `opt_daily` 是权利金），必须先按 `dataset` 切片再用列。

## 数据集

| 数据 | 本包所需字段 | 08:30 可见边界与单位 |
| --- | --- | --- |
| 合并日线 `daily` | `open/high/low/close`、`amount`（元）、`pct_chg`（小数）、`adj_factor`、`turnover_rate`（小数）、`circ_mv`/`total_mv`（元）、`pe_ttm`/`pb`/`ps_ttm`（无量纲倍数）、`dv_ttm`/`dv_ratio`（小数）、`up_limit`/`down_limit`、`is_suspended` | `daily` 行 17:30、`daily_basic` 行 18:00 才可见，`adj_factor` 09:30 盖章，08:30 一律只能用 T-1 及更早。估值与市值列由 `daily_basic` 合并进同一文件，不需要单独取数 |
| `fundamentals.fina_indicator_vip` | `roe`/`roe_waa`/`roe_dt`（百分数）、`debt_to_assets`（百分数）、`eps`/`bps`（元/股）、`grossprofit_margin`、`netprofit_yoy` | 按 `ann_date` 18:00 可见；同一 `(ts_code, end_date)` 取 `available_at` 最新版本，跨 `end_date` 取 `ann_date` 最近一期。**本仓库没有 `q_netprofit_yoy` 列**；`impai_ttm` 被快照剔除 |
| `fundamentals.dividend` | `div_proc`、`cash_div_tax`/`cash_div`（元/股）、`stk_div`（每股送转股数）、`ann_date`、`imp_ann_date`、`ex_date`、`record_date`、`end_date` | 行级可见时间取 `imp_ann_date`，缺则 `ann_date`，当日 18:00。已可见行里的 `ex_date` 可以读，但不能用它去看未公告的安排。同一次派现在 `预案`/`股东大会通过`/`实施` 三个阶段各有一行，只有 `实施` 行带 `ex_date` |
| `macro.index_daily` | `000300.SH`/`000905.SH`/`000852.SH` 的 `close`（指数点）、`pct_chg`（百分数，5% 存成 5.0） | `contract_1730_from:trade_date`（数据日 17:30），T-1 可见、与日线同步（见 README 的可见边界一条）；仓库只落七只核心指数，覆盖自 2020 年起 |
| `macro.fut_daily` | `ts_code`、`close`/`settle`（指数点）、`vol`/`oi`（手）、`oi_chg` | `contract_1730_from:trade_date`（数据日 17:30，快照构建时覆盖原始层的 23:59:59），回放里 T-1 可见，见 README。CFFEX 的 `IF/IH/IC/IM` 每日各 8 行：4 个实际月合约 + `X.CFX`/`XL.CFX`/`XL1`/`XL2`/`XL3` 连续代码 |
| `macro.fut_mapping` | `ts_code`、`mapping_ts_code` | 同上。连续代码是指针：`XL.CFX`/`X.CFX`→当月、`XL1`→次月、`XL2`→当季、`XL3`→下季，实测连续行的 `close`/`oi` 与被指向合约逐行完全相等 |
| `macro.fut_basic` | `ts_code`、`fut_code`、`d_month`、`list_date`、`delist_date`、`last_ddate`、`multiplier` | 注册表，按 `list_date` 盖章，决策快照豁免窗口下限（全生命周期可见）。只有实际月合约有 `delist_date`；五个连续代码的 `delist_date`/`multiplier` 为空，剩余期限必须先经 `fut_mapping` 换成实际合约再查。IM 最早 `list_date` 2022-07-22 |
| `macro.opt_daily` | `ts_code`、`close`/`settle`（权利金）、`vol`/`oi`（张） | 同 `fut_daily`，回放里 T-1 可见。SSE 自 2015-02-09、SZSE 与 CFFEX 自 2019-12-23；只落金融期权（ETF + 股指），无商品期权 |
| `macro.opt_basic` | `ts_code`、`opt_code`、`call_put`、`exercise_price`、`maturity_date`、`per_unit`、`list_date` | 注册表，同 `fut_basic` 的可见规则。`opt_code` 形如 `OP510050.SH`/`OP000852.SH`；`exercise_price` 与标的同量纲（ETF 是元/份，股指是指数点） |
| `universe` | `ts_code`、`name`、`list_date`、`exchange`、`market`、`l1_code`/`l1_name` | 决策日冻结，`name` 是决策日在册名称（ST 判定只用它）。行业中性只用它，不回填今天的行业 |

## 复权、截面与状态

推断时冻结前复权锚：`anchor = adj_factor(T-1)`，`qfq(t) = raw(t) * adj_factor(t) / anchor`，收益用 qfq 收盘；成交额不复权。个别代码的 `adj_factor` 在稀薄成交期会来回跳变，不要假定它逐日非降。

截面处理顺序：有限值 → 方向统一 → 1%/99% 截尾 → 行业中性（对 `universe.l1_code` 哑变量回归取残差）与规模中性（对 `log(circ_mv)` 回归取残差）→ rank（`2*(rank(pct=True)-0.5)`）或 z-score → 家族内等权或 `fit` 合成。截面只含当时已上市、未退市、T-1 未停牌、值有限且有可见源行的股票。

状态变量（家族 5、6）在 `fit(context)` 里算好月度季节均值与基准窗的均值/标准差并写入 `context.state_dir`，`generate_orders` 只读；季节均值只用输入窗估计。macro 域窗口默认 24 个月（约 480–490 个交易日），基准窗因此上限 250 个交易日，可见观测不足 250 时不出状态。

## 执行

- 最长回看：日线约 60 个真实交易观测（ADV20 与除权跳变检查），macro 约 250 个（状态基准窗）。先按 `ts_code` 取尾窗，再留最新截面；模块级缓存只键控 `context.asof_version`，冷启动必须得到相同订单。
- 卖先于买，按 `context.account.cash` 快照本地递减预算，留 3% 费用缓冲，买入向下取整到 100 股，不把未成交卖出当成已到账。30 只篮子的单只预算 3,333 元，实际权重是 `floor(3333/(100*P))*100*P`，欠配现金要汇报。
- 涨跌停：买单成交价 ≥ 当日 `up_limit` 拒单、卖单 ≤ `down_limit` 拒单，当日限价 08:30 不可见，只能用 T-1 代理估计；拒单是结果，汇报它。
- 停牌：`is_suspended=True` 的 Bar 一律拒单，且该标志含复牌日；「真停牌」按日线缺行判断。60 日持有期内出现长停牌时，仓位按 Broker 的实际状态汇报，不要在策略里假设已卖出。
- 除权：Broker 在除权日按已实施派现贷记现金红利、按 `pre_close` 结算股数变化，策略不要自己再调仓位或再算一遍红利。

## 常见失败

- 用当日日线、估值、复权或财报 `end_date`；用未公告派现的 `ex_date` 排期。
- 直接对负 `pe_ttm` 取倒数排序，把巨亏公司排到微亏公司前面。
- 把 `dv_ttm` 的缺失当成零股息填 0；把 `index_daily.pct_chg` 的百分数当成小数。
- 在 `macro` 并集上不按 `dataset` 切片就取 `close`。
- 用连续合约代码去 `fut_basic` 查交割日（那几行是空的），或用当月合约算年化基差（剩余期限 20 天出头时年化被放大数倍）。
- 用全样本分位当状态阈值（前视），或在验证区间上重估季节均值。
- 在 `generate_orders` 里每天全量重读全历史并重算滚动量。
