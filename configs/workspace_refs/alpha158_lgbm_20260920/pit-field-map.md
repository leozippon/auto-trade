# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`：`fundamentals.datasets` 必须列出 `fina_indicator_vip`、`forecast_vip`、`express_vip`、`income_vip`（缺任一，`vipf2` 组不可构造），`events.datasets` 必须列出 `moneyflow`（缺它，`mfflow` 组不可构造）；缺组时只能登记去掉该组的变体，并在结果笔记写明。本文是本包可见时点、单位、去重规则与陷阱的**唯一权威表述**。

## 读法

```text
daily = pd.read_parquet(context.asof_dir + "/daily",
                        columns=["ts_code","trade_date","open","high","low","close","vol","amount",
                                 "adj_factor","is_suspended","up_limit"],
                        filters=[("trade_date",">=",start)])       # 复核 130 个自然日；fit 3 年 + 110 日
fund = pd.read_parquet(context.asof_dir + "/fundamentals", columns=[...],
                       filters=[("dataset","in",["fina_indicator_vip","forecast_vip","express_vip","income_vip"]),
                                ("ann_date",">=",start)])           # 复核 1,100 + 400 个自然日
flow = pd.read_parquet(context.asof_dir + "/events", columns=[...],
                       filters=[("dataset","in",["moneyflow"]),("trade_date",">=",start)])   # 复核 120 个自然日
universe = pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code","name","list_date"])
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。`fundamentals` 与 `events` 是多个数据集的列并集：**必须先按 `dataset` 过滤再谈字段与单位**。`fundamentals` 与 `events` 的行带行级 `available_at`，判可见只看它（`pd.to_datetime` 后与 `context.inference_at` 比；两个域的字符串格式不同，不要按字符串比较）；`daily` 与 `universe` 没有 `available_at` 列，可见性由 as-of 视图本身给定：08:30 决策时 `daily` 的最新一行就是 T-1。`fit` 在调用它的那个决策日拿到同一个 context，所以训练永远看不到当天决策看不到的行。

## 逐表可见边界

| 数据集 | 本包所需字段与单位 | 行级 `available_at` 规则 | 08:30 能看到的最新行 |
|---|---|---|---|
| `daily` | `open`/`high`/`low`/`close`（元/股，未复权）、`vol`（股）、`amount`（元）、`adj_factor`（累计复权因子）、`is_suspended`、`up_limit` | 无列；视图按日级收盘合同构建 | T-1 |
| `universe` | `name`（ST 与退市筛选）、`list_date`（YYYYMMDD） | 无列；决策日冻结 | 当日 |
| `fundamentals.fina_indicator_vip` | `or_yoy`、`netprofit_yoy`、`dt_netprofit_yoy`、`op_yoy`（**百分数**，÷ 100） | `f_ann_date` 或 `ann_date` 当日 18:00 | 公告日 ≤ T-1 |
| `fundamentals.forecast_vip` | `type`（预增/扭亏/略增 → +1，预减/首亏/略减 → −1）、`net_profit_min`/`net_profit_max`/`last_parent_net`（**万元**） | 同上 | T-1 |
| `fundamentals.express_vip` | `n_income`、`yoy_net_profit`（**元**） | 同上 | T-1 |
| `fundamentals.income_vip` | 只用来标记某报告期的正式报表首次可见的时刻（此后该期的预告方向记 0） | 同上 | T-1 |
| `events.moneyflow` | `net_mf_amount`、`buy_elg_amount`、`sell_elg_amount`、`buy_lg_amount`、`sell_lg_amount`（**万元**，× 1e4 后与 `circ_mv`、`amount` 的元相除） | `trade_date` 当日 19:00 | T-1 |

## 去重与版本规则

- 同一 `(dataset, ts_code, end_date)` 的多个版本：在时点 t 生效的是 `available_at ≤ t` 的最新一版；`available_at` 完全相同的重复行取文件顺序的最后一行（稳定排序）。这与防御型包「只取首版」不同，是冻结产物的口径，逐字沿用。
- 增速列取 t 时点**最新可见报告期**（首次可见时刻最晚的那一期）的生效版本；没有可见报告期即 NaN，z 分数后记 0。
- 预告方向取 t 时点最新可见的一条预告；快报方向把最新可见快报与**同一报告期**的预告配对，比较快报增速与预告中值增速。
- `moneyflow` 同一 `(ts_code, trade_date)` 多行时取 `available_at` 最大的一行；窗口按自然日计（5 日 = T-5..T-1），窗口内无行即 NaN。
- 前复权锚定在窗口内每个名字的最后一根 K 线：价格、成交量乘以 `adj_factor / 最后一根的 adj_factor`；成交均价 = `amount / vol` 用原始值（复权因子在比值里抵消）。

## 数据下限（第一个研究年要特别读）

研究期第一个决策视图里，`fundamentals` 与 `events` 只从 2020 年初开始（`fina_indicator_vip` 最早公告日 2020-01-03，`moneyflow` 最早交易日 2020-01-02），日线则回溯五年。所以第一个研究年的 `fit`（训练窗口三年）里，`vipf2` 与 `mfflow` 列只在后一半左右的训练日上有值，前一半全部记 0；到第三个研究年，训练窗口才完全落在下限之后。第一个研究年的读数更接近纯 Alpha158 模型：单独汇报，不据此增删特征组。

## 陷阱

- **两种 `available_at` 格式**：`fundamentals` 是 `2020-01-07T18:00:00+08:00`，`events` 是 `2020-01-02 19:00:00+08:00`；只按解析后的时间戳比较。
- **单位**：`fina_indicator_vip` 的增速是百分数；`forecast_vip` 的净利润是万元、`express_vip` 是元——快报方向在各自表内自洽地算增速，从不跨表混用金额；`moneyflow` 的金额是万元。
- **`income_vip` 的旧期行**：少数行的 `ann_date` 比 `available_at` 早数年（旧报告期落湖晚），`ann_date` 下推过滤多留 400 天余量，只可能漏掉晚到一年以上的公告。
- **标签只在训练里出现**：`O[t+11]` 必须已在 T-1 之前可见；最后 11 根 K 线没有标签，验证集之前隔 10 个交易日。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；池子用「T-1 有 K 线」判断可交易。
- **北交所**在读取时剔除，训练与决策一致；科创板在池子里剔除（200 股起买），但仍参与训练与 z 分数统计。
