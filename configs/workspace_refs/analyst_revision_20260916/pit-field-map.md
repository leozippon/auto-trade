# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`；下表是原始层实测的合同，某一折仍可能缺表或为空。`data_summary.json` 里 `events` 的 `datasets` 列出本轮实际加载的数据集，只有列在其中的才可读。

## `report_rc` 数值切片的确定 schema

数值切片已落地（commit `65edfae`，`src/autotrade/environment/data/snapshot.py` 与 `unit_rules.py`）。它是 events 域的一个数据集，**默认不加载**：`report_rc` 在 `SELECTABLE_DATASETS["events"]` 里而不在 `DEFAULT_DATASETS["events"]` 里，本臂靠创建实验时的 `events_datasets` 显式选入。文本域的 `report_rc` 仍默认加载，只带标题。

`events.parquet` 中 `dataset == "report_rc"` 的列，以及 `unit_reference.json` 里将出现的单位标签：

| 列 | 类型 / 单位标签 | 说明 |
|---|---|---|
| `ts_code`、`org_name`、`author_name`、`report_date`、`create_time`、`quarter`、`report_type`、`classify`、`rating` | 字符串 | `quarter` 登记为 `datetime` 语义并注明是财年；`rating`/`classify` 登记为 `categorical` |
| `op_rt`、`op_pr`、`tp`、`np` | `10k_CNY` | 营业收入 / 营业利润 / **利润总额** / 净利润，全部万元 |
| `eps`、`max_price`、`min_price` | `CNY_per_share` | 每股收益与目标价区间上下沿 |
| `pe`、`ev_ebitda` | `multiple` | 券商给出的**前瞻**倍数，与 `daily_basic.pe_ttm` 不是同一个量 |
| `rd`、`roe` | `percent` | **`rd` 是预测股息率不是研发费用**；`17.9` 就是 17.9%，不要再除以 100 |
| `available_at`、`available_at_rule` | 时间戳 / 字符串 | 规则名 `source:create_time` 与保守回退 `conservative_from:report_date`（22:00） |

**切片里没有的列**：`report_title`（留在文本域，与数值行同夜放行）、`name`、`imp_dg`（全库全空，单位表登记为 `unknown` 并说明「从不填充」）。本包对目标价与评级变动两条的排除结论不受影响。

单位表把 `tp` 登记为 `10k_CNY` 并给出 `np/eps == daily_basic.total_share` 的核验，与本包实测 `tp/np` 中位 1.175 一致：**`tp` 是利润总额，不是目标价**。

## 读法

```text
events   = pd.read_parquet(context.asof_dir + "/events",
              columns=["dataset", "available_at", "ts_code", "report_date",
                       "org_name", "quarter", "eps", "np", "rating"])
events   = events[(events["dataset"] == "report_rc")
                  & (pd.to_datetime(events["available_at"]) <= pd.Timestamp(context.inference_at))]
daily    = pd.read_parquet(context.asof_dir + "/daily", columns=[...],
                           filters=[("trade_date", ">=", start)])
universe = pd.read_parquet(context.asof_dir + "/universe")     # ts_code、name、list_date、l1_code/l1_name
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；不要把 `asof_dir` 读失败回退 `snapshot_dir`。

事件域是多个数据集的列并集：`report_rc` 的行**没有 `trade_date`**（该列来自 `moneyflow` 等按交易日的数据集），日期只有 `report_date`（字符串 `YYYYMMDD`）。先按 `dataset` 过滤，再取该数据集自己的日期列。

## 本包用到的数据集

| 数据 | 本包所需字段 | 08:30 可见边界与单位 |
|---|---|---|
| `events.report_rc` | `ts_code`、`report_date`、`org_name`、`author_name`、`quarter`、`eps`（`CNY_per_share`）、`np`/`tp`（`10k_CNY`）、`rating`、`report_type`、`classify`、`create_time` | 行级 `available_at`：`create_time` 与 `report_date` 相差 −1..+3 天时取 `create_time`（秒级），否则按 `report_date` 当天 22:00 保守盖章；两种盖章都早于夜间文本节点 23:15，因此**到 T-1** |
| 合并日线 `daily` | `close`、`amount`（元）、`turnover_rate`（小数）、`circ_mv`（元）、`adj_factor`、`pe_ttm`、`up_limit`/`down_limit`、`is_suspended` | 当日行 17:30 才可见，08:30 只有 T-1 及更早；`adj_factor` 当日 09:30 盖章 |
| `macro.index_daily` | `000300.SH` 的 `close`、`pct_chg`（百分数，先除以 100） | `contract_1730_from:trade_date`，与日线同步到 T-1；只用于基准与 β |
| `universe` | `ts_code`、`name`（含 ST 标记）、`list_date`、`l1_code`/`l1_name` | 决策日冻结；行业与 ST 只从这里取 |

## 盖章规则与它的年份断层

`report_rc` 有两条 `available_at_rule`，全窗（`report_date ≤ 2025-12-31`，1,407,073 行）分别是 `source:create_time` 866,779 行（61.6%）与 `conservative_from:report_date:implausible_create_time` 540,294 行（38.4%）。**它们不是随机混合，而是按年份分层的**：

| `report_date` 年份 | `source:create_time` | 保守盖章 |
|---|---|---|
| 2020 | 0.0% | 100.0% |
| 2021 | 0.0% | 100.0% |
| 2022 | 58.8% | 41.2% |
| 2023 | 99.8% | 0.2% |
| 2024 | 100.0% | 0.0% |
| 2025 | 97.2% | 2.8% |

原因见数据文档 §4：历史回填里的 `create_time` 常常是 TuShare 的采集时间而不是发布时间（实测 `create_time - report_date` 的 p75 是 248 天、最大 852 天，只有 62.4% 的行落在 −1..+3 天的可信区间）。写入规则已按 §3.3 回退到 `report_date` 当天 22:00，PIT 方向保守、正确。

后果，逐条：

- **「秒级 PIT 时钟」只从 2023 年起成立。** 2020–2021 的每一行都是 22:00 的日级保守盖章（实测 `create_time` 的秒位在 98.7% 的行上非零，但那是采集时刻的秒，不是发布时刻的秒）。任何「盘前发布 vs 盘后发布」的分层在输入窗落在 2020–2021 的折上不可测，不要造。
- **到 T-1，不是 T-2。** 本切片受**夜间文本节点** `cn_nightly_text_full`（23:15）门控，而不是工作日的晚间落库节点；18:00 与 22:00 的盖章都在 23:15 之前，所以 `report_rc` 的数值行与日线一样在次日 08:30 可见，并且**与同一份研报的标题同夜一起放行**。只有 1,908 行（0.136%）盖章晚于 23:15，那些是 T-2。`available_at` 的小时分布：18 时 710,055 行、22 时 551,714 行、21 时 129,322 行，其余零散。
- **保守盖章不会造成前视，但会造成「同日成批」。** 一天里所有保守盖章的行时刻完全相同（22:00:00），按时间排序时大量并列，取「窗口内每券商最新一条」必须有确定的并列打破规则（例如再按 `report_date`、`org_name` 排序），否则同一份数据在两次运行里可能给出不同的篮子。

## 财年语义：本方向最容易错的一个字段

`quarter` 是**财年**标签，不是日历季度。实测：1,389,730 行（98.8%）以 `Q4` 结尾，`Q3`/`Q2`/`Q1` 合计 12,814 行，另有 1,532 行是裸的 `"Q"`、1 行是 `"Q5"`。`2024Q4` = FY2024 全年预测。

一份研报同时给出多个财年，`quarter` 年份减 `report_date` 年份的分布：

| 偏移 | 行数 |
|---|---|
| −1（去年，追溯口径） | 78,100 |
| 0（当前财年） | 497,225 |
| +1 | 468,652 |
| +2 | 357,631 |
| ≥ +3 | 936 |

`(ts_code, org_name, report_date, report_title)` 分组的行数：3 行 398,817 组（最常见）、2 行 58,952 组、4 行 9,695 组、1 行 10,605 组。**一份研报 = 一组行，不是一行。**

## 重复、改写与陈旧

- **业务键不唯一，必须自己去重。** 声明的业务键是 `(ts_code, report_date, org_name, author_name, quarter)`，**它不唯一**：联合层只丢弃逐字节相同的行，而供应商的重推带着不同的 `create_time` 与 `eps`。实测重复率 0.919%（12,936 行、12,838 个键组），其中 11,829 组的 `create_time` 不同、1,812 组的 `eps` 真的变了。**规则：先按 `available_at <= inference_at` 过滤，再对每个业务键取 `create_time` 最大的一条。** 不去重就会把同一份预测的新旧两个版本一起算进一致预期。
- **同一份研报改期重发**：`(ts_code, org_name, report_title)` 里有多个 `report_date` 的组占 0.48%。属于低频噪声，去重后不需要额外处理，但不要用 `report_title` 当业务键。
- **预测陈旧是常态**：同券商同股票同财年的相邻两次预测中位间隔 54 个自然日（p25 21、p75 85），其中 31.23% 的 `eps` 完全没变（上调 25.89%、下调 42.88%，n = 318,315）。因此「最近 20 日的一致预期」里有相当一部分是券商把旧数字重发了一遍。家族 3 的 `flat` 计数就是为量出这件事而存在的；家族 1/2 用「窗口内每券商最新一条」而不是「所有行的均值」也是为此。
- **`W0` 无预测 ≠ 预测为零。** 前者不进截面，后者才参与排序。

## 覆盖偏斜与季节性

- 逐年覆盖股票数 2,516（2020）→ 3,499（2023）→ 3,380（2025）；逐年券商 90–107 家，全窗 144 家，前十家（国泰君安、申万宏源证券、中信证券、中金、华泰证券、中信建投、国信证券、天风证券、长江证券、东吴证券）占 34.70% 的行。
- 逐月行数中位 15,767（区间 5,048–48,018），逐月覆盖股票数中位 1,460（区间 679–2,462）。旺季是 4/5 月与 8–11 月，淡季是 1/2/6/7/12 月，约 3 倍摆动。
- 20 日窗口内每股中位覆盖券商数只有 1–2 家，P90 在 3–14 家之间随季节摆动。
- 直接后果（48 个月末实测，家族 1 口径「两窗各 ≥2 家券商」）：无过滤 318–1,226 只（中位 715）→ 加 ADV20 ≥ 3,000 万元并剔除科创板/北交所 265–979 只（中位 584）→ 再加 T-1 收盘 ≤ 40 元 180–807 只（中位 408），**11/48 个月末跌破 300**。其他家族的同口径覆盖见 `families.md`。

## 代码后缀

`report_rc` 的北交所行用的是 `.BJ` 后缀（`92xxxx.BJ` 263 个代码，另有 `900xxx.BJ` 5 个、`83xxxx.BJ` 2 个），**没有** `stk_surv` 那种「北交所旧编码带 `.SZ`」的问题（实测 0 行）。但 `900xxx.BJ` 与 `83xxxx.BJ` 这 7 个代码未必能对上 `daily`/`universe` 的现行代码，join 后消失属正常。本包无论如何都剔除北交所（10 万元账户买不起一手），所以这一项只影响普查计数，不影响候选。

## 单位

- `eps`：CNY/股。当前财年行中位 1.12、p01 −0.25、p99 13.25、最大 80.48。
- `np`：万元。当前财年行中位 89,300（= 8.93 亿元）。用 `np * 1e4 / eps` 反推股本，中位 8.48 亿股（p05 1.02 亿、p95 128.7 亿），与 A 股股本量级一致——这是本地对两列单位的交叉验证，`np` 与 `eps` 不得混算。
- `tp`：**利润总额（万元），不是目标价**。实测 `tp/np` 中位 1.175（p05 1.014、p95 1.699）。
- `min_price`/`max_price`：目标价区间下沿/上沿，CNY/股。`min_price` 非空率 29.72%（`min_price/close` 中位 1.264、p05 1.085、p95 1.615），`max_price` 非空率 0.79%。
- `pe`：券商给出的**前瞻** PE（中位 17.0），与 `daily_basic.pe_ttm` 不是同一个量，不要互相替代或相除。
- `rd`：预测**股息率**（`percent`），不是研发费用；非空率 23.81%。其余列非空率：`op_rt` 85.45%、`roe` 81.14%（`percent`）、`ev_ebitda` 59.25%、`op_pr` 10.95%。
- `imp_dg`：全库 0 行非空，已被排除出切片。
- `daily.turnover_rate`/`pct_chg` 是小数，`amount`/`circ_mv` 已归一为元；`macro.index_daily.pct_chg` 是百分数。

## 分类列

- `report_type`：点评 981,526、非个股 223,481（15.88%）、一般 113,374、深度 77,050、新股 2,593、调研 1,179、会议纪要 47。**`非个股` 的行仍然带具体 `ts_code` 与 `eps`（99.25% 非空）**——那是行业/主题研报里对成分股给出的预测，按 `report_type` 过滤会砍掉六分之一的数据。要过滤就先量出过滤前后的覆盖与读数差。
- `classify`：一般报告 1,205,601、首次关注 102,270、首份报告 68,923、首次评级 30,279。首次覆盖类合计约 14.3%，它是覆盖度家族的抓手——而覆盖度家族在本臂只能作对照。
- `rating`：48 个不同字符串，前几位是 买入 710,640、增持 276,926、无 107,886、推荐 99,122、强烈推荐 48,859、优于大市 47,466、跑赢行业 44,703，另有 `BUY`/`Buy`/`OVERWEIGHT`/`OUTPERFORM` 等英文写法。15 个词的序数映射覆盖当前财年行的 90.56%；剩下的近一成必须留空而不是猜。

## 复权与截面

推断时冻结前复权锚：`anchor = adj_factor(T-1)`，`qfq(t) = raw(t) * adj_factor(t) / anchor`，收益用 qfq 收盘；成交额不复权。`920627.BJ` 在 2022 年的 `adj_factor` 逐日反复跳变（数据文档 §4），本包剔除北交所，不受影响，但计算复权收益时仍不得假设 `adj_factor` 逐日非降。

截面只含当时已上市、未退市、T-1 未停牌、值有限且有可见源行的股票。

## 执行

- 最长回看约 91 个自然日的事件窗加 60 个交易日的日线控制窗；先按股票取尾窗，再留最新截面；模块级缓存只键控 `context.asof_version`，冷启动必须得到相同订单。
- 卖先于买，按 `context.account.cash` 快照本地递减预算，留 3% 费用缓冲，买入向下取整到 100 股。
- 涨跌停：买单成交价 ≥ 当日 `up_limit` 拒单，当日限价 08:30 不可见，只能用 T-1 代理估计；拒单是结果，汇报它。
- 停牌：`is_suspended=True` 的 Bar 一律拒单，且该标志含复牌日；「真停牌」按日线缺行判断。

## 常见失败

- 不按财年切片，把 FY、FY+1、FY+2 的 `eps` 混在一起做窗口均值。
- 用 `report_date` 或 `create_time` 判可见。
- 把 `tp` 当目标价，或把 `imp_dg` 当评级变动标志（全空）。
- 不去重就对行做均值：业务键不唯一，同一份预测的新旧版本会被一起算进一致预期。
- 把 2020–2021 输入窗的 22:00 保守盖章当成真实发布时刻，据此做日内分层。
- 保守盖章行的时刻完全相同，取「每券商最新一条」时没有确定的并列打破规则，导致同一份数据两次运行给出不同篮子。
- 淡季可选股票数掉到 180–280 只时，靠把券商家数门槛从 ≥2 降到 ≥1 把候选池撑回 300 只，却仍按家族 1 汇报。
- 混单位：`np` 万元、`eps` CNY/股、`pe` 前瞻而非 TTM、`index_daily.pct_chg` 百分数。
- 在 `generate_orders` 里每天全量重读事件域并重算全历史滚动量。

## 本轮快照默认没有的东西

- 研报正文。文本域只保留标题与截断正文，`context.nl` 问不出研报观点。
- `broker_recommend`（券商金股）可见时点是月末后 31 天，不能当月度信号；本包不用。
- 精确开盘竞价与分钟线（`include_intraday=false`）：成交只有 09:30 与 15:00 两个时点。
