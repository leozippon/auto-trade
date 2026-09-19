# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，不得由前者推断后者。

## 本轮选入的数据

| 域 | 数据集 | 行级 `available_at` |
|---|---|---|
| `daily` | 日线、每日指标、复权因子、涨跌停价、停牌（合成一张 30 列的表） | 无列，可见性由 as-of 视图给定 |
| `universe` | 股票基本信息与申万一级成员（`name`、`list_date`、`l1_code`、`l1_name`） | 无列，决策日冻结 |
| `fundamentals` | `income_vip`、`balancesheet_vip`、`cashflow_vip`、`fina_indicator_vip`、`forecast_vip`、`express_vip`、`dividend`、`fina_audit`、`fina_mainbz_vip`、`disclosure_date` | 有 |
| `macro` | `cn_gdp`、`cn_cpi`、`cn_ppi`、`cn_pmi`、`cn_m`、`sf_month`、`shibor`、`shibor_lpr`、`index_daily`、`index_dailybasic`、`sw_daily`、`fut_basic`、`fut_mapping`、`fut_daily`、`opt_basic`、`opt_daily`、`cb_basic`、`cb_daily`、`cb_call` | 有 |
| `events` | `margin`、`margin_detail`、`margin_secs`、`moneyflow`、`cyq_perf`、`bak_daily`、`block_trade`、`stk_holdernumber`、`stk_holdertrade`、`new_share`、`share_float_complete`、`top_list`、`top_inst`、`limit_list_d`、`kpl_list`、`top10_floatholders`、`report_rc` | 有 |
| `text` | 只有 `report_rc`（研报标题索引与正文分片） | 有 |

不在本轮：分钟线与集合竞价（执行只能用 09:30 与 15:00）、`intraday_flow`、`report_rc` 之外的文本、开始得太晚的供应商变体、没有 `available_at` 的 `index_weight`。

## 读法与可见性

```text
daily = pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])
fund  = pd.read_parquet(context.asof_dir + "/fundamentals", columns=["dataset", "available_at", ...],
                        filters=[("dataset", "in", [...]), ("ann_date", ">=", start)])
```

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。
- `fundamentals`、`macro`、`events`、`text` 是多个数据集的列并集：**先按 `dataset` 过滤，再谈字段与单位**；同名列在不同数据集里含义与单位不同。
- 带行级 `available_at` 的域，判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `ann_date`/`trade_date` 推；各域的时间戳字符串格式不同，不要按字符串比较。`ann_date`/`trade_date` 只用来下推过滤、限定读取窗口。
- `daily` 与 `universe` 没有 `available_at` 列：as-of 视图只含决策时刻之前可见的行，08:30 决策时 `daily` 的最新一行就是 T-1；按不存在的列过滤会报错。
- 常见的行级规则：财务报表与预告快报在公告日 18:00 可见，`moneyflow` 在交易日 19:00，`macro` 的日频行情（指数、期货、期权、可转债）在交易日 17:30；逐数据集的规则以 manifest 的 `availability_rules` 为准。
- 同一 `(ts_code, end_date)` 的财务报表可能有多个版本（首次公告、修订、重述）。选「首版」（事件口径）还是「时点生效版」（状态口径）是设计决定，要在 `hypothesis` 里写明；重述版不能当作新事件。

## 申万一级行业是点内的

`universe.l1_name` 不是按当前归属回填的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021，该日之前的决策日用冻结的 SW2014 成分，之后用 SW2021。研究期跨这个切换点，所以：

- **一级行业的个数与名字在研究期内会变**（SW2014 二十八个、SW2021 三十一个，个别名字也重排过）。每个决策日在当天的截面上重算行业，普查表按决策日分别列；**跨决策日比较行业名单是错的**。
- `l1_name` 为空的名字（上市窗口边界、供应商回填缺口）落在 `未分类` 这一个桶里。宿主侧的行业归因用的是同一列与同一个 `未分类` 标签，所以你在篮子里数出来的行业构成与提名门读的 `stats.benchmark.top_industry_weight` 说的是同一件事——只差后者是**时间加权**的。
- 申万成分历史由供应商回填，纳入日期可能异常（有名字的纳入日早于其上市日），**不能当作可靠的行业变更日**；只用「决策日归属」。

## 单位（最容易错的几处）

| 字段 | 单位 |
|---|---|
| `daily` 的价格 | 元/股，未复权；`adj_factor` 是累计复权因子 |
| `daily.vol`、`amount`、`circ_mv`、`total_mv` | 股、元、元、元（已从原始湖的手、千元、万元归一） |
| `daily.pct_chg`、`turnover_rate` | 小数 |
| `macro.index_daily.pct_chg` | **百分数**（0.68 = 0.68%），÷ 100 |
| `macro.sf_month` 的 `inc_month`/`inc_cumval` | **亿元**；`stk_endval` 是**万亿元** |
| `macro.cn_m` 的 `m0`/`m1`/`m2` | **亿元**；`*_yoy`、`*_mom` 是百分数 |
| `macro.shibor`、`macro.shibor_lpr` | 百分数（3.0 = 3.0 %） |
| `events.moneyflow` 的 `*_amount` | **万元** |
| `events.report_rc.np` | **万元**；`quarter` 是财年标签，不是季度 |
| `fundamentals` 的报表金额 | 元，年初至今累计（资产负债表是时点值） |
| `fina_indicator_vip` 的增速与比率 | 百分数 |

## 数据下限

研究期第一个决策视图里，`fundamentals`、`macro` 与 `events` 从 2020 年初开始，`daily` 回溯五年。由报表自己算的同比与 TTM 从 2021 年春季起才可算，八个季度的 SUE 要到 2022 年；三年训练窗口的学习型模型在第一个研究年里，财务、宏观与事件列只在后一半左右的训练日上有值。第一个研究年的读数单列汇报。

## 陷阱

- **整手与板块**：100 股整手；科创板 200 股起、北交所 100 股起后可 1 股递增。起步包剔除两者。收盘价 ≤ 30 元使一手 ≤ 3,000 元；30 只时单只预算约 3,200 元，最低佣金 5 元即 16 bp/边。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp），早期研究年的往返成本更高。
- **`n_income` 含少数股东损益**，归母净利润是 `n_income_attr_p`。
- **解禁面会把同一笔解禁重复计数**：逐年重复公告（股数按股本变动重述、日期有时挪一天）与两次下载并集里的重复行，逐行相加能把单个持有人的一笔持仓读成总股本的 169.9 %。规则是「一个持有人在窗口里只算一行」——取最新公告，其中股数最大，并列取比例最小。
- **月度宏观面整整滞后一个月才可见**：`sf_month`、`cn_m` 与 `cn_cpi`/`cn_ppi`/`cn_pmi`/`cn_gdp` 的行级 `available_at` 按「月末 + 31 天」保守盖章，某个月的数要到次月末之后才进得了决策截面；按 `month` 列推可见性会把整整一个月的前视带进来。这些序列在同一个决策日对全池取同一个值，截面上只能通过各个名字各自的敏感度或暴露产生区分度。
- **`circ_mv` 在原始湖里是万元**，快照里已归一为元：离线普查直接读原始湖会差 1 万倍，而它通常同时是池的截取列与规模中性化列，一处错会静默地把两件事同时弄反。
- **对照不得对自己的面中性化**；训练面板与决策截面必须由同一个函数构建。
