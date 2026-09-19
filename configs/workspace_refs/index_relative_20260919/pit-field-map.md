# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

本轮与别的轮次只差一个数据集：**`macro.index_weight`**——七只核心指数的月末成分与权重。它是本臂存在的理由，所以本文把它放在最前面。

## 成分截面 `macro.index_weight`（本轮新挂，本臂的股票池）

```text
sec = pd.read_parquet(
    context.asof_dir + "/macro",
    columns=["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"],
    filters=[("dataset", "=", "index_weight"), ("index_code", "=", "000300.SH"),
             ("trade_date", ">=", start)],           # start = 决策日往回 120 个自然日
)
sec = sec[pd.to_datetime(sec["available_at"], utc=True)
          <= pd.Timestamp(context.inference_at).tz_convert("UTC")]
latest = sec["trade_date"].max()                      # 日期不晚于决策日的那一张
members = sec[sec["trade_date"] == latest]            # con_code 就是成分的 ts_code
```

| 事项 | 规则 |
|---|---|
| 行的键 | `(index_code, con_code, trade_date)`。成分的代码列叫 **`con_code` 不是 `ts_code`**，所以研究股票池的筛选碰不到它，跨域 join 要自己改名 |
| 可见性 | 行级 `available_at`，规则 `contract_1730_from:trade_date`——截面按它自己那个交易日的 **17:30** 盖章。08:30 决策**看不到当天那张**，读到的是上一张 |
| 频率 | 每指数每月一张，日期是当月最后一个交易日。2020-01 起无缺口 |
| `weight` 的单位 | **百分数**（4.64 = 4.64 %），一张截面求和 100 ± 0.05。当组合权重用必须先 ÷ 100。同一个域里 `repo_daily.weight` 是年化回购利率，同名不同义——**必须先按 `dataset` 过滤再谈单位** |
| 读取窗口 | 120 个自然日足够覆盖两张截面，长假之后也够；不带日期窗口读整段是错的 |
| 可选的指数 | `000300.SH`、`000905.SH`、`000852.SH`、`000688.SH`、`399006.SZ`、`000016.SH`、`000001.SH`。**本臂只用 `000300.SH`**，因为裁决的中性化回归就是对它做的；换成别的指数建的书仍然按沪深 300 判，那是 `families.md` 的点名禁止第 2 条 |

**只有月末截面，这是本臂最大的一条数据限制。** 两张截面之间的真实指数调整、临时剔除与停牌成分都看不见：复核日拿到的是上一张截面的名单，与指数当时的真实成分有偏离，偏离随离上一张截面的天数变大。中证的定期调整在每年 6 月与 12 月的第二个星期五之后生效，临时调整随时可能发生，两者都**先于生效日公告**，所以「取日期不晚于决策日的最新一张」是保守的方向（读到的是旧名单，不是未来名单），但它确实不是当天的真名单。本包因此把复核定在自然月首个决策日——与截面同频；**把复核改得更快并不会让成分更新，只会用同一张截面多交易几次**，这一点要写进任何改节奏的 `hypothesis`。

另一面是修正：该表每个开市日重取当年分区，供应商一旦回写历史权重，改后的数值会带着原来的 `available_at` 进入历史快照。幅度未量化，与宏观整段替换同类。

## 本轮选入的其余数据

| 域 | 数据集 | 行级 `available_at` |
|---|---|---|
| `daily` | 日线、每日指标、复权因子、涨跌停价、停牌（合成一张 30 列的表） | 无列，可见性由 as-of 视图给定 |
| `universe` | 股票基本信息与申万一级成员（`name`、`list_date`、`l1_code`、`l1_name`） | 无列，决策日冻结 |
| `fundamentals` | `income_vip`、`balancesheet_vip`、`cashflow_vip`、`fina_indicator_vip`、`forecast_vip`、`express_vip`、`dividend`、`fina_audit`、`fina_mainbz_vip`、`disclosure_date` | 有 |
| `macro` | `cn_gdp`、`cn_cpi`、`cn_ppi`、`cn_pmi`、`cn_m`、`sf_month`、`shibor`、`shibor_lpr`、`index_daily`、`index_dailybasic`、`sw_daily`、**`index_weight`**、`fut_basic`、`fut_mapping`、`fut_daily`、`opt_basic`、`opt_daily`、`cb_basic`、`cb_daily`、`cb_call` | 有 |
| `events` | `margin`、`margin_detail`、`margin_secs`、`moneyflow`、`cyq_perf`、`bak_daily`、`block_trade`、`stk_holdernumber`、`stk_holdertrade`、`new_share`、`share_float_complete`、`top_list`、`top_inst`、`limit_list_d`、`kpl_list`、`top10_floatholders`、`report_rc` | 有 |
| `text` | 只有 `report_rc`（研报标题索引与正文分片） | 有 |

不在本轮：分钟线与集合竞价（执行只能用 09:30 与 15:00）、`intraday_flow`、`report_rc` 之外的文本、开始得太晚的供应商变体。

**开工第一件事**：核对 `macro.datasets` 里确实有 `index_weight`，以及 `universe` 带 `l1_name`。缺任何一块都**不得静默回退**到全市场池——那不是本臂，按 `families.md` 的终止规则诚实终止。

## 读法与可见性（通则）

```text
daily = pd.read_parquet(context.asof_dir + "/daily", columns=[...], filters=[("trade_date", ">=", start)])
fund  = pd.read_parquet(context.asof_dir + "/fundamentals", columns=["dataset", "available_at", ...],
                        filters=[("dataset", "in", [...]), ("ann_date", ">=", start)])
```

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。
- `fundamentals`、`macro`、`events`、`text` 是多个数据集的列并集：**先按 `dataset` 过滤，再谈字段与单位**；`dataset` 既要进下推过滤也要进列投影。
- 带行级 `available_at` 的域，判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `ann_date`/`trade_date` 推；各域的时间戳字符串格式不同，不要按字符串比较。
- `daily` 与 `universe` 没有 `available_at` 列：as-of 视图只含决策时刻之前可见的行，08:30 决策时 `daily` 的最新一行就是 T-1；按不存在的列过滤会报错。
- 常见的行级规则：财务报表与预告快报在公告日 18:00 可见，`moneyflow` 在交易日 19:00，`macro` 的日频行情与**成分截面**在交易日 17:30；逐数据集的规则以 manifest 的 `availability_rules` 为准。
- 同一 `(ts_code, end_date)` 的财务报表可能有多个版本（首次公告、修订、重述）。选「首版」（事件口径）还是「时点生效版」（状态口径）是设计决定，要在 `hypothesis` 里写明；重述版不能当作新事件。起步包三张主报表取**首版**。

## 申万一级行业是点内的

`universe.l1_name` 不是按当前归属回填的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021，该日之前的决策日用冻结的 SW2014 成分，之后用 SW2021。研究期跨这个切换点，所以：

- **一级行业的个数与名字在研究期内会变**（研究期五个年度锚点上，成分池里的一级行业数实测 26 / 30 / 29 / 28 / 28）。每个决策日在当天的截面上重算行业，普查表按决策日分别列；**跨决策日比较行业名单是错的**。
- `l1_name` 为空的名字落在 `未分类` 这一个桶里。宿主侧的行业归因用的是同一列与同一个标签，所以你在篮子里数出来的行业构成与门 6 读的 `stats.benchmark.top_industry_weight` 说的是同一件事——只差后者是**时间加权**的，季度之内的价格漂移会把两只同业名字的权重推到 `2/N` 之上，所以想靠名额满足那道门要留余量。
- 申万成分历史由供应商回填，纳入日期可能异常，**不能当作可靠的行业变更日**；只用「决策日归属」。

## 单位（最容易错的几处）

| 字段 | 单位 |
|---|---|
| `macro.index_weight.weight` | **百分数**（4.64 = 4.64 %），÷ 100 才是组合权重 |
| `macro.index_daily.pct_chg` | **百分数**（0.68 = 0.68 %），÷ 100 才是日收益 |
| `daily` 的价格 | 元/股，未复权；`adj_factor` 是累计复权因子 |
| `daily.vol`、`amount`、`circ_mv`、`total_mv` | 股、元、元、元（已从原始湖的手、千元、万元归一） |
| `daily.pct_chg`、`turnover_rate` | 小数 |
| `macro.sf_month` 的 `inc_month`/`inc_cumval` | **亿元**；`stk_endval` 是**万亿元** |
| `macro.cn_m` 的 `m0`/`m1`/`m2` | **亿元**；`*_yoy`、`*_mom` 是百分数 |
| `macro.shibor`、`macro.shibor_lpr` | 百分数 |
| `events.moneyflow` 的 `*_amount` | **万元** |
| `events.report_rc.np` | **万元**；`quarter` 是财年标签，不是季度 |
| `fundamentals` 的报表金额 | 元，年初至今累计（资产负债表是时点值） |
| `fina_indicator_vip` 的增速与比率 | 百分数 |

## 数据下限

研究期第一个决策视图里，`fundamentals`、`macro` 与 `events` 从 2020 年初开始，`daily` 回溯五年，`index_weight` 同样从 2020-01 起。由报表自己算的同比与 TTM 从 2021 年春季起才可算，八个季度的 SUE 要到 2022 年；三年训练窗口的学习型模型在第一个研究年里，财务、宏观与事件列只在后一半左右的训练日上有值。第一个研究年的读数单列汇报。

## 陷阱

- **成分的代码列是 `con_code`**。忘了改名再去 join `daily`/`universe` 会得到一张空表，而空表在下游只表现为「池太小」，不会报错。
- **一手常常比一个座位贵。** 100 股整手；科创板 200 股起，研究期五个锚点上有 9 / 13 / 15 / 17 / 20 只科创板成分，起步包整族剔除，代价是每张截面少掉 1.7–4.8 pp 的指数权重——这个数要报。10 万元账户 15 只时 11.0–22.9 % 的成分买不起、30 只时 27.6–42.9 %；100 万元账户 30 只时 0.4–2.4 %、50 只时 2.8–6.6 %。**买得起是座位的函数，随权益每天变**，不是池的静态属性。
- **不要给成分池再加价格上限或成交额下限。** 旧包的 `1 元 < 收盘 ≤ 30 元` 与 `ADV ≥ 3,000 万元` 是为全市场池写的：套到沪深 300 上，价格上限会砍掉一半以上的指数，而成交额下限一条都砍不掉（最薄的成分实测 0.53–0.83 亿元/日）。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。T-1 整日停牌的**持仓**估值必须兜底为 0，否则一个 NaN 会把整个买入预算变成 NaN。
- **卖出估值的价格表要覆盖全部有 K 线的名字**，不是只覆盖池内名字：被指数剔除的名字也要估值。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp），早期研究年的往返成本更高。
- **`n_income` 含少数股东损益**，归母净利润是 `n_income_attr_p`。
- **`circ_mv` 在原始湖里是万元**，快照里已归一为元：离线普查直接读原始湖会差 1 万倍，而它通常同时是筛选列与规模中性化列，一处错会静默地把两件事同时弄反。
- **月度宏观面整整滞后一个月才可见**（`sf_month`、`cn_m`、`cn_cpi`/`cn_ppi`/`cn_pmi`/`cn_gdp` 按「月末 + 31 天」保守盖章）；按 `month` 列推可见性会把整整一个月的前视带进来。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断要拿最新可见交易日与决策日比自然月或 ISO 周，冷 worker 与热 worker 必须逐字节相同。
- **对照不得对自己的面中性化**；训练面板与决策截面必须由同一个函数构建。
