# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

**开工第一件事**：核对 `macro.datasets` 里有没有 `index_weight`，以及 `universe` 带不带 `l1_name`。前者决定能不能建指数内的书（带跟踪授权时是必需的，见 `README.md`），后者是行业敞口那条要求的字段。**缺任何一块都不得静默回退**到别的池子再按同一套口径报数——那是另一本书，按 `families.md` 的终止规则诚实终止或改登记。

## 成分截面 `macro.index_weight`

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
| 可选的指数 | `000300.SH`、`000905.SH`、`000852.SH`、`000688.SH`、`399006.SZ`、`000016.SH`、`000001.SH`。**裁决的中性化与跟踪授权都是对沪深 300 做的**，所以用别的指数建书而仍按沪深 300 报跟踪误差是 `families.md` 的点名禁止第 3 条 |

**只有月末截面，这是这一面最大的限制。** 两张截面之间的真实指数调整、临时剔除与停牌成分都看不见：复核日拿到的是上一张截面的名单，与指数当时的真实成分有偏离，偏离随离上一张截面的天数变大。中证的定期调整在每年 6 月与 12 月的第二个星期五之后生效，临时调整随时可能发生，两者都**先于生效日公告**，所以「取日期不晚于决策日的最新一张」是保守的方向（读到的是旧名单，不是未来名单），但它确实不是当天的真名单。把复核定在自然月首个决策日与截面同频最省事；**把复核改得更快并不会让成分更新，只会用同一张截面多交易几次**，这一点要写进任何改节奏的 `hypothesis`。细节见 `references/index-membership.md`。

另一面是修正：该表每个开市日重取当年分区，供应商一旦回写历史权重，改后的数值会带着原来的 `available_at` 进入历史快照。幅度未量化，与宏观整段替换同类。

## 本轮可能选入的其余数据

具体清单以 `data_summary.json` 为准；下表是本包起步包实际读到的域与常用项。

| 域 | 数据集 | 行级 `available_at` |
|---|---|---|
| `daily` | 日线、每日指标、复权因子、涨跌停价、停牌（合成一张 30 列的表） | 无列，可见性由 as-of 视图给定 |
| `universe` | 股票基本信息与申万一级成员（`name`、`list_date`、`l1_code`、`l1_name`） | 无列，整表按决策日口径；回放中逐槽换到该槽锚点的口径 |
| `fundamentals` | `income_vip`、`balancesheet_vip`、`cashflow_vip`、`fina_indicator_vip`、`forecast_vip`、`express_vip`、`dividend`、`fina_audit`、`fina_mainbz_vip`、`disclosure_date` | 有 |
| `macro` | `cn_gdp`、`cn_cpi`、`cn_ppi`、`cn_pmi`、`cn_m`、`sf_month`、`shibor`、`shibor_lpr`、`index_daily`、`index_dailybasic`、`sw_daily`、`index_weight`、`fut_*`、`opt_*`、`cb_*` | 有 |
| `events` | `margin*`、`moneyflow`、`cyq_perf`、`bak_daily`、`block_trade`、`stk_holdernumber`、`stk_holdertrade`、`new_share`、`share_float_complete`、`top_list`、`top_inst`、`limit_list_d`、`kpl_list`、`top10_floatholders`、`report_rc` | 有 |
| `text` | 研报标题索引与正文分片 | 有 |

分钟线与集合竞价不在本轮：执行只能用 09:30 与 15:00。

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
- 同一 `(ts_code, end_date)` 的财务报表可能有多个版本（首次公告、修订、重述）。选「首版」（事件口径）还是「时点生效版」（状态口径）是设计决定，要在 `hypothesis` 里写明；重述版不能当作新事件。起步包三张主报表取**首版**（`report_type == "1"`）。

## 申万一级行业是点内的

`universe.l1_name` 不是按当前归属回填的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021，该日之前的决策日用冻结的 SW2014 成分，之后用 SW2021。研究期跨这个切换点，所以：

- **一级行业的个数与名字在研究期内会变**。每个决策日在当天的截面上重算行业，普查表按决策日分别列；**跨决策日比较行业名单是错的**。
- `l1_name` 为空的名字落在 `未分类` 这一个桶里。宿主侧的行业归因用的是同一列与同一个标签，所以你在篮子里数出来的行业构成与 `stats.benchmark.top_industry_weight` 说的是同一件事——只差后者是**时间加权**的，复核间隔之内的价格漂移会把两只同业名字的权重推到 `2/N` 之上，所以想靠名额满足那道要求要留余量。
- 申万成分历史由供应商回填，纳入日期可能异常，**不能当作可靠的行业变更日**；只用「决策日归属」。

## `snapshot` 只有一个时点口径

会话挂的 `snapshot`（`/mnt/snapshot`、`context.snapshot_dir`）是**研究期末那一天**的决策视图，只有这一个口径：`universe.parquet` 里的 `name`、ST 状态、`l1_name` 与在册名单都是那一天的，不是历史上每一天的。

- **不要拿它离线重建历史某一天的决策**：那会把后来的更名、ST、退市与行业重分类灌回过去（幸存者偏差），离线读数被系统性抬高。已有实测：其余全部不变、只把这一个文件换成当天口径，重建出的首日持仓与宿主回放的一致度就从 5/12 变成 12/12，而用期末口径的那次离线年化被抬高了约 13 个百分点。
- 宿主回放不受影响：回放里 `asof_dir/universe` 用的是该折决策日的口径，并按回放槽的锚点逐槽换版（槽内不变，槽内新上市的代码在该槽里没有行）。
- `/mnt/tools/screen.py` 扫全历史时读的也是这一个口径，名称与行业相关的筛选结果同样不是当时的口径。
- 离线重建只能用来查构造完整性与相对量级，**不作绝对水平的判断**。

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
| `dividend.cash_div` | **元/股** |
| `balancesheet_vip` 的 `money_cap` / `st_borr` / `lt_borr` / `minority_int` / `total_hldr_eqy_exc_min_int` | 元 |

## 数据下限

研究期第一个决策视图里，`fundamentals`、`macro` 与 `events` 从 2020 年初开始，`daily` 回溯五年，`index_weight` 同样从 2020-01 起。由报表自己算的同比与 TTM 从 2021 年春季起才可算，八个季度的 SUE 要到 2022 年；三年训练窗口的学习型模型在第一个研究年里，财务、宏观与事件列只在后一半左右的训练日上有值。第一个研究年的读数单列汇报。

## 本轮四个开局会碰到的字段

可见时点与单位仍以上面的通则为准。这里只点本轮分数会踩到的坑，不代替 `families.md` 的构造。

| 开局 | 读哪里 | 陷阱 |
|---|---|---|
| 毛利率 overlay | `income_vip.total_revenue` / `oper_cost` / `operate_profit`，`balancesheet_vip.total_assets`。分数是（收入 − 营业成本）/ 资产，年化 ×4/季度序号。缺 `oper_cost` 才允许回退到营业利润 / 资产 | 报表金额是元、年初至今累计。不要 EP/BP，不要叠盈利 / 应计 / 投资三柱。取首版。缺 `index_weight` 不得回退全市场 |
| 指数内 −NOA | `balancesheet_vip`：`total_assets`、`money_cap`、`st_borr`、`lt_borr`、`minority_int`、`total_hldr_eqy_exc_min_int`。NOA = (TA − 货币资金) − (TA − 短债 − 长债 − 少数股东权益 − 归母权益)，分数是 −NOA / 滞后 TA | **缺列诚实失败，不静默改公式。** 不是 percent accrual，不要用利润与经营现金流的差。滞后资产是上一张可见的同季资产，不是当期 TA 自己除自己 |
| −资产增长 overlay | 优先上年同季 `total_assets` 算同比，回退 `fina_indicator_vip.assets_yoy` / 100。分数是 z(−资产同比) | `assets_yoy` 是百分数。不要叠盈利 / 应计。证伪对着已关质量书的投资柱秩相关 |
| 可见现金分红池 | `fundamentals.dividend.cash_div` 近 365 日求和 / `close`。只计 `available_at` 可见、且 `ex_date` 或 `pay_date` 不晚于决策日的记录 | `cash_div` 与 `close` 都是元/股，比值才是拖尾股息率。不要除 `circ_mv`。不要做成盈利收益率或 OCF 收益率，不要 PE。可见性只看 `available_at`，不从 `ann_date` 推 |

## 陷阱

- **成分的代码列是 `con_code`**。忘了改名再去 join `daily`/`universe` 会得到一张空表，而空表在下游只表现为「池太小」，不会报错。
- **一手常常比一个座位贵，而且它是座位的函数不是池的属性**，随权益每天变。科创板 200 股起一手，在小账户上整族装不进一个座位；算术与实测在 `references/capital-arithmetic.md`。
- **不要不假思索地套用别的包的价格与成交额筛子。** `1 元 < 收盘 ≤ 30 元`、`ADV ≥ 3,000 万元` 是为全市场池写的：套到沪深 300 上，价格上限会砍掉一半以上的指数，而成交额下限一条都砍不掉（最薄的成分实测 0.53–0.83 亿元/日）。而且**池的筛子现在会被零技能面板扣掉**，它不再是一个免费的超额来源。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。T-1 整日停牌的**持仓**估值必须兜底为 0，否则一个 NaN 会把整个买入预算变成 NaN。
- **卖出估值的价格表要覆盖全部有 K 线的名字**，不是只覆盖池内名字：已经离开池子的名字也要估值。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp），早期研究年的往返成本更高。
- **`n_income` 含少数股东损益**，归母净利润是 `n_income_attr_p`。本轮毛利率开局不用这两列。
- **`circ_mv` 在原始湖里是万元**，快照里已归一为元：离线普查直接读原始湖会差 1 万倍，而它通常同时是筛选列、规模中性化列与现金分红收益率的分母，一处错会静默地把几件事同时弄反。
- **月度宏观面整整滞后一个月才可见**（`sf_month`、`cn_m`、`cn_cpi`/`cn_ppi`/`cn_pmi`/`cn_gdp` 按「月末 + 31 天」保守盖章）；按 `month` 列推可见性会把整整一个月的前视带进来。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断要拿最新可见交易日与决策日比自然月或 ISO 周，冷 worker 与热 worker 必须逐字节相同。
- **训练面板与决策截面必须由同一个函数构建。**
- **NOA 公式缺列不得改写。** 用别的负债或现金列顶上去，得到的不是本开局登记的分数。
- **现金分红只计已经发生的可见记录。** 用未来的 `ex_date` / `pay_date` 当收益是前视；只用 `ann_date` 当可见是把尚未盖章的预案算进去。
