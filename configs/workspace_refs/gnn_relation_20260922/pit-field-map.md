# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

**开工第一件事**：核对 `macro.datasets` 里有没有 `index_weight` 与 `index_daily`，以及 `universe` 带不带 `l1_name`。前两个决定能不能建指数内的书与残差化的标签，第三个是**行业图与行业敞口那条要求**的字段。**缺任何一块都不得静默回退**到别的池子、别的标签或别的图再按同一套口径报数——那是另一个机制，按 `families.md` 的终止规则诚实终止或改登记。

第二件事是设备：运行事实 `budgets.strategy_gpu_count` 必须是 1，会话容器里 `torch.cuda.is_available()` 为 True，记下卡名与空闲显存。为 0 或不可用时本臂无法运行，把读数写进结果笔记后以 `no_edge` 结束本臂，**不改写成 CPU 版本**。

## 成分截面 `macro.index_weight`

```text
sec = pd.read_parquet(
    context.asof_dir + "/macro",
    columns=["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"],
    filters=[("dataset", "=", "index_weight"), ("index_code", "=", "000300.SH"),
             ("trade_date", ">=", start)],           # start = 决策日往回 N 个自然日
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
| 读取窗口 | 训练窗口要读满整段（起步包读训练年数 + 260 个自然日），只取决策截面时 120 个自然日足够覆盖两张；不带日期窗口读整段是错的 |
| 可选的指数 | `000300.SH`、`000905.SH`、`000852.SH`、`000688.SH`、`399006.SZ`、`000016.SH`、`000001.SH`。**裁决的中性化都是对沪深 300 做的**，用别的指数建书而仍按沪深 300 报跟踪误差是 `families.md` 的点名禁止第 2 条 |

**只有月末截面，这是这一面最大的限制。** 两张截面之间的真实指数调整、临时剔除与停牌成分都看不见：复核日拿到的是上一张截面的名单，与指数当时的真实成分有偏离，偏离随离上一张截面的天数变大。中证的定期调整在每年 6 月与 12 月的第二个星期五之后生效，临时调整随时可能发生，两者都**先于生效日公告**，所以「取日期不晚于决策日的最新一张」是保守的方向（读到的是旧名单，不是未来名单），但它确实不是当天的真名单。**把复核改得更快并不会让成分更新，只会用同一张截面多交易几次**，这一点要写进任何改节奏的 `hypothesis`。细节见 `references/index-membership.md`。

**训练面板上的成员资格按同一条规则逐日重建**：训练日 t 的截面 = 日期不晚于 t 的最新一张。起步包的 `panel._membership` 就是这条规则，它同时保证了标签、特征 z 分数与图的节点集合是同一个集合。

另一面是修正：该表每个开市日重取当年分区，供应商一旦回写历史权重，改后的数值会带着原来的 `available_at` 进入历史快照。幅度未量化，与宏观整段替换同类。

## 基准自己的日线 `macro.index_daily`

标签的基准腿与（另一条臂的）市场状态都从这里读，**不从成分股重构指数**。

| 事项 | 规则 |
|---|---|
| 过滤 | `("dataset", "=", "index_daily")` 与 `("ts_code", "=", "000300.SH")`。这一域里 `ts_code` 同时属于 `index_daily` / `fut_daily` / `opt_daily` / `cb_daily`，**先按 `dataset` 过滤再谈字段** |
| 可见性 | 行级 `available_at`，交易日 **17:30** 盖章，与成分截面同一条收盘合同。08:30 决策读到的最新一行是 T-1 |
| 字段与单位 | `open` / `close` 是指数点位；`pct_chg` 是**百分数**（0.68 = 0.68 %），÷ 100 才是日收益。**起步包用 `open` 与 `close` 直接算比值，不碰 `pct_chg`**，因为标签是开盘到开盘的 |
| 覆盖 | 研究期内逐日无缺口；面板把它按交易日对齐后前向填充，**填不满就报错**，不要静默用 0 代替 |

## 本轮可能读到的其余数据

具体清单以 `data_summary.json` 为准；下表是本包起步包实际读到的域与常用项。

| 域 | 数据集 | 行级 `available_at` |
|---|---|---|
| `daily` | 日线、每日指标、复权因子、涨跌停价、停牌（合成一张 30 列的表） | 无列，可见性由 as-of 视图给定 |
| `universe` | 股票基本信息与申万一级成员（`name`、`list_date`、`l1_code`、`l1_name`） | 无列，整表按决策日口径；回放中逐槽换到该槽锚点的口径 |
| `macro` | `index_weight`、`index_daily`（本包用这两个）；另有 `sw_daily`、`index_dailybasic`、`cn_*`、`shibor`、`fut_*`、`opt_*`、`cb_*` | 有 |
| `fundamentals` | `income_vip`、`balancesheet_vip`、`cashflow_vip`、`fina_indicator_vip`、`forecast_vip`、`express_vip`、`dividend`、`fina_audit`、`fina_mainbz_vip`、`disclosure_date` | 有 |
| `events` | `margin*`、`moneyflow`、`cyq_perf`、`bak_daily`、`block_trade`、`stk_holdernumber`、`stk_holdertrade`、`new_share`、`share_float_complete`、`top_list`、`top_inst`、`limit_list_d`、`kpl_list`、`top10_floatholders`、`report_rc` | 有 |

分钟线与集合竞价不在本轮：执行只能用 09:30 与 15:00。

## 读法与可见性（通则）

```text
daily = pd.read_parquet(context.asof_dir + "/daily", columns=[...],
                        filters=[("trade_date", ">=", start), ("ts_code", "in", codes)])
macro = pd.read_parquet(context.asof_dir + "/macro", columns=["dataset", "available_at", ...],
                        filters=[("dataset", "=", "..."), ("trade_date", ">=", start)])
```

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。as-of 视图的首片只含决策时刻前可见的行、后续片只含之后发布的行，读目录无需去重。
- `fundamentals`、`macro`、`events`、`text` 是多个数据集的列并集：**先按 `dataset` 过滤，再谈字段与单位**；`dataset` 既要进下推过滤也要进列投影。
- 带行级 `available_at` 的域，判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `ann_date`/`trade_date` 推；各域的时间戳字符串格式不同，不要按字符串比较。
- `daily` 与 `universe` 没有 `available_at` 列：as-of 视图只含决策时刻之前可见的行，08:30 决策时 `daily` 的最新一行就是 T-1；按不存在的列过滤会报错。
- 每次读取都带 `columns=`、`filters=` 与日期窗口。**训练窗口的 `daily` 读取必须按成分并集下推 `ts_code`**：不下推就是把全市场三百多万行读进 16 GiB 的容器里。
- **训练面板与决策截面必须由同一个函数构建。**
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断要拿最新可见交易日与决策日比自然月或 ISO 周，冷 worker 与热 worker 必须逐字节相同。

## 申万一级行业是点内的（本包的静态图就是它）

`universe.l1_name` 不是按当前归属回填的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021，该日之前的决策日用冻结的 SW2014 成分，之后用 SW2021。研究期跨这个切换点，所以：

- **一级行业的个数与名字在研究期内会变**。每个决策日在当天的截面上重算行业，普查表按决策日分别列；**跨决策日比较行业名单是错的**。
- **行业图因此也在变**：同一对名字在切换点前后可能从同业变成不同业。这不是缺陷，是点内口径的后果；要在 `hypothesis` 里写明本臂的图是按决策日口径构造的。
- `l1_name` 为空的名字落在 `未分类` 这一个桶里。宿主侧的行业归因用的是同一列与同一个标签，所以你在篮子里数出来的行业构成与 `stats.benchmark.top_industry_weight` 说的是同一件事——只差后者是**时间加权**的。
- 申万成分历史由供应商回填，纳入日期可能异常，**不能当作可靠的行业变更日**；只用「决策日归属」。
- **一个已接受的口径缺口**：`universe` 整表是决策日（回放中是该槽锚点）的一个口径，所以训练窗口里每一天用的是同一份行业标签与同一份名字。起步包因此**只在决策行上做 ST/退与板块的可交易筛选，不把它推回训练窗口**——把今天的 ST 标签推回两年前，等于系统性地删掉后来变坏的名字，那是前视。行业标签没有这个退路（图需要它），它是本包已知的、量级最小的一处点内不一致。

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
| `daily.pct_chg`、`turnover_rate` | 小数（0.0142 = 1.42 %）。原始湖里 `turnover_rate` 是百分数 |
| `events.moneyflow` 的 `*_amount` | **万元** |
| `fundamentals` 的报表金额 | 元，年初至今累计（资产负债表是时点值） |
| `fina_indicator_vip` 的增速与比率 | 百分数 |

复权价 = 未复权价 × `adj_factor`；拆分一致的成交量 = `vol` / `adj_factor`；成交均价 = `amount` / `vol` × `adj_factor`（`vol` 为 0 时记空）。窗口内的比值不受复权锚点影响，Alpha158 的算子全部是比值，所以复权锚点换了也不影响特征。

## 数据下限

研究期第一个决策视图里，`fundamentals`、`macro` 与 `events` 从 2020 年初开始，`daily` 回溯五年，`index_weight` 与 `index_daily` 同样从 2020-01 起。**两年训练窗口在第一个研究年的起点上刚好够**（训练窗口要往前再吃 260 个自然日的算子与相关窗口热身）：第一个研究年的 `fit` 训练日数会比后面的年份少，读数单列汇报，不要把第一年的快与省内存外推到整期。

## 陷阱

- **成分的代码列是 `con_code`**。忘了改名再去 join `daily`/`universe` 会得到一张空表，而空表在下游只表现为「池太小」，不会报错。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。T-1 整日停牌的**持仓**估值必须兜底为 0，否则一个 NaN 会把整个买入预算变成 NaN。
- **卖出估值的价格表要覆盖全部有 K 线的名字**，不是只覆盖池内名字：已经离开池子的名字也要估值。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp），早期研究年的往返成本更高。
- **不要不假思索地套用别的包的价格与成交额筛子。** `1 元 < 收盘 ≤ 30 元`、`ADV ≥ 3,000 万元` 是为全市场池写的：套到沪深 300 上，价格上限会砍掉一半以上的指数，而成交额下限一条都砍不掉（最薄的成分实测 0.53–0.83 亿元/日）。而且**池的筛子现在会被零技能面板扣掉**，它不再是一个免费的超额来源。
- **标签不能越过决策时刻**：`HORIZON` 日标签要用 t+1+`HORIZON` 的开盘价，所以只有 t + 1 + `HORIZON` ≤ T-1 的信号日进训练。验证段与训练段之间还要留 `HORIZON + 2` 个交易日的隔离带，否则验证日的标签与训练日的标签在时间上重叠。
- **图的邻接也要按时点建**：相关窗口只能用 t 及以前的收益，行业只能用决策日口径。用整段面板算一次相关再到处用，是一次安静的前视。
- **截面只有 300 个名字**。全市场每个截面 2,400–3,500 个名字，成分只有 300 上下：同样的训练年数下样本少一个量级，`fit` 会快很多、内存也低，但**过拟合的风险高很多**。加容量（层数、隐藏维、序列长度、特征列）之前先看 G-INC 与逐年读数，不要先加容量再找理由。
- **模型文件不在 `output/` 里**：权重只写 `context.state_dir`，每次回放开始时为空；不要把训练好的权重放进 `models/` 以绕过重训（写也写不进去，那是只读挂载）。
