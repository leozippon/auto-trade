# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

**开工第一件事**：核对三件事——`macro.datasets` 里有没有 `index_weight`（没有它就没有这个臂的股票池）与 `index_daily`（没有它就没有残差标签）；`universe` 带不带 `l1_name`（P2 的行业敞口）；以及运行事实里的 **`benchmark_index`** 与 `lib/index.py` 的 `INDEX_CODE` 是否一致（不一致时改常量去对齐运行事实，反过来不行）。**缺任何一块都不得静默回退**到别的池子、别的标签或别的特征集再按同一套口径报数——那是另一个登记，按 `refs/families.md` 的终止规则诚实终止或改登记。

## 本臂的数据集选择：与 20260925 轮新鲜书臂逐字相同

本臂只动训练种子，**一列新数据都不读**：日线、`index_weight` 与 `index_daily` 就是全部输入，`universe` 供名称与行业。创建参数沿用 20260919 轮的数据集选择（含 `index_weight`），`include_intraday` 保持默认 `False`，预建视图种子是 `data/pit_views_seed_research_20260919`，与 `fresh_book_1m_20260925`、`learner_axis_1m_20260924` 同一棵，不需要为本臂另建种子。起步包、宿主冒烟与作包的种子普查都在这棵种子上跑过（`refs/sources.md`）。

## 每个研究年有它自己的 60 个月输入窗

运行事实 `research_geometry` 给出两种窗口：研究期末那份决策视图（会话的 `/mnt/snapshot`）的 `input_window`，以及每个研究年回放起点那份决策视图的 `years[].input_window`（`window_months` 个月，止于该年开始前一天）。回放里策略看到的是后者再加上逐日落地的回放行：一次完整期验证从第一研究年的视图起一路接续下去。

| 研究年 | 回放起点那份视图的 `input_window` | 起步包 `fit` 实际读到的起点（3 年 + 120 个自然日，再截在基准序列的首根 K 线） |
|---|---|---|
| Y1 | `20160701..20210630` | 2020-01-02（`index_daily` / `index_weight` 的数据下限） |
| Y2 | `20170701..20220630` | 2020-01-02（同上） |
| Y3 | `20180701..20230630` | 从 2020-03 起逐季后移 |
| Y4 | `20190701..20240630` | 从 2021-03 起逐季后移，窗口完整 |

**离线普查只有研究期末那一份视图**（`input_window` 为 `20200701..20250630`，其中 `index_daily` 也从 2020-07-01 起），所以在会话里还原 Y1–Y3 的季度拟合必然被左截断：Y1 第一次重训在宿主上有约 220 根训练 bar，在研究期末视图上只剩约 100 根，**低于 `MIN_SAMPLES` = 200，起步包按设计直接报错**；Y1 第二次重训同样不足。Y2 的训练 bar 是宿主的 75–82 %，Y3 从 88 % 到 100 %，Y4 完整。**所以种子普查只做 Y3 与 Y4**（`refs/references/seed-census.md`），作包时的普查也是这样做的。

## 成分截面 `macro.index_weight` 与基准序列 `macro.index_daily`

```text
sec = pd.read_parquet(
    context.asof_dir + "/macro",
    columns=["dataset", "available_at", "index_code", "con_code", "trade_date", "weight"],
    filters=[("dataset", "=", "index_weight"), ("index_code", "=", INDEX_CODE),
             ("trade_date", ">=", start)],
)
bench = pd.read_parquet(
    context.asof_dir + "/macro",
    columns=["dataset", "available_at", "ts_code", "trade_date", "open", "close"],
    filters=[("dataset", "=", "index_daily"), ("ts_code", "=", INDEX_CODE),
             ("trade_date", ">=", start)],
)
# 两者都再按 available_at <= context.inference_at 截一次
```

| 事项 | 规则 |
|---|---|
| 两张表的键不一样 | `index_weight` 的指数在 `index_code`、成分在 **`con_code`**（`ts_code` 为空）；`index_daily` 的指数在 **`ts_code`**，它的 `index_code` **整列为空**。所以成分按 `index_code` 过滤、基准按 `ts_code` 过滤；**拿 `index_code` 去筛 `index_daily` 会安静地返回 0 行**，不带 `ts_code` 过滤读 `index_daily` 会把七个指数的行混在一起 |
| 可见性 | 两者都有行级 `available_at`，规则 `contract_1730_from:trade_date`——按它自己那个交易日的 **17:30** 盖章。08:30 决策**看不到当天那张截面**，读到的是上一张 |
| 截面频率 | 每指数每月一张，日期是当月最后一个交易日；沪深 300 从 2020-01 起无缺口，每张 300 只 |
| `weight` 的单位 | **百分数**（4.64 = 4.64 %），一张截面求和 100 ± 0.05；当组合权重用先 ÷ 100。同一个域里 `repo_daily.weight` 是年化回购利率，同名不同义——**先按 `dataset` 过滤再谈单位** |
| 读取窗口 | 决策时读 120 个自然日足够覆盖两张截面；训练窗读训练年数 + 120 个自然日；不带日期窗口读整段是错的 |
| 可选的指数 | `000300.SH`、`000905.SH`、`000852.SH`、`000688.SH`、`399006.SZ`、`000016.SH`、`000001.SH`。裁决的中性化、β、跟踪误差与零技能面板的成分换名都对同一个指数做（运行事实 `benchmark_index`） |

**只有月末截面，这是这一面最大的限制。** 两张截面之间的真实调整、临时剔除与停牌成分都看不见；复核日拿到的是上一张截面的名单（方向上保守，细节见 `refs/references/index-membership.md`）。每条腿的月度复核都与截面同频：自然月首个决策日刚好有一张新截面，新纳入的名字在这一次进书、被剔除的在这一次强卖。

**每一个训练日、每一个决策日都按那一天的成分判定**（`starter/lib/index.py` 的 `membership`）；第一张截面之前的日期没有成分。

## 本轮要读的列

| 域 | 列 | 单位 | 行级 `available_at` | 用途 |
|---|---|---|---|---|
| `daily` | `open`/`high`/`low`/`close` | 元/股，未复权 | 无列，最新一行 = T-1 | Alpha158 的四条价格线、标签的两个开盘、买单定价 |
| `daily` | `adj_factor` | 累计复权因子 | 同上 | 前复权（锚在每只名字自己最新可见的那一根） |
| `daily` | `vol` / `amount` | **股** / **元** | 同上 | Alpha158 的量与成交均价 |
| `macro.index_daily` | `open` / `close` | 指数点 | 有 | 残差标签的基准腿、β 的回归自变量（**按 `ts_code` 过滤**） |
| `macro.index_weight` | `weight` | **百分数** | 有 | 成员资格与覆盖度报告 |
| `universe` | `name` / `l1_name` | — | 无列，整表按回放槽锚点口径 | ST/退过滤、P2 的行业 |

本包**不读** `events`、`fundamentals` 与 `text`。

## 读法与可见性（通则）

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。
- 带行级 `available_at` 的域，判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `trade_date` 推。
- `daily` 与 `universe` 没有 `available_at` 列：as-of 视图只含决策时刻之前可见的行，08:30 决策时 `daily` 的最新一行就是 T-1；按不存在的列过滤会报错。**`daily` 的读取没有日期上界是对的**——as-of 视图本身就是上界；离线普查在研究期末视图上还原历史决策日时，这一点反过来变成陷阱，必须自己截在 T-1（`references/stage0-census.md`）。
- 起步包每次读取都带 `columns=`、`filters=` 与日期窗口；不引入跨调用缓存，冷 worker 与热 worker 出单一致。

## 申万一级行业是点内的

`universe.l1_name` 按**决策日**口径取（`in_date ≤ 决策日 < out_date`），申万在 2021-12-13 从 SW2014 切到 SW2021，研究期跨这个切换点，所以**一级行业的个数与名字在研究期内会变**，跨决策日比较行业名单是错的。`l1_name` 为空的名字落在 `未分类` 这一个桶里。回放里 `universe` 是该回放槽锚点那一天的口径，槽内不变；研究期末视图只有期末那一个口径。

## 单位（最容易错的几处）

| 字段 | 单位 |
|---|---|
| `macro.index_weight.weight` | **百分数**，÷ 100 才是组合权重 |
| `macro.index_daily.pct_chg` | **百分数**；`open`/`close` 是指数点。起步包只用 `open` 与 `close` 算比值 |
| `daily` 的价格 | 元/股，未复权；`adj_factor` 是累计复权因子 |
| `daily.vol`、`amount` | 股、元（已从原始湖的手、千元归一） |
| `daily.pct_chg`、`turnover_rate` | 小数 |

## 陷阱

- **成分的代码列是 `con_code`**。忘了改名再去 join `daily`/`universe` 会得到一张空表，而空表在下游只表现为「池太小」，不会报错。
- **截面 z 分数与标签的秩只在当天的成分集合上做。**
- **训练矩阵与决策矩阵必须同一个列顺序**：`primary.score` 每次都把 `state_dir` 里存下的列名与 `data.FEATURE_NAMES` 逐字比对，不一致直接报错。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。这类名字在复核日的买卖单会以 `missing_execution_price` 被拒，要逐次报。
- **一手与整手**：100 万元 50 席上一手买不起的成分逐年 2.8–6.6 %（研究期前段的高价股更多）；买不起只影响买入，已持有的名字不因此强卖；买单按最近整手四舍五入再按现金截断。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp，2023-08-28 起），早期研究年的往返成本更高。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断拿最新可见交易日与决策日比自然月。
- **β 只能用滞后窗口估**（120 个交易日的滚动 OLS，向 1 收缩 0.7、截断 [0.3, 2.0]）。
- **数据下限**：`index_weight` 与 `index_daily` 都从 2020-01 起，所以第一研究年的季度重训只有约一年半的在册训练日；第一研究年的读数要单独汇报，它的 `fit` 秒数不能拿来外推。
