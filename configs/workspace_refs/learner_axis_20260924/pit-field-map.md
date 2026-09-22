# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

**开工第一件事**：核对三件事——`macro.datasets` 里有没有 `index_weight`（没有它就没有这个臂的股票池）与 `index_daily`（没有它就没有残差标签）；`universe` 带不带 `l1_name`（P2 的行业敞口那条要求）；以及实验事实里的 **`benchmark_index`** 与 `lib/index.py` 的 `INDEX_CODE` 是否一致。**缺任何一块都不得静默回退**到别的池子、别的标签或别的特征集再按同一套口径报数——那是另一个登记，按 `families.md` 的终止规则诚实终止或改登记。

## 本臂的数据集选择：与 20260919 轮逐字相同

本臂换的是学习器，**一列新数据都不读**：日线、`index_weight` 与 `index_daily` 就是全部输入。所以创建参数沿用 20260919 轮的数据集选择（含 `index_weight`），`include_intraday` 保持默认 `False`，预建视图种子就是那一轮的 `data/pit_views_seed_research_20260919`，不需要为本臂另建种子。本包的起步包与宿主冒烟都在这棵种子上跑过（`sources.md`）。

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
| 频率 | 每指数每月一张，日期是当月最后一个交易日。2020-01 起无缺口，全库 80 张截面 |
| `weight` 的单位 | **百分数**（4.64 = 4.64 %），一张截面求和 100 ± 0.05。当组合权重用必须先 ÷ 100。同一个域里 `repo_daily.weight` 是年化回购利率，同名不同义——**必须先按 `dataset` 过滤再谈单位** |
| 读取窗口 | 120 个自然日足够覆盖两张截面，长假之后也够；不带日期窗口读整段是错的 |
| 可选的指数 | `000300.SH`、`000905.SH`、`000852.SH`、`000688.SH`、`399006.SZ`、`000016.SH`、`000001.SH`。**裁决的中性化、β、跟踪误差与零技能面板的成分换名都对同一个指数做**，流水线把它作为**顶层实验事实 `benchmark_index`** 发布（纯 `ts_code` 字符串，旁边带 `neutralized_excess_method`），它是一个创建参数因此可以不是沪深 300；用别的指数建书而仍按它报跟踪误差是 `families.md` 的点名禁止 |

**只有月末截面，这是这一面最大的限制。** 两张截面之间的真实指数调整、临时剔除与停牌成分都看不见；复核日拿到的是上一张截面的名单，偏离随离上一张截面的天数变大。方向上保守（读到的是旧名单不是未来名单），细节见 `references/index-membership.md`。**月度复核与月末截面同频，这是本轮构造的一个附带好处**：自然月首个决策日复核时刚好有一张新截面可读，新纳入的名字在这一次进书、被剔除的名字在这一次强卖。

**每一个训练日、每一个决策日都要按那一天的成分判定。** 起步包把窗口内每一张可见截面都读出来，按「不晚于该日的最新一张」铺到面板的日期轴上（`starter/lib/index.py` 的 `membership`），**第一张截面之前的日期没有成分**——把第一张截面的名单回填给更早的训练日，就是把未来的成分名单发给过去。

## 本轮要读的列

| 域 | 列 | 单位 | 行级 `available_at` | 用途 |
|---|---|---|---|---|
| `daily` | `open`/`high`/`low`/`close` | 元/股，未复权 | 无列，最新一行 = T-1 | Alpha158 的四条价格线、标签的两个开盘、买单定价 |
| `daily` | `adj_factor` | 累计复权因子 | 同上 | 前复权（锚在每只名字自己最新可见的那一根） |
| `daily` | `vol` / `amount` | **股** / **元** | 同上 | Alpha158 的量与成交均价 |
| `macro.index_daily` | `open` / `close` | 指数点 | 有，`contract_1730_from:trade_date` | 残差标签的基准腿、β 的回归自变量 |
| `macro.index_weight` | `weight` | **百分数** | 有，同上 | 成员资格与覆盖度报告 |
| `universe` | `name` / `l1_name` | — | 无列，整表按决策日口径 | ST/退过滤、P2 的行业敞口 |

本包**不读** `events`、`fundamentals` 与 `text`，也不读毕业谱系的 `vipf2` / `mfflow` / `newage` 三组点时列：本臂要分离的是学习器，多带别的列会让「赢了是因为什么」说不清。排序目标的**相关度等级**不是一列新数据，它是同一个秩标签按等宽切出来的整数（`starter/lib/labels.py` 的 `grades`），可见性与标签完全相同。

## 读法与可见性（通则）

```text
daily = pd.read_parquet(context.asof_dir + "/daily", columns=[...],
                        filters=[("trade_date", ">=", start), ("ts_code", "in", codes)])
```

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。
- 带行级 `available_at` 的域，判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `trade_date` 推。
- `daily` 与 `universe` 没有 `available_at` 列：as-of 视图只含决策时刻之前可见的行，08:30 决策时 `daily` 的最新一行就是 T-1；按不存在的列过滤会报错。
- 起步包每次读取都带 `columns=`、`filters=` 与日期窗口；不引入跨调用缓存，冷 worker 与热 worker 出单一致。
- **`daily` 的读取没有日期上界，这是对的**：as-of 视图本身就是上界。离线门在研究期末的视图上重建某个历史决策日时，这一点反过来变成陷阱——必须自己把 `daily` 截在该决策日的 T-1、把宏观域按 `available_at` 截在该决策的 08:30，否则起步包会把整段未来的 K 线读进特征与标签（`references/learner-ablation.md`）。

## 申万一级行业是点内的

`universe.l1_name` 不是按当前归属回填的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021，该日之前的决策日用冻结的 SW2014 成分，之后用 SW2021。研究期跨这个切换点，所以**一级行业的个数与名字在研究期内会变**，跨决策日比较行业名单是错的。`l1_name` 为空的名字落在 `未分类` 这一个桶里。宿主侧的行业归因用的是同一列与同一个标签，只差它是**时间加权**的，复核间隔之内的价格漂移会把权重推高，所以 P2 仍要逐次读。申万成分历史由供应商回填，纳入日期可能异常，**不能当作可靠的行业变更日**；只用「决策日归属」。

## `snapshot` 只有一个时点口径

会话挂的 `snapshot`（`/mnt/snapshot`、`context.snapshot_dir`）是**研究期末那一天**的决策视图，只有这一个口径：`universe.parquet` 里的 `name`、ST 状态、`l1_name` 与在册名单都是那一天的，不是历史上每一天的。

- **不要拿它离线重建历史某一天的决策**：那会把后来的更名、ST、退市与行业重分类灌回过去（幸存者偏差），离线读数被系统性抬高。已有实测：其余全部不变、只把这一个文件换成当天口径，重建出的首日持仓与宿主回放的一致度就从 5/12 变成 12/12，而用期末口径的那次离线年化被抬高了约 13 个百分点。
- 宿主回放逐槽换版，但槽内不变：回放里 `asof_dir/universe` 是该回放槽锚点那一天的口径（四年研究回放按年分槽），槽内的上市、更名、ST 与行业重分类要到下一个槽的锚点才可见，槽内新上市的代码在该槽里没有行。实测四年回放期末月仍有 107 个在交易的代码没有 universe 行（`name` 为空、行业落在 `未分类`），所以回放里按名称或行业做的过滤也只准到槽锚点那一天。
- `/mnt/tools/screen.py` 扫全历史时读的也是这一个口径，名称与行业相关的筛选结果同样不是当时的口径。
- 离线重建只能用来查构造完整性与相对量级，**不作绝对水平的判断**。

## 单位（最容易错的几处）

| 字段 | 单位 |
|---|---|
| `macro.index_weight.weight` | **百分数**（4.64 = 4.64 %），÷ 100 才是组合权重 |
| `macro.index_daily.pct_chg` | **百分数**，÷ 100 才是日收益；`open`/`close` 是指数点 |
| `daily` 的价格 | 元/股，未复权；`adj_factor` 是累计复权因子 |
| `daily.vol`、`amount`、`circ_mv`、`total_mv` | 股、元、元、元（已从原始湖的手、千元、万元归一） |
| `daily.pct_chg`、`turnover_rate`、`turnover_rate_f` | 小数 |

## 陷阱

- **成分的代码列是 `con_code`**。忘了改名再去 join `daily`/`universe` 会得到一张空表，而空表在下游只表现为「池太小」，不会报错。
- **截面 z 分数只在当天的成分集合上做**。把全市场或成分并集拿去算中位数与 MAD，得到的分数分布与决策日不是同一个口径。
- **训练矩阵与决策矩阵必须同一个列顺序**。起步包只有 `data.FEATURE_NAMES` 一处定义这个顺序，`primary.score` 每次都把 `state_dir` 里存下的列名与它逐字比对，不一致直接报错。
- **排序目标的查询组就是决策 bar**：训练行按 bar 升序、同一 bar 的行连续（`data.fit_samples` 就是这样堆的），组的大小按行序数出来。把行打乱或跨 bar 分组，lambdarank 学到的就不是「同一天里谁排前面」。
- **训练日与决策日的成分各用各自那天的截面**；第一张截面之前的日期没有成分。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。Alpha158 的输入在无 K 线的日子上必须是空值（起步包把这些格子改回 NaN 再算算子）。
- **一手与整手**：100 万元 50 席上一手买不起的成分约 2–3 %（实测 2.13–2.85 %）。买不起只影响买入，**已持有的名字涨到一手超过一个座位时不得强卖**。买单按最近整手四舍五入再按现金截断，不是向下取整。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp），早期研究年的往返成本更高。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断要拿最新可见交易日与决策日比自然月，冷 worker 与热 worker 必须逐字节相同。**`state_dir` 里的重训计数器不是缓存**：它只在 `fit` 里写，`generate_orders` 只把它抄进订单元数据，不据它做任何决定。
- **β 只能用滞后窗口估**。起步包用 120 个交易日的滚动 OLS，向 1 收缩 0.7、截断 [0.3, 2.0]；把整段研究期的 β 一次算出来再回填是前视。
- **数据下限**：研究期第一个决策视图里 `daily` 回溯五年，但 `index_weight` 从 2020-01 起，所以第一个研究年的训练窗口只有约一年半有成分（首张截面之前的日期没有成分，见上），样本少、`fit` 快；第一个研究年的读数要单独汇报，它的 `fit` 秒数不能拿来外推。
