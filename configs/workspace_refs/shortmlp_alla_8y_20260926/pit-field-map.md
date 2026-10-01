# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

**开工第一件事**：核对 `macro.datasets` 里有没有 `index_weight`（没有它就没有这个臂的股票池）、`macro.datasets` 里有没有 `index_daily`（没有它就没有残差标签的基准腿）、`events.datasets` 里有没有 `margin_detail`（两条两融通道），以及 `universe` 带不带 `l1_name`（行业敞口那条要求的字段）。**缺任何一块都不得静默回退**到别的池子、别的标签或别的通道再按同一套口径报数——那是另一个登记，按 `families.md` 的终止规则诚实终止或改登记。

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
| 可选的指数 | `000300.SH`、`000905.SH`、`000852.SH`、`000688.SH`、`399006.SZ`、`000016.SH`、`000001.SH`。**裁决的中性化与跟踪授权都是对沪深 300 做的**，所以用别的指数建书而仍按沪深 300 报跟踪误差是 `families.md` 的点名禁止 |

**只有月末截面，这是这一面最大的限制。** 两张截面之间的真实指数调整、临时剔除与停牌成分都看不见：复核日拿到的是上一张截面的名单，与指数当时的真实成分有偏离，偏离随离上一张截面的天数变大。中证的定期调整在每年 6 月与 12 月的第二个星期五之后生效，临时调整随时可能发生，两者都**先于生效日公告**，所以「取日期不晚于决策日的最新一张」是保守的方向（读到的是旧名单，不是未来名单），但它确实不是当天的真名单。细节见 `references/index-membership.md`。

**本臂比头包多用一件事：训练日也要按那一天的成分判定。** 起步包把窗口内每一张可见截面都读出来，按「不晚于该训练日的最新一张」铺到面板的日期轴上（`starter/lib/index.py` 的 `membership`）。用今天的名单去标注三年前的训练日是一种幸存者偏差：当年不在指数里的名字会被当成在，当年在而现在不在的名字会整段消失。

另一面是修正：该表每个开市日重取当年分区，供应商一旦回写历史权重，改后的数值会带着原来的 `available_at` 进入历史快照。幅度未量化，与宏观整段替换同类。

## 本轮序列模型要读的列

具体清单以 `data_summary.json` 为准；下表是本包起步包实际读到的域与列，覆盖率是在本轮某个研究期决策视图上实测的（口径与读数见 `sources.md`）。

| 域 | 列 | 单位 | 行级 `available_at` | 实测覆盖 |
|---|---|---|---|---|
| `daily` | `open`/`high`/`low`/`close` | 元/股，未复权 | 无列，可见性由 as-of 视图给定，最新一行 = T-1 | 100 % |
| `daily` | `adj_factor` | 累计复权因子 | 同上 | 100 % |
| `daily` | `vol` / `amount` | **股** / **元**（原始湖是手 / 千元） | 同上 | 100 % |
| `daily` | `turnover_rate` / `turnover_rate_f` | **小数**（0.0142 = 1.42 %） | 同上 | 100 % / 100 % |
| `daily` | `volume_ratio` | 倍（量比） | 同上 | **99.90 %** |
| `daily` | `up_limit` / `down_limit` | 元/股 | 同上 | 100 % / 100 % |
| `daily` | `circ_mv` | **元**（原始湖是万元） | 同上 | 100 % |
| `macro.index_daily` | `open` / `close` / `pct_chg` | 指数点 / 指数点 / **百分数** | 有，`contract_1730_from:trade_date` | 每交易日一行 |
| `macro.index_weight` | `weight` | **百分数** | 有，同上 | 每月一张 |
| `events.margin_detail` | `rzye` / `rzmre` / `rzche` / `rqye` | **元** | 有，`official_next_day_09_from:trade_date` | 覆盖当日成分的 **99.67 %** |
| `universe` | `name` / `l1_name` | — | 无列，整表按决策日口径 | 100 % |

**触板标记是算出来的，不是读出来的**：`high >= up_limit`（触涨停）、`low <= down_limit`（触跌停），用同一天的未复权价与限价比，不要拿复权价去比。实测在一个研究年上，收盘封涨停占全部股票日的 **1.01 %**，盘中触涨停 **1.52 %**，收盘封跌停 **0.56 %**——这是一条稀疏通道，截面 z 标准化之后它主要在「今天有没有人被挡在外面」这件事上出力。

**两融的可见性比看上去晚一天。** `margin_detail` 的规则是「交易日次日 09:00」，**晚于次日 08:30 的决策**，所以交易日 T 的两融数据第一次能用是 T+2 的 08:30。起步包不硬编码这个位移：它把每一行按 `available_at` 映射到**严格晚于该时刻的第一个 08:30 决策**，再落到那次决策读到的 T-1 那一行上，然后沿日期轴前向填充（`starter/lib/panel.py` 的 `_place_by_stamp`）。规则字符串改了，算术仍然对。

## 集合竞价面不在本轮的模型里

`auction` 域在快照里**存在**，但**精确竞价数据从 2025-01-16 才开始**（此前 09:29 没有精确竞价，历史回放用带标记的分钟代理）。实测研究期末的决策视图里它只有 **107 个交易日**、约 58.8 万行，而本臂的训练窗口是三年。把一条 90 % 以上训练日为空的通道喂进模型，模型学到的是「这一段有数据、那一段没有」，即一次隐蔽的时间标记。

**所以本包不把竞价面放进 `g1` 的输入，也不允许作为变体轴。** 它在研究期尾段可作离线诊断（例如检查触板通道与开盘竞价量的关系），但任何用到它的分数都不得进入回放。替代它的是同一类信息里研究期全覆盖的部分：触板标记、自由流通换手、量比与两融流。

## 读法与可见性（通则）

```text
daily = pd.read_parquet(context.asof_dir + "/daily", columns=[...],
                        filters=[("trade_date", ">=", start), ("ts_code", "in", codes)])
events = pd.read_parquet(context.asof_dir + "/events", columns=["dataset", "available_at", ...],
                         filters=[("dataset", "=", "margin_detail"), ("trade_date", ">=", start)])
```

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。
- `fundamentals`、`macro`、`events`、`text` 是多个数据集的列并集：**先按 `dataset` 过滤，再谈字段与单位**；`dataset` 既要进下推过滤也要进列投影。
- 带行级 `available_at` 的域，判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `ann_date`/`trade_date` 推；各域的时间戳字符串格式不同，不要按字符串比较。
- `daily` 与 `universe` 没有 `available_at` 列：as-of 视图只含决策时刻之前可见的行，08:30 决策时 `daily` 的最新一行就是 T-1；按不存在的列过滤会报错。
- **按 `ts_code` 下推过滤是本臂省时间的主要手段**：股票池是成分的并集（三年窗口实测 421 个名字），比全市场少一个量级，读取与面板都跟着变小。
- 起步包每次读取都带 `columns=`、`filters=` 与日期窗口；不引入跨调用缓存，冷 worker 与热 worker 出单一致。

## 申万一级行业是点内的

`universe.l1_name` 不是按当前归属回填的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021，该日之前的决策日用冻结的 SW2014 成分，之后用 SW2021。研究期跨这个切换点，所以：

- **一级行业的个数与名字在研究期内会变**。每个决策日在当天的截面上重算行业；**跨决策日比较行业名单是错的**。
- `l1_name` 为空的名字落在 `未分类` 这一个桶里。宿主侧的行业归因用的是同一列与同一个标签，所以你在篮子里数出来的行业构成与 `stats.benchmark.top_industry_weight` 说的是同一件事——只差后者是**时间加权**的。
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
| `macro.index_daily.pct_chg` | **百分数**（0.68 = 0.68 %），÷ 100 才是日收益；`open`/`close` 是指数点 |
| `daily` 的价格与 `up_limit`/`down_limit` | 元/股，未复权；`adj_factor` 是累计复权因子 |
| `daily.vol`、`amount`、`circ_mv`、`total_mv` | 股、元、元、元（已从原始湖的手、千元、万元归一） |
| `daily.pct_chg`、`turnover_rate`、`turnover_rate_f` | 小数 |
| `daily.volume_ratio` | 倍数，不是百分数 |
| `events.margin_detail` 的 `rzye`/`rzmre`/`rzche`/`rqye` | **元**（与同域 `moneyflow` 的 `*_amount` 是万元不同——先按 `dataset` 过滤再谈单位） |
| `events.moneyflow` 的 `*_amount` | **万元** |

## 陷阱

- **成分的代码列是 `con_code`**。忘了改名再去 join `daily`/`universe` 会得到一张空表，而空表在下游只表现为「池太小」，不会报错。
- **训练日的成分要用当天那张截面**，不是最新那张。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。面板对无 K 线的日子把价格沿用前值、量与换手记 0、有无日线标记记 0，序列因此不会断，但模型看得见那一天没有成交。
- **卖出估值的价格表要覆盖全部有 K 线的名字**，不是只覆盖池内名字：已经离开池子的名字也要估值。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp），早期研究年的往返成本更高。
- **`circ_mv` 在原始湖里是万元**，快照里已归一为元：离线普查直接读原始湖会差 1 万倍，而它同时是两条两融通道的分母。
- **两融的分母别用总市值**：起步包用 `circ_mv`（流通市值），换成 `total_mv` 会把大盘国企那一端整体压扁。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断要拿最新可见交易日与决策日比 ISO 周，冷 worker 与热 worker 必须逐字节相同。**`state_dir` 里的重训计数器不是缓存**：它只在 `fit` 里读写，`generate_orders` 不看它，冷热 worker 的出单与它无关。
- **训练面板与决策截面必须由同一个函数构建**，归一化也必须（`starter/lib/panel.py` 的 `build` 与 `window`）。
- **β 只能用滞后窗口估**。起步包用 120 个交易日的滚动 OLS，向 1 收缩 0.7、截断 [0.3, 2.0]；把整段研究期的 β 一次算出来再回填是前视。
