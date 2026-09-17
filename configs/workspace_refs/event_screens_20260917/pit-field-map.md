# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`。借来那条腿要的：`fundamentals.datasets` 必须列出 `income_vip`、`cashflow_vip`、`balancesheet_vip`（缺任一，应计筛子不可构造，股票池也就和姊妹臂不再逐字相同，按 `families.md` 诚实终止）；`macro.datasets` 必须列出 `index_daily`（缺它时排名列按登记变体退为总波动，买单元数据 `vol_basis` 写 `total`）；`universe` 必须带 `l1_name`（缺它，桶就不存在）。本臂自己要的：`fundamentals.datasets` 必须列出 **`fina_audit`** 与 **`fina_mainbz_vip`**，`events.datasets` 必须列出 **`share_float_complete`**——**缺哪一个，对应的那个筛子就不存在，必须在 `hypothesis` 里声明并按只开其余筛子的变体跑，不得静默跳过**。本文是本包可见时点、单位、去重规则与陷阱的**唯一权威表述**。

## 读法

借来那条腿的四次读取（每个决策都做）：

```text
daily = pd.read_parquet(context.asof_dir + "/daily",
                        columns=["ts_code","trade_date","close","pct_chg","amount","adj_factor",
                                 "turnover_rate","circ_mv","pe_ttm","is_suspended"],
                        filters=[("trade_date",">=",start)])                       # 220 个自然日
stmts = pd.read_parquet(context.asof_dir + "/fundamentals",
                        columns=["dataset","available_at","ts_code","end_date","ann_date","report_type",
                                 "n_income_attr_p","n_cashflow_act","total_assets"],
                        filters=[("dataset","in",["income_vip","cashflow_vip","balancesheet_vip"]),
                                 ("ann_date",">=",start)])                          # 700 个自然日
index = pd.read_parquet(context.asof_dir + "/macro",
                        columns=["dataset","available_at","ts_code","trade_date","pct_chg"],
                        filters=[("dataset","=","index_daily"),("ts_code","=","000300.SH"),
                                 ("trade_date",">=",start)])                       # 220 个自然日
universe = pd.read_parquet(context.asof_dir + "/universe",
                           columns=["ts_code","name","list_date","l1_name"])
```

本臂的三次读取（**只在复核日做**，筛子在 `_shape` 里，非复核日走不到）：

```text
audit  = pd.read_parquet(context.asof_dir + "/fundamentals",
                         columns=["dataset","available_at","ts_code","end_date","audit_result"],
                         filters=[("dataset","=","fina_audit"),("end_date",">=",start)])       # 900 个自然日的报告期
mainbz = pd.read_parquet(context.asof_dir + "/fundamentals",
                         columns=["dataset","available_at","ts_code","end_date","bz_code","bz_item","bz_sales"],
                         filters=[("dataset","=","fina_mainbz_vip"),("end_date",">=",start)])  # 900 个自然日的报告期
float_ = pd.read_parquet(context.asof_dir + "/events",
                         columns=["dataset","available_at","ts_code","float_date","float_share",
                                  "float_ratio","share_type","holder_name"],
                         filters=[("dataset","=","share_float_complete"),
                                  ("float_date",">=",今天),("float_date","<=",今天 + 90 天)])
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。`fundamentals`、`macro` 与 `events` 都是多个数据集的列并集：同名列在不同数据集里含义与单位不同（`pct_chg` 在 `daily` 是小数、在 `macro.index_daily` 是百分数；`end_date` 在报表里是报告期末、在别的事件表里可能是别的东西），**必须先按 `dataset` 过滤再谈字段与单位**。

两类表要分开：`fundamentals`、`macro` 与 `events` 的行带**行级 `available_at` 列**，判可见只看它（字符串，`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比；起步包只有 `face.visible_rows` 一个地方做这件事）；`daily` 与 `universe` **没有 `available_at` 列**（按它过滤会 KeyError），它们的可见性由 as-of 视图本身给定。

**三个新面的过滤列是精心挑的，不要改成看起来更自然的那一列。** `fina_mainbz_vip` 在快照里**根本没有 `ann_date` 列**（它的可见时点全部继承自主表），按 `ann_date` 过滤会直接报错或读空；`fina_audit` 有 `ann_date`，但按报告期 `end_date` 过滤才能同时罩住「缺公告日期、从主表继承时点」的那些行。解禁面按 `float_date` 过滤：它是**已公告的未来日期**，这也是唯一能把九百多万行压到几万行的那一列。

## 逐表可见边界

| 数据集 | 本包所需字段与单位 | 行级 `available_at` 规则 | 08:30 能看到的最新行 |
|---|---|---|---|
| `fundamentals.income_vip` | `ts_code`、`end_date`、`ann_date`、`report_type`（只用 `"1"`）、`n_income_attr_p`（**元**，年初至今累计） | `source:f_ann_date_or_ann_date` 18:00 | 公告日 ≤ T-1 的报表 |
| `fundamentals.cashflow_vip` | 同上键；`n_cashflow_act`（**元**，年初至今累计） | 同上 | T-1 |
| `fundamentals.balancesheet_vip` | 同上键；`total_assets`（**元**，期末时点值） | 同上 | T-1 |
| **`fundamentals.fina_audit`** | `end_date`（报告期，几乎全是年报）、`audit_result`（**中文分类串**，见下）、`audit_agency` | 有公告日期时 `source:ann_date` 18:00；缺公告日期时继承同公司同报告期主表中**最晚**的可见时间；仍无法确定的行**根本不进入 PIT 数据** | 公告日 ≤ T-1 的那一版意见 |
| **`fundamentals.fina_mainbz_vip`** | `end_date`、`bz_code`（**构成类型**：`P` 产品／`D` 地区／`I` 行业）、`bz_item`（分部名）、`bz_sales`（**元**）、`bz_profit`、`curr_type`（全为 `CNY`） | 全部是 `fallback_joined_statement_available_at` 18:00：该数据集**没有自己的公告日期**，时点一律继承同公司同报告期主表中最晚的可见时间 | 对应主表公告日 ≤ T-1 的那个报告期 |
| **`events.share_float_complete`** | `float_date`（**已公告的解禁日**）、`float_share`（**股**）、`float_ratio`（**占总股本的百分数**）、`share_type`（限售类型）、`holder_name`（持有人） | 绝大多数是 `conservative_date_eod`：按公告日当天 **23:59:59** 保守盖章，因此最早用于**下一个交易日**的决策；极少数没有公告日期的行按 `fallback_conservative_from:float_date` 盖章，也就是**到解禁当天才可见** | 公告日 ≤ T-1 的解禁安排 |
| `macro.index_daily` | `ts_code = 000300.SH`、`pct_chg`（**百分数**，÷ 100 才是日收益） | `contract_1730_from:trade_date` 17:30 | T-1 |
| `daily` | `close`（元）、`pct_chg`（小数）、`amount`（元）、`circ_mv`（**元**，已归一化）、`turnover_rate`（小数）、`adj_factor`、`pe_ttm`（倍）、`is_suspended` | **无 `available_at` 列** | 最新一行 = T-1 |
| `universe` | `name`（ST 筛选）、`list_date`、`l1_name`（申万一级行业名，桶的键） | **无 `available_at` 列**；决策日冻结 | 当日 |

原始湖 `data/raw/daily` 的 `amount` 是**千元**、`circ_mv` 是**万元**；离线普查直接读原始湖会差 1000 倍 / 1 万倍，快照里已归一化为元。**池底与规模三分位排的都是 `circ_mv`，所以这一处单位错会静默地把两个桶维度同时截反。**

## 申万一级行业是点内的

`universe.l1_name` 不是当前归属回填出来的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并且按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021。研究期跨这个切换点，所以一级行业的个数与名字在研究期内会变（SW2014 二十八个、SW2021 三十一个，加上 `未分类`），**跨决策日比较行业名单是错的**；切换日附近的那一次季度复核可能因为归属变动而多换几只名字，这是真实的口径变更，不是实现缺陷。`l1_name` 为空的名字落在 `未分类` 这一个桶里，和其它桶一样受 2 只上限约束，宿主侧的行业归因用的是同一个名字与同一个标签。申万成分历史由供应商回填，**纳入日期不能当作可靠的行业变更日**；本包只用「决策日归属」。

## 去重与版本规则

借来那条腿的规则（不得改动）：三张主表同一 `(ts_code, end_date)` 各取 `available_at` **最早**的一版，重述版永远不是事件；三张表按 `(ts_code, end_date)` 内连接，事件时点取三者中**最晚**的 `available_at`；最新报表按**时间戳**取而不是按年龄并列取；年龄 > 130 个交易日即无信号，名字离开股票池；旧报告期的重述不是事件（报告期末必须在公告日前 400 个自然日内）；年初至今流量按季度序号年化。

本臂三个面各自的规则：

- **审计意见**：先取每个名字**最近一个有意见的报告期**（`end_date` 最大），再在这个报告期内取 `available_at` **最晚**的一版。这里和主表相反——主表要首版因为要的是「事件」，审计意见要最新版因为要的是「现在的结论」，一份被更正的意见由更正它的那一份取代。研究期五个决策日实测：池内名字几乎全部落在**上一个完整会计年度**（七月的决策日读到的是去年年报的意见），个别名字还停在更早一年。
- **解禁**：同一笔解禁会被**重复计数两次**，两次都能把单个持有人的一笔持仓读成总股本的一倍以上，必须一起处理。其一，同一笔解禁在落地之前会被**逐年重复公告**，每次按期间的股本变动重述股数、有时把日期挪一天，于是同一笔持仓在面板里出现三到五次（实测：一个持有人的一笔持仓按逐行相加读成总股本的 **169.9 %**；整只票在 90 天窗口里逐行相加最高读到 **358.1 %**）。其二，该数据集是两次下载的并集，**同一份公告里**同一笔持仓会以两个相邻日期出现两次，一次股数被重述、比例不变，一次反过来（实测：同一公告里同一持有人的同一笔 20.87 亿股同时挂着 65.30 % 与 81.35 %，后者是对着更早的总股本算的）。因此规则是：**一个持有人在窗口内释放的限售持仓只算一行**——取该 `(ts_code, holder_name, share_type)` 的最新公告，其中股数最大的一行，并列时取比例最小的那一行（比例最小的那个是对着最新、也就是最大的总股本算出来的）。按此规则，五个研究期决策日上没有任何一只票的合计超过 100 %（最大 65.5–75.1 %）；逐行相加则有两只超过 100 %。**残差误差是单边保守的**：同一个持有人在同一个窗口里真有两笔不同的解禁时只算一笔，老公告里有、新公告里没提的那一笔不算。
- **分部构成**：先按 `bz_code` 过滤到**一种**构成（起步值 `P` 产品），再只取 `end_date` 以 `1231` 结尾的**年度**行，再取每个名字最新的那个年度，再按 `(ts_code, end_date, bz_item)` 取首个可见版本。分母是**去掉合计行之后**分部行的和（见下一节的陷阱）。

## 陷阱

- **`fina_mainbz_vip` 把自己的合计写成了一行分部。** 每个名字在每个报告期都有一行 `bz_item` 等于构成类型自己的名字（产品构成里就是「产品」），它的 `bz_sales` 正好等于其余分部之和；另有一行「合计特别调整」。把它们当分部，分母就是真值的两倍，**每个名字的集中度都塌到约 0.5**——研究期五个决策日实测，筛后池上的集中度中位数 0.500、四分位距 **0.000**，也就是这个筛子在筛一个常数。去掉这两行之后，集中度中位数 0.576–0.600、最集中一成的门槛 0.924–0.936（`sources.md`）。
- **`float_ratio` 是百分数不是小数。** 它是单笔解禁占**总股本**的百分数，99.3 % 的行小于 1、中位数约 0.0003 %，只因为多数解禁批次很小；数据文档已按 78 万行逐行核对并标 `verified`。按小数读，`≥ 3` 这条线指向的是完全不同的一组股票。
- **`float_date` 向前读是合法的，别的事件面不是。** 解禁日期是公告里写明的未来日程，公告一发就可以用；这与「未公告的日程」「当前状态表」完全不同。反过来，极少数没有公告日期的行按解禁日当天盖章，**它们在事前永远不可见**，这是本面无法在环境内修复的覆盖缺口。
- **「读不到」不是「合格」。** 审计意见缺公告日期又无法从主表继承时**根本不进 PIT 数据**；解禁面本身在数据文档里仍标着触顶风险，只做并集不删历史。所以三个筛子一律只能写成「把被标记的名字剔出去」，绝不能写成「只留下有干净读数的名字」——后者筛的是数据覆盖率。研究期五个决策日实测：筛后池里读不到审计行的名字 0–1 只、读不到产品构成的 0–2 只，覆盖不是问题，但规则是合同。
- **一个数据集装着三套构成。** `bz_code` 的 `P`／`D`／`I` 是产品／地区／行业三种口径，混读会把同一笔收入算三遍。起步值只读 `P`，换口径是登记的变体，混读不是。
- **`audit_result` 是中文分类串，`带强调事项段的无保留意见` 仍然是无保留意见。** 起步值只命中 `保留意见`／`无法表示意见`／`否定意见`；把强调事项段加进来会把命中数从 3–9 只抬到 14–22 只，那是变体轴 c，不是口径修正。
- **`pct_chg` 两种单位**：`macro.index_daily.pct_chg` 是百分数，`daily.pct_chg` 是小数；把指数直接当小数用会把 β 压到零、残差波动等于总波动而不报错。
- **`n_income` 不是归母净利润**：`income_vip.n_income` 含少数股东损益，应计只用 `n_income_attr_p`。
- **总资产可以是零或空**：`accr`/`ocf_ta` 任一为空即无质量列，该名字不入池——池必须与姊妹臂逐字相同。
- **`is_suspended` 只标盘中临停**，整天停牌的股票在 `daily` 里没有行；用最新交易日有无行判断可交易。
- **整手与价格上限**：100 股整手，T-1 收盘价 ≤ 30 元使一手 ≤ 3,000 元；科创板 200 股起，本包剔除。
- **构造的顺序是机制的一部分**：池底 → 应计筛子 → **三个排除筛子** → 桶内填位 → 三分位配额。筛子必须在池底与应计筛子**之后**（否则它的命中数不是「筛前池的百分之几」，门 9 读的就不是同一个量），必须在排名与桶**之前**（否则被剔掉的名字还会占着保留带与三分位的位置）。三分位因此是**筛后**池的三分位。
- **上限是只数不是权重**：季度之内价格漂移会把两只同业名字的时间加权权重推到 2/15 之上，提名硬门读的 `stats.benchmark.top_industry_weight` 是时间加权量。
- **对照共用同一个池**：`c_noscreen`、`c_placebo` 与候选的股票池、池底、应计筛子、分数、中性化、桶与节奏必须逐字相同，只差筛子；`c_pool` 不成形也不设桶，是底线对照。池一变，读数就不可比，与姊妹臂 `lowvol_bucketed_20260917` 的主候选也不可比。
