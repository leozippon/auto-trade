# 08:30 字段图

本文是本包可见时点、单位、去重规则与陷阱的**唯一权威表述**。本臂读五个域，两条 sleeve 各读自己需要的那几块——**读得起的前提是复核日才读**：非复核日在确认过最新可见交易日之后直接返回 `[]`。

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`：

- `fundamentals.datasets` 必须列出 `income_vip`、`cashflow_vip`、`balancesheet_vip`（缺任一，sleeve 甲的质量腿不可构造，本臂无从合成），以及 `fina_indicator_vip`、`forecast_vip`、`express_vip`（缺任一，sleeve 乙的 `vipf2` 六列不可构造）；
- `events.datasets` 必须列出 `moneyflow`（缺它，`mfflow` 八列不可构造）；
- `macro.datasets` 必须列出 `index_daily`（缺它，sleeve 甲的低波腿不可构造）；
- `universe` 必须带 `l1_name`（缺它，门 5 的行业集中度与构造里的行业列都没有来源）。

缺任何一块时**不得静默回退**：按 `families.md` 的终止规则诚实终止，或者把「本轮跑的是 `data.FEATURE_GROUPS` 的哪个子集」写进 `hypothesis`（那是变体轴 e，不是意外）。

## 读法

```text
# sleeve 甲（composite）：日线 + 指数 + 三张报表 + universe
daily = pd.read_parquet(context.asof_dir + "/daily",
                        columns=["ts_code","trade_date","close","amount","adj_factor",
                                 "circ_mv","pe_ttm","pb","is_suspended"],
                        filters=[("trade_date",">=",start)])                  # 220 个自然日
index = pd.read_parquet(context.asof_dir + "/macro",
                        columns=["dataset","available_at","ts_code","trade_date","pct_chg"],
                        filters=[("dataset","=","index_daily"),("ts_code","=","000300.SH"),
                                 ("trade_date",">=",start)])                  # 220 个自然日
stmts = pd.read_parquet(context.asof_dir + "/fundamentals",
                        columns=["dataset","available_at","ts_code","end_date","ann_date",
                                 "report_type","n_income_attr_p","n_cashflow_act","total_assets"],
                        filters=[("dataset","in",["income_vip","cashflow_vip","balancesheet_vip"]),
                                 ("ann_date",">=",start)])                    # 400 个自然日

# sleeve 乙（pv）：日线宽表 + 三块点内扩展
daily = pd.read_parquet(context.asof_dir + "/daily",
                        columns=["ts_code","trade_date","open","high","low","close","vol",
                                 "amount","adj_factor","is_suspended","up_limit"],
                        filters=[("trade_date",">=",start)])   # 决策 130 个自然日；fit 训练年 + 110
vipf2 = pd.read_parquet(context.asof_dir + "/fundamentals", columns=[...],
                        filters=[("dataset","in",("fina_indicator_vip","forecast_vip",
                                                  "express_vip","income_vip")),
                                 ("ann_date",">=",start)])
mf    = pd.read_parquet(context.asof_dir + "/events", columns=[...],
                        filters=[("dataset","in",("moneyflow",)),("trade_date",">=",start)])
univ  = pd.read_parquet(context.asof_dir + "/universe",
                        columns=["ts_code","name","list_date","l1_name"])
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；**读失败不得回退 `snapshot_dir`**。`fundamentals`、`macro` 与 `events` 都是多个数据集的列并集：同名列在不同数据集里含义与单位不同，**必须先按 `dataset` 过滤再谈字段与单位**，而且 `dataset` 既要进 pushdown 过滤也要进列投影（多数据集文件共用 `ts_code`）。

两类表要分开：`fundamentals`、`macro`、`events` 的行带**行级 `available_at` 列**，判可见只看它（字符串，`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `ann_date`/`trade_date` 推；`daily` 与 `universe` **没有 `available_at` 列**（决策视图的 `daily.parquet` 30 列，按它过滤会 KeyError），可见性由 as-of 视图本身给定——08:30 决策时 `daily` 的最新一行就是 T-1，不需要也不能再按时间戳过滤，只按 `trade_date < T` 取。

## 逐表可见边界

| 数据集 | 本包所需字段与单位 | 行级 `available_at` 规则 | 08:30 能看到的最新行 |
|---|---|---|---|
| `daily` | `open`/`high`/`low`/`close`（元）、`vol`（股）、`amount`（**元**）、`adj_factor`、`circ_mv`（**元**，已归一化）、`pe_ttm`（倍）、`pb`（倍）、`is_suspended`、`up_limit` | **无该列**；可见性由 as-of 视图隐含 | 最新一行 = T-1 |
| `universe` | `name`（ST 筛选）、`list_date`（`newage` 的唯一来源）、`l1_name`（行业归因的同一列） | **无该列**；决策日冻结 | 当日 |
| `fundamentals.income_vip` | `end_date`、`ann_date`、`report_type`（只用 `"1"`）、`n_income_attr_p`（**元**，年初至今累计） | `source:f_ann_date_or_ann_date` 18:00 | 公告日 ≤ T-1 的报表 |
| `fundamentals.cashflow_vip` | 同上键；`n_cashflow_act`（**元**，年初至今累计） | 同上 | T-1 |
| `fundamentals.balancesheet_vip` | 同上键；`total_assets`（**元**，期末时点值） | 同上 | T-1 |
| `fundamentals.fina_indicator_vip` | `or_yoy`、`netprofit_yoy`、`dt_netprofit_yoy`、`op_yoy`（**百分数**，÷ 100） | 同上 | T-1 |
| `fundamentals.forecast_vip` | `type`、`p_change_min/max`、`net_profit_min/max`（**万元**）、`last_parent_net`（**万元**） | 同上 | T-1 |
| `fundamentals.express_vip` | `n_income`、`yoy_net_profit`、`revenue`（**元**） | 同上 | T-1 |
| `events.moneyflow` | `net_mf_amount`、`buy_elg_amount`/`sell_elg_amount`、`buy_lg_amount`/`sell_lg_amount`（全部 **万元**，× 1e4 转元） | `official_19_from:trade_date` 19:00 | T-1（19:00 > 08:30，当日行永不可见） |
| `macro.index_daily` | `ts_code = 000300.SH`、`trade_date`、`pct_chg`（**百分数**，÷ 100 才是日收益） | `contract_1730_from:trade_date` 17:30 | T-1 |

原始湖 `data/raw/daily` 的 `amount` 是**千元**、`circ_mv` 是**万元**；离线普查直接读原始湖会差 1000 倍 / 1 万倍，快照里已归一化为元。**sleeve 甲的池内市值下限排的就是 `circ_mv`，这一处单位错会静默地把整条构造规则反过来。**

快照决策视图里三张报表只含 `report_type = "1"` 的行，过滤仍要写，别的视图不保证。

## 申万一级行业是点内的

`universe.l1_name` 不是当前归属回填出来的：宿主按**决策日**取成分（`in_date ≤ 决策日 < out_date`），并按当天市场实际在用的口径取——申万在 2021-12-13 从 SW2014 切到 SW2021，该日之前的决策日用冻结的 SW2014 成分、之后用 SW2021。研究期跨这个切换点，所以一级行业的个数与名字在研究期内会变（SW2014 二十八个、SW2021 三十一个，加上 `未分类`），**跨决策日比较行业名单是错的**，普查表要按决策日分别列。`l1_name` 为空的名字落在 `未分类` 这一个桶里，宿主侧的行业归因用同一个名字与同一个标签，所以本包里的行业读数与门 5 读的 `stats.benchmark.top_industry_weight` 说的是同一件事。申万成分历史由供应商回填、**纳入日期可能异常**（有名字的纳入日早于其上市日），所以纳入日期不能当作可靠的行业变更日；本包只用「决策日归属」。

## 去重与版本规则

- **三张主报表取首版**：同一 `(ts_code, end_date)` 在每张表里可能有多个版本（首次公告、修订、年报重述），每张表各取 `available_at` **最早**的一版；重述版永远不是事件，也永远不替换状态。三张表按 `(ts_code, end_date)` 内连接，事件时点取三者中**最晚**的 `available_at`。
- **`vipf2` 取在效版本，不取首版**：`fina_indicator_vip` 的同一 `(ts_code, end_date)` 要取 `available_at ≤ t` 中**最大**的那一版（在效版本），而「最新报告期」是首次可见时间最晚的那个 `end_date`。这条与上一条方向相反，是两条 sleeve 的不同约定，不得互相套用。
- **最新报表按时间戳取，不按年龄并列取**：窗口之前公告的报表年龄全部并列在窗口长度上，按行序打破并列会取到最旧的一份。年龄 > 130 个可见交易日即无信号，名字离开 sleeve 甲的池。
- **旧报告期的重述不是事件**：首版早于读取窗口的重述在窗口里看起来就是首版，所以报告期结束日必须在公告日前 400 个自然日内，否则丢弃。
- **年初至今按季度序号年化**（`× 4 / qn`）：三月末 ×4、六月末 ×2、九月末 ×4/3、十二月末 ×1。同一决策日不同名字可能处在不同的最新季度。
- **`moneyflow` 按天取最新可见行**，同 `(ts_code, trade_date)` 若有重复，按稳定排序取文件里最后一次出现。
- **60 日波动窗**：最近 60 个可见交易日，个股与沪深 300 都有值的日数 ≥ 40 才有 `ivol`；整日停牌的名字在 `daily` 里没有行，窗口按可见行计，不补零。

## 陷阱

- **`pct_chg` 两种单位**：`macro.index_daily.pct_chg` 是百分数（`0.68` = 0.68 %），`daily.pct_chg` 是小数。不除 100，β 会被压到接近零、残差波动退化成总波动而且不报错，sleeve 甲的低波腿整条错掉。
- **`n_income` 不是归母净利润**：`income_vip.n_income` 含少数股东损益，应计只用 `n_income_attr_p`。`express_vip.n_income` 是另一张表的另一个口径，不要混。
- **万元与元三处**：`moneyflow` 的 `*_amount`、`forecast_vip` 的 `net_profit_min/max` 与 `last_parent_net` 都是万元；`express_vip` 的列是元。每个特征内部必须单位自洽（万元/万元、元/元），**永远不跨单位相除**。
- **池内分位 ≠ 全市场分位**：sleeve 甲的市值下限在**已筛过的截面内部**算 `rank(pct=True)`。在池外或筛选前算同一个分位，砍掉的名字完全不同（读数见 `references/composite-legs.md`），那是禁止表里的另一个形状。
- **两条 sleeve 的池本来就不同**，这不是缺陷：sleeve 甲要求四条腿的列非空并加市值下限，sleeve 乙不要求报表列也没有市值下限。合成买的正是这种不同。任何把两个池并起来的改动都直接削掉本臂要买的那个相关系数。
- **`is_suspended` 只标盘中临停**，整天停牌的股票在 `daily` 里没有行；用最新交易日有无行判断可交易。T-1 整日停牌的**持仓**在窗口里有行但最后一根没有价格，估值必须兜底为 0，否则一个 NaN 会把整个买入预算变成 NaN。
- **卖出估值的价格表要覆盖全部有 K 线的名字**，不是只覆盖池内名字：离池强卖也要估值，估成 0 会把买入预算按那笔卖出的现金整块压掉。
- **整手与价格上限**：100 股整手，T-1 收盘价 ≤ 30 元使一手 ≤ 3,000 元；科创板 200 股起，本包剔除。`W_A = 0.30` 时 sleeve 甲单只目标约 2,000 元，**一手就可能超过目标**——买单按最近的整手四舍五入再按现金截断，实际权重因此是一个要汇报的读数（`families.md`）。
- **特征列顺序是合同**：sleeve 乙的 booster 按列位置吃数，`score` 会把 `feature_names.npy` 与 `data.FEATURE_NAMES` 逐字比对。改特征集是变体轴 e，不是随手改。
- **决策窗口长度由最长算子决定**：sleeve 乙的决策路径读 130 个自然日（≥ 63 个交易日）才喂得满 d = 60 的滚动算子；sleeve 甲读 220 个自然日（长于 130 个交易日的报表年龄上限）。缩短窗口会静默地把最长窗口那批列变成 NaN 再被标准化映成 0。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：两条 sleeve 的到期判断都是拿最新可见交易日与决策日比自然月 / ISO 周，冷 worker 与热 worker 必须逐字节相同。
