# 本轮数据与 08:30 字段图

本文是本包可用数据、可见时点、单位与陷阱的**唯一权威表述**。数据集清单、行数、日期覆盖与单位以本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json` 为准：先核对它们，再写阈值或跨域组合；`snapshot` view 与 `valid` view 分别核对，**不得由前者推断后者**。

**开工第一件事**：核对四件事——`macro.datasets` 里有没有 `index_weight`（没有它就没有这个臂的股票池）与 `index_daily`（没有它就没有残差标签）；`events.datasets` 里有没有 **`limit_list_d`、`kpl_list`、`top_list`、`top_inst`** 这四张（少一张就是少一组列）；`universe` 带不带 `l1_name`（P2 的行业敞口那条要求）；以及实验事实里的 **`benchmark_index`** 与 `lib/index.py` 的 `INDEX_CODE` 是否一致。**缺任何一块都不得静默回退**到别的池子、别的标签或别的列集再按同一套口径报数——那是另一个登记，按 `families.md` 的终止规则诚实终止或改登记。起步包的 `lib/events.py` 在窗口里读不到任何可见行时直接报错，就是为了让这种情况在第一次 `smoke_backtest` 就暴露，而不是变成一块全零的特征——**本块的合法值本来就大量是零，所以「整张表没有」和「这段时间没事件」必须由代码而不是由眼睛区分开**。

## 本臂需要的数据集选择（创建参数）

| 域 | 本臂需要 | 状态 |
|---|---|---|
| `events_datasets` | 20260919 轮的 17 个，**一个都不用加** | `limit_list_d`、`kpl_list`、`top_list`、`top_inst` 全部在**默认集合**里，也全部在本轮固定的那棵预建视图种子里 |
| `macro_datasets` | 20260919 轮的 20 个（含 `index_weight`） | 不变 |
| `fundamental_datasets` / `text_datasets` | 20260919 轮的 10 个 / `["report_rc"]` | 不变。本包不读这两个域 |
| `include_intraday` | **False** | 本块的四张表都是日终榜单，与分钟域无关 |

**本臂不需要新的视图种子。** 这是它与姊妹臂的一处实际差别：本臂的四张表已经在 `data/pit_views_seed_research_20260919` 的快照契约里，所以它可以直接挂那棵种子跑，`prepare` 是硬链接、耗时约 0 秒（实测）。**这也正是这一族最刺眼的地方**：这四张表在每一条臂的快照里都躺了几十轮，谁也没读过。

`limit_step`、`limit_cpt_list`、`kpl_concept_cons`、`limit_list_ths`、`ths_hot`、`dc_hot`、`hm_detail` 这七张**默认不加载**，打开任何一张都会改变快照契约、需要一棵新种子；它们在 `families.md` 的变体轴 d 上。其中 `kpl_concept_cons` 另有覆盖问题（原始层 2025-01-02 才有行，本轮研究期只盖到最后半年），`limit_list_ths` 数据源 2023-11-01 才开始，`hm_detail` 2022-08 才开始——**打开它们之前先读覆盖，不要读成「当期没有事件」**。

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

## 本块读的四个事件数据集

事件域与宏观域一样是**多个数据集的列并集**：**先按 `dataset` 过滤，再谈字段与单位**，`dataset` 既要进下推过滤也要进列投影。按 `ts_code` 下推是本臂省时间与省内存的主要手段——这四张表都是全市场的，而成分并集小一个量级。

### `events.limit_list_d`（日终涨跌停 / 炸板标签）

| 列 | 单位 / 取值 |
|---|---|
| `limit` | 分类：**U 涨停、D 跌停、Z 炸板** |
| `limit_times` | 计数，**连板数** |
| `open_times` | 计数，**炸板次数** |
| `first_time` / `last_time` | **HHMMSS 的时钟读数**（`93151` = 09:31:51），不是一个可以求平均的数字；跌停行的 `first_time` 常为空 |
| `fd_amount` | **元**，封单金额 |
| `amount` | **元**。注意：同名列在 `daily` 里是原始湖的千元口径归一之后的元，两边**恰好同为元**，但本包只在 `limit_list_d` 内部取 `fd_amount / amount` 这个比值，不跨数据集相除 |
| `float_mv` / `total_mv` | **元** |
| `pct_chg` / `turnover_ratio` | **百分数** |

- **可见性**：行级 `available_at` = `trade_date` 当日 **16:00**（规则名 `official_16_from:trade_date`），**当日盘中不可见**，T+1 盘前可读。
- **`limit_amount` 这一列不在快照里**：供应商会把历史数值改成空，早期分区另有量级异常，快照按列排除它，只有原始湖保留供审计。不要去找它。
- **实际交易约束不看这张表**，看核心行情里的 `stk_limit`。这张表记的是「事后发生了什么」。

### `events.kpl_list`（开盘啦榜单）

| 列 | 单位 / 取值 |
|---|---|
| `tag` | 分类，五个池：**涨停、自然涨停、炸板、竞价、跌停**。分区键就是它 |
| `theme` | 文本，概念题材，「、」分隔多个 |
| `lu_time` / `ld_time` / `open_time` / `last_time` | 时钟读数，按池不同而有无 |
| `amount` / `net_change` / `free_float` / `limit_order` | **元** |
| `bid_amount` / `bid_change` / `bid_turnover` / `lu_bid_vol` | **单位未登记（`unknown`）**，而且只在 `tag=竞价` 的行上非空。**只能取秩或分位，不能作水平** |
| `pct_chg` / `bid_pct_chg` / `rt_pct_chg` / `turnover_rate` | **百分数**，同样按池不同而有无 |

- **可见性**：行级 `available_at` = **次日 08:30**（规则名 `official_next_day_0830_from:trade_date`）。
- **as-of 视图比行级时点再晚一天**：事件域的视图按每个数据集的刷新作业截断，`kpl_list` 的落地作业是 08:50 的盘前回填（外加 23:35 的晚间那次），所以 08:30 的决策只看得到前一晚 23:35 那次刷新里的行——**周二到周五决策时最新可见的是 T-2 的榜单**（T-1 那张次日 08:30 才盖章，没赶上那次刷新），**周一决策看到的是上周五的**。这与实盘一致（08:50 的作业落在 08:30 之后），是保守而不是泄漏；本块的列全是 20 日与 60 日窗口的聚合，少一天不改结论。`limit_list_d`（16:00）与 `top_list` / `top_inst`（20:00）不受影响，决策时 T-1 可见。**第 0 轮要核的是这条滞后本身**：在一个周二到周五的决策日，视图里最新的 `kpl_list` 行是 T-2；在一个周一决策日，是上周五。
- **同一个 `(ts_code, trade_date)` 可以在多个池里出现**（一只票可以既在「涨停」又在「竞价」）。起步包按池分别做指示列，再各自滚动，不做去重——**这是有意的**：在两个池里出现本身就是信息。
- **起步包不读 `bid_*` 四列**，正是因为它们的单位未登记。`tag=竞价` 的**上榜次数**被用作一列（`kpl_bid_l`），这是一个纯计数，没有单位问题。

### `events.top_list`（龙虎榜）与 `events.top_inst`（机构席位）

| 列 | 单位 |
|---|---|
| `top_list.l_buy` / `l_sell` / `l_amount` / `net_amount` / `amount` | **元** |
| `top_list.float_values` | **元** |
| `top_list.pct_change` / `turnover_rate` / `net_rate` / `amount_rate` | **百分数** |
| `top_list.reason` | 文本，上榜理由 |
| `top_inst.buy` / `sell` / `net_buy` | **元** |
| `top_inst.buy_rate` / `sell_rate` | **百分数** |
| `top_inst.exalter` | 文本，**席位名称**。机构席位统一叫「机构专用」 |
| `top_inst.side` | 分类，**`'0'` 买方席位、`'1'` 卖方席位**（字符串，不是整数） |

- **可见性**：两张表的行级 `available_at` 都是 `trade_date` 当日 **20:00**（规则名 `official_20_from:trade_date`），T+1 盘前可读。
- **重复业务键是这两张表的已知缺陷**：同一个 `(ts_code, trade_date)` 可能有多行，因为同一天被多个上榜理由列出，而其中一个理由可能是「连续三个交易日内累计达到 20 %」这样的**多日口径**——它的 `amount` 会明显大于当天的实际成交额。数据文档把这类键记为 `warning` 并要求**「使用时必须扩展业务键或先聚合，不能假定一键一行」**。
  起步包的归约写死为：`top_list` 先按 `(ts_code, trade_date)` **取均值**；`top_inst` 先按 `(ts_code, trade_date, exalter, side)` 取均值（同一席位在多个理由下的多次列示），再按 `(ts_code, trade_date)` **求和**（一天里多个机构席位）。这把多次列示当作**同一件事的多次重述**。它在 `families.md` 的变体轴 c 上，**换归约方式要重登记**。
- **机构席位的识别只看 `exalter` 是否含「机构专用」**，不看别的。实测某周龙虎榜席位行里机构专用占约 12.5 %。
- **`top_inst` 的 `net_buy` 在买方席位上通常为正、卖方席位上为负**，两侧一起求和得到的是这一天机构席位的净额——这正是本块要的量。

## 本轮要读的列

| 域 | 列 | 单位 | 行级 `available_at` | 用途 |
|---|---|---|---|---|
| `daily` | `open`/`high`/`low`/`close` | 元/股，未复权 | 无列，最新一行 = T-1 | Alpha158 的四条价格线、标签的两个开盘、买单定价 |
| `daily` | `adj_factor` | 累计复权因子 | 同上 | 前复权（锚在每只名字自己最新可见的那一根） |
| `daily` | `vol` / `amount` | **股** / **元** | 同上 | Alpha158 的量与成交均价；`amount` 还是本块所有资金比值的分母 |
| `macro.index_daily` | `open` / `close` | 指数点 | 有，`contract_1730_from:trade_date` | 残差标签的基准腿、β 的回归自变量 |
| `macro.index_weight` | `weight` | **百分数** | 有，同上 | 成员资格与覆盖度报告 |
| `events.limit_list_d` | 见上表 | 见上表 | 有，当日 16:00 | 块的「触板」与「板的质量」两组 |
| `events.kpl_list` | 见上表 | 见上表 | 有，**次日 08:30** | 块的「榜单与概念热度」组 |
| `events.top_list` / `top_inst` | 见上表 | 见上表 | 有，当日 20:00 | 块的「龙虎榜」组 |
| `universe` | `name` / `l1_name` | — | 无列，整表按回放槽锚点口径（槽内不变） | ST/退过滤、P2 的行业敞口 |

本包**不读** `fundamentals` 与 `text`，也不读毕业谱系的 `vipf2` / `mfflow` / `newage` 三组点时列：本臂要分离的是这一块，多带别的列会让「赢了是因为什么」说不清。

## 读法与可见性（通则）

```text
daily = pd.read_parquet(context.asof_dir + "/daily", columns=[...],
                        filters=[("trade_date", ">=", start), ("ts_code", "in", codes)])
```

- `asof_dir` 下每个域是 parquet parts 目录（传目录名，不加 `.parquet`），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。
- 带行级 `available_at` 的域，判可见只看它（`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `trade_date` 推。
- `daily` 与 `universe` 没有 `available_at` 列：as-of 视图只含决策时刻之前可见的行，08:30 决策时 `daily` 的最新一行就是 T-1；按不存在的列过滤会报错。
- 起步包每次读取都带 `columns=`、`filters=` 与日期窗口；不引入跨调用缓存，冷 worker 与热 worker 出单一致。

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
| `limit_list_d.fd_amount` / `amount` / `float_mv`；`top_list` 与 `top_inst` 的全部金额列 | **元** |
| `limit_list_d.open_times` / `limit_times` | **计数** |
| `limit_list_d.first_time` / `last_time`、`kpl_list.lu_time` / `ld_time` | **HHMMSS 的时钟读数**，不是数字 |
| `kpl_list.bid_amount` / `bid_change` / `bid_turnover` / `lu_bid_vol` | **未登记**，只在 `tag=竞价` 上非空；只能取秩或分位，起步包不读 |
| `top_inst.side` | 分类字符串 `'0'` / `'1'`，不是整数 |
| `macro.index_weight.weight` | **百分数**（4.64 = 4.64 %），÷ 100 才是组合权重 |
| `macro.index_daily.pct_chg` | **百分数**，÷ 100 才是日收益；`open`/`close` 是指数点 |
| `daily` 的价格 | 元/股，未复权；`adj_factor` 是累计复权因子 |
| `daily.vol`、`amount`、`circ_mv`、`total_mv` | 股、元、元、元（已从原始湖的手、千元、万元归一） |
| `daily.pct_chg`、`turnover_rate`、`turnover_rate_f` | 小数 |

## 陷阱

- **成分的代码列是 `con_code`**。忘了改名再去 join `daily`/`universe` 会得到一张空表，而空表在下游只表现为「池太小」，不会报错。
- **训练日与决策日的成分各用各自那天的截面**；第一张截面之前的日期没有成分。
- **截面 z 分数只在当天的成分集合上做**。把全市场或成分并集拿去算中位数与 MAD，得到的分数分布与决策日不是同一个口径。本块的 14 列走的是**与 Alpha158 完全相同的稳健 z 分数**，在同一个 bar 的同一个成分集合上，这样 booster 看到的是一张同质矩阵。
- **训练矩阵与决策矩阵必须同一个列顺序**。起步包只有 `data.feature_names(candidate)` 一处定义这个顺序，`primary.score` 每次都把 `state_dir` 里存下的列名与当前候选的列名逐字比对，不一致直接报错。
- **「整张表没有行」与「这段时间没有事件」是两件事。** 前者是数据集没被选进快照或到了覆盖终点，起步包**直接报错**；后者是本块最常见的合法观测，编码成显式的 0.0。**把后者也当成缺失会扔掉这块的大部分信号，把前者当成零会伪造一个观测。**
- **本块的「非空占比」恒等于 1，不要拿它当覆盖度。** 要报的是**非零占比**（`board_any_x` 为 1 的成分比例），起步包把它做成了买单元数据里的 `block_coverage`。
- **`first_time` 是时钟不是数字。** `93151` 与 `140448` 之间的差不是 47,297，起步包先折算成「离开盘有多近」的 [0, 1] 比例再用。
- **重复业务键**：见上，`top_list` / `top_inst` 必须先聚合。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；用 T-1 有无行判断可交易。Alpha158 的输入在无 K 线的日子上必须是空值（起步包把这些格子改回 NaN 再算算子）。
- **一手与整手**：100 万元 50 席上一手买不起的成分约 2–3 %（实测 2.13–2.85 %）。买不起只影响买入，**已持有的名字涨到一手超过一个座位时不得强卖**。买单按最近整手四舍五入再按现金截断，不是向下取整。
- **印花税在研究期内切换**（卖出 10 bp → 5 bp），早期研究年的往返成本更高。
- **复核节奏不得依赖模块级计数器或跨调用缓存**：到期判断要拿最新可见交易日与决策日比自然月，冷 worker 与热 worker 必须逐字节相同。**`state_dir` 里的重训计数器不是缓存**：它只在 `fit` 里读写，`generate_orders` 不看它。
- **β 只能用滞后窗口估**。起步包用 120 个交易日的滚动 OLS，向 1 收缩 0.7、截断 [0.3, 2.0]；把整段研究期的 β 一次算出来再回填是前视。
- **数据下限**：`index_weight` 从 2020-01 起，所以研究期第一个研究年的 `fit` 只有约一年半的在册训练日（三年窗口的前一半没有成分，因而没有训练行）；本块的四张表同样从 2020-01-02 起，与在册日期对齐，**本块在训练窗口里不会出现「前半段整段为空」的时间标记**。第一个研究年的读数仍要单独汇报。
