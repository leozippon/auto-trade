# PIT 字段图：这份机制真正读到的东西

只列继承产物**实际读取**的列。它一共只碰四个 as-of 域，其余全部不读。本臂不许加列，所以这张表同时也是「允许读的全集」——出现任何不在表里的读取，就是改了特征，属于禁止行为。

判可见只看 `available_at <= context.inference_at`，推断时点是决策日 08:30+08:00。所有读取都要带 `columns=`、`dataset` 过滤与日期窗口；as-of 域读失败**不得**回退 `snapshot_dir`。

## `asof_dir/daily`

一张已经归一化并联结好的日频表（日线 + 估值 + 交易约束）。日频域没有 `available_at` 列，可见性就是 `trade_date < T`（as-of 视图本身已经按时点切好，代码里再显式过滤一次）。底层各来源的盖章合同分别是 `daily` 交易日 17:30、`daily_basic` 18:00、`adj_factor` 09:30、`stk_limit` 与 `suspend_d` 08:45——在 08:30 的决策上，**全部只到 T-1**。

| 列 | 谁读 | 单位（快照已归一化） | 备注 |
| --- | --- | --- | --- |
| `ts_code` / `trade_date` | `lib/data.py`、`lib/mfflow.py` | — | `trade_date` 是 `YYYYMMDD` 字符串 |
| `open` / `high` / `low` / `close` | `lib/data.py` | 元/股 | 158 算子与 `VWAP` 的输入；按每只票最后一根可见 bar 的 `adj_factor` 折成前复权 |
| `vol` | `lib/data.py` | 股（源为手，快照 ×100） | 复权因子在 `VWAP = amount / vol` 里抵消，所以 `VWAP` 用未复权值 |
| `amount` | `lib/data.py`、`lib/mfflow.py` | 元（源为千元，快照 ×1000） | `ADV20` 与 `mf_amtturn20` 的分母；`ADV20_MIN = 3e7` 就是 3000 万元 |
| `adj_factor` | `lib/data.py` | 无量纲 | 复权锚固定在 T-1：每只票的因子 = 该行因子 ÷ 该票最后一根可见行的因子 |
| `up_limit` | `lib/data.py` | 元/股 | 过滤 T-1 收在涨停的票；Broker 另有自己的涨跌停拒单，两者不是一回事 |
| `is_suspended` | `lib/data.py` → `universe()` | 分类 | **v11 语义变了，见下** |
| `circ_mv` | `lib/mfflow.py` | 元（源为万元，快照 ×10000） | 8 列资金流特征里 6 列的分母；缺失 / ≤ 0 / NaN → 该格 NaN |

**`is_suspended` 的 v11 语义（本臂第一个要确认的事）。** 该标志按 `suspend_d` 的停牌行（`suspend_type="S"`）置位。全日停牌当天供应商根本不发日线行，所以**有日线的那天被置位只可能是盘中临时停牌**，当天成交量额完整。v10 及更早**把复牌行（`"R"`）也一并置位**，而复牌日是正常交易日：审计窗里 946 个置位行中 747 行是正常交易的复牌日。继承来的 `universe()` 第一道过滤就是 `~suspended`，因此它在 v9 视图上误剔除了大量正常可交易的名字，在 v11 上这些名字回到可选池。**可选池会变宽，top16 的成分可能变，这不是「修好了」，是换了一个池子。** 日线上永远不存在连续多根停牌 bar，按「连续停牌 N 根」写的规则取不到任何事件——本臂不需要构造停复牌事件，只需要知道这个池子变了。

## `asof_dir/fundamentals`（`lib/vipf2.py`，6 列）

只读四个数据集：`fina_indicator_vip`、`forecast_vip`、`express_vip`、`income_vip`。行级 `available_at` = 公告日 18:00，因此在 08:30 的决策上等价于 `ann_date <= T-1`。同一 `(dataset, ts_code, end_date)` 有多版本时，t 时刻生效的是 `available_at <= t` 里最大的那一版；完全重复的行按稳定排序取最后一次出现。**修正版本必须按这条规则取，不能按 `end_date` 取最新一行。**

| 数据集 | 列 | 单位 | 用途 |
| --- | --- | --- | --- |
| `fina_indicator_vip` | `or_yoy` / `netprofit_yoy` / `dt_netprofit_yoy` / `op_yoy` | 百分数 → 除以 100 | 4 列成长特征；无可见行 → NaN |
| `forecast_vip` | `type` / `net_profit_min` / `net_profit_max` / `last_parent_net` | 分类 / 万元 | `fc_dir`（预增·扭亏·略增 → +1，预减·首亏·略减 → −1，其他 → 0）与 `fc_exp_rev` 的预告腿 |
| `express_vip` | `n_income` / `yoy_net_profit` | 元 | `fc_exp_rev` 的快报腿；与同一 `(ts_code, end_date)` 的预告配对 |
| `income_vip` | 仅 `(ts_code, end_date, available_at)` | — | 只用来标记「正式报告已披露」，据此把 `fc_dir` 归零；不取任何数值 |

单位在每个特征内部自洽（万元/万元、元/元），**绝不跨表混用**。缺数据不造值：`fina_indicator` 四列保持 NaN（RobustZScore 之后映射为 0，与 Alpha 列同一个中性值），两个事件列按构造为 0。

已记录的张力（源实验 PRIOR 的待审计项，本臂**不动**、只需知道）：`fc_dir` 与 `fc_exp_rev` 取值在 {−1, 0, +1}，无预告/无快报的标的构造性为 0，决策截面上零占比可能过半；过半时 MAD = 0 会把该列机械压平成常数，成为死重。这是编码问题不是机制问题，树模型忽略常数列因而无害。**本臂禁止以此为由改动这两列**——改列就是改特征。

## `asof_dir/events`（`lib/mfflow.py`，只读 `moneyflow`）

行级 `available_at` = `trade_date` 当日 19:00（规则名 `official_19_from:trade_date`），因此在 08:30 的决策上只到 T-1，当日盘中不可见。过滤只看 `available_at`，**绝不用 `trade_date < T` 的捷径**：一条 `trade_date == T` 的行被排除，唯一理由是它的 `available_at`（T 19:00）晚于推断时点。同 `(ts_code, trade_date)` 多版本时取 `available_at <= t` 里最大的那一版，去重发生在开窗之前。

| 列 | 单位 | 用途 |
| --- | --- | --- |
| `net_mf_amount` | 万元（×1e4 → 元） | `mf_amt_5`、`mf_amt_20`、`mf_pos20`、`mf_trend520`、`mf_vola5_20`、`mf_amtturn20` |
| `buy_elg_amount` / `sell_elg_amount` | 万元 | `mf_elg_net5`（特大单净额） |
| `buy_lg_amount` / `sell_lg_amount` | 万元 | `mf_lg_net20`（大单净额） |

窗口是**日历天**：「T 之前 K 个日历日」= 可见行里 `trade_date >= T − K` 天的那些，即 `{T-K, …, T-1}`。窗内一行都没有 → 该格 **NaN，不是 0**：`moneyflow` 覆盖接近满，用 0 会造出一团中位数原子把这一列在 RobustZScore 下压平。NaN 让截面统计只在有行的票上算，无行的票在 zscore **之后**才被填 0。

## `asof_dir/universe`（`lib/newage.py`，1 列）

静态上市维表，只读 `ts_code` 与 `list_date`。`age = log1p(决策日 − list_date)`，单位是日历天；决策路径与 fit 路径共用同一个构造器。as-of 视图可能已按 `list_date <= T` 裁过，也可能没裁，两种形态都要能处理——**表里没有这只票 → NaN**，不要用当日日期或 0 顶上。`T < list_date` 也是 NaN（防御性，带 bar 的票不会触发）。

## 明确不读的域

`macro`、`text`、`intraday_1min`，以及 `events` 里除 `moneyflow` 之外的全部数据集、`fundamentals` 里除上述四个之外的全部数据集。本臂不得增加任何一个。

一条直接后果：v10 把当日发布的宏观日频表改成收盘合同盖章（`contract_1730_from:<日期列>`，因此比 v9 早一个交易日可用）**对这份机制完全没有影响**，因为它一列宏观都不读。种子里仍然选了宏观数据集，那是本轮各臂共用一棵视图树的要求，不是这份策略的输入。不要为这条变更写任何适配代码，也不要把它当成 v9 → v11 的风险项——本臂真正的数据合同风险只有上面那一条 `is_suspended`。
