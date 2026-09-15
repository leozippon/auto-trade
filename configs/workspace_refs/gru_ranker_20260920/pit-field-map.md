# 08:30 字段图

先核对本轮 `data_summary.json`、snapshot manifest 与 `unit_reference.json`。主候选只读 `daily` 与 `universe` 两个域；变体轴 a 才会读 `events.moneyflow` 或 `fundamentals`。本文是本包可见时点、单位与陷阱的**唯一权威表述**。

## 读法

```text
daily = pd.read_parquet(context.asof_dir + "/daily",
                        columns=["ts_code","trade_date","open","high","low","close","vol","amount",
                                 "adj_factor","turnover_rate"],
                        filters=[("trade_date",">=",start)])   # fit：3 年 + 150 个自然日；复核：130 个自然日
universe = pd.read_parquet(context.asof_dir + "/universe", columns=["ts_code","name","list_date"])
```

`asof_dir` 下每个域是 parquet parts 目录（传目录名），`snapshot_dir` 下是平铺文件；读失败**不得**回退 `snapshot_dir`。as-of 视图的首片只含决策时刻前可见的行、后续片只含之后发布的行，读目录无需去重。

## 逐表可见边界

| 数据集 | 本包所需字段与单位 | 行级 `available_at` | 08:30 能看到的最新行 |
|---|---|---|---|
| `daily` | `open`/`high`/`low`/`close`（元/股，未复权）、`vol`（**股**）、`amount`（**元**）、`adj_factor`（累计复权因子）、`turnover_rate`（**小数**，0.0142 = 1.42%）；变体可用 `pe_ttm`、`pb`（倍）、`circ_mv`（元） | **无 `available_at` 列**（按它过滤会 KeyError）；可见性由 as-of 视图本身给定，视图按收盘合同构建 | 最新一行 = T-1，不需要也不能再按时间戳过滤 |
| `universe` | `name`（ST/退筛选）、`list_date`（`YYYYMMDD` 字符串） | **无 `available_at` 列**；决策日冻结 | 当日 |
| `events.moneyflow`（仅变体轴 a） | 分档买卖金额 `buy_sm_amount` … `sell_elg_amount`、`net_mf_amount`（**万元**） | 有行级列，规则 `official_19_from:trade_date`（交易日 19:00） | T-1 |
| `fundamentals.*`（仅变体轴 a） | 按数据集取字段；`income_vip` 等报表金额为元 | 有行级列：报表 `source:f_ann_date_or_ann_date`，`fina_indicator_vip` `source:ann_date` | 公告日 ≤ T-1 |

有行级 `available_at` 的域（`events`、`fundamentals`、`macro`）判可见只看这一列（字符串，`pd.to_datetime(..., utc=True)` 后与 `context.inference_at` 比），不从 `trade_date`/`ann_date` 推；它们是多个数据集的列并集，**必须先按 `dataset` 过滤再谈字段与单位**。

## 面板规则（`starter/lib/panel.py`）

- 复权价 = 未复权价 × `adj_factor`；拆分一致的成交量 = `vol` / `adj_factor`；成交均价 = `amount` / `vol` × `adj_factor`（`vol` 为 0 时记空）。窗口内比值不受复权锚点影响。
- 面板的日期轴是窗口内出现过的交易日，名字轴是窗口内有过日线的全部非北交所名字。某名字某日没有日线（整日停牌）时：价格沿用前值、成交量与换手记 0、有无日线标记记 0；`c_lgbm` 的 Alpha158 在这些日子上把价格与成交量改回空值后再算。
- 「完整序列」= 截至该日的窗口内有日线的天数 ≥ 60；训练、打分与股票池都要求它。
- 标签只用于 `fit`：信号日 t 的标签需要 t+11 日的开盘价，因此只有 t + 11 ≤ T-1 的信号日进入训练，任何标签都不会越过决策时刻。
- 股票池在 T-1 这一行上判：ADV20 = 最近 20 个交易日中有成交额的日子的平均（至少 10 日），上市天数 = 决策日 − `list_date`。
- 变体轴 a 的新列：事件与报表按行级 `available_at` 换算到上海时间的日期，映射到**严格晚于**该时刻的第一个 08:30 决策所对应的信号日（即 19:00 盖章的 T 日资金流进入 T 日这一行，因为 T+1 08:30 才用它；报表同理按公告时刻），再前向填充到后续交易日；训练面板与决策截面必须用同一个函数完成这一步。

## 陷阱

- **`turnover_rate` 是小数**：快照已归一化；方向研究的原始湖是百分数，起步包乘 100 后取 `log1p` 以对齐研究时的尺度。
- **原始湖单位不同**：离线普查直接读 `data/raw` 时 `vol` 是手、`amount` 是千元、`circ_mv` 是万元；快照里分别已是股、元、元。
- **`is_suspended` 只标盘中临停**，整天停牌的名字在 `daily` 里没有行；判可交易用 T-1 有无行。
- **模型文件不在 `output/` 里**：权重与 booster 只写 `context.state_dir`，每次回放开始时为空；不要把训练好的权重放进 `models/` 以绕过重训。
- **显存**：训练把全部训练日的输入窗口以 float16 缓存在卡上（60 日序列约 3 GiB）；加长序列或加特征会线性放大，先量峰值再跑正式验证。
- **科创板与北交所**：北交所在面板里就被剔除；科创板 200 股起买，只在股票池里剔除，仍参与训练。
