# 研究会话探索计划

目标：在本臂唯一的研究会话里，判定**振幅切割动量 `a1`** 作一个不训练的规则分数，能不能在中证 500 上选出主动收益与零技能可区分的一本书；能就按冻结门走，不能就诚实地以 `no_edge` 结束。判据、预登记与验证计划只在 `refs/families.md`，数据与单位只在 `refs/pit-field-map.md`，离线门的做法只在 `refs/references/offline-screen.md`，本文只排顺序与预算。

**本臂的命题有且只有一个**：`a1` 的主动序列越过零技能（HB）。`c_shuf` 是同批的一次零技能抽样，`c_mom` 是同批的机制对照（切割 vs 不切），两者都不是门。

## 第 0 轮：不占回放预算的核对

用 `shell` 在研究期末的决策视图上跑，不启动回放。任一条不合格就在第一批的 `hypothesis` 里声明，不要硬凑。

1. **运行事实核对（最先做的一件事）**：`broker_replay.initial_cash`（应为 100 万）、**`benchmark_index`**（应为 `000905.SH`）、`acceptance_rules`（没有跟踪授权；`max_drawdown` 0.50、`active_max_drawdown` 0.30；`freeze_gate` 的阈值与试验数口径）、`budgets`、`max_replay_years`，以及 `research_geometry.years[].input_window`。`benchmark_index` 与 `lib/index.py` 的 `INDEX_CODE` 不一致时**改常量去对齐运行事实**。
2. **数据核对**：`macro` 里有 `index_weight`；`daily` 带 `high`、`low`、`pre_close`、`pct_chg`，`unit_reference.json` 把 `pct_chg` 登记为小数；`universe` 带 `l1_name`。
3. **池与覆盖普查（G-AM 的 K0 与 K1，不看收益）**：48 个复核日上的截面只数、剔科创板之后的池、有分数且买得起的只数（逐研究年最小）；`a1` 与 range20、与 `c_mom` 的截面相关。做法在 `refs/references/offline-screen.md`。
4. **账户算术表（直接对着 P3）**：座位金额、买不起的池内份额、5 元最低佣金的 bp/边、一次往返总成本（`refs/references/capital-arithmetic.md`）。
5. **换手与活动量实测（P3 的硬要求）**：`smoke_backtest` 的窗口**上限是 5 个交易日**，量不出月度复核的换手；在研究期靠后的一个月初（例如某个研究年的 7 月第一个交易日）跑 5 天，读首日建仓的 `strategy` 秒数、成交与拒单、买单元数据（`pool`、`scored`、`unbuyable_share`、`top_industry_names`）。**年换手、年成本与每月完成回合数从第一批完整期验证的 `turnover`、`fees_paid`、成交记录读**，包作者的第四研究年 70 天宿主冒烟读数在 `refs/sources.md`，只作量级参考。整个会话只用一个 `start` 做靠后窗口的探针。
6. **零技能的形状读数**：本池 50 席上随机书的主动分布（`refs/references/zero-skill-panel.md`），先知道地板在哪里。

## 第 0.5 轮：离线门 G-AM（0 回放年）

判据在 `refs/families.md`，做法在 `refs/references/offline-screen.md`。全部在 `shell` 里做，**不发起任何回放**。

1. 按时点重建每个交易日的 `a1` 与 `c_mom`（分数记在它用到的最后一个交易日）。
2. **K2**：`/mnt/tools/screen.py --signal <你的文件> --horizons 10,20,40 --start 20210701 --end 20250630 --top-fraction 0.107`，`a1` 与 `c_mom` 各一次；读 h = 20 的 `ic_mean`、`t_stat`、`ic_size_neutral`。
3. **K3**：48 个复核日的月度书（`a1` 前 50 只、行业配额 12）对同池等权均值，逐研究年；十分位剖面；200 本同池随机 50 只书的主动分位；`c_mom` 同一张表。
4. 判定并写进第一批的 `hypothesis`：K0–K3 各自的读数与是否命中。**不论读数，第一批都照登记开**（`no_edge` 本来就要一次完整期验证）；离线否决不改 HB，也不改登记。包作者的读数是 K2 与 K3 命中。

**这一步不产生提名依据。** 离线读数只有秩与方向可信；门只用来否决，裁决是宿主的整期批次。重算登记的 `a1` 与对照 `c_mom` 不计 trial；**多算任何别的配置**（别的 N、λ、振幅定义、席数、配额）就是筛过的配置，没提交就照 `offline_trials` 申报。

## 第 1 轮：唯一的第一批 `batch_validate`

**`c_shuf`（第一条，`control: true`）、`a1`（`control: false`）、`c_mom`（`control: true`）**，`span` = `full`，12 个回放年，`offline_trials` = 0（会话另筛而没提交的配置每个加 1）。

`hypothesis` 必须写清：逐行引用 `refs/families.md` 的分数表、腿表与符号预登记表；经济解释与**证伪条件**（HB：主动 IR ≥ 0.64 且 ≥ 3/4 研究年为正）；席数、复核节奏、保留带、行业配额；预期的主动 IR（0.2–0.5）与逐年符号；G-AM 的四个判据量；P2 / P3 / P5 的判定规则；本臂的 trial 账。

**不要再加零技能对照腿**——宿主每次正式验证都会抽面板，`c_shuf` 已经是同批的那一次抽样。回放一旦开跑就计入预算，策略自身抛错的候选照样花掉，环境失败的候选退还。第一批之后，对 `a1` 的完整期节点跑一次 `run_null_control`：不花回放年，**它是诊断不是判据**。

## 判定分支，逐条写死（判据本身见 `refs/families.md`）

- **`a1` 主动 IR < 0.64，或正年 < 3** → HB 没过，`finish_session(no_edge)`，读数写全（G-AM、`a1` / `c_shuf` / `c_mom` 的同批主动读数、逐年、换手与回合数）。
- **`a1` 越过 HB 但过不了冻结门** → 登记的唯一变体 `a1i`（行业内秩）一批，4 个回放年，写下此刻的 `trials_to_date` 与门槛；之后过门就冻结，不过就 `no_edge`。
- **`a1` 越过 HB 且过冻结门** → 按提名条件补齐读数（第一批已有 3 条完整期验证）后冻结。
- **`a1` 与 `c_shuf` 的主动读数相近** → 与零技能分不开，按 HB 读，不要把两者之差当增量。
- **`a1` 不高于 `c_mom`** → 结论只能写成「这个窗口里的动量」，不能写成「振幅切割」（提名条件第 4 条）。
- **绝对超额高而主动超额低** → 赢的是池不是选名；看 `benchmark.panel_neutralized_excess`。
- **某一年主动收益极端** → 只在一年成立的读数不是效应；包作者的离线读数里 Y3 与 Y4 方向相反。
- **换手、成本或回合数越线** → 先按 `refs/references/capital-arithmetic.md` 重算；**不得改成周度复核**，也不得加换名上限去压换手（那是一个新的候选配置）。
- **实现读数不对**（座位没填满、买不起份额偏高、拒单率高、池比 445 只明显少）→ 先修实现再复测，不算证伪。

## 收官轮

只做两件事之一：对已经满足全部提名条件的候选补齐缺的读数并冻结；或者以 `finish_session(outcome="no_edge")` 携读数结束本臂。

## 计算预算

- **决策**：每次复核或补座位读一次约 284 个自然日的日线（截面成员加持仓，约 500 只）与一张截面，容器内不到 1 秒；非复核日且书满时直接返回 `[]`，不读数据。实测在 `refs/sources.md`。
- **`fit`**：没有。
- **回放年**：G-AM 与 `smoke_backtest` 都是 0 年；第一批 12 年，变体批 4 年；**全臂 ≤ 16 年、host trial ≤ 2**，以运行事实 `budgets.max_replay_years` 为准。

## 每个候选必须汇报的读数

除标准结果块外，写进结果笔记：逐行对照 `refs/families.md` 的分数表与腿表；经济解释与证伪条件；**主动中性化超额、主动跟踪误差、主动信息比率**与**面板自己的中性化超额**；逐年分块的主动超额与原始超额；`c_shuf` 与 `c_mom` 的同批读数；`top_industry_weight`、`size_tilt` 与 `size_beta` 并排；G-AM 的四个读数；**实测换手、年完成回合数、每月完成回合数、按本臂 `initial_cash` 重算的佣金账与 2 倍滑点下的超额**；拒单率与原因；平均总仓位；`pnl_concentration`；空对照的三个数；一次决策的实测平均墙钟；`trials_to_date` 与此刻的 `information_ratio_bar`；账户面四项（每次复核的买不起份额、座位有没有填满、实际权重对目标权重的偏离、书覆盖的指数权重）。缺任何一项的结果不作为提名依据。

## Validation 否证（任何一条命中，结果无效，不讨论收益）

- 股票池不是按时点读出来的：硬编码名单、用别的指数建书而仍按 `benchmark_index` 报跟踪误差、或用最新一张截面去标注历史决策日。
- 分数用了决策日当天或之后的日线；`pct_chg` 当百分数又拿去做绝对阈值；把缺的交易日当 0 收益。
- 成分的 `con_code` 没有改名就去 join；读 `index_daily` 没有按 `ts_code` 过滤。
- `c_shuf` 的打乱依赖模块级状态或跨调用缓存（冷热 worker 不一致）；复核节奏依赖计数器。
- 买不起的**持仓**被当成离池强卖；或者买单向下取整而不是按最近整手四舍五入。
- 把任何东西放进 `models/`。
- 没有第 0 轮的普查、账户算术与 G-AM 重算就开了第一批；或者把 G-AM 做成一次子跨度回放。
- `c_shuf` / `c_mom` 登记成 `control: false`，候选登记成 `control: true`，或 `offline_trials` 少报了筛过而没提交的候选配置、又或把本批提交的候选重复申报。
- 看过 IC 之后改了登记符号或登记参数而没有重写 `hypothesis`。
- 用绝对中性化超额或绝对信息比率去论证边际，而不看主动读数与面板。
