# 研究会话探索计划

目标：在本臂唯一的研究会话里，按登记量一次「中证 500 内的沪深 300 纳入预判」这本书，写全读数后以 `no_edge` 收尾（K0 已失败，`refs/families.md`）。判据、预登记与验证计划只在 `refs/families.md`，数据与单位只在 `refs/pit-field-map.md`，重算的做法只在 `refs/references/offline-screen.md`，本文只排顺序与预算。

## 第 0 轮：不占回放预算的核对

用 `shell` 在 `/mnt/snapshot` 上跑，不启动回放。任一条不合格就在第一批的 `hypothesis` 里声明，不要硬凑。

1. **运行事实核对（最先做）**：`broker_replay.initial_cash`（本包按 100 万元写）、**`benchmark_index`**（应为 `000905.SH`；与 `lib/knobs.py` 的 `INDEX` 不一致时改常量去对齐运行事实）、`acceptance_rules`（两个回撤上限应为 0.50 / 0.30，不设授权；`freeze_gate` 的阈值与试验数口径）、`budgets`、`max_replay_years` 与 `research_geometry.years[].input_window`。
2. **数据核对**：`macro` 里 `index_weight` 有 `000905.SH` 与 `000300.SH` 两只指数的截面；`daily` 带 `amount`、`total_mv`、`close`、`is_suspended`；`universe` 带 `list_date`、`name`、`l1_name`；`unit_reference.json` 对 `amount` 与 `total_mv` 的登记（元）。
3. **K0 重算（不读收益）**：八次审核的命中份额、精度与纳入者来源（`refs/references/offline-screen.md`）。与 `refs/sources.md` 的逐次表一致就照登记走；不一致，先查截面的取法（是不是用了决策日之后的截面）与样本空间的口径，再按登记走。
4. **离线筛 S1–S4 重算**：`screen.py` 全期与逐年各一次（`m1` 与散列随机），月度书与窗口份额。这些是收尾读数，不改变那一批的登记。
5. **冒烟与活动量**：`smoke_backtest` 至多 5 天，只用来确认合同、`strategy` 秒数与第一天的 30 张买单（买不起份额、实际权重）；**换手、每月完成回合数与平均总仓位读第一批完整期结果**，不要用冒烟外推。整个会话靠后窗口的探针只用一个 `start`。

**这一步不产生提名依据。** 离线读数只有秩与方向可信；K0 已经决定了本臂的路径。

## 第 1 轮：唯一一批完整研究期 `batch_validate`

腿与登记按 `refs/families.md`：**`c_shuf`（第一条，`control: true`）、`m1`（`control: false`）**，`span="full"`，8 个回放年，`offline_trials` = 0（会话若另筛了任何变体，每个加 1，如实申报）。

`hypothesis` 必须写清：逐行引用 `refs/families.md` 的分数表、腿表与符号 / 量级表；K0 的读数与它的登记后果（这一批之后 `no_edge`）；S1–S4 的读数；宿主线 H 的定义；本臂的 trial 账（M = 1、N_eff = 1、门槛约 0.98）。

**不要再加零技能对照腿**——宿主每次正式验证都会抽面板，`c_shuf` 已经是同批的那一次抽样。第一批之后对 `m1` 的完整期节点跑一次 `run_null_control`：不花回放年，它是诊断不是判据。

## 收官轮

写全 `refs/families.md`「宿主读法」那一段的读数：`m1` 与 `c_shuf` 的主动中性化超额、主动跟踪误差、主动 IR、面板自己的中性化超额、逐年主动超额，H 是否越过，窗口内外的拆分（P5），`size_tilt` 与 `size_beta`，实测年换手、每月完成回合数（逐月看有没有零回合的月份）、`excess_at_2x_slippage`、平均总仓位，空对照三个数；然后以 `finish_session(outcome="no_edge")` 结束，理由里交代 K0 的结构原因。写一条 skill：纳入规则的时点复刻与 K0 的做法（引用节点 id 与读数），供以后在别的池（例如不在任何宽基里的「前 300 候补」）上开这一族时复用。

## 计算预算

- **决策**：复核日与补座日读全部 A 股一年的 `amount` / `total_mv`（约 120 万行）与两张截面，容器内几秒；非复核日只读 12 天的交易日列就返回 `[]`。实测读数在 `refs/sources.md`。
- **回放年**：K0、离线筛与 `smoke_backtest` 都是 0 年；唯一一批 8 年。
- **不做的事**：不开变体批、不换池、不加腿去追一个读数——K0 失败之后，任何一批额外的回放都是在一个已经证伪的形状上花钱。

## Validation 否证（任何一条命中，结果无效，不讨论收益）

- 股票池或在位名单不是按时点读的：用了日期不早于决策日的截面、硬编码名单、或用最新一张截面标注历史决策日。
- 纳入规则的窗口越过了 T-1，或把 `total_mv` / `amount` 当成别的口径去设阈值。
- `c_shuf` 的打乱依赖模块级状态或跨调用缓存；复核节奏依赖计数器。
- 买不起的**持仓**被当成离池强卖；或者买单向下取整而不是按最近整手四舍五入。
- `c_shuf` 登记成 `control: false`、`m1` 登记成 `control: true`，或 `offline_trials` 少报了会话另筛过的变体。
- 看过读数之后改了登记而没有重写 `hypothesis`；或在 K0 失败之后开了登记之外的批次。
- 用绝对中性化超额或绝对信息比率去论证边际，而不看主动读数与面板。
