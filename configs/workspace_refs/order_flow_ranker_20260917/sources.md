# 本地实测事实与复现命令

全部统计都在 **≤ 2026-03-31** 的行上计算；没有读过任何 2026 年 6–9 月的行。本地环境 `~/miniconda3/envs/quant`，沙箱镜像 `autotrade-sandbox:latest`。

## 探针来源

本包的机制证据不是在这里重算的，它来自 `logs/notes/review_20260910/DIR2_two_more_arms.md` §2.5 与其 `logs/scratch/dir2_20260917/` 下的脚本与 JSON：

| 脚本 | 作用 |
|---|---|
| `agg_intraday.py` | 流式扫过 1,070 个分钟按日分区（20211101–20260331，1,205 秒），产出 `intraday_daily2/{2021..2026}.parquet`，5,309,485 个股票日 |
| `probe_intraday.py` | 十个日内构造的 rank-IC 表（控制集 规模 + mom20 + mom60 + turn20 + vol20 + max20） |
| `probe_mf.py` | `ofi_21` 与供应商 `mfimb_21` 的 Spearman、以及把 `mfimb_21` 加进控制集之后的复测 |
| `harness.py` / `build_panel2.py` | 日频面板（20210401–20260331，5,945,190 行）与 rank-IC 评估器 |

头条读数（控制集含 `max20`、自身 5 日反转与 `mfimb_21`）：`ofi_21` H=5 **+0.0193 (t +3.49, IR 0.24)**、五年全部同号；H=20 **+0.0285 (t +2.30, IR 0.32)**、五年中四年同号；只做多顶档五分位 +0.086%/5 日 (t 1.98)、+0.269%/20 日 (t 1.52)。`Spearman(ofi_21, mfimb_21) = 0.575`（5,094,191 个重叠股票日）。九个兄弟构造的表在 `families.md`。

## 数据面：`intraday_flow`

`ofi` 的逐日值现在由派生事件数据集 `intraday_flow` 提供，本臂不挂分钟域。数据层代码在 `src/autotrade/environment/data/intraday_flow.py`，构建脚本 `scripts/data/build_intraday_flow.py`，合同与列说明在 `docs/data-documentation.md`。

- 原始湖 `data/raw/intraday_flow/trade_date=<YYYYMMDD>.parquet`：**1,606 个分区，20200102–20260818，共 211 MB（约 0.13 MB/交易日）**。对照分钟按日层 26 GB、快照形态约 29.8 MB/交易日——一棵带分钟的种子约需 235 GB，这张表在种子里的增量可以忽略。
- 2026-03-31 分区 5,179 行；`sealed_limit` 占 0.04%、`nret < 200` 占 0.00%。
- 行级 `available_at` = 交易日 17:30（`contract_1730_from:trade_date`），08:30 只到 T-1。

## 本包新做的核对

```bash
cd /Data2/lzp/ADMCubeQuant
~/miniconda3/envs/quant/bin/python logs/scratch/refs_e_20260917/probe_min.py         # 分钟表形状（背景事实）
~/miniconda3/envs/quant/bin/python logs/scratch/refs_e_20260917/build_asof.py        # 造一棵小 as-of 树
~/miniconda3/envs/quant/bin/python logs/scratch/refs_e_20260917/add_flow_events.py   # 把 intraday_flow 并进 events 部件
docker run --rm --network none --cpus 8 --memory 24g --user $(id -u):$(id -g) \
  -v /Data2/lzp/ADMCubeQuant:/work -w /work -e HOME=/tmp \
  autotrade-sandbox:latest python logs/scratch/refs_e_20260917/identity.py           # 五列恒等
docker run --rm ... autotrade-sandbox:latest python logs/scratch/refs_e_20260917/smoke2.py   # 四条腿冒烟
PYTHONPATH=src ~/miniconda3/envs/quant/bin/python -c \
  "from autotrade.environment.strategy_loader import validate_strategy_package as v; \
   print(v('configs/workspace_refs/order_flow_ranker_20260917/starter/main.py'))"
```

### 恒等核对：事件表路径 vs 原分钟路径

同一棵 as-of 树、同一段日期，用旧的分钟流式实现（保留在 `logs/scratch/refs_e_20260917/flow_minute_ref.py`）与新的 `intraday_flow` 读取各构建一次 A 面，逐格比较：

| 比较 | 结果 |
|---|---|
| 逐股票日 `ofi` | 134,537 个股票日，**最大绝对差 0.0** |
| 逐股票日 `ofi_amt` | 同上，**最大绝对差 0.0** |
| 股票日集合 | 两边都是 134,537 行，索引逐行相同 |
| 五列 `ofi_21` / `ofi_5` / `ofi_amt_21` / `ofi_up_21` / `ofi_dn_21` | 25,734 个满窗股票日上**最大绝对差 ≤ 5.6e-17，NaN 位置零处不一致** |

五列的比较限定在「21 日窗口整个落在旧路径那 26 个交易日之内」的股票日上：旧路径没有更早的历史，窗口不满时它按 `min_periods=15` 取部分均值，而新路径能从更早的日子补满——那是历史长度的差别，不是构造的差别。协调方另在 134,537 个股票日的样本上做过 `ofi`/`ofi_amt`/`nret`/`zero_share` 的逐字节核对，结论一致。

### starter 冒烟（沙箱镜像，8 CPU，真实 as-of 布局）

as-of 树：`daily` 20250603–20260331（697,669 行，已归一化）、`events` 为 `moneyflow` 与 `intraday_flow` 的列并集（各 1,041,284 行，202 个交易日）、`universe` 5,900 只。

| 读数 | 值 |
|---|---|
| A 面构建（202 个交易日） | **6.6 秒** |
| B 面构建 | 5.8 秒 |
| 池大小（2026-03-31） | 3,343 只 |
| 覆盖率（末日） | `ofi_21` 99.86%、`mfimb_21` 99.92% |
| `o1` / `c_mf_ew` 一次决策（冷 worker） | 6.0 / 6.1 秒，各 15 张买单，严格 JSON 通过 |
| `o2` 一次 `fit` | **17.2 秒，`ok=1`**（训练正常，无退化） |
| `c_mf` 一次 `fit` | 17.9 秒，`ok=1` |
| `o2` / `c_mf` 一次决策 | 6.4 / 6.6 秒，各 15 张买单 |

对照：改用 `intraday_flow` 之前，同样五列的分钟流式实现在 26 个交易日上要 30.8 秒、一次决策 35 秒，且 `o2` 的首次 `fit` 必然 `ok=0`（决策快照只冻结 21 个交易日的分钟）。这条限制随数据面一起消失。

`validate_strategy_package` 返回 `FitSchedule(refit_period='quarter')`。

## 背景：分钟表形状（抽样 2022-01-05 / 2023-01-03 / 2024-01-02 / 2025-01-06 / 2026-01-05 / 2026-03-31）

本臂不再读这些，但它们是 `intraday_flow` 那几条不变量的来源：

- 非 BJ 股票日的 Bar 数**恒为 241**（每个抽样日 100%）；栅格 09:30–11:30 与 13:01–15:00，无 09:25、无 13:00；15:00 之后的 Bar 只有 `.BJ` 有。
- 签名分钟数中位数 **239**、1% 分位 239，`nret >= 200` 实测剔除率 **0**。
- 零成交 Bar 占 2.0%–6.5%；`close > 0`、OHLC 四值恒等、`amount = 0`。
- `zero_share >= 0.5` 的封板剔除率：0.37% / 3.43% / 0.60% / 0.91% / 0.42% / 0.31%。
- 覆盖：2026-03-31 日线 5,482 只、分钟 5,496 只，日线里**没有一只缺分钟**。

## `moneyflow` 的结构

- 八个 `buy_*`/`sell_*_amount` 是同一笔成交额的规模分档：**71.5% 的股票日买卖两侧逐分相等**（2024-05-15，5,095 只），`median |Σbuy − Σsell| = 0.0`，`gross = Σ(buy+sell)` = 当日成交额的 **2.000** 倍。
- `net_mf_amount` 与八列的代数净额几乎不相关（三日 corr 0.003 / −0.005 / 0.023），与 lg+elg 净额的 Spearman 只有 0.11–0.14；`net/(gross/2)` 落在 [−1, +1]。
- 档位占 `gross` 的中位份额：sm 0.465、md 0.337、lg 0.159、elg 0.019。
- 结论：对照必须是「八列 **+** `net_mf_amount`」。只用八列的对照代数净额恒为 0，是一个被人为削弱的对照。

## 建种子前必须核对的一条先决条件

`intraday_flow` 跟随分钟按日层，**当前终点是 2026-08-18**，而 `data/raw/daily` 已到 2026-09-09。Held-out 若定在 `20260601..20260930`，这张表在 8 月 18 日之后没有行，`ofi_21` 会在 Held-out 后段整段失效。建种子前必须确认覆盖到 Held-out 终点；补不齐就把终点收到最后一个有行的交易日（当前是 `20260601..20260818`），并在实验记录里写明这是数据边界而不是选择。

种子本身没有额外磁盘代价：`include_intraday` 保持为假，只在 `events_datasets` 里多选一个 `intraday_flow`（原始湖 211 MB，快照里的增量与它同量级）。

## 已知的口径限制（诚实记录）

- `ofi` 是分钟收盘价的 tick 规则，不是真正的成交方向分类。本地没有逐笔或 Level-2 数据，无法对照验证；与供应商分类的 Spearman 0.575 是能拿到的最接近的核对。
- `ofi_1d` 的分母是未做开盘竞价校正的 `vol`，深圳名字的 09:30 那根偏大 24%–42%，对分母的影响约 1%（开盘竞价通常占全天量 1%–3%）。数据层按探针口径固定了这一点；要测校正版本必须改数据层并重建这张表，本臂做不到，也不登记。
- 九个已证伪的兄弟构造与原计划的机制无关对照 `c_path` 都需要逐分钟 Bar，本臂读不到，因此对本臂是范围之外而不只是被证伪。
- 探针是几千只股票的五分位 rank-IC，环境交易的是 10 万元上的 10–15 只只做多篮子，翻译损失很大且没有估计。每个 IR 只值一折。
- 探针的控制腿与前向收益在同一面板上算，对探针是样本内；对一折不是，因为一折在 `fit()` 里重训。
- 探针没有做行业中性化。行业条件对日频反转腿没有增量，但一个实际上是行业押注的候选仍能通过这些探针，所以折内的申万一级中性化腿是强制的。
