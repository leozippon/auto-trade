# 来源与诚实边界

这些材料只提供机制先验。论文里的收益、IC、分位表不是本环境 Validation 预期；不要粘贴第三方完整源码，只引用并转述机制。

## 文献先验

- **主证据（A 股复制）**：Hou, Qiao, Zhang 等，"Replicating and Digesting Anomalies in the Chinese A-Share Market"，*Management Science* 2024——在 A 股复制美股异象清单时，分析师预测变化（analyst forecast change）多空组合月度 0.32%、t = 2.32（12 个月持有），是少数几个复制成功的基础异象之一。同一篇报告 CH-3/CH-4 因子模型能吸收其中相当一部分，**这是本包把 EP 价值腿写成强制对照而不是稳健性附加项的直接原因**，也是家族 1 的否证条件 (c) 的来源。
- **原始机制**：Givoly–Lakonishok（*JAE* 1979）与 Stickel（*JF* 1991）给出分析师预测修正后的价格漂移；Chan–Jegadeesh–Lakonishok（*JF* 1996）把它与盈余动量并列为两条独立的信息不足反应通道。修正被视为「关于基本面的新信息」，而不是价格的函数——这正是它在本环境里与已关的六个价量/被动披露臂结构不同的地方。
- **配对口径**：修正研究的标准做法是只比较**同一分析师**前后两次预测，避免一致预期的构成变化被读成修正。本包家族 2 就是这条，实测支持它的必要性：20 日窗口内每股中位只有 1–2 家券商，换一家发报告就能整体平移一致预期。
- **离散度**：Diether–Malloy–Scherbina（*JF* 2002）——分析师意见分歧大的股票随后收益更低，机制是卖空受限下由乐观者定价。A 股卖空高度受限，方向先验比美股更强，因此家族 4 的方向是预设的负号而不是待定。
- **不作数值预期**：以上论文的样本期、股票池、持有期与交易成本都与本折不同。D1 的本地实测（20 日控制后 IC +0.025、t +2.83、IR 0.41，多头前五分位控制后 +0.20%/20d、t +1.08）才是本臂唯一的本地先验，而它离成本线只有几个基点。

## 与已试家族的结构差异，以及必须避开的已关邻域

现有六臂的事件族全部是**公司被动披露的事件**（业绩预告与快报、解禁、股东增减持、分红、ST）或**市场自身的成交痕迹**（龙虎榜、涨停炸板、资金流、大宗、两融）。分析师盈利预测是**卖方主动生产的、关于未来的数值判断**，在全库里独一份。

但 `report_rc` 这张表本身**不是没被碰过**，必须精确区分：

- open_mechanism fold11 判定：`report_rc` 的文本通道 body 均 29 字符 ≈ 只有标题，「可用机制仅剩计数/覆盖 = 已关卖方族邻域 → 文本通道无载体」。
- ml_ranker fold10 第二次 screen 弃权后，「文本通道永久关闭」，其中含「research_report/评级事件全部关闭」。

因此本臂**唯一新增的机制是数值预测水平的修正**。覆盖度计数、首次覆盖、评级事件都在已关邻域内，在本包里只能以家族 5 的对照腿形式出现，提名它们等于重开一个已关家族。这一条写进了 README 的臂级终止裁决与 `families.md` 的「明确不做」。

## 已知不可实现与已被排除

- **数值切片已落地**（commit `65edfae`）：`report_rc` 的数值/评级列现在是 events 域的一个数据集，单位已登记；文本域仍只有标题。切片**默认不加载**，实验必须在 `events_datasets` 里显式选入，否则本臂不可测（见 `exploration-plan.md` 第 1 步的生死门）。
- **目标价家族不可测**：`tp` 是利润总额（万元）不是目标价；`max_price` 非空率 0.79%，`min_price` 29.72% 且只是区间下沿。按 `min_price` 建修正家族的可选股票数中位 183 只。
- **评级变动家族不可测**：`imp_dg` 全库 0 行非空；自建序数映射后每月末发生变动的股票数中位 36 只。
- **研报正文不可得**：文本域只有标题，`context.nl` 问不出研报观点。
- **前几轮已被证伪、本臂不得复用作候选**：`factor_cs` 的价值/反转/成长/趋势线性打分与日频彩票族、`value_regime` 的 EP 与股息两腿、`site_visits` 的五个调研家族、`cb_linkage` 的五个转债家族、`explore_platform` 的涨停与热榜 playbook、`explore_github` 的业绩预告漂移与资金流混合。它们只能作为 `hypothesis` 里写明的机制无关对照。

## 本仓库

- 本地笔记：`logs/notes/review_20260909/D1_directions_from_data_20260916.md`（§1.1 数据清单、§2.1 探针表的 P6 两行、§2.2 P6 的预登记否证、§4 R1 推荐）；`logs/notes/review_20260909/D2_directions_from_literature_20260916.md`（§1 与 §5 的 `report_rc` 文本通道已关记录，§6 的账户容量约束；其中「`report_rc` 数值列在本环境不可实现」一条已被 commit `65edfae` 取代）。D1 的探针脚本在 `logs/scratch/d1_20260916/`。
- 唯一总闸是 `available_at <= inference_at`；可见时间规则见 `docs/data-documentation.md` §3.3 与 §4 的 `report_rc.create_time` 回填风险条目，单位见 `docs/units-reference.md`（沙箱里以 `unit_reference.json` 为准）。
- 正式 ABI、允许的库、`fit(context)`/`context.state_dir`、`models/` 只读挂载见只读 `output/README.md`。
- 离线筛查脚本 `/mnt/tools/screen.py`（只经 `shell` 运行）给 rank IC、ICIR、衰减、规模中性 IC、头部十分位超额、换手与可交易性；它只读 `/mnt/snapshot` 决策视图，不替代 Validation。匹配对照的普查要用自己的脚本，同样只在输入窗上做。
- `starter/main.py` 是家族 1 的最小合同示例，不是可提交产物：它按已落地的切片列名书写并通过 `validate_strategy_package` 静态检查，仍必须先按本折 `data_summary.json` 核对再重写。

## 本包数字的出处

`README.md`、`families.md`、`pit-field-map.md` 里的每个覆盖、重复率、盖章占比、单位与可选股票数都来自 2026-09-16 对 `data/raw/report_rc/`（80 个月分区，取 `month <= 202512` 的 72 个、1,407,073 行）、`data/raw/daily/`、`data/raw/trade_cal/exchange=SSE/` 的只读普查，**全部限定在 `report_date`/`available_at` ≤ 2025-12-31 的行上**。脚本在 `logs/scratch/refs_a_20260916/`，沙箱里以本轮 `data_summary.json` 为准：

```bash
PY=~/miniconda3/envs/quant/bin/python
$PY logs/scratch/refs_a_20260916/probe_report_rc.py   # 行数、逐年覆盖、quarter 语义、imp_dg、盖章规则、重复率、单位、券商集中度
$PY logs/scratch/refs_a_20260916/probe_coverage.py    # 23:35 晚间节点、.BJ 代码口径、家族 1 的 48 个月末可选股票数
$PY logs/scratch/refs_a_20260916/probe_tradable.py    # 陈旧/上下调比例、季末口径的流动性+股价过滤后可选股票数
$PY logs/scratch/refs_a_20260916/probe_variants.py    # 家族 1 / 变体 1b / 家族 2 三种口径 × 三级过滤的 48 个月末曲线
$PY logs/scratch/refs_a_20260916/probe_families.py    # 目标价与评级变动两条的覆盖（据此判为不可测）、覆盖度腿厚度
$PY logs/scratch/refs_a_20260916/probe_disp.py        # 家族 4 离散度与 FY+1 修正的覆盖
```

普查为 CPU-only 只读读取，单次峰值内存约 1 GB（1.4 M 行 × 10 列加 20 个交易日的日线切片）。执行前后 `free -h` 可用内存 290Gi → 282Gi（总 503Gi），八张 L20 的显存占用由同机 vLLM 服务持有、本次未变化，`/Data2` 由 446G 降至 445G。
