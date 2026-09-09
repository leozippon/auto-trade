# 事件/资金特征族排序器参考包（20260916）

本目录是只读 refs：把**披露与资金事件面**收成一个特征集，交给一个在 `fit(context)` 里当场训练的 walk-forward 截面排序器，并按预登记的归因对照判定它是否真的挣到了日频价量之外的边际。它不是可提交策略，也不是预期收益。

任务只有一句话：用一个特征集——卖方一致预期（`report_rc`）、两融余额（`margin_detail`）、大宗折溢价与席位（`block_trade`）、股东人数/增减持/十大流通股东（`stk_holdernumber`/`stk_holdertrade`/`top10_floatholders`）、筹码成本分布（`cyq_perf`）、解禁供给（`share_float_complete`）——训 LightGBM 排序器，并且必须同批比过两个对照：**同学习器同标签只用日频价量特征**（证明边际来自事件面而不是模型），以及**同特征不学习的等权 rank 合成**（证明学习器本身有价值）。比不过第一个对照，本机制就关闭。

按以下顺序读：

1. `exploration-plan.md`：普查、轮次、拟合预算、首折判定规则与臂级终止规则。
2. `families.md`：五个特征族的逐列定义与窗口、学习器规格、两个对照、消融、阈值与「什么会杀死它」。
3. `pit-field-map.md`：逐表可见时点、单位、去重键与常见失败。
4. `sources.md`：本地实测的数据事实、上游文献与已证伪清单。
5. `starter/`：一份合同内的最小可运行包骨架（`main.py` + `lib/`），供起步与改写，**不是**候选本身。它通过了 `validate_strategy_package` 的静态检查，并在沙箱镜像里用一份合成 as-of 切片跑通了「特征构建 → 带禁运的切分 → 拟合 → 落盘 → 决策出单」全链路与三个 `CONTROL` 分支；但那是手写 harness，**不能替代 `smoke_backtest`**，正式回放的第一步仍然是 `smoke_backtest`。`main.py` 的 `CONTROL` 常量在主候选与两个对照之间切换，三者除这一行外必须逐字相同。

## 为什么这是一个新机制家族

`docs/agent-design.md` 的守则把机制家族定义为收益来源的经济解释，并明写「**基于新特征集的学习排序器**是不同家族，同一信号换估计器、持有期、篮子大小或中性化方式只是变体」。本仓库已有的两条 ML 谱系是：`explore_github` 的 Alpha158 + vipF2 成长/预告 + mfflow 资金流（172 列，季度重训 LightGBM），以及 `ml_ranker` 的 17 个手写价量特征（LightGBM 后换 MLP）。本包的特征集与它们**没有一列重合**：全部来自事件与披露表，其中 `report_rc` 数值切片、`margin_detail`、`block_trade`、`cyq_perf`、`stk_holdernumber`/`stk_holdertrade`/`top10_floatholders`、`share_float_complete` 从未进入过任何排序器。因此它是新家族，而不是既有排序器的估计器变体。

反过来也要诚实：其中几条腿**作为单因子已经被证伪或判为空**——`cyq_perf` 的 `winner_rate` 被判为动量/波动换皮（控制后 t = −1.66 / −1.46），股东人数集中度在只控规模那一步就是空的，单因子解禁事件族已被 `explore_platform` 关闭。本包允许它们**作为多个特征之一**进入排序器，正是因为家族的主张变了：主张不再是「某一列单独可交易」，而是「这些列的联合信息里有日频价量学不到的东西」。这个主张只有一个检验，就是归因对照，见下。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块放在 `output/` 下并用绝对导入，每个 `.py` 受同一套静态检查，文件数与总字节数有上限（`max_strategy_files`/`max_strategy_bytes`，以运行事实为准）。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch` 及其子模块，以及 `output/` 内自己的模块。qlib / joblib / pickle 不得 import。本臂 `gpu_count=0`，策略容器没有 GPU，全部在 CPU 上跑。
- 拟合全部放在 `fit(context)`（模块级 `REFIT_PERIOD="quarter"`，一次 `fit` 有 `budgets.strategy_fit_timeout_seconds` 的独立预算，本轮为 3600 秒），结果写入 `context.state_dir`；`generate_orders` 只读它，受 `budgets.strategy_inference_timeout_seconds` 约束。`models/` 以只读 `context.models_dir` 挂载。
- 持久化只有三种形式：`np.save`/`np.savez`/`np.savez_compressed`、`DataFrame.to_parquet`、booster 的 `save_model`，全部写到 `context.state_dir`。booster 的加载**必须**写成 `lgb.Booster(model_file=path)` 或 `lgb.Booster(model_str=s)`；位置传参在 LightGBM 4.7 会把路径当成 `params` 并抛出与「文件缺失」逐字相同的 `TypeError`（见 `sources.md` 的运行教训一节）。
- 沙箱无网络，不要 `pip install`。不要写死 `/mnt/agent/workspace`。先核对本轮 `data_summary.json` 与 `unit_reference.json`，再经 `context.asof_dir` / `context.snapshot_dir` 读数，每次读取都给 `columns=`、`dataset` 过滤与日期窗口；as-of 域读失败**不得**回退 `snapshot_dir`。
- 每一行输入必须 `available_at <= context.inference_at`。判可见只看 `available_at`，不看 `trade_date`/`ann_date`/`report_date`——本包六张表里有三张在 08:30 只到 **T-2**，逐表边界见 `pit-field-map.md`。
- 标签只用推断时已经完全实现的收益；训练与验证之间必须留出持有期长度的禁运。
- 不要把 refs 拷进 `output`。Broker 负责 T+1、费用、涨跌停与成交；策略只发订单草图。

## 本轮几何与账户

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段，父本对照也只在这一段上真正是样本外；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。

账户是 10 万元真实资金：佣金万一、最低 5 元/笔，过户费 0.1 bp，卖出印花税按成交日切换（2023-08-28 前万十，之后万五），方向滑点 5 bp。本包的起步声明是 **topk=15、每 20 个交易日再平衡、单次最多换掉半个篮子**：15 只等权、95% 仓位时单只预算约 6,300 元，最低佣金 5 元即 7.9 bp/边，一次半篮换手往返约 25–35 bp，一年约 12 次半篮轮换≈600% 年换手、约 1.8–2.2% 的年成本拖累。100 股整手：声明 T-1 收盘价 ≤ 30 元，使一手 ≤ 3,000 元、不超过单只预算的一半；科创板 200 股起、北交所 100 股起且流动性不足，整体剔除。

## 「没有边际」的读法

每个候选都按下面的清单读，不看整窗总收益：

- **归因对照未被超过**：相对「同学习器、同超参、同标签、只用日频价量特征」的 `benchmark.neutralized_excess_return` ≤ 0，或空对照分位不高于它。这是本臂的第一判据，命中即整个机制关闭，不是换个变体再试。
- **学习器没有价值**：相对「同特征、等权 rank 合成、不训练」的中性化超额 ≤ 0。此时保留更简单的对照版本。
- `benchmark.neutralized_excess_return` ≈ 0 或为负：只是规模/β 暴露。
- `vs_parent.beats_parent=false`，或父本对照在本折新季度（`sub_windows` 最后一行）为负而候选也没有转正。
- 四个季度子窗的超额同号少于三个，或超额只来自单个季度。
- `cost_sensitivity.excess_at_2x_slippage` ≤ 0；`pnl_concentration.top5_share_of_gross_gains` 过高。
- `null_control.excess_percentile` 在 0.5 附近（`run_null_control(node_id)`，K=500，每折上限 `max_null_controls_per_fold`，只给决赛候选用）；`selection_statistics.deflated_sharpe_probability` 接近 0，按 `trials_so_far` 读——网格里的每个点都是一次试验。
- **事件块没被用上**：五个事件族的特征重要性合计 < 25%，或没有任何两个族各自 ≥ 5%。
- **覆盖伪装成信号**：某一族的边际全部来自它的 `has_*` 指示列，或该族在可选池里的覆盖率 < 20% 却被判为唯一功臣。逐族覆盖率实测见 `sources.md`：筹码 95–100%、股东人数 ≥ 99%、两融 46–67%、分析师 36–46%、大宗 9–16%、未来 90 日解禁 3–6%。

没有候选证明边际时，用 `finish_fold(outcome="no_edge", reason=...)` 明确弃权（有父本则保留父本）是正当结果；提名任何通过硬规则的节点都会被冻结。

## 禁止退回通用因子

本包的候选形态只有两种：本包登记的特征集排序器及其预登记变体（特征族消融、标签口径、中性化层次、集成），以及登记为对照的等权 rank 合成。手写的价值/反转/动量/低波/彩票打分不得作为候选，只能以「日频价量归因对照」或「机制无关对照」的身份出现在 `hypothesis` 里，且**对照节点不得被 `finish_fold` 提名**。事件族全部落败后的剩余预算只能用于本包登记的其他族或其窗口/持有期变体，不得改成通用因子回退。

## 运行教训

跨实验的运行教训不重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节。本包只把与「fit 写、决策读」直接相关的两条指到 `sources.md`：booster 跨容器持久化与静默退化的指纹诊断（`fit-state-pitfalls`），以及「可见性语义必须单一事实源」（`panel-semantics-parity`）。
