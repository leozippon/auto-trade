# 平台机制重定向参考包（20260912）

本目录是只读 refs：把股票讨论平台（雪球、股吧、同花顺、财联社、短线论坛）里流传的交易逻辑收成能用前一日信息在 08:30 决策、次日 09:30 成交的可证伪机制，不是可提交策略，也不是预期收益。

任务只有一句话：涨停板类机制在日频 08:30 决策下已经整体证伪（`explore_platform_strategies` 包的九个 playbook 两轮全部落败，臂随后漂成通用因子回退），本轮只测六个用 T-1 可见数据就能执行、且各自带一个机制无关匹配对照的新机制——板块领涨-跟涨溢出、市场宽度情绪周期门控、跌停超卖开盘反转、解禁后压力释放、放量滞涨与地量、大宗折溢价。每个机制单独验证，对照同批比较，落败就淘汰，不用通用因子救场。

按以下顺序读：

1. `exploration-plan.md`：离线筛查、预登记、轮次与验收口径。
2. `playbooks.md`：六个机制的 PIT 改写、入场出场、对照与否证。
3. `pit-field-map.md`：可见时间、单位、本轮快照里没有的数据集。
4. `sources.md`：沿革、已证伪清单与外部先验。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入，每个 `.py` 都受同一套静态检查。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch`（本臂 gpu_count=0，策略容器无 GPU，只跑 CPU；以运行事实 `budgets.strategy_gpu_count` 为准）及其子模块，以及 `output/` 内自己的模块。qlib / joblib / pickle 不得 import；拟合结果用 NumPy 数组或 booster 的 `save_model(context.state_dir + ...)` 持久化。
- 需要拟合的量放在 `fit(context)` 并写入 `context.state_dir`；`generate_orders` 只读它。`models/` 以只读 `context.models_dir` 挂载。
- 沙箱无网络，任何时候都不得抓取站点数据；平台原文只是机制来源，不是数据。
- 每一行必须 `available_at <= context.inference_at`。默认 08:30 决策时，当日日线、竞价、当日 `stk_limit`/`suspend_d`、当日榜单都不存在。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 `data_summary.json` 与单位表，再经 `context.asof_dir` / `context.snapshot_dir` 读数，每次读取都给 `columns=` 与日期窗口。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停、停牌与成交；策略只发订单草图。封死的涨停买不进、跌停卖不出、复牌当日常被拒单，拒单是结果。

## 本轮几何与账户

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段，父本对照也只在这一段上真正是样本外；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。可用的完整 Validation 次数远多于机制数量（上限以本轮事实为准）。

账户是 10 万元真实资金：佣金万一、最低 5 元/笔，过户费 0.1 bp，卖出印花税按成交日切换（切换前万十，切换后万五），方向滑点 5 bp。单只 7,000–10,000 元时最低佣金就是 5–7 bp/边，一次完整换仓的往返成本约 25–35 bp。事件类 cohort 天然短持有、高换手：一个持有 3 日的 cohort 一年要付几十次往返，超额必须显著大于成本才算数。100 股整手：股价 40 元的一手占单只预算四成以上，要么限制股价（T-1 收盘 ≤ 40 元）要么接受权重偏离；科创板 200 股起、北交所 100 股起，通常买不起一手。篮子 8–15 只、等额现金、无信号日持币。

## 「没有边际」的读法

每个候选都按下面的清单读，不看整窗总收益：

- `benchmark.neutralized_excess_return` ≈ 0 或为负：只是规模/β 暴露。
- 候选相对其匹配对照的中性化超额 ≤ 0：机制本身没有信息。
- `vs_parent.beats_parent=false`，或父本对照在本折新季度（`sub_windows` 最后一行）为负而候选也没有转正。
- 四个季度子窗的超额符号不一致（同号少于三个）；事件类机制还要报每个子窗的事件数，事件数 < 30 的子窗不算证据。
- `cost_sensitivity.excess_at_2x_slippage` ≤ 0：毕业按 2 倍滑点压力裁决。
- `pnl_concentration.top5_share_of_gross_gains` 过高：靠几笔事件撑起的收益不算。
- `null_control.excess_percentile` 在 0.5 附近，即与同规模随机组合无法区分；选择之前就能对入围节点用 Fold 专用的 `run_null_control(node_id)` 工具算出来（K=500，约 3.5 分钟一次，每折上限 `max_null_controls_per_fold`，默认 3，只给最终候选用）。`selection_statistics.deflated_sharpe_probability` 接近 0，即 N 次尝试里的最大噪声；每一行验证结果都带它，按 `trials_so_far` 读。

没有候选证明边际时，用 `finish_fold(outcome="no_edge", reason=...)` 明确弃权（有父本则保留父本，首折记为 `baseline_missing`）是正当结果；提名任何通过硬规则的节点都会被冻结。

## 禁止退回通用因子

机制落败后剩下的预算只能用于本包登记的其他机制或它们的门控/持有期变体，不得改成价值/反转/动量/低波/规模打分——除非该打分在 `hypothesis` 里被明确登记为某个机制的「机制无关对照」。前两轮这个臂正是在机制证伪后交出了通用因子回退，本轮不允许。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
