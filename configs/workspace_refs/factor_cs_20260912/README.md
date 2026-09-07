# 截面异象探索参考包（20260912）

本目录是只读 refs：只提供可证伪假说、家族公式和 PIT 检查，不是可提交策略，也不是预期收益。

任务只有一句话：在未筛选的全 A 股票池上，检验六个有文献依据、能用 T-1 可见数据重算的 A 股特有异象家族，每个家族先做规模与 β 中性，再用 `fit(context)` 里的正则化合成或树模型排序器把站得住的家族合成一个 10–15 只、持有期 ≥ 5 个交易日的多头篮子，并按本包的「没有边际」读法如实淘汰。前几轮（`factor_cs_20260826` 包的价值/反转/成长/趋势线性打分）已经走到头：胜出的都是小市值暴露，规模因子修正后中性化超额转负，本轮不再测它们。

按以下顺序读：

1. `exploration-plan.md`：本折的流程、合成纪律、多重检验与验收口径。
2. `families.md`：六个家族的机制、精确字段、预登记检验与失效模式。
3. `pit-field-map.md`：每个字段的可见时间、单位与陷阱。
4. `sources.md`：文献与仓库交叉核对；对方收益表不作本折预期。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入（如 `from lib.features import x`），每个 `.py` 都受同一套静态检查。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch`（策略容器的 GPU 数量见运行事实 `budgets.strategy_gpu_count`，本臂为 0，即只跑 CPU）及其子模块，以及 `output/` 内自己的模块。qlib / joblib / pickle 不得 import；模型参数用 NumPy 数组（`np.save`/`np.savez`）、booster 的 `save_model(context.state_dir + ...)` 或 `torch.save(obj, context.state_dir + ...)` 持久化；静态检查只拒绝绝对路径字面量与只读根写入。
- 需要拟合的量放在 `fit(context)` 并写入 `context.state_dir`（一次 `fit` 有 `budgets.strategy_fit_timeout_seconds` 的独立预算，按模块级 `REFIT_PERIOD` 重训）；`generate_orders` 只读它，且受单次决策上限 `budgets.strategy_inference_timeout_seconds` 约束。`models/` 以只读 `context.models_dir` 挂载。
- 沙箱无网络，不要抓网页。
- 每一行必须 `available_at <= context.inference_at`；财务与事件用 `available_at`，不用 `end_date`/`trade_date` 偷看。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 `data_summary.json` 与单位表，再经 `context.asof_dir` / `context.snapshot_dir` 读数，每次读取都给 `columns=` 与日期窗口。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停、停牌与成交；策略只发订单草图。

## 本轮几何与账户

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段，父本对照也只在这一段上真正是样本外；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。可用的完整 Validation 次数远多于家族数量（上限以本轮事实为准）。

账户是 10 万元真实资金：佣金万一、最低 5 元/笔，过户费 0.1 bp，卖出印花税按成交日切换（切换前万十，切换后万五），方向滑点 5 bp。单只 7,000–10,000 元时最低佣金就是 5–7 bp/边，一次完整换仓的往返成本约 25–35 bp；每日全换仓一年要付六成以上，因此持有期 ≥ 5 个交易日（或 n_drop 式部分换仓）是硬约束。100 股整手：股价 40 元的一手占单只预算四成以上，要么限制股价（例如 T-1 收盘 ≤ 40 元）要么接受权重偏离；科创板 200 股起、北交所 100 股起，通常买不起一手，直接从候选里剔除并说明。默认 `inference_time=08:30`：当日日线、`daily_basic`、`adj_factor` 都不可见，只能用 T-1 及更早；成交只有次日 09:30 开盘价与 15:00 收盘价两个时点。

## 「没有边际」的读法

每个候选都按下面的清单读，不看整窗总收益：

- `benchmark.neutralized_excess_return` ≈ 0 或为负：只是规模/β 暴露。`style_analysis.json` 的规模倾斜同样要看。
- `vs_parent.beats_parent=false`，或父本对照在本折新季度（`sub_windows` 最后一行）为负而候选也没有转正。
- 四个季度子窗的超额符号不一致（同号少于三个）。
- `cost_sensitivity.excess_at_2x_slippage` ≤ 0：毕业按 2 倍滑点压力裁决，只在假设滑点下才有的边际不算。
- `pnl_concentration.top5_share_of_gross_gains` 或单票占比过高：靠一两只票撑起的收益不算。
- `null_control.excess_percentile` 在 0.5 附近，即与同规模随机组合无法区分；选择之前就能对入围节点用 Fold 专用的 `run_null_control(node_id)` 工具算出来（K=500，约 3.5 分钟一次，每折上限 `max_null_controls_per_fold`，默认 3，只给最终候选用）。`selection_statistics.deflated_sharpe_probability` 接近 0，即 N 次尝试里的最大噪声；每一行验证结果都带它，按 `trials_so_far` 读——本折评估了多少候选就要按多少次试验读 Sharpe。

没有候选证明边际时，用 `finish_fold(outcome="no_edge", reason=...)` 明确弃权（有父本则保留父本，首折记为 `baseline_missing`）是正当结果；提名任何通过硬规则的节点都会被冻结，把没有证明边际的节点冻结进链条对后续折没有价值。

## 禁止退回通用因子

价值、反转、动量、低波、成长、规模等通用截面打分不得作为本轮候选，除非在 `batch_validate` 的 `hypothesis` 里明确写成某个家族的「机制无关对照」（例如规模中性 20 日反转作为隔夜-日内分解的对照）。前一轮 explore_platform 就是这样漂成了通用因子回退，本轮不允许。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
