# GitHub 强基线复现与创新参考包（20260912）

本目录是只读 refs：把 Microsoft Qlib 公开工作流里的 **Alpha158 特征 + LightGBM 排序器 + TopkDropout** 收成可在本 ABI 内重写的强基线，再在它之上做四个预登记的创新家族。它不是可提交策略，也不是预期收益；不要 import qlib，不要移植它的数据层、训练器、Recorder 或回测器。

任务只有一句话：先用一折的第一轮把有据可查的基线（158 个日频算子、LightGBM、10 日开盘到开盘的截面 rank 标签、topk + n_drop）在本环境的含成本 Validation 里复现出来，与等权符号打分和父本对照同批比较；再逐轮只改一个可分离的组件——标签口径、特征中性化、跨重训日集成、WorldQuant-101 / 国泰君安-191 算子族——每个改动都必须与基线同批比较并按本包的「没有边际」读法判定。前几轮 explore_github 谱系（低波+股息、内部人−解禁、资金流、五日量价反转、EP+异常换手）的产物没有可信优势，本轮不再重测。

按以下顺序读：

1. `exploration-plan.md`：轮次、拟合纪律、多重检验与验收口径。
2. `alpha158.md`：158 个算子的本地定义、标签、标准化与 TopkDropout 的转述。
3. `families.md`：四个创新家族的预登记检验与失效模式。
4. `sources.md`：上游仓库与文件路径、文献、已证伪清单。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入，每个 `.py` 都受同一套静态检查。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch`（策略容器无 GPU，只跑 CPU）及其子模块，以及 `output/` 内自己的模块。qlib / joblib / pickle 不得 import；模型用 booster 的 `save_model(context.state_dir + "/model.txt")` 与 `lgb.Booster(model_file=context.state_dir + "/model.txt")` 持久化，数组用 `np.save`/`np.savez`，`torch.save` 被静态拒绝。
- 拟合全部放在 `fit(context)`（一次 `fit` 有 `budgets.strategy_fit_timeout_seconds` 的独立预算，按模块级 `REFIT_PERIOD` 重训），结果写入 `context.state_dir`；`generate_orders` 只读它，且受单次决策上限 `budgets.strategy_inference_timeout_seconds` 约束。`models/` 以只读 `context.models_dir` 挂载。
- 沙箱无网络，不要 `pip install`，不要抓 GitHub。
- 每一行必须 `available_at <= context.inference_at`；默认 08:30 只用 T-1 及更早日线。标签只用推断时已经实现的收益。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 `data_summary.json` 与单位表，再经 `context.asof_dir` / `context.snapshot_dir` 读数，每次读取都给 `columns=` 与日期窗口。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停与成交；策略只发订单草图，不要在策略里再写一套回测器。

## 本轮几何与账户

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段，父本对照也只在这一段上真正是样本外；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。首折的第一次 `fit` 大约能看到两年日线，之后随回放推进而增长。

账户是 10 万元真实资金：佣金万一、最低 5 元/笔，过户费 0.1 bp，卖出印花税按成交日切换（切换前万十，切换后万五），方向滑点 5 bp。单只 7,000–10,000 元时最低佣金就是 5–7 bp/边，一次完整换仓的往返成本约 25–35 bp；Qlib 示例的 `topk=50, n_drop=5` 每日换仓在这个账户上既买不起 50 只也付不起换手，本包的起步声明是 `topk=12, n_drop=2`、每 5 个交易日再平衡一次。100 股整手：股价 40 元的一手占单只预算四成以上，要么限制股价（T-1 收盘 ≤ 40 元）要么接受权重偏离；科创板 200 股起、北交所 100 股起，通常买不起一手。

## 「没有边际」的读法

每个候选都按下面的清单读，不看整窗总收益：

- `benchmark.neutralized_excess_return` ≈ 0 或为负：只是规模/β 暴露。排序器最容易学到的就是小市值。
- 相对基线（第一轮的 LightGBM 复现节点）的中性化超额 ≤ 0：改动没有信息。
- `vs_parent.beats_parent=false`，或父本对照在本折新季度（`sub_windows` 最后一行）为负而候选也没有转正。
- 四个季度子窗的超额符号不一致（同号少于三个）。
- `cost_sensitivity.excess_at_2x_slippage` ≤ 0：毕业按 2 倍滑点压力裁决。
- `pnl_concentration.top5_share_of_gross_gains` 过高。
- `null_control.excess_percentile` 在 0.5 附近，即与同规模随机组合无法区分；选择之前就能对入围节点用 Fold 专用的 `run_null_control(node_id)` 工具算出来（K=500，约 3.5 分钟一次，每折上限 `max_null_controls_per_fold`，默认 3，只给最终候选用）。`selection_statistics.deflated_sharpe_probability` 接近 0，即 N 次尝试里的最大噪声；每一行验证结果都带它，按 `trials_so_far` 读——超参网格里的每个点都是一次试验。

没有候选证明边际时，用 `finish_fold(outcome="no_edge", reason=...)` 明确弃权（有父本则保留父本，首折记为 `baseline_missing`）是正当结果；提名任何通过硬规则的节点都会被冻结。

## 禁止退回通用因子

本包的候选形态只有两种：Alpha158 基线及其预登记变体，以及 `families.md` 的算子族。手写的价值/反转/动量/低波打分不得作为候选，除非在 `hypothesis` 里被登记为「等权符号打分」这一机制无关对照。前几轮 explore_github 就是在源仓库思路证伪后回到了小市值与资金流打分，本轮不允许。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
