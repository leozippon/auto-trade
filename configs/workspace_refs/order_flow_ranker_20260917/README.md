# 日内签名订单流排序器参考包（20260917）

本目录是只读 refs：把 **tick-rule 签名成交量失衡**收成一个特征面，交给一个只做多的截面排序器，并按预登记的归因对照判定它是否真的挣到了供应商日频资金流之外的边际。它不是可提交策略，也不是预期收益。

信号本身来自 1 分钟 Bar，但**本臂不挂分钟域**：派生事件数据集 `intraday_flow` 已经把每个分钟分区就地约简成每个股票日一行（`ofi_1d`、`ofi_amt_1d` 加两个诊断列），策略只读这张日频表。阈值与滚动窗口留在策略里。

任务只有一句话：用 `ofi` 家族（逐分钟收盘价的涨跌方向给该分钟成交量定号、按日汇总，再取 21 个交易日均值）做 08:30 决策的截面信号，并且必须同批比过 **`moneyflow` 面**——同一个篮子、同一个标签、同一个打分器，特征换成供应商自己的日频资金流分类。**比不过这个对照，本机制立即关闭**，因为那说明分钟级 tick 规则只是把供应商已经算过的东西重算了一遍，而那几列早已在 `explore_github` 的冻结产物里。

按以下顺序读：

1. `exploration-plan.md`：不占预算的普查、首折四路批次、特征构建与拟合的成本预算、判定与臂级终止规则。
2. `families.md`：两个特征面的逐列定义、硬门、九个已关闭的兄弟构造、阈值与「什么会杀死它」。
3. `pit-field-map.md`：`intraday_flow` 的列、盖章与覆盖终点、逐表可见时点、单位、以及数据层已经处理掉与仍须策略处理的两组事项。
4. `sources.md`：本地实测的数据事实与复现命令。
5. `starter/`：一份合同内的最小可运行包骨架（`main.py` + `lib/`），供起步与改写，**不是**候选本身。它通过 `validate_strategy_package` 的静态检查，并在沙箱镜像里用一份真实的 202 个交易日 as-of 切片（≤ 2026-03-31）跑通了「事件表读取 → 阈值过滤 → 21 日窗口 → 中性化打分 → 出单」全链路与四个分支；那是手写 harness，**不能替代 `smoke_backtest`**，正式回放的第一步仍然是 `smoke_backtest`。`main.py` 的 `CANDIDATE` 常量在四条腿之间切换，四者除这一行外必须逐字相同。

## 为什么这是一个新机制家族

一天的 OHLCV 与任意一条日内订单流路径都相容：同样的开高低收和成交量，既可以由整天的持续买压产生，也可以由整天的持续卖压产生。在 T+1、有涨跌停、散户占比高的市场里，有信息的参与者无法一笔打完，订单被拆到几百个分钟里执行，留下一条能跨日持续的签名足迹。本包的主张是：这条足迹在剔除动量、换手、波动、彩票需求、短期反转**以及供应商自己的日频净流入口径**之后，仍然对 5–20 日的截面有解释力。

按 `docs/agent-design.md` 的家族定义（「基于新特征集的学习排序器」是不同家族），这是新家族：分钟 Bar 是本仓库唯一还没有任何一条 arm 读过的大数据面，`ofi_1d` 没有一列能从 Alpha158、WQ101、GTJA191、`factor_cs` 与 `ml_ranker` 已经穷尽的日频面构造出来。

反过来也要诚实。同一次研究里对同一个分钟面测了十个构造，**九个在控制变量下归零**（分段收益、尾盘 OFI、量能集中度、竞价量占比、分钟 Amihud、VWAP 偏离……逐条见 `families.md`，它们都需要分钟域、本臂读不到），只有 `ofi` 活下来；而且它活下来的方式是控制后**变强**（t 从 +0.19 升到 +3.28），这正是非价格信号的指纹。它最弱的地方也写死在这里：**只做多的那条腿很薄**——顶档五分位超额 +0.086%/5 日（t 1.98）、+0.269%/20 日（t 1.52），而供应商日频口径的同一条腿是它的三倍（+0.366%/20 日，t 3.01）。所以硬门不是形式，它是这个包的核心。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块放在 `output/` 下并用绝对导入，每个 `.py` 受同一套静态检查，文件数与总字节数有上限（`max_strategy_files`/`max_strategy_bytes`，以运行事实为准）。
- 正式 import 只允许：纯计算标准库、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch` 及其子模块，以及 `output/` 内自己的模块。qlib / joblib / pickle 不得 import。本臂 `gpu_count=0`，策略容器没有 GPU，全部在 CPU 上跑。
- 拟合全部放在 `fit(context)`（模块级 `REFIT_PERIOD="quarter"`），结果写入 `context.state_dir`；`generate_orders` 只读它。一次 `fit` 的墙钟上限是 `budgets.strategy_fit_timeout_seconds`（本轮 3600 秒），**`batch_validate` 并发池存续期间按本批实际路数放大**（三路即 10,800 秒），规则以运行事实 `budgets.batch_validate_fit_timeout_note` 为准；串行回放仍是基准值，所以预算要按 3600 秒设计。
- 持久化只有三种形式：`np.save`/`np.savez`/`np.savez_compressed`、`DataFrame.to_parquet`、booster 的 `save_model`，全部写到 `context.state_dir`。booster 的加载**必须**写成 `lgb.Booster(model_file=path)` 或 `lgb.Booster(model_str=s)`；位置传参在 LightGBM 4.7 会把路径当成 `params`。
- 沙箱无网络，不要 `pip install`。不要写死 `/mnt/agent/workspace`。先核对本轮 `data_summary.json` 与 `unit_reference.json`，再经 `context.asof_dir` 读数，每次读取都给 `columns=`、`filters=` 与日期窗口；as-of 域读失败**不得**回退 `snapshot_dir`。
- 两个特征面都在 `context.asof_dir + "/events"` 这个 **parts 目录**里，按 `dataset` 区分（`intraday_flow` / `moneyflow`）。事件域是列并集，同名列在不同数据集里含义不同，**必须先按 `dataset` 过滤再谈字段与单位**。
- 每一行输入必须 `available_at <= context.inference_at`。两张表都是交易日盖章（`intraday_flow` 17:30、`moneyflow` 19:00），08:30 只到 **T-1**。判可见只看 `available_at`，不看 `trade_date`。
- **本臂不挂分钟域**：`include_intraday` 为假，`intraday_1min` 是零行文件，`historical_minutes_available` 为假，因此 09:30 与 15:00 之外的 `execute_at` 一律拒单。执行时点只用这两个。
- 标签只用推断时已经完全实现的收益；训练与验证之间必须留出持有期长度的禁运。
- 不要把 refs 拷进 `output`。Broker 负责 T+1、费用、涨跌停与成交；策略只发订单草图。

## 本轮几何与账户

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。

**首折没有父产物**，因此适用基线锚点规则（`docs/pipeline-design.md` §2.2）：只要本会话有一个过 `AcceptanceRules` 硬门的完整 Validation，`finish_fold(outcome="no_edge")` 就会以 `baseline_anchor_required` 被拒，必须提名其一，账本记 `baseline_anchor=true`。本包因此把首折批次设计成**一定产生至少一个可锚定候选**，并把「锚点是谁」与「家族是否成立」分开判：**过硬门的候选里最好的那个做锚点**，即便它是对照腿；这时折内记录必须写明锚点不是本臂的机制，且下一折若不重新登记就终止。锚点是待替换的弱基线，不是已证明的边际。

账户是 10 万元真实资金：佣金万一、最低 5 元/笔，过户费 0.1 bp，卖出印花税按成交日切换（2023-08-28 前万十，之后万五），方向滑点 5 bp。本包的起步声明是 **topk=15、每 20 个交易日再平衡、单次最多换掉半个篮子**：15 只等权、95% 仓位时单只预算约 6,300 元，最低佣金 5 元即 7.9 bp/边，一次半篮换手往返约 25–35 bp，一年约 12 次半篮轮换 ≈ 600% 年换手、约 **1.8–2.2% 的年成本拖累**。100 股整手：声明 T-1 收盘价 ≤ 30 元，使一手 ≤ 3,000 元、不超过单只预算的一半；科创板 200 股起、北交所 100 股起且流动性不足，整体剔除。

**成本门写死在这里：探针的五分位多头腿折成年化约 +3.3%，低于 15 只集中篮子必须达到的水平。一个 15 只的篮子要把它集中到 ≥ 5%/年的毛超额，否则结论是「机制真实但在本账户不可交易」——那是一个正当的关闭理由，不是失败。**

## 「没有边际」的读法

每个候选都按下面的清单读，不看整窗总收益：

- **硬门未过**：判据逐字见 `families.md`「硬门」段，那是本包关闭判据的唯一权威表述，本文不复述。它是本臂的第一判据，命中即整个机制关闭，不是换个变体再试。哪些读数关闭机制、哪些只让候选不可提名，同样以 `families.md`「什么会杀死它」为准。
- `benchmark.neutralized_excess_return` ≈ 0 或为负：只是规模/β 暴露。
- `vs_parent.beats_parent=false`，或父本对照在本折新季度（`sub_windows` 最后一行）为负而候选也没有转正。
- 四个季度子窗的超额同号少于三个，或超额只来自单个季度。
- `cost_sensitivity.excess_at_2x_slippage` ≤ 0；`pnl_concentration.top5_share_of_gross_gains` 过高。
- `null_control.excess_percentile` 在 0.5 附近（`run_null_control(node_id)`，K=500，每折上限 `max_null_controls_per_fold`，只给决赛候选用）；`selection_statistics.deflated_sharpe_probability` 接近 0，按 `trials_so_far` 读。
- **分钟面没被用上**：`o2` 里 `ofi` 块的特征重要性合计 < 5%（等于没用上）。
- **过滤伪装成信号**：封板过滤（`sealed_limit`，即零成交 Bar 占比 ≥ 50%）剔掉的股票日超过 40%，那 `ofi` 就是一个涨跌停日的伪影，不是订单流。实测基线 0.3%–3.4%/日（2026-03-31 为 0.04%），逐折汇报。

没有候选证明边际时，用 `finish_fold(outcome="no_edge", reason=...)` 明确弃权是正当结果——除非首折的基线锚点规则要求提名（见上）。

## 禁止退回通用因子

本包的候选形态只有两种：本包登记的 `ofi` 面排序器及其预登记变体（窗口、口径、打分器、执行时点），以及登记为对照的 `moneyflow` 面与日频价量面。手写的价值/反转/动量/低波/彩票打分不得作为候选，只能以对照身份出现在 `hypothesis` 里，且**对照节点不得被 `finish_fold` 提名**（首折基线锚点是唯一例外，且必须在折内记录里写明）。`families.md` 列出的九个已证伪兄弟构造只能作对照，不得作候选。

## 运行教训

跨实验的运行教训不重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节。本包只指两条：booster 跨容器持久化与静默退化的指纹诊断（`fit-state-pitfalls`），以及「可见性语义必须单一事实源」（`panel-semantics-parity`）——后者在本包里表现为一条硬要求：**训练面板与决策截面必须由同一个函数构建**（`starter/lib/flow.py` 的 `build_face`）。
