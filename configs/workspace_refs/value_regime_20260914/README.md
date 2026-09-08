# 估值与派现的慢因子探索参考包（20260914）

本目录是只读 refs：只提供可证伪假说、家族公式和 PIT 检查，不是可提交策略，也不是预期收益。

任务只有一句话：在剔除最小 30% 市值后的全 A 上，检验 EP、BP、股息率与质量这四条慢速、高容量的估值/派现腿（持有 60 个交易日、30 只等权），把站得住的部分收敛成一个带收缩的复合分数，再用 IC/IM 当季年化基差读出的杠杆与对冲需求状态，把同一本账在小盘价值腿与大盘价值腿之间做风格倾斜——不是清仓开关。四个季度子窗一致性是主要读数，整窗汇总不作论据。

按以下顺序读：

1. `exploration-plan.md`：本折的轮次、普查门槛、多重检验与验收口径。
2. `families.md`：六个家族的机制、精确字段、预登记检验与失效模式。
3. `pit-field-map.md`：每个字段的可见时间、单位与陷阱。
4. `sources.md`：文献与业界复盘的交叉核对，以及本轮不再重测的清单。

## 为什么是这三件事

价值在 A 股是少数被反复复制出来的截面异象。Liu–Stambaugh–Yuan 的 CH-3 直接用 EP 作价值腿，并剔除最小 30% 市值以剥离壳价值；Hou–Qiao–Zhang 与 Jacobs–Müller 的复制研究都显示价值与交易摩擦类在 A 股存活，而盈利/投资类多数不显著。它是低频补偿与错误定价的混合物，信息来自估值水平本身，不是近期价格。

派现是 2022–2024 的主导风格，买方是险资、国资与红利指数化产品这类对久期不敏感的长钱。它与 EP 高度同向（本包实测截面 Spearman 0.681），必须当成同一大类里的两种度量，而不是两个独立信号。

杠杆与对冲需求的价格写在股指期货基差上。IC/IM 的当季年化贴水是中性策略、雪球与 DMA 的对冲成本，也是小微盘拥挤度的外生读数；2024-02 的微盘流动性危机在基差上比在任何权益价量指标上都更早、幅度更大。本包把它做成风格切换而不是 on/off 闸门：闸门家族在前几轮里反复失败，根因是季度 walk-forward 下有效观测太少，一旦门关上就没有观测了。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入（如 `from lib.value import score`），每个 `.py` 都受同一套静态检查。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch`（本臂 `budgets.strategy_gpu_count=0`，只跑 CPU）及其子模块，以及 `output/` 内自己的模块。qlib / joblib / pickle 不得 import；参数用 `np.save`/`np.savez`、booster 的 `save_model(context.state_dir + ...)` 或 `torch.save(obj, context.state_dir + ...)` 持久化；静态检查只拒绝绝对路径字面量与只读根写入，以 `output/README.md` 的合同为准。
- 需要拟合的量（复合权重、状态阈值所用的季节均值与标准差）放在 `fit(context)` 并写入 `context.state_dir`，`REFIT_PERIOD="quarter"`；`generate_orders` 只读它。`models/` 以只读 `context.models_dir` 挂载。
- 沙箱无网络，不要抓网页。
- 每一行必须 `available_at <= context.inference_at`。估值列随 `daily_basic` 在 T-1 18:00 可见；财务与派现按各自 `ann_date`（派现优先 `imp_ann_date`）18:00 可见；`fut_*`/`opt_*` 按交易日收盘后盖章，08:30 只能用 T-1 及更早。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 `data_summary.json` 与单位表，再经 `context.asof_dir` / `context.snapshot_dir` 读数，每次读取都给 `columns=` 与日期窗口。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停、停牌、除权现金红利与成交；策略只发订单草图。

## 本轮几何与账户

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段，父本对照也只在这一段上真正是样本外；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。

24 个月是硬上限，对本包有两处直接后果：状态变量的基准窗最长只能取 250 个交易日（快照 macro 窗约 480–490 个交易日，回放起点处更长的窗会被截断）；60 日持有的标签在输入窗里只能拿到约 8 段不重叠的前瞻收益，复合权重必须靠收缩而不是靠拟合。

账户是 10 万元真实资金：佣金万一、最低 5 元/笔，过户费 0.1 bp，卖出印花税按成交日切换（切换前万十，切换后万五），方向滑点 5 bp。30 只篮子意味着单只约 3,300 元，最低佣金就是 15 bp/边，一次完整换仓的往返成本约 35–45 bp。默认节奏是持有 60 个交易日、每 15 个交易日换掉分数最差的 1/4，年单边换手约 4 倍账面，成本年拖累约 1.4–1.8%——这就是本包必须挣到的门槛，写进每个 `hypothesis`。

100 股整手：3,333 元的单只预算在股价 33.3 元以上连一手都买不起，实际权重是 `floor(3333/(100*P))*100*P`。价格上限对本包几乎不花代价——2024-06-28 剔除科创板/北交所与最小 30% 市值后的 3,335 只里，收盘价中位数 10.22 元，≤25 元占 83.3%，而 EP 前 200 名里 189 只、股息率前 200 名里 182 只都在 25 元以下（便宜的票本来就便宜）。因此默认宇宙加一条 T-1 收盘价 ≤ 25 元，并逐决策汇报被这条剔除的候选数与欠配现金比例；若某个家族的头部被它砍掉超过 40%，改成 20 只篮子（单只 5,000 元）并写进 `hypothesis`，不要默默接受权重偏离。科创板 200 股起、北交所 100 股起，直接剔除并说明。

默认 `inference_time=08:30`：当日日线、`daily_basic`、复权因子都不可见；成交只有决策日当天 09:30 开盘价与 15:00 收盘价两个时点（相对 T-1 数据日是次日；`execute_at` ≥ 决策时刻即可）。

## 「没有边际」的读法

每个候选都按下面的清单读，不看整窗总收益：

- `benchmark.neutralized_excess_return` ≈ 0 或为负。本包的规模风险与前几轮方向相反：实测 EP 与 `log(circ_mv)` 的截面 Spearman 为 +0.248、`dv_ttm` 为 +0.149，所以赢的可能是「大盘低波（银行）」而不是小市值。`style_analysis.json` 的规模倾斜两个方向都要看。
- `vs_parent.beats_parent=false`，或父本对照在本折新季度（`sub_windows` 最后一行）为负而候选也没有转正。
- 四个季度子窗的超额符号不一致（同号少于三个）。风格因子最容易整窗好看而只靠一个季度——12 个月验证窗完全可能整段落在一个风格里。
- `cost_sensitivity.excess_at_2x_slippage` ≤ 0：毕业按 2 倍滑点压力裁决。
- `pnl_concentration.top5_share_of_gross_gains` 或单票占比过高。
- `null_control.excess_percentile` 在 0.5 附近，即与同规模随机组合无法区分；选择之前就能对入围节点用 Fold 专用的 `run_null_control(node_id)` 工具算出来（K=500，约 3.5 分钟一次，每折上限 `max_null_controls_per_fold`，默认 3，只给最终候选用）。`selection_statistics.deflated_sharpe_probability` 接近 0，即 N 次尝试里的最大噪声；按 `trials_so_far` 读。
- 状态类候选（家族 5、6）另有一条：验证窗触发日少于 20 个交易日、多于 60%，或独立触发段少于 3 段时，判「不可测」，不作为候选结果汇报——不要用一个几乎不触发或几乎恒触发的门去解释收益。

没有候选证明边际时，用 `finish_fold(outcome="no_edge", reason=...)` 明确弃权（有父本则保留父本，首折记为 `baseline_missing`）是正当结果；提名任何通过硬规则的节点都会被冻结，把没有证明边际的节点冻结进链条对后续折没有价值。

## 禁止退回通用快因子

反转、动量、低波、换手、成长等通用截面打分不得作为本轮候选，除非在 `batch_validate` 的 `hypothesis` 里明确写成某个家族的「机制无关对照」（例如规模中性 20 日动量作为 EP 的对照）。前几轮 explore_platform 与 explore_github 就是这样漂成了通用因子回退，本轮不允许。同样禁止把基差做成清仓闸门：本包只做满仓内部的风格倾斜。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
