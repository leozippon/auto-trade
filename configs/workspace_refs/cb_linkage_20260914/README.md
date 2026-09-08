# 可转债→正股联动参考包（20260914）

本目录是只读 refs：只提供可证伪假说、家族公式和 PIT 检查，不是可提交策略，也不是预期收益。

任务只有一句话：在「T-1 有存续可转债的正股」这个约 300–500 只的股票池上，检验五个以转债市场为信息源、能用 T-1 可见的 `cb_daily`/`cb_basic`/`cb_call` 重算的正股多头假说家族——转债残差动量、转股溢价率状态、强赎条款前后的正股行为、转债相对正股的成交额异动、纯债溢价率的债底缓冲——每个家族带同池等权对照与匹配的无转债对照，只做正股多头、不交易转债，并按本包的「没有边际」读法如实淘汰。前几批六臂全部在同一只证券的价量、事件、文本上做变换，这是第一条用同一发行人的另一只证券做信息源的臂；`cb_*` 三个数据集此前 refs=0、PRIOR=0、策略代码=0。

按以下顺序读：

1. `exploration-plan.md`：第 0 轮离线普查、预登记轮次、家族切换规则与验收口径。
2. `families.md`：五个家族与一个对照家族的机制、精确公式、股票池、持有、对照与否证条件。
3. `pit-field-map.md`：每个字段的域、单位、可见时间、覆盖与陷阱。
4. `sources.md`：文献与数据接口；对方的收益数字不作本折预期。

## 机制

- 两个市场、两套交易制度。转债 T+0；2022-08-01 起上市首日 +57.3%/−43.3%、此后每日 ±20%（此前无涨跌幅限制、只有临时停牌），持有人以机构与专业投资者为主（2022-06 起个人开通转债交易须满足两年经验与 10 万元资产的适当性要求）。正股 T+1，主板 ±10%、创业板/科创板 ±20%。同一家公司的信息在两个市场定价，约束更少、参与者更专业的市场先反映，文献直接支持转债交易预测正股收益、价格发现发生在转债市场（见 `sources.md`）。正股涨停被截断的那部分信息，正好在转债价格里可见。
- 发行人的促转股动机。转股消灭债务，强赎条款（通常连续 30 个交易日中 15 日收盘不低于当期转股价的 130%）附近正股价格有内生性；下修条款让发行人在正股远低于转股价时也有动作空间。这是 A 股特有的公司金融机制，与任何权益内部因子无关。
- 转债价格的两个锚。转股价值（`cb_value`）与纯债价值（`bond_value`）分别是转债的股性锚与债性锚，`cb_over_rate` 与 `bond_over_rate` 给出转债相对两个锚的位置，是转债市场对正股期权价值与发行人信用的日频定价；转债成交额相对正股成交额的异动是知情交易的候选痕迹。
- 规模与容量。2022–2024 年存续转债 400–560 只（本地核对：季末在市 474/558/508 只，2025 年末 387 只），正股集中在中小盘制造业：2024-06 基准日 519 只发行人，流通市值中位 39 亿元、68% 日成交额 ≥ 3,000 万元、95% 收盘价 ≤ 40 元、无北交所；转债日成交额中位数是正股的 0.4 倍，一成以上的券超过正股 3 倍。10 万元账户不受容量约束。

## 本臂必须与不得

- 只做正股多头，转债只是信息源；不交易转债、不做转债套利、不做任何多空。
- 一切转债字段只用 T-1 及更早：`cb_daily` 与 `cb_call` 的行级 `available_at` 是 `trade_date`/`ann_date` 当日 23:59:59，08:30 决策时 T-1 行可见、当日行不可见，与 `daily` 同节奏。
- 每个家族带两个对照：同池同过滤的等权篮子（证明信号本身有信息），以及按同日、同规模十分位、同申万一级行业、成交额最接近抽取的无存续转债匹配篮子（证明边际不是「转债发行人」这个池子本身）。池子有结构性偏差：正股涨到触发强赎的发行人会因转债退市而离开池子，留下的是没涨够的——等权池篮子天然带这个倾斜，匹配对照必须每次调仓重新抽。
- 存续与转股价只能从 `cb_daily` 重算：`cb_basic.conv_price`/`remain_size`/`newest_rating`/`delist_date` 是当前状态、已被快照剔除；历史转股价 `conv_price_t = 100 × 正股close_t / cb_value_t`，存续按 T-1 有无 `cb_daily` 行判断。
- 通用截面因子不得作候选，只能作 `families.md` 里命名的对照（第 6 家族与各家族的机制无关对照）。

## 硬合同

- 正式策略写在 `output/` 包内：入口固定为 `output/main.py` 的 `generate_orders(context)`，返回严格 JSON 订单数组；辅助模块可放在 `output/` 下并用绝对导入（如 `from lib.features import x`），每个 `.py` 都受同一套静态检查。
- 正式 import 只允许：纯计算标准库（`__future__`、`collections`、`dataclasses`、`datetime`、`decimal`、`functools`、`itertools`、`math`、`statistics`、`typing`）、`numpy`、`pandas`、`scipy`、`sklearn`、`lightgbm`、`xgboost`、`statsmodels`、`torch`（本臂 gpu_count=0，策略容器无 GPU，只跑 CPU；以运行事实 `budgets.strategy_gpu_count` 为准）及其子模块，以及 `output/` 内自己的模块。qlib / joblib / pickle 不得 import；拟合结果用 NumPy 数组、booster 的 `save_model(context.state_dir + ...)` 或 `torch.save(obj, context.state_dir + ...)` 持久化；静态检查只拒绝绝对路径字面量与只读根写入，以 `output/README.md` 的合同为准。
- 需要拟合的量放在 `fit(context)` 并写入 `context.state_dir`（一次 `fit` 有 `budgets.strategy_fit_timeout_seconds` 的独立预算，按模块级 `REFIT_PERIOD` 重训）；`generate_orders` 只读它，且受单次决策上限 `budgets.strategy_inference_timeout_seconds` 约束。`models/` 以只读 `context.models_dir` 挂载。
- 沙箱无网络，不要抓网页。
- 每一行必须 `available_at <= context.inference_at`；转债与公告用 `available_at`，不用 `trade_date`/`ann_date`/`call_date` 偷看。
- 不要写死 `/mnt/agent/workspace`。先核对本轮 `data_summary.json` 里 `macro` 的 `datasets` 是否含 `cb_daily`/`cb_basic`/`cb_call` 与单位表，再经 `context.asof_dir` / `context.snapshot_dir` 读数，每次读取都给 `columns=`、`dataset` 过滤与日期窗口。
- 不要把 refs 拷进 `output`。
- Broker 负责 T+1、费用、涨跌停、停牌与成交；策略只发订单草图。

## 本轮几何与账户

开发窗口按季度步进切成常规 Fold：每折的验证区间是截至本折季度的连续四个季度（滚动四季窗），相邻两折只相差一个季度，因此只有最后一个季度是父本没见过的新数据段，父本对照也只在这一段上真正是样本外；没有 Test 阶段，相邻两折之间跑一次元学习；本折输入窗是验证区间之前约 24 个月，精确窗口以运行事实为准。`cb_daily` 自 2018-01-02 起、`cb_call` 自 2019 年起基本完整，开发窗口 2022Q1–2025Q4 的输入窗全部覆盖，没有日历缺口；可用的完整 Validation 次数远多于家族数量（上限以本轮事实为准）。

账户是 10 万元真实资金：佣金万一、最低 5 元/笔，过户费 0.1 bp，卖出印花税按成交日切换（切换前万十，切换后万五），方向滑点 5 bp。单只 7,000–10,000 元时最低佣金就是 5–7 bp/边，一次完整换仓的往返成本约 25–35 bp；5 日持有的家族一年要付几十次往返，超额必须显著大于成本才算数。100 股整手：股价 40 元的一手占单只预算四成以上，本包把 T-1 收盘 ≤ 40 元写成声明过滤；科创板 200 股起、北交所 100 股起，通常买不起一手，本池科创板占 8%、无北交所，默认剔除科创板并说明。默认 `inference_time=08:30`：当日日线、`daily_basic`、`adj_factor`、当日转债行情都不可见，只能用 T-1 及更早；成交只有决策日当天 09:30 开盘价与 15:00 收盘价两个时点（相对 T-1 数据日是次日；`execute_at` ≥ 决策时刻即可）。

## 「没有边际」的读法

每个候选都按下面的清单读，不看整窗总收益：

- `benchmark.neutralized_excess_return` ≈ 0 或为负：只是规模/β 暴露。`style_analysis.json` 的规模倾斜同样要看——本池天然偏小盘。
- 候选相对同池等权对照的中性化超额 ≤ 0：信号没有信息；等权池相对匹配无转债对照的超额才是「池子效应」，两者分开报。
- `vs_parent.beats_parent=false`，或父本对照在本折新季度（`sub_windows` 最后一行）为负而候选也没有转正。
- 四个季度子窗的超额符号不一致（同号少于三个）；事件类家族（强赎）还要报每个子窗的事件数，事件数 < 30 的子窗不算证据。
- `cost_sensitivity.excess_at_2x_slippage` ≤ 0：毕业按 2 倍滑点压力裁决，只在假设滑点下才有的边际不算。
- `pnl_concentration.top5_share_of_gross_gains` 或 `top_name_share_of_gross_gains` 过高：靠一两只票撑起的收益不算。
- 拒单率：转债领先的信号最强的日子往往正股次日开盘涨停，买单被拒；意图买单拒单率 > 30% 判「不可执行」而不是「有边际」。
- `null_control.excess_percentile` 在 0.5 附近，即与同规模随机组合无法区分；选择之前就能对入围节点用 Fold 专用的 `run_null_control(node_id)` 工具算出来（K=500，约 3.5 分钟一次，每折上限 `max_null_controls_per_fold`，默认 3，只给最终候选用）。`selection_statistics.deflated_sharpe_probability` 接近 0，即 N 次尝试里的最大噪声；每一行验证结果都带它，按 `trials_so_far` 读。

没有候选证明边际时，用 `finish_fold(outcome="no_edge", reason=...)` 明确弃权（有父本则保留父本，首折记为 `baseline_missing`）是正当结果；提名任何通过硬规则的节点都会被冻结，把没有证明边际的节点冻结进链条对后续折没有价值。

## 禁止退回通用因子

价值、反转、动量、低波、成长、规模等通用截面打分不得作为本轮候选，也不得在转债家族落败后用剩余预算补上；它们只能以 `families.md` 命名的形式出现在 `batch_validate` 的 `hypothesis` 里，并明确写成某个转债家族的「机制无关对照」（例如第 6 家族的正股 5 日反转作为残差动量的对照，20 日动量作为溢价率与强赎家族的对照，BP 作为债底家族的对照）。前几轮 explore_platform 就是这样漂成了通用因子回退，本轮不允许；对照节点不得被 `finish_fold` 提名。

## 运行教训

跨实验的运行教训不再重复写在参考包里：默认挂载的运行记忆在工作区 `memory/<来源>/` 下只读可读，索引见 `inputs/skills_index.json` 的 `operating_memory` 一节（`origin=curated` 为策展经验，`origin=graduated` 为通过最终评估的历史实验留下的 skills）。
