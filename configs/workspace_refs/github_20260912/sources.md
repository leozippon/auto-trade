# 来源、路径与已证伪清单

只引用公开页面并转述机制。不要 vendor 任何仓库，不要粘贴 yaml/源码全文。沙箱不挂载上游仓库；宿主若在 `external_references/` 下留有 Qlib 或 RD-Agent 的克隆，下面的仓库相对路径同样适用，但 Agent 在沙箱里读不到它们，`alpha158.md` 里的转述就是全部可用材料。

## Qlib（引用，禁止拷贝）

- 项目：<https://github.com/microsoft/qlib>。
- `qlib/contrib/data/handler.py`：`Alpha158`、`Alpha360` 处理器类与默认 `learn_processors`/`infer_processors`。
- `qlib/contrib/data/loader.py`：`Alpha158DL.get_feature_config`（158 列表达式的生成器）、`Alpha360DL`。
- `qlib/data/dataset/processor.py`：`RobustZScoreNorm`、`CSRankNorm`、`CSZScoreNorm`、`DropnaLabel`、`Fillna` 的定义。
- `qlib/contrib/model/gbdt.py`：`LGBModel`（早停、`num_boost_round`）。
- `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`、`workflow_config_lightgbm_Alpha360.yaml`：示例参数、标签 `Ref($close,-2)/Ref($close,-1)-1`、`TopkDropoutStrategy(topk=50, n_drop=5)`、CSI300 市场与费用设置。
- `qlib/contrib/strategy/signal_strategy.py`：`TopkDropoutStrategy` 的换手规则。
- `examples/benchmarks/README.md`：各模型在 CSI300 上的公开表——那是他们的数据、复权、费用与 executor，不是本折预期。
- 文档：[Workflow Management](https://qlib.readthedocs.io/en/latest/component/workflow.html)、[Data Layer](https://qlib.readthedocs.io/en/latest/component/data.html)。本环境没有 `qrun`、`DatasetH`、`Recorder`；正式策略只有 `fit` 与 `generate_orders`。

## RD-Agent（过程先验）

- 项目：<https://github.com/microsoft/RD-Agent>；`rdagent/scenarios/qlib/` 下的因子与模型研发循环（`rdagent/app/qlib_rd_loop/factor.py`、`model.py`，评估在 `rdagent/scenarios/qlib/developer/`）。
- 只借用它的过程形态：先写假说与预期，再实现、回测、写反馈，把「为什么失败」记进下一轮假说。本环境的对应物是 `batch_validate` 的 `hypothesis`、`TODO.md` 与 skills；不要移植它的 LLM 因子生成循环。

## 算子族

- WorldQuant 101：Kakushadze, *101 Formulaic Alphas*，[arXiv:1601.00991](https://arxiv.org/abs/1601.00991)。公开的是公式与截面 rank 习惯。
- 国泰君安 191：《基于短周期价量特征的多因子选股体系》（2017）。公式见 `families.md`，社区复现仓库只作核对。

## 前几轮已被证伪、本轮不再测

- `explore_github_strategies` 包的四个思路：业绩预告漂移、增持−解禁、五日量价反转、EP+异常换手。两轮下来无可信优势（首折空对照分位 0.64、去膨胀 Sharpe 概率 0.03），更早谱系的四个冻结策略在 12 次窗外回放里均值超额 −1.8%、严格 walk-forward 链只有 +2.2%。
- 更早的 `github_strategies` 包与 `ref_github_strats`：反复交出小市值轮动与父本克隆。
- `ml_numpy_ranker_20260826` 谱系（Alpha158 的 numpy 改写）：只跑到闭式 ridge 与手工阈值的流动性 z-score，一轮即收工，从未训练树模型；最弱一折 −6.87%。这不是「Alpha158 + LightGBM 被证伪」，而是它从未被完整复现——本包第一轮就是补这一课。
- factor_cs 谱系里 LightGBM IC 门 vs ridge 的样本外对照两者都输给等权：那是用树模型做单因子门控，不是排序器。

## 本仓库

- 日线（含复权因子、换手、市值、涨跌停、停牌标记）与指数日线已落为 PIT parquet；正式策略从这些原料重算。
- 唯一总闸是 `available_at <= inference_at`；可见时间规则见 `docs/data-documentation.md §3.3`，单位见 `docs/units-reference.md`（沙箱里以 `unit_reference.json` 为准）。
- 正式 ABI、允许的库、`fit(context)`/`context.state_dir`、booster 的 `save_model`/`load_model`、`models/` 只读挂载见只读 `output/README.md`。
- 离线筛查脚本 `/mnt/tools/screen.py`（只经 `shell` 运行）给 rank IC、ICIR、衰减、规模中性 IC、头部十分位超额、换手与可交易性；它只读 `/mnt/snapshot` 决策视图，不替代 Validation。
