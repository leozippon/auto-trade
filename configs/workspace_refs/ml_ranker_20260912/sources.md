# 来源、文献与已证伪清单

只引用公开页面并转述机制。不要 vendor 任何仓库，不要下载权重。沙箱不挂载上游仓库；宿主若在 `external_references/` 下留有 Qlib 或 RD-Agent 的克隆，下面的仓库相对路径同样适用，但 Agent 在沙箱里读不到它们，`models.md` 与 `protocol.md` 的转述就是全部可用材料。

## 上游（引用，禁止拷贝）

- Qlib：<https://github.com/microsoft/qlib>。`qlib/contrib/data/loader.py`（`Alpha158DL`、`Alpha360DL` 的特征表达式）、`qlib/contrib/data/handler.py`（处理器与默认标签）、`qlib/data/dataset/processor.py`（`RobustZScoreNorm`、`CSRankNorm`）、`qlib/contrib/model/gbdt.py`（LightGBM）、`qlib/contrib/model/pytorch_nn.py`（`DNNModelPytorch`，MLP）、`qlib/contrib/model/pytorch_gru.py`、`qlib/contrib/model/pytorch_transformer.py`（序列模型的公开结构：GRU `hidden_size=64, num_layers=2`，Transformer `d_model=64, nhead=2, num_layers=2`，均对 Alpha360 输入）、`qlib/contrib/strategy/signal_strategy.py`（`TopkDropoutStrategy`）、`examples/benchmarks/` 下各模型的 yaml 与 README 表。那些表是他们的数据、复权、费用与 executor 跑出来的，不是本折预期。
- RD-Agent：<https://github.com/microsoft/RD-Agent>，`rdagent/scenarios/qlib/` 的因子/模型研发循环。只借用「假说→实现→回测→反馈」的过程形态；本环境的对应物是 `batch_validate` 的 `hypothesis`、`TODO.md` 与 skills。

## 文献

- Gu, Kelly and Xiu, *Empirical Asset Pricing via Machine Learning*（RFS 2020）：树模型与浅层网络在截面预测上的相对表现，以及验证集选超参、滚动重训的做法。
- Bailey and López de Prado, *The Deflated Sharpe Ratio*（JPM 2014）：本环境 `selection_statistics.deflated_sharpe_probability` 的来源。
- López de Prado, *Advances in Financial Machine Learning*（2018）第 7 章：清洗与禁运的时间序列交叉验证。
- Ke et al., *LightGBM*（NeurIPS 2017）。

## 前几轮已被证伪、本轮不再重复

- `ml_numpy_ranker_20260826` 谱系：季度切分下退化成一条手工阈值的流动性 z-score；改成年度窗后只训了闭式 ridge、`REFIT_PERIOD="quarter"`、一轮三候选即在预算 14–21% 处收工，最弱一折 −6.87%；从未训练过树模型或网络（当时合同只给 numpy/pandas）。本包第一轮就是把树模型与网络跑出来，且不允许一轮收工。
- factor_cs 谱系里 LightGBM IC 门 vs ridge 的样本外对照两者都输给等权：那是单因子门控，不是排序器。
- 一批三个候选曾在宿主争用下被决策超时误杀：策略自身只需约 7 s，环境侧启动占了时钟。现已把容器就绪移出决策时钟，但策略计算仍要留余量。
- 更早的深度学习实验没有留下可复用的记录；本包不把任何未记录的结果当先验。

## 本仓库

- 日线（含复权因子、换手、市值、涨跌停、停牌标记）已落为 PIT parquet；正式策略从这些原料重算。
- 唯一总闸是 `available_at <= inference_at`；可见时间规则见 `docs/data-documentation.md §3.3`，单位见 `docs/units-reference.md`（沙箱里以 `unit_reference.json` 为准）。
- 正式 ABI、允许的库、`fit(context)`/`context.state_dir`、booster 的 `save_model`/`load_model`、`np.savez` 持久化、`models/` 只读挂载见只读 `output/README.md`。
- 本臂的 GPU 由实验参数 `gpu_count` 分配，会话容器（8 核、8 GiB 内存、1 GiB `/tmp`）与正式回放的策略容器取同一个值：大于 0 时两者都挂 GPU，各自在启动时按空闲显存挑卡，同一实验的容器共享这些卡；为 0 时都没有。策略容器的生效数量是运行事实 `budgets.strategy_gpu_count`；`runtime_env.json` 的 `sandbox_spec` 只记本会话自己的容器（Meta 会话无容器，为 `null`）。策略代码一律探测 `torch.cuda.is_available()`，不得写死。
- 离线筛查脚本 `/mnt/tools/screen.py`（只经 `shell` 运行）给单信号的 rank IC、ICIR、衰减、规模中性 IC、头部十分位超额、换手与可交易性；模型验证 IC 用自己的脚本在 `/mnt/snapshot` 上算，不碰验证区间。
