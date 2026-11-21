# AlphaQCM：论文说了什么、代码里有什么、搬到本项目要改什么

本文是包作者对论文与官方代码的摘要，供会话判断从哪里下手。数字都摘自论文（`refs/papers/alphaqcm/paper.txt`，以 `paper.pdf` 为准）或代码（`refs/papers/alphaqcm/repo/`，提交 `dffc695f9ea5`）；包作者没能核实的地方写明了。

## 问题：公式化因子挖掘

一个公式化因子（formulaic alpha）是一个由算子和操作数组成的表达式，把每只股票过去的价量映射成一个截面打分，例如 `Corr(Close, Volume, 20)`。论文的操作数是六个日频字段 Open、High、Low、Close、Vwap、Volume，加上常数（{−30, −10, −5, −2, −1, −0.5, −0.01, 0.01, 0.5, 1, 2, 5, 10, 30}）与时间窗（{10, 20, 30, 40, 50} 天）；算子分时间序列算子（Ref、TsRank、Mean、Med、Sum、Std、Var、Max、Min、WMA、EMA、Cov、Corr）和截面算子（Sign、Abs、Log、Rank、Add、Sub、Mul、Div、Greater、Less），见论文 Table B.5。表达式用逆波兰记法（RPN）写成一串 token，最长 20 个。

目标不是找一个最好的公式，而是找一个最多 P 个公式的因子池（alpha pool），用线性组合得到一个 meta-alpha，使它与未来收益的截面相关（IC）尽量高。池的维护（论文 Algorithm 1）：每个因子先按日截面标准化；新公式加进池后拟合线性权重；池超过 P 个时去掉权重绝对值最小的那个；奖励是新池的 IC 减旧池的 IC。代码里的权重不是对收益直接做 OLS，而是在训练集上用 Adam 最小化 `1 − 2·wᵀIC + wᵀM w + 0.005·‖w‖₁`（IC 是各因子与目标的 IC，M 是因子两两之间的 IC），与按日标准化后的最小二乘等价，另加 L1；与池里已有因子相关超过 0.99 的新公式直接不收（`alphagen/models/alpha_pool.py`）。

## 方法：AlphaQCM 加了什么

AlphaGen（Yu 等，KDD 2023）把生成公式看成一个 MDP：状态是已生成的 token 序列，动作是下一个 token（不合语法的动作被屏蔽），只有公式完成时才有奖励，γ = 1。AlphaGen 用 PPO 解这个 MDP。AlphaQCM 指出两点：因子池一变奖励函数就变（非平稳），而大多数公式对池没有贡献（奖励稀疏）。它的做法：

- 用 DQN 学 Q 函数（一个网络，平方 TD 误差），用 IQN 学累计奖励的分位数（另一个网络，分位数 Huber 损失）；两者都是 2 层、隐藏 128 的 LSTM 编码 token 序列，加两层 64 的全连接头（Appendix E.2）。
- 关键部件是 QCM 方差估计：按 Cornish-Fisher 展开，分位数 θ_k ≈ 截距 + φ_k·√h + (φ_k²−1)·√h·s/6 + (φ_k³−3φ_k)·√h·(κ−3)/24，其中 φ_k 是标准正态分位数；对 K 个分位数做 OLS，第二个系数的平方就是方差 h 的估计。论文的论点是：非平稳使分位数有偏，偏差落进截距，方差估计仍然一致（Proposition 3.1）。
- 选动作：a_t = argmax_a [Q(x_t, a) + λ·√ĥ(x_t, a)]（式 5），λ ∈ {0.5, 1, 2}；另有 ε = 0.01 的随机动作，前 10,000 步全随机。代码在 `fqf_iqn_qrdqn/agent/iqcm_agent.py` 的 `exploit` 与 `fqf_iqn_qrdqn/model/alpha_iqn.py` 的 `calculate_higher_moments`。
- 带优先级的经验回放，回放池 100,000，批 128，学习率 5e-5，目标网络每 5,000 步同步（Table E.6）。

论文自己的消融（Table 2，各 10 个种子的均值与标准差，测试期 IC）：

| 骨干 | 方差项 | CSI 300 | CSI 500 | Market |
|---|---|---|---|---|
| IQN | 无（λ = 0） | 7.17% ± 2.40% | 8.58% ± 1.47% | 7.04% ± 1.82% |
| IQN | 朴素分位数方差 | 6.16% ± 1.73% | 8.75% ± 1.03% | 8.42% ± 1.59% |
| IQN | QCM | 8.49% ± 1.03% | 9.55% ± 1.16% | 9.16% ± 1.61% |
| QRDQN | 无 | 6.96% ± 1.64% | 8.54% ± 1.56% | 7.06% ± 1.92% |
| QRDQN | QCM | 7.59% ± 0.81% | 9.08% ± 1.07% | 9.12% ± 1.74% |

去掉方差项的差距在 CSI 300 与 CSI 500 上约为一个种子间标准差，在全市场上稍大；论文还做了参数量对齐的对照（Table G.7：AlphaGen 加大到约 57 万参数、AlphaQCM 缩小到约 30 万参数，排序不变）和预置 Alpha101 经验的对照（Table 3：训练全程跑完后不预置更好）。

## 实验设定

- 数据：中国 A 股，Qlib 格式，由 baostock 下载（`data_collection/fetch_baostock_data.py`），三个股票池：CSI 300、CSI 500、沪深全市场（Market）。另有 S&P 500 的一组（Table 4）。
- 时段：训练 2010-01-01 至 2019-12-31，验证 2020 年，测试 2021-01-01 至 2022-12-31（第 4 节）；附录 G.3 把测试期延长到 2024-12-31 重做了一遍。
- 目标：20 个交易日的未来收益，代码里是 `Ref(close, -20) / close − 1`，即从当天收盘到 20 天后收盘；公式用到当天收盘为止的数据，所以默认能按当天收盘成交。
- 指标：只有 IC（逐日 Pearson 相关的均值，Appendix A）。没有 RankIC、没有 ICIR、没有组合回测、没有交易成本；代码里有一个 Qlib 的 top-k 策略文件（`alphagen_qlib/strategy.py`），论文没有用它报告数字。
- 选择：池大小 P ∈ {10, 20, 50, 100} 按验证集选（4.1 节）；每个设定 10 个种子。总步数按 P 为 250k、300k、350k、400k（Table E.6）。
- 基线：Alpha101（线性组合）、MLP、XGBoost、LightGBM（用 Qlib 实现，「预设超参数」，喂给它们的特征论文没写）、带或不带互相关过滤的 GP、带过滤的 PPO、AlphaGen。

## 主要数字

Table 1（测试期 2021–2022 的 IC，10 个种子的均值 ± 标准差）：AlphaQCM 在 CSI 300、CSI 500、Market 上为 8.49% ± 1.03%、9.55% ± 1.16%、9.16% ± 1.61%；AlphaGen 为 8.13% ± 0.94%、8.08% ± 1.23%、6.04% ± 1.78%；LightGBM 为 2.93%、4.16%、4.28%；Alpha101 为 3.44%、4.38%、3.15%。论文的主张是「股票越多，AlphaQCM 相对 AlphaGen 的优势越大」。在 CSI 300 上它与 AlphaGen 的差在一个标准差之内。

这些数是 20 天收益上的 IC，不是组合收益，也没有扣成本；附录 G.3 延长测试期的数字本文不转述。

## 官方代码里有什么

仓库在 GitHub 上标约 62 KB，解压后 58 个文件、约 250 KB，除配置与说明外全是 Python，没有数据和权重。

| 部分 | 文件 | 说明 |
|---|---|---|
| 表达式 | `alphagen/data/expression.py`、`tokens.py`、`tree.py` | 算子类（一元、二元、滚动、成对滚动），每个算子在 `[天, 股票]` 张量上求值，滚动算子用 `unfold`；`tree.py` 按 RPN 建树并给出合法动作 |
| 动作空间 | `alphagen/config.py`、`alphagen/rl/env/wrapper.py` | 实际启用的算子是 Abs、Log、Add、Sub、Mul、Div、Greater、Less、Ref、Mean、Sum、Std、Var、Max、Min、Med、Mad、Delta、WMA、EMA、Cov、Corr；Sign、截面 Rank、TsRank、Skew、Kurt 在代码里被注释掉，与 Table B.5 不完全一致。动作数 = 22 个算子 + 6 个字段 + 5 个时间窗 + 14 个常数 + 结束符 = 48 |
| 环境 | `alphagen/rl/env/core.py` | 公式完成时调用池的 `try_new_expr`，非法公式奖励 −1 |
| 因子池 | `alphagen/models/alpha_pool.py` | 见上文「问题」一节 |
| 数据 | `alphagen_qlib/stock_data.py`、`calculator.py` | 用 Qlib 把六个字段读成 `[天, 字段, 股票]` 张量，前留 100 天、后留 30 天；按日标准化、缺失填 0 后算 IC |
| 智能体 | `fqf_iqn_qrdqn/agent/iqcm_agent.py`、`qrqcm_agent.py`、`fqcm_agent.py`、`base_agent.py` | QCM 与 IQN、QR-DQN、FQF 三种骨干的组合；`base_agent.py` 是训练循环、每 1,000 步在验证集和测试集上评估并存表 |
| 网络、回放 | `fqf_iqn_qrdqn/network.py`、`model/`、`memory/` | LSTM 编码器、余弦分位数嵌入、优先级回放 |
| 脚本 | `train_qcm_csi300.py`、`train_qcm_csi500.py`、`train_qcm.py`（全市场）、`train_drl_csi300.py`（不带 QCM 的原版 QR-DQN、IQN、FQF 智能体；论文的「无方差」消融是 QCM 智能体取 `--std-lam 0`）、`qcm_config/*.yaml` | 命令行参数 `--model {qrdqn, iqn, fqf} --pool --std-lam --seed` |

代码与论文对不上、会话复现时要自己定的地方：

- 奖励：论文写新池 IC 减旧池 IC，`try_new_expr` 返回的是新池 IC 本身（AlphaGen 的写法）。包作者没有核实论文的数字用的是哪一种。
- 步数：`qcm_config/*.yaml` 的 `num_steps` 是 2,000,000，论文 Table E.6 是 25 万到 40 万。
- 评估：`base_agent.py` 同时存「验证集最好」和「测试集最好」两张表（`valid_best_table.csv`、`test_best_table.csv`）；后者是在测试集上挑检查点。论文说池大小按验证集选，但没说检查点怎样选；复现时只用验证集挑。
- 代码对 QCM 回归的标准差项没有限制非负，λ·√ĥ 实际用的是 OLS 系数本身，可能为负。

## 镜像之外的依赖与替换

| 依赖 | 用在哪里 | 替换 |
|---|---|---|
| Qlib | `alphagen_qlib/stock_data.py` 读数据 | 用本项目日线自己拼 `[天, 字段, 股票]` 的 torch 张量（字段与可见时点见 `refs/README.md` 的「本臂的数据与账户」），保留 `StockData` 的属性（`data`、`n_days`、`n_stocks`、`max_backtrack_days`、`max_future_days`），表达式类就能原样用；`StockData.__init__` 会初始化 Qlib，子类要绕开它 |
| gymnasium | `core.py`、`wrapper.py` 只用作基类和 `spaces.Discrete`/`Box` | 换成普通类：`action_space.n = 48`、`observation_space.shape = (20,)`、一个从合法动作里均匀抽样的函数；包装类还要把属性访问转给内层环境（`base_agent.py` 经包装类读 `env.pool`、`env.env.pool`，并调用 `env.close()`） |
| tensorboard（`torch.utils.tensorboard.SummaryWriter`） | `base_agent.py` 记日志 | 写 CSV |
| pyyaml | 读 `qcm_config/*.yaml` | 镜像里有 |
| baostock、pandera、cv2、stable-baselines3 | 数据下载、AlphaGen 的 PPO、未用到的 Atari 包装 | 不需要 |

其余（torch、numpy、pandas）都在镜像里，也都在策略可导入的名单里。策略包不能导入 `refs/`，所以表达式、池、QCM 智能体要复制进策略包并改成绝对导入。训练脚本按当前目录读 `qcm_config/*.yaml`，把日志、检查点与因子表写到 `AlphaQCM_data/alpha_logs/`；`refs/` 每次尝试开头重新复制，所以在工作区别处运行，产物才留得下。

## 已知的批评与注意事项

- 只有 IC：没有成本、没有组合、没有换手；20 天收益的 IC 与一个 20–60 只、带成本的多头组合之间隔着很远。
- 测试期短（两年），且只有一次切分；10 个种子的散布与方法之间的差同量级。
- 公开数据的问题：价格按下载当天做前复权（`fetch_baostock_data.py` 的 `forward_adjust_date`），价格水平里带着以后的复权信息，含价格水平的公式（例如 `Close + 5`）因此不是按时点可见的；指数成分来自 Qlib 的成分文件，质量参差；停牌日的处理没有专门写。
- 多重检验：一次挖掘评估数万个公式，池的权重和成员都在训练集上按 IC 挑出，训练 IC 必然偏高；论文只报告测试期 IC，但检查点与 P 的选择用了验证集，代码还保留了按测试集挑的表。
- 后来的论文（例如 AlphaSAGE，ICLR 2026）在另一时段把 AlphaQCM 当基线重跑，它与 AlphaGen 的相对位置在不同股票池上不一致。
- 论文没有给训练用时与硬件，只说时间主要花在与环境的交互（公式求值）上（附录 G.2）。

## 与本项目问题的差别，以及搬过来时怎样做

| 论文 | 本项目 |
|---|---|
| 固定股票池，当天收盘可成交 | 按时点可见的全市场面板（主板与创业板），有停牌、涨跌停与真实成本；决策在开盘前，只看到前一交易日为止的日线 |
| 目标是 20 天收盘到收盘收益，指标是 IC | 目标是持仓期的收益；最终看 20–60 只多头组合扣成本后的读数 |
| 一次切分、一次挖掘 | 回放里按会话选定的节奏再拟合 |

由此而来的做法：

- 挖掘放在 `fit` 里，只用当时可见的数据；训练窗口的最后一段留作验证，用来选 P 与检查点，不在测试上选任何东西。
- 目标换成从决策后第一个可成交价（例如次日开盘或当日均价）起、与会话持有期相同的收益；`target` 只在 `fit` 里、只对已经到期的样本计算。
- 公式在决策时只用可见的日线求值；价格字段用当时可见的复权因子做后复权（收盘价 × 复权因子），不要用以某个未来日期为基准的前复权；Vwap 用成交额除以成交量。
- 停牌与缺失：求值前在可交易掩码之外置 NaN，算 IC 时按 NaN 掩掉（代码里标准化后把缺失填 0 再算 IC：两边都缺时不影响 IC，只有一边缺时会把 IC 往 0 拉一点；排序与组合里则必须把不可交易的股票去掉），打分只在可交易股票上排序。
- 再挖掘的节奏与每次的算力：一次挖掘的耗时先在研究年上量出来，再定是每次再拟合都重挖、隔几次重挖、还是只重拟合池的权重（权重重拟合很便宜）；也可以把智能体与池放在 `context.state_dir` 里接着挖。每次 `fit` 的时限在运行事实的 `budgets` 里。
- 防止在噪声上挑选：研究年的最后一段不给挖掘器看，只用来读结果；对照同样大小的随机公式池（从同一文法随机抽、权重同样拟合）和同样操作数喂给 LightGBM 的排序器；比较时按日配对，并考虑 20 天收益的重叠（用不重叠的子样本或按块的标准误）。
- 基本面字段可以作为额外的操作数加入，但要在忠实版本之后作为扩展，并单独记录。
