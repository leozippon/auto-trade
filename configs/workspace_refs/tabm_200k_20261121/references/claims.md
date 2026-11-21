# TabM：论文说了什么、代码里有什么、与本项目差在哪里

本文是包作者对论文与作者仓库的摘要，供会话参考；数字都摘自论文（以表、图编号标出）或仓库里随附的结果文件，摘不到的地方写明。论文以 `refs/papers/tabm/paper.pdf` 为准，`paper.txt` 的表格可能错位，表格的原始 LaTeX 在 `tex/tables/`。

## 1. 问题

表格数据的监督学习（每个对象一行特征、一个标签，分类或回归）。论文的出发点：近年的注意力类、检索类表格深度模型相对简单 MLP 的提升不稳定、也更慢；而梯度提升树（GBDT）仍是强基线。论文主张，「参数高效集成」能让普通 MLP 变成稳定而强的表格模型，并给出 TabM（Tabular DL model that makes Multiple predictions）。

## 2. 方法与关键部件

**关键部件：一个 MLP 同时表示 k 个子模型（implicit submodels），它们共享大部分权重、一起训练、各自给出预测，推理时取平均。** 论文固定 k = 32、不调（§3.3）。

基础 MLP（§3.3）：`MLP(x) = Linear(Block_N(...Block_1(x)))`，`Block(x) = Dropout(ReLU(Linear(x)))`，没有归一化层，没有残差。

BatchEnsemble 线性层（§3.2）：第 i 个子模型的线性层是 `l_i(x_i) = s_i ⊙ (W (r_i ⊙ x_i)) + b_i`。`W` 在 k 个子模型间共享；`r_i, s_i, b_i`（论文称 adapters）每个子模型各一份，相当于 `W_i = W ⊙ (s_i r_iᵀ)`。k 个子模型打包成一次前向：输入形状 `(B, k, d)`，`l(X) = ((X ⊙ R) W) ⊙ S + B`。每多一个子模型，每层只多 3d 个参数。

什么共享、什么各自一份：

| 部分 | TabM | TabM_mini | TabM_packed |
|---|---|---|---|
| 隐藏层权重 `W` | 共享 | 共享 | 每个子模型各一份（等于 k 个独立 MLP） |
| 第一层前的逐元素缩放 `R`（第一个 adapter） | 各自一份，随机初始化 | 各自一份，随机初始化（唯一的非共享部分） | 不适用 |
| 其余各层的 `R, S` 与偏置 `B` | 各自一份；除第一个 `R` 外，乘性 adapter 初始化为 1 | 没有 | 不适用 |
| 输出头 | 各自一份（`LinearEnsemble`） | 各自一份 | 各自一份 |
| 数值特征嵌入（如有） | 共享 | 共享 | 共享 |

论文从基线一步步推到 TabM（§3.3，图 2，46 个数据集上相对 MLP 的平均提升，均值 ± 标准差，跨数据集）：

| 模型 | 做法 | 相对 MLP |
|---|---|---|
| MLP×32 | 调好一个 MLP，再用 32 个种子独立训练、各自早停，预测取平均 | 0.96 ± 1.6% |
| TabM_packed | 32 个独立 MLP 打包成一个模型并行训练，按整个集成的验证分早停、按一个模型调参 | 1.42 ± 2.3% |
| TabM_naive | 按原始 BatchEnsemble 共享权重（所有 adapter 随机 ±1 初始化） | 1.86 ± 2.4% |
| TabM_bad | TabM_naive 去掉第一个 adapter `R` | 1.00 ± 2.3% |
| TabM_mini | 只保留第一个 adapter `R`（MiniEnsemble） | 1.96 ± 2.8% |
| TabM | TabM_naive 的结构，除第一个 `R` 外乘性 adapter 初始化为 1 | 2.15 ± 2.8% |
| TabM♠ | TabM，k 个子模型用同一批训练样本 | 2.07 ± 3.6% |
| TabM†_mini | TabM_mini 加分段线性特征嵌入（piecewise-linear embeddings） | 2.92 ± 4.1% |
| TabM†_mini ×5 | 5 个独立 TabM†_mini 再集成 | 3.20 ± 4.4% |

论文自己的归因（§3.5）：两点最重要——子模型同时训练（于是能按集成整体的验证分早停、按一个模型调参），以及子模型间权重共享（看起来是一种正则化）。第一个 adapter 是关键：它在特征第一次被 `W` 混合之前把同一个输入变成 k 个不同的表示（§3.3 TabM_mini 段；仓库 README 的 "The order of layers" 段把这条写成实现规则）。

训练与推理和别的做法的区别：

- 与普通 MLP 比：训练时优化 k 个预测各自损失的平均（mean loss），不是平均预测的损失；推理时把 k 个预测平均（分类要平均概率，不是 logits）。仓库 README 的 Training 段把这条标为 crucial。超参只多一个 k。论文观察到 TabM 的最优学习率通常比 MLP 高（附录 A.3）。
- 与 k 个独立训练的 MLP 比：参数量接近一个 MLP；一次前向算完 k 个子模型；早停与调参针对集成整体，而不是每个成员各自早停。
- 批次：默认（论文的 TabM）每个子模型看各自独立打乱的批次，输入形状 `(B, k, d)`；TabM♠ 让 k 个子模型共用同一批，输入 `(B, d)` 再复制成 k 份，实现更简单更快、平均性能略降（§3.4，图 2）。仓库 README 推荐默认用共享批次起步。
- 特征嵌入（†，§3.4、附录 D.8）：数值特征先经分段线性嵌入（按分位数分箱，`rtdl_num_embeddings.PiecewiseLinearEmbeddings(..., version='B')`）再展平送入骨干；对 TabM 帮助很大。
- 训练后可以裁剪子模型（§5.2）：贪心挑出的子集 TabM[G] 平均用 8.8 ± 6.6 个子模型，性能 2.02 ± 2.6%，略低于完整 TabM。

论文推荐：TabM 与 TabM_mini 是最有效的两个变体，TabM_packed 只在概念上重要（§3.3）；仓库 README 默认用 `arch_type='tabm'`，`tabm-packed` 「仅为完整性而实现，多数情况下更慢、更重、更弱」。实测最好的是带分段线性嵌入的 TabM†_mini（图 3 排名第一）。

分析里与本项目有关的几条：

- 单个子模型弱、集合强（§5.1，图 5、图 6）：训练不早停时，单个子模型相对普通 MLP 是过拟合的，平均后泛化明显更好；最好的单个子模型 TabM[B] 相对 MLP 是 −0.06 ± 1.8%。也就是说，TabM 的提升几乎全部来自集合预测。
- k 的影响（§5.3，图 7）：网络越宽越深，能容纳的有效子模型越多；k 过大可能变差；太窄（d = 64）或太浅（1 层）会吃亏。
- 死神经元比例（§5.4）：k = 1 与 k = 32 时分别是 0.29 ± 0.17 与 0.14 ± 0.09（46 个数据集平均）。

## 3. 实验设定

- 数据（§3.1，表 1，附录 C）：46 个公开数据集，训练集 1.8K–723K 行（中位 12K），特征 3–986 个（中位 20），28 个回归、18 个分类。37 个是沿用前人的随机划分；9 个是「domain-aware」划分：TabReD 的 8 个数据集（按时间划分）与 Microsoft（MSLR-WEB10K，原始划分）。
- 预处理（附录 D.2）：数值特征用 scikit-learn 分位数归一化的改动版（训练集上加微小噪声再拟合 `QuantileTransformer`，仓库里叫 `noisy-quantile`），类别特征 one-hot，二值特征映到 {0, 1}；TabReD 数据集完全按 TabReD 的预处理。
- 训练（附录 D.2）：回归用 MSE、分类用交叉熵；AdamW；不用学习率调度、不做数据增强；全局梯度裁剪 1.0；每个数据集固定批大小；验证集上连续 16 个 epoch 不提升就停。仓库里随附的 TabM 运行记录显示用了 bf16 自动混合精度（`amp = true`）。
- 调参（附录 D.2、D.9）：Optuna TPE，小数据集 100 次、大数据集（Covertype、Microsoft、TabReD）50 次。TabM 的搜索空间（表 6）：层数 UniformInt[1, 5]，宽度 UniformInt[64, 1024]，dropout {0, Uniform[0, 0.5]}，学习率 LogUniform[1e-4, 5e-3]，weight decay {0, LogUniform[1e-4, 1e-1]}；带嵌入的版本（表 7）层数 [1, 4]、学习率 [5e-5, 3e-3]、分箱数 [8, 32]。
- 评估（附录 D.2、D.3）：调好的配置从头用 15 个种子训练，报告测试指标的均值与标准差。回归用 RMSE，TabReD 的分类用 ROC-AUC，其余分类用准确率。排名用「层级」规则：A 只有在 `A_mean − A_std > B_mean` 时才算胜过 B（附录 D.3），所以名次范围比别的论文窄。
- 基线（§4.1）：MLP、FT-Transformer、SAINT、T2G-Former、ExcelFormer、TabR、ModernNCA，以及 XGBoost、LightGBM、CatBoost；附录 B.1 另有 ResNet、SNN、DCNv2、AutoInt、MLP-Mixer、Trompt、TabPFN。
- 硬件：大多数实验在单张 A100 上（附录 D.1）。

## 4. 主要数字

总体（图 3，46 个数据集）：平均名次 TabM†_mini 1.7 ± 1.2、TabM 2.8 ± 2.1、CatBoost 3.2 ± 2.0、XGBoost 3.3 ± 2.1、LightGBM 3.4 ± 2.0、MLP 5.5 ± 3.2。图 3 右栏单列 9 个 domain-aware 数据集（只有图，没有给数）。

按时间划分的 TabReD（表 20，单模型，15 个种子的均值 ± 标准差；↓ 越小越好，↑ 越大越好）：

| 数据集 | MLP | TabM | TabM†_mini | 最好的 GBDT |
|---|---|---|---|---|
| sberbank-housing ↓ | 0.2529 ± 0.0078 | 0.2439 ± 0.0021 | 0.2334 ± 0.0018 | XGBoost 0.2419 ± 0.0012 |
| ecom-offers ↑ | 0.5989 ± 0.0017 | 0.5941 ± 0.0003 | 0.5910 ± 0.0012 | XGBoost 0.5763 ± 0.0072 |
| maps-routing ↓ | 0.1625 ± 0.0001 | 0.1612 ± 0.0001 | 0.1610 ± 0.0001 | XGBoost 0.1616 ± 0.0001 |
| homesite-insurance ↑ | 0.9506 ± 0.0005 | 0.9640 ± 0.0002 | 0.9631 ± 0.0003 | CatBoost 0.9606 ± 0.0003 |
| cooking-time ↓ | 0.4828 ± 0.0002 | 0.4804 ± 0.0002 | 0.4804 ± 0.0001 | XGBoost / CatBoost 0.4823 ± 0.0001 |
| homecredit-default ↑ | 0.8538 ± 0.0014 | 0.8599 ± 0.0010 | 0.8635 ± 0.0008 | XGBoost 0.8670 ± 0.0005 |
| delivery-eta ↓ | 0.5493 ± 0.0007 | 0.5494 ± 0.0004 | 0.5510 ± 0.0019 | CatBoost 0.5465 ± 0.0001 |
| weather ↓ | 1.5378 ± 0.0054 | 1.4722 ± 0.0024 | 1.4651 ± 0.0020 | LightGBM 1.4625 ± 0.0008 |

读法：在这 8 个按时间划分的数据集上，不带嵌入的 TabM 胜过最好的 GBDT 4 个（ecom-offers、maps-routing、homesite-insurance、cooking-time），输 4 个；TabM†_mini 胜 5 个、输 3 个（homecredit-default、delivery-eta、weather）。胜负的幅度多在千分之几；ecom-offers 上普通 MLP 比 TabM 还好；delivery-eta 上 TabM 与 MLP 持平。

数据量变大时（§4.3，表 2，Weather 与 Maps Routing 的完整版，训练集分别为 13M 与 6.5M 行，验证与测试集与小版本相同）：Weather 上 XGBoost 1.4234、MLP 1.4842、TabM†_mini 1.4112、TabM†♠∗_mini 1.4090；Maps Routing 上 XGBoost 0.1601、MLP 0.1592、TabM†_mini 0.1582。训练时间（A100）：Weather 上 XGBoost 10 分钟、MLP 15 分钟、TabM†♠∗_mini 1.3 小时、TabM†_mini 3.3 小时；Maps Routing（986 个特征）上分别是 28 分钟、15 分钟、2 小时、13.5 小时。∗ 表示用了自动混合精度与 `torch.compile`。注意：这组实验沿用小版本上调好的超参（附录 D.4），GBDT 在大数据上未必调到最好；TabM 只用 3 个种子。

一个低信噪比的随机划分回归（表 19，year，即 YearPredictionMSD 的子集，RMSE ↓）：MLP 8.9628 ± 0.0232、TabM 8.8705 ± 0.0043、XGBoost 9.0307 ± 0.0028、LightGBM 9.0200 ± 0.0025、CatBoost 9.0370 ± 0.0073。这里 TabM 明显好于树。

Microsoft（MSLR-WEB10K，一个排序学习数据集，论文按回归用 RMSE 评估，没有报告排序指标；表 18，↓）：MLP 0.7475、TabM 0.7432、TabM†_mini 0.7423、XGBoost 0.7413、CatBoost 0.7412。树略好。

阶段一要复现的 Adult（表 18，准确率 ↑，单模型，15 个种子）：

| 模型 | 准确率 |
|---|---|
| MLP | 0.8540 ± 0.0018 |
| TabM | 0.8575 ± 0.0008 |
| TabM♠ | 0.8582 ± 0.0011 |
| TabM_mini | 0.8598 ± 0.0011 |
| TabM_packed | 0.8580（仓库 `paper/exp/tabm-packed/adult` 的 15 份结果文件的均值；表 18 没有这一行） |
| MLP† | 0.8603 ± 0.0009 |
| TabM†_mini | 0.8700 ± 0.0007 |
| XGBoost / LightGBM / CatBoost | 0.8720 ± 0.0006 / 0.8713 ± 0.0007 / 0.8714 ± 0.0012 |

Adult 上集成机制的增益（TabM − MLP = 0.0035）比特征嵌入的增益小，而树最好。这正适合做阶段一：数字小而稳定，跑得快，能同时检查实现、关键部件的消融和与树的比较。表 4 记 Adult 的划分：训练 26,048、验证 6,513、测试 16,281，批大小 256；测试集大小恰好是 UCI 的 `adult.test`，训练加验证恰好是 `adult.data` 的 32,561 行。表 4 记它有 6 个数值、1 个二值、8 个类别特征（共 15 个），而 UCI 原始文件是 14 列：这一处包作者没能对上，会话按 6 个数值、8 个类别列处理即可。重建时要注意：`adult.test` 的首行是说明行，它的标签写作 `<=50K.` / `>50K.`（带句点）；缺失值写作 `?`，可以当作一个类别；字段之间是逗号加空格。仓库里 `exp/tabm/adult/0-evaluation/0.toml` 的设置是 `num_policy = "noisy-quantile"`、`amp = true`、`share_training_batches = false`（即每个子模型独立打乱批次），`exp/mlp` 与 `exp/tabm-packed` 下的同名文件给出另两个模型的超参。

## 5. 作者代码里有什么

仓库 `yandex-research/tabm`，提交 `b53fd7cf2d9be2d66b918ca6cf66e0365e4ce11c`，Apache-2.0。剥离后约 3.6 MB。

- `tabm.py`：`tabm` 包本身，单文件，版本 0.0.3。主要对象：`TabM`（`TabM.make(n_num_features=..., cat_cardinalities=..., d_out=..., num_embeddings=..., arch_type='tabm'|'tabm-mini'|'tabm-packed', k=32, n_blocks, d_block, dropout)`，输出形状 `(B, k, d_out)`）；搭积木用的 `EnsembleView`、`LinearBatchEnsemble`、`LinearEnsemble`、`ElementwiseAffine`、`MLPBackbone`、`MLPBackboneBatchEnsemble`、`MLPBackboneMiniEnsemble`、`MLPBackboneEnsemble`、`make_tabm_backbone`，以及把现成模型的线性层换成集成版的 `ensemble_linear_layers_`、`batchensemble_linear_layers_` 等。`TabM.make` 的默认值：无嵌入时 3 层、有嵌入时 2 层，宽 512，dropout 0.1，k = 32；默认优化器 `AdamW(lr=0.002, weight_decay=0.0003)`（仓库 README）。
- `README.md`：包的用法、实现细节（层的顺序、权重共享、初始化）、超参建议。
- `example.ipynb`：端到端训练示例（California Housing、分位数归一化、分段线性嵌入、mean loss、共享或独立批次、早停）。notebook 是 JSON，可以当文本读；它开头的 `pip install` 与 `fetch_california_housing` 在离线会话里不能用。
- `tabm_reference.py`：已弃用的旧实现。
- `paper/`：论文实验代码。`bin/model.py`（论文用的初版 TabM 与 MLP 及训练循环，已被 `tabm` 包取代）、`bin/tune.py`、`bin/evaluate.py`、`bin/ensemble.py`、`bin/go.py`、`lib/`（数据、预处理、指标）、`tools/prepare_tabred.py`；`exp/<模型>/<数据集>/` 下是调参空间（`0-tuning.toml`）、调好的配置（`0-evaluation/<种子>.toml`）与结果（`report.json`）。本包只保留了 mlp、mlp-piecewiselinear、tabm、tabm-mini、tabm-packed、tabm-mini-piecewiselinear 六个模型在 Adult 与 TabReD 上的部分。`paper/README.md` 里有汇总这些结果文件的代码片段；它示例输出里 california 一行（−0.4509）与 `exp/tabm/california` 的结果（−0.4414，即表 18 的 TabM）对不上，而等于 TabM♠ 的值，示例输出应是旧的。
- 不包含：任何数据（论文数据集要从外部下载，TabReD 要 Kaggle 账号）；预训练权重（不需要）。

依赖与替换：

| 依赖 | 谁用 | 镜像里有吗 | 怎么办 |
|---|---|---|---|
| torch | `tabm.py` | 有（2.13） | 直接用 |
| typing_extensions | `tabm.py` | 随 torch 安装 | 直接用 |
| rtdl_num_embeddings 0.0.12 | `tabm.py` 顶层导入；分段线性与周期嵌入 | 没有 | 纯 Python wheel 在 `refs/wheels/`，用 `pip install --no-index --no-deps --target <工作区目录>` 装，加进 `PYTHONPATH`。它只依赖 torch；只有用决策树分箱时才要 scikit-learn |
| loguru、delu、tomli、tomli_w、optuna、rtdl_revisiting_models、tqdm、tensorboard | 只有 `paper/bin`、`paper/lib` | 多半没有 | 不需要：阶段一用 `tabm` 包加自己写的几十行训练循环（照 `example.ipynb`）即可；调参用小网格或随机搜索代替 Optuna |

策略包能导入的库里没有 `tabm` 与 `rtdl_num_embeddings`（也没有 typing_extensions），所以移进策略时要用 torch 重写用到的部分：`LinearBatchEnsemble`（逐元素乘 `r`、共享矩阵乘、逐元素乘 `s`、加偏置）、`LinearEnsemble`（k 个独立的头，用 `einsum` 或批量矩阵乘）、`ElementwiseAffine`、`EnsembleView`、初始化（`random-signs` / `normal` 与 TabM 的「除第一个外初始化为 1」），以及如需要的分段线性嵌入。这部分代码量小；Apache-2.0 允许照抄并改写，保留出处说明即可。

## 6. 批评与需要注意的地方

- 与 GBDT 的胜负并不一边倒。表 20 的 8 个按时间划分的数据集上，TabM 只胜一半，幅度多在千分之几；Adult、Microsoft 上树更好。论文的总体排名用宽松的「层级」规则（附录 D.3），小差别不计胜负。
- 增益是相对 MLP 报告的平均数，跨数据集的标准差（图 2 约 ±2.8 个百分点）比均值（2.15%）还大；在 domain-aware 划分上图 3 只有图没有数。
- 集成之后的提升几乎全部来自平均（TabM[B] ≈ MLP），所以 TabM 的价值取决于数据里「子模型的误差有多不相关」。低信噪比的数据上平均通常有用，但这一点论文没有在金融数据上验证。
- 每个数据集都做了 50–100 次 TPE 调参、15 个种子；会话每次再拟合不可能这样调。
- 论文的任务都是逐行独立的点预测（RMSE、准确率、ROC-AUC）；没有截面排序、没有标签重叠、没有滚动再拟合。Microsoft 本是排序数据集，论文也只当回归评估。
- 大数据集上的比较沿用小版本的超参（附录 D.4），GBDT 与 TabM 都未必最好。
- 同组的 TabReD（ICLR 2025）是理解这篇论文在「时间划分」上读数的背景：它收集了 8 个带时间戳、特征很多的工业数据集，发现按时间划分会改变方法的排名，在 TabReD 上表现最好的是简单的 MLP 类结构与 GBDT（TabReD 的 arXiv 摘要）。TabM 表 20 的数据就来自它。TabReD 不在本包里。

## 7. 与本项目问题的差别，以及对移植的含义

| 方面 | 论文 | 本项目 |
|---|---|---|
| 行 | 独立同分布的对象，最多 723K 训练行（完整版 13M） | 股票-交易日；研究年里每天约 2,500 到 4,700 只股票，一次拟合的训练窗口约数百万行 |
| 标签 | 点标签，信噪比通常不低 | 其后若干交易日的收益，信噪比很低，厚尾 |
| 目标 | RMSE、准确率、ROC-AUC | 每天截面上的排序（选出前几十只），读数是逐日截面秩相关与组合收益 |
| 标签时间结构 | 无 | 期限 h 的标签在相邻 h 天里重叠 |
| 划分 | 一次固定的训练 / 验证 / 测试 | 按会话选定的节奏滚动再拟合，每次只用当时可见的数据 |
| 预处理 | 全局分位数归一化 | 特征分布随市场状态漂移，截面上的相对位置更有意义 |

对移植的含义（建议，会话可以不采纳，但改动要写理由）：

- 特征按日做截面归一化（每天在当天可交易的股票上取秩再映到正态，或去均值除以标准差并截尾），代替论文的全局分位数归一化；如用分段线性嵌入，分箱在截面归一化之后的训练分布上算。
- 标签也按日处理：截面秩或 z 分数。损失可以先用 MSE（论文的做法；标签按日标准化后，它拟合的是截面上的相对高低，不再被整个市场的涨跌主导），再试逐日的相关损失（每个子模型在每一天上算预测与标签的 Pearson 相关，取负、对天与子模型平均）或成对、列表式的排序损失。不论哪种，都要对 k 个子模型各自算损失再平均，不能先平均预测再算损失（这是论文与仓库都强调的一条）。
- 批次：逐日的相关或排序损失需要一批里是完整的（或抽样的）同一天截面，一批放一天或几天。共享批次（TabM♠）最容易实现；要让每个子模型看不同的批，可以让每个子模型抽不同的日子，但每天股票数不同，要固定每天抽的只数或做掩码。
- 推理：k 个子模型的输出先各自按日标准化再平均，或直接平均（用 MSE 训练时两者差别不大）；排序看平均后的打分。
- 验证与早停：早停用最近一段训练期之后的验证段，验证段与训练段之间空出至少 h 个交易日，否则重叠的标签会把验证分抬高；早停的指标用验证段上的逐日截面秩相关，而不是 MSE。再拟合时沿用同样的规则，超参在研究年里选定后固定。
- 规模：论文的 TabM 在 A100 上处理百万到千万行是可行的（表 2）；本项目每次拟合约两百万行、50–200 个特征，属于这个范围。每次 `fit` 有时限（运行事实 `budgets`），先用冒烟量出一次拟合的时间与显存。
