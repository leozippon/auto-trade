# 论文索引

这里是 25 篇关于测试时训练与测试时适应的论文，2019–2026 年，每篇一个文件：标题、作者、出处、arXiv 编号与日期、许可与摘要。许可允许再分发的 16 篇另收正文，取自 arXiv 的 HTML 版，去掉了参考文献，公式保留为 LaTeX；其余 9 篇只有摘要，因为它们的许可（arXiv 的非独占许可，或论文集没有声明开放许可）不允许再分发正文，而 arXiv 的元数据（含摘要）是 CC0。文件名里的年份是第一个公开版本的年份。2026-10-27 下载。

每段写这篇做了什么，以及它与本包的哪个问题有关：测试时用什么目标更新、更新放在哪里、什么时候更新反而有害、标签延迟与时点，以及怎样判断模型在分布外的时期是否真的适应了。

## 经典的测试时训练与测试时适应

`2019_ttt_self_supervision.md`：Sun 等，Test-Time Training with Self-Supervision for Generalization under Distribution Shifts（ICML 2020，CC0，全文）。提出测试时训练：主任务与一个自监督辅助任务（图像旋转预测）共用特征提取器，测试时先用辅助任务在测试样本上更新共享部分、再做主任务预测；有逐样本重置与沿测试流一直更新的在线两种版本，并给出辅助任务与主任务梯度正相关时更新有益的分析。与本包有关的是：自监督目标怎样经共享表示把测试分布的信息传给主任务，以及在线累积与逐次重置的区别。

`2020_tent.md`：Wang 等，Tent: Fully Test-time Adaptation by Entropy Minimization（ICLR 2021，arXiv 非独占许可，只有摘要）。测试时只最小化预测的熵，只更新归一化层的统计量与逐通道仿射参数，不用源数据也不用辅助任务。它是最便宜的一类无标签更新；分类之外（回归、排序）没有现成的熵，要会话自己定义对应的量。

`2021_ttt_plus_plus.md`：Liu 等，TTT++: When Does Self-Supervised Test-Time Training Fail or Thrive?（NeurIPS 2021，未声明开放许可，只有摘要）。指出分布偏移严重时测试时训练可能反而变差；用训练时离线保存的特征统计摘要、测试时在线做矩匹配来约束适应，并换用信息更多的对比式自监督。直接关系本包的主要问题：更新什么时候有害，以及怎样用训练时留下的摘要约束它。

`2021_memo.md`：Zhang、Levine、Finn，MEMO: Test Time Robustness via Adaptation and Augmentation（NeurIPS 2022，arXiv 非独占许可，只有摘要）。对单个测试样本做多种增广，最小化增广上平均预测分布的熵，更新全部参数后预测，再重置。关系：只有一个样本时的适应，以及增广下的预测一致性作目标；价量序列上什么算保持语义的增广要自己定义。

## 测试时训练层与测试时记忆

`2024_ttt_layers.md`：Sun 等，Learning to (Learn at Test Time): RNNs with Expressive Hidden States（arXiv 预印本，CC BY 4.0，全文）。TTT 层把序列层的隐状态做成一个小模型（线性或两层 MLP）的权重，每读一个 token 就用自监督重建损失对它走一步梯度；小批量 TTT 与对偶形式让它能并行训练，外层训练连自监督任务本身（几组投影）一起学。关系：测试时训练可以放在模型自己的前向计算里，快权重在一条序列内更新、序列之间重置。代码在 `../code/ttt-lm-pytorch`。

`2024_titans.md`：Behrouz、Zhong、Mirrokni，Titans: Learning to Memorize at Test Time（arXiv 预印本，CC BY 4.0，全文）。一个神经长期记忆模块（MLP）在测试时按「惊奇度」更新：惊奇度是联想记忆损失的梯度，带动量，另有按数据与容量衰减的遗忘；三种把记忆接进架构的方式（作上下文、作门、作层），实验含时间序列。关系：测试时记住什么、遗忘多快；分布漂移时旧记忆要能被冲掉。非官方的实现在 `../code/titans-pytorch`。

`2025_test_time_regression.md`：Wang 等，Test-time regression: a unifying framework for designing sequence models with associative memory（arXiv 预印本，CC BY 4.0，全文）。把联想回忆写成测试时的回归：线性注意力、状态空间模型、快权重、在线学习器与 softmax 注意力都是三项选择（回归权重、回归函数类、测试时优化算法）的特例。关系：「推断时拟合一个小回归」的统一设计空间，经典回归里的加权、正则与递推解都落在其中。

`2025_ttt_video.md`：Dalal 等，One-Minute Video Generation with Test-Time Training（CVPR 2025，CC BY 4.0，全文）。在一个预训练的 5B 扩散 Transformer 里加入 TTT 层（隐状态是两层 MLP）后微调，生成一分钟、多场景的连贯视频，人评胜过 Mamba 2、Gated DeltaNet 与滑窗注意力。关系：把 TTT 层加进已训练好的模型再微调，而不是从头训练。

`2025_miras.md`：Behrouz 等，It's All Connected: A Journey Through Test-Time Memorization, Attentional Bias, Retention, and Online Optimization（arXiv 预印本，CC BY 4.0，全文）。把各类序列模型看作学习键到值映射的联想记忆，由四项选择决定：记忆的架构、内部目标（注意力偏置）、保持门（把遗忘看作正则）与记忆的学习算法；提出 L2 之外的内部目标（如 Huber）与新的遗忘门。关系：测试时目标与遗忘形式的选择；厚尾的收益序列上内部目标怎样选是开放的问题。

`2025_lact.md`：Zhang 等，Test-Time Training Done Right（arXiv 预印本，CC BY 4.0，全文）。大块测试时训练：快权重每次更新的块从 16–64 个 token 放大到 2K–1M，GPU 利用率高出几个数量级，非线性状态可以大到模型参数的约四成，在线更新可以用 Muon 等优化器；在新视角合成、语言模型与自回归视频扩散上验证。关系：以一整块数据（例如一天的截面或一段日子）为单位更新，以及在卡上成批计算的效率。代码在 `../code/LaCT`，其中有两个不依赖专用内核的最小实现。

`2025_ttt_e2e.md`：Tandon 等，End-to-End Test-Time Training for Long Context（arXiv 预印本，CC BY 4.0，全文）。把长上下文建模当作持续学习：标准的滑窗注意力 Transformer 在测试时对读到的上下文做下一 token 预测，把上下文压进权重；训练时用元学习优化这份测试时学习的初始化，即直接优化「测试时更新之后」的损失。关系：外层训练为内层的测试时更新而优化，与旧的动态评估（dynamic evaluation）的区别也在这里。官方代码没有许可声明，未收。

`2025_nested_learning.md`：Behrouz 等，Nested Learning: The Illusion of Deep Learning Architectures（NeurIPS 2025 发表过其一个版本，CC BY 4.0，全文）。把模型看作多层嵌套、各有自己上下文流与更新频率的优化问题；常用优化器本身是压缩梯度的联想记忆；提出按不同频率更新的连续谱记忆系统与自修改的序列模型 Hope。关系：多时间尺度的更新；`../families.md` 里关闭的分层持续记忆实现（逐日梯度更新、分层时钟）与它相近，读时可以对照那里的读数。

## 语言模型上的测试时训练

`2023_ttt_nearest_neighbors.md`：Hardt、Sun，Test-Time Training on Nearest Neighbors for Large Language Models（ICLR 2024，arXiv 非独占许可，只有摘要）。对每个测试输入从大规模文本索引里取最近邻，用这些近邻的文本对语言模型各做一步梯度再评估；只要 20 个近邻就在 Pile 的二十多个任务上大幅改善，但索引的质量与规模是必要条件。关系：近邻用来更新参数，而不是平均它们的结局（上一方向关闭的是后者）。代码在 `../code/tttlm`。

`2024_ttt_few_shot.md`：Akyürek 等，The Surprising Effectiveness of Test-Time Training for Few-Shot Learning（ICML 2025，arXiv 非独占许可，只有摘要）。推断时用任务给出的少量示例构造训练数据（留一与增广），为每个任务临时训练适配器再预测；在 ARC 上准确率最高到微调基线的 6 倍，在 BIG-Bench Hard 的 10 示例设定上比少样本提示高 7.3 个百分点。关系：把「当下」当作一个只有少量样本的新任务、临时训练一个小适配器。代码在 `../code/marc`。

`2025_specialization_after_generalization.md`：Hübotter 等，Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models（ICLR 2026，CC BY 4.0，全文）。解释测试时训练为什么、何时有效：模型在全局上参数不足，测试时训练把容量集中到与当前任务相关的少数概念上，因此即使在分布内也能降低误差；有线性表示假设下的模型与图像、语言上的规模研究。关系：本包问的是分布外的适应，而这篇说收益也可能来自分布内的专门化；读开与关的配对时，分布漂移的时期与平稳的时期分开看，能把两种来源分开。

`2026_ttt_discover.md`：Yuksekgonul 等，Learning to Discover at Test Time（ICML 2026，CC BY 4.0，全文）。针对单个科学或工程问题在测试时做强化学习，让模型在这个问题上的尝试中继续训练，目标是一个最好的解而不是平均表现；在数学、GPU 内核、算法竞赛与单细胞去噪上达到新水平。关系较远：它的测试时是对一个固定问题反复试错，收在这里作测试时训练在推理一端的最新形态。

## 时间序列与金融

`2022_fsnet.md`：Pham 等，Learning Fast and Slow for Online Time Series Forecasting（ICLR 2023，CC BY 4.0，全文）。在线时间序列预测：慢学的主干，每层加一个按梯度快慢变化触发的快适配器，再加一个联想记忆，遇到反复出现的模式时取回旧的适配。关系：同时应对新出现与反复出现的分布；它每一步都用刚到的真值更新，本包里真值要等持有期走完才到。代码在 `../code/fsnet`。

`2023_doubleadapt.md`：Zhao 等，DoubleAdapt: A Meta-learning Approach to Incremental Learning for Stock Trend Forecasting（KDD 2023，arXiv 非独占许可，只有摘要）。股票趋势预测的增量学习：数据适配器把增量数据变换到局部平稳的分布，模型适配器给出参数的初始化，两者在把每次增量更新当作一个任务的元学习里一起优化；实验在中证 300 与中证 500 成分股上。关系：A 股上滚动更新应对概念漂移的最直接先例，也是滚动重拟合之外一种更强的对照。官方代码没有许可声明，未收。

`2025_tafas.md`：Kim 等，Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation（AAAI 2025，CC BY-NC-SA 4.0，全文）。时间序列预测的测试时适应：主干冻结，用预测窗口里已经到来的那部分真值，更新输入与输出两端带门控的校准模块。关系：标签只部分到达时怎样更新。代码的许可只允许非商业使用，未收。

`2025_petsa.md`：Medeiros 等，Accurate Parameter-Efficient Test-Time Adaptation for Time Series Forecasting（ICML 2025 研讨会，CC BY 4.0，全文）。只更新输入与输出两端的小校准模块（低秩适配器加动态门），损失含稳健项、频域项与分块结构项。关系：冻结主干、只动很少的参数，降低每次更新的算力与过拟合。代码的许可是 CC BY-NC-SA，未收。

`2026_tta_financial_regime_shifts.md`：Wu 等，Test-Time Adaptation for Non-stationary Time Series: From Synthetic Regime Shifts to Financial Markets（arXiv 预印本，CC BY 4.0，全文）。主干冻结，只更新归一化层的仿射参数：分类用熵与时间一致性，回归用弱增广下的预测方差，可选一个 EMA 教师，另加二次的漂移惩罚与按不确定性触发的回退。合成的渐变漂移上有效；在美股指数 ETF 与汇率的日频数据上，只更新批归一化统计量是稳健的默认，更激进的归一化层适应反而有害。关系：金融序列上的正反两面读数，以及让更新保持稳定的办法。

`2026_delay_aware_tta_audit.md`：Fentazi 等，Training on the Future: A Delay-Aware Audit of Test-Time Adaptation for Time-Series Forecasting（arXiv 预印本，arXiv 非独占许可，只有摘要）。在严格的延迟标签下（预测原点 s 的标签只在 s + d、d ≥ H 时放出）重跑 TAFAS、COSA、PETSA 与 DynaTTA：多数数据集上对冻结模型的显著改进最多半个百分点，每个数据集上都有方法显著更差；漂移大的数据上，标签延迟越长，偏离冻结模型的适配器越有害；不调参的递归最小二乘校正器组以极小的代价胜过多数方法；泄漏的下一步更新把简单适配器的增益夸大到 110 %。关系：本包时点规则的直接对应，以及廉价闭式基线的分量。

`2026_dpal_financial_tta.md`：Lian、Wang、Deng，DPAL: A Test-Time Adaptation Loss for Financial Time Series Forecasting（arXiv 预印本，CC BY 4.0，全文）。金融序列的测试时适应损失：逐点的收益拟合之外，加方向项与由收益重建价格路径的校准项，接在 TAFAS 与 PETSA 上；按成熟标签的协议评估，一个窗口的整个目标段都已观测后才能用于更新。关系：更新目标与最终评价（方向、累计路径）对齐，以及成熟标签的协议。

## 综述

`2023_tta_survey.md`：Liang、He、Tan，A Comprehensive Survey on Test-Time Adaptation under Distribution Shifts（IJCV 2024，arXiv 非独占许可，只有摘要）。按测试数据的形式把测试时适应分成测试时域适应、测试时批适应与在线测试时适应，逐类整理算法与应用。用来查经典方法的谱系。

`2026_self_improving_test_time_survey.md`：Niu 等，A Survey on Self-Improving Test-Time Intelligence: Feedback-Driven Adapting, Learning, and Scaling at Inference（Machine Intelligence Research 已接收，arXiv 非独占许可，只有摘要）。把测试时适应、测试时学习与测试时扩展（多采样、用工具）统一成反馈驱动的测试时智能来综述。用来看这块领域到 2026 年的全景。
