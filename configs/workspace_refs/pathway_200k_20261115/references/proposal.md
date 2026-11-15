# 所有者的提案：按奖励强化或抑制推断用到的通路

所有者在「持续学习」这个总题目下写了几条方向。下面是本臂这一条和三个臂共用的领域框架，原文一字未改。

## Direction 2: Overcoming the Lack of Global Optimization in Continual Learning

- **Updating a model's deeper understanding rather than merely its memory:** Record the neural pathways activated during each inference and directly reinforce or suppress those pathways. The reward signal is flexible; it could be surprise, an actual reward, or something else.

## Direction 3: Domain Applications

- **Continual memory adaptation for financial time-series data:** Financial data arrives rapidly and in large volumes, making daily updates challenging. Investigate how memory systems can effectively adapt to the temporal structure and continuous arrival of financial data.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。它写的是一般的模型；本臂把它落到 A 股的截面选股上，每个词的所指如下，`refs/README.md` 的方向与对照都按这张表读。右列只说明所指，不规定做法：机制的实现由会话设计。

| 提案里的词 | 本臂里的所指 |
|---|---|
| model | 截面选股模型：在每个决策日给可交易的股票打分，书买分高的名字；模型的种类不限 |
| deeper understanding rather than merely its memory | 改变模型怎样把输入算成分数（参数与它们组成的计算），而不只是往一个存储里添样本或记录 |
| neural pathways activated during each inference | 一次决策里实际参与计算的那部分网络：被激活的单元与连到它们上的权重 |
| record | 记下某次决策用到的通路。决策在 `generate_orders` 的容器里；更新若在 `fit` 里做，记录要经 `state_dir` 传过去，或在 `fit` 里用当时的输入与参数重算 |
| directly reinforce or suppress | 按奖励的符号与大小改动这些通路上的权重，正的加强、负的抑制，而不是在带标签的样本上重新训练 |
| surprise | 结局与模型预期之差的大小，或模型对自己输入的预测误差 |
| an actual reward | 这次决策的已实现结果：所选名字其后的收益或它的截面秩，或账户的收益；期限走完才可见 |
| something else | 会话自己定义的一个标量 |
| Global Optimization | 在全部可见数据上重新求解的训练，例如按日程从头重训；本臂里它是对照 |
| continual memory adaptation | 模型带着学到的状态逐日走过一场回放，按每天新到期的证据更新，而不是按日程从头重训 |
| arrives rapidly and in large volumes | 研究年里每个交易日新到约 2,500（第一个研究年）到 4,700（最后一个研究年）只股票的日线行，一年约 54 万到 76 万条公告标题 |
| daily updates are challenging | 逐日更新要在每次 `fit` 的时限里做完，而一场研究年回放约 1,701 个决策日，单日开销按天数放大（`refs/README.md` 的「规模与时间」） |
| temporal structure | 交易日的节奏、结局的期限滞后、研究年里市场状态的切换；切换发生在哪些时期，由会话自己从可见的数据里找 |
| memory systems | 任何把学到的东西从一次决策带到下一次的状态：参数、资格记录、缓存的样本、键值记忆 |

## 会话查不到的事实（运营者的转述，不是读数）

- 三因子更新是 `Δw = η · r · e`：e 是推断时记下的资格（eligibility）记录，r 是之后才到的一个标量。取 `e = ∂ŷ/∂w`、`r = y − ŷ`，它正好是在 ½(y − ŷ)² 上走一步梯度下降，所以普通的在线梯度下降是这一族里的一员；其余成员的差别在于什么算资格（例如只算推断时被激活的单元）、奖励是什么（已实现收益的秩、惊奇度、奖励减去它的滑动期望），以及负奖励是否抑制。
- 对概率为 π(a) 的随机选择，`Δθ = η (r − b) ∇θ log π(a)` 是期望奖励梯度的无偏估计，只要基线 b 不依赖于 a；b 取得合适（例如奖励的期望）时降低方差，取得不好也会增大方差。
- 在分段线性单元（例如 ReLU）组成的网络里，一个输入的输出是由哪些单元被激活决定的仿射函数（没有偏置时是线性的），未激活单元进出的权重上梯度为零。所以「一次推断用到的通路」有确切的含义：激活集合与其上的权重。
- 在 t 日做出、期限为 H 个交易日的决策，结局要到 t + H 才知道，所以 t 日的资格记录要留到那时。

上面的式子运营者都在合成数据上用 numpy 核对过（有限差分与精确期望）。会话没有网络；这几段是运营者的转述与核对，不是市场上的测量。
