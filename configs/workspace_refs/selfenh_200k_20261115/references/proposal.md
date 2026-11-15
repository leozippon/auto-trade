# 所有者的提案：放大模型偏好的并行持续学习

所有者在「持续学习」这个总题目下写了几条方向。下面是本臂这一条和三个臂共用的领域框架，原文一字未改。

## Direction 1: Learning from Humans

- **Learning and amplifying model biases:** Explore Self-Enhance (amplifying the model's inherent biases) combined with parallel continual learning. Allow some models to deteriorate during learning, then select those that improve. The specific learning method is not important; it could involve Test-Time Training (TTT), full-parameter fine-tuning, or other approaches.

## Direction 3: Domain Applications

- **Continual memory adaptation for financial time-series data:** Financial data arrives rapidly and in large volumes, making daily updates challenging. Investigate how memory systems can effectively adapt to the temporal structure and continuous arrival of financial data.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。它写的是一般的模型；本臂把它落到 A 股的截面选股上，每个词的所指如下，`refs/README.md` 的方向与对照都按这张表读。右列只说明所指，不规定做法：机制的实现由会话设计。

| 提案里的词 | 本臂里的所指 |
|---|---|
| model | 截面选股模型：在每个决策日给可交易的股票打分，书买分高的名字；模型的种类不限 |
| Self-Enhance / the model's inherent biases | 一个副本自己的偏好：它在打分上已经倾向的方向（偏爱的特征、风格或名字）；「放大」指让副本朝这个方向走得更远，例如在它自己的输出上继续训练 |
| parallel continual learning | 同一场回放里并行的多个副本，各自持续更新；它们活在 `fit` 容器的内存与 `state_dir` 里。宿主层面没有副本群：每个 `validate` 候选是一场独立的回放，看不到别的候选 |
| deteriorate / improve | 按已经到期的结局，在一段窗口上读出的表现变差或变好；结局要等期限走完才可见 |
| select those that improve | 在回放里按已到期的结局淘汰、复制或加权副本；选择规则是策略自己的 |
| Test-Time Training (TTT) | 在决策时用当时可见的数据更新模型的一部分：无标签的自监督目标，或已经到期的标签 |
| full-parameter fine-tuning | 更新模型的全部参数；预训练模型的副本可以在回放里微调（合同 README 的 Pretrained models） |
| Learning from Humans | 本臂里没有人的反馈；模型之外的证据只有市场给出的已实现结局 |
| continual memory adaptation | 模型带着学到的状态逐日走过一场回放，按每天新到期的证据更新，而不是按日程从头重训 |
| arrives rapidly and in large volumes | 研究年里每个交易日新到约 2,500（第一个研究年）到 4,700（最后一个研究年）只股票的日线行，一年约 54 万到 76 万条公告标题 |
| daily updates are challenging | 逐日更新要在每次 `fit` 的时限里做完，而一场研究年回放约 1,701 个决策日，单日开销按天数放大（`refs/README.md` 的「规模与时间」） |
| temporal structure | 交易日的节奏、结局的期限滞后、研究年里市场状态的切换；切换发生在哪些时期，由会话自己从可见的数据里找 |
| memory systems | 任何把学到的东西从一次决策带到下一次的状态：参数、副本群、缓存的样本、键值记忆 |

## 会话查不到的事实（运营者的转述，不是读数）

- 让模型在自己置信的输出上训练，会让它朝已经倾向的方向走得更远；没有外部证据时，它的错误和正确的倾向一样被加强。本臂的外部证据是已实现的结局，它只经由选择、或经由已经到期的标签到达策略。运营者的核对：一个部分偏错的线性分类器在自己最有把握的一半样本上反复训练五轮，权重范数从 3.5 涨到 7.1，对真值的准确率从 0.715 落到 0.707，它在自己错判的样本上的平均置信度从 0.79 升到 0.88。
- 在 N 个有噪声的副本里挑最好的，会高估胜者：若各副本在评估窗口上的得分相互独立、均值相同，最好的那个平均高出「N 个标准正态变量的最大值的期望」个标准误：N = 4 时 1.03，N = 10 时 1.54，N = 16 时 1.77，N = 30 时 2.04（数值积分）。选择规则只有超出这个量才算挣到东西。
- 把最差的副本换成最好的副本的拷贝、再让它们重新分化，是 population-based training 的做法；副本只重新播种、彼此从不相互作用的一群是 bagging 的集成，它是对照。
- 每个副本都有自己的更新时间与内存；N 个副本每天的更新成本是一个副本的 N 倍，除非在卡上成批计算。

会话没有网络；上面这几段是运营者的转述与在合成数据上的核对，不是市场上的测量。
