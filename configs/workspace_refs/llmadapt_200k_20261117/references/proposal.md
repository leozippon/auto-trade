# 所有者的提案：读标题、持续学习的语言模型

所有者在「持续学习」这个总题目下写了几条方向。下面是与本臂有关的两条，以及所有者 2026-11-16 关于算力的指示，原文一字未改。

## Direction 1: Learning from Humans

- **Learning and amplifying model biases:** Explore Self-Enhance (amplifying the model's inherent biases) combined with parallel continual learning. Allow some models to deteriorate during learning, then select those that improve. The specific learning method is not important; it could involve Test-Time Training (TTT), full-parameter fine-tuning, or other approaches.

## Direction 3: Domain Applications

- **Continual memory adaptation for financial time-series data:** Financial data arrives rapidly and in large volumes, making daily updates challenging. Investigate how memory systems can effectively adapt to the temporal structure and continuous arrival of financial data.

## 2026-11-16 的指示（标题是运营者加的）

The current strategies are still too simplistic and do not make sufficient use of GPU resources. Please consider adding guidance to the System Prompt or research Packs to encourage more extensive use of GPU compute, or even explicitly specify the use of an LLM.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。它写的是一般的模型；本臂把它落到 A 股的截面选股上，每个词的所指如下，`refs/README.md` 的方向、算力与对照都按这张表读。右列只说明所指，不规定做法：模型、学习方式与更新节奏由会话设计。

| 提案里的词 | 本臂里的所指 |
|---|---|
| LLM | 模型库里的语言模型或嵌入模型（清单、发布月份与加载写法在合同 README 的 Pretrained models），在回放里读公告标题 |
| more extensive use of GPU compute | 语言模型在回放里、在卡上读标题，持续学习的版本还在卡上更新（`refs/README.md` 的「算力」） |
| the model's inherent biases | 预训练语言模型读一条标题时已有的理解：它给出的表示或判断，来自预训练，没有经过本实验的结局 |
| Self-Enhance (amplifying …) | 在模型自己的输出上继续训练，让它朝已有的倾向走得更远；本臂不以它为方向，会话可以把它当作一种学习方式 |
| parallel continual learning / select those that improve | 同一场回放里并行更新的几份模型，按已到期的结局留下变好的；本臂不要求副本与选择 |
| Test-Time Training (TTT) | 在决策时用当时可见的数据更新模型的一部分：无标签的目标（例如在新到的标题上做语言建模），或已经到期的标签 |
| full-parameter fine-tuning | 更新语言模型的全部参数；与之相对的是适配器（例如 LoRA）或只训练顶层。训练出的权重写在 `context.state_dir`，不写进模型库 |
| The specific learning method is not important | 学习方式由会话选；本臂问的是持续学习比冻结多出什么 |
| 冻结、只作编码器（运营者的词） | 语言模型的权重在整场回放里不变，只用它的输出作表示；它上面学到的部分照常更新 |
| continual memory adaptation | 模型带着学到的状态逐日走过一场回放，按每天新到的标题与新到期的结局更新，而不是按日程从头重训 |
| arrives rapidly and in large volumes | 研究年里每个交易日新到约 2,500（第一个研究年）到 4,700（最后一个研究年）只股票的日线行，一年约 54 万到 76 万条公告标题 |
| daily updates are challenging | 每天都有新标题与新到期的结局要学，一场研究年回放约 1,701 个决策日；每次 `fit` 有时限（运行事实 `budgets`） |
| temporal structure | 标题的可见时点、结局的期限滞后、研究年里市场状态的切换；切换发生在哪些时期，由会话自己从可见的数据里找 |
| memory systems | 任何把学到的东西从一次决策带到下一次的状态：语言模型的适配器或权重、打分头、优化器状态、缓存的标题表示 |

## 会话查不到的事实（运营者的转述，不是读数）

- 此前用语言模型的实验都把它冻结着。20261019 轮两个提案的四个实验，冻结的小语言模型（现已不在模型库里）读匿名化的标题，只更新上面约 3.6 万参数的头；复盘记下「没测的有适配器或 LoRA」。20261022 轮的检索用冻结的嵌入模型给标题向量。没有一场训练过适配器或语言模型本身。
- 在 20261019 轮里，去掉文本让逐日更新的头明显变差，按季冷重训的头几乎不变；两个实验各只在一个种子基上做了这一对，它们的窗口越过本轮研究年，这里不写数字。登记册的读法：「标题只在逐日更新的头上起作用」。
- 关键词计数与冻结嵌入在研究年上读不出增量。20261113b 轮一场会话（研究年与本轮相同）把标题的正则事件计数、以及冻结的 Qwen3-Embedding-0.6B 嵌入加进树排序器，研究年读数与同批不读文本的树相当，差别在换种子的散布之内。20261025b 到 20261027c 的六个检索实验把文本做进键、类别或特征，没有一条读出文本的增量；20261115 轮的键值臂加上哈希双字与事件计数的标题特征，两个种子基都比不加的低（`refs/references/families.md`）。
- 很多股票在一个决策日或一个月里没有标题。20261026 的一场检索会话给没有标题的股票零向量，它们占了组合前排的大半，整个组合其实是一个无消息的低波动价量筛；登记册的教训是「文本检索的组合要先量「没有文本」的那一群占了多少」。
- 一条 D 日标题最早用于 D+2 的决策（`refs/README.md` 的「本臂的数据与账户」）；以那天的打分为起点、期限 h 个交易日的结局，要等期限末那天的价格进入决策的视图之后，才能用来更新模型。
