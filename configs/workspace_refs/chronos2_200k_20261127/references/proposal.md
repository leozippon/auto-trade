# 所有者的提案：复现近三年论文里可能用于金融市场的方法

所有者的指示，原文一字未改：

For subsequent experiments, please proceed as follows: Search for papers published at ICLR, ICML, and NeurIPS over the past three years on finance, time-series data, and other methods that could potentially be applied to financial markets. For each selected paper, provide the paper itself and its open-source repository (if available) in a research Pack. Let the Agent reproduce and fine-tune the proposed methods, then evaluate whether they have the potential to generate profitable trading strategies.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。本臂选的论文是 Chronos-2（Amazon 的技术报告，arXiv 2510.15821）：它不在提案点名的三个会议里，也没有金融实验；选它是因为它的特有部件是信息的通道，一组序列之间的注意力与协变量，正好能在同一批里对照「同组股票与外生序列经一个预训练的预测器传进来的信息」与「没有这些信息的同一个模型」。每个词的所指如下，`refs/README.md` 的阶段、要求与对照都按这张表读。右列只说明所指，不规定做法。

| 提案里的词 | 本臂里的所指 |
|---|---|
| papers … on time-series data, and other methods that could potentially be applied to financial markets | 本臂的论文：Ansari、Shchur、Küken、Auer 等 23 位作者，Chronos-2: From Univariate to Universal Forecasting，Amazon 的技术报告（arXiv 2510.15821，2025-10） |
| the paper itself | `refs/papers/chronos2/paper.pdf`（arXiv v1）、它的文本抽取 `paper.txt` 与 LaTeX 源 `tex/` |
| its open-source repository | `refs/papers/chronos2/repo/`：作者仓库在一个固定提交上的快照，CI 配置、云部署笔记本与测试用的小检查点已剥离；作者公开的权重在模型库里 |
| the proposed methods | 一个只有编码器的 Transformer，每层先沿时间、再在一组序列之间做注意力；一组可以是相关的几条序列，也可以是目标连同它只有过去值的协变量与未来值已知的协变量；一次前向给出 21 个分位的多步预测；本臂用模型库里的 `chronos-2` 检查点 |
| reproduce | 阶段一：用作者代码核对移植，再在包里带的 fev-bench 任务上复现带协变量与分组的预测，并与同一模型逐条单独、不用协变量的预测比 |
| fine-tune | 阶段二：在本项目研究年上只用当时可见的数据微调它（全部参数或 LoRA），或者把它冻结、在它的分位预测与编码器状态上训练打分头；分组由同行业或同涨同跌的股票组成，协变量取决策时可见的指数、行业、换手、事件与文本计数 |
| evaluate | 阶段三：在同一批里与关掉分组、去掉协变量的同一模型，冻结的零样本模型，随机初始化的同结构，以及同样输入上的 LightGBM 排序器对照验证，读预测层面与组合层面的差 |
| profitable trading strategies | 20 万元账户、按本项目成本与持仓规则的回放读数，账户扣费后的收益与基准指数、随机对照组合并列；论文没有金融任务，也没有组合 |
