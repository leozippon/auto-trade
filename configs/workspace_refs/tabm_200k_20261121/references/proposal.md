# 所有者的提案：从顶会论文里找可用于金融市场的方法

下面是所有者关于后续实验的指示，原文一字未改。

## 所有者的指示

For subsequent experiments, please proceed as follows: Search for papers published at ICLR, ICML, and NeurIPS over the past three years on finance, time-series data, and other methods that could potentially be applied to financial markets. For each selected paper, provide the paper itself and its open-source repository (if available) in a research Pack. Let the Agent reproduce and fine-tune the proposed methods, then evaluate whether they have the potential to generate profitable trading strategies.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。本臂选的论文是 TabM（ICLR 2025），一个通用的表格深度学习模型；右列说明每个词在本臂里的所指，不规定做法。

| 提案里的词 | 本臂里的所指 |
|---|---|
| papers published at ICLR, ICML, and NeurIPS over the past three years | 本臂的论文：TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling，ICLR 2025 |
| other methods that could potentially be applied to financial markets | TabM 不是金融论文；它是表格监督学习的方法，本项目的打分本身就是逐只股票、逐日的一行特征对应一个标签，所以可以直接套用 |
| the paper itself and its open-source repository | `refs/papers/tabm/` 下的 PDF、文本抽取、LaTeX 源与作者仓库的一个固定提交（`refs/README.md` 的表） |
| reproduce | 阶段一：在论文自己用过的公开数据集（UCI Adult，随包附带）上用作者的包复现论文报告的数，并复现 TabM 与普通 MLP 的差 |
| fine-tune | 阶段二：把 TabM 用策略允许的库重写进策略包，按 A 股的截面排序问题调整输入归一化、损失、批次构成与超参，每个改动写一行理由 |
| the proposed methods | 一个 MLP 里同时训练 k 个共享大部分权重的子模型、推理时取平均（参数高效集成），以及论文的变体 TabM、TabM_mini、TabM_packed 与分段线性特征嵌入 |
| evaluate whether they have the potential to generate profitable trading strategies | 阶段三：TabM 的打分在研究年上逐日截面秩相关与组合读数上，比去掉集成机制的同一网络、比同输入的 LightGBM 排序器多出什么（`refs/README.md` 的对照） |
