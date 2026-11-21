# 所有者的提案：复现近三年顶会论文里可能用于金融市场的方法

所有者的指示，原文一字未改：

For subsequent experiments, please proceed as follows: Search for papers published at ICLR, ICML, and NeurIPS over the past three years on finance, time-series data, and other methods that could potentially be applied to financial markets. For each selected paper, provide the paper itself and its open-source repository (if available) in a research Pack. Let the Agent reproduce and fine-tune the proposed methods, then evaluate whether they have the potential to generate profitable trading strategies.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。本臂选的论文是 TimesFM（ICML 2024）；每个词的所指如下，`refs/README.md` 的阶段、要求与对照都按这张表读。右列只说明所指，不规定做法。

| 提案里的词 | 本臂里的所指 |
|---|---|
| papers … on time-series data | 本臂的论文：Das、Kong、Sen、Zhou，A decoder-only foundation model for time-series forecasting，ICML 2024（PMLR 235） |
| the paper itself | `refs/papers/timesfm/paper.pdf`（PMLR 版）、它的文本抽取 `paper.txt` 与 arXiv v4 的 LaTeX 源 `tex/` |
| its open-source repository | `refs/papers/timesfm/repo/`：作者仓库在一个固定提交上的快照，图片、3.0 的代码与面向代理的安装说明已剥离 |
| the proposed methods | 分块输入、只有解码器的 Transformer，在大规模序列语料上预训练后零样本预测；本臂用的是模型库里的 TimesFM 2.5 检查点，它与论文里的 1.0 的差别在 `refs/references/claims.md` |
| reproduce | 阶段一：用 2.5 检查点按论文的 ETT 协议做零样本预测，与论文表 5 的数对照 |
| fine-tune | 阶段二：在本项目研究年上只用当时可见的数据更新这个检查点（全部参数、LoRA 适配器或新的输出头），或者把它冻结、在它的输出上训练打分头 |
| evaluate | 阶段三：在同一批里与去掉预训练权重的同结构、冻结的零样本模型和 LightGBM 排序器对照验证，读预测层面与组合层面的差 |
| profitable trading strategies | 20 万元账户、按本项目成本与持仓规则的回放读数；论文与引用的金融研究给的都是不计成本的数字 |
