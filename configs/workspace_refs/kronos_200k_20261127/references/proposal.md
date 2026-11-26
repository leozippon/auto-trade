# 所有者的提案：复现近三年论文里可能用于金融市场的方法

所有者的指示，原文一字未改：

For subsequent experiments, please proceed as follows: Search for papers published at ICLR, ICML, and NeurIPS over the past three years on finance, time-series data, and other methods that could potentially be applied to financial markets. For each selected paper, provide the paper itself and its open-source repository (if available) in a research Pack. Let the Agent reproduce and fine-tune the proposed methods, then evaluate whether they have the potential to generate profitable trading strategies.

---

## 术语对照：提案的词在本臂里指什么

上文是所有者的原话。本臂选的论文是 Kronos（AAAI 2026）：它不在提案点名的三个会议里，选它是因为此前三臂换的都是学习器，而它换的是信息（一分钟 K 线作输入）与规模（预训练过的 K 线模型在卡上微调）。每个词的所指如下，`refs/README.md` 的阶段、要求与对照都按这张表读。右列只说明所指，不规定做法。

| 提案里的词 | 本臂里的所指 |
|---|---|
| papers … on finance, time-series data | 本臂的论文：Shi、Fu、Chen、Zhao、Xu、Zhang、Li，Kronos: A Foundation Model for the Language of Financial Markets，AAAI 2026（arXiv 2508.02739） |
| the paper itself | `refs/papers/kronos/paper.pdf`（arXiv v1）、它的文本抽取 `paper.txt` 与 LaTeX 源 `tex/` |
| its open-source repository | `refs/papers/kronos/repo/`：作者仓库在一个固定提交上的快照，图片、抓取实时行情的脚本、网页界面与自带的行情数据已剥离；作者公开的权重在模型库里 |
| the proposed methods | 把 K 线量化成层级离散子词的分词器，加上在多市场 K 线上自回归预训练的只有解码器的 Transformer；本臂用模型库里的 Kronos-mini、small、base 三个检查点 |
| reproduce | 阶段一：用作者代码核对移植，再按论文投资模拟的设定零样本地算出信号，在研究年的 CSI 300 成分上读它的截面秩相关 |
| fine-tune | 阶段二：在本项目研究年上只用当时可见的数据更新分词器与预测器，或者把它冻结、在它的隐状态与预测上训练打分头；输入可以是日线或一分钟 K 线 |
| evaluate | 阶段三：在同一批里与随机初始化的同结构、只读日线的同一模型、冻结的零样本模型和同一批 K 线上的 LightGBM 排序器对照验证，读预测层面与组合层面的差 |
| profitable trading strategies | 20 万元账户、按本项目成本与持仓规则的回放读数，账户扣费后的收益与基准指数、随机对照组合并列；论文的模拟是另一时段、另一种组合与成本 |
