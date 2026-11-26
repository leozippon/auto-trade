# Kronos：论文的主张、检查点与本项目的差别

这是包作者对论文与代码的摘要，供会话快速定位；数字都标了出处，以 `refs/papers/kronos/paper.pdf`（arXiv v1，2025-08-02）与 `tex/` 为准。AAAI 2026 的正式版没有打开过（出版页拒绝自动访问），与 v1 有没有差别未核实。包作者无法核实的地方标为「未核实」。

## 问题

论文要做一个专门读金融 K 线的基础模型：一个模型读一条 K 线序列（每根是开、高、低、收、成交量、成交额六个数，D = 6），给出其后 H 根（第 2 节），不在目标数据上训练就能用于价格与收益预测、波动率预测和合成 K 线生成。作者的出发点是：通用的时间序列基础模型在金融数据上常常不如不预训练的专门模型（iTransformer 等），而金融序列在它们的预训练语料里占比很小（附录表 3 列了各模型语料里金融数据的比例）。

## 方法与关键部件

- 分词器（第 3 节，图 3，附录 Implementation Details）：一个 Transformer 自编码器，编码器与解码器各 3 层、256 维、前馈 512、4 头，用 Binary Spherical Quantization 把每根 K 线编码成 k = 20 位的二值码，拆成粗、细两个各 10 位的子词（词表各 1,024）。训练目标是只用粗子词重建、用完整码重建两项 MSE 之和加 BSQ 的量化损失（式 2；β = 0.05、γ0 = 1.0、γ = 1.1、ζ = 0.05、λ = 1、分组 5）。输入先逐个字段在窗口内做 z-score，再截在 ±5（附录 Input Preprocessing）。
- 预测器：只有解码器的因果 Transformer，RoPE、Pre-RMSNorm；每一步先由隐状态预测粗子词，再以粗子词的嵌入作查询、隐状态作键值做一次交叉注意力，预测细子词（式 4–8）。训练时细子词的条件用模型自己抽样出的粗子词，不用真值（作者说这减轻了暴露偏差）。每根 K 线另加分、时、星期、日、月五个可学的时间嵌入（附录 Model Architecture）。
- 规模（表 1，附录表 5）：small 8 层、512 维、前馈 1024、8 头、24.7M；base 12 层、832 维、2048、16 头、102.3M；large 18 层、1664 维、3072、32 头、499.2M。最大上下文 512 根。AdamW，余弦学习率，前 15,000 步从峰值的 10% 线性预热；学习率 small 1e-3、base 5e-4、large 2e-4，权重衰减 0.01、0.05、0.10，dropout 随规模减小。论文没有给出预训练的总步数、批大小与耗时；实验环境是三个各 8 张 RTX 4090D 的节点（附录 Experimental Environment）。
- 预训练语料（第 3 节 Dataset，附录 Dataset Details）：逾 120 亿根 K 线、45 个交易所、30 多个国家，股票、加密货币、期货、外汇；正文说 7 种频率，没有列出是哪几种；附录表 4 的清洗参数列了一分钟到周线 12 种，其中有 5 分钟与 30 分钟。清洗去掉价格跳变、长时间无成交与价格停滞的段；对加密货币、期货、外汇加大采样权重。预训练数据到 2024-06（附录 Forecasting Task Setup），起点与逐项清单没有公开。
- 推理（第 3 节 Inference，附录表 6）：像生成文本一样逐根自回归解码，温度与 top-p 采样，多条路径取平均。价格与收益预测 T = 0.6、top-p 0.9、N = 10；波动率 T = 0.9、N = 1；投资模拟 T = 0.6、top-p 0.9、N = 10。第 4.4 节图 7：路径数加多，IC 与 RankIC 单调上升。

论文主张的来源是预训练：分词器加预测器在多市场 K 线上的自回归预训练，然后零样本使用。

## 实验设定与主要数字

测试期从 2024-07 开始（预训练截止之后）。预测任务的测试期在哪天结束，论文没有写；投资模拟的累计收益图（附录图 9）横轴从 2024-07 到 2025-06 之后，曲线约 225 个点，与作者微调示例的回测区间 2024-07-01 到 2025-06-05（`repo/finetune/config.py`）相符，`sources.json` 的 `data_end` 记为 2025-06-05；预测任务的终点未核实。

各任务的读法（附录 Metric Calculation Details）要先认清，它们都不是本项目的逐日截面秩相关：

- 价格序列预测：每个样本（一条序列、一个时点）内，预测的开、高、低、收四条路径与真实路径之间的 IC、RankIC，四个通道平均：量的是一条序列未来路径的形状。
- 收益预测：预测收益 = 预测期末收盘价 / 当前收盘价 − 1，IC 与 RankIC 在同一资产类、同一频率的全部样本上合在一起算，不是按日截面。
- 回看与预测长度按频率定（附录表 8）：5 分钟 480/96、15 分钟 160/32、1 小时 80/12、日线 40/12 等。一分钟线与 30 分钟线都不在预测任务里。
- 测试资产：九个股票交易所（分布内：上海 XSHG、纳斯达克、日本、印度、韩国、香港；分布外：印尼、马来西亚、台湾），币安现货、千余个外汇对。

| 任务（各资产、各频率的平均） | 指标 | Kronos-small / base / large | 最好的从头训练模型 | 最好的通用基础模型 | 出处 |
|---|---|---|---|---|---|
| 价格序列预测 | RankIC | 0.0254 / 0.0258 / 0.0267 | TimesNet 0.0143 | Time-MoE-base 0.0138 | 附录表 14、15 |
| 同上，只看上海 XSHG | RankIC | 0.0375 / 0.0390 / 0.0381 | DLinear 0.0181 | Time-MoE-base 0.0317 | 同上 |
| 收益预测 | RankIC | 0.0622 / 0.0634 / 0.0675 | DLinear 0.0423 | Moirai-base 0.0533 | 附录表 16、17 |
| 同上，只看上海 XSHG | IC / RankIC | 0.0677 / 0.0652 / 0.0662；0.0617 / 0.0653 / 0.0642 | NSTransformer IC 0.0777、RankIC 0.0595 | — | 附录表 16 |
| 已实现波动率预测 | MAE / R² | 0.0384 / 0.0372 / 0.0370；0.2490 / 0.2470 / 0.2624 | — | — | 附录表 18、19 |

摘要的「RankIC 高 93% / 87%」是价格序列预测上 Kronos-large 的 0.0267 对 Time-MoE-base 的 0.0138 与 TimesNet 的 0.0143。上海 XSHG 的收益 IC 上，Kronos 三个规模都不如从头训练的 NSTransformer。从头训练的模型都用 MSE 加 IC 项的损失、批 256、学习率 5e-4、最多 12 个 epoch，各试两组大小（附录 Baselines and Configurations）。

投资模拟（附录 Investment Simulation Setup 与表 10）：A 股日线，Qlib 的原始数据不加处理，在 Qlib 里回测；信号是回看 90 根日线、预测 10 根收盘价的均值相对当前收盘价的涨幅；每天按信号取前 k 只等权多头，每天最多换 n 只，最短持有 5 天；CSI 300 上 k = 50、n = 5，CSI 800 上 k = 200、n = 10；每笔成本 0.15%。年化超额收益（AER）与信息比率（IR）：

| 模型 | CSI 300 AER / IR | CSI 800 AER / IR |
|---|---|---|
| Kronos-small | 0.1805 / 1.2394 | 0.1772 / 1.6050 |
| Kronos-base | 0.1911 / 1.3782 | 0.1867 / 1.6652 |
| Kronos-large | 0.2193 / 1.4177 | 0.1974 / 1.8805 |
| Moment-large（最好的通用基础模型） | 0.1655 / 1.1993 | 0.1707 / 1.5361 |
| PatchTST（从头训练） | 0.1289 / 0.9895 | 0.1620 / 1.5033 |
| TimeXer（从头训练） | 0.1035 / 0.7988 | 0.1509 / 1.5471 |

测试期约 11 个月，其间有 2024-09 下旬的大涨；论文没有给出这些数的误差或显著性。

消融（表 2，Kronos-small 规模）：把离散空间换成连续空间的直接回归（MSE）或 Student-t 混合（NLL），或者让两个子词并行预测，价格 RankIC 从 0.0254 降到 0.0149、0.0102、0.0226，收益 RankIC 从 0.0622 降到 0.0399、0.0329、0.0505。另有词表大小（图 6）、分词器结构（附录表 9）、采样参数（附录图 8）的消融。论文没有「同结构、随机初始化、同样训练量」与预训练版的比较：表 2 的变体与从头训练的基线都不是这个对照。论文也没有微调后的读数，全部 Kronos 读数是零样本。

## 论文没有回答的，以及已知的证据

- 包作者没有找到第三方的复现或独立评测（2026-11-26 检索）；论文的全部读数来自作者。
- 论文里通用基础模型在上海 XSHG 的价格 RankIC 多在 0 附近（附录表 15），与另一些工作的发现一致：通用时间序列基础模型零样本预测日度个股收益接近噪声（例如 Rahimikia 等，arXiv 2511.18578，评的是 TimesFM、Chronos 等，没有评 Kronos，未核实其中是否提到 Kronos）。Kronos 的主张正是专门的金融预训练能改变这一点。
- 收益预测的 IC 是把全部样本合在一起算的，截面上（同一天的股票之间）能排得多好，论文没有报。
- 投资模拟是 Qlib 的大账户、等权 50 或 200 只、按比例计成本，没有整手、最低佣金与 20 万元账户的约束；也没有报残差或风格暴露。
- 预训练截止 2024-06 之前的任何时段，模型都可能见过那段 K 线；论文的测试都在截止之后，本项目的研究年都在截止之前。

## 作者代码里有什么

仓库 `shiyu-coder/Kronos`，提交 `67b630e67f6a`（2026-04-13），MIT。作者 README 写明公开 mini、small、base 三个规模，large 不公开；权重在 Hugging Face 的 NeoQuasar 名下，同为 MIT。

| 位置 | 内容 |
|---|---|
| `model/module.py` | BSQ 量化器、RMSNorm、SwiGLU 前馈、RoPE 自注意力与交叉注意力（都用 `F.scaled_dot_product_attention`）、层级嵌入、依赖感知层、双输出头与损失（两个子词交叉熵的平均）、时间嵌入 |
| `model/kronos.py` | `KronosTokenizer`（`encode(x, half=True)` 给粗、细两个子词的序号，`decode` 还原）、`Kronos`（`forward` 训练用；`decode_s1` 给粗子词的 logits 与隐状态，`decode_s2` 给细子词的 logits）、采样函数、自回归推理循环 `auto_regressive_inference`、`KronosPredictor`（标准化、截断、反标准化与按批预测） |
| `finetune/` | A 股微调示例：`qlib_data_preprocess.py` 用 Qlib 读 CSI 300 日线并切分，`dataset.py` 随机抽 101 根的窗口、在前 90 根上标准化，`train_tokenizer.py` 与 `train_predictor.py` 用 `torchrun` 多卡训练（comet_ml 记日志），`qlib_test.py` 推理出信号并用 Qlib 的 TopkDropout 回测 |
| `finetune_csv/` | 读 CSV 的单资产微调流程与 YAML 配置（配套的 5 分钟数据已剥离） |
| `examples/prediction_example.py` 等三个 | 单条与按批预测的最小示例（读的示例 CSV 不在仓库里） |

剥离的部分：图片；抓取 akshare、baostock 实时行情的脚本与依赖它们的预测、回测脚本；Flask 网页界面与它存下的 2025-08 预测结果；`finetune_csv/data`（港股 5 分钟 K 线，到 2025-09-17）；`tests`（回归测试，它的输入是 2024-06 到 2024-08 的一只 A 股 5 分钟线，晚于研究年）。核对移植改用阶段一的做法：在研究年 K 线上把作者代码与移植并排跑。

依赖与替换：

| 依赖 | 用在哪里 | 镜像里 | 怎样替换 |
|---|---|---|---|
| `torch`、`einops`、`numpy`、`pandas` | 模型与预测器 | 有 | 不需要，策略可以导入 |
| `huggingface_hub`（1.33.0） | `PyTorchModelHubMixin`：`from_pretrained` 读目录里的 `config.json` 与 `model.safetensors` | 有 | 会话的 shell 里可以照用（给本地目录时离线可用，包作者在镜像里断网验证过）；策略不能导入，去掉这个基类，按下节加载 |
| `safetensors` | 权重文件格式 | 有 | 策略不能导入，用 `accelerate` 读（下节） |
| `tqdm`、`sys.path.append` | 进度条、路径 | `tqdm` 有 | 策略里删掉 |
| `qlib`、`comet_ml`、`matplotlib`、`akshare`、`flask`、`plotly` | 数据读取、日志、画图、抓数、网页 | 无 | 不需要：数据从本项目的视图读，单卡训练去掉 `torch.distributed` 与 `torchrun` |

## 怎样加载模型库里的检查点

模型库里五个目录，各有 `config.json`、`model.safetensors`（float32）与 `README.md`，取自 Hugging Face（经镜像）在固定修订上的文件，每个文件按仓库的文件清单核对过 SHA-256 或 git 摘要：

| 目录 | 修订 | `config.json` 的要点 | 参数个数（包作者实测） |
|---|---|---|---|
| `Kronos-mini` | f4e68697d9d5 | 4 层、256 维、前馈 512、4 头，dropout 0.2 | 4,108,032 |
| `Kronos-small` | 901c26c13326 | 8 层、512 维、1024、8 头，dropout 0.25/0.1 | 24,741,376 |
| `Kronos-base` | 2b554741eca4 | 12 层、832 维、2048、16 头，dropout 0.2 | 102,310,592 |
| `Kronos-Tokenizer-2k` | 26966d003506 | 编码、解码各 3 个块、256 维，粗细各 10 位，分组 5 | 3,958,042 |
| `Kronos-Tokenizer-base` | 0e0117387f39 | 同上，分组 4 | 3,958,042 |

mini 配 Tokenizer-2k，作者 README 说上下文 2,048 根；small 与 base 配 Tokenizer-base，上下文 512 根。权重在 2025-06-30（small、base 与 Tokenizer-base）与 2025-07-01（mini 与 Tokenizer-2k）上传，之后的提交按标题只改 README。

会话的 shell 里（`PYTHONPATH=refs/papers/kronos/repo`）：`from model import Kronos, KronosTokenizer, KronosPredictor`，`Kronos.from_pretrained("/mnt/pretrained/Kronos-small")` 与 `KronosTokenizer.from_pretrained("/mnt/pretrained/Kronos-Tokenizer-base")`。包作者在沙箱镜像里断网、只用 CPU 验证过：五个目录都能这样加载；small 与 base 对 4 只股票各 90 根日线贪心解 2 步、mini 对 4 只各 960 根 K 线（包作者手里的一分钟线；会话里没有一分钟线，只有 5 或 30 分钟线）贪心解 2 步，输出有限，同样输入两遍逐位相同；`decode_s1` 的第二个输出是隐状态，形状（序列，长度，维数），取最后一个位置即最后一根 K 线处的表示。

策略里：把 `model/module.py` 与 `model/kronos.py` 复制进策略包，去掉 `PyTorchModelHubMixin` 基类、`tqdm` 与 `sys.path` 那两行，按目录里 `config.json` 的数构造模块，再读权重。包作者验证过两种读法与 `from_pretrained` 得到的张量逐个相等，镜像里都在：

```python
from accelerate.utils import load_checkpoint_in_model
model = Kronos(s1_bits=10, s2_bits=10, n_layers=8, d_model=512, n_heads=8, ff_dim=1024, ffn_dropout_p=0.25,
               attn_dropout_p=0.1, resid_dropout_p=0.25, token_dropout_p=0.1, learn_te=True)
load_checkpoint_in_model(model, context.pretrained_dir + "/Kronos-small/model.safetensors")
# 或：from transformers.modeling_utils import load_state_dict; model.load_state_dict(load_state_dict(path), strict=True)
```

用法与要注意的地方（都来自 `model/kronos.py`）：

- `KronosPredictor.predict` 对每条序列在整个输入窗口上逐字段 z-score（标准差加 1e-5），截在 ±5，预测完再反变换；缺成交量与成交额时补零，缺成交额时用成交量乘均价。时间特征是 `timestamps` 的分、时、星期、日、月；日内 K 线的时间取它的 `trade_time`（这一根的结束时刻）。窗口里的价格要先复权（README 阶段二），否则除权日会成为窗口内的一次跳变。
- 窗口长于 `max_context` 时只留最后 `max_context` 根。
- 解码循环每生成一根，就把整个窗口（最多 `max_context` 根）重新送进模型，没有键值缓存；`sample_count` 条路径是把批复制这么多份。抽样用 `torch.multinomial`；`top_k=1` 是贪心。
- 训练时的 `forward`（`use_teacher_forcing=False`，作者微调脚本的用法）同样对粗子词做多项式抽样，训练要复现就得固定 torch 的随机源；`model.train()` 下各处 dropout 生效。
- 训练出的权重写在 `context.state_dir`，不写进模型库。

## 代价

下面按矩阵乘法的 FLOP 估算，没有在卡上实测，先用冒烟量。每处理一个位置，small 约 4,700 万 FLOP、base 约 2 亿、mini 约 700 万（2,048 根的满上下文时注意力再加约 800 万）；全市场按约 4,700 个名字算：

| 用法 | 每个决策日 | 一场研究年（约 1,701 天） |
|---|---|---|
| 一次前向取隐状态，日线 90 根 | small 约 2e13，base 约 8.5e13 | small 约 3.4e16，base 约 1.4e17 |
| 贪心解 3 根，日线 90 根 | small 约 6e13，base 约 2.6e14 | 约三倍于上一行 |
| 论文的抽样解码，10 条路径 × 10 根 | small 约 2.1e15，base 约 8.9e15 | small 约 3.6e18，base 约 1.5e19 |
| small 或 base 一次前向，日内 K 线 512 根（5 分钟约 10.7 个交易日，30 分钟 64 个） | small 约 1.3e14，base 约 5.3e14 | small 约 2.3e17，base 约 9e17 |
| mini 一次前向，日内 K 线 2,048 根（5 分钟约 42.7 个交易日，30 分钟 256 个） | 约 1.4e14 | 约 2.4e17 |

作个量级换算：按 A100 bf16 峰值的三成（约 1e14 FLOP/s），论文的抽样解码在 base 上每天约 90 秒、一场研究年约 40 小时，fp32 下约慢十倍；一次前向取隐状态每天不到一秒。作者微调示例的训练量（300 万个 101 根的窗口，前向加反向约三倍前向）在 small 上约 4e16、base 上约 2e17 FLOP，按同样的换算是几分钟到半小时。显存：base 的权重 0.4 GB，AdamW 状态再加约 1.2 GB，激活按批大小增长，批大小按冒烟时的峰值定。

## 与本项目问题的差别，以及它意味着什么

| | 论文 | 本项目 |
|---|---|---|
| 目标 | 一条序列未来的价格路径（序列内的相关）；收益预测的相关在全部样本上合算 | 同一天整个截面上其后若干天收益的排序 |
| 输入 | 单条 K 线序列，看不到其他股票与财务 | 价量、财务、指数、文本都可见；Kronos 只读单条序列，截面信息只能进打分头或对照 |
| 时段 | 预训练截止之后的约一年 | 研究年整个落在预训练的时间范围内 |
| 使用方式 | 零样本 | 每个决策只能用当时可见的数据训练与选择 |
| 组合 | Qlib 大账户、等权 50 或 200 只、0.15% 成本 | 20 万元、整手、最低佣金、按本项目成本与持仓规则 |

意味着：直接拿零样本的价格预测排序，读数会受预训练见过研究年的影响而偏高，与论文的读数也不可比；更有希望的是把它当作 K 线编码器，在本项目自己的截面标签上学打分头，再看微调与日内 K 线能不能加东西。研究年上任何带预训练权重的读数都是上界，2024-07 以后的验证期才是干净的检验。
