# Chronos-2：论文的主张、检查点与本项目的差别

这是包作者对论文与代码的摘要，供会话快速定位；数字都标了出处，以 `refs/papers/chronos2/paper.pdf`（arXiv v1，2025-10-17）与 `tex/` 为准。论文是 Amazon 的技术报告，没有经过会议评审。包作者无法核实的地方标为「未核实」。

## 问题

论文要做一个能处理各种预测任务的预训练模型：单变量、多变量（几条一起变化的序列同时预测）与带协变量（外生变量，只有过去值的或未来值事先已知的，数值或类别）的预测，都不在目标数据上训练（零样本）。作者的出发点：此前的预训练模型多半只读单条序列，少数支持多变量或协变量的（Moirai-1、Toto、COSMIC、TabPFN-TS）各有缺口，经验上也比单变量模型好得不多（第 2 节、表 1）。

## 方法与关键部件

- 输入（第 3.1 节）：目标与协变量的历史拼在一起，未来部分只在已知协变量的位置有值，其余记为缺失。类别协变量先变成数：单个目标时用目标编码，多个目标时用序号编码（作者包的预处理里做，模型本身只收数）。
- 缩放：每一维用自己历史的均值与标准差标准化（缺失值不计），再取反双曲正弦（式 1、2），预测时反变换（式 5）。所以模型看不到一条序列的水平，只看到形状。
- 元特征与分块：每一维附上时间索引与掩码（观测到为 1），缺失值置零；历史与未来分别切成长 16 的不重叠块，块与它的时间索引、掩码拼起来经一个残差网络映成隐向量；历史与未来之间插一个 REG 记号，兼作分隔与注意力汇（第 3.1 节）。
- 结构（第 3.2 节）：只有编码器的 Transformer，按 T5 编码器的设计。每层先做时间注意力（同一维的各块之间，RoPE 位置编码），再做组注意力（同一个块的位置上，同组的各条序列之间；组内没有顺序，所以没有位置编码），再接前馈。组由每条序列的组号指定，层内把组号变成掩码，信息不跨组。
- 输出：未来各块的表示经一个残差块，直接给出多步、21 个分位（0.01、0.05、0.1、0.15 … 0.9、0.95、0.99）的预测，不逐步自回归。
- 训练（第 3.3 节）：分位损失（式 4），只在目标维上算，已知协变量与缺失的目标不计；一批里混着单变量、多变量与带已知协变量的任务，每个任务一个组号；输出块数逐批随机。分两段：先在 2,048 点的上下文、较少的输出块上预训练，再把上下文扩到 8,192 点、加多输出块。论文没有给步数、批大小、优化器、硬件与耗时，也没有写预训练时组的大小上限（作者包的说明只说交叉学习时一批约 100 条与技术报告一致）。
- 推理（第 3.4 节、表 2）：单变量时每条序列一个组号；多变量时同一条序列的各维同组；带协变量时目标、只有过去值的与已知的协变量同组，协变量那几行的预测不用。另有「完全交叉学习」：整批同一个组号，信息在整批里共享，主表的读数都用这个模式（第 5.2 节）。

论文主张的来源是组注意力，加上用合成数据教会它的多变量与协变量的用法：作者说模型超出单变量的本事全部来自合成数据（第 4.2 节、第 6 节）。

## 训练数据与发布时间

- 真实数据（第 4.1 节、附录表 6）：取自第一代 Chronos 与 GIFT-Eval 预训练语料的 23 个单变量数据集，领域是能源、自然、交通、网页、云运维，另有 M4 的日、时、月、周四种频率。按名称没有行情序列。M4 的序列按 M4 原文分六个领域，其中有金融一类（日度 4,227 条里约 1,559 条），匿名、来源与时间不明（据 M4 的论文，包作者没有在本包里核实）。作者说语料与 GIFT-Eval 各任务的测试部分不重叠，但与它一些数据集的训练部分有重叠（第 5.1 节）；fev-bench 与 Chronos Benchmark II 的数据集都没有进训练。
- 合成数据（第 4 节）：单变量用 TSI（趋势、季节与不规则成分的组合）与 TCM（随机因果图上的自回归）两种生成器；多变量与协变量任务全部来自合成数据，由「多变量化器」从 AR、ETS、TSI 与 KernelSynth 抽出几条单变量序列，再加同一时刻的线性或非线性变换（同期相关），或跨时刻的依赖（领先滞后、协整），随机指定其中一部分为已知协变量。
- 截止日期：论文没有写训练数据的截止日期。
- 发布：arXiv v1 2025-10-17；作者仓库的新闻一栏写 2025-10-20 发布；Hugging Face 上 `amazon/chronos-2` 的权重与配置在 2025-10-30 上传（仓库的文件清单）。都晚于全部研究年与验证期。

## 实验设定与主要数字

技能分 = 1 减去「相对季节朴素预测的误差比」的几何平均，乘 100；胜率是两两比较里赢的比例（第 5.1 节，沿用 fev-bench 的定义）。

| 基准 | 指标 | Chronos-2 | 次好的模型 | 出处 |
|---|---|---|---|---|
| fev-bench（100 个任务，含多变量与协变量） | SQL 技能分 / 胜率 | 47.3 / 90.7 | TiRex 42.6 / 80.8，TimesFM-2.5 42.3 / 75.9 | 表 3 |
| GIFT-Eval（97 个任务） | WQL 技能分；MASE 技能分 | 51.4；30.2 | TimesFM-2.5 51.0；29.5 | 表 4 |
| Chronos Benchmark II（27 个任务，多为短序列） | WQL 技能分；MASE 技能分 | 46.6；26.5 | TimesFM-2.5 42.4；23.3 | 表 5 |

fev-bench 上 Chronos-2 对 TiRex 的两两技能分差 8.2，95% 自助法区间 5.6 到 11.2（图 2）；这是论文里唯一的误差范围。

同一个模型的单变量模式（逐条单独、忽略协变量）对交叉学习模式，技能分（图 3、图 4、图 5）：

| 子集 | 单变量模式 | 交叉学习 | 次好的模型 |
|---|---|---|---|
| fev-bench 单变量子集（32 个任务） | 36.0 | 37.0 | TiRex 35.0 |
| GIFT-Eval | 50.9 | 51.4 | TimesFM-2.5 51.0 |
| Chronos Benchmark II | 44.0 | 46.6 | TimesFM-2.5 42.4 |
| fev-bench 多变量子集（26 个） | 57.6 | 57.9 | Toto-1.0 57.1 |
| fev-bench 协变量子集（42 个） | 39.9 | 47.0 | TabPFN-TS 40.0 |
| 能源，带动态协变量（16 个，SQL） | 40.9 | 51.3 | TabPFN-TS 46.5 |
| 零售，带动态协变量（17 个，WQL） | 42.1 | 48.6 | TabPFN-TS 44.3 |

多变量的增益小，作者借 Takens 嵌入定理解释：历史够长时，单变量模型能抓住大部分同样的结构（第 5.2 节）。协变量的增益最大；用来说明的两个例子都是未来已知的协变量（图 6 德国电价的日前负荷与风光发电预测，图 7 Rossmann 的促销与假日），论文没有把只有过去值的与未来已知的协变量的增益分开报。

消融（图 8，技能分，依次是 fev-bench、GIFT-Eval、Chronos Benchmark II）：主模型（1.2 亿参数）47.3、51.4、46.6；2,800 万参数的小模型 45.3、50.4、44.1；只用合成数据预训练的 Chronos-2-Synth 45.9、50.4、46.4；只做了 2,048 点那一段的 Chronos-2-2K 46.9、50.1、45.8。论文没有「同结构、随机初始化、同样训练量」与预训练版的比较，也没有微调后的读数：全部读数是零样本。

吞吐（第 1 节脚注）：一张 A10G 上每秒约 300 条序列（一批 1,024 条、上下文 2,048 点、预测 64 步）。

## 论文没有回答的，以及已知的证据

- 论文没有金融实验。fev-bench 的 100 个任务里有几个宏观经济数据集（FRED-MD、FRED-QD、GVAR，按 fev 的任务清单），论文没有单独报；没有个股或逐日行情的任务。
- 论文的读法都是预测误差（分位损失、MASE），没有截面排序或组合的读法。
- 包作者没有检索 Chronos-2 的第三方复现或它在金融数据上的评测。另有工作在全球市场的日度超额收益上评测通用时间序列基础模型，发现现成的模型零样本表现普遍很差，在金融数据上预训练才有改善（Rahimikia、Ni、Wang，Re(Visiting) Time Series Foundation Models in Finance，arXiv 2511.18578，据其摘要）；其中评了哪些模型、是否包括 Chronos-2，未核实。
- 交叉学习的结果随一批的大小变，批太大会偏离预训练时组的大小（作者包的说明）；股票的组该多大、按什么分，论文没有可参照的设定。

## 作者代码里有什么

仓库 `amazon-science/chronos-forecasting`，提交 `1224fafa91e1`（2026-10-05），Apache-2.0，包版本 2.3.3.dev0。

| 位置 | 内容 |
|---|---|
| `src/chronos/chronos2/config.py` | `Chronos2CoreConfig`（`transformers` 的 `PretrainedConfig`）与 `Chronos2ForecastingConfig`（上下文、块长、分位、REG、反双曲正弦、输出块数） |
| `src/chronos/chronos2/layers.py` | RoPE、层归一化、前馈、多头注意力（eager 或 `scaled_dot_product_attention`）、时间注意力、组注意力（把隐状态转成「时间 × 批」后在批的维上做注意力）、残差块 |
| `src/chronos/chronos2/model.py` | `Chronos2Model`（`PreTrainedModel`）：缩放、分块、组掩码、`encode`、`forward`（给 `quantile_preds`，传 `future_target` 时另给 `loss`）、初始化 |
| `src/chronos/chronos2/pipeline.py` | `Chronos2Pipeline`：`from_pretrained`、`predict`、`predict_quantiles`、`predict_df`、`embed`、`fit`（微调）、`predict_fev`（fev 任务的评测入口） |
| `src/chronos/chronos2/dataset.py`、`preprocess.py`、`trainer.py` | 把各种输入排成批（目标、只有过去值的协变量、已知协变量的次序）、类别编码、`transformers.Trainer` 的子类 |
| `src/chronos/chronos_bolt.py`、`chronos.py`、`base.py` | 前两代模型；`InstanceNorm` 与 `Patch` 在 `chronos_bolt.py` 里 |
| `scripts/` | 第一代 Chronos（T5）的训练与评测脚本、KernelSynth 生成器（依赖 `gluonts`、`typer`，镜像里没有）；没有 Chronos-2 的预训练代码，也没有它的 TSI、TCM 与多变量化器 |
| `notebooks/chronos-2-quickstart.ipynb` | 快速上手：单变量、协变量、交叉学习、numpy 接口、全参数与 LoRA 微调（读的数据在 S3 上，会话里取不到） |
| `test/` | 单元测试，展示各种输入的写法（用到的小检查点已剥离，不能直接跑） |

剥离的部分：`.github` 的 CI 配置；SageMaker 部署笔记本；`test/` 下三个测试用的小模型目录。

依赖与替换（`pyproject.toml`）：

| 依赖 | 用在哪里 | 镜像里 | 怎样替换 |
|---|---|---|---|
| `torch`、`transformers`（5.15.0）、`accelerate`、`einops`、`numpy`、`pandas` | 模型与管线 | 有 | 会话的 shell 里作者的包可以照用；策略只能用副本（下节） |
| `copy`、`packaging`、`contextlib`、`logging`、`warnings`、`pathlib`、`time` | 深拷贝、版本判断、回退、日志、管线 | 有，但策略不能导入 | 模型只用到前三个，副本里删掉（下节）；管线不进策略 |
| `peft`（0.20.0） | LoRA 微调 | 有 | 策略可以导入 |
| `fev`、`datasets`、`boto3`、`gluonts`、`matplotlib` | 基准评测、数据读取、S3、第一代训练、画图 | 无 | 不需要：阶段一的任务与 SQL 按下文自己写 |

## 怎样加载模型库里的检查点

模型库里的 `chronos-2` 目录有 `config.json`、`model.safetensors`（float32，477,930,472 字节）与 `README.md`，取自 Hugging Face（经镜像）`amazon/chronos-2` 的修订 29ec3766d36d，每个文件按仓库的文件清单核对过 SHA-256 或 git 摘要。权重与配置在 2025-10-30 上传，之后的提交按标题只改 README。`config.json` 的要点：12 层、768 维、12 个头、前馈 3,072、dropout 0.1；上下文 8,192、块长 16、最多 64 个输出块（1,024 步）、21 个分位、带 REG、用反双曲正弦。参数个数 119,477,664（包作者实测）。

会话的 shell 里（`PYTHONPATH=refs/papers/chronos2/repo/src`）：`from chronos import Chronos2Pipeline`，`pipeline = Chronos2Pipeline.from_pretrained("/mnt/pretrained/chronos-2", device_map="cuda")`（没有卡时用 `"cpu"`）。包作者在沙箱镜像里断网、只用 CPU 验证过：作者的载入与策略副本的载入都不到 0.5 s，每个张量相等；在 4 只股票 2021 年 100 个交易日的对数收盘价上，副本的直接前向与 `pipeline.predict` 在三种输入上逐位相同（各自成组；四只同组；一个目标带一个只有过去值的协变量与一个已知协变量）；同样输入两遍前向逐位相同；`Chronos2Model(config)` 的随机初始化与检查点不同。

策略里：把作者的 `chronos2/config.py`、`chronos2/layers.py`、`chronos2/model.py` 复制进策略包（例如 `lib/chronos2/`），把 `chronos_bolt.py` 里的 `Patch` 与 `InstanceNorm` 两个类原样复制成一个文件，再改下面几处。包作者照此做过，副本通过了策略的静态检查：

- `layers.py`：`maybe_autocast` 的 `try` 导入连同它用 `contextlib` 写的回退，换成直接 `from transformers.utils.generic import maybe_autocast`；`from .config import …` 改成绝对导入（`from lib.chronos2.config import …`）。
- `model.py`：删掉 `import copy`、`from packaging import version` 与 `from transformers import __version__ as transformers_version`；版本判断那一段换成 `from transformers import initialization as init` 与 `_TRANSFORMERS_V5 = True`；`from chronos.chronos_bolt import InstanceNorm, Patch` 与两处相对导入改成包里的绝对导入；`encoder_config = copy.deepcopy(config)` 改成 `encoder_config = config`（拷贝只为把 `is_decoder` 设成 false，它本来就是 false）。
- `config.py` 不用改。

载入与调用见合同 README 的 Chronos-2 一段：`Chronos2Model.from_pretrained(context.pretrained_dir + "/chronos-2")`，前向 `model(context=…, group_ids=…, future_covariates=…, num_output_patches=…)`。用法与要注意的地方（都来自 `model.py` 与 `pipeline.py`）：

- `context` 是（序列数，长度），缺失为 NaN；长度不同的序列在左边补 NaN 对齐（作者的管线这样做）。超过 8,192 点只留最后 8,192 点；长度不是 16 的倍数时左边补 NaN 成整块。
- 组号：同值的序列在每个块的位置上互相做注意力；不给组号时每条序列各自成组。组掩码是（块数 × 批 × 批），一次调用里的序列两两之间都算一遍再遮掉，所以一次调用放多少条决定组注意力的开销。
- 一组里的次序：先目标，再只有过去值的协变量，最后已知协变量（`preprocess.py` 的 `_build_prepared_inputs`）。`future_covariates` 是（序列数，预测长度），已知协变量的行放未来值，其余行放 NaN；已知协变量的未来值按它自己历史的均值与标准差缩放。
- 输出 `quantile_preds` 是（序列数，21，输出块数 × 16），已经反变换回序列自己的单位；协变量行也有预测，不用。`model.encode(...)` 返回的第一个值的 `last_hidden_state` 是（序列数，历史块数 + 1 + 输出块数，768），倒数第「输出块数 + 1」个位置是 REG。
- 训练：`forward` 传 `future_target`（序列数，长度不超过输出块数 × 16；协变量行放 NaN）时另返回 `loss`：分位损失在预测步上取平均（遮掉的位置按 0 计）、在 21 个分位上求和、在批上取平均（协变量行按 0 计）；已知协变量的位置不计。`model.train()` 下 dropout 0.1 生效。作者的 `Chronos2Pipeline.fit` 在会话的 shell 里可以照用（它复制一份模型，经 `transformers.Trainer` 训练，把结果存进 `output_dir`）；策略里自己写训练循环，或者用 `transformers.Trainer` 并把输出目录放在 `context.state_dir` 下。
- 随机初始化：`Chronos2Model(config)`，`config` 取已载入模型的 `model.config` 或 `Chronos2CoreConfig.from_pretrained(context.pretrained_dir + "/chronos-2")`；作者的 `_init_weights` 按 T5 的方式初始化。
- 训练出的权重写在 `context.state_dir`，不写进模型库。

## 阶段一的任务与读法

- 数据：`refs/data/chronos2/train-00000-of-00001.parquet`，一行：`id`、`timestamp`（9,357 个整点，2004-03-10 18:00 到 2005-04-04 14:00）与 13 个序列列；缺失为 NaN（UCI 原文件记作 −200）：CO(GT) 1,683 个、NOx(GT) 1,639 个、NO2(GT) 1,642 个，C6H6(GT) 与 T、RH、AH 各 366 个。来源是 UCI Air Quality（De Vito 等，Sensors and Actuators B 129(2)，2008；UCI 机器学习库，CC BY 4.0，doi 10.24432/C59K5F），这里是 fev 作者转换后的格式，转换细节未核实。
- 任务（fev-bench 的 `tasks.yaml`）：目标 CO(GT)、C6H6(GT)、NOx(GT)、NO2(GT)；已知协变量 T、RH、AH；其余列不用；预测 168 步，20 个窗口，步长 168，第一个窗口的历史到倒数第 3,361 行为止；季节长度 24；分位 0.1 到 0.9。
- SQL（fev 的 `metrics.py`）：每个窗口里，对每个目标维、每个分位 q 与每个预测步算 2·|(y − ŷ_q)·(1{y ≤ ŷ_q} − q)|，除以这一维在窗口历史上的季节误差 mean|y_t − y_{t−24}|（缺失不计）；在预测步上取平均（真值缺失的步不计），在 9 个分位上取平均，在 4 个目标上取平均；最后在 20 个窗口上取平均。
- 公布的数（fev 仓库 `benchmarks/fev_bench/results/`，提交 ecce67120aae，fev 0.8.0）：Chronos-2 0.7983（交叉学习、一批 100 条、float32、在卡上），TiRex 0.8650，TimesFM-2.5 0.8769，TabPFN-TS 0.9157，季节朴素 1.3840。论文里没有这个任务单独的数；这份结果文件与论文表 3 是否是同一次运行，未核实。
- 包作者的读数（镜像、4 个 CPU、float32，作者的 `predict_quantiles`，字典形式的输入）：（a）四个目标与三个协变量同组 0.798304；（b）每个目标单独、不用协变量 0.915703；20 个窗口都是（a）更好，按窗口配对的 t 为 7.26；两者合计约 7.5 分钟。（c）（d）没有跑。按上面的写法算的季节朴素是 1.4418，与公布的 1.3840 不同（缺失值的处理不同），不作目标。
- 三个协变量在预测窗口里给的是事后的真实气象值：这是 fev 对已知协变量的设定，放到行情上就是前视。

## 代价

下面按矩阵乘法的 FLOP 估算，没有在卡上实测，先用冒烟量。每处理一个位置（一个 16 点的块）约 2.4 亿 FLOP（参数个数的两倍）；时间注意力的开销随块数的平方增长，上下文 8,192 点（512 块）时每个位置约再加 8%，512 日（32 块）时可以忽略；组注意力每层每个位置每条约 4·B·768 FLOP（B 是一次调用里的条数）。全市场按约 4,700 个名字算：

| 用法 | 每个决策日 | 一场研究年（约 1,701 天） |
|---|---|---|
| 一次前向，512 日上下文（32 块加 REG 与 1 个输出块） | 约 3.8e13 FLOP | 约 6.5e16 FLOP |
| 一次前向，2,048 日上下文 | 约 1.5e14 FLOP | 约 2.5e17 FLOP |
| 全市场放进一次调用（B = 4,700） | 组注意力与线性部分同量级，另有约 3 GB 的组掩码 | — |
| 微调一步，256 条 × 2,048 点（前向加反向） | 约 2.4e13 FLOP | 作者默认的 1,000 步约 2.4e16 FLOP |

量级换算：按论文的 A10G 吞吐（每秒约 300 条，上下文 2,048），全市场每天约 16 s、一场研究年约 7.5 小时，上下文 512 日时约四分之一；A100 上没有数。包作者在 4 个 CPU 上量到 256 条 512 点一次前向约 35 s，约每秒 6e10 FLOP，所以 CPU 只够阶段一这样的小任务。显存：权重 0.45 GiB，全参数微调时权重、梯度与 AdamW 状态约 1.8 GiB，激活随批大小增长，批大小按冒烟时的峰值定。

## 与本项目问题的差别，以及它意味着什么

| | 论文 | 本项目 |
|---|---|---|
| 目标 | 一条或几条序列未来的值，按分位损失与 MASE 读 | 同一天整个截面上其后若干天收益的排序 |
| 组 | 一个任务的目标与协变量，或交叉学习时的一批 | 同行业或同涨同跌的股票，或目标连同它的指数、行业与事件序列 |
| 协变量 | 能源与零售任务里多有未来已知的值（日前负荷预测、促销、假日） | 几乎都只有过去值；未来已知的只有日历一类 |
| 时段 | 各基准的测试段，fev-bench 与 Chronos Benchmark II 没有进训练 | 研究年与验证期都早于发布，语料是否覆盖这些行情无法证明 |
| 使用方式 | 零样本 | 每个决策只能用当时可见的数据训练与选择 |
| 组合 | 没有 | 20 万元、整手、最低佣金、按本项目成本与持仓规则 |

意味着：直接拿零样本的中位数排序，读到的是一个为预测误差训练的模型在截面排序上的表现，与论文的读数不可比。论文最强的证据在未来已知的协变量上，行情里这类信息几乎没有，所以本臂实际检验的是同组的股票与只有过去值的外生序列能不能经它传进来；分组与协变量的对照两边用同样的权重，是本臂最干净的比较。
