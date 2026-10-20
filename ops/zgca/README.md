# ZGCA A100 工作流

`ops/zgca/zgca.py` 把在 ZGCA A100 开发机上跑一轮研究的整个周期做成七个子命令：`up`、`deploy`、`push`、`run`、`status`、`sync`、`down`。它只调用 `~/zgca-compute` 的平台入口：`instance` 查看与启停实例，`dev` 经跳板登录开发机，并用它的四个原语补装基础工具（`bootstrap`）、做带比对的传输（`copy`）、写仅属主可读的文件（`put`）和报告 GPU、磁盘与 tmux（`report`）。实例生命周期、传输吞吐、个人存储和火山 Docker 这些与项目无关的平台事实只记在 `~/zgca-compute/README.md`，这里只写本项目的步骤。沙箱隔离按执行器原样：平台提供的 rootless Docker、`--network none`、非 root 的 agent 用户、只读的策略挂载，这里一概不动。开发机那一侧的步骤都在 `remote.sh` 里，每次调用把本地这份脚本原样发过去，在登录 shell 里执行，开发机上不会留下过期副本。

```bash
python ops/zgca/zgca.py up
python ops/zgca/zgca.py deploy data/pit_views_seed_research_8y_fundanns_20261019
python ops/zgca/zgca.py push hcm_100k_8y_sol_qwen_20261014d hcm_100k_8y_sol_dsf_20261014d
python ops/zgca/zgca.py run scripts/experiments/create_round_<轮次>.py --dry-run
python ops/zgca/zgca.py status
python ops/zgca/zgca.py sync <臂 id>
python ops/zgca/zgca.py down
python ops/zgca/zgca.py --instance <平台实例 id> status
```

在本机的 `quant` 环境里运行。除 `push` 外，脚本只用标准库和本检出里的模型档案，系统 `python3` 也能跑；`push` 要用控制台自己的“在跑”判定，那个模块依赖 numpy。`ZGCA_COMPUTE` 可以指向别处的 zgca-compute 检出，默认 `~/zgca-compute`；`ZGCA_GITHUB_PROXY` 是下载 GitHub 发布包时先试的代理前缀（例如 `https://ghfast.top/`）。实例缺省是 `auto-trade-dev-a100x2`（两张 A100-80G，平台 id `mpo06t71crmmtdsrb98ckwk29`）；放在子命令之前的 `--instance <平台实例 id>` 让同一个脚本驱动另一台，`instance` 和 `dev` 都被指到所选的这台，别的实例同时在跑也不会选错。开发机上的仓库根目录不随实例改变，缺省是 `/personal_fast_storage/auto-trade`；`ZGCA_REMOTE_ROOT=/personal_fast_storage/<名称>` 把所有传输和开发机步骤移到个人存储下的另一个目录（其他写法在动开发机之前就被拒绝），供验证部署与在用的树并存：同一资源区的实例看到的是同一份个人存储。另设的根要用 `--instance` 配一台非缺省的实例：沙箱镜像、tmux 会话和控制台端口属于实例而不属于根，缺省实例服务在用的树，在它上面另设根可能重建在跑臂所用的镜像（重建前的检查只看本根的臂）或经在用的控制台建臂，所以被拒绝。

## 各子命令

`up` 就是 `instance start`：已是 RUNNING 就只确认 SSH，否则发出一次启动请求（平台暂时拒绝时由 `instance` 退避重试），等到实例显示 RUNNING 且 SSH 能登录，最后打印本周卡时。

`deploy <种子目录>` 先在本机核对：种子目录在 `data/` 下且预构建已完成（没有 `prebuild_unfinished.json`），它的 `provider.json` 指向的 research release 已经发布在 `data/research_releases/` 下，`.env` 存在且设了模型档案点名的键，预训练模型库 `data/pretrained/` 带着它的 `MANIFEST.sha256`（开发机上新建的臂钉住并挂载这份库）。镜像构建要装的 DuckDB CLI 发布包和 causal-conv1d 轮子（约 197 MB）都在 GitHub 上，开发机访问不到，所以由本机下载一次、缓存在 `.runtime/zgca/`：设了 `ZGCA_GITHUB_PROXY` 时先经代理、再直连 GitHub，只有与钉住的 SHA-256 相符的下载才进缓存，缓存里摘要不符的文件直接拒绝。核对都通过、实例在运行，才开始动开发机：先 `dev bootstrap`（新实例的镜像不带 rsync 和 tmux），建好运行时目录，再依次传输，每一次都由 `dev copy` 传完后以同样参数 dry-run 比对，有差异就停下：

| 内容 | 方式 |
| --- | --- |
| `git ls-files` 列出的受管文件 | 按清单复制，不删除开发机上的多余文件 |
| 模型网关的 key：本机 `.env` 里模型档案点名的那些行（各档案的 key 与地址变量，加上中转站的备用 key） | `dev put` 写成开发机的 `.env`，仅 root 可读，本机不落临时文件，写完核对摘要 |
| 两个发布包和受管文件清单 | 复制到开发机的 `.runtime/zgca/` |
| 种子目录、该 release 目录、预训练模型库 | 精确镜像（`--delete`，硬链接保留） |

TuShare 令牌、`HF_TOKEN` 等其余变量不离开本机。清单到位后、传种子之前，开发机按清单清理代码：在含受管文件的顶层目录里删除清单上没有的文件，运行时目录（`data/`、`experiments/`、`logs/`、`results/`、`.runtime/`）从不进入，符号链接、`__pycache__` 和 `*.egg-info` 保留，这样退役的轮次文件不会留在开发机上被启动器或测试读到。最后是准备步骤：装 ripgrep（worker 在会话开始前就拒绝没有 `rg` 的宿主）；venv 不存在或缺包时用开发机的 miniconda 建 `/personal_fast_storage/envs/quant`（另设的根用自己里面的 `.venv`，以免改指在用的树所跑的那个），以可编辑方式安装本仓库（平台 PyPI 镜像缺 tushare 1.4.29，所以加清华作为额外索引）；确认 Docker 可达，再用执行器自己的指纹检查判断沙箱镜像。镜像不存在、或者它烤入的策略契约与这份检出不一致，就按平台的构建办法（旧构建器、默认网桥、`--force-rm`）重建，两个发布包由开发机在网桥网关上临时提供；有臂在跑时拒绝重建，因为根盘容不下第二个镜像；构建因磁盘用尽失败时清掉不用的镜像和已停止的容器，冷构建一次。重建后再检查一次，仍不一致即失败。`deploy` 以 `dev report` 收尾。

`push <臂 id>...` 把本机的臂送到开发机，供在那里新建的臂当作谱系（`lineage_arms`）读取：新臂创建时要读它点名的每一条早先臂，它们必须在那台机器的 `experiments/` 下，而 `deploy` 不传实验目录。每条臂必须是本机的臂（`hitl/params.json` 记的实验根不是开发机的，从开发机同步来的副本一律拒绝），而且不在本机运行（控制台自己的规则：worker 活着且状态没有结束），全部核对通过才开始传；用 `dev copy` 复制，不含可从种子重建的 `pit_views/`，不带 `--delete`，在 `deploy` 之后运行。例如 2026-10-19 第二轮的 HCM 臂以 `_profiles.HCM_LINEAGE` 的四条臂为谱系，其中两条 `zgca_20261014c` 臂本来就建在开发机的个人存储上，只在本机的两条在 `run` 之前送过去：

```bash
python ops/zgca/zgca.py push hcm_100k_8y_sol_qwen_20261014d hcm_100k_8y_sol_dsf_20261014d
```

`run <轮次文件> [启动器参数...]` 要求轮次文件是仓库里的受管文件（否则 `deploy` 不会把它带过去）。它先确保控制台在 tmux 会话 `autotrade-console` 里以 `AUTOTRADE_GPU_NAME=A100` 运行并回答 `/api/health`，再把启动器放进 tmux 会话 `autotrade-round`，同样带着 `AUTOTRADE_GPU_NAME=A100`，端口由脚本补上，其余参数（`--dry-run`、`--fill`、臂 id）原样传给轮次文件；输出写到开发机的 `logs/zgca/<轮次文件名>_<UTC 时间>.log`，结束时追加一行 `round-exit <退出码>`。上一轮启动器还在跑时拒绝再启动。启动器创建臂之前要先做起步策略的冒烟回放，在 A100 上实测约两小时；通过的记录在开发机的 `logs/research/smoke/` 下按种子、几何与起步策略字节保存，下次直接复用。

`status` 先用 `instance show` 打印实例状态与卡时；实例在运行时再用 `dev report` 打印 GPU、根盘与个人存储余量和 tmux 会话，然后打印控制台健康（含空闲槽位与空闲 GPU）、每条在跑臂的状态、阶段与心跳距今的秒数（worker 已不在却没有结束的臂单独标出）、沙箱镜像，以及最近一份 `create_round_*` 启动器日志的末尾三行（镜像构建日志也在同一目录，不算在内）。

`sync <臂 id>` 把开发机上这条臂的实验目录（不含 `pit_views/`，那是可从种子重建的 12 GB 缓存）和 `logs/workers/<臂>.log` 用 `dev copy` 拉回本仓库。本地已有同名目录时，只有它的 `hitl/params.json` 记着开发机的实验根（即以前从开发机同步来的副本）才会更新，本机自己的臂一律拒绝覆盖。同步前先打印臂当前的状态；臂还在跑时拒绝同步，因为它的 `status.json` 每几秒改写一次、日志不断增长，拷贝过不了比对，等它结束再同步。“在跑”与 `status`、`push` 和镜像重建的判定都用控制台自己的规则（`hitl_state.running_arms`）。

`down` 就是 `instance stop`：等到 STOPPED 后打印卡时。它不问也不检查有没有臂在跑：先看 `status`，由调用者决定。

## 幂等与失败

启停的幂等意图（键何时写下、保留或删除）和平台暂时拒绝时的退避重试都由 `instance` 负责，见 `~/zgca-compute/README.md` 的“API 与生命周期”；启停没有确认到目标状态时，重跑同一个子命令即可。每一步都校验自己的结果，失败就以非零退出并说明原因，没有静默回退。

## 为什么不再改写元数据

2026-10-14 的手工部署要把种子 `provider.json`、两个 as-of stash `contract.json` 和 release `manifest.json` 里的绝对仓库路径改成开发机路径，还要建空的 `data/raw` 与 `data/pit/fundamental_events`。根因是 worker、种子链接和 stash 绑定把这些构建时记下的宿主路径当作契约来比较，并要求活数据湖的目录存在。自 0038e3f 起，release 由 generation id 标识，记录里的路径只作记录；manifest 的来源按相对仓库的位置比较；活数据湖的根目录不再要求存在，因为臂读的是钉住的 release。所以 `deploy` 原样复制，开发机上的记录也就不再因此而与本机不同。

## 一个周期的开销

卡时规则见 `~/zgca-compute/README.md`。全新的根要把种子（约 16 GB）、release（约 48 GB）和预训练模型库（约 12 GB）全量传一遍，按那里记的 3.1–3.7 MB/s 要六到七小时：2026-10-19 向两卡实例部署时，种子用了 73 分钟，模型库 61 分钟（release 已在开发机上，比对 34 秒）。之后 `deploy` 只传差异，几分钟内完成。同一次部署里冷构建镜像约 6 分钟，缓存命中约 15 秒；起步策略的冒烟约两小时；一条 HCM 整期臂约七小时。

## 改动后的验证

本工作流或工具包的 `instance`、`dev` 改过、只过了离线测试时，要在一台新实例上把整个周期跑一遍，部署到另设的根，不碰在用的树。截至 2026-10-20，平台没有给本账户分配不带 GPU 的规格，组织又只许同时开两卡（见工具包 README 的“配额、规格与存储”），所以等两卡实例 `down` 之后，按工具包 README 的创建步骤在资源区 5 用当前镜像建一台单卡（`A100-80GB-1`）实例 `auto-trade-validate`，再依次运行：

```bash
V=<新实例 id>
export ZGCA_REMOTE_ROOT=/personal_fast_storage/auto-trade-validate ZGCA_GITHUB_PROXY=https://ghfast.top/
python ops/zgca/zgca.py --instance $V up
python ops/zgca/zgca.py --instance $V deploy data/pit_views_seed_research_8y_fundanns_20261019
python ops/zgca/zgca.py --instance $V push range20_csi300_8y_20260927
python ops/zgca/zgca.py --instance $V status
~/zgca-compute/instance stop $V && ~/zgca-compute/instance start $V
ZGCA_INSTANCE=$V ~/zgca-compute/dev bash -lc 'ls -d /personal_fast_storage/auto-trade-validate && rm -rf /personal_fast_storage/auto-trade-validate'
python ops/zgca/zgca.py --instance $V down
ssh-keygen -R zenith-$V -f ~/.config/zgca-compute/known_hosts
unset ZGCA_REMOTE_ROOT
```

`deploy` 要跑六到七小时（见上一节），放进 tmux 或 nohup，输出写进 `logs/zgca/`；加上 venv 安装和冷构建镜像，整个验证约七卡时。按输出里的 UTC 时刻记下每步耗时，并核对：`dev bootstrap` 在新镜像上真的装了 rsync 和 tmux；每次 `dev copy` 和 `dev put` 都没有报差异或摘要不符（成功时它们不输出）；准备步骤在验证根的 `.venv` 建好 venv 并构建镜像；`push` 的臂（2026-10-20 本机最小的已结束、没有 Paper 账户的臂，不含 `pit_views/` 约 9 MB）送到后 `status` 数出一条已结束的臂；停机后立即启动时平台多半先拒绝，`instance` 应打印 “start refused for now … trying again”，退避后启动成功。验证根要在验证实例还在运行时删除，先 `ls -d` 确认路径。个人 Token 没有删除实例的接口，`down` 只是停机：停下的实例不计卡时，连续停满 30 天由平台软删除（此前连发三天提醒），7 天恢复期过后删除。

## 已接受的局限

`/personal_fast_storage` 按资源区隔离、同一资源区内的实例共用，所以用 `--instance` 驱动的另一台若与缺省实例在同一资源区，两台读写的是同一个仓库根目录：代码、`.env`、种子、臂目录和日志都是同一份，一台上的 `deploy` 改的也是另一台正在用的代码，一台的 `status` 会把另一台在跑的臂标成 worker 已不在。两台同时跑臂时要放在不同资源区，或者用 `ZGCA_REMOTE_ROOT` 给另一台一个自己的根，或者同一时间只让一台有臂。

`deploy` 只在仍含受管文件的顶层目录里清理：整个退出版本库的顶层目录、仓库根下不再受管的文件和清理后留下的空目录仍留在开发机上。镜像只在合同指纹说它缺失或过期时重建，所以只改 Dockerfile、合同文件不变的改动不会让实例重建，要先在实例上删掉镜像 tag 再部署。控制台不随 `deploy` 重启，改了代码之后要在没有臂运行时 `~/zgca-compute/dev tmux kill-session -t autotrade-console` 再 `run`。`status` 只认 `create_round_*` 命名的启动器日志。

模型服务的 key 在实例存在期间一直留在开发机的 `.env` 里（仅 root 可读），停机不清除；其中本机 vLLM 网关的 key 在开发机上没有用处（那里没有这个网关），但 worker 启动时要求每个角色都有凭据，而 NL 总在本机模型上。`sync` 不拉回开发机上的模型服务会话日志（`data/llm_conversations/`）：日志按服务方、模型、日期和 worker 进程号分文件，几条臂写在同一棵目录里，续跑换了进程号的旧文件也无从按臂认出，一条 gpt-6 臂就有数百 MB；要用时整棵取回，`~/zgca-compute/dev rsync -a dev:/personal_fast_storage/auto-trade/data/llm_conversations/ <本地目录>/`（有臂在跑时目录仍在增长，所以不用 `dev copy` 比对）。

开发机上的容器遥测缺少峰值内存和 GPU 峰值，开发机上的一切又都以 root 运行，原因见工具包 README 的“火山平台 Docker”；所以在开发机宿主上跑单元测试时，依赖目录权限位的测试会失败，Docker 沙箱以 uid 61000 运行，不受影响。离线测试在 `tests/unit/test_zgca_workflow.py`，用模拟的 `instance` 和 `dev`，不访问网络也不启停实例；那些入口本身的测试在 `~/zgca-compute/tests`。
