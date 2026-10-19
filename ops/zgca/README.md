# ZGCA A100 工作流

`ops/zgca/zgca.py` 把在 ZGCA A100 开发机上跑一轮研究的整个周期做成六个子命令：`up`、`deploy`、`run`、`status`、`sync`、`down`。它只用 `~/zgca-compute` 里的三个平台入口——`instance`（查看与启停实例）、`dev`（经跳板 SSH、rsync 到开发机），以及二者共用的 `zenith`（经跳板访问 Zenith API）——不另建调度层，不缓存开发机地址，仓库里不保存任何密钥。沙箱隔离按执行器原样：平台提供的 rootless Docker、`--network none`、非 root 的 agent 用户、只读的策略挂载，这里一概不动。开发机那一侧的步骤都在 `remote.sh` 里，每次调用把本地这份脚本原样发过去执行，开发机上不会留下过期副本。平台本身的用法、配额和历史实测见 `~/zgca-compute/README.md` 与 `NOTES.md`。

```bash
python ops/zgca/zgca.py up
python ops/zgca/zgca.py deploy data/pit_views_seed_research_8y_fund_20261014b
python ops/zgca/zgca.py run scripts/experiments/create_round_<轮次>.py --dry-run
python ops/zgca/zgca.py status
python ops/zgca/zgca.py sync hcm_100k_8y_sol_dsf_zgca_20261014c
python ops/zgca/zgca.py down
python ops/zgca/zgca.py --instance <平台实例 id> status
```

脚本只用标准库和本检出里的模型档案（同样只用标准库），本机的 `quant` 环境或系统 `python3` 都能运行；`ZGCA_COMPUTE` 可以指向别处的 zgca-compute 检出，默认 `~/zgca-compute`。实例缺省是 `auto-trade-dev-a100`（一张 A100-80G）；放在子命令之前的 `--instance <平台实例 id>` 让同一个脚本驱动另一台。`instance` 和 `dev` 都被指到所选的这台机器，所以别的实例同时在跑也不会选错。开发机上的仓库根目录不随实例改变，仍是 `/personal_fast_storage/auto-trade`。

## 六步各做什么

`up` 就是 `instance start`：已是 RUNNING 就直接返回，否则发出一次启动请求，等到实例显示 RUNNING 且 SSH 能登录，最后打印本周卡时。

`deploy <种子目录>` 先在本机核对：种子目录在 `data/` 下且预构建已完成（没有 `prebuild_unfinished.json`），它的 `provider.json` 指向的 research release 已经发布在 `data/research_releases/` 下，`.env` 存在且设了模型档案点名的键，预训练模型库 `data/pretrained/` 带着它的 `MANIFEST.sha256`（开发机上新建的臂钉住并挂载这份库）。然后在开发机上建好运行时目录，缺 rsync 时先装上（平台镜像不带它，而之后的每一次传输都是 rsync），再按顺序同步到开发机的 `/personal_fast_storage/auto-trade`：`git ls-files` 列出的受管文件、模型网关的 key（只取本机 `.env` 里模型档案点名的那些行——各档案的 key 与地址变量，加上中转站的备用 key——经 `dev` 以 umask 077 直接流进开发机的 `.env`，本机不落临时文件，写完核对摘要；TuShare 令牌、`HF_TOKEN` 等其余变量不离开本机，以前整份拷过去的 `.env` 在下一次 `deploy` 时被覆盖）、镜像构建要装的 DuckDB CLI 发布包与 causal-conv1d 轮子（开发机访问不到 GitHub，所以由本机下载、按各自钉住的 SHA-256 核对后带过去，构建时由开发机在网桥网关上临时提供）连同这份受管文件清单、种子目录、该 release 目录、预训练模型库（后三者是精确镜像，带 `--delete`，硬链接保留；模型库首次约 11 GB）；每一次传输之后都用 rsync 的 dry-run 逐项比对，有差异就停下。清单到位后、传种子之前，开发机按清单清理代码：在含受管文件的顶层目录里删除清单上没有的文件，运行时目录（`data/`、`experiments/`、`logs/`、`results/`、`.runtime/`）从不进入，符号链接、`__pycache__` 和 `*.egg-info` 保留，这样退役的轮次文件不会留在开发机上被启动器或测试读到。最后在开发机上执行准备步骤：装 ripgrep（worker 在会话开始前就拒绝没有 `rg` 的宿主）和 tmux，venv 不存在或缺包时用开发机的 miniconda 建 `/personal_fast_storage/envs/quant` 并以可编辑方式安装本仓库（平台镜像缺 tushare 1.4.29，所以加清华作为额外索引），确认 Docker 可达并打印磁盘余量，再用执行器自己的指纹检查判断沙箱镜像：镜像不存在、或者它烤入的策略契约与这份检出不一致，就用旧构建器在默认网桥上重建（`DOCKER_BUILDKIT=0`，平台没有 buildx 也不允许 `--network host`；停机重开后 tag 丢失而层都在时，全部命中缓存约 15 秒），重建后再检查一次，仍不一致即失败。

`run <轮次文件> [启动器参数...]` 要求轮次文件是仓库里的受管文件（否则 `deploy` 不会把它带过去）。它先确保控制台在 tmux 会话 `autotrade-console` 里以 `AUTOTRADE_GPU_NAME=A100` 运行并回答 `/api/health`，再把启动器放进 tmux 会话 `autotrade-round`，同样带着 `AUTOTRADE_GPU_NAME=A100`，端口由脚本补上，其余参数（`--dry-run`、`--fill`、臂 id）原样传给轮次文件；输出写到开发机的 `logs/zgca/<轮次>_<UTC 时间>.log`，结束时追加一行 `round-exit <退出码>`。上一轮启动器还在跑时拒绝再启动。启动器创建臂之前要先做起步策略的冒烟回放，在 A100 上实测约两小时；通过的记录在开发机的 `logs/research/smoke/` 下按种子、几何与起步策略字节保存，下次直接复用。

`status` 先用 `instance show` 打印实例状态与卡时；实例在运行时再登录一次，打印 tmux 会话、控制台健康（含空闲槽位与空闲 GPU）、每条在跑臂的状态、阶段与心跳距今的秒数（worker 已不在却没有结束的臂单独标出）、GPU 占用、根盘与个人存储余量、沙箱镜像，以及最近一份启动器日志的末尾三行。

`sync <臂 id>` 把开发机上这条臂的实验目录（不含 `pit_views/`，那是可从种子重建的 12 GB 缓存）和 `logs/workers/<臂>.log` 拉回本仓库，拉回后同样 dry-run 比对。本地已有同名目录时，只有它的 `hitl/params.json` 记着开发机的实验根（即以前从开发机同步来的副本）才会更新，本机自己的臂一律拒绝覆盖。同步前先打印臂当前的状态；臂还在跑时拒绝同步，因为它的 `status.json` 每几秒改写一次、日志不断增长，拷贝过不了比对，等它结束再同步。“在跑”与 `status` 和镜像重建的判定都用控制台自己的规则（`hitl_state.running_arms`：worker 进程活着且状态没有结束）。

`down` 就是 `instance stop`：等到 STOPPED 后打印卡时。它不问也不检查有没有臂在跑：先看 `status`，由调用者决定。

## 幂等与失败

启动和停止各是一个幂等意图，键何时写下、保留或删除由共享入口 `instance` 决定，见 `~/zgca-compute/README.md` 的“API 与生命周期”；启停没有确认到目标状态时，重跑同一个子命令即可。每一步都校验自己的结果，失败就以非零退出并说明原因，没有静默回退。

## 为什么不再改写元数据

2026-10-14 的手工部署要把种子 `provider.json`、两个 as-of stash `contract.json` 和 release `manifest.json` 里的绝对仓库路径改成开发机路径，还要建空的 `data/raw` 与 `data/pit/fundamental_events`。根因是 worker、种子链接和 stash 绑定把这些构建时记下的宿主路径当作契约来比较，并要求活数据湖的目录存在。自 0038e3f 起，release 由 generation id 标识，记录里的路径只作记录；manifest 的来源按相对仓库的位置比较；活数据湖的根目录不再要求存在，因为臂读的是钉住的 release。所以 `deploy` 原样复制，开发机上的记录也就不再因此而与本机不同。

## 一个周期的开销

实例按卡数乘开机时长计卡时，每周 168，北京时间周一零点重置，停机后约一分钟才更新计量。首次向 `/personal_fast_storage` 传种子（12 GB）和 release（49 GB）在本机约 12 MB/s 的出口上要一个半小时左右，之后 `deploy` 只传差异，几分钟内完成；冷构建镜像约十分钟，缓存命中约 15 秒；起步策略的冒烟约两小时；一条 HCM 整期臂约七小时。开机后先看 `status` 里的磁盘：根盘约 20 GB 与镜像盘共用。

## 已接受的局限

`/personal_fast_storage` 按资源区隔离、同一资源区内的实例共用，所以用 `--instance` 驱动的第二台若与第一台在同一资源区，两台读写的是同一个仓库根目录：代码、`.env`、种子、臂目录和日志都是同一份，一台上的 `deploy` 改的也是另一台正在用的代码，一台的 `status` 会把另一台在跑的臂标成 worker 已不在。两台同时跑臂时要放在不同资源区，或者同一时间只让一台有臂。`deploy` 只在仍含受管文件的顶层目录里清理：整个退出版本库的顶层目录、仓库根下不再受管的文件和清理后留下的空目录仍留在开发机上。控制台不随 `deploy` 重启，改了代码之后要在没有臂运行时 `~/zgca-compute/dev tmux kill-session -t autotrade-console` 再 `run`。模型服务的 key 在实例存在期间一直留在开发机的 `.env` 里（仅 root 可读），停机不清除；其中本机 vLLM 网关的 key 在开发机上没有用处（那里没有这个网关），但 worker 启动时要求每个角色都有凭据，而 NL 总在本机模型上。`sync` 不拉回开发机上的模型服务会话日志（`data/llm_conversations/`）：日志按服务方、模型、日期和 worker 进程号分文件，几条臂写在同一棵目录里，续跑换了进程号的旧文件也无从按臂认出，一条 gpt-6 臂就有数百 MB；要用时整棵取回，`~/zgca-compute/dev rsync -a dev:/personal_fast_storage/auto-trade/data/llm_conversations/ <本地目录>/`。开发机上的容器遥测缺少峰值内存和 GPU 峰值，那是平台 daemon 在另一个 PID 命名空间里的缘故（见 `NOTES.md`）。离线测试在 `tests/unit/test_zgca_workflow.py`，用模拟的 `instance` 和 `dev`，不访问网络也不启停实例；启停意图本身的测试在 `~/zgca-compute/tests`。

`deploy` 从 GitHub 下载 causal-conv1d 轮子（约 197 MB）时每次尝试限时 15 分钟；本机到 GitHub 慢时（2026-10-19 实测约 34 KB/s）下载不完，`deploy` 就在传输之前失败。这时一次性手工取回（例如经 ghfast.top 代理）放进本机的 `.runtime/zgca/`，`deploy` 照样按钉住的 SHA-256 核对；本机现已缓存这一份，之后不再下载。

带 `lineage_arms` 的臂在创建时要读它点名的每一条早先臂，这些臂必须在创建它的那台机器的 `experiments/` 下，而 `deploy` 不传实验目录。点名的臂只在本机时，先在本机仓库根目录下逐条手工拷过去（不含可从种子重建的 `pit_views/`），同一命令加 `--dry-run --itemize-changes` 再跑一次应当没有输出：

```bash
ZGCA_INSTANCE=<平台实例 id> ~/zgca-compute/dev rsync -a --no-owner --no-group --exclude /pit_views/ \
    experiments/<臂 id>/ dev:/personal_fast_storage/auto-trade/experiments/<臂 id>/
```
