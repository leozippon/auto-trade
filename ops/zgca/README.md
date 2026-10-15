# ZGCA A100 工作流

`ops/zgca/zgca.py` 把在 ZGCA A100 开发机上跑一轮研究的整个周期做成六个子命令：`up`、`deploy`、`run`、`status`、`sync`、`down`。它只调用 `~/zgca-compute` 里的两个平台入口——`zenith`（经跳板访问 Zenith API）和 `dev`（经跳板 SSH、rsync 到开发机）——不另建调度层，不缓存开发机地址，仓库里不保存任何密钥。沙箱隔离按执行器原样：平台提供的 rootless Docker、`--network none`、非 root 的 agent 用户、只读的策略挂载，这里一概不动。开发机那一侧的步骤都在 `remote.sh` 里，每次调用把本地这份脚本原样发过去执行，开发机上不会留下过期副本。平台本身的用法、配额和历史实测见 `~/zgca-compute/README.md` 与 `NOTES.md`。

```bash
python ops/zgca/zgca.py up
python ops/zgca/zgca.py deploy data/pit_views_seed_research_8y_fund_20261014b
python ops/zgca/zgca.py run scripts/experiments/create_round_20261014c.py --dry-run
python ops/zgca/zgca.py status
python ops/zgca/zgca.py sync hcm_100k_8y_sol_dsf_zgca_20261014c
python ops/zgca/zgca.py down
```

脚本只用标准库，本机的 `quant` 环境或系统 `python3` 都能运行；`ZGCA_COMPUTE` 可以指向别处的 zgca-compute 检出，默认 `~/zgca-compute`。实例固定为 `auto-trade-dev-a100`（一张 A100-80G），`dev` 也被指到这台机器，所以别的实例同时在跑也不会选错。

## 六步各做什么

`up` 读实例状态；已是 RUNNING 就直接返回，否则发出一次启动请求，每 5 秒轮询操作直到 SUCCEEDED、再等实例显示 RUNNING，最后确认 SSH 能登录，并打印本周卡时。

`deploy <种子目录>` 先在本机核对：种子目录在 `data/` 下且预构建已完成（没有 `prebuild_unfinished.json`），它的 `provider.json` 指向的 research release 已经发布在 `data/research_releases/` 下，`.env` 存在。然后按顺序同步到开发机的 `/personal_fast_storage/auto-trade`：`git ls-files` 列出的受管文件、`.env`、种子目录、该 release 目录（后两者是精确镜像，带 `--delete`，硬链接保留）；每一次传输之后都用 rsync 的 dry-run 逐项比对，有差异就停下。最后在开发机上执行准备步骤：装 ripgrep（worker 在会话开始前就拒绝没有 `rg` 的宿主）和 tmux，venv 不存在或缺包时用开发机的 miniconda 建 `/personal_fast_storage/envs/quant` 并以可编辑方式安装本仓库（平台镜像缺 tushare 1.4.29，所以加清华作为额外索引），确认 Docker 可达并打印磁盘余量，再用执行器自己的指纹检查判断沙箱镜像：镜像不存在、或者它烤入的策略契约与这份检出不一致，就用旧构建器在默认网桥上重建（`DOCKER_BUILDKIT=0`，平台没有 buildx 也不允许 `--network host`；停机重开后 tag 丢失而层都在时，全部命中缓存约 15 秒），重建后再检查一次，仍不一致即失败。

`run <轮次文件> [启动器参数...]` 要求轮次文件是仓库里的受管文件（否则 `deploy` 不会把它带过去）。它先确保控制台在 tmux 会话 `autotrade-console` 里以 `AUTOTRADE_GPU_NAME=A100` 运行并回答 `/api/health`，再把启动器放进 tmux 会话 `autotrade-round`，同样带着 `AUTOTRADE_GPU_NAME=A100`，端口由脚本补上，其余参数（`--dry-run`、`--fill`、臂 id）原样传给轮次文件；输出写到开发机的 `logs/zgca/<轮次>_<UTC 时间>.log`，结束时追加一行 `round-exit <退出码>`。上一轮启动器还在跑时拒绝再启动。启动器创建臂之前要先做起步策略的冒烟回放，在 A100 上实测约两小时；通过的记录在开发机的 `logs/research/smoke/` 下按种子、几何与起步策略字节保存，下次直接复用。

`status` 打印实例状态与卡时；实例在运行时再登录一次，打印 tmux 会话、控制台健康（含空闲槽位与空闲 GPU）、每条活动臂的状态、阶段与心跳距今的秒数、GPU 占用、根盘与个人存储余量、沙箱镜像，以及最近一份启动器日志的末尾三行。

`sync <臂 id>` 把开发机上这条臂的实验目录（不含 `pit_views/`，那是可从种子重建的 12 GB 缓存）和 `logs/workers/<臂>.log` 拉回本仓库，拉回后同样 dry-run 比对。本地已有同名目录时，只有它的 `hitl/params.json` 记着开发机的实验根（即以前从开发机同步来的副本）才会更新，本机自己的臂一律拒绝覆盖。臂还在跑时也能同步，先打印它当前的状态。

`down` 发出停止请求并等到 STOPPED，然后打印卡时。它不问也不检查有没有臂在跑：先看 `status`，由调用者决定。

## 幂等与失败

启动和停止各是一个意图，键写在本仓库 `.runtime/zgca/<实例>.start.key` 或 `.stop.key`，请求发出之前就落盘。操作 SUCCEEDED 且实例到达目标状态后删除；平台报 FAILED 也删除，因为再试是新的意图。请求被拒（503、429）、传输中断、操作停在 UNKNOWN 时键都保留，重跑同一个子命令就带着同一个键再发，平台按幂等回放，不会多启一次；只有确认要开始新的意图时才手动删掉键文件。每一步都校验自己的结果，失败就以非零退出并说明原因，没有静默回退。

## 为什么不再改写元数据

2026-10-14 的手工部署要把种子 `provider.json`、两个 as-of stash `contract.json` 和 release `manifest.json` 里的绝对仓库路径改成开发机路径，还要建空的 `data/raw` 与 `data/pit/fundamental_events`。根因是 worker、种子链接和 stash 绑定把这些构建时记下的宿主路径当作契约来比较，并要求活数据湖的目录存在。自 0038e3f 起，release 由 generation id 标识，记录里的路径只作记录；manifest 的来源按相对仓库的位置比较；活数据湖的根目录不再要求存在，因为臂读的是钉住的 release。所以 `deploy` 原样复制，开发机上的记录也就不再因此而与本机不同。

## 一个周期的开销

实例按卡数乘开机时长计卡时，每周 168，北京时间周一零点重置，停机后约一分钟才更新计量。首次向 `/personal_fast_storage` 传种子（12 GB）和 release（49 GB）在本机约 12 MB/s 的出口上要一个半小时左右，之后 `deploy` 只传差异，几分钟内完成；冷构建镜像约十分钟，缓存命中约 15 秒；起步策略的冒烟约两小时；一条 HCM 整期臂约七小时。开机后先看 `status` 里的磁盘：根盘约 20 GB 与镜像盘共用。

## 已接受的局限

`deploy` 对代码只增不删：本机删除了的受管文件不会从开发机删除。控制台不随 `deploy` 重启，改了代码之后要在没有臂运行时 `~/zgca-compute/dev tmux kill-session -t autotrade-console` 再 `run`。开发机上的容器遥测缺少峰值内存和 GPU 峰值，那是平台 daemon 在另一个 PID 命名空间里的缘故（见 `NOTES.md`）。离线测试在 `tests/unit/test_zgca_workflow.py`，用模拟的 `zenith` 和 `dev`，不访问网络也不启停实例。
