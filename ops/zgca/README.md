# ZGCA A100 工作流

`ops/zgca/zgca.py` 把在 ZGCA A100 开发机上跑一轮研究的整个周期做成八个子命令：`up`、`deploy`、`run`、`resume`、`status`、`sync`、`sync-rounds`、`down`。它只调用 `~/zgca-compute` 的平台入口：`instance` 查看与启停实例，`dev` 经跳板登录开发机，并用它的四个原语补装基础工具（`bootstrap`）、做带比对的传输（`copy`）、写仅属主可读的文件（`put`）和报告 GPU、磁盘与 tmux（`report`）。实例生命周期、传输吞吐、个人存储和火山 Docker 这些与项目无关的平台事实只记在 `~/zgca-compute`（规则在 `README.md`，带日期的观察在 `PLATFORM_RECORDS.md`），这里只写本项目的步骤。沙箱隔离按执行器原样：平台提供的 rootless Docker、`--network none`、非 root 的 agent 用户、只读的策略挂载，这里一概不动。开发机那一侧的步骤都在 `remote.sh` 里，每次调用把本地这份脚本原样发过去，在登录 shell 里执行，开发机上不会留下过期副本。

```bash
python ops/zgca/zgca.py up
python ops/zgca/zgca.py deploy data/pit_views_seed_research_10y_fundanns_20261110
python ops/zgca/zgca.py run scripts/experiments/create_round_<轮次>.py --dry-run
python ops/zgca/zgca.py resume <实验 id>
python ops/zgca/zgca.py status
python ops/zgca/zgca.py sync <实验 id>
python ops/zgca/zgca.py sync-rounds scripts/experiments/create_round_<轮次>.py
python ops/zgca/zgca.py down
python ops/zgca/zgca.py --instance <平台实例 id> status
```

在本机的 `quant` 环境里运行。`deploy` 要读种子记下的快照配置，`sync-rounds` 要导入轮次文件，这些模块依赖 numpy、pandas 等第三方包；其余子命令不需要它们，系统 `python3` 也能跑。`ZGCA_COMPUTE` 可以指向别处的 zgca-compute 检出，默认 `~/zgca-compute`；`ZGCA_GITHUB_PROXY` 是下载 GitHub 发布包时先试的代理前缀（例如 `https://ghfast.top/`）。实例缺省是 `auto-trade-dev-a100x2`（平台 id `mpo06t71crmmtdsrb98ckwk29`；2026-11-02 起是一张 A100-80G，名字沿用建时的两卡）；放在子命令之前的 `--instance <平台实例 id>` 让同一个脚本驱动另一台，`instance` 和 `dev` 都被指到所选的这台，别的实例同时在跑也不会选错。开发机上的布局不随实例改变：仓库根目录是 `/personal_fast_storage/auto-trade`，控制台和 worker 用的 venv 在它旁边的 `/personal_fast_storage/envs/quant`，预训练模型库在实例自己的盘上（见[存储布局](#存储布局)）；这些路径只写在 `zgca.py` 里，每次调用时传给 `remote.sh`。

## 各子命令

`up` 就是 `instance start`：已是 RUNNING 就只确认 SSH，否则发出一次启动请求（平台暂时拒绝时由 `instance` 退避重试），等到实例显示 RUNNING 且 SSH 能登录，最后打印本周卡时。

`deploy <种子目录>` 先在本机核对：预构建视图目录在 `data/` 下且预构建已完成（没有 `prebuild_unfinished.json`），它的 `provider.json` 指向的 research release 已经发布在 `data/research_releases/` 下、并记着种子的快照配置，`.env` 存在且设了模型档案点名的键，预训练模型库 `data/pretrained/` 带着它的 `MANIFEST.sha256`（开发机上新建的实验钉住并挂载这份库），工作树就是它的提交（`git status --porcelain` 为空，否则要求先提交：受管文件按工作树里的内容传过去，开发机上记下的是这个提交）。镜像构建要装的两个文件开发机都取不到：GitHub 上的 DuckDB CLI 发布包，和 NVIDIA 的 CUDA 前向兼容包（见下文 [GPU 驱动](#gpu-驱动)）。所以由本机下载一次、缓存在 `.runtime/zgca/`：设了 `ZGCA_GITHUB_PROXY` 时 GitHub 的发布包先经代理、再直连，只有与钉住的 SHA-256 相符的下载才进缓存，缓存里摘要不符的文件直接拒绝（causal-conv1d 在镜像里从 PyPI 源码编译，开发机经清华镜像取得，不必带过去）。核对都通过、实例在运行，才开始动开发机：有实验在跑时直接拒绝、什么也不改，因为部署会改写和清理它们导入的代码，以 `--delete` 镜像它们读的种子、release 与模型库，还可能重建它们的镜像；tmux 会话 `autotrade-round` 或 `autotrade-fill` 里还有启动器时同样拒绝，因为它可能在部署中途建出实验；否则先 `dev bootstrap`（新实例的镜像不带 rsync 和 tmux），建好运行时目录，并让 `data/pretrained` 成为指向实例盘上模型库目录的链接（那里是真实目录或指向别处时拒绝），再依次传输，每一次都由 `dev copy` 传完后以同样参数 dry-run 比对，有差异就停下：

| 内容 | 方式 |
| --- | --- |
| `git ls-files` 列出的受管文件 | 按清单复制，不删除开发机上的多余文件 |
| 模型网关的 key：本机 `.env` 里模型档案点名的那些行（各档案的 key 与地址变量，加上中转站的备用 key） | `dev put` 写成开发机的 `.env`，仅 root 可读，本机不落临时文件，写完核对摘要 |
| 两个构建用文件、受管文件清单和这次部署的提交号 | 复制到开发机的 `.runtime/zgca/`，提交号在 `deployed_commit` 里，`status` 会显示 |
| 预构建视图目录 | 精确镜像（`--delete`，硬链接保留） |
| 该 release 目录 | 同上，但不带种子快照配置不读的文本数据集与分钟线（新闻与分钟线是 release 的大头：b50c220f 的 51.75 GB 里 38.5 GB）；跳过的目录在开发机上保留不删（`--delete` 不带 `--delete-excluded`），同一 release 的另一份预构建视图可能要读 |
| 预训练模型库 | 最后一步，精确镜像到实例自己的盘 `/ebs/auto-trade/pretrained`，只在放得下时（见下） |

TuShare 令牌、`HF_TOKEN` 等其余变量不离开本机。种子、release 和模型库都原样复制、不改写元数据：其中记下的宿主路径只是记录而不是契约（release 由 generation id 标识，manifest 的来源按相对仓库的位置比较）。清单到位后、传种子之前，开发机按清单清理代码：在含受管文件的顶层目录里删除清单上没有的文件，运行时目录（`data/`、`experiments/`、`logs/`、`results/`、`.runtime/`）从不进入，符号链接、`__pycache__` 和 `*.egg-info` 保留，这样退役的轮次文件不会留在开发机上被启动器或测试读到。然后是准备步骤：装 ripgrep（worker 在会话开始前就拒绝没有 `rg` 的宿主）；venv 不存在时用开发机的 miniconda 建 `/personal_fast_storage/envs/quant`，每次部署都以可编辑方式把本仓库装进这个 venv：钉住的版本都已满足时只重做本仓库自身的安装，改过的钉住版本也就由此生效；确认 Docker 可达，再用执行器自己的指纹检查判断沙箱镜像。镜像不存在、或者它烤入的策略契约与这份检出不一致，就按平台的构建办法（旧构建器、默认网桥、`--force-rm`）重建，DuckDB CLI 与兼容包由开发机在网桥网关上临时提供；有实验在跑时拒绝重建，因为实例盘容不下第二个镜像；构建因磁盘用尽失败时清掉不用的镜像和已停止的容器，冷构建一次。重建后再检查一次，仍不一致即失败。准备步骤最后在镜像里跑一次 GPU 检查，参数与策略容器相同（只读、断网、agent 用户、tmpfs 的 `/tmp`、一张卡）：torch 在卡上算一次矩阵乘、导入 vLLM，跑不起来就失败。

模型库最后传：实例盘上的余量要容下库里还缺的字节，另留 20 GiB 给镜像重建的新层和容器的临时文件。容不下就不传库，打印 `dev report` 后以非零退出，写明还缺多少、盘上空多少：代码、种子、release 和镜像都已部署，库没有，开发机上就不能新建或启动任何实验（建实验要读库的清单）。这样放不下的库不会把盘写满，也不会让人误以为实例已就绪。`deploy` 以 `dev report` 收尾。

`run <轮次文件> [启动器参数...]` 要求轮次文件是仓库里的受管文件（否则 `deploy` 不会把它带过去），并且开发机上次部署记下的提交（`deployed_commit`）就是本机的 HEAD：开发机跑的是部署过去的那棵树，不一致或从未部署时拒绝，什么也不启动，先部署这个提交。核对通过后，它先确保控制台在 tmux 会话 `autotrade-console` 里以 `AUTOTRADE_GPU_NAME=A100` 运行并回答 `/api/health`：控制台只用它启动时的代码，所以没有实验在跑时，已有的控制台先被结束、在当前检出上重启，`deploy` 之后建实验就按新代码校验；有实验在跑时保留它，直到实验都结束（等待中的填充循环每一遍都重新连控制台，正赶上重启那几秒的一遍以非零退出、结束循环，`status` 里看得到）。然后把启动器放进 tmux 会话 `autotrade-round`，同样带着 `AUTOTRADE_GPU_NAME=A100`，端口由脚本补上，其余参数（`--dry-run`、`--fill`、实验 id）原样传给轮次文件；输出写到开发机的 `logs/zgca/<轮次文件名>_<UTC 时间>.log`，结束时追加一行 `round-exit <退出码>`。上一轮启动器还在跑时拒绝再启动。启动器创建实验之前要先做起步策略的冒烟回放，在 A100 上实测约两小时；通过的记录在开发机的 `logs/research/smoke/` 下按种子、几何与起步策略字节保存，下次直接复用。

带 `--fill` 而不是 `--dry-run` 时，启动器改放进 tmux 会话 `autotrade-fill`，每十分钟跑一遍，直到轮次里没有待建的实验。一遍 `--fill` 只建当下空闲槽位与空闲卡容得下的实验，其余留作待建；本机由研究填充 cron（`ops/cron/research_fill.cron`）每十分钟再跑一遍，开发机上没有 crontab，所以由这个循环来跑。每一遍都追加到同一份 `logs/zgca/fill_<轮次文件名>_<UTC 时间>.log`：某一遍的汇总行报 `0 pending` 时循环以 `round-exit 0` 结束；某一遍以非零退出（创建被拒、控制台不回答）时循环以那个退出码结束，不再重试，留给操作者处理。每一遍开始前先量一次个人存储：配额下的余量不足 40 GiB 或者量不出来（见[存储布局](#存储布局)）时，这一遍不跑启动器，只记一行，例如 `<时间> fill held, no experiment is created this pass: storage: 61.0 of 100 GiB used, 39.0 free (du of /personal_fast_storage), below the 40 GiB floor`，量不出来时冒号后是 `storage: unknown, du of /personal_fast_storage failed (exit 1): <du 的报错>`，十分钟后的下一遍再量。它与 `autotrade-round` 分开，之后别的轮次的 `run` 既不等它也不会结束它；同一时间只有一个填充循环，要提前停下就 `~/zgca-compute/dev tmux kill-session -t autotrade-fill`。

`resume <实验 id>` 让控制台续跑一个 worker 已不在、却没有完成的实验：以 failed 结束的，或者停在被终止、被中断（例如实例重启后）状态的；completed 和 stopped 的拒绝。前提与 `run` 相同，开发机上次部署的提交就是本机的 HEAD；另外实验当前不在跑，配额下的余量也不低于填充循环的 40 GiB 下限。任何一条不满足都直接拒绝，什么也不改。核对通过后它像 `run` 一样确保控制台在跑，删掉本机这个实验的同步标记（见下文 `sync`），再在开发机上向控制台发它自己的续跑请求（`POST /api/experiments/<实验 id>/control`，`{"action": "resume"}`），打印控制台的回答；回答里没有新起的 worker 就以错误退出。删掉标记是为了实验再结束时被定时同步重新取回；控制台拒绝时标记也已删掉，下一遍同步只是再拷一次，结果不变。续跑接着用实验的沙箱工作区，所以可续跑的实验什么都不释放（见 `sync`）。实例重启之后，先用 `resume` 续跑被中断的实验，再启动填充循环。

`status` 先用 `instance show` 打印实例状态与卡时；实例在运行时再用 `dev report` 打印 GPU、根盘与个人存储余量和 tmux 会话。其中个人存储的余量是整个共享文件系统的，看不到配额，所以接着打印一行配额下的用量，例如 `storage: 56.2 of 100 GiB used, 43.8 free (du of /personal_fast_storage)`，低于下限时末尾加 `, below the 40 GiB floor`，量不出来时是 `storage: unknown, ...`。然后打印最近一次部署的提交、控制台健康（含空闲槽位与空闲 GPU）、每条在跑实验的状态、阶段与心跳距今的秒数（worker 已不在却没有结束的实验单独标出）、沙箱镜像，最近一份 `create_round_*` 启动器日志的末尾三行（只在它比最近一份填充日志新时显示，更早的是已被填充循环接替的旧轮次；镜像构建日志也在同一目录，不算在内），以及填充循环：会话 `autotrade-fill` 是否在跑、最近一份填充日志的最后一行汇总，结束了的还有它的 `round-exit`。

`sync <实验 id>` 把开发机上这个实验的实验目录（不含 `pit_views/`，那是可从种子重建的 12 GB 缓存）和 `logs/workers/<实验>.log` 用 `dev copy` 拉回本仓库。本地已有同名目录时，只有它的 `hitl/params.json` 记着开发机的实验根（即以前从开发机同步来的副本）才会更新，本机自己的实验一律拒绝覆盖。同步前先打印实验当前的状态；实验还在跑时拒绝同步，因为它的 `status.json` 每几秒改写一次、日志不断增长，拷贝过不了比对，等它结束再同步。“在跑”与 `status`、`deploy`、`down` 和镜像重建的判定都用控制台自己的规则（`hitl_state.running_experiments`）。拷贝完成后，`sync` 再让实验钉住的沙箱镜像 tag（`hitl/sandbox_image.json` 的 `image_ref`）在本机存在，做法与本机实验建 tag 相同：把本机的 `autotrade-sandbox:latest` 打上这个 tag。前提是开发机上这个 tag 的镜像与本机 `autotrade-sandbox:latest` 内置的策略运行时指纹相同，两边各用一个离线、不占 GPU 的容器读出（开发机一侧是 `remote.sh` 的 `image-runtime` 步骤）；指纹不同，或开发机上已没有这个 tag（停机再开可能丢 tag），就不打 tag、以错误退出，说明这个策略在两边镜像一致之前不能在本机跟随。没钉镜像的实验没有 tag 可打。`sync` 把进度记在 `.runtime/zgca/synced/<实验 id>`：开始拷贝前删掉，两次拷贝都核对无误后写 `copied`，打 tag 这一步也成功（或没有 tag 要打）后写 `synced`，所以中途被打断的拷贝不留标记。

拷回来的副本是已完成的实验（它的 `hitl/status.json` 是 completed）时，`sync` 接着在开发机上释放这个实验的大件，成功后标记写 `released`：`pit_views/`、沙箱工作区 `.runtime/sandboxes/<实验 id>`，以及每次尝试结束时归档进来的工作区 `artifacts/run_*/workspace`（单个超过 64 MiB 的文件不归档；本机副本里留着）。实验目录和其余记录都保留，所以填充循环仍把它算作已建，控制台照常列出。释放由 `remote.sh` 的 `release` 步骤做，“已彻底结束”的判定和删除沿用 `scripts/experiments/prune_pit_views.py`（没有活着的 worker、账本有裁决），此外开发机上的状态也必须是 completed；删除的路径都由实验 id 和固定的子路径拼成，id 带路径分隔符或只由点组成、或路径上有符号链接时拒绝，任何一处被拒都什么也不删，每次释放记一行各项释放了多少。failed、被终止、被中断这些可续跑的实验什么都不释放，续跑要用它们的工作区。重复 `sync` 同一个实验结果相同，再次释放什么也不删。

`sync-rounds [轮次文件...]` 是定时同步的一遍，由本机 cron（`ops/cron/zgca_sync.cron`，安装用 `install_cron.py --block zgca`）每 30 分钟跑一次，输出追加到 `logs/zgca/sync_cron.log`。上一遍还占着锁时（一次拷贝约一小时），到点的这一遍什么也不做，只记一行带时间的 `zgca sync skipped: the previous run still holds .runtime/zgca/sync_cron.lock`；拷贝期间有一两行是正常的，接连很多行说明占着锁的那一遍卡住了，要由操作者结束它所在的整个进程组（cron 给每个任务一个组，复用连接的共享 ssh 自成一组、不在其中，不要动它）；同步本身失败时照常记它自己的行，不会记成占锁。它读出这些轮次文件定义的实验：标记是 `released` 的，或者是 `synced` 而副本不是已完成实验的，算已同步，不碰平台；标记是 `synced`、副本是已完成实验的（释放上线之前同步的都是这样）只在开发机上释放；其余的逐个用 `experiment-state` 问一次。要碰开发机时先确认实例在运行。开发机上还没有的算待建，worker 还在跑的算在跑，都不拷贝；worker 已不在却没到结束状态（completed、failed、stopped 之外，例如被终止）的算中断，每遍记一行、不同步，要续跑就 `resume`，要按原样取回就手工 `sync`；到了结束状态的才像 `sync` 一样拷贝、打 tag，已完成的再释放。标记是 `copied` 的（拷贝完了、tag 被拒）每遍只重试打 tag、不重新拷贝，并记一行原因，直到两边镜像一致。被拒的释放算失败、标记留在 `synced`，下一遍再试。实例没在运行或平台查询失败时这一遍只记一行，下一遍再试；某个实验失败只记它一行，后面的照常进行。每遍最后一行是带时间的汇总，已同步、未打 tag、在跑、中断、待建、失败各几个（实例问不到时另加未检查的个数与原因），所以 `tail -1` 连同它的时间就是现状；有失败、未打 tag 或实例问不到时以 1 退出。它不开 Paper 账户。cron 文件里的 `ROUNDS` 列出还有实验要取回的 ZGCA 轮次文件：轮次关闭后仍留在里面，直到它的实验都同步回来，并在轮次文件离开仓库之前移出；不在其中的已完成实验由手工 `sync <实验 id>` 释放。

`down` 就是 `instance stop`：等到 STOPPED 后打印卡时。实例在运行时先做与 `deploy` 相同的检查，有实验或启动器在跑就拒绝，因为停机会结束它们；确要放弃这些实验，直接用 `~/zgca-compute/instance stop <平台实例 id>`。

## GPU 驱动

镜像里的 torch 2.13 要 CUDA 13，开发机的内核驱动却是 535（CUDA 12.2），`--gpus` 把宿主的这份驱动注入容器。A100 是数据中心卡，可以用 NVIDIA 的前向兼容：较新的用户态驱动跑在较旧的内核驱动上。开发机的构建因此带上构建参数 `CUDA_COMPAT_URL`，Dockerfile 的最后一层把 `cuda-compat-13-0` 解到 `/usr/local/cuda-compat`，并只在这时把 `LD_LIBRARY_PATH` 指向它；平台的 GPU 钩子不会自己选用兼容库。本机构建不传这个参数（驱动是 580 以上），那一层是空的。

钉住的是 580.196.07（`zgca.py` 的 `CUDA_COMPAT_*`，摘要与 NVIDIA Debian 12 源的包索引一致；驱动版本与兼容包的探测见工具包的平台记录）。平台若把内核驱动升到兼容包的版本或更高，兼容包就比内核驱动旧，NVIDIA 不支持这种组合，准备步骤的 GPU 检查会失败：那时换钉更新的兼容包，或在 `remote.sh` 里不再传这个参数。

## 幂等与失败

启停的幂等意图（键何时写下、保留或删除）和平台暂时拒绝时的退避重试都由 `instance` 负责，见 `~/zgca-compute/README.md` 的“API 与生命周期”；启停没有确认到目标状态时，重跑同一个子命令即可。每一步都校验自己的结果，失败就以非零退出并说明原因，没有静默回退。一次平台查询 120 秒得不到回答时，`~/zgca-compute` 的入口以一行错误失败，不会无限期地等下去。

## 存储布局

本项目在每个资源区用两处存储。个人存储上的 `/personal_fast_storage/auto-trade` 是这个区的仓库根：代码、`.env`、种子、release、实验和日志都在这里，同一资源区的实例共用这一份。实验的 `pit_views/` 硬链接种子的文件，所以种子和 `experiments/` 必须在同一个文件系统上。

个人存储 `/personal_fast_storage` 是 GPFS 上有 100 GiB 硬配额的一块（工具包的平台记录；`zgca.py` 的 `STORAGE_QUOTA`）。`df` 和 `statvfs` 在那里看到的是整个 3.8 PB 的共享文件系统，所以 `dev report`、控制台建实验时的余量检查和 Agent 工具的余量下限都看不到配额；容器里也没有任何配额工具，`remote.sh` 只能用 `du` 量整棵树，硬链接只算一次，与配额的算法一致，2026-11-20 量一次约一秒。量的结果由 `status` 显示；填充循环和 `resume` 在配额下的余量不足 40 GiB（`STORAGE_FLOOR`）或量不出来时不起新实验，因为一次会话的沙箱工作区长到过 20 GiB，worker 在每次尝试结束时又把工作区（单个超过 64 MiB 的文件除外）归档进实验的 `artifacts/run_*/workspace`。2026-11-20 配额写满：一条跑了 18 小时的会话以 `Disk quota exceeded` 失败，随后的填充又建了同轮的两条实验，都当场失败。已完成的实验同步回本机之后由 `sync` 释放它在开发机上的大件（见上文），其余的配额由种子、release、venv、日志和还可能续跑的实验占着。

预训练模型库放在实例自己的盘上，`/ebs/auto-trade/pretrained`，仓库根的 `data/pretrained` 是指向它的链接。库放在个人存储上会和种子、release 与实验抢这份配额。实验只钉住库清单的摘要，容器启动时解析这条链接、挂载它指向的目录，所以除了这条链接没有别处知道库在哪。容器挂载的只有库和实验的视图；release 只由宿主侧的 worker 读，所以留在个人存储上。

实例盘同时是实例的根和 Docker 的镜像存储，缺省实例的是 100 GiB 的数据盘。它放不下本机原先十个模型的库（133.6 GiB）再加一个约 15 GiB 的镜像，所以两台主机用同一份精简的库：现在是六个模型、约 50 GiB，连同镜像和 20 GiB 余量放得下。

公共存储不放本项目的任何东西：组织的公共对象存储对组织内每个账户可读、没有删除接口，上传还可能产生费用；`/public_fast_storage` 是 POSIX 共享，资源区 5 的任何实例都能以 root 改写。由 TuShare 数据派生的种子和 release 受许可约束，只能放私有存储；开源模型库也不放在那里，它在实例盘上（见上）。2026-10-14 测吞吐时在对象存储 `s-lzp26-probe/` 下留了六个零字节对象，无法删除。

## 一个周期的开销

卡时规则见 `~/zgca-compute/README.md`。全新的根要把种子（十年那颗约 20 GB）、release 里种子要读的部分（约 13 GB）和模型库（约 54 GB）传一遍。按工具包平台记录里的 3.1–3.7 MB/s，种子与 release 要两个多小时，模型库四到五小时；实例盘上已有的模型不再传。之后 `deploy` 只传差异，几分钟内完成；准备步骤冷构建镜像加 GPU 检查约二十分钟，镜像缓存命中时不到一分钟。起步策略的冒烟约两小时；一条 HCM 整期实验约七小时。

## 已接受的局限

`/personal_fast_storage` 按资源区隔离、同一资源区内的实例共用，所以用 `--instance` 驱动的另一台若与缺省实例在同一资源区，两台读写的是同一个仓库根目录：代码、`.env`、种子、实验目录和日志都是同一份，一台上的 `deploy` 改的也是另一台正在用的代码，一台的 `status` 会把另一台在跑的实验标成 worker 已不在，`deploy` 和 `down` 的在跑检查也看不到另一台的实验。两台同时跑实验时要放在不同资源区，或者同一时间只让一台有实验。

实例盘上的模型库和镜像停机再开后仍在，这是观察而不是保证；丢了就要重传模型库。

`deploy` 只在仍含受管文件的顶层目录里清理：整个退出版本库的顶层目录和仓库根下不再受管的文件仍留在开发机上。镜像只在合同指纹说它缺失或过期时重建，所以只改 Dockerfile、合同文件不变的改动不会让实例重建，要先在实例上删掉镜像 tag 再部署。`deploy` 本身不重启控制台，下一次 `run` 在没有实验运行时才重启它；部署之后若先有实验经旧控制台建起（例如填充循环），新的建实验校验或参数表要等它们结束、再 `run` 时才生效。`status` 只认 `create_round_*` 与 `fill_create_round_*` 命名的启动器日志。

`deploy` 只镜像这次点名的种子和它的 release，从不删除开发机上以前部署的种子与 release，而它们都占 100 GiB 的个人存储配额：2026-11-10 两份种子（各约 20 GB）、两份 release（各约 13 GB）连同实验与 smoke 暂存把配额写满，一遍填充的 smoke 以 `Disk quota exceeded` 失败。换种子之后，等开发机上没有未结束的实验再钉旧种子和旧 release，就手工删掉它们：`~/zgca-compute/dev rm -rf /personal_fast_storage/auto-trade/data/<旧种子目录> /personal_fast_storage/auto-trade/data/research_releases/<旧 generation>`。实验的 `pit_views/` 硬链接着种子的文件（可从种子重建，`sync` 也不拉回）；已完成实验的那份在同步后已释放，其余结束状态的实验还留着它，要连这些 `pit_views/` 一起删掉才真正腾出配额。

模型服务的 key 在实例存在期间一直留在开发机的 `.env` 里（仅 root 可读），停机不清除；其中本机 vLLM 网关的 key 只有在某个对话角色跑在本机模型上时 worker 才要求，开发机上没有这个网关，所以用不上。`sync` 不拉回开发机上的模型服务会话日志（`data/llm_conversations/`）：日志按服务方、模型、日期和 worker 进程号分文件，几个实验写在同一棵目录里，续跑换了进程号的旧文件也无从按实验认出，一条 gpt-6 实验就有数百 MB；要用时整棵取回，`~/zgca-compute/dev rsync -a dev:/personal_fast_storage/auto-trade/data/llm_conversations/ <本地目录>/`（有实验在跑时目录仍在增长，所以不用 `dev copy` 比对）。

在实例上定版、同步回本机的策略和本机的一样可以开 Paper 账户跟随：建账户只按本机的实验目录重建设置，不读参数里记着的实例实验根和沙箱根；控制台对标为“通过”的策略显示建账户的按钮。建账户时本机还要核对两件事，缺一就拒绝且什么都不写：注册的策略字节与账本为它记下的指纹一致，钉住的镜像 tag 在本机存在并内置本机代码要求的策略运行时，这个 tag 由上面的 `sync` 或定时同步打上。

定时同步对每个实验只取回一次：`resume` 删掉标记，经它续跑的实验结束后会再取回；在控制台页面上直接续跑的不会，结束后要手工 `sync`（手工 `sync` 也重写标记）。这样的实验在那之前不会被释放，因为本机副本还是它续跑前的状态。

配额的检查只是某一时刻的量：40 GiB 是留出的余量而不是保证，已经在跑的会话仍可能把配额写满；`du` 遍历时正赶上会话删除文件会报错，那一次按量不出来处理（`status` 显示 unknown，填充那一遍不建）。释放只覆盖同步回来的已完成实验，failed 和 stopped 的实验、旧种子与 release、对话日志都不在其内。worker 在每次会话结束时把工作区复制（而不是移动）进实验的 `artifacts`，所以那一刻要另有一份工作区大小的余量。

开发机上的容器遥测缺少峰值内存和 GPU 峰值，开发机上的一切又都以 root 运行，原因见工具包 README 的“火山平台 Docker”；所以在开发机宿主上跑单元测试时，依赖目录权限位的测试会失败，Docker 沙箱以 uid 61000 运行，不受影响。离线测试在 `tests/unit/test_zgca_workflow.py`，用模拟的 `instance`、`dev` 和本机 `docker`，不访问网络也不启停实例；那些入口本身的测试在 `~/zgca-compute/tests`。
