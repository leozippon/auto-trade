# 两套路径写法不要混用

同一个「工作区」在两类工具里是两种写法，这是新会话、尤其是新子代理最常踩的第一个坑。

- `shell` 跑在沙箱真实文件系统上：`cwd` 是工作区内的相对路径（默认就是工作区根），而命令自己读的数据路径是真实绝对路径，只读挂载点用 `/mnt/...` 写。
- `read_file` / `write_file` / `edit_file` / `grep` / `glob` 用的是「授权根 + 该根下的相对路径」，这是一个虚拟地址而不是文件系统路径；填成绝对路径的信号是「路径必须相对于所选根」。

| 目标 | `shell` 里怎么写 | 文件与搜索工具怎么写 |
| --- | --- | --- |
| 工作区里的探数脚本 | `notes/probe.py` | `root="workspace"`, `path="notes/probe.py"` |
| 正式策略入口 | `output/main.py` | `root="output"`, `path="main.py"` |
| 快照里的日线域 | `/mnt/snapshot/daily.parquet` | `root="snapshot"`, `path="daily.parquet"` |

## 做法

- 写脚本和运行脚本用同一个不带 `workspace/` 前缀的相对路径。`write_file` 会把多余的 `workspace/` 前缀吃掉，`shell` 不会：两边写法不一致，写入成功、运行时却「文件不存在」。以 `write_file` 返回的那个路径为准。
- 可写根只有 `workspace` / `output` / `models`；快照、父产物、结果、步骤树等根都是只读的，填给写入工具会被直接拒。
- 先用 `glob` 在授权根内定位，再用 `read_file` 分页读。大结果会落盘并返回可续读的根与路径，按它续读，不要重跑一遍再截断一次。
- 正式策略里不写死任何宿主路径：运行时只用 `context.snapshot_dir` / `context.asof_dir` / `context.state_dir` / `context.models_dir` 拼出域名目录，且这个拼接必须是读写调用的首个位置参数。
- 委派时留意：描述子代理可写范围时用的是沙箱绝对路径，而它的第一次 `read_file` 很容易照抄那个前缀。task 里要么把两套写法都写清，要么只给相对路径。

## 教训

路径与根名类错误在多个折里反复出现，单个折一度到两位数。最典型的一幕是：刚启动的子代理第一轮就把绝对路径填给 `read_file`，而它读到的委派文本里正是那个绝对路径——两处都没错，只是分属两套写法，中间从没有人把它们并排放过。
