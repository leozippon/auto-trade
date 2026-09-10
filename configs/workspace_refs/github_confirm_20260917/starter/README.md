# 本包没有 starter

继承来的产物**就是** starter。

实验创建时 `inherit_from` 已经把源实验最新冻结 Fold 的 `output/` 与 `models/` 整棵拷进本实验并锁为只读快照，首个 Fold 的父产物就是它，你的工作区 `output/` 从它起步。它是一份经过完整 Validation、通过全部硬门、并且在多个新季度上被父本对照重放过的真实产物——13 个文件、约 1,600 行，包含它自己 25 KB 的 `README.md` 与逐模块的 docstring，讲清了每一列特征的构造、单位、PIT 规则与已知失效模式。

再给一份合成骨架只会有害：

- 它会与真正的父产物形成**第二份**机制描述，而本臂的全部纪律就是「机制冻结、只改登记的那一处」。两份描述一旦分叉，就没人知道该信哪份。
- 本臂的两个变体各自只改一个地方——`lib/trade.py` 里 `build_orders` 的换名规则，或同一函数里 `execute_at` 那一个字符串。这两处改动都应当直接在父产物的副本上做，改完与父本逐字 diff 应当只剩登记的那一处。从骨架重写反而制造出无数计划外差异。
- 冒烟不能省。`smoke_backtest` 是进 `batch_validate` 之前的必经门，源实验有过真 bug 正是被它在 0 槽拦下的。任何离线的手写 harness 都不能替代它。

所以起步动作是：读 `output/README.md` 与 `lib/` 各模块的 docstring，对未经改动的父产物跑一次 `smoke_backtest`（见 `exploration-plan.md` 第 0 步），确认它在本轮视图上原样跑得动且没有落进 `fallback_ewsign`，然后再动那一行。
