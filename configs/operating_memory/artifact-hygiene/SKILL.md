# 冻结候选只含在用的代码

被提名的是节点的不可变快照：研究期之后宿主原样回放它，毕业后进入 Paper，再没有 Agent 改它。`modification_check` 只拒绝它看得见的错（隐藏路径、`__pycache__`、符号链接、不支持的 import）；多出来的模块和死分支不让任何检查失败，却计入指纹与文件、字节上限，跟着被冻结，也误导之后读它的人。

## 什么时候清

- 在候选第一次正式验证之前清，清完的字节就是要验证、要提名的字节。试验按字节计：探针之后再改任何一个字节（删死代码、改注释都算）然后上整期，就是另一个试验，要重新验证才能提名。
- 已验证的节点不再顺手清理。清理是写候选时的一部分：不单独派子代理，不在提交前改写别的候选目录。

## `__pycache__`

在脚本里 import 候选目录或 `output/` 里的模块，Python 会在被 import 的文件旁边写 `__pycache__`，与当前目录无关；`python -m py_compile` 不论加不加 `-B` 都会写。冒烟与批次先跑的 `modification_check` 随即以 `hidden or runtime file is forbidden` 拒绝。

- 要 import 时用 `python -B` 运行；只查语法用 `python -B -c "import ast, sys; [ast.parse(open(p, encoding='utf-8').read(), p) for p in sys.argv[1:]]" <文件…>`，不用 `py_compile`。
- 已经留下的：`find output candidates -name __pycache__ -prune -exec rm -rf {} +`。

## 清什么

- 恒为假的分支、只被它调用的函数、不再被引用的参数与常量：删掉，不留开关，需要时再写；删之前先证明零引用（全包只剩定义处）。
- 一轮被证伪的变体，在下一轮的候选里拆掉接线，不把半截实现留在胜者里。
- 改了窗口、阈值或口径就同步改注释与 docstring：写着一个数、算着另一个数的注释比没有更糟。
