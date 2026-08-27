# Repository Guidelines

## Repository Principles

- Maintain a minimalist code architecture and implementation while ensuring logical correctness and completeness.
- Achieve optimal performance while keeping the environment as close to real-world conditions as possible.
- Maximize the Agent's autonomy while lowering the complexity of interactions between the Agent and the environment.


## Repository Guardrails

- Treat resource checks and logging as mandatory steps, not optional cleanup.


## Environment

- The Python environment is `quant` at `~/miniconda3/envs/quant` with Python 3.11; conda is not initialized in non-interactive shells.
- The Docker sandbox Python is independent from the local conda environment; rebuild `ops/docker/sandbox.Dockerfile` when sandbox dependencies change.


## Resource Checks

- Check GPU memory and system memory before and after each experiment, training, inference, evaluation, or data-processing run:

```bash
nvidia-smi
free -h
```

- Routine read-only inspection, small documentation edits, prompt export, formatting checks, and targeted lightweight unit tests do not require these checks unless they are expected to be resource-intensive.
- Stop and adjust the workload if memory usage becomes unsafe.


## Logging

- Record logs promptly for every training, inference, evaluation, or data-processing run.
- Use `LOGBOOK.md` as the concise current logbook: keep entries short and focused on what was tried, the key result, and the current conclusion.
- Use `docs/logbook/DETAILED_LOGBOOK.md` as the detailed durable logbook: include the date, task, key command or config, resource checks, important artifact and log paths, and final result or conclusion.
- For routine context gathering, read `LOGBOOK.md` first; read `docs/logbook/DETAILED_LOGBOOK.md` only when detailed historical commands, paths, or experiment context are needed.
- Runtime log files under ignored `logs/` may be kept locally for debugging.


## Living Documentation

- Treat the current design docs as the communication layer between audit, research decisions, and implementation.
- Keep five authoritative living docs aligned by scope:
  - `docs/data-documentation.md`: data sources, downloads, audits, PIT availability rules, unit rules, and known data risks.
  - `docs/agent-design.md`: Agent-visible inputs, writable strategy artifacts, prompt protocol, tool usage semantics, and forbidden behavior.
  - `docs/environment-design.md`: PIT snapshots, Sandbox/runtime paths, trusted tools, Broker/backtest/NL scoring, LLM API boundary, and run logs.
  - `docs/pipeline-design.md`: Fold/Epoch/Held-out orchestration, artifact handoff, freeze/fallback rules, ledgers, and reporting.
  - `docs/deployment-documentation.md`: deployment surfaces — the local research console, Paper paper-trading page, and the live-trading page that remains frontend-only and empty.
- Keep `docs/parameters-reference.md` as a derived quick reference, not a sixth authoritative design doc. When defaults, CLI/config knobs, Broker profile fields, replay budgets, sandbox/tool limits, data-task limits, or QMT constants change, update the relevant authoritative doc and this parameter reference in the same work item. Code and run/snapshot manifests remain the source of truth for actually effective values.
- When a change materially affects one of these areas, update the relevant document in the same work item. Do not rely on code or logs alone to communicate a changed design, data contract, operating procedure, or parameter default.
- Keep these documents concise and current: describe the latest accepted state rather than earlier attempts, old names, or superseded workflows. Detailed historical traces belong in the logbooks, and obsolete version labels or migration notes stay out unless still operationally relevant.
- Add a new durable document under `docs/` when a new project area becomes important enough that the existing documents would become confusing or overloaded.


# 全局准则

## 多智能体协作

*若任务提示标明你是子代理，忽略本节其余规则，且不要再派生子代理。*

*除非任务非常简单，否则启动多智能体协作。*

- 你的职责是抽象设计、全局协调与最终验收。只有必要时才亲自做穷尽阅读和修改。
- 有意压缩自己的上下文占用，以保持端到端推理和架构判断连贯。
- 启动子代理时，在其任务提示中标明它是子代理，使其忽略本节。
- 委托只一层。设计子代理任务时，须能在不再委托的情况下完成。
- 日常读库和信息搜集，交给中等能力子代理。
- 审计和根因定位，交给最高能力、中等推理强度的子代理，避免无效过思和臆测铺陈。
- 关键文档与核心代码的审阅、开发、修改，交给最高性能子代理以保证保真执行。
- 有意选择子代理的上下文：全新窗口切断路径依赖，便于独立重看；继承上下文则延续先前对齐的推理。
- 向全新上下文的子代理委托前，必须让它拿到当前 `AGENTS.md`。
- 在已有子代理的后续任务和新拉起之间取得平衡；既不要为碎片任务频繁丢弃短命子代理，也不要把单个子代理推到上下文上限。
- 并行子代理范围互斥；必须接触同一区域的工作串行。
- 仅在不再需要或明显跑偏时中断子代理，不要只为催促而打断。
- 不要轮询正在运行的子代理；那会空耗上下文。去做独立工作，或等到结果回来。
- 若子代理必须等待，应让它 Sleep；否则它会在等定时器时让出并退出。
- 已定结论带入后续审查，不要无故重开。
- 不要进行不必要的迭代审计，容易陷入空转。

### AutoTrade Fold 与 Meta 会话

本节与「多智能体协作」「开发原则」「操作护栏」一并注入普通 Fold 和 Meta 的系统提示词。

| 角色 | 策略与模型 | PRIOR | 共享 skills | 正式回测与结束 |
| --- | --- | --- | --- | --- |
| Fold 父 Agent | 可写；设计、实现、协调、验收 | 只读 | 可写 | 可回测、可结束 Fold |
| Fold `developer` / `general-purpose` | 可写 | 不可 | 可写 | 否 |
| Fold `auditor` / `Explore` | 只读 | 不可 | 只读 | 否 |
| Meta 父 Agent | 可小幅正则化 | 唯一可写 | 可写 | 不可回测；可结束 Meta |
| Meta 任一子角色 | 只读提议 | 不可 | 只读 | 否 |

- `explore` 只一层。除非任务非常简单，否则父 Agent 应当委托。子代理不得嵌套、正式回测、结束会话、修改 PRIOR 或自行验收；由父 Agent 验收。Fold 父 Agent 不得用 shell 修改策略产物。
- 每个 Fold 和 Meta 从 `inputs/skills_index.json` 起步，再按需读取 skill 正文、PIT 可见数据、单位引用、制品和 how-to。可写角色把可迁移知识写入 `skills/<kebab-name>/SKILL.md`，不要堆进提示词或 PRIOR。skill 脚本不会自动执行，skills 不进入策略、revision、frozen、Test 或 Held-out。
- PRIOR 由 Meta 维护、Fold 只读。只保留简洁的策略方向、编排和 skill 路径引用；不要复制 skill 正文、目录、how-to 或 raw trace。没有有效改进时保持原文。首版必须非空，且遵守长度、日历、Test 与 Held-out 泄漏规则。
- PIT `available_at`、Test/Held-out 隔离、`generate_orders(context)` JSON ABI、写权、结束门和不透明引用保持 fail-closed。历史分钟和竞价只作证据或精确价格来源，不是策略时钟。不得伪造工具、Validation 或完成结果。

## 开发原则

- **实现原则**：只实现并保留当前需求所证明的最小完整方案。偏好简单、直接、优雅的设计；避免没有现成证据支持的泛化、冗余守卫和功能。
- **审计原则**：冻结范围，定义必须始终成立的行为与条件；要求可复现的实质影响证据，区分缺陷、建议和已接受限制；除非另有指示，权衡收益与增加的复杂度和冗余。不要把低影响风险做成不成比例的机制；仍须暴露实质缺陷和低成本修复。
- **修复原则**：每次小而自洽的改动只修一个根因，并让代码库整体更健康；复杂度持续膨胀时，重构根因而不是叠例外。
- **失败原则**：正确性无法保证时，快速显式失败，而不是静默回退或报告假成功。
- **测试原则**：测试必须始终成立的条件、负路径和真实端到端行为，而不是只测当前实现的快乐路径。
- **克制原则**：如实记录不可消除的限制；不要把未支持行为伪装成兼容或恢复。
- **单一来源原则**：共享且定义行为的信息只保留一个来源。仅在组件无法共享时复制，仅在分歧会实质影响正确性或运行时才做一致性检查。

这些原则冲突时，先保住明确需求、正确性和诚实失败；然后选最简单的完整实现。

## 操作护栏

- 保持仓库整齐、干净。
- 在写或改代码之前，读够相关代码和配套文档，形成可靠设计。
- 保持独立判断。当请求与证据、文档要求、安全约束或更高优先级指令冲突时，及时提出。
- 删除共享代码、持久数据、公开接口或操作入口之前，先查清谁在用。
- 开发中不要反复打补丁。同一组件需要反复修复时，停下来重新评估底层设计。只有根因重构是当前需求下最小完整方案时才做。
- 避免过多测试用例和过度依赖 mock。保留验证必要行为与失败路径的测试，必要时做真实测试。


## Documentation

- Before a substantial addition or restructuring, identify the document's purpose and scope, then read the full affected document and any relevant neighboring documents. Integrate the change into the existing narrative.
- When accumulated patches have obscured the document, reorganize instead of appending another fragment.
- Apply the Single-Source Principle across documents. Keep each fact, design decision, and procedure in one authoritative section. Use links or brief pointers elsewhere; when sections overlap, clarify their boundaries instead of repeating the same content.
- Write Markdown prose as logical lines; do not hard-wrap it at 80 columns.
- Keep documentation aligned with current behavior.
- Use filename casing as a soft audience convention: retain ecosystem-standard names such as `README.md`, `LICENSE.md`, and `CHANGELOG.md`; use `lowercase-kebab-case.md` for ordinary user-facing documents and `UPPER_SNAKE_CASE.md` for agent, process, or internal-control documents.
- Structure user-facing documentation for human readability in plain, approachable language. Follow these practices:
  - Use headings only for meaningful divisions at the same level, group related ideas together, and move from overview to detail and normal use to exceptions where that order fits.
  - Maintain a clear logical progression and fluent, natural language within and between sections.
  - Keep the content as concise as possible without sacrificing logical completeness.
  - Use tables and lists for genuinely parallel information; use concise prose for reasoning, sequences, and qualifications.
  - Avoid canned introductions, repetitive summaries, excessive headings, artificial parallelism, and unnecessary bold emphasis.
  - Avoid unnecessary abstraction and redundant concepts; retain necessary standard technical terminology.
  - Omit variable names, filenames, and similar details unless necessary.
  - Keep unnecessary cross-references to a minimum.


## Git and Delivery

- Keep commits focused and self-contained. Code, tests, and living documentation for the same behavior change should usually be committed together.
- Use concise imperative commit subjects, preferably in English for tooling and search consistency. Add a short body when validation or operational impact matters.
- Before committing, remove generated caches such as `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `*.pyc`, and `*.pyo`; never commit runtime logs, local state, data dumps, API keys, scratch notebooks, or ignored artifacts.
- Run `git status` before and after changes, review `git diff --cached` before committing, and leave unrelated local changes unstaged.
- Pull and rebase or merge carefully before pushing when the remote branch has moved. Do not rewrite shared history, force-push, or use destructive Git commands unless explicitly approved.
- Commit and push to the configured remote repository as needed; prefer SSH Git operations and keep `origin` aligned with the canonical remote.
- Keep `main` as the only long-lived branch. Temporary branches are allowed when useful and must ultimately be merged into `main`.
