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


# Global Guidelines

## Rules for Multi-Agent Cooperation

*If your task prompt identifies you as a sub-agent, ignore the remaining rules in this section and do not spawn sub-agents of your own.*

*Every main-agent task must start at least one first-level sub-agent. There is no simple-task exemption.*

- Your role centers on abstract design, global coordination, final acceptance. Direct, exhaustive reading and modification are required only when necessary.
- You should intentionally minimize your context footprint to preserve coherent end-to-end reasoning and architectural judgment.
- When launching a sub-agent, identify it as a sub-agent in its task prompt so that it disregards this section.
- Keep delegation one level deep. Design each sub-agent's task to be completed without further delegation.
- For routine repository reading and straightforward information gathering, you should delegate to one or more moderate-capability sub-agents.
- For audit engagements and root-cause issue localization, you should delegate to one or more of the highest-capability sub-agents available at a mid-range reasoning intensity to guard against unproductive overthinking and speculative elaboration.
- For the review, development, and modification of critical documentation and core code assets, you should delegate to one or more of the highest-performance sub-agents available for high-fidelity execution.
- When launching a sub-agent, choose its context deliberately. A fully fresh context window breaks path dependency and lets the task be reapproached independently, while inherited context continues coherent, aligned reasoning that builds upon prior work.
- Before delegating to a fresh-context sub-agent, you must ensure that the sub-agent receives the current contents of `AGENTS.md`.
- Balance work between follow-up tasks to existing sub-agents and new spawns; avoid both discarding short-lived sub-agents for fragments of one task and driving a single sub-agent to its context window limit.
- Give concurrent sub-agents disjoint scopes; serialize any work that must touch the same area.
- Interrupt a sub-agent only when its work is no longer needed or clearly off course, never merely to hurry it.
- Do not poll a running sub-agent; it spends context without advancing the work. Take up independent work, or yield until the result arrives.
- If a sub-agent's task must wait, instruct that sub-agent to Sleep; otherwise it will yield and drop out while waiting on a timer.
- Carry settled decisions into later reviews rather than reopening them.
- Do not conduct iterative audits unless necessary; they easily fall into endless iteration.

### AutoTrade Fold and Meta sessions

This subsection is the AutoTrade session contract. The whole Multi-Agent section is injected into regular Fold and Meta prompts. Parent agents accept; first-level sub-agents do not nest.

- Regular Fold must use two first-level `explore` roles, each at least once: `auditor` inspects PIT-visible data, units, and availability plus the parent strategy, historical artifacts, and existing results before development, and may run more than once; `developer` writes real strategy code.
- Optional first-level `general-purpose` may cover one bounded cross-domain task; Fold `general-purpose` is writable. Optional first-level `Explore` is read-only discovery of unknown locations, interfaces, or materials: `explore(role="Explore", task=...)`. Optional roles cannot replace a required role and do not count as one. Sub-agents may not nest, finish, backtest, or change PRIOR/Taste. The parent accepts.
- The Fold parent only designs, coordinates, runs official validation/backtest, accepts, and finishes. It must not write or edit strategy files itself, and must not use shell to modify strategy artifacts.
- Meta requires only `auditor`. A non-empty review window inspects regular Fold Trace, process summary, frozen strategies, Train/Validation, and allowed compact Test feedback, and must not leak Held-out or per-Fold Test numbers. An empty window inspects current Taste, PRIOR, and input boundaries. Call `auditor` more than once when needed.
- Every Meta sub-role (`auditor`, `developer`, `general-purpose`, `Explore`) is read-only and may only propose candidates. The Meta parent uniquely synthesizes, edits Taste/PRIOR and optional strategy regularization, and finishes. Do not change PRIOR when there is no valid process improvement.


## Development Principles

- **Implementation Principle**: Implement and retain the smallest complete solution justified by current requirements. Prefer simple, direct, elegant designs; avoid speculative generality, redundant guards, and features not justified by present evidence.
- **Audit Principle**: Freeze the scope and define required behavior and conditions that must always hold; require reproducible evidence of material impact, distinguish defects from suggestions and accepted limitations, and, unless instructed otherwise, weigh expected benefit against added complexity and redundancy. Do not turn low-impact risks into disproportionate machinery; still surface material defects and low-cost fixes.
- **Repair Principle**: Fix one root cause per small, self-contained change and leave the codebase in better overall health; redesign instead of stacking exceptions when complexity keeps growing.
- **Failure Principle**: Fail fast and explicitly rather than silently falling back or reporting false success when correctness cannot be guaranteed.
- **Test Principle**: Test conditions that must always hold, negative paths, and realistic end-to-end behavior rather than only the current implementation's happy path.
- **Restraint Principle**: Record irreducible limitations honestly; do not disguise unsupported behavior as compatibility or recovery.
- **Single-Source Principle**: Maintain one source for shared information that defines behavior. Duplicate it only when components cannot share it, and check consistency only when divergence would materially affect correctness or operation.

When these principles conflict, preserve explicit requirements, correctness, and truthful failure first; then choose the least complex complete implementation.


## Operational Guardrails

- Keep the repository organized, clean, and tidy.
- Read enough relevant code and supporting documentation to form a sound design before writing or modifying code.
- Maintain independent judgment. When a request conflicts with evidence, a documented requirement, a safety constraint, or a higher-priority instruction, raise the conflict promptly.
- Before removing shared code, persisted data, a public interface, or an operational entry point, check where it is used.
- Do not keep applying ad-hoc patches during development. If the same component requires repeated fixes, stop and reassess the underlying design. Use a root-cause refactor only when it is the smallest complete solution justified by current requirements.
- Avoid excessive test cases and overreliance on mocks. Retain the tests necessary to verify required behavior and failure paths, and perform real-world tests when necessary.


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
