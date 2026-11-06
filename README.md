# ADMCubeQuant

ADMCubeQuant is a research system in which an LLM agent designs and tests A-share trading strategies on its own, under conditions that stay close to a real research workflow.

The agent works inside a networkless Docker sandbox. Everything it can read is a point-in-time snapshot built from a local TuShare data lake: every row carries the timestamp at which it actually became available, so a strategy can never see a number that did not exist yet at the moment it claims to decide. The agent writes a strategy package exposing `generate_orders(context)` (and optionally `fit(context)`), backtests it against a simulated A-share broker (price limits, suspensions, lot sizes, stamp duty), and iterates.

An experiment comprises one research session on fixed dates, a validation period the session lives through step by step, at most one registered strategy with its validation record, and one continuous forward backtest. The research period is whole years (ten August-to-July years in the current profiles). The session reads data only up to the end of its first research years (5, 6 or 7) and backtests candidates there with `validate`; the remaining years are the validation period. Each `test` replays one candidate as a fresh account from the validation start to the next step (a quarter by default) and records that newest step out of sample, so the validation years reach the agent only through test results. A candidate is tested only when it registers: it covers the whole research span, is not a control, has finite metrics, stays within the drawdown limit and, when it trains a model, names a seed replicate. The last tested candidate is the registered strategy. The session works under experiment-level budgets of inference time, randomized controls, model calls and hosted-model API spend (50 USD by default, priced from the model catalog; local models are free); it compacts its own context on the host's notice and can read its own transcript back, and an attempt that fails is resumed in place from its test horizon, its last compaction summary, its workspace and the tree of validated candidates. Every validation and test reports the statistic the readout uses, and no pass or fail.

The registered strategy and its seed replicates are then replayed without the agent as one continuous account over the twelve months after the research period and the Held-out months. Every reading is taken on the active return, the strategy's daily return minus random baselines matched to its own trades, regressed on market, size and value: the readout states the validation record's residual information ratio against the threshold z(1 − α)/√years with its one-sided p-value, and the forward reading's lower bound and residual, pooled across registered strategies. Nothing passes or fails. The operator may open a paper-trading account for a registered strategy whose forward replay completed. A local web console drives this process.

Live trading is deliberately not implemented. The console has a live-trading page, but it is an empty frontend placeholder: no backend, no broker connection, no order path.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/autotrade/agent/` | Agent session runner, prompts, sub-agent and context-compaction machinery |
| `src/autotrade/environment/` | PIT snapshot building, unit registry, sandbox, trusted tools, broker and replay engine, LLM gateway |
| `src/autotrade/pipelines/` | Research calendar, sessions, the validation record and its readout, strategy registration, forward evaluation, ledger, interactive worker |
| `src/autotrade/data_sources/tushare/` | TuShare download, audit and scheduled-update logic |
| `src/autotrade/webui/` | Console backend and static frontend |
| `src/autotrade/paper/` | Paper-trading engine |
| `scripts/` | Entry points: `webui/`, `experiments/`, `data/`, `paper/`, `dev/` |
| `configs/` | Update schedule, column inventory, exported prompt snapshot, strategy output template, reference workspaces |
| `ops/` | Sandbox Dockerfile, cron templates, nginx deployment configs, the ZGCA A100 instance workflow (`ops/zgca/`) |
| `tests/unit/` | The whole test suite |

Runtime directories (`data/`, `experiments/`, `logs/`, `results/`, `.runtime/`) are created locally and are not tracked; pytest and ruff keep their caches under `.runtime/cache/`. Worker stdout/stderr for a console-launched experiment is appended to `logs/workers/<experiment_id>.log`.

## Setup

The Python environment is a conda env named `quant` on Python 3.11. Install the package with the console extras:

```bash
conda activate quant
pip install -e '.[webui]'
```

Build the sandbox image before running anything that executes strategy code or starts an agent session. The build context must be the repository root:

```bash
docker build -t autotrade-sandbox:latest -f ops/docker/sandbox.Dockerfile .
```

Copy `.env.example` to `.env` and fill in the credentials you need: a TuShare token for data, and a model endpoint. Model serving is external to this repository — the default local Qwen model is reached through a gateway at `VLLM_BASE_URL`, and hosted models (DeepSeek, Xiaomi MiMo, and GPT-6, Claude and DeepSeek ids through the LightingTheWord relay) can be selected per experiment and per role. The console fails to start a session rather than silently falling back when a selected model has no key.

Data is not bundled. The lake is populated by the TuShare download and audit scripts under `scripts/data/`, normally installed as a nightly cron job via `ops/cron/`.

## Running

Start the console on the machine that holds the data, the Docker daemon and the model endpoint:

```bash
python scripts/webui/run_webui.py
```

It listens on `127.0.0.1:38888` and refuses any non-loopback bind. Create an experiment from the homepage: its research, forward and Held-out dates, the models and the budgets. The console then spawns a detached worker process, and the experiment detail page shows live status, agent traces and validation results, the forward verdict once the replay has run, and the controls for pausing, stopping, restarting and injecting a message into the running session.

An experiment that needs a GPU is placed on a card that keeps `AUTOTRADE_GPU_MIN_FREE_GIB` GiB free (12 by default), not only on a card nobody uses, so a host whose cards all carry other people's work still runs experiments; a card a running experiment already holds is never taken. Set the margin in the console's environment, and restart the console for it to take effect. Use the same setting for round-launcher commands, whose startup smoke test uses the same admission rule:

```bash
AUTOTRADE_GPU_MIN_FREE_GIB=12 python scripts/webui/run_webui.py
```

It is an admission margin, not a cap on the strategy: nothing reserves the memory, and the experiment shares the card with whatever else runs on it.

Experiments that belong together are created from a checked-in round file, which fixes their shared dates, dataset selection and prebuilt view cache; `--dry-run` validates all of it offline before anything is sent to the console:

```bash
python scripts/experiments/create_round_<round id>.py 38888 --dry-run
```

The same worker can be launched by hand for a headless run, and two smaller entry points exist for narrower work — replaying a single strategy against a daily parquet, and running exactly one research session in isolation to inspect its prompts, traces and artifacts:

```bash
python scripts/experiments/run_interactive_experiment.py --help
python scripts/experiments/run_experiment.py --help
python scripts/experiments/run_audit_session.py --help
```

Run the tests with plain pytest:

```bash
python -m pytest -q tests/unit
```

## Documentation

`AGENTS.md` is the contract for anyone — human or agent — working in this repository: development principles, documentation rules, resource checks and the rules for multi-agent work.

The design documentation lives in `docs/` and the project log in `docs/logbook/PROJECT_LOG_YYYY-MM-DD.md`, one file per day. All of these, `AGENTS.md` included, are deliberately kept local and are excluded from version control, so a fresh clone will not contain them. Start with `docs/summary.md`, the owner's single design document: one table per area (Data, Environment, Agent, Pipeline, WebUI), the known limits, and a pointer to the detailed documents, which use `UPPER_SNAKE_CASE` names. Six documents are authoritative, each owning one area: data sources and PIT rules, agent-visible inputs and protocol, the environment and broker, the research pipeline, deployment, and — in `docs/RESEARCH_LESSONS.md` — the research-lessons register: settled lessons, closed directions with the evidence that closed them, and the open hypothesis space, which is what each new round is designed from. The rest are derived: a quick reference for parameter defaults and a unit table generated from the code.
