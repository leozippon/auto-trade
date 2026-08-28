# ADMCubeQuant

ADMCubeQuant is a research system in which an LLM agent designs and tests A-share trading strategies on its own, under conditions that stay close to a real research workflow.

The agent works inside a networkless Docker sandbox. Everything it can read is a point-in-time snapshot built from a local TuShare data lake: every row carries the timestamp at which it actually became available, so a strategy can never see a number that did not exist yet at the moment it claims to decide. The agent writes a single-file strategy exposing `generate_orders(context)`, backtests it against a simulated A-share broker (price limits, suspensions, lot sizes, stamp duty), and iterates.

Research is organised as a rolling experiment. A Fold is one development session over one test period; an Epoch is a full sweep of folds; a Meta session periodically reviews completed folds and rewrites the standing research direction the next folds start from. Each fold either freezes a new strategy or keeps the previous one, and a held-out period is scored exactly once at the very end. A local web console drives all of this, with optional human gating between sessions.

Live trading is deliberately not implemented. The console has a live-trading page, but it is an empty frontend placeholder: no backend, no broker connection, no order path.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/autotrade/agent/` | Agent session runner, prompts, sub-agent and context-compaction machinery |
| `src/autotrade/environment/` | PIT snapshot building, unit registry, sandbox, trusted tools, broker and replay engine, LLM gateway |
| `src/autotrade/pipelines/` | Fold/Epoch/Meta/Held-out orchestration, ledger, interactive worker, reporting |
| `src/autotrade/data_sources/tushare/` | TuShare download, audit and scheduled-update logic |
| `src/autotrade/webui/` | Console backend and static frontend |
| `src/autotrade/paper/` | Paper-trading engine |
| `scripts/` | Entry points: `webui/`, `experiments/`, `data/`, `paper/`, `dev/` |
| `configs/` | Update schedule, column inventory, exported prompt snapshot, strategy output template, workspace reference packs |
| `ops/` | Sandbox Dockerfile, cron templates, nginx deployment configs |
| `tests/unit/` | The whole test suite |

Runtime directories (`data/`, `experiments/`, `logs/`, `results/`, `.runtime/`) are created locally and are not tracked. Worker stdout/stderr for a console-launched experiment is appended to `logs/workers/<experiment_id>.log`.

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

Copy `.env.example` to `.env` and fill in the credentials you need: a TuShare token for data, and a model endpoint. Model serving is external to this repository — the default local Qwen model is reached through a gateway at `VLLM_BASE_URL`, and DeepSeek is available as an alternative. The console fails to start a session rather than silently falling back when a selected model has no key.

Data is not bundled. The lake is populated by the TuShare download and audit scripts under `scripts/data/`, normally installed as a nightly cron job via `ops/cron/`.

## Running

Start the console on the machine that holds the data, the Docker daemon and the model endpoint:

```bash
python scripts/webui/run_webui.py
```

It listens on `127.0.0.1:38888` and refuses any non-loopback bind. Create an experiment from the homepage: choose the fold calendar, the held-out periods, the models and the run mode (`auto` runs straight through, `manual` waits for approval before each session, `step` also pauses after each validation backtest). The console then spawns a detached worker process, and the experiment detail page shows live status, agent traces, backtest results and the controls for pausing, injecting a message, re-running a fold or rolling back.

The same worker can be launched by hand for a headless run, and two smaller entry points exist for narrower work — replaying a single strategy against a daily parquet, and running exactly one Fold or Meta session in isolation to inspect its prompts, traces and artifacts:

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

`AGENTS.md` is the tracked contract for anyone — human or agent — working in this repository: development principles, documentation rules, resource checks and the Fold/Meta session rules.

The design documentation lives in `docs/` and the logbooks in `LOGBOOK.md` and `docs/logbook/`. Both are deliberately kept local and are excluded from version control, so a fresh clone will not contain them. Five documents are authoritative, each owning one area: data sources and PIT rules, agent-visible inputs and protocol, the environment and broker, the rolling pipeline, and deployment. The rest are derived: a quick reference for parameter defaults, and a unit table generated from the code.
