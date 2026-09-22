# ADMCubeQuant

ADMCubeQuant is a research system in which an LLM agent designs and tests A-share trading strategies on its own, under conditions that stay close to a real research workflow.

The agent works inside a networkless Docker sandbox. Everything it can read is a point-in-time snapshot built from a local TuShare data lake: every row carries the timestamp at which it actually became available, so a strategy can never see a number that did not exist yet at the moment it claims to decide. The agent writes a strategy package exposing `generate_orders(context)` (and optionally `fit(context)`), backtests it against a simulated A-share broker (price limits, suspensions, lot sizes, stamp duty), and iterates.

An experiment is one research arm on fixed dates. The agent works through one research session on a research period of whole July-to-June years (2021-07 to 2025-06 by default), with arm-level budgets of inference time, replay-years, null controls and model calls; the session compacts its own context on the host's notice and can read its own transcript back, and an attempt that fails is resumed in place from its last compaction summary, its workspace and the tree of validated candidates. The session may nominate one validated candidate for freezing, and the freeze is accepted only if its research-period result clears a deflated-Sharpe gate that charges every revision the arm tried. The frozen strategy is then replayed once, without the agent, as one continuous book over the twelve months after research and the Held-out quarter after those. It graduates only if the forward slice holds up statistically (a positive bootstrap lower bound on its neutralized excess return, a non-negative last six months, positive excess under doubled slippage, enough trading and exposure, drawdown inside the limit) and the Held-out slice shows no catastrophic failure. A graduate is the candidate for paper trading, which the operator starts by hand. A local web console drives all of this.

Live trading is deliberately not implemented. The console has a live-trading page, but it is an empty frontend placeholder: no backend, no broker connection, no order path.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/autotrade/agent/` | Agent session runner, prompts, sub-agent and context-compaction machinery |
| `src/autotrade/environment/` | PIT snapshot building, unit registry, sandbox, trusted tools, broker and replay engine, LLM gateway |
| `src/autotrade/pipelines/` | Research calendar, research sessions, freeze, forward replay and verdict, ledger, interactive worker |
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

It listens on `127.0.0.1:38888` and refuses any non-loopback bind. Create an experiment from the homepage: its research, forward and Held-out dates, the models and the budgets. The console then spawns a detached worker process, and the experiment detail page shows live status, agent traces and validation results, the forward verdict once the replay has run, and the controls for pausing, stopping, restarting and injecting a message into the running session.

Arms that belong together are created from a checked-in round file, which fixes their shared dates, dataset selection and prebuilt view seed; `--dry-run` validates all of it offline before anything is sent to the console:

```bash
python scripts/experiments/create_round_20260920.py 38888 --dry-run
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

The design documentation lives in `docs/` and the logbooks in `LOGBOOK.md` and `docs/logbook/`. All of these, `AGENTS.md` included, are deliberately kept local and are excluded from version control, so a fresh clone will not contain them. Six documents are authoritative, each owning one area: data sources and PIT rules, agent-visible inputs and protocol, the environment and broker, the research pipeline, deployment, and — in `docs/research-lessons.md` — the research-lessons register: settled lessons, closed directions with the evidence that closed them, and the open hypothesis space, which is what each new round is designed from. The rest are derived: `docs/system-overview.md`, an entry point that walks the whole system in workflow order with worked examples and points back into the authoritative sections; a quick reference for parameter defaults; and a unit table generated from the code.
