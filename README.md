# ADMCubeQuant

ADMCubeQuant is a research system in which an LLM agent designs and tests A-share trading strategies on its own, under conditions that stay close to a real research workflow.

The agent works inside a networkless Docker sandbox. Everything it can read is a point-in-time snapshot built from a local TuShare data lake: every row carries the timestamp at which it actually became available, so a strategy can never see a number that did not exist yet at the moment it claims to decide. The agent writes a strategy package exposing `generate_orders(context)` (and optionally `fit(context)`), backtests it against a simulated A-share broker (price limits, suspensions, lot sizes, stamp duty), and iterates.

An experiment comprises one research session on fixed dates, one registered strategy package with its validation record, and a replay of the period after it. The research period is whole years that start on the first of a month (2015-08 to 2025-07 on the current ten-year prebuilt views), split per experiment into research years the session sees from the start (5, 6 or 7) and validation years it never reads directly: its data view stays fixed at the end of the research years for the whole session. It researches with `validate`, which replays up to six candidates over research years only and reports the statistics and the significance test with no pass or fail, and it reveals the validation years with `test`: each test reruns one registrable candidate on the host from the validation start to the next step (a quarter by default) and returns a per-step table, and the newest step enters the validation record. The session ends when the last step is tested, which registers the last tested candidate. It works under experiment-level budgets of inference time, model calls, randomized controls and hosted-model API spend (50 USD by default; local models are free), with six `validate` calls allowed before each test; it compacts its own context, can read its own transcript back, and an attempt that fails resumes in place from the test horizon, the record and its last compaction summary.

The validation record is read as one series: the strategy's daily return minus random baselines matched to its own trades, regressed on market, size and value, with the residual information ratio set against the threshold z(1 − α)/√years. The registered package is then replayed without the agent as one continuous account over the following twelve months and the Held-out months; that forward reading and its lower bound are recorded and pooled across registered strategies, and gate nothing. For a registered strategy the operator chooses to follow, he opens a Paper account (CNY 200,000 by default): before each trading day it issues an order sheet that a person trades by hand, and it confirms or terminates itself by rules pinned at creation. A local web console drives this process.

Live trading is deliberately not implemented. The console has a live-trading page, but it is an empty frontend placeholder: no backend, no broker connection, no order path.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/autotrade/agent/` | Agent session runner, prompts, sub-agent and context-compaction machinery |
| `src/autotrade/environment/` | PIT snapshot building, unit registry, sandbox, trusted tools, broker and replay engine, LLM gateway |
| `src/autotrade/pipelines/` | Research geometry, sessions, test steps and the validation record, forward replay and readout, ledger, interactive worker |
| `src/autotrade/data_sources/tushare/` | TuShare download, audit and scheduled-update logic |
| `src/autotrade/webui/` | Console backend and static frontend |
| `src/autotrade/paper/` | Paper-trading engine |
| `scripts/` | Entry points: `webui/`, `experiments/`, `data/`, `paper/`, `dev/` |
| `configs/` | Update schedule, column inventory, exported prompt snapshot, strategy output template, reference workspaces, the skill library |
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

Copy `.env.example` to `.env` and fill in the credentials you need: a TuShare token for data, and a model endpoint. Model serving is external to this repository — by default the main conversation runs on `gpt-6-astra` and sub-agents and compaction on `DeepSeek-V4.1-Flash`, both through the LightingTheWord relay; DeepSeek, Xiaomi MiMo, the relay's `gemini-4-argon` and a local Qwen model behind a gateway at `VLLM_BASE_URL` can be selected per experiment and per role. The console fails to start a session rather than silently falling back when a selected model has no key.

Data is not bundled. The lake is populated by the TuShare download and audit scripts under `scripts/data/`, normally installed as a nightly cron job via `ops/cron/`.

## Running

Start the console on the machine that holds the data, the Docker daemon and the model endpoint:

```bash
python scripts/webui/run_webui.py
```

It listens on `127.0.0.1:38888` and refuses any non-loopback bind. Its warnings and errors go to stderr and to `logs/webui/webui.log`, rotated by size, so they survive a supervisor that discards its output. Create an experiment from the homepage: its research, forward and Held-out dates, the models and the budgets. The console then spawns a detached worker process, and the experiment detail page shows live status, agent traces, validation results and the validation record, the forward reading once the replay has run, and the controls for pausing, stopping, restarting and injecting a message into the running session.

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
