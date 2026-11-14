# ADMCubeQuant

ADMCubeQuant is a research system in which an LLM agent designs and tests A-share trading strategies on its own, under conditions that stay close to a real research workflow.

The agent works inside a networkless Docker sandbox. Everything it can read is a point-in-time snapshot built from a local TuShare data lake: every row carries the timestamp at which it actually became available, so a strategy can never see a number that did not exist yet at the moment it claims to decide. The agent writes a strategy package exposing `generate_orders(context)` (and optionally `fit(context)`), backtests it against a simulated A-share broker (price limits, suspensions, lot sizes, stamp duty), and iterates.

An experiment is one research session and at most one registered strategy. Its ten August–July years (2015-08 to 2025-07 by default) are split into research years, which the agent sees as one data view fixed for the whole session, and validation years it never reads directly. It researches with `validate`, which replays up to six candidates over the research years and, once a step is tested, also over the validation steps already revealed, reporting statistics with no pass or fail. It reveals the validation years one step (a quarter by default) at a time with `test`, which reruns one registrable candidate from the validation start through the next step and adds that step to the validation record. The session ends when the last step is tested, registering the last tested candidate. Experiment-level budgets bound inference time, model calls, randomized controls and hosted-model spend (USD 50 by default, USD 150 in current rounds; local models are free), and a failed attempt resumes in place.

The validation record is read as one series: the strategy's daily return minus random baselines matched to its own trades, regressed on market, size and value, with the residual information ratio set against the threshold z(1 − α)/√years. The registered package is then replayed without the agent as one continuous account over the following twelve months and the Held-out months; those readings are recorded and pooled across registered strategies, and gate nothing. A registered strategy the operator chooses to follow runs as a Paper account (CNY 200,000 by default) that issues an order sheet before each trading day for a person to trade by hand, and confirms or terminates itself by rules pinned at creation. A local web console drives this process.

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
| `src/autotrade/common/` | Shared helpers: JSON records, ids, time |
| `scripts/` | Entry points: `webui/`, `experiments/`, `data/`, `paper/`, `dev/`, `llm/` |
| `configs/` | Update schedule, column inventory, exported prompt snapshot, strategy output template, reference workspaces, the skill library |
| `ops/` | Sandbox Dockerfile, cron templates, nginx deployment configs, the entry host's login gateway (`ops/login-gateway/`), the ZGCA A100 instance workflow (`ops/zgca/`) |
| `tests/unit/` | The whole test suite |

Runtime directories (`data/`, `experiments/`, `logs/`, `results/`, `.runtime/`) are created locally and are not tracked; pytest and ruff keep their caches under `.runtime/cache/`. Worker stdout/stderr for a console-launched experiment is appended to `logs/workers/<experiment_id>.log`.

## Setup

The Python environment is a conda env named `quant` on Python 3.11. Install the package with the console extras, and the test and lint tools:

```bash
conda activate quant
pip install -e '.[webui,dev]'
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

An experiment that needs a GPU takes only cards no running experiment holds that keep `AUTOTRADE_GPU_MIN_FREE_GIB` GiB free (12 by default). Set it in the environment of the console (restart it after a change) and of round launchers, whose startup smoke test applies the same rule.

Experiments that belong together are created from a checked-in round file, which fixes their shared dates, dataset selection and prebuilt view cache; `--dry-run` validates all of it without creating anything, and `--fill` creates the pending experiments as running slots free up:

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

The design documentation lives in `docs/` and the project log in `docs/logbook/PROJECT_LOG_YYYY-MM-DD.md`, one file per day. These, `AGENTS.md` included, are deliberately kept local and excluded from version control, so a fresh clone does not contain them. Where they exist, start with `docs/summary.md`, the owner's single design page, which ends with a map of the detailed documents.
