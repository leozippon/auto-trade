# Strategy output contract

This directory is the formal strategy artifact. `main.py` is the only runtime entrypoint and must define exactly one synchronous function:

```python
def generate_orders(context):
    return []
```

It may also define one synchronous `fit(context)` and a module-level `REFIT_PERIOD` literal:

```python
REFIT_PERIOD = "quarter"  # "day" / "month" / "quarter" / "year"; omitted or None = fit once per replay


def fit(context):
    ...  # train on the visible PIT history and np.save(...) the result under context.state_dir
```

The Environment calls `fit` once before the first decision of every replay and again at the first decision that falls in a new `REFIT_PERIOD`, always with the same context object that day's `generate_orders` receives, so `fit` can never see a row the decision could not. `fit` may write files under `context.state_dir`; `generate_orders` can only read them. That directory is empty at the start of every replay (Validation, frozen Test and Held-out all re-fit from PIT data), is never part of a revision or frozen artifact, and is discarded with the replay. One `fit` invocation has its own wall-clock budget (`budgets.strategy_fit_timeout_seconds`, default 30 minutes); a timeout or an exception in `fit` fails the whole backtest.

The strategy returns orders; it never receives or calls the Broker. The Environment validates the complete return value, queues accepted orders, and applies market timing, cash, positions, T+1, trading constraints, costs, and account updates.

The official working copy is `/mnt/agent/workspace/output` (search root `output`); there is no sibling `/mnt/agent/output`. The loader only loads `main.py`; user-module and helper imports are unsupported. Do not write caches, logs, data dumps, model weights, notebooks, hidden files, or secrets here.

Static assets to inherit across Folds (hand-curated tables, priors, reference parameters) belong in `/mnt/agent/workspace/models` (search root `models`), not in `output/`; the running strategy sees that directory read-only as `context.models_dir`. It may contain subdirectories and files such as `.npy`, `.npz`, `.parquet`, `.json` or `.txt`; at runtime only `np.load` (`.npy`/`.npz`) and `pd.read_parquet` can read them, so pickle or torch formats are not loadable by a strategy. Anything fitted at replay time belongs in `context.state_dir`, never in `models/`. Temporary training files stay elsewhere in `/mnt/agent/workspace/` (never under `output/` or `models/`).

## Data units are part of the strategy contract

Read `/mnt/artifacts/data_summary.json` and its `unit_contract` before writing thresholds or combining domains. The system prompt intentionally keeps only critical unit examples; this run-specific file is authoritative. `daily.parquet` is normalized: prices are CNY/share, volume and share fields are shares, amount/market-value fields are CNY, and `pct_chg`, turnover, and ratio fields are decimals (`5%=0.05`, `-9.5%=-0.095`). Heterogeneous research unions retain source units and must be interpreted by the tuple (file, `dataset`, column): `events.parquet` moneyflow `*_amount` is 万元 (`500` = CNY 5m), while `macro.parquet` `index_daily.pct_chg` is a percent-number (`5%=5.0`, do not multiply by 100 again). Verify and explicitly convert any unlabelled source unit before using it in a signal or threshold.

## Schedule

When creating an experiment, the user sets `strategy_period` and a fixed `Asia/Shanghai` `inference_time`:

| Period | Invocation rule |
| --- | --- |
| `day` | Every available trading day |
| `month` | First available trading day of a new month |
| `quarter` | First available trading day of a new quarter |
| `year` | First available trading day of a new year |

The first available trading day in a replay is always due. `inference_time` is any valid 24-hour `HH:MM`; the default is `08:30`.

The schedule decides only when the strategy is asked. The strategy decides its own rebalance cadence — returning `[]` on a non-rebalance day is normal — and, through `REFIT_PERIOD`, its own re-fitting cadence. Historical one-minute and auction records can be point-in-time research features, but they never create strategy ticks. Static historical minute closes also provide exact execution prices outside the daily open and close timestamps.

## Read-only context

Each invocation receives an immutable market-level context:

| Surface | Contract |
| --- | --- |
| `context.inference_at` | Timezone-aware current decision time, normalized to `Asia/Shanghai` |
| `context.bars` | Strict-JSON daily records from the current evaluation interval that are visible at the decision time; it may be empty at the interval's first inference |
| `context.account.cash` | Non-negative finite cash snapshot at invocation entry |
| `context.account.positions` | Read-only `{symbol: quantity}` long-position mapping at invocation entry |
| `context.history(symbol)` | Visible `bars` rows for one symbol |
| `context.latest(symbol)` | Last visible row for one symbol, or `None` |
| `context.snapshot_dir` | Read-only frozen research snapshot path string |
| `context.asof_dir` | Read-only PIT view path string for this decision |
| `context.asof_version` | Version label for the current as-of view |
| `context.state_dir` | Per-replay state directory path string: writable while `fit` runs, read-only for `generate_orders`; empty when the strategy defines no `fit` |
| `context.models_dir` | Read-only path string of the artifact's `models/` directory; empty when the artifact has none |
| `context.nl(...)` | Optional host-mediated local evidence query; absent configurations fail explicitly |

Every bar has a timezone-aware `available_at` no later than `context.inference_at`. The host also rejects any explicit future `available_at` nested in an NL request or response.

The strategy receives no Broker, Shell, experiment controls or previous results. The runtime mounts `main.py`, the read-only snapshot/as-of directories, the read-only `models/` directory and the state directory (read-write only for `fit`), so executable strategy logic must be self-contained in `main.py`.

## Reading PIT data

`context.bars` is an evaluation-interval market surface, not the complete strategy input history. For a long daily lookback, use the PIT `daily` domain under `context.asof_dir`; it contains the frozen input history plus rows that have become visible by the current inference time. Read only confirmed columns and bound the strategy's date window after inspecting the current schema. Additional configured domains may use the same context-rooted pandas access.

The two path strings have **different layouts**, and mixing them up is the single most common way a backtest dies on its first decision:

| Path | Layout | Read it as |
| --- | --- | --- |
| `context.asof_dir` | one **directory of parquet parts per domain**: `asof_dir/daily/part_0000.parquet`, `part_0001.parquet`, … | `pd.read_parquet(context.asof_dir + "/daily")` — pass the **directory**, pandas concatenates the parts |
| `context.snapshot_dir` | one **flat file per domain**: `snapshot_dir/daily.parquet` | `pd.read_parquet(context.snapshot_dir + "/daily.parquet")` |

`context.asof_dir + "/daily.parquet"` does not exist and raises `FileNotFoundError`. `modification_check` rejects that spelling before a backtest starts. The same applies to every as-of domain (`daily`, `events`, `macro`, `fundamentals`, `intraday_1min`, `auction`, `text_index`, `universe`); only text bodies under `asof_dir/text_library/` are individual `<name>.parquet` shards.

**Never fall back from `asof_dir` to `snapshot_dir` when an as-of read fails.** The frozen snapshot stops at the decision time, so reading it during the replay silently substitutes stale data for the rolling view — a point-in-time violation, not a recovery. Fix the read instead, and let a genuine failure fail.

Run `smoke_backtest` before `daily_backtest`: it replays the current `output/` over the first few trading days on the real path — real as-of layout, real `AccountSnapshot`, same executor and per-decision timeout — and returns the exact exception text plus the as-of domain directory names. A hand-written shell script that assigns `context.asof_dir = "/mnt/snapshot"` or fakes an account object proves nothing about the replay.

The static contract accepts `pandas.read_parquet` and `numpy.load` only when the first positional argument is a path expression directly rooted at `context.snapshot_dir`, `context.asof_dir`, `context.state_dir` or `context.models_dir` (`context.<dir> + "/<literal>"`), and `numpy.save`/`savez`/`savez_compressed` and `DataFrame.to_parquet` only when it is rooted at `context.state_dir`. At runtime the state directory is read-only outside `fit`, so a write from `generate_orders` fails the backtest. For example:

```python
import pandas as pd

daily = pd.read_parquet(
    context.asof_dir + "/daily",
    columns=["ts_code", "close"],
)
```

Do not import a path helper or use an arbitrary file path. The exact file layout, schema, dataset labels, coverage, and units come from the current run's data summary, snapshot manifest, Parquet metadata, and unit reference.

If the confirmed signal and execution conditions produce no eligible security, return `[]` — an empty array is a valid and correct result. A genuine strategy order may carry concise metadata describing its signal basis.

Units follow the contract above: normalized daily fields are already converted, combined event, macro, and fundamental domains can retain source units, and such fields must be interpreted by the full (file, `dataset`, column) identity — never infer a unit from a shared column name.

The formal executor keeps one strategy worker alive across the inference calls of a replay. A module-level cache may reuse PIT-data-derived values only while its recorded `context.asof_version` still matches the current call. The version identifies the as-of data view only; values that depend on `context.inference_at`, `context.bars`, or `context.account` must be recomputed per call or keyed separately. When the version changes, update the cache from only newly visible rows or replace it from the required columns and an exact finite tail. If the incremental merge cannot be proved exact for the confirmed schema, use the bounded-tail reload. Every read must remain rooted in the current context and must not admit a row beyond `context.inference_at`.

Worker and revision restarts naturally clear module state, so a cache is only an optimization: strategy results must remain correct from a cold cache. A refit replaces the files under `context.state_dir` without restarting the inference worker, so load them inside `generate_orders` (a small `np.load` per call), not at import time and not into a cache that outlives a refit. Do not reread the full PIT directory and rerun full-history sort, group, percentage-change, and rolling calculations on every daily invocation. Project only required columns, retain the exact per-symbol tail the longest factor needs, then run cross-sectional work. Validate an optimization once in development against a full-history reference for factors, ranks, candidates, and orders to a floating tolerance no looser than `1e-12`; this equivalence check is not part of every inference.

## Strict JSON order array

The top-level return value must be a JSON array, including when there are no orders. `NaN`, infinity, Python objects, sets, tuples used as object keys, or any other non-JSON value are forbidden.

Each order requires four fields:

```json
{
  "symbol": "000001.SZ",
  "action": "buy",
  "quantity": 100,
  "execute_at": "2026-01-05T09:30:00+08:00"
}
```

Rules:

- `symbol` is a non-empty security code.
- `action` is exactly `buy` or `sell`.
- `quantity` is a positive integer share count; booleans are not integers for this contract.
- `execute_at` is a timezone-aware ISO-8601 string and cannot precede `context.inference_at`.
- Additional JSON fields are preserved as order metadata. A short `reason` is useful for review.

The account snapshot does not change while `generate_orders` is running. For a batch, read cash once, keep a local remaining budget, and leave room for price movement and fees.

Sort candidates explicitly whenever order expresses investment priority: iteration order over an unordered container is not a substitute for stating intent, and it decides which orders get the remaining budget.

## Exact execution time

The fixed inference time controls when the strategy runs; each order independently chooses a timezone-aware `execute_at` no earlier than that invocation. The Environment processes pending orders in timestamp order and resolves prices without rounding or delaying the requested timestamp:

- `09:30` uses that trading day's normalized daily `open`.
- `15:00` uses that trading day's normalized daily `close`.
- Any other timestamp requires the same symbol's static historical one-minute row at that exact minute and uses its `close`.

If the exact price source is absent or invalid, including when the required daily row is unavailable, the order is rejected as `missing_execution_price`; the Environment does not substitute an adjacent minute or a daily endpoint.

## DailyBroker quick reference

`DailyBroker` is a deterministic long-only A-share account:

- Buys and sells use share quantities. A buy declares whole 100-share lots, except on the STAR board (`688`/`689.SH`), where it declares at least 200 shares and then any 1-share increment, and on the BSE, where it declares at least 100 shares and then any 1-share increment.
- A sell declares whole lots, or one declaration carrying the entire odd-lot tail a corporate action left behind (a STAR/BSE position below its minimum declaration is likewise exitable only in full).
- A position bought today becomes sellable on the next trading day's `open_day` transition.
- Commission applies on both sides with a minimum fee, and a transfer fee (过户费) applies on both sides with no minimum. Stamp duty applies on sells only, at a rate that depends on the execution date. Directional slippage adjusts the event price.
- Suspension, `missing_execution_price`, daily price limits, insufficient cash, insufficient sellable quantity, and invalid buy lots reject the whole order. Missing or non-finite direction-related limit-up/limit-down prices reject the whole order as `missing_daily_price_limit`.
- Orders are all-or-reject. There is no partial fill, order-book depth, queue position, market impact, or liquidity-capacity model.

Initial cash is an experiment setting, so size from `context.account.cash` instead of assuming a fixed account. The default cost profile uses 1 bp commission with a CNY 5 minimum, a 0.1 bp transfer fee, and 5 bp directional slippage. Sell stamp duty is **10 bp for executions before 2023-08-28 and 5 bp from 2023-08-28 onward** — size pre-cutover windows against the higher rate, because the Broker charges it. Round-trip cost on a pre-cutover sell is therefore about twice the post-cutover figure.

The Broker processes due orders sequentially and updates its true account after each fill. `context.account` remains the pre-call snapshot, so batch sizing must not assume that later strategy statements can observe earlier fills.

## Local evidence

If configured, call the host evidence service with:

```python
result = context.nl(query="利润预告", mode="search", limit=5)
```

`mode="search"` returns bounded local evidence. `mode="answer"` allows a model to answer only from retrieved evidence and requires valid evidence citations. No evidence returns `status="no_evidence"` and does not invoke the model.

Treat text as fallible supporting evidence. Publish time, ingest time, retrieval recall, model priors, parsing, and look-ahead contamination remain risks. NL never overrides cash, positions, tradability, costs, or PIT.

## Runtime restrictions

The loader requires one synchronous single-argument `generate_orders`, accepts at most one synchronous single-argument `fit`, and requires `REFIT_PERIOD`, when present, to be a single module-level assignment of a period literal or `None`. Supported imports are limited to:

```text
__future__, collections, datetime, decimal, math, statistics, numpy, pandas
```

User-module imports, relative imports, dynamic import/execution, arbitrary file access, process calls, and general external I/O are rejected. Common NumPy and pandas load/save methods are blocked; the only supported strategy file I/O is the context-rooted form described above.

The default executor uses network-disabled, read-only Docker containers with bounded CPU, memory, process count, inference time, fit time, protocol output, and temporary storage; a strategy with `fit` gets a second identical container whose only difference is the writable state mount. If the container boundary cannot be established, execution fails instead of changing modes.

## Default strategy

The shipped `main.py` is a deliberately small working baseline. `fit` reads a bounded window of the PIT `daily` domain, builds cross-sectionally standardized 5-day and 20-day adjusted returns, fits a ridge regression against the realized 5-day forward return, and saves the three coefficients as `ridge_coef.npy` under `context.state_dir` (all zeros when fewer than 200 samples are visible, which makes the ranking flat and alphabetical). `generate_orders`, while flat, loads those coefficients, scores the latest visible cross-section, and submits strict JSON buy orders for up to ten top-ranked symbols with an equal-budget basket and a cash buffer at the next same-day daily price timestamp: `09:30` before the open or `15:00` before the close. An after-close invocation emits no order because the strategy does not receive a future trading calendar. Replace the features with a mechanism-backed PIT signal and add an explicit exit/rebalance lifecycle before treating it as a research strategy.

Before finishing a Fold, verify that:

- `main.py` is self-contained and passes static validation.
- If `fit` is defined, it writes everything `generate_orders` needs under `context.state_dir` and completes within `budgets.strategy_fit_timeout_seconds`.
- Every field and unit used by the signal has been confirmed from the current data contract.
- Orders pass strict JSON validation and respect the configured schedule.
- Batch sizing leaves a cost buffer and does not depend on same-call account mutation.
- The current revision completed full daily Validation.
- The formal output contains no hidden paths, caches, temporary data, notebooks, credentials, or unused code.
