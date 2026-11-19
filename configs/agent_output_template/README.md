# Strategy output contract

## Entry points

This directory is the formal strategy artifact: a Python package whose entry module `main.py` must define exactly one synchronous, single-argument `generate_orders`. It may also define one synchronous, single-argument `fit` and, only together with `fit`, a module-level `REFIT_PERIOD` assigned once to a period literal or `None`:

```python
REFIT_PERIOD = "quarter"  # "day" / "month" / "quarter" / "year"; omitted or None = fit once per replay


def fit(context):
    ...  # train on the visible PIT history and np.save(...) the result under context.state_dir


def generate_orders(context):
    return []
```

`fit` is recognised by its definition in `main.py`, not by its body: delegating to a sibling module (`def fit(context): return trainer.fit(context)`) behaves exactly like an inline body. The Environment calls `fit` once before the first decision of every replay and again at the first decision that falls in a new `REFIT_PERIOD`, always with the same context object that day's `generate_orders` receives, so `fit` never sees a row the decision could not; it hands what it trained to `generate_orders` through `context.state_dir`.

The run facts' `broker_replay.schedule` sets when the strategy is asked: a period and an `Asia/Shanghai` `inference_time` (any valid `HH:MM`, default `08:00`). A `day` period asks on every available trading day; `month`, `quarter` and `year` on the first available trading day of a new one; a replay's first available trading day is always due. The strategy chooses its own rebalance cadence (returning `[]` on a non-rebalance day is normal) and, through `REFIT_PERIOD`, its re-fitting cadence. Historical one-minute and auction records can be point-in-time features but never create strategy ticks.

A starter from a reference workspace replaces `output/main.py`, and `refs/README.md` describes it; otherwise `output/main.py` returns no orders and is not a baseline to extend.

## Package and assets

The official working copy is `/mnt/agent/workspace/output` (search root `output`); there is no sibling `/mnt/agent/output`. `main.py` imports sibling modules and packages of this directory absolutely (`from lib.features import momentum` for `output/lib/features.py`); relative imports are rejected. A package module is one module however it is imported: an `import` inside a function body (the usual way to break an import cycle) resolves to the same module object, with the same module-level state, as one at the top of the file. Every `.py` file here is part of the strategy, under the same import and file-I/O rules as `main.py`, and counts toward the artifact fingerprint and the file and byte limits; a single-file strategy is a package with one module. Do not write caches, logs, data dumps, model weights, notebooks, hidden files or secrets here.

Static assets that travel with the artifact (lookup tables, vocabularies, precomputed static features, reference parameters) belong in `/mnt/agent/workspace/models` (search root `models`), which the strategy reads as `context.models_dir`: subdirectories and files such as `.npy`, `.npz`, `.parquet`, `.json` or `.txt`, loaded with `np.load`, `pd.read_parquet`, `torch.load` or a booster's `load_model` (`pickle`- and `joblib`-only formats are not loadable). Weights trained in the session on research-year outcomes and shipped in `models/` are look-ahead in every replay of those years. Whatever is fitted or cached at replay time goes in `context.state_dir`, never in `models/`; temporary training files stay elsewhere in `/mnt/agent/workspace/`.

## Limits

Strategy code runs in network-disabled, read-only Docker containers bounded in CPU, memory, processes, call time, protocol output, temporary storage and state-directory disk; a strategy with `fit` gets a second, identical container for it, and both mount the state directory read-write. If that boundary cannot be established, execution fails instead of changing modes. This experiment's limits are in the run facts:

| Limit | Run facts |
| --- | --- |
| Wall clock of one `fit` call | `budgets.strategy_fit_timeout_seconds` |
| Wall clock of one `generate_orders` call | `budgets.strategy_inference_timeout_seconds` |
| Container memory | `budgets.strategy_memory_bytes` |
| `context.state_dir` on disk, measured after every call | `budgets.strategy_state_disk_bytes` |
| Files and bytes of `output/` and `models/` | `artifact_contract.modification_constraints` (`max_strategy_files`, `max_strategy_bytes`, `max_model_artifact_files`, `max_model_artifact_bytes`) |

A timeout or an exception in either entry point, crossing the memory or state cap, or a subdirectory of `context.state_dir` the Environment cannot read (one created with a restrictive mode such as `0o700`) fails the whole backtest as the strategy's own failure.

## Read-only context

Each invocation receives an immutable market-level context:

| Surface | Contract |
| --- | --- |
| `context.inference_at` | Timezone-aware decision time, normalized to `Asia/Shanghai` |
| `context.bars` | Strict-JSON daily records of the current evaluation interval visible at the decision; may be empty at the interval's first inference |
| `context.account.cash` | Non-negative finite cash at invocation entry |
| `context.account.positions` | Read-only `{symbol: quantity}` long positions at invocation entry |
| `context.history(symbol)` | Visible `bars` rows for one symbol |
| `context.latest(symbol)` | Last visible row for one symbol, or `None` |
| `context.snapshot_dir` | Read-only frozen research snapshot path string |
| `context.asof_dir` | Read-only PIT view path string of this decision |
| `context.asof_version` | Version label of that view |
| `context.state_dir` | Path string of the replay's working store, read-write for `fit` and `generate_orders`, empty when the replay starts |
| `context.models_dir` | Read-only path string of the artifact's `models/`; empty when it has none |
| `context.pretrained_dir` | Read-only path string of the pretrained store; empty when the experiment mounts none |

Every bar has a timezone-aware `available_at` no later than `context.inference_at`. The strategy receives no Broker, shell, experiment controls or previous results. Its containers mount this package, the snapshot and as-of directories, `models/` and the pretrained store read-only and the state directory read-write; nothing else of the workspace is visible, so all executable logic lives in this package.

## Reading PIT data

`context.bars` is an evaluation-interval market surface, not the complete strategy input history. A long daily lookback reads the PIT `daily` domain under `context.asof_dir`: the frozen input history plus the rows visible by the current inference time. Its parts never repeat a row (the frozen first part holds everything available at or before the decision time, later parts only rows published after it), so it needs no deduplication. Read only confirmed columns, bound the date window after inspecting the current schema, and read other configured domains the same context-rooted way.

The two path strings have different layouts, and mixing them up is the most common way a backtest dies on its first decision:

| Path | Layout | Read it as |
| --- | --- | --- |
| `context.asof_dir` | one directory of parquet parts per domain: `asof_dir/daily/part_0000.parquet` plus later parts, renamed as each finished month is merged into one | `pd.read_parquet(context.asof_dir + "/daily")`: pass the directory, pandas concatenates the parts |
| `context.snapshot_dir` | one flat file per domain: `snapshot_dir/daily.parquet` | `pd.read_parquet(context.snapshot_dir + "/daily.parquet")` |

`context.asof_dir + "/daily.parquet"` does not exist and raises `FileNotFoundError`; `modification_check` rejects that spelling before a backtest starts. The same holds for every as-of domain (`daily`, `events`, `macro`, `fundamentals`, `intraday_1min`, `auction`, `text_index`, `universe`); only text bodies under `asof_dir/text_library/` are individual `<name>.parquet` shards. Never fall back from `asof_dir` to `snapshot_dir` when an as-of read fails: the frozen snapshot stops at the decision time, so reading it during the replay silently substitutes stale data for the rolling view, a point-in-time violation rather than a recovery. Fix the read, and let a genuine failure fail. `smoke_backtest` exercises the real path; a shell script that assigns `context.asof_dir = "/mnt/snapshot"` or fakes an account object proves nothing about the replay.

The supported file I/O is `pandas.read_parquet`, `numpy.load`, `torch.load`, a booster's `load_model` and `from_pretrained` for reading, and `numpy.save`/`savez`/`savez_compressed`, `DataFrame.to_parquet`, `torch.save`, a booster's `save_model` and `save_pretrained` for writing. Build each path any way (a helper, a variable, a loop, an f-string, a module constant) from the context directory it belongs under: reads from `context.snapshot_dir`, `context.asof_dir`, `context.state_dir`, `context.models_dir` or `context.pretrained_dir`, writes from `context.state_dir`. The static check refuses only what the source proves wrong: an absolute path literal (no host or sandbox path exists in the replay container) and a write rooted at a read-only directory. At runtime every context directory except `context.state_dir` is mounted read-only.

File layout, schema, dataset labels, coverage and units come from the run's data summary, snapshot manifest, Parquet metadata and unit reference. Read `/mnt/artifacts/data_summary.json` and its `unit_contract`, which with the unit reference it points to are authoritative, before writing thresholds or combining domains. `daily.parquet` is normalized: prices are CNY/share, volume and share fields shares, amount and market-value fields CNY, and `pct_chg`, turnover-rate and dividend-yield fields decimals (`5%=0.05`). Research unions keep source units and are read by the tuple (file, `dataset`, column): `events.parquet` moneyflow `*_amount` is 万元 (`500` = CNY 5m), and `macro.parquet` `index_daily.pct_chg` is a percent-number (`5%=5.0`). Verify and explicitly convert any unlabelled unit before using it in a signal or threshold.

## State across calls and days

Within a replay the executor keeps one strategy worker alive across the inference calls, so module-level state persists from one decision day to the next; `fit` runs in a container of its own, started at the first `fit` and kept for the later ones, with its own module state. `context.state_dir` persists across calls and across both containers, and both read and write it. It is empty at the start of every replay (every `validate`, `test` or `smoke_backtest` replay, and the later replay of a frozen artifact), is never part of a revision or frozen artifact, and is discarded with the replay; nothing carries from one replay to the next.

Keep anything expensive that later calls reuse (feature panels, fitted models, scores, intermediate tables) in module state or in the directory. The state cap counts everything left in the directory, and Paper keeps it from day to day, so overwrite or drop what later calls no longer need instead of adding a file per day. A value derived from an earlier day's PIT view can be reused on any later day without look-ahead, because the replay only moves forward.

Build the warm-up window once, at the first call that needs it, then extend it by each new day: read only the rows dated after the last day it holds, drop what falls out of the longest lookback, and run the cross-sectional work on that tail. Do not reread the full PIT directory and rerun the full-history sorts, groups, percentage changes and rolling calculations on every decision day. Track what a cache holds by the rows' dates, not by part file names, which change as finished months merge. Check an incremental path once in development against a full recomputation of factors, ranks, candidates and orders, to a tolerance fit for the dtype (`1e-12` for float64 CPU arithmetic), not in every inference.

Later views may revise what earlier rows said: a restated fundamental arrives as a new row for an old period, and a new universe vintage changes names, industries and listings; `context.asof_version` changes with each, as with every new part. If the features `fit` trained on and those a decision uses must agree exactly, recompute the trailing window the rolling calculations depend on, or compare `context.asof_version` where exactness matters.

A replay starts cold. The first call builds what it needs from PIT data, and module state must be something the calls of this replay, re-issued in order, would rebuild. Paper, the daily account a registered strategy may later run in, does exactly that every morning: a fresh worker re-issues every earlier `generate_orders` call in order, each with its own day's view and booked account, while the state directory persists from day to day, so the re-issued calls may find what later days wrote there, the latest fit included; the module state they leave must still be the one the replay would carry into the new day. Paper re-issues the `generate_orders` calls only: `fit` runs in a fresh container on each day it is due and earlier fits are not re-issued, so whatever one `fit` hands the next (weights, optimizer state, running statistics, predictions awaiting their outcome) must be in `context.state_dir`, while what a fresh worker can rebuild from the PIT view (tables, panels, features) may stay in module state. `generate_orders` must not persist anything a re-issue would repeat: a counter in module state counts the same on every re-issue, but one written to `context.state_dir` on every call is advanced again by each re-issued call and diverges on Paper.

A refit replaces files under `context.state_dir` without restarting the inference worker, so reload fitted artifacts when the stamp `fit` writes there changes, never at import time. Values derived from `context.inference_at`, `context.bars` or `context.account` belong to the call that received them. Every read stays rooted in the current context and must not admit a row beyond `context.inference_at`.

## Orders

The strategy returns orders and never receives or calls the Broker; the Environment validates the complete return value, queues accepted orders, and applies market timing, cash, positions, T+1, trading constraints, costs and account updates. The return value must be a JSON array, also when there are no orders; `NaN`, infinity, Python objects, sets, tuples used as object keys and any other non-JSON value are forbidden. Each order requires four fields and may name a session and a limit price:

```json
{
  "symbol": "000001.SZ",
  "action": "buy",
  "quantity": 100,
  "execute_at": "2026-01-05T09:30:00+08:00",
  "session": "intraday",
  "limit_price": 10.52
}
```

- `symbol` is a non-empty security code; `action` is exactly `buy` or `sell`.
- `quantity` is a positive integer share count; booleans are not integers for this contract.
- `execute_at` is a timezone-aware ISO-8601 string and cannot precede `context.inference_at`.
- `session` is optional and, when given, is exactly `open`, `close` or `intraday`: how the order meets the market on `execute_at`'s trading day. A session order's `execute_at` is that day's session minute: `09:30` for `open` and `intraday`, `15:00` for `close`.
- `limit_price` is optional, allowed only with `"session": "intraday"`, and is a positive price in whole 0.01 CNY ticks; it makes the order a day limit order.
- Additional JSON fields are kept as order metadata; a short `reason` helps review. The list one call returns, metadata included, serializes to at most 1,000,000 characters of JSON (counted in UTF-8 bytes), so keep metadata to short summaries rather than logs.

A return value that breaks any of these rules fails the backtest at that decision; a well-formed order the Broker cannot fill is rejected on its own and the replay goes on. Sort candidates explicitly whenever order expresses investment priority: iteration order over an unordered container does not state intent, and it decides which orders get the remaining budget.

The fixed inference time controls when the strategy runs; each order independently chooses a timezone-aware `execute_at` no earlier than that invocation. The Environment processes pending orders in timestamp order, orders sharing a timestamp in the order the strategy returned them. An order with a `session` is priced from its trading day's normalized daily bar; the adverse daily limit is `up_limit` for a buy and `down_limit` for a sell:

| Session | Fills at | Does not fill when |
| --- | --- | --- |
| `open` | the day's `open` (the opening call auction) plus directional slippage | the slipped price reaches the adverse limit: `unfilled_price_limit` |
| `close` | the day's `close` (the closing call auction) plus directional slippage | the slipped price reaches the adverse limit: `unfilled_price_limit` |
| `intraday` | the day's volume-weighted average price, `amount / vol`, plus directional slippage, at most the adverse limit | the day is locked at the adverse limit (`high` and `low` both on it): `unfilled_locked` |
| `intraday` with `limit_price` | a buy at the lower of `open` and the limit once the day's `low` reaches the limit, a sell at the higher of `open` and the limit once the `high` reaches it; no slippage, so an open beyond the limit fills at the open | the limit is not reached (`unfilled_limit`) or the day is locked at the adverse limit (`unfilled_locked`) |

An unfilled session order expires at the end of its day with that reason; nothing carries over and nothing fills in part. An order without a `session` is priced at its exact `execute_at`, neither rounded nor delayed, and is rejected as `daily_price_limit` when its slipped price reaches the adverse limit: `09:30` takes that trading day's normalized daily `open`, `15:00` its `close`, and any other timestamp the `close` of the same symbol's static historical one-minute row at that exact minute. An absent or invalid price source, including an unavailable daily row or a field its rule reads, rejects the order as `missing_execution_price`; no adjacent minute or daily endpoint is substituted. Minute bars are off by default: with the run facts' `visible_timeline.execution_policy.historical_minutes_available` false, an order without a session fills only at `09:30` and `15:00`, and every other `execute_at` is rejected.

## Broker

`DailyBroker` is a deterministic long-only A-share account:

- A buy declares whole 100-share lots, except on the STAR board (`688`/`689.SH`), where it declares at least 200 shares and then any 1-share increment, and on the BSE, where it declares at least 100 shares and then any 1-share increment.
- A sell declares whole lots, or one declaration carrying the entire odd-lot tail a corporate action left behind (a STAR/BSE position below its minimum declaration is likewise exitable only in full).
- A position bought today becomes sellable from the next trading day's open.
- On an ex-date, before any fill or decision, the account is credited each held position's gross cash dividend less any policy-required withholding, and a bonus issue or split re-sizes the position to the whole shares the ex-rights reference price (`pre_close`) implies, any fractional share paid in cash; shares created that day become sellable the next trading day.
- Commission applies on both sides with a minimum fee, and a transfer fee (过户费) on both sides with no minimum. Stamp duty applies on sells only, at a rate that depends on the execution date. Directional slippage adjusts the event price of every fill except a day limit order's.
- When the run facts' `broker_replay.dividend_tax_policy.charged` is true, dividend tax reduces available cash at distribution and/or sale under the dividend's record-date rule; that policy is authoritative for historical regimes, withholding, FIFO holding periods and taxable bonus shares, so do not assume dividends are credited tax-free until sale.
- Suspension, `missing_execution_price`, daily price limits (including the `unfilled_*` outcomes above), insufficient cash, insufficient sellable quantity and invalid buy lots reject the whole order; missing or non-finite direction-related limit-up/limit-down prices reject it as `missing_daily_price_limit`.
- Orders are all-or-reject: there is no partial fill, order-book depth, queue position, market impact or liquidity-capacity model.

Initial cash is an experiment setting, so size from `context.account.cash`; the run facts' `broker_replay` gives this experiment's cash, commission, slippage, stamp duty and dividend-tax policy. The default cost profile is 1 bp commission with a CNY 5 minimum, a 0.1 bp transfer fee and 5 bp directional slippage. Sell stamp duty is 10 bp for executions before 2023-08-28 and 5 bp from that day on; size pre-cutover windows against the higher rate, because the Broker charges it. At the default profile a round trip of fills of at least CNY 50,000 each costs about 22 bp before the cutover and 17 bp from it, slippage included; smaller fills pay more, because the commission minimum binds below that size.

Each trading day opens with the T+1 release and ex-date settlement, then fills the orders due at or before the inference time, so `context.account` is the account as it stood at the decision. The Broker fills due orders sequentially against its true account, while `context.account` stays the snapshot taken before the call: batch sizing must not assume that later statements see earlier fills. Read cash once, keep a local remaining budget, and leave room for price movement and fees.

## Runtime restrictions

Supported imports, in every module of the package, are the package's own modules plus:

```text
__future__, collections, dataclasses, datetime, decimal, functools, itertools, math, statistics, typing,
numpy, pandas, scipy, sklearn, lightgbm, xgboost, statsmodels, torch, transformers, peft, accelerate, einops, vllm
```

Submodules of these libraries (`scipy.stats`, `sklearn.linear_model`, `torch.nn`) are covered. Relative imports, dynamic import or execution, arbitrary file access, process calls and general external I/O are rejected, and so are the other NumPy and pandas load/save methods and pickle-based persistence (`pickle`, `joblib`, `read_pickle`/`to_pickle`). With file I/O limited to the context-rooted calls of Reading PIT data, fitted parameters persist as NumPy arrays, a `torch.save` checkpoint (a `peft` adapter's own weights from `get_peft_model_state_dict`), a `save_pretrained` model directory, or a LightGBM/XGBoost booster file (`booster.save_model(context.state_dir + "/model.txt")`, read back with `lgb.Booster(model_file=...)` or `xgb.Booster().load_model(...)`). Both containers are offline, so a `transformers` model either loads from the pretrained store or is built from its config (`AutoModel.from_config(LlamaConfig(...))`); a hub id passed to `from_pretrained` fails at once.

GPU devices are visible to strategy code only when the run facts' `budgets.strategy_gpu_count` is above zero, so select the device from `torch.cuda.is_available()` and keep the CPU path working; on CPU, torch's thread count is tied to the container's CPU quota. A replay running alone (a smoke backtest, a one-candidate batch, the forward replay) attaches all `budgets.strategy_gpu_count` cards, which lets one model run across them (`tensor_parallel_size`); the candidates of a `validate` call that replay side by side each attach one of the experiment's cards in turn, so they share a card only when the batch runs more candidates at once than the experiment has cards. A replay's `fit` and `generate_orders` containers attach the same cards. Read the count from `torch.cuda.device_count()` rather than assuming one.

Random draws must be reproducible, because two replays of the same span that place different orders are not reproducible evidence. Write the seed once, as one integer assignment to a seed name — `SEED`, `SEED_BASE`, or a name ending in `_SEED` or `_SEED_BASE` (`SEED_BASE = 1000`) — and derive every draw from it, optionally together with the decision date: `day = int(context.inference_at.strftime("%Y%m%d"))`, then `rng = np.random.default_rng([SEED_BASE, day])` with a further fixed integer for each independent stream, and an integer derived from the seed for any library that samples (`random_state=`, `seed=`, `torch.manual_seed`). Another draw of the same strategy is then that one line changed. Never seed from the wall clock or draw from numpy's unseeded global state (`np.random.choice`, `np.random.default_rng()`); `modification_check` refuses both.

## Pretrained models

The only pretrained weights available are the six Hugging Face model directories of a read-only store, `context.pretrained_dir` for the strategy and `/mnt/pretrained` in the session's shell; its `MANIFEST.sha256` lists every file. In an experiment whose `runtime_env.json` shows no `sandbox_spec.pretrained_manifest_sha256`, neither is mounted and `context.pretrained_dir` is empty. The hub is unreachable from both containers, so any other model id fails at once. Measured load times, throughput and memory are in the skill `pretrained-costs`.

| Directory | Weights | Released | Architecture | Use |
| --- | --- | --- | --- | --- |
| `Qwen3.8-27B-FP8` | 27B parameters, block FP8, 28.8 GiB | 2026-08 | Qwen3.8 hybrid: 48 Gated DeltaNet (linear-attention) and 16 gated-attention layers | generation in process with `vllm` on one card (below) |
| `Qwen3.5-2B` | 2B, bfloat16, 4.3 GiB | 2026-02 | Qwen3.5 hybrid: 18 Gated DeltaNet and 6 full-attention layers | hidden-state features, fine-tuning and adapters with `transformers` and `peft`; the 27B's architecture, so it also runs under `vllm` |
| `Qwen3-Embedding-0.6B` | 0.6B, bfloat16, 1.1 GiB | 2025-06 | text embedding: Qwen3, 28 layers, last-token pooling, 1024 dimensions | text embeddings with `transformers`, on a card or on CPU |
| `Qwen3-Embedding-4B` | 4B, bfloat16, 7.5 GiB | 2025-06 | text embedding: Qwen3, 36 layers, last-token pooling, 2560 dimensions | text embeddings with `transformers` |
| `Qwen3-Reranker-4B` | 4B, bfloat16, 7.5 GiB | 2025-06 | reranker: a Qwen3 language model, 36 layers, answering yes or no for one query-document pair | relevance scores of query-document pairs with `transformers` |
| `timesfm-2.5-200m-transformers` | 231M, float32, 0.9 GiB | 2025-09 | TimesFM 2.5: decoder-only time-series transformer, 20 layers, input patches of 32 points, context up to 16,384 points | point and quantile forecasts of univariate series, 128 steps ahead, with `transformers` |

The Qwen3.5 and Qwen3.8 checkpoints are multimodal; they load and serve as text-only language models (`AutoModelForCausalLM` loads the Qwen3.5 ones that way). The embedding models take the last token's hidden state (left padding), L2-normalised; a prefix of the vector, normalised again, is itself an embedding (their cards' Matryoshka training), and their cards put a one-line task instruction before a query (`Instruct: <task>\nQuery:<query>`) but not before a document. The reranker reads one pair through its own chat template and scores it as the probability of "yes" against "no" as the next token. TimesFM 2.5 forecasts each series from its own past alone; this directory is the 2026-02 `transformers` conversion of the 2025-09 checkpoint.

The model cards publish no training-data cutoff, so a model may remember replay dates before its release month and readings through it may be optimistic: an accepted limitation of the experiment, not a reason to avoid the models; encoding what was visible at the decision is legitimate use, asking a model to recall what followed a name or a date is not.

```python
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, TimesFm2_5ModelForPrediction

root = context.pretrained_dir + "/Qwen3.5-2B"
tokenizer = AutoTokenizer.from_pretrained(root)
model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.bfloat16)

encoder_root = context.pretrained_dir + "/Qwen3-Embedding-0.6B"  # "/Qwen3-Embedding-4B" loads the same way
encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_root, padding_side="left")
encoder = AutoModel.from_pretrained(encoder_root, dtype=torch.bfloat16).eval()
with torch.inference_mode():
    batch = encoder_tokenizer(titles, padding=True, truncation=True, max_length=128, return_tensors="pt")
    vectors = torch.nn.functional.normalize(encoder(**batch).last_hidden_state[:, -1].float(), dim=-1)

ranker_root = context.pretrained_dir + "/Qwen3-Reranker-4B"
ranker_tokenizer = AutoTokenizer.from_pretrained(ranker_root, padding_side="left")
ranker = AutoModelForCausalLM.from_pretrained(ranker_root, dtype=torch.bfloat16).eval()
no_yes = ranker_tokenizer.convert_tokens_to_ids(["no", "yes"])
prompts = [ranker_tokenizer.apply_chat_template(
    [{"role": "system", "content": instruction}, {"role": "query", "content": query}, {"role": "document", "content": document}],
    tokenize=False) for query, document in pairs]
with torch.inference_mode():
    batch = ranker_tokenizer(prompts, padding=True, return_tensors="pt")
    relevance = ranker(**batch, logits_to_keep=1).logits[:, -1, no_yes].float().softmax(-1)[:, 1]

forecaster = TimesFm2_5ModelForPrediction.from_pretrained(context.pretrained_dir + "/timesfm-2.5-200m-transformers").eval()
with torch.inference_mode():
    forecast = forecaster(past_values=[torch.tensor(window, dtype=torch.float32) for window in windows], forecast_context_len=512)
median, bands = forecast.mean_predictions, forecast.full_predictions  # (series, 128) and (series, 128, 10)
```

`forecast_context_len` is a multiple of 32: shorter series are padded and longer ones cut to their last points; left at its default of 16,384, every series is padded to that length. `full_predictions` holds a mean forecast and the 10th to 90th percentiles (channels 0 and 1-9), and `mean_predictions` is its channel 5, the median; forecasts are clamped at zero when every input is non-negative (`truncate_negative=False` turns that off).

`peft` is installed for adapters, and `flash-linear-attention` and `causal-conv1d` give the Qwen3.5 linear-attention layers fused GPU kernels that each container compiles on first use, so its first GPU forward and first backward each take up to about a minute longer, on the clock of the call that makes them. Whatever is trained during a replay is written under `context.state_dir` (`model.save_pretrained(context.state_dir + "/adapter")`, or a `torch.save` of the weights), never into the store.

`vllm` runs a store model in process for batched generation: there is no model service and no network, and the engine lives in the strategy's own container, on the card the replay attaches.

```python
from vllm import LLM, SamplingParams

ENGINE = []  # one engine per container, kept for the following calls

def engine(context):
    if not ENGINE:
        ENGINE.append(LLM(model=context.pretrained_dir + "/<model>", max_model_len=8192,
                          max_num_seqs=128, enforce_eager=True, seed=SEED))
    return ENGINE[0]

outputs = engine(context).chat(
    prompts,  # one message list per prompt, the same batch in the same order every time
    SamplingParams(temperature=0.0, max_tokens=8, seed=SEED),
    chat_template_kwargs={"enable_thinking": False},
    use_tqdm=False,
)
texts = [output.outputs[0].text for output in outputs]
```

- An engine claims 90% of its card's memory (`gpu_memory_utilization`), and `fit` and `generate_orders` share the card, so build it in one of them: in `generate_orders`, where it stays loaded for the following calls, or in `fit` with `REFIT_PERIOD = "day"`, writing the day's readings under `context.state_dir` for `generate_orders`. When replays share a card, lower `gpu_memory_utilization` until all of them fit.
- Loading counts on the clock of the call that loads. Without `enforce_eager=True` the engine also runs torch.compile and CUDA-graph capture, redone in every container because their caches live in `/tmp`.
- On one card, Qwen3.8-27B-FP8's linear-attention state holds about 180 concurrent sequences, so `max_num_seqs` must stay below that (vLLM's default 256 refuses to start).
- Two replays of the same bytes must place the same orders. With `enforce_eager=True`, temperature 0 and each decision's prompts sent as one fixed batch in a fixed order (for example the day's shortlist sorted by code), outputs were identical in every container and on every card tried; a compiled engine (the default) gave different outputs in another container, and with prefix caching on (vLLM's default) even on a repeated batch; in both modes any other batch (one prompt alone, a subset, another order) changed most outputs.
- `chat_template_kwargs={"enable_thinking": False}` switches the Qwen3.5-family thinking block off; the template then renders an empty one and the answer follows it.
- The `vllm` package is importable; its server (`vllm serve`) is a session tool, not something a strategy starts.

## Example

The same contract as a two-module package; it shows the contract's shape, and the method and its scale are the session's own. `fit` trains a scikit-learn classifier on the visible PIT window and persists its parameters as arrays with a stamp; `generate_orders` keeps its feature window in module state, extending it by each new day's rows, reloads the parameters only when the stamp changes, and scores the latest cross-section with NumPy. Blocks are labelled with their path under `output/`.

```python
# output/lib/ranker.py
"""Feature construction and a scikit-learn ranker persisted as NumPy arrays."""

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES = ["ret_5", "ret_20", "vol_20"]


def daily_rows(context, filters):
    """Visible daily rows of the PIT view that pass ``filters``, with adjusted closes."""

    frame = pd.read_parquet(
        context.asof_dir + "/daily",
        columns=["ts_code", "trade_date", "close", "adj_factor"],
        filters=filters,
    ).dropna()
    frame = frame[frame["close"] > 0].copy()
    frame["adj_close"] = frame["close"] * frame["adj_factor"]
    return frame


def daily_since(context, lookback_days, held=None):
    """Visible daily rows over a calendar lookback, by symbol and date.

    ``held`` is the window an earlier call of this replay returned: only the
    days after its last one are read, and the days now out of reach dropped.
    """

    start = (context.inference_at - timedelta(days=lookback_days)).strftime("%Y%m%d")
    if held is None or held.empty:
        frame = daily_rows(context, [("trade_date", ">=", start)])
    else:
        frame = pd.concat([held, daily_rows(context, [("trade_date", ">", held["trade_date"].max())])])
    frame = frame[frame["trade_date"] >= start]
    return frame.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def features(frame):
    """Per row: cross-sectionally standardized momentum and volatility."""

    grouped = frame.groupby("ts_code")["adj_close"]
    out = frame[["ts_code", "trade_date"]].copy()
    out["ret_5"] = grouped.pct_change(5).to_numpy()
    out["ret_20"] = grouped.pct_change(20).to_numpy()
    daily_return = grouped.pct_change()
    out["vol_20"] = daily_return.groupby(frame["ts_code"]).transform(
        lambda series: series.rolling(20).std()
    ).to_numpy()
    out = out.dropna()
    for column in FEATURES:
        by_day = out.groupby("trade_date")[column]
        spread = by_day.transform("std").replace(0.0, np.nan)
        out[column] = ((out[column] - by_day.transform("mean")) / spread).fillna(0.0)
    return out


def fit_classifier(frame, min_samples):
    """Logistic model of beating the cross-sectional median 5-day forward return."""

    sample = features(frame)
    forward = frame.groupby("ts_code")["adj_close"].shift(-5) / frame["adj_close"] - 1.0
    sample = sample.join(forward.rename("forward"), how="inner").dropna()
    median = sample.groupby("trade_date")["forward"].transform("median")
    sample["label"] = (sample["forward"] > median).astype(int)
    if len(sample) < min_samples or sample["label"].nunique() < 2:
        return np.zeros(len(FEATURES)), 0.0
    model = LogisticRegression(C=0.5).fit(sample[FEATURES].to_numpy(), sample["label"].to_numpy())
    return model.coef_[0], float(model.intercept_[0])


def score(cross_section, coef, intercept):
    logits = cross_section[FEATURES].to_numpy() @ coef + intercept
    return 1.0 / (1.0 + np.exp(-logits))
```

```python
# output/main.py
"""Entry module: fit persists array parameters, generate_orders scores with NumPy."""

import numpy as np

from lib.ranker import daily_since, features, fit_classifier, score

REFIT_PERIOD = "quarter"
FIT_LOOKBACK_DAYS = 400
FEATURE_LOOKBACK_DAYS = 45
MIN_FIT_SAMPLES = 100
TOP_N = 10
CASH_FRACTION = 0.95
# Module state of the inference worker, kept from one decision to the next.
WINDOW = {}
MODEL = {}


def fit(context):
    coef, intercept = fit_classifier(daily_since(context, FIT_LOOKBACK_DAYS), MIN_FIT_SAMPLES)
    np.savez(context.state_dir + "/ranker.npz", coef=coef, intercept=np.array([intercept]))
    np.save(context.state_dir + "/stamp.npy", np.array([int(context.inference_at.strftime("%Y%m%d"))]))


def fitted(context):
    """The parameters of the latest fit, read from disk only after a refit."""

    stamp = int(np.load(context.state_dir + "/stamp.npy")[0])
    if MODEL.get("stamp") != stamp:
        params = np.load(context.state_dir + "/ranker.npz")
        MODEL.update(stamp=stamp, coef=params["coef"], intercept=float(params["intercept"][0]))
    return MODEL["coef"], MODEL["intercept"]


def generate_orders(context):
    if context.account.positions:
        return []
    execution_at = context.inference_at.replace(hour=9, minute=30, second=0, microsecond=0)
    if context.inference_at > execution_at:
        return []
    frame = daily_since(context, FEATURE_LOOKBACK_DAYS, WINDOW.get("daily"))
    WINDOW["daily"] = frame
    if frame.empty:
        return []
    latest = frame["trade_date"].max()
    cross_section = features(frame)
    cross_section = cross_section[cross_section["trade_date"] == latest].copy()
    if cross_section.empty:
        return []
    coef, intercept = fitted(context)
    cross_section["score"] = score(cross_section, coef, intercept)
    ranked = cross_section.sort_values(["score", "ts_code"], ascending=[False, True])
    prices = frame[frame["trade_date"] == latest].set_index("ts_code")["close"]
    symbols = ranked["ts_code"].tolist()[:TOP_N]
    remaining = float(context.account.cash) * CASH_FRACTION
    orders = []
    for index, symbol in enumerate(symbols):
        price = float(prices[symbol])
        quantity = int(remaining / (len(symbols) - index) / price // 100 * 100)
        if quantity <= 0:
            continue
        orders.append(
            {
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "execute_at": execution_at.isoformat(),
                "reason": "logistic_rank_equal_budget",
            }
        )
        remaining -= quantity * price
    return orders
```
