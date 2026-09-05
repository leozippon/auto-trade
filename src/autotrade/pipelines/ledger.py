"""The single experiment ledger (docs/pipeline-design.md §4.1).

One JSONL file per experiment. Records are distinguished by ``record_type``
(``fold`` / ``meta_learning`` / ``heldout``); Steps are lightweight
summaries inside the fold record's ``steps[]``, never separate files.
``attempt_failed`` records are appended when a run throws before its success
record — they carry the error evidence and are ignored by every reader that
selects the success types, so a failed attempt is re-runnable but auditable.
A run that is killed outright cannot append that record itself, so every run
also leaves a host-only marker (:class:`RunMarkers`) that the next worker start
turns into the missing ``attempt_failed``.

A ``fold`` or ``heldout`` row with ``state_changed_during_test=true`` is an
integrity failure, not a success: it is persisted before fail-fast so the
corruption is auditable, then every resume, retry, held-out, and parent
selection must refuse until a human rolls the dirty frozen trees back.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist

from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.environment.runtime import (
    append_versioned_jsonl,
    read_versioned_jsonl,
    utc_now_iso,
    write_json_atomic,
)

# Stamped on every appended record; bump when the record shape changes.
LEDGER_RECORD_SCHEMA_VERSION = 1
RECORD_TYPES = ("fold", "meta_learning", "heldout", "attempt_failed")
LINK_KEYS = ("experiment_id", "epoch_id", "fold_id", "run_id")
DURABLE_SUCCESS_TYPES = ("fold", "meta_learning", "heldout")
_INTEGRITY_RECORD_TYPES = frozenset({"fold", "heldout"})

# Host-only, never mounted into a sandbox and never Agent-visible.
RUN_MARKER_DIR = ".host/runs"
INTERRUPTED_RUN_ERROR = (
    "RunInterrupted: the run process exited before it wrote a ledger record"
)
# The marker exists to survive exactly the events that can also tear it (the
# atomic write renames without fsync, so a SIGKILL/OOM/host reset can leave a
# zero-length or truncated file). Such a marker still proves a run died, so it
# becomes an ``attempt_failed`` too, with its unknown link keys named as unknown.
UNREADABLE_RUN_MARKER_ERROR = (
    "RunMarkerUnreadable: the run process exited before it wrote a ledger "
    "record and its run marker could not be read back"
)
UNKNOWN_MARKER_LINK_KEY = "unknown"


class FrozenArtifactMutated(RuntimeError):
    """Frozen output/models changed during Test or Held-out.

    The integrity record is already in the ledger. Same-process retries,
    resume, rerun, held-out, and parent selection must refuse until a human
    rolls back the dirty trees.
    """


class FrozenArtifactRestoreFailed(FrozenArtifactMutated):
    """Mutation was detected, but pre-evaluation frozen bytes could not be restored.

    Worse than ``FrozenArtifactMutated``: the live trees must not be treated as
    clean. The integrity record is still written when the caller can append it.
    """


def is_frozen_artifact_mutation(record: Mapping[str, object]) -> bool:
    """True when a fold/held-out row flags frozen output/models as changed."""
    return (
        record.get("record_type") in _INTEGRITY_RECORD_TYPES
        and record.get("state_changed_during_test") is True
    )


def is_durable_success_record(
    record: Mapping[str, object],
    *,
    record_types: tuple[str, ...] | None = None,
) -> bool:
    """Fold/meta/held-out rows that may be treated as completed work."""
    types = record_types if record_types is not None else DURABLE_SUCCESS_TYPES
    if record.get("record_type") not in types:
        return False
    return not is_frozen_artifact_mutation(record)


def assert_no_frozen_artifact_mutation(records: list[dict[str, object]]) -> None:
    """Refuse further pipeline work while an integrity-failure row remains."""
    for record in records:
        if not is_frozen_artifact_mutation(record):
            continue
        phase = (
            "held-out" if record.get("record_type") == "heldout" else "frozen test"
        )
        raise FrozenArtifactMutated(
            "strategy or model artifacts changed during "
            f"{phase}; refuse retry, resume, rerun, held-out, and parent "
            "selection until the frozen trees are rolled back"
        )


def latest_fold_records(records: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    """Latest successful fold record per (epoch, fold): the ledger is append-only, so a
    re-run appends a superseding record. Formal consumers (reporting, console)
    must never double-count earlier attempts. Integrity-failure rows are not
    adopted as the official latest result."""
    latest: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        if not is_durable_success_record(record, record_types=("fold",)):
            continue
        latest[(str(record.get("epoch_id")), str(record.get("fold_id")))] = record
    return latest


def latest_heldout_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Latest successful record per held-out period (a fold re-run replays held-out, so
    earlier period records are superseded, not removed). Integrity-failure rows
    are not adopted as the official latest result."""
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        if not is_durable_success_record(record, record_types=("heldout",)):
            continue
        latest[str(record.get("fold_id"))] = record
    return [latest[key] for key in sorted(latest)]


def walk_forward_transitions(
    fold_records: list[dict[str, object]], *, epoch_id: str, test_stage: bool
) -> dict[str, object]:
    """Out-of-sample transitions of one Epoch for graduation term (b).

    Without a Test stage a transition is the host's ``parent_control`` of every
    Fold after the Epoch's first: the previous Fold's frozen strategy replayed
    on this Fold's Validation window, scored on its new period alone when the
    window trails over several (``transition_result``). With a Test stage it is
    each Fold's frozen Test. A transition counts as positive only when its
    result exists and its excess return over the benchmark is > 0; a failed or
    missing result is a transition that proved nothing.
    """
    folds = sorted(
        (
            record
            for record in latest_fold_records(fold_records).values()
            if str(record.get("epoch_id")) == epoch_id
        ),
        key=lambda record: str(record.get("validation_period") or ""),
    )
    if test_stage:
        source = "frozen_test"
        results = [record.get("test_result") for record in folds]
        percentiles: list[float] = []
    else:
        source = "parent_control"
        results = [
            transition_result(record.get("parent_control"))
            for record in folds[1:]
        ]
        # The null percentile of each counted transition on the span it is
        # scored on; a control that ran no null control contributes nothing.
        percentiles = [
            value
            for value in (
                _finite(
                    (transition_null_control(record.get("parent_control")) or {}).get(
                        "excess_percentile"
                    )
                )
                for record in folds[1:]
            )
            if value is not None
        ]
    return {
        "source": source,
        "epoch_id": epoch_id,
        "transitions": len(results),
        "positive_excess": sum(1 for result in results if _excess_positive(result)),
        # Diagnostic beside the count: where the transitions sat inside
        # random-name replays of their own trade skeletons, on average. Never
        # part of the term; None when no transition carried a null control.
        "mean_excess_percentile": (
            sum(percentiles) / len(percentiles) if percentiles else None
        ),
    }


def frozen_selection(
    fold_records: list[dict[str, object]], *, artifact_id: str
) -> dict[str, object] | None:
    """Selection evidence of the Fold that froze ``artifact_id``.

    The strategy a Held-out replays was nominated out of one Fold's search,
    and that Fold's record is the only place the search's deflated-Sharpe
    probability (§2.4) and the frozen node's own null percentile live. Both
    are diagnostics; the verdict carries them beside its gating metrics so a
    graduation is read together with how much of it selection alone explains.
    ``None`` when no Fold record froze the artifact (a Meta-regularized
    artifact that never went through a Fold), and each field is ``None`` when
    the record predates its block.
    """

    matches = [
        record
        for record in latest_fold_records(fold_records).values()
        if record.get("frozen_strategy_artifact_id") == artifact_id
    ]
    if not matches:
        return None
    record = matches[-1]
    selection = record.get("selection_statistics")
    selection = selection if isinstance(selection, Mapping) else {}
    null = record.get("null_control")
    null = null if isinstance(null, Mapping) else {}
    candidates = selection.get("candidates_evaluated")
    return {
        "fold_id": record.get("fold_id"),
        "candidates_evaluated": (
            candidates
            if isinstance(candidates, int) and not isinstance(candidates, bool)
            else None
        ),
        "deflated_sharpe_probability": _finite(selection.get("deflated_sharpe_probability")),
        "validation_excess_percentile": _finite(null.get("excess_percentile")),
    }


def transition_result(control: object) -> Mapping[str, object] | None:
    """The result one parent control is graded on: its new period, else the window.

    A Fold whose Validation window trails over several periods has already seen
    all but its last one, so its ``parent_control`` records ``step_result`` --
    that last period alone -- and the transition is scored on it. A single-period
    window (and every ledger written before the rolling schedule) has none, and
    the whole ``validation_result`` is the transition. The single source for
    both the graduation term and the report, so the two can never disagree.
    """

    if not isinstance(control, Mapping):
        return None
    step = control.get("step_result")
    if isinstance(step, Mapping):
        return step
    result = control.get("validation_result")
    return result if isinstance(result, Mapping) else None


def transition_null_control(control: object) -> Mapping[str, object] | None:
    """The null-control block of the span ``transition_result`` scores.

    A control graded on its ``step_result`` is ranked against the ``step``
    sub-block of its null control, never the whole window's; otherwise the
    window's block is the one. None when no null control ran.
    """

    if not isinstance(control, Mapping):
        return None
    null = control.get("null_control")
    if not isinstance(null, Mapping):
        return None
    if isinstance(control.get("step_result"), Mapping):
        step = null.get("step")
        return step if isinstance(step, Mapping) else None
    return null


def _excess_positive(result: object) -> bool:
    if not isinstance(result, Mapping) or result.get("status") == "failed":
        return False
    benchmark = result.get("benchmark")
    total = result.get("total_return")
    bench = benchmark.get("benchmark_return") if isinstance(benchmark, Mapping) else None
    if isinstance(total, bool) or isinstance(bench, bool):
        return False
    if not isinstance(total, (int, float)) or not isinstance(bench, (int, float)):
        return False
    return float(total) - float(bench) > 0


# --- selection statistics (docs/pipeline-design.md §2.4) ---------------------
#
# A Fold nominates one winner out of the candidates it evaluated on the very
# window their returns are quoted on, so the winner's Sharpe is the maximum of
# a search, not a draw. The deflated Sharpe ratio (Bailey & López de Prado,
# "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
# Overfitting and Non-Normality", Journal of Portfolio Management 40(5), 2014)
# is the probability that the selected Sharpe exceeds what the best of N
# equally plausible trials would reach by chance alone. It is informational:
# nothing in the pipeline gates on it.
_EULER_MASCHERONI = 0.5772156649015329
# Skew and kurtosis of a return series shorter than a trading month say more
# about the sample than about the strategy, and √(T−1) barely separates
# anything there. Below this the probability is reported as unavailable.
DEFLATED_SHARPE_MIN_RETURN_DAYS = 20


def deflated_sharpe(
    *,
    observed_sharpe: object,
    trial_sharpes: Sequence[object],
    returns: Sequence[object],
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> dict[str, object]:
    """Deflated-Sharpe block for one selected candidate out of N trials.

    ``observed_sharpe`` and ``trial_sharpes`` are annualized exactly as the
    replay summaries report them; ``returns`` are the selected candidate's
    per-period (daily) returns. The formula works in per-period units, so
    every Sharpe is divided by ``√periods_per_year`` going in and
    ``sharpe_star`` is annualized again coming out — the returned figures
    therefore read directly beside the Sharpe the console shows.

    With V the sample variance of the N trial Sharpes and γ the
    Euler-Mascheroni constant, the expected maximum Sharpe under N trials is

        SR* = √V · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

    and the probability that the observed Sharpe exceeds it is

        Φ[ (SR − SR*)·√(T−1) / √(1 − γ₃·SR + (γ₄−1)/4·SR²) ]

    with γ₃ the skew and γ₄ the (non-excess, normal = 3) kurtosis of the T
    returns. ``deflated_sharpe_probability`` is ``None`` — never 0 — whenever it is
    undefined; ``unavailable_reason`` then names which condition failed.
    """

    scale = math.sqrt(float(periods_per_year))
    trials = [value for value in map(_finite, trial_sharpes) if value is not None]
    series = [value for value in map(_finite, returns) if value is not None]
    block: dict[str, object] = {
        "deflated_sharpe_probability": None,
        "trials": len(trials),
        "sharpe_star": None,
        "trial_sharpe_std": None,
        "observed_sharpe": _finite(observed_sharpe),
        "return_days": len(series),
        "return_skew": None,
        "return_kurtosis": None,
        "unavailable_reason": None,
    }
    if block["observed_sharpe"] is None:
        block["unavailable_reason"] = "no_observed_sharpe"
        return block
    if len(trials) < 2:
        # One trial has no dispersion to estimate the maximum's spread from,
        # and zero trials is not a search at all.
        block["unavailable_reason"] = "fewer_than_two_trials"
        return block
    if len(series) < DEFLATED_SHARPE_MIN_RETURN_DAYS:
        block["unavailable_reason"] = "return_series_too_short"
        return block
    count = len(trials)
    mean_trial = sum(trials) / count
    variance = sum((value - mean_trial) ** 2 for value in trials) / (count - 1)
    # Identical trials leave V = 0, hence SR* = 0: with no dispersion there is
    # nothing to deflate and the statistic degenerates to the probabilistic
    # Sharpe ratio against zero, which is the honest answer.
    trial_std = math.sqrt(variance)
    normal = NormalDist()
    sharpe_star = trial_std * (
        (1.0 - _EULER_MASCHERONI) * normal.inv_cdf(1.0 - 1.0 / count)
        + _EULER_MASCHERONI * normal.inv_cdf(1.0 - 1.0 / (count * math.e))
    )
    block["trial_sharpe_std"] = trial_std
    block["sharpe_star"] = sharpe_star
    days = len(series)
    mean_return = sum(series) / days
    centered = [value - mean_return for value in series]
    second = sum(value**2 for value in centered) / days
    if second <= 0:
        block["unavailable_reason"] = "zero_return_variance"
        return block
    skew = (sum(value**3 for value in centered) / days) / second**1.5
    kurtosis = (sum(value**4 for value in centered) / days) / second**2
    block["return_skew"] = skew
    block["return_kurtosis"] = kurtosis
    sharpe = float(block["observed_sharpe"]) / scale
    variance_term = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if variance_term <= 0:
        # The Sharpe estimator's own variance is not positive under these
        # moments; reporting a probability from it would be a fabrication.
        block["unavailable_reason"] = "undefined_sharpe_variance"
        return block
    statistic = (sharpe - sharpe_star / scale) * math.sqrt(days - 1) / math.sqrt(
        variance_term
    )
    block["deflated_sharpe_probability"] = normal.cdf(statistic)
    return block


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def experiment_verdict(
    records: list[dict[str, object]], *, strict: bool = True
) -> dict[str, object] | None:
    """The experiment's graduation verdict from its latest Held-out records.

    Each ``heldout`` row carries the per-period verdict the pipeline computed
    from ``AcceptanceRules.heldout_verdict``; the experiment graduates only when
    every Held-out period did. ``None`` until a Held-out record exists. A row
    without a verdict block is not the current ledger format: ``strict``
    callers (report, terminal status) refuse it, the console read model
    (``strict=False``) shows no verdict rather than inventing one.
    """
    latest = latest_heldout_records(records)
    if not latest:
        return None
    reasons: list[str] = []
    periods: list[dict[str, object]] = []
    graduated = True
    for record in latest:
        verdict = record.get("verdict")
        if not isinstance(verdict, Mapping):
            if not strict:
                return None
            raise ValueError(
                f"heldout record {record.get('fold_id')!r} carries no verdict block"
            )
        label = str(record.get("period") or record.get("fold_id") or "")
        status = str(verdict.get("status") or "")
        if status != "graduated":
            graduated = False
        for reason in verdict.get("reasons") or ():
            text = f"{label}: {reason}" if len(latest) > 1 else str(reason)
            if text not in reasons:
                reasons.append(text)
        periods.append({"period": label, **dict(verdict)})
    return {
        "status": "graduated" if graduated else "discarded",
        "reasons": reasons,
        "periods": periods,
    }


class ExperimentLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, object]) -> None:
        record_type = record.get("record_type")
        if record_type not in RECORD_TYPES:
            raise ValueError(f"unsupported record_type: {record_type!r}")
        missing = [key for key in LINK_KEYS if not record.get(key)]
        if missing:
            raise ValueError(f"ledger record missing link keys: {missing}")
        append_versioned_jsonl(
            self.path, record, schema_version=LEDGER_RECORD_SCHEMA_VERSION
        )

    def rewrite(self, records: list[dict[str, object]]) -> None:
        """Atomic full rewrite for migrations and Fold/Held-out rollback.

        The rolling-upgrade write-isolation guard is part of the primitive,
        not a procedural convention: the rewrite refuses while the owning
        experiment worker is alive, and refuses records that do not already
        carry the current schema stamp (a migration or rollback must hand over
        fully migrated records). Callers must go through this method instead of
        editing the file by hand.
        """
        # Local import: hitl_state pulls in the config/session stack, which
        # this module must not load for its plain append/read paths.
        from autotrade.pipelines.hitl_state import assert_no_live_writer

        assert_no_live_writer(self.path.parent.parent)
        for record in records:
            version = record.get("schema_version")
            if type(version) is not int or version != LEDGER_RECORD_SCHEMA_VERSION:
                raise ValueError(
                    f"rewrite requires fully migrated records; got schema_version {version!r}"
                )
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def read(self, record_type: str | None = None) -> list[dict[str, object]]:
        records = read_versioned_jsonl(
            self.path,
            schema_version=LEDGER_RECORD_SCHEMA_VERSION,
            label="ledger record",
        )
        if record_type is None:
            return records
        return [record for record in records if record.get("record_type") == record_type]


class RunMarkers:
    """In-flight run markers that make a killed run auditable.

    Fold, Meta and Held-out runs append ``attempt_failed`` themselves when they
    catch an exception, but a SIGKILL, an OOM kill or a host reset leaves no
    code to run: the trace file stops mid-event and the ledger under-reports the
    attempt. Each run therefore writes a marker holding its link keys before it
    starts and deletes it once its ledger record — a business record or an
    ``attempt_failed`` — is durable, so a leftover marker is exactly the
    evidence of a run that died silently.
    """

    def __init__(self, experiment_dir: str | Path) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.root = self.experiment_dir / RUN_MARKER_DIR

    def begin(self, attempt: Mapping[str, object]) -> None:
        missing = [key for key in LINK_KEYS if not attempt.get(key)]
        if missing:
            raise ValueError(f"run marker missing link keys: {missing}")
        self.root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.mkdir(exist_ok=True, mode=0o700)
        write_json_atomic(
            self.root / f"{attempt['run_id']}.json",
            {**dict(attempt), "started_at": utc_now_iso()},
        )

    def finish(self, run_id: str) -> None:
        (self.root / f"{run_id}.json").unlink(missing_ok=True)

    def recover(self, ledger: ExperimentLedger) -> list[dict[str, object]]:
        """Record every run that died before its ledger record, then forget it.

        Called once when an experiment worker starts, the only moment at which
        no run of this experiment is in flight. A marker whose run already
        reached the ledger (the process died between the append and the marker
        cleanup) is dropped without a second record, and every handled marker is
        removed, so repeated restarts never duplicate a record.

        An unreadable marker is handled the same way: it is still the evidence
        of a dead run, and it must never be able to fail every later worker
        start. Its file name is the run id, the experiment directory is the
        experiment id, and the link keys it cannot supply are recorded as
        unknown rather than guessed.
        """
        if not self.root.is_dir():
            return []
        recorded = {
            str(record.get("run_id"))
            for record in ledger.read()
            if record.get("run_id")
        }
        appended: list[dict[str, object]] = []
        for path in sorted(self.root.glob("*.json")):
            marker, unreadable = _read_run_marker(path)
            run_id = str(marker.get("run_id") or path.stem)
            if run_id not in recorded:
                record = {
                    **marker,
                    "record_type": "attempt_failed",
                    "experiment_id": marker.get("experiment_id")
                    or self.experiment_dir.name,
                    "epoch_id": marker.get("epoch_id") or UNKNOWN_MARKER_LINK_KEY,
                    "fold_id": marker.get("fold_id") or UNKNOWN_MARKER_LINK_KEY,
                    "run_id": run_id,
                    "error": (
                        f"{UNREADABLE_RUN_MARKER_ERROR}: {path.name}: {unreadable}"
                        if unreadable
                        else INTERRUPTED_RUN_ERROR
                    ),
                }
                ledger.append(record)
                appended.append(record)
            path.unlink(missing_ok=True)
        return appended


def _read_run_marker(path: Path) -> tuple[dict[str, object], str]:
    """A marker's fields, or ``({}, reason)`` when the file is not a JSON object."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON ({exc})"
    if not isinstance(payload, Mapping):
        return {}, f"payload is {type(payload).__name__}, not a JSON object"
    return dict(payload), ""
