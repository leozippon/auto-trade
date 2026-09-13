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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist

from autotrade.environment.replay.stats import TRADING_DAYS_PER_YEAR
from autotrade.environment.replay.style import (
    STYLE_ARTIFACT_NAME,
    daily_returns_from_curve,
    window_neutralized_excess,
)
from autotrade.environment.runtime import (
    append_versioned_jsonl,
    read_versioned_jsonl,
    utc_now_iso,
    write_json_atomic,
)

# Stamped on every appended record; bump when the record shape changes.
LEDGER_RECORD_SCHEMA_VERSION = 1
RECORD_TYPES = (
    "fold",
    "meta_learning",
    "heldout",
    "deployment_adjustment",
    "attempt_failed",
)
LINK_KEYS = ("experiment_id", "epoch_id", "fold_id", "run_id")
DURABLE_SUCCESS_TYPES = ("fold", "meta_learning", "heldout", "deployment_adjustment")
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


def latest_meta_records(
    records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Latest successful meta-learning record per session key. A re-run appends
    a superseding record, so only the last one describes what that session
    left behind -- its published PRIOR, and any artifact it regularized."""
    latest: dict[str, dict[str, object]] = {}
    for record in records:
        if not is_durable_success_record(record, record_types=("meta_learning",)):
            continue
        session_key = str(record.get("session_key") or "")
        if session_key:
            latest[session_key] = record
    return latest


def preceding_meta_regularization(
    records: list[dict[str, object]],
) -> dict[str, object] | None:
    """The Meta outcome this Fold inherits, when a Meta ran right before it.

    A Meta may claim a cleanup in its PRIOR that the Pipeline then refused --
    the check rejected it, or the package would not run -- and the Fold mounts
    the unchanged parent while reading a PRIOR that says otherwise. The ledger
    is append-ordered and every Meta is followed by the Fold it prepares, so
    the preceding session is the last durable record: a meta row means this
    Fold inherits its verdict, anything else means no Meta ran in between.
    ``reason`` is the refusal text when there is one, so a PRIOR claim about a
    regularization can be checked against what was actually frozen.
    """

    for record in reversed(records):
        if not is_durable_success_record(record):
            continue
        if record.get("record_type") != "meta_learning":
            return None
        status = str(record.get("status") or "")
        smoke = record.get("regularization_smoke")
        reasons: list[str] = []
        if isinstance(smoke, Mapping) and smoke.get("status") == "failed":
            reasons.append(f"regularization_smoke: {smoke.get('error')}")
        check = record.get("modification_check")
        if isinstance(check, Mapping) and check.get("allowed_to_backtest") is False:
            listed = check.get("reasons")
            reasons.extend(
                str(item)
                for item in (listed if isinstance(listed, Sequence) and not isinstance(listed, str) else [])
            )
        reason = "; ".join(item for item in reasons if item)
        return {
            "status": status,
            **({"reason": reason[:_META_REGULARIZATION_REASON_MAX_CHARS]} if reason else {}),
        }
    return None


def rerun_absorbed(
    records: list[dict[str, object]], session_key: str, token: str
) -> bool:
    """Whether one console re-run request already has its fold record.

    The single definition of "this token is spent", for the runner deciding
    whether to re-run the session and for the worker deciding whether a
    completed experiment still owes work. An integrity-flagged row is not a
    completed re-run, so a token it carries stays outstanding.
    """
    latest = next(
        (
            record
            for record in reversed(records)
            if str(record.get("session_key") or "") == session_key
            and is_durable_success_record(record, record_types=("fold",))
        ),
        None,
    )
    return latest is not None and str(latest.get("rerun_id") or "") == token


# Same bound and purpose as a failed control's reason: enough to act on, and
# it travels in the next Fold's system prompt.
_META_REGULARIZATION_REASON_MAX_CHARS = 400


def _artifact_id(value: object) -> str | None:
    """A non-empty artifact id, or None when the record names none."""

    return value if isinstance(value, str) and value else None


def _epoch_folds(
    fold_records: list[dict[str, object]], *, epoch_id: str
) -> list[dict[str, object]]:
    """One Epoch's latest Fold records, in schedule order."""

    return sorted(
        (
            record
            for record in latest_fold_records(fold_records).values()
            if str(record.get("epoch_id")) == epoch_id
        ),
        key=lambda record: str(record.get("validation_period") or ""),
    )


@dataclass(frozen=True)
class Transition:
    """One walk-forward transition: whose it is, its result, and where it lives.

    ``result_ref`` points at the replay record the result was projected from,
    which is where the neutralized excess of a span is recomputed when the
    record itself does not carry one (:func:`transition_neutralized_excess`).
    """

    artifact_id: str | None
    result: object
    result_ref: str


def baseline_anchor_artifacts(fold_records: list[dict[str, object]]) -> frozenset[str]:
    """Every artifact frozen as a baseline anchor (docs/pipeline-design.md §2.2).

    An anchor is the lineage's control -- the weak baseline a parentless Fold
    must freeze so the next Fold has something to beat -- not a candidate
    anyone judged worth shipping. Its id is the single handle on that: the
    transitions it replayed score a control and are excluded from graduation,
    and the experiment refuses to deliver one to Held-out.
    """

    return frozenset(
        str(record["frozen_strategy_artifact_id"])
        for record in latest_fold_records(fold_records).values()
        if record.get("baseline_anchor") is True
        and _artifact_id(record.get("frozen_strategy_artifact_id"))
    )


def _transition_rows(
    folds: list[dict[str, object]], *, test_stage: bool
) -> list[Transition]:
    """The transitions of one Epoch, in schedule order.

    Without a Test stage a transition is the host's ``parent_control`` of every
    Fold after the Epoch's first, and the artifact it scores is the parent that
    control replayed. With a Test stage it is each Fold's frozen Test, which
    scores that Fold's own frozen artifact. The single source for both the
    chain-wide count (:func:`walk_forward_transitions`) and the shipped
    artifact's own count (:func:`final_artifact_transitions`), so the two can
    never disagree about which artifact a transition belongs to. The id is
    ``None`` when the record names none, which matches no artifact.

    Every transition the schedule produced, anchors included: the callers drop
    the ones that replayed an anchor (:func:`_counted`) but still need the full
    row count to tell "this schedule confirmed nothing" from "this schedule
    could confirm nothing".
    """

    if test_stage:
        return [
            Transition(
                _artifact_id(record.get("frozen_strategy_artifact_id")),
                record.get("test_result"),
                str(record.get("test_result_ref") or ""),
            )
            for record in folds
        ]
    return [
        _parent_control_transition(record.get("parent_control")) for record in folds[1:]
    ]


def _counted(rows: list[Transition], anchors: frozenset[str]) -> list[Transition]:
    """The transitions that score something: an anchor's do not.

    A baseline anchor is a control the experiment never delivers, so its
    forward record leaves the numerator and the denominator alike -- counting
    it either way would let a placebo's record stand in for a candidate's.
    """

    return [row for row in rows if row.artifact_id not in anchors]


def _parent_control_transition(control: object) -> Transition:
    """One host parent control as the transition it is."""

    control = control if isinstance(control, Mapping) else {}
    return Transition(
        _artifact_id(control.get("parent_strategy_artifact_id")),
        transition_result(control),
        str(control.get("validation_result_ref") or ""),
    )


def parent_control_excess(control: object) -> float | None:
    """The neutralized excess one parent control is graded on, for readers that
    hold a single Fold record rather than an Epoch's rows (the console row, the
    report). Same function, same number as the count above it."""

    return transition_neutralized_excess(_parent_control_transition(control))


def epoch_transitions(
    fold_records: list[dict[str, object]], *, epoch_id: str, test_stage: bool
) -> list[Transition]:
    """One Epoch's transitions as the counts were taken on them.

    The public seam over :func:`_transition_rows` for readers that need the
    transitions themselves rather than the totals -- the console averages the
    same figures the count is taken on, so a tile and the number above it
    cannot describe different sets.
    """

    return _counted(
        _transition_rows(
            _epoch_folds(fold_records, epoch_id=epoch_id), test_stage=test_stage
        ),
        baseline_anchor_artifacts(fold_records),
    )


def walk_forward_transitions(
    fold_records: list[dict[str, object]], *, epoch_id: str, test_stage: bool
) -> dict[str, object]:
    """Out-of-sample transitions of one Epoch for graduation term (b).

    Without a Test stage a transition is the host's ``parent_control`` of every
    Fold after the Epoch's first: the previous Fold's frozen strategy replayed
    on this Fold's Validation window, scored on its new period alone when the
    window trails over several (``transition_result``). With a Test stage it is
    each Fold's frozen Test. A transition counts as positive only when its
    size/beta-neutralized excess is > 0 (``transition_neutralized_excess``); a
    failed or missing result is a transition that proved nothing, and one whose
    neutralized excess cannot be established at all is reported as
    ``unmeasured`` rather than graded on its raw excess.

    This is the development *chain's* record: the transitions it counts mostly
    replay earlier artifacts of the lineage, not the one Held-out ships. What
    that shipped artifact proved forward on its own is
    :func:`final_artifact_transitions`. Transitions that replayed a baseline
    anchor are not part of either: an anchor is a control the experiment never
    delivers, so its forward record says nothing about a graduating strategy.
    """
    folds = _epoch_folds(fold_records, epoch_id=epoch_id)
    scheduled = _transition_rows(folds, test_stage=test_stage)
    rows = _counted(scheduled, baseline_anchor_artifacts(fold_records))
    if test_stage:
        source = "frozen_test"
        percentiles: list[float] = []
    else:
        source = "parent_control"
        # The null percentile of each counted transition on the span it is
        # scored on; a control that ran no null control contributes nothing.
        percentiles = [
            value
            for value in (
                finite_number(
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
        # What the schedule produced, before the anchors came out: a term that
        # counts nothing because every transition replayed a control is not the
        # same as a schedule with no transitions to count.
        "scheduled": len(scheduled),
        **_counts(rows),
        # Diagnostic beside the count: where the transitions sat inside
        # random-name replays of their own trade skeletons, on average. Never
        # part of the term; None when no transition carried a null control.
        "mean_excess_percentile": (
            sum(percentiles) / len(percentiles) if percentiles else None
        ),
    }


def final_artifact_transitions(
    fold_records: list[dict[str, object]],
    *,
    epoch_id: str,
    test_stage: bool,
    artifact_id: str,
) -> dict[str, object]:
    """The transitions of one Epoch that scored ``artifact_id`` itself.

    Graduation's walk-forward term counts the whole chain, and a mechanism
    first frozen in the Epoch's last Fold inherits none of that record: every
    transition replayed the parent it replaced. This is the subset that
    actually replayed the artifact Held-out ships, counted the same way
    (:func:`_transition_rows`), so an artifact with no forward quarter of its
    own is visible as ``transitions == 0`` rather than hidden behind the
    chain's average.
    """

    rows = _counted(
        _transition_rows(
            _epoch_folds(fold_records, epoch_id=epoch_id), test_stage=test_stage
        ),
        baseline_anchor_artifacts(fold_records),
    )
    own = [row for row in rows if row.artifact_id == artifact_id]
    return {
        "artifact_id": artifact_id,
        "epoch_id": epoch_id,
        **_counts(own),
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
    return {
        "fold_id": record.get("fold_id"),
        "candidates_evaluated": _count(selection.get("candidates_evaluated")),
        "deflated_sharpe_probability": finite_number(selection.get("deflated_sharpe_probability")),
        # N as the formula actually used it: the finite trial Sharpes, which is
        # not always ``candidates_evaluated`` (a kept parent joins the trials,
        # a non-finite Sharpe drops out). A probability from two trials barely
        # deflates anything, so the count has to be read beside it.
        "deflated_sharpe_trials": _count(selection.get("trials")),
        "validation_excess_percentile": finite_number(null.get("excess_percentile")),
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


def transition_neutralized_excess(transition: Transition) -> float | None:
    """The size/beta-neutralized excess one transition is graded on.

    The single source for every reader of a transition's sign: the graduation
    terms, the report and the console all call it, so none of them can grade on
    a different number. CSI 300 fell 21.6% in 2022 and rose 16-18% in the 2024
    and 2025 quarters, so a raw excess over it says as much about which quarter
    a transition landed in as about the strategy; the neutralized figure is the
    one that survives that.

    Read in this order, and never past it: the figure the record carries; else
    the same figure recomputed on the graded span from the replay's own style
    sidecar (records written before results carried a per-quarter figure);
    else ``None``. ``None`` means the transition's sign is unknown, which the
    counts report as ``unmeasured`` and the verdict fails on -- falling back to
    the raw excess would silently grade graduation on the number this rule
    exists to stop using.
    """

    result = transition.result
    if not isinstance(result, Mapping) or result.get("status") == "failed":
        return None
    benchmark = result.get("benchmark")
    recorded = finite_number(
        benchmark.get("neutralized_excess_return") if isinstance(benchmark, Mapping) else None
    )
    if recorded is not None:
        return recorded
    return _derived_excess(
        transition.result_ref,
        str(result.get("start") or ""),
        str(result.get("end") or ""),
    )


@lru_cache(maxsize=512)
def _derived_excess(result_ref: str, start: str, end: str) -> float | None:
    """The neutralized excess of one span, recomputed from the style sidecar.

    Cached because the console re-reads the same transitions on every list and
    detail request and a replay result never changes once written (a re-run
    writes a new result directory): without it one experiment's twelve
    transitions cost ~40 ms of sidecar parsing and regression per pass.
    """

    path = Path(result_ref or "")
    if not result_ref:
        return None
    if path.name == "result.json":
        path = path.parent
    try:
        analysis = json.loads((path / STYLE_ARTIFACT_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(analysis, Mapping):
        return None
    return window_neutralized_excess(analysis, start=start, end=end)


def _counts(rows: Sequence[Transition]) -> dict[str, int]:
    """How many transitions were counted, positive, and unmeasurable.

    A failed or absent result proved nothing and is simply not positive, as it
    always was. ``unmeasured`` is the narrower case this rule introduces: a
    result that exists but whose neutralized excess could not be established
    at all, which must fail the gate explicitly rather than be graded on the
    raw excess instead.
    """

    measured = [transition_neutralized_excess(row) for row in rows]
    return {
        "transitions": len(rows),
        "positive_excess": sum(1 for value in measured if value is not None and value > 0),
        "unmeasured": sum(
            1
            for row, value in zip(rows, measured, strict=True)
            if value is None
            and isinstance(row.result, Mapping)
            and row.result.get("status") != "failed"
        ),
    }


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
    trials = [value for value in map(finite_number, trial_sharpes) if value is not None]
    series = [value for value in map(finite_number, returns) if value is not None]
    block: dict[str, object] = {
        "deflated_sharpe_probability": None,
        "trials": len(trials),
        "sharpe_star": None,
        "trial_sharpe_std": None,
        "observed_sharpe": finite_number(observed_sharpe),
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


def candidate_deflated_sharpe(
    *,
    observed_sharpe: object,
    trial_sharpes: Sequence[object],
    result_ref: str,
) -> dict[str, object]:
    """Deflated-Sharpe block of one completed Validation out of ``trial_sharpes``.

    The single source for the Fold record's ``selection_statistics`` and for
    the provisional block a session sees on each candidate row: both read the
    candidate's daily returns from its own replay record and hand the same
    inputs to :func:`deflated_sharpe`. A record that cannot be read at all is
    reported as ``return_series_missing``, which is a different fact from a
    window that is genuinely too short.
    """

    series = validation_daily_returns(result_ref)
    block = deflated_sharpe(
        observed_sharpe=observed_sharpe,
        trial_sharpes=trial_sharpes,
        returns=series if series is not None else (),
    )
    if series is None and block["unavailable_reason"] == "return_series_too_short":
        block["unavailable_reason"] = "return_series_missing"
    return block


def validation_daily_returns(result_ref: str) -> list[float] | None:
    """Daily returns of one completed Validation, read from its own record.

    The equity curve lives in the replay's ``result.json``, never in the
    summary, and it is the only place the return series exists. ``None`` says
    the series could not be read at all -- an absent, unreadable or
    curve-less record.
    """

    path = Path(str(result_ref or ""))
    if path.is_dir():
        path = path / "result.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    curve = payload.get("equity_curve") if isinstance(payload, dict) else None
    if not isinstance(curve, list):
        return None
    return [
        value
        for _day, value in daily_returns_from_curve(
            [row for row in curve if isinstance(row, Mapping)]
        )
    ]


def finite_number(value: object) -> float | None:
    """``value`` as a finite float, or None when it is not a real number."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _count(value: object) -> int | None:
    """``value`` as a count, or None when the record does not carry one."""

    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def latest_deployment_record(
    records: list[dict[str, object]],
) -> dict[str, object] | None:
    """The latest ``deployment_adjustment`` row (append-only, latest wins)."""
    latest = None
    for record in records:
        if is_durable_success_record(record, record_types=("deployment_adjustment",)):
            latest = record
    return latest


def deployment_adjustment_due(records: list[dict[str, object]], *, start: str) -> bool:
    """Whether the post-Held-out deployment adjustment still has to run: the
    knob names a window, the experiment graduated, and no adjustment row is
    durable yet. A crashed attempt (``attempt_failed`` only) is still due."""
    if not start:
        return False
    verdict = experiment_verdict(records, strict=False)
    if verdict is None or verdict.get("status") != "graduated":
        return False
    return latest_deployment_record(records) is None


def paper_candidate(records: list[dict[str, object]]) -> dict[str, object] | None:
    """The one artifact Paper pins (docs/pipeline-design.md §3.4).

    None unless the experiment graduated. The adjusted artifact when the
    latest deployment adjustment recorded ``status="adjusted"`` and refit the
    very artifact the latest Held-out rows scored; otherwise the graduated
    artifact -- a failed or abstained adjustment, or one whose parent is not
    the current graduate (a rollback moved the frontier), leaves the graduate
    as the candidate. The single source for the terminal status, the console
    and the report.
    """
    verdict = experiment_verdict(records, strict=False)
    if verdict is None or verdict.get("status") != "graduated":
        return None
    heldout = latest_heldout_records(records)
    graduated_id = str(heldout[-1].get("strategy_artifact_id") or "")
    if not graduated_id:
        return None
    adjustment = latest_deployment_record(records)
    if (
        adjustment is not None
        and adjustment.get("status") == "adjusted"
        and str(adjustment.get("parent_strategy_artifact_id") or "") == graduated_id
        and adjustment.get("adjusted_strategy_artifact_id")
    ):
        return {
            "artifact_id": str(adjustment["adjusted_strategy_artifact_id"]),
            "output_path": str(adjustment.get("adjusted_strategy_artifact_path") or ""),
            "models_path": str(adjustment.get("adjusted_model_artifact_path") or "") or None,
            "source": "adjusted",
            "graduated_artifact_id": graduated_id,
        }
    frozen = next(
        (
            record
            for record in reversed(records)
            if is_durable_success_record(record, record_types=("fold", "meta_learning"))
            and str(record.get("frozen_strategy_artifact_id") or "") == graduated_id
        ),
        None,
    )
    output = str((frozen or {}).get("frozen_strategy_artifact_path") or "")
    return {
        "artifact_id": graduated_id,
        "output_path": output,
        # The store freezes ``output/`` and ``models/`` side by side.
        "models_path": str(Path(output).parent / "models") if output else None,
        "source": "graduated",
        "graduated_artifact_id": graduated_id,
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
