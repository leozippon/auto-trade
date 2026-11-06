"""A Paper account's frozen identity: which artifact it trades, in what environment.

A book is created once from an experiment's Paper candidate and never changes
afterwards. Creation copies the registered artifact into the book and resolves the
experiment's own parameters with the pipeline's resolver, so the book trades
the snapshot configuration, Broker profile, schedule and strategy sandbox the
research replays used, but for the decision clock: every Paper decision runs
under ``PAPER_DECISION_TIMEOUT_SECONDS``. Everything is then read from
``book.json``, never from the experiment again, and the experiment may be
archived. Creation also copies the artifact's out-of-sample curve (the forward
and Held-out replay its verdict named) into ``source_history.json``, so the
book keeps the history its own days continue. A different capital, artifact or
environment is a new book in a new state root.

A book trades the registered strategy of an experiment with its forward record (the
experiment's Paper candidate), recorded as its ``candidate_source``, and is opened by
the operator from the console's ranking view; at most ``BOOK_CAP`` books are open at
once. Creation pins the rules its Paper verdict is read by (``paper/verdict.py``),
so a later change cannot move a book already trading, and the boards its account
may buy on, in its Broker profile: those of the experiment's stamp (every board
when it has none) that the book's own initial cash qualifies for
(``broker.default_permitted_boards``), so a Paper account opened with less capital
than its experiment does not keep the STAR Market or the Beijing exchange. The
book's Broker and its verdict's random baseline both read that one field.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from datetime import datetime, time
from pathlib import Path

from autotrade.common.records import read_json_object, write_json_atomic
from autotrade.common.time import CN_TZ
from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.broker import BrokerProfile, default_permitted_boards
from autotrade.environment.broker_core import BOARDS
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.replay.curve import result_curve
from autotrade.environment.sandbox import SandboxConfig
from autotrade.environment.strategy import StrategySchedule
from autotrade.environment.strategy_loader import validate_strategy_package
from autotrade.pipelines.experiment import null_control_seed
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    forward_record,
    paper_candidate,
    recorded_path,
)
from autotrade.pipelines.worker import (
    resolve_worker_options,
    strategy_boundary,
    strategy_sandbox_from_spec,
)

from .pit import require_audited_fundamentals
from .storage import read_jsonl

BOOK_NAME = "book.json"
# 3: the pinned verdict rules, and the snapshot configuration and the strategy
# sandbox as their own records (``SnapshotConfig.to_record``,
# ``SandboxConfig.to_record``). A book of another schema is refused.
BOOK_SCHEMA_VERSION = 3
STRATEGY_COPY_NAME = "strategy"
# What every book trades, as its ``candidate_source`` records it.
CANDIDATE_SOURCE = "frozen"
# Books open at once; a killed book does not count.
BOOK_CAP = 4
# The capital, in CNY, a new book starts with unless its creator names another,
# whatever the experiment's own initial_cash (the owner's decision of 2026-11-06); a
# book keeps the capital it was created with.
PAPER_INITIAL_CASH = 200_000.0
# When every book's order sheet is due, Beijing time: five minutes before the
# 09:15 opening call auction, whatever the book's decision time (the owner's
# decision of 2026-11-06). ops/cron/paper.cron starts its last retry at least
# the container-start clock plus PAPER_DECISION_TIMEOUT_SECONDS before it.
PAPER_SHEET_DUE = time(9, 10)
# The wall clock of one decision call in every Paper run -- the day's decision
# and the replay of each earlier one -- in place of the experiment's
# strategy_inference_timeout_seconds, which is sized for research replays.
PAPER_DECISION_TIMEOUT_SECONDS = 1800.0
# The Paper verdict's rules, pinned into every new book. A statistical check
# runs only every CHECKPOINT_DAYS settled Paper sessions (half a year): kill
# when the KILL_CONFIDENCE upper bound of the plain active mean is below zero
# (t < -1), confirm when its CONFIRM_CONFIDENCE lower bound is above zero
# (t > 2).
CHECKPOINT_DAYS = 126
KILL_CONFIDENCE = 0.841
CONFIRM_CONFIDENCE = 0.977
# The book's terminal verdict transitions, appended by ``paper/verdict.py``.
VERDICT_LOG_NAME = "verdict.jsonl"
# The out-of-sample history the book continues: the forward and Held-out
# replay of the very artifact it trades, copied out of the experiment at
# creation so the book keeps it after the experiment is archived.
SOURCE_HISTORY_NAME = "source_history.json"
SOURCE_HISTORY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Book:
    root: Path
    experiment_id: str
    artifact_id: str
    # ``CANDIDATE_SOURCE``.
    candidate_source: str
    note: str
    strategy_path: Path
    models_dir: Path | None
    schedule: StrategySchedule
    profile: BrokerProfile
    sandbox: SandboxConfig
    snapshot_config: SnapshotConfig
    max_intraday_row_group_rows: int
    raw_dir: Path
    fundamental_events_root: Path
    fundamental_events_status: Path
    benchmark_index: str
    # Pinned at creation: checkpoint_days, kill_confidence,
    # confirm_confidence, max_drawdown, active_max_drawdown, baseline_seed.
    verdict_rules: dict[str, object]
    artifact_fingerprint: str = ""


def create_book(
    state_root: str | Path,
    *,
    experiment_dir: str | Path,
    artifact_id: str,
    repo_root: str | Path,
    initial_cash: float = PAPER_INITIAL_CASH,
    note: str = "",
) -> Book:
    """Create the book at ``state_root`` (``<Paper state root>/<book id>``).

    The book trades the experiment's Paper candidate, copies the replay its forward
    record names and buys on the experiment's stamped boards (every board when the
    experiment has none). It starts with ``initial_cash``, not the experiment's own,
    and keeps only the boards that capital qualifies for. A book is refused while
    ``BOOK_CAP`` other books beside it are not killed, and, before anything is
    written, when the experiment reads fundamentals and the PIT fundamental events
    audit does not cover the book's first decision view
    (``pit.require_audited_fundamentals``).
    """

    root = Path(state_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Paper state root is not empty; a new book needs a new state root: {root}")
    experiment = Path(experiment_dir).resolve(strict=True)
    candidate = paper_candidate(
        ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    )
    if candidate is None:
        raise ValueError(f"{experiment.name} has no Paper candidate: no registered strategy with a forward record")
    if artifact_id != candidate["artifact_id"]:
        raise ValueError(
            f"{artifact_id} is not the Paper candidate of {experiment.name}; "
            f"choose {candidate['artifact_id']}"
        )
    require_book_place(root.parent, root.name)
    source = experiment / "artifacts" / "strategy" / "frozen" / artifact_id
    if not (source / "output" / "main.py").is_file():
        raise FileNotFoundError(f"registered artifact has no output/main.py: {source}")

    params = read_json_object(experiment / "hitl" / "params.json", missing_ok=True)
    if not params:
        raise ValueError(f"missing experiment params: {experiment / 'hitl' / 'params.json'}")
    params = {**params, "initial_cash": initial_cash}
    options = resolve_worker_options(params, experiment_dir=experiment, repo_root=repo_root, preflight=True)
    if options.data_backend != "pit":
        raise ValueError("Paper accounts trade on the PIT research release only")
    require_audited_fundamentals(
        options.snapshot_config,
        raw_dir=options.raw_dir,
        fundamental_events_root=options.fundamental_events_root,
        fundamental_events_status=options.fundamental_events_status,
        today=datetime.now(CN_TZ).strftime("%Y%m%d"),
    )
    profile = options.rolling.broker_profile
    permitted_boards = profile.permitted_boards or BOARDS
    eligible = default_permitted_boards(profile.initial_cash)
    profile = replace(profile, permitted_boards=tuple(board for board in permitted_boards if board in eligible))
    sandbox = strategy_sandbox_from_spec(
        options.agent_sandbox, **strategy_boundary(options.rolling)
    )
    image = read_json_object(experiment / "hitl" / "sandbox_image.json", missing_ok=True).get("image_ref")
    if image:
        # The image the experiment's formal replays actually ran.
        sandbox = replace(sandbox, spec=replace(sandbox.spec, image=str(image)))

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    copy = root / STRATEGY_COPY_NAME
    shutil.copytree(source / "output", copy / "output")
    models = None
    if (source / "models").is_dir():
        shutil.copytree(source / "models", copy / "models")
        models = copy / "models"
    fingerprint = artifact_fingerprint(copy / "output", models)
    validate_strategy_package(copy / "output" / "main.py")
    record = {
        "schema_version": BOOK_SCHEMA_VERSION,
        "created_at": datetime.now(CN_TZ).isoformat(),
        "experiment_id": experiment.name,
        "artifact_id": artifact_id,
        "candidate_source": CANDIDATE_SOURCE,
        "artifact_fingerprint": fingerprint,
        "note": note,
        "strategy_path": str((copy / "output" / "main.py").relative_to(root)),
        "models_dir": str(models.relative_to(root)) if models is not None else None,
        "schedule": options.rolling.schedule.to_record(),
        # The index the source experiment was graded against, so the book's own page
        # continues the copied out-of-sample curve against the same benchmark
        # instead of splicing two different indexes into one line.
        "benchmark_index": options.rolling.benchmark_index,
        "profile": profile.to_record(),
        # Without a decision clock: every Paper decision runs under
        # PAPER_DECISION_TIMEOUT_SECONDS (load_book).
        "sandbox": {key: value for key, value in sandbox.to_record().items() if key != "timeout_seconds"},
        "snapshot_config": options.snapshot_config.to_record(),
        "max_intraday_row_group_rows": options.max_intraday_row_group_rows,
        "raw_dir": str(options.raw_dir),
        "fundamental_events_root": str(options.fundamental_events_root),
        "fundamental_events_status": str(options.fundamental_events_status),
        "verdict_rules": {
            "checkpoint_days": CHECKPOINT_DAYS,
            "kill_confidence": KILL_CONFIDENCE,
            "confirm_confidence": CONFIRM_CONFIDENCE,
            # The experiment's own drawdown limits, which its verdict and gate used.
            "max_drawdown": options.rolling.acceptance.max_drawdown,
            "active_max_drawdown": options.rolling.acceptance.active_max_drawdown,
            "baseline_seed": null_control_seed(artifact_id, "paper"),
        },
    }
    write_json_atomic(root / BOOK_NAME, record, private=True, sort_keys=False)
    copy_source_history(root, experiment)
    return load_book(root)


def write_source_history(
    state_root: str | Path,
    *,
    experiment_id: str,
    result_file: str | Path,
    heldout_start: str | None = None,
) -> Path:
    """Copy one out-of-sample replay curve into the book.

    The curve is projected exactly as the console projects a result of that
    experiment, so the book's copy and the console's read the same numbers.
    Copying frees the book from ``experiments/``: its page keeps drawing the
    artifact's history after the experiment is archived.
    """

    root = Path(state_root).resolve(strict=True)
    path = Path(result_file).resolve(strict=True)
    if path.is_dir():
        path = path / "result.json"
    if not path.is_file():
        raise FileNotFoundError(f"replay result is missing: {path}")
    curve = result_curve(path)
    if not curve["series"]:
        raise ValueError(f"replay result has no daily returns: {path}")
    target = root / SOURCE_HISTORY_NAME
    write_json_atomic(
        target,
        {
            "schema_version": SOURCE_HISTORY_SCHEMA_VERSION,
            "copied_at": datetime.now(CN_TZ).isoformat(),
            "experiment_id": experiment_id,
            "result": path.parent.name,
            # The session the Held-out slice opens on, drawn as a divider; a
            # replay that is Held-out end to end has none.
            "heldout_start": heldout_start,
            "series": curve["series"],
            "benchmark": curve["benchmark"],
        },
        private=True,
        sort_keys=False,
    )
    return target


def copy_source_history(state_root: str | Path, experiment_dir: str | Path) -> Path:
    """The book's copy of its source experiment's out-of-sample curve.

    The forward record names the one replay carrying the experiment's forward and
    Held-out slices — the curve the console draws beside its readout — so an
    experiment without one has no history to copy and says so. The recorded path is
    read under the experiment's current directory (``ledger.recorded_path``), so an
    experiment moved since its replay copies too.
    """

    experiment = Path(experiment_dir).resolve(strict=True)
    record = forward_record(ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read())
    if record is None:
        raise ValueError(
            f"{experiment.name} has no forward verdict record: there is no out-of-sample history to copy"
        )
    reference = record.get("result_ref")
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{experiment.name}'s forward record names no replay result")
    result_file = recorded_path(experiment, reference)
    replay = record.get("replay") if isinstance(record.get("replay"), dict) else {}
    heldout_start = replay.get("heldout_start")
    return write_source_history(
        state_root,
        experiment_id=experiment.name,
        result_file=result_file,
        heldout_start=str(heldout_start) if heldout_start else None,
    )


def load_book(state_root: str | Path) -> Book:
    root = Path(state_root).resolve()
    record = read_json_object(root / BOOK_NAME, missing_ok=True)
    if not record:
        raise FileNotFoundError(f"no Paper account at {root}; create one with `run_paper.py init`")
    if record.get("schema_version") != BOOK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported Paper account schema {record.get('schema_version')!r} in {root / BOOK_NAME} "
            f"(this code reads {BOOK_SCHEMA_VERSION}); delete it and create it again"
        )
    return Book(
        root=root,
        experiment_id=str(record["experiment_id"]),
        artifact_id=str(record["artifact_id"]),
        candidate_source=str(record["candidate_source"]),
        note=str(record.get("note") or ""),
        strategy_path=root / str(record["strategy_path"]),
        models_dir=root / str(record["models_dir"]) if record.get("models_dir") else None,
        schedule=StrategySchedule(**record["schedule"]),
        profile=BrokerProfile.from_record(record["profile"]),
        # Every Paper decision runs under PAPER_DECISION_TIMEOUT_SECONDS.
        sandbox=SandboxConfig.from_record(
            {**record["sandbox"], "timeout_seconds": PAPER_DECISION_TIMEOUT_SECONDS}
        ),
        snapshot_config=SnapshotConfig.from_record(record["snapshot_config"]),
        max_intraday_row_group_rows=int(record["max_intraday_row_group_rows"]),
        raw_dir=Path(record["raw_dir"]),
        fundamental_events_root=Path(record["fundamental_events_root"]),
        fundamental_events_status=Path(record["fundamental_events_status"]),
        benchmark_index=str(record["benchmark_index"]),
        verdict_rules=dict(record["verdict_rules"]),
        artifact_fingerprint=str(record.get("artifact_fingerprint") or ""),
    )


def verify_book_artifact(book: Book) -> None:
    """Verify the complete pinned package before execution, under writer_lock.

    Never refresh the pin: differing bytes are a failed run, not a revision.
    """
    actual = artifact_fingerprint(book.strategy_path.parent, book.models_dir)
    if not book.artifact_fingerprint or actual != book.artifact_fingerprint:
        raise ValueError("Paper artifact fingerprint mismatch; refusing to execute the strategy or models")


def book_status(state_root: str | Path) -> str:
    """``observing`` until the book's one terminal transition, then its status
    (``confirmed`` or ``killed``) for good."""

    path = Path(state_root) / VERDICT_LOG_NAME
    rows, skipped = read_jsonl(path)
    if skipped or len(rows) > 1:
        raise ValueError(f"{path} must hold at most one whole transition")
    return str(rows[0]["status"]) if rows else "observing"


def open_capped_books(paper_root: str | Path) -> list[str]:
    """The ids of the books under ``paper_root`` that are not killed. A hidden
    directory is a book being deleted (``books.delete_book``)."""

    root = Path(paper_root)
    books = sorted(path.parent for path in root.glob(f"*/{BOOK_NAME}")) if root.is_dir() else []
    return [
        book.name
        for book in books
        if not book.name.startswith(".") and book_status(book) != "killed"
    ]


def require_book_place(paper_root: str | Path, book_id: str) -> None:
    """Refuse a new book ``book_id`` while ``BOOK_CAP`` other books under
    ``paper_root`` are not killed; the book's own id never counts against itself."""

    others = [book for book in open_capped_books(paper_root) if book != book_id]
    if len(others) >= BOOK_CAP:
        raise ValueError(
            f"{len(others)} Paper accounts are open ({', '.join(others)}), "
            f"the cap is {BOOK_CAP}; a book frees its place when it is killed"
        )


__all__ = [
    "BOOK_CAP",
    "BOOK_NAME",
    "CANDIDATE_SOURCE",
    "PAPER_DECISION_TIMEOUT_SECONDS",
    "PAPER_INITIAL_CASH",
    "PAPER_SHEET_DUE",
    "SOURCE_HISTORY_NAME",
    "VERDICT_LOG_NAME",
    "Book",
    "book_status",
    "copy_source_history",
    "create_book",
    "load_book",
    "open_capped_books",
    "require_book_place",
    "verify_book_artifact",
    "write_source_history",
]
