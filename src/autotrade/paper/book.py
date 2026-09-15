"""A Paper book's frozen identity: which artifact it trades, in what environment.

A book is created once from an experiment's Paper candidate and never changes
afterwards. Creation copies the frozen artifact into the book and resolves the
experiment's own parameters with the pipeline's resolver, so the book trades
the snapshot configuration, Broker profile, schedule and strategy sandbox the
research replays used; everything is then read from ``book.json``, never from
the experiment again, and the experiment may be archived. Creation also copies
the artifact's out-of-sample curve (the forward and Held-out replay its verdict
named) into ``source_history.json``, so the book keeps the history its own days
continue. A different capital, artifact or environment is a new book in a new
state root.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from autotrade.environment.broker import BrokerProfile
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.nl import NLConfig
from autotrade.environment.replay.curve import result_curve
from autotrade.environment.sandbox import SandboxConfig, SandboxLimits
from autotrade.environment.strategy import CN_TZ, StrategySchedule
from autotrade.environment.strategy_loader import validate_strategy_package
from autotrade.pipelines.ledger import ExperimentLedger, forward_record, paper_candidate
from autotrade.pipelines.worker import (
    _strategy_sandbox_from_spec,
    resolve_worker_options,
)

from .storage import read_json, write_json_atomic

BOOK_NAME = "book.json"
BOOK_SCHEMA_VERSION = 1
STRATEGY_COPY_NAME = "strategy"
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
    candidate_source: str
    note: str
    strategy_path: Path
    models_dir: Path | None
    schedule: StrategySchedule
    profile: BrokerProfile
    sandbox: SandboxConfig
    snapshot_config: SnapshotConfig
    # build_model_gateway keyword arguments of the experiment's NL role, or
    # None when the experiment had no LLM settings.
    nl_gateway: dict[str, object] | None
    nl_config: NLConfig
    nl_failure_policy: str
    max_intraday_row_group_rows: int
    raw_dir: Path
    fundamental_events_root: Path
    fundamental_events_status: Path


def create_book(
    state_root: str | Path,
    *,
    experiment_dir: str | Path,
    artifact_id: str,
    repo_root: str | Path,
    initial_cash: float | None = None,
    note: str = "",
) -> Book:
    root = Path(state_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Paper state root is not empty; a new book needs a new state root: {root}")
    experiment = Path(experiment_dir).resolve(strict=True)
    candidate = paper_candidate(
        ExperimentLedger(experiment / "ledgers" / "experiment_ledger.jsonl").read()
    )
    if candidate is None:
        raise ValueError(f"{experiment.name} has no Paper candidate: it did not graduate")
    if artifact_id != candidate["artifact_id"]:
        raise ValueError(
            f"{artifact_id} is not the Paper candidate of {experiment.name}; "
            f"choose {candidate['artifact_id']}"
        )
    source = experiment / "artifacts" / "strategy" / "frozen" / artifact_id
    if not (source / "output" / "main.py").is_file():
        raise FileNotFoundError(f"frozen artifact has no output/main.py: {source}")

    params = read_json(experiment / "hitl" / "params.json")
    if not params:
        raise ValueError(f"missing experiment params: {experiment / 'hitl' / 'params.json'}")
    if initial_cash is not None:
        params = {**params, "initial_cash": initial_cash}
    options = resolve_worker_options(params, experiment_dir=experiment, repo_root=repo_root, preflight=True)
    if options.data_backend != "pit":
        raise ValueError("Paper books trade on the PIT research release only")
    sandbox = _strategy_sandbox_from_spec(
        options.agent_sandbox, fit_timeout_seconds=options.rolling.strategy_fit_timeout_seconds
    )
    image = read_json(experiment / "hitl" / "sandbox_image.json").get("image_ref")
    if image:
        # The image the experiment's formal replays actually ran.
        sandbox = replace(sandbox, image=str(image))
    llm = options.llm
    nl_gateway = None
    if llm is not None:
        nl_model = llm.model_for("nl")
        nl_gateway = {
            "model": nl_model,
            "env_file": str(Path(repo_root).resolve() / llm.env_file) if not Path(llm.env_file).is_absolute() else str(llm.env_file),
            "timeout_seconds": llm.timeout_seconds,
            "max_retries": llm.max_retries,
            "retry_backoff_seconds": llm.retry_backoff_seconds,
            "max_tokens": llm.max_tokens_for("nl", model=nl_model),
            "temperature": llm.temperature,
            "thinking_enabled": llm.thinking_enabled,
            "reasoning_effort": llm.reasoning_effort_for("nl"),
        }

    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    copy = root / STRATEGY_COPY_NAME
    shutil.copytree(source / "output", copy / "output")
    models = None
    if (source / "models").is_dir():
        shutil.copytree(source / "models", copy / "models")
        models = copy / "models"
    fingerprint = _tree_fingerprint(source, ("output", "models"))
    if _tree_fingerprint(copy, ("output", "models")) != fingerprint:
        raise RuntimeError(f"artifact copy differs from its source: {source}")
    validate_strategy_package(copy / "output" / "main.py")
    record = {
        "schema_version": BOOK_SCHEMA_VERSION,
        "created_at": datetime.now(CN_TZ).isoformat(),
        "experiment_id": experiment.name,
        "artifact_id": artifact_id,
        "candidate_source": "graduated",
        "artifact_fingerprint": fingerprint,
        "note": note,
        "strategy_path": str((copy / "output" / "main.py").relative_to(root)),
        "models_dir": str(models.relative_to(root)) if models is not None else None,
        "schedule": options.rolling.schedule.to_record(),
        "profile": asdict(options.rolling.broker_profile),
        "sandbox": {
            "image": sandbox.image,
            "docker_executable": sandbox.docker_executable,
            "limits": asdict(sandbox.limits),
        },
        "snapshot_config": asdict(options.snapshot_config),
        "nl_gateway": nl_gateway,
        "nl_config": asdict(options.nl_config),
        "nl_failure_policy": options.rolling.nl_failure_policy,
        "max_intraday_row_group_rows": options.max_intraday_row_group_rows,
        "raw_dir": str(options.raw_dir),
        "fundamental_events_root": str(options.fundamental_events_root),
        "fundamental_events_status": str(options.fundamental_events_status),
    }
    write_json_atomic(root / BOOK_NAME, record)
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
    )
    return target


def copy_source_history(state_root: str | Path, experiment_dir: str | Path) -> Path:
    """The book's copy of its source experiment's out-of-sample curve.

    The forward record names the one replay carrying the arm's forward and
    Held-out slices — the curve the console draws for that verdict — so an arm
    without one has no history to copy and says so.
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
    result_file = (experiment / reference).resolve()
    if not result_file.is_relative_to(experiment):
        raise ValueError(f"{experiment.name}'s forward result is outside the experiment: {result_file}")
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
    record = read_json(root / BOOK_NAME)
    if not record:
        raise FileNotFoundError(f"no Paper book at {root}; create one with `run_paper.py init`")
    if record.get("schema_version") != BOOK_SCHEMA_VERSION:
        raise ValueError(f"unsupported Paper book schema in {root / BOOK_NAME}")
    sandbox = record["sandbox"]
    return Book(
        root=root,
        experiment_id=str(record["experiment_id"]),
        artifact_id=str(record["artifact_id"]),
        candidate_source=str(record["candidate_source"]),
        note=str(record.get("note") or ""),
        strategy_path=root / str(record["strategy_path"]),
        models_dir=root / str(record["models_dir"]) if record.get("models_dir") else None,
        schedule=StrategySchedule(**record["schedule"]),
        profile=BrokerProfile(**record["profile"]),
        sandbox=SandboxConfig(
            image=str(sandbox["image"]),
            docker_executable=str(sandbox["docker_executable"]),
            limits=SandboxLimits(**_tuples(sandbox["limits"])),
        ),
        snapshot_config=SnapshotConfig(**_tuples(record["snapshot_config"])),
        nl_gateway=dict(record["nl_gateway"]) if record.get("nl_gateway") else None,
        nl_config=NLConfig(**record["nl_config"]),
        nl_failure_policy=str(record["nl_failure_policy"]),
        max_intraday_row_group_rows=int(record["max_intraday_row_group_rows"]),
        raw_dir=Path(record["raw_dir"]),
        fundamental_events_root=Path(record["fundamental_events_root"]),
        fundamental_events_status=Path(record["fundamental_events_status"]),
    )


def _tuples(record: dict[str, object]) -> dict[str, object]:
    """JSON arrays back to the tuples the frozen dataclasses hold."""

    return {key: tuple(value) if isinstance(value, list) else value for key, value in record.items()}


def _tree_fingerprint(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in names:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            digest.update(json.dumps(str(path.relative_to(root))).encode())
            digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


__all__ = [
    "BOOK_NAME",
    "SOURCE_HISTORY_NAME",
    "Book",
    "copy_source_history",
    "create_book",
    "load_book",
    "write_source_history",
]
