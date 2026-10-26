#!/usr/bin/env python3
"""Replay one recorded Validation node of a finished arm again, outside its session.

An independent reading of a node: its recorded revision, checked file by file
against the revision manifest and the fingerprint the arm's ledger recorded,
is replayed by the evaluator a session's ``batch_validate`` calls, assembled
from the arm's own ``hitl/params.json`` exactly as the round smoke assembles a
scratch experiment (``_round._smoke_replays``): the arm's PIT view seed
hardlinked in, the arm's own sandbox image and pretrained-store pin, cards
taken by ``hitl_state.select_gpus``. A span is a research span (``full``,
``Yk`` or ``Yi..Yj``), so nothing past research end is ever replayed here; the
designed path to forward and Held-out is ``incubate.py``. What may be changed
is recorded: ``--variant`` replays a modified copy of the node's ``output/``
instead (an ablation; the reading keeps the diff against the recorded bytes),
``--seed`` gives the strategy's one seed line another integer (the one change
a seed replicate may make, ``experiment.seed_change``), ``--slippage-bps``
replaces the arm's slippage knob, and ``--start-day``/``--max-days`` truncate
the replay to a window as a smoke does (no zero-skill panel then).

Everything is written below ``.runtime/revalidate/<arm>/<node>/<label>/``:
``reading.json`` (what was replayed, then the result reference or the error),
the strategy copy that ran, a scratch experiment holding the pin, the PIT
views and the result, and ``active_daily.csv`` beside the result of a whole
span. The arm's directory is only read. A label is used once.

usage: revalidate_node.py --experiment <arm> --node valid_012 --label exact
       [--span full] [--variant DIR] [--seed N] [--slippage-bps X]
       [--start-day YYYYMMDD] [--max-days N]
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.identity import AgentRefStore
from autotrade.environment.runtime import utc_now_iso, write_json_atomic
from autotrade.environment.sandbox_images import (
    SANDBOX_IMAGE_STATE_NAME,
    prepare_experiment_sandbox_image,
)
from autotrade.environment.step_tree import node_handle
from autotrade.pipelines.config import ArtifactRevision, research_span
from autotrade.pipelines.experiment import _SEED_LINE, _SEED_NAME
from autotrade.pipelines.hitl_state import read_json, select_gpus
from autotrade.pipelines.ledger import ExperimentLedger
from autotrade.pipelines.session_resume import REVISIONS_DIR
from autotrade.pipelines.session_tools import write_active_daily
from autotrade.pipelines.worker import build_experiment_pipeline, resolve_worker_options

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
REVALIDATE_ROOT = REPO_ROOT / ".runtime" / "revalidate"
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
DAY = re.compile(r"\d{8}")


def session_step(records: list[dict], node: str) -> dict:
    """The row of ``node`` (its step id or short handle) in the arm's one
    research session that ended research."""

    ended = [
        record
        for record in records
        if record.get("record_type") == "research_session" and (record.get("frozen") or record.get("arm_end"))
    ]
    if len(ended) != 1:
        raise ValueError("the arm has no research session that ended its research")
    rows = [row for row in ended[0]["steps"] if node in (row["step_id"], node_handle(row["step_id"]))]
    if len(rows) != 1:
        raise ValueError(f"{node!r} names {len(rows)} Steps of the arm's research session, not one")
    return rows[0]


def label_dir(experiment_id: str, node: str, label: str) -> Path:
    """Where one reading goes; refused for an unsafe name or a label already used."""

    for value, what in ((experiment_id, "experiment id"), (node, "node"), (label, "label")):
        if not LABEL.fullmatch(value):
            raise ValueError(f"{what} must match {LABEL.pattern}: {value!r}")
    target = REVALIDATE_ROOT / experiment_id / node / label
    if target.exists():
        raise ValueError(f"label {label!r} is already used: {target}")
    return target


def copy_revision(experiment_dir: Path, row: dict, dest: Path) -> ArtifactRevision:
    """The recorded revision's files, copied to ``dest`` and checked against
    its manifest, whose fingerprint must be the one the ledger recorded."""

    source = experiment_dir / REVISIONS_DIR / str(row["revision_id"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["fingerprint"] != row["fingerprint"]:
        raise ValueError(f"revision {row['revision_id']} is not the bytes the ledger recorded for the Step")
    for entry in manifest["files"]:
        data = (source / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']} of revision {row['revision_id']} differs from its manifest")
        target = dest / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (dest / "models").mkdir(exist_ok=True)
    return ArtifactRevision(str(row["revision_id"]), dest / "output", dest / "models")


def use_variant(output: Path, variant: Path) -> dict[str, object]:
    """Replace the recorded ``output`` by the files of ``variant``; what was
    replayed instead: each file's SHA-256 and the unified diff against the
    recorded bytes, which must not be empty."""

    def files(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts
        }

    recorded, changed = files(output), files(variant)
    diff = "".join(
        line
        for name in sorted({*recorded, *changed})
        for line in difflib.unified_diff(
            recorded.get(name, b"").decode("utf-8").splitlines(keepends=True),
            changed.get(name, b"").decode("utf-8").splitlines(keepends=True),
            f"recorded/{name}",
            f"variant/{name}",
        )
    )
    if not diff:
        raise ValueError(f"variant {variant} holds the recorded bytes")
    shutil.rmtree(output)
    shutil.copytree(variant, output, ignore=shutil.ignore_patterns("__pycache__"))
    return {
        "source": str(variant),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in changed.items()},
        "diff": diff,
    }


def set_seed(output: Path, seed: int) -> str:
    """Give the strategy's one seed line ``seed``; ``"<path>: <line>"``."""

    found = []
    for path in sorted(output.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for index, line in enumerate(lines):
            match = _SEED_LINE.fullmatch(line.rstrip("\n"))
            if match and _SEED_NAME.fullmatch(match[1]):
                found.append((path, lines, index, match))
    if len(found) != 1:
        raise ValueError(f"the strategy has {len(found)} seed lines, not one, so a seed cannot be set")
    path, lines, index, match = found[0]
    lines[index] = lines[index][: match.start(2)] + str(seed) + lines[index][match.end(2) :]
    path.write_text("".join(lines), encoding="utf-8")
    return f"{path.relative_to(output)}: {lines[index].strip()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", required=True, help="Experiment id of a finished arm.")
    parser.add_argument("--node", required=True, help="Step id or short handle (valid_012).")
    parser.add_argument("--label", required=True, help="Name of this reading, used once.")
    parser.add_argument("--span", default="full", help="Research span: full, Yk or Yi..Yj.")
    parser.add_argument("--variant", type=Path, help="Modified copy of the node's output/ to replay.")
    parser.add_argument("--seed", type=int, help="New value of the strategy's one seed line.")
    parser.add_argument("--slippage-bps", type=float, help="Replaces the arm's slippage_bps.")
    parser.add_argument("--start-day", help="First trading day of a truncated window (YYYYMMDD).")
    parser.add_argument("--max-days", type=int, help="Trading days of a truncated window.")
    args = parser.parse_args(argv)
    if args.start_day is not None and not DAY.fullmatch(args.start_day):
        parser.error("--start-day must be YYYYMMDD")
    if args.max_days is not None and args.max_days <= 0:
        parser.error("--max-days must be positive")
    if args.slippage_bps is not None and not 0 <= args.slippage_bps < 1e4:
        parser.error("--slippage-bps must be in [0, 10000)")
    if args.variant is not None and not (args.variant / "main.py").is_file():
        parser.error("--variant must be a directory holding main.py")
    arm = EXPERIMENTS_ROOT / args.experiment
    try:
        params = read_json(arm / "hitl" / "params.json")
        if not params:
            raise ValueError(f"missing experiment params: {arm / 'hitl' / 'params.json'}")
        row = session_step(ExperimentLedger(arm / "ledgers" / "experiment_ledger.jsonl").read(), args.node)
        out = label_dir(args.experiment, node_handle(row["step_id"]), args.label)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    scratch = out / "scratch"
    experiment_dir = scratch / args.experiment
    (experiment_dir / "hitl").mkdir(parents=True)
    reading: dict[str, object] = {
        "experiment_id": args.experiment,
        "step_id": row["step_id"],
        "revision_id": row["revision_id"],
        "fingerprint": row["fingerprint"],
        "recorded_result_ref": row.get("validation_result_ref"),
        "label": args.label,
        "span": args.span,
        "start_day": args.start_day,
        "max_days": args.max_days,
        "variant": None,
        "seed_line": None,
        "slippage_bps": params.get("slippage_bps") if args.slippage_bps is None else args.slippage_bps,
        "started_at": utc_now_iso(),
        "status": "running",
    }
    write_json_atomic(out / "reading.json", reading)
    started = time.monotonic()
    try:
        revision = copy_revision(arm, row, out / "strategy")
        if args.variant is not None:
            reading["variant"] = use_variant(revision.output_path, args.variant.resolve())
        if args.seed is not None:
            reading["seed_line"] = set_seed(revision.output_path, args.seed)
        overrides = {
            "experiments_root": str(scratch),
            "work_root": str(out / "work"),
            **({} if args.slippage_bps is None else {"slippage_bps": args.slippage_bps}),
        }
        options = resolve_worker_options({**params, **overrides}, experiment_dir=experiment_dir, repo_root=REPO_ROOT)
        spec = options.agent_sandbox
        if spec is not None:
            if spec.gpu == "auto":
                devices = select_gpus(EXPERIMENTS_ROOT, spec.gpu_count, require_name=spec.gpu_name_filter)
                spec = replace(spec, gpu=tuple(devices))
            # The arm's own image tag, as its worker and incubate.py resolve it.
            shutil.copyfile(arm / "hitl" / SANDBOX_IMAGE_STATE_NAME, experiment_dir / "hitl" / SANDBOX_IMAGE_STATE_NAME)
            spec = prepare_experiment_sandbox_image(spec, experiment_id=args.experiment, experiment_dir=experiment_dir)
            options = replace(options, agent_sandbox=spec)
            reading["gpu_devices"] = list(spec.gpu) if isinstance(spec.gpu, tuple) else spec.gpu
            reading["sandbox_image"] = spec.image
        write_json_atomic(out / "reading.json", reading)
        pipeline = build_experiment_pipeline(
            options,
            ledger=ExperimentLedger(experiment_dir / "ledgers" / "experiment_ledger.jsonl"),
            store=FilesystemArtifactStore(experiment_dir / "artifacts" / "strategy"),
            ref_store=AgentRefStore(experiment_dir),
        ).pipeline
        _, years = pipeline.research_inputs()
        request = research_span(years, args.span).request(
            revision, schedule=options.rolling.schedule, broker_profile=options.rolling.broker_profile
        )
        result = pipeline.evaluator.evaluate(request, max_days=args.max_days, start_day=args.start_day)
        if args.max_days is None and args.start_day is None:
            reading["active_daily"] = str(write_active_daily(result.result_ref))
        reading.update(status="ok", result_ref=result.result_ref)
    except BaseException as exc:
        reading.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        reading.update(finished_at=utc_now_iso(), wall_seconds=round(time.monotonic() - started, 1))
        write_json_atomic(out / "reading.json", reading)
    stats = result.summary
    benchmark = stats.get("benchmark") or {}
    print(
        json.dumps(
            {
                "reading": str(out / "reading.json"),
                "total_return": stats.get("total_return"),
                "max_drawdown": stats.get("max_drawdown"),
                "active_information_ratio": benchmark.get("active_information_ratio"),
                "active_neutralized_excess": benchmark.get("active_neutralized_excess"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
