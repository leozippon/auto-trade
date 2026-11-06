"""Archiving experiments out of the console's home list (``scripts/experiments/archive_experiment.py``).

An archived experiment keeps its directory and gains one marker file. The script
refuses an experiment that still holds something live, archives and restores
idempotently, and the experiment stays exactly what it was to everything but the
console's home list, the console's start of its worker and the PIT-view prune:
it stays an experiment of the consistency check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autotrade.environment.runtime import utc_now_iso
from autotrade.pipelines.ledger import ExperimentLedger, verdict_void
from autotrade.webui.registry import ARCHIVED_NAME, experiment_listing
from scripts.dev.check_verdicts import current_experiments
from tests.unit.paper_book_fixture import paper_root, write_book_record
from tests.unit.webui_research_experiment import build_experiment

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/experiments/archive_experiment.py"


def _archive(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--experiments-root",
            str(tmp_path / "experiments"),
            "--paper-state-root",
            str(paper_root(tmp_path)),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _listed(root: Path) -> list[str]:
    return sorted(row["experiment_id"] for row in experiment_listing(root)["experiments"])


def test_an_experiment_holding_anything_live_is_refused_and_nothing_changes(tmp_path: Path) -> None:
    """A live worker, a standing graduation, a Paper account or an unknown id
    refuses the whole command before any marker is written; a voided
    graduate is archived like any other ended experiment."""

    root = tmp_path / "experiments"
    build_experiment(root, "ended", "no_deliverable")
    build_experiment(root, "running", "research", alive=True)
    build_experiment(root, "graduate", "graduated")
    build_experiment(root, "traded", "no_deliverable")
    write_book_record(paper_root(tmp_path) / "traded_book", experiment_id="traded")
    for experiment, phrase in (
        ("running", "running has a live worker"),
        ("graduate", "graduate is a graduate; void the graduation first"),
        ("traded", "traded has Paper account traded_book"),
        ("nowhere", "nowhere: unknown experiment: nowhere"),
    ):
        refused = _archive(tmp_path, "ended", experiment, "--reason", "historical")
        assert refused.returncode == 2 and phrase in refused.stderr, refused.stderr
        assert "nothing was changed" in refused.stderr
        assert not any(root.glob(f"*/hitl/{ARCHIVED_NAME}"))
    assert _archive(tmp_path, "nowhere", "--restore").returncode == 2
    assert _archive(tmp_path, "ended", "--reason", " ").returncode == 2
    assert _archive(tmp_path, "ended").returncode == 2  # neither a reason nor --restore

    graduate = root / "graduate"
    ExperimentLedger(graduate / "ledgers/experiment_ledger.jsonl").append(
        verdict_void("graduate", voided_by="operator", reason="leak", evidence_ref="src")
    )
    voided = _archive(tmp_path, "graduate", "--reason", "withdrawn")
    assert voided.returncode == 0, voided.stderr
    assert (graduate / "hitl" / ARCHIVED_NAME).is_file()


def test_archive_and_restore_are_idempotent_and_move_only_the_home_list(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    for experiment in ("first", "second", "third"):
        build_experiment(root, experiment, "no_deliverable")
    marker = root / "first" / "hitl" / ARCHIVED_NAME

    archived = _archive(tmp_path, "first", "second", "--reason", "historical experiment")
    assert archived.returncode == 0, archived.stderr
    assert json.loads(archived.stdout) == {"action": "archive", "changed": ["first", "second"]}
    written = json.loads(marker.read_text(encoding="utf-8"))
    assert written["reason"] == "historical experiment" and written["archived_at"].endswith("+00:00")
    assert _listed(root) == ["third"]
    shelf = experiment_listing(root, archived=True)
    assert sorted(row["experiment_id"] for row in shelf["experiments"]) == ["first", "second"]
    assert shelf["archived"] == experiment_listing(root)["archived"] == 2

    # Archiving again keeps the first marker; restoring twice removes it once.
    again = _archive(tmp_path, "first", "--reason", "another reason")
    assert again.returncode == 0 and json.loads(again.stdout)["changed"] == []
    assert json.loads(marker.read_text(encoding="utf-8")) == written
    restored = _archive(tmp_path, "first", "--restore")
    assert restored.returncode == 0 and json.loads(restored.stdout)["changed"] == ["first"]
    assert not marker.exists() and (root / "first").is_dir()
    assert _archive(tmp_path, "first", "--restore").stdout.strip().endswith('"changed": []}')
    assert _listed(root) == ["first", "third"]
    assert experiment_listing(root)["archived"] == 1


def test_an_archived_experiment_stays_in_the_consistency_check(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    experiment = build_experiment(root, "current", "discarded")
    params = experiment / "hitl/params.json"
    params.write_text(
        json.dumps({**json.loads(params.read_text(encoding="utf-8")), "_created_at": utc_now_iso()}),
        encoding="utf-8",
    )
    archived = _archive(tmp_path, "current", "--reason", "historical")
    assert archived.returncode == 0, archived.stderr
    assert current_experiments(root) == [experiment]
