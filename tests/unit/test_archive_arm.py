"""Archiving arms out of the console's home list (``scripts/experiments/archive_arm.py``).

An archived arm keeps its directory and gains one marker file. The script
refuses an arm that still holds something live, archives and restores
idempotently, and the arm stays exactly what it was to everything but the
console's home list: a lineage arm and an arm of the consistency check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from autotrade.environment.runtime import utc_now_iso
from autotrade.pipelines.ledger import ExperimentLedger, verdict_void
from autotrade.pipelines.lineage import extract_lineage
from autotrade.webui.registry import ARCHIVED_NAME, experiment_listing, list_experiments
from scripts.dev.check_verdicts import current_arms
from tests.unit.paper_book_fixture import paper_root, write_book_record
from tests.unit.test_lineage import RESEARCH_END, RESEARCH_START, _arm
from tests.unit.webui_research_arm import build_arm

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/experiments/archive_arm.py"


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
    return sorted(row["experiment_id"] for row in list_experiments(root))


def test_an_arm_holding_anything_live_is_refused_and_nothing_changes(tmp_path: Path) -> None:
    """A live worker, a standing graduation, a Paper book or an unknown id
    refuses the whole command before any marker is written; a voided
    graduate is archived like any other ended arm."""

    root = tmp_path / "experiments"
    build_arm(root, "ended", "no_deliverable")
    build_arm(root, "running", "research", alive=True)
    build_arm(root, "graduate", "graduated")
    build_arm(root, "traded", "no_deliverable")
    write_book_record(paper_root(tmp_path) / "traded_book", experiment_id="traded")
    for arm, phrase in (
        ("running", "running has a live worker"),
        ("graduate", "graduate is a graduate; void the graduation first"),
        ("traded", "traded has Paper book traded_book"),
        ("nowhere", "nowhere: unknown experiment: nowhere"),
    ):
        refused = _archive(tmp_path, "ended", arm, "--reason", "historical")
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
    for arm in ("first", "second", "third"):
        build_arm(root, arm, "no_deliverable")
    marker = root / "first" / "hitl" / ARCHIVED_NAME

    archived = _archive(tmp_path, "first", "second", "--reason", "historical arm")
    assert archived.returncode == 0, archived.stderr
    assert json.loads(archived.stdout) == {"action": "archive", "changed": ["first", "second"]}
    written = json.loads(marker.read_text(encoding="utf-8"))
    assert written["reason"] == "historical arm" and written["archived_at"].endswith("+00:00")
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


def test_an_archived_arm_stays_a_lineage_arm_and_in_the_consistency_check(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    ancestor = _arm(root, "ancestor", [{"seed": 2, "loading": 0.6}, {"seed": 3, "loading": 0.4}])
    params = ancestor / "hitl/params.json"
    params.write_text(
        json.dumps({**json.loads(params.read_text(encoding="utf-8")), "_created_at": utc_now_iso()}),
        encoding="utf-8",
    )
    period = {"research_start": RESEARCH_START, "research_end": RESEARCH_END}
    before = extract_lineage(root, ["ancestor"], **period)
    archived = _archive(tmp_path, "ancestor", "--reason", "historical")
    assert archived.returncode == 0, archived.stderr
    assert extract_lineage(root, ["ancestor"], **period) == before
    assert current_arms(root) == [ancestor]
