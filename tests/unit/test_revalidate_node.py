"""The node replay outside a session: what it refuses before replaying, and
where a reading goes. The replay itself is the evaluator's, tested with it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.experiments import revalidate_node as tool

STEP = "research__session_ref_a__run_ref_b__valid_012"


def _arm(root: Path, *, tamper: bool = False) -> Path:
    """A finished arm with one recorded Step and its revision."""

    arm = root / "arm"
    revision = arm / "artifacts" / "strategy" / "revisions" / "revision_1"
    files = {"output/main.py": "SEED_BASE = 1000\nMODE = 'x'\n", "output/lib.py": "def f(seed=1000):\n    return seed\n"}
    entries = []
    for path, text in files.items():
        (revision / path).parent.mkdir(parents=True, exist_ok=True)
        (revision / path).write_text(text, encoding="utf-8")
        entries.append({"path": path, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    (revision / "manifest.json").write_text(json.dumps({"fingerprint": "fp", "files": entries}), encoding="utf-8")
    if tamper:
        (revision / "output/lib.py").write_text("def f(seed=2000):\n    return seed\n", encoding="utf-8")
    ledger = arm / "ledgers" / "experiment_ledger.jsonl"
    ledger.parent.mkdir(parents=True)
    row = {"step_id": STEP, "revision_id": "revision_1", "fingerprint": "fp"}
    session = {"record_type": "research_session", "arm_end": {"status": "no_deliverable"}, "steps": [row]}
    ledger.write_text(json.dumps(session) + "\n", encoding="utf-8")
    (arm / "hitl").mkdir()
    (arm / "hitl" / "params.json").write_text(json.dumps({"experiment_id": "arm"}), encoding="utf-8")
    return arm


def test_a_reading_goes_below_runtime_revalidate_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert tool.REVALIDATE_ROOT == tool.REPO_ROOT / ".runtime" / "revalidate"
    monkeypatch.setattr(tool, "REVALIDATE_ROOT", tmp_path)
    assert tool.label_dir("arm", "valid_012", "exact") == tmp_path / "arm" / "valid_012" / "exact"
    (tmp_path / "arm" / "valid_012" / "exact").mkdir(parents=True)
    with pytest.raises(ValueError, match="already used"):
        tool.label_dir("arm", "valid_012", "exact")
    for label in ("", "../hitl", "a/b", ".hidden"):
        with pytest.raises(ValueError, match="label must match"):
            tool.label_dir("arm", "valid_012", label)


@pytest.mark.parametrize(
    "extra",
    [["--max-days", "0"], ["--start-day", "2024-07-01"], ["--slippage-bps", "-1"], ["--seed", "x"], ["--gpus", "1"]],
)
def test_bad_arguments_are_refused(extra: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        tool.main(["--experiment", "arm", "--node", "valid_012", "--label", "l", *extra])
    assert exit_info.value.code == 2


def test_a_refused_node_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    arm = _arm(tmp_path)
    before = sorted(path.relative_to(arm) for path in arm.rglob("*"))
    monkeypatch.setattr(tool, "EXPERIMENTS_ROOT", tmp_path)
    monkeypatch.setattr(tool, "REVALIDATE_ROOT", tmp_path / "revalidate")
    assert tool.main(["--experiment", "arm", "--node", "valid_099", "--label", "l"]) == 2
    assert not (tmp_path / "revalidate").exists()
    assert sorted(path.relative_to(arm) for path in arm.rglob("*")) == before


def test_the_step_is_named_by_id_or_handle_in_the_session_that_ended_research(tmp_path: Path) -> None:
    records = [json.loads(line) for line in (_arm(tmp_path) / "ledgers" / "experiment_ledger.jsonl").open()]
    assert tool.session_step(records, "valid_012")["step_id"] == STEP
    assert tool.session_step(records, STEP)["revision_id"] == "revision_1"
    with pytest.raises(ValueError, match="names 0 Steps"):
        tool.session_step(records, "valid_013")
    with pytest.raises(ValueError, match="no research session"):
        tool.session_step([{**records[0], "arm_end": None}], "valid_012")


def test_only_the_recorded_bytes_are_replayed(tmp_path: Path) -> None:
    arm = _arm(tmp_path)
    row = {"revision_id": "revision_1", "fingerprint": "fp"}
    revision = tool.copy_revision(arm, row, tmp_path / "copy")
    assert (revision.output_path / "main.py").read_text(encoding="utf-8").startswith("SEED_BASE = 1000")
    with pytest.raises(ValueError, match="not the bytes the ledger recorded"):
        tool.copy_revision(arm, {**row, "fingerprint": "other"}, tmp_path / "copy2")
    with pytest.raises(ValueError, match="differs from its manifest"):
        tool.copy_revision(_arm(tmp_path / "tampered", tamper=True), row, tmp_path / "copy3")


def test_a_seed_changes_the_one_seed_line_only(tmp_path: Path) -> None:
    revision = tool.copy_revision(_arm(tmp_path), {"revision_id": "revision_1", "fingerprint": "fp"}, tmp_path / "c")
    assert tool.set_seed(revision.output_path, 2000) == "main.py: SEED_BASE = 2000"
    assert (revision.output_path / "main.py").read_text(encoding="utf-8") == "SEED_BASE = 2000\nMODE = 'x'\n"
    assert (revision.output_path / "lib.py").read_text(encoding="utf-8") == "def f(seed=1000):\n    return seed\n"
    (revision.output_path / "extra.py").write_text("FIT_SEED = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="2 seed lines"):
        tool.set_seed(revision.output_path, 4000)


def test_a_variant_replays_its_own_files_and_records_the_diff(tmp_path: Path) -> None:
    revision = tool.copy_revision(_arm(tmp_path), {"revision_id": "revision_1", "fingerprint": "fp"}, tmp_path / "c")
    variant = tmp_path / "variant"
    variant.mkdir()
    (variant / "main.py").write_text("SEED_BASE = 1000\nMODE = 'y'\n", encoding="utf-8")
    (variant / "lib.py").write_text("def f(seed=1000):\n    return seed\n", encoding="utf-8")
    record = tool.use_variant(revision.output_path, variant)
    assert "-MODE = 'x'\n+MODE = 'y'" in record["diff"] and "lib.py" not in record["diff"]
    assert (revision.output_path / "main.py").read_text(encoding="utf-8").endswith("MODE = 'y'\n")
    with pytest.raises(ValueError, match="holds the recorded bytes"):
        tool.use_variant(revision.output_path, variant)
