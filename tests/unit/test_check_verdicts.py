"""The consistency check (``scripts/dev/check_verdicts.py``) on one synthetic
arm the real worker ran to its verdict.

The run over the real arms reads ``experiments/`` and is made by hand (the
script's docstring says how); here it reads an arm small enough for the unit
suite: every reading the arm stores reproduces, a tampered one is named and
fails the run, and an arm created under rules since retired is not judged.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

from autotrade.environment.runtime import utc_now_iso
from autotrade.pipelines import worker
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker
from scripts.dev.check_verdicts import (
    CURRENT_RULES_SINCE,
    check_arm,
    current_arms,
    main,
)
from tests.unit.synthetic_arm import SyntheticPITProvider, make_arm

LEDGER = Path("ledgers") / "experiment_ledger.jsonl"


@pytest.fixture(scope="module")
def arm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An arm created now, with a recorded freeze gate and forward verdict."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worker, "ResearchPITSnapshotProvider", SyntheticPITProvider)
        repo, experiment = make_arm(tmp_path_factory.mktemp("judged"), _created_at=utc_now_iso())
        result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert result["verdict"]["status"] in ("graduated", "discarded")
    return experiment


def _copy(arm: Path, root: Path, edit=None, **params: object) -> Path:
    """The arm copied under ``root``, its ledger lines passed through
    ``edit`` and its parameters updated with ``params``. The records still
    name the paths the worker wrote at, and are read under the copy."""

    target = root / arm.name
    shutil.copytree(arm, target)
    path = target / "hitl" / "params.json"
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), **params}), encoding="utf-8")
    records = [json.loads(line) for line in (target / LEDGER).read_text(encoding="utf-8").splitlines()]
    if edit is not None:
        edit(records)
    (target / LEDGER).write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    return target


def _record(records: list[dict], kind: str) -> dict:
    return next(row for row in records if row["record_type"] == kind)


def test_every_reading_of_a_current_arm_reproduces(arm: Path) -> None:
    judgements = check_arm(arm)
    assert [item.reading for item in judgements] == ["freeze_gate", "forward"]
    assert [item.differences for item in judgements] == [[], []]
    assert current_arms(arm.parent) == [arm]
    assert main(["--experiments", str(arm.parent)]) == 0


def test_a_moved_arm_is_judged_from_where_it_now_lives(arm: Path, tmp_path: Path) -> None:
    """An arm whose records name a root it no longer lives at is judged from
    its own files; a record naming a place outside the arm fails the check
    instead of being read from there."""

    gone = tmp_path / "gone" / "experiments" / arm.name

    def moved(records: list[dict]) -> None:
        records[:] = json.loads(json.dumps(records).replace(str(arm), str(gone)))

    copy = _copy(arm, tmp_path / "experiments", moved)
    records = [json.loads(line) for line in (copy / LEDGER).read_text(encoding="utf-8").splitlines()]
    assert _record(records, "forward")["result_ref"].startswith(str(gone))
    judgements = check_arm(copy)
    assert [(item.reading, item.differences) for item in judgements] == [("freeze_gate", []), ("forward", [])]

    def elsewhere(records: list[dict]) -> None:
        moved(records)
        forward = _record(records, "forward")
        forward["result_ref"] = forward["result_ref"].replace(f"/{arm.name}/", "/another_arm/")

    with pytest.raises(FileNotFoundError, match="outside the experiment"):
        check_arm(_copy(arm, tmp_path / "elsewhere" / "experiments", elsewhere))


def test_a_tampered_reading_is_named_and_fails_the_run(arm: Path, tmp_path: Path) -> None:
    """One bit of a stored probability and a recorded status are enough. The
    switches a reading written before their retirement states in its
    thresholds are the one thing dropped before comparing."""

    def tampered(records: list[dict]) -> None:
        gate = _record(records, "research_session")["freeze_gate"]
        dsr = gate["deflated_sharpe"]
        dsr["deflated_sharpe_probability"] = math.nextafter(dsr["deflated_sharpe_probability"], 0.0)
        gate["thresholds"]["require_seed_replicates"] = True
        forward = _record(records, "forward")
        forward["verdict"]["status"] = "graduated" if forward["verdict"]["status"] != "graduated" else "discarded"
        forward["verdict"]["thresholds"]["require_forward_plain_selection"] = True

    copy = _copy(arm, tmp_path / "experiments", tampered)
    judgements = {item.reading: item.differences for item in check_arm(copy)}
    assert judgements == {
        "freeze_gate": ["deflated_sharpe.deflated_sharpe_probability"],
        "forward": ["verdict.status"],
    }
    assert main(["--experiments", str(copy.parent)]) == 1


def test_an_arm_created_under_retired_rules_is_not_judged(arm: Path, tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    earlier = CURRENT_RULES_SINCE.replace(year=CURRENT_RULES_SINCE.year - 1).isoformat()
    _copy(arm, root, _created_at=earlier)
    assert current_arms(root) == []
    assert main(["--experiments", str(root)]) == 0


def test_a_terminal_offline_delta_is_rejudged_from_the_record(arm: Path, tmp_path: Path) -> None:
    """The re-judge reads the declared terminal delta off the research-session
    record, so a gate stored without it no longer reproduces once the record
    declares screens."""

    def declared(records: list[dict]) -> None:
        _record(records, "research_session")["terminal_offline_trials"] = 5

    copy = _copy(arm, tmp_path / "experiments", declared)
    judgements = {item.reading: item.differences for item in check_arm(copy)}
    assert judgements["freeze_gate"]
    assert any("trials" in path for path in judgements["freeze_gate"])
    assert main(["--experiments", str(copy.parent)]) == 1
