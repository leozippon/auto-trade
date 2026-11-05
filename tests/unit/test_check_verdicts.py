"""The consistency check (``scripts/dev/check_verdicts.py``) on one synthetic
arm the real worker ran to its readout.

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
from autotrade.pipelines.experiment import _keep_frozen_artifact_ids
from autotrade.pipelines.ledger import RETIRED_ABLATION_KEY
from autotrade.pipelines.worker import load_worker_options, run_local_interactive_worker
from scripts.dev.check_verdicts import (
    check_arm,
    current_arms,
    main,
)
from tests.unit.synthetic_arm import SyntheticPITProvider, make_arm

LEDGER = Path("ledgers") / "experiment_ledger.jsonl"


@pytest.fixture(scope="module")
def arm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An arm created now, with a recorded freeze gate, rolling record and readout."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(worker, "ResearchPITSnapshotProvider", SyntheticPITProvider)
        repo, experiment = make_arm(tmp_path_factory.mktemp("judged"), _created_at=utc_now_iso())
        result = run_local_interactive_worker(load_worker_options(experiment, repo_root=repo))
    assert result["verdict"] == {"status": "frozen_with_record", "reasons": []}
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
    assert [item.reading for item in judgements] == ["freeze_registration", "forward", "readout"]
    assert [item.differences for item in judgements] == [[], [], []]
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
    assert [(item.reading, item.differences) for item in judgements] == [
        ("freeze_registration", []),
        ("forward", []),
        ("readout", []),
    ]

    def elsewhere(records: list[dict]) -> None:
        moved(records)
        forward = _record(records, "forward")
        forward["result_ref"] = forward["result_ref"].replace(f"/{arm.name}/", "/another_arm/")

    with pytest.raises(FileNotFoundError, match="outside the experiment"):
        check_arm(_copy(arm, tmp_path / "elsewhere" / "experiments", elsewhere))


def test_a_tampered_reading_is_named_and_fails_the_run(arm: Path, tmp_path: Path) -> None:
    """One bit of a stored slice, of the readout's record and of a
    registration's verdict are enough."""

    def tampered(records: list[dict]) -> None:
        _record(records, "research_session")["freeze_gate"]["passed"] = False
        forward = _record(records, "forward")
        block = forward["slices"]["forward"]
        block["lower_bound"] = math.nextafter(block["lower_bound"], 0.0)
        record = forward["readout"]["record"]
        record["residual_information_ratio"] = math.nextafter(record["residual_information_ratio"], 0.0)

    copy = _copy(arm, tmp_path / "experiments", tampered)
    judgements = {item.reading: item.differences for item in check_arm(copy)}
    assert judgements == {
        "freeze_registration": ["passed"],
        "forward": ["forward.lower_bound"],
        "readout": ["record.residual_information_ratio"],
    }
    assert main(["--experiments", str(copy.parent)]) == 1


def test_an_arm_created_under_retired_rules_is_not_judged(arm: Path, tmp_path: Path) -> None:
    """An arm is judged by the rules it states, whenever it was created: one
    whose rules name no research years is never judged again."""

    root = tmp_path / "experiments"
    _copy(arm, root, research_years=None)
    assert current_arms(root) == []
    assert main(["--experiments", str(root)]) == 0


def test_an_arm_frozen_with_the_retired_ablation_is_not_judged_and_keeps_it(arm: Path, tmp_path: Path) -> None:
    """An arm frozen while the ablation existed states the sealed gate's
    rules, not ``research_years``, so it is never judged again; the ablation
    it froze beside its nominee (``ledger.RETIRED_ABLATION_KEY``) stays out
    of the prune."""

    ablation = {"artifact_id": "strategy_research_0ab1a7e0f00d", "source_step_id": "s", "line": "main.py:7: A -> B"}

    def recorded(records: list[dict]) -> None:
        _record(records, "research_session")["frozen"][RETIRED_ABLATION_KEY] = ablation

    root = tmp_path / "experiments"
    copy = _copy(arm, root, recorded, research_years=None, sealed_years=1, ablation_required=True)
    assert current_arms(root) == []
    records = [json.loads(line) for line in (copy / LEDGER).read_text(encoding="utf-8").splitlines()]
    assert ablation["artifact_id"] in _keep_frozen_artifact_ids(records)
