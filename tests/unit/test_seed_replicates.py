"""Seed replicates: a strategy that trains a model is frozen with its seeds.

The arm's own pipeline runs end to end -- the real artifact store, ledger,
freeze registration, sealed reading and verdict -- over scripted Validations
whose daily active return is set per training seed (``SEED_BASE`` in the
candidate's ``main.py``) and per stage, so the readings each seed gets are
known before the pipeline reads them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from autotrade.environment.artifacts import FilesystemArtifactStore
from autotrade.environment.replay.style import STYLE_ARTIFACT_NAME
from autotrade.pipelines.config import (
    ArtifactRevision,
    EvaluationRequest,
    EvaluationResult,
    ResearchSessionRequest,
    ResearchSessionResult,
    RollingExperimentConfig,
    StepResult,
    acceptance_for,
    research_span,
)
from autotrade.pipelines.experiment import RollingExperimentPipeline
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    experiment_verdict,
    forward_record,
)
from tests.unit.acceptance_fixture import REQUIRED_RULES
from tests.unit.test_rolling_pipeline import (
    CONFIG_PROFILE,
    CONFIG_SCHEDULE,
    DAYS,
    GEOMETRY,
    RELEASE_END,
    Snapshots,
    _write_result,
)

FIT = '''REFIT_PERIOD = "quarter"
SEED_BASE = {seed}
SEEDS_PER_HEAD = {heads}
MAX_SWAPS = {swaps}
MECHANISM = {mechanism}


def fit(context):
    return None


def generate_orders(context):
    return []
'''
NO_FIT = '''SEED_BASE = {seed}
SEEDS_PER_HEAD = {heads}
MAX_SWAPS = {swaps}
MECHANISM = {mechanism}


def generate_orders(context):
    return []
'''
# Daily active return by stage and training seed: research Validations and the
# sealed reading replay research years in mode ``valid``, the forward stage in
# ``heldout``. An active return of 0.0009 a day or more reads a residual IR far
# above the one-year bar; -0.0010 reads far below zero. Forward, seed 1000
# beats its panel well clear of its noise, seed 2000 barely and seed 3000
# loses nearly as much as the other two win: their mean stays above zero and
# inside its noise.
ALPHAS = {
    ("valid", 1000): 0.0012,
    ("valid", 2000): 0.0010,
    ("valid", 3000): 0.0009,
    ("valid", 4000): -0.0010,
    ("heldout", 1000): 0.0015,
    ("heldout", 2000): 0.0002,
    ("heldout", 3000): -0.0014,
}
# A candidate with its mechanism switched off (``MECHANISM = False``, an
# ablation) earns this daily active return whatever its seed.
ABLATED = {"valid": 0.0003, "heldout": 0.0}


def _source(
    seed: int, *, swaps: int = 4, heads: int = 2, fit: bool = True, mechanism: bool = True
) -> str:
    return (FIT if fit else NO_FIT).format(seed=seed, swaps=swaps, heads=heads, mechanism=mechanism)


class SeedEvaluator:
    """A result whose zero-skill panel is the replay's market and size
    exposure, so the active series is the seed's daily ``alpha`` plus noise."""

    def __init__(
        self, root: Path, *, fail_seed: int | None = None, bare_seed: int | None = None
    ) -> None:
        self.root = root
        self.requests: list[EvaluationRequest] = []
        # Forward, this seed's replay raises; that one's stores no panel.
        self.fail_seed = fail_seed
        self.bare_seed = bare_seed

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        self.requests.append(request)
        source = (Path(request.revision.output_path) / "main.py").read_text(encoding="utf-8")
        seed = int(re.search(r"SEED_BASE = (\d+)", source)[1])  # type: ignore[index]
        if seed == self.fail_seed and request.mode == "heldout":
            raise TimeoutError("strategy inference exceeded 360s")
        ablated = "MECHANISM = False" in source
        result = _write_result(
            self.root / f"{request.mode}_{len(self.requests):03d}",
            [day for day in DAYS if request.start <= day <= min(request.end, RELEASE_END)],
            alpha=ABLATED[request.mode] if ablated else ALPHAS[(request.mode, seed)],
            # The draw is the seed's and the stage's, whatever replayed before.
            seed=seed + (28 if request.mode == "heldout" else 0) + (50 if ablated else 0),
        )
        sidecar = Path(result.result_ref).parent / STYLE_ARTIFACT_NAME
        analysis = json.loads(sidecar.read_text(encoding="utf-8"))
        if seed == self.bare_seed and request.mode == "heldout":
            del analysis["panel_daily"]
        else:
            analysis["panel_daily"] = [
                [day, 0.9 * market + 0.2 * size]
                for (day, market), (_day, size) in zip(
                    analysis["benchmark_daily"], analysis["size_factor_daily"], strict=True
                )
            ]
        sidecar.write_text(json.dumps(analysis), encoding="utf-8")
        return result


class SeedDeveloper:
    """Validates each scripted ``(source, span)`` and freezes ``nominee`` with
    the candidates ``replicates`` names as its seed replicates and the one
    ``ablation`` names as its ablation."""

    def __init__(
        self,
        store,
        evaluator,
        candidates,
        *,
        nominee: int,
        replicates,
        deterministic: bool,
        ablation: int | None = None,
    ):
        self.store = store
        self.evaluator = evaluator
        self.candidates = candidates
        self.nominee = nominee
        self.replicates = tuple(replicates)
        self.deterministic = deterministic
        self.ablation = ablation

    def __call__(self, request: ResearchSessionRequest) -> ResearchSessionResult:
        steps = []
        for number, (source, span) in enumerate(self.candidates):
            directory = self.store.root.parent / "candidates" / f"candidate_{number}"
            directory.mkdir(parents=True)
            (directory / "main.py").write_text(source, encoding="utf-8")
            revision = self.store.create_revision(directory)
            typed = ArtifactRevision(str(revision.revision_id), Path(revision.output_path))
            replayed = research_span(request.research_years, span)
            validation = self.evaluator.evaluate(
                replayed.request(typed, schedule=CONFIG_SCHEDULE, broker_profile=CONFIG_PROFILE)
            )
            steps.append(
                StepResult(f"research_step_{number}", typed.revision_id, validation, span=replayed.label)
            )
        return ResearchSessionResult(
            f"conversation_{request.run_id}",
            tuple(steps),
            "freeze",
            node_id=steps[self.nominee].step_id,
            seed_replicates=tuple(steps[index].step_id for index in self.replicates),
            deterministic=self.deterministic,
            ablation=None if self.ablation is None else steps[self.ablation].step_id,
        )


def _arm(
    tmp_path: Path,
    candidates,
    *,
    nominee=0,
    replicates=(),
    deterministic=False,
    ablation=None,
    ablation_required=False,
    **replays,
):
    config = RollingExperimentConfig(
        experiment_id="arm",
        experiments_root=tmp_path / "experiments",
        geometry=GEOMETRY,
        acceptance=acceptance_for({**REQUIRED_RULES, "ablation_required": ablation_required}),
    )
    store = FilesystemArtifactStore(config.experiment_dir / "artifacts" / "strategy")
    evaluator = SeedEvaluator(config.experiment_dir / "artifacts" / "results", **replays)
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=store,
        evaluator=evaluator,
        developer=SeedDeveloper(
            store,
            evaluator,
            candidates,
            nominee=nominee,
            replicates=replicates,
            deterministic=deterministic,
            ablation=ablation,
        ),
        trading_days=DAYS,
        ledger=ExperimentLedger(config.ledger_path),
    )
    return pipeline, evaluator


def test_a_model_nominee_freezes_with_its_seed_replicate_and_both_replay_forward(tmp_path: Path):
    """The registered replicates are checked, frozen beside the nominee, and
    replayed exactly like it on the sealed years and then forward; the
    forward record carries their readings beside the nominee's."""

    pipeline, evaluator = _arm(
        tmp_path,
        [(_source(1000), "full"), (_source(2000), "full"), (_source(3000), "full")],
        replicates=(1, 2),
    )
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    seeds = gate["seed_replicates"]
    assert gate["passed"] is True and gate["reasons"] == []
    # Nothing was declared, so no record states a declaration.
    assert seeds["trains_a_model"] is True and "deterministic" not in seeds
    assert "deterministic" not in record["frozen"]
    assert [entry["seed_line"] for entry in seeds["replicates"]] == [
        "main.py: SEED_BASE = 2000",
        "main.py: SEED_BASE = 3000",
    ]
    # Registration reads no score: no bar, and no mean over the seeds.
    assert "mean_information_ratio" not in seeds and "information_ratio_bar" not in seeds
    frozen = record["frozen"]
    assert [item["source_step_id"] for item in frozen["seed_replicates"]] == [
        "research_step_1",
        "research_step_2",
    ]
    assert all(Path(item["output_path"], "main.py").is_file() for item in frozen["seed_replicates"])

    research = len(evaluator.requests)
    forward = pipeline.run_forward()
    replays = evaluator.requests[research:]

    # Three replays of the sealed years, then three of forward and Held-out,
    # each under one Broker, the nominee first.
    artifacts = [frozen["artifact_id"], *(item["artifact_id"] for item in frozen["seed_replicates"])]
    assert [request.revision.revision_id for request in replays] == artifacts * 2
    shapes = [(r.mode, r.start, r.end, r.continuation, r.broker_profile) for r in replays]
    assert shapes[:3] == [shapes[0]] * 3 and shapes[3:] == [shapes[3]] * 3
    sealed = forward["sealed"]
    assert (shapes[0][1], shapes[0][2]) == (sealed["start"], sealed["end"]) == ("20230701", "20240630")
    assert [row["source_step_id"] for row in sealed["seed_replicates"]] == [
        "research_step_1",
        "research_step_2",
    ]
    readings = sealed["slice"]["seed_replicates"]
    assert [row["source_step_id"] for row in readings] == ["research_step_1", "research_step_2"]
    assert all(row["residual_information_ratio"] > row["bar"] for row in readings)
    assert sealed["slice"]["reasons"] == []
    assert [row["source_step_id"] for row in forward["seed_replicates"]] == [
        "research_step_1",
        "research_step_2",
    ]
    assert forward["seed_replicates"][0]["refits_executed"] == forward["refits_executed"]
    for name in ("forward", "heldout"):
        block = forward["slices"][name]
        members = [block, *block["seed_replicates"]]
        assert block["seed_mean"]["members"] == 3
        for key in ("plain_selection", "strategy_return", "panel_return"):
            assert block["seed_mean"]["raw_readings"][key] == pytest.approx(
                sum(member["raw_readings"][key] for member in members) / 3
            )
    block = forward["slices"]["forward"]
    # The nominee itself beat its panel, and so does the mean over the three
    # seeds; the mean's lower bound does not clear zero.
    assert block["raw_readings"]["plain_selection"] > 0
    assert block["seed_mean"]["raw_readings"]["plain_selection"] > 0
    assert block["lower_bound"] < 0 < block["seed_mean"]["plain_excess"]
    assert forward["verdict"]["reasons"] == ["forward_lower_bound_not_positive"]
    assert experiment_verdict(pipeline.ledger.read())["status"] == "discarded"


def test_seeds_that_agree_forward_graduate_together(tmp_path: Path):
    pipeline, _evaluator = _arm(
        tmp_path, [(_source(1000), "full"), (_source(2000), "full")], replicates=(1,)
    )
    pipeline.run_research_session()
    forward = pipeline.run_forward()
    block = forward["slices"]["forward"]
    assert block["seed_mean"]["members"] == 2 and block["lower_bound"] > 0
    assert forward["verdict"]["status"] == "graduated"


def _refused(tmp_path: Path, candidates, **arm) -> dict[str, object]:
    pipeline, _evaluator = _arm(tmp_path, candidates, **arm)
    record = pipeline.run_research_session()
    assert record["frozen"] is None
    assert record["arm_end"]["status"] == "no_deliverable"
    return record["freeze_gate"]


def test_a_model_nominee_without_a_replicate_is_refused(tmp_path: Path):
    gate = _refused(tmp_path, [(_source(1000), "full"), (_source(1000, swaps=3), "full")])
    assert gate["reasons"] == ["freeze_too_few_seed_replicates"]
    assert gate["seed_replicates"]["replicates"] == []


@pytest.mark.parametrize(
    ("replicate", "problem"),
    [
        ((_source(1000), "full"), "holds the same bytes as the nominee"),
        ((_source(2000, swaps=3), "full"), "more than one line"),
        ((_source(1000, swaps=3), "full"), "does not give a seed name"),
        ((_source(1000, heads=4), "full"), "does not give a seed name"),
    ],
    ids=["same_bytes", "seed_and_a_knob", "a_knob_not_the_seed", "a_seed_count"],
)
def test_a_replicate_that_is_not_the_nominee_on_another_seed_is_refused(
    tmp_path: Path, replicate, problem
):
    gate = _refused(
        tmp_path,
        [(_source(1000), "full"), (_source(3000, swaps=2), "full"), replicate],
        replicates=(2,),
    )
    assert gate["reasons"] == ["freeze_seed_replicate_invalid"]
    [entry] = gate["seed_replicates"]["replicates"]
    assert problem in entry["problem"]


def test_a_replicate_that_loses_does_not_hold_back_the_registration(tmp_path: Path):
    """Registration reads no score: a valid replicate whose seed loses what
    the nominee's wins registers beside it."""

    pipeline, _evaluator = _arm(
        tmp_path, [(_source(1000), "full"), (_source(4000), "full")], replicates=(1,)
    )
    record = pipeline.run_research_session()
    assert record["freeze_gate"]["passed"] is True
    assert [item["source_step_id"] for item in record["frozen"]["seed_replicates"]] == [
        "research_step_1"
    ]


@pytest.mark.parametrize("replicates", [(), (1,)], ids=["alone", "with_a_worse_seed"])
def test_a_deterministic_model_nominee_owes_no_replicate(tmp_path: Path, replicates):
    """A nominee that trains a model but declares its package deterministic
    freezes without a replicate. The gate and the frozen block record the
    declaration for the operator's review."""

    pipeline, _evaluator = _arm(
        tmp_path,
        [(_source(1000), "full"), (_source(4000), "full")],
        replicates=replicates,
        deterministic=True,
    )
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    seeds = gate["seed_replicates"]
    assert gate["passed"] is True and gate["reasons"] == []
    assert seeds["trains_a_model"] is True and seeds["deterministic"] is True
    assert record["frozen"]["deterministic"] is True


def test_a_deterministic_nominee_is_held_to_its_replicates_validity(tmp_path: Path):
    gate = _refused(
        tmp_path,
        [(_source(1000), "full"), (_source(3000, swaps=2), "full"), (_source(1000), "full")],
        replicates=(2,),
        deterministic=True,
    )
    assert gate["reasons"] == ["freeze_seed_replicate_invalid"]
    assert gate["seed_replicates"]["deterministic"] is True


def test_a_nominee_that_trains_no_model_is_judged_on_its_own_series(tmp_path: Path):
    """A nominee that trains no model needs no replicate: it freezes alone,
    its gate reads that it had nothing to replicate, and forward its own
    series is judged, with its bound."""

    pipeline, evaluator = _arm(
        tmp_path, [(_source(1000, fit=False), "full"), (_source(2000, fit=False), "full")]
    )
    record = pipeline.run_research_session()
    gate = record["freeze_gate"]
    assert gate["passed"] is True
    assert gate["seed_replicates"]["trains_a_model"] is False
    assert gate["seed_replicates"]["replicates"] == []
    assert "seed_replicates" not in record["frozen"]
    research = len(evaluator.requests)
    forward = pipeline.run_forward()
    # One sealed replay, one forward.
    assert len(evaluator.requests) == research + 2
    block = forward["slices"]["forward"]
    assert "seed_mean" not in block
    assert 0 < block["lower_bound"] < block["plain_excess"]
    assert forward["verdict"]["status"] == "graduated"


@pytest.mark.parametrize(
    ("replays", "error", "message"),
    [
        (
            {"fail_seed": 2000},
            RuntimeError,
            r"seed replicate research_step_1 \(.*\) did not complete",
        ),
        ({"bare_seed": 2000}, ValueError, "graded another series than the frozen book"),
    ],
    ids=["its_replay_fails", "its_replay_has_no_panel"],
)
def test_a_replicate_missing_from_the_mean_fails_the_stage_and_leaves_no_verdict(
    tmp_path: Path, replays, error, message
):
    pipeline, _evaluator = _arm(
        tmp_path,
        [(_source(1000), "full"), (_source(2000), "full")],
        replicates=(1,),
        **replays,
    )
    pipeline.run_research_session()
    with pytest.raises(error, match=message):
        pipeline.run_forward()
    rows = pipeline.ledger.read()
    assert rows[-1]["record_type"] == "attempt_failed"
    assert forward_record(rows) is None
    assert experiment_verdict(rows) is None
