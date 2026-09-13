"""Meta regularization: three statuses, one freeze owner, one safety rule.

The Meta session may make small regularizing edits to the strategy working
copy, but it never freezes them — the Pipeline does, exactly as it does for a
Fold's selected Step. And because a regularized artifact enters the next Fold
without ever having been backtested, that Fold may keep it only after
validating identical content itself; a Fold that cannot reverts the lineage to
the artifact the Meta edited instead of failing the experiment.
"""

from __future__ import annotations

import shutil
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from autotrade.pipelines import (
    ArtifactRevision,
    EvaluationResult,
    FoldSessionResult,
    FrozenArtifact,
    RollingExperimentConfig,
    RollingExperimentPipeline,
    StepResult,
)
from autotrade.pipelines.agent_views import freeze_flags
from autotrade.pipelines.config import MetaSessionResult, SnapshotBundle
from autotrade.pipelines.experiment import REGULARIZATION_SMOKE_DAYS
from autotrade.pipelines.folds import build_fold_schedule
from autotrade.pipelines.hitl_state import iter_development_sessions
from autotrade.pipelines.ledger import (
    ExperimentLedger,
    preceding_meta_regularization,
)

MAIN = "def generate_orders(context):\n    return []\n"
DAYS = [stamp.strftime("%Y%m%d") for stamp in pd.bdate_range("2025-09-29", "2026-06-30")]


class Snapshots:
    def prepare(self, *, fold, phase, start, end, decision_time):
        return SnapshotBundle(f"{phase}_{start}_{end}", "decision", "replay")


class Artifacts:
    """A filesystem-backed double: revisions are real trees, so the guard's
    tree comparison is exercised rather than stubbed."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.revisions: dict[str, ArtifactRevision] = {}
        self.frozen_root = root / "frozen"
        self.frozen_root.mkdir(parents=True, exist_ok=True)

    def add_revision(self, revision_id: str, body: str) -> ArtifactRevision:
        output = self.root / "revisions" / revision_id
        output.mkdir(parents=True, exist_ok=True)
        (output / "main.py").write_text(body, encoding="utf-8")
        revision = ArtifactRevision(revision_id, output)
        self.revisions[revision_id] = revision
        return revision

    def revision(self, revision_id: str) -> ArtifactRevision:
        return self.revisions[revision_id]

    def freeze_revision(self, revision_id: str, **values):
        source = self.revisions[revision_id]
        target = self.frozen_root / values["artifact_id"]
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source.output_path, target)
        return FrozenArtifact(
            values["artifact_id"], target, None, values["run_id"], values["fold_id"],
            values["step_id"],
        )


class Evaluator:
    def evaluate(self, request, *, max_days: int | None = None):
        return EvaluationResult({"total_return": 0.02, "max_drawdown": 0.03}, f"result/{request.mode}")


class CrashingEvaluator:
    """Replays everything but a package carrying ``marker``, which raises.

    ``only_full_window`` makes it run the truncated rehearsal and fail the
    whole replay -- what a few days of smoke cannot rule out.
    """

    def __init__(self, marker: str, *, only_full_window: bool = False) -> None:
        self.marker = marker
        self.only_full_window = only_full_window

    def evaluate(self, request, *, max_days: int | None = None):
        source = (Path(request.revision.output_path) / "main.py").read_text(encoding="utf-8")
        if self.marker in source and not (self.only_full_window and max_days is not None):
            raise RuntimeError("TypeError: unsupported operand type(s) on day 1")
        return EvaluationResult(
            {"total_return": 0.02, "max_drawdown": 0.03}, f"result/{request.mode}"
        )


def _pipeline(root: Path, *, meta_learner, developer=lambda request: None, evaluator=None):
    config = RollingExperimentConfig(
        "exp", root / "experiments", "2026Q1", "2026Q1", "2026Q2", "2026Q2", fold_period="quarter", epochs=1
    )
    artifacts = Artifacts(root / "artifacts")
    pipeline = RollingExperimentPipeline(
        config,
        snapshots=Snapshots(),
        artifacts=artifacts,
        evaluator=evaluator or Evaluator(),
        developer=developer,
        meta_learner=meta_learner,
        ledger=ExperimentLedger(config.ledger_path),
    )
    return pipeline, artifacts, config


def _parent(artifacts: Artifacts, body: str = MAIN) -> FrozenArtifact:
    revision = artifacts.add_revision("revision_parent", body)
    return FrozenArtifact(
        "strategy_parent", revision.output_path, None, "run_seed", "fold_seed", "step_seed"
    )


class MetaSessionStatusTest(unittest.TestCase):
    """The three Meta outcomes are each recorded on the ledger row."""

    def _run(self, root: Path, session: MetaSessionResult, artifacts_body: str | None = None):
        pipeline, artifacts, config = _pipeline(root, meta_learner=lambda facts: session)
        parent = _parent(artifacts)
        if session.revision_id:
            artifacts.add_revision(session.revision_id, artifacts_body or (MAIN + "# regularized\n"))
        fold = build_fold_schedule("2026Q1", "2026Q1", DAYS, window_months=24)[0]
        prior, next_parent = pipeline.run_meta_session(
            "epoch_001", 1, fold, parent=parent, previous_prior=""
        )
        record = ExperimentLedger(config.ledger_path).read("meta_learning")[-1]
        return prior, next_parent, record

    def test_an_allowed_edit_is_frozen_and_becomes_the_next_parent(self) -> None:
        with TemporaryDirectory() as tmp:
            prior, next_parent, record = self._run(
                Path(tmp),
                MetaSessionResult(prior="prefer simple", revision_id="revision_meta", allowed=True),
            )
            self.assertEqual(prior, "prefer simple")
            self.assertEqual(record["status"], "meta_regularized")
            self.assertEqual(record["frozen_strategy_artifact_id"], next_parent.artifact_id)
            self.assertTrue(record["frozen_strategy_artifact_path"])
            self.assertIn("regularized", (next_parent.path / "main.py").read_text(encoding="utf-8"))
            # Never backtested: the next Fold must validate it before falling back.
            self.assertTrue(next_parent.requires_validation)

    def test_a_prior_only_session_keeps_the_parent(self) -> None:
        with TemporaryDirectory() as tmp:
            prior, next_parent, record = self._run(
                Path(tmp), MetaSessionResult(prior="prefer robust", allowed=True)
            )
            self.assertEqual(prior, "prefer robust")
            self.assertEqual(record["status"], "prior_only_kept_parent")
            self.assertEqual(next_parent.artifact_id, "strategy_parent")
            self.assertIsNone(record["frozen_strategy_artifact_id"])
            self.assertFalse(next_parent.requires_validation)

    def test_a_refused_edit_keeps_the_parent_and_prior(self) -> None:
        with TemporaryDirectory() as tmp:
            # A refused check makes the learner withhold the revision, exactly
            # as LLMMetaLearner does (`if parent_id and allowed and changed`).
            prior, next_parent, record = self._run(
                Path(tmp),
                MetaSessionResult(
                    prior="prefer robust",
                    allowed=False,
                    modification_check={"allowed_to_backtest": False,
                                        "reasons": ["diff lines 50 > 5"]},
                ),
            )
            # A refused regularization is an audited verdict, not lost PRIOR.
            self.assertEqual(prior, "prefer robust")
            self.assertEqual(record["status"], "rejected_kept_parent")
            self.assertEqual(next_parent.artifact_id, "strategy_parent")
            self.assertEqual(record["modification_check"]["reasons"], ["diff lines 50 > 5"])

    def test_the_pipeline_refuses_to_freeze_a_revision_the_check_disallowed(self) -> None:
        """Defence in depth: the Pipeline gates the freeze on the check verdict,
        not only on the learner's nomination."""
        with TemporaryDirectory() as tmp:
            _prior, next_parent, record = self._run(
                Path(tmp),
                MetaSessionResult(
                    prior="candidate", revision_id="revision_meta", allowed=False,
                    modification_check={"allowed_to_backtest": False},
                ),
            )
            self.assertEqual(record["status"], "rejected_kept_parent")
            self.assertEqual(next_parent.artifact_id, "strategy_parent")

    def test_the_learner_nominates_and_the_pipeline_freezes(self) -> None:
        with TemporaryDirectory() as tmp:
            _prior, next_parent, _record = self._run(
                Path(tmp),
                MetaSessionResult(prior="candidate", revision_id="revision_meta", allowed=True),
            )
            # The frozen artifact is a Pipeline-owned copy, not the learner's
            # working revision directory.
            self.assertIn("frozen", str(next_parent.path))
            self.assertTrue(next_parent.artifact_id.endswith("_meta_learning"))


class UnvalidatedParentFallbackTest(unittest.TestCase):
    """What counts as this Fold's own complete Validation of its parent.

    A meta-regularized parent may stay the lineage head only when the Fold it
    entered validated it: the host's parent control, or a Step whose trees are
    the parent's byte for byte.
    """

    def _guard(self, root: Path, *, steps, parent_body=MAIN):
        pipeline, artifacts, _config = _pipeline(root, meta_learner=lambda facts: None)
        parent = replace(_parent(artifacts, parent_body), requires_validation=True)
        return pipeline, artifacts, parent, steps

    def _step(self, artifacts: Artifacts, revision_id: str, body: str, *, summary=None):
        artifacts.add_revision(revision_id, body)
        return StepResult(
            f"step_{revision_id}",
            revision_id,
            EvaluationResult(
                summary or {"total_return": 0.02, "max_drawdown": 0.03},
                "result/valid",
            ),
        )

    def test_no_steps_at_all_is_not_a_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline, _artifacts, parent, _ = self._guard(Path(tmp), steps=())
            self.assertFalse(pipeline._parent_validated_in_fold(parent, ()))

    def test_an_acceptable_parent_control_is_the_validation(self) -> None:
        # The host replayed the meta-regularized parent on this Fold's window
        # before the session: that completed Validation satisfies the rule
        # even when the session itself recorded no Step (deadline).
        with TemporaryDirectory() as tmp:
            pipeline, _artifacts, parent, _ = self._guard(Path(tmp), steps=())
            control = EvaluationResult({"total_return": 0.02, "max_drawdown": 0.03}, "result/valid")
            self.assertTrue(
                pipeline._parent_validated_in_fold(parent, (), control=control)
            )
            # Only a hard reject disqualifies the control; a drawdown over the
            # cap now warns, so the unprovable case is a non-finite metric.
            broken = EvaluationResult(
                {"total_return": float("nan"), "max_drawdown": 0.03}, "result/valid"
            )
            self.assertFalse(
                pipeline._parent_validated_in_fold(parent, (), control=broken)
            )
            over_cap = EvaluationResult(
                {"total_return": 0.02, "max_drawdown": 0.9}, "result/valid"
            )
            self.assertTrue(
                pipeline._parent_validated_in_fold(parent, (), control=over_cap)
            )

    def test_a_step_validating_identical_content_is_the_validation(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline, artifacts, parent, _ = self._guard(Path(tmp), steps=())
            step = self._step(artifacts, "revision_same", MAIN)
            self.assertTrue(pipeline._parent_validated_in_fold(parent, (step,)))

    def test_a_step_validating_different_content_is_not(self) -> None:
        # Including the in-window repair of a broken parent: it proves the
        # repair, not the package the Meta froze.
        with TemporaryDirectory() as tmp:
            pipeline, artifacts, parent, _ = self._guard(Path(tmp), steps=())
            step = self._step(artifacts, "revision_other", MAIN + "# different\n")
            self.assertFalse(pipeline._parent_validated_in_fold(parent, (step,)))

    def test_identical_content_that_fails_acceptance_does_not_count(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline, artifacts, parent, _ = self._guard(Path(tmp), steps=())
            step = self._step(
                artifacts,
                "revision_same",
                MAIN,
                # Nothing was proven: the metric cannot even be compared.
                summary={"total_return": float("nan"), "max_drawdown": 0.03},
            )
            self.assertFalse(pipeline._parent_validated_in_fold(parent, (step,)))

    def test_only_a_requires_validation_parent_is_asked(self) -> None:
        with TemporaryDirectory() as tmp:
            _pipe, artifacts, parent, _ = self._guard(Path(tmp), steps=())
            self.assertTrue(parent.requires_validation)
            # A plain frozen Fold artifact was validated when it was frozen, so
            # run_fold never asks the question for it.
            self.assertFalse(_parent(artifacts).requires_validation)


BROKEN = MAIN + "# regularized, and broken on the first decision day\n"


class RejectedMetaRegularizationTest(unittest.TestCase):
    """A regularization the next Fold never validates is reverted, not fatal.

    The incident this covers: a Meta shrank the parent, the frozen package
    crashed on the first decision day of the next Fold, that Fold's parent
    control failed and its session abstained -- and the run died with the whole
    arm behind it. The regularization is the only unvalidated thing in the
    lineage, so rejecting it and continuing from the artifact it edited is a
    validated fallback; only a regularization with no validated predecessor
    leaves nothing to fall back to.
    """

    def _regularized_fold(self, root: Path, *, no_edge: bool = True):
        """Meta regularizes into a package that crashes, then the Fold runs."""

        def developer(request):
            return FoldSessionResult(
                "conversation",
                (),
                None,
                no_edge_reason=(
                    "the mounted parent crashes on day 1 and no candidate "
                    "proved an edge on this window"
                    if no_edge
                    else ""
                ),
            )

        pipeline, artifacts, config = _pipeline(
            root,
            meta_learner=lambda facts: MetaSessionResult(
                prior="shrink it", revision_id="revision_meta", allowed=True
            ),
            developer=developer,
            evaluator=CrashingEvaluator(
                "broken on the first decision day",
                # It survives the Pipeline's pre-freeze rehearsal and dies on
                # the full window, so the regularization is frozen and the next
                # Fold is where it fails.
                only_full_window=True,
            ),
        )
        artifacts.add_revision("revision_meta", BROKEN)
        fold = build_fold_schedule("2026Q1", "2026Q1", DAYS, window_months=24)[0]
        predecessor = _parent(artifacts)
        _prior, meta_parent = pipeline.run_meta_session(
            "epoch_001", 0, fold, parent=predecessor, previous_prior=""
        )
        return pipeline, config, fold, predecessor, meta_parent

    def test_the_meta_row_names_the_artifact_it_regularized(self) -> None:
        with TemporaryDirectory() as tmp:
            _pipe, config, _fold, predecessor, meta_parent = self._regularized_fold(
                Path(tmp)
            )
            record = ExperimentLedger(config.ledger_path).read("meta_learning")[-1]
            self.assertEqual(
                record["parent_strategy_artifact_id"], predecessor.artifact_id
            )
            self.assertEqual(
                record["frozen_strategy_artifact_id"], meta_parent.artifact_id
            )
            # The freeze carries its own fallback, so the next Fold never has to
            # rediscover which artifact the regularization replaced.
            self.assertEqual(
                meta_parent.validated_predecessor.artifact_id, predecessor.artifact_id
            )

    def test_a_fold_that_never_validates_it_reverts_to_the_predecessor(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline, config, fold, predecessor, meta_parent = self._regularized_fold(
                Path(tmp)
            )

            outcome = pipeline.run_fold("epoch_001", fold, parent=meta_parent)

            # The experiment continues, with the lineage it had before the Meta.
            self.assertEqual(outcome.frozen.artifact_id, predecessor.artifact_id)
            self.assertFalse(outcome.frozen.requires_validation)
            record = ExperimentLedger(config.ledger_path).read("fold")[-1]
            self.assertTrue(record["meta_regularization_rejected"])
            # The row names both ends of the revert: what was rejected, and what
            # the lineage went back to.
            self.assertEqual(
                record["parent_strategy_artifact_id"], meta_parent.artifact_id
            )
            self.assertEqual(
                record["frozen_strategy_artifact_id"], predecessor.artifact_id
            )
            self.assertEqual(record["fold_status"], "no_update")
            self.assertEqual(record["parent_control"]["status"], "failed")
            # And the next sessions read it: freeze_flags is what the Fold run
            # facts and the Meta reviews project.
            self.assertEqual(
                freeze_flags(record), {"meta_regularization_rejected": True}
            )

    def test_a_regularization_without_a_validated_predecessor_is_refused(self) -> None:
        # Nothing validated to revert to (a pre-existing ledger whose lineage
        # cannot be reconstructed): the run still refuses rather than shipping
        # a package no Fold ever backtested.
        with TemporaryDirectory() as tmp:
            pipeline, _config, fold, _predecessor, meta_parent = self._regularized_fold(
                Path(tmp)
            )
            orphan = replace(meta_parent, validated_predecessor=None)
            with self.assertRaisesRegex(RuntimeError, "refusing unvalidated fallback"):
                pipeline.run_fold("epoch_001", fold, parent=orphan)


class RegularizationSmokeTest(unittest.TestCase):
    """The Pipeline runs a regularized package before it freezes it.

    The Meta session has no evaluator and no backtest tool: it cannot execute
    what it edits, and a package that crashed on its first decision day was
    frozen and mounted as the next Fold's parent. The freeze owner rehearses it
    over that Fold's first trading days instead, and a package that cannot run
    is not adopted.
    """

    def _run(self, root: Path, evaluator):
        pipeline, artifacts, config = _pipeline(
            root,
            meta_learner=lambda facts: MetaSessionResult(
                prior="shrink it", revision_id="revision_meta", allowed=True
            ),
            evaluator=evaluator,
        )
        artifacts.add_revision("revision_meta", BROKEN)
        fold = build_fold_schedule("2026Q1", "2026Q1", DAYS, window_months=24)[0]
        _prior, next_parent = pipeline.run_meta_session(
            "epoch_001", 0, fold, parent=_parent(artifacts), previous_prior=""
        )
        record = ExperimentLedger(config.ledger_path).read("meta_learning")[-1]
        return next_parent, record

    def test_a_package_that_crashes_in_the_rehearsal_is_not_frozen(self) -> None:
        with TemporaryDirectory() as tmp:
            next_parent, record = self._run(
                Path(tmp), CrashingEvaluator("broken on the first decision day")
            )
            self.assertEqual(record["status"], "rejected_kept_parent")
            self.assertIsNone(record["frozen_strategy_artifact_id"])
            # The lineage head is untouched and still validated.
            self.assertEqual(next_parent.artifact_id, "strategy_parent")
            self.assertFalse(next_parent.requires_validation)
            self.assertEqual(record["regularization_smoke"]["status"], "failed")
            self.assertIn("unsupported operand", record["regularization_smoke"]["error"])

    def test_a_package_that_runs_is_frozen_and_the_row_records_the_rehearsal(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            next_parent, record = self._run(Path(tmp), Evaluator())
            self.assertEqual(record["status"], "meta_regularized")
            self.assertTrue(next_parent.requires_validation)
            self.assertEqual(
                record["regularization_smoke"],
                {"status": "ok", "days": REGULARIZATION_SMOKE_DAYS},
            )


class PrecedingMetaRegularizationTest(unittest.TestCase):
    """What the next Fold is told about the Meta that ran right before it.

    The Fold reads that Meta's PRIOR in full. A PRIOR that claims a cleanup the
    Pipeline then refused would otherwise describe an artifact the Fold never
    mounted, with nothing in the session to check it against.
    """

    def test_a_refused_regularization_reaches_the_next_fold_with_its_reason(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            seen: list[object] = []

            def developer(request):
                seen.append(request.meta_regularization)
                return FoldSessionResult(
                    "conversation",
                    (),
                    None,
                    no_edge_reason=(
                        "no candidate proved an edge on this window and the "
                        "parent stays the lineage head"
                    ),
                )

            pipeline, artifacts, config = _pipeline(
                Path(tmp),
                meta_learner=lambda facts: MetaSessionResult(
                    prior="shrink it", revision_id="revision_meta", allowed=True
                ),
                developer=developer,
                # The regularized package cannot run, so it is never frozen.
                evaluator=CrashingEvaluator("broken on the first decision day"),
            )
            artifacts.add_revision("revision_meta", BROKEN)
            fold = build_fold_schedule("2026Q1", "2026Q1", DAYS, window_months=24)[0]
            _prior, parent = pipeline.run_meta_session(
                "epoch_001", 0, fold, parent=_parent(artifacts), previous_prior=""
            )

            pipeline.run_fold("epoch_001", fold, parent=parent)

            self.assertEqual(seen[0]["status"], "rejected_kept_parent")
            self.assertIn("regularization_smoke", seen[0]["reason"])
            # And once this Fold has its own record, the Meta is no longer the
            # session that ran immediately before.
            records = ExperimentLedger(config.ledger_path).read()
            self.assertIsNone(preceding_meta_regularization(records))

    def test_a_frozen_regularization_is_reported_without_a_reason(self) -> None:
        records = [
            {
                "record_type": "meta_learning",
                "status": "meta_regularized",
                "frozen_strategy_artifact_id": "strategy_meta",
                "modification_check": {"allowed_to_backtest": True},
                "regularization_smoke": {"status": "ok", "days": 3},
            }
        ]
        self.assertEqual(
            preceding_meta_regularization(records), {"status": "meta_regularized"}
        )

    def test_a_check_refusal_carries_its_own_reasons(self) -> None:
        records = [
            {
                "record_type": "meta_learning",
                "status": "rejected_kept_parent",
                "modification_check": {
                    "allowed_to_backtest": False,
                    "reasons": ["unsupported models file type: deadcode_backup/old.py"],
                },
            }
        ]
        verdict = preceding_meta_regularization(records)
        assert verdict is not None
        self.assertEqual(verdict["status"], "rejected_kept_parent")
        self.assertIn("unsupported models file type", verdict["reason"])

    def test_an_empty_or_fold_led_ledger_says_nothing(self) -> None:
        self.assertIsNone(preceding_meta_regularization([]))
        self.assertIsNone(
            preceding_meta_regularization(
                [
                    {"record_type": "meta_learning", "status": "rejected_kept_parent"},
                    {"record_type": "fold", "fold_status": "frozen"},
                ]
            )
        )


class RegularizedParentKeepsItsIdentityTest(unittest.TestCase):
    """The Fold that re-validates a regularized parent must not rename it.

    A regularized artifact reaches the next Fold unbacktested, and that Fold
    normally hands it straight back — the host's parent control replayed it,
    and the Agent nominates that node. Freezing the same bytes under a third id
    would throw away the forward transition the control just produced, which is
    the only evidence graduation term (c) has about the artifact Held-out
    ships.
    """

    def test_a_fold_that_renominates_the_regularized_parent_keeps_its_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            regularized = MAIN + "# regularized\n"
            session = MetaSessionResult(
                prior="prefer simple", revision_id="revision_meta", allowed=True
            )

            def developer(request):
                return FoldSessionResult(
                    "conversation",
                    (
                        StepResult(
                            "control",
                            "revision_same",
                            request.parent_control,
                            parent_control=True,
                        ),
                    ),
                    "control",
                )

            pipeline, artifacts, config = _pipeline(
                root, meta_learner=lambda facts: session, developer=developer
            )
            artifacts.add_revision("revision_meta", regularized)
            fold = build_fold_schedule("2026Q1", "2026Q1", DAYS, window_months=24)[0]
            _prior, meta_parent = pipeline.run_meta_session(
                "epoch_001", 0, fold, parent=_parent(artifacts), previous_prior=""
            )
            self.assertTrue(meta_parent.requires_validation)
            # The Fold nominates the content the Meta published, byte for byte.
            artifacts.add_revision("revision_same", regularized)

            outcome = pipeline.run_fold("epoch_001", fold, parent=meta_parent)

            self.assertEqual(outcome.fold_status, "no_update")
            self.assertEqual(outcome.frozen.artifact_id, meta_parent.artifact_id)
            # This Fold's complete Validation is the one the Meta artifact owed.
            self.assertFalse(outcome.frozen.requires_validation)
            record = ExperimentLedger(config.ledger_path).read("fold")[-1]
            self.assertTrue(record["nominated_identical_to_parent"])
            self.assertEqual(
                record["frozen_strategy_artifact_id"], meta_parent.artifact_id
            )
            self.assertEqual(record["hard_reject_reasons"], [])


class MetaAlwaysHasAFollowingFoldTest(unittest.TestCase):
    """No Meta session is ever scheduled after the last development Fold.

    This is what makes a regularized artifact safe to publish: it is never the
    last thing a run produces. The Fold that follows re-validates it (the guard
    above) and replays it as that Fold's parent control, which is the forward
    transition graduation term (c) asks of the artifact Held-out ships. A
    schedule that ended on a Meta would ship a strategy no Fold ever backtested
    and no transition ever scored.
    """

    def test_every_meta_session_is_followed_by_the_fold_it_prepares(self) -> None:
        folds = build_fold_schedule("2025Q4", "2026Q2", DAYS, window_months=24)
        self.assertEqual(len(folds), 3)
        for interval in range(len(folds) + 2):
            sessions = iter_development_sessions(
                2, folds, meta_enabled=True, meta_learning_fold_interval=interval
            )
            kinds = [session.kind for session in sessions]
            self.assertEqual(kinds[-1], "fold", f"interval={interval}")
            for index, session in enumerate(sessions):
                if session.kind != "meta":
                    continue
                following = sessions[index + 1]
                # The Meta's own visible_fold is that very Fold, so run_meta
                # cannot be reached without a Fold left to re-validate it.
                self.assertEqual(following.kind, "fold")
                self.assertEqual(following.fold.fold_id, session.fold.fold_id)
                self.assertEqual(following.epoch_id, session.epoch_id)


if __name__ == "__main__":
    unittest.main()
