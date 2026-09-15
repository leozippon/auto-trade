from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_audit_session
from autotrade.environment.sandbox import SandboxSpec
from autotrade.pipelines import worker as worker_module


class _ProviderConstructed(RuntimeError):
    pass


def _options(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    """One options set, shaped like the worker's validated options."""

    options = SimpleNamespace(
        rolling=SimpleNamespace(
            ledger_path=tmp_path / "ledger.jsonl",
            schedule=object(),
            broker_profile=object(),
            step_tree_enabled=True,
            research_directive="directive",
            max_llm_calls=800,
            max_research_minutes=20,
            strategy_fit_timeout_seconds=1800,
            nl_failure_policy="fail",
            workspace_reference="configs/workspace_refs/pack",
            operating_memory="curated+graduated",
        ),
        experiment_dir=tmp_path / "experiments" / "audit",
        experiment_id="audit",
        work_root=tmp_path / "work",
        repo_root=tmp_path,
        baseline_strategy=tmp_path / "strategy",
        developer_mode="llm",
        llm=SimpleNamespace(
            compact_enabled=True,
            compaction=object(),
            compaction_for=lambda role: object(),
            build_gateway=lambda role, **kwargs: object(),
            max_tokens_for=lambda role: 4096,
        ),
        agent_sandbox=SandboxSpec(image="audit-image:test"),
        data_backend="daily",
        daily_path=tmp_path / "daily.parquet",
        execution_mode="sandbox",
        max_intraday_row_group_rows=2_000_000,
        nl_config=object(),
    )
    for key, value in overrides.items():
        setattr(options, key, value)
    return options


def _build(options: SimpleNamespace):
    return worker_module.build_experiment_pipeline(
        options,
        ledger=object(),
        store=object(),
        ref_store=object(),
    )


def test_the_audit_entrypoint_assembles_through_the_shared_builder() -> None:
    """The audit script must not grow a second assembly.

    A hand-written copy is what drifted last time: the console built the
    compaction gateway without retries and the audit path did not, so the
    audited session was not the session the console runs. The script may
    therefore name the Agent adapters and the backends nowhere.
    """

    source = Path(run_audit_session.__file__).read_text(encoding="utf-8")
    assert "build_experiment_pipeline(" in source
    for name in (
        "LLMResearchDeveloper",
        "LLMMetaLearner",
        "RollingExperimentPipeline",
        "ResearchPITSnapshotProvider",
        "PITDailyEvaluationBackend",
        "LocalDailySnapshotProvider",
        "LocalDailyEvaluationBackend",
    ):
        assert name not in source, name


def test_the_compaction_gateway_is_built_without_provider_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed compaction falls through to the emergency fit path by design,
    so provider retries would only add their full latency to that failure.
    Every surface that assembles an experiment gets that decision, because
    there is exactly one assembly."""

    roles: list[tuple[str, dict[str, object]]] = []

    def build_gateway(role: str, **kwargs: object) -> object:
        roles.append((role, kwargs))
        return object()

    monkeypatch.setattr(
        worker_module, "LocalDailySnapshotProvider", lambda path: object()
    )
    monkeypatch.setattr(
        worker_module,
        "LocalDailyEvaluationBackend",
        lambda *args, **kwargs: SimpleNamespace(
            trading_days=["20240102"], sandbox=kwargs.get("sandbox")
        ),
    )
    monkeypatch.setattr(
        worker_module, "LLMResearchDeveloper", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        worker_module,
        "RollingExperimentPipeline",
        lambda *args, **kwargs: SimpleNamespace(**kwargs),
    )
    options = _options(tmp_path)
    options.llm.build_gateway = build_gateway

    build = _build(options)

    assert build.developer_label == "llm_research_agent"
    assert dict(roles)["compact"] == {"max_retries": 0}
    assert [role for role, _kwargs in roles] == [
        "main",
        "subagent",
        "nl",
        "compact",
    ]


def test_the_assembled_session_mounts_the_image_refs_and_memory_it_was_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--sandbox-image has to reach the research developer, and the strategy
    wall clock the Agent is promised has to be the one the options carry rather
    than a library default."""

    captured: dict[str, dict[str, object]] = {}

    def capture(name: str):
        def build(**kwargs: object) -> object:
            captured[name] = kwargs
            return object()

        return build

    monkeypatch.setattr(
        worker_module, "LocalDailySnapshotProvider", lambda path: object()
    )
    monkeypatch.setattr(
        worker_module,
        "LocalDailyEvaluationBackend",
        lambda *args, **kwargs: SimpleNamespace(
            trading_days=["20240102"], sandbox=kwargs.get("sandbox")
        ),
    )
    monkeypatch.setattr(worker_module, "LLMResearchDeveloper", capture("developer"))
    monkeypatch.setattr(
        worker_module,
        "RollingExperimentPipeline",
        lambda *args, **kwargs: SimpleNamespace(**kwargs),
    )

    options = _options(tmp_path)
    build = _build(options)

    assert build.trading_days == ["20240102"]
    assert build.pipeline.evaluator.sandbox.image == "audit-image:test"
    assert build.pipeline.evaluator.sandbox.limits.fit_timeout_seconds == 1800
    developer = captured["developer"]
    assert developer["sandbox_spec"] is options.agent_sandbox
    assert developer["evaluator"] is build.pipeline.evaluator
    assert developer["workspace_reference"] == "configs/workspace_refs/pack"
    assert developer["operating_memory"] == "curated+graduated"
    assert developer["repo_root"] == tmp_path


def test_the_pipeline_uses_the_experiments_own_pit_view_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seed is the experiment's resolved one, not a hardcoded default.

    An arm whose dataset selection needs its own prebuilt tree must read that
    tree here too, and on the same terms: an explicitly chosen seed has to
    apply rather than fall back to a cold build.
    """

    captured: dict[str, object] = {}

    def build_provider(**kwargs: object) -> object:
        captured.update(kwargs)
        raise _ProviderConstructed

    monkeypatch.setattr(
        worker_module, "ResearchPITSnapshotProvider", build_provider
    )
    options = _options(
        tmp_path,
        data_backend="pit",
        raw_dir=tmp_path / "raw",
        fundamental_events_root=tmp_path / "fundamentals",
        fundamental_events_status=tmp_path / "fundamentals-status.json",
        snapshot_config=object(),
        pit_cache_root=tmp_path / "pit-cache",
        pit_views_seed=tmp_path / "data/pit_views_seed_ext",
        pit_views_seed_required=True,
    )

    with pytest.raises(_ProviderConstructed):
        _build(options)

    assert captured["pit_views_seed"] == tmp_path / "data/pit_views_seed_ext"
    assert captured["pit_views_seed_required"] is True


def test_the_audit_calendar_defaults_are_the_console_geometry() -> None:
    """The audit CLI and the console must launch the same research calendar."""
    from argparse import ArgumentParser

    from autotrade.pipelines.calendar import GEOMETRY_PARAMETERS
    from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
    from scripts.experiments._cli import add_calendar_arguments

    parser = ArgumentParser()
    add_calendar_arguments(parser)
    args = parser.parse_args([])
    assert {name: getattr(args, name) for name in GEOMETRY_PARAMETERS} == {
        name: WEB_CREATE_DEFAULTS[name] for name in GEOMETRY_PARAMETERS
    }
