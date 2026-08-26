from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import run_audit_session
from autotrade.pipelines.pit_views_seed import DEFAULT_PIT_VIEWS_SEED


class _ProviderConstructed(RuntimeError):
    pass


def test_audit_pipeline_uses_the_default_pit_view_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def build_provider(**kwargs: object) -> object:
        captured.update(kwargs)
        raise _ProviderConstructed

    monkeypatch.setattr(run_audit_session, "ExperimentLedger", lambda path: object())
    monkeypatch.setattr(
        run_audit_session, "FilesystemArtifactStore", lambda path: object()
    )
    monkeypatch.setattr(
        run_audit_session, "ResearchPITSnapshotProvider", build_provider
    )

    llm = SimpleNamespace(
        compact_enabled=False,
        build_gateway=lambda role: object(),
    )
    options = SimpleNamespace(
        rolling=SimpleNamespace(ledger_path=tmp_path / "ledger.jsonl"),
        experiment_dir=tmp_path / "experiments" / "audit",
        llm=llm,
        agent_sandbox=object(),
        data_backend="pit",
        raw_dir=tmp_path / "raw",
        fundamental_events_root=tmp_path / "fundamentals",
        fundamental_events_status=tmp_path / "fundamentals-status.json",
        snapshot_config=object(),
        pit_cache_root=tmp_path / "pit-cache",
        repo_root=tmp_path,
    )

    with pytest.raises(_ProviderConstructed):
        run_audit_session._build_pipeline(options)

    assert captured["pit_views_seed"] == tmp_path / DEFAULT_PIT_VIEWS_SEED
