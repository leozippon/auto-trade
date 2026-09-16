"""The run facts a session is told: the research period and the session's
place in the arm, universe policy and strategy call cadence — one sentence
each, with every date after research end absent — plus the budgets and
decision-input windows the same object publishes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from autotrade.agent.experiment_facts import (
    BATCH_VALIDATE_FIT_TIMEOUT_NOTE,
    SEARCH_ROOT_SHELL_NOTE,
    SMOKE_PROBE_NOTE,
    build_experiment_facts,
)
from autotrade.agent.prompts import build_system_prompt
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.identity import AgentRefStore


def _facts(
    *,
    data_summary: dict[str, object] | None = None,
    **manifest_overrides: object,
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "experiment_id": "exp",
        "run_id": "run_x",
        "epoch_id": "research",
        "fold_id": "s4",
        "kind": "research",
        "research": {
            "decision_time": "2025-06-30T23:59:59+08:00",
            "input_window": "20230701..20250630",
            "research_period": "20210701..20250630",
            "years": [{"label": "Y1", "start": "20210701", "end": "20220630"}],
        },
        "schedule": {"period": "day", "inference_time": "08:30"},
        "snapshot_config": SnapshotConfig().to_record(),
    }
    manifest.update(manifest_overrides)
    with tempfile.TemporaryDirectory() as tmp:
        return build_experiment_facts(
            manifest=manifest,
            ref_store=AgentRefStore(Path(tmp) / "experiment"),
            data_summary=data_summary,
        )


def _data_summary(rows: dict[str, int]) -> dict[str, object]:
    """A snapshot view whose files carry the given row counts."""

    return {
        "views": {
            "snapshot": {
                "mount_path": "/mnt/snapshot",
                "files": [
                    {"path": name, "mount_path": f"/mnt/snapshot/{name}", "rows": count}
                    for name, count in rows.items()
                ],
            }
        }
    }


def test_the_session_facts_state_the_research_period_and_the_session_place() -> None:
    facts = _facts()
    scope = facts["research_scope"]
    assert "one research session researches the period 20210701..20250630" in scope["research"]
    assert "no other session follows" in scope["research"]
    assert "sealed data" in scope["research"]
    assert scope["universe"].startswith("The universe is unfiltered")
    assert "ST names included" in scope["universe"]
    assert "every trading day at 08:30" in scope["strategy_cadence"]
    assert "own rebalance cadence" in scope["strategy_cadence"]
    assert "session" not in facts["identity"]
    assert facts["visibility_policy"]["research_period_visible"] is True
    assert "不进入任何会话" in facts["visibility_policy"]["after_research_end"]
    rendered = json.dumps(facts, ensure_ascii=False)
    for later in ("2026", "202507", "2025-07"):
        assert later not in rendered
    assert '"s4"' not in rendered


def test_the_signal_screen_path_is_a_fact_only_where_the_mount_exists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = AgentRefStore(Path(tmp) / "experiment")
        docker_facts = build_experiment_facts(
            manifest={"kind": "research", "experiment_id": "exp", "run_id": "run_x"},
            ref_store=store,
            runtime_env={"mode": "docker"},
        )
        local_facts = build_experiment_facts(
            manifest={"kind": "research", "experiment_id": "exp", "run_id": "run_x"},
            ref_store=store,
            runtime_env={"mode": "local"},
        )
    screen = docker_facts["source_refs"]["signal_screen_ref"]
    assert screen["path"] == "/mnt/tools/screen.py"
    # Sub-agents kept handing the bare path to read_file; the fact now carries
    # the argv contract and says which tools cannot open it.
    assert '["python", "/mnt/tools/screen.py", "--help"]' in screen["usage"]
    assert "shell only" in screen["usage"]
    assert "read_file" in screen["usage"]
    assert "signal_screen_ref" not in local_facts["source_refs"]


def test_a_screened_universe_and_a_monthly_cadence_are_described() -> None:
    screened = SnapshotConfig(
        screen_exclude_st=True, screen_exclude_new_listed_days=180, screen_boards=("main",)
    ).to_record()
    facts = _facts(
        snapshot_config=screened,
        schedule={"period": "month", "inference_time": "09:00"},
    )
    scope = facts["research_scope"]
    assert scope["universe"].startswith("The universe is screened")
    assert "exclude_st=True" in scope["universe"]
    assert "boards=['main']" in scope["universe"]
    assert "first available trading day of each month at 09:00" in scope["strategy_cadence"]


def test_both_strategy_wall_clocks_reach_the_session() -> None:
    """The prompts tell the agent to read both, so both must be projected."""

    budgets = _facts(
        budgets={
            "max_llm_calls": 1600,
            "deadline_seconds": 43200,
            "strategy_inference_timeout_seconds": 30.0,
            "strategy_fit_timeout_seconds": 1800,
        }
    )["budgets"]
    assert budgets["strategy_inference_timeout_seconds"] == 30.0
    assert budgets["strategy_fit_timeout_seconds"] == 1800


def test_the_session_deadline_names_the_wrap_up_grace_inside_it() -> None:
    """``deadline_seconds`` is main deadline PLUS grace.

    Without the split the session plans against a wall clock ten minutes later
    than the one its directive names and the one hard finalization uses.
    """

    budgets = _facts(
        budgets={"deadline_seconds": 43800.0, "deadline_grace_seconds": 600.0}
    )["budgets"]
    assert budgets["deadline_seconds"] == 43800.0
    assert budgets["deadline_grace_seconds"] == 600.0


def test_the_session_is_told_deadline_seconds_is_pausable_effective_time() -> None:
    """The session must see the pause clock next to ``deadline_seconds``.

    Every replay tool -- ``smoke_backtest`` and ``run_null_control`` included
    -- pauses the budget; shell and sub-agent waits do not, and must not be
    written as exemptions in the sentence the sessions actually read.
    """

    expected = (
        "`deadline_seconds` 统计可暂停的有效推理时间；"
        "`smoke_backtest`、`batch_validate` 和 `run_null_control` "
        "调用期间暂停计时，"
        "因此会话总墙钟可能更长。"
    )
    facts = _facts(
        budgets={"deadline_seconds": 43800.0, "deadline_grace_seconds": 600.0}
    )
    assert facts["budgets"]["deadline_seconds_note"] == expected
    for name in ("shell", "subagent", "sub-agent", "子代理"):
        assert name not in facts["budgets"]["deadline_seconds_note"]
    assert expected in build_system_prompt(experiment_facts=facts)


def test_the_facts_say_where_the_session_started() -> None:
    """The template and a handed-on node are both stated; a later session
    sees the node id its working copy was seeded from."""

    template = _facts(start={"kind": "template", "template_ref": "agent_output_template"})
    assert template["artifact_contract"]["start"]["kind"] == "template"
    node = _facts(
        start={"kind": "step_node", "node_id": "research__session_ref_a__run_ref_b__valid_002"},
    )
    assert node["artifact_contract"]["start"] == {
        "kind": "step_node",
        "node_id": "research__session_ref_a__run_ref_b__valid_002",
    }


def test_the_replay_year_budget_travels_with_how_it_is_spent() -> None:
    budgets = _facts(budgets={"max_replay_years": 24, "max_null_controls": 3})["budgets"]
    assert budgets["max_replay_years"] == 24
    assert budgets["max_null_controls"] == 3
    assert "replay-year" in budgets["max_replay_years_note"]
    assert "max_replay_years_note" not in _facts(budgets={})["budgets"]


def test_the_intraday_lookback_is_named_only_when_minutes_are_built() -> None:
    """Without minute bars the execution policy reports none available, so no
    minute lookback window may be advertised either."""

    timeline = _facts()["visible_timeline"]
    assert "intraday_trade_days" not in timeline["snapshot_windows"]
    assert "decision_snapshot_intraday_lookback_trade_days" not in timeline
    assert timeline["execution_policy"]["historical_minutes_available"] is False

    with_minutes = _facts(
        snapshot_config=SnapshotConfig(include_intraday=True).to_record()
    )["visible_timeline"]
    assert with_minutes["snapshot_windows"]["intraday_trade_days"] == 21
    assert with_minutes["decision_snapshot_intraday_lookback_trade_days"] == 21


def test_a_zero_row_domain_file_is_reported_as_unavailable() -> None:
    """A switched-off domain (minutes) and a domain with nothing in the visible
    window (the auction before 2025) are still written as zero-row Parquet
    files. Reading availability off file presence told the session it could
    price orders at an exact minute or at the auction, and every such order was
    rejected as ``missing_execution_price``."""

    empty = _facts(
        data_summary=_data_summary(
            {
                "intraday_1min.parquet": 0,
                "auction.parquet": 0,
                "events.parquet": 13_770_524,
                "text_index.parquet": 0,
            }
        )
    )["visible_timeline"]["execution_policy"]
    assert empty["historical_minutes_available"] is False
    assert empty["auction_available"] is False
    assert empty["text_available"] is False
    # A populated domain in the same summary still reports available.
    assert empty["events_available"] is True

    populated = _facts(
        data_summary=_data_summary(
            {"intraday_1min.parquet": 4_800_000, "auction.parquet": 5_067}
        )
    )["visible_timeline"]["execution_policy"]
    assert populated["historical_minutes_available"] is True
    assert populated["auction_available"] is True
    assert populated["events_available"] is False


def test_a_cpu_only_experiment_still_states_its_strategy_gpu_count() -> None:
    """0 is the answer "every formal replay runs on CPU", not a missing fact:
    dropping it would leave the session unable to tell a CPU-only arm from an
    older manifest that never published the field."""

    budgets = _facts(
        budgets={
            "strategy_fit_timeout_seconds": 3600.0,
            "strategy_gpu_count": 0,
        }
    )["budgets"]
    assert budgets["strategy_gpu_count"] == 0
    assert "strategy_gpu_count" not in _facts(budgets={})["budgets"]


def test_the_facts_publish_the_strategy_containers_cpu_quota_and_batch_width() -> None:
    """A session sizing its own thread pools must not have to guess.

    The formal strategy container is started with a fixed CPU quota and has its
    OMP/MKL/OPENBLAS/NUMEXPR variables set from it; one session guessed half of
    it and lost two backtest slots to fit timeouts. Both this quota and how many
    replays a batch runs at once are read from the code that enforces them, so
    the fact cannot drift from the container.
    """

    from autotrade.environment.sandbox import SandboxLimits
    from autotrade.pipelines.local_backend import BATCH_VALIDATE_MAX_CONCURRENCY

    budgets = _facts()["budgets"]
    assert budgets["strategy_cpus"] == SandboxLimits().cpus
    assert budgets["batch_validate_max_concurrency"] == BATCH_VALIDATE_MAX_CONCURRENCY
    # The width alone would mislead: the fit clock the batch is judged against
    # scales with it, so the rule travels with the number.
    assert budgets["batch_validate_fit_timeout_note"] == BATCH_VALIDATE_FIT_TIMEOUT_NOTE
    assert "strategy_cpus" in build_system_prompt(experiment_facts=_facts())


def test_the_facts_publish_the_strategy_containers_memory_ceiling() -> None:
    """The container the replay runs in is not the one the session lives in.

    An arm extrapolated a 7.34 GiB fit, read it against the session Sandbox's
    own limit, concluded the replay was swapping, and spent hours on a cause
    the strategy container cannot have. The ceiling it should have compared
    against is published here, in the same unit a replay reports its measured
    peak in.
    """

    from autotrade.environment.sandbox import SandboxLimits

    budgets = _facts()["budgets"]
    assert budgets["strategy_memory_bytes"] == SandboxLimits().memory_bytes
    assert budgets["strategy_memory_bytes"] == 32 * 1024**3
    # And why a rehearsal at the start of the span does not size a batch.
    assert budgets["smoke_backtest_probe_note"] == SMOKE_PROBE_NOTE
    assert "start" in SMOKE_PROBE_NOTE


def test_the_facts_place_every_read_root_in_the_shell_filesystem() -> None:
    """Root names are a tool convention, not paths.

    Two arms read ``steps`` as a shell path, failed, and concluded the Step
    tree was invisible to ``shell``; it is mounted, under /mnt/artifacts. The
    fact states where each root really is, and that ``trace`` is the one with
    no path in the container.
    """

    note = _facts()["runtime_tools"]["file_root_shell_paths"]
    assert note == SEARCH_ROOT_SHELL_NOTE
    for path in ("/mnt/artifacts/steps", "/mnt/snapshot", "/mnt/agent/workspace"):
        assert path in note
    assert "trace" in note
