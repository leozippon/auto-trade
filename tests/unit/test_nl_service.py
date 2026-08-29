"""``ctx.nl()``: PIT retrieval, the bounded Sub Agent, and the evidence gate.

Every test drives the production composition — one :class:`TextRetriever` over
an on-disk snapshot layout, one :class:`CompanyContextStore`, one
:class:`NLSubAgentEngine` — because the failure mode this stack has repeatedly
had is a restored module with no live caller.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from autotrade.environment.llm import ProviderResponse, ScriptedLLM, ToolCall
from autotrade.environment.nl import NLConfig, NLService
from autotrade.environment.nl.service import NL_CALLS_PER_TRADING_DAY

NOW = datetime(2026, 1, 2, tzinfo=UTC)
INDEX_COLUMNS = [
    "text_id",
    "dataset",
    "ts_codes",
    "title",
    "available_at",
    "library_file",
]


def write_snapshot(
    root: Path,
    rows: list[tuple[str, str, str, str, str]],
    bodies: dict[str, str] | None = None,
    *,
    universe: bool = False,
) -> Path:
    """A minimal PIT text snapshot: index + per-dataset library shards."""
    root.mkdir(parents=True, exist_ok=True)
    library = root / "text_library"
    library.mkdir(exist_ok=True)
    frame = pd.DataFrame(
        [
            {
                "text_id": text_id,
                "dataset": dataset,
                "ts_codes": ts_codes,
                "title": title,
                "available_at": available_at,
                "library_file": f"{dataset}.parquet",
            }
            for text_id, dataset, ts_codes, title, available_at in rows
        ],
        columns=INDEX_COLUMNS,
    )
    frame.to_parquet(root / "text_index.parquet", index=False)
    for dataset, group in frame.groupby("dataset"):
        pd.DataFrame(
            {
                "text_id": list(group["text_id"]),
                "body": [
                    (bodies or {}).get(text_id, "") for text_id in group["text_id"]
                ],
            }
        ).to_parquet(library / f"{dataset}.parquet", index=False)
    if universe:
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["平安银行"],
                "exchange": ["SZSE"],
                "l1_name": ["银行"],
            }
        ).to_parquet(root / "universe.parquet", index=False)
        pd.DataFrame(
            {
                "dataset": ["fina_mainbz_vip"],
                "ts_code": ["000001.SZ"],
                "bz_item": ["公司银行业务"],
                "end_date": ["20251231"],
                "available_at": ["2026-01-01T18:00:00+08:00"],
            }
        ).to_parquet(root / "fundamentals.parquet", index=False)
    return root


def stock_snapshot(root: Path) -> Path:
    return write_snapshot(
        root,
        [
            (
                "t1",
                "anns_d",
                "000001.SZ",
                "平安银行减持公告",
                "2026-01-01T08:00:00+08:00",
            ),
            ("t2", "news", "", "市场综述", "2026-01-01T09:00:00+08:00"),
            (
                "t3",
                "anns_d",
                "000001.SZ",
                "平安银行利润快报",
                "2026-01-01T10:00:00+08:00",
            ),
            (
                "t5",
                "anns_d",
                "000001.SZ",
                "平安银行历史处罚",
                "2025-01-02T08:00:00+08:00",
            ),
            (
                "t4",
                "anns_d",
                "000001.SZ",
                "平安银行未来公告",
                "2026-02-01T08:00:00+08:00",
            ),
        ],
        {
            "t1": "股东减持不超过 1%",
            "t2": "两市成交放量",
            "t3": "利润同比增长",
            "t4": "未来内容",
            "t5": "历史行政处罚",
        },
        universe=True,
    )


# ---- retrieval and PIT visibility -------------------------------------------


def test_future_evidence_is_invisible_and_a_search_miss_skips_the_model(tmp_path: Path):
    llm = ScriptedLLM([ProviderResponse(content="must not run")])
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)
    result = service.query({"query": "利润", "mode": "search"}, inference_at=NOW)
    assert [item["record_id"] for item in result["evidence"]] == ["t3"]
    assert "t4" not in json.dumps(result)  # available_at is in the future
    assert llm.calls == []  # search mode never reaches the provider
    service.close()


def test_stock_scope_bounds_retrieval_to_company_linked_candidates(tmp_path: Path):
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"))
    scoped = service.query({"query": "减持", "ts_code": "000001.SZ"}, inference_at=NOW)
    assert [item["record_id"] for item in scoped["evidence"]] == ["t1"]
    # The unlinked market row is reachable only without the stock scope.
    broad = service.query({"query": "成交"}, inference_at=NOW)
    assert [item["record_id"] for item in broad["evidence"]] == ["t2"]
    service.close()


def test_company_context_is_live_and_scopes_by_name_as_well_as_code(tmp_path: Path):
    root = stock_snapshot(tmp_path / "snap")
    service = NLService.from_snapshot(root)
    assert service.company_context_store is not None
    context = service.company_context_store.context("000001.SZ")
    assert context["name"] == "平安银行"
    assert context["main_business"] == ["公司银行业务"]
    # The company name is a retrieval term, so a name-only query still scopes.
    named = service.query(
        {"query": "平安银行", "ts_code": "000001.SZ"}, inference_at=NOW
    )
    # Every visible announcement linked to the company, and only those: the
    # unlinked market row and the future row stay out.
    assert {item["record_id"] for item in named["evidence"]} == {"t1", "t3", "t5"}
    service.close()


def test_evidence_revision_is_stable_distinct_and_carries_no_hash(tmp_path: Path):
    root = stock_snapshot(tmp_path / "snap")
    service = NLService.from_snapshot(root)
    service.retriever.as_of = NOW
    first = service.retriever.candidate_evidence_state(
        "000001.SZ", patterns=("减持",), lookback_days=400
    )
    again = service.retriever.candidate_evidence_state(
        "000001.SZ", patterns=("减持",), lookback_days=400
    )
    other = service.retriever.candidate_evidence_state(
        "000001.SZ", patterns=("利润",), lookback_days=400
    )
    assert first.revision == again.revision
    assert first.revision != other.revision
    assert first.match_count == 1
    # Item 5: the identity is the structural row list, never a digest.
    assert first.revision.startswith("rows:")
    assert "sha256" not in first.revision
    service.close()


def test_lookback_days_bounds_the_candidate_window(tmp_path: Path):
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"))
    service.retriever.as_of = NOW
    near = service.retriever.candidate_evidence_state(
        "000001.SZ", patterns=("处罚",), lookback_days=400
    )
    far = service.retriever.candidate_evidence_state(
        "000001.SZ", patterns=("处罚",), lookback_days=30
    )
    assert near.match_count == 1
    assert far.match_count == 0
    service.close()


def test_pattern_must_stay_inside_the_re2_contract(tmp_path: Path):
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"))
    service.retriever.as_of = NOW
    with pytest.raises(ValueError, match="RE2"):
        service.retriever.candidate_evidence_state(
            "000001.SZ", patterns=("(?<=a)b",), lookback_days=30
        )
    service.close()


# ---- the Sub Agent ----------------------------------------------------------


def test_sub_agent_answers_over_pit_evidence_and_may_retrieve_for_itself(
    tmp_path: Path,
):
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "r", "text_retrieve", {"pattern": "减持", "max_results": 3}
                    ),
                )
            ),
            ProviderResponse(content="利润改善，但存在减持压力。"),
        ]
    )
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)
    result = service.query(
        {"query": "利润", "mode": "answer", "ts_code": "000001.SZ"}, inference_at=NOW
    )
    assert result["status"] == "ok"
    assert result["answer"] == "利润改善，但存在减持压力。"
    task = result["task"]
    assert task["state"] == "completed"
    assert task["rounds"] >= 1
    # The Sub Agent's own retrieval widened the evidence beyond the prefetch,
    # and every row it can see is PIT-visible.
    seen = {item["text_id"] for item in task["evidence"]}
    assert "t1" in seen
    assert "t4" not in seen
    assert len(llm.calls) == 2
    service.close()


def test_sub_agent_replays_reasoning_with_tool_call_on_next_round(tmp_path: Path):
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "r",
                        "text_retrieve",
                        {"pattern": "减持", "max_results": 3},
                    ),
                ),
                reasoning_content="先检索减持证据",
            ),
            ProviderResponse(content="存在减持压力。"),
        ]
    )
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)

    result = service.query(
        {"query": "利润", "mode": "answer", "ts_code": "000001.SZ"},
        inference_at=NOW,
    )

    assert result["status"] == "ok"
    assistant = next(
        message for message in llm.calls[1]["messages"] if message.role == "assistant"
    )
    assert assistant.reasoning_content == "先检索减持证据"
    service.close()


def test_tool_round_budget_ends_with_one_final_no_tool_turn(tmp_path: Path):
    retrieve = ToolCall("r", "text_retrieve", {"pattern": "利润"})
    llm = ScriptedLLM(
        [
            ProviderResponse(tool_calls=(retrieve,)),
            ProviderResponse(tool_calls=(retrieve,)),
            ProviderResponse(content="预算用尽后的结论。"),
        ]
    )
    service = NLService.from_snapshot(
        stock_snapshot(tmp_path / "snap"), llm=llm, config=NLConfig(max_llm_rounds=2)
    )
    result = service.query(
        {"query": "利润", "mode": "answer", "ts_code": "000001.SZ"}, inference_at=NOW
    )
    assert result["status"] == "ok"
    assert result["task"]["rounds"] == 2
    # Exhausting the budget sends a final turn with no tools offered, rather
    # than raising: the model still gets to answer from what it has.
    assert llm.calls[-1]["tool_choice"] == "none"
    final_prompt = llm.calls[-1]["messages"][-1].content or ""
    assert "budget for this NL Sub Agent task is exhausted" in final_prompt
    service.close()


def test_enum_response_contract_constrains_the_answer(tmp_path: Path):
    llm = ScriptedLLM([ProviderResponse(content="判定：偏空")])
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)
    result = service.query(
        {
            "query": "减持影响",
            "mode": "answer",
            "ts_code": "000001.SZ",
            "response_format": {"type": "enum", "choices": ["偏多", "偏空", "中性"]},
        },
        inference_at=NOW,
    )
    assert result["status"] == "ok"
    assert result["answer"] == "偏空"
    service.close()


@pytest.mark.parametrize(
    ("policy", "expects_raise"),
    [("return_error_with_audit", False), ("fail", True)],
)
def test_failure_policy_decides_between_an_audit_record_and_a_raise(
    tmp_path: Path, policy: str, expects_raise: bool
):
    llm = ScriptedLLM([ProviderResponse(content="不在候选内")])
    service = NLService.from_snapshot(
        stock_snapshot(tmp_path / "snap"), llm=llm, failure_policy=policy
    )
    request = {
        "query": "减持影响",
        "mode": "answer",
        "ts_code": "000001.SZ",
        "response_format": {"type": "enum", "choices": ["偏多", "偏空"]},
    }
    if expects_raise:
        with pytest.raises(RuntimeError, match="NL sub agent failed"):
            service.query(request, inference_at=NOW)
    else:
        result = service.query(request, inference_at=NOW)
        assert result["status"] == "error"
        assert result["task"]["state"] == "failed_with_policy"
        assert result["task"]["error"]
    service.close()


def test_failure_policy_is_validated_at_construction(tmp_path: Path):
    with pytest.raises(ValueError, match="failure_policy"):
        NLService.from_snapshot(
            stock_snapshot(tmp_path / "snap"), failure_policy="explode"
        )


# ---- the declared evidence predicate (event_filter) --------------------------


def test_event_filter_gate_skips_the_model_when_nothing_matches(tmp_path: Path):
    llm = ScriptedLLM([ProviderResponse(content="must not run")])
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)
    result = service.query(
        {
            "query": "是否有减持",
            "mode": "answer",
            "ts_code": "000001.SZ",
            "event_filter": {"patterns": ["回购注销"], "lookback_days": 400},
        },
        inference_at=NOW,
    )
    assert result["status"] == "no_matching_evidence"
    assert result["event_filter"]["matching_evidence_count"] == 0
    assert llm.calls == []
    assert service.no_evidence_skips == 1
    assert service.event_filter_calls == 1
    service.close()


def test_event_filter_match_prefetches_the_rows_for_the_sub_agent(tmp_path: Path):
    llm = ScriptedLLM([ProviderResponse(content="存在减持。")])
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)
    result = service.query(
        {
            "query": "是否有减持",
            "mode": "answer",
            "ts_code": "000001.SZ",
            "event_filter": {"patterns": ["减持"], "lookback_days": 400},
        },
        inference_at=NOW,
    )
    assert result["status"] == "ok"
    assert result["event_filter"]["matching_evidence_count"] == 1
    assert result["event_filter"]["evidence_revision"].startswith("rows:")
    assert len(llm.calls) == 1
    service.close()


def test_event_filter_lookback_excludes_an_older_match(tmp_path: Path):
    llm = ScriptedLLM([ProviderResponse(content="must not run")])
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)
    result = service.query(
        {
            "query": "是否有减持",
            "mode": "answer",
            "ts_code": "000001.SZ",
            "event_filter": {"patterns": ["处罚"], "lookback_days": 30},
        },
        inference_at=NOW,
    )
    assert result["status"] == "no_matching_evidence"
    assert llm.calls == []
    service.close()


def test_a_query_without_a_predicate_still_lets_the_sub_agent_search(tmp_path: Path):
    # The gate belongs to the declared predicate. Without one, a literal-text
    # miss must NOT short-circuit — the Sub Agent retrieves for itself.
    llm = ScriptedLLM(
        [
            ProviderResponse(
                tool_calls=(ToolCall("r", "text_retrieve", {"pattern": "减持"}),)
            ),
            ProviderResponse(content="找到了减持公告。"),
        ]
    )
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)
    result = service.query(
        {"query": "毫无匹配的字面问题", "mode": "answer", "ts_code": "000001.SZ"},
        inference_at=NOW,
    )
    assert result["status"] == "ok"
    assert result["answer"] == "找到了减持公告。"
    assert {item["text_id"] for item in result["task"]["evidence"]} == {"t1"}
    service.close()


def test_event_filter_validation_rejects_every_malformed_shape(tmp_path: Path):
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"))
    base = {"query": "x", "mode": "answer", "ts_code": "000001.SZ"}
    cases = [
        (
            {
                **base,
                "ts_code": "",
                "event_filter": {"patterns": ["a"], "lookback_days": 30},
            },
            "stock-scoped",
        ),
        ({**base, "event_filter": {"patterns": [], "lookback_days": 30}}, "1 to 16"),
        (
            {**base, "event_filter": {"patterns": ["a"], "lookback_days": 0}},
            "between 1 and 3660",
        ),
        (
            {**base, "event_filter": {"patterns": ["(?<=a)b"], "lookback_days": 30}},
            "RE2",
        ),
        (
            {
                **base,
                "event_filter": {"patterns": ["a"], "lookback_days": 30, "extra": 1},
            },
            "unsupported fields",
        ),
        (
            {
                **base,
                "mode": "search",
                "event_filter": {"patterns": ["a"], "lookback_days": 30},
            },
            "mode=answer",
        ),
    ]
    for request, message in cases:
        with (
            pytest.subTests(request=request)
            if hasattr(pytest, "subTests")
            else _noop(),
            pytest.raises(ValueError, match=message),
        ):
            service.query(request, inference_at=NOW)
    service.close()


class _noop:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ---- request validation and budgets -----------------------------------------


def test_nl_request_rejects_unknown_mode_field_and_oversized_limit(tmp_path: Path):
    service = NLService.from_snapshot(
        write_snapshot(tmp_path / "snap", []), config=NLConfig(max_results=2)
    )
    with pytest.raises(ValueError, match="mode"):
        service.query({"query": "x", "mode": "trade"}, inference_at=NOW)
    with pytest.raises(ValueError, match="1 to 2"):
        service.query({"query": "x", "limit": 3}, inference_at=NOW)
    with pytest.raises(ValueError, match="unknown NL request fields"):
        service.query({"query": "x", "depth": 2}, inference_at=NOW)
    with pytest.raises(ValueError, match="non-empty string"):
        service.query({"query": "  "}, inference_at=NOW)
    service.close()


def test_nl_call_budget_is_per_decision_and_total(tmp_path: Path):
    service = NLService.from_snapshot(
        stock_snapshot(tmp_path / "snap"),
        config=NLConfig(max_calls_per_decision=1, max_total_calls=2),
    )
    later = datetime(2026, 1, 3, tzinfo=UTC)
    assert service.query({"query": "利润"}, inference_at=NOW)["status"] in {
        "ok",
        "no_evidence",
    }
    with pytest.raises(RuntimeError, match="nl_max_calls_per_decision"):
        service.query({"query": "利润"}, inference_at=NOW)
    service.query({"query": "利润"}, inference_at=later)
    # An exhausted per-backtest budget is an explicit error naming the knob,
    # never a silently skipped call that reads as "no evidence".
    with pytest.raises(RuntimeError, match="nl_max_total_calls=2"):
        service.query({"query": "利润"}, inference_at=datetime(2026, 1, 4, tzinfo=UTC))
    counters = service.counters()
    assert counters["nl_calls"] == 2
    assert counters["nl_max_total_calls"] == 2
    # Both refusals are recorded, so a strategy that swallows the exception
    # still leaves the exhausted budget visible in the backtest summary.
    assert counters["nl_budget_rejected_calls"] == 2
    service.close()


def test_nl_total_call_budget_scales_with_the_replay_it_runs():
    # NL is the only real LLM inference inside a replay's wall clock, so the
    # ceiling must bound it — derived from the replay length, not a constant.
    assert NLConfig().max_total_calls is None
    long_window = NLConfig().for_replay(242)
    assert long_window.max_total_calls == NL_CALLS_PER_TRADING_DAY * 242
    short_window = NLConfig().for_replay(61)
    assert short_window.max_total_calls == NL_CALLS_PER_TRADING_DAY * 61
    assert short_window.max_total_calls < long_window.max_total_calls
    # A very short replay still affords one full decision's per-decision peak.
    assert NLConfig().for_replay(1).max_total_calls == NLConfig().max_calls_per_decision
    # An explicit ceiling is the operator's own and is never re-derived.
    assert NLConfig(max_total_calls=7).for_replay(242).max_total_calls == 7
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError, match="trading_days"):
            NLConfig().for_replay(bad)  # type: ignore[arg-type]


def test_nl_counters_separate_gated_calls_from_calls_that_reached_the_model(
    tmp_path: Path,
):
    llm = ScriptedLLM([ProviderResponse(content="减持")])
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)
    gated = service.query(
        {
            "query": "有减持吗",
            "mode": "answer",
            "ts_code": "000001.SZ",
            "event_filter": {"patterns": ["从未出现的事件"], "lookback_days": 30},
        },
        inference_at=NOW,
    )
    assert gated["status"] == "no_matching_evidence"
    answered = service.query(
        {"query": "减持规模", "mode": "answer", "ts_code": "000001.SZ"},
        inference_at=NOW,
    )
    assert answered["status"] == "ok"

    counters = service.counters()
    assert counters["nl_calls"] == 2
    assert counters["nl_event_filter_calls"] == 1
    assert counters["nl_evidence_gated_calls"] == 1
    # Only the ungated call reached the Sub Agent, and it cost one LLM round.
    assert counters["nl_executed_calls"] == 1
    assert counters["nl_llm_calls"] == 1
    assert counters["nl_wall_seconds"] >= 0.0
    service.close()


def test_every_charged_nl_call_lands_in_exactly_one_outcome_bucket(tmp_path: Path):
    """``nl_calls`` must be explained, not just counted.

    An unexplained residue is what makes the summary unreadable: a rejected
    request that still charged the budget, or a retrieval-only call no bucket
    mentions, both make the arithmetic silently wrong.
    """
    llm = ScriptedLLM([ProviderResponse(content="减持")])
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"), llm=llm)

    # Invalid request: refused before the budget is charged, so it leaves nothing.
    with pytest.raises(ValueError, match="mode"):
        service.query({"query": "利润", "mode": "trade"}, inference_at=NOW)
    assert service.counters()["nl_calls"] == 0

    service.query({"query": "利润", "mode": "search"}, inference_at=NOW)
    service.query({"query": "不存在的词", "mode": "search"}, inference_at=NOW)
    service.query(
        {
            "query": "有减持吗",
            "mode": "answer",
            "ts_code": "000001.SZ",
            "event_filter": {"patterns": ["从未出现的事件"], "lookback_days": 30},
        },
        inference_at=NOW,
    )
    service.query(
        {"query": "减持规模", "mode": "answer", "ts_code": "000001.SZ"},
        inference_at=NOW,
    )

    counters = service.counters()
    # A search hit and a search miss are both retrieval-only outcomes.
    assert counters["nl_search_calls"] == 2
    assert counters["nl_evidence_gated_calls"] == 1
    assert counters["nl_executed_calls"] == 1
    assert counters["nl_calls"] == 4
    assert counters["nl_calls"] == (
        counters["nl_executed_calls"]
        + counters["nl_search_calls"]
        + counters["nl_evidence_gated_calls"]
    )
    service.close()


def test_answer_without_a_configured_model_counts_as_retrieval_only(tmp_path: Path):
    service = NLService.from_snapshot(stock_snapshot(tmp_path / "snap"))
    service.query({"query": "利润", "mode": "answer"}, inference_at=NOW)
    counters = service.counters()
    assert counters["nl_executed_calls"] == 0
    assert counters["nl_search_calls"] == 1
    assert counters["nl_calls"] == 1
    service.close()


def test_snapshot_index_without_available_at_fails_closed(tmp_path: Path):
    root = tmp_path / "snap"
    root.mkdir()
    pd.DataFrame(
        {
            "text_id": ["1"],
            "dataset": ["news"],
            "title": ["利润"],
            "ts_codes": [""],
            "library_file": ["news.parquet"],
        }
    ).to_parquet(root / "text_index.parquet", index=False)
    service = NLService.from_snapshot(root)
    with pytest.raises(ValueError, match="available_at"):
        service.query({"query": "利润"}, inference_at=NOW)
    service.close()


def test_snapshot_timeview_directory_refreshes_and_enforces_evidence_budgets(
    tmp_path: Path,
):
    index_dir = tmp_path / "text_index"
    library_dir = tmp_path / "text_library"
    index_dir.mkdir()
    library_dir.mkdir()
    service = NLService.from_snapshot(
        tmp_path,
        config=NLConfig(max_results=3, max_record_chars=4, max_total_evidence_chars=6),
    )
    assert service.query({"query": "利润"}, inference_at=NOW)["status"] == "no_evidence"

    pd.DataFrame(
        {
            "text_id": ["older", "newer"],
            "dataset": ["news", "news"],
            "title": ["利润旧闻", "利润新讯"],
            "ts_codes": ["", ""],
            "available_at": ["2026-01-01T08:00:00+08:00", "2026-01-02T07:00:00+08:00"],
            "library_file": ["news.parquet", "news.parquet"],
        }
    ).to_parquet(index_dir / "part_0000.parquet", index=False)
    pd.DataFrame(
        {
            "text_id": ["older", "newer"],
            "body": ["戊己庚辛壬癸", "甲乙丙丁子丑"],
        }
    ).to_parquet(library_dir / "news.parquet", index=False)

    result = service.query({"query": "利润"}, inference_at=NOW)
    texts = [item["text"] for item in result["evidence"]]
    assert texts == ["甲乙丙丁", "戊己"]
    assert all(len(text) <= 4 for text in texts)
    assert sum(map(len, texts)) == 6
    service.close()


def test_snapshot_library_file_cannot_escape_library_root(tmp_path: Path):
    root = tmp_path / "snap"
    root.mkdir()
    pd.DataFrame(
        {
            "text_id": ["escape"],
            "dataset": ["news"],
            "title": ["利润"],
            "ts_codes": [""],
            "available_at": ["2026-01-01T08:00:00+08:00"],
            "library_file": ["../outside.parquet"],
        }
    ).to_parquet(root / "text_index.parquet", index=False)
    service = NLService.from_snapshot(root)
    with pytest.raises(ValueError, match="invalid text library file"):
        service.query({"query": "利润"}, inference_at=NOW)
    service.close()


def test_nl_config_rejects_non_integer_and_non_finite_budgets():
    with pytest.raises(ValueError, match="integer budgets"):
        NLConfig(max_results=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        NLConfig(deadline_seconds=float("nan"))


def test_service_requires_a_retriever():
    with pytest.raises(ValueError, match="retriever"):
        NLService(None)  # type: ignore[arg-type]
