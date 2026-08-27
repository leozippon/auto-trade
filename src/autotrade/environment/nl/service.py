"""Bounded NL reasoning over an already PIT-filtered local evidence set."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from autotrade.environment.llm import LLMProxy

from .context import CompanyContextStore
from .engine import (
    ENUM_MAX_RESULTS,
    ENUM_MAX_TOKENS,
    ENUM_SNIPPET_CHARS,
    NLSubAgentConfig,
    NLSubAgentEngine,
    company_terms,
)
from .retrieval import TextRetriever, validate_pattern

NLMode = Literal["search", "answer"]

# Per-backtest NL call ceiling. NL is the only real LLM inference inside a
# backtest's wall clock, so an unbounded total budget lets one strategy turn an
# official backtest into an LLM latency test. Two calls per decision day over a
# quarterly (~61 trading day) validation window keeps the 10-call per-decision
# peak usable on the days that need it while bounding worst-case NL wall to
# roughly 120 x nl_deadline_seconds.
DEFAULT_MAX_TOTAL_CALLS = 120


@dataclass(frozen=True)
class _EventFilter:
    patterns: tuple[str, ...]
    lookback_days: int


@dataclass(frozen=True)
class NLConfig:
    max_results: int = 8
    max_query_chars: int = 2000
    max_record_chars: int = 4000
    max_total_evidence_chars: int = 16_000
    max_llm_rounds: int = 3
    deadline_seconds: float = 20.0
    max_tokens: int = 1200
    max_calls_per_decision: int = 10
    # None keeps the per-decision cap as the only bound; the shipped default is
    # a real ceiling (see DEFAULT_MAX_TOTAL_CALLS).
    max_total_calls: int | None = DEFAULT_MAX_TOTAL_CALLS

    def __post_init__(self) -> None:
        values = (
            self.max_results,
            self.max_query_chars,
            self.max_record_chars,
            self.max_total_evidence_chars,
            self.max_llm_rounds,
            self.max_tokens,
            self.max_calls_per_decision,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in values
        ):
            raise ValueError("NL integer budgets must be positive")
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not math.isfinite(self.deadline_seconds)
            or self.deadline_seconds <= 0
        ):
            raise ValueError("NL deadline_seconds must be a positive finite number")
        if self.max_total_calls is not None and (
            isinstance(self.max_total_calls, bool)
            or not isinstance(self.max_total_calls, int)
            or self.max_total_calls <= 0
        ):
            raise ValueError("max_total_calls must be a positive integer or None")


@dataclass(frozen=True)
class NLResult:
    status: Literal["ok", "no_evidence", "no_matching_evidence", "error"]
    query: str
    mode: NLMode
    evidence: tuple[Mapping[str, object], ...]
    answer: str = ""
    task: Mapping[str, object] | None = None
    # Only set when an event_filter declared the validity predicate.
    evidence_revision: str | None = None
    matching_evidence_count: int | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "query": self.query,
            "mode": self.mode,
            "answer": self.answer,
            "evidence": [dict(item) for item in self.evidence],
        }
        if self.task is not None:
            record["task"] = dict(self.task)
        if self.evidence_revision is not None:
            record["event_filter"] = {
                "evidence_revision": self.evidence_revision,
                "matching_evidence_count": self.matching_evidence_count,
            }
        return record


class NLService:
    """The host side of ``ctx.nl()``: PIT retrieval plus one bounded Sub Agent.

    Retrieval is the single :class:`TextRetriever`; the answer path is the
    single :class:`NLSubAgentEngine`, which may run further ``text_retrieve``
    rounds of its own over the same retriever. Company context comes from the
    memoized :class:`CompanyContextStore` and both scopes the candidate
    evidence and is handed to the Sub Agent.
    """

    def __init__(
        self,
        retriever: TextRetriever,
        *,
        llm: LLMProxy | None = None,
        config: NLConfig | None = None,
        company_context_store: CompanyContextStore | None = None,
        failure_policy: str = "return_error_with_audit",
    ) -> None:
        if retriever is None:
            raise ValueError("NL service requires a snapshot text retriever")
        if failure_policy not in {"fail", "return_error_with_audit"}:
            raise ValueError(f"unsupported failure_policy={failure_policy}")
        self.retriever = retriever
        self.llm = llm
        self.config = config or NLConfig()
        self.company_context_store = company_context_store
        self.failure_policy = failure_policy
        self.engine = (
            NLSubAgentEngine(llm, retriever) if llm is not None else None
        )
        self.calls = 0
        self.event_filter_calls = 0
        self.no_evidence_skips = 0
        # Cost accounting surfaced in the backtest summary: NL is the only real
        # LLM inference inside a backtest, so its calls and wall clock are the
        # first thing a slow backtest has to be explained by.
        self.executed_calls = 0
        self.search_calls = 0
        self.llm_calls = 0
        self.budget_rejected_calls = 0
        self.wall_seconds = 0.0
        self._calls_by_decision: dict[str, int] = {}

    def counters(self) -> dict[str, object]:
        """NL cost accounting for one backtest, merged into its summary.

        Every accepted call lands in exactly one outcome bucket, so
        ``nl_calls == nl_executed_calls + nl_evidence_gated_calls +
        nl_search_calls`` for calls that returned. Rejected requests never reach
        a bucket: an invalid request is refused before the budget is charged, and
        an exhausted budget is counted only in ``nl_budget_rejected_calls``.
        ``nl_event_filter_calls`` cuts across the buckets — it counts the calls
        that declared an evidence predicate, gated or not.
        """
        return {
            "nl_calls": self.calls,
            "nl_executed_calls": self.executed_calls,
            "nl_search_calls": self.search_calls,
            "nl_llm_calls": self.llm_calls,
            "nl_event_filter_calls": self.event_filter_calls,
            "nl_evidence_gated_calls": self.no_evidence_skips,
            "nl_budget_rejected_calls": self.budget_rejected_calls,
            "nl_wall_seconds": round(self.wall_seconds, 3),
            "nl_max_total_calls": self.config.max_total_calls,
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot_dir,
        *,
        llm: LLMProxy | None = None,
        config: NLConfig | None = None,
        failure_policy: str = "return_error_with_audit",
    ) -> NLService:
        from pathlib import Path

        root = Path(snapshot_dir)
        settings = config or NLConfig()
        index_path = root / "text_index" if (root / "text_index").is_dir() else root / "text_index.parquet"
        return cls(
            TextRetriever(
                index_path,
                root / "text_library",
                snippet_chars=settings.max_record_chars,
            ),
            llm=llm,
            config=settings,
            company_context_store=CompanyContextStore(root),
            failure_policy=failure_policy,
        )

    def close(self) -> None:
        self.retriever.close()

    def query(
        self,
        request: Mapping[str, object],
        *,
        inference_at: datetime,
    ) -> dict[str, object]:
        started = time.perf_counter()
        try:
            return self._query(request, inference_at=inference_at)
        finally:
            self.wall_seconds += time.perf_counter() - started

    def _query(
        self,
        request: Mapping[str, object],
        *,
        inference_at: datetime,
    ) -> dict[str, object]:
        # Validate first: an unusable request must not consume a call budget or
        # leave an accepted-call residue that no outcome bucket explains.
        query, mode, limit, ts_code, choices, event_filter = self._validate_request(request)
        self._consume_call_budget(inference_at)
        context: dict[str, object] = {}
        terms: list[str] = []
        if ts_code and self.company_context_store is not None:
            context = dict(self.company_context_store.context(ts_code))
            terms = company_terms(context, ts_code)
        self.retriever.as_of = inference_at
        if event_filter is not None:
            # A declared validity predicate is the gate: when nothing inside the
            # rolling window matches it, the model is not called at all. It is
            # also the prefetch for an enum request, which needs no exploration.
            state = self.retriever.candidate_evidence_state(
                ts_code,
                company_terms=terms or None,
                patterns=event_filter.patterns,
                lookback_days=event_filter.lookback_days,
                max_results=ENUM_MAX_RESULTS if choices else 0,
            )
            self.event_filter_calls += 1
            if state.match_count == 0:
                self.no_evidence_skips += 1
                return NLResult(
                    "no_matching_evidence",
                    query,
                    mode,
                    (),
                    evidence_revision=state.revision,
                    matching_evidence_count=0,
                ).to_record()
            rows = list(state.evidence)
            evidence = self._snapshot_evidence(rows)
            return self._answer(
                query,
                mode,
                evidence,
                prefetched=rows,
                ts_code=ts_code,
                context=context,
                terms=terms,
                choices=choices,
                evidence_revision=state.revision,
                matching_evidence_count=state.match_count,
            )
        rows = self.retriever.search(
            query,
            ts_code=ts_code,
            max_results=limit,
            company_terms=terms or None,
        )
        evidence = self._snapshot_evidence(rows)
        if mode == "search" or self.engine is None:
            # Retrieval-only outcome: no model was called, with or without a hit.
            self.search_calls += 1
            if not evidence:
                return NLResult("no_evidence", query, mode, ()).to_record()
            return NLResult("ok", query, mode, evidence).to_record()
        # No declared predicate: the Sub Agent still runs and may retrieve for
        # itself, so a prompt whose literal text matches nothing is not a dead
        # end. Only a declared event_filter skips the model.
        return self._answer(
            query,
            mode,
            evidence,
            prefetched=rows,
            ts_code=ts_code,
            context=context,
            terms=terms,
            choices=choices,
        )

    def _consume_call_budget(self, inference_at: datetime) -> None:
        """Fail closed on an exhausted budget; never downgrade to a silent skip.

        The refusal is counted so a strategy that swallows the error still
        leaves the exhausted budget visible in the backtest summary.
        """
        key = inference_at.isoformat()
        used = self._calls_by_decision.get(key, 0)
        if used >= self.config.max_calls_per_decision:
            self.budget_rejected_calls += 1
            raise RuntimeError(
                f"NL call budget exhausted for decision {key} "
                f"(nl_max_calls_per_decision={self.config.max_calls_per_decision})"
            )
        if self.config.max_total_calls is not None and self.calls >= self.config.max_total_calls:
            self.budget_rejected_calls += 1
            raise RuntimeError(
                "NL call budget exhausted for this backtest "
                f"(nl_max_total_calls={self.config.max_total_calls})"
            )
        self._calls_by_decision[key] = used + 1
        self.calls += 1

    def _validate_request(
        self, request: Mapping[str, object]
    ) -> tuple[str, NLMode, int, str, tuple[str, ...], "_EventFilter | None"]:
        if not isinstance(request, Mapping):
            raise TypeError("NL request must be an object")
        unknown = sorted(
            set(request).difference(
                {"query", "mode", "limit", "ts_code", "response_format", "event_filter"}
            )
        )
        if unknown:
            raise ValueError(f"unknown NL request fields: {unknown}")
        query = request.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("NL query must be a non-empty string")
        query = query.strip()
        if len(query) > self.config.max_query_chars:
            raise ValueError("NL query exceeds the configured character budget")
        mode = request.get("mode", "search")
        if mode not in {"search", "answer"}:
            raise ValueError("NL mode must be search or answer")
        limit = request.get("limit", self.config.max_results)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.config.max_results:
            raise ValueError(f"NL limit must be an integer from 1 to {self.config.max_results}")
        # An optional stock scope bounds retrieval to code/name-linked candidate
        # evidence and gives the Sub Agent that company's PIT context.
        ts_code = request.get("ts_code", "")
        if not isinstance(ts_code, str):
            raise ValueError("NL ts_code must be a string")
        ts_code = ts_code.strip()
        choices = _response_choices(request.get("response_format"))
        event_filter = _parse_event_filter(request.get("event_filter"), ts_code=ts_code)
        if event_filter is not None and mode != "answer":
            raise ValueError("event_filter applies only to mode=answer")
        return query, mode, limit, ts_code, choices, event_filter  # type: ignore[return-value]

    def _snapshot_evidence(
        self,
        rows: list[dict[str, object]],
    ) -> tuple[Mapping[str, object], ...]:
        result: list[Mapping[str, object]] = []
        remaining = self.config.max_total_evidence_chars
        for row in rows:
            if remaining <= 0:
                break
            budget = min(self.config.max_record_chars, remaining)
            text = str(row.get("snippet") or "")[:budget]
            result.append(
                {
                    "evidence_id": f"{row['dataset']}:{row['text_id']}",
                    "source": row["dataset"],
                    "record_id": row["text_id"],
                    "available_at": row["available_at"],
                    "title": row["title"],
                    "symbol": row["ts_codes"],
                    "text": text,
                }
            )
            remaining -= len(text)
        return tuple(result)

    def _answer(
        self,
        query: str,
        mode: NLMode,
        evidence: tuple[Mapping[str, object], ...],
        *,
        prefetched: list[dict[str, object]],
        ts_code: str,
        context: Mapping[str, object],
        terms: list[str],
        choices: tuple[str, ...],
        evidence_revision: str | None = None,
        matching_evidence_count: int | None = None,
    ) -> dict[str, object]:
        """One bounded NL Sub Agent task over the retrieved PIT evidence.

        The evidence already found is handed over as prefetched context; the
        Sub Agent may run further ``text_retrieve`` rounds against the same
        retriever before answering.
        """
        assert self.engine is not None
        config = NLSubAgentConfig(
            max_tokens=ENUM_MAX_TOKENS if choices else self.config.max_tokens,
            max_tool_rounds=self.config.max_llm_rounds,
            failure_policy=self.failure_policy,
            deadline_at=time.monotonic() + self.config.deadline_seconds,
            response_choices=choices,
            max_results_per_search=ENUM_MAX_RESULTS if choices else self.config.max_results,
            max_evidence_snippet_chars=(
                ENUM_SNIPPET_CHARS if choices else self.config.max_record_chars
            ),
        )
        task = self.engine.run(
            ts_code=ts_code,
            prompt=query,
            request_kwargs={"mode": mode, "limit": len(evidence)},
            config=config,
            prefetched_evidence=prefetched,
            company_context=context or None,
            candidate_terms=terms or None,
        )
        # Counted together off the returned task: the sub-agent converts its own
        # failures into an audited result, so a task that never returns leaves
        # neither counter half-incremented.
        self.executed_calls += 1
        self.llm_calls += len(task.llm_calls)
        record = task.to_record()
        if not task.ok:
            if self.failure_policy == "fail":
                raise RuntimeError(f"NL sub agent failed: {task.error or task.state}")
            return NLResult(
                "error", query, mode, evidence, task=record,
                evidence_revision=evidence_revision,
                matching_evidence_count=matching_evidence_count,
            ).to_record()
        return NLResult(
            "ok", query, mode, evidence, task.content, task=record,
            evidence_revision=evidence_revision,
            matching_evidence_count=matching_evidence_count,
        ).to_record()


def _parse_event_filter(raw: object, *, ts_code: str) -> "_EventFilter | None":
    """The strategy's declared evidence-validity predicate for this call."""
    if raw is None:
        return None
    if not ts_code:
        raise ValueError("event_filter is supported only for stock-scoped nl() calls")
    if not isinstance(raw, Mapping):
        raise ValueError("event_filter must be an object with patterns and lookback_days")
    unknown = set(raw) - {"patterns", "lookback_days"}
    if unknown:
        raise ValueError(f"event_filter has unsupported fields: {', '.join(sorted(unknown))}")
    raw_patterns = raw.get("patterns")
    if not isinstance(raw_patterns, list) or not 1 <= len(raw_patterns) <= 16:
        raise ValueError("event_filter.patterns must contain 1 to 16 strings")
    patterns: list[str] = []
    seen: set[str] = set()
    for raw_pattern in raw_patterns:
        if not isinstance(raw_pattern, str):
            raise ValueError("event_filter.patterns must contain only strings")
        pattern = validate_pattern(raw_pattern.strip())
        if pattern not in seen:
            seen.add(pattern)
            patterns.append(pattern)
    if not patterns:
        raise ValueError("event_filter.patterns must contain at least one non-empty pattern")
    # The candidate scan uses one RE2 alternation. Validate the aggregate too so
    # the existing bounded-pattern contract remains true regardless of list size.
    validate_pattern("|".join(f"(?:{pattern})" for pattern in patterns))
    lookback_days = raw.get("lookback_days")
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int):
        raise ValueError("event_filter.lookback_days must be an integer")
    if not 1 <= lookback_days <= 3660:
        raise ValueError("event_filter.lookback_days must be between 1 and 3660")
    return _EventFilter(tuple(patterns), lookback_days)


def _response_choices(value: object) -> tuple[str, ...]:
    """Optional enum answer contract: the Sub Agent must return one listed value."""
    if value is None:
        return ()
    if not isinstance(value, Mapping):
        raise ValueError("NL response_format must be an object")
    unknown = sorted(set(value).difference({"type", "choices"}))
    if unknown:
        raise ValueError(f"unknown NL response_format fields: {unknown}")
    if str(value.get("type", "enum")) != "enum":
        raise ValueError("NL response_format type must be enum")
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("NL response_format requires a non-empty choices array")
    if any(not isinstance(item, str) or not item.strip() for item in choices):
        raise ValueError("NL response_format choices must be non-empty strings")
    return tuple(str(item).strip() for item in choices)


__all__ = [
    "DEFAULT_MAX_TOTAL_CALLS",
    "NLConfig",
    "NLMode",
    "NLResult",
    "NLService",
]
