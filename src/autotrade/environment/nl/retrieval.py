"""Point-in-time text retrieval over the snapshot/as-of text libraries.

Pure data access for the NL Sub Agent: DuckDB/RE2 regex over the text index
and per-dataset body shards (column projection + LIMIT; the multi-GB corpus is
never resident in host memory). No LLM dependency — the agent loop lives in
``nl/engine.py``.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from autotrade.environment.data.pit import to_cn_timestamps

MAX_PATTERN_CHARS = 256
_CANDIDATE_CACHE_SIZE = 128
# One retriever's DuckDB connection, sized for the bounded candidate scans it
# runs: unconfigured, DuckDB takes one thread per host core (192 on the shared
# host, per replay) and 80 % of host memory as its limit.
DUCKDB_THREADS = 4
DUCKDB_MEMORY_LIMIT = "4GB"


@dataclass
class _CandidateCorpus:
    """Static candidate rows plus incrementally loaded PIT-visible bodies."""

    index: pd.DataFrame
    bodies: pd.DataFrame | None = None
    loaded_body_ids: set[tuple[str, str]] = field(default_factory=set)
    # Per-pattern regex verdicts keyed by corpus row label. Rows and bodies are
    # immutable once in the corpus, so a (pattern, row) verdict never changes;
    # the event gate only regexes rows it has never tested (measured: the gate
    # re-scanned the full corpus every ctx.nl() call — 78% of NL wall).
    match_cache: dict[str, dict[object, bool]] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateEvidenceState:
    """Content identity and cardinality for one PIT candidate evidence scope."""

    revision: str
    match_count: int
    evidence: tuple[dict[str, object], ...] = ()


class TextRetriever:
    """Grep-style retrieval over the snapshot text index and as-of text library.

    The NL Sub Agent supplies a regex pattern (case-insensitive grep semantics,
    RE2 engine — linear-time matching, so an adversarial or accidental
    catastrophic-backtracking pattern cannot pin the host CPU; unsupported
    constructs are rejected with a fixable error). Titles/codes are matched
    first, then full bodies when more results are needed. Stock-scoped searches
    stay inside code/name-linked candidate rows; calls without a stock code use
    the broad market corpus.

    Bodies live in per-dataset parquet shards under ``text_library/`` and are
    scanned in place via DuckDB with column projection and result limits — the
    multi-GB corpus is never resident in host memory. A bounded LRU retains the
    PIT-visible body subset linked to recently requested candidates and extends
    it as the decision clock advances. ``as_of`` bounds every read to rows whose
    ``available_at`` has passed; ``as_of`` None keeps the whole index visible.
    A directory index is an as-of view the Timeview appends shards to while a
    backtest runs, so it is re-read whenever its shard set changes.
    """

    def __init__(
        self,
        text_index_path: str | Path,
        text_library_dir: str | Path,
        *,
        snippet_chars: int = 4000,
        as_of: datetime | None = None,
    ) -> None:
        if (
            isinstance(snippet_chars, bool)
            or not isinstance(snippet_chars, int)
            or snippet_chars <= 0
        ):
            raise ValueError("snippet_chars must be a positive integer")
        self.index_path = Path(text_index_path)
        self.library_dir = Path(text_library_dir)
        self.snippet_chars = snippet_chars
        self.as_of = as_of
        self._query_lock = threading.Lock()
        self._connection = duckdb.connect(
            config={"threads": DUCKDB_THREADS, "memory_limit": DUCKDB_MEMORY_LIMIT}
        )
        self._snippets: dict[tuple[str, str], str] = {}
        self._candidate_cache: OrderedDict[tuple[str, ...], _CandidateCorpus] = OrderedDict()
        self._index_signature: tuple[tuple[str, int, int], ...] = ()
        self._load_index()

    def close(self) -> None:
        """Release the persistent DuckDB connection after a replay."""
        with self._query_lock:
            self._connection.close()

    # ---- index ----

    def _load_index(self) -> None:
        self.index = self._read_index()
        self._index_signature = self._shard_signature()
        self._available_at = (
            to_cn_timestamps(self.index["available_at"])
            if not self.index.empty and "available_at" in self.index.columns
            else pd.Series([], dtype="datetime64[ns, Asia/Shanghai]")
        )
        self._datasets = self.index.get("dataset", pd.Series("", index=self.index.index)).astype(str)
        self._codes = self.index.get("ts_codes", pd.Series("", index=self.index.index)).fillna("").astype(str)
        linked = self._codes[self._codes.ne("")]
        self._code_rows = {
            str(code): pd.Index(rows)
            for code, rows in linked.groupby(linked, sort=False).groups.items()
        }
        # Candidate scoping and body caches are keyed by row label, so a
        # re-read invalidates them rather than mixing two index generations.
        self._candidate_cache.clear()
        self._candidate_titles = pd.DataFrame(
            {
                "row": self.index.index,
                "title": self.index.get("title", pd.Series("", index=self.index.index)).astype(str),
            }
        )
        with self._query_lock:
            self._connection.register("candidate_titles", self._candidate_titles)

    def _read_index(self) -> pd.DataFrame:
        if self.index_path.is_file():
            return pd.read_parquet(self.index_path)
        if not self.index_path.is_dir() or not any(self.index_path.glob("*.parquet")):
            return pd.DataFrame()
        return self._connection.execute(
            "SELECT * FROM read_parquet(?, union_by_name = true)",
            [str(self.index_path / "*.parquet")],
        ).fetchdf()

    def _shard_signature(self) -> tuple[tuple[str, int, int], ...]:
        """Identity of the index files on disk, so a directory index is re-read
        only when the Timeview has actually appended or replaced a shard."""
        if not self.index_path.is_dir():
            return ()
        signature: list[tuple[str, int, int]] = []
        for path in sorted(self.index_path.glob("*.parquet")):
            stat = path.stat()
            signature.append((path.name, stat.st_mtime_ns, stat.st_size))
        return tuple(signature)

    def _index_source_exists(self) -> bool:
        return self.index_path.is_file() or (
            self.index_path.is_dir() and any(self.index_path.glob("*.parquet"))
        )

    def _refresh_index(self) -> None:
        """Re-read a directory index the Timeview has appended shards to.

        Every entry point that reads the index runs this first: a retriever
        lives for the whole replay, so an entry point that skipped it would
        keep answering from the shard set of the first ``ctx.nl()`` call and
        silently lose the evidence that became visible afterwards. Re-reading
        clears the candidate caches, so no answer mixes two index generations.
        """
        if self.index_path.is_dir() and self._shard_signature() != self._index_signature:
            self._load_index()

    def visible_index(self) -> pd.DataFrame:
        """Index rows whose ``available_at`` has passed at ``self.as_of``."""
        self._refresh_index()
        if self._index_source_exists() and "available_at" not in self.index.columns:
            raise ValueError(f"text index has no available_at column: {self.index_path}")
        if self.index.empty or self.as_of is None:
            return self.index
        return self.index[self._available_at <= self._anchor()]

    def _anchor(self) -> pd.Timestamp:
        cutoff = pd.Timestamp(self.as_of)
        return (
            cutoff.tz_localize("Asia/Shanghai")
            if cutoff.tzinfo is None
            else cutoff.tz_convert("Asia/Shanghai")
        )

    # ---- retrieval ----

    def search(
        self,
        pattern: str,
        *,
        ts_code: str = "",
        max_results: int = 5,
        search_bodies: bool = True,
        company_terms: list[str] | None = None,
        lookback_days: int | None = None,
    ) -> list[dict[str, object]]:
        """Raises ValueError for patterns outside the RE2/grep contract."""
        regex = validate_pattern(pattern)
        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or max_results <= 0
        ):
            raise ValueError("max_results must be a positive integer")
        visible = self.visible_index()
        candidate_key = self._candidate_key(ts_code, company_terms)
        corpus = self._candidate_corpus(candidate_key) if candidate_key else None
        index = (
            self._visible_candidate_index(corpus, lookback_days=lookback_days)
            if corpus is not None
            else visible
        )
        if index.empty:
            return []
        pattern_hit = self._title_code_match(index, regex)
        hits = index[pattern_hit].copy()
        hits["_relevance"] = "candidate" if corpus is not None else "background"
        hits["_rank"] = 40 if corpus is not None else 20
        if search_bodies and len(hits) < max_results:
            body_idx = (
                self._grep_candidate_bodies(
                    corpus,
                    index,
                    regex,
                    exclude=set(hits["text_id"].astype(str)),
                    limit=max_results * 3,
                )
                if corpus is not None
                else self._grep_bodies(
                    index,
                    regex,
                    exclude=set(hits["text_id"].astype(str)),
                    limit=max_results * 3,
                )
            )
            if body_idx:
                body_rows = index[index["text_id"].astype(str).isin(body_idx)].copy()
                body_rows["_relevance"] = "candidate" if corpus is not None else "background"
                body_rows["_rank"] = 30 if corpus is not None else 10
                hits = pd.concat([hits, body_rows], ignore_index=False)
        if hits.empty:
            return []
        hits = hits.drop_duplicates(subset=["text_id"], keep="first")
        sort_cols = ["_rank"] + (["available_at"] if "available_at" in hits.columns else [])
        hits = hits.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        selected = hits.head(max_results)
        if corpus is not None and search_bodies:
            # The first stock query commonly hits titles, followed by several body
            # patterns. Prime the currently visible candidate rows once so later
            # regex rounds do not rescan the multi-GB shards; newly visible rows
            # are appended on demand as the decision clock advances.
            self._candidate_bodies(corpus, index)
        self._prime_snippets(selected)
        return self._evidence_records(selected)

    def candidate_evidence_state(
        self,
        ts_code: str,
        *,
        company_terms: list[str] | None = None,
        patterns: tuple[str, ...] = (),
        lookback_days: int | None = None,
        max_results: int = 0,
    ) -> CandidateEvidenceState:
        """Identify and count the matching evidence visible inside a rolling PIT window.

        ``revision`` is the canonical listing of the matching rows themselves,
        not a digest of them: it is only ever compared for equality against a
        cached value, and stating the rows outright keeps the identity readable
        in an audit instead of collapsing it into an opaque stamp.
        """
        if not str(ts_code or "").strip():
            raise ValueError("candidate_evidence_state requires a non-empty ts_code")
        self._refresh_index()
        key = self._candidate_key(ts_code, company_terms)
        corpus = self._candidate_corpus(key)
        visible = self._visible_candidate_index(corpus, lookback_days=lookback_days)
        if patterns and not visible.empty:
            visible = self._matching_candidate_rows(corpus, visible, patterns)
        columns = [c for c in ("dataset", "text_id", "available_at", "title") if c in visible]
        rows: list[list[str]] = []
        if columns:
            ordered = visible[columns].fillna("").astype(str).sort_values(columns)
            rows = [list(row) for row in ordered.itertuples(index=False, name=None)]
        evidence: tuple[dict[str, object], ...] = ()
        if max_results > 0 and not visible.empty:
            selected = visible.copy()
            selected["_relevance"] = "candidate"
            selected["_rank"] = 40
            sort_cols = ["_rank"] + (["available_at"] if "available_at" in selected.columns else [])
            selected = selected.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(max_results)
            self._prime_snippets(selected)
            evidence = tuple(self._evidence_records(selected))
        return CandidateEvidenceState(
            revision="rows:" + json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
            match_count=int(len(visible)),
            evidence=evidence,
        )

    def _evidence_records(self, rows: pd.DataFrame) -> list[dict[str, object]]:
        return [
            {
                "text_id": str(row.get("text_id", "")),
                "dataset": str(row.get("dataset", "")),
                "title": str(row.get("title", "")),
                "available_at": str(row.get("available_at", "")),
                "ts_codes": str(row.get("ts_codes", "")),
                "relevance": str(row.get("_relevance", "background")),
                "snippet": self._snippet(str(row.get("dataset", "")), str(row.get("text_id", ""))),
            }
            for row in rows.to_dict("records")
        ]

    def _matching_candidate_rows(
        self,
        corpus: _CandidateCorpus,
        visible: pd.DataFrame,
        patterns: tuple[str, ...],
    ) -> pd.DataFrame:
        valid = tuple(validate_pattern(pattern) for pattern in patterns)
        combined = "|".join(f"(?:{pattern})" for pattern in valid)
        # Bound distinct pattern keys: a strategy that varies its pattern per
        # call would otherwise grow one verdict dict per pattern for the
        # corpus's lifetime (FIFO eviction; normal strategies reuse one gate).
        if combined not in corpus.match_cache and len(corpus.match_cache) >= 32:
            corpus.match_cache.pop(next(iter(corpus.match_cache)))
        cache = corpus.match_cache.setdefault(combined, {})
        pending = visible.loc[[label for label in visible.index if label not in cache]]
        if not pending.empty:
            matched = self._title_code_match(pending, combined)
            bodies = self._candidate_bodies(corpus, pending)
            body_match_ids: set[str] = set()
            if not bodies.empty:
                pending_ids = set(pending["text_id"].astype(str))
                body_frame = bodies[bodies["text_id"].isin(pending_ids)]
                if not body_frame.empty:
                    with self._query_lock:
                        try:
                            self._connection.register("candidate_bodies", body_frame)
                            body_ids = self._connection.execute(
                                "SELECT DISTINCT text_id FROM candidate_bodies "
                                "WHERE regexp_matches(body, ?, 'i')",
                                [combined],
                            ).fetchall()
                        except duckdb.Error as exc:
                            raise ValueError(
                                f"text_retrieve body query failed (RE2/grep semantics): {exc}"
                            ) from exc
                        finally:
                            self._connection.unregister("candidate_bodies")
                    body_match_ids = {str(row[0]) for row in body_ids}
            verdict = matched | pending["text_id"].astype(str).isin(body_match_ids)
            for label, value in verdict.items():
                cache[label] = bool(value)
        return visible[pd.Series([cache[label] for label in visible.index], index=visible.index, dtype=bool)]

    def _candidate_key(self, ts_code: str, company_terms: list[str] | None) -> tuple[str, ...]:
        return tuple(_candidate_terms(ts_code, company_terms))

    def _candidate_corpus(self, key: tuple[str, ...]) -> _CandidateCorpus:
        cached = self._candidate_cache.pop(key, None)
        if cached is not None:
            self._candidate_cache[key] = cached
            return cached
        code = key[0]
        code_rows = self._code_rows.get(code, pd.Index([]))
        title_terms = [term for term in key if term]
        if title_terms and not self.index.empty:
            clauses = " OR ".join("contains(lower(title), lower(?))" for _ in title_terms)
            with self._query_lock:
                title_rows = pd.Index(
                    row[0]
                    for row in self._connection.execute(
                        f"SELECT row FROM candidate_titles WHERE {clauses}",
                        title_terms,
                    ).fetchall()
                )
            rows = code_rows.union(title_rows, sort=False).sort_values()
        else:
            rows = code_rows
        corpus = _CandidateCorpus(self.index.loc[rows].copy())
        self._candidate_cache[key] = corpus
        if len(self._candidate_cache) > _CANDIDATE_CACHE_SIZE:
            _, evicted = self._candidate_cache.popitem(last=False)
            if evicted.bodies is not None:
                for row in evicted.bodies.itertuples(index=False):
                    self._snippets.pop((str(row.dataset), str(row.text_id)), None)
        return corpus

    def _visible_candidate_index(
        self,
        corpus: _CandidateCorpus,
        *,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        if lookback_days is not None:
            if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days <= 0:
                raise ValueError("lookback_days must be a positive integer")
            if self.as_of is None:
                raise ValueError("lookback_days requires a decision as_of time")
        index = corpus.index
        if index.empty:
            return index
        if self.as_of is None:
            visible = index
        else:
            visible = index[self._available_at.loc[index.index] <= self._anchor()]
        if lookback_days is None or visible.empty:
            return visible
        earliest = self._anchor() - pd.Timedelta(days=lookback_days)
        return visible[self._available_at.loc[visible.index] >= earliest]

    # ---- body shards ----

    def _shards(self, dataset: str) -> list[str]:
        """Validated body-shard paths for one dataset, inside the library root."""
        rows = self.index[self._datasets == str(dataset)] if not self.index.empty else self.index
        names: list[str] = []
        if not rows.empty and "library_file" in rows.columns:
            names = sorted({str(name) for name in rows["library_file"].fillna("").astype(str) if name})
        if not names:
            names = [f"{dataset}.parquet"]
        paths = [self._library_path(name) for name in names]
        return [str(path) for path in paths if path.exists()]

    def _library_path(self, library_file: str) -> Path:
        relative = Path(str(library_file))
        if (
            not str(library_file).strip()
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"invalid text library file: {library_file!r}")
        root = self.library_dir.resolve()
        candidate = (root / relative).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise ValueError(f"text library file escapes its root: {library_file!r}")
        if candidate.exists() and not candidate.is_file():
            raise ValueError(f"text library entry is not a file: {library_file!r}")
        return candidate

    def _title_code_match(self, index: pd.DataFrame, regex: str) -> pd.Series:
        """RE2 title/code match over the visible index (boolean mask)."""
        frame = pd.DataFrame(
            {
                "row": index.index,
                "title": index.get("title", pd.Series("", index=index.index)).astype(str),
                "ts_codes": index.get("ts_codes", pd.Series("", index=index.index)).fillna("").astype(str),
            }
        )
        with self._query_lock:
            try:
                self._connection.register("visible_index", frame)
                rows = self._connection.execute(
                    "SELECT row FROM visible_index "
                    "WHERE regexp_matches(title, ?, 'i') OR regexp_matches(ts_codes, ?, 'i')",
                    [regex, regex],
                ).fetchall()
            except duckdb.Error as exc:
                raise ValueError(f"unsupported regex (RE2/grep semantics): {exc}") from exc
            finally:
                self._connection.unregister("visible_index")
        return pd.Series(index.index.isin([row[0] for row in rows]), index=index.index)

    def _grep_candidate_bodies(
        self,
        corpus: _CandidateCorpus,
        visible_index: pd.DataFrame,
        regex: str,
        *,
        exclude: set[str],
        limit: int,
    ) -> set[str]:
        bodies = self._candidate_bodies(corpus, visible_index)
        if bodies.empty:
            return set()
        allowed = set(visible_index["text_id"].astype(str)) - exclude
        frame = bodies[bodies["text_id"].isin(allowed)]
        if frame.empty:
            return set()
        with self._query_lock:
            try:
                self._connection.register("candidate_bodies", frame)
                rows = self._connection.execute(
                    "SELECT text_id FROM candidate_bodies WHERE regexp_matches(body, ?, 'i') LIMIT ?",
                    [regex, limit],
                ).fetchall()
            except duckdb.Error as exc:
                raise ValueError(f"text_retrieve body query failed (RE2/grep semantics): {exc}") from exc
            finally:
                self._connection.unregister("candidate_bodies")
        return {str(row[0]) for row in rows}

    def _candidate_bodies(
        self, corpus: _CandidateCorpus, visible_index: pd.DataFrame
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        datasets = visible_index.get("dataset")
        if datasets is not None:
            dataset_values = datasets.astype(str)
            for dataset in dataset_values.unique():
                shards = self._shards(dataset)
                ids = visible_index.loc[dataset_values == dataset, "text_id"].astype(str).drop_duplicates()
                loaded = {
                    text_id
                    for loaded_dataset, text_id in corpus.loaded_body_ids
                    if loaded_dataset == dataset
                }
                ids = ids[~ids.isin(loaded)]
                if not shards or ids.empty:
                    continue
                placeholders = ",".join("?" for _ in ids)
                rows = self._body_query(
                    "SELECT CAST(text_id AS VARCHAR), CAST(body AS VARCHAR) FROM read_parquet(?) "
                    f"WHERE CAST(text_id AS VARCHAR) IN ({placeholders})",
                    [shards, *ids.tolist()],
                )
                if rows:
                    frame = pd.DataFrame(rows, columns=["text_id", "body"])
                    frame["dataset"] = dataset
                    frames.append(frame)
                corpus.loaded_body_ids.update((dataset, text_id) for text_id in ids)
        if corpus.bodies is None:
            corpus.bodies = pd.DataFrame(columns=["text_id", "body", "dataset"])
        if frames:
            added = pd.concat(frames, ignore_index=True)
            corpus.bodies = pd.concat([corpus.bodies, added], ignore_index=True).drop_duplicates(
                ["dataset", "text_id"], keep="first"
            )
            for row in added.itertuples(index=False):
                self._snippets.setdefault(
                    (str(row.dataset), str(row.text_id)),
                    str(row.body or "")[: self.snippet_chars],
                )
        return corpus.bodies

    def _grep_bodies(
        self,
        index: pd.DataFrame,
        regex: str,
        *,
        exclude: set[str],
        limit: int,
    ) -> set[str]:
        """Linear-time full-body grep with column projection and a result cap;
        matched snippets are cached so ranking never re-reads the shards.

        Shards also hold rows outside the PIT boundary, so the scan semi-joins
        the visible ids first: the LIMIT must count visible matches only, or a
        burst of future-dated matches could crowd valid evidence out of the cap.
        """
        found: set[str] = set()
        datasets = index.get("dataset")
        if datasets is None:
            return found
        dataset_values = datasets.astype(str)
        for dataset in dataset_values.unique():
            shards = self._shards(dataset)
            if not shards:
                continue
            ids = index.loc[dataset_values == dataset, "text_id"].astype(str).drop_duplicates()
            ids = ids[~ids.isin(exclude)]
            if ids.empty:
                continue
            with self._query_lock:
                try:
                    self._connection.register("visible_text_ids", ids.to_frame(name="text_id"))
                    rows = self._connection.execute(
                        "SELECT text_id, substr(body, 1, ?) FROM read_parquet(?) "
                        "WHERE CAST(text_id AS VARCHAR) IN (SELECT text_id FROM visible_text_ids) "
                        "AND regexp_matches(body, ?, 'i') LIMIT ?",
                        [self.snippet_chars, shards, regex, limit],
                    ).fetchall()
                except duckdb.Error as exc:
                    raise ValueError(f"text_retrieve body query failed (RE2/grep semantics): {exc}") from exc
                finally:
                    self._connection.unregister("visible_text_ids")
            for text_id, snippet in rows:
                tid = str(text_id)
                self._snippets.setdefault((dataset, tid), str(snippet or ""))
                found.add(tid)
            if len(found) >= limit:
                break
        return found

    def _body_query(self, query: str, params: list[object]) -> list[tuple]:
        with self._query_lock:
            try:
                return self._connection.execute(query, params).fetchall()
            except duckdb.Error as exc:
                raise ValueError(f"text_retrieve body query failed (RE2/grep semantics): {exc}") from exc

    def _prime_snippets(self, rows: pd.DataFrame) -> None:
        if rows.empty:
            return
        datasets = rows.get("dataset")
        if datasets is None:
            return
        dataset_values = datasets.astype(str)
        for dataset in dataset_values.unique():
            ids = rows.loc[dataset_values == dataset, "text_id"].astype(str).drop_duplicates()
            missing = [text_id for text_id in ids if (dataset, text_id) not in self._snippets]
            shards = self._shards(dataset)
            if not missing or not shards:
                continue
            placeholders = ",".join("?" for _ in missing)
            body_rows = self._body_query(
                "SELECT CAST(text_id AS VARCHAR), substr(body, 1, ?) FROM read_parquet(?) "
                f"WHERE CAST(text_id AS VARCHAR) IN ({placeholders})",
                [self.snippet_chars, shards, *missing],
            )
            for text_id, snippet in body_rows:
                self._snippets[(dataset, str(text_id))] = str(snippet or "")

    def _snippet(self, dataset: str, text_id: str) -> str:
        if not dataset or not text_id:
            return ""
        key = (dataset, text_id)
        cached = self._snippets.get(key)
        if cached is None:
            shards = self._shards(dataset)
            rows = (
                self._body_query(
                    "SELECT substr(body, 1, ?) FROM read_parquet(?) WHERE text_id = ? LIMIT 1",
                    [self.snippet_chars, shards, text_id],
                )
                if shards
                else []
            )
            cached = str(rows[0][0]) if rows and rows[0][0] is not None else ""
            self._snippets[key] = cached
        return cached


def validate_pattern(pattern: str) -> str:
    """Gate the sub-agent pattern to the RE2/grep contract before any scan.

    Length is capped and the pattern is compiled by DuckDB's RE2 up front:
    unsupported constructs (backreferences, lookaround) fail here with a
    fixable message instead of silently matching nothing or falling back to a
    backtracking engine."""
    text = str(pattern or "").strip()
    if not text:
        raise ValueError("text pattern must be non-empty")
    if len(text) > MAX_PATTERN_CHARS:
        raise ValueError(f"text pattern exceeds {MAX_PATTERN_CHARS} characters")
    try:
        duckdb.execute("SELECT regexp_matches('', ?)", [text]).fetchall()
    except duckdb.Error as exc:
        raise ValueError(f"text pattern is outside the RE2 contract: {exc}") from exc
    return text


def _candidate_terms(ts_code: str, company_terms: list[str] | None = None) -> list[str]:
    code = str(ts_code or "").strip()
    terms = [code] if code else []
    if "." in code:
        terms.append(code.split(".", 1)[0])
    terms.extend(str(term).strip() for term in (company_terms or []) if str(term).strip())
    seen: set[str] = set()
    ordered: list[str] = []
    for term in terms:
        if term and term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered
