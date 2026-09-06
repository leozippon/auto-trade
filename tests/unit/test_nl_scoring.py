"""Company context and the PIT text retriever's internals.

``TextRetriever`` is the single retrieval implementation behind ``ctx.nl()``:
its candidate cache, incremental body loading and rolling index re-read are
what keep a backtest's NL calls both point-in-time correct and affordable.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from autotrade.environment.nl import TextRetriever

CN_TZ = ZoneInfo("Asia/Shanghai")
INDEX_COLUMNS = ["text_id", "dataset", "ts_codes", "title", "available_at", "library_file"]


class CompanyContextStoreTest(unittest.TestCase):
    """The frozen snapshot is immutable for a backtest, so the company-context
    sources are read once and each ts_code's context is memoized (R17)."""

    def _make_snapshot(self, tmp: Path) -> Path:
        snap = tmp / "snap"
        snap.mkdir(parents=True)
        pd.DataFrame(
            {"ts_code": ["000001.SZ"], "name": ["平安银行"], "exchange": ["SZSE"], "l1_name": ["银行"]}
        ).to_parquet(snap / "universe.parquet", index=False)
        pd.DataFrame(
            {
                "dataset": ["fina_mainbz_vip"],
                "ts_code": ["000001.SZ"],
                "bz_item": ["零售金融"],
                "end_date": ["20211231"],
                "available_at": ["2022-01-04T18:00:00+08:00"],
            }
        ).to_parquet(snap / "fundamentals.parquet", index=False)
        return snap

    def test_sources_read_once_and_contexts_memoized(self):
        from unittest import mock

        from autotrade.environment.nl.context import CompanyContextStore

        with tempfile.TemporaryDirectory() as tmp:
            snap = self._make_snapshot(Path(tmp))
            with mock.patch(
                "autotrade.environment.nl.context.pd.read_parquet", wraps=pd.read_parquet
            ) as spy:
                store = CompanyContextStore(snap)
                self.assertEqual(spy.call_count, 0)  # lazy: nothing read at construction
                first = store.context("000001.SZ")
                again = store.context("000001.SZ")
                other = store.context("999999.SZ")
            # universe.parquet + fundamentals.parquet read exactly once across all calls.
            self.assertEqual(spy.call_count, 2)
            self.assertIs(first, again)  # memoized object, not rebuilt
            self.assertEqual(first["name"], "平安银行")
            self.assertEqual(first["main_business"], ["零售金融"])
            self.assertEqual(other["context"], "insufficient_company_information")


class TextRetrieverRollingTest(unittest.TestCase):
    """``ctx.nl()`` text rolls with the as-of clock: a row is invisible until
    its ``available_at`` has passed, and the Timeview appends shards mid-run."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.library = self.root / "text_library"
        self.library.mkdir()

    def _write_index(self, rows: list[tuple[str, str, str, str, str]], *, directory: bool = False):
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
        if directory:
            index_dir = self.root / "text_index"
            index_dir.mkdir(exist_ok=True)
            frame.to_parquet(index_dir / f"part_{len(list(index_dir.iterdir())):04d}.parquet", index=False)
            return index_dir
        frame.to_parquet(self.root / "text_index.parquet", index=False)
        return self.root / "text_index.parquet"

    def _write_bodies(self, dataset: str, bodies: dict[str, str]) -> None:
        pd.DataFrame(
            {"text_id": list(bodies), "body": list(bodies.values())}
        ).to_parquet(self.library / f"{dataset}.parquet", index=False)

    def _retriever(self, index_path) -> TextRetriever:
        retriever = TextRetriever(index_path, self.library)
        self.addCleanup(retriever.close)
        return retriever

    def test_rows_appear_only_once_their_available_at_has_passed(self):
        index = self._write_index(
            [
                ("f1", "anns_d", "000001.SZ", "Frozen announcement", "2021-10-01T18:00:00+08:00"),
                ("r1", "anns_d", "000001.SZ", "Replay announcement", "2022-01-04T18:00:00+08:00"),
            ]
        )
        self._write_bodies("anns_d", {"f1": "frozen body", "r1": "replay body"})
        retriever = self._retriever(index)

        def ids(as_of):
            retriever.as_of = as_of
            return {hit["text_id"] for hit in retriever.search("announcement", ts_code="000001.SZ", max_results=10)}

        # Noon on the announcement's own day: the 18:00 row is not out yet.
        self.assertEqual(ids(datetime(2022, 1, 4, 12, 0, tzinfo=CN_TZ)), {"f1"})
        self.assertEqual(ids(datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)), {"f1", "r1"})

    def test_as_of_is_required_before_any_visibility_decision(self):
        index = self._write_index(
            [("f1", "anns_d", "000001.SZ", "Frozen announcement", "2021-10-01T18:00:00+08:00")]
        )
        self._write_bodies("anns_d", {"f1": "frozen body"})
        retriever = self._retriever(index)
        retriever.as_of = None
        with self.assertRaisesRegex(ValueError, "as_of"):
            retriever.candidate_evidence_state("000001.SZ", patterns=("announcement",), lookback_days=3660)

    def test_candidate_bodies_load_only_pit_visible_rows(self):
        index = self._write_index(
            [
                ("f1", "anns_d", "000001.SZ", "Frozen announcement", "2021-10-01T18:00:00+08:00"),
                ("r1", "anns_d", "000001.SZ", "Replay announcement", "2022-01-04T18:00:00+08:00"),
            ]
        )
        self._write_bodies("anns_d", {"f1": "frozen body", "r1": "replay body"})
        retriever = self._retriever(index)

        retriever.as_of = datetime(2022, 1, 4, 12, 0, tzinfo=CN_TZ)
        retriever.search("body", ts_code="000001.SZ", max_results=10)
        corpus = next(iter(retriever._candidate_cache.values()))
        self.assertEqual(corpus.loaded_body_ids, {("anns_d", "f1")})

        retriever.as_of = datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)
        retriever.search("body", ts_code="000001.SZ", max_results=10)
        self.assertEqual(corpus.loaded_body_ids, {("anns_d", "f1"), ("anns_d", "r1")})

    def test_candidate_corpus_is_reused_across_calls_at_one_as_of_instant(self):
        index = self._write_index(
            [("f1", "anns_d", "000001.SZ", "Frozen announcement", "2021-10-01T18:00:00+08:00")]
        )
        self._write_bodies("anns_d", {"f1": "frozen body"})
        retriever = self._retriever(index)
        retriever.as_of = datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)
        key = retriever._candidate_key("000001.SZ", None)
        first = retriever._candidate_corpus(key)
        # The corpus (rows, loaded bodies and per-pattern verdicts) is cached:
        # re-scanning the whole index per ctx.nl() call was 78% of NL wall time.
        self.assertIs(retriever._candidate_corpus(key), first)
        retriever.search("body", ts_code="000001.SZ", max_results=10)
        self.assertIs(retriever._candidate_corpus(key), first)
        self.assertIn(key, retriever._candidate_cache)

    def test_candidate_revision_moves_only_when_linked_evidence_becomes_visible(self):
        index = self._write_index(
            [
                ("f1", "anns_d", "000001.SZ", "Frozen announcement", "2021-10-01T18:00:00+08:00"),
                ("r1", "anns_d", "000001.SZ", "Replay announcement", "2022-01-04T18:00:00+08:00"),
            ]
        )
        self._write_bodies("anns_d", {"f1": "frozen body", "r1": "replay body"})
        retriever = self._retriever(index)

        retriever.as_of = datetime(2022, 1, 4, 12, 0, tzinfo=CN_TZ)
        before = retriever.candidate_evidence_state("000001.SZ", lookback_days=3660).revision
        retriever.as_of = datetime(2022, 1, 4, 13, 0, tzinfo=CN_TZ)
        self.assertEqual(
            retriever.candidate_evidence_state("000001.SZ", lookback_days=3660).revision, before
        )
        retriever.as_of = datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)
        self.assertNotEqual(
            retriever.candidate_evidence_state("000001.SZ", lookback_days=3660).revision, before
        )

    def test_candidate_revision_follows_the_declared_patterns(self):
        index = self._write_index(
            [
                ("f1", "anns_d", "000001.SZ", "Frozen announcement", "2021-10-01T18:00:00+08:00"),
                ("r1", "anns_d", "000001.SZ", "Replay announcement", "2022-01-04T18:00:00+08:00"),
            ]
        )
        self._write_bodies("anns_d", {"f1": "frozen body", "r1": "replay body"})
        retriever = self._retriever(index)

        retriever.as_of = datetime(2022, 1, 4, 12, 0, tzinfo=CN_TZ)
        all_before = retriever.candidate_evidence_state("000001.SZ", lookback_days=3660).revision
        risk_before = retriever.candidate_evidence_state(
            "000001.SZ", patterns=("replay body",), lookback_days=3660
        ).revision
        other_before = retriever.candidate_evidence_state(
            "000001.SZ", patterns=("处罚",), lookback_days=3660
        ).revision

        retriever.as_of = datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)

        self.assertNotEqual(
            retriever.candidate_evidence_state("000001.SZ", lookback_days=3660).revision, all_before
        )
        self.assertNotEqual(
            retriever.candidate_evidence_state(
                "000001.SZ", patterns=("replay body",), lookback_days=3660
            ).revision,
            risk_before,
        )
        # A predicate nothing matches is unaffected by the new visible row.
        self.assertEqual(
            retriever.candidate_evidence_state(
                "000001.SZ", patterns=("处罚",), lookback_days=3660
            ).revision,
            other_before,
        )

    def test_candidate_scope_uses_a_rolling_window_and_expires(self):
        index = self._write_index(
            [("r1", "anns_d", "000001.SZ", "Replay announcement", "2022-01-04T18:00:00+08:00")]
        )
        self._write_bodies("anns_d", {"r1": "replay body"})
        retriever = self._retriever(index)

        retriever.as_of = datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)
        active = retriever.candidate_evidence_state(
            "000001.SZ", patterns=("body",), lookback_days=30, max_results=5
        )
        self.assertEqual(active.match_count, 1)
        self.assertEqual([item["text_id"] for item in active.evidence], ["r1"])
        self.assertEqual(
            [
                hit["text_id"]
                for hit in retriever.search(
                    "body", ts_code="000001.SZ", max_results=10, lookback_days=30
                )
            ],
            ["r1"],
        )

        retriever.as_of = datetime(2022, 2, 5, 9, 0, tzinfo=CN_TZ)
        expired = retriever.candidate_evidence_state(
            "000001.SZ", patterns=("body",), lookback_days=30
        )
        self.assertEqual(expired.match_count, 0)
        self.assertNotEqual(expired.revision, active.revision)
        self.assertEqual(retriever.search("body", ts_code="000001.SZ", lookback_days=30), [])

    def test_candidate_lookback_rejects_invalid_values_even_for_an_empty_scope(self):
        index = self._write_index(
            [("r1", "anns_d", "000001.SZ", "Replay announcement", "2022-01-04T18:00:00+08:00")]
        )
        self._write_bodies("anns_d", {"r1": "replay body"})
        retriever = self._retriever(index)
        retriever.as_of = datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)
        for value in (True, 1.5, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                retriever.candidate_evidence_state("999999.SZ", lookback_days=value)

    def test_a_directory_index_picks_up_shards_appended_mid_run(self):
        index_dir = self._write_index(
            [("f1", "anns_d", "000001.SZ", "First announcement", "2021-10-01T18:00:00+08:00")],
            directory=True,
        )
        self._write_bodies("anns_d", {"f1": "first body", "later": "later body"})
        retriever = self._retriever(index_dir)
        retriever.as_of = datetime(2022, 1, 5, 9, 0, tzinfo=CN_TZ)
        self.assertEqual(
            {hit["text_id"] for hit in retriever.search("body", ts_code="000001.SZ", max_results=10)},
            {"f1"},
        )
        # The Timeview appends a shard while the backtest is running.
        self._write_index(
            [("later", "anns_d", "000001.SZ", "Later announcement", "2022-01-04T18:00:00+08:00")],
            directory=True,
        )
        self.assertEqual(
            {hit["text_id"] for hit in retriever.search("body", ts_code="000001.SZ", max_results=10)},
            {"f1", "later"},
        )


if __name__ == "__main__":
    unittest.main()
