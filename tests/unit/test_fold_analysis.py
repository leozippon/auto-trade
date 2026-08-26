"""Fold/Step strategy analysis: guarded inputs, persistence, failure honesty.

The analysis prompt is assembled from a completed Fold's own record, so the
guard that strips Test evidence out of it is the whole reason this module can
run at all: an analysis that quoted the frozen-Test number would put it in front
of a researcher who is still steering development.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from autotrade.environment.identity import AgentRefStore
from autotrade.pipelines.fold_analysis import (
    DEFAULT_MAX_TOKENS,
    analysis_key,
    analysis_paths,
    analyze_fold,
    analyze_step,
    build_fold_analysis_messages,
    guarded_record_view,
    read_strategy_files,
)

RECORD = {
    "epoch_id": "epoch_001",
    "fold_id": "fold_2022Q1",
    "validation_period": "20211001..20211231",
    "test_period": "20220101..20220331",
    "validation_result": {"total_return": 0.01},
    "test_result": {"total_return": 0.09},
    "test_result_ref": "/experiments/exp/artifacts/results/frozen_test_000/result.json",
    "fold_status": "frozen",
}


class FakeProxy:
    provider = "fake"
    model = "fake-model"

    def __init__(self, content: str = "## 策略逻辑概述\n看多动量。") -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def complete(self, messages, *, tools=(), tool_choice="auto", max_tokens=None):
        self.calls.append({"messages": messages, "tools": tools, "tool_choice": tool_choice,
                           "max_tokens": max_tokens})
        return SimpleNamespace(content=self.content, usage={"total_tokens": 10})


class ExplodingProxy(FakeProxy):
    def complete(self, messages, **kwargs):
        raise RuntimeError("provider unavailable")


class FoldAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self._refs_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._refs_tmp.cleanup)
        self.ref_store = AgentRefStore(Path(self._refs_tmp.name) / "experiment")
        self.fold_ref = self.ref_store.get_or_create("fold", "fold_2022Q1")

    def test_guarded_view_excludes_test_evidence(self) -> None:
        view = guarded_record_view(RECORD, ref_store=self.ref_store)
        self.assertNotIn("test_result", view)
        self.assertNotIn("test_period", view)
        self.assertNotIn("test_result_ref", view)
        self.assertEqual(view["validation_result"], {"total_return": 0.01})
        self.assertEqual(view["fold_status"], "frozen")

    def test_prompt_never_carries_the_test_number_or_window(self) -> None:
        messages = build_fold_analysis_messages(
            RECORD,
            [{"path": "main.py", "content": "print(1)", "truncated": False}],
            ref_store=self.ref_store,
        )
        user = messages[1].content
        self.assertNotIn("0.09", user)
        self.assertNotIn("20220101..20220331", user)
        self.assertIn("print(1)", user)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")

    def test_analyze_fold_writes_markdown_and_a_provenance_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strategy = Path(tmp) / "strategy"
            strategy.mkdir()
            (strategy / "main.py").write_text(
                "def generate_orders(context):\n    return []\n", encoding="utf-8"
            )
            (strategy / "manifest.json").write_text("{}", encoding="utf-8")
            out_dir = Path(tmp) / "analysis"
            proxy = FakeProxy()
            md_path = analyze_fold(
                proxy,
                ledger_record=RECORD,
                ref_store=self.ref_store,
                strategy_dir=strategy,
                model_dir=None,
                out_dir=out_dir,
                output_identity=("epoch_001", self.fold_ref),
            )
            self.assertIn("看多动量", md_path.read_text(encoding="utf-8"))
            self.assertEqual(md_path.name, f"epoch_001__{self.fold_ref}.md")
            meta = json.loads(
                (out_dir / f"epoch_001__{self.fold_ref}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(meta["status"], "ok")
            self.assertEqual(meta["guarded_view"], "validation_only")
            self.assertEqual(meta["analysis_kind"], "fold")
            self.assertEqual((meta["provider"], meta["model"]), ("fake", "fake-model"))
            self.assertEqual(meta["usage"], {"total_tokens": 10})
            self.assertEqual(meta["fold_ref"], self.fold_ref)
            self.assertEqual(meta["output_ref"], self.fold_ref)
            self.assertEqual(meta["analysis_path"], md_path.name)
            self.assertFalse(Path(meta["analysis_path"]).is_absolute())
            # The analysis call is text-only: no tool surface is offered.
            self.assertEqual(proxy.calls[0]["tools"], ())
            self.assertEqual(proxy.calls[0]["tool_choice"], "none")
            self.assertEqual(proxy.calls[0]["max_tokens"], DEFAULT_MAX_TOKENS)
            self.assertNotIn("0.09", json.dumps(meta, ensure_ascii=False))

    def test_provider_failure_records_an_error_sidecar_and_re_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strategy = Path(tmp) / "strategy"
            strategy.mkdir()
            (strategy / "main.py").write_text("pass\n", encoding="utf-8")
            out_dir = Path(tmp) / "analysis"
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                analyze_fold(
                    ExplodingProxy(), ledger_record=RECORD, ref_store=self.ref_store,
                    strategy_dir=strategy, model_dir=None, out_dir=out_dir,
                    output_identity=("epoch_001", self.fold_ref),
                )
            meta = json.loads(
                (out_dir / f"epoch_001__{self.fold_ref}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(meta["status"], "error")
            self.assertIn("provider unavailable", meta["error"])
            self.assertFalse((out_dir / f"epoch_001__{self.fold_ref}.md").exists())

    def test_empty_analysis_content_is_a_failure_not_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strategy = Path(tmp) / "strategy"
            strategy.mkdir()
            (strategy / "main.py").write_text("pass\n", encoding="utf-8")
            out_dir = Path(tmp) / "analysis"
            with self.assertRaisesRegex(RuntimeError, "empty"):
                analyze_fold(
                    FakeProxy(content="   \n"), ledger_record=RECORD, ref_store=self.ref_store,
                    strategy_dir=strategy, model_dir=None, out_dir=out_dir,
                    output_identity=("epoch_001", self.fold_ref),
                )
            meta = json.loads(
                (out_dir / f"epoch_001__{self.fold_ref}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(meta["status"], "error")
            self.assertFalse((out_dir / f"epoch_001__{self.fold_ref}.md").exists())

    def test_analyze_step_writes_under_its_node_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strategy = Path(tmp) / "strategy"
            strategy.mkdir()
            (strategy / "main.py").write_text("pass\n", encoding="utf-8")
            out_dir = Path(tmp) / "analysis"
            node_id = "epoch_001__fold_ref_ab__run_x__valid_000"
            proxy = FakeProxy(content="## 步骤解读\nok")
            md_path = analyze_step(
                proxy,
                step_record={"epoch_id": "epoch_001", "fold_id": "fold_2022Q1",
                             "validation_result": {"total_return": 0.01}},
                ref_store=self.ref_store,
                strategy_dir=strategy,
                model_dir=None,
                out_dir=out_dir,
                node_id=node_id,
            )
            self.assertEqual(md_path.name, f"step__{node_id}.md")
            meta = json.loads(md_path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(meta["analysis_kind"], "step")
            # Only public record identities survive in the sidecar; the output
            # itself is keyed by the immutable Step node.
            self.assertEqual(meta["epoch_id"], "step")
            self.assertEqual(meta["fold_ref"], self.fold_ref)
            self.assertEqual(meta["output_ref"], node_id)
            self.assertNotIn("fold_id", meta)

    def test_analysis_paths_are_derived_from_one_key(self) -> None:
        self.assertEqual(analysis_key("epoch_001", "fold_2022Q1"), "epoch_001__fold_2022Q1")
        md_path, meta_path = analysis_paths(Path("/out"), "epoch_001", "fold_2022Q1")
        self.assertEqual(md_path, Path("/out/epoch_001__fold_2022Q1.md"))
        self.assertEqual(meta_path, Path("/out/epoch_001__fold_2022Q1.json"))

    def test_strategy_files_are_bounded_and_report_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            strategy = Path(tmp) / "strategy"
            strategy.mkdir()
            (strategy / "main.py").write_text("x" * 5_000, encoding="utf-8")
            (strategy / "notes.md").write_text("small\n", encoding="utf-8")
            files = read_strategy_files(strategy, max_file_chars=100, max_total_chars=1_000)
            by_path = {item["path"]: item for item in files}
            self.assertTrue(by_path["main.py"]["truncated"])
            self.assertLessEqual(len(by_path["main.py"]["content"]), 100)
            self.assertFalse(by_path["notes.md"]["truncated"])


if __name__ == "__main__":
    unittest.main()
