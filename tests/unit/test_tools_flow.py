"""Agent tool primitives: shell, structured search, and typed artifact I/O.

These are the Agent's whole filesystem surface, so their negative paths are
security boundaries, not conveniences: path traversal, symlink escape, hidden
paths, size caps, and stale/ambiguous edits must all fail loudly. The tools are
exercised through the registry contract the runner actually dispatches, over a
synthetic sandbox layout with a real PIT root and a real prior-fold artifact.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from autotrade.environment.runtime import SandboxPaths
from autotrade.environment.tools import (
    CommandResult,
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadFileTool,
    SafeWorkspace,
    SandboxShellTool,
    SearchRoots,
    ToolError,
    ToolRegistry,
    WriteFileTool,
)
from autotrade.environment.tools.shell import (
    DEFAULT_SHELL_TIMEOUT_SECONDS,
    FORBIDDEN_WAIT,
    argv_is_forbidden_wait,
)
from autotrade.environment.tools import search as search_module

RG_AVAILABLE = shutil.which("rg") is not None


def build_sandbox(root: Path) -> tuple[SandboxPaths, SearchRoots, SafeWorkspace]:
    """A synthetic sandbox layout with every read root the Agent may reach."""
    paths = SandboxPaths(root)
    for directory in (
        paths.workspace,
        paths.agent_output,
        paths.model_artifacts,
        paths.current_snapshot,
        paths.train,
        paths.valid,
        paths.artifacts,
        paths.parent_output,
        paths.parent_model_artifacts,
        paths.results,
        paths.steps,
        paths.logs,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (paths.agent_output / "README.md").write_text("readonly\n", encoding="utf-8")
    workspace = SafeWorkspace(paths.agent)
    return paths, SearchRoots(workspace, paths=paths), workspace


class FakeRunner:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.calls: list[tuple] = []
        self.result = result or CommandResult(0, stdout="ok")

    def run(self, argv, *, cwd, timeout_seconds, max_output_chars, input_text=None):
        self.calls.append((tuple(argv), cwd, timeout_seconds, max_output_chars, input_text))
        return self.result


class ShellToolTest(unittest.TestCase):
    """Shell is the one unstructured surface, so its audit record and its
    output budget carry the whole guarantee."""

    def test_shell_calls_only_the_injected_runner_with_the_configured_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, workspace = build_sandbox(Path(tmp))
            runner = FakeRunner(CommandResult(0, stdout="hello"))
            tool = SandboxShellTool(workspace, runner, timeout_seconds=5, max_output_chars=100)
            result = tool.invoke({"argv": ["echo", "hello"], "cwd": ".", "timeout_seconds": 900})
            self.assertTrue(result.ok)
            self.assertEqual(result.value["stdout"], "hello")
            self.assertEqual(result.value["exit_code"], 0)
            # The session's configured caps win over anything the model asks for.
            self.assertEqual(runner.calls, [(("echo", "hello"), ".", 5, 100, None)])

    def test_shell_reports_a_nonzero_exit_without_inventing_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, workspace = build_sandbox(Path(tmp))
            runner = FakeRunner(CommandResult(2, stdout="", stderr="boom"))
            result = SandboxShellTool(workspace, runner).invoke({"argv": ["false"]})
            self.assertEqual(result.value["exit_code"], 2)
            self.assertEqual(result.value["stderr"], "boom")

    def test_shell_rejects_a_cwd_outside_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, workspace = build_sandbox(Path(tmp))
            runner = FakeRunner()
            registry = ToolRegistry([SandboxShellTool(workspace, runner)])
            for cwd in ("../snapshot", "/etc", "output/../../snapshot"):
                result = registry.invoke("shell", {"argv": ["ls"], "cwd": cwd})
                self.assertFalse(result.ok, cwd)
            self.assertEqual(runner.calls, [])

    def test_shell_rejects_an_empty_or_non_string_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, workspace = build_sandbox(Path(tmp))
            runner = FakeRunner()
            registry = ToolRegistry([SandboxShellTool(workspace, runner)])
            for argv in ([], [""], ["ls", 3]):
                self.assertFalse(registry.invoke("shell", {"argv": argv}).ok, argv)
            self.assertEqual(runner.calls, [])

    def test_shell_is_declared_mutating_so_the_finish_lock_reaches_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, workspace = build_sandbox(Path(tmp))
            tool = SandboxShellTool(workspace, FakeRunner())
            self.assertTrue(tool.spec.mutating)

    def test_shell_schema_timeout_maximum_matches_instance_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, workspace = build_sandbox(Path(tmp))
            default = SandboxShellTool(workspace, FakeRunner())
            custom = SandboxShellTool(workspace, FakeRunner(), timeout_seconds=5)
            default_schema = json.loads(json.dumps(default.spec.input_schema))
            custom_schema = json.loads(json.dumps(custom.spec.input_schema))
            self.assertEqual(default.timeout_seconds, DEFAULT_SHELL_TIMEOUT_SECONDS)
            self.assertEqual(
                default_schema["properties"]["timeout_seconds"]["maximum"],
                DEFAULT_SHELL_TIMEOUT_SECONDS,
            )
            self.assertEqual(
                custom_schema["properties"]["timeout_seconds"]["maximum"], 5
            )
            registry = ToolRegistry([default])
            blocked = registry.invoke(
                "shell", {"argv": ["echo", "ok"], "timeout_seconds": 120}
            )
            self.assertFalse(blocked.ok)
            self.assertIn("above its maximum", blocked.error)

    def test_shell_rejects_forbidden_wait_and_allows_foreground_commands(self) -> None:
        rejected = (
            ["sleep", "5"],
            ["/bin/sleep", "1"],
            ["usleep", "1000"],
            ["env", "sleep", "1"],
            ["timeout", "5", "sleep", "1"],
            ["nice", "-n", "10", "sleep", "1"],
            ["stdbuf", "-oL", "sleep", "1"],
            ["nohup", "sleep", "1"],
            ["time", "sleep", "1"],
            ["bash", "-c", "sleep 5"],
            ["bash", "-lc", "sleep 1"],
            ["sh", "-c", "sleep 1"],
            ["dash", "-c", "env sleep 1"],
            ["timeout", "5", "env", "sleep", "1"],
        )
        allowed = (
            ["pyright"],
            ["timeout", "pyright"],
            ["timeout", "5", "pyright"],
            ["python", "-c", "import time; time.sleep(1)"],
            ["grep", "sleep", "file.py"],
            ["echo", "sleep"],
            ["bash", "-c", "echo hello"],
            ["env", "FOO=1", "pyright"],
        )
        for argv in rejected:
            self.assertTrue(argv_is_forbidden_wait(argv), argv)
        for argv in allowed:
            self.assertFalse(argv_is_forbidden_wait(argv), argv)
        with tempfile.TemporaryDirectory() as tmp:
            _, _, workspace = build_sandbox(Path(tmp))
            runner = FakeRunner()
            registry = ToolRegistry([SandboxShellTool(workspace, runner)])
            result = registry.invoke("shell", {"argv": ["sleep", "5"]})
            self.assertFalse(result.ok)
            self.assertEqual(result.value.get("error_type"), FORBIDDEN_WAIT)
            self.assertEqual(runner.calls, [])
            ok = registry.invoke("shell", {"argv": ["echo", "hello"]})
            self.assertTrue(ok.ok)
            self.assertEqual(len(runner.calls), 1)


@unittest.skipUnless(RG_AVAILABLE, "ripgrep is required for the structured search tools")
class StructuredSearchToolTest(unittest.TestCase):
    def _tools(self, root: Path):
        paths, roots, workspace = build_sandbox(root)
        return paths, roots, ToolRegistry([GrepTool(roots), GlobTool(roots), ReadFileTool(roots)])

    def test_grep_and_glob_are_structured_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, roots, registry = self._tools(Path(tmp))
            (paths.workspace / "alpha.txt").write_text("alpha\nbeta\n", encoding="utf-8")
            (paths.workspace / "foo-bar.txt").write_text("alpha\n", encoding="utf-8")
            (paths.workspace / ".hidden.txt").write_text("alpha\n", encoding="utf-8")
            (paths.workspace / "nested").mkdir()
            (paths.workspace / "nested" / "gamma.json").write_text('{"key": "alpha"}', encoding="utf-8")
            (paths.agent_output / "main.py").write_text(
                "def generate_orders(context):\n    unique_output_marker = True\n    return []\n",
                encoding="utf-8",
            )
            (paths.model_artifacts / "model.json").write_text('{"name": "agent-model"}\n', encoding="utf-8")
            (paths.parent_model_artifacts / "parent.json").write_text('{"name": "parent-model"}\n', encoding="utf-8")

            files = registry.invoke(
                "grep", {"pattern": "alpha", "root": "workspace", "output_mode": "files"}
            ).value
            self.assertEqual(files["mode"], "files")
            self.assertEqual(files["num_files"], 3)
            self.assertIn("workspace/alpha.txt", files["filenames"])
            self.assertNotIn("workspace/.hidden.txt", files["filenames"])

            content = registry.invoke(
                "grep",
                {"pattern": "alpha", "root": "workspace", "output_mode": "content", "glob": "*.txt"},
            ).value
            self.assertEqual(content["mode"], "content")
            self.assertIn("workspace/alpha.txt:1:alpha", content["content"])
            self.assertIn("workspace/foo-bar.txt", content["filenames"])

            counts = registry.invoke(
                "grep", {"pattern": "alpha", "root": "workspace", "output_mode": "count"}
            ).value
            self.assertEqual(counts["page_matches"], 3)

            listing = registry.invoke("glob", {"pattern": "**/*.json", "root": "workspace"}).value
            # `workspace` is the whole agent tree, so the model artifact is in scope.
            self.assertEqual(
                listing["filenames"], ["models/model.json", "workspace/nested/gamma.json"]
            )
            (paths.workspace / "b.py").write_text("", encoding="utf-8")
            (paths.workspace / "a.py").write_text("", encoding="utf-8")
            (paths.workspace / "nested" / "c.py").write_text("", encoding="utf-8")
            top_py = registry.invoke("glob", {"pattern": "workspace/*.py", "root": "workspace"}).value
            self.assertEqual(top_py["filenames"], ["workspace/a.py", "workspace/b.py"])
            page_1 = registry.invoke(
                "glob", {"pattern": "workspace/**/*.py", "root": "workspace", "head_limit": 2}
            ).value
            page_2 = registry.invoke(
                "glob",
                {"pattern": "workspace/**/*.py", "root": "workspace", "head_limit": 2, "offset": 2},
            ).value
            self.assertEqual(page_1["filenames"], ["workspace/a.py", "workspace/b.py"])
            self.assertEqual(page_2["filenames"], ["workspace/nested/c.py"])

            # A symlink loop inside a read root must not be followed.
            (paths.workspace / "loop").mkdir()
            os.symlink(paths.workspace, paths.workspace / "loop" / "self")
            looped = registry.invoke(
                "glob", {"pattern": "workspace/**/*.py", "root": "workspace", "head_limit": 10}
            ).value
            self.assertEqual(
                looped["filenames"], ["workspace/a.py", "workspace/b.py", "workspace/nested/c.py"]
            )

            output_files = registry.invoke(
                "grep", {"pattern": "unique_output_marker", "root": "output", "output_mode": "files"}
            ).value
            self.assertEqual(output_files["filenames"], ["main.py"])
            model_files = registry.invoke(
                "grep", {"pattern": "agent-model", "root": "models", "output_mode": "files"}
            ).value
            self.assertEqual(model_files["filenames"], ["model.json"])
            parent_files = registry.invoke(
                "grep", {"pattern": "parent-model", "root": "parent_models", "output_mode": "files"}
            ).value
            self.assertEqual(parent_files["filenames"], ["parent.json"])

            limited = registry.invoke(
                "grep",
                {"pattern": "alpha", "root": "workspace", "output_mode": "files", "head_limit": 1},
            ).value
            self.assertEqual(limited["returned"], 1)
            self.assertTrue(limited["truncated"])

    def test_search_roots_are_an_enum_and_traversal_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, roots, registry = self._tools(Path(tmp))
            (paths.workspace / ".hidden.txt").write_text("alpha\n", encoding="utf-8")
            # `test` is never offered: the schema enum is built from the roots
            # that exist, and `test` is not one of them.
            self.assertNotIn("test", roots.names)
            refused = registry.invoke("grep", {"pattern": "x", "root": "test"})
            self.assertFalse(refused.ok)
            traversal = registry.invoke("glob", {"pattern": "../*.json", "root": "workspace"})
            self.assertFalse(traversal.ok)
            self.assertEqual(traversal.value["error_type"], "path_error")
            hidden = registry.invoke("glob", {"pattern": "workspace/.hidden.txt", "root": "workspace"})
            self.assertFalse(hidden.ok)

    def test_read_returns_line_numbered_paginated_and_guarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _, registry = self._tools(Path(tmp))
            (paths.workspace / "f.txt").write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")
            full = registry.invoke("read_file", {"root": "workspace", "path": "workspace/f.txt"}).value
            self.assertEqual(full["line_count"], 4)
            self.assertIn("1\tl1", full["content"])  # cat -n style
            self.assertIn("4\tl4", full["content"])
            page = registry.invoke(
                "read_file", {"root": "workspace", "path": "workspace/f.txt", "offset": 1, "limit": 2}
            ).value
            self.assertIn("2\tl2", page["content"])
            self.assertNotIn("1\tl1", page["content"])
            self.assertNotIn("4\tl4", page["content"])
            # Guards: empty path, directories, unknown roots and hidden paths.
            self.assertFalse(registry.invoke("read_file", {"root": "workspace", "path": ""}).ok)
            (paths.workspace / "sub").mkdir()
            self.assertFalse(registry.invoke("read_file", {"root": "workspace", "path": "workspace/sub"}).ok)
            self.assertFalse(
                registry.invoke("read_file", {"root": "test", "path": "daily.parquet"}).ok
            )
            self.assertFalse(registry.invoke("read_file", {"root": "workspace", "path": "workspace/.secret"}).ok)
            escaped = registry.invoke(
                "read_file", {"root": "workspace", "path": "../../etc/passwd"}
            )
            self.assertFalse(escaped.ok)
            self.assertEqual(escaped.value["error_type"], "path_error")

    def test_read_follows_no_symlink_out_of_its_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _, registry = self._tools(Path(tmp))
            secret = Path(tmp) / "outside.txt"
            secret.write_text("classified\n", encoding="utf-8")
            os.symlink(secret, paths.workspace / "link.txt")
            escaped = registry.invoke("read_file", {"root": "workspace", "path": "workspace/link.txt"})
            self.assertFalse(escaped.ok)
            self.assertNotIn("classified", json.dumps(escaped.value))

    def test_read_rejects_files_over_the_size_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _, registry = self._tools(Path(tmp))
            (paths.workspace / "big.bin").write_bytes(b"x" * 2048)
            (paths.workspace / "small.txt").write_text("ok\n", encoding="utf-8")
            with patch.object(search_module, "MAX_READ_BYTES", 1024):
                too_large = registry.invoke("read_file", {"root": "workspace", "path": "workspace/big.bin"})
                small = registry.invoke("read_file", {"root": "workspace", "path": "workspace/small.txt"})
            self.assertFalse(too_large.ok)
            self.assertEqual(too_large.value["error_type"], "too_large")
            self.assertIn("read cap", too_large.error)
            self.assertEqual(small.value["line_count"], 1)

    def test_files_and_glob_pages_are_budgeted_as_a_whole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _, registry = self._tools(Path(tmp))
            names = [f"workspace/file_{index:02d}_{'x' * 24}.txt" for index in range(30)]
            for name in names:
                (paths.agent / name).write_text("alpha\n", encoding="utf-8")
            with patch.object(search_module, "MAX_RESULT_CHARS", 200):
                files = registry.invoke(
                    "grep",
                    {"pattern": "alpha", "root": "workspace", "output_mode": "files", "head_limit": 1000},
                ).value
                listing = registry.invoke(
                    "glob", {"pattern": "workspace/*.txt", "root": "workspace", "head_limit": 1000}
                ).value
            for record in (files, listing):
                # The page rides the observation exactly once, inside the char
                # budget: no duplicated joined-text copy escaping the cap.
                self.assertNotIn("content", record)
                self.assertTrue(record["truncated_by_chars"])
                self.assertTrue(record["filenames"])
                self.assertLessEqual(len("\n".join(record["filenames"])), 200)
                for name in record["filenames"]:
                    self.assertIn(name, names)  # no char-cut partial paths
                self.assertEqual(record["returned"], len(record["filenames"]))
                # The full page is persisted outside the model context budget.
                stored = Path(str(record["result_path"])).read_text(encoding="utf-8")
                self.assertEqual(len(stored.split("\n")), len(names))

    def test_content_filenames_come_from_the_budgeted_page(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _, registry = self._tools(Path(tmp))
            names = [f"workspace/match_{index:02d}.txt" for index in range(40)]
            for name in names:
                (paths.agent / name).write_text("needle\n", encoding="utf-8")
            with patch.object(search_module, "MAX_RESULT_CHARS", 150):
                record = registry.invoke(
                    "grep",
                    {"pattern": "needle", "root": "workspace", "output_mode": "content", "head_limit": 1000},
                ).value
            self.assertTrue(record["truncated_by_chars"])
            for name in record["filenames"]:
                # Derived from the truncated visible page only: every listed
                # file appears in the visible content and really exists.
                self.assertIn(name, record["content"])
                self.assertIn(name, names)

    def test_grep_does_not_honor_stray_ignore_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _, registry = self._tools(Path(tmp))
            (paths.workspace / ".ignore").write_text("sub/\n", encoding="utf-8")
            (paths.workspace / "sub").mkdir()
            (paths.workspace / "sub" / "data.txt").write_text("needle\n", encoding="utf-8")
            # A stray .ignore in a data root must not silently filter grep
            # results out of sync with the glob walker.
            files = registry.invoke(
                "grep", {"pattern": "needle", "root": "workspace", "output_mode": "files"}
            ).value
            self.assertIn("workspace/sub/data.txt", files["filenames"])
            listing = registry.invoke("glob", {"pattern": "**/*.txt", "root": "workspace"}).value
            self.assertIn("workspace/sub/data.txt", listing["filenames"])

    def test_pit_and_prior_fold_roots_are_reachable(self) -> None:
        # The whole point of the multi-root design: authoring against PIT views
        # and inherited artifacts without shelling out.
        with tempfile.TemporaryDirectory() as tmp:
            paths, roots, registry = self._tools(Path(tmp))
            (paths.current_snapshot / "manifest.json").write_text('{"kind": "decision_input"}\n', encoding="utf-8")
            (paths.parent_output / "main.py").write_text("# parent strategy\n", encoding="utf-8")
            (paths.results / "valid_000.json").write_text('{"total_return": 0.0}\n', encoding="utf-8")
            (paths.steps / "tree.txt").write_text("- epoch_001__fold_ref_ab__run_x__valid_000\n", encoding="utf-8")
            for root in ("snapshot", "parent_output", "results", "steps"):
                self.assertIn(root, roots.names)
            snapshot = registry.invoke("read_file", {"root": "snapshot", "path": "manifest.json"})
            self.assertTrue(snapshot.ok, snapshot.error)
            self.assertIn("decision_input", snapshot.value["content"])
            steps = registry.invoke("read_file", {"root": "steps", "path": "tree.txt"})
            self.assertIn("fold_ref_ab", steps.value["content"])

    def test_standalone_authoring_degrades_to_the_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "work"
            workspace_dir.mkdir()
            (workspace_dir / "main.py").write_text("# standalone\n", encoding="utf-8")
            roots = SearchRoots(SafeWorkspace(workspace_dir))
            self.assertEqual(roots.names, ("workspace",))
            registry = ToolRegistry([ReadFileTool(roots)])
            result = registry.invoke("read_file", {"path": "main.py"})
            self.assertTrue(result.ok, result.error)
            self.assertIn("standalone", result.value["content"])


class ArtifactIOToolTest(unittest.TestCase):
    def _registry(self, root: Path):
        paths, _, workspace = build_sandbox(root)
        return paths, ToolRegistry([WriteFileTool(workspace), EditFileTool(workspace)])

    def test_write_then_edit_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, registry = self._registry(Path(tmp))
            written = registry.invoke(
                "write_file", {"path": "output/helpers/sig.py", "content": "x = 1\ny = 2\n"}
            )
            self.assertTrue(written.ok, written.error)
            self.assertTrue(written.value["created"])
            self.assertGreater(written.value["bytes_written"], 0)
            self.assertTrue((paths.agent_output / "helpers" / "sig.py").exists())
            edited = registry.invoke(
                "edit_file",
                {"path": "output/helpers/sig.py", "old_text": "x = 1", "new_text": "x = 42"},
            )
            self.assertTrue(edited.ok, edited.error)
            self.assertEqual(edited.value["replacements"], 1)
            self.assertIn(
                "x = 42", (paths.agent_output / "helpers" / "sig.py").read_text(encoding="utf-8")
            )
            overwrite = registry.invoke(
                "write_file", {"path": "output/helpers/sig.py", "content": "x = 0\n"}
            )
            self.assertFalse(overwrite.value["created"])

    def test_edit_missing_and_stale_are_typed_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, registry = self._registry(Path(tmp))
            miss = registry.invoke(
                "edit_file", {"path": "output/nope.py", "old_text": "a", "new_text": "b"}
            )
            self.assertFalse(miss.ok)
            registry.invoke("write_file", {"path": "t.txt", "content": "hello world"})
            stale = registry.invoke(
                "edit_file", {"path": "t.txt", "old_text": "absent", "new_text": "x"}
            )
            self.assertFalse(stale.ok)
            self.assertEqual(stale.value["error_type"], "stale")
            self.assertTrue(stale.value["retry_hint"])

    def test_edit_ambiguous_requires_replace_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, registry = self._registry(Path(tmp))
            registry.invoke("write_file", {"path": "d.txt", "content": "a\na\na"})
            ambiguous = registry.invoke(
                "edit_file", {"path": "d.txt", "old_text": "a", "new_text": "b"}
            )
            self.assertFalse(ambiguous.ok)
            self.assertEqual(ambiguous.value["error_type"], "ambiguous")
            self.assertEqual(ambiguous.value["details"]["matches"], 3)
            replaced = registry.invoke(
                "edit_file",
                {"path": "d.txt", "old_text": "a", "new_text": "b", "replace_all": True},
            )
            self.assertTrue(replaced.ok, replaced.error)
            self.assertEqual(replaced.value["replacements"], 3)

    def test_write_rejects_escape_and_readonly_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, registry = self._registry(Path(tmp))
            escape = registry.invoke(
                "write_file", {"path": "output/../../snapshot/x.py", "content": "x"}
            )
            self.assertFalse(escape.ok)
            self.assertFalse((paths.current_snapshot / "x.py").exists())
            readonly = registry.invoke("write_file", {"path": "output/README.md", "content": "x"})
            self.assertFalse(readonly.ok)
            self.assertEqual(readonly.value["error_type"], "readonly")
            self.assertEqual(
                (paths.agent_output / "README.md").read_text(encoding="utf-8"), "readonly\n"
            )

    def test_write_rejects_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, registry = self._registry(Path(tmp))
            result = registry.invoke(
                "write_file", {"path": "/mnt/agent/output/abs_bug.py", "content": "x = 1\n"}
            )
            self.assertFalse(result.ok)
            self.assertFalse((paths.agent_output / "mnt").exists())

    def test_write_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, registry = self._registry(Path(tmp))
            outside = Path(tmp) / "outside"
            outside.mkdir()
            os.symlink(outside, paths.workspace / "escape")
            result = registry.invoke("write_file", {"path": "workspace/escape/x.py", "content": "x"})
            self.assertFalse(result.ok)
            self.assertFalse((outside / "x.py").exists())

    def test_write_rejects_oversized_content(self) -> None:
        from autotrade.environment.tools.files import MAX_WRITE_CHARS

        with tempfile.TemporaryDirectory() as tmp:
            paths, registry = self._registry(Path(tmp))
            result = registry.invoke(
                "write_file", {"path": "big.txt", "content": "x" * (MAX_WRITE_CHARS + 1)}
            )
            self.assertFalse(result.ok)
            self.assertFalse((paths.agent / "big.txt").exists())

    def test_write_refusals_carry_an_actionable_error_type(self) -> None:
        # The structured ToolError fields exist so the Agent can act on the
        # refusal instead of re-reading prose. A refusal that degrades to the
        # generic "tool_error" tells it nothing.
        from autotrade.environment.tools.files import MAX_WRITE_CHARS

        with tempfile.TemporaryDirectory() as tmp:
            paths, registry = self._registry(Path(tmp))
            outside = Path(tmp) / "outside"
            outside.mkdir()
            os.symlink(outside, paths.workspace / "escape")
            cases = (
                ("path_error", "write_file", {"path": "output/../../snapshot/x.py", "content": "x"}),
                ("path_error", "write_file", {"path": "/mnt/agent/output/abs.py", "content": "x"}),
                ("path_error", "write_file", {"path": "workspace/escape/x.py", "content": "x"}),
                ("too_large", "write_file", {"path": "b.txt", "content": "x" * (MAX_WRITE_CHARS + 1)}),
                ("not_found", "edit_file", {"path": "nope.py", "old_text": "a", "new_text": "b"}),
            )
            observed = {}
            for expected, tool, arguments in cases:
                result = registry.invoke(tool, arguments)
                self.assertFalse(result.ok, (tool, arguments))
                observed[(tool, expected, str(arguments["path"]))] = result.value["error_type"]
            self.assertEqual(
                observed,
                {key: key[1] for key in observed},
                "write/edit refusals degraded to the generic tool_error",
            )

    def test_workspace_guard_rejects_traversal_hidden_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _ = self._registry(Path(tmp))
            outside = Path(tmp) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")
            os.symlink(outside, paths.agent / "link")
            safe = SafeWorkspace(paths.agent)
            for path in ("../outside.txt", str(outside), ".env", "link"):
                with self.subTest(path=path), self.assertRaises(ToolError):
                    safe.resolve(path, must_exist=path == "link")


if __name__ == "__main__":
    unittest.main()


class TerminalToolWriteLockTest(unittest.TestCase):
    """Finishing locks the workspace: after a terminal tool succeeds, the
    registry refuses every mutating tool while read-only ones keep working."""

    def _finish_tool(self, root: Path):
        from autotrade.environment.artifacts import new_revision_id
        from autotrade.environment.step_tree import StepTree
        from autotrade.environment.tools import FinishFoldTool

        paths, roots, workspace = build_sandbox(root)
        (paths.agent_output / "main.py").write_text(
            "def generate_orders(context):\n    return []\n", encoding="utf-8"
        )
        tree = StepTree(paths.steps)
        node_id = tree.record_step(
            paths.agent_output,
            epoch_id="epoch_001",
            fold_id="fold_ref_ab",
            run_id="run_x",
            result_name="valid_000",
            revision_id=new_revision_id("revision"),
            metrics={},
        )
        finish = FinishFoldTool(tree, fold_id="fold_ref_ab", run_id="run_x")
        return workspace, roots, finish, node_id

    def test_finish_locks_mutating_tools_and_leaves_reads_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, roots, finish, node_id = self._finish_tool(Path(tmp))
            registry = ToolRegistry(
                [WriteFileTool(workspace), EditFileTool(workspace), ReadFileTool(roots), finish]
            )
            # Before finishing, writes go through.
            assert registry.invoke("write_file", {"path": "draft.txt", "content": "x"}).ok
            finished = registry.invoke("finish_fold", {"node_id": node_id})
            assert finished.ok, finished.error
            assert finished.value["write_locked"] is True

            for tool, arguments in (
                ("write_file", {"path": "output/main.py", "content": "changed"}),
                ("edit_file", {"path": "draft.txt", "old_text": "x", "new_text": "y"}),
            ):
                denied = registry.invoke(tool, arguments)
                assert not denied.ok, tool
                assert "locked" in denied.error, tool
            # The formal artifact is untouched, and reads still work.
            assert "generate_orders" in (
                Path(tmp) / "agent/output/main.py"
            ).read_text(encoding="utf-8")
            assert registry.invoke("read_file", {"root": "output", "path": "main.py"}).ok

    def test_shell_is_locked_after_finish_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace, _roots, finish, node_id = self._finish_tool(Path(tmp))
            runner = FakeRunner()
            registry = ToolRegistry([SandboxShellTool(workspace, runner), finish])
            assert registry.invoke("shell", {"argv": ["echo", "before"]}).ok
            assert registry.invoke("finish_fold", {"node_id": node_id}).ok
            denied = registry.invoke("shell", {"argv": ["echo", "after"]})
            assert not denied.ok
            assert "locked" in denied.error
            assert len(runner.calls) == 1  # the locked call never reached the sandbox


class StrategyOrderContractTest(unittest.TestCase):
    """A sandboxed strategy cannot bypass the host order contract: the payload
    it returns is revalidated on the trusted side before it reaches the Broker."""

    def test_a_sandbox_payload_with_an_invalid_action_is_rejected(self) -> None:
        from autotrade.environment.strategy import CN_TZ, StrategyContractError, validate_order_payload

        inference_at = datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ)
        payload = [
            {
                "symbol": "000001.SZ",
                "action": "hold",
                "quantity": 100,
                "execute_at": "2026-01-02T09:30:00+08:00",
            }
        ]
        with self.assertRaisesRegex(StrategyContractError, "action"):
            validate_order_payload(payload, inference_at=inference_at)

    def test_the_replay_engine_revalidates_every_sandbox_payload(self) -> None:
        import pandas as pd

        from autotrade.environment.replay import BacktestError, run_daily_replay
        from autotrade.environment.strategy import StrategySchedule

        daily = pd.DataFrame(
            [{"trade_date": "20260102", "symbol": "000001.SZ", "open": 10.0, "close": 11.0}]
        )

        def rogue(_context):
            return [
                {
                    "symbol": "000001.SZ",
                    "action": "hold",
                    "quantity": 100,
                    "execute_at": "2026-01-02T09:30:00+08:00",
                }
            ]

        with self.assertRaises(BacktestError):
            run_daily_replay(daily=daily, strategy=rogue, schedule=StrategySchedule("day", "08:30"))


class SequentialDispatchClassificationTest(unittest.TestCase):
    """A batch runs concurrently only when no call in it must be ordered.

    ``daily_backtest`` commits a Step, so it declares ``mutating`` on its own
    spec rather than relying on the name list; the list is left holding only
    the tools that genuinely have no flag."""

    def test_daily_backtest_is_sequential_through_its_spec_flag(self) -> None:
        from autotrade.environment.tools import (
            SEQUENTIAL_TOOL_NAMES,
            is_sequential_tool,
        )
        from autotrade.pipelines.local_backend import FoldBacktestTool

        self.assertTrue(FoldBacktestTool.spec.mutating)
        self.assertTrue(is_sequential_tool(FoldBacktestTool.spec))
        self.assertNotIn("daily_backtest", SEQUENTIAL_TOOL_NAMES)

    def test_read_only_specs_are_concurrent_and_unknown_names_are_not(self) -> None:
        from autotrade.environment.tools import (
            SEQUENTIAL_TOOL_NAMES,
            ToolSpec,
            is_sequential_tool,
        )

        read_only = ToolSpec("read_file", "read", {"type": "object", "properties": {}})
        self.assertFalse(is_sequential_tool(read_only))
        # An unregistered name has no spec: its rejection keeps the batch order.
        self.assertTrue(is_sequential_tool(None))
        for name in SEQUENTIAL_TOOL_NAMES:
            spec = ToolSpec(name, "gate", {"type": "object", "properties": {}})
            self.assertFalse(spec.mutating, name)
            self.assertTrue(is_sequential_tool(spec), name)


class ToolContractHintTest(unittest.TestCase):
    """What the schema tells the model must match what the tools accept."""

    def test_inputs_resolve_under_the_default_workspace_root(self) -> None:
        # The real session layout: the writable workspace is the default root
        # and the session facts live in its `inputs/`, so the model's natural
        # `inputs/...` path resolves with no root at all.
        with tempfile.TemporaryDirectory() as tmp:
            paths = SandboxPaths(Path(tmp))
            for directory in (paths.workspace / "inputs", paths.logs):
                directory.mkdir(parents=True, exist_ok=True)
            (paths.workspace / "inputs" / "skills_index.json").write_text("[]\n", encoding="utf-8")
            roots = SearchRoots(SafeWorkspace(paths.workspace), paths=paths)
            self.assertNotIn("agent", search_module.SEARCH_ROOTS)
            self.assertEqual(roots.names[0], "workspace")
            registry = ToolRegistry([ReadFileTool(roots), GlobTool(roots)])
            read = registry.invoke("read_file", {"path": "inputs/skills_index.json"})
            self.assertTrue(read.ok, read.error)
            self.assertEqual(read.value["root"], "workspace")
            stale = registry.invoke(
                "read_file", {"root": "agent", "path": "inputs/skills_index.json"}
            )
            self.assertFalse(stale.ok)
            root_help = ReadFileTool(roots).spec.input_schema["properties"]["root"]["description"]
            self.assertIn("path='inputs/skills_index.json'", root_help)

    def test_shell_description_states_argv_shape_timeout_and_workspace_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, workspace = build_sandbox(Path(tmp))
            default = SandboxShellTool(workspace, FakeRunner()).spec.description
            custom = SandboxShellTool(workspace, FakeRunner(), timeout_seconds=5).spec.description
            self.assertIn('["python", "-c", "print(1)"]', default)
            self.assertIn("inside the workspace", default)
            self.assertIn("at most 30", default)
            self.assertIn("at most 5", custom)
