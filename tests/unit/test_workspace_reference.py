"""Workspace reference pack: the read-only ``refs/`` tree and the seeded ``output/``."""

from __future__ import annotations

import shutil
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from autotrade.environment.artifacts import (
    artifact_fingerprint,
    copy_artifact,
    readonly_baseline,
    restore_working_artifacts_writable,
)
from autotrade.environment.tools.base import ToolError
from autotrade.environment.tools.files import EditFileTool
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.environment.tools.workspace import SafeWorkspace
from autotrade.pipelines.local_backend import (
    install_workspace_reference,
    seed_output_from_starter,
    session_workspace_map,
)
from autotrade.webui.manager import _reclaim_sandbox_containers

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = REPO_ROOT / "configs" / "agent_output_template"

STARTER_MAIN = """from lib.pick import first


def generate_orders(context):
    return first(context)
"""
STARTER_PICK = """def first(context):
    del context
    return []
"""


def _pack(root: Path, *, starter: bool) -> Path:
    """A reference pack as the repository checks one in."""

    pack = root / "pack"
    pack.mkdir(parents=True)
    (pack / "README.md").write_text("pack notes\n", encoding="utf-8")
    if starter:
        lib = pack / "starter" / "lib"
        lib.mkdir(parents=True)
        (pack / "starter" / "main.py").write_text(STARTER_MAIN, encoding="utf-8")
        (lib / "__init__.py").write_text("", encoding="utf-8")
        (lib / "pick.py").write_text(STARTER_PICK, encoding="utf-8")
    return pack


def _seed_session_workspace(workspace: Path, pack: Path | None) -> tuple[Path, dict[str, str]]:
    """Seed ``output/`` the way a research session does, in the same order.

    The read-only baseline is digested last, so it pins the bytes the session
    actually received rather than the ones the template alone would have given.
    """

    output = workspace / "output"
    copy_artifact(TEMPLATE_DIR, output)
    seed_output_from_starter(output, pack)
    restore_working_artifacts_writable(output)
    return output, readonly_baseline(output)


class InstallWorkspaceReferenceTest(unittest.TestCase):
    def test_copies_seed_into_refs_and_does_not_touch_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "seed"
            seed.mkdir()
            (seed / "note.md").write_text("factor notes\n", encoding="utf-8")
            workspace = root / "workspace"
            output = workspace / "output"
            output.mkdir(parents=True)
            (output / "main.py").write_text("def generate_orders(context):\n    return []\n")
            install_workspace_reference(workspace, seed)
            self.assertEqual(
                (workspace / "refs" / "note.md").read_text(encoding="utf-8"),
                "factor notes\n",
            )
            self.assertEqual(
                (output / "main.py").read_text(encoding="utf-8"),
                "def generate_orders(context):\n    return []\n",
            )
            self.assertEqual([path.name for path in output.iterdir()], ["main.py"])
            self.assertFalse((output / "refs").exists())
            self.assertFalse((workspace / "inputs").exists())
            self.assertFalse((workspace / "models").exists())

    def test_missing_directory_fails(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            missing = Path(tmp) / "missing_seed"
            with self.assertRaisesRegex(FileNotFoundError, "workspace_reference"):
                install_workspace_reference(workspace, missing)
            self.assertFalse((workspace / "refs").exists())

    def test_unset_param_is_a_noop(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            (workspace / "output").mkdir()
            install_workspace_reference(workspace, "")
            install_workspace_reference(workspace, None)
            install_workspace_reference(workspace, "   ")
            self.assertFalse((workspace / "refs").exists())
            self.assertEqual(list((workspace / "output").iterdir()), [])

    def test_worker_requires_an_existing_repo_directory_when_set(self) -> None:
        from autotrade.pipelines.worker import _optional_workspace_reference

        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            self.assertEqual(_optional_workspace_reference("", repo), "")
            self.assertEqual(_optional_workspace_reference(None, repo), "")
            with self.assertRaises(FileNotFoundError):
                _optional_workspace_reference("missing_seed", repo)
            with self.assertRaisesRegex(ValueError, "must be a string"):
                _optional_workspace_reference(["seed"], repo)
            seed = repo / "seed"
            seed.mkdir()
            self.assertEqual(_optional_workspace_reference("seed", repo), "seed")

    def test_session_facts_include_refs_only_when_the_directory_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            mapping = session_workspace_map(workspace)
            self.assertNotIn("refs", mapping)
            (workspace / "refs").mkdir()
            mapping = session_workspace_map(workspace)
            self.assertEqual(mapping["refs"], "refs/")
            self.assertEqual(mapping["strategy"], "output/main.py")

    def test_create_form_does_not_render_the_param(self) -> None:
        from autotrade.webui.params_schema import parameter_schema

        fields = {
            field["key"]
            for group in parameter_schema()["groups"]
            for field in group["fields"]
        }
        self.assertNotIn("workspace_reference", fields)


class SeedOutputFromStarterTest(unittest.TestCase):
    """``output/`` starts as the pack's starter, so no arm has to port it."""

    def test_a_pack_with_a_starter_seeds_output_and_keeps_the_contract_readme(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = _pack(root, starter=True)
            workspace = root / "workspace"
            output, _baseline = _seed_session_workspace(workspace, pack)
            self.assertEqual(
                sorted(path.relative_to(output).as_posix() for path in output.rglob("*.py")),
                ["lib/__init__.py", "lib/pick.py", "main.py"],
            )
            self.assertEqual((output / "main.py").read_text(encoding="utf-8"), STARTER_MAIN)
            # The contract file is the template's, never the pack's own README.
            self.assertEqual(
                (output / "README.md").read_text(encoding="utf-8"),
                (TEMPLATE_DIR / "README.md").read_text(encoding="utf-8"),
            )

    def test_a_pack_without_a_starter_leaves_the_template_output(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = _pack(root, starter=False)
            workspace = root / "workspace"
            output, _baseline = _seed_session_workspace(workspace, pack)
            self.assertEqual(
                sorted(path.name for path in output.iterdir()), ["README.md", "main.py"]
            )
            self.assertEqual(
                (output / "main.py").read_text(encoding="utf-8"),
                (TEMPLATE_DIR / "main.py").read_text(encoding="utf-8"),
            )
            self.assertEqual(seed_output_from_starter(output, None), ())

    def test_nothing_under_the_workspace_is_mode_protected(self) -> None:
        """Every arm that fanned candidates out of a copy hit "is not writable"
        and recovered with the same ``chmod -R a+w``: ``cp`` reproduces the
        source's mode, so one 0o444 file anywhere under the workspace locks the
        typed writers out of every copy made from it. Nothing there carries a
        mode of its own -- the contract README included, which is held to its
        seeded digest instead."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = _pack(root, starter=True)
            workspace = root / "workspace"
            output, baseline = _seed_session_workspace(workspace, pack)
            install_workspace_reference(workspace, pack)
            for path in (workspace / "refs", output):
                for child in (path, *path.rglob("*")):
                    expected = 0o777 if child.is_dir() else 0o666
                    self.assertEqual(
                        stat.S_IMODE(child.stat().st_mode),
                        expected,
                        child.relative_to(workspace),
                    )
            self.assertEqual(
                (workspace / "refs" / "starter" / "main.py").read_text(encoding="utf-8"),
                STARTER_MAIN,
            )
            # The fan-out the arms actually run: copy2 reproduces the source
            # mode exactly as `cp` does under the sandbox's zero umask.
            candidate = workspace / "candidates" / "c_v15"
            (candidate / "lib").mkdir(parents=True)
            for relative in ("main.py", "lib/pick.py"):
                shutil.copy2(workspace / "refs" / "starter" / relative, candidate / relative)
            writer = EditFileTool(SafeWorkspace(workspace))
            edited = writer.invoke(
                {
                    "path": "candidates/c_v15/main.py",
                    "old_text": "return first(context)",
                    "new_text": "return first(context)[:15]",
                }
            )
            self.assertTrue(edited.ok, edited.error)
            # The README the Agent may not edit is refused by path, and the
            # session is still held to the bytes it was seeded with.
            with self.assertRaises(ToolError) as refused:
                writer.invoke({"path": "output/README.md", "old_text": "#", "new_text": "##"})
            self.assertEqual(refused.exception.error_type, "readonly")
            (output / "README.md").write_text("tampered\n", encoding="utf-8")
            check = ModificationCheckTool(
                output, parent_dir=TEMPLATE_DIR, readonly_baseline=baseline
            )
            with self.assertRaisesRegex(ToolError, "readonly files modified"):
                check.invoke({})

    def test_a_freshly_seeded_workspace_passes_the_smoke_backtest_gate(self) -> None:
        """``smoke_backtest`` runs ``modification_check`` before it replays, and
        the read-only baseline it is held to is digested after seeding."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = _pack(root, starter=True)
            workspace = root / "workspace"
            output, baseline = _seed_session_workspace(workspace, pack)
            check = ModificationCheckTool(
                output, parent_dir=TEMPLATE_DIR, readonly_baseline=baseline
            )
            result = check.invoke({})
            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.value["fingerprint"], artifact_fingerprint(output))
            self.assertEqual(result.value["delta"]["readonly_violations"], [])
            self.assertIn("lib/pick.py", result.value["delta"]["changed_files"])

    def test_the_host_restores_the_contract_file_a_session_lost(self) -> None:
        """One flattened ``rm -rf`` took ``output/`` down with the seeded
        README, and nothing in the session could write that file back: the
        typed writers refuse it by path and the check only compared digests,
        so every replay stayed blocked. The host owns those bytes, so the
        host restores them -- and says so in the result."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = _pack(root, starter=True)
            workspace = root / "workspace"
            output, baseline = _seed_session_workspace(workspace, pack)
            seeded_text = (output / "README.md").read_text(encoding="utf-8")
            check = ModificationCheckTool(
                output,
                parent_dir=TEMPLATE_DIR,
                readonly_baseline=baseline,
                readonly_seed=TEMPLATE_DIR,
            )

            (output / "README.md").unlink()
            result = check.invoke({})
            self.assertTrue(result.ok, result.error)
            self.assertEqual((output / "README.md").read_text(encoding="utf-8"), seeded_text)
            self.assertIn("README.md", str(result.value["restored_readonly"]))
            # One tree: the restored file is inside the counts and the address
            # the formal replay is pinned to.
            self.assertEqual(result.value["fingerprint"], artifact_fingerprint(output))
            self.assertEqual(result.value["delta"]["readonly_violations"], [])
            self.assertEqual(stat.S_IMODE((output / "README.md").stat().st_mode), 0o666)

            (output / "README.md").write_text("rewritten contract\n", encoding="utf-8")
            overwritten = check.invoke({})
            self.assertTrue(overwritten.ok, overwritten.error)
            self.assertEqual((output / "README.md").read_text(encoding="utf-8"), seeded_text)
            self.assertIn("README.md", str(overwritten.value["restored_readonly"]))

            # Nothing to restore, nothing reported.
            self.assertNotIn("restored_readonly", check.invoke({}).value)

    def test_a_contract_file_is_not_restored_from_a_seed_that_has_moved(self) -> None:
        """The seed of an initial artifact is the live repository template,
        which a maintainer may edit while the session runs. Writing those
        bytes would swap the contract the session was seeded with, so a moved
        seed restores nothing and the violation is reported as before."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = _pack(root, starter=True)
            workspace = root / "workspace"
            output, baseline = _seed_session_workspace(workspace, pack)
            moved = root / "moved_template"
            copy_artifact(TEMPLATE_DIR, moved)
            (moved / "README.md").write_text("a new contract section\n", encoding="utf-8")

            (output / "README.md").unlink()
            check = ModificationCheckTool(
                output,
                parent_dir=TEMPLATE_DIR,
                readonly_baseline=baseline,
                readonly_seed=moved,
            )
            with self.assertRaisesRegex(ToolError, "readonly files modified"):
                check.invoke({})
            self.assertFalse((output / "README.md").exists())


class ReclaimSandboxContainersTest(unittest.TestCase):
    def test_reclaim_filters_adm_experiment_label(self) -> None:
        listing = Mock(stdout="", returncode=0)
        with patch("autotrade.webui.manager.subprocess.run", return_value=listing) as run:
            self.assertEqual(_reclaim_sandbox_containers("exp_demo"), [])
        command = run.call_args_list[0].args[0]
        self.assertEqual(command[:3], ["docker", "ps", "-aq"])
        self.assertIn("label=adm.experiment=exp_demo", command)
        self.assertFalse(any("mq.experiment" in part for part in command))


if __name__ == "__main__":
    unittest.main()
