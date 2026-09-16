"""Workspace reference pack: the read-only ``refs/`` tree and the seeded ``output/``."""

from __future__ import annotations

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
from autotrade.environment.tools.modification_check import ModificationCheckTool
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

    def test_the_seeded_copy_is_writable_while_the_reference_copy_stays_read_only(self) -> None:
        """The whole point: the Agent edits ``output/``, and ``refs/starter/``
        stays a read-only reference it never has to copy out of."""

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = _pack(root, starter=True)
            workspace = root / "workspace"
            output, _baseline = _seed_session_workspace(workspace, pack)
            install_workspace_reference(workspace, pack)
            self.assertEqual(stat.S_IMODE((output / "main.py").stat().st_mode), 0o666)
            self.assertEqual(stat.S_IMODE((output / "lib").stat().st_mode), 0o777)
            reference = workspace / "refs" / "starter" / "main.py"
            self.assertEqual(reference.read_text(encoding="utf-8"), STARTER_MAIN)
            self.assertEqual(stat.S_IMODE(reference.stat().st_mode), 0o444)
            self.assertEqual(
                stat.S_IMODE((workspace / "refs" / "README.md").stat().st_mode), 0o444
            )

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
