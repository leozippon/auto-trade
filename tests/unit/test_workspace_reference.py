"""Workspace reference seed copy and the sandbox reclaim label."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from autotrade.pipelines.local_backend import fold_workspace_map, install_workspace_reference
from autotrade.webui.manager import _reclaim_sandbox_containers


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

    def test_fold_facts_include_refs_only_when_the_directory_exists(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            mapping = fold_workspace_map(workspace)
            self.assertNotIn("refs", mapping)
            (workspace / "refs").mkdir()
            mapping = fold_workspace_map(workspace)
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
