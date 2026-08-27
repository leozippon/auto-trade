import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from autotrade.environment.artifacts import (
    ArtifactError,
    ModificationConstraints,
    _atomic_replace_copied,
    _unlock_directory,
    copy_artifact,
    copy_model_artifacts,
    init_from_template,
    load_model_artifacts,
    load_strategy_artifact,
    modification_delta,
    model_artifact_delta,
    new_revision_id,
    restore_frozen_artifact_trees,
)

from .fixtures_sandbox import TEMPLATE_DIR

VALID_MAIN = """
def generate_orders(context):
    return []
"""


def write_artifact(root: Path, *, main: str = VALID_MAIN) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("readonly", encoding="utf-8")
    (root / "main.py").write_text(main, encoding="utf-8")
    (root / "notes.md").write_text("neutral when evidence is thin\n", encoding="utf-8")
    return root


class ArtifactContractTest(unittest.TestCase):
    def test_default_template_is_a_loadable_strategy_artifact(self):
        module = ModuleType("agent_output_template_main")
        source = (TEMPLATE_DIR / "main.py").read_text(encoding="utf-8")
        exec(compile(source, str(TEMPLATE_DIR / "main.py"), "exec"), module.__dict__)
        context = SimpleNamespace(
            bars=(
                {"symbol": "000002.SZ", "close": 20.0},
                {"symbol": "000001.SZ", "close": 10.0},
            ),
            account=SimpleNamespace(cash=100_000.0, positions={}),
            inference_at=None,
        )
        self.assertEqual(
            module._visible_prices(context), {"000002.SZ": 20.0, "000001.SZ": 10.0}
        )
        # A holder of any position emits nothing, so the template never
        # double-enters while flat logic is the only implemented lifecycle.
        held = SimpleNamespace(**{**context.__dict__})
        held.account = SimpleNamespace(cash=0.0, positions={"000001.SZ": 100})
        self.assertEqual(module.generate_orders(held), [])

    def test_loads_valid_artifact_directory_and_stamps_a_revision_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(Path(tmp))
            artifact = load_strategy_artifact(root)
            self.assertIn("main.py", artifact.files)
            self.assertTrue(artifact.revision_id.startswith("artifact_"))
            self.assertNotEqual(load_strategy_artifact(root).revision_id, artifact.revision_id)
            pinned = load_strategy_artifact(root, revision_id="revision_fixed")
            self.assertEqual(pinned.revision_id, "revision_fixed")
            self.assertTrue(new_revision_id("revision").startswith("revision_"))

    def test_rejects_missing_entrypoint_and_forbidden_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(Path(tmp), main="x = 1\n")
            with self.assertRaisesRegex(ArtifactError, "must define"):
                load_strategy_artifact(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(Path(tmp), main="def run_strategy(context):\n    return {}\n")
            with self.assertRaisesRegex(ArtifactError, "generate_orders"):
                load_strategy_artifact(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(
                Path(tmp),
                main='def generate_orders(context):\n    return open("/mnt/artifacts/x").read()\n',
            )
            with self.assertRaisesRegex(ArtifactError, "stage directories"):
                load_strategy_artifact(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(
                Path(tmp),
                main='def generate_orders(context):\n    return open("/mnt/agent/workspace/secret").read()\n',
            )
            with self.assertRaisesRegex(ArtifactError, "stage directories"):
                load_strategy_artifact(root)

    def test_forbidden_path_scan_ignores_docstrings(self):
        main = '''
"""Documentation may mention /mnt/artifacts without becoming executable access."""


def helper():
    """Function docs may mention /mnt/snapshots/ for user guidance."""
    return None


def generate_orders(context):
    helper()
    return []
'''
        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(Path(tmp), main=main)
            artifact = load_strategy_artifact(root)
            self.assertIn("main.py", artifact.files)

    def test_init_from_template_skips_runtime_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = Path(tmp) / "template"
            shutil.copytree(TEMPLATE_DIR, template)
            cache_dir = template / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "x.pyc").write_bytes(b"x")
            dest = Path(tmp) / "dest"
            init_from_template(template, dest)
            self.assertTrue((dest / "main.py").exists())
            self.assertFalse((dest / "__pycache__").exists())
            load_strategy_artifact(dest)

    def test_copy_artifact_skips_runtime_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_artifact(Path(tmp) / "source")
            cache_dir = source / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "x.pyc").write_bytes(b"x")
            dest = Path(tmp) / "dest"
            copy_artifact(source, dest)
            self.assertTrue((dest / "main.py").exists())
            self.assertFalse((dest / "__pycache__").exists())
            load_strategy_artifact(dest)

    def test_allows_subdirectories_and_rejects_unsupported_files_cache_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(Path(tmp))
            helper_dir = root / "helpers"
            helper_dir.mkdir()
            (helper_dir / "signals.py").write_text("def score(context):\n    return 0.0\n", encoding="utf-8")
            artifact = load_strategy_artifact(root)
            self.assertIn("helpers/signals.py", artifact.files)

        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(Path(tmp))
            (root / "lookup.csv").write_text("ts_code,score\n000001.SZ,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "unsupported"):
                load_strategy_artifact(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(Path(tmp))
            (root / "helpers").mkdir()
            (root / "helpers" / "bad.py").write_text(
                'def leak():\n    return open("/mnt/artifacts/x").read()\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ArtifactError, "stage directories"):
                load_strategy_artifact(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = write_artifact(Path(tmp))
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "x.pyc").write_bytes(b"x")
            with self.assertRaisesRegex(ArtifactError, "runtime cache"):
                load_strategy_artifact(root)

        if hasattr(Path, "symlink_to"):
            with tempfile.TemporaryDirectory() as tmp:
                root = write_artifact(Path(tmp))
                (root / "linked.py").symlink_to(root / "main.py")
                with self.assertRaisesRegex(ArtifactError, "symlinks"):
                    load_strategy_artifact(root)

    def test_modification_delta_counts_files_and_code_lines_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = write_artifact(Path(tmp) / "parent")
            work = Path(tmp) / "work"
            copy_artifact(parent, work)
            (work / "main.py").write_text(VALID_MAIN + "\n# new condition\n", encoding="utf-8")
            (work / "notes.md").write_text("short prompt\nwith detail\n", encoding="utf-8")

            delta = modification_delta(parent, work)
            self.assertEqual(set(delta.changed_files), {"main.py", "notes.md"})
            self.assertGreaterEqual(delta.diff_lines, 2)
            self.assertGreaterEqual(delta.code_diff_lines, 1)
            self.assertEqual(delta.total_files, 3)

    def test_constraints_ignore_factor_prior_counts_and_tighten_after_early_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = write_artifact(Path(tmp) / "parent")
            work = Path(tmp) / "work"
            copy_artifact(parent, work)
            (work / "README.md").write_text("tampered", encoding="utf-8")
            delta = modification_delta(parent, work)
            allowed, reasons = ModificationConstraints().evaluate(delta)
            self.assertFalse(allowed)
            self.assertTrue(any("readonly" in reason for reason in reasons))

            loose = ModificationConstraints(max_diff_lines=1, early_max_diff_lines=100).for_epoch(1)
            strict = ModificationConstraints(max_diff_lines=1, early_max_diff_lines=100).for_epoch(3)
            self.assertEqual(loose.max_diff_lines, 100)
            self.assertEqual(strict.max_diff_lines, 1)

    def test_evaluate_reports_the_offending_numbers_not_a_bare_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = write_artifact(Path(tmp) / "parent")
            work = Path(tmp) / "work"
            copy_artifact(parent, work)
            for index in range(4):
                (work / f"extra_{index}.py").write_text(f"VALUE = {index}\n", encoding="utf-8")
            delta = modification_delta(parent, work)
            allowed, reasons = ModificationConstraints(max_changed_files=2).evaluate(delta)
            self.assertFalse(allowed)
            self.assertIn("changed files 4 > 2", reasons)

    def test_model_artifacts_are_separate_revisioned_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            (root / "ranker").mkdir(parents=True)
            (root / "params.json").write_text('{"alpha": 1}\n', encoding="utf-8")
            (root / "ranker" / "weights.pt").write_bytes(b"weights")

            artifact = load_model_artifacts(root)

            self.assertEqual(set(artifact.files), {"params.json", "ranker/weights.pt"})
            self.assertTrue(artifact.revision_id.startswith("models_"))
            self.assertGreater(artifact.total_bytes, 0)

            dest = Path(tmp) / "copied"
            copy_model_artifacts(root, dest)
            self.assertEqual(set(load_model_artifacts(dest).files), set(artifact.files))

            delta = model_artifact_delta(Path(tmp) / "empty_parent", dest)
            self.assertEqual(delta.total_files, 2)
            self.assertEqual(set(delta.changed_files), {"params.json", "ranker/weights.pt"})

    def test_model_comparison_streams_without_whole_file_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            large_payload = (b"0123456789abcdef" * (1024 * 128)) + b"tail"
            parent = tmp_path / "parent_models"
            work = tmp_path / "work_models"
            parent.mkdir()
            work.mkdir()
            (parent / "weights.bin").write_bytes(large_payload)
            (work / "weights.bin").write_bytes(large_payload)

            with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
                self.assertEqual(model_artifact_delta(parent, work).changed_files, ())

                with (work / "weights.bin").open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"X")
                self.assertEqual(model_artifact_delta(parent, work).changed_files, ("weights.bin",))

    def test_model_artifacts_reject_hidden_cache_and_unsupported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            (root / "data.parquet").write_bytes(b"not a model")
            with self.assertRaisesRegex(ArtifactError, "unsupported"):
                load_model_artifacts(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "unsupported"):
                load_model_artifacts(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            root.mkdir()
            (root / ".secret.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "hidden"):
                load_model_artifacts(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            hidden_dir = root / ".hidden"
            hidden_dir.mkdir(parents=True)
            (hidden_dir / "params.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "hidden"):
                load_model_artifacts(root)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "models"
            cache = root / "nested" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "x.pyc").write_bytes(b"x")
            with self.assertRaisesRegex(ArtifactError, "runtime cache"):
                load_model_artifacts(root)


class FrozenRestoreReplaceTest(unittest.TestCase):
    def test_restore_frozen_artifact_trees_replaces_and_relocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            snapshot_output = write_artifact(tmp_path / "snap_out")
            snapshot_models = tmp_path / "snap_models"
            snapshot_models.mkdir()
            (snapshot_models / "weights.bin").write_bytes(b"snap")
            live_output = write_artifact(
                tmp_path / "frozen" / "output",
                main="def generate_orders(context):\n    return [1]\n",
            )
            live_models = tmp_path / "frozen" / "models"
            live_models.mkdir()
            (live_models / "old.bin").write_bytes(b"old")

            restore_frozen_artifact_trees(
                output_path=live_output,
                snapshot_output=snapshot_output,
                models_path=live_models,
                snapshot_models=snapshot_models,
            )

            self.assertEqual(
                (live_output / "main.py").read_text(encoding="utf-8"),
                (snapshot_output / "main.py").read_text(encoding="utf-8"),
            )
            self.assertEqual((live_models / "weights.bin").read_bytes(), b"snap")
            self.assertFalse((live_models / "old.bin").exists())
            self.assertEqual(
                stat.S_IMODE((live_output / "main.py").stat().st_mode) & 0o222, 0
            )
            self.assertEqual(
                stat.S_IMODE((live_models / "weights.bin").stat().st_mode) & 0o222, 0
            )

    def test_unlock_directory_propagates_parent_chmod_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "output"
            dest.mkdir()
            parent = dest.parent
            real_chmod = Path.chmod

            def chmod(self, mode, *args, **kwargs):
                if Path(self).resolve() == parent.resolve():
                    raise OSError(1, "chmod denied")
                return real_chmod(self, mode, *args, **kwargs)

            with patch.object(Path, "chmod", chmod):
                with self.assertRaises(OSError) as ctx:
                    _unlock_directory(dest)
            self.assertIn("chmod denied", str(ctx.exception))

    def test_atomic_replace_restores_backup_then_reraises_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_artifact(Path(tmp) / "source")
            dest = write_artifact(Path(tmp) / "dest")
            (dest / "main.py").write_text(VALID_MAIN + "# live\n", encoding="utf-8")
            real_rename = Path.rename

            def rename(self, target):
                if self.name.startswith(f".{dest.name}.restore_"):
                    raise OSError(1, "staging commit failed")
                return real_rename(self, target)

            with patch.object(Path, "rename", rename):
                with self.assertRaises(OSError) as ctx:
                    _atomic_replace_copied(source, dest, copy_artifact)
            self.assertIn("staging commit failed", str(ctx.exception))
            self.assertNotIn("failed to restore", str(ctx.exception))
            self.assertTrue(dest.is_dir())
            self.assertIn(
                "# live", (dest / "main.py").read_text(encoding="utf-8")
            )

    def test_atomic_replace_propagates_backup_restore_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = write_artifact(Path(tmp) / "source")
            dest = write_artifact(Path(tmp) / "dest")
            (dest / "main.py").write_text(VALID_MAIN + "# live\n", encoding="utf-8")
            real_rename = Path.rename

            def rename(self, target):
                if self.name.startswith(f".{dest.name}.restore_"):
                    raise OSError(1, "staging commit failed")
                if self.name.startswith(f".{dest.name}.backup_"):
                    raise OSError(1, "backup restore failed")
                return real_rename(self, target)

            with patch.object(Path, "rename", rename):
                with self.assertRaisesRegex(OSError, "failed to restore .* after replace failure"):
                    _atomic_replace_copied(source, dest, copy_artifact)
            self.assertFalse(dest.exists())
            backups = [
                path
                for path in dest.parent.iterdir()
                if path.name.startswith(f".{dest.name}.backup_")
            ]
            self.assertEqual(len(backups), 1)
            self.assertIn(
                "# live", (backups[0] / "main.py").read_text(encoding="utf-8")
            )
            self.assertFalse(
                any(
                    path.name.startswith(f".{dest.name}.restore_")
                    for path in dest.parent.iterdir()
                )
            )


if __name__ == "__main__":
    unittest.main()
