"""The meta-learning derived-image subsystem, end to end and reachable.

`maybe_rebuild_sandbox_image` and `write_sandbox_environment_example` were 373
lines with no production caller, `sandbox_image_update` was projected and
rendered but never written, and three docs described it as live. These tests
pin the outcome record for every branch that does not need Docker, and the
Agent-visible projection of it.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from autotrade.environment.runtime import RunManifest, SandboxPaths
from autotrade.environment.sandbox import DEFAULT_IMAGE, SandboxSpec
from autotrade.environment.sandbox_images import (
    SANDBOX_ENVIRONMENT_REQUEST_NAME,
    maybe_rebuild_sandbox_image,
    write_sandbox_environment_example,
)

REQUEST_REF = f"/mnt/agent/workspace/{SANDBOX_ENVIRONMENT_REQUEST_NAME}"


class SandboxEnvironmentRequestTest(unittest.TestCase):
    def _manifest(self, root: Path) -> RunManifest:
        return RunManifest.create(
            SandboxPaths(root).run_manifest, {"kind": "meta_learning", "experiment_id": "exp"}
        )

    def _rebuild(self, root: Path, request: object | None, **overrides):
        workspace = root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        request_path = workspace / SANDBOX_ENVIRONMENT_REQUEST_NAME
        if request is not None:
            request_path.write_text(json.dumps(request), encoding="utf-8")
        manifest = self._manifest(root)
        options = {
            "base_spec": SandboxSpec(gpu=None),
            "experiment_id": "exp",
            "epoch_id": "epoch_001",
            "experiment_dir": root / "experiment",
            "manifest": manifest,
            "use_docker": False,
            "rebuild_enabled": True,
            "timeout_seconds": 1800,
            "image_keep": 3,
        }
        options.update(overrides)
        result, spec = maybe_rebuild_sandbox_image(request_path, **options)
        return result, spec, manifest

    def test_the_example_the_meta_prompt_names_is_written_and_is_not_a_request(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            write_sandbox_environment_example(workspace)
            example = workspace / "sandbox_environment.example.json"
            self.assertTrue(example.is_file())
            payload = json.loads(example.read_text(encoding="utf-8"))
            self.assertTrue(set(payload) & {"python_packages", "npm_packages", "apt_packages"})
            # The example must never trigger a build: only the real name does.
            self.assertFalse((workspace / SANDBOX_ENVIRONMENT_REQUEST_NAME).exists())
            result, spec, _manifest = self._rebuild(Path(tmp), None)
            self.assertIsNone(result)
            self.assertEqual(spec.image, DEFAULT_IMAGE)

    def test_a_request_with_no_packages_is_skipped_and_recorded(self) -> None:
        with TemporaryDirectory() as tmp:
            result, spec, manifest = self._rebuild(Path(tmp), {"python_packages": []})
            self.assertEqual(result, {"status": "skipped_empty", "request_ref": REQUEST_REF})
            self.assertEqual(spec.image, DEFAULT_IMAGE)
            self.assertEqual(manifest.data["sandbox_image_update"]["status"], "skipped_empty")

    def test_local_development_records_the_request_without_touching_docker(self) -> None:
        with TemporaryDirectory() as tmp:
            result, spec, manifest = self._rebuild(
                Path(tmp), {"python_packages": ["scipy==1.14.0"]}, use_docker=False
            )
            self.assertEqual(result["status"], "skipped_local_dev")
            self.assertEqual(result["request_ref"], REQUEST_REF)
            self.assertEqual(spec.image, DEFAULT_IMAGE)
            self.assertEqual(manifest.data["sandbox_image_update"], result)

    def test_disabling_the_rebuild_records_disabled_rather_than_silently_ignoring(self) -> None:
        with TemporaryDirectory() as tmp:
            result, spec, manifest = self._rebuild(
                Path(tmp),
                {"python_packages": ["scipy==1.14.0"]},
                use_docker=True,
                rebuild_enabled=False,
            )
            self.assertEqual(result, {"status": "disabled", "request_ref": REQUEST_REF})
            self.assertEqual(spec.image, DEFAULT_IMAGE)
            self.assertEqual(manifest.data["sandbox_image_update"]["status"], "disabled")
            # No build directory was created.
            self.assertFalse((Path(tmp) / "experiment/sandbox_images").exists())

    def test_a_malformed_request_is_recorded_before_it_fails_the_run(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "rejected"):
                self._rebuild(Path(tmp), {"python_packages": ["scipy; rm -rf /"]})
            manifest = json.loads(
                (SandboxPaths(Path(tmp)).host_run_manifest).read_text(encoding="utf-8")
            )
            # Recorded BEFORE the hard failure: the audit trail survives fail-fast.
            self.assertEqual(manifest["sandbox_image_update"]["status"], "rejected")
            self.assertTrue(manifest["sandbox_image_update"]["reason"])

    def test_a_non_object_request_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                self._rebuild(Path(tmp), ["scipy"])

    def test_the_outcome_reaches_the_agent_visible_manifest_without_host_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _result, _spec, manifest = self._rebuild(
                root, {"python_packages": ["scipy==1.14.0"]}, use_docker=False
            )
            manifest.update(agents_md_sections_sha256="ignored legacy field")
            public = json.loads(SandboxPaths(root).run_manifest.read_text(encoding="utf-8"))
            update = public["sandbox_image_update"]
            self.assertEqual(update["status"], "skipped_local_dev")
            self.assertEqual(update["request_ref"], REQUEST_REF)
            # Host build coordinates stay out of the Agent-visible copy.
            for withheld in ("build_dir", "dockerfile_ref", "host_request_ref", "build_id",
                             "stdout_tail", "stderr_tail"):
                self.assertNotIn(withheld, update)
            self.assertNotIn(str(root), json.dumps(public))
            self.assertNotIn("agents_md_sections_sha256", public)
            # The host copy keeps everything for the audit.
            host = json.loads(SandboxPaths(root).host_run_manifest.read_text(encoding="utf-8"))
            self.assertEqual(host["sandbox_image_update"], manifest.data["sandbox_image_update"])

    def test_the_knobs_that_drive_it_are_real_config_fields(self) -> None:
        from autotrade.pipelines.config import RollingExperimentConfig

        config = RollingExperimentConfig(
            "exp", Path("/tmp/experiments"), "2022Q1", "2022Q1", "2023Q1", "2023Q1"
        )
        self.assertTrue(config.meta_sandbox_rebuild_enabled)
        self.assertEqual(config.meta_sandbox_rebuild_timeout_seconds, 1800)
        self.assertEqual(config.meta_sandbox_image_keep, 3)
        for field, value in (
            ("meta_sandbox_rebuild_timeout_seconds", -1),
            ("meta_sandbox_image_keep", -1),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                RollingExperimentConfig(
                    "exp", Path("/tmp/experiments"), "2022Q1", "2022Q1", "2023Q1", "2023Q1",
                    **{field: value},
                )

    def test_the_meta_prompt_and_the_meta_tool_set_make_the_request_writable(self) -> None:
        from autotrade.agent.prompts import META_SYSTEM_PROMPT
        from autotrade.agent.runner import _META_TOOLS

        # The subsystem is only reachable if the Meta session is told about it
        # AND can write the file: it had neither before R4.
        self.assertIn("sandbox_environment.json", META_SYSTEM_PROMPT)
        self.assertIn("sandbox_environment.example.json", META_SYSTEM_PROMPT)
        self.assertIn("write_file", _META_TOOLS)
        self.assertIn("edit_file", _META_TOOLS)


if __name__ == "__main__":
    unittest.main()
