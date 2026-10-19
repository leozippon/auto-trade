"""Offline contract of ops/zgca/zgca.py: the platform layer is a mock, no network.

The mock `instance` (the shared lifecycle entry, tested in ~/zgca-compute) and
the mock `dev` record their arguments and the instance they were pointed at,
and answer from the environment; the mock `zenith` only records, since the
workflow reaches the API through those two. The remote script's `tracked`
step, which deletes files, runs for real against a temporary root.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "ops" / "zgca" / "zgca.py"
SPEC = importlib.util.spec_from_file_location("zgca", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
zgca = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zgca  # dataclasses resolve the module's annotations through sys.modules
SPEC.loader.exec_module(zgca)

INSTANCE = zgca.INSTANCE_ID
MOCK_ZENITH = r"""#!/usr/bin/env python3
import json, os, sys
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"tool": "zenith", "args": sys.argv[1:]}) + "\n")
sys.exit("the workflow reaches the API only through instance and dev")
"""
MOCK_INSTANCE = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"tool": "instance", "args": args, "instance": os.environ.get("ZGCA_INSTANCE")}) + "\n")
if args[0] == "state":
    print(os.environ.get("MOCK_STATE", "RUNNING"))
sys.exit(int(os.environ.get("MOCK_INSTANCE_EXIT_" + args[0].upper(), "0")))
"""
MOCK_DEV = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
record = {"tool": "dev", "args": args, "instance": os.environ.get("ZGCA_INSTANCE")}
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps(record) + "\n")
if args[:1] == ["rsync"] and args[1] == "--dry-run":
    sys.stdout.write(os.environ.get("MOCK_RSYNC_DIFF", ""))
if args[:2] == ["bash", "-lc"]:
    sys.exit(int(os.environ.get("MOCK_STEP_EXIT_" + args[4].replace("-", "_").upper(), "0")))
"""
MOCK_CURL = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"tool": "curl", "args": args}) + "\n")
with open(args[args.index("-o") + 1], "wb") as out:
    out.write(os.environ.get("MOCK_CURL_BODY", "").encode())
"""
TOOLS = {"zenith": MOCK_ZENITH, "instance": MOCK_INSTANCE, "dev": MOCK_DEV, "curl": MOCK_CURL}


class ZgcaWorkflowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="zgca-workflow-")
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.tools = base / "tools"
        self.tools.mkdir()
        for name, text in TOOLS.items():
            (self.tools / name).write_text(text, encoding="utf-8")
            (self.tools / name).chmod(0o700)
        self.repo = base / "repo"
        self.repo.mkdir()
        self.log = base / "calls.jsonl"
        self.env = {name: value for name, value in os.environ.items() if not name.startswith(("ZGCA_", "ZENITH_", "MOCK_"))}
        # The DuckDB CLI the deploy ships is a fixture here: the mock curl writes
        # it and the pinned digest is the fixture's for the test's duration.
        body = "fixture, not the DuckDB CLI\n"
        self.env.update(MOCK_LOG=str(self.log), MOCK_CURL_BODY=body, PATH=str(self.tools) + os.pathsep + os.environ["PATH"])
        self.pinned = zgca.DUCKDB_CLI_SHA256
        zgca.DUCKDB_CLI_SHA256 = hashlib.sha256(body.encode()).hexdigest()
        self.addCleanup(setattr, zgca, "DUCKDB_CLI_SHA256", self.pinned)

    def calls(self, tool: str | None = None) -> list[dict[str, object]]:
        lines = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        return [call for call in map(json.loads, lines) if tool is None or call["tool"] == tool]

    def run_main(self, *argv: str, **env: str) -> tuple[int, str]:
        """Run the CLI against the mocks; (exit code, stderr)."""
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.env | env)
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                code = zgca.main(argv, repo_root=self.repo, tools=self.tools)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        return code, err.getvalue()

    # ----- instance lifecycle -----

    def test_up_and_down_are_the_shared_entry_for_the_pinned_instance(self) -> None:
        for command, action in (("up", "start"), ("down", "stop")):
            with self.subTest(command=command):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main(command)
                self.assertEqual(code, 0, err)
                self.assertEqual(self.calls(), [{"tool": "instance", "args": [action, INSTANCE], "instance": INSTANCE}])
                # Its unverified result (a refused request, a timeout, no SSH) is this command's failure.
                code, err = self.run_main(command, **{f"MOCK_INSTANCE_EXIT_{action.upper()}": "1"})
                self.assertEqual(code, 1)
                self.assertIn(f"instance {action} {INSTANCE} failed", err)

    def test_status_shows_the_instance_and_logs_in_only_when_it_is_running(self) -> None:
        self.assertEqual(self.run_main("status", MOCK_STATE="STOPPED")[0], 0)
        self.assertEqual([call["args"] for call in self.calls()], [["show", INSTANCE], ["state", INSTANCE]])
        self.log.unlink()
        self.assertEqual(self.run_main("status")[0], 0)
        self.assertEqual([(call["tool"], call["args"][4] if call["tool"] == "dev" else call["args"][0]) for call in self.calls()],
                         [("instance", "show"), ("instance", "state"), ("dev", "status")])
        self.log.unlink()
        code, err = self.run_main("status", MOCK_INSTANCE_EXIT_SHOW="1")
        self.assertEqual(code, 1)
        self.assertIn(f"instance show {INSTANCE} failed", err)
        self.assertEqual(len(self.calls()), 1)

    def test_deploy_run_and_sync_refuse_an_instance_that_is_not_running_before_any_transfer(self) -> None:
        self._tracked_repo()
        self._seed()
        (self.repo / ".env").write_text("KEY=value\n", encoding="utf-8")
        for argv in (("deploy", "data/pit_views_seed_test"), ("run", "scripts/experiments/create_round_test.py"), ("sync", "arm_new")):
            for env, reason in (({"MOCK_STATE": "STOPPING"}, "is STOPPING, not RUNNING"), ({"MOCK_INSTANCE_EXIT_STATE": "1"}, "instance state")):
                with self.subTest(command=argv[0], reason=reason):
                    code, err = self.run_main(*argv, **env)
                    self.assertEqual(code, 1)
                    self.assertIn(reason, err)
        self.assertFalse(self.calls("dev"))
        self.assertEqual({tuple(call["args"]) for call in self.calls("instance")}, {("state", INSTANCE)})

    # ----- deploy -----

    def _tracked_repo(self) -> None:
        (self.repo / "scripts" / "experiments").mkdir(parents=True)
        (self.repo / "scripts" / "experiments" / "create_round_test.py").write_text("ROUND = None\n", encoding="utf-8")
        (self.repo / "untracked.py").write_text("", encoding="utf-8")
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run([*git, "init", "-q"], cwd=self.repo, check=True)
        subprocess.run([*git, "add", "scripts"], cwd=self.repo, check=True)
        subprocess.run([*git, "commit", "-q", "-m", "fixture"], cwd=self.repo, check=True)

    def _seed(self, name: str = "data/pit_views_seed_test", generation: str = "gen_test", *, release: bool = True) -> None:
        seed = self.repo / name
        seed.mkdir(parents=True, exist_ok=True)
        (seed / "provider.json").write_text(json.dumps({"generation_id": generation}), encoding="utf-8")
        if release:
            release_dir = self.repo / zgca.RELEASES / generation
            release_dir.mkdir(parents=True, exist_ok=True)
            (release_dir / "manifest.json").write_text("{}", encoding="utf-8")

    def test_deploy_refuses_before_any_transfer(self) -> None:
        self._tracked_repo()
        refusals = []
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "existing directory"))
        self._seed(release=False)
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "not published"))
        self._seed()
        (self.repo / "data/pit_views_seed_test" / zgca.UNFINISHED_SEED_MARKER).write_text("{}", encoding="utf-8")
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "unfinished prebuild"))
        (self.repo / "data/pit_views_seed_test" / zgca.UNFINISHED_SEED_MARKER).unlink()
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), ".env is missing"))
        (self.repo / ".env").write_text("KEY=value\n", encoding="utf-8")
        refusals.append((self.run_main("deploy", "/etc/passwd"), "below data/"))
        refusals.append((self.run_main("deploy", "scripts/experiments"), "below data/"))
        # A shipped DuckDB CLI that is not the pinned release is refused, not refetched.
        cli = self.repo / ".runtime" / "zgca" / zgca.DUCKDB_CLI_NAME
        cli.parent.mkdir(parents=True)
        cli.write_text("another release\n", encoding="utf-8")
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "not the pinned"))
        cli.unlink()
        for (code, err), reason in refusals:
            with self.subTest(reason=reason):
                self.assertEqual(code, 1)
                self.assertIn(reason, err)
        self.assertFalse(self.calls())

    def test_deploy_carries_tracked_files_env_seed_and_release_then_prepares(self) -> None:
        self._tracked_repo()
        self._seed()
        (self.repo / ".env").write_text("KEY=value\n", encoding="utf-8")
        code, err = self.run_main("deploy", "data/pit_views_seed_test")
        self.assertEqual(code, 0, err)
        dev = self.calls("dev")
        self.assertTrue(all(call["instance"] == INSTANCE for call in dev))
        self.assertEqual(dev[0]["args"][2], zgca.REMOTE_SCRIPT.read_text(encoding="utf-8"))
        # Transfers (by destination) and remote steps in the order they ran: the
        # tracked list lands before the `tracked` step prunes by it, and both
        # come before the seed and release.
        remote = zgca.REMOTE_ROOT
        ran = [call["args"][-1] if call["args"][0] == "rsync" else call["args"][4:]
               for call in dev if call["args"][1] != "--dry-run"]
        self.assertEqual(ran, [
            ["dirs"], f"dev:{remote}/", f"dev:{remote}/.env", f"dev:{remote}/.runtime/zgca/", ["tracked"],
            f"dev:{remote}/data/pit_views_seed_test/", f"dev:{remote}/data/research_releases/gen_test/",
            ["prepare", zgca.DUCKDB_CLI_SHA256],
        ])
        transfers = [call for call in dev if call["args"][:1] == ["rsync"] and call["args"][1] != "--dry-run"]
        state = self.repo / ".runtime" / "zgca"
        manifest = state / zgca.TRACKED_LIST_NAME
        self.assertEqual(manifest.read_bytes(), b"scripts/experiments/create_round_test.py\0")
        self.assertIn(f"--files-from={manifest}", transfers[0]["args"])
        self.assertIn("--from0", transfers[0]["args"])
        self.assertEqual(transfers[2]["args"][-3:-1], [str(state / zgca.DUCKDB_CLI_NAME), str(manifest)])
        self.assertIn("--delete", transfers[3]["args"])
        self.assertIn("-H", transfers[4]["args"])
        self.assertNotIn("--delete", transfers[0]["args"])
        # The release was fetched once, from the official URL, and is reused by the next deploy.
        fetches = self.calls("curl")
        self.assertEqual([call["args"][-1] for call in fetches], [zgca.DUCKDB_CLI_URL])
        self.assertTrue((state / zgca.DUCKDB_CLI_NAME).is_file())
        self.assertEqual(self.run_main("deploy", "data/pit_views_seed_test")[0], 0)
        self.assertEqual(len(self.calls("curl")), 1)
        # Every transfer is followed by its own dry-run itemization.
        verifications = [call for call in dev if call["args"][:2] == ["rsync", "--dry-run"]]
        self.assertEqual([call["args"][3:] for call in verifications], [call["args"][1:] for call in transfers])

    def test_the_tracked_step_removes_only_untracked_files_below_tracked_directories(self) -> None:
        root = Path(self.temp.name) / "instance"
        files = {  # path on the instance: whether the shipped list names it
            "README.md": True,
            ".env": False,
            "scripts/experiments/create_round_kept.py": True,
            "scripts/experiments/create_round_retired.py": False,
            "src/pkg/mod.py": True,
            "src/pkg/__pycache__/mod.cpython-311.pyc": False,
            "src/pkg.egg-info/PKG-INFO": False,
            "data/README.md": True,  # a tracked file does not open a runtime directory
            "data/release/manifest.json": False,
            "experiments/arm/hitl/status.json": False,
            "logs/workers/arm.log": False,
        }
        for name in files:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("", encoding="utf-8")
        (root / "scripts" / "alias.py").symlink_to("experiments/create_round_kept.py")
        (root / "scripts" / "data_link").symlink_to("../data")
        script = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        line = f"ROOT={zgca.REMOTE_ROOT}\n"
        self.assertEqual(script.count(line), 1)
        script = script.replace(line, f"ROOT={root}\n")

        def step() -> subprocess.CompletedProcess[str]:
            return subprocess.run(["bash", "-c", script, "remote.sh", "tracked"], capture_output=True, text=True, check=False)

        missing = step()
        self.assertEqual(missing.returncode, 1)
        self.assertIn("no tracked file list", missing.stderr)
        listed = root / ".runtime" / "zgca" / zgca.TRACKED_LIST_NAME
        listed.parent.mkdir(parents=True)
        listed.write_bytes(b"".join(name.encode() + b"\0" for name, named in files.items() if named))
        pruned = step()
        self.assertEqual(pruned.returncode, 0, pruned.stderr)
        self.assertIn("1 stale removed under scripts, src: scripts/experiments/create_round_retired.py", pruned.stdout)
        left = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink() or path.is_file()}
        self.assertEqual(left, set(files) - {"scripts/experiments/create_round_retired.py"}
                         | {"scripts/alias.py", "scripts/data_link", f".runtime/zgca/{zgca.TRACKED_LIST_NAME}"})

    def test_deploy_stops_at_a_transfer_that_left_differences(self) -> None:
        self._tracked_repo()
        self._seed()
        (self.repo / ".env").write_text("KEY=value\n", encoding="utf-8")
        code, err = self.run_main("deploy", "data/pit_views_seed_test", MOCK_RSYNC_DIFF=">f.st...... scripts/x.py\n")
        self.assertEqual(code, 1)
        self.assertIn("differs from its source", err)
        steps = [call["args"][4] for call in self.calls("dev") if call["args"][:2] == ["bash", "-lc"]]
        self.assertEqual(steps, ["dirs"])

    # ----- run, sync -----

    def test_run_forwards_the_launcher_arguments_and_refuses_an_untracked_round(self) -> None:
        self._tracked_repo()
        code, err = self.run_main("run", "untracked.py", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("not a tracked file", err)
        self.assertFalse(self.calls("dev"))
        code, err = self.run_main("run", "scripts/experiments/create_round_test.py", "--dry-run", "arm_one")
        self.assertEqual(code, 0, err)
        steps = [call["args"][4:] for call in self.calls("dev")]
        self.assertEqual(steps, [["console-up"], ["round", "scripts/experiments/create_round_test.py", "--dry-run", "arm_one"]])
        code, err = self.run_main("run", "scripts/experiments/create_round_test.py", MOCK_STEP_EXIT_CONSOLE_UP="1")
        self.assertEqual(code, 1)
        self.assertIn("console-up failed", err)
        self.assertEqual(self.calls()[-1]["args"][4], "console-up")

    def test_sync_pulls_an_instance_arm_and_refuses_to_overwrite_a_local_one(self) -> None:
        local = self.repo / "experiments" / "arm_local"
        (local / "hitl").mkdir(parents=True)
        (local / "hitl" / "params.json").write_text(json.dumps({"experiments_root": str(self.repo / "experiments")}), encoding="utf-8")
        code, err = self.run_main("sync", "arm_local")
        self.assertEqual(code, 1)
        self.assertIn("arm of this host", err)
        (local / "hitl" / "params.json").unlink()
        code, err = self.run_main("sync", "arm_local")
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        code, err = self.run_main("sync", "../escape")
        self.assertEqual(code, 1)
        self.assertFalse(self.calls("dev"))
        copied = self.repo / "experiments" / "arm_zgca"
        (copied / "hitl").mkdir(parents=True)
        (copied / "hitl" / "params.json").write_text(json.dumps({"experiments_root": f"{zgca.REMOTE_ROOT}/experiments"}), encoding="utf-8")
        for arm in ("arm_new", "arm_zgca"):
            with self.subTest(arm=arm):
                code, err = self.run_main("sync", arm)
                self.assertEqual(code, 0, err)
                dev = self.calls("dev")
                self.assertEqual(dev[-5]["args"][4:], ["arm-state", arm])
                pull = dev[-4]["args"]
                self.assertEqual(pull[-2:], [f"dev:{zgca.REMOTE_ROOT}/experiments/{arm}/", f"{self.repo / 'experiments' / arm}/"])
                self.assertEqual(pull[pull.index("--exclude") + 1], "/pit_views/")
                self.assertNotIn("--delete", pull)
                self.assertEqual(dev[-2]["args"][-2:], [f"dev:{zgca.REMOTE_ROOT}/logs/workers/{arm}.log", f"{self.repo / 'logs' / 'workers' / arm}.log"])
                self.log.unlink()

    def test_sync_stops_before_any_transfer_when_the_arm_state_step_refuses(self) -> None:
        code, err = self.run_main("sync", "arm_running", MOCK_STEP_EXIT_ARM_STATE="1")
        self.assertEqual(code, 1)
        self.assertIn("remote step arm-state failed", err)
        self.assertEqual([call["args"][:1] for call in self.calls("dev")], [["bash"]])

    def test_the_arm_state_step_refuses_an_arm_whose_worker_runs(self) -> None:
        """The console's own notion of a running arm decides, run for real against a temporary root."""
        from autotrade.pipelines.hitl_state import proc_start_ticks

        base = Path(self.temp.name)
        root, venv = base / "instance", base / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        (venv / "bin" / "python").chmod(0o700)
        live = {"state": "running_session", "pid": os.getpid(), "pid_start_ticks": proc_start_ticks(os.getpid())}
        arms = {
            "arm_live": live,
            # A worker that is gone no longer writes, whatever its last state said.
            "arm_gone": {**live, "pid_start_ticks": live["pid_start_ticks"] + 1},
            "arm_done": {"state": "completed", "verdict": "pass", "pid": None},
        }
        for arm, record in arms.items():
            (root / "experiments" / arm / "hitl").mkdir(parents=True)
            (root / "experiments" / arm / "hitl" / "status.json").write_text(json.dumps(record), encoding="utf-8")
        script = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        for line, replacement in ((f"ROOT={zgca.REMOTE_ROOT}\n", f"ROOT={root}\n"),
                                  ("VENV=/personal_fast_storage/envs/quant\n", f"VENV={venv}\n")):
            self.assertEqual(script.count(line), 1)
            script = script.replace(line, replacement)

        def step(arm: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(["bash", "-c", script, "remote.sh", "arm-state", arm], capture_output=True, text=True, check=False)

        refused = step("arm_live")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("arm_live: state running_session", refused.stdout)
        self.assertIn("arm_live is still running", refused.stderr)
        for arm in ("arm_gone", "arm_done"):
            with self.subTest(arm=arm):
                shown = step(arm)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                self.assertIn(f"{arm}: state {arms[arm]['state']}", shown.stdout)
        missing = step("arm_absent")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("no arm arm_absent", missing.stderr)

    def test_the_tools_must_exist_and_the_remote_script_rejects_unknown_steps(self) -> None:
        for name in ("zenith", "instance", "dev"):
            with self.subTest(tool=name):
                (self.tools / name).unlink()
                code, err = self.run_main("status")
                self.assertEqual(code, 1)
                self.assertIn(f"{self.tools / name} is not executable", err)
                (self.tools / name).write_text(TOOLS[name], encoding="utf-8")
                (self.tools / name).chmod(0o700)
        self.assertFalse(self.calls())
        with self.assertRaises(SystemExit) as caught:
            zgca.main([], repo_root=self.repo, tools=self.tools)
        self.assertEqual(caught.exception.code, 2)
        remote = subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), "bogus"], capture_output=True, text=True, check=False)
        self.assertEqual(remote.returncode, 1)
        self.assertIn("unknown remote step", remote.stderr)
        self.assertEqual(subprocess.run(["bash", "-n", str(zgca.REMOTE_SCRIPT)], check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
