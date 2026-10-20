"""Offline contract of ops/zgca/zgca.py: the platform layer is a mock, no network.

The mock `instance` (the shared lifecycle entry) and the mock `dev` (whose
copy, put, bootstrap and report primitives verify their own results) record
their arguments and the instance they were pointed at, and answer from the
environment; both are tested for real in ~/zgca-compute. The mock `zenith`
only records, since the workflow reaches the API through those two. The
remote script's `tracked`, `arm-state` and `status` steps run for real
against a temporary root.
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
if args[:1] == ["put"]:  # stdin is the file
    record["stdin"] = sys.stdin.buffer.read().decode()
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps(record) + "\n")
if "remote.sh" in args:  # env ZGCA_REMOTE_ROOT=<root> bash -lc <script> remote.sh <step> ...
    step = args[args.index("remote.sh") + 1]
    sys.exit(int(os.environ.get("MOCK_STEP_EXIT_" + step.replace("-", "_").upper(), "0")))
sys.exit(int(os.environ.get("MOCK_DEV_EXIT_" + args[0].upper(), "0")))
"""
PROXY = "https://proxy.invalid/"
MOCK_CURL = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"tool": "curl", "args": args}) + "\n")
# Through the proxy: as the origin does, or a failure, or other bytes.
answer = os.environ.get("MOCK_PROXY_ANSWER", "") if args[-1].startswith("https://proxy.invalid/") else ""
if answer == "fail":
    sys.exit(28)
with open(args[args.index("-o") + 1], "wb") as out:
    out.write(b"another release\n" if answer == "wrong" else os.environ.get("MOCK_CURL_BODY", "").encode())
"""
TOOLS = {"zenith": MOCK_ZENITH, "instance": MOCK_INSTANCE, "dev": MOCK_DEV, "curl": MOCK_CURL}
# A local .env as the host keeps it: data tokens beside the model gateway keys.
LOCAL_ENV = (
    "TUSHARE_TOKEN=tushare-relay\nTUSHARE_TOKEN_TMP=tushare-official\nHF_TOKEN=hf\n"
    "# DEEPSEEK_API_KEY=commented-out\nexport LIGHTING_API_KEY=relay-primary\nLIGHTING_API_KEY_ALT=relay-alt\n"
)
SHIPPED_ENV = "export LIGHTING_API_KEY=relay-primary\nLIGHTING_API_KEY_ALT=relay-alt\n"


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
        # The release files the deploy ships (the DuckDB CLI, the causal-conv1d
        # wheel) are fixtures here: the mock curl writes them and the pinned
        # digests are the fixture's for the test's duration.
        body = "fixture, not a release file\n"
        self.env.update(MOCK_LOG=str(self.log), MOCK_CURL_BODY=body, PATH=str(self.tools) + os.pathsep + os.environ["PATH"])
        for name in ("DUCKDB_CLI_SHA256", "CAUSAL_CONV1D_WHEEL_SHA256"):
            self.addCleanup(setattr, zgca, name, getattr(zgca, name))
            setattr(zgca, name, hashlib.sha256(body.encode()).hexdigest())

    def calls(self, tool: str | None = None) -> list[dict[str, object]]:
        lines = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        return [call for call in map(json.loads, lines) if tool is None or call["tool"] == tool]

    @staticmethod
    def remote(call: dict[str, object]) -> list[str] | None:
        """The remote.sh step and its arguments of a `dev env ZGCA_REMOTE_ROOT=<root> bash -lc <script> remote.sh ...` call, else None."""
        args = call["args"]
        return args[args.index("remote.sh") + 1:] if "remote.sh" in args else None

    def step(self, call: dict[str, object]) -> str:
        """A call's command: the remote.sh step for a remote step, else the entry's subcommand."""
        remote = self.remote(call)
        return remote[0] if remote is not None else call["args"][0]

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
        self.assertEqual([(call["tool"], self.step(call)) for call in self.calls()],
                         [("instance", "show"), ("instance", "state"), ("dev", "report"), ("dev", "status")])
        self.log.unlink()
        code, err = self.run_main("status", MOCK_INSTANCE_EXIT_SHOW="1")
        self.assertEqual(code, 1)
        self.assertIn(f"instance show {INSTANCE} failed", err)
        self.assertEqual(len(self.calls()), 1)

    def test_another_instance_is_addressed_by_every_entry_when_named(self) -> None:
        other = "secondinstance01"
        self.assertEqual(self.run_main("--instance", other, "status")[0], 0)
        calls = self.calls()
        self.assertEqual([call["tool"] for call in calls], ["instance", "instance", "dev", "dev"])
        self.assertTrue(all(call["instance"] == other for call in calls))
        self.assertEqual([call["args"][1] for call in calls if call["tool"] == "instance"], [other, other])
        # An empty or malformed id would let `dev` pick whichever machine runs.
        for bad in ("", "a b", "../x"):
            with self.subTest(instance=bad), self.assertRaises(SystemExit) as caught, \
                    contextlib.redirect_stderr(io.StringIO()):
                zgca.main(["--instance", bad, "status"], repo_root=self.repo, tools=self.tools)
            self.assertEqual(caught.exception.code, 2)
        self.assertEqual(len(self.calls()), 4)

    def test_another_remote_root_takes_every_transfer_and_remote_step(self) -> None:
        """ZGCA_REMOTE_ROOT, e.g. a validation deploy beside the live tree on the zone's shared storage."""
        other = "/personal_fast_storage/auto-trade-validate"
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        self._arm("arm_here")
        for argv in (("deploy", "data/pit_views_seed_test"), ("push", "arm_here"), ("sync", "arm_there")):
            code, err = self.run_main(*argv, ZGCA_REMOTE_ROOT=other)
            self.assertEqual(code, 0, err)
        dev = self.calls("dev")
        self.assertEqual({self.step(call) for call in dev},
                         {"bootstrap", "dirs", "copy", "put", "tracked", "prepare", "report", "arm-state"})
        for call in dev:
            args = call["args"]
            if self.remote(call) is not None:
                self.assertEqual(args[:2], ["env", f"ZGCA_REMOTE_ROOT={other}"])
            elif args[0] == "copy":
                (remote,) = [arg for arg in args if arg.startswith("dev:")]
                self.assertTrue(remote.startswith(f"dev:{other}/"), remote)
            elif args[0] == "put":
                self.assertEqual(args[1], f"{other}/.env")
        # Unset is the default root; anything but a directory directly below
        # personal storage is refused before any call, an empty value included.
        self.log.unlink()
        for bad in ("", "auto-trade-validate", "/tmp/auto-trade", "/personal_fast_storage", "/personal_fast_storage/",
                    "/personal_fast_storage/../auto-trade", "/personal_fast_storage/a/b", "/personal_fast_storage/a b"):
            with self.subTest(root=bad):
                code, err = self.run_main("status", ZGCA_REMOTE_ROOT=bad)
                self.assertEqual(code, 1)
                self.assertIn("ZGCA_REMOTE_ROOT=", err)
        self.assertFalse(self.calls())

    def test_deploy_run_and_sync_refuse_an_instance_that_is_not_running_before_any_transfer(self) -> None:
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        self._arm("arm_here")
        for argv in (("deploy", "data/pit_views_seed_test"), ("run", "scripts/experiments/create_round_test.py"),
                     ("sync", "arm_new"), ("push", "arm_here")):
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

    def _arm(self, arm: str, *, created_under: str | None = None, **status: object) -> None:
        hitl = self.repo / "experiments" / arm / "hitl"
        hitl.mkdir(parents=True)
        root = created_under or str(self.repo / "experiments")
        (hitl / "params.json").write_text(json.dumps({"experiments_root": root}), encoding="utf-8")
        (hitl / "status.json").write_text(json.dumps(status or {"state": "completed", "pid": None}), encoding="utf-8")

    def _pretrained(self) -> None:
        store = self.repo / zgca.PRETRAINED
        (store / "Model").mkdir(parents=True, exist_ok=True)
        (store / "Model" / "config.json").write_text("{}\n", encoding="utf-8")
        (store / zgca.PRETRAINED_MANIFEST).write_text("0" * 64 + "  Model/config.json\n", encoding="utf-8")

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
        (self.repo / ".env").write_text("TUSHARE_TOKEN=tushare-relay\n", encoding="utf-8")
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "none of the keys"))
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "data/pretrained/MANIFEST.sha256 is missing"))
        self._pretrained()
        refusals.append((self.run_main("deploy", "/etc/passwd"), "below data/"))
        refusals.append((self.run_main("deploy", "scripts/experiments"), "below data/"))
        # A shipped release file that is not the pinned release is refused, not refetched.
        state = self.repo / ".runtime" / "zgca"
        state.mkdir(parents=True)
        names = (zgca.DUCKDB_CLI_NAME, zgca.CAUSAL_CONV1D_WHEEL_NAME)
        for name in names:
            for other in names:
                (state / other).write_text("another release\n" if other == name else self.env["MOCK_CURL_BODY"], encoding="utf-8")
            refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), f"{name} has digest"))
        for name in names:
            (state / name).unlink()
        for (code, err), reason in refusals:
            with self.subTest(reason=reason):
                self.assertEqual(code, 1)
                self.assertIn(reason, err)
        self.assertFalse(self.calls())

    def test_deploy_carries_tracked_files_env_seed_release_and_store_then_prepares(self) -> None:
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        code, err = self.run_main("deploy", "data/pit_views_seed_test")
        self.assertEqual(code, 0, err)
        dev = self.calls("dev")
        self.assertTrue(all(call["instance"] == INSTANCE for call in dev))
        # Each remote step is this checkout's script in a login shell, told the root.
        remote = zgca.REMOTE_ROOT
        script = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        for call in dev:
            if self.remote(call) is not None:
                self.assertEqual(call["args"][:6], ["env", f"ZGCA_REMOTE_ROOT={remote}", "bash", "-lc", script, "remote.sh"])
        # The base tools first; then transfers (by destination) and remote steps
        # in the order they ran: the tracked list lands before the `tracked`
        # step prunes by it, and both come before the seed, the release and the
        # pretrained store; the host report last.
        ran = [self.remote(call) or call["args"][:1] + call["args"][1:][-1:] for call in dev]
        self.assertEqual(ran, [
            ["bootstrap"], ["dirs"], ["copy", f"dev:{remote}/"], ["put", f"{remote}/.env"],
            ["copy", f"dev:{remote}/.runtime/zgca/"], ["tracked"],
            ["copy", f"dev:{remote}/data/pit_views_seed_test/"], ["copy", f"dev:{remote}/data/research_releases/gen_test/"],
            ["copy", f"dev:{remote}/data/pretrained/"],
            ["prepare", zgca.DUCKDB_CLI_SHA256, zgca.CAUSAL_CONV1D_WHEEL_SHA256], ["report"],
        ])
        # Only the keys the model profiles name leave this host, through the
        # owner-only put: nothing else from the .env.
        (written,) = [call for call in dev if call["args"][0] == "put"]
        self.assertEqual(written["stdin"], SHIPPED_ENV)
        transfers = [call for call in dev if call["args"][0] == "copy"]
        state = self.repo / ".runtime" / "zgca"
        manifest = state / zgca.TRACKED_LIST_NAME
        self.assertEqual(manifest.read_bytes(), b"scripts/experiments/create_round_test.py\0")
        self.assertIn(f"--files-from={manifest}", transfers[0]["args"])
        self.assertIn("--from0", transfers[0]["args"])
        self.assertEqual(
            transfers[1]["args"][-4:-1],
            [str(state / zgca.DUCKDB_CLI_NAME), str(state / zgca.CAUSAL_CONV1D_WHEEL_NAME), str(manifest)],
        )
        # The seed, the release and the store are mirrored, hard links kept.
        for transfer in transfers[2:5]:
            self.assertIn("--delete", transfer["args"])
            self.assertIn("-H", transfer["args"])
        self.assertEqual(transfers[4]["args"][-2], f"{self.repo / zgca.PRETRAINED}/")
        self.assertNotIn("--delete", transfers[0]["args"])
        # Each release was fetched once, from its official URL, and is reused by the next deploy.
        fetches = self.calls("curl")
        self.assertEqual([call["args"][-1] for call in fetches], [zgca.DUCKDB_CLI_URL, zgca.CAUSAL_CONV1D_WHEEL_URL])
        self.assertTrue((state / zgca.DUCKDB_CLI_NAME).is_file())
        self.assertTrue((state / zgca.CAUSAL_CONV1D_WHEEL_NAME).is_file())
        self.assertEqual(self.run_main("deploy", "data/pit_views_seed_test")[0], 0)
        self.assertEqual(len(self.calls("curl")), 2)

    def test_release_files_come_through_the_proxy_first_then_github_and_only_with_the_pinned_digest(self) -> None:
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        state = self.repo / ".runtime" / "zgca"
        urls = (zgca.DUCKDB_CLI_URL, zgca.CAUSAL_CONV1D_WHEEL_URL)
        # The deploy stops at the stopped instance, after fetching.
        for answer, sources in (("", [PROXY + url for url in urls]),
                                ("fail", [s for url in urls for s in (PROXY + url, url)]),
                                ("wrong", [s for url in urls for s in (PROXY + url, url)])):
            with self.subTest(proxy=answer or "works"):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main("deploy", "data/pit_views_seed_test", MOCK_STATE="STOPPED",
                                          ZGCA_GITHUB_PROXY=PROXY, MOCK_PROXY_ANSWER=answer)
                self.assertIn("not RUNNING", err)
                self.assertEqual([call["args"][-1] for call in self.calls("curl")], sources)
                self.assertEqual({path.name for path in state.iterdir()} - {zgca.TRACKED_LIST_NAME},
                                 {zgca.DUCKDB_CLI_NAME, zgca.CAUSAL_CONV1D_WHEEL_NAME})
                for name in (zgca.DUCKDB_CLI_NAME, zgca.CAUSAL_CONV1D_WHEEL_NAME):
                    (state / name).unlink()
        # No source with the pinned bytes: nothing enters the cache, nothing is shipped.
        self.log.unlink()
        code, err = self.run_main("deploy", "data/pit_views_seed_test", ZGCA_GITHUB_PROXY=PROXY,
                                  MOCK_PROXY_ANSWER="wrong", MOCK_CURL_BODY="truncated\n")
        self.assertEqual(code, 1)
        self.assertIn(f"no source gave {zgca.DUCKDB_CLI_NAME} with the pinned digest", err)
        self.assertEqual([path.name for path in state.iterdir()], [zgca.TRACKED_LIST_NAME])
        self.assertFalse(self.calls("dev"))

    def test_the_remote_root_is_the_workflows_and_another_root_keeps_its_own_venv(self) -> None:
        script = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(script.count(f"ROOT={zgca.REMOTE_ROOT}\n"), 1)
        probe = script.replace("\nstep=${1:-}\n", '\necho "$ROOT $VENV"; exit 0\n')
        self.assertNotEqual(probe, script)
        other = "/personal_fast_storage/auto-trade-validate"
        live = f"{zgca.REMOTE_ROOT} /personal_fast_storage/envs/quant"
        for given, expected in ((None, live), (zgca.REMOTE_ROOT, live), (other, f"{other} {other}/.venv")):
            with self.subTest(root=given):
                env = {name: value for name, value in os.environ.items() if name != "ZGCA_REMOTE_ROOT"}
                shown = subprocess.run(["bash", "-c", probe, "remote.sh"], env=env | ({"ZGCA_REMOTE_ROOT": given} if given else {}),
                                       capture_output=True, text=True, check=False)
                self.assertEqual((shown.returncode, shown.stdout.strip()), (0, expected), shown.stderr)

    def test_the_tracked_step_removes_only_untracked_files_below_tracked_directories(self) -> None:
        root = Path(self.temp.name) / "instance"
        env = {**os.environ, "ZGCA_REMOTE_ROOT": str(root)}
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

        def step() -> subprocess.CompletedProcess[str]:
            return subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), "tracked"], env=env, capture_output=True, text=True, check=False)

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

    def test_the_image_build_and_the_instance_name_the_wheel_the_deploy_ships(self) -> None:
        """Three places name one file: the Dockerfile fetches it by default,
        the deploy pins and ships it, remote.sh verifies and serves it."""
        dockerfile = (REPO / "ops" / "docker" / "sandbox.Dockerfile").read_text(encoding="utf-8")
        self.assertIn(f"ARG CAUSAL_CONV1D_WHEEL_URL={zgca.CAUSAL_CONV1D_WHEEL_URL}\n", dockerfile)
        self.assertIn(f"/tmp/{zgca.CAUSAL_CONV1D_WHEEL_NAME}", dockerfile)
        remote = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        for name in (zgca.DUCKDB_CLI_NAME, zgca.CAUSAL_CONV1D_WHEEL_NAME):
            self.assertIn(f"=.runtime/zgca/{name}\n", remote)

    def test_deploy_stops_at_the_first_primitive_that_fails(self) -> None:
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        remote = zgca.REMOTE_ROOT
        for env, reason, ran in (
            ({"MOCK_DEV_EXIT_BOOTSTRAP": "100"}, "dev bootstrap failed (exit 100)", ["bootstrap"]),
            ({"MOCK_DEV_EXIT_COPY": "1"}, f"dev copy failed (exit 1): {self.repo}/ dev:{remote}/", ["bootstrap", "dirs", "copy"]),
            ({"MOCK_DEV_EXIT_PUT": "1"}, f"dev put failed (exit 1): {remote}/.env", ["bootstrap", "dirs", "copy", "put"]),
        ):
            with self.subTest(reason=reason):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main("deploy", "data/pit_views_seed_test", **env)
                self.assertEqual(code, 1)
                self.assertIn(reason, err)
                self.assertEqual([self.step(call) for call in self.calls("dev")], ran)

    # ----- run, sync -----

    def test_run_forwards_the_launcher_arguments_and_refuses_an_untracked_round(self) -> None:
        self._tracked_repo()
        code, err = self.run_main("run", "untracked.py", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("not a tracked file", err)
        self.assertFalse(self.calls("dev"))
        code, err = self.run_main("run", "scripts/experiments/create_round_test.py", "--dry-run", "arm_one")
        self.assertEqual(code, 0, err)
        steps = [self.remote(call) for call in self.calls("dev")]
        self.assertEqual(steps, [["console-up"], ["round", "scripts/experiments/create_round_test.py", "--dry-run", "arm_one"]])
        code, err = self.run_main("run", "scripts/experiments/create_round_test.py", MOCK_STEP_EXIT_CONSOLE_UP="1")
        self.assertEqual(code, 1)
        self.assertIn("console-up failed", err)
        self.assertEqual(self.step(self.calls()[-1]), "console-up")

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
                self.assertEqual([self.step(call) for call in dev], ["arm-state", "copy", "copy"])
                self.assertEqual(self.remote(dev[0]), ["arm-state", arm])
                pull = dev[1]["args"]
                self.assertEqual(pull[-2:], [f"dev:{zgca.REMOTE_ROOT}/experiments/{arm}/", f"{self.repo / 'experiments' / arm}/"])
                self.assertEqual(pull[pull.index("--exclude") + 1], "/pit_views/")
                self.assertNotIn("--delete", pull)
                self.assertEqual(dev[2]["args"][-2:], [f"dev:{zgca.REMOTE_ROOT}/logs/workers/{arm}.log", f"{self.repo / 'logs' / 'workers' / arm}.log"])
                self.log.unlink()

    def test_push_ships_ended_arms_of_this_host_and_refuses_the_rest_before_any_call(self) -> None:
        """The way lineage arms reach the instance: the reverse of sync, under the same rules."""
        from autotrade.pipelines.hitl_state import proc_start_ticks

        self._arm("arm_one")
        self._arm("arm_two", state="stopped", pid=None)
        self._arm("arm_copy", created_under=f"{zgca.REMOTE_ROOT}/experiments")
        self._arm("arm_live", state="running_session", pid=os.getpid(), pid_start_ticks=proc_start_ticks(os.getpid()))
        (self.repo / "experiments" / "arm_bare").mkdir()
        for arms, reason in ((("arm_one", "../escape"), "not an experiment id"),
                             (("arm_bare",), "not a readable arm of this host"),
                             (("arm_one", "arm_copy"), "copy of an arm of the instance"),
                             (("arm_live", "arm_one"), "arm_live still running here")):
            with self.subTest(reason=reason):
                code, err = self.run_main("push", *arms)
                self.assertEqual(code, 1)
                self.assertIn(reason, err)
        self.assertFalse(self.calls())
        code, err = self.run_main("push", "arm_one", "arm_two")
        self.assertEqual(code, 0, err)
        self.assertEqual([(call["tool"], self.step(call)) for call in self.calls()],
                         [("instance", "state"), ("dev", "copy"), ("dev", "copy")])
        for arm, call in zip(("arm_one", "arm_two"), self.calls("dev"), strict=True):
            args = call["args"]
            self.assertEqual(args[-2:], [f"{self.repo / 'experiments' / arm}/", f"dev:{zgca.REMOTE_ROOT}/experiments/{arm}/"])
            self.assertEqual(args[args.index("--exclude") + 1], "/pit_views/")
            self.assertNotIn("--delete", args)

    def test_sync_stops_before_any_transfer_when_the_arm_state_step_refuses(self) -> None:
        code, err = self.run_main("sync", "arm_running", MOCK_STEP_EXIT_ARM_STATE="1")
        self.assertEqual(code, 1)
        self.assertIn("remote step arm-state failed", err)
        self.assertEqual([self.step(call) for call in self.calls("dev")], ["arm-state"])

    def test_the_arm_state_step_refuses_an_arm_whose_worker_runs(self) -> None:
        """The console's own notion of a running arm decides, run for real against a temporary root."""
        from autotrade.pipelines.hitl_state import proc_start_ticks

        # Another root than the default keeps its venv inside it.
        root = Path(self.temp.name) / "instance"
        venv = root / ".venv"
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
        env = {**os.environ, "ZGCA_REMOTE_ROOT": str(root)}

        def step(arm: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), "arm-state", arm], env=env, capture_output=True, text=True, check=False)

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

    def test_the_status_step_shows_the_newest_round_log_not_an_image_build_log(self) -> None:
        root = Path(self.temp.name) / "instance"
        logs = root / "logs" / "zgca"
        logs.mkdir(parents=True)
        (logs / "create_round_test_20261019T190000Z.log").write_text("smoke\narms created\nround-exit 0\n", encoding="utf-8")
        build = logs / "image_build_20261019T200000Z.log"
        build.write_text("Successfully built\n", encoding="utf-8")
        os.utime(logs / "create_round_test_20261019T190000Z.log", (1, 1))
        shown = subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), "status"], env={**os.environ, "ZGCA_REMOTE_ROOT": str(root)},
                               capture_output=True, text=True, check=False)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("round log: logs/zgca/create_round_test_20261019T190000Z.log\n  smoke\n  arms created\n  round-exit 0", shown.stdout)
        self.assertNotIn(build.name, shown.stdout)

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
        # The build verifies and serves both shipped release files, so prepare takes both digests.
        prepare = subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), "prepare", "0" * 64], capture_output=True, text=True, check=False)
        self.assertEqual(prepare.returncode, 1)
        self.assertIn("DuckDB CLI and causal-conv1d wheel digests", prepare.stderr)
        self.assertEqual(subprocess.run(["bash", "-n", str(zgca.REMOTE_SCRIPT)], check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
