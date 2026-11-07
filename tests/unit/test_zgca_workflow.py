"""Offline contract of ops/zgca/zgca.py: the platform layer is a mock, no network.

The mock `instance` (the shared lifecycle entry) and the mock `dev` (whose
copy, put, bootstrap and report primitives verify their own results) record
their arguments and the instance they were pointed at, and answer from the
environment; both are tested for real in ~/zgca-compute. The mock `zenith`
only records, since the workflow reaches the API through those two. The
remote script's steps run for real against a temporary layout (a root with its
venv inside, a local directory beside it), with stubs for the venv's python,
docker, tmux and sleep where a step would reach them.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
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
if "remote.sh" in args:  # bash -lc <script> remote.sh <root> <venv> <local dir> <step> ...
    step = args[args.index("remote.sh") + 4]
    if step == "local-disk":  # bytes already in the store's directory, bytes free on its disk
        print(os.environ.get("MOCK_LOCAL_DISK", "0 100000000000000"))
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
        # The release files the deploy ships (the DuckDB CLI, the CUDA
        # compatibility package) are a fixture here: the mock curl writes them
        # and the pinned digests are the fixture's for the test's duration.
        body = "fixture, not a release file\n"
        self.env.update(MOCK_LOG=str(self.log), MOCK_CURL_BODY=body, PATH=str(self.tools) + os.pathsep + os.environ["PATH"])
        for name in ("DUCKDB_CLI_SHA256", "CUDA_COMPAT_SHA256"):
            self.addCleanup(setattr, zgca, name, getattr(zgca, name))
            setattr(zgca, name, hashlib.sha256(body.encode()).hexdigest())

    def calls(self, tool: str | None = None) -> list[dict[str, object]]:
        lines = self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []
        return [call for call in map(json.loads, lines) if tool is None or call["tool"] == tool]

    @staticmethod
    def remote(call: dict[str, object]) -> list[str] | None:
        """The remote.sh step and its arguments of a `dev bash -lc <script> remote.sh <layout> ...` call, else None."""
        args = call["args"]
        return args[args.index("remote.sh") + 4:] if "remote.sh" in args else None

    @staticmethod
    def remote_step(root: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        """remote.sh run for real against a temporary layout: ``root``, its venv in root/.venv, the local directory root/../ebs."""
        return subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), str(root), str(root / ".venv"), str(root.parent / "ebs"), *args],
                              env=env, capture_output=True, text=True, check=False)

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
        # A down first asks the instance's state: a stopped one runs no experiment to guard.
        for command, action, before in (("up", "start", []), ("down", "stop", [["state", INSTANCE]])):
            with self.subTest(command=command):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main(command, MOCK_STATE="STOPPED")
                self.assertEqual(code, 0, err)
                self.assertEqual([call["args"] for call in self.calls()], [*before, [action, INSTANCE]])
                self.assertTrue(all(call["tool"] == "instance" and call["instance"] == INSTANCE for call in self.calls()))
                # Its unverified result (a refused request, a timeout, no SSH) is this command's failure.
                code, err = self.run_main(command, MOCK_STATE="STOPPED", **{f"MOCK_INSTANCE_EXIT_{action.upper()}": "1"})
                self.assertEqual(code, 1)
                self.assertIn(f"instance {action} {INSTANCE} failed", err)

    def test_down_stops_a_running_instance_only_when_no_experiment_runs(self) -> None:
        code, err = self.run_main("down")
        self.assertEqual(code, 0, err)
        self.assertEqual([(call["tool"], self.step(call)) for call in self.calls()],
                         [("instance", "state"), ("dev", "idle"), ("instance", "stop")])
        self.assertEqual(self.remote(self.calls("dev")[0]), ["idle", "down"])
        self.log.unlink()
        code, err = self.run_main("down", MOCK_STEP_EXIT_IDLE="1")
        self.assertEqual(code, 1)
        self.assertIn("remote step idle failed", err)
        self.assertEqual([self.step(call) for call in self.calls()], ["state", "idle"])

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

    def test_deploy_run_and_sync_refuse_an_instance_that_is_not_running_before_any_transfer(self) -> None:
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        for argv in (("deploy", "data/pit_views_seed_test"), ("run", "scripts/experiments/create_round_test.py"),
                     ("sync", "experiment_new")):
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
        # As the repository's .gitignore does: the runtime trees and the .env are not the tree a deploy ships.
        (self.repo / ".git" / "info" / "exclude").write_text("/.env\n/.runtime/\n/data/\n/experiments/\n/logs/\n/untracked.py\n", encoding="utf-8")
        subprocess.run([*git, "add", "scripts"], cwd=self.repo, check=True)
        subprocess.run([*git, "commit", "-q", "-m", "fixture"], cwd=self.repo, check=True)

    @staticmethod
    def _provider(generation: str = "gen_test", *, text: tuple[str, ...] = ("anns_d",), intraday: bool = False) -> dict[str, object]:
        """A seed's provider record, its snapshot configuration as the prebuild records it (the fields deploy reads)."""
        return {
            "generation_id": generation,
            "snapshot_config": {
                "datasets": {"events": [], "fundamentals": ["income_vip"], "macro": ["index_daily"], "text": list(text)},
                "include_intraday": intraday,
            },
        }

    def _seed(self, name: str = "data/pit_views_seed_test", generation: str = "gen_test", *, release: bool = True) -> None:
        seed = self.repo / name
        seed.mkdir(parents=True, exist_ok=True)
        (seed / "provider.json").write_text(json.dumps(self._provider(generation)), encoding="utf-8")
        if release:
            release_dir = self.repo / zgca.RELEASES / generation
            release_dir.mkdir(parents=True, exist_ok=True)
            (release_dir / "manifest.json").write_text("{}", encoding="utf-8")

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
        (self.repo / "data/pit_views_seed_test" / "provider.json").write_text(json.dumps({"generation_id": "gen_test"}), encoding="utf-8")
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "no usable snapshot configuration"))
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
        # The working tree must be its commit: neither a file not yet added nor a modified one ships.
        (self.repo / "scripts" / "new_module.py").write_text("", encoding="utf-8")
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "commit first (deploy records the commit it ships): ?? scripts/new_module.py"))
        (self.repo / "scripts" / "new_module.py").unlink()
        (self.repo / "scripts" / "experiments" / "create_round_test.py").write_text("ROUND = 1\n", encoding="utf-8")
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), "commit first (deploy records the commit it ships): M scripts/experiments/create_round_test.py"))
        subprocess.run(["git", "checkout", "-q", "--", "scripts"], cwd=self.repo, check=True)
        refusals.append((self.run_main("deploy", "/etc/passwd"), "below data/"))
        refusals.append((self.run_main("deploy", "scripts/experiments"), "below data/"))
        # A shipped release file that is not the pinned release is refused, not refetched.
        state = self.repo / ".runtime" / "zgca"
        state.mkdir(parents=True)
        (state / zgca.DUCKDB_CLI_NAME).write_text("another release\n", encoding="utf-8")
        refusals.append((self.run_main("deploy", "data/pit_views_seed_test"), f"{zgca.DUCKDB_CLI_NAME} has digest"))
        (state / zgca.DUCKDB_CLI_NAME).unlink()
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
        # Each remote step is this checkout's script in a login shell, told the
        # instance's layout: the root, its venv, and the root's directory on
        # the instance's own disk.
        remote, local = zgca.REMOTE_ROOT, "/ebs/auto-trade"
        script = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        for call in dev:
            if self.remote(call) is not None:
                self.assertEqual(call["args"][:7], ["bash", "-lc", script, "remote.sh", remote, "/personal_fast_storage/envs/quant", local])
        # No experiment runs, then the base tools; then transfers (by
        # destination) and remote steps in the order they ran: the tracked
        # list lands before the `tracked` step prunes by it, and both come
        # before the seed and the release; then the image, and only then the
        # pretrained store, to the instance's own disk once it is known to
        # fit; the host report last.
        ran = [self.remote(call) or call["args"][:1] + call["args"][1:][-1:] for call in dev]
        self.assertEqual(ran, [
            ["idle", "deploy"], ["bootstrap"], ["dirs"], ["copy", f"dev:{remote}/"], ["put", f"{remote}/.env"],
            ["copy", f"dev:{remote}/.runtime/zgca/"], ["tracked"],
            ["copy", f"dev:{remote}/data/pit_views_seed_test/"], ["copy", f"dev:{remote}/data/research_releases/gen_test/"],
            ["prepare", zgca.DUCKDB_CLI_SHA256, zgca.CUDA_COMPAT_SHA256], ["local-disk"],
            ["copy", f"dev:{local}/pretrained/"], ["report"],
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
        # The commit those files are travels beside the list.
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True).stdout
        commit = state / zgca.DEPLOYED_COMMIT_NAME
        self.assertEqual(commit.read_text(encoding="utf-8"), head)
        self.assertEqual(
            transfers[1]["args"][-5:-1],
            [str(state / zgca.DUCKDB_CLI_NAME), str(state / zgca.CUDA_COMPAT_NAME), str(manifest), str(commit)],
        )
        # The seed, the release and the store are mirrored, hard links kept.
        for transfer in transfers[2:5]:
            self.assertIn("--delete", transfer["args"])
            self.assertIn("-H", transfer["args"])
        self.assertEqual(transfers[4]["args"][-2], f"{self.repo / zgca.PRETRAINED}/")
        self.assertNotIn("--delete", transfers[0]["args"])
        # The release leaves behind exactly the text and minute datasets the
        # seed's configuration does not read, and keeps them where an earlier
        # deploy put them (no --delete-excluded).
        release = transfers[3]["args"]
        self.assertEqual(
            [arg for arg in release if arg.startswith("--exclude")],
            [f"--exclude=/raw/{name}/" for name in ("research_report", "report_rc", "irm_qa_sh", "irm_qa_sz",
                                                    "major_news", "news", "cctv_news", "npr", "stk_mins_1min_by_date")],
        )
        self.assertNotIn("--delete-excluded", release)
        # The release files were fetched once, from their official URLs, and are reused by the next deploy.
        fetches = self.calls("curl")
        self.assertEqual([call["args"][-1] for call in fetches], [zgca.DUCKDB_CLI_URL, zgca.CUDA_COMPAT_URL])
        self.assertTrue((state / zgca.DUCKDB_CLI_NAME).is_file())
        self.assertTrue((state / zgca.CUDA_COMPAT_NAME).is_file())
        self.assertEqual(self.run_main("deploy", "data/pit_views_seed_test")[0], 0)
        self.assertEqual(len(self.calls("curl")), 2)

    def test_release_exclusions_follow_the_seeds_configuration(self) -> None:
        """Only text and minute datasets the configuration does not name stay behind; a seed naming them takes them."""
        self.assertEqual(
            zgca.release_exclusions(self._provider(text=("anns_d", "major_news", "news"), intraday=True)),
            ["research_report", "report_rc", "irm_qa_sh", "irm_qa_sz", "cctv_news", "npr"],
        )
        everything = zgca.release_exclusions(self._provider(text=()))
        self.assertIn("stk_mins_1min_by_date", everything)
        # Datasets every experiment reads outside its configuration never stay behind.
        for name in ("trade_cal", "stock_basic", "namechange", "index_member_all", "index_classify", "daily"):
            self.assertNotIn(name, everything)

    def test_deploy_finishes_everything_but_the_store_when_the_store_does_not_fit_and_fails(self) -> None:
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        store = sum(path.stat().st_size for path in (self.repo / zgca.PRETRAINED).rglob("*") if path.is_file())
        reserve = zgca.LOCAL_DISK_RESERVE
        # What is already there counts: exactly the reserve plus what is missing fits.
        for present, free, fits in ((0, reserve + store, True), (0, reserve + store - 1, False),
                                    (store, reserve, True), (store - 1, reserve, False)):
            with self.subTest(present=present, free=free):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main("deploy", "data/pit_views_seed_test", MOCK_LOCAL_DISK=f"{present} {free}")
                ran = [self.remote(call) or call["args"][:1] + call["args"][1:][-1:] for call in self.calls("dev")]
                done = ran.index(["local-disk"])
                self.assertEqual(ran[done - 1], ["prepare", zgca.DUCKDB_CLI_SHA256, zgca.CUDA_COMPAT_SHA256])
                if fits:
                    self.assertEqual((code, ran[done + 1:]), (0, [["copy", "dev:/ebs/auto-trade/pretrained/"], ["report"]]), err)
                else:
                    # Everything else is done, the report shows the disks, and the deploy fails saying why.
                    self.assertEqual((code, ran[done + 1:]), (1, [["report"]]))
                    self.assertIn("the pretrained store does not fit on the instance's own disk", err)
                    self.assertIn(f"{zgca.LOCAL_DISK} has {free / zgca.GIB:.1f} GiB free", err)
                    self.assertIn("Code, seed, release and image are deployed; the store is not", err)

    def test_release_files_come_through_the_proxy_first_then_github_and_only_with_the_pinned_digest(self) -> None:
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        state = self.repo / ".runtime" / "zgca"
        github, nvidia = zgca.DUCKDB_CLI_URL, zgca.CUDA_COMPAT_URL
        # The deploy stops at the stopped instance, after fetching; the proxy
        # is tried for the GitHub release only.
        for answer, sources in (("", [PROXY + github, nvidia]),
                                ("fail", [PROXY + github, github, nvidia]),
                                ("wrong", [PROXY + github, github, nvidia])):
            with self.subTest(proxy=answer or "works"):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main("deploy", "data/pit_views_seed_test", MOCK_STATE="STOPPED",
                                          ZGCA_GITHUB_PROXY=PROXY, MOCK_PROXY_ANSWER=answer)
                self.assertIn("not RUNNING", err)
                self.assertEqual([call["args"][-1] for call in self.calls("curl")], sources)
                self.assertEqual({path.name for path in state.iterdir()} - {zgca.TRACKED_LIST_NAME, zgca.DEPLOYED_COMMIT_NAME},
                                 {zgca.DUCKDB_CLI_NAME, zgca.CUDA_COMPAT_NAME})
                (state / zgca.DUCKDB_CLI_NAME).unlink()
                (state / zgca.CUDA_COMPAT_NAME).unlink()
        # No source with the pinned bytes: nothing enters the cache, nothing is shipped.
        self.log.unlink()
        code, err = self.run_main("deploy", "data/pit_views_seed_test", ZGCA_GITHUB_PROXY=PROXY,
                                  MOCK_PROXY_ANSWER="wrong", MOCK_CURL_BODY="truncated\n")
        self.assertEqual(code, 1)
        self.assertIn(f"no source gave {zgca.DUCKDB_CLI_NAME} with the pinned digest", err)
        self.assertEqual({path.name for path in state.iterdir()}, {zgca.TRACKED_LIST_NAME, zgca.DEPLOYED_COMMIT_NAME})
        self.assertFalse(self.calls("dev"))

    def test_the_tracked_step_removes_only_untracked_files_below_tracked_directories(self) -> None:
        root = Path(self.temp.name) / "instance"
        files = {  # path on the instance: whether the shipped list names it
            "README.md": True,
            ".env": False,
            "configs/skills/kept/SKILL.md": True,
            "scripts/experiments/create_round_kept.py": True,
            "scripts/experiments/create_round_retired.py": False,
            "src/pkg/mod.py": True,
            "src/pkg/__pycache__/mod.cpython-311.pyc": False,
            "src/pkg.egg-info/PKG-INFO": False,
            "data/README.md": True,  # a tracked file does not open a runtime directory
            "data/release/manifest.json": False,
            "experiments/experiment/hitl/status.json": False,
            "logs/workers/experiment.log": False,
        }
        for name in files:
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text("", encoding="utf-8")
        (root / "scripts" / "alias.py").symlink_to("experiments/create_round_kept.py")
        (root / "scripts" / "data_link").symlink_to("../data")
        # A removed skill item left as empty directories, which the library
        # would refuse; empty directories in a runtime tree are not entered.
        (root / "configs" / "skills" / "removed" / "references").mkdir(parents=True)
        (root / "experiments" / "empty").mkdir()

        missing = self.remote_step(root, "tracked")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("no tracked file list", missing.stderr)
        listed = root / ".runtime" / "zgca" / zgca.TRACKED_LIST_NAME
        listed.parent.mkdir(parents=True)
        listed.write_bytes(b"".join(name.encode() + b"\0" for name, named in files.items() if named))
        pruned = self.remote_step(root, "tracked")
        self.assertEqual(pruned.returncode, 0, pruned.stderr)
        self.assertIn("1 stale removed, 2 empty directories pruned under configs, scripts, src: "
                      "scripts/experiments/create_round_retired.py", pruned.stdout)
        left = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink() or path.is_file()}
        self.assertEqual(left, set(files) - {"scripts/experiments/create_round_retired.py"}
                         | {"scripts/alias.py", "scripts/data_link", f".runtime/zgca/{zgca.TRACKED_LIST_NAME}"})
        self.assertFalse((root / "configs" / "skills" / "removed").exists())
        self.assertTrue((root / "experiments" / "empty").is_dir())

    def test_the_image_build_and_the_instance_name_the_files_the_deploy_ships(self) -> None:
        """Three places name each file: the Dockerfile fetches the DuckDB CLI
        by default and the compatibility package only when told, the deploy
        pins and ships both, remote.sh verifies and serves them."""
        dockerfile = (REPO / "ops" / "docker" / "sandbox.Dockerfile").read_text(encoding="utf-8")
        self.assertIn(zgca.DUCKDB_CLI_URL, dockerfile)
        self.assertIn("ARG CUDA_COMPAT_URL=\n", dockerfile)
        self.assertIn("ENV LD_LIBRARY_PATH=${CUDA_COMPAT_URL:+/usr/local/cuda-compat}\n", dockerfile)
        self.assertEqual(zgca.CUDA_COMPAT_URL.rsplit("/", 1)[1], zgca.CUDA_COMPAT_NAME)
        remote = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(f"=.runtime/zgca/{zgca.DUCKDB_CLI_NAME}\n", remote)
        self.assertIn(f"=.runtime/zgca/{zgca.CUDA_COMPAT_NAME}\n", remote)

    def test_the_dirs_step_links_the_store_to_the_instance_disk_and_refuses_anything_else(self) -> None:
        """Run for real against temporary roots: the link, its idempotence, and the refusals."""
        base = Path(self.temp.name)
        root, local = base / "instance", base / "ebs"

        def step(name: str) -> subprocess.CompletedProcess[str]:
            return self.remote_step(root, name)

        link = root / "data" / "pretrained"
        for _ in range(2):
            made = step("dirs")
            self.assertEqual(made.returncode, 0, made.stderr)
            self.assertEqual(os.readlink(link), str(local / "pretrained"))
        (local / "pretrained" / "Model").mkdir()
        (local / "pretrained" / "Model" / "config.json").write_text("{}\n", encoding="utf-8")
        # Counted as the deploy counts the local store: the bytes of its files,
        # past 2^31 too (a sparse file takes no disk).
        with open(local / "pretrained" / "Model" / "model.safetensors", "wb") as large:
            large.truncate(3 * 2**30)
        present, free = map(int, step("local-disk").stdout.split())
        self.assertEqual(present, zgca._tree_bytes(local / "pretrained"))
        self.assertEqual(present, 3 + 3 * 2**30)
        self.assertGreater(free, 0)
        link.unlink()
        link.symlink_to(base / "elsewhere")
        refused = step("dirs")
        self.assertEqual(refused.returncode, 1)
        self.assertIn(f"not to the store's directory {local / 'pretrained'}", refused.stderr)
        link.unlink()
        link.mkdir()
        refused = step("dirs")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("is a directory, but the store lives in", refused.stderr)
        self.assertTrue(link.is_dir() and not link.is_symlink())

    def test_deploy_stops_at_the_first_primitive_that_fails(self) -> None:
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        remote = zgca.REMOTE_ROOT
        for env, reason, ran in (
            # Running experiments: nothing on the instance changes.
            ({"MOCK_STEP_EXIT_IDLE": "1"}, "remote step idle failed (exit 1)", ["idle"]),
            ({"MOCK_DEV_EXIT_BOOTSTRAP": "100"}, "dev bootstrap failed (exit 100)", ["idle", "bootstrap"]),
            ({"MOCK_DEV_EXIT_COPY": "1"}, f"dev copy failed (exit 1): {self.repo}/ dev:{remote}/", ["idle", "bootstrap", "dirs", "copy"]),
            ({"MOCK_DEV_EXIT_PUT": "1"}, f"dev put failed (exit 1): {remote}/.env", ["idle", "bootstrap", "dirs", "copy", "put"]),
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
        code, err = self.run_main("run", "scripts/experiments/create_round_test.py", "--dry-run", "experiment_one")
        self.assertEqual(code, 0, err)
        steps = [self.remote(call) for call in self.calls("dev")]
        self.assertEqual(steps, [["console-up"], ["round", "scripts/experiments/create_round_test.py", "--dry-run", "experiment_one"]])
        code, err = self.run_main("run", "scripts/experiments/create_round_test.py", MOCK_STEP_EXIT_CONSOLE_UP="1")
        self.assertEqual(code, 1)
        self.assertIn("console-up failed", err)
        self.assertEqual(self.step(self.calls()[-1]), "console-up")

    def test_sync_pulls_an_instance_experiment_and_refuses_to_overwrite_a_local_one(self) -> None:
        local = self.repo / "experiments" / "experiment_local"
        (local / "hitl").mkdir(parents=True)
        (local / "hitl" / "params.json").write_text(json.dumps({"experiments_root": str(self.repo / "experiments")}), encoding="utf-8")
        code, err = self.run_main("sync", "experiment_local")
        self.assertEqual(code, 1)
        self.assertIn("experiment of this host", err)
        (local / "hitl" / "params.json").unlink()
        code, err = self.run_main("sync", "experiment_local")
        self.assertEqual(code, 1)
        self.assertIn("refusing to overwrite", err)
        code, err = self.run_main("sync", "../escape")
        self.assertEqual(code, 1)
        self.assertFalse(self.calls("dev"))
        copied = self.repo / "experiments" / "experiment_zgca"
        (copied / "hitl").mkdir(parents=True)
        (copied / "hitl" / "params.json").write_text(json.dumps({"experiments_root": f"{zgca.REMOTE_ROOT}/experiments"}), encoding="utf-8")
        for experiment in ("experiment_new", "experiment_zgca"):
            with self.subTest(experiment=experiment):
                code, err = self.run_main("sync", experiment)
                self.assertEqual(code, 0, err)
                dev = self.calls("dev")
                self.assertEqual([self.step(call) for call in dev], ["experiment-state", "copy", "copy"])
                self.assertEqual(self.remote(dev[0]), ["experiment-state", experiment])
                pull = dev[1]["args"]
                self.assertEqual(pull[-2:], [f"dev:{zgca.REMOTE_ROOT}/experiments/{experiment}/", f"{self.repo / 'experiments' / experiment}/"])
                self.assertEqual(pull[pull.index("--exclude") + 1], "/pit_views/")
                self.assertNotIn("--delete", pull)
                self.assertEqual(dev[2]["args"][-2:], [f"dev:{zgca.REMOTE_ROOT}/logs/workers/{experiment}.log", f"{self.repo / 'logs' / 'workers' / experiment}.log"])
                self.log.unlink()

    def test_sync_stops_before_any_transfer_when_the_experiment_state_step_refuses(self) -> None:
        code, err = self.run_main("sync", "experiment_running", MOCK_STEP_EXIT_EXPERIMENT_STATE="1")
        self.assertEqual(code, 1)
        self.assertIn("remote step experiment-state failed", err)
        self.assertEqual([self.step(call) for call in self.calls("dev")], ["experiment-state"])

    def test_the_experiment_state_and_idle_steps_refuse_while_a_worker_runs(self) -> None:
        """The console's own notion of a running experiment decides, run for real against a temporary root."""
        from autotrade.pipelines.hitl_state import proc_start_ticks

        root = Path(self.temp.name) / "instance"
        venv = root / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        (venv / "bin" / "python").chmod(0o700)
        live = {"state": "running_session", "pid": os.getpid(), "pid_start_ticks": proc_start_ticks(os.getpid())}
        experiments = {
            "experiment_live": live,
            # A worker that is gone no longer writes, whatever its last state said.
            "experiment_gone": {**live, "pid_start_ticks": live["pid_start_ticks"] + 1},
            "experiment_done": {"state": "completed", "verdict": "pass", "pid": None},
        }
        for experiment, record in experiments.items():
            (root / "experiments" / experiment / "hitl").mkdir(parents=True)
            (root / "experiments" / experiment / "hitl" / "status.json").write_text(json.dumps(record), encoding="utf-8")

        def step(experiment: str) -> subprocess.CompletedProcess[str]:
            return self.remote_step(root, "experiment-state", experiment)

        refused = step("experiment_live")
        self.assertEqual(refused.returncode, 1)
        self.assertIn("experiment_live: state running_session", refused.stdout)
        self.assertIn("experiment_live is still running", refused.stderr)
        for experiment in ("experiment_gone", "experiment_done"):
            with self.subTest(experiment=experiment):
                shown = step(experiment)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                self.assertIn(f"{experiment}: state {experiments[experiment]['state']}", shown.stdout)
        missing = step("experiment_absent")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("no experiment experiment_absent", missing.stderr)
        # The first step of deploy and down refuses while one runs, naming it.
        # A stub tmux answers has-session for $SESSION alone, so no session of
        # this host's tmux server is asked.
        stubs = Path(self.temp.name) / "stubs"
        stubs.mkdir()
        (stubs / "tmux").write_text('#!/bin/sh\n[ "$1" = has-session ] && [ "$3" = "$SESSION" ]\n', encoding="utf-8")
        (stubs / "tmux").chmod(0o700)
        env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}"}
        refused = self.remote_step(root, "idle", "deploy", env=env)
        self.assertEqual(refused.returncode, 1)
        self.assertIn("deploy refused: experiments are running (experiment_live)", refused.stderr)
        # A venv without autotrade (a first install that failed) and no venv at
        # all run no worker, so a deploy can proceed to the install.
        (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "$@"\n', encoding="utf-8")
        idle = [self.remote_step(root, "idle", "down", env=env)]
        shutil.rmtree(venv)
        idle.append(self.remote_step(root, "idle", "down", env=env))
        for case, shown in zip(("no autotrade", "no venv"), idle, strict=True):
            with self.subTest(venv=case):
                self.assertEqual((shown.returncode, shown.stdout, shown.stderr), (0, "", ""))
        # With no experiment running, a round launcher or fill loop that could
        # create one meanwhile is refused too.
        for session in ("autotrade-round", "autotrade-fill"):
            with self.subTest(session=session):
                refused = self.remote_step(root, "idle", "deploy", env=env | {"SESSION": session})
                self.assertEqual(refused.returncode, 1)
                self.assertIn(f"deploy refused: a round launcher is running in tmux session {session}", refused.stderr)

    def test_the_status_step_shows_the_newest_round_log_not_an_image_build_log(self) -> None:
        root = Path(self.temp.name) / "instance"
        logs = root / "logs" / "zgca"
        logs.mkdir(parents=True)
        (logs / "create_round_test_20261019T190000Z.log").write_text("smoke\nexperiments created\nround-exit 0\n", encoding="utf-8")
        build = logs / "image_build_20261019T200000Z.log"
        build.write_text("Successfully built\n", encoding="utf-8")
        os.utime(logs / "create_round_test_20261019T190000Z.log", (1, 1))
        # The image line asks Docker; a stub answers for the host's daemon.
        stubs = Path(self.temp.name) / "stubs"
        stubs.mkdir()
        (stubs / "docker").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        (stubs / "docker").chmod(0o700)
        env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}"}
        shown = self.remote_step(root, "status", env=env)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertNotIn("deployed commit", shown.stdout)
        self.assertIn("image: docker unavailable\n", shown.stdout)
        self.assertIn("round log: logs/zgca/create_round_test_20261019T190000Z.log\n  smoke\n  experiments created\n  round-exit 0", shown.stdout)
        self.assertNotIn(build.name, shown.stdout)
        self.assertIn("fill: no session\n", shown.stdout)
        # A fill log is newer but is not the round log; its last pass and its end are shown.
        fill = logs / "fill_create_round_test_20261019T210000Z.log"
        fill.write_text("fill create_round_test: slots; queue 1: 0 created, 1 pending (...)\n"
                        "fill create_round_test: slots; queue 1: 1 created, 0 pending (...)\nround-exit 0\n", encoding="utf-8")
        # The commit the last deploy shipped is shown once it is recorded.
        (root / ".runtime" / "zgca").mkdir(parents=True)
        (root / ".runtime" / "zgca" / zgca.DEPLOYED_COMMIT_NAME).write_text("0123abcd\n", encoding="utf-8")
        shown = self.remote_step(root, "status", env=env)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("deployed commit: 0123abcd\n", shown.stdout)
        self.assertIn("round log: logs/zgca/create_round_test_20261019T190000Z.log", shown.stdout)
        self.assertIn(f"fill: no session; last log logs/zgca/{fill.name}\n  fill create_round_test: slots; queue 1: 1 created, 0 pending (...)\n  round-exit 0", shown.stdout)

    def test_a_fill_repeats_until_nothing_is_pending_and_ends_at_a_failed_pass(self) -> None:
        """The round step's --fill loop, run for real against a temporary root with
        tmux, sleep and the launcher stubbed: every pass appends to one log, the
        loop sleeps between passes and ends at the first pass that reports 0
        pending, or at the first that exits non-zero, which is never retried."""
        temp = Path(self.temp.name)
        root, stubs = temp / "instance", temp / "stubs"
        for directory in (root / ".venv" / "bin", root / "scripts" / "experiments", root / "logs" / "zgca", stubs):
            directory.mkdir(parents=True)
        (root / "scripts" / "experiments" / "create_round_test.py").write_text("", encoding="utf-8")
        scripts = {
            # The console's health probe answers; each launcher pass prints and exits as planned.
            root / ".venv" / "bin" / "python": '#!/bin/sh\n[ "$1" = -c ] && { echo "{}"; exit 0; }\n'
            'n=$(($(cat "$PASSES" 2>/dev/null || echo 0) + 1)); echo $n >"$PASSES"\n'
            'line=$(sed -n "${n}p" "$PLAN"); echo "${line#*|}"; exit "${line%%|*}"\n',
            # No session exists yet; a new one runs its command at once, in its directory.
            stubs / "tmux": '#!/bin/sh\n[ "$1" = new-session ] || exit 1\n'
            'while [ $# -gt 1 ]; do [ "$1" = -c ] && { shift; cd "$1"; }; shift; done\nexec sh -c "$1"\n',
            stubs / "sleep": '#!/bin/sh\necho "$1" >>"$SLEPT"\n',
        }
        for script, body in scripts.items():
            script.write_text(body, encoding="utf-8")
            script.chmod(0o700)
        held = "fill create_round_test: slots; queue 1: 0 skipped (created already), 0 created, 0 refused, 1 pending (1 by the free GPUs)"
        done = held.replace("0 created", "1 created").replace("1 pending", "0 pending")
        env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}",
               "PLAN": str(temp / "plan"), "PASSES": str(temp / "passes"), "SLEPT": str(temp / "slept")}
        for plan, ending in (([f"0|{held}", f"0|{held}", f"0|{done}"], "round-exit 0"),
                             ([f"0|{held}", "1|experiment_one: refused, HTTP 400"], "round-exit 1")):
            with self.subTest(ending=ending):
                for leftover in (temp / "passes", temp / "slept", *(root / "logs" / "zgca").glob("*.log")):
                    leftover.unlink(missing_ok=True)
                (temp / "plan").write_text("\n".join(plan) + "\n", encoding="utf-8")
                shown = self.remote_step(root, "round", "scripts/experiments/create_round_test.py", "--fill", env=env)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                self.assertIn("fill launcher started in tmux session autotrade-fill", shown.stdout)
                (log,) = (root / "logs" / "zgca").glob("fill_create_round_test_*.log")
                self.assertEqual(log.read_text(encoding="utf-8").splitlines(), [step.split("|", 1)[1] for step in plan] + [ending])
                self.assertEqual((temp / "slept").read_text(encoding="utf-8").split(), ["600"] * (len(plan) - 1))

    def test_console_up_restarts_an_idle_console_onto_the_checkout_and_keeps_a_busy_one(self) -> None:
        """A console serves the code it was started with: with no experiment running an
        existing one is killed, its port let go and a new one started before any
        experiment is created; with experiments running it is kept. Run for real against a
        temporary root with tmux, sleep and the venv's python stubbed."""
        temp = Path(self.temp.name)
        root, stubs = temp / "instance", temp / "stubs"
        for directory in (root / ".venv" / "bin", stubs):
            directory.mkdir(parents=True)
        scripts = {
            # running_experiments lists $EXPERIMENTS; the health probe answers while $UP exists.
            root / ".venv" / "bin" / "python": '#!/bin/sh\ncase "$2" in *running_experiments*) cat "$EXPERIMENTS" ;;\n'
            '*) [ -f "$UP" ] || exit 1; echo "{}" ;; esac\n',
            # has-session answers $HAS; killing a session takes the console down, a new one brings it up.
            stubs / "tmux": '#!/bin/sh\necho "$1" >>"$CALLS"\ncase "$1" in has-session) exit "$HAS" ;;\n'
            'kill-session) rm -f "$UP" ;; new-session) touch "$UP" ;; esac\n',
            stubs / "sleep": "#!/bin/sh\n",
        }
        for script, body in scripts.items():
            script.write_text(body, encoding="utf-8")
            script.chmod(0o700)
        cases = (
            ("1", "", "console started in tmux session autotrade-console", ["has-session", "new-session"]),
            ("0", "", "console restarted on the current checkout", ["has-session", "kill-session", "new-session"]),
            ("0", "experiment_a\nexperiment_b\n", "console kept: experiments are running (experiment_a experiment_b)", ["has-session"]),
        )
        for has, experiments, said, calls in cases:
            with self.subTest(said=said):
                (temp / "experiments").write_text(experiments, encoding="utf-8")
                (temp / "calls").unlink(missing_ok=True)
                (temp / "up").unlink(missing_ok=True)
                if has == "0":  # an existing console answers
                    (temp / "up").touch()
                env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "HAS": has,
                       "EXPERIMENTS": str(temp / "experiments"), "UP": str(temp / "up"), "CALLS": str(temp / "calls")}
                shown = self.remote_step(root, "console-up", env=env)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                self.assertIn(said, shown.stdout)
                self.assertIn("console health: {}", shown.stdout)
                self.assertEqual((temp / "calls").read_text(encoding="utf-8").split(), calls)

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
        root = Path(self.temp.name) / "instance"
        remote = self.remote_step(root, "bogus")
        self.assertEqual(remote.returncode, 1)
        self.assertIn("unknown remote step", remote.stderr)
        # The layout comes from zgca.py on every call; a step without it is refused.
        bare = subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), "status"], capture_output=True, text=True, check=False)
        self.assertEqual(bare.returncode, 1)
        self.assertIn("usage: remote.sh <root> <venv> <local dir> <step>", bare.stderr)
        # The build verifies and serves the shipped release files, so prepare takes their digests.
        prepare = self.remote_step(root, "prepare", "0" * 64)
        self.assertEqual(prepare.returncode, 1)
        self.assertIn("pinned DuckDB CLI and CUDA compatibility package digests", prepare.stderr)
        self.assertEqual(subprocess.run(["bash", "-n", str(zgca.REMOTE_SCRIPT)], check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
