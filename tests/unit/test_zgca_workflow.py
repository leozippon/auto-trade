"""Offline contract of ops/zgca/zgca.py: the platform layer is a mock, no network.

The mock `instance` (the shared lifecycle entry) and the mock `dev` (whose
copy, put, bootstrap and report primitives verify their own results) record
their arguments and the instance they were pointed at, and answer from the
environment; both are tested for real in ~/zgca-compute, and the workflow
reaches the API through those two alone. The mock copy moves nothing or,
given a stand-in of the instance's file system, is a real rsync verified by a
dry run as `dev copy` verifies. The remote script's steps run for real against a temporary layout (a root with its
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

import pytest

from autotrade.pipelines.pack_payload import (
    PAYLOAD_MANIFEST,
    format_manifest,
    payload_digests,
)

# The ZGCA scripts run for real in bash against stand-in tools.
pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "ops" / "zgca" / "zgca.py"
SPEC = importlib.util.spec_from_file_location("zgca", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
zgca = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = zgca  # dataclasses resolve the module's annotations through sys.modules
SPEC.loader.exec_module(zgca)

INSTANCE = zgca.INSTANCE_ID
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
if "remote.sh" in args:  # bash -lc <script> remote.sh <root> <venv> <local dir> <storage> <quota> <floor> <step> ...
    step = args[args.index("remote.sh") + 7]
    root = os.environ.get("MOCK_REMOTE_ROOT")
    if step == "experiment-state" and root:  # the real step, against a temporary instance root
        import subprocess
        at = args.index("remote.sh")
        sys.exit(subprocess.run(["bash", "-c", args[at - 1], "remote.sh", root, root + "/.venv", root + "/../ebs", *args[at + 4:]]).returncode)
    if step == "local-disk":  # bytes already in the store's directory, bytes free on its disk
        print(os.environ.get("MOCK_LOCAL_DISK", "0 100000000000000"))
    if step == "storage":  # the storage line, and whether the floor is free
        print(os.environ.get("MOCK_STORAGE", "storage: 56.2 of 100 GiB used, 43.8 free (du of /personal_fast_storage)"))
    if step == "deployed-commit":
        print(os.environ.get("MOCK_DEPLOYED_COMMIT", ""))
    if step == "image-runtime":  # the runtime an image on the instance bakes
        print(os.environ.get("MOCK_REMOTE_RUNTIME", ""))
    sys.exit(int(os.environ.get("MOCK_STEP_EXIT_" + step.replace("-", "_").upper(), "0")))
if args[:1] == ["copy"] and os.environ.get("MOCK_RSYNC"):  # a real rsync, dev:<path> being MOCK_RSYNC<path>, verified as dev copy verifies
    import subprocess
    paths = [os.environ["MOCK_RSYNC"] + arg[4:] if arg.startswith("dev:") else arg for arg in args[1:]]
    if os.environ.get("MOCK_RSYNC_CUT"):  # the link drops once rsync is past artifacts/: hitl/ never arrives
        subprocess.run(["rsync", "--exclude=/hitl/", *paths], check=True)
        sys.exit(12)
    subprocess.run(["rsync", *paths], check=True)
    sys.exit(1 if subprocess.run(["rsync", "--dry-run", "--itemize-changes", *paths], capture_output=True, check=True).stdout else 0)
code = int(os.environ.get("MOCK_DEV_EXIT_" + args[0].upper(), "0"))
if args[:1] == ["copy"] and not code and args[-1].endswith("/") and not args[-1].startswith("dev:"):
    os.makedirs(args[-1], exist_ok=True)  # it moves nothing, yet leaves the directory it copied into, as rsync does
sys.exit(code)
"""
# This host's Docker: `run` is contract_fingerprint.read_image_contract's one
# offline container, answering the runtime the image bakes; the image is logged.
MOCK_DOCKER = r"""#!/usr/bin/env python3
import json, os, sys
args = sys.argv[1:]
logged = ["run", args[args.index("--entrypoint") + 2]] if args[:1] == ["run"] else args
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"tool": "docker", "args": logged}) + "\n")
if args[:1] == ["run"]:
    print(json.dumps({"runtime": os.environ.get("MOCK_LOCAL_RUNTIME", ""), "contract": ""}))
sys.exit(int(os.environ.get("MOCK_DOCKER_EXIT_" + args[0].upper(), "0")))
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
TOOLS = {"instance": MOCK_INSTANCE, "dev": MOCK_DEV, "curl": MOCK_CURL, "docker": MOCK_DOCKER}
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
        return args[args.index("remote.sh") + 7:] if "remote.sh" in args else None

    @staticmethod
    def remote_step(root: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        """remote.sh run for real against a temporary layout: ``root``, its venv in root/.venv, the local directory
        root/../ebs, and root's parent as the storage under zgca.py's quota and floor."""
        layout = (root, root / ".venv", root.parent / "ebs", root.parent, zgca.STORAGE_QUOTA, zgca.STORAGE_FLOOR)
        return subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), *map(str, layout), *args],
                              env=env, capture_output=True, text=True, check=False)

    def step(self, call: dict[str, object]) -> str:
        """A call's command: the remote.sh step for a remote step, else the entry's subcommand."""
        remote = self.remote(call)
        return remote[0] if remote is not None else call["args"][0]

    def run_main(self, *argv: str, **env: str) -> tuple[int, str]:
        """Run the CLI against the mocks; (exit code, stderr), and its stdout in ``self.out``."""
        saved = dict(os.environ)
        os.environ.clear()
        os.environ.update(self.env | env)
        err, out = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
                code = zgca.main(argv, repo_root=self.repo, tools=self.tools)
        finally:
            os.environ.clear()
            os.environ.update(saved)
        self.out = out.getvalue()
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
        # instance's layout: the root, its venv, the root's directory on the
        # instance's own disk, and the storage with its quota and floor in bytes.
        remote, local = zgca.REMOTE_ROOT, "/ebs/auto-trade"
        script = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        for call in dev:
            if self.remote(call) is not None:
                self.assertEqual(call["args"][:10], ["bash", "-lc", script, "remote.sh", remote, "/personal_fast_storage/envs/quant", local,
                                                     "/personal_fast_storage", str(100 * 2**30), str(40 * 2**30)])
        # No experiment runs, then the base tools; then transfers (by
        # destination) and remote steps in the order they ran: the tracked
        # list lands before the `tracked` step prunes by it, and both come
        # before the seed and the release; then the image, and only then the
        # pretrained store, to the instance's own disk once it is known to
        # fit; then the deployed commit, and the host report last.
        ran = [self.remote(call) or call["args"][:1] + call["args"][1:][-1:] for call in dev]
        self.assertEqual(ran, [
            ["idle", "deploy"], ["bootstrap"], ["dirs"], ["copy", f"dev:{remote}/"], ["put", f"{remote}/.env"],
            ["copy", f"dev:{remote}/.runtime/zgca/"], ["tracked"],
            ["copy", f"dev:{remote}/data/pit_views_seed_test/"], ["copy", f"dev:{remote}/data/pack_payloads/"],
            ["copy", f"dev:{remote}/data/research_releases/gen_test/"],
            ["prepare", zgca.DUCKDB_CLI_SHA256, zgca.CUDA_COMPAT_SHA256], ["local-disk"],
            ["copy", f"dev:{local}/pretrained/"], ["copy", f"dev:{remote}/.runtime/zgca/"], ["report"],
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
            [str(state / zgca.DUCKDB_CLI_NAME), str(state / zgca.CUDA_COMPAT_NAME), str(manifest)],
        )
        # The commit those files are travels alone, as the last transfer.
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True).stdout
        commit = state / zgca.DEPLOYED_COMMIT_NAME
        self.assertEqual(commit.read_text(encoding="utf-8"), head)
        self.assertEqual(transfers[6]["args"][-2:], [str(commit), f"dev:{remote}/.runtime/zgca/"])
        # The seed, the release and the store are mirrored, hard links kept;
        # so is the Packs' payload store, empty while no Pack has a payload.
        seed, payloads, release, store = transfers[2:6]
        for transfer in (seed, release, store):
            self.assertIn("--delete", transfer["args"])
            self.assertIn("-H", transfer["args"])
        self.assertEqual(store["args"][-2], f"{self.repo / zgca.PRETRAINED}/")
        self.assertIn("--delete", payloads["args"])
        self.assertEqual(payloads["args"][-2], f"{self.repo / 'data' / 'pack_payloads'}/")
        self.assertNotIn("--delete", transfers[0]["args"])
        # The release leaves behind exactly the text and minute datasets the
        # seed's configuration does not read, and keeps them where an earlier
        # deploy put them (no --delete-excluded).
        release = release["args"]
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

    def test_deploy_holds_every_paper_pack_to_its_manifest_and_mirrors_the_payloads(self) -> None:
        """A tracked Pack's payload that differs from its manifest stops the deploy
        before anything is fetched or reaches the instance; one as listed is
        mirrored there (rsync makes the directory, as it makes the seed's), and
        a payload no longer here is removed there."""
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        pack = self.repo / "configs" / "workspace_refs" / "paper_test"
        pack.mkdir(parents=True)
        (pack / "README.md").write_text("Reproduce papers/demo/.\n", encoding="utf-8")
        text = self.repo / "data" / "pack_payloads" / "paper_test" / "papers" / "demo" / "paper.txt"
        text.parent.mkdir(parents=True)
        text.write_text("Abstract\n", encoding="utf-8")
        (pack / PAYLOAD_MANIFEST).write_text(format_manifest(payload_digests(self.repo / "data" / "pack_payloads" / "paper_test", "store")), encoding="utf-8")
        git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        subprocess.run([*git, "add", "configs"], cwd=self.repo, check=True)
        subprocess.run([*git, "commit", "-q", "-m", "paper pack"], cwd=self.repo, check=True)
        text.write_text("Another text\n", encoding="utf-8")
        code, err = self.run_main("deploy", "data/pit_views_seed_test")
        self.assertEqual(code, 1)
        self.assertIn("Pack paper_test: data/pack_payloads/paper_test/papers/demo/paper.txt has sha256", err)
        self.assertFalse(self.calls())
        text.write_text("Abstract\n", encoding="utf-8")
        # The instance's file system as the dirs step leaves it.
        instance = Path(self.temp.name) / "instance"
        root = Path(str(instance) + zgca.REMOTE_ROOT)
        for directory in (root / ".runtime" / "zgca", root / "data" / "research_releases", Path(str(instance) + zgca.LOCAL_ROOT)):
            directory.mkdir(parents=True)
        shipped = root / "data" / "pack_payloads"
        for retired in ((), ("retired_pack", "paper.txt")):
            with self.subTest(retired=retired):
                if retired:
                    (shipped / retired[0]).mkdir()
                    (shipped.joinpath(*retired)).write_text("old\n", encoding="utf-8")
                code, err = self.run_main("deploy", "data/pit_views_seed_test", MOCK_RSYNC=str(instance))
                self.assertEqual(code, 0, err)
                self.assertEqual(sorted(path.relative_to(shipped).as_posix() for path in shipped.rglob("*") if path.is_file()),
                                 ["paper_test/papers/demo/paper.txt"])
                self.assertEqual((shipped / "paper_test" / "papers" / "demo" / "paper.txt").read_text(encoding="utf-8"), "Abstract\n")
        self.assertTrue((root / "configs" / "workspace_refs" / "paper_test" / PAYLOAD_MANIFEST).is_file())

    def test_a_deploy_that_stops_at_the_image_records_no_commit_on_the_instance(self) -> None:
        """The steps before the failure have shipped their files, but the commit
        `run` and `resume` require is only recorded by a deploy that completes."""
        self._tracked_repo()
        self._seed()
        self._pretrained()
        (self.repo / ".env").write_text(LOCAL_ENV, encoding="utf-8")
        instance = Path(self.temp.name) / "instance"
        root = Path(str(instance) + zgca.REMOTE_ROOT)
        for directory in (root / ".runtime" / "zgca", root / "data" / "research_releases", Path(str(instance) + zgca.LOCAL_ROOT)):
            directory.mkdir(parents=True)
        shipped = root / ".runtime" / "zgca"
        code, err = self.run_main("deploy", "data/pit_views_seed_test", MOCK_RSYNC=str(instance), MOCK_STEP_EXIT_PREPARE="1")
        self.assertEqual(code, 1)
        self.assertIn("remote step prepare failed", err)
        self.assertEqual(sorted(path.name for path in shipped.iterdir()),
                         sorted([zgca.CUDA_COMPAT_NAME, zgca.DUCKDB_CLI_NAME, zgca.TRACKED_LIST_NAME]))
        code, err = self.run_main("deploy", "data/pit_views_seed_test", MOCK_RSYNC=str(instance))
        self.assertEqual(code, 0, err)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True).stdout
        self.assertEqual((shipped / zgca.DEPLOYED_COMMIT_NAME).read_text(encoding="utf-8"), head)

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
                    self.assertEqual((code, ran[done + 1:]), (0, [["copy", "dev:/ebs/auto-trade/pretrained/"],
                                                                  ["copy", f"dev:{zgca.REMOTE_ROOT}/.runtime/zgca/"], ["report"]]), err)
                else:
                    # Everything else is done, the report shows the disks, and the deploy fails saying why,
                    # recording no commit.
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
        """Three places name each file: the Dockerfile pins and fetches the
        DuckDB CLI by default and the compatibility package only when told,
        with the digest told beside it; the deploy reads the CLI's pins from
        the Dockerfile, pins the package and ships both; remote.sh verifies
        and serves them and tells the build the package's digest."""
        dockerfile = zgca.SANDBOX_DOCKERFILE.read_text(encoding="utf-8")
        self.assertEqual(zgca.SANDBOX_DOCKERFILE, REPO / "ops" / "docker" / "sandbox.Dockerfile")
        self.assertIn(zgca.DUCKDB_CLI_URL, dockerfile)
        self.assertTrue(zgca.DUCKDB_CLI_URL.startswith("https://github.com/"))
        self.assertIn("ARG CUDA_COMPAT_URL=\nARG CUDA_COMPAT_SHA256=\n", dockerfile)
        self.assertIn("ENV LD_LIBRARY_PATH=${CUDA_COMPAT_URL:+/usr/local/cuda-compat}\n", dockerfile)
        self.assertEqual(zgca.CUDA_COMPAT_URL.rsplit("/", 1)[1], zgca.CUDA_COMPAT_NAME)
        remote = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(f"=.runtime/zgca/{zgca.DUCKDB_CLI_NAME}\n", remote)
        self.assertIn(f"=.runtime/zgca/{zgca.CUDA_COMPAT_NAME}\n", remote)
        self.assertIn('--build-arg "CUDA_COMPAT_SHA256=$compat_digest"', remote)

    def test_a_dockerfile_that_does_not_pin_a_release_once_is_an_error(self) -> None:
        """The deploy reads the CLI's pins out of the Dockerfile: a default that
        is missing, malformed or given twice stops it rather than being guessed."""
        dockerfile = Path(self.temp.name) / "sandbox.Dockerfile"
        self.addCleanup(setattr, zgca, "SANDBOX_DOCKERFILE", zgca.SANDBOX_DOCKERFILE)
        zgca.SANDBOX_DOCKERFILE = dockerfile
        pinned = "ARG DUCKDB_CLI_SHA256=" + "0" * 64 + "\n"
        dockerfile.write_text(pinned, encoding="utf-8")
        self.assertEqual(zgca.dockerfile_pin("DUCKDB_CLI_SHA256", "([0-9a-f]{64})"), "0" * 64)
        for text in ("", "ARG DUCKDB_CLI_SHA256=00\n", pinned * 2):
            dockerfile.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not give build arg DUCKDB_CLI_SHA256 one default"):
                zgca.dockerfile_pin("DUCKDB_CLI_SHA256", "([0-9a-f]{64})")

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

    def _head(self) -> str:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True).stdout.strip()

    def test_run_forwards_the_launcher_arguments_and_refuses_an_untracked_round(self) -> None:
        self._tracked_repo()
        code, err = self.run_main("run", "untracked.py", "--dry-run")
        self.assertEqual(code, 1)
        self.assertIn("not a tracked file", err)
        self.assertFalse(self.calls("dev"))
        deployed = self._head()
        code, err = self.run_main("run", "scripts/experiments/create_round_test.py", "--dry-run", "experiment_one", MOCK_DEPLOYED_COMMIT=deployed)
        self.assertEqual(code, 0, err)
        steps = [self.remote(call) for call in self.calls("dev")]
        self.assertEqual(steps, [["deployed-commit"], ["console-up"], ["round", "scripts/experiments/create_round_test.py", "--dry-run", "experiment_one"]])
        code, err = self.run_main("run", "scripts/experiments/create_round_test.py", MOCK_DEPLOYED_COMMIT=deployed, MOCK_STEP_EXIT_CONSOLE_UP="1")
        self.assertEqual(code, 1)
        self.assertIn("console-up failed", err)
        self.assertEqual(self.step(self.calls()[-1]), "console-up")

    def test_run_refuses_while_the_instance_runs_another_commit(self) -> None:
        """The round file is checked here, but the instance runs what the last deploy shipped: a run is refused
        before the console or a launcher starts unless that is the local HEAD, and with no deploy recorded."""
        self._tracked_repo()
        head = self._head()
        for recorded, said in (("0" * 40, f"the instance runs commit {'0' * 40}, not this checkout's HEAD {head}: deploy this commit first"),
                               ("", "the instance runs commit (none recorded)")):
            with self.subTest(recorded=recorded):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main("run", "scripts/experiments/create_round_test.py", "--fill", MOCK_DEPLOYED_COMMIT=recorded)
                self.assertEqual(code, 1)
                self.assertIn(said, err)
                self.assertEqual([self.step(call) for call in self.calls("dev")], ["deployed-commit"])
        # The instance's side: the commit the deploy shipped, or a refusal before any deploy.
        root = Path(self.temp.name) / "instance"
        missing = self.remote_step(root, "deployed-commit")
        self.assertEqual(missing.returncode, 1)
        self.assertIn("no deployed commit is recorded", missing.stderr)
        (root / ".runtime" / "zgca").mkdir(parents=True)
        (root / ".runtime" / "zgca" / zgca.DEPLOYED_COMMIT_NAME).write_text(head + "\n", encoding="utf-8")
        shown = self.remote_step(root, "deployed-commit")
        self.assertEqual((shown.returncode, shown.stdout), (0, head + "\n"), shown.stderr)

    def test_resume_restarts_an_unfinished_experiment_under_runs_preconditions_and_drops_its_marker(self) -> None:
        """Before anything changes, `resume` refuses an id that is none, an instance on another
        commit, an experiment that is running, absent or completed, and a storage below the floor;
        otherwise it brings the console up as `run` does, drops the sync marker so the experiment is
        copied again when it ends, and sends the console's resume, whose refusal is the command's."""
        from autotrade.pipelines.hitl_state import (
            ENDED_STATES,
            ExperimentState,
            proc_start_ticks,
        )

        # zgca.py names the states sync and resume decide by (importing hitl_state needs numpy); they are the console's.
        self.assertEqual(zgca.COMPLETED, ExperimentState.COMPLETED)
        self.assertEqual(set(zgca.NOT_RESUMED), ENDED_STATES - {ExperimentState.FAILED})
        self._tracked_repo()
        head = self._head()
        live = {"state": "running_session", "pid": os.getpid(), "pid_start_ticks": proc_start_ticks(os.getpid())}
        root, _ = self._sync_round({
            "exp_failed": {"state": "failed", "pid": None},
            "exp_terminated": {"state": "terminated", "pid": None},
            "exp_done": {"state": "completed", "pid": None},
            "exp_live": live,
            "exp_absent": None,
        }, name="create_round_resume")
        markers = self.repo / ".runtime" / "zgca" / zgca.SYNCED_DIR_NAME
        markers.mkdir(parents=True)
        for experiment in ("exp_failed", "exp_done"):
            (markers / experiment).write_text("synced\n", encoding="utf-8")
        env = {"MOCK_REMOTE_ROOT": str(root), "MOCK_DEPLOYED_COMMIT": head}
        below = "storage: 61.0 of 100 GiB used, 39.0 free (du of /personal_fast_storage), below the 40 GiB floor"
        for experiment, extra, said, ran in (
            ("../exp_failed", {}, "not an experiment id", []),
            ("exp_failed", {"MOCK_DEPLOYED_COMMIT": "0" * 40}, "deploy this commit first", ["deployed-commit"]),
            ("exp_done", {}, "exp_done ended completed: resume restarts a failed, terminated or interrupted experiment",
             ["deployed-commit", "experiment-state"]),
            ("exp_live", {}, "exp_live is still running", ["deployed-commit", "experiment-state"]),
            ("exp_absent", {}, "no experiment exp_absent on the instance", ["deployed-commit", "experiment-state"]),
            ("exp_failed", {"MOCK_STORAGE": below, "MOCK_STEP_EXIT_STORAGE": "1"},
             f"{below}; nothing is started on the instance while less than 40 GiB is free under the quota",
             ["deployed-commit", "experiment-state", "storage"]),
        ):
            with self.subTest(experiment=experiment, refused=said):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main("resume", experiment, **env | extra)
                self.assertEqual(code, 1)
                self.assertIn(said, err)
                self.assertEqual([self.step(call) for call in self.calls("dev")], ran)
                self.assertEqual(self.marker("exp_failed"), "synced\n")
        for experiment in ("exp_failed", "exp_terminated"):
            with self.subTest(experiment=experiment):
                self.log.unlink()
                code, err = self.run_main("resume", experiment, **env)
                self.assertEqual(code, 0, err)
                self.assertEqual([self.remote(call) for call in self.calls("dev")],
                                 [["deployed-commit"], ["experiment-state", experiment], ["storage"], ["console-up"], ["resume", experiment]])
                self.assertIn("storage: 56.2 of 100 GiB used, 43.8 free (du of /personal_fast_storage)", self.out)
                self.assertIsNone(self.marker(experiment))
        # A resume the console refuses fails the command; the marker is gone already, so the next sync pass copies it again.
        (markers / "exp_failed").write_text("synced\n", encoding="utf-8")
        code, err = self.run_main("resume", "exp_failed", **env, MOCK_STEP_EXIT_RESUME="1")
        self.assertEqual(code, 1)
        self.assertIn("remote step resume failed (exit 1)", err)
        self.assertIsNone(self.marker("exp_failed"))
        self.assertEqual(self.marker("exp_done"), "synced\n")

    def _copy_here(self, experiment: str, state: str, image: str = "") -> None:
        """The copy a sync left here (the mock copy moves nothing): created on the instance, in ``state``, pinning ``image``."""
        hitl = self.repo / "experiments" / experiment / "hitl"
        hitl.mkdir(parents=True)
        (hitl / "params.json").write_text(json.dumps({"experiments_root": f"{zgca.REMOTE_ROOT}/experiments"}), encoding="utf-8")
        (hitl / "status.json").write_text(json.dumps({"schema_version": 1, "state": state}), encoding="utf-8")
        if image:
            (hitl / "sandbox_image.json").write_text(json.dumps({"experiment_id": experiment, "image_ref": image}), encoding="utf-8")

    def test_only_a_synced_copy_of_a_completed_experiment_is_released_on_the_instance(self) -> None:
        """The release follows the verified copy alone: one synced and completed is released (one synced
        before releases existed included) and marked `released`, which costs nothing afterwards; a
        synced copy of a failed experiment releases nothing; a completed copy whose tag was refused is
        released all the same, stays `copied` and has its tag (and release) tried again by every pass
        until the images agree; a refused release is one line and is tried again next pass; a sync by
        hand releases as a pass does."""
        image = "autotrade-sandbox:exp_copied-base-11111111-1111-4111-8111-111111111111"
        root, round_file = self._sync_round({
            "exp_done": {"state": "completed", "pid": None},
            "exp_old": {"state": "completed", "pid": None},
            "exp_failed": {"state": "failed", "pid": None},
            "exp_copied": {"state": "completed", "pid": None},
        })
        for experiment, state in (("exp_done", "completed"), ("exp_old", "completed"), ("exp_failed", "failed")):
            self._copy_here(experiment, state)
        self._copy_here("exp_copied", "completed", image)
        markers = self.repo / ".runtime" / "zgca" / zgca.SYNCED_DIR_NAME
        markers.mkdir(parents=True)
        for experiment, stage in (("exp_old", "synced"), ("exp_failed", "synced"), ("exp_copied", "copied")):
            (markers / experiment).write_text(stage + "\n", encoding="utf-8")
        env = {"MOCK_REMOTE_ROOT": str(root), "MOCK_REMOTE_RUNTIME": "a" * 64}

        def ran() -> list[list[str]]:
            return [self.remote(call) or call["args"][-1:] for call in self.calls("dev")]

        code, err = self.run_main("sync-rounds", round_file, **env, MOCK_LOCAL_RUNTIME="b" * 64)
        self.assertEqual(code, 1, err)
        self.assertEqual(ran(), [
            ["experiment-state", "exp_done"], [f"{self.repo}/experiments/exp_done/"], [f"{self.repo}/logs/workers/exp_done.log"],
            ["release", "exp_done"], ["release", "exp_old"], ["image-runtime", image], ["release", "exp_copied"],
        ])
        self.assertIn("exp_copied: copied, not tagged, tried again next pass: experiments/exp_copied is synced, but its strategy cannot be followed", self.out)
        self.assertTrue(self.out.splitlines()[-1].endswith(": 4 experiments: 3 synced, 1 untagged, 0 running, 0 interrupted, 0 pending, 0 failed"))
        self.assertEqual({experiment: self.marker(experiment) for experiment in ("exp_done", "exp_old", "exp_failed", "exp_copied")},
                         {"exp_done": "released\n", "exp_old": "released\n", "exp_failed": "synced\n", "exp_copied": "copied\n"})

        # The instance refuses a release (it does not show the experiment completed and finished for good).
        (markers / "exp_old").write_text("synced\n", encoding="utf-8")
        self.log.unlink()
        code, err = self.run_main("sync-rounds", round_file, **env, MOCK_LOCAL_RUNTIME="b" * 64, MOCK_STEP_EXIT_RELEASE="1")
        self.assertEqual(code, 1)
        self.assertEqual(ran(), [["release", "exp_old"], ["image-runtime", image], ["release", "exp_copied"]])
        for experiment in ("exp_old", "exp_copied"):
            self.assertIn(f"{experiment}: failed, retried next pass: remote step release failed (exit 1)", self.out)
        self.assertTrue(self.out.splitlines()[-1].endswith(": 4 experiments: 2 synced, 0 untagged, 0 running, 0 interrupted, 0 pending, 2 failed"))
        self.assertEqual([self.marker("exp_old"), self.marker("exp_copied")], ["synced\n", "copied\n"])

        # Released next pass; the copied one, tagged once the images agree, is released in the same pass.
        self.log.unlink()
        code, err = self.run_main("sync-rounds", round_file, **env, MOCK_LOCAL_RUNTIME="a" * 64)
        self.assertEqual(code, 0, err)
        self.assertEqual(ran(), [["release", "exp_old"], ["image-runtime", image], ["release", "exp_copied"]])
        self.assertTrue(self.out.splitlines()[-1].endswith(": 4 experiments: 4 synced, 0 untagged, 0 running, 0 interrupted, 0 pending, 0 failed"))
        self.assertEqual([self.marker(experiment) for experiment in ("exp_old", "exp_copied", "exp_failed")], ["released\n", "released\n", "synced\n"])
        # Now the pass asks neither the platform nor the instance anything.
        self.log.unlink()
        self.assertEqual(self.run_main("sync-rounds", round_file, **env)[0], 0)
        self.assertFalse(self.calls())

        # By hand: a failed copy keeps everything; a completed one is released again, which frees what is left, if anything.
        for experiment, after in (("exp_failed", [["experiment-state", "exp_failed"], ["copy"], ["copy"]]),
                                  ("exp_done", [["experiment-state", "exp_done"], ["copy"], ["copy"], ["release", "exp_done"]])):
            with self.subTest(sync=experiment):
                self.log.unlink(missing_ok=True)
                code, err = self.run_main("sync", experiment, **env)
                self.assertEqual(code, 0, err)
                self.assertEqual([self.remote(call) or call["args"][:1] for call in self.calls("dev")], after)
                self.assertEqual(self.marker(experiment), "synced\n" if experiment == "exp_failed" else "released\n")

    def test_a_copy_whose_pinned_image_is_gone_from_the_instance_is_released_and_settled_in_one_line(self) -> None:
        """Tags are lost across stop/start and an out-of-disk rebuild prunes every unused image, so a
        copied experiment's pinned image can be gone for good. The pass still releases it, says so in one
        line, marks it settled and succeeds, and no later pass asks about it again; a sync by hand
        releases it the same way and fails saying that no tag was made."""
        image = "autotrade-sandbox:exp_lost-base-11111111-1111-4111-8111-111111111111"
        root, round_file = self._sync_round({"exp_lost": {"state": "completed", "pid": None}})
        self._copy_here("exp_lost", "completed", image)
        markers = self.repo / ".runtime" / "zgca" / zgca.SYNCED_DIR_NAME
        markers.mkdir(parents=True)
        (markers / "exp_lost").write_text("copied\n", encoding="utf-8")
        env = {"MOCK_REMOTE_ROOT": str(root), "MOCK_STEP_EXIT_IMAGE_RUNTIME": str(zgca.REMOTE_ABSENT)}

        code, err = self.run_main("sync-rounds", round_file, **env)
        self.assertEqual(code, 0, err)
        self.assertEqual([self.remote(call) for call in self.calls("dev")], [["image-runtime", image], ["release", "exp_lost"]])
        self.assertFalse(self.calls("docker"))
        said = self.out.splitlines()
        self.assertEqual(len(said), 2, said)
        self.assertIn(f"exp_lost: synced without a tag, and no later pass tries again: experiments/exp_lost is synced, but its pinned "
                      f"image {image} is no longer on the instance", said[0])
        self.assertTrue(said[1].endswith(": 1 experiments: 1 synced, 0 untagged, 0 running, 0 interrupted, 0 pending, 0 failed"), said[1])
        self.assertEqual(self.marker("exp_lost"), "released\n")
        for _ in range(2):
            self.log.unlink(missing_ok=True)
            self.assertEqual(self.run_main("sync-rounds", round_file, **env)[0], 0)
            self.assertFalse(self.calls())

        self.log.unlink(missing_ok=True)
        code, err = self.run_main("sync", "exp_lost", **env)
        self.assertEqual(code, 1)
        self.assertIn(f"its pinned image {image} is no longer on the instance", err)
        self.assertEqual([self.step(call) for call in self.calls("dev")], ["experiment-state", "copy", "copy", "image-runtime", "release"])
        self.assertEqual(self.marker("exp_lost"), "released\n")

    def test_the_release_step_frees_a_completed_experiments_bulk_once_and_refuses_anything_else(self) -> None:
        """Run for real against a temporary instance root, with prune_pit_views.py deciding: a completed
        experiment loses its PIT views, sandbox work root and archived session workspaces and keeps its
        directory with every record; a second release frees nothing. A failed experiment, an id that is
        none, an absent experiment and a tree reached through a link are refused, and nothing is deleted."""
        from tests.unit.support.webui_research_experiment import build_experiment

        root = Path(self.temp.name) / "instance"
        (root / ".venv" / "bin").mkdir(parents=True)
        (root / ".venv" / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        (root / ".venv" / "bin" / "python").chmod(0o700)
        (root / "scripts").symlink_to(REPO / "scripts", target_is_directory=True)
        bulk = {}
        for experiment, stage in (("exp_done", "registered_with_record"), ("exp_failed", "broken")):
            directory = build_experiment(root / "experiments", experiment, stage)
            bulk[experiment] = [directory / "pit_views" / "decision" / "daily.parquet",
                                directory / "artifacts" / "run_research_1" / "workspace" / "notes" / "notes.md",
                                root / ".runtime" / "sandboxes" / experiment / "research" / "agent" / "notes.md"]
            for path in bulk[experiment]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * 65536)
            (directory / "artifacts" / "run_research_1" / "run_manifest.json").write_text("{}", encoding="utf-8")
        done = root / "experiments" / "exp_done"
        records = sorted(path for path in done.rglob("*") if path.is_file() and not any(path.is_relative_to(tree.parents[1]) for tree in bulk["exp_done"][:2]))
        self.assertIn(done / "artifacts" / "run_research_1" / "run_manifest.json", records)

        def release(experiment: str) -> subprocess.CompletedProcess[str]:
            return self.remote_step(root, "release", experiment)

        link = done / "artifacts" / "run_research_2"
        link.symlink_to(root / "experiments" / "exp_failed" / "artifacts" / "run_research_1", target_is_directory=True)
        for experiment, said in (
            ("exp_failed", "exp_failed is kept whole, state 'failed' is not completed: a resume continues in its workspace"),
            ("..", "not an experiment id: '..'"),
            ("../exp_done", "not an experiment id: '../exp_done'"),
            ("exp_absent", "no experiment directory experiments/exp_absent on the instance"),
            ("exp_done", f"{Path('experiments/exp_done/artifacts/run_research_2')} is a symbolic link; nothing of exp_done was released"),
        ):
            with self.subTest(experiment=experiment):
                refused = release(experiment)
                self.assertEqual(refused.returncode, 1)
                self.assertIn(said, refused.stderr)
                self.assertTrue(all(path.is_file() for paths in bulk.values() for path in paths))
        link.unlink()

        freed = release("exp_done")
        self.assertEqual(freed.returncode, 0, freed.stderr)
        self.assertRegex(freed.stdout, r"^exp_done: released on the instance, freed \d+\.\d\d GiB \(pit_views \d+\.\d\d GiB, sandbox \d+\.\d\d GiB, "
                                       r"run_research_1/workspace \d+\.\d\d GiB\); its records stay\n$")
        self.assertFalse((done / "pit_views").exists())
        self.assertFalse((done / "artifacts" / "run_research_1" / "workspace").exists())
        self.assertFalse((root / ".runtime" / "sandboxes" / "exp_done").exists())
        self.assertEqual(sorted(path for path in done.rglob("*") if path.is_file()), records)
        self.assertTrue(all(path.is_file() for path in bulk["exp_failed"]))
        again = release("exp_done")
        self.assertEqual((again.returncode, again.stdout),
                         (0, "exp_done: released on the instance, freed 0.00 GiB, nothing was left; its records stay\n"), again.stderr)

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
                # A first copy is made in the state directory and enters experiments/ once verified; a copy over one is made in place.
                landing = self.repo / ".runtime" / "zgca" / zgca.INCOMING_DIR_NAME if experiment == "experiment_new" else self.repo / "experiments"
                pull = dev[1]["args"]
                self.assertEqual(pull[-2:], [f"dev:{zgca.REMOTE_ROOT}/experiments/{experiment}/", f"{landing / experiment}/"])
                self.assertTrue((self.repo / "experiments" / experiment).is_dir())
                self.assertFalse((self.repo / ".runtime" / "zgca" / zgca.INCOMING_DIR_NAME / experiment).exists())
                self.assertEqual(pull[pull.index("--exclude") + 1], "/pit_views/")
                self.assertNotIn("--delete", pull)
                self.assertEqual(dev[2]["args"][-2:], [f"dev:{zgca.REMOTE_ROOT}/logs/workers/{experiment}.log", f"{self.repo / 'logs' / 'workers' / experiment}.log"])
                self.log.unlink()

    def test_sync_tags_the_pinned_image_here_only_when_both_images_bake_one_runtime(self) -> None:
        """After the copy, the experiment's pinned tag is made here as a clone of the
        base image, exactly as a local experiment's is, but only when the image it
        names on the instance and this host's base bake the same strategy runtime.
        Otherwise the records are synced and the command fails saying why, with no
        tag; a sync run again does the same again."""

        experiment = "experiment_zgca"
        copied = self.repo / "experiments" / experiment / "hitl"
        copied.mkdir(parents=True)
        (copied / "params.json").write_text(json.dumps({"experiments_root": f"{zgca.REMOTE_ROOT}/experiments"}), encoding="utf-8")
        image = f"autotrade-sandbox:{experiment}-base-11111111-1111-4111-8111-111111111111"
        (copied / "sandbox_image.json").write_text(json.dumps({"experiment_id": experiment, "image_ref": image}), encoding="utf-8")
        runtime, other = "a" * 64, "b" * 64

        code, err = self.run_main("sync", experiment, MOCK_REMOTE_RUNTIME=runtime, MOCK_LOCAL_RUNTIME=other)
        self.assertEqual(code, 1)
        self.assertIn(f"cannot be followed on Paper here until the images agree: its image {image} bakes strategy runtime {runtime}", err)
        self.assertEqual([self.step(call) for call in self.calls("dev")], ["experiment-state", "copy", "copy", "image-runtime"])
        self.assertEqual(self.remote(self.calls("dev")[-1]), ["image-runtime", image])
        self.assertEqual([call["args"] for call in self.calls("docker")], [["run", "autotrade-sandbox:latest"]])
        for attempt in (1, 2):
            with self.subTest(attempt=attempt):
                self.log.unlink()
                code, err = self.run_main("sync", experiment, MOCK_REMOTE_RUNTIME=runtime, MOCK_LOCAL_RUNTIME=runtime)
                self.assertEqual(code, 0, err)
                self.assertEqual([call["args"] for call in self.calls("docker")],
                                 [["run", "autotrade-sandbox:latest"], ["tag", "autotrade-sandbox:latest", image]])
        # An image the instance no longer has cannot say what it baked.
        self.log.unlink()
        code, err = self.run_main("sync", experiment, MOCK_STEP_EXIT_IMAGE_RUNTIME="1")
        self.assertEqual(code, 1)
        self.assertIn("remote step image-runtime failed", err)
        self.assertFalse(self.calls("docker"))

    def test_the_image_runtime_step_reads_an_image_on_the_instance_or_refuses_an_absent_one(self) -> None:
        root = Path(self.temp.name) / "instance"
        venv = root / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        (venv / "bin" / "python").chmod(0o700)
        stubs = Path(self.temp.name) / "stubs"
        stubs.mkdir()
        (stubs / "docker").write_text(
            '#!/bin/sh\nif [ "$1" = image ]; then [ "$3" = "$PRESENT" ]; exit; fi\n'
            'echo \'{"runtime": "r" , "contract": ""}\'\n', encoding="utf-8",
        )
        (stubs / "docker").chmod(0o700)
        env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "PYTHONPATH": str(REPO / "src")}
        image = "autotrade-sandbox:experiment-base-11111111-1111-4111-8111-111111111111"
        absent = self.remote_step(root, "image-runtime", image, env=env)
        self.assertEqual(absent.returncode, zgca.REMOTE_ABSENT)
        self.assertIn(f"image {image} is not on the instance", absent.stderr)
        shown = self.remote_step(root, "image-runtime", image, env=env | {"PRESENT": image})
        self.assertEqual((shown.returncode, shown.stdout), (0, "r\n"), shown.stderr)

    def test_sync_stops_before_any_transfer_when_the_experiment_state_step_refuses(self) -> None:
        code, err = self.run_main("sync", "experiment_running", MOCK_STEP_EXIT_EXPERIMENT_STATE="1")
        self.assertEqual(code, 1)
        self.assertIn("remote step experiment-state failed", err)
        self.assertEqual([self.step(call) for call in self.calls("dev")], ["experiment-state"])

    def _sync_round(self, experiments: dict[str, object], name: str = "create_round_test") -> tuple[Path, str]:
        """A temporary instance root whose venv runs this interpreter, holding the status of each experiment
        given one (a string is written as is), and a round file here listing them all, in order."""
        root = Path(self.temp.name) / "instance"
        (root / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
        (root / ".venv" / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
        (root / ".venv" / "bin" / "python").chmod(0o700)
        for experiment, record in experiments.items():
            if record is not None:
                (root / "experiments" / experiment / "hitl").mkdir(parents=True, exist_ok=True)
                (root / "experiments" / experiment / "hitl" / "status.json").write_text(
                    record if isinstance(record, str) else json.dumps(record), encoding="utf-8")
        round_file = self.repo / "scripts" / "experiments" / f"{name}.py"
        round_file.parent.mkdir(parents=True, exist_ok=True)
        round_file.write_text(f"from types import SimpleNamespace\nROUND = SimpleNamespace(experiments=dict.fromkeys({list(experiments)!r}))\n", encoding="utf-8")
        return root, round_file.relative_to(self.repo).as_posix()

    def marker(self, experiment: str) -> str | None:
        path = self.repo / ".runtime" / "zgca" / zgca.SYNCED_DIR_NAME / experiment
        return path.read_text(encoding="utf-8") if path.exists() else None

    def test_sync_rounds_brings_back_each_ended_experiment_once(self) -> None:
        """One cron pass against the real experiment-state step on a temporary instance root:
        only an ended experiment is copied, then tagged and marked; a running, interrupted or
        absent one costs one query and nothing is copied; a failure is one line and the next
        experiment goes ahead; a refused tag is tried again without a new copy; an experiment
        marked synced is never asked about again."""
        from autotrade.pipelines.hitl_state import proc_start_ticks

        live = {"state": "running_session", "pid": os.getpid(), "pid_start_ticks": proc_start_ticks(os.getpid())}
        root, round_file = self._sync_round({
            "exp_broken": "{",  # a status the instance cannot read: the query fails
            "exp_done": {"state": "completed", "verdict": {"status": "registered_with_record"}, "pid": None},
            "exp_live": live,
            "exp_gone": {**live, "pid_start_ticks": live["pid_start_ticks"] + 1},
            "exp_image": {"state": "failed", "pid": None},
            "exp_absent": None,
        })
        # The mock copy moves nothing, so exp_image's records are here already, pinning an image.
        hitl = self.repo / "experiments" / "exp_image" / "hitl"
        hitl.mkdir(parents=True)
        (hitl / "params.json").write_text(json.dumps({"experiments_root": f"{zgca.REMOTE_ROOT}/experiments"}), encoding="utf-8")
        image = "autotrade-sandbox:exp_image-base-11111111-1111-4111-8111-111111111111"
        (hitl / "sandbox_image.json").write_text(json.dumps({"experiment_id": "exp_image", "image_ref": image}), encoding="utf-8")
        env = {"MOCK_REMOTE_ROOT": str(root), "MOCK_REMOTE_RUNTIME": "a" * 64}

        def ran() -> list[list[str]]:
            return [self.remote(call)[:2] if self.remote(call) else call["args"][-1:] for call in self.calls("dev")]

        code, err = self.run_main("sync-rounds", round_file, **env, MOCK_LOCAL_RUNTIME="b" * 64)
        self.assertEqual(code, 1, err)
        local, incoming = self.repo / "experiments", self.repo / ".runtime" / "zgca" / zgca.INCOMING_DIR_NAME
        self.assertEqual(ran(), [
            ["experiment-state", "exp_broken"],
            ["experiment-state", "exp_done"], [f"{incoming}/exp_done/"], [f"{self.repo}/logs/workers/exp_done.log"],
            ["experiment-state", "exp_live"], ["experiment-state", "exp_gone"],
            ["experiment-state", "exp_image"], [f"{local}/exp_image/"], [f"{self.repo}/logs/workers/exp_image.log"],
            ["image-runtime", image],
            ["experiment-state", "exp_absent"],
        ])
        self.assertEqual([call["args"] for call in self.calls("instance")], [["state", INSTANCE]])
        lines = self.out.splitlines()
        self.assertTrue(lines[-1].endswith(
            " zgca sync create_round_test: 6 experiments: 1 synced, 1 untagged, 1 running, 1 interrupted, 1 pending, 1 failed"), lines[-1])
        self.assertIn("exp_broken: failed, retried next pass: remote step experiment-state failed (exit 1): json.decoder.JSONDecodeError", self.out)
        self.assertIn("exp_gone: state running_session, verdict -, sessions None/None: not ended, yet no worker runs it", self.out)
        self.assertIn(f"exp_image: copied, not tagged, tried again next pass: experiments/exp_image is synced, but its strategy cannot "
                      f"be followed on Paper here until the images agree: its image {image}", self.out)
        self.assertEqual({experiment: self.marker(experiment) for experiment in ("exp_broken", "exp_done", "exp_live", "exp_gone", "exp_image", "exp_absent")},
                         {"exp_broken": None, "exp_done": "synced\n", "exp_live": None, "exp_gone": None, "exp_image": "copied\n", "exp_absent": None})

        # Once the images agree the copied experiment is only tagged; the synced one is not asked about.
        self.log.unlink()
        code, err = self.run_main("sync-rounds", round_file, **env, MOCK_LOCAL_RUNTIME="a" * 64)
        self.assertEqual(code, 1, err)
        self.assertEqual(ran(), [["experiment-state", "exp_broken"], ["experiment-state", "exp_live"], ["experiment-state", "exp_gone"],
                                 ["image-runtime", image], ["experiment-state", "exp_absent"]])
        self.assertEqual([call["args"] for call in self.calls("docker")], [["run", "autotrade-sandbox:latest"], ["tag", "autotrade-sandbox:latest", image]])
        self.assertTrue(self.out.splitlines()[-1].endswith(": 6 experiments: 2 synced, 0 untagged, 1 running, 1 interrupted, 1 pending, 1 failed"))
        self.assertEqual(self.marker("exp_image"), "synced\n")

        # A round whose experiments are all back asks neither the platform nor the instance anything.
        _, synced_round = self._sync_round({"exp_done": None, "exp_image": None}, name="create_round_back")
        self.log.unlink()
        code, err = self.run_main("sync-rounds", synced_round, **env)
        self.assertEqual(code, 0, err)
        self.assertFalse(self.calls())
        self.assertTrue(self.out.splitlines()[-1].endswith(" zgca sync create_round_back: 2 experiments: 2 synced, 0 untagged, 0 running, 0 interrupted, 0 pending, 0 failed"))
        # No round at all is the steady state between rounds; a round file that is not there fails the pass.
        self.assertEqual(self.run_main("sync-rounds")[0], 0)
        self.assertTrue(self.out.splitlines()[-1].endswith(" zgca sync no round: 0 experiments: 0 synced, 0 untagged, 0 running, 0 interrupted, 0 pending, 0 failed"))
        code, err = self.run_main("sync-rounds", "scripts/experiments/create_round_missing.py")
        self.assertEqual(code, 1)
        self.assertIn("create_round_missing.py is not a round file", err)
        self.assertFalse(self.calls())

    def test_sync_rounds_retries_what_a_pass_could_not_finish_and_never_marks_a_partial_copy(self) -> None:
        """A failed copy leaves no marker and the next pass copies again; a sync run by hand
        removes the marker before it copies, so a copy it does not finish is synced by the
        next pass; a stopped or unreachable instance is one summary line and nothing is asked."""
        root, round_file = self._sync_round({
            "exp_one": {"state": "completed", "pid": None},
            "exp_two": {"state": "stopped", "pid": None},
            "exp_absent": None,
        })
        env = {"MOCK_REMOTE_ROOT": str(root)}
        code, err = self.run_main("sync-rounds", round_file, **env, MOCK_DEV_EXIT_COPY="1")
        self.assertEqual(code, 1, err)
        self.assertEqual([self.step(call) for call in self.calls("dev")],
                         ["experiment-state", "copy", "experiment-state", "copy", "experiment-state"])
        self.assertIn("exp_one: failed, retried next pass: dev copy failed (exit 1)", self.out)
        self.assertIn("exp_two: failed, retried next pass: dev copy failed (exit 1)", self.out)
        self.assertTrue(self.out.splitlines()[-1].endswith(": 3 experiments: 0 synced, 0 untagged, 0 running, 0 interrupted, 1 pending, 2 failed"))
        self.assertEqual([self.marker("exp_one"), self.marker("exp_two")], [None, None])
        code, err = self.run_main("sync-rounds", round_file, **env)
        self.assertEqual(code, 0, err)
        self.assertTrue(self.out.splitlines()[-1].endswith(": 3 experiments: 2 synced, 0 untagged, 0 running, 0 interrupted, 1 pending, 0 failed"))
        self.assertEqual([self.marker("exp_one"), self.marker("exp_two")], ["synced\n", "synced\n"])

        # A sync by hand that dies in its copy takes the marker with it (the mock copy moved nothing: the records it brought stand here).
        (self.repo / "experiments" / "exp_one" / "hitl").mkdir()
        (self.repo / "experiments" / "exp_one" / "hitl" / "params.json").write_text(
            json.dumps({"experiments_root": f"{zgca.REMOTE_ROOT}/experiments"}), encoding="utf-8")
        code, err = self.run_main("sync", "exp_one", **env, MOCK_DEV_EXIT_COPY="1")
        self.assertEqual(code, 1)
        self.assertIsNone(self.marker("exp_one"))
        self.log.unlink()
        self.assertEqual(self.run_main("sync-rounds", round_file, **env)[0], 0)
        self.assertEqual([self.remote(call) or call["args"][:1] for call in self.calls("dev")][:3],
                         [["experiment-state", "exp_one"], ["copy"], ["copy"]])
        self.assertEqual(self.marker("exp_one"), "synced\n")

        # The instance cannot be asked: one line, no query, the next pass tries again.
        for env_case, reason in (({"MOCK_STATE": "STOPPED"}, f"instance {INSTANCE} is STOPPED, not RUNNING"),
                                 ({"MOCK_INSTANCE_EXIT_STATE": "1"}, f"instance state {INSTANCE} failed (exit 1)")):
            with self.subTest(reason=reason):
                self.log.unlink()
                code, err = self.run_main("sync-rounds", round_file, **env, **env_case)
                self.assertEqual(code, 1)
                self.assertFalse(self.calls("dev"))
                self.assertEqual(len(self.out.splitlines()), 1)
                self.assertIn(": 3 experiments: 2 synced, 0 untagged, 0 running, 0 interrupted, 0 pending, 0 failed, 1 not checked "
                              f"({reason}", self.out)

    def test_a_copy_cut_short_is_continued_by_the_next_pass_and_a_directory_of_another_origin_is_still_refused(self) -> None:
        """Real rsync copies from a stand-in of the instance's file system, cut as a dropped link cuts them:
        rsync passes artifacts/ before hitl/, so the bulk arrives and hitl/params.json does not. A first
        copy cut short waits in the state directory, unmarked, and nothing is in experiments/; a directory
        that appears there meanwhile under the same id, of unknown origin or a local experiment, is refused
        and neither it nor the waiting copy is touched; the next pass continues the copy, verifies it and
        renames it into place. A copy over an earlier one, cut short in place, keeps that copy's
        hitl/params.json and is continued by the next pass too."""
        root, round_file = self._sync_round({"exp_cut": {"state": "failed", "pid": None}})
        remote = root / "experiments" / "exp_cut"
        (remote / "hitl" / "params.json").write_text(json.dumps({"experiments_root": f"{zgca.REMOTE_ROOT}/experiments"}), encoding="utf-8")
        for name, size in (("artifacts/run_001/workspace/model.bin", 1 << 20), ("artifacts/run_001/result.json", 100), ("pit_views/view.parquet", 100)):
            (remote / name).parent.mkdir(parents=True, exist_ok=True)
            (remote / name).write_bytes(os.urandom(size))
        (root / "logs" / "workers").mkdir(parents=True)
        (root / "logs" / "workers" / "exp_cut.log").write_text("worker log\n", encoding="utf-8")
        # The instance's file system, whose repository root is the instance root above.
        fs = Path(self.temp.name) / "instance_fs"
        (fs / zgca.REMOTE_ROOT.lstrip("/")).parent.mkdir(parents=True)
        (fs / zgca.REMOTE_ROOT.lstrip("/")).symlink_to(root)
        env = {"MOCK_REMOTE_ROOT": str(root), "MOCK_RSYNC": str(fs)}
        local = self.repo / "experiments" / "exp_cut"
        waiting = self.repo / ".runtime" / "zgca" / zgca.INCOMING_DIR_NAME / "exp_cut"

        def tree(path: Path) -> dict[str, bytes]:
            return {file.relative_to(path).as_posix(): file.read_bytes() for file in sorted(path.rglob("*")) if file.is_file()}

        def copied() -> dict[str, bytes]:
            return {name: data for name, data in tree(remote).items() if not name.startswith("pit_views/")}

        code, err = self.run_main("sync-rounds", round_file, **env, MOCK_RSYNC_CUT="1")
        self.assertEqual(code, 1, err)
        self.assertIn("exp_cut: failed, retried next pass: dev copy failed (exit 12)", self.out)
        self.assertFalse(local.exists())
        self.assertEqual(sorted(tree(waiting)), ["artifacts/run_001/result.json", "artifacts/run_001/workspace/model.bin"])
        self.assertIsNone(self.marker("exp_cut"))

        for origin, params in (("unknown", None), ("this host", {"experiments_root": str(self.repo / "experiments")})):
            with self.subTest(origin=origin):
                (local / "hitl").mkdir(parents=True)
                if params:
                    (local / "hitl" / "params.json").write_text(json.dumps(params), encoding="utf-8")
                (local / "notes.txt").write_text("not from the instance\n", encoding="utf-8")
                foreign, kept = tree(local), tree(waiting)
                self.log.unlink()
                code, err = self.run_main("sync-rounds", round_file, **env)
                self.assertEqual(code, 1)
                self.assertRegex(self.out, r"exp_cut: failed, retried next pass: experiments/exp_cut .*; refusing to overwrite it")
                self.assertFalse(self.calls("dev"))
                self.assertEqual((tree(local), tree(waiting)), (foreign, kept))
                shutil.rmtree(local)

        self.log.unlink()
        code, err = self.run_main("sync-rounds", round_file, **env)
        self.assertEqual(code, 0, err)
        self.assertEqual([self.step(call) for call in self.calls("dev")], ["experiment-state", "copy", "copy"])
        self.assertEqual(tree(local), copied())
        self.assertFalse(waiting.exists())
        self.assertEqual((self.repo / "logs" / "workers" / "exp_cut.log").read_text(encoding="utf-8"), "worker log\n")
        self.assertEqual(self.marker("exp_cut"), "synced\n")

        # Resumed and ended again: a sync by hand cut short in place keeps the copy's params.json, and the next pass finishes it.
        (remote / "artifacts" / "run_002").mkdir()
        (remote / "artifacts" / "run_002" / "result.json").write_bytes(os.urandom(100))
        (remote / "hitl" / "status.json").write_text(json.dumps({"state": "failed", "pid": None, "attempts": 2}), encoding="utf-8")
        code, err = self.run_main("sync", "exp_cut", **env, MOCK_RSYNC_CUT="1")
        self.assertEqual(code, 1)
        self.assertIn("dev copy failed (exit 12)", err)
        self.assertIsNone(self.marker("exp_cut"))
        self.assertIn("artifacts/run_002/result.json", tree(local))
        self.assertNotEqual(tree(local), copied())
        self.assertEqual((local / "hitl" / "params.json").read_bytes(), (remote / "hitl" / "params.json").read_bytes())
        code, err = self.run_main("sync-rounds", round_file, **env)
        self.assertEqual(code, 0, err)
        self.assertEqual(tree(local), copied())
        self.assertFalse(waiting.exists())
        self.assertEqual(self.marker("exp_cut"), "synced\n")

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
        self.assertEqual(refused.returncode, zgca.REMOTE_RUNNING)
        self.assertIn("experiment_live: state running_session", refused.stdout)
        self.assertIn("experiment_live is still running", refused.stderr)
        for experiment in ("experiment_gone", "experiment_done"):
            with self.subTest(experiment=experiment):
                shown = step(experiment)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                self.assertIn(f"{experiment}: state {experiments[experiment]['state']}", shown.stdout)
        missing = step("experiment_absent")
        self.assertEqual(missing.returncode, zgca.REMOTE_ABSENT)
        self.assertIn("no experiment experiment_absent", missing.stderr)
        # Any other failure is neither: a status the instance cannot read.
        (root / "experiments" / "experiment_broken" / "hitl").mkdir(parents=True)
        (root / "experiments" / "experiment_broken" / "hitl" / "status.json").write_text("{", encoding="utf-8")
        self.assertEqual(step("experiment_broken").returncode, 1)
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
        # A fill log is not the round log; its last pass and its end are shown,
        # and a round log older than it is a past round's, so it is not.
        fill = logs / "fill_create_round_test_20261019T210000Z.log"
        fill.write_text("fill create_round_test: slots; queue 1: 0 created, 1 pending (...)\n"
                        "fill create_round_test: slots; queue 1: 1 created, 0 pending (...)\nround-exit 0\n", encoding="utf-8")
        # The commit the last deploy shipped is shown once it is recorded.
        (root / ".runtime" / "zgca").mkdir(parents=True)
        (root / ".runtime" / "zgca" / zgca.DEPLOYED_COMMIT_NAME).write_text("0123abcd\n", encoding="utf-8")
        shown = self.remote_step(root, "status", env=env)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("deployed commit: 0123abcd\n", shown.stdout)
        self.assertNotIn("round log:", shown.stdout)
        self.assertIn(f"fill: no session; last log logs/zgca/{fill.name}\n  fill create_round_test: slots; queue 1: 1 created, 0 pending (...)\n  round-exit 0", shown.stdout)
        # A round launched after the fill loop is shown beside it.
        os.utime(fill, (2, 2))
        os.utime(logs / "create_round_test_20261019T190000Z.log", (3, 3))
        shown = self.remote_step(root, "status", env=env)
        self.assertIn("round log: logs/zgca/create_round_test_20261019T190000Z.log", shown.stdout)
        self.assertIn(f"fill: no session; last log logs/zgca/{fill.name}", shown.stdout)

    FILL_HELD = "fill create_round_test: slots; queue 1: 0 skipped (created already), 0 created, 0 refused, 1 pending (1 by the free GPUs)"
    FILL_DONE = FILL_HELD.replace("0 created", "1 created").replace("1 pending", "0 pending")

    def _fill(self, plan: list[str], du: list[str] | None = None) -> tuple[list[str], list[str]]:
        """One round step with --fill, run for real against a temporary root with tmux, sleep,
        the launcher and (given ``du``) du stubbed: each launcher pass prints and exits as its
        ``<exit>|<line>`` in ``plan`` says, each du answers its line of ``du`` (bytes used,
        ``fail``, or ``hang``). The fill log's lines and the seconds slept between passes."""
        temp = Path(self.temp.name)
        root, stubs = temp / "instance", temp / "stubs"
        for directory in (root / ".venv" / "bin", root / "scripts" / "experiments", root / "logs" / "zgca", stubs):
            directory.mkdir(parents=True, exist_ok=True)
        (root / "scripts" / "experiments" / "create_round_test.py").write_text("", encoding="utf-8")
        scripts = {
            # The console's health probe answers; each launcher pass prints and exits as planned.
            root / ".venv" / "bin" / "python": '#!/bin/sh\n[ "$1" = -c ] && { echo "{}"; exit 0; }\n'
            'n=$(($(cat "$PASSES" 2>/dev/null || echo 0) + 1)); echo $n >"$PASSES"\n'
            'line=$(sed -n "${n}p" "$PLAN"); echo "${line#*|}"; exit "${line%%|*}"\n',
            # No session exists yet; a new one runs its command at once, in its directory, under sh.
            stubs / "tmux": '#!/bin/sh\n[ "$1" = new-session ] || exit 1\n'
            'while [ $# -gt 1 ]; do [ "$1" = -c ] && { shift; cd "$1"; }; shift; done\nexec sh -c "$1"\n',
            stubs / "sleep": '#!/bin/sh\necho "$1" >>"$SLEPT"\n',
        }
        if du is not None:  # du -sxB1 <storage>: the planned total, or a failure as du reports one
            scripts[stubs / "du"] = ('#!/bin/sh\nn=$(($(cat "$MEASURED" 2>/dev/null || echo 0) + 1)); echo $n >"$MEASURED"\n'
                                     'line=$(sed -n "${n}p" "$DU_PLAN")\n'
                                     '[ "$line" = fail ] && { printf "%s\\n" "du: cannot read directory \'$2/x\': Permission denied" >&2; exit 1; }\n'
                                     '[ "$line" = hang ] && exec tail -f /dev/null\n'
                                     'printf "%s\\t%s\\n" "$line" "$2"\n')
            (temp / "du_plan").write_text("\n".join(du) + "\n", encoding="utf-8")
        for script, body in scripts.items():
            script.write_text(body, encoding="utf-8")
            script.chmod(0o700)
        for leftover in (temp / "passes", temp / "slept", temp / "measured", *(root / "logs" / "zgca").glob("*.log")):
            leftover.unlink(missing_ok=True)
        (temp / "plan").write_text("\n".join(plan) + "\n", encoding="utf-8")
        env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "PLAN": str(temp / "plan"), "PASSES": str(temp / "passes"),
               "SLEPT": str(temp / "slept"), "DU_PLAN": str(temp / "du_plan"), "MEASURED": str(temp / "measured")}
        shown = self.remote_step(root, "round", "scripts/experiments/create_round_test.py", "--fill", env=env)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("fill launcher started in tmux session autotrade-fill", shown.stdout)
        self.assertIn("each held while less than 40 GiB is free under the storage quota", shown.stdout)
        (log,) = (root / "logs" / "zgca").glob("fill_create_round_test_*.log")
        slept = temp / "slept"
        return log.read_text(encoding="utf-8").splitlines(), slept.read_text(encoding="utf-8").split() if slept.exists() else []

    def test_a_fill_repeats_until_nothing_is_pending_and_ends_at_a_failed_pass(self) -> None:
        """The round step's --fill loop: every pass appends to one log, the loop sleeps between
        passes and ends at the first pass that reports 0 pending, or at the first that exits
        non-zero, which is never retried. The storage here is a small temporary tree, measured by
        the real du, so no pass is held."""
        for plan, ending in (([f"0|{self.FILL_HELD}", f"0|{self.FILL_HELD}", f"0|{self.FILL_DONE}"], "round-exit 0"),
                             ([f"0|{self.FILL_HELD}", "1|experiment_one: refused, HTTP 400"], "round-exit 1")):
            with self.subTest(ending=ending):
                lines, slept = self._fill(plan)
                self.assertEqual(lines, [step.split("|", 1)[1] for step in plan] + [ending])
                self.assertEqual(slept, ["600"] * (len(plan) - 1))

    def test_a_fill_pass_is_held_while_the_storage_floor_is_not_free_or_unmeasured(self) -> None:
        """No launcher runs in a pass whose du fails, outlasts its limit (1 s here instead of 300 s)
        or leaves less than the floor free under the quota: the pass logs one line saying so and
        the next pass measures again; the first pass that finds the floor free runs the launcher."""
        script = Path(self.temp.name) / "remote.sh"
        text = zgca.REMOTE_SCRIPT.read_text(encoding="utf-8")
        self.assertEqual(text.count("limit=300"), 1)
        script.write_text(text.replace("limit=300", "limit=1"), encoding="utf-8")
        self.addCleanup(setattr, zgca, "REMOTE_SCRIPT", zgca.REMOTE_SCRIPT)
        zgca.REMOTE_SCRIPT = script
        storage = Path(self.temp.name)
        below = zgca.STORAGE_QUOTA - zgca.STORAGE_FLOOR + 2**30  # a GiB short of the floor
        lines, slept = self._fill([f"0|{self.FILL_DONE}"], du=["fail", "hang", str(below), str(zgca.STORAGE_QUOTA - zgca.STORAGE_FLOOR)])
        self.assertEqual(len(lines), 5, lines)
        self.assertRegex(lines[0], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d fill held, no experiment is created this pass: ")
        self.assertTrue(lines[0].endswith(f"storage: unknown, du of {storage} failed (exit 1): "
                                          f"du: cannot read directory '{storage}/x': Permission denied"), lines[0])
        self.assertTrue(lines[1].endswith(f"fill held, no experiment is created this pass: storage: unknown, du of {storage} "
                                          "did not finish within 1 s"), lines[1])
        self.assertTrue(lines[2].endswith(f"fill held, no experiment is created this pass: storage: 61.0 of 100 GiB used, 39.0 free "
                                          f"(du of {storage}), below the 40 GiB floor"), lines[2])
        self.assertEqual(lines[3:], [self.FILL_DONE, "round-exit 0"])
        self.assertEqual(slept, ["600", "600", "600"])

    def test_the_storage_step_measures_the_quota_by_du_and_status_shows_it(self) -> None:
        """The storage line against the quota, exit 0 only with the floor free; a failed du is no
        measure and never free. `status` shows the same line whatever it says, before anything else."""
        temp = Path(self.temp.name)
        root, stubs = temp / "instance", temp / "stubs"
        stubs.mkdir()
        (stubs / "du").write_text('#!/bin/sh\n[ -n "$USED" ] || { echo "du: cannot access \'$2\': Stale file handle" >&2; exit 1; }\n'
                                  'printf "%s\\t%s\\n" "$USED" "$2"\n', encoding="utf-8")
        (stubs / "docker").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        for stub in stubs.iterdir():
            stub.chmod(0o700)
        cases = (
            (str(int(56.2 * 2**30)), 0, f"storage: 56.2 of 100 GiB used, 43.8 free (du of {temp})"),
            (str(zgca.STORAGE_QUOTA - zgca.STORAGE_FLOOR), 0, f"storage: 60.0 of 100 GiB used, 40.0 free (du of {temp})"),
            (str(zgca.STORAGE_QUOTA - zgca.STORAGE_FLOOR + 1), 1,
             f"storage: 60.0 of 100 GiB used, 40.0 free (du of {temp}), below the 40 GiB floor"),
            ("", 1, f"storage: unknown, du of {temp} failed (exit 1): du: cannot access '{temp}': Stale file handle"),
        )
        for used, code, line in cases:
            with self.subTest(used=used or "du fails"):
                env = {**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "USED": used}
                measured = self.remote_step(root, "storage", env=env)
                self.assertEqual((measured.returncode, measured.stdout), (code, line + "\n"), measured.stderr)
                shown = self.remote_step(root, "status", env=env)
                self.assertEqual(shown.returncode, 0, shown.stderr)
                self.assertEqual(shown.stdout.splitlines()[:2], [line, f"repository: not deployed at {root}"])

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
        for name in ("instance", "dev"):
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
        # The layout comes from zgca.py on every call; a step without it is refused, and so is a quota that is no byte count.
        bare = subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), "status"], capture_output=True, text=True, check=False)
        self.assertEqual(bare.returncode, 1)
        self.assertIn("usage: remote.sh <root> <venv> <local dir> <storage> <quota bytes> <floor bytes> <step>", bare.stderr)
        garbled = subprocess.run(["bash", str(zgca.REMOTE_SCRIPT), str(root), str(root / ".venv"), str(root.parent / "ebs"),
                                  str(root.parent), "100GiB", str(zgca.STORAGE_FLOOR), "storage"], capture_output=True, text=True, check=False)
        self.assertEqual(garbled.returncode, 1)
        self.assertIn("the storage quota and floor are byte counts", garbled.stderr)
        # The build verifies and serves the shipped release files, so prepare takes their digests.
        prepare = self.remote_step(root, "prepare", "0" * 64)
        self.assertEqual(prepare.returncode, 1)
        self.assertIn("pinned DuckDB CLI and CUDA compatibility package digests", prepare.stderr)
        self.assertEqual(subprocess.run(["bash", "-n", str(zgca.REMOTE_SCRIPT)], check=False).returncode, 0)


if __name__ == "__main__":
    unittest.main()
