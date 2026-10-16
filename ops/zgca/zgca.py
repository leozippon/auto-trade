#!/usr/bin/env python3
"""The ZGCA A100 development instance, end to end: up, deploy, run, status, sync, down.

Every step goes through the platform layer in ~/zgca-compute (`instance` to
show, start and stop the one instance as idempotent intents, `dev` for SSH and
rsync to it, both over `zenith`) and verifies its own result: up and down are
`instance start` and `instance stop`; a deploy carries the tracked files, the
local .env, the named PIT view seed and the research release that seed was
built from, prunes code the tree no longer tracks, then prepares the instance
(venv, ripgrep, tmux, sandbox image) through ops/zgca/remote.sh; a run starts
the console and a round launcher in tmux with AUTOTRADE_GPU_NAME set; status
and sync read and copy back. Nothing here schedules, caches an endpoint, holds
a secret or changes how the executor isolates a strategy container. See
ops/zgca/README.md.

usage: zgca.py up | deploy <seed dir> | run <round file> [launcher args...] |
       status | sync <arm id> | down
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# auto-trade-dev-a100: one Tesla-A100-80G in the Volcano zone; its id is the
# platform ledger id the API addresses it by, not a secret.
INSTANCE_ID = "ozaeun8qxljr1niyoz9flg6uc"
# The instance's persistent personal storage: the only place code, the .env,
# seeds and releases go (~/zgca-compute/README.md, 配额与存储).
REMOTE_ROOT = "/personal_fast_storage/auto-trade"
RELEASES = "data/research_releases"
UNFINISHED_SEED_MARKER = "prebuild_unfinished.json"
# The DuckDB CLI release the sandbox image installs (ops/docker/sandbox.Dockerfile,
# build arg DUCKDB_CLI_URL). GitHub is unreachable from the instance and its
# release proxies answer or refuse by the day, so the deploy fetches the
# release where GitHub is reachable, verifies it against this digest (two
# independent routes agreed on it on 2026-10-15), ships it, and the build reads
# it from the instance host through that build arg.
DUCKDB_CLI_URL = "https://github.com/duckdb/duckdb/releases/download/v1.1.3/duckdb_cli-linux-amd64.zip"
DUCKDB_CLI_NAME = "duckdb_cli-linux-amd64.zip"
DUCKDB_CLI_SHA256 = "efd0fccdb1a28d9ec7a6ebfcde59900068b8ba43a846c9b553c0fd2bbe4acf43"
TRACKED_LIST_NAME = "tracked.txt"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REMOTE_SCRIPT = Path(__file__).with_name("remote.sh")
REPO_ROOT = Path(__file__).resolve().parents[2]


class WorkflowError(Exception):
    """A step that did not reach its verified result; the message says why."""


def say(message: str) -> None:
    """One timestamped progress line, flushed so it lands before the output of the step it announces."""

    print(f"{datetime.now(tz=UTC):%H:%M:%S}Z {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Workflow:
    repo_root: Path
    tools: Path
    instance: str = INSTANCE_ID

    def __post_init__(self) -> None:
        for name in ("zenith", "dev", "instance"):
            if not os.access(self.tools / name, os.X_OK):
                raise WorkflowError(
                    f"{self.tools / name} is not executable: ZGCA_COMPUTE must name the "
                    "zgca-compute checkout (default ~/zgca-compute)"
                )

    @property
    def state_dir(self) -> Path:
        return self.repo_root / ".runtime" / "zgca"

    # ----- platform entries -----

    def call(self, tool: str, *args: str, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
        """One run of a ~/zgca-compute entry (`instance`, `dev`), pointed at the pinned instance."""
        env = {**os.environ, "ZGCA_INSTANCE": self.instance}
        return subprocess.run(
            [str(self.tools / tool), *args],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE if capture else None, env=env, check=False,
        )

    def lifecycle(self, command: str, *, capture: bool = False) -> str:
        """`instance <command> <id>`, which verifies its own result and says why on stderr when it cannot."""
        completed = self.call("instance", command, self.instance, capture=capture)
        if completed.returncode != 0:
            raise WorkflowError(f"instance {command} {self.instance} failed (exit {completed.returncode})")
        return (completed.stdout or b"").decode("utf-8", errors="replace").strip()

    def require_running(self) -> None:
        state = self.lifecycle("state", capture=True)
        if state != "RUNNING":
            raise WorkflowError(f"instance {self.instance} is {state}, not RUNNING: run `up` first")

    # ----- instance shell and transfers -----

    def remote(self, step: str, *args: str) -> None:
        """Run one step of remote.sh on the instance, in a login shell for the platform's environment."""
        script = REMOTE_SCRIPT.read_text(encoding="utf-8")
        completed = self.call("dev", "bash", "-lc", script, "remote.sh", step, *args)
        if completed.returncode != 0:
            raise WorkflowError(f"remote step {step} failed (exit {completed.returncode})")

    def rsync(self, *args: str) -> None:
        completed = self.call("dev", "rsync", *args)
        if completed.returncode != 0:
            raise WorkflowError(f"rsync failed (exit {completed.returncode}): {' '.join(args[-2:])}")
        verify = self.call("dev", "rsync", "--dry-run", "--itemize-changes", *args, capture=True)
        if verify.returncode != 0:
            raise WorkflowError(f"rsync verification failed (exit {verify.returncode}): {' '.join(args[-2:])}")
        if verify.stdout.strip():
            raise WorkflowError(
                f"the copy differs from its source after the transfer ({args[-1]}):\n"
                + verify.stdout.decode("utf-8", errors="replace")
            )

    def duckdb_cli(self) -> Path:
        """The verified DuckDB CLI release the deploy ships, fetched once into the state directory."""

        path = self.state_dir / DUCKDB_CLI_NAME
        if not path.is_file():
            self.state_dir.mkdir(parents=True, exist_ok=True)
            say(f"deploy: fetching {DUCKDB_CLI_URL}")
            fetched = subprocess.run(
                ["curl", "-fsSL", "--retry", "5", "--retry-all-errors", "--retry-delay", "3",
                 "--max-time", "900", "-C", "-", "-o", str(path), DUCKDB_CLI_URL],
                stdin=subprocess.DEVNULL, check=False,
            )
            if fetched.returncode != 0 or not path.is_file():
                raise WorkflowError(f"could not fetch {DUCKDB_CLI_URL} (curl exit {fetched.returncode})")
        digest = _sha256(path)
        if digest != DUCKDB_CLI_SHA256:
            raise WorkflowError(
                f"{path} has digest {digest}, not the pinned {DUCKDB_CLI_SHA256}: remove it to fetch "
                "the release again, or re-pin the digest after verifying the new release"
            )
        return path

    # ----- commands -----

    def up(self) -> None:
        self.lifecycle("start")

    def deploy(self, seed: str) -> None:
        seed_path = Path(seed)
        seed_dir = (self.repo_root / seed_path).resolve()
        if seed_path.is_absolute() or seed_path.parts[:1] != ("data",) or not seed_dir.is_dir():
            raise WorkflowError(f"the seed must be an existing directory below data/, named relative to the repository: {seed}")
        if (seed_dir / UNFINISHED_SEED_MARKER).exists():
            raise WorkflowError(f"{seed} has an unfinished prebuild ({UNFINISHED_SEED_MARKER} is present)")
        try:
            generation = str(json.loads((seed_dir / "provider.json").read_text(encoding="utf-8")).get("generation_id") or "")
        except (OSError, ValueError) as exc:
            raise WorkflowError(f"{seed}/provider.json is unreadable: {exc}") from exc
        if not SAFE_ID.fullmatch(generation):
            raise WorkflowError(f"{seed}/provider.json names no research release generation")
        release_dir = self.repo_root / RELEASES / generation
        if not (release_dir / "manifest.json").is_file():
            raise WorkflowError(f"{seed} was built from research release {generation}, which is not published at {release_dir}")
        env_file = self.repo_root / ".env"
        if not env_file.is_file():
            raise WorkflowError(f"{env_file} is missing: the instance's sessions read the same credentials")
        listed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=self.repo_root, stdin=subprocess.DEVNULL, capture_output=True, check=False,
        )
        if listed.returncode != 0 or not listed.stdout:
            raise WorkflowError("git ls-files listed no tracked files in " + str(self.repo_root))
        duckdb_cli = self.duckdb_cli()
        # The list rsync copies from is also what the instance keeps: shipped
        # beside the DuckDB CLI, it lets the `tracked` step remove what is no
        # longer tracked (a retired round file would otherwise stay and be read).
        manifest = self.state_dir / TRACKED_LIST_NAME
        manifest.write_bytes(listed.stdout)
        self.require_running()
        self.remote("dirs")
        preserve = ("-a", "--no-owner", "--no-group")
        say("deploy: tracked files")
        self.rsync(*preserve, f"--files-from={manifest}", "--from0", f"{self.repo_root}/", f"dev:{REMOTE_ROOT}/")
        say("deploy: .env")
        self.rsync(*preserve, str(env_file), f"dev:{REMOTE_ROOT}/.env")
        say(f"deploy: {DUCKDB_CLI_NAME} and {TRACKED_LIST_NAME}")
        self.rsync(*preserve, str(duckdb_cli), str(manifest), f"dev:{REMOTE_ROOT}/.runtime/zgca/")
        self.remote("tracked")
        say(f"deploy: {seed}")
        self.rsync(*preserve, "-H", "--delete", f"{seed_dir}/", f"dev:{REMOTE_ROOT}/{seed_path.as_posix()}/")
        say(f"deploy: {RELEASES}/{generation}")
        self.rsync(*preserve, "-H", "--delete", f"{release_dir}/", f"dev:{REMOTE_ROOT}/{RELEASES}/{generation}/")
        say("deploy: preparing the instance")
        self.remote("prepare", DUCKDB_CLI_SHA256)
        say("deploy: done")

    def run(self, round_file: str, args: Sequence[str]) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", round_file],
            cwd=self.repo_root, stdin=subprocess.DEVNULL, capture_output=True, check=False,
        )
        if tracked.returncode != 0 or not (self.repo_root / round_file).is_file():
            raise WorkflowError(f"{round_file} is not a tracked file of the repository, so deploy would not have carried it")
        self.require_running()
        self.remote("console-up")
        self.remote("round", round_file, *args)

    def status(self) -> None:
        self.lifecycle("show")
        if self.lifecycle("state", capture=True) == "RUNNING":
            self.remote("status")

    def sync(self, arm: str) -> None:
        if not SAFE_ID.fullmatch(arm):
            raise WorkflowError(f"not an experiment id: {arm!r}")
        local = self.repo_root / "experiments" / arm
        if local.exists():
            try:
                created_under = json.loads((local / "hitl" / "params.json").read_text(encoding="utf-8")).get("experiments_root")
            except (OSError, ValueError) as exc:
                raise WorkflowError(f"experiments/{arm} exists here and its hitl/params.json cannot say where it was created ({exc}); refusing to overwrite it") from exc
            if created_under != f"{REMOTE_ROOT}/experiments":
                raise WorkflowError(f"experiments/{arm} is an arm of this host (created under {created_under}), not a copy from the instance; refusing to overwrite it")
        self.require_running()
        self.remote("arm-state", arm)
        preserve = ("-a", "--no-owner", "--no-group")
        local.parent.mkdir(parents=True, exist_ok=True)
        self.rsync(*preserve, "--exclude", "/pit_views/", f"dev:{REMOTE_ROOT}/experiments/{arm}/", f"{local}/")
        logs = self.repo_root / "logs" / "workers"
        logs.mkdir(parents=True, exist_ok=True)
        self.rsync(*preserve, f"dev:{REMOTE_ROOT}/logs/workers/{arm}.log", f"{logs / arm}.log")
        say(f"synced experiments/{arm} (without pit_views) and logs/workers/{arm}.log")

    def down(self) -> None:
        self.lifecycle("stop")


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    tools: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="zgca.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("up", help="instance start: wait for RUNNING and SSH")
    deploy = commands.add_parser("deploy", help="copy code, .env, a seed and its release; prepare venv, ripgrep, tmux, image")
    deploy.add_argument("seed", help="PIT view seed directory, relative to the repository (data/pit_views_seed_...)")
    run = commands.add_parser("run", help="start the console if needed and a round launcher in tmux")
    run.add_argument("round_file", help="scripts/experiments/create_round_*.py")
    run.add_argument("args", nargs=argparse.REMAINDER, help="launcher arguments after the port (--dry-run, --fill, arm ids)")
    commands.add_parser("status", help="instance, card-hours, console, arms, GPU, disk, image, round log")
    sync = commands.add_parser("sync", help="copy an arm's records and worker log back here")
    sync.add_argument("arm", help="experiment id")
    commands.add_parser("down", help="instance stop: wait for STOPPED")
    options = parser.parse_args(argv)
    try:
        workflow = Workflow(
            repo_root=repo_root,
            tools=tools or Path(os.environ.get("ZGCA_COMPUTE") or "~/zgca-compute").expanduser(),
        )
        if options.command == "up":
            workflow.up()
        elif options.command == "deploy":
            workflow.deploy(options.seed)
        elif options.command == "run":
            workflow.run(options.round_file, options.args)
        elif options.command == "status":
            workflow.status()
        elif options.command == "sync":
            workflow.sync(options.arm)
        else:
            workflow.down()
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
