#!/usr/bin/env python3
"""The ZGCA A100 development instance, end to end: up, deploy, run, status, sync, down.

Every step goes through the platform layer in ~/zgca-compute (`zenith` for the
Zenith API behind the jump host, `dev` for SSH and rsync to the one instance)
and verifies its own result: an instance start or stop is one idempotent
intent whose key lives in .runtime/zgca/ until the operation reaches a terminal
state; a deploy carries the tracked files, the local .env, the named PIT view
seed and the research release that seed was built from, then prepares the
instance (venv, ripgrep, tmux, sandbox image) through ops/zgca/remote.sh; a
run starts the console and a round launcher in tmux with AUTOTRADE_GPU_NAME
set; status, sync and down read, copy back and stop. Nothing here schedules,
caches an endpoint, holds a secret or changes how the executor isolates a
strategy container. See ops/zgca/README.md.

usage: zgca.py up | deploy <seed dir> | run <round file> [launcher args...] |
       status | sync <arm id> | down
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
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
POLL_SECONDS = 5
OPERATION_TIMEOUT_SECONDS = 15 * 60
SSH_TIMEOUT_SECONDS = 3 * 60
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REMOTE_SCRIPT = Path(__file__).with_name("remote.sh")
REPO_ROOT = Path(__file__).resolve().parents[2]


class WorkflowError(Exception):
    """A step that did not reach its verified result; the message says why."""


def _text(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict | list) else str(payload)


def say(message: str) -> None:
    """One timestamped progress line, flushed so it lands before the output of the step it announces."""

    print(f"{datetime.now(tz=UTC):%H:%M:%S}Z {message}", flush=True)


@dataclass
class Workflow:
    repo_root: Path
    tools: Path
    instance: str = INSTANCE_ID
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        for name in ("zenith", "dev"):
            if not os.access(self.tools / name, os.X_OK):
                raise WorkflowError(
                    f"{self.tools / name} is not executable: ZGCA_COMPUTE must name the "
                    "zgca-compute checkout (default ~/zgca-compute)"
                )

    @property
    def state_dir(self) -> Path:
        return self.repo_root / ".runtime" / "zgca"

    # ----- platform API -----

    def api(self, method: str, path: str, *, key: str | None = None) -> tuple[bool, object]:
        """One `zenith` call: (True, body) on 2xx, else (False, error JSON or transport text)."""
        env = {name: value for name, value in os.environ.items() if name != "ZENITH_IDEMPOTENCY_KEY"}
        if key:
            env["ZENITH_IDEMPOTENCY_KEY"] = key
        completed = subprocess.run(
            [str(self.tools / "zenith"), method, path],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, check=False,
        )
        payload: object = None
        if completed.stdout.strip():
            try:
                payload = json.loads(completed.stdout)
            except ValueError:
                payload = None
        if completed.returncode == 0:
            if not isinstance(payload, dict):
                raise WorkflowError(f"{method} {path}: the response is not a JSON object")
            return True, payload
        if payload is None:
            payload = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
        return False, payload

    def get(self, path: str) -> dict[str, object]:
        ok, payload = self.api("GET", path)
        if not ok:
            raise WorkflowError(f"GET {path} failed: {_text(payload)}")
        assert isinstance(payload, dict)
        return payload

    def instance_row(self) -> dict[str, object]:
        row = self.get(f"/pt/api/ai/instances/{self.instance}").get("instance")
        if not isinstance(row, dict):
            raise WorkflowError(f"instance {self.instance}: the detail response carries no instance")
        return row

    def instance_state(self) -> str:
        return str(self.instance_row().get("unified_status") or "")

    def require_running(self) -> None:
        state = self.instance_state()
        if state != "RUNNING":
            raise WorkflowError(f"instance {self.instance} is {state or 'in no known state'}, not RUNNING: run `up` first")

    def card_hours(self) -> str:
        personal = self.get("/pt/api/me/card-hours").get("personal")
        if not isinstance(personal, dict):
            raise WorkflowError("card-hours: the response carries no personal allowance")
        try:
            end = datetime.fromtimestamp(float(personal["current_period_end_at"]) / 1000, tz=UTC)
            return (
                f"card-hours this week: {float(personal['used_card_hours']):.2f} used, "
                f"{float(personal['remaining_card_hours']):.2f} left of "
                f"{float(personal['effective_allowance_card_hours']):.0f}, resets {end:%Y-%m-%d %H:%M} UTC"
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"card-hours: the personal allowance lacks a readable field ({exc})") from exc

    # ----- instance lifecycle -----

    def transition(self, action: str, target: str) -> None:
        """Start or stop the instance as one idempotent intent and wait for `target`.

        The key is written before the request and removed only when the
        operation it names succeeded and the instance shows the target state,
        or when the platform reports the operation FAILED (a new attempt is a
        new intent). A refused or lost request, or an UNKNOWN operation, keeps
        it, so a re-run carries the same key, as the platform requires.
        """

        key_file = self.state_dir / f"{self.instance}.{action}.key"
        if self.instance_state() == target:
            key_file.unlink(missing_ok=True)
            say(f"instance {self.instance} is {target}")
            return
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
            say(f"{action}: re-sending the pending intent {key}")
        else:
            key = f"autotrade-{action}-{uuid.uuid4().hex}"
            self.state_dir.mkdir(parents=True, exist_ok=True)
            key_file.write_text(key + "\n", encoding="utf-8")
        ok, receipt = self.api("POST", f"/pt/api/ai/instances/{self.instance}/{action}", key=key)
        if not ok:
            if self.instance_state() == target:
                key_file.unlink(missing_ok=True)
                say(f"instance {self.instance} is {target}")
                return
            raise WorkflowError(
                f"{action} was not accepted: {_text(receipt)}. The intent key stays in {key_file}: "
                "re-run to send it again (after a timeout, a 503 or a 429); remove the file only "
                "for a new intent"
            )
        assert isinstance(receipt, dict)
        operation_id = str(receipt.get("operation_id") or "")
        if not SAFE_ID.fullmatch(operation_id):
            raise WorkflowError(f"{action} receipt names no operation: {_text(receipt)}")
        say(f"{action} accepted: operation {operation_id} {receipt.get('state') or ''}".rstrip())
        record = self.wait_operation(operation_id)
        state = str(record.get("state") or "")
        if state == "FAILED":
            key_file.unlink(missing_ok=True)
            raise WorkflowError(f"operation {operation_id} failed: {_text(record)}")
        if state != "SUCCEEDED":
            raise WorkflowError(
                f"operation {operation_id} is {state}: re-run to reconcile it under the same key"
            )
        self.wait_state(target)
        key_file.unlink(missing_ok=True)
        say(f"instance {self.instance} is {target}")

    def wait_operation(self, operation_id: str) -> dict[str, object]:
        deadline = self.clock() + OPERATION_TIMEOUT_SECONDS
        while True:
            record = self.get(f"/pt/api/operations/{operation_id}").get("operation")
            if not isinstance(record, dict):
                raise WorkflowError(f"operation {operation_id}: the response carries no operation")
            if record.get("state") in ("SUCCEEDED", "FAILED", "UNKNOWN"):
                return record
            if self.clock() >= deadline:
                raise WorkflowError(
                    f"operation {operation_id} is still {record.get('state')} after "
                    f"{OPERATION_TIMEOUT_SECONDS} s: re-run to keep waiting under the same key"
                )
            self.sleep(POLL_SECONDS)

    def wait_state(self, target: str) -> None:
        deadline = self.clock() + OPERATION_TIMEOUT_SECONDS
        while True:
            state = self.instance_state()
            if state == target:
                return
            if self.clock() >= deadline:
                raise WorkflowError(f"instance {self.instance} is {state}, not {target}, after {OPERATION_TIMEOUT_SECONDS} s")
            self.sleep(POLL_SECONDS)

    # ----- instance shell and transfers -----

    def dev(self, *args: str, stdin: bytes | None = None, capture: bool = False) -> subprocess.CompletedProcess[bytes]:
        env = {**os.environ, "ZGCA_INSTANCE": self.instance}
        return subprocess.run(
            [str(self.tools / "dev"), *args],
            input=stdin, stdin=None if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else None, env=env, check=False,
        )

    def remote(self, step: str, *args: str) -> None:
        """Run one step of remote.sh on the instance, in a login shell for the platform's environment."""
        script = REMOTE_SCRIPT.read_text(encoding="utf-8")
        completed = self.dev("bash", "-lc", script, "remote.sh", step, *args)
        if completed.returncode != 0:
            raise WorkflowError(f"remote step {step} failed (exit {completed.returncode})")

    def rsync(self, *args: str, stdin: bytes | None = None) -> None:
        completed = self.dev("rsync", *args, stdin=stdin)
        if completed.returncode != 0:
            raise WorkflowError(f"rsync failed (exit {completed.returncode}): {' '.join(args[-2:])}")
        verify = self.dev("rsync", "--dry-run", "--itemize-changes", *args, stdin=stdin, capture=True)
        if verify.returncode != 0:
            raise WorkflowError(f"rsync verification failed (exit {verify.returncode}): {' '.join(args[-2:])}")
        if verify.stdout.strip():
            raise WorkflowError(
                f"the copy differs from its source after the transfer ({args[-1]}):\n"
                + verify.stdout.decode("utf-8", errors="replace")
            )

    def wait_ssh(self) -> None:
        deadline = self.clock() + SSH_TIMEOUT_SECONDS
        while True:
            if self.dev("true").returncode == 0:
                return
            if self.clock() >= deadline:
                raise WorkflowError(f"instance {self.instance} is RUNNING but SSH did not answer within {SSH_TIMEOUT_SECONDS} s")
            self.sleep(10)

    # ----- commands -----

    def up(self) -> None:
        self.transition("start", "RUNNING")
        self.wait_ssh()
        say(f"ssh reachable; {self.card_hours()}")

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
        self.require_running()
        self.remote("dirs")
        preserve = ("-a", "--no-owner", "--no-group")
        say("deploy: tracked files")
        self.rsync(*preserve, "--files-from=-", "--from0", f"{self.repo_root}/", f"dev:{REMOTE_ROOT}/", stdin=listed.stdout)
        say("deploy: .env")
        self.rsync(*preserve, str(env_file), f"dev:{REMOTE_ROOT}/.env")
        say(f"deploy: {seed}")
        self.rsync(*preserve, "-H", "--delete", f"{seed_dir}/", f"dev:{REMOTE_ROOT}/{seed_path.as_posix()}/")
        say(f"deploy: {RELEASES}/{generation}")
        self.rsync(*preserve, "-H", "--delete", f"{release_dir}/", f"dev:{REMOTE_ROOT}/{RELEASES}/{generation}/")
        say("deploy: preparing the instance")
        self.remote("prepare")
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
        row = self.instance_row()
        state = str(row.get("unified_status") or "")
        print(f"instance {row.get('name')} ({self.instance}): {state}, {row.get('gpu_type')} x{row.get('gpu_count')}")
        retention = row.get("retention")
        stopped_at = retention.get("stopped_at") if isinstance(retention, dict) else None
        if stopped_at:
            print(f"  stopped since {datetime.fromtimestamp(float(stopped_at) / 1000, tz=UTC):%Y-%m-%d %H:%M} UTC")
        print(self.card_hours())
        if state == "RUNNING":
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
        self.transition("stop", "STOPPED")
        print(self.card_hours())


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    tools: Path | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    parser = argparse.ArgumentParser(prog="zgca.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("up", help="start the instance, wait for RUNNING and SSH")
    deploy = commands.add_parser("deploy", help="copy code, .env, a seed and its release; prepare venv, ripgrep, tmux, image")
    deploy.add_argument("seed", help="PIT view seed directory, relative to the repository (data/pit_views_seed_...)")
    run = commands.add_parser("run", help="start the console if needed and a round launcher in tmux")
    run.add_argument("round_file", help="scripts/experiments/create_round_*.py")
    run.add_argument("args", nargs=argparse.REMAINDER, help="launcher arguments after the port (--dry-run, --fill, arm ids)")
    commands.add_parser("status", help="instance, card-hours, console, arms, GPU, disk, image, round log")
    sync = commands.add_parser("sync", help="copy an arm's records and worker log back here")
    sync.add_argument("arm", help="experiment id")
    commands.add_parser("down", help="stop the instance and wait for STOPPED")
    options = parser.parse_args(argv)
    try:
        workflow = Workflow(
            repo_root=repo_root,
            tools=tools or Path(os.environ.get("ZGCA_COMPUTE") or "~/zgca-compute").expanduser(),
            sleep=sleep,
            clock=clock,
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
