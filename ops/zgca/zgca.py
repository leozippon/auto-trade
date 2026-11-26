#!/usr/bin/env python3
"""The ZGCA A100 development instance, end to end: up, deploy, run, resume, status, sync, sync-rounds, down.

Every command reaches the platform through ~/zgca-compute's `instance` and
`dev` alone, runs its instance-side steps from ops/zgca/remote.sh and verifies
its own result; the command list below says what each does. The rules behind
them (deploy's checks and transfers, the fill loop and its storage floor,
sync's markers, tag and release, the sync cron) are in ops/zgca/README.md.

usage: zgca.py [--instance <id>] up | deploy <seed dir> |
       run <round file> [launcher args...] | resume <experiment id> | status |
       sync <experiment id> | sync-rounds [round file...] | down
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# auto-trade-dev-a100x2, the default of --instance, in the Volcano zone: one
# Tesla-A100-80G since 2026-11-02 (the name keeps the two cards it was created
# with); its id is the platform ledger id the API addresses it by, not a secret.
INSTANCE_ID = "mpo06t71crmmtdsrb98ckwk29"
# The platform's own shape for an instance id (~/zgca-compute/dev).
INSTANCE_ID_SHAPE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
GIB = 2**30
# The instance's layout, stated here alone: remote.sh receives it on every
# call. The repository root is on the instance's persistent personal storage,
# the only place code, the .env, seeds and releases go (~/zgca-compute/README.md,
# 配额、规格与存储), and the venv the console and its workers run on sits
# beside it. The pretrained store lives in <LOCAL_ROOT>/pretrained on the
# instance's own disk, which also holds its root and Docker's images
# (~/zgca-compute/README.md, 火山平台 Docker); the root's data/pretrained links to it.
STORAGE = "/personal_fast_storage"
REMOTE_ROOT = f"{STORAGE}/auto-trade"
REMOTE_VENV = f"{STORAGE}/envs/quant"
# That storage is a GPFS fileset under a hard byte quota. df and statvfs there
# show the whole shared file system, so neither they nor anything reading them
# (the console's create check, the agent tools' floor) sees the quota, and the
# container has no quota tool: remote.sh measures it as du of the whole tree,
# which counts a hard link once, as the quota does (about a second on 2026-11-20).
# Filling it on 2026-11-20 killed a session 18 hours in with EDQUOT.
STORAGE_QUOTA = 100 * GIB
# Free room under the quota below which the fill loop creates no experiment and
# `resume` starts none: a session's workspace reached 20 GiB in its sandbox, and
# the end of each attempt archives that workspace, less its files over 64 MiB,
# into the experiment's artifacts.
STORAGE_FLOOR = 40 * GIB
LOCAL_DISK = "/ebs"
LOCAL_ROOT = f"{LOCAL_DISK}/auto-trade"
# Left free on that disk beside the store: the new layers of an image rebuild
# and the containers' scratch. Filling it would stop Docker.
LOCAL_DISK_RESERVE = 20 * 2**30
RELEASES = "data/research_releases"
UNFINISHED_SEED_MARKER = "prebuild_unfinished.json"
# The DuckDB CLI release the sandbox image installs, pinned by version and
# SHA-256 in the image's Dockerfile alone (build args DUCKDB_CLI_URL and
# DUCKDB_CLI_SHA256) and read from there: the release's GitHub URL out of the
# default, which reaches it through a release proxy, and the digest. GitHub is
# unreachable from the instance and its release proxies answer or refuse by
# the day, so the deploy fetches the release where GitHub is reachable
# (through the proxy RELEASE_PROXY_ENV names, then directly), verifies it
# against the digest, ships it, and the build reads it from the instance host
# through that build arg and verifies it again.
SANDBOX_DOCKERFILE = Path(__file__).resolve().parents[1] / "docker" / "sandbox.Dockerfile"


def dockerfile_pin(name: str, value: str) -> str:
    """What the group in the pattern ``value`` matches of the one default SANDBOX_DOCKERFILE gives build arg ``name``."""

    found = re.findall(rf"^ARG {name}={value}$", SANDBOX_DOCKERFILE.read_text(encoding="utf-8"), re.MULTILINE)
    if len(found) != 1:
        raise RuntimeError(f"{SANDBOX_DOCKERFILE} does not give build arg {name} one default matching {value}")
    return found[0]


DUCKDB_CLI_URL = dockerfile_pin("DUCKDB_CLI_URL", r"\S*?(https://github\.com/\S+)")
DUCKDB_CLI_NAME = DUCKDB_CLI_URL.rsplit("/", 1)[1]
DUCKDB_CLI_SHA256 = dockerfile_pin("DUCKDB_CLI_SHA256", "([0-9a-f]{64})")
# A URL prefix tried before a GitHub release's own URL, e.g. https://ghfast.top/.
RELEASE_PROXY_ENV = "ZGCA_GITHUB_PROXY"
# NVIDIA's CUDA forward-compatibility package for CUDA 13.0 (the user-mode
# driver of the 580 series). The image's torch needs CUDA 13, the instance's
# kernel driver is 535 (CUDA 12.2), and an A100 may run a newer user-mode
# driver on an older kernel driver; the instance's image build installs it
# (build arg CUDA_COMPAT_URL), other builds do not. Pinned to the digest of
# NVIDIA's Debian 12 package index (rechecked 2026-11-21, with the index held
# to the repository's Release file), after a probe on the instance
# (~/zgca-compute/PLATFORM_RECORDS.md, GPU 驱动); the build checks its
# download against the digest too (build arg CUDA_COMPAT_SHA256).
CUDA_COMPAT_URL = "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-0_580.196.07-1_amd64.deb"
CUDA_COMPAT_NAME = "cuda-compat-13-0_580.196.07-1_amd64.deb"
CUDA_COMPAT_SHA256 = "d2d3262512046b23215ff197fe79a6ca0d44299ed0e665275ac05bb28b79e4e3"
# The pretrained-model store (environment.pretrained): the console stamps its
# manifest digest into every experiment it creates and every container compares it.
PRETRAINED = "data/pretrained"
PRETRAINED_MANIFEST = "MANIFEST.sha256"
TRACKED_LIST_NAME = "tracked.txt"
# The commit whose tracked files the last deploy shipped; remote.sh's status shows it.
DEPLOYED_COMMIT_NAME = "deployed_commit"
# One marker per experiment `sync` brought back, in the state directory, saying
# how far it got: `copied`, `synced` or `released` (ops/zgca/README.md, sync).
SYNCED_DIR_NAME = "synced"
# Where the first copy of an experiment is made, in the state directory, under
# its id: it enters experiments/ by one rename once verified, so a copy cut
# short never leaves there a directory without hitl/params.json, which
# sync_target would refuse (rsync copies artifacts/ before hitl/). A copy over
# an earlier one is made in place, where that copy's params.json stays.
INCOMING_DIR_NAME = "incoming"
# The experiment states (autotrade.pipelines.hitl_state.ExperimentState) sync
# and resume decide by, named here because importing that module needs numpy,
# which neither otherwise does: only a copy of a completed experiment is
# released on the instance, and an experiment that ended any other way than
# failed is not resumed.
COMPLETED = "completed"
NOT_RESUMED = ("completed", "stopped")
# What a sync-rounds pass counts, in its summary line's order.
SYNC_COUNTS = ("synced", "untagged", "running", "interrupted", "pending", "failed")
# remote.sh's exit statuses (its ABSENT and RUNNING) for a step whose subject
# is not on the instance, or is still running there.
REMOTE_ABSENT = 3
REMOTE_RUNNING = 4
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Every transfer keeps contents, times and modes; the instance's root owns what lands there.
PRESERVE = ("-a", "--no-owner", "--no-group")
REMOTE_SCRIPT = Path(__file__).with_name("remote.sh")
REPO_ROOT = Path(__file__).resolve().parents[2]


class WorkflowError(Exception):
    """A step that did not reach its verified result; the message says why."""


class PinnedImageAbsent(WorkflowError):
    """The image a synced experiment pins is gone from the instance; retrying its tag cannot bring it back."""


def say(message: str) -> None:
    """One timestamped progress line, flushed so it lands before the output of the step it announces."""

    print(f"{datetime.now(tz=UTC):%H:%M:%S}Z {message}", flush=True)


def instance_id(value: str) -> str:
    """An --instance value; an empty one would let `dev` pick any RUNNING machine."""

    if not INSTANCE_ID_SHAPE.fullmatch(value):
        raise argparse.ArgumentTypeError(f"not a platform instance id: {value!r}")
    return value


def _with_project_src() -> None:
    """Make this checkout's `autotrade` importable; the script also runs outside the project's environment."""
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def gateway_env(env_file: Path) -> bytes:
    """The lines of ``env_file`` that set a key some model gateway reads.

    The model profiles name those keys (``gateway_env_names``), parsed as the
    gateways parse the file; everything else in it (the TuShare tokens,
    HF_TOKEN) stays on this host.
    """

    _with_project_src()
    from autotrade.environment.llm.model_profiles import gateway_env_names

    names = set(gateway_env_names())
    kept = []
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if key in names:
            kept.append(stripped + "\n")
    return "".join(kept).encode("utf-8")


def release_exclusions(provider: dict[str, object]) -> list[str]:
    """The release's ``raw`` datasets a deploy of this seed leaves behind.

    Only the text datasets and the minute bars, the bulk of a release, which
    nothing reads unless the snapshot configuration names them; every other
    dataset ships, since the release's readers also open some no configuration
    names (the SSE calendar, the listing and industry tables). The seed's
    ``provider.json`` records its configuration; an experiment whose configuration
    needs a dataset left behind is refused when it pins the release.
    """

    _with_project_src()
    from autotrade.environment.data.intraday_flow import MINUTE_DATASET
    from autotrade.environment.data.snapshot import SELECTABLE_DATASETS, SnapshotConfig
    from autotrade.pipelines.pit_backend import required_release_raw_datasets

    record = provider["snapshot_config"]
    datasets = record["datasets"]
    config = SnapshotConfig(
        fundamental_datasets=tuple(datasets["fundamentals"]),
        events_datasets=tuple(datasets["events"]),
        macro_datasets=tuple(datasets["macro"]),
        text_datasets=tuple(datasets["text"]),
        include_intraday=bool(record["include_intraday"]),
    )
    required = set(required_release_raw_datasets(config))
    return [name for name in (*SELECTABLE_DATASETS["text"], MINUTE_DATASET) if name not in required]


def _tree_bytes(root: Path) -> int:
    """The bytes of the regular files below ``root``, as remote.sh's local-disk step adds them up."""

    return sum(
        info.st_size
        for directory, _, names in os.walk(root)
        for info in (os.lstat(os.path.join(directory, name)) for name in names)
        if stat.S_ISREG(info.st_mode)
    )


def _said(completed: subprocess.CompletedProcess[bytes]) -> str:
    """': <the last line a captured stderr said>', or nothing."""

    lines = (completed.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()
    return f": {lines[-1]}" if lines else ""


def shown_state(experiment_id: str, shown: str) -> tuple[str, str]:
    """The line remote.sh's experiment-state step showed for ``experiment_id`` and the state it names; "" for either it lacks."""

    line = next((text for text in shown.splitlines() if text.startswith(f"{experiment_id}: state ")), "")
    state = re.match(rf"{re.escape(experiment_id)}: state (\S+?), ", line)
    return line, state.group(1) if state else ""


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
        # The workflow calls these two alone; they reach the API through `zenith` and fail without it.
        for name in ("dev", "instance"):
            if not os.access(self.tools / name, os.X_OK):
                raise WorkflowError(
                    f"{self.tools / name} is not executable: ZGCA_COMPUTE must name the "
                    "zgca-compute checkout (default ~/zgca-compute)"
                )

    @property
    def state_dir(self) -> Path:
        return self.repo_root / ".runtime" / "zgca"

    # ----- platform entries -----

    def call(
        self, tool: str, *args: str, capture: bool = False, data: bytes | None = None, quiet: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        """One run of a ~/zgca-compute entry (`instance`, `dev`), pointed at this workflow's instance; ``data`` is its stdin, ``quiet`` captures its stderr."""
        env = {**os.environ, "ZGCA_INSTANCE": self.instance}
        return subprocess.run(
            [str(self.tools / tool), *args],
            stdin=subprocess.DEVNULL if data is None else None, input=data,
            stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if quiet else None, env=env, check=False,
        )

    def lifecycle(self, command: str, *, capture: bool = False, quiet: bool = False) -> str:
        """`instance <command> <id>`, which verifies its own result and says why on stderr when it cannot (in the error, when quiet)."""
        completed = self.call("instance", command, self.instance, capture=capture, quiet=quiet)
        if completed.returncode != 0:
            raise WorkflowError(f"instance {command} {self.instance} failed (exit {completed.returncode})" + _said(completed))
        return (completed.stdout or b"").decode("utf-8", errors="replace").strip()

    def require_running(self, *, quiet: bool = False) -> None:
        state = self.lifecycle("state", capture=True, quiet=quiet)
        if state != "RUNNING":
            raise WorkflowError(f"instance {self.instance} is {state}, not RUNNING: run `up` first")

    # ----- instance shell and transfers -----

    def remote_run(self, step: str, *args: str, capture: bool = False, quiet: bool = False) -> subprocess.CompletedProcess[bytes]:
        """Run one step of remote.sh on the instance, in a login shell for the platform's environment, told the instance's layout."""
        script = REMOTE_SCRIPT.read_text(encoding="utf-8")
        return self.call(
            "dev", "bash", "-lc", script, "remote.sh", REMOTE_ROOT, REMOTE_VENV, LOCAL_ROOT,
            STORAGE, str(STORAGE_QUOTA), str(STORAGE_FLOOR), step, *args, capture=capture, quiet=quiet,
        )

    def remote(self, step: str, *args: str, capture: bool = False, quiet: bool = False) -> str:
        """One remote step that must succeed; its output when captured, and what it said on stderr in the error when quiet."""
        completed = self.remote_run(step, *args, capture=capture, quiet=quiet)
        if completed.returncode != 0:
            raise WorkflowError(f"remote step {step} failed (exit {completed.returncode})" + _said(completed))
        return (completed.stdout or b"").decode("utf-8", errors="replace")

    def dev(self, command: str, *args: str, data: bytes | None = None) -> None:
        """One `dev` primitive (bootstrap, copy, put, report), which verifies its own result and says why on stderr."""
        completed = self.call("dev", command, *args, data=data)
        if completed.returncode != 0:
            raise WorkflowError(f"dev {command} failed (exit {completed.returncode})" + (f": {' '.join(args[-2:])}" if args else ""))

    def git(self, *args: str) -> bytes:
        """The output of one git command in this repository; a git that fails fails the step."""
        completed = subprocess.run(["git", *args], cwd=self.repo_root, stdin=subprocess.DEVNULL, capture_output=True, check=False)
        if completed.returncode != 0:
            raise WorkflowError(f"git {' '.join(args)} failed in {self.repo_root}: {completed.stderr.decode('utf-8', errors='replace').strip()}")
        return completed.stdout

    def release_file(self, url: str, name: str, pinned: str) -> Path:
        """A verified release file the image build needs and the deploy ships, fetched once into the state directory.

        The sources are tried in order, the proxy first when one is set and
        the file is a GitHub release; a download lands beside the cache and
        enters it only with the pinned digest, and a partial one is resumed by
        the next source.
        """

        path = self.state_dir / name
        if not path.is_file():
            self.state_dir.mkdir(parents=True, exist_ok=True)
            part = path.with_name(name + ".part")
            proxy = os.environ.get(RELEASE_PROXY_ENV, "")
            sources = ([proxy + url] if proxy and url.startswith("https://github.com/") else []) + [url]
            for source in sources:
                say(f"deploy: fetching {source}")
                fetched = subprocess.run(
                    ["curl", "-fsSL", "--retry", "5", "--retry-all-errors", "--retry-delay", "3",
                     "--max-time", "900", "-C", "-", "-o", str(part), source],
                    stdin=subprocess.DEVNULL, check=False,
                )
                if fetched.returncode != 0 or not part.is_file():
                    say(f"deploy: {source} failed (curl exit {fetched.returncode})")
                    continue
                if _sha256(part) == pinned:
                    part.replace(path)
                    break
                part.unlink()
                say(f"deploy: {source} did not give the pinned release")
            else:
                raise WorkflowError(
                    f"no source gave {name} with the pinned digest ({', '.join(sources)}); "
                    f"{RELEASE_PROXY_ENV} names a proxy prefix to try before GitHub"
                )
        digest = _sha256(path)
        if digest != pinned:
            raise WorkflowError(
                f"{path} has digest {digest}, not the pinned {pinned}: remove it to fetch "
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
            provider = json.loads((seed_dir / "provider.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WorkflowError(f"{seed}/provider.json is unreadable: {exc}") from exc
        generation = str(provider.get("generation_id") or "")
        if not SAFE_ID.fullmatch(generation):
            raise WorkflowError(f"{seed}/provider.json names no research release generation")
        release_dir = self.repo_root / RELEASES / generation
        if not (release_dir / "manifest.json").is_file():
            raise WorkflowError(f"{seed} was built from research release {generation}, which is not published at {release_dir}")
        try:
            excluded = release_exclusions(provider)
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkflowError(f"{seed}/provider.json records no usable snapshot configuration: {exc!r}") from exc
        env_file = self.repo_root / ".env"
        if not env_file.is_file():
            raise WorkflowError(f"{env_file} is missing: the instance's sessions read the same model gateway keys")
        gateway_keys = gateway_env(env_file)
        if not gateway_keys:
            raise WorkflowError(f"{env_file} sets none of the keys the model profiles name")
        pretrained = self.repo_root / PRETRAINED
        if not (pretrained / PRETRAINED_MANIFEST).is_file():
            raise WorkflowError(
                f"{PRETRAINED}/{PRETRAINED_MANIFEST} is missing: the instance's experiments pin and mount that store"
            )
        # The tracked files ship as the working tree has them and the instance
        # records the commit they are, so the tree must be exactly that commit:
        # nothing modified or staged, and nothing untracked the tracked code
        # might import.
        changes = [line.strip() for line in self.git("status", "--porcelain").decode("utf-8", errors="replace").splitlines()]
        if changes:
            raise WorkflowError(
                "the working tree is not its commit, commit first (deploy records the commit it ships): "
                + ", ".join(changes[:5]) + (", ..." if len(changes) > 5 else "")
            )
        commit = self.git("rev-parse", "HEAD").decode("ascii").strip()
        listed = self.git("ls-files", "-z")
        if not listed:
            raise WorkflowError("git ls-files listed no tracked files in " + str(self.repo_root))
        payloads = self.require_pack_payloads(listed)
        store_bytes = _tree_bytes(pretrained)
        duckdb_cli = self.release_file(DUCKDB_CLI_URL, DUCKDB_CLI_NAME, DUCKDB_CLI_SHA256)
        cuda_compat = self.release_file(CUDA_COMPAT_URL, CUDA_COMPAT_NAME, CUDA_COMPAT_SHA256)
        # The list rsync copies from is also what the instance keeps: shipped
        # beside the release files, it lets the `tracked` step remove what is no
        # longer tracked (a retired round file would otherwise stay and be read).
        manifest = self.state_dir / TRACKED_LIST_NAME
        manifest.write_bytes(listed)
        deployed = self.state_dir / DEPLOYED_COMMIT_NAME
        deployed.write_text(commit + "\n", encoding="utf-8")
        self.require_running()
        root = REMOTE_ROOT
        # Nothing on the instance changes while experiments run there, or a
        # round launcher that could create one: the steps below overwrite and
        # prune the code they import, mirror the seed, release and store they
        # read with --delete, and may rebuild their image.
        self.remote("idle", "deploy")
        self.dev("bootstrap")
        self.remote("dirs")
        say(f"deploy: tracked files of commit {commit}")
        self.dev("copy", *PRESERVE, f"--files-from={manifest}", "--from0", f"{self.repo_root}/", f"dev:{root}/")
        say("deploy: .env (the model gateway keys only)")
        self.dev("put", f"{root}/.env", data=gateway_keys)
        say(f"deploy: {DUCKDB_CLI_NAME}, {CUDA_COMPAT_NAME} and {TRACKED_LIST_NAME}")
        self.dev("copy", *PRESERVE, str(duckdb_cli), str(cuda_compat), str(manifest), f"dev:{root}/.runtime/zgca/")
        self.remote("tracked")
        say(f"deploy: {seed}")
        self.dev("copy", *PRESERVE, "-H", "--delete", f"{seed_dir}/", f"dev:{root}/{seed_path.as_posix()}/")
        # Mirrored whole, so a payload removed here (its Pack left the tree) goes there too.
        shipped = payloads.relative_to(self.repo_root).as_posix()
        say(f"deploy: {shipped}")
        self.dev("copy", *PRESERVE, "--delete", f"{payloads}/", f"dev:{root}/{shipped}/")
        # Excluded directories are also kept on the instance (--delete without
        # --delete-excluded): another seed of the same release may read them.
        say(f"deploy: {RELEASES}/{generation} without {', '.join(excluded) or 'nothing'}")
        self.dev("copy", *PRESERVE, "-H", "--delete", *(f"--exclude=/raw/{name}/" for name in excluded),
                 f"{release_dir}/", f"dev:{root}/{RELEASES}/{generation}/")
        say("deploy: preparing the instance")
        self.remote("prepare", DUCKDB_CLI_SHA256, CUDA_COMPAT_SHA256)
        # The store goes last and only where it fits, so the steps above are
        # done whatever its size; a store that does not fit fails the deploy.
        store = f"{LOCAL_ROOT}/pretrained"
        present, free = (int(field) for field in self.remote("local-disk", capture=True).split()[-2:])
        missing = max(store_bytes - present, 0)
        if missing + LOCAL_DISK_RESERVE > free:
            self.dev("report")
            raise WorkflowError(
                f"the pretrained store does not fit on the instance's own disk: it needs {missing / GIB:.1f} GiB more "
                f"({store_bytes / GIB:.1f} GiB in all, {present / GIB:.1f} GiB already in {store}) plus "
                f"{LOCAL_DISK_RESERVE / GIB:.0f} GiB kept free for image rebuilds and container scratch, and "
                f"{LOCAL_DISK} has {free / GIB:.1f} GiB free. Code, seed, release and image are deployed; the store "
                "is not, so no experiment can be created or started there until it has room and deploy runs again"
            )
        say(f"deploy: {PRETRAINED} to {store}")
        self.dev("copy", *PRESERVE, "-H", "--delete", f"{pretrained}/", f"dev:{store}/")
        # The commit goes once everything else is in place: run and resume
        # take the recorded commit for a complete deploy of it.
        say(f"deploy: {DEPLOYED_COMMIT_NAME}")
        self.dev("copy", *PRESERVE, str(deployed), f"dev:{root}/.runtime/zgca/")
        self.dev("report")
        say("deploy: done")

    def require_pack_payloads(self, listed: bytes) -> Path:
        """Hold every tracked Pack with a payload manifest to it, as an experiment's pre-flight does on the instance.

        Returns the payload store, created empty here when no Pack has a
        payload yet so the deploy mirrors it all the same.
        """

        _with_project_src()
        from autotrade.pipelines.pack_payload import (
            PAYLOAD_MANIFEST,
            PAYLOAD_ROOT,
            PackPayloadError,
            pack_payload,
        )

        for entry in listed.split(b"\0"):
            path = Path(os.fsdecode(entry))
            if entry and path.name == PAYLOAD_MANIFEST:
                try:
                    pack_payload(self.repo_root / path.parent, self.repo_root)
                except PackPayloadError as exc:
                    raise WorkflowError(f"{exc}: scripts/experiments/fetch_pack_payload.py fetches it") from exc
        payloads = self.repo_root / PAYLOAD_ROOT
        payloads.mkdir(parents=True, exist_ok=True)
        return payloads

    def run(self, round_file: str, args: Sequence[str]) -> None:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", round_file],
            cwd=self.repo_root, stdin=subprocess.DEVNULL, capture_output=True, check=False,
        )
        if tracked.returncode != 0 or not (self.repo_root / round_file).is_file():
            raise WorkflowError(f"{round_file} is not a tracked file of the repository, so deploy would not have carried it")
        self.require_running()
        self.require_deployed_head()
        self.remote("console-up")
        self.remote("round", round_file, *args)

    def require_deployed_head(self) -> None:
        """Refuse unless the last deploy shipped this checkout's HEAD.

        The instance runs the tree the last deploy shipped, which no local check
        says anything about; the console and what it starts must run this one.
        """
        head = self.git("rev-parse", "HEAD").decode("ascii").strip()
        deployed = self.remote("deployed-commit", capture=True).split()[-1:]
        if deployed != [head]:
            raise WorkflowError(
                f"the instance runs commit {deployed[0] if deployed else '(none recorded)'}, not this checkout's HEAD {head}: "
                "deploy this commit first"
            )

    def resume(self, experiment_id: str) -> None:
        """Restart, through the console's own `resume`, an experiment whose worker is gone before it completed.

        Under `run`'s preconditions (the instance runs this checkout's HEAD; the
        console is brought up as `run` brings it up), for an experiment that
        ended `failed` or was left terminated or interrupted, and only while
        the storage floor is free. The marker goes before the request, so the
        experiment is copied back again when it ends; a refused request only
        costs the next sync pass a copy that changes nothing.
        """

        if not SAFE_ID.fullmatch(experiment_id):
            raise WorkflowError(f"not an experiment id: {experiment_id!r}")
        self.require_running()
        self.require_deployed_head()
        line, state = shown_state(experiment_id, self.remote("experiment-state", experiment_id, capture=True, quiet=True))
        if not state:
            raise WorkflowError(f"remote step experiment-state showed no state for {experiment_id}")
        say(line)
        if state in NOT_RESUMED:
            raise WorkflowError(f"{experiment_id} ended {state}: resume restarts a failed, terminated or interrupted experiment")
        self.require_storage()
        self.remote("console-up")
        self.sync_marker(experiment_id).unlink(missing_ok=True)
        self.remote("resume", experiment_id)

    def require_storage(self) -> None:
        """Refuse unless the storage floor is free under the quota; an unmeasured storage is not free."""
        completed = self.remote_run("storage", capture=True, quiet=True)
        line = (completed.stdout or b"").decode("utf-8", errors="replace").strip()
        if completed.returncode != 0:
            raise WorkflowError(
                (line or "storage: unmeasured" + _said(completed))
                + f"; nothing is started on the instance while less than {STORAGE_FLOOR / GIB:.0f} GiB is free under the quota"
            )
        say(line)

    def status(self) -> None:
        self.lifecycle("show")
        if self.lifecycle("state", capture=True) == "RUNNING":
            self.dev("report")
            self.remote("status")

    def sync(self, experiment_id: str) -> None:
        local = self.sync_target(experiment_id)
        self.require_running()
        self.remote("experiment-state", experiment_id)
        self.copy_back(experiment_id, local)
        untagged = self.tag_and_release(experiment_id, local)
        if untagged:
            raise untagged

    def sync_target(self, experiment_id: str) -> Path:
        """Where a sync of ``experiment_id`` lands: never over an experiment of this host."""
        if not SAFE_ID.fullmatch(experiment_id):
            raise WorkflowError(f"not an experiment id: {experiment_id!r}")
        local = self.repo_root / "experiments" / experiment_id
        if local.exists():
            try:
                created_under = json.loads((local / "hitl" / "params.json").read_text(encoding="utf-8")).get("experiments_root")
            except (OSError, ValueError) as exc:
                raise WorkflowError(f"experiments/{experiment_id} exists here and its hitl/params.json cannot say where it was created ({exc}); refusing to overwrite it") from exc
            if created_under != f"{REMOTE_ROOT}/experiments":
                raise WorkflowError(f"experiments/{experiment_id} is an experiment of this host (created under {created_under}), not a copy from the instance; refusing to overwrite it")
        return local

    def sync_marker(self, experiment_id: str) -> Path:
        return self.state_dir / SYNCED_DIR_NAME / experiment_id

    def sync_stage(self, experiment_id: str) -> str:
        """What the marker says (`copied`, `synced`, `released`), or nothing; anything else is no completed stage."""
        marker = self.sync_marker(experiment_id)
        return marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""

    @staticmethod
    def completed_copy(local: Path) -> bool:
        """Whether the copy here is of a completed experiment, by the status it copied; an unreadable one is not."""
        try:
            state = json.loads((local / "hitl" / "status.json").read_text(encoding="utf-8")).get("state")
        except (OSError, ValueError, AttributeError):
            return False
        return state == COMPLETED

    def tag_and_release(self, experiment_id: str, local: Path, *, tag: bool = True) -> WorkflowError | None:
        """Tag a verified copy (when ``tag``) and release a completed one's bulk on the instance, neither waiting on the other.

        Advances the marker as ops/zgca/README.md (sync) says; a release the
        instance refuses raises. Returns why no tag was made.
        """
        untagged = None
        if tag:
            try:
                self.tag_pinned_image(experiment_id, local)
            except WorkflowError as exc:
                untagged = exc
        settled = untagged is None or isinstance(untagged, PinnedImageAbsent)
        marker = self.sync_marker(experiment_id)
        if settled:
            marker.write_text("synced\n", encoding="utf-8")
        if self.completed_copy(local):
            self.remote("release", experiment_id, quiet=True)
            if settled:
                marker.write_text("released\n", encoding="utf-8")
        return untagged

    def copy_back(self, experiment_id: str, local: Path) -> None:
        """Copy the experiment without pit_views and its worker log; the marker is gone meanwhile and says `copied` once both copies are verified.

        A first copy is made in the incoming directory and renamed into place once verified (INCOMING_DIR_NAME).
        """
        marker = self.sync_marker(experiment_id)
        marker.unlink(missing_ok=True)
        target = local if local.exists() else self.state_dir / INCOMING_DIR_NAME / experiment_id
        target.parent.mkdir(parents=True, exist_ok=True)
        self.dev("copy", *PRESERVE, "--exclude", "/pit_views/", f"dev:{REMOTE_ROOT}/experiments/{experiment_id}/", f"{target}/")
        if target != local:
            local.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.rename(local)
            except OSError as exc:
                raise WorkflowError(f"the verified copy in {target} cannot enter experiments/{experiment_id}: {exc}") from exc
        logs = self.repo_root / "logs" / "workers"
        logs.mkdir(parents=True, exist_ok=True)
        self.dev("copy", *PRESERVE, f"dev:{REMOTE_ROOT}/logs/workers/{experiment_id}.log", f"{logs / experiment_id}.log")
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("copied\n", encoding="utf-8")
        say(f"synced experiments/{experiment_id} (without pit_views) and logs/workers/{experiment_id}.log")

    def round_experiments(self, round_file: str) -> list[str]:
        """The experiment ids a round file defines (``ROUND.experiments``), in file order."""
        path = self.repo_root / round_file
        spec = importlib.util.spec_from_file_location(f"zgca_round_{path.stem}", path)
        if not path.is_file() or spec is None or spec.loader is None:
            raise WorkflowError(f"{round_file} is not a round file of {self.repo_root}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return list(module.ROUND.experiments)

    def sync_rounds(self, round_files: Sequence[str]) -> bool:
        """One pass of the sync cron over these rounds' experiments (ops/zgca/README.md, sync-rounds).

        False when an experiment failed or is untagged, or the instance could not be asked.
        """

        _with_project_src()
        from autotrade.pipelines.hitl_state import ENDED_STATES

        experiments = [experiment for round_file in round_files for experiment in self.round_experiments(round_file)]
        counts = dict.fromkeys(SYNC_COUNTS, 0)
        unsynced = []
        for experiment in experiments:
            stage = self.sync_stage(experiment)
            if stage == "released" or (stage == "synced" and not self.completed_copy(self.repo_root / "experiments" / experiment)):
                counts["synced"] += 1
            else:
                unsynced.append(experiment)
        problem = ""
        if unsynced:
            try:
                self.require_running(quiet=True)
            except WorkflowError as exc:
                problem = str(exc)
        for experiment in [] if problem else unsynced:
            try:
                counts[self.sync_ended(experiment, ENDED_STATES)] += 1
            except WorkflowError as exc:
                say(f"{experiment}: failed, retried next pass: {exc}")
                counts["failed"] += 1
        names = ", ".join(Path(round_file).stem for round_file in round_files) or "no round"
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} zgca sync {names}: {len(experiments)} experiments: "
        line += ", ".join(f"{counts[name]} {name}" for name in SYNC_COUNTS)
        if problem:
            line += f", {len(unsynced)} not checked ({problem})"
        print(line, flush=True)
        return not (problem or counts["failed"] or counts["untagged"])

    def sync_ended(self, experiment_id: str, ended_states: frozenset[str]) -> str:
        """Bring one experiment back if it has ended, and release it there once it is back completed; what the pass counts it as."""
        local = self.sync_target(experiment_id)
        stage = self.sync_stage(experiment_id)
        if stage not in ("copied", "synced"):
            completed = self.remote_run("experiment-state", experiment_id, capture=True, quiet=True)
            line, state = shown_state(experiment_id, (completed.stdout or b"").decode("utf-8", errors="replace"))
            if completed.returncode == REMOTE_ABSENT:
                return "pending"
            if completed.returncode == REMOTE_RUNNING:
                return "running"
            if completed.returncode != 0:
                raise WorkflowError(f"remote step experiment-state failed (exit {completed.returncode})" + _said(completed))
            if not state:
                raise WorkflowError(f"remote step experiment-state showed no state for {experiment_id}")
            if state not in ended_states:
                say(f"{line}: not ended, yet no worker runs it (the console shows it interrupted); "
                    f"not synced, `zgca.py resume {experiment_id}` restarts it, `zgca.py sync {experiment_id}` copies it as it is")
                return "interrupted"
            say(f"{line}: ended, syncing")
            self.copy_back(experiment_id, local)
        untagged = self.tag_and_release(experiment_id, local, tag=stage != "synced")
        if isinstance(untagged, PinnedImageAbsent):
            say(f"{experiment_id}: synced without a tag, and no later pass tries again: {untagged}")
        elif untagged:
            say(f"{experiment_id}: copied, not tagged, tried again next pass: {untagged}")
            return "untagged"
        return "synced"

    def tag_pinned_image(self, experiment_id: str, local: Path) -> None:
        """Give a synced experiment the sandbox image tag it pins on this host, as every local experiment's is made.

        A local experiment's tag is a clone of the base image
        (``sandbox_images.prepare_experiment_sandbox_image``), and a Paper
        account runs the experiment's tag; the instance's tag is not here. The
        tag is made only when the image it names on the instance and this host's
        base bake the same strategy runtime, each read in one offline container
        (``contract_fingerprint.read_image_contract``), so the strategy runs here
        under the rules it was validated by. ``docker tag`` is idempotent, so a
        sync run again re-points the tag at the base image as it is then.
        """

        _with_project_src()
        from autotrade.environment.contract_fingerprint import (
            SandboxImageContractMismatch,
            read_image_contract,
        )
        from autotrade.environment.sandbox import DEFAULT_IMAGE, SandboxSpec
        from autotrade.environment.sandbox_images import SANDBOX_IMAGE_STATE_NAME

        state = local / "hitl" / SANDBOX_IMAGE_STATE_NAME
        if not state.is_file():
            say(f"experiments/{experiment_id} pins no sandbox image: no tag to make")
            return
        try:
            image = str(json.loads(state.read_text(encoding="utf-8"))["image_ref"])
            SandboxSpec(image=image)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise WorkflowError(f"experiments/{experiment_id}/hitl/{SANDBOX_IMAGE_STATE_NAME} names no usable image tag: {exc!r}") from exc
        shown = self.remote_run("image-runtime", image, capture=True, quiet=True)
        if shown.returncode == REMOTE_ABSENT:
            raise PinnedImageAbsent(
                f"experiments/{experiment_id} is synced, but its pinned image {image} is no longer on the instance "
                "(tags are lost across stop/start), so the strategy runtime it baked cannot be read: no tag is made "
                "here and its strategy cannot be followed on Paper here"
            )
        if shown.returncode != 0:
            raise WorkflowError(f"remote step image-runtime failed (exit {shown.returncode})" + _said(shown))
        remote = (shown.stdout or b"").decode("utf-8", errors="replace").split()[-1:]
        try:
            base = read_image_contract(DEFAULT_IMAGE).runtime
        except SandboxImageContractMismatch as exc:
            raise WorkflowError(str(exc)) from exc
        if remote != [base]:
            raise WorkflowError(
                f"experiments/{experiment_id} is synced, but its strategy cannot be followed on Paper here until the "
                f"images agree: its image {image} bakes strategy runtime {remote[0] if remote else '(none read)'} on the "
                f"instance, this host's {DEFAULT_IMAGE} bakes {base}; no tag was made"
            )
        tagged = subprocess.run(["docker", "tag", DEFAULT_IMAGE, image], stdin=subprocess.DEVNULL, capture_output=True, check=False)
        if tagged.returncode != 0:
            raise WorkflowError(f"docker tag {DEFAULT_IMAGE} {image} failed: {tagged.stderr.decode('utf-8', errors='replace').strip()}")
        say(f"tagged {DEFAULT_IMAGE} as {image}: both bake strategy runtime {base}")

    def down(self) -> None:
        # Stopping kills every worker and round launcher on the instance; one
        # that is not RUNNING runs none.
        if self.lifecycle("state", capture=True) == "RUNNING":
            self.remote("idle", "down")
        self.lifecycle("stop")


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    tools: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(prog="zgca.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--instance", type=instance_id, default=INSTANCE_ID,
        help=f"platform id of the instance every step addresses (default {INSTANCE_ID}: auto-trade-dev-a100x2, one A100-80G despite its name)",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("up", help="instance start: wait for RUNNING and SSH")
    deploy = commands.add_parser("deploy", help="refuse an uncommitted tree, a Pack payload unlike its manifest or running experiments; copy code, the model gateway keys, a seed, the Pack payloads and the seed's release; prepare ripgrep, venv, image; copy the pretrained store")
    deploy.add_argument("seed", help="prebuilt view cache directory, relative to the repository (data/pit_views_seed_...)")
    run = commands.add_parser("run", help="refuse unless the instance runs this checkout's HEAD; start the console, or restart it when no experiment runs, then a round launcher in tmux; --fill repeats it every ten minutes, holding a pass while the storage floor is not free, until nothing is pending")
    run.add_argument("round_file", help="scripts/experiments/create_round_*.py")
    run.add_argument("args", nargs=argparse.REMAINDER, help="launcher arguments after the port (--dry-run, --fill, experiment ids)")
    resume = commands.add_parser("resume", help="refuse unless the instance runs this checkout's HEAD and the storage floor is free; restart a failed, terminated or interrupted experiment through the console and drop its sync marker")
    resume.add_argument("experiment_id", help="experiment id")
    commands.add_parser("status", help="instance, card-hours, GPU, disk, tmux, storage against its quota, deployed commit, console, experiments, image, round log")
    sync = commands.add_parser("sync", help="copy an experiment's records and worker log back here, release a completed one's bulk on the instance, and tag its pinned sandbox image here when the runtimes agree")
    sync.add_argument("experiment_id", help="experiment id")
    sync_rounds = commands.add_parser("sync-rounds", help="one pass of the sync cron: sync, once, every experiment of these rounds that has ended on the instance, releasing the completed ones there; one summary line")
    sync_rounds.add_argument("round_files", nargs="*", help="scripts/experiments/create_round_*.py")
    commands.add_parser("down", help="instance stop, refused while experiments run: wait for STOPPED")
    options = parser.parse_args(argv)
    try:
        workflow = Workflow(
            repo_root=repo_root,
            tools=tools or Path(os.environ.get("ZGCA_COMPUTE") or "~/zgca-compute").expanduser(),
            instance=options.instance,
        )
        if options.command == "up":
            workflow.up()
        elif options.command == "deploy":
            workflow.deploy(options.seed)
        elif options.command == "run":
            workflow.run(options.round_file, options.args)
        elif options.command == "resume":
            workflow.resume(options.experiment_id)
        elif options.command == "status":
            workflow.status()
        elif options.command == "sync":
            workflow.sync(options.experiment_id)
        elif options.command == "sync-rounds":
            return 0 if workflow.sync_rounds(options.round_files) else 1
        else:
            workflow.down()
    except WorkflowError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
