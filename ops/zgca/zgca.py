#!/usr/bin/env python3
"""The ZGCA A100 development instance, end to end: up, deploy, push, run, status, sync, down.

Every step goes through the platform layer in ~/zgca-compute (`instance` to
show, start and stop the instance as idempotent intents; `dev` to reach it:
`bootstrap` for its base tools, `copy` for verified rsync transfers, `put`
for an owner-only file, `report` for its GPUs, disks and tmux sessions, and
plain commands for the steps of ops/zgca/remote.sh; both over `zenith`) and
verifies its own result. up and down are `instance start` and `instance
stop`; a deploy carries the tracked files, the model gateways' keys from the
local .env (only the keys the model profiles name), the named PIT view seed
and the research release that seed was built from (without the text and
minute datasets its configuration does not read), prunes code the tree no
longer tracks, prepares the instance (ripgrep, venv, a sandbox image that runs
on the instance's GPU driver), and last copies the pretrained-model store every
new arm pins and mounts to the instance's own disk, or refuses when it does
not fit there; push carries arms of this host a round names as lineage; a run
starts the console and a round launcher in tmux with AUTOTRADE_GPU_NAME set;
status and sync read and copy back. ZGCA_REMOTE_ROOT moves all of it to
another root on the instance's personal storage (a validation deploy beside
the live tree, on an instance other than the default). Nothing here
schedules, caches an endpoint, keeps a secret in the repository or changes how
the executor isolates a strategy container. See ops/zgca/README.md.

usage: zgca.py [--instance <id>] up | deploy <seed dir> | push <arm id>... |
       run <round file> [launcher args...] | status | sync <arm id> | down
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

# auto-trade-dev-a100x2: two Tesla-A100-80G in the Volcano zone, the default
# of --instance; its id is the platform ledger id the API addresses it by, not
# a secret.
INSTANCE_ID = "mpo06t71crmmtdsrb98ckwk29"
# The platform's own shape for an instance id (~/zgca-compute/dev).
INSTANCE_ID_SHAPE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
# The instance's persistent personal storage: the only place code, the .env,
# seeds and releases go (~/zgca-compute/README.md, 配额、规格与存储).
REMOTE_ROOT = "/personal_fast_storage/auto-trade"
# Names another root directly below that storage for every transfer and
# remote step, e.g. a validation deploy beside the live tree, which every
# instance of the zone sees; remote.sh receives the root on each call and
# keeps such a root's venv inside it.
REMOTE_ROOT_ENV = "ZGCA_REMOTE_ROOT"
REMOTE_ROOT_SHAPE = re.compile(r"^/personal_fast_storage/[A-Za-z0-9][A-Za-z0-9._-]*$")
# The instance's own disk, which also holds its root and Docker's images
# (~/zgca-compute/README.md, 火山平台 Docker): the pretrained store lives there,
# in <LOCAL_DISK>/<name of the remote root>/pretrained, and the root's
# data/pretrained links to it. remote.sh receives that directory on each call.
LOCAL_DISK = "/ebs"
LOCAL_ROOT_ENV = "ZGCA_LOCAL_ROOT"
# Left free on that disk beside the store: the new layers of an image rebuild
# and the containers' scratch. Filling it would stop Docker.
LOCAL_DISK_RESERVE = 20 * 2**30
RELEASES = "data/research_releases"
UNFINISHED_SEED_MARKER = "prebuild_unfinished.json"
# The DuckDB CLI release the sandbox image installs (ops/docker/sandbox.Dockerfile,
# build arg DUCKDB_CLI_URL). GitHub is unreachable from the instance and its
# release proxies answer or refuse by the day, so the deploy fetches the
# release where GitHub is reachable (through the proxy RELEASE_PROXY_ENV names,
# then directly), verifies it against this digest (two independent routes
# agreed on it on 2026-10-15), ships it, and the build reads it from the
# instance host through that build arg.
DUCKDB_CLI_URL = "https://github.com/duckdb/duckdb/releases/download/v1.1.3/duckdb_cli-linux-amd64.zip"
DUCKDB_CLI_NAME = "duckdb_cli-linux-amd64.zip"
DUCKDB_CLI_SHA256 = "efd0fccdb1a28d9ec7a6ebfcde59900068b8ba43a846c9b553c0fd2bbe4acf43"
# A URL prefix tried before a GitHub release's own URL, e.g. https://ghfast.top/.
RELEASE_PROXY_ENV = "ZGCA_GITHUB_PROXY"
# NVIDIA's CUDA forward-compatibility package for CUDA 13.0 (the user-mode
# driver of the 580 series). The image's torch needs CUDA 13, the instance's
# kernel driver is 535 (CUDA 12.2), and an A100 may run a newer user-mode
# driver on an older kernel driver; the instance's image build installs it
# (build arg CUDA_COMPAT_URL), other builds do not. Pinned to the digest of
# NVIDIA's Debian 12 package index, after a probe on the instance on
# 2026-10-24 (ops/zgca/README.md).
CUDA_COMPAT_URL = "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-compat-13-0_580.196.07-1_amd64.deb"
CUDA_COMPAT_NAME = "cuda-compat-13-0_580.196.07-1_amd64.deb"
CUDA_COMPAT_SHA256 = "d2d3262512046b23215ff197fe79a6ca0d44299ed0e665275ac05bb28b79e4e3"
# The pretrained-model store (environment.pretrained): the console stamps its
# manifest digest into every arm it creates and every container compares it.
PRETRAINED = "data/pretrained"
PRETRAINED_MANIFEST = "MANIFEST.sha256"
TRACKED_LIST_NAME = "tracked.txt"
GIB = 2**30
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
# Every transfer keeps contents, times and modes; the instance's root owns what lands there.
PRESERVE = ("-a", "--no-owner", "--no-group")
REMOTE_SCRIPT = Path(__file__).with_name("remote.sh")
REPO_ROOT = Path(__file__).resolve().parents[2]


class WorkflowError(Exception):
    """A step that did not reach its verified result; the message says why."""


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
    ``provider.json`` records its configuration; an arm whose configuration
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
    remote_root: str = REMOTE_ROOT

    def __post_init__(self) -> None:
        if not REMOTE_ROOT_SHAPE.fullmatch(self.remote_root):
            raise WorkflowError(
                f"{REMOTE_ROOT_ENV}={self.remote_root!r} is not a directory directly below /personal_fast_storage"
            )
        # An instance's sandbox image, tmux sessions and console port serve one
        # root: another root there could rebuild the image the live arms run on
        # (its guard sees only its own root's arms) or launch through the live
        # console, so the default instance keeps the live tree's.
        if self.remote_root != REMOTE_ROOT and self.instance == INSTANCE_ID:
            raise WorkflowError(
                f"{REMOTE_ROOT_ENV}={self.remote_root} is not the live tree's root, which the default instance "
                f"{INSTANCE_ID} serves: name the instance this root is for with --instance"
            )
        for name in ("zenith", "dev", "instance"):
            if not os.access(self.tools / name, os.X_OK):
                raise WorkflowError(
                    f"{self.tools / name} is not executable: ZGCA_COMPUTE must name the "
                    "zgca-compute checkout (default ~/zgca-compute)"
                )

    @property
    def state_dir(self) -> Path:
        return self.repo_root / ".runtime" / "zgca"

    @property
    def local_root(self) -> str:
        """This root's directory on the instance's own disk, apart from any other root's."""
        return f"{LOCAL_DISK}/{PurePosixPath(self.remote_root).name}"

    # ----- platform entries -----

    def call(
        self, tool: str, *args: str, capture: bool = False, data: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        """One run of a ~/zgca-compute entry (`instance`, `dev`), pointed at this workflow's instance; ``data`` is its stdin."""
        env = {**os.environ, "ZGCA_INSTANCE": self.instance}
        return subprocess.run(
            [str(self.tools / tool), *args],
            stdin=subprocess.DEVNULL if data is None else None, input=data,
            stdout=subprocess.PIPE if capture else None, env=env, check=False,
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

    def remote(self, step: str, *args: str, capture: bool = False) -> str:
        """Run one step of remote.sh on the instance, in a login shell for the platform's environment, under this workflow's roots; its output when captured."""
        script = REMOTE_SCRIPT.read_text(encoding="utf-8")
        completed = self.call(
            "dev", "env", f"{REMOTE_ROOT_ENV}={self.remote_root}", f"{LOCAL_ROOT_ENV}={self.local_root}",
            "bash", "-lc", script, "remote.sh", step, *args, capture=capture,
        )
        if completed.returncode != 0:
            raise WorkflowError(f"remote step {step} failed (exit {completed.returncode})")
        return (completed.stdout or b"").decode("utf-8", errors="replace")

    def dev(self, command: str, *args: str, data: bytes | None = None) -> None:
        """One `dev` primitive (bootstrap, copy, put, report), which verifies its own result and says why on stderr."""
        completed = self.call("dev", command, *args, data=data)
        if completed.returncode != 0:
            raise WorkflowError(f"dev {command} failed (exit {completed.returncode})" + (f": {' '.join(args[-2:])}" if args else ""))

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
                f"{PRETRAINED}/{PRETRAINED_MANIFEST} is missing: the instance's arms pin and mount that store"
            )
        listed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=self.repo_root, stdin=subprocess.DEVNULL, capture_output=True, check=False,
        )
        if listed.returncode != 0 or not listed.stdout:
            raise WorkflowError("git ls-files listed no tracked files in " + str(self.repo_root))
        store_bytes = _tree_bytes(pretrained)
        duckdb_cli = self.release_file(DUCKDB_CLI_URL, DUCKDB_CLI_NAME, DUCKDB_CLI_SHA256)
        cuda_compat = self.release_file(CUDA_COMPAT_URL, CUDA_COMPAT_NAME, CUDA_COMPAT_SHA256)
        # The list rsync copies from is also what the instance keeps: shipped
        # beside the release files, it lets the `tracked` step remove what is no
        # longer tracked (a retired round file would otherwise stay and be read).
        manifest = self.state_dir / TRACKED_LIST_NAME
        manifest.write_bytes(listed.stdout)
        self.require_running()
        root = self.remote_root
        self.dev("bootstrap")
        self.remote("dirs")
        say("deploy: tracked files")
        self.dev("copy", *PRESERVE, f"--files-from={manifest}", "--from0", f"{self.repo_root}/", f"dev:{root}/")
        say("deploy: .env (the model gateway keys only)")
        self.dev("put", f"{root}/.env", data=gateway_keys)
        say(f"deploy: {DUCKDB_CLI_NAME}, {CUDA_COMPAT_NAME} and {TRACKED_LIST_NAME}")
        self.dev("copy", *PRESERVE, str(duckdb_cli), str(cuda_compat), str(manifest), f"dev:{root}/.runtime/zgca/")
        self.remote("tracked")
        say(f"deploy: {seed}")
        self.dev("copy", *PRESERVE, "-H", "--delete", f"{seed_dir}/", f"dev:{root}/{seed_path.as_posix()}/")
        # Excluded directories are also kept on the instance (--delete without
        # --delete-excluded): another seed of the same release may read them.
        say(f"deploy: {RELEASES}/{generation} without {', '.join(excluded) or 'nothing'}")
        self.dev("copy", *PRESERVE, "-H", "--delete", *(f"--exclude=/raw/{name}/" for name in excluded),
                 f"{release_dir}/", f"dev:{root}/{RELEASES}/{generation}/")
        say("deploy: preparing the instance")
        self.remote("prepare", DUCKDB_CLI_SHA256, CUDA_COMPAT_SHA256)
        # The store goes last and only where it fits, so the steps above are
        # done whatever its size; a store that does not fit fails the deploy.
        store = f"{self.local_root}/pretrained"
        present, free = (int(field) for field in self.remote("local-disk", capture=True).split()[-2:])
        missing = max(store_bytes - present, 0)
        if missing + LOCAL_DISK_RESERVE > free:
            self.dev("report")
            raise WorkflowError(
                f"the pretrained store does not fit on the instance's own disk: it needs {missing / GIB:.1f} GiB more "
                f"({store_bytes / GIB:.1f} GiB in all, {present / GIB:.1f} GiB already in {store}) plus "
                f"{LOCAL_DISK_RESERVE / GIB:.0f} GiB kept free for image rebuilds and container scratch, and "
                f"{LOCAL_DISK} has {free / GIB:.1f} GiB free. Code, seed, release and image are deployed; the store "
                "is not, so no arm can be created or started there until it has room and deploy runs again"
            )
        say(f"deploy: {PRETRAINED} to {store}")
        self.dev("copy", *PRESERVE, "-H", "--delete", f"{pretrained}/", f"dev:{store}/")
        self.dev("report")
        say("deploy: done")

    def push(self, arms: Sequence[str]) -> None:
        """Copy arms of this host to the instance, where an arm naming them as lineage is created."""
        # The console's own notion of a running arm; its module needs the project's environment (numpy).
        _with_project_src()
        from autotrade.pipelines.hitl_state import running_arms

        experiments = self.repo_root / "experiments"
        for arm in arms:
            if not SAFE_ID.fullmatch(arm):
                raise WorkflowError(f"not an experiment id: {arm!r}")
            try:
                created_under = json.loads((experiments / arm / "hitl" / "params.json").read_text(encoding="utf-8")).get("experiments_root")
            except (OSError, ValueError) as exc:
                raise WorkflowError(f"experiments/{arm} is not a readable arm of this host ({exc})") from exc
            if created_under == f"{self.remote_root}/experiments":
                raise WorkflowError(f"experiments/{arm} is a copy of an arm of the instance; its original is there")
        running = sorted(set(arms) & set(running_arms(experiments)))
        if running:
            raise WorkflowError(f"{', '.join(running)} still running here: a copy cannot be verified while it changes")
        self.require_running()
        for arm in arms:
            say(f"push: experiments/{arm} (without pit_views)")
            self.dev("copy", *PRESERVE, "--exclude", "/pit_views/", f"{experiments / arm}/", f"dev:{self.remote_root}/experiments/{arm}/")

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
            self.dev("report")
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
            if created_under != f"{self.remote_root}/experiments":
                raise WorkflowError(f"experiments/{arm} is an arm of this host (created under {created_under}), not a copy from the instance; refusing to overwrite it")
        self.require_running()
        self.remote("arm-state", arm)
        local.parent.mkdir(parents=True, exist_ok=True)
        self.dev("copy", *PRESERVE, "--exclude", "/pit_views/", f"dev:{self.remote_root}/experiments/{arm}/", f"{local}/")
        logs = self.repo_root / "logs" / "workers"
        logs.mkdir(parents=True, exist_ok=True)
        self.dev("copy", *PRESERVE, f"dev:{self.remote_root}/logs/workers/{arm}.log", f"{logs / arm}.log")
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
    parser.add_argument(
        "--instance", type=instance_id, default=INSTANCE_ID,
        help=f"platform id of the instance every step addresses (default {INSTANCE_ID}, auto-trade-dev-a100x2)",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    commands.add_parser("up", help="instance start: wait for RUNNING and SSH")
    deploy = commands.add_parser("deploy", help="copy code, the model gateway keys, a seed and its release; prepare ripgrep, venv, image; copy the pretrained store")
    deploy.add_argument("seed", help="PIT view seed directory, relative to the repository (data/pit_views_seed_...)")
    push = commands.add_parser("push", help="copy arms of this host to the instance (lineage of an arm created there)")
    push.add_argument("arms", nargs="+", metavar="arm", help="experiment id")
    run = commands.add_parser("run", help="start the console if needed and a round launcher in tmux")
    run.add_argument("round_file", help="scripts/experiments/create_round_*.py")
    run.add_argument("args", nargs=argparse.REMAINDER, help="launcher arguments after the port (--dry-run, --fill, arm ids)")
    commands.add_parser("status", help="instance, card-hours, GPU, disk, tmux, console, arms, image, round log")
    sync = commands.add_parser("sync", help="copy an arm's records and worker log back here")
    sync.add_argument("arm", help="experiment id")
    commands.add_parser("down", help="instance stop: wait for STOPPED")
    options = parser.parse_args(argv)
    try:
        workflow = Workflow(
            repo_root=repo_root,
            tools=tools or Path(os.environ.get("ZGCA_COMPUTE") or "~/zgca-compute").expanduser(),
            instance=options.instance,
            remote_root=os.environ.get(REMOTE_ROOT_ENV, REMOTE_ROOT),
        )
        if options.command == "up":
            workflow.up()
        elif options.command == "deploy":
            workflow.deploy(options.seed)
        elif options.command == "push":
            workflow.push(options.arms)
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
