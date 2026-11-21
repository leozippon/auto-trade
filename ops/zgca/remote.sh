#!/usr/bin/env bash
# The instance-side half of ops/zgca/zgca.py: one step per call, sent by the
# workflow as `bash -lc "$(cat remote.sh)" remote.sh <root> <venv> <local dir>
# <storage> <quota bytes> <floor bytes> <step> [args]` so the platform's login
# environment (DOCKER_HOST) is present, the script is always the local
# checkout's, never a stale copy on the instance, and the instance's layout and
# storage quota are stated in zgca.py alone. Runs as the instance's root;
# everything persistent lives on the quota-bound storage except the pretrained
# store, which is too large for that quota and lives in <local dir>/pretrained
# on the instance's own disk, linked from the root's data/pretrained. The
# generic host work (base tools, transfers, GPU/disk/tmux report) is
# ~/zgca-compute's `dev`, which the workflow calls around these steps.
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
# A step whose subject is not on the instance exits ABSENT, one whose subject
# is still running there RUNNING (zgca.py's REMOTE_ABSENT and REMOTE_RUNNING),
# which the workflow tells from a step that failed.
ABSENT=3
RUNNING=4
absent() { echo "ERROR: $*" >&2; exit "$ABSENT"; }

[ $# -ge 7 ] || die "usage: remote.sh <root> <venv> <local dir> <storage> <quota bytes> <floor bytes> <step> [args...]"
ROOT=$1
VENV=$2
LOCAL=$3
STORAGE=$4
QUOTA=$5
FLOOR=$6
shift 6
for bytes in "$QUOTA" "$FLOOR"; do
    case "$bytes" in ''|*[!0-9]*) die "the storage quota and floor are byte counts, not '$QUOTA' and '$FLOOR'" ;; esac
done
BOX_PYTHON=/root/miniconda3/bin/python
IMAGE=autotrade-sandbox:latest
# Shipped by the deploy with the digest it pinned; GitHub and its release
# proxies are not reliably reachable from the instance, so the build reads the
# DuckDB CLI from this host over the bridge network instead.
DUCKDB_CLI=.runtime/zgca/duckdb_cli-linux-amd64.zip
# NVIDIA's CUDA 13.0 forward-compatibility package, shipped and served the
# same way: the image's torch needs CUDA 13 and this host's driver is older.
CUDA_COMPAT=.runtime/zgca/cuda-compat-13-0_580.196.07-1_amd64.deb
SERVE_PORT=8765
PORT=38888
GPU_NAME=A100
CONSOLE_SESSION=autotrade-console
ROUND_SESSION=autotrade-round
# A `--fill` launcher repeats in a session of its own until its round has
# nothing pending, so a later `run` of another round neither kills it nor
# waits for it; FILL_INTERVAL is the local research-fill cron's period.
FILL_SESSION=autotrade-fill
FILL_INTERVAL=600
HEALTH="http://127.0.0.1:$PORT/api/health"

require_venv() { [ -x "$VENV/bin/python" ] || die "no venv at $VENV: run deploy first"; }

# storage_check <storage> <quota bytes> <floor bytes>: one line on the storage
# used against its quota, and exit 0 only while at least the floor is free.
# Neither df nor statvfs sees the quota (zgca.py, STORAGE_QUOTA), so this
# measures du of the whole tree, which counts a hard link once, as the quota
# does; a du that fails or outlasts its limit (about a second on 2026-11-20;
# GPFS has hung a du for an hour) is no measure, and an unknown is never free.
# `status` prints the line; the fill loop, which runs it in a shell of its own
# (written for sh), and `resume` hold below the floor.
storage_check() {
    local said used rc limit=300
    said=$(timeout "$limit" du -sxB1 "$1" 2>&1) || {
        rc=$?
        if [ "$rc" -eq 124 ]; then
            echo "storage: unknown, du of $1 did not finish within $limit s"
        else
            echo "storage: unknown, du of $1 failed (exit $rc): $(printf '%s\n' "$said" | grep -v '^[0-9]' | tail -n 1)"
        fi
        return 1
    }
    used=${said%%[!0-9]*}
    [ -n "$used" ] || { echo "storage: unknown, du of $1 printed no total: $said"; return 1; }
    awk -v used="$used" -v quota="$2" -v floor="$3" -v storage="$1" 'BEGIN {
        printf "storage: %.1f of %.0f GiB used, %.1f free (du of %s)", used / 2^30, quota / 2^30, (quota - used) / 2^30, storage
        if (quota - used < floor) { printf ", below the %.0f GiB floor\n", floor / 2^30; exit 1 }
        printf "\n"
    }'
}

health() {
    "$VENV/bin/python" -c 'import sys, urllib.request; sys.stdout.write(urllib.request.urlopen(sys.argv[1], timeout=10).read().decode())' "$HEALTH" 2>/dev/null
}

# The executor's own check (environment.contract_fingerprint): the image must
# bake this checkout's strategy runtime and have been built from its template
# README, or a session would enforce rules the Agent was not told.
image_current() {
    docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "image $IMAGE is absent (tags are lost across stop/start)"; return 1; }
    "$VENV/bin/python" - "$IMAGE" <<'PY'
import sys
from autotrade.environment.contract_fingerprint import (
    SandboxImageContractMismatch, assert_image_contract_current, host_runtime_fingerprint,
)
try:
    assert_image_contract_current(
        sys.argv[1], runtime=host_runtime_fingerprint(), readme="configs/agent_output_template/README.md"
    )
except SandboxImageContractMismatch as exc:
    sys.exit(str(exc))
PY
}

# Before a deploy's first transfer (after `dev bootstrap`): the directories the
# transfers and the steps write into, and data/pretrained as the link to the
# store on the instance's own disk. Experiments pin only the store's manifest digest
# and containers bind the link's target, so nothing else knows where it is. A
# real directory there (a store from before the move) is not touched.
dirs() {
    mkdir -p "$ROOT/data/research_releases" "$ROOT/experiments" "$ROOT/logs/workers" \
        "$ROOT/logs/webui" "$ROOT/logs/zgca" "$ROOT/.runtime/zgca" "$LOCAL/pretrained"
    local link="$ROOT/data/pretrained"
    if [ -L "$link" ]; then
        [ "$(readlink "$link")" = "$LOCAL/pretrained" ] \
            || die "$link links to $(readlink "$link"), not to the store's directory $LOCAL/pretrained; remove the link and deploy again"
    elif [ -e "$link" ]; then
        die "$link is a directory, but the store lives in $LOCAL/pretrained on the instance's own disk: move its models there (ops/zgca/README.md, 存储布局), remove it, and deploy again"
    else
        ln -s "$LOCAL/pretrained" "$link"
    fi
}

# Bytes of the files already in the store's directory and bytes free on its
# disk, for the deploy to decide whether the store fits (the disk is local, so
# walking it is cheap). The sum is printed as a float: mawk's %d stops at
# 2^31 - 1.
local_disk() {
    [ -d "$LOCAL/pretrained" ] || die "no $LOCAL/pretrained: the dirs step creates it"
    echo "$(find "$LOCAL/pretrained" -type f -printf '%s\n' | awk '{ s += $1 } END { printf "%.0f", s }') $(df -B1 --output=avail "$LOCAL" | tail -n 1 | tr -d ' ')"
}

# Keep only tracked files inside the top-level directories that hold them, by
# the NUL-separated list the deploy shipped (what rsync copied from). The
# runtime directories are never entered, symlinks are left alone, and the
# install's own byproducts (__pycache__, *.egg-info) stay; a retired round
# file, which the launcher and the tests would still read, goes.
tracked() {
    cd "$ROOT"
    [ -s .runtime/zgca/tracked.txt ] || die "no tracked file list at $ROOT/.runtime/zgca/tracked.txt: deploy ships it"
    python3 - <<'PY'
from pathlib import Path

tracked = set(Path(".runtime/zgca/tracked.txt").read_text(encoding="utf-8").split("\0")) - {""}
runtime = {"data", "experiments", "logs", "results", ".runtime"}
tops = {entry.split("/", 1)[0] for entry in tracked if "/" in entry} - runtime
removed = []
for top in sorted(tops):
    for path in sorted(Path(top).rglob("*")):
        if path.is_symlink() or not path.is_file() or path.as_posix() in tracked:
            continue
        if "__pycache__" in path.parts or any(part.endswith(".egg-info") for part in path.parts):
            continue
        path.unlink()
        removed.append(path.as_posix())
# A directory left empty is stale too: the skill library refuses an item
# directory without its SKILL.md, so a removed item must not leave a shell.
pruned = 0
for top in sorted(tops):
    for path in sorted(Path(top).rglob("*"), reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
            pruned += 1
print(f"tracked: {len(tracked)} listed; {len(removed)} stale removed, {pruned} empty directories pruned "
      f"under {', '.join(sorted(tops))}" + (": " + ", ".join(removed) if removed else ""))
PY
}

# The experiments whose worker is alive and has not ended -- the console's own
# notion, hitl_state.running_experiments -- one id per line (none: empty output).
# A venv that is absent or has no autotrade installed (before the first
# deploy's install succeeded) runs no console and no worker.
running_experiments() {
    [ -x "$VENV/bin/python" ] || return 0
    "$VENV/bin/python" -c 'import importlib.util, sys
if importlib.util.find_spec("autotrade") is None:
    sys.exit()
from autotrade.pipelines.hitl_state import running_experiments
sys.stdout.writelines(f"{experiment}\n" for experiment in running_experiments(sys.argv[1]))' "$ROOT/experiments"
}

# The first remote step of a deploy and of a down ($1), refused while
# experiments run: a deploy overwrites and prunes the code they import, mirrors
# the seed, release and store they read with --delete and may rebuild their
# image; a down kills them. A round launcher or fill loop still in its tmux
# session could create one meanwhile, so it is refused too (no tmux, as before
# a first `dev bootstrap`, runs no session).
idle() {
    local running session
    running=$(running_experiments | tr '\n' ' ')
    [ -z "$running" ] || die "$1 refused: experiments are running (${running% }), and $1 would change or stop what they run on; wait for them to end (status shows them)"
    for session in "$ROUND_SESSION" "$FILL_SESSION"; do
        ! tmux has-session -t "$session" 2>/dev/null \
            || die "$1 refused: a round launcher is running in tmux session $session and could create an experiment meanwhile; wait for it to end or kill the session"
    done
}

# The legacy builder on the platform's bridge network (no buildx, no host
# networking); --force-rm so a failed step leaves no multi-GB container behind.
# The shipped DuckDB CLI and CUDA compatibility package are served to the build
# from this host, on the bridge gateway only, for the build's duration.
build_image() {
    local log=$1 duckdb_digest=$2 compat_digest=$3 gateway server status base
    echo "$duckdb_digest  $DUCKDB_CLI" | sha256sum -c --quiet \
        || die "$ROOT/$DUCKDB_CLI is missing or is not the release the deploy pinned"
    echo "$compat_digest  $CUDA_COMPAT" | sha256sum -c --quiet \
        || die "$ROOT/$CUDA_COMPAT is missing or is not the package the deploy pinned"
    gateway=$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}')
    [ -n "$gateway" ] || die "the bridge network names no gateway to serve $(dirname "$DUCKDB_CLI") on"
    (cd "$(dirname "$DUCKDB_CLI")" && exec python3 -m http.server "$SERVE_PORT" --bind "$gateway" >/dev/null 2>&1) &
    server=$!
    sleep 1
    kill -0 "$server" 2>/dev/null || die "could not serve $(dirname "$DUCKDB_CLI") on $gateway:$SERVE_PORT"
    base="http://$gateway:$SERVE_PORT"
    echo "building $IMAGE (log $ROOT/$log)"
    DOCKER_BUILDKIT=0 docker build --force-rm -t "$IMAGE" -f ops/docker/sandbox.Dockerfile \
        --build-arg "DUCKDB_CLI_URL=$base/$(basename "$DUCKDB_CLI")" \
        --build-arg "CUDA_COMPAT_URL=$base/$(basename "$CUDA_COMPAT")" . 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    kill "$server" 2>/dev/null
    wait "$server" 2>/dev/null || true
    return "$status"
}

# The instance's own disk is also the image store, and the pretrained store
# takes most of it: it holds one sandbox image and its layer cache, not two. A
# rebuild that reuses the cache (a lost tag, a contract-only change) fits
# beside the old layers; one that rebuilds the pip layer does not and dies of
# ENOSPC after downloading everything. Nothing predicts which it will be, so
# the build is tried as is and only an out-of-disk failure clears the store --
# every unused image and stopped container -- for one cold rebuild. A running experiment runs on the image it
# pinned, so no rebuild is attempted beside it.
rebuild_image() {
    local running log
    running=$(running_experiments | tr '\n' ' ')
    [ -z "$running" ] || die "the sandbox image must be rebuilt, but experiments are running (${running% }) and the disk cannot hold a second image beside the one they run on; wait for them to end"
    log="logs/zgca/image_build_$(date -u +%Y%m%dT%H%M%SZ).log"
    if ! build_image "$log" "$@"; then
        grep -q "No space left on device" "$log" || die "the image build failed (see $ROOT/$log)"
        echo "the build ran out of disk beside the previous image's layers: removing every unused image and stopped container, then building cold"
        docker container prune -f
        docker image prune -af
        df -h --output=target,size,used,avail,pcent / | sed 's/^/disk: /'
        log="logs/zgca/image_build_$(date -u +%Y%m%dT%H%M%SZ).log"
        build_image "$log" "$@" || die "the image build failed again on an empty store (see $ROOT/$log)"
    fi
}

# The image's CUDA must run on this host's driver: torch on one card, run as a
# strategy container runs (read-only, no network, the agent user, a tmpfs
# /tmp), with stdin held open until the container exits because the
# platform's Docker proxy drops the output of a container whose stdin closed.
gpu_check() {
    "$VENV/bin/python" - "$IMAGE" <<'PY' || die "$IMAGE does not run on this host's GPU driver ($(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)); the build installs the CUDA compatibility package $CUDA_COMPAT the deploy pins: check the build log under logs/zgca/, remove the image tag and deploy again"
import subprocess
import sys

check = (
    "import torch, vllm; "
    "assert torch.cuda.is_available(), 'torch finds no usable GPU'; "
    "x = torch.ones(1024, 1024, device='cuda'); "
    "print('GPU check:', torch.cuda.get_device_name(0), 'torch', torch.__version__, "
    "'CUDA', torch.version.cuda, 'vllm', vllm.__version__, 'matmul', float((x @ x).sum()))"
)
argv = ["docker", "run", "-i", "--rm", "--gpus", "device=0", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,exec,size=1g", "--user", "61000", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", sys.argv[1], "python", "-c", check]
sys.exit(subprocess.Popen(argv, stdin=subprocess.PIPE).wait())
PY
}

# The arguments are the pinned digests of the DuckDB CLI and the CUDA
# compatibility package the deploy shipped.
prepare() {
    cd "$ROOT"
    export DEBIAN_FRONTEND=noninteractive
    # The Agent's typed grep tool runs rg on the host; the worker refuses an
    # LLM experiment without it (7fdf506). rsync and tmux come from `dev bootstrap`.
    command -v rg >/dev/null || { echo "installing ripgrep"; apt-get -qq update && apt-get -qq install -y ripgrep; }
    if [ ! -x "$VENV/bin/python" ]; then
        [ -x "$BOX_PYTHON" ] || die "no interpreter at $BOX_PYTHON to create the venv from"
        echo "creating venv $VENV"
        "$BOX_PYTHON" -m venv "$VENV"
    fi
    # On every deploy, so a changed pin reaches the venv; when every pin is
    # already satisfied nothing but the editable install itself changes.
    echo "installing the project into $VENV"
    "$VENV/bin/pip" install -q -e '.[webui]' pytest httpx
    "$VENV/bin/python" -c "import autotrade, sys; sys.exit(0 if autotrade.__file__.startswith('$ROOT/') else 1)" \
        || die "$VENV imports autotrade from outside $ROOT; remove the venv and deploy again"
    docker info >/dev/null 2>&1 || die "Docker is not reachable (DOCKER_HOST=${DOCKER_HOST:-unset})"
    if image_current; then
        echo "sandbox image $IMAGE carries this checkout's strategy contract"
    else
        rebuild_image "$@"
        image_current || die "$IMAGE was rebuilt but does not carry this checkout's strategy contract"
    fi
    docker images --format 'image: {{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' "$IMAGE"
    gpu_check
}

# The console in tmux session CONSOLE_SESSION, answering /api/health. A console
# serves the code it was started with, and `run` comes here before every
# launcher, so with no experiment running an existing console is restarted onto the
# checkout the last deploy shipped before any experiment is created: one left from
# before a deploy validated creations against the old parameter schema and
# refused them. While experiments run it is kept, on the code it was started with.
console_up() {
    cd "$ROOT"
    require_venv
    local started="console started in tmux session $CONSOLE_SESSION (logs/webui/console.log)" running
    if tmux has-session -t "$CONSOLE_SESSION" 2>/dev/null; then
        running=$(running_experiments | tr '\n' ' ')
        if [ -n "$running" ]; then
            started=""
            echo "console kept: experiments are running (${running% }), so it serves the code it was started with until they end"
        else
            tmux kill-session -t "$CONSOLE_SESSION"
            # The old process must let go of the port first, or the health
            # check below would read it instead of the new one.
            for _ in $(seq 1 30); do health >/dev/null || break; sleep 1; done
            ! health >/dev/null || die "the console killed in tmux session $CONSOLE_SESSION still answers at $HEALTH"
            started="console restarted on the current checkout in tmux session $CONSOLE_SESSION (no experiment running)"
        fi
    fi
    if [ -n "$started" ]; then
        tmux new-session -d -s "$CONSOLE_SESSION" -c "$ROOT" \
            "AUTOTRADE_GPU_NAME=$GPU_NAME exec $VENV/bin/python -B scripts/webui/run_webui.py --host 127.0.0.1 --port $PORT >>logs/webui/console.log 2>&1"
        echo "$started"
    fi
    local record
    for _ in $(seq 1 30); do
        if record=$(health); then echo "console health: $record"; return 0; fi
        sleep 2
    done
    die "the console does not answer at $HEALTH; see logs/webui/console.log and tmux session $CONSOLE_SESSION"
}

round() {
    local round_file=$1
    shift
    cd "$ROOT"
    require_venv
    [ -f "$round_file" ] || die "$round_file is not on the instance: deploy first"
    # One `--fill` pass creates what the free slots and cards allow and leaves
    # the rest pending (scripts/experiments/_round.py); nothing on the instance
    # repeats it the way the local cron does, so a fill that is not a dry run
    # loops in FILL_SESSION, holding each pass while the storage floor is not free.
    local session=$ROUND_SESSION fill=0 dry=0 prefix="" arg
    for arg in "$@"; do
        case "$arg" in --fill) fill=1 ;; --dry-run) dry=1 ;; esac
    done
    if [ "$fill$dry" = 10 ]; then session=$FILL_SESSION; prefix=fill_; fi
    ! tmux has-session -t "$session" 2>/dev/null \
        || die "a round launcher is still running in tmux session $session; wait for it (status shows its log) or kill the session"
    health >/dev/null || die "the console does not answer at $HEALTH"
    local log qlog command
    log="logs/zgca/$prefix$(basename "$round_file" .py)_$(date -u +%Y%m%dT%H%M%SZ).log"
    # The log and the command are quoted for the shell tmux runs them in.
    # Unbuffered, so a pass that smokes the starter for an hour shows its
    # progress in the log instead of nothing until it exits.
    printf -v qlog %q "$log"
    printf -v command 'AUTOTRADE_GPU_NAME=%q %q -u %q %q' "$GPU_NAME" "$VENV/bin/python" "$round_file" "$PORT"
    for arg in "$@"; do printf -v arg ' %q' "$arg"; command+=$arg; done
    if [ "$session" = "$FILL_SESSION" ]; then
        # Every pass appends to one log. The loop ends at the first pass that
        # leaves nothing pending, or at the first that exits non-zero: a
        # refused creation must reach the operator, not be retried forever.
        # A pass held by the storage floor runs no launcher and logs one line;
        # the next pass measures again. The session's shell gets storage_check
        # as this script defines it.
        local qstorage
        printf -v qstorage %q "$STORAGE"
        tmux new-session -d -s "$session" -c "$ROOT" \
            "$(declare -f storage_check)
while :; do if held=\$(storage_check $qstorage $QUOTA $FLOOR); then $command >>$qlog 2>&1 || { echo round-exit \$? >>$qlog; break; }; grep '; queue ' $qlog | tail -n 1 | grep -q ', 0 pending (' && { echo round-exit 0 >>$qlog; break; }; else echo \"\$(date -Iseconds) fill held, no experiment is created this pass: \$held\" >>$qlog; fi; sleep $FILL_INTERVAL; done"
        echo "fill launcher started in tmux session $session, one pass every $FILL_INTERVAL s until nothing is pending, each held while less than $((FLOOR >> 30)) GiB is free under the storage quota; log $ROOT/$log"
    else
        tmux new-session -d -s "$session" -c "$ROOT" "$command >$qlog 2>&1; echo round-exit \$? >>$qlog"
        echo "round launcher started in tmux session $session; log $ROOT/$log"
    fi
}

# The project's part of `status`; `dev report` has shown GPUs, disks and tmux,
# but its free space on the storage is the shared file system's, not the quota's.
status() {
    storage_check "$STORAGE" "$QUOTA" "$FLOOR" || true
    cd "$ROOT" 2>/dev/null || { echo "repository: not deployed at $ROOT"; return 0; }
    [ ! -f .runtime/zgca/deployed_commit ] || echo "deployed commit: $(cat .runtime/zgca/deployed_commit)"
    local record
    if [ ! -x "$VENV/bin/python" ]; then
        echo "venv: missing at $VENV (run deploy)"
    elif record=$(health); then
        echo "console: $record"
    else
        echo "console: not answering at $HEALTH"
    fi
    if [ -x "$VENV/bin/python" ]; then
        "$VENV/bin/python" - <<'PY'
import json
import time
from datetime import datetime
from pathlib import Path

from autotrade.pipelines.hitl_state import ENDED_STATES, running_experiments

running = set(running_experiments("experiments"))
rows, ended = [], 0
for status in sorted(Path("experiments").glob("*/hitl/status.json")):
    name = status.parent.parent.name
    try:
        record = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        rows.append(f"{name}: status unreadable ({exc})")
        continue
    state = str(record.get("state") or "")
    if name not in running:
        if state in ENDED_STATES:
            ended += 1
        else:
            rows.append(f"{name}: {state}, but its worker is gone (interrupted)")
        continue
    beat = record.get("heartbeat_at")
    age = f"{time.time() - datetime.fromisoformat(str(beat)).timestamp():.0f} s ago" if beat else "none"
    rows.append(f"{name}: {state}, stage {record.get('environment_stage') or '-'}, heartbeat {age}, pid {record.get('pid')}")
print(f"experiments: {len(running)} running, {ended} ended")
for row in rows:
    print("  " + row)
PY
    fi
    local image
    if image=$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedSince}}' "$IMAGE" 2>/dev/null); then
        echo "image: ${image:-$IMAGE absent (tags are lost across stop/start; deploy rebuilds it)}"
    else
        echo "image: docker unavailable"
    fi
    # The newest launcher log `round` wrote (named after its create_round_*
    # file; image build logs sit beside them), only when it is newer than the
    # newest fill log: an older one is a past round's, not what runs now.
    local latest fill_log
    fill_log=$(ls -t logs/zgca/fill_create_round_*.log 2>/dev/null | head -n 1 || true)
    latest=$(ls -t logs/zgca/create_round_*.log 2>/dev/null | head -n 1 || true)
    if [ -n "$latest" ] && [ "$latest" -nt "$fill_log" ]; then echo "round log: $latest"; tail -n 3 "$latest" | sed 's/^/  /'; fi
    # The fill loop: whether its session runs, and the newest fill log's last
    # pass summary and, once it has ended, how.
    latest=$fill_log
    if tmux has-session -t "$FILL_SESSION" 2>/dev/null; then
        echo "fill: running in tmux session $FILL_SESSION${latest:+; log $latest}"
    else
        echo "fill: no session${latest:+; last log $latest}"
    fi
    if [ -n "$latest" ]; then
        local line pass end
        pass=$(grep '; queue ' "$latest" | tail -n 1) || true
        end=$(grep '^round-exit' "$latest" | tail -n 1) || true
        for line in "$pass" "$end"; do [ -z "$line" ] || echo "  $line"; done
    fi
}

# The commit the last deploy shipped, which `run` compares with the local HEAD.
deployed_commit() {
    [ -s "$ROOT/.runtime/zgca/deployed_commit" ] || die "no deployed commit is recorded at $ROOT/.runtime/zgca/deployed_commit: deploy first"
    cat "$ROOT/.runtime/zgca/deployed_commit"
}

# One experiment's recorded state, and a refusal while it runs: a running experiment
# rewrites its status every few seconds and grows its worker log, so no copy
# of it could pass the transfer's verification.
experiment_state() {
    local experiment=$1 status="$ROOT/experiments/$1/hitl/status.json"
    require_venv
    [ -f "$status" ] || absent "no experiment $experiment on the instance ($status is missing)"
    "$VENV/bin/python" - "$ROOT/experiments" "$experiment" "$RUNNING" <<'PY'
import json
import sys
from pathlib import Path

from autotrade.pipelines.hitl_state import running_experiments

root, experiment = Path(sys.argv[1]), sys.argv[2]
record = json.loads((root / experiment / "hitl" / "status.json").read_text(encoding="utf-8"))
state, verdict = record.get("state"), record.get("verdict") or "-"
print(f"{experiment}: state {state}, verdict {verdict}, sessions {record.get('completed_sessions')}/{record.get('total_sessions')}")
if experiment in running_experiments(root):
    print(
        f"ERROR: {experiment} is still running (worker pid {record.get('pid')}): its status and log change "
        "every few seconds, so a copy cannot be verified; sync it after it ends",
        file=sys.stderr,
    )
    sys.exit(int(sys.argv[3]))
PY
}

# The console's own resume of one experiment (POST .../control, action
# resume), which starts a worker for an experiment whose worker is gone before
# it completed and answers what it did; `resume` has checked the rest. An
# answer that started no worker is a refusal.
resume() {
    require_venv
    "$VENV/bin/python" - "http://127.0.0.1:$PORT/api/experiments/$1/control" <<'PY'
import json
import sys
import urllib.error
import urllib.request

request = urllib.request.Request(
    sys.argv[1], data=json.dumps({"action": "resume"}).encode(), headers={"Content-Type": "application/json"}, method="POST"
)
try:
    with urllib.request.urlopen(request, timeout=300) as response:
        body = response.read().decode("utf-8", "replace")
except urllib.error.HTTPError as exc:
    sys.exit(f"ERROR: the console refused the resume: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')}")
except urllib.error.URLError as exc:
    sys.exit(f"ERROR: the console is unreachable: {exc.reason}")
print(f"console: {body}")
if not json.loads(body).get("spawned"):
    sys.exit("ERROR: the console recorded the resume but started no worker")
PY
}

# Free what a completed experiment no longer needs once `sync` holds its
# verified copy: its PIT views, its sandbox work root and the session
# workspaces its runs archived (artifacts/run_*/workspace, each a copy of a
# session's workspace without its files over 64 MiB), every path built from the
# id and a fixed sub-path.
# Its directory stays with the records, so the fill loop still counts it as
# created and the console still lists it. prune_pit_views.py decides "finished
# for good" and deletes as it does; on top of that only a status of completed
# passes, since a failed, terminated or interrupted experiment resumes in its
# workspace. Nothing is deleted when anything is refused, and a second release
# frees nothing.
release() {
    cd "$ROOT"
    require_venv
    "$VENV/bin/python" - "$1" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "scripts/experiments")
from prune_pit_views import GIB, _delete, derived_trees, freed_bytes, kept_reason

from autotrade.pipelines.hitl_state import HITL_DIR_NAME, STATUS_NAME, ExperimentState, read_status

experiment = sys.argv[1]
if Path(experiment).name != experiment or not experiment.strip("."):
    sys.exit(f"ERROR: not an experiment id: {experiment!r}")
directory = Path("experiments") / experiment
if directory.is_symlink() or not directory.is_dir():
    sys.exit(f"ERROR: no experiment directory experiments/{experiment} on the instance")
state = read_status(directory / HITL_DIR_NAME / STATUS_NAME).get("state")
reason = kept_reason(directory) if state == ExperimentState.COMPLETED else f"state {state!r} is not completed"
if reason:
    sys.exit(f"ERROR: {experiment} is kept whole, {reason}: a resume continues in its workspace")
trees = derived_trees(Path("."), directory)
artifacts = directory / "artifacts"
runs = sorted(artifacts.glob("run_*")) if artifacts.is_dir() else []
for link in (artifacts, *runs, *(run / "workspace" for run in runs)):
    if link.is_symlink():
        sys.exit(f"ERROR: {link} is a symbolic link; nothing of {experiment} was released")
trees.update((f"{run.name}/workspace", run / "workspace") for run in runs if (run / "workspace").is_dir())
sizes = {label: freed_bytes(path) for label, path in trees.items()}
for label, path in trees.items():
    _delete(label, path)
detail = ", ".join(f"{label} {size / GIB:.2f} GiB" for label, size in sizes.items())
print(f"{experiment}: released on the instance, freed {sum(sizes.values()) / GIB:.2f} GiB"
      + (f" ({detail})" if detail else ", nothing was left") + "; its records stay")
PY
}

# The strategy runtime digest an image on the instance bakes
# (environment.contract_fingerprint), read in one offline container without a
# GPU: `sync` compares the one a synced experiment's pinned image bakes with
# this host's base image before giving the experiment its tag there.
image_runtime() {
    require_venv
    docker image inspect "$1" >/dev/null 2>&1 \
        || absent "image $1 is not on the instance (tags are lost across stop/start), so the strategy runtime it baked cannot be read"
    "$VENV/bin/python" - "$1" <<'PY'
import sys

from autotrade.environment.contract_fingerprint import SandboxImageContractMismatch, read_image_contract

try:
    print(read_image_contract(sys.argv[1]).runtime)
except SandboxImageContractMismatch as exc:
    sys.exit(f"ERROR: {exc}")
PY
}

step=$1
shift
case "$step" in
    idle) [ $# -eq 1 ] || die "idle needs the command it guards"; idle "$1" ;;
    dirs) dirs ;;
    tracked) tracked ;;
    local-disk) local_disk ;;
    prepare) [ $# -eq 2 ] || die "prepare needs the pinned DuckDB CLI and CUDA compatibility package digests"; prepare "$1" "$2" ;;
    console-up) console_up ;;
    round) [ $# -ge 1 ] || die "round needs the round file"; round "$@" ;;
    status) status ;;
    storage) storage_check "$STORAGE" "$QUOTA" "$FLOOR" ;;
    deployed-commit) deployed_commit ;;
    experiment-state) [ $# -eq 1 ] || die "experiment-state needs one experiment id"; experiment_state "$1" ;;
    resume) [ $# -eq 1 ] || die "resume needs one experiment id"; resume "$1" ;;
    release) [ $# -eq 1 ] || die "release needs one experiment id"; release "$1" ;;
    image-runtime) [ $# -eq 1 ] || die "image-runtime needs one image tag"; image_runtime "$1" ;;
    *) die "unknown remote step: '$step'" ;;
esac
