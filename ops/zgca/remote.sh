#!/usr/bin/env bash
# The instance-side half of ops/zgca/zgca.py: one step per call, sent by the
# workflow as `bash -lc "$(cat remote.sh)" remote.sh <step> [args]` so the
# platform's login environment (DOCKER_HOST) is present and the script is
# always the local checkout's, never a stale copy on the instance. Runs as the
# instance's root; everything persistent lives on /personal_fast_storage.
set -euo pipefail

ROOT=/personal_fast_storage/auto-trade
VENV=/personal_fast_storage/envs/quant
BOX_PYTHON=/root/miniconda3/bin/python
# The platform's PyPI mirror lacks tushare 1.4.29 (~/zgca-compute/NOTES.md).
EXTRA_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
IMAGE=autotrade-sandbox:latest
# Shipped by the deploy with the digests it pinned; GitHub and its release
# proxies are not reliably reachable from the instance, so the build reads the
# DuckDB CLI and the causal-conv1d wheel from this host over the bridge
# network instead.
DUCKDB_CLI=.runtime/zgca/duckdb_cli-linux-amd64.zip
CAUSAL_CONV1D_WHEEL=.runtime/zgca/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
SERVE_PORT=8765
PORT=38888
GPU_NAME=A100
CONSOLE_SESSION=autotrade-console
ROUND_SESSION=autotrade-round
HEALTH="http://127.0.0.1:$PORT/api/health"

die() { echo "ERROR: $*" >&2; exit 1; }

require_venv() { [ -x "$VENV/bin/python" ] || die "no venv at $VENV: run deploy first"; }

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

dirs() {
    mkdir -p "$ROOT/data/research_releases" "$ROOT/experiments" "$ROOT/logs/workers" \
        "$ROOT/logs/webui" "$ROOT/logs/zgca" "$ROOT/.runtime/zgca"
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
print(f"tracked: {len(tracked)} listed; {len(removed)} stale removed under {', '.join(sorted(tops))}"
      + (": " + ", ".join(removed) if removed else ""))
PY
}

# The arms whose worker is alive and has not ended -- the console's own
# notion, hitl_state.running_arms -- one id per line (none: empty output).
running_arms() {
    "$VENV/bin/python" -c 'import sys
from autotrade.pipelines.hitl_state import running_arms
sys.stdout.writelines(f"{arm}\n" for arm in running_arms(sys.argv[1]))' "$ROOT/experiments"
}

# The legacy builder on the platform's bridge network (no buildx, no host
# networking); --force-rm so a failed step leaves no multi-GB container behind.
# The shipped DuckDB CLI and causal-conv1d wheel are served to the build from
# this host, on the bridge gateway only, for the build's duration.
build_image() {
    local log=$1 duckdb_digest=$2 wheel_digest=$3 gateway server status base
    echo "$duckdb_digest  $DUCKDB_CLI" | sha256sum -c --quiet \
        || die "$ROOT/$DUCKDB_CLI is missing or is not the release the deploy pinned"
    echo "$wheel_digest  $CAUSAL_CONV1D_WHEEL" | sha256sum -c --quiet \
        || die "$ROOT/$CAUSAL_CONV1D_WHEEL is missing or is not the release the deploy pinned"
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
        --build-arg "CAUSAL_CONV1D_WHEEL_URL=$base/$(basename "$CAUSAL_CONV1D_WHEEL")" . 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    kill "$server" 2>/dev/null
    wait "$server" 2>/dev/null || true
    return "$status"
}

# The 20 GB root disk is also the image store: it holds one sandbox image and
# its layer cache, not two. A rebuild that reuses the cache (a lost tag, a
# contract-only change) fits beside the old layers; one that rebuilds the pip
# layer does not and dies of ENOSPC after downloading everything. Nothing
# predicts which it will be, so the build is tried as is and only an
# out-of-disk failure clears the store -- every unused image and stopped
# container -- for one cold rebuild. A running arm runs on the image it
# pinned, so no rebuild is attempted beside it.
rebuild_image() {
    local running log
    running=$(running_arms | tr '\n' ' ')
    [ -z "$running" ] || die "the sandbox image must be rebuilt, but arms are running (${running% }) and the disk cannot hold a second image beside the one they run on; wait for them to end"
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

# The two arguments are the pinned digests of the DuckDB CLI and the
# causal-conv1d wheel the deploy shipped.
prepare() {
    cd "$ROOT"
    export DEBIAN_FRONTEND=noninteractive
    # The Agent's typed grep tool runs rg on the host; the worker refuses an
    # LLM arm without it (7fdf506).
    command -v rg >/dev/null || { echo "installing ripgrep"; apt-get -qq update && apt-get -qq install -y ripgrep; }
    command -v tmux >/dev/null || { echo "installing tmux"; apt-get -qq update && apt-get -qq install -y tmux; }
    if ! "$VENV/bin/python" -c 'import autotrade, fastapi, uvicorn, tushare' >/dev/null 2>&1; then
        [ -x "$BOX_PYTHON" ] || die "no interpreter at $BOX_PYTHON to create the venv from"
        [ -x "$VENV/bin/python" ] || { echo "creating venv $VENV"; "$BOX_PYTHON" -m venv "$VENV"; }
        echo "installing the project into $VENV"
        "$VENV/bin/pip" install -q --extra-index-url "$EXTRA_INDEX" -e '.[webui]' pytest httpx
    fi
    "$VENV/bin/python" -c "import autotrade, sys; sys.exit(0 if autotrade.__file__.startswith('$ROOT/') else 1)" \
        || die "$VENV imports autotrade from outside $ROOT; remove the venv and deploy again"
    docker info >/dev/null 2>&1 || die "Docker is not reachable (DOCKER_HOST=${DOCKER_HOST:-unset})"
    df -h --output=target,size,used,avail,pcent / "$(dirname "$ROOT")" | sed 's/^/disk: /'
    if image_current; then
        echo "sandbox image $IMAGE carries this checkout's strategy contract"
    else
        rebuild_image "$@"
        image_current || die "$IMAGE was rebuilt but does not carry this checkout's strategy contract"
    fi
    docker images --format 'image: {{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}' "$IMAGE"
    df -h --output=target,size,used,avail,pcent / | sed 's/^/disk: /'
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | sed 's/^/gpu: /'
}

console_up() {
    cd "$ROOT"
    require_venv
    if ! tmux has-session -t "$CONSOLE_SESSION" 2>/dev/null; then
        tmux new-session -d -s "$CONSOLE_SESSION" -c "$ROOT" \
            "AUTOTRADE_GPU_NAME=$GPU_NAME exec $VENV/bin/python -B scripts/webui/run_webui.py --host 127.0.0.1 --port $PORT >>logs/webui/console.log 2>&1"
        echo "console started in tmux session $CONSOLE_SESSION (logs/webui/console.log)"
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
    ! tmux has-session -t "$ROUND_SESSION" 2>/dev/null \
        || die "a round launcher is still running in tmux session $ROUND_SESSION; wait for it (status shows its log) or kill the session"
    health >/dev/null || die "the console does not answer at $HEALTH"
    local log command arg
    log="logs/zgca/$(basename "$round_file" .py)_$(date -u +%Y%m%dT%H%M%SZ).log"
    printf -v command 'AUTOTRADE_GPU_NAME=%q %q %q %q' "$GPU_NAME" "$VENV/bin/python" "$round_file" "$PORT"
    for arg in "$@"; do printf -v arg ' %q' "$arg"; command+=$arg; done
    tmux new-session -d -s "$ROUND_SESSION" -c "$ROOT" "$command >$log 2>&1; echo round-exit \$? >>$log"
    echo "round launcher started in tmux session $ROUND_SESSION; log $ROOT/$log"
}

status() {
    cd "$ROOT" 2>/dev/null || { echo "repository: not deployed at $ROOT"; return 0; }
    echo "tmux sessions: $(tmux ls -F '#{session_name}' 2>/dev/null | tr '\n' ' ')"
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

from autotrade.pipelines.hitl_state import ENDED_STATES, running_arms

running = set(running_arms("experiments"))
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
print(f"arms: {len(running)} running, {ended} ended")
for row in rows:
    print("  " + row)
PY
    fi
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/gpu: /'
    df -h --output=target,size,used,avail,pcent / "$(dirname "$ROOT")" | sed 's/^/disk: /'
    local image
    if image=$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedSince}}' "$IMAGE" 2>/dev/null); then
        echo "image: ${image:-$IMAGE absent (tags are lost across stop/start; deploy rebuilds it)}"
    else
        echo "image: docker unavailable"
    fi
    local latest
    latest=$(ls -t logs/zgca/*.log 2>/dev/null | head -n 1 || true)
    if [ -n "$latest" ]; then echo "round log: $latest"; tail -n 3 "$latest" | sed 's/^/  /'; fi
}

# One arm's recorded state, and a refusal while it runs: a running arm
# rewrites its status every few seconds and grows its worker log, so no copy
# of it could pass the transfer's verification.
arm_state() {
    local arm=$1 status="$ROOT/experiments/$1/hitl/status.json"
    require_venv
    [ -f "$status" ] || die "no arm $arm on the instance ($status is missing)"
    "$VENV/bin/python" - "$ROOT/experiments" "$arm" <<'PY'
import json
import sys
from pathlib import Path

from autotrade.pipelines.hitl_state import running_arms

root, arm = Path(sys.argv[1]), sys.argv[2]
record = json.loads((root / arm / "hitl" / "status.json").read_text(encoding="utf-8"))
state, verdict = record.get("state"), record.get("verdict") or "-"
print(f"{arm}: state {state}, verdict {verdict}, sessions {record.get('completed_sessions')}/{record.get('total_sessions')}")
if arm in running_arms(root):
    sys.exit(
        f"ERROR: {arm} is still running (worker pid {record.get('pid')}): its status and log change "
        "every few seconds, so a copy cannot be verified; sync it after it ends"
    )
PY
}

step=${1:-}
[ $# -eq 0 ] || shift
case "$step" in
    dirs) dirs ;;
    tracked) tracked ;;
    prepare) [ $# -eq 2 ] || die "prepare needs the pinned DuckDB CLI and causal-conv1d wheel digests"; prepare "$1" "$2" ;;
    console-up) console_up ;;
    round) [ $# -ge 1 ] || die "round needs the round file"; round "$@" ;;
    status) status ;;
    arm-state) [ $# -eq 1 ] || die "arm-state needs one arm id"; arm_state "$1" ;;
    *) die "unknown remote step: '$step'" ;;
esac
