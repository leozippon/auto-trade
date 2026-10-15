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
        "$ROOT/logs/webui" "$ROOT/logs/zgca" "$ROOT/.runtime"
}

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
        # No buildx and no host networking on the platform: the legacy
        # builder over the default bridge; a cached rebuild takes ~15 s.
        echo "building $IMAGE"
        DOCKER_BUILDKIT=0 docker build -t "$IMAGE" -f ops/docker/sandbox.Dockerfile .
        image_current || die "$IMAGE was rebuilt but does not carry this checkout's strategy contract"
    fi
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

active = {"launching", "initializing", "running_session", "paused"}
rows, ended = [], 0
for status in sorted(Path("experiments").glob("*/hitl/status.json")):
    name = status.parent.parent.name
    try:
        record = json.loads(status.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        rows.append(f"{name}: status unreadable ({exc})")
        continue
    state = str(record.get("state") or "")
    if state not in active:
        ended += 1
        continue
    beat = record.get("heartbeat_at")
    age = f"{time.time() - datetime.fromisoformat(str(beat)).timestamp():.0f} s ago" if beat else "none"
    rows.append(f"{name}: {state}, stage {record.get('environment_stage') or '-'}, heartbeat {age}, pid {record.get('pid')}")
print(f"arms: {len(rows)} active, {ended} ended")
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

arm_state() {
    local arm=$1 status="$ROOT/experiments/$1/hitl/status.json"
    require_venv
    [ -f "$status" ] || die "no arm $arm on the instance ($status is missing)"
    "$VENV/bin/python" - "$status" "$arm" <<'PY'
import json
import sys

record = json.load(open(sys.argv[1], encoding="utf-8"))
state, verdict = record.get("state"), record.get("verdict") or "-"
print(f"{sys.argv[2]}: state {state}, verdict {verdict}, sessions {record.get('completed_sessions')}/{record.get('total_sessions')}")
PY
}

step=${1:-}
[ $# -eq 0 ] || shift
case "$step" in
    dirs) dirs ;;
    prepare) prepare ;;
    console-up) console_up ;;
    round) [ $# -ge 1 ] || die "round needs the round file"; round "$@" ;;
    status) status ;;
    arm-state) [ $# -eq 1 ] || die "arm-state needs one arm id"; arm_state "$1" ;;
    *) die "unknown remote step: '$step'" ;;
esac
