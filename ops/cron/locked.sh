#!/usr/bin/env bash
# One cron job under its lock and a time limit, leaving a dated line in the
# job's log for every outcome the job itself cannot report:
#
#   ops/cron/locked.sh <lock file> <limit> <label> <command> [args...]
#
# A run that finds the lock held runs nothing, writes "<label> skipped: the
# previous run still holds <lock file>" and exits 0. Otherwise the command runs
# under `timeout <limit>` (timeout(1)'s duration, e.g. 4h), in a process group
# of its own that inherits the lock; a run the limit ends writes "<label>
# stopped: the run reached its time limit" and exits 124, and any other run
# passes the command's own output and exit status through. At the limit TERM
# reaches the whole group, and whatever of the group is still there GRACE
# seconds after the command's own process ended is killed, so nothing the run
# started outlives it or keeps the lock. The command's own process must end on
# TERM, as python and bash do. Each line is stamped as `date -Iseconds` stamps.
set -u
[ $# -ge 4 ] || { echo "usage: $0 <lock file> <limit> <label> <command> [args...]" >&2; exit 2; }
lock=$1 limit=$2 label=$3
shift 3
GRACE=10
exec 9>>"$lock" || exit 1
flock -n -E 75 9 || {
    status=$?
    [ "$status" -ne 75 ] || { echo "$(date -Iseconds) $label skipped: the previous run still holds $lock"; exit 0; }
    exit "$status"
}
# timeout makes itself the leader of a new process group, so its pid names the group.
timeout "$limit" "$@" &
group=$!
wait "$group"
status=$?
for _ in $(seq "$GRACE"); do kill -0 -- "-$group" 2>/dev/null || break; sleep 1; done
kill -KILL -- "-$group" 2>/dev/null
[ "$status" -ne 124 ] || echo "$(date -Iseconds) $label stopped: the run reached its time limit"
exit "$status"
