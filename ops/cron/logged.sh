#!/usr/bin/env bash
# One cron job with its output in log files instead of cron's mail:
#
#   ops/cron/logged.sh <log> <command> [args...]
#
# The command's stdout and stderr are appended to <log>_<YYYYMMDD>.log, one
# file per job and day (the day the run starts, in the host's time zone), and
# each run ends there with one dated line "<name> exited <status>", <name>
# being the last component of <log>. A run first removes the files of <log>
# untouched for 14 days. Nothing reaches cron, so cron mails nothing, unless
# the run fails: a non-zero exit status that RETRY_EXIT_STATUSES (the block's
# space-separated planned retry statuses) does not list also writes that line,
# with the file's path, to stderr, the one line cron mails. The command's own
# exit status is passed through. Lines are stamped as `date -Iseconds` stamps.
set -u
[ $# -ge 2 ] || { echo "usage: $0 <log> <command> [args...]" >&2; exit 2; }
dir=$(dirname "$1") name=$(basename "$1")
shift
mkdir -p "$dir" || exit 1
find "$dir" -maxdepth 1 -name "${name}_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9].log" -mmin +20160 -delete
log="$dir/${name}_$(date +%Y%m%d).log"
"$@" >> "$log" 2>&1
status=$?
line="$(date -Iseconds) $name exited $status"
echo "$line" >> "$log"
if [ "$status" -ne 0 ] && [[ " ${RETRY_EXIT_STATUSES:-} " != *" $status "* ]]; then
    echo "$line; output in $log" >&2
fi
exit "$status"
