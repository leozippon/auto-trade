"""The updater lock: the one exclusive kernel flock every lake writer holds,
and the proof a child process gives that it runs under the runner's hold.

The cron runner (and the operator commands of lake.py) take the lock with
:func:`acquire_lock`; the runner passes its fd to its children, which prove
the inheritance with :func:`inherited_updater_lock_fd`. A manual command
against the production lake is refused unless it can
(:func:`manual_production_write`).
"""

from __future__ import annotations

import fcntl
import os
import time
from dataclasses import dataclass
from fcntl import LOCK_NB, LOCK_SH, LOCK_UN, flock
from pathlib import Path

from autotrade.common.time import utc_now
from autotrade.environment.data.contracts import updater_lock_path


@dataclass
class FileLock:
    """A held kernel flock; the file itself is never deleted (unlinking a
    flock-backed lock file races a concurrent opener onto a dead inode)."""

    path: Path
    fd: int

    def release(self) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)


def acquire_lock(repo_root: Path, wait_seconds: int) -> FileLock:
    """Exclusive kernel flock on the updater lock of ``repo_root``: released
    automatically when the holder exits, so a crashed/killed run can never
    leave a permanently stale lock (the old PID-file scheme broke on PID
    reuse). pid/started_at are diagnostics only."""
    lock = updater_lock_path(repo_root)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR)
    deadline = time.monotonic() + max(0, wait_seconds)
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                os.close(fd)
                raise RuntimeError(
                    f"lock is held after waiting {wait_seconds}s, another run may be active: {lock}"
                ) from None
            time.sleep(min(15.0, remaining))
    os.ftruncate(fd, 0)
    os.write(fd, f"pid={os.getpid()}\nstarted_at={utc_now()}\n".encode())
    return FileLock(lock, fd)


UPDATER_LOCK_FD_ENV = "TUSHARE_UPDATE_LOCK_FD"


def inherited_updater_lock_fd(repo_root: Path) -> int | None:
    """The runner's updater-lock fd this process inherited, or None.

    The cron runner holds the exclusive updater flock and passes that very file
    descriptor to its children (``pass_fds``) plus its number in
    ``TUSHARE_UPDATE_LOCK_FD``. A child proves it really is a runner child by
    showing all three:

      1. the named fd is open in THIS process,
      2. it refers to the same file as the updater lock path (dev+ino), and
      3. that lock is currently held exclusively (a fresh shared, non-blocking
         probe fails).

    An environment variable alone proves nothing, and neither does "somebody
    holds the lock" -- a manual run could hold it itself. Honest boundary: an
    operator who deliberately opens the lock file, flocks it exclusively and
    exports the fd number has reimplemented the runner; this is a discipline
    boundary against accident and drift, not a security boundary against a
    determined local user.
    """
    raw = os.environ.get(UPDATER_LOCK_FD_ENV, "")
    if not raw.isdigit():
        return None
    fd = int(raw)
    lock_path = updater_lock_path(repo_root)
    try:
        inherited = os.fstat(fd)
        on_disk = os.stat(str(lock_path))
    except OSError:
        return None
    if (inherited.st_dev, inherited.st_ino) != (on_disk.st_dev, on_disk.st_ino):
        return None
    try:
        probe = os.open(str(lock_path), os.O_RDONLY)
    except OSError:
        return None
    try:
        flock(probe, LOCK_SH | LOCK_NB)
    except BlockingIOError:
        return fd  # exclusively held: the runner's lock is live
    else:
        flock(probe, LOCK_UN)
        return None
    finally:
        os.close(probe)


def manual_production_write(repo_root: Path, protected_dir: Path, target: object) -> bool:
    """Whether writing ``target`` would change ``protected_dir`` (or anything
    under it) from outside the cron runner.

    The runner is the only production write entrance: it alone provides the
    generation record, per-job state, failure alerting and the exclusive
    flock. Only a genuine runner child -- proven by the inherited lock fd, not
    by an environment flag -- may write there; an empty ``target`` writes
    nothing."""
    if not target or inherited_updater_lock_fd(repo_root) is not None:
        return False
    try:
        resolved = Path(str(target)).resolve()
        protected = protected_dir.resolve()
    except OSError:
        return False
    return resolved == protected or protected in resolved.parents
