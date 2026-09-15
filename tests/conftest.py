"""Pin THIS repo's src ahead of any sibling editable install.

A sibling checkout pip-installed editable into the shared environment sorts
its .pth entry before ours, so bare `pytest` imported the sibling's
`autotrade` and 90 tests failed against foreign code. Tests must always run
against the tree they live in.
"""

import os
import stat
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
while _SRC in sys.path:
    sys.path.remove(_SRC)
sys.path.insert(0, _SRC)


def _restore_writable_tree(root: Path) -> None:
    """Add u+rwX as we descend so 000 / 0555 publish cannot block pytest
    basetemp retention or TemporaryDirectory cleanup."""
    try:
        info = root.lstat()
    except OSError:
        return
    if stat.S_ISLNK(info.st_mode):
        return
    try:
        os.chmod(root, stat.S_IMODE(info.st_mode) | stat.S_IRWXU)
    except OSError:
        return
    if not stat.S_ISDIR(info.st_mode):
        return
    try:
        with os.scandir(root) as entries:
            children = [Path(entry.path) for entry in entries]
    except OSError:
        return
    for child in children:
        _restore_writable_tree(child)


@pytest.fixture(autouse=True)
def _restore_tmp_path_writable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    nested = tmp_path / "tmp"
    nested.mkdir()
    monkeypatch.setenv("TMPDIR", str(nested))
    monkeypatch.setenv("TEMP", str(nested))
    monkeypatch.setenv("TMP", str(nested))
    yield
    _restore_writable_tree(tmp_path)


def pytest_sessionfinish(session, exitstatus):
    # Only this session's basetemp: the retained ``pytest-of-<user>`` root is
    # shared with every other pytest process of the same user, and walking it
    # would lift the 0600/0444 modes a concurrent session is asserting on.
    factory = getattr(session.config, "_tmp_path_factory", None)
    if factory is None:
        return
    root = Path(factory.getbasetemp())
    if root.is_dir():
        _restore_writable_tree(root)
