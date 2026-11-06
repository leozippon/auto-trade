"""The shared primitives of ``autotrade.common``."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path

import pytest

from autotrade.common.ids import is_experiment_id
from autotrade.common.records import (
    dataclass_from_record,
    dataclass_record,
    read_json_object,
    read_json_object_or_empty,
    write_json_atomic,
)
from autotrade.common.time import CN_TZ
from autotrade.environment import strategy


def test_the_sandbox_contract_keeps_the_same_market_timezone() -> None:
    assert strategy.CN_TZ == CN_TZ


def test_a_record_file_reads_back_what_was_written_and_a_private_one_is_owner_only(tmp_path: Path) -> None:
    target = tmp_path / "new" / "record.json"
    write_json_atomic(target, {"b": 1, "a": ["x", 2]}, private=True)
    assert read_json_object(target) == {"a": ["x", 2], "b": 1}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not [path for path in target.parent.iterdir() if path.name.endswith(".tmp")]
    with pytest.raises(ValueError):
        write_json_atomic(target, {"x": float("nan")})


def test_a_missing_or_malformed_record_fails_unless_the_reader_tolerates_it(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError):
        read_json_object(missing)
    assert read_json_object(missing, missing_ok=True) == {}
    for name, text in (("broken.json", "{broken"), ("list.json", "[1, 2]")):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=name):
            read_json_object(path, missing_ok=True)
        assert read_json_object_or_empty(path) == {}


def test_a_record_the_host_cannot_read_is_no_malformed_record(tmp_path: Path) -> None:
    """Permission denied (or any I/O error) is the OSError it is, missing_ok
    or not: the file may hold a valid record, so a reader that tolerates
    malformed records must not take it for one."""

    path = tmp_path / "locked.json"
    write_json_atomic(path, {"state": "running_session"})
    path.chmod(0)
    try:
        for missing_ok in (False, True):
            with pytest.raises(PermissionError):
                read_json_object(path, missing_ok=missing_ok)
    finally:
        path.chmod(0o600)


@dataclass(frozen=True)
class _Pinned:
    name: str
    boards: tuple[str, ...] = ()
    limit: float | None = None


def test_a_dataclass_record_round_trips_and_refuses_another_key_set() -> None:
    value = _Pinned("a", ("main", "gem"), 1.5)
    record = dataclass_record(value)
    assert record == {"name": "a", "boards": ["main", "gem"], "limit": 1.5}
    assert dataclass_from_record(_Pinned, record) == value
    with pytest.raises(ValueError, match="missing keys \\['limit'\\]"):
        dataclass_from_record(_Pinned, {"name": "a", "boards": []})
    with pytest.raises(ValueError, match="unknown keys \\['extra'\\]"):
        dataclass_from_record(_Pinned, {**record, "extra": 1})


@pytest.mark.parametrize("value", ["a", "experiment_1-x", "9" * 100])
def test_an_experiment_id_is_a_safe_directory_name(value: str) -> None:
    assert is_experiment_id(value)


@pytest.mark.parametrize("value", ["", "-experiment", "_experiment", "a/b", "a b", "a.b", "9" * 101, None, 1])
def test_anything_else_is_not_an_experiment_id(value: object) -> None:
    assert not is_experiment_id(value)
