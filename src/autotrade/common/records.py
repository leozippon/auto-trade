"""JSON record files, host-path redaction, and the strict dataclass record pair.

One reader and one atomic writer serve every JSON record file the host keeps;
a module that owes its callers another error type translates the reader's
``ValueError`` at its own boundary. Redaction scrubs host paths out of text
that leaves the host process. A frozen dataclass whose fields are JSON scalars,
tuples, string-keyed mappings or such dataclasses persists as
``dataclass_record`` and comes back through ``dataclass_from_record``, which
refuses a missing or unknown key (a nested dataclass is rebuilt by its owner's
``from_record`` and handed in).
"""

from __future__ import annotations

import json
import os
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def read_json_object(path: str | Path, *, missing_ok: bool = False) -> dict[str, object]:
    """The JSON object stored at ``path``.

    A missing file raises ``FileNotFoundError``, or reads as ``{}`` when
    ``missing_ok``. Text that is not UTF-8 JSON and a payload that is not an
    object raise ``ValueError`` naming the path. Any other ``OSError``
    (permission denied, an I/O error) propagates as it is: the file may hold
    a valid record the host could not read, which is no malformed record.
    """

    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        if missing_ok:
            return {}
        raise
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"unreadable JSON record {target}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON record is not an object: {target}")
    return value


def read_json_object_or_empty(path: str | Path) -> dict[str, object]:
    """``read_json_object`` for a reader that must not fail on one file: a
    missing or unreadable file reads as ``{}``."""

    try:
        return read_json_object(path, missing_ok=True)
    except (OSError, ValueError):
        return {}


def write_json_atomic(
    path: str | Path,
    payload: object,
    *,
    sort_keys: bool = True,
    private: bool = False,
    default: Callable[[object], object] | None = str,
) -> None:
    """Publish one JSON file atomically.

    ``sort_keys`` keeps a record diffable; pass False for a file whose builder
    orders keys deliberately (an index the Agent reads in chunks needs its
    identifying keys first). ``private`` makes the file owner-only and creates
    missing parents owner-only. ``default`` renders a value JSON has no type
    for (``str``: paths, timestamps); None refuses it with ``TypeError``, for
    a record whose every value must already be JSON. NaN is refused: in a
    record it is an upstream bug.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o777)
    # A unique temp name: concurrent writers must never share one, or
    # interleaved chunks are replaced into place.
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                sort_keys=sort_keys,
                default=default,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if private:
            temporary.chmod(0o600)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


# Host-path redaction. A match has to look like a path: a top-level directory
# this host keeps its files under, then at least one more segment of ordinary
# path characters. Redacting every ``/`` at a token boundary instead rewrote the
# transcript the Agent reads its own work back from: 464 substitutions in one
# audited arm, nearly all of them division, backquoted names, ``2>/dev/null``
# and ``<domain>/part_0000.parquet``.
HOST_PATH_MARKER = "[host_path]"


def _host_path_prefixes() -> tuple[str, ...]:
    """The top-level directories an absolute host path can start with.

    The home directories plus the top of this checkout: the data, snapshot,
    experiment and sandbox work roots all live inside the repository, so one
    prefix covers every host tree a session could name. ``/Data2`` is named
    outright because that is where the worker runs from, and the derived one
    keeps this correct for a checkout elsewhere. ``/mnt``, ``/tmp``, ``/opt``
    and ``/usr`` are the Agent's view of its own sandbox, never host paths, so
    they can never become a prefix.
    """

    tops = {"/Data2", "/home", "/root"}
    repo = Path(__file__).resolve().parents[3]
    if len(repo.parts) > 1:
        tops.add("/" + repo.parts[1])
    return tuple(sorted(tops - {"/mnt", "/tmp", "/opt", "/usr"}))


HOST_PATH_PREFIXES = _host_path_prefixes()
HOST_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.\-/])(?:"
    + "|".join(re.escape(prefix) for prefix in HOST_PATH_PREFIXES)
    + r")(?:/[A-Za-z0-9._\-+@~%]+)+"
)


def redact_host_paths(text: str) -> str:
    return HOST_PATH_RE.sub(HOST_PATH_MARKER, text)


def dataclass_record(instance: object) -> dict[str, object]:
    """Every init field of a dataclass by name, tuples as lists and a nested
    dataclass as its own record: the JSON record ``dataclass_from_record``
    inverts."""

    if not is_dataclass(instance) or isinstance(instance, type):
        raise TypeError(f"not a dataclass instance: {instance!r}")
    return {item.name: _json_value(getattr(instance, item.name)) for item in fields(instance) if item.init}


def dataclass_from_record(cls: type[T], record: object) -> T:
    """``cls`` rebuilt from its ``dataclass_record``; a missing or unknown key
    fails, so a record written under other fields never loads as defaults."""

    if not isinstance(record, Mapping):
        raise TypeError(f"{cls.__name__} record must be a JSON object")
    names = [item.name for item in fields(cls) if item.init]  # type: ignore[arg-type]
    missing = sorted(set(names) - set(record))
    unknown = sorted(set(record) - set(names))
    if missing or unknown:
        raise ValueError(f"{cls.__name__} record has missing keys {missing} and unknown keys {unknown}")
    return cls(**{name: _from_json(record[name]) for name in names})


def _json_value(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return dataclass_record(value)
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _from_json(value: object) -> object:
    # JSON arrays come back as the tuples a frozen dataclass holds.
    if isinstance(value, list):
        return tuple(_from_json(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _from_json(item) for key, item in value.items()}
    return value


__all__ = [
    "HOST_PATH_MARKER",
    "HOST_PATH_PREFIXES",
    "HOST_PATH_RE",
    "dataclass_from_record",
    "dataclass_record",
    "read_json_object",
    "read_json_object_or_empty",
    "redact_host_paths",
    "write_json_atomic",
]
