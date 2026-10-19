"""The pretrained-model store: the only pretrained weights a strategy or session may load.

The host keeps Hugging Face model directories under ``data/pretrained/`` beside
a ``MANIFEST.sha256`` that lists every file as ``<sha256>  <model>/<file>``
(``sha256sum -c`` verifies it). An arm pins the digest of that manifest file
when it is created (the experiment parameter ``pretrained_manifest_sha256``,
stamped by the console), and every container that mounts the store -- each
strategy container, research and Paper alike, and the Agent's session
container -- first compares the store's manifest with the pin and refuses to
start on a missing store or a different one. An arm or a Paper book without
the pin mounts nothing.

This is pin-and-compare only: hashing the weights themselves is the one-off
``sha256sum -c MANIFEST.sha256`` after the store is copied or deployed, not a
cost every container start pays.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# The store this checkout's processes mount; tests point it at a fixture.
PRETRAINED_STORE = Path(__file__).resolve().parents[3] / "data" / "pretrained"
MANIFEST_NAME = "MANIFEST.sha256"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class PretrainedStoreError(RuntimeError):
    """The pretrained store an arm pinned is absent or is a different store."""


def pretrained_pin(value: object) -> str:
    """A ``pretrained_manifest_sha256`` value: empty (no store) or a SHA-256 hex digest."""

    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(
            "pretrained_manifest_sha256 must be empty or a lowercase SHA-256 hex digest"
        )
    return value


def manifest_digest(store: Path | None = None) -> str:
    """The SHA-256 of the store's ``MANIFEST.sha256``, the value an arm pins."""

    root = PRETRAINED_STORE if store is None else Path(store)
    try:
        payload = (root / MANIFEST_NAME).read_bytes()
    except OSError as exc:
        raise PretrainedStoreError(
            f"the pretrained store at {root} has no readable {MANIFEST_NAME} ({exc}); "
            "copy the store there and verify it with `sha256sum -c` "
            "(docs/deployment-documentation.md)"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def require_pretrained_store(pin: str, store: Path | None = None) -> Path:
    """The store's resolved directory, when its manifest is the one ``pin`` names."""

    root = (PRETRAINED_STORE if store is None else Path(store)).resolve()
    digest = manifest_digest(root)
    if digest != pin:
        raise PretrainedStoreError(
            f"the pretrained store at {root} is not the one this arm or book pinned "
            f"(manifest {digest}, pinned {pin}): restore the pinned store; a new store "
            "reaches only arms created after it"
        )
    return root


__all__ = [
    "MANIFEST_NAME",
    "PRETRAINED_STORE",
    "PretrainedStoreError",
    "manifest_digest",
    "pretrained_pin",
    "require_pretrained_store",
]
