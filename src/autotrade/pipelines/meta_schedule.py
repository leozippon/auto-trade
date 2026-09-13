"""Session keys of Fold-era Meta ledger records, for the console that still reads them."""

from __future__ import annotations

from collections.abc import Mapping


def meta_session_key(epoch_id: str, trigger_after_folds: int = 0) -> str:
    """HITL session key of one Fold-era Meta session."""

    if trigger_after_folds <= 0:
        return f"{epoch_id}/meta_learning"
    return f"{epoch_id}/meta_learning_after_fold_{trigger_after_folds:03d}"


def meta_record_session_key(record: Mapping[str, object]) -> str:
    return meta_session_key(
        str(record.get("epoch_id") or ""),
        int(record.get("trigger_after_folds") or 0),
    )
