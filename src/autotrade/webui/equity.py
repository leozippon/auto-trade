"""Daily equity series of one ledger-named replay result.

The curve itself is ``replay.curve``'s projection of the result file and its
style sidecar — the same one a Paper book copies at creation — so this module
only answers which results exist at all: ``registry.ledger_result``'s rule, by
which the forward replay has no curve until its verdict is recorded.
"""

from __future__ import annotations

from pathlib import Path

from autotrade.environment.replay.curve import result_curve

from . import registry


def result_equity_payload(root: Path, experiment_id: str, name: str) -> dict[str, object]:
    return {
        "experiment_id": experiment_id,
        "result": name,
        **result_curve(registry.ledger_result(root, experiment_id, name)),
    }
