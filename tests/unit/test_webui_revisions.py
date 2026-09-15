"""The arm's retained strategy revisions on the console's Step-tree surface.

The store is host state. What must hold is that the console publishes the
lineage as opaque refs and nothing else, resolves a diff back from those refs,
reads the revisions the older layout left behind rather than hiding them, and
refuses a reference it does not know instead of guessing a revision.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from autotrade.environment.artifacts import (
    REVISION_MANIFEST_FILE,
    FilesystemArtifactStore,
    copy_artifact,
)
from autotrade.environment.runtime import chmod_tree
from autotrade.pipelines.config import EvaluationResult, StepResult
from autotrade.pipelines.session_resume import record_step_sidecar
from autotrade.webui.server import create_app
from tests.unit.webui_research_arm import build_arm

STRATEGY = "def generate_orders(context):\n    return []\n"


def _arm(tmp_path: Path) -> tuple[TestClient, Path]:
    experiments = tmp_path / "experiments"
    directory = build_arm(experiments, "arm", "research")
    return TestClient(create_app(tmp_path, experiments)), directory


def _working_output(tmp_path: Path, strategy: str) -> Path:
    output = tmp_path / "work" / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "main.py").write_text(strategy, encoding="utf-8")
    return output


def _two_revisions(tmp_path: Path, directory: Path) -> FilesystemArtifactStore:
    """``revision_a``, then ``revision_b`` descending from it: one file changed
    and one added, which is what a candidate of the next round looks like."""

    store = FilesystemArtifactStore(directory / "artifacts/strategy")
    output = _working_output(tmp_path, STRATEGY)
    store.create_revision(output, revision_id="revision_a")
    (output / "main.py").write_text(STRATEGY + "# second\n", encoding="utf-8")
    (output / "signals.py").write_text("SIGNAL = 1\n", encoding="utf-8")
    store.create_revision(
        output, revision_id="revision_b", parent_revision_id="revision_a"
    )
    return store


def _lineage(client: TestClient) -> list[dict[str, object]]:
    response = client.get("/api/experiments/arm/revisions")
    assert response.status_code == 200, response.text
    return response.json()["revisions"]


def test_an_arm_with_no_revisions_publishes_an_empty_lineage(tmp_path: Path) -> None:
    """Nothing to draw is an empty list, not an error the panel would swallow."""

    client, _directory = _arm(tmp_path)
    assert client.get("/api/experiments/arm/revisions").json() == {"revisions": []}


def test_the_lineage_names_revisions_by_ref_and_joins_them_to_their_step_node(
    tmp_path: Path,
) -> None:
    client, directory = _arm(tmp_path)
    _two_revisions(tmp_path, directory)
    record_step_sidecar(
        directory,
        StepResult("node_1", "revision_a", EvaluationResult({}, "ref_1"), span="full"),
    )

    first, second = _lineage(client)

    assert first["strategy_ref"].startswith("strategy_ref_")
    assert first["parent_strategy_ref"] is None
    assert second["parent_strategy_ref"] == first["strategy_ref"]
    # The Step node id is already public; the revision ids are not.
    assert (first["node_id"], second["node_id"]) == ("node_1", None)
    assert (first["layout"], second["layout"]) == ("objects", "objects")
    assert (first["file_count"], second["file_count"]) == (1, 2)
    assert first["total_bytes"] == len(STRATEGY.encode("utf-8"))
    body = json.dumps([first, second])
    assert "revision_a" not in body and "revision_b" not in body
    assert str(tmp_path) not in body


def test_a_revision_from_the_older_layout_is_shown_without_a_parent(
    tmp_path: Path,
) -> None:
    """A revision copied before the store was content-addressed has no recorded
    lineage. It is still the arm's history, so it reads as a root, marked."""

    client, directory = _arm(tmp_path)
    store = FilesystemArtifactStore(directory / "artifacts/strategy")
    legacy = store.revisions_root / "revision_legacy"
    copy_artifact(_working_output(tmp_path, STRATEGY), legacy / "output")
    chmod_tree(legacy, file_mode=0o444, dir_mode=0o555)
    assert not (legacy / REVISION_MANIFEST_FILE).exists()

    [row] = _lineage(client)

    assert row["layout"] == "legacy"
    assert row["parent_strategy_ref"] is None and row["created_at"] is None
    assert row["file_count"] == 1


def test_the_diff_returns_the_lines_the_agent_wrote(tmp_path: Path) -> None:
    client, directory = _arm(tmp_path)
    _two_revisions(tmp_path, directory)
    first, second = _lineage(client)

    payload = client.get(
        "/api/experiments/arm/revisions/diff",
        params={"a": first["strategy_ref"], "b": second["strategy_ref"]},
    ).json()

    assert payload["strategy_ref_a"] == first["strategy_ref"]
    assert payload["strategy_ref_b"] == second["strategy_ref"]
    assert (payload["added"], payload["removed"], payload["modified"]) == (1, 0, 1)
    changes = {str(item["path"]): item for item in payload["files"]}
    assert set(changes) == {"output/main.py", "output/signals.py"}
    assert changes["output/signals.py"]["change"] == "added"
    assert changes["output/signals.py"]["size_before"] is None
    assert "+SIGNAL = 1" in changes["output/signals.py"]["diff"]
    modified = changes["output/main.py"]
    assert modified["change"] == "modified"
    assert modified["size_before"] == len(STRATEGY.encode("utf-8"))
    assert "+# second" in modified["diff"]
    assert modified["diff_truncated"] is False


def test_a_revision_compared_with_itself_reports_no_change(tmp_path: Path) -> None:
    client, directory = _arm(tmp_path)
    _two_revisions(tmp_path, directory)
    ref = _lineage(client)[1]["strategy_ref"]

    payload = client.get(
        "/api/experiments/arm/revisions/diff", params={"a": ref, "b": ref}
    ).json()

    assert payload["files"] == []
    assert (payload["added"], payload["removed"], payload["modified"]) == (0, 0, 0)


@pytest.mark.parametrize("unknown", ["revision_a", f"strategy_ref_{uuid4()}"])
def test_a_reference_the_console_never_issued_is_refused(
    tmp_path: Path, unknown: str
) -> None:
    """A raw revision id is not a reference, and a well-formed reference the
    experiment never issued resolves to nothing: both are named, not guessed."""

    client, directory = _arm(tmp_path)
    _two_revisions(tmp_path, directory)
    known = _lineage(client)[0]["strategy_ref"]

    response = client.get(
        "/api/experiments/arm/revisions/diff", params={"a": unknown, "b": known}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "unknown strategy revision"
