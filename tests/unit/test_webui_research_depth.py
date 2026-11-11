"""How much research preceded each test, as the console reads it.

The console states, per step the session tested, the model calls and inference
time charged since the previous test, the counted ``validate`` calls and their
candidates, and the wall-clock between the two ledger stamps. It reads them
off what the worker already writes: the ledger's ``test_step`` stamps, the
``budget_used`` block each ``test`` call's trace event carries, and the step
tree. The experiments here are written as those files, on one explicit
timeline.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autotrade.common.records import write_json_atomic
from autotrade.environment.step_tree import StepTree
from autotrade.pipelines import session_tools
from autotrade.pipelines.hitl_state import proc_start_ticks
from autotrade.pipelines.ledger import LEDGER_RECORD_SCHEMA_VERSION
from autotrade.webui import traces
from autotrade.webui.registry import research_depth
from autotrade.webui.server import create_app
from tests.unit.test_webui_inject_message import _fn
from tests.unit.webui_research_experiment import PARAMS

APP_JS = Path(__file__).resolve().parents[2] / "src/autotrade/webui/static/app.js"
T0 = datetime(2025, 1, 1, 0, 0, 0, 123456, tzinfo=UTC)


def _at(minutes: float) -> str:
    return (T0 + timedelta(minutes=minutes)).isoformat()


def _budget(calls: int, seconds: float) -> dict[str, object]:
    return {"llm_calls": calls, "inference_seconds": seconds, "test_replays": 0, "null_controls": 0}


class _Experiment:
    """One experiment's files, written in the order its worker would."""

    def __init__(self, root: Path, experiment_id: str = "experiment") -> None:
        self.root = root
        self.directory = root / experiment_id
        self.experiment_id = experiment_id
        self.nodes: list[dict[str, object]] = []
        hitl = self.directory / "hitl"
        hitl.mkdir(parents=True)
        write_json_atomic(hitl / "params.json", {"experiment_id": experiment_id, **PARAMS, "validates_per_step": 3})
        self.status({"state": "stopped"})

    def status(self, fields: dict[str, object]) -> None:
        write_json_atomic(self.directory / "hitl/status.json", {"schema_version": 1, **fields})

    def live(self, run_id: str) -> None:
        self.status(
            {
                "pid": os.getpid(),
                "pid_start_ticks": proc_start_ticks(os.getpid()),
                "state": "running_session",
                "session_key": "research",
                "run_id": run_id,
            }
        )

    def ledger(self, minutes: float, run_id: str, record: dict[str, object]) -> None:
        path = self.directory / "ledgers/experiment_ledger.jsonl"
        path.parent.mkdir(exist_ok=True)
        line = {
            "experiment_id": self.experiment_id,
            "epoch_id": "research",
            "fold_id": "research",
            "session_key": "research",
            "run_id": run_id,
            **record,
            "schema_version": LEDGER_RECORD_SCHEMA_VERSION,
            "recorded_at": _at(minutes),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")

    def step(self, minutes: float, run_id: str, step: int, source: str = "test") -> None:
        self.ledger(minutes, run_id, {"record_type": "test_step", "step": step, "source": source})

    def trace(self, run_id: str, events: list[tuple[float, dict[str, object]]]) -> Path:
        path = self.directory / "artifacts/traces" / f"{run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for minutes, event in events:
                handle.write(json.dumps({**event, "ts": _at(minutes)}) + "\n")
        return path

    def validate(self, minutes: float, batch: str, *, complete: int, failed: int = 0) -> None:
        """One ``validate`` call's nodes: ``complete`` Validations and ``failed`` dead ends."""

        for index in range(complete + failed):
            done = index < complete
            self.nodes.append(
                {
                    "node_id": f"node_{batch}_{index}",
                    "epoch_id": "research",
                    "complete_validation": done,
                    **({} if done else {"status": "failed", "error": "BacktestError"}),
                    "created_at": _at(minutes),
                    "metadata": {"batch_id": batch, "span": "full"},
                }
            )
        write_json_atomic(self.directory / "steps/tree.json", {"current_node_id": None, "nodes": self.nodes})


def _test_call(minutes: float, calls: int, seconds: float) -> tuple[float, dict[str, object]]:
    return minutes, {"event_type": "tool_call", "tool": "test", "budget_used": _budget(calls, seconds)}


def _resumed_session(root: Path) -> _Experiment:
    """Two attempts: the first validates once and tests steps 1 and 2 back to
    back, is refused a third test and fails; the second validates, tests step 3,
    keeps working and validates again. It is still running."""

    experiment = _Experiment(root)
    experiment.validate(60, "batch_a", complete=2, failed=1)
    experiment.step(90, "run_a", 1)
    experiment.step(100, "run_a", 2)
    experiment.trace(
        "run_a",
        [
            (0, {"event_type": "environment_stage", "stage": "pit_snapshot"}),
            (50, {"event_type": "llm_call", "budget_used": _budget(39, 580.0)}),
            _test_call(90.01, 40, 600.0),
            _test_call(100.01, 41, 620.0),
            # Refused: no step was recorded for it.
            _test_call(105, 45, 700.0),
        ],
    )
    experiment.ledger(110, "run_a", {"record_type": "attempt_failed", "phase": "research", "error": "OSError: host lost"})
    experiment.validate(150, "batch_b", complete=3)
    experiment.step(180, "run_b", 3)
    experiment.validate(200, "batch_c", complete=1)
    experiment.trace(
        "run_b",
        [
            (120, {"event_type": "environment_stage", "stage": "pit_snapshot"}),
            _test_call(180.01, 100, 2000.0),
            (210, {"event_type": "llm_call", "budget_used": _budget(130, 2600.0)}),
        ],
    )
    experiment.live("run_b")
    return experiment


def _figures(row: dict[str, object]) -> tuple[object, ...]:
    return (row["step"], row["llm_calls"], row["inference_seconds"], row["validates"], row["candidates"])


def test_each_tested_step_reads_the_research_that_preceded_it(tmp_path: Path) -> None:
    """A step's interval is charged what the budget block of its own ``test``
    call holds beyond the previous step's, across attempts; its validate calls
    are the distinct batches of the complete Validations created in it, a failed
    candidate and a refused test counting for nothing; and what followed the
    last test is one more interval, open while the session runs."""

    experiment = _resumed_session(tmp_path)
    depth = research_depth(tmp_path, "experiment")
    first, second, third, tail = depth["intervals"]
    assert _figures(first) == (1, 40, 600.0, 1, 2)
    assert _figures(second) == (2, 1, 20.0, 0, 0)
    assert _figures(third) == (3, 59, 1380.0, 1, 3)
    assert _figures(tail) == (None, 30, 600.0, 1, 1)
    assert [row["llm_calls_total"] for row in depth["intervals"]] == [40, 41, 100, 130]
    # Wall-clock runs from the first attempt's first event, then stamp to stamp.
    assert [row["wall_seconds"] for row in (first, second, third)] == [90 * 60, 10 * 60, 80 * 60]
    assert (first["ended_at"], third["ended_at"]) == (_at(90), _at(180))
    assert tail["open"] is True and tail["ended_at"] is None and tail["wall_seconds"] > 0
    assert depth["validates_per_step"] == 3
    # The open interval counts what the session's own allowance would resume with.
    assert tail["validates"] == session_tools._validates_since(
        StepTree(experiment.directory / "steps"), _at(180)
    )
    assert traces.TEST_TOOL == session_tools.TestTool.spec.name


def test_a_stopped_session_keeps_its_last_interval_open_without_a_running_clock(tmp_path: Path) -> None:
    experiment = _resumed_session(tmp_path)
    experiment.status({"state": "terminated", "session_key": "research", "run_id": "run_b"})
    tail = research_depth(tmp_path, "experiment")["intervals"][-1]
    assert (tail["open"], tail["wall_seconds"], tail["llm_calls"]) == (True, None, 30)


def test_a_step_whose_trace_is_gone_is_unmetered_and_still_counts_its_validates(tmp_path: Path) -> None:
    """Without an attempt's trace its steps' spend is unknown, and so is the
    spend of the interval that starts from them; nothing is guessed. The step
    tree still gives every interval its validate calls."""

    experiment = _resumed_session(tmp_path)
    (experiment.directory / "artifacts/traces/run_a.jsonl").unlink()
    first, second, third, tail = research_depth(tmp_path, "experiment")["intervals"]
    for row in (first, second, third):
        assert (row["llm_calls"], row["inference_seconds"]) == (None, None)
    assert (first["wall_seconds"], first["validates"], first["candidates"]) == (None, 1, 2)
    assert (third["llm_calls_total"], third["validates"]) == (100, 1)
    assert tail["llm_calls"] == 30


def test_a_session_its_last_test_completed_has_no_interval_after_it(tmp_path: Path) -> None:
    experiment = _Experiment(tmp_path)
    experiment.validate(10, "batch_a", complete=1)
    experiment.step(20, "run_a", 1)
    experiment.trace("run_a", [(0, {"event_type": "session_start"}), _test_call(20.01, 12, 100.0)])
    experiment.ledger(
        21, "run_a", {"record_type": "research_session", "outcome": "completed", "budget_used": _budget(12, 100.0)}
    )
    experiment.status({"state": "completed"})
    assert [_figures(row) for row in research_depth(tmp_path, "experiment")["intervals"]] == [(1, 12, 100.0, 1, 1)]


def test_a_session_the_host_completed_shows_what_followed_its_last_test(tmp_path: Path) -> None:
    """A budget that ran out after step 1: the research that followed the last
    test is one closed interval up to the session's record, and the steps the
    host wrote are no research."""

    experiment = _Experiment(tmp_path)
    experiment.step(20, "run_a", 1)
    experiment.validate(30, "batch_a", complete=2)
    experiment.step(40, "run_a", 2, source="host_completion")
    experiment.step(40.001, "run_a", 3, source="host_completion")
    experiment.trace("run_a", [(0, {"event_type": "session_start"}), _test_call(20.01, 12, 100.0)])
    experiment.ledger(
        41, "run_a", {"record_type": "research_session", "outcome": "deadline", "budget_used": _budget(400, 9000.0)}
    )
    experiment.status({"state": "completed"})
    first, tail = research_depth(tmp_path, "experiment")["intervals"]
    assert _figures(first) == (1, 12, 100.0, 0, 0)
    assert _figures(tail) == (None, 388, 8900.0, 1, 2)
    assert (tail["open"], tail["ended_at"], tail["wall_seconds"]) == (False, _at(41), 21 * 60)


def test_an_experiment_that_never_ran_has_no_interval_and_an_unknown_one_no_route(tmp_path: Path) -> None:
    _Experiment(tmp_path)
    client = TestClient(create_app(tmp_path, tmp_path))
    assert client.get("/api/experiments/experiment/research-depth").json()["intervals"] == []
    assert client.get("/api/experiments/missing/research-depth").status_code == 404


def test_an_unreadable_ledger_answers_a_stated_conflict(tmp_path: Path) -> None:
    experiment = _resumed_session(tmp_path)
    (experiment.directory / "ledgers/experiment_ledger.jsonl").write_text("{not json\n", encoding="utf-8")
    response = TestClient(create_app(tmp_path, tmp_path)).get("/api/experiments/experiment/research-depth")
    assert response.status_code == 409
    assert str(tmp_path) not in response.text


def test_the_test_marks_of_a_trace_follow_its_appends(tmp_path: Path) -> None:
    """The marks ride the incremental scan the counters use: a trace read again
    after it grew adds the new calls and keeps its first event's time; a
    ``test`` call still running, and a call of another tool, are no marks."""

    experiment = _Experiment(tmp_path)
    path = experiment.trace(
        "run_a",
        [
            (0, {"event_type": "session_start"}),
            (1, {"event_type": "tool_call_started", "tool": "test"}),
            _test_call(2, 5, 50.0),
            (3, {"event_type": "tool_call", "tool": "validate", "budget_used": _budget(6, 60.0)}),
        ],
    )
    marks = traces.trace_test_marks(path)
    assert marks == {"started_at": _at(0), "tests": [{"ts": _at(2), "budget_used": _budget(5, 50.0)}]}
    experiment.trace("run_a", [_test_call(4, 9, 90.0)])
    again = traces.trace_test_marks(path)
    assert again["started_at"] == _at(0)
    assert [mark["budget_used"]["llm_calls"] for mark in again["tests"]] == [5, 9]
    assert traces.trace_stats(path)["tool_counts"] == {"test": 2, "validate": 1}


@pytest.mark.skipif(shutil.which("node") is None, reason="node required for the JS check")
def test_the_depth_table_words_each_interval_and_dashes_what_is_unmetered(tmp_path: Path) -> None:
    """The panel is the server's rows, one per interval: a tested step by its
    number, what followed the last test worded by whether the session can still
    run, and a dash wherever the records carry no figure. The cumulative calls
    ride in the cell's hover against the experiment's limit."""

    experiment = _resumed_session(tmp_path)
    (experiment.directory / "artifacts/traces/run_a.jsonl").unlink()
    payload = research_depth(tmp_path, "experiment")
    script = APP_JS.read_text(encoding="utf-8")
    harness = "\n".join(
        [
            """
const node = (tag) => ({
  tag, nodeType: 1, style: {}, children: [], attrs: {},
  append(...nodes) { this.children.push(...nodes); },
  replaceChildren(...nodes) { this.children = nodes; },
  addEventListener() {}, setAttribute(key, value) { this.attrs[key] = value; },
});
const document = { createElement: node, createTextNode: (text) => ({ nodeType: 3, text }) };
// The router's listeners sit between dataTable and the next function.
const window = { addEventListener() {} };
const route = null;
const liveTimers = [];
const setPollInterval = () => 1;
const fmtTs = (iso) => `at ${iso.slice(11, 16)}`;
""",
            f"const payload = {json.dumps(payload)};",
            "const api = async () => payload;",
            *(_fn(script, name) for name in ("el", "panelHead", "dataTable", "fmtDuration", "fmtPct", "researchDepthPanel")),
            """
const text = (item) => item.nodeType === 3 ? item.text : item.children.map(text).join("");
const find = (item, tag) => item.nodeType !== 1 ? [] : [
  ...(item.tag === tag ? [item] : []), ...item.children.flatMap((child) => find(child, tag)),
];
const host = researchDepthPanel({ experiment_id: "experiment", budget: { llm_calls: 1000 } }, true);
setTimeout(() => {
  const rows = find(host, "tr").slice(1).map((row) => find(row, "td").map(text));
  const hovers = find(host, "tr").slice(1).map((row) => find(row, "td")[3].attrs.title || null);
  console.log(JSON.stringify({ rows, hovers, polls: liveTimers.length }));
}, 0);
""",
        ]
    )
    result = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    drawn = json.loads(result.stdout)
    assert [row[0] for row in drawn["rows"]] == [
        "第 1 步之前",
        "第 2 步之前",
        "第 3 步之前",
        "末次 test 之后（会话未结束）",
    ]
    assert drawn["rows"][0][1:] == ["at 01:30", "—", "—", "—", "1", "2"]
    assert drawn["rows"][1][2:] == ["10:00", "—", "—", "0", "0"]
    assert drawn["rows"][3][1] == "—" and drawn["rows"][3][3:] == ["30", "10:00", "1", "1"]
    assert drawn["hovers"] == [None, None, "累计 100 / 1000（10.0%）", "累计 130 / 1000（13.0%）"]
    assert drawn["polls"] == 1
