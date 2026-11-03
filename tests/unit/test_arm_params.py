"""The arm parameter record: one declaration every parameter surface reads.

An arm's params.json is flat; ``ArmParams`` groups its parameters and is the
only list of them, their types, their defaults and who may set them. These
tests pin the conditions that must hold for that to stay true: every surface
is read off the record, a file reads back to the parameters it states, an
older arm still loads, and a key or a value the record does not declare is
refused.
"""

from __future__ import annotations

import pytest

from autotrade.pipelines.arm_params import (
    NON_PERSISTABLE_PARAMS,
    PARAMETERS,
    ArmParams,
    Surface,
    console_request,
    parameters,
    read_param,
    read_param_or_default,
    session_models,
)
from autotrade.pipelines.hitl_state import (
    WEB_CLOSED_PARAMS,
    WEB_CREATE_DEFAULTS,
    WEB_INTERNAL_PARAMS,
    WEB_REQUIRED_PARAMS,
)
from tests.unit.acceptance_fixture import CREATE_RULES

ROOTS = {"experiments_root": "experiments", "work_root": ".runtime/sandboxes"}


def _created(**request: object) -> dict[str, object]:
    return console_request(
        {"experiment_id": "arm_a", **CREATE_RULES, **request},
        defaults=WEB_CREATE_DEFAULTS,
        **ROOTS,
    )


def test_every_console_surface_is_read_off_the_record() -> None:
    assert list(WEB_CREATE_DEFAULTS) == parameters(Surface.CONSOLE)
    assert WEB_INTERNAL_PARAMS == {name: PARAMETERS[name].stamp for name in parameters(Surface.MANAGED)}
    assert WEB_CLOSED_PARAMS == {
        *parameters(Surface.MANAGED),
        *parameters(Surface.HOST),
        *NON_PERSISTABLE_PARAMS,
    }
    assert WEB_REQUIRED_PARAMS == {"experiment_id", *CREATE_RULES, "research_start", "research_end", "forward_end", "heldout_end"}
    # The rules a create request must state, sealed_years and
    # ablation_required, have no value anywhere; alpha, the third required
    # rule, carries the console's value 0.05 and is no create-time requirement.
    assert all(WEB_CREATE_DEFAULTS[name] is None for name in ("experiment_id", *CREATE_RULES))
    assert WEB_CREATE_DEFAULTS["alpha"] == 0.05 and "alpha" not in WEB_REQUIRED_PARAMS


def test_a_created_arm_reads_back_to_the_parameters_it_states() -> None:
    created = _created(initial_cash=100_000, screen_boards=["main"], compact_token_threshold=None)
    params = ArmParams.from_record(created)

    assert params.experiment_id == "arm_a"
    assert params.broker.initial_cash == 100_000
    assert params.data.screen_boards == ("main",)
    assert params.acceptance.sealed_years == CREATE_RULES["sealed_years"]
    assert params.environment.data_backend == "pit"
    # Every parameter holds what the file states, or its default where the
    # file states nothing (the host parameters a create never writes).
    for name, spec in PARAMETERS.items():
        owner = params if spec.group is None else getattr(params, spec.group)
        stated = created.get(name, spec.default)
        assert getattr(owner, name) == (tuple(stated) if isinstance(stated, list) else stated), name


def test_an_arm_created_before_a_parameter_existed_still_loads() -> None:
    # An arm of 2026-09-25 as its params.json stands: the retired gate rules
    # and NL parameters, and none of the parameters added since.
    legacy = {
        "experiment_id": "lstm_alla_8y_20260926",
        "research_start": "20170701",
        "research_end": "20250630",
        "forward_end": "20260630",
        "heldout_end": "20260930",
        "heldout_tolerance_z": 1.28,
        "min_active_ir": 0.75,
        "min_dsr_probability": 0.975,
        "min_full_span_validations": 2,
        "nl_failure_policy": "return_error_with_audit",
        "nl_model": "qwen-3.8-27b-fp8",
        "recency_months": 6,
        "_created_at": "2026-09-25T18:46:50.236327+00:00",
        "_creation_surface": "webui",
    }
    params = ArmParams.from_record(legacy)

    assert params.retired.min_dsr_probability == 0.975
    assert params.retired.recency_months == 6
    assert params.retired.nl_max_results is None
    assert params.acceptance.sealed_years is None
    assert params.environment.daily_path is None


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("max_null_controls", True, "max_null_controls must be an integer, not bool"),
        ("include_macro", 1, "include_macro must be a boolean, not int"),
        ("initial_cash", "100000", "initial_cash must be a number, not str"),
        ("screen_boards", "main", "screen_boards must be a list of strings, not str"),
        ("research_directive", ["a"], "research_directive must be a string, not list"),
    ],
)
def test_a_value_of_another_type_is_refused(key: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ArmParams.from_record({"experiment_id": "arm_a", key: value})
    with pytest.raises(ValueError, match=message):
        _created(**{key: value})


def test_null_reads_as_unset_and_a_missing_key_as_its_default() -> None:
    params = ArmParams.from_record(
        {"experiment_id": "arm_a", "window_months": None, "compact_token_threshold": "", "research_directive": ""}
    )

    assert params.data.window_months is None
    # A cleared form field sends "": unset for a number, the text itself for text.
    assert params.agent.compact_token_threshold is None
    assert params.packs.research_directive == ""
    assert params.budgets.max_replay_years == WEB_CREATE_DEFAULTS["max_replay_years"]


def test_an_undeclared_key_is_refused_and_a_stamp_is_skipped() -> None:
    with pytest.raises(ValueError, match=r"unknown experiment parameters: \['nl_budget'\]"):
        ArmParams.from_record({"experiment_id": "arm_a", "nl_budget": 1})
    assert ArmParams.from_record({"experiment_id": "arm_a", "_anything": 1}).experiment_id == "arm_a"


@pytest.mark.parametrize(
    ("request_", "message"),
    [
        ({"min_dsr_probability": 0.9}, "unknown experiment parameters: min_dsr_probability"),
        ({"data_backend": "daily"}, "console-managed parameters are not accepted: data_backend"),
        ({"work_root": "/elsewhere"}, "console-managed parameters are not accepted: work_root"),
        ({"llm_base_url": "http://x"}, "console-managed parameters are not accepted: llm_base_url"),
        ({"sealed_years": None}, "missing required experiment parameters: sealed_years"),
        ({"experiment_id": "_hidden"}, "experiment_id must match"),
    ],
)
def test_a_create_request_states_only_console_parameters(request_: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _created(**request_)


def test_a_reader_takes_the_one_declared_default_for_a_key_the_file_leaves_out() -> None:
    created = _created()
    # What the console stamps is what a reader of a file without the key reads.
    for name in set(parameters(Surface.CONSOLE)) - {"experiment_id", *CREATE_RULES}:
        assert read_param({}, name) == created[name] == PARAMETERS[name].default, name
    # A stated value, null included, reads as stated; a reader that takes an
    # empty value as unset gets the default for it too.
    assert read_param({"compact_token_threshold": None}, "compact_token_threshold") is None
    assert read_param({"compact_token_threshold": 5}, "compact_token_threshold") == 5
    assert read_param_or_default({"strategy_period": ""}, "strategy_period") == "day"
    assert read_param_or_default({"developer_mode": None}, "developer_mode") == "baseline"
    # A host parameter the console never writes declares the worker's value.
    assert read_param({}, "work_root") == ".runtime/sandboxes"
    with pytest.raises(KeyError):
        read_param({"no_such_parameter": 1}, "no_such_parameter")


def test_the_session_roles_take_their_defaults_and_derive_only_what_they_name() -> None:
    flash = PARAMETERS["model"].default
    assert session_models({}) == (flash, flash, PARAMETERS["compact_model"].default)
    assert session_models({"model": "m", "subagent_model": "", "compact_model": None}) == (
        "m",
        "m",
        PARAMETERS["compact_model"].default,
    )
    # A worker-side llm_model stands in for an empty main role only.
    assert session_models({"llm_model": "x"})[0] == "x"
    assert session_models({"model": "m", "llm_model": "x"})[0] == "m"
