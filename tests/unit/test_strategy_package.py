"""The formal strategy is a package: ``main.py`` plus sibling modules.

Every ``.py`` below ``output/`` is held to the same import and I/O rules, the
artifact fingerprint and the executable structure cover all of them, and the
budgets the strategy runs under come from one source each.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autotrade.environment.artifacts import artifact_fingerprint
from autotrade.environment.executor import (
    DockerStrategyExecutor,
    TrustedStrategyExecutor,
    docker_available,
)
from autotrade.environment.sandbox import SandboxLimits
from autotrade.environment.strategy import CN_TZ, AccountSnapshot, StrategyContext
from autotrade.environment.strategy_loader import (
    StrategyLoadError,
    load_strategy_module,
    validate_strategy_package,
    validate_strategy_source,
)
from autotrade.environment.tools import ToolError
from autotrade.environment.tools.finish_fold import executable_output_structure
from autotrade.environment.tools.modification_check import ModificationCheckTool
from autotrade.pipelines.config import rolling_default
from autotrade.pipelines.experiment import _MAX_DEADLINE_OVERRIDE_MINUTES, _session_budgets
from autotrade.pipelines.hitl_state import WEB_CREATE_DEFAULTS
from autotrade.pipelines.local_backend import LLMMetaLearner
from autotrade.pipelines.worker import _strategy_sandbox_from_spec

MAIN = '''import numpy as np

from lib.features import scaled


def fit(context):
    np.save(context.state_dir + "/w.npy", np.array([scaled(1.0)]))


def generate_orders(context):
    return [{"weight": float(np.load(context.state_dir + "/w.npy")[0]), "live": scaled(2.0)}]
'''
HELPER = "SCALE = {scale}\n\n\ndef scaled(value):\n    return value * SCALE\n"


def _write_package(root: Path, *, helper: str = HELPER.format(scale=2)) -> Path:
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "__init__.py").write_text("", encoding="utf-8")
    (root / "lib" / "features.py").write_text(helper, encoding="utf-8")
    (root / "main.py").write_text(MAIN, encoding="utf-8")
    return root / "main.py"


def _context(state_dir: str) -> StrategyContext:
    return StrategyContext(
        inference_at=datetime(2026, 1, 2, 8, 30, tzinfo=CN_TZ),
        bars=(),
        account=AccountSnapshot(cash=1.0, positions={}),
        state_dir=state_dir,
    )


def test_package_loads_in_host_and_helper_modules_do_not_leak_between_strategies(tmp_path: Path):
    first = _write_package(tmp_path / "first")
    second = _write_package(tmp_path / "second", helper=HELPER.format(scale=3))
    state = tmp_path / "state"
    state.mkdir()

    executor = TrustedStrategyExecutor.from_path(first, state_dir=state)
    context = _context(str(state))
    executor.fit(context)
    (order,) = executor.execute(context)
    assert (order["weight"], order["live"]) == (2.0, 4.0)

    # The second package has its own lib.features; the first one's module must
    # not be served from sys.modules, and the package root must not linger.
    (second_order,) = TrustedStrategyExecutor.from_path(second, state_dir=state).execute(context)
    assert second_order["live"] == 6.0
    assert "lib" not in sys.modules and "lib.features" not in sys.modules
    assert str(tmp_path / "first") not in sys.path and str(tmp_path / "second") not in sys.path
    assert not list(tmp_path.rglob("__pycache__"))
    # The first strategy still uses the module it imported.
    assert executor.execute(context)[0]["live"] == 4.0


@pytest.mark.parametrize(
    ("helper", "message"),
    [
        ("import subprocess\n", "lib/features.py: strategy imports unsupported module: subprocess"),
        ("from . import other\n", "lib/features.py: strategy uses a relative import"),
        (
            "import numpy as np\ndef dump(ctx):\n    np.save('/tmp/x.npy', [])\n",
            "lib/features.py: strategy may save only below context.state_dir",
        ),
        (
            "import pandas as pd\ndef dump(ctx):\n    pd.read_pickle(ctx.models_dir + '/m.pkl')\n",
            "lib/features.py: strategy calls unsupported external I/O method: read_pickle",
        ),
    ],
)
def test_a_sibling_module_is_held_to_the_same_rules_everywhere(tmp_path: Path, helper, message):
    main = _write_package(tmp_path / "output", helper=helper + "def scaled(value):\n    return value\n")
    with pytest.raises(StrategyLoadError, match=message):
        validate_strategy_package(main)
    with pytest.raises(ToolError, match=message):
        ModificationCheckTool(tmp_path / "output").invoke({})
    with pytest.raises(StrategyLoadError, match=message):
        TrustedStrategyExecutor.from_path(main, state_dir=tmp_path)


def test_helper_reads_are_rooted_by_attribute_not_by_parameter_name(tmp_path: Path):
    """A helper receives the context under any name; the root attribute is the rule."""

    helper = (
        "import pandas as pd\n\n"
        "def daily(ctx, columns):\n"
        "    return pd.read_parquet(ctx.asof_dir + '/daily', columns=columns)\n\n"
        "def scaled(value):\n    return value\n"
    )
    assert validate_strategy_package(_write_package(tmp_path / "output", helper=helper)) is not None


def test_package_shadowing_a_library_and_missing_entry_are_rejected(tmp_path: Path):
    root = tmp_path / "output"
    _write_package(root)
    (root / "numpy.py").write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(StrategyLoadError, match="shadows a library module: numpy"):
        validate_strategy_package(root / "main.py")
    with pytest.raises(StrategyLoadError, match="does not exist"):
        validate_strategy_package(tmp_path / "absent" / "main.py")


def test_fingerprint_and_executable_structure_cover_sibling_modules(tmp_path: Path):
    root = tmp_path / "output"
    _write_package(root)
    before_fingerprint = artifact_fingerprint(root)
    before_structure = executable_output_structure(root)

    (root / "lib" / "features.py").write_text(
        "# a comment only\n" + HELPER.format(scale=2), encoding="utf-8"
    )
    assert artifact_fingerprint(root) != before_fingerprint
    assert executable_output_structure(root) == before_structure

    (root / "lib" / "features.py").write_text(HELPER.format(scale=3), encoding="utf-8")
    assert executable_output_structure(root) != before_structure


def test_library_imports_and_booster_files_follow_the_rooted_io_rule():
    validate_strategy_source(
        "from sklearn.linear_model import Ridge\nimport scipy.stats\nimport torch.nn\n"
        "import lightgbm as lgb\nimport xgboost\nimport statsmodels.api as sm\n"
        "from dataclasses import dataclass\nfrom typing import Sequence\n"
        "def fit(context):\n"
        "    lgb.Booster().save_model(context.state_dir + '/m.txt')\n"
        "def generate_orders(context):\n"
        "    xgboost.Booster().load_model(context.state_dir + '/m.json')\n    return []\n"
    )
    with pytest.raises(StrategyLoadError, match="save_model only below context.state_dir"):
        validate_strategy_source(
            "import lightgbm as lgb\ndef fit(context):\n    lgb.Booster().save_model('/tmp/m.txt')\n"
            "def generate_orders(context): return []\n"
        )
    # torch.save puts the path second: rejected, tensors go through NumPy.
    with pytest.raises(StrategyLoadError, match="first positional argument"):
        validate_strategy_source(
            "import torch\ndef fit(context):\n    torch.save({}, context.state_dir + '/m.pt')\n"
            "def generate_orders(context): return []\n"
        )
    with pytest.raises(StrategyLoadError, match="unsupported module: joblib"):
        validate_strategy_source("import joblib\ndef generate_orders(context): return []\n")


def test_budgets_come_from_one_source_each():
    limits = SandboxLimits()
    assert (limits.cpus, limits.memory, limits.pids) == (16.0, "32g", 256)
    assert (limits.timeout_seconds, limits.fit_timeout_seconds) == (180.0, 3600.0)
    # The pipeline knob defaults to the executor's fit wall clock and the
    # WebUI defaults read the pipeline dataclass.
    assert rolling_default("strategy_fit_timeout_seconds") == limits.fit_timeout_seconds
    strategy_config = _strategy_sandbox_from_spec(
        None, fit_timeout_seconds=rolling_default("strategy_fit_timeout_seconds")
    )
    assert strategy_config.limits == limits
    for name in ("max_fold_minutes", "max_backtests_per_fold", "max_steps_per_fold", "max_llm_calls"):
        assert WEB_CREATE_DEFAULTS[name] == rolling_default(name)
    assert (
        rolling_default("max_fold_minutes"),
        rolling_default("max_backtests_per_fold"),
        rolling_default("max_steps_per_fold"),
        rolling_default("max_llm_calls"),
    ) == (720, 30, 30, 1600)
    assert _MAX_DEADLINE_OVERRIDE_MINUTES == 2 * rolling_default("max_fold_minutes") == 1440
    # The per-decision wall clock the Meta session publishes is the executor default.
    meta_defaults = inspect.signature(LLMMetaLearner.__init__).parameters
    assert meta_defaults["decision_timeout_seconds"].default == limits.timeout_seconds
    assert meta_defaults["fit_timeout_seconds"].default == limits.fit_timeout_seconds

    config = SimpleNamespace(
        max_steps_per_fold=rolling_default("max_steps_per_fold"),
        max_backtests_per_fold=rolling_default("max_backtests_per_fold"),
        max_llm_calls=rolling_default("max_llm_calls"),
        max_fold_minutes=rolling_default("max_fold_minutes"),
        deadline_grace_minutes=rolling_default("deadline_grace_minutes"),
    )
    budgets = _session_budgets(config, {"deadline_seconds": _MAX_DEADLINE_OVERRIDE_MINUTES * 60})
    grace = rolling_default("deadline_grace_minutes") * 60.0
    assert budgets["deadline_seconds"] == 1440 * 60.0 + grace
    with pytest.raises(ValueError, match="cannot exceed 1440 minutes"):
        _session_budgets(config, {"deadline_seconds": _MAX_DEADLINE_OVERRIDE_MINUTES * 60 + 1})


@pytest.mark.skipif(not docker_available(), reason="Docker is unavailable")
def test_real_sandbox_runs_a_package_with_the_shipped_libraries(tmp_path: Path):
    image = "autotrade-sandbox:latest"
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False).returncode:
        pytest.skip(f"local sandbox image is unavailable: {image}")
    package = tmp_path / "output"
    helper = (
        "import lightgbm, scipy, sklearn, statsmodels, torch, xgboost\n"
        "torch.set_num_threads(2)\n"
        "SCALE = torch.get_num_threads()\n\n\n"
        "def scaled(value):\n    return value * SCALE\n"
    )
    strategy = _write_package(package, helper=helper)
    package.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    state.chmod(0o777)
    executor = DockerStrategyExecutor(strategy, state_dir=state)
    try:
        context = _context(executor.context_state_dir)
        executor.fit(context)
        (order,) = executor.execute(context)
    finally:
        executor.close()
    assert np.load(state / "w.npy").tolist() == [2.0]
    assert (order["weight"], order["live"]) == (2.0, 4.0)
