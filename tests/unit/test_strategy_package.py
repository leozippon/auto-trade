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


DEFERRED_MAIN = '''import scorer
import state


def generate_orders(context):
    state.remember(context.inference_at.strftime("%Y%m%d"))
    return [{"seen": scorer.seen()}]
'''
DEFERRED_SCORER = '''def seen():
    import state  # inside the call, the way a package breaks an import cycle

    return {tag}state.seen()
'''
DEFERRED_STATE = '''SEEN = None


def remember(value):
    global SEEN
    SEEN = value


def seen():
    return SEEN
'''


def _write_deferred_package(root: Path, *, tag: str = "") -> Path:
    root.mkdir(parents=True)
    (root / "state.py").write_text(DEFERRED_STATE, encoding="utf-8")
    (root / "scorer.py").write_text(DEFERRED_SCORER.format(tag=tag), encoding="utf-8")
    (root / "main.py").write_text(DEFERRED_MAIN, encoding="utf-8")
    return root / "main.py"


def test_a_module_imported_inside_a_call_is_the_one_the_package_already_holds(tmp_path: Path):
    """A package module is one module, wherever the import statement sits.

    Python resolves an import written inside a function body when that function
    runs, which is after the entry module finished importing. A Fold lost a
    Validation to this: the sibling module its ``generate_orders`` had just
    written to was not the one the scoring module imported and read back --
    on the host that import raises, and in the replay container (whose working
    directory is the package root) it silently executes the file a second time
    into a module whose module-level state is still empty.
    """

    first = TrustedStrategyExecutor.from_path(_write_deferred_package(tmp_path / "first"))
    second = TrustedStrategyExecutor.from_path(
        _write_deferred_package(tmp_path / "second", tag="'second:' + ")
    )
    context = _context("")

    assert first.execute(context) == [{"seen": "20260102"}]
    # Each strategy keeps its own sibling modules, before and after the other
    # one runs, and neither leaves anything behind between calls.
    assert second.execute(context) == [{"seen": "second:20260102"}]
    assert first.execute(context) == [{"seen": "20260102"}]
    assert "state" not in sys.modules and "scorer" not in sys.modules
    assert str(tmp_path / "first") not in sys.path and str(tmp_path / "second") not in sys.path
    assert not list(tmp_path.rglob("__pycache__"))


@pytest.mark.parametrize(
    ("helper", "message"),
    [
        ("import subprocess\n", "lib/features.py: strategy imports unsupported module: subprocess"),
        ("from . import other\n", "lib/features.py: strategy uses a relative import"),
        (
            "import numpy as np\ndef dump(ctx):\n    np.save('/tmp/x.npy', [])\n",
            "lib/features.py: strategy passes an absolute path literal to save",
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


def test_a_path_may_be_built_any_way_the_strategy_likes(tmp_path: Path):
    """The check cannot see where a computed path points, so it does not guess.

    Reviewed Folds lost rounds to the old shape rule, which only recognized
    ``context.<root> + "<literal>"`` and rejected every helper, variable,
    f-string and named constant that builds the same path.
    """

    helper = (
        "import pandas as pd\nimport numpy as np\n\n"
        "DOMAIN = '/daily'\n\n"
        "def _path(ctx, name):\n    return ctx.state_dir + '/' + name\n\n"
        "def daily(ctx, columns):\n"
        "    return pd.read_parquet(ctx.asof_dir + DOMAIN, columns=columns)\n\n"
        "def cache(ctx, frames):\n"
        "    for index, frame in enumerate(frames):\n"
        "        frame.to_parquet(f'{ctx.state_dir}/part_{index}.parquet')\n"
        "    np.save(_path(ctx, 'w.npy'), np.zeros(3))\n\n"
        "def scaled(value):\n    return value\n"
    )
    assert validate_strategy_package(_write_package(tmp_path / "output", helper=helper)) is not None


def test_a_path_passed_by_keyword_is_read_like_a_positional_one():
    """``np.load(file=...)`` is the same call as ``np.load(...)``.

    Reading only the positional arguments let the real parameter names --
    ``file``, ``path``, ``filename`` -- carry an absolute literal, or a write
    below a read-only root, straight past both rules.
    """

    with pytest.raises(StrategyLoadError, match="absolute path literal to load"):
        validate_strategy_source(
            "import numpy as np\ndef generate_orders(context):\n"
            "    np.load(file='/mnt/snapshot/daily.parquet')\n    return []\n"
        )
    with pytest.raises(StrategyLoadError, match="may not save below context.models_dir"):
        validate_strategy_source(
            "import numpy as np\ndef fit(context):\n"
            "    np.save(file=context.models_dir + '/w.npy', arr=1)\n"
            "def generate_orders(context): return []\n"
        )
    # The same call below a writable root still passes.
    validate_strategy_source(
        "import numpy as np\ndef fit(context):\n"
        "    np.save(file=context.state_dir + '/w.npy', arr=1)\n"
        "def generate_orders(context): return []\n"
    )


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
    with pytest.raises(StrategyLoadError, match="absolute path literal to save_model"):
        validate_strategy_source(
            "import lightgbm as lgb\ndef fit(context):\n    lgb.Booster().save_model('/tmp/m.txt')\n"
            "def generate_orders(context): return []\n"
        )
    # torch.save puts the path second: the check reads every positional
    # argument, so the rooted call passes and the absolute literal does not.
    validate_strategy_source(
        "import torch\ndef fit(context):\n    torch.save({}, context.state_dir + '/m.pt')\n"
        "def generate_orders(context): return []\n"
    )
    with pytest.raises(StrategyLoadError, match="absolute path literal to save"):
        validate_strategy_source(
            "import torch\ndef fit(context):\n    torch.save({}, '/mnt/agent/workspace/m.pt')\n"
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


@pytest.mark.skipif(not docker_available(), reason="Docker is unavailable")
def test_real_sandbox_resolves_a_deferred_package_import_to_the_same_module(tmp_path: Path):
    """The container is where a split module used to go unnoticed.

    Its working directory is the package root, so an import inside a call
    found the file again instead of failing, and executed a second copy of it
    whose module-level state was empty.
    """

    image = "autotrade-sandbox:latest"
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False).returncode:
        pytest.skip(f"local sandbox image is unavailable: {image}")
    package = tmp_path / "output"
    strategy = _write_deferred_package(package)
    package.chmod(0o755)
    executor = DockerStrategyExecutor(strategy)
    try:
        orders = executor.execute(_context(""))
    finally:
        executor.close()
    assert orders == [{"seen": "20260102"}]
