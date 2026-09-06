"""Statically preflight and load a trusted Agent-authored strategy package."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .strategy import FitSchedule, StrategyContext, StrategyContractError, StrategyFunction

# Pure-computation standard library plus the numerical/ML stack the sandbox
# image ships. Submodules (``scipy.stats``, ``sklearn.linear_model``,
# ``torch.nn``) are covered by their top-level name; modules that live inside
# the strategy package itself are added per package by ``local_modules``.
ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "functools",
        "itertools",
        "math",
        "statistics",
        "typing",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "lightgbm",
        "xgboost",
        "statsmodels",
        "torch",
    }
)
FORBIDDEN_CALLS = frozenset({"compile", "eval", "exec", "open", "__import__"})
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "read_csv",
        "read_clipboard",
        "read_excel",
        "read_feather",
        "read_fwf",
        "read_gbq",
        "read_hdf",
        "read_html",
        "read_json",
        "read_orc",
        "read_pickle",
        "read_sas",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "read_spss",
        "read_stata",
        "read_table",
        "read_xml",
        "fromfile",
        "fromregex",
        "genfromtxt",
        "loadtxt",
        "memmap",
        "savetxt",
        "to_csv",
        "to_excel",
        "to_feather",
        "tofile",
        "to_hdf",
        "to_json",
        "to_pickle",
        "to_sql",
        "to_stata",
        "to_xml",
        "urlopen",
    }
)
# The only file I/O a strategy may perform: reads below a read-only data root,
# writes below the per-replay state directory that ``fit`` owns.
# ``save_model``/``load_model`` are the LightGBM and XGBoost booster files.
# The static check on their path arguments is a denylist, not a shape rule: a
# path may be built by a helper, a variable, a loop or an f-string, and which
# positional argument holds it differs per API (``np.save(path, obj)`` against
# ``torch.save(obj, path)``), so only what the source itself proves wrong is
# refused — a literal absolute path, and a write aimed at a read-only context
# root. The replay container's mounts and ``StrategyContext`` are the real
# boundary; this check only reports the mistakes it can see before the replay.
ROOTED_READS = frozenset({"read_parquet", "load", "load_model"})
ROOTED_WRITES = frozenset({"to_parquet", "save", "savez", "savez_compressed", "save_model"})
READ_ROOTS = ("snapshot_dir", "asof_dir", "state_dir", "models_dir")
WRITE_ROOTS = ("state_dir",)
REFIT_PERIOD_NAME = "REFIT_PERIOD"
ENTRYPOINT_NAME = "main.py"


class StrategyLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedStrategy:
    """The entrypoints one ``main.py`` exposes; ``fit`` is optional."""

    generate_orders: StrategyFunction
    fit: Callable[[StrategyContext], object] | None
    fit_schedule: FitSchedule | None


def validate_strategy_source(
    source: str,
    *,
    filename: str = ENTRYPOINT_NAME,
    local_modules: frozenset[str] = frozenset(),
    entrypoints: bool = True,
) -> FitSchedule | None:
    """Reject common direct capability and external-I/O calls before import.

    ``local_modules`` names the top-level modules and packages that live next
    to ``main.py`` and may be imported absolutely. With ``entrypoints`` the
    file is the entry module and must declare ``generate_orders`` (and may
    declare ``fit``/``REFIT_PERIOD``); the returned schedule is the declared
    ``fit`` schedule (``None`` when there is no ``fit``). Helper modules pass
    ``entrypoints=False`` and get the same import and I/O rules only.

    This is a convenience check for trusted, reviewed strategies, not a
    sandbox or a security boundary: it names mistakes the source proves
    (unsupported imports, direct capability calls, external I/O methods,
    absolute path literals, writes at a read-only root) and leaves everything
    it cannot judge to the replay container's mounts and ``StrategyContext``.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise StrategyLoadError(f"invalid strategy syntax: {exc}") from exc
    fit_schedule = None
    if entrypoints:
        _entrypoint(tree, "generate_orders", required=True)
        fit = _entrypoint(tree, "fit", required=False)
        refit_period = _refit_period(tree, has_fit=fit is not None)
        fit_schedule = FitSchedule(refit_period) if fit is not None else None
    allowed = ALLOWED_MODULES | local_modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise StrategyLoadError(
                    "strategy uses a relative import; import package modules absolutely"
                )
            modules = [str(node.module or "").split(".", 1)[0]]
        else:
            modules = []
        unsupported = sorted(set(modules).difference(allowed))
        if unsupported:
            raise StrategyLoadError(f"strategy imports unsupported module: {unsupported[0]}")
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            raise StrategyLoadError(f"strategy calls forbidden builtin: {node.func.id}")
        if not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method in FORBIDDEN_ATTRIBUTES:
            raise StrategyLoadError(f"strategy calls unsupported external I/O method: {method}")
        if method in ROOTED_READS:
            roots = READ_ROOTS
        elif method in ROOTED_WRITES:
            roots = WRITE_ROOTS
        else:
            continue
        _reject_unusable_paths(node, method=method, roots=roots)
    return fit_schedule


def validate_strategy_package(main_py: str | Path) -> FitSchedule | None:
    """Preflight ``main.py`` and every ``.py`` file below its directory.

    The directory of ``main.py`` is the strategy package: ``main.py`` is the
    entry module and may import sibling modules and packages absolutely
    (``import lib.features``). Every file gets the same import and I/O rules;
    hidden and ``__pycache__`` paths are not part of the package.
    """
    main_py = Path(main_py)
    if not main_py.is_file():
        raise StrategyLoadError(f"strategy file does not exist: {main_py}")
    root = main_py.parent
    local_modules = package_modules(root)
    clash = sorted(local_modules & ALLOWED_MODULES)
    if clash:
        raise StrategyLoadError(f"strategy package shadows a library module: {clash[0]}")
    schedule: FitSchedule | None = None
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise StrategyLoadError(f"{relative}: cannot read strategy source: {exc}") from exc
        try:
            result = validate_strategy_source(
                source,
                filename=str(relative),
                local_modules=local_modules,
                entrypoints=path == main_py,
            )
        except StrategyLoadError as exc:
            raise StrategyLoadError(f"{relative}: {exc}") from exc
        if path == main_py:
            schedule = result
    return schedule


def package_modules(root: Path) -> frozenset[str]:
    """Top-level module and package names importable from a strategy directory."""
    names = set()
    if not root.is_dir():
        return frozenset()
    for entry in root.iterdir():
        if entry.name.startswith(".") or entry.name == "__pycache__":
            continue
        if entry.is_dir():
            names.add(entry.name)
        elif entry.suffix == ".py" and entry.name != ENTRYPOINT_NAME:
            names.add(entry.stem)
    return frozenset(names)


def _entrypoint(tree: ast.Module, name: str, *, required: bool) -> ast.FunctionDef | None:
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if not definitions and not required:
        return None
    if len(definitions) != 1 or isinstance(definitions[0], ast.AsyncFunctionDef):
        raise StrategyLoadError(f"strategy must define exactly one synchronous {name}(context)")
    definition = definitions[0]
    if len(definition.args.args) != 1:
        raise StrategyLoadError(f"{name} must accept exactly one context argument")
    return definition


def _refit_period(tree: ast.Module, *, has_fit: bool) -> str | None:
    """Read the module-level ``REFIT_PERIOD`` constant declaration, if any."""

    assignments = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(target, ast.Name) and target.id == REFIT_PERIOD_NAME for target in targets):
            assignments.append(node)
    if not assignments:
        return None
    if not has_fit:
        raise StrategyLoadError(f"{REFIT_PERIOD_NAME} requires a fit(context) entrypoint")
    node = assignments[0]
    value = node.value
    if (
        len(assignments) != 1
        or (isinstance(node, ast.Assign) and len(node.targets) != 1)
        or not isinstance(value, ast.Constant)
        or (value.value is not None and not isinstance(value.value, str))
    ):
        raise StrategyLoadError(
            f"{REFIT_PERIOD_NAME} must be assigned once to a period string literal or None"
        )
    try:
        FitSchedule(value.value)
    except StrategyContractError as exc:
        raise StrategyLoadError(f"{REFIT_PERIOD_NAME} {exc}") from exc
    return value.value


def _reject_unusable_paths(node: ast.Call, *, method: str, roots: tuple[str, ...]) -> None:
    """Refuse the path arguments the source itself shows cannot work.

    Every positional argument and every keyword value is read as a possible
    path, because the path sits first in ``np.save(path, obj)``, second in
    ``torch.save(obj, path)`` and behind a name in ``np.load(file=...)``, and
    two errors are refused: a literal absolute path, which
    addresses the host or the Agent sandbox instead of the replay container,
    and a write rooted at a context directory that is mounted read-only.
    Everything else — a name, a helper call, a constant, a loop variable —
    passes, since the source cannot say where it points.
    """

    allowed = ", ".join(f"context.{root}" for root in roots)
    for argument in (*node.args, *(keyword.value for keyword in node.keywords)):
        head = _leading_path_atom(argument)
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            if head.value.startswith("/") or "/mnt/" in head.value:
                raise StrategyLoadError(
                    f"strategy passes an absolute path literal to {method}: "
                    f"{head.value!r}; a strategy reaches files only through the "
                    f"directories on its context ({allowed}), and may build that "
                    "path any way it likes"
                )
        elif isinstance(head, ast.Attribute) and head.attr in READ_ROOTS and head.attr not in roots:
            raise StrategyLoadError(
                f"strategy may not {method} below context.{head.attr}: that "
                f"directory is mounted read-only; write below {allowed}"
            )


def _leading_path_atom(node: ast.AST) -> ast.AST:
    """The leftmost atom of a path expression.

    Concatenations and f-strings are followed to whatever starts the string,
    so ``context.state_dir + "/w.npy"`` and ``f"{context.state_dir}/w.npy"``
    both resolve to the ``state_dir`` attribute and ``"/tmp/" + name`` to the
    literal that starts it. Anything else is its own atom.
    """

    while True:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            node = node.left
        elif isinstance(node, ast.JoinedStr) and node.values:
            node = node.values[0]
        elif isinstance(node, ast.FormattedValue):
            node = node.value
        else:
            return node


def load_strategy_module(path: str | Path) -> LoadedStrategy:
    """Validate the package around ``main.py`` and import its entry module.

    The package directory is importable only while the entry module executes,
    and the modules it pulled in are forgotten from ``sys.modules`` afterwards
    (the entry module keeps its own references), so several strategies whose
    helper modules share names can be loaded into one host process. Bytecode
    caches are never written next to the artifact.
    """
    strategy_path = Path(path).resolve()
    if not strategy_path.is_file():
        raise StrategyLoadError(f"strategy file does not exist: {strategy_path}")
    fit_schedule = validate_strategy_package(strategy_path)
    spec = importlib.util.spec_from_file_location("autotrade_user_strategy", strategy_path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"cannot load strategy: {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    root = str(strategy_path.parent)
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, root)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise StrategyLoadError(f"strategy import failed: {exc}") from exc
    finally:
        sys.path.remove(root)
        sys.dont_write_bytecode = write_bytecode
        _forget_package_modules(root)
    strategy = getattr(module, "generate_orders", None)
    if not callable(strategy):
        raise StrategyLoadError("strategy does not expose generate_orders")
    fit = None
    if fit_schedule is not None:
        fit = getattr(module, "fit", None)
        if not callable(fit):
            raise StrategyLoadError("strategy does not expose fit")
    return LoadedStrategy(strategy, fit, fit_schedule)


def _forget_package_modules(root: str) -> None:
    prefix = root + os.sep
    for name, module in list(sys.modules.items()):
        # Read the module's own namespace: some library modules answer any
        # attribute lookup (torch._classes), so getattr would fabricate a path.
        namespace = getattr(module, "__dict__", None) or {}
        locations = [namespace.get("__file__")]
        try:
            locations.extend(namespace.get("__path__") or ())
        except TypeError:
            pass
        if any(isinstance(location, str) and location.startswith(prefix) for location in locations):
            del sys.modules[name]


def load_strategy(path: str | Path) -> StrategyFunction:
    """Load the bare ``generate_orders`` of a strategy that declares no ``fit``."""

    loaded = load_strategy_module(path)
    if loaded.fit is not None:
        raise StrategyLoadError(
            "strategy defines fit(context); load it with load_strategy_module and a state_dir"
        )
    return loaded.generate_orders


__all__ = [
    "ALLOWED_MODULES",
    "LoadedStrategy",
    "StrategyLoadError",
    "load_strategy",
    "load_strategy_module",
    "package_modules",
    "validate_strategy_package",
    "validate_strategy_source",
]
