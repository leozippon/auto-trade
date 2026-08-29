"""Statically preflight and load a trusted Agent-authored strategy."""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .strategy import FitSchedule, StrategyContext, StrategyContractError, StrategyFunction

ALLOWED_MODULES = frozenset(
    {"__future__", "collections", "datetime", "decimal", "math", "numpy", "pandas", "statistics"}
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
# The only file I/O a strategy may perform, and only on a path expression rooted
# at one of the context directories: reads below any read-only data root, writes
# below the per-replay state directory that ``fit`` owns.
ROOTED_READS = frozenset({"read_parquet", "load"})
ROOTED_WRITES = frozenset({"to_parquet", "save", "savez", "savez_compressed"})
READ_ROOTS = ("snapshot_dir", "asof_dir", "state_dir", "models_dir")
WRITE_ROOTS = ("state_dir",)
REFIT_PERIOD_NAME = "REFIT_PERIOD"


class StrategyLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedStrategy:
    """The entrypoints one ``main.py`` exposes; ``fit`` is optional."""

    generate_orders: StrategyFunction
    fit: Callable[[StrategyContext], object] | None
    fit_schedule: FitSchedule | None


def validate_strategy_source(source: str, *, filename: str = "main.py") -> FitSchedule | None:
    """Reject common direct capability and external-I/O calls before import.

    Returns the declared ``fit`` schedule (``None`` when ``main.py`` defines no
    ``fit``). This denylist is a convenience check for trusted, reviewed
    strategies, not a sandbox or a security boundary.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise StrategyLoadError(f"invalid strategy syntax: {exc}") from exc
    generate = _entrypoint(tree, "generate_orders", required=True)
    fit = _entrypoint(tree, "fit", required=False)
    refit_period = _refit_period(tree, has_fit=fit is not None)
    context_args = frozenset(
        node.args.args[0].arg for node in (generate, fit) if node is not None
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".", 1)[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [str(node.module or "").split(".", 1)[0]]
        else:
            modules = []
        unsupported = sorted(set(modules).difference(ALLOWED_MODULES))
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
        if not node.args or not _is_context_data_path(
            node.args[0], context_args=context_args, roots=roots
        ):
            allowed = " or ".join(f"context.{root}" for root in roots)
            raise StrategyLoadError(f"strategy may {method} only below {allowed}")
    return FitSchedule(refit_period) if fit is not None else None


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


def _is_context_data_path(
    node: ast.AST, *, context_args: frozenset[str], roots: tuple[str, ...]
) -> bool:
    """Recognize a path expression rooted in one of the named context directories."""

    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id in context_args
            and node.attr in roots
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (
            _is_context_data_path(node.left, context_args=context_args, roots=roots)
            and isinstance(node.right, ast.Constant)
            and isinstance(node.right.value, str)
        )
    return False


def load_strategy_module(path: str | Path) -> LoadedStrategy:
    strategy_path = Path(path).resolve()
    if not strategy_path.is_file():
        raise StrategyLoadError(f"strategy file does not exist: {strategy_path}")
    source = strategy_path.read_text(encoding="utf-8")
    fit_schedule = validate_strategy_source(source, filename=strategy_path.name)
    spec = importlib.util.spec_from_file_location("autotrade_user_strategy", strategy_path)
    if spec is None or spec.loader is None:
        raise StrategyLoadError(f"cannot load strategy: {strategy_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise StrategyLoadError(f"strategy import failed: {exc}") from exc
    strategy = getattr(module, "generate_orders", None)
    if not callable(strategy):
        raise StrategyLoadError("strategy does not expose generate_orders")
    fit = None
    if fit_schedule is not None:
        fit = getattr(module, "fit", None)
        if not callable(fit):
            raise StrategyLoadError("strategy does not expose fit")
    return LoadedStrategy(strategy, fit, fit_schedule)


def load_strategy(path: str | Path) -> StrategyFunction:
    """Load the bare ``generate_orders`` of a strategy that declares no ``fit``."""

    loaded = load_strategy_module(path)
    if loaded.fit is not None:
        raise StrategyLoadError(
            "strategy defines fit(context); load it with load_strategy_module and a state_dir"
        )
    return loaded.generate_orders


__all__ = [
    "LoadedStrategy",
    "StrategyLoadError",
    "load_strategy",
    "load_strategy_module",
    "validate_strategy_source",
]
