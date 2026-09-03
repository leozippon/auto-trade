#!/usr/bin/env python
"""Create the 2026-09-10 round: five research directions on the quarterly walk-forward design.

This script lives in scripts/experiments/ and supersedes the gitignored
logs/launch/ location where earlier round definitions were stranded. One round
definition is kept here at a time; superseded rounds stay in git history.

The slate is five arms: factor_cs, explore_platform, explore_github,
open_mechanism and corner_cases are all stopped and restarted here under the new
schedule, each keeping its reference pack and model settings. Nothing is
inherited: corner_cases restarts from the template too, and the previous run
(corner_cases_20260907) stays archived in its own directory.

What this round changes is the research design. The Development window is read
in quarters and every Fold is validated on the trailing four quarters ending at
its own quarter, so the window steps forward one quarter at a time
(fold_2022Q4 ... fold_2025Q4: 13 Folds, 12 walk-forward transitions) and only
the last quarter of each window is new. One Epoch is enough because the chain
never revisits a window it has already seen. The account is the researcher's
real capital (CNY 100k, where the CNY 5 minimum commission and lot sizes bite),
graduation additionally requires the Held-out excess to survive twice the
profile slippage and at least 20 closed round trips, and the per-Fold budgets
are halved because three of the four quarters and the parent's own result are
already known at each step.

Everything else stays on the console creation defaults (no Test stage, explicit
held-out range, Meta between every two Folds, 24-month input window, unfiltered
universe, intraday domain off, curated+graduated operating memory, one hour per
fit(context) and 180 s per decision, local Qwen for every role, xhigh, auto).

WEB_CREATE_DEFAULTS is the base -- no stale params template is read -- and
EXPECTED_DEFAULTS pins the values this round depends on, so a drift in the
console defaults stops the script instead of silently re-scoping five
experiments.

Every parameter set is validated offline with every request-level check the
console applies on POST /api/experiments (ExperimentManager.create_experiment's
key, id and stamp rules plus the worker's own resolve_worker_options
pre-flight), and additionally refuses a directive the PRIOR calendar policy
would reject. The console's deployment-state checks -- an experiment directory
that already exists and a free running slot (five arms fill
webui.manager.MAX_RUNNING_EXPERIMENTS exactly, so every experiment of the
previous round, corner_cases_20260907 included, has to be stopped first) -- can
only be decided against the live server and still happen at POST time, so
--dry-run answers whether the parameters are acceptable, not whether the server
will take the experiment now. It needs no PIT views either: the pre-flight
deliberately skips the calendar-dependent fold schedule and every data root, so
building the quarterly views is a separate step
(scripts/data/prebuild_pit_views_seed.py) and never a hidden requirement here.

Usage:
  PYTHONPATH=src ~/miniconda3/envs/quant/bin/python \
      scripts/experiments/create_round_20260910.py <port> [--dry-run] [experiment_id ...]
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.tools.prior_policy import calendar_policy_violation
from autotrade.pipelines.hitl_state import (
    WEB_CLOSED_PARAMS,
    WEB_CREATE_DEFAULTS,
    WEB_INTERNAL_PARAMS,
    WEB_REQUIRED_PARAMS,
)
from autotrade.pipelines.worker import resolve_worker_options

# The console's own id rule and roots; importing them keeps this script from
# growing a second copy of the create contract.
from autotrade.webui.manager import _ID as EXPERIMENT_ID_RE

EXPERIMENTS_ROOT = REPO_ROOT / "experiments"

# Console defaults this round relies on. Values, not commentary: if the console
# changes any of them the round has to be re-decided, not silently re-run.
EXPECTED_DEFAULTS: dict[str, object] = {
    "test_stage": False,
    "heldout_first_period": "20260101..20260630",
    "heldout_last_period": "20260101..20260630",
    "meta_learning_fold_interval": 1,
    "window_months": 24,
    "include_fundamentals": True,
    "include_macro": True,
    "include_events": True,
    "include_text": True,
    "include_intraday": False,
    "screen_exclude_st": False,
    "screen_exclude_new_listed_days": 0,
    "screen_boards": (),
    "screen_min_circ_mv_yi": None,
    "screen_max_circ_mv_yi": None,
    # Derived from SandboxLimits.fit_timeout_seconds; the directives promise the
    # Agent a fit budget of this size.
    "strategy_fit_timeout_seconds": 3600,
    "operating_memory": "curated+graduated",
    "initial_control_mode": "auto",
    "reasoning_effort": "xhigh",
    "inference_time": "08:30",
    "strategy_period": "day",
    "inherit_from": "",
    "model": "qwen-3.8-27b-fp8",
    "meta_model": "qwen-3.8-27b-fp8",
    "nl_model": "qwen-3.8-27b-fp8",
    "compact_model": "qwen-3.8-27b-fp8",
}

# What this round decides for all five arms. Values, not commentary: the
# schedule, the account, the graduation gates and the per-step budgets.
COMMON_OVERRIDES: dict[str, object] = {
    # No experiment takes an L20.
    "gpu_count": 0,
    # Quarterly walk-forward: 13 Folds, each validated on the trailing four
    # quarters ending at its own quarter, so every step adds exactly one new
    # quarter and the chain never revisits a window.
    "fold_period": "quarter",
    "validation_periods": 4,
    "development_first_period": "2022Q1",
    "development_last_period": "2025Q4",
    # One pass: with 12 forward transitions there is nothing a second pass over
    # the same windows could add that would still be out of sample.
    "epochs": 1,
    # The researcher's real account, where the CNY 5 minimum commission and lot
    # sizes are a real cost rather than a rounding error.
    "initial_cash": 100_000,
    # Graduation must survive twice the profile slippage and rest on more than a
    # handful of trades.
    "cost_stress_multiplier": 2.0,
    "heldout_min_trades": 20,
    # Halved per-step budgets: at each step three of the four quarters and the
    # parent's own result on the window are already known.
    "max_fold_minutes": 360,
    "max_steps_per_fold": 15,
    "max_backtests_per_fold": 15,
    "max_llm_calls": 800,
}

# Five lines shared by every directive. They say how the session should be
# spent, what the host already ran, how candidates are pre-registered and
# compared, that a hypothesis with parameters gets fitted, and what decides
# between two candidates. Direction, not procedure: no calendar labels, no
# per-tool recipes.
SESSION_LINE = (
    "本 Fold 的预算是推理墙钟 370 分钟（运行事实 `budgets.deadline_seconds`：360 分钟主截止加最后 "
    "10 分钟收尾宽限 `deadline_grace_seconds`）、15 个 Step、15 次回测、800 次模型调用"
    "（回测墙钟独立计时并回补）。"
    "预算是用来持续探索的：一批候选跑出好结果，意味着可以进入下一轮细化与加固，而不是提前收工；"
    "把整段预算花在一条不断收敛的探索链上，直到时间或回测次数真的用完。"
)
PARENT_CONTROL_LINE = (
    "本 Fold 的验证窗口是截至本季的连续四个季度，其中只有最后一个季度是上一 Fold 之后新出现的。"
    "父产物是已冻结策略时，宿主已在会话开始前把它原样在这整个窗口上跑完一次完整 Validation，"
    "它是 Step 树的第一个节点、不占用上述预算；它的最后一个子窗口就是父策略在这个新季度上的样本外记录，"
    "更早的子窗口父策略已经见过。直接把它当作对照基线，不要再花一次回测重跑父本。"
    "父产物是初始模板时没有这个节点"
    "（运行事实 `parent_control` 为空、`artifact_contract.parent.parent_control_available` 为假），"
    "需要基线就自己跑一次并计入上述预算。"
)
PREREGISTER_LINE = (
    "互斥候选一律先预登记：写清机制、预期方向与证伪条件，再用 batch_validate 在同一父节点下并排跑完整 Validation，"
    "看到结果之后才做取舍；下一批候选从上一批的结论里长出来，被证伪的方向如实结束并换下一条。"
)
FITTING_LINE = (
    "假设里有参数就把参数拟合出来，而不是手调常数：训练写在 fit(context) 里、只用当时可见的 PIT 窗口，"
    "结果经 context.state_dir 交给 generate_orders。output/ 是一个可以拆分模块的包，运行时库含 scipy、"
    "sklearn、lightgbm、xgboost、statsmodels 与 CPU 版 torch；调仓与重训的节奏（模块级 REFIT_PERIOD）"
    "由策略自己决定并在设计中给出依据，环境只规定何时询问策略。"
)
ROBUSTNESS_LINE = (
    "并列候选之间的取舍看稳健，而不是看简单：子窗口内的一致性、中性化后的超额、对参数与阈值的敏感度，"
    "比单一总量指标更有说服力；不同口径不一致时如实说明，不得只挑有利口径。"
)

# Verbatim from experiments/open_mechanism_20260903/hitl/params.json
# ("fold_exploration_directive"), copied here because that experiment is
# archived out of the repository before this round starts.
OPEN_MECHANISM_PRIOR_DIRECTIVE = (
    "跳开因子库、横截面打分、线性排序和常规技术指标堆叠。本实验无参考仓库。只根据当前 PIT 可见结构自行提出"
    "一种不同机制，写成最小可执行策略，并用完整 Validation 证伪。事件状态、微观结构、制度约束、行为路径、"
    "非对称执行只是类型，不是指定答案。本轮以机制新颖与可证伪为目标，不要求稳定或正收益；证伪后如实结束该方向。"
    "仍须遵守 PIT、禁止硬编码股票或日期、真实回测 ABI 与诚实失败。本指令不放宽提交合同、回撤硬限制或 finish_fold。"
)

ROUND: dict[str, dict[str, object]] = {
    "factor_cs_20260910": {
        "workspace_reference": "configs/workspace_refs/factor_cs_20260826",
        "fold_exploration_directive": "\n".join(
            [
                "方向：在未筛选的全 A 股票池上做截面多因子选股，每一步只推进一个可分离的因子家族，"
                "正式产物写在 output/ 包内。",
                "先读 refs/README.md 与 exploration-plan.md，再用 PIT parquet 自算因子并核对覆盖率与单位；"
                "参考包阅读、因子重算与 IC 统计适合交给子代理并行完成，你的精力放在设计、决策与验收上。",
                "股票池不做任何 ST、板块、次新、市值或价格筛选；可交易性、停牌与涨跌停由策略自己处理并说明理由。",
                SESSION_LINE,
                PARENT_CONTROL_LINE,
                PREREGISTER_LINE,
                FITTING_LINE,
                ROBUSTNESS_LINE,
                "禁止克隆父策略、禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足"
                " available_at <= 推断时点，财务用 available_at。可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "explore_platform_strategies_20260910": {
        "workspace_reference": "configs/workspace_refs/explore_platform_strategies",
        "fold_exploration_directive": "\n".join(
            [
                "方向：把 refs 里已筛好的股票讨论平台机制重写成本项目 ABI 下的可执行策略；先读 refs/README.md、"
                "pit-field-map.md 与 playbooks.md，按其中的优先级推进，一次只验证一个机制。",
                "预登记的机制先过离线筛查：事件计数、覆盖窗口、各组前瞻收益的符号都要看过，"
                "再决定它值不值得一次完整 Validation；不要一开始就把几个弱机制堆成不透明打分。",
                "股票池未经任何筛选，可交易性、停牌与涨跌停由策略自理。沙箱无网络，任何时候都不得抓取站点数据。",
                SESSION_LINE,
                PARENT_CONTROL_LINE,
                PREREGISTER_LINE,
                FITTING_LINE,
                ROBUSTNESS_LINE,
                "禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足 available_at <= 推断时点。"
                "可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "explore_github_strategies_20260910": {
        "workspace_reference": "configs/workspace_refs/explore_github_strategies",
        "fold_exploration_directive": "\n".join(
            [
                "方向：把 refs 里已筛好的 GitHub A 股策略思路在本项目 ABI 下重写（是重写，不是移植代码）；"
                "先读 refs/README.md、playbook.md 与 screening.md，再看 vendor/*/SOURCE.md 与相邻公式摘录。",
                "一次只验证一个思路；源仓库的数据库、下载器、调度器、broker、日志与框架适配一律删除，"
                "撮合、T+1、费用与涨跌停属于环境，策略只发订单意图。",
                "不要把源仓库 README 的收益数字当成预期，只用本项目的 Validation 复现其行为。"
                "股票池未经任何筛选，可交易性、停牌与涨跌停由策略自理。",
                SESSION_LINE,
                PARENT_CONTROL_LINE,
                PREREGISTER_LINE,
                FITTING_LINE,
                ROBUSTNESS_LINE,
                "禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足 available_at <= 推断时点。"
                "可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "corner_cases_20260910": {
        # Same direction text as corner_cases_20260907; the strategy is not
        # inherited, so the arm restarts from the template on the quarterly
        # schedule while that run's artifacts stay archived in its own directory.
        "workspace_reference": "configs/workspace_refs/corner_cases_20260907",
        "fold_exploration_directive": "\n".join(
            [
                "方向：专注 A 股市场的角落状态——涨跌停板动力学、停牌复牌再定价、ST 与摘帽戴帽、新股上市初期、"
                "除权除息、解禁事件窗、极端市场日宽度——把普通截面策略过滤掉或处理错的边缘状态写成可证伪的最小机制，"
                "每一步只推进一个机制，正式产物写在 output/ 包内。",
                "先读 refs/README.md 与 exploration-plan.md，再按 pit-field-map.md 核对每个字段的可见时间与单位；"
                "先在输入窗做各家族的可交易事件数普查，事件数不足的机制如实判「不可测」，诚实的零结果与被证伪的方向"
                "都是本方向的合格产出。",
                "角落状态的可执行性必须显式论证：封死的涨停买不进、复牌当日常被拒单、跌停票出不去、新股首日没有可见行情；"
                "每个候选都要写明真实可达的入场与出场路径并汇报拒单。角落组合天然集中，仓位上限、篮子分散与无信号日持币"
                "必须把最大回撤守在硬限之内。",
                "股票池不做任何 ST、板块、次新、市值或价格筛选；可交易性、停牌与涨跌停由策略自己处理并说明理由。",
                SESSION_LINE,
                PARENT_CONTROL_LINE,
                PREREGISTER_LINE,
                FITTING_LINE,
                ROBUSTNESS_LINE,
                "禁止克隆父策略、禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足"
                " available_at <= 推断时点，财务用 available_at。可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "open_mechanism_20260910": {
        # No workspace_reference and no inherit_from: the arm starts from the
        # empty template with nothing mounted but the operating memory.
        "fold_exploration_directive": "\n".join(
            [
                OPEN_MECHANISM_PRIOR_DIRECTIVE,
                "本轮环境提供 output/ 包结构、可选的 fit(context) 与并排跑完整 Validation 的 batch_validate，"
                "可按需使用；它们只是手段，不改变本方向对机制新颖与可证伪的要求。",
                SESSION_LINE,
                PARENT_CONTROL_LINE,
                PREREGISTER_LINE,
                FITTING_LINE,
                ROBUSTNESS_LINE,
            ]
        ),
    },
}

# Reported for every arm on --dry-run: what this round decides, plus the
# defaults it depends on.
REPORT_KEYS = (
    "experiment_id",
    "workspace_reference",
    "inherit_from",
    "gpu_count",
    "reasoning_effort",
    "initial_control_mode",
    "fold_period",
    "development_first_period",
    "development_last_period",
    "validation_periods",
    "test_stage",
    "heldout_first_period",
    "heldout_last_period",
    "epochs",
    "meta_learning_fold_interval",
    "window_months",
    "include_intraday",
    "operating_memory",
    "max_fold_minutes",
    "max_steps_per_fold",
    "max_backtests_per_fold",
    "max_llm_calls",
    "initial_cash",
    "cost_stress_multiplier",
    "heldout_min_trades",
    "strategy_fit_timeout_seconds",
    "inference_time",
    "strategy_period",
    "model",
)


def check_console_defaults() -> None:
    drift = {
        key: (value, WEB_CREATE_DEFAULTS[key])
        for key, value in EXPECTED_DEFAULTS.items()
        if WEB_CREATE_DEFAULTS[key] != value
    }
    if drift:
        raise SystemExit(
            "console creation defaults drifted from what this round assumes; "
            "re-decide the round before creating: "
            + json.dumps(
                {k: {"round": str(v[0]), "console": str(v[1])} for k, v in drift.items()},
                ensure_ascii=False,
            )
        )


def request_params(experiment_id: str) -> dict[str, object]:
    """The create request body: console defaults, the round's overrides, the id."""
    base = {
        key: (list(value) if isinstance(value, tuple) else value)
        for key, value in WEB_CREATE_DEFAULTS.items()
    }
    return {
        **base,
        **COMMON_OVERRIDES,
        **ROUND[experiment_id],
        "experiment_id": experiment_id,
    }


def normalize(params: dict[str, object]) -> dict[str, object]:
    """Run the console's create-time validation offline and return params.json.

    Same order and same request-level checks as
    ExperimentManager.create_experiment: closed keys, unknown keys, required
    keys, the id rule, the console-managed stamp, then the worker's own
    resolve_worker_options pre-flight (which is what actually type-checks every
    knob). The console's duplicate-directory and running-slot checks need the
    live deployment and stay at POST time; the calendar-policy gate below is
    stricter than create.
    """
    closed = sorted(set(params) & WEB_CLOSED_PARAMS)
    if closed:
        raise ValueError("console-managed parameters are not accepted: " + ", ".join(closed))
    unknown = sorted(set(params) - set(WEB_CREATE_DEFAULTS))
    if unknown:
        raise ValueError("unknown experiment parameters: " + ", ".join(unknown))
    merged = {**WEB_CREATE_DEFAULTS, **params}
    missing = sorted(key for key in WEB_REQUIRED_PARAMS if merged.get(key) in (None, ""))
    if missing:
        raise ValueError("missing required experiment parameters: " + ", ".join(missing))
    experiment_id = str(params.get("experiment_id") or "").strip()
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise ValueError("experiment_id must match [A-Za-z0-9][A-Za-z0-9_-]{0,99}")
    directive = str(merged.get("fold_exploration_directive") or "")
    # Not enforced on create, but the same text is injected into PRIOR-facing
    # prompts and can be re-sent through set_directive, which does enforce it.
    violation = calendar_policy_violation(directive)
    if violation:
        raise ValueError(f"fold_exploration_directive {violation}")
    merged.update(
        {
            **WEB_INTERNAL_PARAMS,
            "experiment_id": experiment_id,
            "experiments_root": str(EXPERIMENTS_ROOT),
            "work_root": str(REPO_ROOT / ".runtime/sandboxes"),
            "_creation_surface": "webui",
        }
    )
    resolve_worker_options(
        merged,
        experiment_dir=EXPERIMENTS_ROOT / experiment_id,
        repo_root=REPO_ROOT,
        preflight=True,
    )
    return merged


def post(port: int, params: dict[str, object]) -> bool:
    """POST one create request; report whether the console accepted it."""
    experiment_id = params["experiment_id"]
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/experiments",
        data=json.dumps(params).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            print(experiment_id, response.status, response.read(400).decode("utf-8", "replace"))
            return True
    except urllib.error.HTTPError as exc:
        print(experiment_id, "HTTP", exc.code, exc.read(800).decode("utf-8", "replace"), file=sys.stderr)
    except urllib.error.URLError as exc:
        # No console on that port, or it dropped the connection: an operator
        # error, not a traceback.
        print(experiment_id, "console unreachable:", exc.reason, file=sys.stderr)
    return False


def main() -> int:
    # A mistyped flag must never fall through to the real POST path: without
    # this, --dryrun is read as an experiment-id filter and creates the round.
    mistyped = [
        arg for arg in sys.argv[1:] if arg.startswith("--") and arg != "--dry-run"
    ]
    if mistyped:
        print("unknown option: " + ", ".join(mistyped), file=sys.stderr)
        return 2
    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        raise SystemExit(__doc__)
    port = int(sys.argv[1])
    dry_run = "--dry-run" in sys.argv
    wanted = {arg for arg in sys.argv[2:] if not arg.startswith("--")}
    unknown_ids = sorted(wanted - set(ROUND))
    if unknown_ids:
        raise SystemExit("not in this round: " + ", ".join(unknown_ids))
    check_console_defaults()
    failed: list[str] = []
    for experiment_id in ROUND:
        if wanted and experiment_id not in wanted:
            continue
        params = request_params(experiment_id)
        merged = normalize(params)  # raises before anything is sent
        if dry_run:
            directive = str(merged["fold_exploration_directive"])
            print(json.dumps({key: merged[key] for key in REPORT_KEYS}, ensure_ascii=False))
            print(f"  directive: {len(directive.splitlines())} lines, {len(directive)} chars")
            for line in directive.splitlines():
                print("   |", line)
            continue
        if not post(port, params):
            failed.append(str(experiment_id))
    if failed:
        print("not created: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
