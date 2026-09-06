#!/usr/bin/env python
"""Create the 2026-09-10 round: six research directions on the quarterly walk-forward design.

This script lives in scripts/experiments/ and supersedes the gitignored
logs/launch/ location where earlier round definitions were stranded. One round
definition is kept here at a time; superseded rounds stay in git history.

The slate is six arms, which fill the console's running slots exactly.
factor_cs, explore_platform and explore_github were rebuilt on refreshed
reference packs (the *_20260912 packs: A-share anomaly families instead of the
value/reversal/growth ridge, previous-day mechanisms instead of the falsified
limit-up playbooks, and an Alpha158 + LightGBM baseline to innovate on) after
every arm's first-fold winner proved statistically weak; open_mechanism and
corner_cases keep their earlier definitions. ml_ranker is new: a GPU-assisted
machine-learning ranker arm on the same budgets. Nothing is inherited: every
arm starts from the template.

Every model role of every arm runs on the local qwen-3.8-27b-fp8: Fold parent,
Meta parent, sub-agents, strategy analysis, NL and compaction alike. That is
already the console creation default for all six roles, so this round overrides
none of them and pins them in EXPECTED_DEFAULTS instead. The hosted DeepSeek
models stay selectable in the catalog, but no arm uses one; the arms differ
only in direction, reference pack and GPU request.

ml_ranker is the only arm with gpu_count=1, and that request travels with the
experiment: it attaches a GPU to the Agent's persistent session sandbox
(exploration shell: offline screening and hyper-parameter pre-selection) and to
the strategy container of every formal replay, fit worker included, so fit()
may really train on the GPU. The card is shared with the concurrent replays,
and the same strategy has to replay unchanged in an experiment without one, so
the directive and the pack require the code to probe the device and keep a
working CPU path.

What this round changes is the research design. The Development window is read
in quarters and every Fold is validated on the trailing four quarters ending at
its own quarter, so the window steps forward one quarter at a time
(fold_2022Q4 ... fold_2025Q4: 13 Folds, 12 walk-forward transitions) and only
the last quarter of each window is new. One Epoch is enough because the chain
never revisits a window it has already seen. The account is the researcher's
real capital (CNY 100k, where the CNY 5 minimum commission and lot sizes bite),
graduation additionally requires the Held-out excess to survive twice the
profile slippage and at least 20 closed round trips, and the per-Fold budgets
sit below the yearly-window defaults -- at each step three of the four quarters
and the parent's own result are already known -- while staying wide enough for
several pre-registered rounds in one Fold.

Everything else stays on the console creation defaults (no Test stage, explicit
held-out range, Meta between every two Folds, 24-month input window, unfiltered
universe, intraday domain off, curated+graduated operating memory, one hour per
fit(context) and 180 s per decision, local Qwen for every role, xhigh, auto).

WEB_CREATE_DEFAULTS is the base -- no stale params template is read -- and
EXPECTED_DEFAULTS pins the values this round depends on, so a drift in the
console defaults stops the script instead of silently re-scoping six
experiments.

Every parameter set is validated offline with every request-level check the
console applies on POST /api/experiments (ExperimentManager.create_experiment's
key, id and stamp rules plus the worker's own resolve_worker_options
pre-flight), and additionally refuses a directive the PRIOR calendar policy
would reject. The console's deployment-state checks -- an experiment directory
that already exists and a free running slot (the six arms fill
webui.manager.MAX_RUNNING_EXPERIMENTS exactly, so every experiment of the
previous round has to be stopped first) -- can only be decided against the
live server and still happen at POST time, so --dry-run answers whether the
parameters are acceptable, not whether the server will take the experiment
now. It needs no PIT views either: the pre-flight
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
    # All six model roles. The round runs entirely on the local model and
    # overrides none of them, so the console default is what actually decides
    # them; spelled out as literals on purpose, since a rename of the local
    # model is exactly the drift this has to catch.
    "model": "qwen-3.8-27b-fp8",
    "meta_model": "qwen-3.8-27b-fp8",
    "subagent_model": "qwen-3.8-27b-fp8",
    "analysis_model": "qwen-3.8-27b-fp8",
    "nl_model": "qwen-3.8-27b-fp8",
    "compact_model": "qwen-3.8-27b-fp8",
}

# What this round decides for every arm. Values, not commentary: the
# schedule, the account, the graduation gates and the per-step budgets.
COMMON_OVERRIDES: dict[str, object] = {
    # No GPU. The request travels with the experiment: it reaches the Agent
    # session sandbox and the strategy container of every formal replay alike;
    # ml_ranker overrides this with 1.
    "gpu_count": 0,
    # No model role is overridden: every one of the six already defaults to the
    # local model, and EXPECTED_DEFAULTS pins that so a default drift cannot
    # silently move an arm onto a hosted model.
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
    # Per-Fold budgets: under the yearly-window defaults because at each step
    # three of the four quarters and the parent's own result on the window are
    # already known, but wide enough for several pre-registered rounds -- no
    # code gate forces a round count any more, so the budget has to leave room
    # for the rounds the prompts ask for.
    "max_fold_minutes": 600,
    "max_steps_per_fold": 24,
    "max_backtests_per_fold": 24,
    "max_llm_calls": 1600,
}

# Two lines shared by every directive: what this round's own schedule makes of
# the host's parent control, and the one selection criterion the static prompt
# does not already state. Everything else the directives used to repeat -- the
# budget figures, pre-registration, fitting parameters in fit(context) -- is in
# the static system prompt or the injected run facts, and is not restated here.
PARENT_CONTROL_LINE = (
    "本 Fold 的验证窗口是截至本季的连续四个季度，其中只有最后一个季度是上一 Fold 之后新出现的："
    "父本对照节点的最后一个子窗口才是父策略在这个新季度上的样本外记录，更早的子窗口它已经见过。"
    "比较对照与候选时按这个口径读 `sub_windows`。"
)
ROBUSTNESS_LINE = (
    "并列候选之间还要看对参数与阈值的敏感度：结论在邻域里翻转的候选不算稳健。"
)

# From experiments/open_mechanism_20260903/hitl/params.json
# ("fold_exploration_directive"), copied here because that experiment is
# archived out of the repository before this round starts; only the drawdown
# clause is updated, since the cap is now enforced at graduation.
OPEN_MECHANISM_PRIOR_DIRECTIVE = (
    "跳开因子库、横截面打分、线性排序和常规技术指标堆叠。本实验无参考仓库。只根据当前 PIT 可见结构自行提出"
    "一种不同机制，写成最小可执行策略，并用完整 Validation 证伪。事件状态、微观结构、制度约束、行为路径、"
    "非对称执行只是类型，不是指定答案。本轮以机制新颖与可证伪为目标，不要求稳定或正收益；证伪后如实结束该方向。"
    "仍须遵守 PIT、禁止硬编码股票或日期、真实回测 ABI 与诚实失败。本指令不放宽提交合同、毕业裁决的回撤上限或 finish_fold。"
)

ROUND: dict[str, dict[str, object]] = {
    "factor_cs_20260910": {
        "workspace_reference": "configs/workspace_refs/factor_cs_20260912",
        "fold_exploration_directive": "\n".join(
            [
                "方向：在未筛选的全 A 股票池上做截面选股，但只测 refs 登记的 A 股特有异象家族（流动性与换手、"
                "彩票需求、隔夜-日内分解、筹码浮盈、资金拥挤、基本面动量），每一步只推进一个可分离的家族，"
                "正式产物写在 output/ 包内。",
                "先读 refs/README.md、exploration-plan.md 与 families.md，再按 pit-field-map.md 核对字段的可见时间与单位；"
                "参考包阅读、因子重算与 IC 筛查适合交给子代理并行完成，你的精力放在设计、决策与验收上。",
                "价值/反转/成长的线性打分与小市值暴露已在前几轮被证伪，本轮不得作为候选；所有信号先做规模与 β 中性，"
                "只按中性化超额与规模倾斜判定。家族合成只在 fit(context) 里做（固定 λ 的 ridge 或树模型排序器，"
                "带禁运的时间序列验证），持有期与篮子按 refs 的手数与成本纪律。",
                "股票池不做任何 ST、板块、次新、市值或价格筛选；可交易性、停牌与涨跌停由策略自己处理并说明理由。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止克隆父策略、禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足"
                " available_at <= 推断时点，财务用 available_at。可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "explore_platform_strategies_20260910": {
        "workspace_reference": "configs/workspace_refs/platform_20260912",
        "fold_exploration_directive": "\n".join(
            [
                "方向：股票讨论平台的涨停板类机制已在日频 08:30 决策下被整体证伪，本轮只测 refs 重新登记的六个"
                "用前一日信息就能执行的机制（板块领涨-跟涨溢出、市场宽度情绪周期门控、跌停超卖开盘反转、"
                "解禁后压力释放、放量滞涨与地量、大宗折溢价），一次只验证一个机制，每个机制必须与其无机制的匹配对照同批比较。",
                "先读 refs/README.md、exploration-plan.md 与 playbooks.md，按 pit-field-map.md 核对可见时间、单位与本轮快照里"
                "没有的数据集；预登记的机制先过离线筛查（事件计数、覆盖窗口、各组前瞻收益的符号），再决定值不值得一次完整 Validation。",
                "机制落败后的剩余预算只能用于本包的其他机制或其门控与持有期变体，不得改成通用因子打分，除非该打分在假说里"
                "被登记为对照。股票池未经任何筛选，可交易性、停牌与涨跌停由策略自理。沙箱无网络，任何时候都不得抓取站点数据。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足 available_at <= 推断时点。"
                "可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "explore_github_strategies_20260910": {
        "workspace_reference": "configs/workspace_refs/github_20260912",
        "fold_exploration_directive": "\n".join(
            [
                "方向：在本项目 ABI 内重写 Qlib 风格的 Alpha158 特征 + LightGBM 截面排序器作为有据可查的强基线，"
                "再在它之上做 refs 登记的创新家族（标签口径、特征中性化、跨重训日集成、WorldQuant-101 与国泰君安-191 算子族），"
                "一次只验证一个改动；是重写，不是移植代码。",
                "先读 refs/README.md、exploration-plan.md、alpha158.md 与 families.md；源仓库的数据层、训练器、Recorder 与回测器"
                "一律不搬，撮合、T+1、费用与涨跌停属于环境，策略只发订单意图。不要把 Qlib 文档里的收益表当成预期，"
                "只用本项目的 Validation 复现其行为。",
                "拟合全部放在 fit(context) 内，标签只用推断时已实现的开盘到开盘收益并留禁运，超参网格预先写死并在 fit 内选点；"
                "股票池未经任何筛选，可交易性、停牌与涨跌停由策略自理。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足 available_at <= 推断时点。"
                "可执行指纹必须不同于父策略。",
            ]
        ),
    },
    "ml_ranker_20260910": {
        # The only arm with a GPU. The request reaches the Agent's session
        # sandbox and the strategy container (fit worker included) of every
        # formal replay, so fit() may train on the device; the card is shared
        # with concurrent replays and the strategy must replay unchanged in an
        # experiment without one, so the directive demands a device probe and
        # a working CPU path.
        "gpu_count": 1,
        "workspace_reference": "configs/workspace_refs/ml_ranker_20260912",
        "fold_exploration_directive": "\n".join(
            [
                "方向：用机器学习与深度学习做截面排序器：先在 fit(context) 里训出 LightGBM 基线，再比较 MLP 与序列模型"
                "（GRU 或小型 Transformer，读 20–60 日 K 线与特征序列），每一步只改一个可分离的组件（模型族、标签口径、"
                "中性化、集成、换手控制），正式产物写在 output/ 包内。",
                "先读 refs/README.md、exploration-plan.md、models.md 与 protocol.md。本臂挂了一块 GPU：开发沙箱用它做"
                "离线筛查与超参预选，正式回放的策略容器（含 fit worker）同样能看到它，fit 可以真的在 GPU 上训练；"
                "但这块卡与并发回放共享，同一份策略还要能在没有 GPU 的实验里原样回放，因此代码必须探测设备并保留"
                "可用的 CPU 路径，fit 的墙钟先在沙箱实测、落在 fit 预算内，模型状态只能以 NumPy 数组或 booster 文件"
                "写入 context.state_dir。",
                "反过拟合是硬纪律：滚动重训、带禁运的清洗时间序列验证、预先写死的小网格并在 fit 内选点、按候选数读去膨胀 Sharpe；"
                "标签只用推断时已实现的开盘到开盘收益。持有期与篮子按 refs 的手数与成本纪律。",
                "股票池不做任何 ST、板块、次新、市值或价格筛选；可交易性、停牌与涨跌停由策略自己处理并说明理由。",
                PARENT_CONTROL_LINE,
                ROBUSTNESS_LINE,
                "禁止克隆父策略、禁止把 refs 拷进 output、禁止写死路径与股票代码；每一行输入都必须满足"
                " available_at <= 推断时点。可执行指纹必须不同于父策略。",
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
                "必须把最大回撤守在毕业裁决的回撤上限之内。",
                "股票池不做任何 ST、板块、次新、市值或价格筛选；可交易性、停牌与涨跌停由策略自己处理并说明理由。",
                PARENT_CONTROL_LINE,
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
                PARENT_CONTROL_LINE,
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
    "meta_model",
    "subagent_model",
    "analysis_model",
    "compact_model",
    "nl_model",
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
