"""Whether the announcement titles a Paper decision reads have landed.

A book whose data selection mounts announcement titles (``anns_d``) sees them
through its replay slot, cut at the last run of their landing job that
finished before the inference time (``contracts.TEXT_DATASET_REFRESH_NODES``):
for an 08:30 decision, the 23:15 run of the evening before. The view assumes
that run landed, as the research replays the strategy was validated on did.
The job is separate from the rest of the lake (it has its own token), so it
can fail while the lake still commits, and the view would then lack the newest
titles without anything saying so. A decision is therefore refused unless the
job's latest success, as the scheduler records it, covers the end date the
scheduler gives that run and finished after that run's launch. The sessions
before it still settle: settling reads no titles.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from autotrade.data_sources.tushare.common import (
    NO_MUTATION_FAILURE_EXIT_CODE,
    OFFICIAL_TOKEN_ENV,
)
from autotrade.data_sources.tushare.cron_update import (
    DEFAULT_CONFIG,
    load_config,
    read_job_state,
    resolve_job_end_date,
)
from autotrade.environment.data.contracts import (
    REFRESH_NODES,
    TEXT_DATASET_REFRESH_NODES,
    node_visible_cutoff,
)
from autotrade.environment.data.snapshot import SnapshotConfig
from autotrade.environment.strategy import CN_TZ

from .engine import PaperTitlesNotLanded

TITLES = "anns_d"


def require_landed_titles(config: SnapshotConfig, inference_at: datetime, repo_root: Path) -> None:
    """Raise ``PaperTitlesNotLanded`` unless every title the decision at
    ``inference_at`` assumes has landed; a book without titles passes.

    ``repo_root`` is the repository whose scheduler lands the titles: its
    schedule gives the end date of the due run, its job state the latest
    success.
    """

    if not (config.replay_include_text and TITLES in config.text_datasets):
        return
    (job,) = TEXT_DATASET_REFRESH_NODES[TITLES]
    # The launch of the run the view's title cutoff stands for. The node runs
    # every calendar evening, as the crontab does, so a Monday or post-holiday
    # decision needs the run of its own previous evening.
    node = node_visible_cutoff(REFRESH_NODES[job], inference_at)
    schedule = load_config(repo_root / DEFAULT_CONFIG)
    root = repo_root / schedule.get("repo_root", ".")
    spec = schedule["jobs"][job]
    # The end date the scheduler gives the run launched then (build_context).
    target = (node.date() - timedelta(days=int(spec.get("end_date_offset_days", 0)))).strftime("%Y%m%d")
    needed = resolve_job_end_date(spec, root, schedule.get("default_raw_dir", "data/raw"), target)
    state = read_job_state(job, root)
    ok = state.get("status") == "ok"
    # The latest success must end on or after the due run's end date and have
    # finished after its launch: an earlier run of the same window fetched
    # only the titles published before it, and the scheduler skips the due
    # run as already done.
    end, at = (state.get("end_date"), state.get("updated_at")) if ok else (
        state.get("last_ok_end_date"), state.get("last_ok_at")
    )
    if end and at and str(end) >= needed and datetime.fromisoformat(str(at)) >= node:
        return
    # The due run itself, forced past the scheduler's skip of a window it
    # already ran; a later run's window then differs, so it is not skipped.
    rerun = f"`python scripts/data/tushare_cron_update.py --job {job} --end-date {needed} --force-run`"
    then = "成功后再重跑 Paper（下面的命令）。"
    last = f"{_local(str(at))} 结束、覆盖到 {end} 的那一次" if end and at else "状态里没有记录"
    if not state or ok:
        facts = f"它最近一次成功是{last}，不包括这一轮" if state else "它没有任何运行记录"
        action = f"处理：确认标题任务已装进 crontab，补跑这一轮 {rerun}，{then}"
    else:
        error = " ".join(str(state.get("error") or "").split())
        facts = (
            f"它最近一次运行（{_local(state['updated_at'])} 结束，区间 {state.get('start_date')}–{state.get('end_date')}）"
            f"失败，返回码 {state.get('returncode')}：{error}；最近一次成功：{last}"
        )
        if state.get("returncode") == NO_MUTATION_FAILURE_EXIT_CODE:
            action = (
                f"处理：标题任务用 {OFFICIAL_TOKEN_ENV} 访问官方 TuShare 服务，没有取到权限（没有设置或已经失效）。"
                f"把有效的 {OFFICIAL_TOKEN_ENV} 写进仓库根目录的 .env，补跑这一轮 {rerun}，{then}"
            )
        else:
            action = f"处理：按任务日志 {state.get('log_path')} 排除原因，补跑这一轮 {rerun}，{then}"
    raise PaperTitlesNotLanded(
        f"公告标题没有落地：{inference_at:%Y-%m-%d %H:%M} 的决策要读到标题任务 {job} "
        f"{node:%Y-%m-%d %H:%M} 那一轮落地的公告标题（成功运行要覆盖到 {needed}），但{facts}",
        action,
    )


def _local(stamp: str) -> str:
    return datetime.fromisoformat(stamp).astimezone(CN_TZ).strftime("%Y-%m-%d %H:%M")


__all__ = ["TITLES", "require_landed_titles"]
