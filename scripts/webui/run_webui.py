#!/usr/bin/env python3
"""HITL experiment console server (docs/deployment-documentation.md).

Serves the web console (homepage + experiment detail SPA) and the JSON control
API over the interactive experiment pipeline. Run on the workstation that
hosts the pipeline, data, and Docker. No auth layer: the default bind is
loopback TCP 127.0.0.1:38888 so a public tunnel can reach the console; an
optional Unix domain socket is same-machine only. A non-loopback bind is
refused outright.

  ~/miniconda3/envs/quant/bin/python scripts/webui/run_webui.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

add_repo_src(__file__)

from autotrade.webui.server import is_loopback_host, run


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; loopback only.")
    parser.add_argument("--port", type=int, default=38888, help="Listen port (default 38888).")
    parser.add_argument(
        "--uds",
        type=Path,
        default=None,
        help="Optional same-machine Unix domain socket instead of TCP; access control "
        "is the socket directory's permissions (overrides --host/--port).",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=repo_root / "experiments",
        help="Experiments root directory shared with the pipeline CLIs.",
    )
    args = parser.parse_args(argv)
    if args.uds is None and not is_loopback_host(args.host):
        parser.error(
            "refusing unauthenticated non-loopback WebUI bind; use a protected Unix "
            "socket or a loopback address"
        )
    run(
        repo_root,
        host=args.host,
        port=args.port,
        uds=args.uds,
        experiments_root=args.experiments_root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
