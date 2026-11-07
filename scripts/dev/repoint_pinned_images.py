#!/usr/bin/env python3
"""Re-point every experiment's pinned sandbox tag at its freshly rebuilt base image.

``sandbox_images.prepare_experiment_sandbox_image`` clones the configured base
image once per experiment into an immutable tag, records both in
``hitl/sandbox_image.json`` and resolves the tag by name at ``docker run`` time,
so an experiment is re-pointed by tagging its rebuilt base under its own tag. The
state file is never rewritten. Each experiment prints its tag and the image id it
names before and after (``<missing>`` where this host lacks the tag, which the
re-tag then creates). Without ``--apply`` only the plan is printed. A state of
another kind than ``base_clone`` is a failure; exit status 1 on any failure.

    python scripts/dev/repoint_pinned_images.py [--apply] [--experiments DIR]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.sandbox_images import SANDBOX_IMAGE_STATE_NAME


def image_id(ref: str) -> str:
    completed = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", ref],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "<missing>"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="re-tag; without it only the plan is printed")
    parser.add_argument(
        "--experiments",
        type=Path,
        default=REPO_ROOT / "experiments",
        help="the experiments root (default: this checkout's experiments/)",
    )
    args = parser.parse_args(argv)
    states = sorted(args.experiments.glob(f"*/hitl/{SANDBOX_IMAGE_STATE_NAME}"))
    if not states:
        print(f"no experiment under {args.experiments} pins a sandbox image")
    failures = 0
    for path in states:
        state = json.loads(path.read_text(encoding="utf-8"))
        active = str(state["image_ref"])
        print(f"\n== {state['experiment_id']} ({state['kind']})\n  active {active}\n  before {image_id(active)}")
        if state["kind"] != "base_clone":
            print("  FAILED: unsupported kind")
            failures += 1
            continue
        command = ["docker", "tag", str(state["base_image_ref"]), active]
        print("  $", " ".join(command))
        if args.apply:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode:
                print("  FAILED:", completed.stderr.strip()[-2000:])
                failures += 1
        print("  after ", image_id(active))
    print("\nfailures:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
