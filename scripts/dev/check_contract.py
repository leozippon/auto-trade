"""Run the sandbox image's baked strategy loader against three strategy sources.

The check reads the loader the image bakes, not this checkout's, so it runs
inside the image from standard input and refuses to run anywhere else:

    docker run --rm -i --pull never --network none --entrypoint python \
        autotrade-sandbox:latest - < scripts/dev/check_contract.py

It prints the image's contract fingerprint and PASS or REJECT for each source;
exit status 1 when the loader rejects any of them.
"""

import sys

from autotrade.environment.strategy_loader import validate_strategy_source

FINGERPRINT = "/opt/autotrade/SOURCE_FINGERPRINT"
SOURCES = {
    "literal_dir_read": (
        "import pandas as pd\n"
        "def generate_orders(context):\n"
        '    frame = pd.read_parquet(context.snapshot_dir + "/daily")\n'
        "    return []\n"
    ),
    "helper_built_path": (
        "import pandas as pd\n"
        "def _path(root, name):\n"
        '    return root + "/" + name\n'
        "def generate_orders(context):\n"
        '    frame = pd.read_parquet(_path(context.snapshot_dir, "daily"))\n'
        "    return []\n"
    ),
    "torch_save_context_path": (
        "import torch\n"
        'REFIT_PERIOD = "quarter"\n'
        "def fit(context):\n"
        '    torch.save({"w": 1}, context.state_dir + "/m.pt")\n'
        "def generate_orders(context):\n"
        "    return []\n"
    ),
}

try:
    with open(FINGERPRINT, encoding="utf-8") as handle:
        fingerprint = handle.read().strip()
except OSError:
    sys.exit(f"{FINGERPRINT} is absent: run this inside the sandbox image (see the docstring)")
print("SOURCE_FINGERPRINT:", fingerprint)
failed = 0
for name, source in SOURCES.items():
    try:
        validate_strategy_source(source)
    except Exception as exc:  # noqa: BLE001 - report whatever the loader says
        failed += 1
        print(f"{name}: REJECT ({type(exc).__name__}: {exc})")
    else:
        print(f"{name}: PASS")
sys.exit(1 if failed else 0)
