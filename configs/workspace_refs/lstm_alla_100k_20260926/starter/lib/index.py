"""The arm's benchmark index code, read from the macro domain.

`INDEX_CODE` = 000852.SH is this arm's benchmark (`benchmark_index`):
`lib/panel.py::_read_benchmark` reads its adjusted-free daily opens and closes
from the macro domain's `index_daily` dataset, and the label (`lib/label.py`)
takes its residual against that same series, so the label and the adjudicating
neutralisation stay on one reference.

`latest` is kept because `lib/trade.py` reads the book's section metadata from
it: this arm's panel carries a fake one-section `sections` frame (every visible
name, zero weight), so the book's reported index weight is 0.0 by construction.
"""

INDEX_CODE = "000852.SH"


def latest(frame):
    """(members of the newest visible section, that section's trade_date)."""

    section = str(frame["trade_date"].max())
    rows = frame[frame["trade_date"] == section]
    return rows[["ts_code", "weight"]].reset_index(drop=True), section
