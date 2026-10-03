"""Skeleton book with the placeholder score p0 (lib/score.py); no fit.

Replace lib/score.py with the mechanism under test. A fitted mechanism adds
`fit(context)` and `REFIT_PERIOD` here, as output/README.md describes.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
