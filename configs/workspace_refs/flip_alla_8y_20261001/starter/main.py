"""Entry: the other end of the census's strongly negative scores, on an equal-cash book.

`knobs.CANDIDATE` in lib/knobs.py selects the leg; `refs/README.md` is the
authority for which legs run in which batch and how each is registered:

    f_all   the 21 flipped census scores, percentile-ranked each day and averaged
    f_vol   the volume / turnover family alone (7 scores), flipped
    f_beta  the beta family alone (3 scores), flipped
    c_orig  the 21 scores in the census direction: what the census arms bought
    c_shuf  f_all's values permuted among the scored names each decision day

lib/census.py computes the census scores point in time, lib/score.py ranks,
flips and averages them, lib/data.py is the pool, and lib/trade.py is the
canonical book, copied unchanged. Changing the candidate changes nothing else.
No fit, no state.
"""

from lib import trade


def generate_orders(context):
    return trade.run(context)
