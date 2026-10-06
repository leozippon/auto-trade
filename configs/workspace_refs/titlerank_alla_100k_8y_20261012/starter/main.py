"""Entry module: a ranker trained from scratch on announcement titles, in a 20-seat event book.

`lib/knobs.py` holds every switch; `refs/README.md` is the authority for which
legs may run and in which batch. With every switch at its first value this
package is `nn`:

    nn     every name's titles that became visible since the last weekly
           review form one unit (lib/ranker.py); a character-level
           convolutional net (lib/models.py), trained from scratch on the
           card each quarter on the trailing three years of units, scores
           the unit for its size-matched return over the next 40 trading
           days (lib/label.py); the units in the top 1 % of the model's
           validation scores are the review's events, which fill empty
           seats of a 20-seat equal-cash book highest score first, bought at
           the 15:00 close and sold at 09:30 of the first review 40 trading
           days later (lib/book.py)

`fit` runs before every decision (REFIT_PERIOD below): it refits in a new
quarter and, on a review day, records the review's events under
`context.state_dir` (lib/events.py); `generate_orders` trades from those
records. The training seed is `knobs.SEED`.
"""

from lib import book, events

REFIT_PERIOD = "day"


def fit(context):
    events.update(context)


def generate_orders(context):
    return book.run(context)
