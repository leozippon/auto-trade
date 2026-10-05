"""Entry module: the incentive-plan-draft event book and its late-entry control.

`lib/knobs.py` holds every switch; `refs/README.md` is the authority for which
values may run and in which batch. With every switch at its first value this
package is `b16`:

    b16    the first restricted-stock or option plan draft a company announces
           after 90 quiet days (lib/titles.py), bought at the 15:00 close of
           the first weekly review after the title becomes visible, oldest
           first into 16 equal-cash seats, sold at 09:30 of the first review
           40 trading days later (lib/book.py)

No fit, no state, no seed: the book is a function of the point-in-time view
at each weekly review, and one replay characterises a configuration.
"""

from lib import book


def generate_orders(context):
    return book.run(context)
