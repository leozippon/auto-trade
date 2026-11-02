"""Strategy entrypoint. This file does not trade.

``generate_orders`` is the required entry and returns no orders. The session
replaces this file with its own strategy before a research result means anything.
"""


def generate_orders(context):
    return []
