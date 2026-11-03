"""Shared primitives every subpackage may import: the market timezone, JSON
record I/O and the strict dataclass record pair, and the experiment-id rule.

This package imports nothing else from ``autotrade``, so any layer can depend
on it. It holds helpers that would otherwise be repeated per subpackage, not
types: a record type stays with the subpackage that owns it.
"""
