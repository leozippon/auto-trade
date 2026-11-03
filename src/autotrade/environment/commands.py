"""The result of one bounded command, beside the runner that produces it.

``executor._run_limited_capture`` builds it for every sandbox command; the tool
layer (``tools.base``, which re-exports it) hands it to the Agent. It imports
nothing from either, so both can import it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    # The runner discarded output beyond its capture cap: the stream held
    # here is a prefix of what the command produced, not the whole of it.
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise TypeError("exit_code must be an integer")

    def to_record(self) -> dict[str, object]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


__all__ = ["CommandResult"]
