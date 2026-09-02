"""Trusted research tooling mounted read-only into the Agent session sandbox.

``screen.py`` is self-contained (numpy/pandas/pyarrow only) so the sandbox
image needs no rebuild; ``sandbox.DockerSandbox`` binds it at
``SCREENING_TOOL_MOUNT``.
"""
