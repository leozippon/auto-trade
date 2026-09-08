"""Environment layer: PIT snapshots, Sandbox, tools, simulated Broker, LLM proxy.

Architecture boundaries: this package must not import from ``autotrade.agent``,
and must not import from ``autotrade.data_sources`` (the ingest adapter depends
on the environment's raw-lake contract in ``data/contracts.py``, never the
reverse).

Layout — subpackages are multi-module subsystems, top-level modules are
single-purpose components:

- ``data/``: PIT availability contracts (incl. the raw-lake research contract),
  raw-lake store, dataset transforms, snapshot builder, research-release
  pinning, agent data summary.
- ``replay/``: daily replay core (host engine, market source, result stats,
  rolling PIT timeview, post-replay style attribution).
- ``strategy.py``/``strategy_loader.py``/``strategy_worker.py``: the scheduled
  strategy contract, its static source validation, and the sandbox-side worker
  that speaks the JSON order protocol.
- ``tools/``: Agent-facing tool contracts dispatched by the session runner.
- ``llm/``: host-side LLM provider boundary (keys never reach the sandbox).
- ``nl/``: NL Sub Agent stack (engine, PIT text retrieval, company context,
  host service).
- ``broker.py``/``broker_core.py``: simulated Broker and its pure fill math.
- ``sandbox.py``/``sandbox_images.py``/``executor.py``/``gpu.py``: sandbox
  lifecycle, derived images, command executors, GPU selection.
- ``contract_fingerprint.py``: digest of the strategy-contract sources baked
  into the sandbox image, checked before every strategy container start.
- ``runtime.py``/``identity.py``/``artifacts.py``/``step_tree.py``:
  cross-cutting run primitives (paths/manifest/trace, agent-visible refs,
  artifact contracts, validated-step lineage).
"""
