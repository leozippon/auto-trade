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
  rolling PIT timeview, post-replay style attribution, the random-portfolio
  null control and its baseline, result curves).
- ``screening/``: research tooling mounted read-only into the Agent session
  (the signal screen).
- ``strategy.py``/``strategy_loader.py``/``strategy_worker.py``: the scheduled
  strategy contract, its static source validation, and the sandbox-side worker
  that speaks the JSON order protocol.
- ``tools/``: Agent-facing tool contracts dispatched by the session runner.
- ``llm/``: host-side LLM provider boundary (keys never reach the sandbox).
- ``broker.py``/``broker_core.py``/``distributions.py``/``dividend_tax.py``:
  simulated Broker, its board and lot rules, ex-date distributions, and
  dividend tax.
- ``sandbox.py``/``sandbox_images.py``/``executor.py``/``commands.py``/
  ``container_stats.py``/``gpu.py``/``pretrained.py``: container boundary and
  session container, each experiment's pinned image clone, strategy executors
  and the bounded command runner, container resource telemetry, GPU
  attachment, the pinned pretrained-model store.
- ``contract_fingerprint.py``: digests of the strategy contract a sandbox image
  carries — the runtime modules it bakes, checked before every strategy
  container start, and the Agent-facing contract text, checked where an Agent
  reads it.
- ``runtime.py``/``trace.py``/``identity.py``/``artifacts.py``/``step_tree.py``/
  ``time_budget.py``: cross-cutting run primitives (paths/manifest, the Agent
  trace and transcript, agent-visible refs, artifact contracts, validated-step
  lineage, the session's effective-time budget).
"""
