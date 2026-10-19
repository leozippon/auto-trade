"""The online-update hook: where a continual-learning rule plugs into the baseline.

`daily(context, state)` is called by `learn.fit` at every trading day's fit
except the ones that fit the head from its initialisation (the first fit of
a replay; under "cold_quarter" the first fit of every quarter), after the
state has been loaded from `context.state_dir`. `state` holds live objects:

    state["model"]     the head on the device (lib/model.Head)
    state["last_day"]  the newest bar whose label the last fit trained on
    state["steps"]     a counter the rule may advance
    state["extra"]     a mapping the rule owns for whatever it carries between
                       days -- reference weights, optimiser moments, pending
                       gradients, evidence buffers, per-name states, an event
                       log -- as tensors and plain values

What a rule may do here: read the PIT view through lib/features.panel,
lib/text.read and lib/encoder.embed (the label of bar t is known from the day
its forward window closes, `panel.due[t]`, never before); change the head, or
train parameters of its own -- adapters on the encoder through peft, the
recurrent states of its linear-attention layers, anything the contract's
libraries build -- and keep them in `extra`, remembering that a trained
encoder makes the cache in lib/encoder.py stale; then call
`learn.save(context, state)`. The checkpoint under `context.state_dir` is the
only carrier: the inference worker rebuilds from that directory and never
sees this process's memory, and a Paper account replays its decisions one by
one from it, so anything expensive is checkpointed, not recomputed. The hook
runs only inside `fit`, the one place the state directory is writable, on
the same morning and context as the day's decision; a rule whose decision
needs more than the head's score changes lib/score.py to match.

The starter's body does nothing: the two shipped modes, "cold_quarter" and
"frozen", are the controls, and the rule is the Agent's design.
"""


def daily(context, state):
    """A no-op in the baseline; the module docstring says what a rule
    receives, may touch and must save."""

    return
