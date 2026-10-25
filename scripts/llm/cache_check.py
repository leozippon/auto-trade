#!/usr/bin/env python3
"""Check that a hosted model's prompt cache works today, before an arm uses it.

Sends one production-shaped request (system prompt, user task, a tool call
and its ~3,000-token result, so a Claude id carries its cache marker on the
tool result as in production) through the project's role gateway, then the
same request again about 20 seconds later. A fresh nonce leads the system
prompt, so the first call cannot read an earlier run's cache. Prints each
call's prompt, cached, write and output tokens, its list price and the
model id the upstream reported.

    <quant python> scripts/llm/cache_check.py claude-opus-5-5
    <quant python> scripts/llm/cache_check.py DeepSeek-V4.1-Flash --calls 3

Exit status 1 when no call after the first shows a cache read or write. The
relay serves DeepSeek from two upstreams with separate caches and DeepSeek
reports no writes, so its second call misses whenever it lands on the other
upstream; a third call lands on one of the two already written. A few cents
per run. The key is read from ``.env`` by the gateway and never printed.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _bootstrap import add_repo_src

REPO_ROOT = add_repo_src(__file__)

from autotrade.environment.llm import (
    ChatMessage,
    ToolCall,
    billed_tokens,
    build_model_gateway,
    cache_write_tokens,
    cached_prompt_tokens,
    token_prices,
)

TOOLS = [{"type": "function", "function": {
    "name": "read_file",
    "description": "Read one text file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}}]
WORDS = ("river", "harbor", "lantern", "orchard", "meadow", "quarry", "valley", "summit")


def request(nonce: str) -> list[ChatMessage]:
    rng = random.Random(nonce)
    entries = [
        f"Entry {n}: the {rng.choice(WORDS)} beside the {rng.choice(WORDS)} was surveyed on day "
        f"{rng.randint(1, 365)} at {rng.randint(100, 4000)} metres, {rng.randint(1, 99)} paces from marker {rng.randint(1, 999)}."
        for n in range(1, 71)
    ]
    return [
        ChatMessage("system", f"Cache check {nonce}. You are a terse assistant: answer in at most five words."),
        ChatMessage("user", "Read notes.txt, then reply with the number of its last entry only."),
        ChatMessage("assistant", None, tool_calls=(ToolCall("call_1", "read_file", {"path": "notes.txt"}),)),
        ChatMessage("tool", "\n".join(entries), tool_call_id="call_1"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model")
    parser.add_argument("--calls", type=int, default=2)
    parser.add_argument("--gap", type=float, default=20.0, help="seconds between calls")
    args = parser.parse_args()
    if args.calls < 2:
        parser.error("--calls must be at least 2: the first has nothing to read")
    gateway = build_model_gateway(
        args.model, env_file=REPO_ROOT / ".env", max_tokens=200, conversation_log_dir=None
    )
    prices = token_prices(gateway.provider, args.model)
    messages = request(uuid.uuid4().hex[:12])
    seen = False
    for call in range(1, args.calls + 1):
        if call > 1:
            time.sleep(args.gap)
        response = gateway.complete(messages, tools=TOOLS)
        usage = response.usage
        cached, written = cached_prompt_tokens(usage), cache_write_tokens(usage)
        cost = prices.cost_usd(billed_tokens(response, messages=messages, tools=TOOLS))
        print(
            f"call {call}: prompt {usage.get('prompt_tokens')}  "
            f"cached {'n/a' if cached is None else cached}  write {written}  "
            f"output {usage.get('completion_tokens')}  list price ${cost:.4f}  upstream {response.model}",
            flush=True,
        )
        seen = seen or (call > 1 and bool(cached or written))
    print(f"{args.model}: {'cache read or write seen' if seen else 'NO cache read or write after the first call'}")
    return 0 if seen else 1


if __name__ == "__main__":
    sys.exit(main())
