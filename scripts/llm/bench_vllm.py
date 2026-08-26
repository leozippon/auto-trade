#!/usr/bin/env python3
"""Benchmark a vLLM OpenAI-compatible endpoint: TTFT / decode speed / concurrency.

Usage:
  bench_vllm.py --base-url http://127.0.0.1:8010 --model qwen-3.8-27b-fp8 \
      [--api-key $VLLM_API_KEY] [--prompt-tokens 512] [--output-tokens 128] \
      [--concurrency 1,4,8,16] [--long-context-tokens 30000]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import httpx

FILLER = (
    "The quick brown fox jumps over the lazy dog near the river bank at dawn. "
    "Engineers debate whether attention is all you need, but linear attention "
    "variants keep evolving with hybrid architectures and cached states. "
)


def build_prompt(target_tokens: int) -> str:
    # English text averages ~0.75 tokens per word, ~4.3 chars per token.
    approx_chars = int(target_tokens * 4.3)
    reps = approx_chars // len(FILLER) + 1
    text = (FILLER * reps)[:approx_chars]
    # End with a deterministic question so output is well-defined.
    text += "\n\nIn ONE word, what color is the fox mentioned above? Answer now."
    return text


async def one_request(
    client: httpx.AsyncClient,
    base: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    tag: str,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "min_tokens": max_tokens,  # vLLM extension: force full-length generation
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    ttft = None
    out_tokens = 0
    ptoks = 0
    finish = None
    try:
        async with client.stream(
            "POST", f"{base}/v1/chat/completions", json=payload, headers=headers
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")[:300]
                return {"tag": tag, "error": f"HTTP {resp.status_code}: {body}"}
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    out_tokens = obj["usage"].get("completion_tokens", 0) or out_tokens
                    ptoks = obj["usage"].get("prompt_tokens", 0)
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {}) or {}
                    if delta.get("content"):
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                    if choices[0].get("finish_reason"):
                        finish = choices[0]["finish_reason"]
    except Exception as exc:  # noqa: BLE001
        return {"tag": tag, "error": f"{type(exc).__name__}: {exc}"}
    total = time.perf_counter() - t0
    return {
        "tag": tag,
        "ttft": ttft,
        "total": total,
        "out_tokens": out_tokens,
        "prompt_tokens": ptoks,
        "decode_tps": (out_tokens - 1) / (total - ttft) if ttft and out_tokens > 1 else None,
        "finish": finish,
    }


def fmt(x):
    return f"{x:7.1f}" if isinstance(x, float) else str(x)


async def run_level(base, model, api_key, prompt, max_tokens, conc, timeout):
    async with httpx.AsyncClient(timeout=timeout) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[
            one_request(client, base, model, api_key, prompt, max_tokens, f"c{conc}-r{i}")
            for i in range(conc)
        ])
    wall = time.perf_counter() - t0
    errs = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    ttfts = [r["ttft"] for r in ok if r["ttft"]]
    tps = [r["decode_tps"] for r in ok if r["decode_tps"]]
    tot_out = sum(r["out_tokens"] for r in ok)
    tot_in = sum(r.get("prompt_tokens", 0) for r in ok)
    print(
        f"conc={conc:2d} | ok/err={len(ok)}/{len(errs)} "
        f"| TTFT avg={fmt(sum(ttfts)/len(ttfts) if ttfts else 0)}s "
        f"max={fmt(max(ttfts) if ttfts else 0)}s "
        f"| decode avg={fmt(sum(tps)/len(tps) if tps else 0)} t/s "
        f"| agg out={fmt(tot_out / wall)} t/s "
        f"| prefill={fmt(tot_in / wall)} t/s "
        f"| wall={wall:5.1f}s"
    )
    for e in errs[:3]:
        print(f"    ERROR {e['tag']}: {e['error'][:200]}")
    return wall, tot_out


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""), help="defaults to $VLLM_API_KEY")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--concurrency", default="1,4,8,16")
    p.add_argument("--long-context-tokens", type=int, default=0,
                   help="if >0, run an extra single long-context request")
    p.add_argument("--timeout", type=float, default=900)
    args = p.parse_args()

    prompt = build_prompt(args.prompt_tokens)
    print(f"# model={args.model} base={args.base_url} "
          f"prompt~{args.prompt_tokens}tok out={args.output_tokens}tok")

    for level in [int(x) for x in args.concurrency.split(",")]:
        await run_level(args.base_url, args.model, args.api_key,
                        prompt, args.output_tokens, level, args.timeout)

    if args.long_context_tokens:
        long_prompt = build_prompt(args.long_context_tokens)
        print(f"# long-context single request (~{args.long_context_tokens} tok)")
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            r = await one_request(client, args.base_url, args.model, args.api_key,
                                  long_prompt, 64, "long")
        if "error" in r:
            print(f"    ERROR: {r['error'][:300]}")
        else:
            print(f"    prompt_tokens={r.get('prompt_tokens')} ttft={r['ttft']:.1f}s "
                  f"(prefill {r.get('prompt_tokens', 0) / r['ttft']:.0f} t/s) "
                  f"decode={r['decode_tps']:.1f} t/s finish={r['finish']}")


if __name__ == "__main__":
    asyncio.run(main())
