"""The in-context path, as an example the starter does not call: a served model reads each shortlisted name's retrieved cases and predicts.

Copy this file to `output/lib/llm_cases.py` and call `rerank` from
lib/learn.score right after a retrieval, on the days the book trades
(`needed`). It is outside the starter because the starter's bytes must not
depend on it: the starter runs without vLLM, and an arm's smoke replays it
without one.

    needed(context)                 whether today's decision can trade: a
                                    review day, or a book with an empty seat
    engine(context, model, ...)     one vLLM engine per process, on this leg's
                                    card, from the pinned pretrained store
    describe(key)                   a key's pattern as four numbers -- no
                                    name, code, date or title
    messages(query, keys, outcomes) the chat for one shortlisted name: its own
                                    description, then its cases' with their
                                    outcomes as percentiles, labelled A, B, ...
    predict(llm, chats)             one percentile per chat, NaN where the
                                    reply holds none; greedy, thinking off
    parse(text)                     the first integer 0..100 in a reply
    random_cases(memory, code, day, n)  the shuffled-retrieval placebo: n cases
                                    drawn by a fixed hash of (code, day), not
                                    by similarity
    redact(titles)                  the only form in which title text may
                                    enter a prompt (this example puts none in)
    rerank(context, memory, codes, query, values, idx, ...)
                                    the shortlist reordered by the model's
                                    percentiles, every other name unchanged

What a prompt may carry (README, "匿名化与泄漏"): numbers derived from a key
and an outcome percentile, nothing else -- no security code, company name,
date, year, price level or title text. Titles reach this path only through
the embeddings that chose the neighbours; an Agent that wants title text in a
prompt passes it through `redact` first and accepts that a served model may
recall an event from its words. A shortlisted name's reply that holds no
number keeps its retrieval score, below every name the model scored.

Determinism: an eager engine (no torch.compile, no CUDA graphs),
temperature 0 and the package's seed, and one batch per day in a fixed order
(the shortlist's own). vLLM gives other tokens for the same prompt in a batch
of another shape or order, and a compiled engine differs between containers,
so a replay of the same span reproduces its fills only under these settings
(output/README.md, "Pretrained models"); prefix caching is off as well, which
an eager engine does not need but which leaves one condition fewer to check.

Cost: a 27B engine loads in minutes, once per fit worker (once per replay;
each Paper morning starts a new worker). A chat here is about 400-600 prompt
tokens and a few output tokens, so a shortlist of SHORTLIST names is a few
tens of thousands of prompt tokens a trading day; call it only when `needed`.
"""

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

from lib import features, knobs, trade

MODEL = "Qwen3.8-27B-FP8"
SHORTLIST = 3 * knobs.SEATS
CASES = 8
MAX_TOKENS = 8
SYSTEM = (
    "You predict how a stock will do over the next month relative to all other stocks, "
    "from its recent price-volume pattern and the patterns of similar past cases whose outcomes are known. "
    "An outcome percentile of 0 is the worst stock of its day and 100 the best. "
    "Reply with one integer from 0 to 100 and nothing else."
)
_ENGINES = {}


def needed(context):
    positions = {code: qty for code, qty in dict(context.account.positions).items() if int(qty) > 0}
    if len(positions) < knobs.SEATS:
        return True
    return bool(trade._due(context, pd.Timestamp(context.inference_at)))


def engine(context, model=MODEL, *, tensor_parallel_size=1, gpu_memory_utilization=0.80, max_model_len=8192, max_num_seqs=32):
    """The process's engine for `model`, built on first use. The weights are
    `context.pretrained_dir/<model>`; the engine runs on the card the leg was
    given, or on both at batch width 1 with tensor_parallel_size=2. The
    engine takes gpu_memory_utilization of the whole card, so this process's
    cached blocks are handed back first."""

    key = (model, tensor_parallel_size)
    if key not in _ENGINES:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _ENGINES[key] = LLM(
            model=context.pretrained_dir + "/" + model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt={"image": 0, "video": 0},
            enforce_eager=True,
            enable_prefix_caching=False,
            seed=knobs.SEED,
        )
    return _ENGINES[key]


def describe(key):
    """'20d +1.8σ, 5d -0.4σ, turnover x1.3, news' from a key (lib/learn.keys):
    the window's cumulative excess return over the benchmark and its last five
    days', in units of the name's own volatility, the last five days' turnover
    over its usual level, and whether it had titles in the text window."""

    w = knobs.WINDOW
    path = np.asarray(key[:w], dtype="float64") * np.sqrt(np.arange(1, w + 1))
    flow = np.asarray(key[w:2 * w], dtype="float64") * features.FLOW_SCALE
    news = len(key) > 2 * w and bool(np.any(np.asarray(key[2 * w:]) != 0))
    return (
        f"20d {path[-1]:+.1f}σ, 5d {path[-1] - path[-6]:+.1f}σ, "
        f"turnover x{np.exp(flow[-5:].mean()):.1f}, {'news' if news else 'no news'}"
    )


def messages(query, keys, outcomes):
    lines = [f"Stock: {describe(query)}", "Similar past cases:"]
    for i, (key, outcome) in enumerate(zip(keys, outcomes)):
        lines.append(f"{chr(65 + i)}: {describe(key)} -> outcome percentile {int(round((float(outcome) + 0.5) * 100))}")
    lines.append("Predicted outcome percentile of the stock:")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def predict(llm, chats):
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=knobs.SEED)
    outputs = llm.chat(chats, params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
    return np.array([parse(output.outputs[0].text) for output in outputs], dtype="float64")


def parse(text):
    found = pd.Series([str(text)]).str.extract(r"(?<!\d)(\d{1,3})(?!\d)")[0].iloc[0]
    if pd.isna(found) or not 0 <= int(found) <= 100:
        return float("nan")
    return float(found)


def random_cases(memory, code, day, n):
    """n distinct memory rows drawn by a fixed hash of (code, day): what the
    model sees in the shuffled-retrieval placebo instead of the neighbours."""

    seed = int(pd.util.hash_array(np.array([f"{code}|{day}|cases"], dtype=object))[0] % (2**63))
    rng = np.random.default_rng([knobs.SEED, seed])
    return np.sort(rng.choice(memory.n, size=n, replace=False))


def redact(titles):
    """Titles (already anonymised by lib/text) without digits, Latin letters
    or anything quoted or bracketed, cut to 24 characters: an event type
    without the dates, amounts, product names and codes that could name it."""

    cleaned = pd.Series([str(title) for title in titles], dtype="object")
    cleaned = cleaned.str.replace(r"[《“\"「【\[(（][^》”\"」】\])）]*[》”\"」】\])）]", "", regex=True)
    cleaned = cleaned.str.replace(r"[0-9０-９A-Za-zＡ-Ｚａ-ｚ.%．％]+", "", regex=True)
    cleaned = cleaned.str.replace(r"\s+", "", regex=True)
    return cleaned.str.slice(0, 24).tolist()


def rerank(context, memory, codes, query, values, idx, *, shortlist=SHORTLIST, cases=CASES, placebo=False, model=MODEL):
    """`values` with the top `shortlist` names reordered by the model: each
    such name's chat shows its `cases` nearest cases (`idx`, nearest first) or,
    with `placebo`, as many drawn at random; a percentile p becomes 1 + p/100,
    above every retrieval score, and a name without one keeps its own."""

    values = np.asarray(values, dtype="float64")
    order = np.argsort(-values, kind="stable")[:shortlist]
    day = context.inference_at.strftime("%Y%m%d")
    chats = []
    for row in order:
        picks = random_cases(memory, codes[row], day, cases) if placebo else np.asarray(idx[row, :cases])
        chats.append(messages(query[row], memory.key[picks].float().cpu().numpy(), memory.y[picks]))
    predicted = predict(engine(context, model), chats)
    out = values.copy()
    scored = np.isfinite(predicted)
    out[order[scored]] = 1.0 + predicted[scored] / 100.0
    return out
