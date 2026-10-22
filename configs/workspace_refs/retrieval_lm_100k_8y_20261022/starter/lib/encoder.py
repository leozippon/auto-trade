"""The frozen title encoder: Qwen3-Embedding under `context.pretrained_dir`, last-token pooled.

    embed(context, titles) -> (len(titles), TEXT_DIM) float32 unit vectors
    retain(titles)            drop every cached title not in `titles`

A title's vector is the encoder's last hidden state at its final token (the
tokenizer appends <|endoftext|>; padding is on the left, so the final token
is the last position), L2-normalised, cut to its first TEXT_DIM dimensions
(the model is trained so that a prefix of its embedding is itself an
embedding) and normalised again. Titles are documents to the model: no
instruction is prepended, since a query and its neighbours are both titles.

The weights are `context.pretrained_dir + "/" + knobs.ENCODER` -- the only
weights a container holds; a hub id fails offline -- loaded once per process
in bfloat16 on the card and float32 on the CPU, every parameter frozen.
Vectors are cached in this process by anonymised title, so a title is encoded
once per worker per replay: a day's fit encodes only the titles first seen
that day. Batches are formed by token length, then title, so two replays of
the same span encode the same batches and reproduce each other; the same
title encoded in a batch of another shape can differ in the last bits of
bfloat16, which is the card's arithmetic.
"""

import numpy as np
import torch

from lib import device, knobs

MAX_TOKENS = 64
_LOADED = {}
_CACHE = {}


def embed(context, titles):
    titles = [str(title) for title in titles]
    missing = sorted({title for title in titles if title not in _CACHE})
    if missing:
        _encode(context, missing)
    if not titles:
        return np.zeros((0, knobs.TEXT_DIM), dtype="float32")
    return np.stack([_CACHE[title] for title in titles]).astype("float32")


def retain(titles):
    keep = {str(title) for title in titles}
    for title in [title for title in _CACHE if title not in keep]:
        del _CACHE[title]


def _load(context):
    if knobs.ENCODER not in _LOADED:
        # Imported here: the text-off twin never loads an encoder.
        from transformers import AutoModel, AutoTokenizer

        root = context.pretrained_dir + "/" + knobs.ENCODER
        dev = device.device()
        tokenizer = AutoTokenizer.from_pretrained(root, padding_side="left")
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        model = AutoModel.from_pretrained(root, dtype=dtype).to(dev).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _LOADED[knobs.ENCODER] = (tokenizer, model)
    return _LOADED[knobs.ENCODER]


def _encode(context, titles):
    tokenizer, model = _load(context)
    dev = next(model.parameters()).device
    ids = tokenizer(titles, truncation=True, max_length=MAX_TOKENS)["input_ids"]
    order = sorted(range(len(titles)), key=lambda i: (len(ids[i]), titles[i]))
    for start in range(0, len(order), knobs.ENCODE_BATCH):
        picked = order[start:start + knobs.ENCODE_BATCH]
        batch = tokenizer.pad({"input_ids": [ids[i] for i in picked]}, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(
                input_ids=batch["input_ids"].to(dev), attention_mask=batch["attention_mask"].to(dev)
            ).last_hidden_state[:, -1].float()
            unit = torch.nn.functional.normalize(hidden, dim=1)[:, : knobs.TEXT_DIM]
            vectors = torch.nn.functional.normalize(unit, dim=1).to(torch.float16).cpu().numpy()
        for row, i in enumerate(picked):
            _CACHE[titles[i]] = vectors[row]
