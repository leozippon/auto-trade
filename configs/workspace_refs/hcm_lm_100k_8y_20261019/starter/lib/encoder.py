"""The frozen title encoder: a pretrained language model under `context.pretrained_dir`, mean-pooled.

    embed(context, titles) -> (len(titles), dim(context)) float32 array
    dim(context)           -> the backbone's hidden size
    retain(titles)            drop every cached title not in `titles`

A title's embedding is the mean over its tokens of the backbone's last
hidden state (the decoder's final norm applied), padding masked out. The
backbone is `context.pretrained_dir + "/" + knobs.LM` -- the only weights a
container holds; a hub id fails offline -- loaded once per process in
bfloat16 on the card and float32 on the CPU, every parameter frozen.
Embeddings are cached in this process by anonymised title, so a title is
encoded once per worker per replay: a decision encodes only the day's new
titles, a quarterly refit only the titles its window gained. Nothing here
survives a replay, and the cache is never written to state_dir.

Batches are formed by token length, then title, so two replays of the same
span encode the same batches and reproduce each other. The same title
encoded in a batch of another shape -- the fit worker against the inference
worker -- can differ in the last bits of bfloat16; that is the card's
arithmetic, and the head is trained on embeddings with that noise in them.
A rule that trains the encoder (adapters through peft, the recurrent states
of the linear-attention layers) must replace this cache with its own: a
cached embedding is valid only while the weights that produced it are.
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        return np.zeros((0, dim(context)), dtype="float32")
    return np.stack([_CACHE[title] for title in titles]).astype("float32")


def dim(context):
    return int(_load(context)[1].config.hidden_size)


def retain(titles):
    keep = {str(title) for title in titles}
    for title in [title for title in _CACHE if title not in keep]:
        del _CACHE[title]


def _load(context):
    if knobs.LM not in _LOADED:
        root = context.pretrained_dir + "/" + knobs.LM
        dev = device.device()
        tokenizer = AutoTokenizer.from_pretrained(root)
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
        backbone = AutoModelForCausalLM.from_pretrained(root, dtype=dtype).model.to(dev).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        _LOADED[knobs.LM] = (tokenizer, backbone)
    return _LOADED[knobs.LM]


def _encode(context, titles):
    tokenizer, backbone = _load(context)
    dev = next(backbone.parameters()).device
    ids = tokenizer(titles, truncation=True, max_length=MAX_TOKENS)["input_ids"]
    order = sorted(range(len(titles)), key=lambda i: (len(ids[i]), titles[i]))
    for start in range(0, len(order), knobs.ENCODE_BATCH):
        picked = order[start:start + knobs.ENCODE_BATCH]
        batch = tokenizer.pad({"input_ids": [ids[i] for i in picked]}, return_tensors="pt")
        mask = batch["attention_mask"].to(dev)
        with torch.inference_mode():
            hidden = backbone(input_ids=batch["input_ids"].to(dev), attention_mask=mask).last_hidden_state
            pooled = (hidden.float() * mask.unsqueeze(2)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        vectors = pooled.to(torch.float16).cpu().numpy()
        for row, i in enumerate(picked):
            _CACHE[titles[i]] = vectors[row]
