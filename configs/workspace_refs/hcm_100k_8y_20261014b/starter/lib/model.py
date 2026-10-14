"""The scorer and its episode store: the three memory levels as modules.

    HCM(features, hidden)   input projection -> GRU cell, whose per-name
                            state is the short-term memory -> two FFN blocks,
                            the body -> a head over [embedding, memory signal]
    groups(model)           the parameter groups "input", "body", "head"
    Store(size, hidden)     the external memory: a capped, reservoir-sampled
                            set of episodes (embedding, body input, label,
                            signal day, due day); `signal` is the mean label
                            of a query's KNN cosine neighbours whose label was
                            already known on the query's day

The GRU cell is the one recurrence: `step` advances every name's state by one
day, `run` applies it over a sequence. The body and head are plain
feed-forward layers, so a stored body input replays through them exactly as a
fresh one. Parameters are about 92k at HIDDEN 80.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional

CHUNK = 512


class Block(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))

    def forward(self, h):
        return h + self.ffn(self.norm(h))


class HCM(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.proj = nn.Linear(features, hidden)
        self.cell = nn.GRUCell(hidden, hidden)
        self.body = nn.ModuleList([Block(hidden), Block(hidden)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden + 1, 1)

    def step(self, x, h):
        """x (names, features), h (names, hidden) -> the next state."""

        return self.cell(torch.tanh(self.proj(x)), h)

    def run(self, x, h):
        """x (days, names, features) -> the state after every day in turn."""

        for day in range(x.shape[0]):
            h = self.step(x[day], h)
        return h

    def embed(self, h):
        for block in self.body:
            h = block(h)
        return self.norm(h)

    def score(self, e, m):
        """e (names, hidden), m (names,) -> (names,)."""

        return self.head(torch.cat([e, m.unsqueeze(1)], dim=1)).squeeze(1)


def groups(model):
    return {
        "input": [*model.proj.parameters(), *model.cell.parameters()],
        "body": [*model.body.parameters(), *model.norm.parameters()],
        "head": list(model.head.parameters()),
    }


class Store:
    FIELDS = ("emb", "hin", "label", "day", "due")

    def __init__(self, size, hidden, device):
        self.size = int(size)
        self.emb = torch.zeros(self.size, hidden, device=device)
        self.hin = torch.zeros(self.size, hidden, device=device)
        self.label = torch.zeros(self.size, device=device)
        self.day = torch.zeros(self.size, dtype=torch.int64, device=device)
        self.due = torch.zeros(self.size, dtype=torch.int64, device=device)
        self.offered = 0

    @property
    def filled(self):
        return min(self.offered, self.size)

    def add(self, emb, hin, label, day, due, rng):
        """Offer episodes (tensors aligned on their first axis; `day` and `due`
        as YYYYMMDD) to the reservoir: the n-th episode ever offered is kept
        with probability size / n, in a uniformly drawn slot. The last write
        to a slot within one call wins, so the result does not depend on the
        order the device applies duplicate indices in."""

        count = emb.shape[0]
        if count == 0:
            return
        n = self.offered + np.arange(1, count + 1)
        direct = n <= self.size
        slots = np.where(direct, n - 1, rng.integers(0, self.size, count))
        kept = direct | (rng.random(count) * n < self.size)
        rows = np.flatnonzero(kept)
        _, last = np.unique(slots[rows][::-1], return_index=True)
        rows = rows[::-1][last]
        target = torch.as_tensor(slots[rows], device=emb.device)
        source = torch.as_tensor(rows, device=emb.device)
        self.emb[target] = emb[source]
        self.hin[target] = hin[source]
        self.label[target] = label[source]
        self.day[target] = day[source]
        self.due[target] = due[source]
        self.offered += count

    def signal(self, query, day, k):
        """(queries,) mean label of each query's k nearest stored embeddings by
        cosine among the episodes due on or before its `day` (YYYYMMDD, one per
        query); 0 for a query with no eligible neighbour."""

        out = torch.zeros(query.shape[0], device=query.device)
        filled = self.filled
        if filled == 0:
            return out
        keys = functional.normalize(self.emb[:filled], dim=1)
        labels = self.label[:filled]
        due = self.due[:filled]
        queries = functional.normalize(query.detach(), dim=1)
        for start in range(0, queries.shape[0], CHUNK):
            stop = start + CHUNK
            sims = queries[start:stop] @ keys.T
            sims = sims.masked_fill(due.unsqueeze(0) > day[start:stop].unsqueeze(1), float("-inf"))
            top = sims.topk(min(k, filled), dim=1)
            hit = torch.isfinite(top.values)
            picked = labels[top.indices] * hit
            out[start:stop] = picked.sum(dim=1) / hit.sum(dim=1).clamp(min=1)
        return out

    def state(self):
        return {
            "size": self.size,
            "offered": self.offered,
            **{name: getattr(self, name)[: self.filled].cpu() for name in self.FIELDS},
        }

    @classmethod
    def load(cls, record, hidden, device):
        store = cls(record["size"], hidden, device)
        filled = len(record["label"])
        for name in cls.FIELDS:
            getattr(store, name)[:filled] = record[name].to(device)
        store.offered = int(record["offered"])
        return store
