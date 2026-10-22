"""C1, the carrier: a small network on the same keys and the same matured cases, refitted cold each quarter.

    fit(memory, day, dev) -> Head   fitted from its initialisation on at most
                                    TRAIN_CASES of the memory's cases
    Head(width)                     key -> one score, through HIDDEN units

The network sees what a retrieval sees -- the keys of every filed case and
their outcomes -- and differs from it only in how it uses them: it compresses
them into fixed weights once a quarter, where a retrieval reads the nearest
cases afresh each day. Its draws (initialisation, the training sample, batch
order) come from the package's one seed line and the fit day.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from lib import knobs


class Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(width, knobs.HIDDEN), nn.GELU(), nn.Linear(knobs.HIDDEN, 1))

    def forward(self, key):
        return self.body(key).squeeze(1)


def fit(memory, day, dev):
    torch.manual_seed(knobs.SEED)
    rng = np.random.default_rng([knobs.SEED, int(day)])
    sample = np.sort(rng.choice(memory.n, size=min(knobs.TRAIN_CASES, memory.n), replace=False))
    keys = memory.key[torch.as_tensor(sample, device=memory.key.device)].float()
    y = torch.as_tensor(memory.y[sample], device=dev)
    net = Head(keys.shape[1]).to(dev)
    optim = torch.optim.AdamW(net.parameters(), lr=knobs.LR)
    net.train()
    for _ in range(knobs.EPOCHS):
        order = torch.as_tensor(rng.permutation(len(sample)), device=dev)
        for start in range(0, len(sample), knobs.BATCH):
            batch = order[start:start + knobs.BATCH]
            optim.zero_grad()
            functional.mse_loss(net(keys[batch]), y[batch]).backward()
            optim.step()
    return net.eval()
