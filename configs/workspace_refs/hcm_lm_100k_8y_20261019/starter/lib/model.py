"""The selector head, and the pooling of a name's titles into its text input.

    Head(text_dim)      the text embedding, normalised and projected to
                        TEXT_DIM, beside a has-text flag and the ten numeric
                        features, through one HIDDEN-wide layer to a score;
                        about 36k parameters at the shipped widths
    gather(table, ids)  (rows, dim) mean of the title embeddings each row's
                        ids name in `table`, and (rows,) has-text flags

The head is deliberately small: the title encoder (lib/encoder.py) is frozen
in the baseline, and the trainable set is what a rule decides to update.
"""

import torch
from torch import nn

from lib import features, knobs


class Head(nn.Module):
    def __init__(self, text_dim):
        super().__init__()
        self.norm = nn.LayerNorm(text_dim, elementwise_affine=False)
        self.text = nn.Linear(text_dim, knobs.TEXT_DIM)
        self.body = nn.Sequential(
            nn.Linear(knobs.TEXT_DIM + 1 + features.FEATURES, knobs.HIDDEN),
            nn.GELU(),
            nn.Linear(knobs.HIDDEN, 1),
        )

    def forward(self, e, has, x):
        """e (names, text_dim) the mean title embedding, anything where `has`
        is 0; has (names,) 1.0 where the name has text; x (names, FEATURES)."""

        t = torch.tanh(self.text(self.norm(e))) * has.unsqueeze(1)
        return self.body(torch.cat([t, has.unsqueeze(1), x], dim=1)).squeeze(1)


def gather(table, ids):
    """table (titles, dim); ids (rows, MAX_TITLES) with -1 for an empty slot."""

    mask = (ids >= 0).to(table.dtype)
    picked = table[ids.clamp(min=0)] * mask.unsqueeze(2)
    count = mask.sum(dim=1)
    return (picked.sum(dim=1) / count.clamp(min=1).unsqueeze(1)).float(), (count > 0).float()
