"""The torch device a fitted model uses: the arm's card when the container has one, else the CPU.

This arm's replays carry a card (run fact `budgets.strategy_gpu_count`), and the
same bytes must still run without one (output/README.md), so the device is
looked up where a tensor is made, never fixed at import. A model trained on the
card must also reproduce its own fills: seed torch from the package's one seed
line, and remember that some CUDA kernels are not bitwise deterministic by
default -- a book whose two replays train two different models does not
reproduce its fills.
"""

import torch


def device():
    """`cuda` when this container carries a card, else `cpu`."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
