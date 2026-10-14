"""Small deterministic model used only to verify export/benchmark plumbing."""

from __future__ import annotations

from typing import Any


def tiny_cnn() -> Any:
    import torch

    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(4, 2),
    ).eval()
