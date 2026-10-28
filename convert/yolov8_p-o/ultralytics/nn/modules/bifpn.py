# Ultralytics YOLO 🚀, AGPL-3.0 license
"""BiFPN modules."""

import torch
import torch.nn as nn

__all__ = ("BiFPN_Concat2",)


class BiFPN_Concat2(nn.Module):
    """Weighted concatenation of two feature maps for BiFPN-style feature fusion."""

    def __init__(self, dimension=1, epsilon=1e-4):
        """Initialize learnable fusion weights and concatenation dimension."""
        super().__init__()
        self.d = dimension
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """Apply normalized learnable weights to two inputs, then concatenate them."""
        w = self.w / (torch.sum(self.w, dim=0) + self.epsilon)
        return torch.cat((w[0] * x[0], w[1] * x[1]), self.d)
