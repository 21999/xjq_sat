import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AngleLoss(nn.Module):
    """
    Given two input tensors of shape (bs, D1) and (bs, D2),
    this loss projects each into a common latent space of dimension L,
    then computes a cosine-similarity-based loss to minimize the angle between them.
    """
    def __init__(self, D1: int, D2: int):
        """
        Args:
            D1 (int): dimensionality of the first tensor.
            D2 (int): dimensionality of the second tensor.
            L  (int): dimension of the common latent space.
        """
        super(AngleLoss, self).__init__()
        L = min(D1, D2)
        # Linear layer to map D1 -> L
        self.proj1 = nn.Linear(D1, L, bias=False)
        # Linear layer to map D2 -> L
        self.proj2 = nn.Linear(D2, L, bias=False)

        # It can help to initialize these so that the initial mapping is roughly identity‐like
        # (e.g., orthonormal). Here we use Kaiming for generality:
        nn.init.kaiming_uniform_(self.proj1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj2.weight, a=math.sqrt(5))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1 (torch.Tensor): shape (bs, D1), not necessarily normalized.
            x2 (torch.Tensor): shape (bs, D2), not necessarily normalized.

        Returns:
            loss (torch.Tensor): scalar tensor that encourages the angle between
                                 proj1(x1) and proj2(x2) to be small.
        """
        # Project both into common space: (bs, L)
        z1 = self.proj1(x1)
        z2 = self.proj2(x2)

        # Compute L2 norms: (bs, 1)
        norm1 = torch.norm(z1, p=2, dim=1, keepdim=True).clamp(min=1e-6)
        norm2 = torch.norm(z2, p=2, dim=1, keepdim=True).clamp(min=1e-6)

        # Normalize to unit vectors: (bs, L)
        z1_normalized = z1 / norm1
        z2_normalized = z2 / norm2

        # Cosine similarity per batch element: (bs,)
        # z1_normalized · z2_normalized = cos(theta)
        cos_sim = torch.sum(z1_normalized * z2_normalized, dim=1)

        # Clamp to [-1, 1] for numerical stability before arccos (if you need the actual angle)
        cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Option 1: Direct “angle” regression (loss = angle in radians)
        # angles = torch.acos(cos_sim)                  # (bs,)
        # loss = angles.mean()

        # Option 2: Use 1 - cosine as a surrogate (simpler, differentiable everywhere except at bounds)
        loss = 1.0 - cos_sim

        return loss
