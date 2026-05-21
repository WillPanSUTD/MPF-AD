"""Cross-attention fusion of per-modality CLIP token maps.

Drop-in replacement for PointAD's single-RGB plug-and-play branch. Takes
M token maps (one per photometric modality, all coming from the same CLIP
backbone), adds learned modality-type embeddings, and applies a single
multi-head cross-attention block along the modality axis per token
position. Output: a single fused token map with the same shape as any
input.

The cross-attention uses Query = each token's own modality, Key/Value =
all modalities for that token, then averages across modalities to produce
the fused token. With `init="mean"`, weights are initialised so the
attention output is approximately the mean of inputs — useful as a
zero-shot starting point before any training.
"""
from __future__ import annotations

from typing import List, Literal

import torch
from torch import nn

MAX_MODALITIES = 6  # lut, phong, diffuse, specular, normal, depth (reserve slot)


class MultiPhotometricFusion(nn.Module):
    def __init__(
        self,
        clip_dim: int = 1024,
        num_heads: int = 8,
        num_modalities_max: int = MAX_MODALITIES,
        init: Literal["mean", "random"] = "mean",
    ) -> None:
        super().__init__()
        self.clip_dim = clip_dim
        self.modality_embed = nn.Embedding(num_modalities_max, clip_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=clip_dim, num_heads=num_heads, batch_first=True
        )
        self.ln = nn.LayerNorm(clip_dim)
        if init == "mean":
            nn.init.zeros_(self.modality_embed.weight)
            # MultiheadAttention has out_proj.bias initialized to 0 by default;
            # zero out out_proj.weight too so output ≈ mean-pooled values.
            nn.init.zeros_(self.attn.out_proj.weight)

    def forward(self, channels: List[torch.Tensor]) -> torch.Tensor:
        """
        channels: list of M tensors, each (B, N, D).
        returns:  (B, N, D) fused token map.
        """
        if len(channels) == 1:
            return channels[0]
        B, N, D = channels[0].shape
        M = len(channels)
        # Stack: (B, M, N, D)
        stacked = torch.stack(channels, dim=1)
        # Add modality embedding
        ids = torch.arange(M, device=stacked.device)
        emb = self.modality_embed(ids).view(1, M, 1, D)  # (1, M, 1, D)
        stacked = stacked + emb
        # Reshape for per-token attention: (B*N, M, D)
        permuted = stacked.permute(0, 2, 1, 3).contiguous()  # (B, N, M, D)
        flat = permuted.view(B * N, M, D)
        # Self-attention along modality axis
        attended, _ = self.attn(flat, flat, flat)  # (B*N, M, D)
        # Mean-pool across modalities
        pooled = attended.mean(dim=1)  # (B*N, D)
        fused = pooled.view(B, N, D)
        # Residual + LN (identity-friendly when out_proj is zero-inited)
        anchor = torch.stack(channels, dim=0).mean(dim=0)  # (B, N, D) mean of inputs
        return self.ln(anchor + fused)
