"""Tests for the SSL modality-masking trainer."""
import torch
import pytest

from src.pointad_plus.photometric_fusion import MultiPhotometricFusion
from src.pointad_plus.ssl_modality_masking import masking_loss


def test_masking_loss_returns_scalar():
    mod = MultiPhotometricFusion(clip_dim=32, num_heads=4)
    channels = [torch.randn(2, 8, 32) for _ in range(5)]
    loss = masking_loss(mod, channels, dropout_seed=0)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_masking_loss_is_zero_when_all_inputs_equal():
    """If all 5 modalities are identical, dropping one shouldn't change the fused output."""
    mod = MultiPhotometricFusion(clip_dim=32, num_heads=4, init="mean")
    same = torch.randn(2, 8, 32)
    channels = [same.clone() for _ in range(5)]
    loss = masking_loss(mod, channels, dropout_seed=0)
    assert loss.item() < 1e-4
