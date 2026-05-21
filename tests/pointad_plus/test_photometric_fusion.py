"""Tests for the multi-photometric fusion block."""
import torch
import pytest

from src.pointad_plus.photometric_fusion import MultiPhotometricFusion


def test_forward_shape_single_modality():
    """M=1 degenerate path: output matches input single token map."""
    mod = MultiPhotometricFusion(clip_dim=64, num_heads=4)
    tokens = [torch.randn(2, 16, 64)]
    out = mod(tokens)
    assert out.shape == (2, 16, 64)


def test_forward_shape_multi_modality():
    mod = MultiPhotometricFusion(clip_dim=64, num_heads=4)
    tokens = [torch.randn(2, 16, 64) for _ in range(5)]
    out = mod(tokens)
    assert out.shape == (2, 16, 64)


def test_mean_init_approximates_mean_fusion():
    """When initialized to identity attention, output ≈ mean of inputs."""
    torch.manual_seed(0)
    mod = MultiPhotometricFusion(clip_dim=64, num_heads=4, init="mean")
    tokens = [torch.randn(1, 8, 64) for _ in range(5)]
    out = mod(tokens)
    target = torch.stack(tokens).mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(
        out.reshape(-1, 64), target.reshape(-1, 64)
    ).mean()
    # Cosine similarity should be close to 1 at mean init
    assert cos > 0.95


def test_param_count_under_5M():
    """Sanity: fusion block is small (~2M params per spec)."""
    mod = MultiPhotometricFusion(clip_dim=1024, num_heads=8)
    n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    assert n < 5_000_000
