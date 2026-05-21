"""Tests for sliding-window utilities."""
import numpy as np
import pytest

from src.pointad_plus.sliding_window import patch_coords, aggregate_overlap_mean


def test_patch_coords_512_image():
    """For 512×512 image with patch=256, stride=128: u/v ∈ {0, 128, 256} → 3×3 = 9 patches."""
    coords = patch_coords(512, 512, patch_size=256, stride=128)
    expected = [(u, v) for v in (0, 128, 256) for u in (0, 128, 256)]
    assert sorted(coords) == sorted(expected)


def test_patch_coords_640x580_image():
    """For 640H × 580W image: u ∈ {0, 128, 256}, v ∈ {0, 128, 256, 384} → 3×4 = 12 patches."""
    coords = patch_coords(640, 580, patch_size=256, stride=128)
    expected = [(u, v) for v in (0, 128, 256, 384) for u in (0, 128, 256)]
    assert sorted(coords) == sorted(expected)


def test_patch_coords_256x256_no_sliding():
    """For exact 256×256 image: single patch at (0, 0)."""
    coords = patch_coords(256, 256, patch_size=256, stride=128)
    assert coords == [(0, 0)]


def test_aggregate_overlap_mean_single_patch():
    """Single patch covers the entire image (which is exactly 256×256)."""
    patch = np.ones((256, 256), dtype=np.float32) * 0.5
    agg = aggregate_overlap_mean([patch], [(0, 0)], 256, 256)
    assert agg.shape == (256, 256)
    assert np.allclose(agg, 0.5)


def test_aggregate_overlap_mean_two_patches_overlap():
    """Two patches with 50% overlap: overlap region = mean of the two."""
    p0 = np.ones((4, 4), dtype=np.float32) * 0.2
    p1 = np.ones((4, 4), dtype=np.float32) * 0.6
    agg = aggregate_overlap_mean(
        [p0, p1], [(0, 0), (2, 0)],
        img_h=4, img_w=6, patch_size=4
    )
    # Cols 0,1: only p0; cols 4,5: only p1; cols 2,3: mean(0.2, 0.6) = 0.4
    assert np.allclose(agg[:, 0:2], 0.2)
    assert np.allclose(agg[:, 4:6], 0.6)
    assert np.allclose(agg[:, 2:4], 0.4)


def test_aggregate_overlap_mean_uncovered_pixels_zero():
    """Pixels not covered by any patch get 0."""
    p = np.ones((4, 4), dtype=np.float32)
    agg = aggregate_overlap_mean([p], [(0, 0)], img_h=8, img_w=8, patch_size=4)
    assert np.all(agg[0:4, 0:4] == 1.0)
    assert np.all(agg[4:8, 4:8] == 0.0)
