"""Sliding-window patch coordinates + overlap-mean aggregation."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def patch_coords(
    img_h: int, img_w: int,
    patch_size: int = 256, stride: int = 128,
) -> List[Tuple[int, int]]:
    """Top-left (u_col, v_row) for each non-overflowing patch.
    Convention: u is column (x), v is row (y), matching harvest_good convention.
    For img dim D and patch P, valid starts are 0, stride, 2*stride, ..., D-P.
    """
    if img_h < patch_size or img_w < patch_size:
        raise ValueError(f"image {img_h}x{img_w} smaller than patch {patch_size}")
    us: List[int] = []
    for u in range(0, img_w - patch_size + 1, stride):
        us.append(u)
    if us[-1] != img_w - patch_size:
        # don't enforce last-fit; some images have dims that aren't (P + k*stride)
        pass
    vs: List[int] = []
    for v in range(0, img_h - patch_size + 1, stride):
        vs.append(v)
    return [(u, v) for v in vs for u in us]


def aggregate_overlap_mean(
    patch_maps: List[np.ndarray],
    coords: List[Tuple[int, int]],
    img_h: int, img_w: int,
    patch_size: int = 256,
) -> np.ndarray:
    """Overlap-mean aggregation of per-patch anomaly maps.
    patch_maps[i] is (patch_size, patch_size). coords[i] is (u, v).
    Returns (img_h, img_w) array. Pixels not covered by any patch → 0.
    """
    if not patch_maps:
        return np.zeros((img_h, img_w), dtype=np.float32)
    # Use float32 instead of float64 to halve memory (the overlap counts
    # are small integers; float32 has plenty of precision for the mean).
    acc = np.zeros((img_h, img_w), dtype=np.float32)
    cov = np.zeros((img_h, img_w), dtype=np.float32)
    for pm, (u, v) in zip(patch_maps, coords):
        pm32 = pm.astype(np.float32, copy=False) if pm.dtype != np.float32 else pm
        acc[v:v + patch_size, u:u + patch_size] += pm32
        cov[v:v + patch_size, u:u + patch_size] += 1.0
    out = np.zeros((img_h, img_w), dtype=np.float32)
    mask = cov > 0
    out[mask] = acc[mask] / cov[mask]
    return out
