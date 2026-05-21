# PointAD+ Phase 3 — Template-Free Sliding-Window Inference

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Eliminate the 336×336 input-resize bottleneck. Slide 256×256 windows over native-resolution welds images, run PointAD+ (mean fusion from Phase 2) on each patch, aggregate per-pixel via overlap-mean, recompute per-defect metrics. Target: lift `pseudo_soldering` integrate image-AUROC above its 76.36 mean-fusion ceiling, and lift aggregate pixel-AUROC/AUPRO (currently 93.7 / 70.1 on the resized inputs).

**Architecture (delta from Phase 2):**

```
Test sample (anomalous, native ≈ 640×580):
  5 photometric channels at native res
        │
        ▼
[Sliding 256×256 window, stride 128]  →  ~12 patches per image
        │
        ▼ (per-patch)
[CLIP-ViT-L × 5 modalities]  →  5 token maps
        │
        ▼
[Mean fusion (Phase 2 finding: no learned params)]
        │
        ▼
[PointAD prompt head]  →  per-patch color anomaly map (256×256)
        │
        ▼
[Overlap-mean aggregation]  →  full-image anomaly map (native res)
        │
        ▼
Image score = max over per-pixel scores
Combine with point branch as Phase 1/2: integrate = (color + point) / 2
```

**Test sample types:**
- Anomalous test images: native size (640×{512, 526, 551, 640}) — multi-patch sliding window
- Good test patches: already 256×256 — single-patch (no sliding) for parity
- All images get the same multi-photometric mean fusion

**Phase 1 + 2 deltas folded in:**
- Mean fusion (no learned params) is the headline; no fusion checkpoint needed.
- The unchanged point branch comes from PointAD's render-multi-view path with shipped `epoch_15.pth`.
- Comparison baseline: Phase 2 mean-fusion results (`results/welds_pointad_plus/ablations/mean_fusion/`).

**Spec:** `docs/superpowers/specs/2026-05-17-pointad-plus-design.md` §5.

**Repo state:** Not a git repo. No commits. Pytest passes + result file existence = gates.

---

## File structure

```
src/pointad_plus/
├── sliding_window.py                  # P3-A — patch generation + aggregation utilities
└── run_welds_pointad_plus_sw.py       # P3-B — sliding-window runner

tests/pointad_plus/
└── test_sliding_window.py             # P3-A

scripts/pointad_plus/
└── eval_welds_pointad_plus_sw.sh      # P3-C — wrapper

results/
└── welds_pointad_plus_sw/
    ├── raw_results.pkl
    └── SUMMARY.md                     # P3-D — per-defect vs Phase 2 mean fusion
```

---

## Task P3-A: Sliding-window utilities

**Files:**
- Create: `src/pointad_plus/sliding_window.py`
- Create: `tests/pointad_plus/test_sliding_window.py`

**Spec:**

```python
def patch_coords(img_h: int, img_w: int, patch_size: int = 256,
                 stride: int = 128) -> list[tuple[int, int]]:
    """Return list of (u, v) top-left pixel coords for non-overflowing patches."""

def aggregate_overlap_mean(
    patch_maps: list[np.ndarray],
    coords: list[tuple[int, int]],
    img_h: int, img_w: int,
    patch_size: int = 256,
) -> np.ndarray:
    """Overlap-mean aggregation of per-patch anomaly maps to full-image map.
    patch_maps[i] is (patch_size, patch_size). coords[i] is (u, v).
    Returns (img_h, img_w) array. Pixels not covered by any patch → 0.
    """
```

- [ ] **P3-A.1: Write failing tests**

```python
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
```

- [ ] **P3-A.2: Confirm failure**

```
python -m pytest tests/pointad_plus/test_sliding_window.py -v
```

Expected: ImportError or all fail.

- [ ] **P3-A.3: Implement**

```python
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
    acc = np.zeros((img_h, img_w), dtype=np.float64)
    cov = np.zeros((img_h, img_w), dtype=np.float64)
    for pm, (u, v) in zip(patch_maps, coords):
        acc[v:v + patch_size, u:u + patch_size] += pm
        cov[v:v + patch_size, u:u + patch_size] += 1.0
    out = np.zeros((img_h, img_w), dtype=np.float32)
    mask = cov > 0
    out[mask] = (acc[mask] / cov[mask]).astype(np.float32)
    return out
```

- [ ] **P3-A.4: Run tests — expect pass**

Expected: 6 passed.

---

## Task P3-B: Sliding-window runner

**Files:**
- Create: `src/pointad_plus/run_welds_pointad_plus_sw.py`

**Approach:** Fork the existing Phase 2 runner (`src/pointad_plus/run_welds_pointad_plus.py`). The integration changes:

1. **Per-sample loop:** for each test sample, load 5 native-resolution photometric PNGs.
2. **Determine patch coords:** for a 256×256 input image, single patch at (0,0). For larger, slide-window.
3. **Per-patch CLIP encoding:** stack the 5 channels' patch crops as a (5, 3, 256, 256) batch, run CLIP-ViT-L (with the appropriate resize/normalize PointAD uses), produce 5 token maps of shape (5, N, D).
4. **Mean fusion** (Phase 2 finding): `fused_tokens = stacked_tokens.mean(dim=0)` → (N, D). Equivalent to instantiating `MultiPhotometricFusion(init="mean")` with no checkpoint — keep it simple, just take mean.
5. **PointAD prompt head:** same as Phase 2 runner.
6. **Per-patch anomaly map:** native PointAD output, resize to 256×256 if needed.
7. **Aggregate:** call `aggregate_overlap_mean` across the patches of this sample.
8. **Image score:** `agg_map.max()`.
9. **Save** per-sample image score + aggregated map shape to `raw_results.pkl`.

**CLI:**

```
python -m src.pointad_plus.run_welds_pointad_plus_sw \
  --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
  --pointad_ckpt external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth \
  --patch_size 256 --stride 128 \
  --out results/welds_pointad_plus_sw/
```

Note: **the manifest from P2-A is reused unchanged** — each test sample already lists 5 native-resolution photometric paths.

- [ ] **P3-B.1: Read the Phase 2 runner**

`src/pointad_plus/run_welds_pointad_plus.py` is the template. Note exactly where CLIP is invoked, where the color-branch tokens become the anomaly map, and where the late fusion with the point branch happens. Re-use the import-and-CLIP-setup block.

- [ ] **P3-B.2: Implement the sliding-window runner**

(Subagent will copy and adapt Phase 2 runner. The per-patch encode + mean + aggregate is the new logic.)

- [ ] **P3-B.3: Smoke-run on 5 samples**

Run with `--limit 5` (if the flag exists; else manually filter). Verify:
- Per-sample patch count is sensible (1 for 256×256 inputs; 12-ish for 640×~580)
- No NaN in output
- Aggregate map shape matches the original image dims
- Image score is in (0, 1)

- [ ] **P3-B.4: Run full eval on 694 samples**

Background run. Expected runtime: ~30-60 min on RTX 4090 (~12× more CLIP forwards per sample on full-size images, but only ~70% of samples are full-size since good samples are 256×256).

---

## Task P3-C: Eval + SUMMARY vs Phase 2 mean fusion

**Files:**
- Create: `results/welds_pointad_plus_sw/SUMMARY.md`

- [ ] **P3-C.1: Extract per-defect numbers**

From the saved `raw_results.pkl`, compute per-defect image-AUROC, pixel-AUROC (if available), and pixel-AUPRO (if available). Compare to:
- Phase 2 mean fusion (`results/welds_pointad_plus/ablations/mean_fusion/raw_results.pkl`)
- Phase 1 single-Phong baseline (`results/welds_zero_shot/raw_results.pkl`)

- [ ] **P3-C.2: Write SUMMARY**

```markdown
# PointAD+ Phase 3 — Sliding-Window Inference at Native Resolution

Method: mean fusion (Phase 2 finding) + sliding 256×256 window with stride 128
PointAD source checkpoint: `exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth`
Test set: 694 samples (same as Phase 2)

## Aggregate

| Method | pixel-AUROC | pixel-AUPRO | image-AUROC | image-AP |
|---|---|---|---|---|
| Phase 1 single Phong | 94.1 | 71.7 | 88.5 | 92.9 |
| Phase 2 mean fusion | 93.7 | 69.1 | 89.8 | 96.8 |
| **Phase 3 mean fusion + sliding window** | … | … | … | … |

## Per-defect image-AUROC (integrate)

| Defect | n | Phase 1 | Phase 2 | Phase 3 | Phase 3 Δ vs Phase 2 |
|---|---|---|---|---|---|
| pseudo_soldering | 206 | 75.98 | 76.36 | … | … |
| pinhole | 11 | 92.28 | 96.49 | … | … |
| pit | 10 | 97.45 | 99.48 | … | … |
| burst | 3 | 97.17 | 99.78 | … | … |
| fish_scale_welding | 21 | 85.31 | 90.16 | … | … |
| bump | 19 | 96.59 | 98.00 | … | … |
| combined | 271 | 97.15 | 98.71 | … | … |

## Phase 3 success-gate check
- Phase 3 vs Phase 2 mean fusion: ≥3 absolute pixel-AUROC improvement on aggregate? …
- pseudo_soldering lift beyond 76.36? …
- Image-AUROC aggregate beats 89.8? …

## Interpretation
[2-3 sentences. Did sliding window help where it should have (high-res-sensitive defects: pinhole, pit, pseudo_soldering)?]
```

---

## Phase 3 success gates (from spec §5.5)

- Sliding-window-aggregated PointAD+ beats PointAD baseline by ≥5 absolute P-AUROC on welds in ≥2 of 3 train-aux scenarios. Phase 1+2 covered carrot only; this Phase covers carrot-only too. So the gate becomes: Phase 3 (carrot) ≥ Phase 1 (carrot) + 5 P-AUROC.
- OR (revised, given Phase 2's mean-fusion baseline): Phase 3 ≥ Phase 2 mean fusion + 2 absolute pixel-AUROC AND pseudo_soldering ≥ 78 (target +2 above current ceiling).

Soft gates:
- Aggregate pixel-AUROC up — primary
- pseudo_soldering integrate up — secondary (most important per-class number)
- No regression worse than -0.5 anywhere

---

## Notes for the implementing engineer

- Working dir: `F:/dataset/LUT_AD_DataSet`.
- No git commits. Pytest + result-file existence = gate.
- Don't edit anything inside `external/PointAD/`.
- Phase 2 runner at `src/pointad_plus/run_welds_pointad_plus.py` is the template — copy and adapt.
- Phase 2 manifest at `external/datasets/welds_pointad_mp/weld/all_meta.json` is reused; each sample has 5 native-resolution photometric paths.
- The `multi_photo` field in each sample points at the native-resolution PNGs (e.g., 640×580 for anomalous test). `Dataset_3D/Eyecandies_Weld/weld/test/<class>/{lut,phong,…}/<id>.png`.
- Mean fusion is parameter-free: `fused_tokens = stack([T1..T5], dim=0).mean(dim=0)`. No fusion-checkpoint loading needed.
- PointAD's expected input to its CLIP wrapper is 336×336 — yes, even for our 256×256 patches. The CLIP backbone resizes internally. So a "256×256 patch" through CLIP becomes 336×336 internally — this is fine; the key is the SLIDING preserves spatial detail across an image larger than 336.
- For the aggregate map: each patch's anomaly map is at the CLIP output resolution (which is some upsample of the 24×24 patch tokens). Resize each patch map to 256×256 (the patch input size) before overlap-mean.
- Sliding doesn't apply to good test patches (already 256×256) — single-patch path. The runner must handle both cases.
