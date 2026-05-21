# 3D Dataset Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate three sibling 3D-anomaly-detection–style dataset variants (MVTec3D-style, Eyecandies-style, Real3D-AD-style) from the existing 549 `.tif` depth maps + YOLO bbox labels, reproducible via a one-shot script pipeline.

**Architecture:** Single shared preprocessing pass (good-patch harvesting, dual mask generation, XYZ unprojection) caches intermediate artifacts; three build scripts consume the cache to materialize the three output layouts; a verification script enforces invariants across all three outputs.

**Tech Stack:** Python 3.10+, `numpy`, `tifffile`, `Pillow`, `PyYAML`, `pytest`, Meta's `sam2` package for SAM2-Hiera-Large segmentation. Pure-Python ASCII PCD writer (no `open3d` dependency).

**Repo state note:** This directory is **not** a git repository. Skip all "commit" steps that would otherwise appear at the end of each task. Use the verification script and task-end smoke runs as the integration checkpoint instead.

**Source spec:** `docs/superpowers/specs/2026-05-16-3d-dataset-extension-design.md`

---

## File structure

```
cfg/dataset_3d.yaml                          (Task 0)

scripts/dataset_3d/__init__.py               (Task 0 — empty marker)
scripts/dataset_3d/common.py                 (Task 0 — config, paths, IO)
scripts/dataset_3d/harvest_good.py           (Task 1)
scripts/dataset_3d/segment_depth_threshold.py (Task 2)
scripts/dataset_3d/segment_sam2.py           (Task 3)
scripts/dataset_3d/unproject_xyz.py          (Task 4)
scripts/dataset_3d/build_mvtec3d.py          (Task 5)
scripts/dataset_3d/build_eyecandies.py       (Task 6)
scripts/dataset_3d/build_real3d.py           (Task 7)
scripts/dataset_3d/verify_dataset.py         (Task 8)

tests/dataset_3d/__init__.py                 (Task 0)
tests/dataset_3d/conftest.py                 (Task 0 — shared fixtures)
tests/dataset_3d/test_common.py              (Task 0)
tests/dataset_3d/test_harvest_good.py        (Task 1)
tests/dataset_3d/test_segment_depth.py       (Task 2)
tests/dataset_3d/test_segment_sam2.py        (Task 3)
tests/dataset_3d/test_unproject.py           (Task 4)
tests/dataset_3d/test_build_mvtec3d.py       (Task 5)
tests/dataset_3d/test_build_eyecandies.py    (Task 6)
tests/dataset_3d/test_build_real3d.py        (Task 7)
tests/dataset_3d/test_verify_dataset.py      (Task 8)

docs/dataset_3d.md                           (Task 9)
docs/dataset_3d.zh-CN.md                     (Task 9)

.gitignore                                   (Task 0 — append Dataset_3D/ and .cache/)
```

Each file in `scripts/dataset_3d/` has **one responsibility** matching its name. `common.py` is the only place that loads config and resolves source-data paths; every other script imports from it.

---

## Task 0: Bootstrap — config, common utilities, test scaffolding

**Files:**
- Create: `cfg/dataset_3d.yaml`
- Create: `scripts/dataset_3d/__init__.py`
- Create: `scripts/dataset_3d/common.py`
- Create: `tests/dataset_3d/__init__.py`
- Create: `tests/dataset_3d/conftest.py`
- Create: `tests/dataset_3d/test_common.py`
- Modify (or create): `.gitignore` — append two lines

### Step 0.1: Write `cfg/dataset_3d.yaml`

- [ ] Create `cfg/dataset_3d.yaml` with exactly this content:

```yaml
source:
  tif_dir: F:/dataset/LUT_AD_DataSet/Crop_Data/Tif
  label_dir: F:/dataset/LUT_AD_DataSet/Crop_Data/label
  modality_roots:
    lut: F:/dataset/LUT_AD_DataSet/Train_Data/LUT
    phong: F:/dataset/LUT_AD_DataSet/Train_Data/Phong
    diffuse: F:/dataset/LUT_AD_DataSet/Train_Data/Diffuse
    specular: F:/dataset/LUT_AD_DataSet/Train_Data/Specular
    normal: F:/dataset/LUT_AD_DataSet/Train_Data/Normal
    depth: F:/dataset/LUT_AD_DataSet/Train_Data/DepthMap

cache_dir: F:/dataset/LUT_AD_DataSet/.cache
output_root: F:/dataset/LUT_AD_DataSet/Dataset_3D

camera:
  pixel_size_mm_x: 0.016
  pixel_size_mm_y: 0.016
  depth_scale_mm: 1.0
  invalid_threshold: -1.0e30

good_harvest:
  patch_size: 256
  stride: 128
  max_patches_per_source: 4
  max_invalid_fraction: 0.05
  split_ratios: [0.70, 0.15, 0.15]
  shuffle_seed: 42

sam2:
  checkpoint: F:/dataset/LUT_AD_DataSet/.cache/sam2/sam2_hiera_large.pt
  config: sam2_hiera_l.yaml
  device: cuda:0

depth_threshold:
  k: 2.5
  neighborhood: 5

real3d:
  n_templates: 4
  template_selection: first_after_shuffle
  pcd_format: ascii

classes:
  0: pseudo_soldering
  1: pinhole
  2: pit
  3: burst
  4: fish_scale_welding
  5: bump
```

### Step 0.2: Write `scripts/dataset_3d/__init__.py`

- [ ] Create an empty file at `scripts/dataset_3d/__init__.py` (zero bytes is fine).

### Step 0.3: Write the failing tests for `common.py`

- [ ] Create `tests/dataset_3d/__init__.py` (empty).

- [ ] Create `tests/dataset_3d/conftest.py`:

```python
"""Shared fixtures for dataset_3d tests."""
from pathlib import Path
import pytest

REPO_ROOT = Path("F:/dataset/LUT_AD_DataSet")
CONFIG_PATH = REPO_ROOT / "cfg" / "dataset_3d.yaml"


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def config_path() -> Path:
    return CONFIG_PATH
```

- [ ] Create `tests/dataset_3d/test_common.py`:

```python
"""Tests for scripts.dataset_3d.common."""
from pathlib import Path
import numpy as np
import pytest

from scripts.dataset_3d import common


def test_load_config_returns_dict(config_path):
    cfg = common.load_config(config_path)
    assert cfg["camera"]["pixel_size_mm_x"] == 0.016
    assert cfg["good_harvest"]["patch_size"] == 256
    assert cfg["classes"][0] == "pseudo_soldering"
    assert cfg["classes"][5] == "bump"


def test_iter_source_ids_returns_549_ids(config_path):
    cfg = common.load_config(config_path)
    ids = list(common.iter_source_ids(cfg))
    assert len(ids) == 549
    assert all(isinstance(i, str) for i in ids)
    assert "1" in ids


def test_read_depth_tif_returns_float32_2d(config_path):
    cfg = common.load_config(config_path)
    arr = common.read_depth_tif(cfg, "1")
    assert arr.dtype == np.float32
    assert arr.ndim == 2


def test_read_labels_for_id_1_has_one_box(config_path):
    cfg = common.load_config(config_path)
    boxes = common.read_labels(cfg, "1")
    assert len(boxes) == 1
    cls, cx, cy, w, h = boxes[0]
    assert cls == 0
    assert 0 < cx < 1 and 0 < cy < 1
    assert 0 < w <= 1 and 0 < h <= 1


def test_read_labels_excludes_classes_txt(config_path):
    cfg = common.load_config(config_path)
    # If classes.txt were not excluded, the loader would crash trying to
    # parse "Pseudo soldering" as a class index. The fixture just needs
    # any valid id to trigger directory iteration in the loader.
    common.read_labels(cfg, "1")  # must not raise


def test_resolve_modality_image_finds_phong(config_path):
    cfg = common.load_config(config_path)
    p = common.resolve_modality_image(cfg, "phong", "1")
    assert p.exists()
    assert p.suffix == ".png"


def test_resolve_modality_image_missing_raises(config_path):
    cfg = common.load_config(config_path)
    with pytest.raises(FileNotFoundError):
        common.resolve_modality_image(cfg, "phong", "999999")


def test_invalid_mask_helper_uses_threshold(config_path):
    cfg = common.load_config(config_path)
    a = np.array([[-1e30, 0.0, 1.5], [-1e38, 2.0, -1e29]], dtype=np.float32)
    invalid = common.invalid_mask(a, cfg["camera"]["invalid_threshold"])
    expected = np.array([[True, False, False], [True, False, False]])
    assert (invalid == expected).all()
```

### Step 0.4: Run tests — expect ALL to fail

- [ ] Run from repo root:

```
pytest tests/dataset_3d/test_common.py -v
```

- Expected: `ModuleNotFoundError: No module named 'scripts.dataset_3d.common'` (or all collected tests fail).

### Step 0.5: Implement `scripts/dataset_3d/common.py`

- [ ] Create `scripts/dataset_3d/common.py` with this content:

```python
"""Shared helpers for the dataset_3d pipeline.

Loads the YAML config, resolves source-data paths, reads .tif depth maps
and YOLO labels. Imported by every other script in this package.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tifffile
import yaml


@dataclass(frozen=True)
class Box:
    """YOLO bbox in normalized coords: cx, cy, w, h in (0, 1)."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    def to_pixel_xyxy(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Return integer pixel (x1, y1, x2, y2), clamped to image bounds."""
        x1 = int(round((self.cx - self.w / 2) * img_w))
        y1 = int(round((self.cy - self.h / 2) * img_h))
        x2 = int(round((self.cx + self.w / 2) * img_w))
        y2 = int(round((self.cy + self.h / 2) * img_h))
        x1 = max(0, min(img_w, x1))
        x2 = max(0, min(img_w, x2))
        y1 = max(0, min(img_h, y1))
        y2 = max(0, min(img_h, y2))
        return x1, y1, x2, y2


def load_config(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_source_ids(cfg: dict) -> Iterable[str]:
    tif_dir = Path(cfg["source"]["tif_dir"])
    for p in sorted(tif_dir.glob("*.tif")):
        yield p.stem


def read_depth_tif(cfg: dict, source_id: str) -> np.ndarray:
    p = Path(cfg["source"]["tif_dir"]) / f"{source_id}.tif"
    arr = tifffile.imread(str(p))
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    return arr


def read_labels(cfg: dict, source_id: str) -> List[Box]:
    """Read YOLO labels for a single source id. Returns [] for empty files.

    Skips the `classes.txt` sidecar that lives alongside the labels.
    """
    p = Path(cfg["source"]["label_dir"]) / f"{source_id}.txt"
    if not p.exists():
        return []
    boxes: List[Box] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:])
            boxes.append(Box(cls=cls, cx=cx, cy=cy, w=w, h=h))
    return boxes


def resolve_modality_image(cfg: dict, modality: str, source_id: str) -> Path:
    """Find <source_id>.png under one of train/val/test under the modality root.

    Each source id lives in exactly one of the three splits. Raises
    FileNotFoundError if the id is not found in any split.
    """
    root = Path(cfg["source"]["modality_roots"][modality])
    for split in ("train", "val", "test"):
        candidate = root / split / "images" / f"{source_id}.png"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {modality} image for source id {source_id} under {root}"
    )


def resolve_modality_depth_tif(cfg: dict, source_id: str) -> Path | None:
    """Find <source_id>.tif under DepthMap/{train,val,test}/images/.

    Returns None if the DepthMap modality is unavailable (some setups
    don't ship rendered DepthMap; Eyecandies builder will fall back to
    re-encoding the raw .tif in that case).
    """
    root = Path(cfg["source"]["modality_roots"]["depth"])
    for split in ("train", "val", "test"):
        for ext in (".tif", ".tiff", ".png"):
            candidate = root / split / "images" / f"{source_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def invalid_mask(depth: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of pixels considered invalid (depth <= threshold)."""
    return depth <= threshold
```

### Step 0.6: Run tests — expect ALL to pass

- [ ] Run:

```
pytest tests/dataset_3d/test_common.py -v
```

- Expected: 8 passed.

### Step 0.7: Append cache + output paths to `.gitignore`

- [ ] If `.gitignore` exists at repo root, append:

```
# 3D dataset extension generated artifacts
.cache/
Dataset_3D/
```

- [ ] If `.gitignore` does not exist, create it with just those three lines.

---

## Task 1: Good-patch harvesting

**Files:**
- Create: `scripts/dataset_3d/harvest_good.py`
- Create: `tests/dataset_3d/test_harvest_good.py`

**Behavior to implement (from spec §3.1):**
- Slide a `patch_size`×`patch_size` window with `stride` over each `.tif`.
- For anomalous images: keep windows whose IoU with every defect bbox is 0.
- For native-good images (empty label file): keep all windows.
- Drop any window whose fraction of invalid pixels (`depth <= invalid_threshold`) exceeds `max_invalid_fraction`.
- Cap ≤`max_patches_per_source` per source image. If more candidates remain, sample uniformly at random with `shuffle_seed`.
- Aggregate all kept patches across all 549 sources, shuffle once with `shuffle_seed`, split 70/15/15 → train/val/test_good.
- Write each kept patch as a 256×256 float32 .tif to `.cache/good_patches/{train,val,test_good}/<source_stem>_<u>_<v>.tif` where `(u, v)` is the patch's top-left pixel.

### Step 1.1: Write failing tests

- [ ] Create `tests/dataset_3d/test_harvest_good.py`:

```python
"""Tests for harvest_good module."""
from pathlib import Path
import numpy as np
import pytest

from scripts.dataset_3d import common, harvest_good


def test_iou_zero_when_disjoint():
    # patch at (0,0)-(10,10), box at (20,20)-(30,30)
    assert harvest_good.box_iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_positive_when_overlap():
    iou = harvest_good.box_iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0 < iou < 1


def test_iou_one_when_identical():
    assert harvest_good.box_iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_invalid_fraction_threshold():
    patch = np.full((4, 4), -1e38, dtype=np.float32)
    patch[0, 0] = 1.0  # 1/16 invalid → 15/16 invalid
    frac = harvest_good.invalid_fraction(patch, -1e30)
    assert frac == pytest.approx(15 / 16)


def test_collect_candidates_excludes_bbox_overlap():
    # 512x512 image, one bbox covering top-left quadrant
    img_h, img_w = 512, 512
    boxes_px = [(0, 0, 256, 256)]
    patches = list(
        harvest_good.iter_candidate_patches(
            img_h=img_h, img_w=img_w, patch_size=256, stride=128, boxes_px=boxes_px
        )
    )
    # Patches whose top-left is at u ∈ {0,128} AND v ∈ {0,128} have IoU>0
    # with the bbox; only patches starting at u>=128 AND v>=128 should pass
    # ... and the only fully-non-overlapping patch is (256, 256).
    assert (256, 256) in patches
    assert (0, 0) not in patches
    assert (128, 128) not in patches


def test_collect_candidates_native_good_keeps_all():
    patches = list(
        harvest_good.iter_candidate_patches(
            img_h=512, img_w=512, patch_size=256, stride=128, boxes_px=[]
        )
    )
    # Strides: u in {0,128,256}, v in {0,128,256} (patches must fit), → 9
    assert len(patches) == 9


def test_split_indices_deterministic():
    # Same seed must give same partition.
    a = harvest_good.split_indices(n=100, ratios=(0.7, 0.15, 0.15), seed=42)
    b = harvest_good.split_indices(n=100, ratios=(0.7, 0.15, 0.15), seed=42)
    assert a == b
    assert len(a[0]) == 70
    assert len(a[1]) == 15
    assert len(a[2]) == 15


def test_end_to_end_writes_patches(tmp_path, config_path, monkeypatch):
    cfg = common.load_config(config_path)
    # Redirect cache to tmp_path
    cfg["cache_dir"] = str(tmp_path)
    # Use a tiny subset by capping to 2 patches/source
    cfg["good_harvest"]["max_patches_per_source"] = 2
    harvest_good.run(cfg, source_ids=["1", "2", "3", "4", "5", "6", "7", "8"])
    cache = tmp_path / "good_patches"
    train_files = list((cache / "train").glob("*.tif"))
    val_files = list((cache / "val").glob("*.tif"))
    test_files = list((cache / "test_good").glob("*.tif"))
    total = len(train_files) + len(val_files) + len(test_files)
    assert total > 0
    # Every file is named <stem>_<u>_<v>.tif
    for p in train_files:
        parts = p.stem.rsplit("_", 2)
        assert len(parts) == 3
        int(parts[1])
        int(parts[2])
```

### Step 1.2: Run tests — expect failure

- [ ] Run:

```
pytest tests/dataset_3d/test_harvest_good.py -v
```

- Expected: ModuleNotFoundError on `harvest_good`, all 8 tests fail.

### Step 1.3: Implement `scripts/dataset_3d/harvest_good.py`

- [ ] Create `scripts/dataset_3d/harvest_good.py`:

```python
"""Step 1 of the dataset_3d pipeline: harvest 256x256 'good' depth patches.

For each .tif, slide a patch_size window with `stride`. Anomalous source
images keep patches whose IoU with every defect bbox is 0; native-good
images (empty label file) keep all valid patches. Invalid-pixel fraction
must be <= max_invalid_fraction. At most max_patches_per_source kept per
image (uniform random subsample with the run seed if more candidates
remain). All kept patches are aggregated, shuffled once with seed, and
split 70/15/15 into train/val/test_good directories.

Usage:
    python -m scripts.dataset_3d.harvest_good [--force]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import tifffile

from scripts.dataset_3d import common

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cfg" / "dataset_3d.yaml"


def box_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def invalid_fraction(patch: np.ndarray, invalid_threshold: float) -> float:
    invalid = patch <= invalid_threshold
    return float(invalid.sum()) / float(patch.size)


def iter_candidate_patches(
    img_h: int,
    img_w: int,
    patch_size: int,
    stride: int,
    boxes_px: Sequence[Tuple[int, int, int, int]],
) -> Iterable[Tuple[int, int]]:
    """Yield (u, v) top-left pixel coords for candidate patches.

    A candidate is kept only if its bbox-overlap (IoU) with every defect
    box is exactly 0. Empty `boxes_px` keeps every in-bounds patch.
    """
    for v in range(0, img_h - patch_size + 1, stride):
        for u in range(0, img_w - patch_size + 1, stride):
            patch_box = (u, v, u + patch_size, v + patch_size)
            ok = all(box_iou_xyxy(patch_box, b) == 0 for b in boxes_px)
            if ok:
                yield (u, v)


def split_indices(
    n: int, ratios: Tuple[float, float, float], seed: int
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministic 3-way partition of range(n) under `seed`."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test_good = order[n_train + n_val :]
    return train, val, test_good


def _yolo_boxes_to_pixel(
    boxes: List[common.Box], img_w: int, img_h: int
) -> List[Tuple[int, int, int, int]]:
    return [b.to_pixel_xyxy(img_w, img_h) for b in boxes]


def run(cfg: dict, source_ids: Iterable[str] | None = None) -> None:
    cache_root = Path(cfg["cache_dir"]) / "good_patches"
    for split in ("train", "val", "test_good"):
        (cache_root / split).mkdir(parents=True, exist_ok=True)

    hcfg = cfg["good_harvest"]
    patch_size = hcfg["patch_size"]
    stride = hcfg["stride"]
    max_per_src = hcfg["max_patches_per_source"]
    max_invalid = hcfg["max_invalid_fraction"]
    seed = hcfg["shuffle_seed"]
    invalid_threshold = cfg["camera"]["invalid_threshold"]

    rng = random.Random(seed)
    kept: List[Tuple[str, int, int, np.ndarray]] = []

    if source_ids is None:
        source_ids = list(common.iter_source_ids(cfg))
    else:
        source_ids = list(source_ids)

    for src_id in source_ids:
        depth = common.read_depth_tif(cfg, src_id)
        h, w = depth.shape[:2]
        boxes = common.read_labels(cfg, src_id)
        boxes_px = _yolo_boxes_to_pixel(boxes, img_w=w, img_h=h)

        candidates: List[Tuple[int, int]] = []
        for u, v in iter_candidate_patches(h, w, patch_size, stride, boxes_px):
            patch = depth[v : v + patch_size, u : u + patch_size]
            if invalid_fraction(patch, invalid_threshold) <= max_invalid:
                candidates.append((u, v))

        if len(candidates) > max_per_src:
            rng.shuffle(candidates)
            candidates = candidates[:max_per_src]

        for u, v in candidates:
            patch = depth[v : v + patch_size, u : u + patch_size].copy()
            kept.append((src_id, u, v, patch))

    print(f"[harvest_good] collected {len(kept)} candidate patches across "
          f"{len(source_ids)} source images")

    train_ix, val_ix, test_ix = split_indices(
        n=len(kept), ratios=tuple(hcfg["split_ratios"]), seed=seed
    )

    split_map = {"train": train_ix, "val": val_ix, "test_good": test_ix}
    for split, indices in split_map.items():
        out_dir = cache_root / split
        for i in indices:
            src_id, u, v, patch = kept[i]
            out_path = out_dir / f"{src_id}_{u}_{v}.tif"
            tifffile.imwrite(str(out_path), patch)
        print(f"[harvest_good] {split}: {len(indices)} patches → {out_dir}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true", help="ignored placeholder for"
                    " idempotency parity with other pipeline scripts")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    _main()
```

### Step 1.4: Run tests — expect pass

- [ ] Run:

```
pytest tests/dataset_3d/test_harvest_good.py -v
```

- Expected: 8 passed. The `test_end_to_end_writes_patches` test reads 8 real `.tif`s from `Crop_Data/Tif/` — that is intentional. If it fails on file IO, debug paths in `conftest.py`/`common.py` before proceeding.

### Step 1.5: Smoke-run the script on full dataset

- [ ] Run:

```
python -m scripts.dataset_3d.harvest_good
```

- Expected: prints "collected N candidate patches across 549 source images" where N is in the range 1500–2500, then three split-line totals that sum to N. Each `.cache/good_patches/{train,val,test_good}/` is populated.

---

## Task 2: Depth-threshold segmentation (baseline mask)

**Files:**
- Create: `scripts/dataset_3d/segment_depth_threshold.py`
- Create: `tests/dataset_3d/test_segment_depth.py`

**Behavior (spec §3.2 — depth-threshold):**
- For each anomalous source `.tif`, for each defect bbox:
  - Extract the bbox region from depth, dilated by `neighborhood` pixels.
  - Compute local median and MAD-based robust σ (= 1.4826 × MAD) **on valid pixels only**.
  - Inside the bbox (un-dilated), mark pixels where `|depth − median| > k · σ` AND `depth > invalid_threshold`.
- Output per source image:
  - `.cache/masks/depth/<id>_full.png` — union of all per-bbox masks (255 = defect).
  - `.cache/masks/depth/<id>_cls{0..5}.png` — per-class union (only emitted for classes present in that image).
- Skip writing the cls-{c} mask if class c does not appear in that image.

### Step 2.1: Write failing tests

- [ ] Create `tests/dataset_3d/test_segment_depth.py`:

```python
"""Tests for segment_depth_threshold module."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from scripts.dataset_3d import common, segment_depth_threshold as sdt


def test_mad_sigma_constant_array_is_zero():
    arr = np.full(100, 5.0, dtype=np.float32)
    assert sdt.mad_sigma(arr) == 0.0


def test_mad_sigma_matches_known_value():
    # MAD of [1,2,3,4,5] is 1.0 → sigma = 1.4826
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    sigma = sdt.mad_sigma(arr)
    assert sigma == pytest.approx(1.4826, rel=1e-3)


def test_mad_sigma_ignores_invalid():
    arr = np.array([-1e38, 1, 2, 3, 4, 5, -1e38], dtype=np.float32)
    valid = arr > -1e30
    sigma = sdt.mad_sigma(arr[valid])
    assert sigma == pytest.approx(1.4826, rel=1e-3)


def test_threshold_mask_marks_outliers_only():
    depth = np.full((10, 10), 1.0, dtype=np.float32)
    depth[5, 5] = 10.0  # one big outlier
    depth[5, 6] = 1.05  # tiny perturbation, well within k*sigma
    bbox = (3, 3, 8, 8)  # x1, y1, x2, y2
    mask = sdt.depth_threshold_mask(
        depth=depth,
        bbox=bbox,
        neighborhood=2,
        k=2.5,
        invalid_threshold=-1e30,
    )
    assert mask.shape == depth.shape
    assert mask.dtype == np.uint8
    assert mask[5, 5] == 255
    assert mask[5, 6] == 0
    # All pixels outside the bbox are 0
    assert mask[0, 0] == 0
    assert mask[9, 9] == 0


def test_run_writes_full_and_per_class_masks(tmp_path, config_path):
    cfg = common.load_config(config_path)
    cfg["cache_dir"] = str(tmp_path)
    # source id 1 has exactly one class-0 box → expect _full and _cls0
    sdt.run(cfg, source_ids=["1"])
    out = tmp_path / "masks" / "depth"
    assert (out / "1_full.png").exists()
    assert (out / "1_cls0.png").exists()
    full = np.array(Image.open(out / "1_full.png"))
    cls0 = np.array(Image.open(out / "1_cls0.png"))
    assert full.dtype == np.uint8
    # For a one-class-one-box image, full and cls0 are identical
    assert (full == cls0).all()
    # Other classes must NOT have files written
    for c in range(1, 6):
        assert not (out / f"1_cls{c}.png").exists()
```

### Step 2.2: Run tests — expect failure

- [ ] Run:

```
pytest tests/dataset_3d/test_segment_depth.py -v
```

- Expected: 5 fails (module missing).

### Step 2.3: Implement `scripts/dataset_3d/segment_depth_threshold.py`

- [ ] Create `scripts/dataset_3d/segment_depth_threshold.py`:

```python
"""Step 3 of the pipeline: depth-threshold (baseline) mask generation.

Independent of SAM2 — provides a robust fallback in case SAM2 produces
poor masks on welded surfaces. Each defect bbox is masked by an
MAD-thresholded outlier test on the local neighborhood.

Usage:
    python -m scripts.dataset_3d.segment_depth_threshold [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from scripts.dataset_3d import common

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cfg" / "dataset_3d.yaml"


def mad_sigma(values: np.ndarray) -> float:
    """Robust σ estimate from a 1-D array of valid samples.

    σ ≈ 1.4826 · median(|x - median(x)|). Returns 0 on an empty or
    constant input.
    """
    if values.size == 0:
        return 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return 1.4826 * mad


def depth_threshold_mask(
    depth: np.ndarray,
    bbox: Tuple[int, int, int, int],
    neighborhood: int,
    k: float,
    invalid_threshold: float,
) -> np.ndarray:
    """Return uint8 mask (full image size), 255 inside-bbox outliers, else 0.

    The local median and σ are estimated on the *dilated* bbox to give a
    larger sample. Pixels are marked only inside the un-dilated bbox.
    Invalid pixels (depth ≤ invalid_threshold) are excluded from both
    sigma estimation and the final mask.
    """
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = bbox
    # Dilated region for sigma estimate
    dx1 = max(0, x1 - neighborhood)
    dy1 = max(0, y1 - neighborhood)
    dx2 = min(w, x2 + neighborhood)
    dy2 = min(h, y2 + neighborhood)

    region = depth[dy1:dy2, dx1:dx2]
    valid = region > invalid_threshold
    region_valid = region[valid]

    mask = np.zeros((h, w), dtype=np.uint8)
    if region_valid.size == 0:
        return mask

    med = float(np.median(region_valid))
    sigma = mad_sigma(region_valid)
    if sigma == 0.0:
        return mask

    # Mark outliers inside the un-dilated bbox only
    inner = depth[y1:y2, x1:x2]
    inner_valid = inner > invalid_threshold
    outlier = np.abs(inner - med) > k * sigma
    keep = inner_valid & outlier
    mask[y1:y2, x1:x2][keep] = 255
    return mask


def _per_image(cfg: dict, source_id: str, out_dir: Path) -> None:
    depth = common.read_depth_tif(cfg, source_id)
    h, w = depth.shape[:2]
    boxes = common.read_labels(cfg, source_id)
    if not boxes:
        # Native-good image: no defect bbox → nothing to write.
        return

    dt_cfg = cfg["depth_threshold"]
    invalid_threshold = cfg["camera"]["invalid_threshold"]

    full = np.zeros((h, w), dtype=np.uint8)
    per_class: dict[int, np.ndarray] = {}

    for b in boxes:
        bbox_px = b.to_pixel_xyxy(img_w=w, img_h=h)
        m = depth_threshold_mask(
            depth=depth,
            bbox=bbox_px,
            neighborhood=dt_cfg["neighborhood"],
            k=dt_cfg["k"],
            invalid_threshold=invalid_threshold,
        )
        np.maximum(full, m, out=full)
        per_class.setdefault(b.cls, np.zeros((h, w), dtype=np.uint8))
        np.maximum(per_class[b.cls], m, out=per_class[b.cls])

    Image.fromarray(full).save(out_dir / f"{source_id}_full.png")
    for cls, m in per_class.items():
        Image.fromarray(m).save(out_dir / f"{source_id}_cls{cls}.png")


def run(cfg: dict, source_ids: Iterable[str] | None = None) -> None:
    out_dir = Path(cfg["cache_dir"]) / "masks" / "depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    if source_ids is None:
        source_ids = list(common.iter_source_ids(cfg))
    else:
        source_ids = list(source_ids)
    for sid in source_ids:
        _per_image(cfg, sid, out_dir)
    print(f"[segment_depth_threshold] wrote masks for {len(source_ids)} images → {out_dir}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    _main()
```

### Step 2.4: Run tests — expect pass

- [ ] Run:

```
pytest tests/dataset_3d/test_segment_depth.py -v
```

- Expected: 5 passed.

### Step 2.5: Smoke-run on the full dataset

- [ ] Run:

```
python -m scripts.dataset_3d.segment_depth_threshold
```

- Expected: prints "wrote masks for 549 images → .cache/masks/depth". Spot-check that `.cache/masks/depth/1_full.png` is a 640×551 grayscale PNG with some white pixels (open it; do not just check file existence).

---

## Task 3: SAM2 segmentation (primary mask)

**Files:**
- Create: `scripts/dataset_3d/segment_sam2.py`
- Create: `tests/dataset_3d/test_segment_sam2.py`

**Behavior (spec §3.2 — SAM2):**
- Download SAM2-Hiera-Large checkpoint to `.cache/sam2/sam2_hiera_large.pt` if absent.
- For each anomalous source `.tif`:
  - Load the Phong-render PNG from one of `Train_Data/Phong/{train,val,test}/images/`.
  - For each defect bbox in its label file, run SAM2 with that bbox as a box prompt.
  - Union per-class.
- Write `.cache/masks/sam2/<id>_full.png` and `.cache/masks/sam2/<id>_cls{c}.png` (same convention as depth-threshold).
- The SAM2 import path uses Meta's official package layout (`sam2.build_sam.build_sam2`, `sam2.sam2_image_predictor.SAM2ImagePredictor`).

### Step 3.1: Write smoke tests

These cannot be pure-unit because SAM2 itself is heavy. We write a smoke test gated on checkpoint availability, plus a unit test for the mask-union helper.

- [ ] Create `tests/dataset_3d/test_segment_sam2.py`:

```python
"""Tests for segment_sam2.

The full SAM2 pipeline requires a downloaded checkpoint and a CUDA GPU.
Tests that exercise the real model are auto-skipped unless the checkpoint
file is present at the configured path AND torch.cuda.is_available().
"""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from scripts.dataset_3d import common, segment_sam2 as ss2


def test_union_masks_takes_pixel_wise_max():
    a = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    b = np.array([[0, 0], [255, 0]], dtype=np.uint8)
    out = ss2.union_masks([a, b])
    assert (out == np.array([[0, 255], [255, 0]], dtype=np.uint8)).all()


def test_union_masks_empty_returns_zero(tmp_path):
    out = ss2.union_masks([], shape=(3, 4))
    assert out.shape == (3, 4)
    assert (out == 0).all()


def _sam2_available(cfg) -> bool:
    try:
        import torch
    except Exception:
        return False
    ckpt = Path(cfg["sam2"]["checkpoint"])
    return ckpt.exists() and torch.cuda.is_available()


@pytest.mark.slow
def test_segment_one_image_smoke(tmp_path, config_path):
    cfg = common.load_config(config_path)
    if not _sam2_available(cfg):
        pytest.skip("SAM2 checkpoint or CUDA not available")
    cfg["cache_dir"] = str(tmp_path)
    ss2.run(cfg, source_ids=["1"])
    out = tmp_path / "masks" / "sam2"
    full = np.array(Image.open(out / "1_full.png"))
    assert full.dtype == np.uint8
    assert (full == 255).sum() > 0  # SAM2 segmented at least one pixel
```

### Step 3.2: Run tests — expect failure (or skip on the slow one)

- [ ] Run:

```
pytest tests/dataset_3d/test_segment_sam2.py -v
```

- Expected: `test_union_masks_*` fail with ModuleNotFoundError; `test_segment_one_image_smoke` is collected and (likely) skipped.

### Step 3.3: Implement `scripts/dataset_3d/segment_sam2.py`

- [ ] Create `scripts/dataset_3d/segment_sam2.py`:

```python
"""Step 2 of the pipeline: SAM2-Hiera-Large mask generation.

Loads SAM2 once, iterates over every anomalous source image, and uses
each defect bbox as a box prompt. Writes the union-of-prompts mask and
per-class union masks to .cache/masks/sam2/.

Usage:
    python -m scripts.dataset_3d.segment_sam2 [--force]

Heavy dependency: requires `sam2` package and a CUDA GPU. The checkpoint
(~900 MB) is downloaded on first run if absent.
"""
from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

from scripts.dataset_3d import common

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cfg" / "dataset_3d.yaml"

SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
)


def union_masks(masks: Sequence[np.ndarray], shape: tuple | None = None) -> np.ndarray:
    if not masks:
        if shape is None:
            raise ValueError("Need shape to build an empty union mask")
        return np.zeros(shape, dtype=np.uint8)
    out = np.zeros_like(masks[0])
    for m in masks:
        np.maximum(out, m, out=out)
    return out


def _ensure_checkpoint(ckpt_path: Path) -> None:
    if ckpt_path.exists():
        return
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[segment_sam2] downloading SAM2 checkpoint → {ckpt_path}")
    urllib.request.urlretrieve(SAM2_CHECKPOINT_URL, str(ckpt_path))


def _load_predictor(cfg: dict):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    ckpt = Path(cfg["sam2"]["checkpoint"])
    _ensure_checkpoint(ckpt)
    sam2_model = build_sam2(
        cfg["sam2"]["config"],
        str(ckpt),
        device=cfg["sam2"]["device"],
    )
    return SAM2ImagePredictor(sam2_model)


def _per_image(cfg: dict, predictor, source_id: str, out_dir: Path) -> None:
    boxes = common.read_labels(cfg, source_id)
    if not boxes:
        return

    phong_path = common.resolve_modality_image(cfg, "phong", source_id)
    img = np.array(Image.open(phong_path).convert("RGB"))
    h, w = img.shape[:2]

    predictor.set_image(img)

    full = np.zeros((h, w), dtype=np.uint8)
    per_class: dict[int, np.ndarray] = {}

    for b in boxes:
        bbox_px = np.array(b.to_pixel_xyxy(img_w=w, img_h=h))[None, :]
        masks, _scores, _logits = predictor.predict(
            box=bbox_px, multimask_output=False
        )
        # SAM2 returns shape (1, H, W) bool
        m = (masks[0].astype(np.uint8)) * 255
        np.maximum(full, m, out=full)
        per_class.setdefault(b.cls, np.zeros((h, w), dtype=np.uint8))
        np.maximum(per_class[b.cls], m, out=per_class[b.cls])

    Image.fromarray(full).save(out_dir / f"{source_id}_full.png")
    for cls, m in per_class.items():
        Image.fromarray(m).save(out_dir / f"{source_id}_cls{cls}.png")


def run(cfg: dict, source_ids: Iterable[str] | None = None) -> None:
    out_dir = Path(cfg["cache_dir"]) / "masks" / "sam2"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictor = _load_predictor(cfg)
    if source_ids is None:
        source_ids = list(common.iter_source_ids(cfg))
    else:
        source_ids = list(source_ids)
    for i, sid in enumerate(source_ids):
        if (i + 1) % 25 == 0:
            print(f"[segment_sam2] {i + 1}/{len(source_ids)}")
        _per_image(cfg, predictor, sid, out_dir)
    print(f"[segment_sam2] wrote masks for {len(source_ids)} images → {out_dir}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    _main()
```

### Step 3.4: Run unit tests — expect pass

- [ ] Run:

```
pytest tests/dataset_3d/test_segment_sam2.py::test_union_masks_takes_pixel_wise_max tests/dataset_3d/test_segment_sam2.py::test_union_masks_empty_returns_zero -v
```

- Expected: 2 passed.

### Step 3.5: Install SAM2 (one-time; do NOT skip if testing on the workstation)

- [ ] Run from a shell with the project's Python active:

```
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

- If install fails on Windows: try the official wheel mirror, or skip and document the SAM2 step as a separate manual run in the dataset card. The smoke test above is the dependency-availability check.

### Step 3.6: Smoke-run on 5 images (do NOT run the full 549 in this task)

- [ ] Run a one-shot inside Python (replace the script invocation):

```
python -c "from pathlib import Path; from scripts.dataset_3d import common, segment_sam2; cfg = common.load_config(Path('cfg/dataset_3d.yaml')); segment_sam2.run(cfg, source_ids=['1','2','3','4','5'])"
```

- Expected: 5 `*_full.png` files under `.cache/masks/sam2/`. Visually inspect at least one — defect region should be non-zero.

- The full 549-image SAM2 pass takes ~1h and is deferred to Task 10.

---

## Task 4: XYZ unprojection (organized point clouds for MVTec3D layout)

**Files:**
- Create: `scripts/dataset_3d/unproject_xyz.py`
- Create: `tests/dataset_3d/test_unproject.py`

**Behavior (spec §3.3):**
- For each source `.tif` (all 549), produce `H×W×3` float32 .tiff under `.cache/xyz/<id>.tiff`.
- `X[v,u] = (u - W/2) · pixel_size_mm_x` (column offset from image center)
- `Y[v,u] = (v - H/2) · pixel_size_mm_y` (row offset from image center)
- `Z[v,u] = depth[v,u]` (already in mm; `depth_scale_mm` factor applied if not 1.0)
- Invalid pixels (`depth ≤ invalid_threshold`): write `(0, 0, 0)` at all three channels.
- Float32, exactly 3 channels, last axis = (X, Y, Z), no NaN.

### Step 4.1: Write failing tests

- [ ] Create `tests/dataset_3d/test_unproject.py`:

```python
"""Tests for unproject_xyz."""
from pathlib import Path
import numpy as np
import pytest
import tifffile

from scripts.dataset_3d import common, unproject_xyz


def test_center_pixel_xy_is_near_zero_for_even_size():
    depth = np.ones((4, 4), dtype=np.float32)
    xyz = unproject_xyz.depth_to_xyz(depth, px_x=0.016, px_y=0.016, depth_scale=1.0,
                                     invalid_threshold=-1e30)
    assert xyz.shape == (4, 4, 3)
    assert xyz.dtype == np.float32
    # u=2, v=2 (center of 4-wide image is between cols 1 and 2)
    # u-W/2 = 2-2 = 0 → X[2,2] = 0
    assert xyz[2, 2, 0] == pytest.approx(0.0)
    assert xyz[2, 2, 1] == pytest.approx(0.0)
    assert xyz[2, 2, 2] == pytest.approx(1.0)


def test_invalid_pixels_become_zero_triplet():
    depth = np.array([[1.0, -1e38], [2.0, -1e29]], dtype=np.float32)
    xyz = unproject_xyz.depth_to_xyz(depth, px_x=0.016, px_y=0.016, depth_scale=1.0,
                                     invalid_threshold=-1e30)
    # Pixel (0,1) is invalid → all three channels 0
    assert (xyz[0, 1] == 0.0).all()
    # Pixel (1,1) value -1e29 is > -1e30 → considered valid
    assert xyz[1, 1, 2] == pytest.approx(-1e29, rel=1e-3)


def test_depth_scale_applied():
    depth = np.full((2, 2), 5.0, dtype=np.float32)
    xyz = unproject_xyz.depth_to_xyz(depth, px_x=0.016, px_y=0.016, depth_scale=2.0,
                                     invalid_threshold=-1e30)
    # Z should be 5.0 * 2.0 = 10.0
    assert (xyz[..., 2] == 10.0).all()


def test_run_writes_xyz_tiff(tmp_path, config_path):
    cfg = common.load_config(config_path)
    cfg["cache_dir"] = str(tmp_path)
    unproject_xyz.run(cfg, source_ids=["1"])
    out = tmp_path / "xyz" / "1.tiff"
    assert out.exists()
    arr = tifffile.imread(str(out))
    assert arr.dtype == np.float32
    assert arr.shape[-1] == 3
    assert not np.isnan(arr).any()
```

### Step 4.2: Run tests — expect failure

- [ ] Run:

```
pytest tests/dataset_3d/test_unproject.py -v
```

- Expected: 4 fails (module missing).

### Step 4.3: Implement `scripts/dataset_3d/unproject_xyz.py`

- [ ] Create `scripts/dataset_3d/unproject_xyz.py`:

```python
"""Step 4 of the pipeline: unproject each depth .tif into an organized
XYZ point cloud (HxWx3 float32 .tiff) for the MVTec3D-style layout.

Usage:
    python -m scripts.dataset_3d.unproject_xyz [--force]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

from scripts.dataset_3d import common

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cfg" / "dataset_3d.yaml"


def depth_to_xyz(
    depth: np.ndarray,
    px_x: float,
    px_y: float,
    depth_scale: float,
    invalid_threshold: float,
) -> np.ndarray:
    h, w = depth.shape[:2]
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    x = (u - w / 2.0) * px_x  # broadcasts to (h, w)
    y = (v - h / 2.0) * px_y
    x = np.broadcast_to(x, (h, w)).copy()
    y = np.broadcast_to(y, (h, w)).copy()
    z = (depth * depth_scale).astype(np.float32)

    invalid = depth <= invalid_threshold
    x[invalid] = 0.0
    y[invalid] = 0.0
    z[invalid] = 0.0

    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)
    return xyz


def _per_image(cfg: dict, source_id: str, out_dir: Path) -> None:
    depth = common.read_depth_tif(cfg, source_id)
    cam = cfg["camera"]
    xyz = depth_to_xyz(
        depth=depth,
        px_x=cam["pixel_size_mm_x"],
        px_y=cam["pixel_size_mm_y"],
        depth_scale=cam["depth_scale_mm"],
        invalid_threshold=cam["invalid_threshold"],
    )
    tifffile.imwrite(str(out_dir / f"{source_id}.tiff"), xyz)


def run(cfg: dict, source_ids: Iterable[str] | None = None) -> None:
    out_dir = Path(cfg["cache_dir"]) / "xyz"
    out_dir.mkdir(parents=True, exist_ok=True)
    if source_ids is None:
        source_ids = list(common.iter_source_ids(cfg))
    else:
        source_ids = list(source_ids)
    for sid in source_ids:
        _per_image(cfg, sid, out_dir)
    print(f"[unproject_xyz] wrote {len(source_ids)} files → {out_dir}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    _main()
```

### Step 4.4: Run tests — expect pass

- [ ] Run:

```
pytest tests/dataset_3d/test_unproject.py -v
```

- Expected: 4 passed.

### Step 4.5: Smoke-run on full dataset

- [ ] Run:

```
python -m scripts.dataset_3d.unproject_xyz
```

- Expected: prints "wrote 549 files → …/.cache/xyz".

---

## Task 5: Build MVTec3D_Weld

**Files:**
- Create: `scripts/dataset_3d/build_mvtec3d.py`
- Create: `tests/dataset_3d/test_build_mvtec3d.py`

**Behavior (spec §4.1):**
- Materialize `Dataset_3D/MVTec3D_Weld/weld/{train,validation,test}/`:
  - `train/good/{rgb,xyz,gt}/` — RGB = Phong render of the source the patch came from, cropped to the patch window; XYZ = same crop of `.cache/xyz/<src>.tiff`; gt = all-zero mask. Source for these is the cached `train` split of good patches.
  - `validation/good/{rgb,xyz,gt}/` — same from `val` split.
  - `test/good/{rgb,xyz,gt}/` — same from `test_good` split.
  - `test/<class>/{rgb,xyz,gt,gt_geom}/` for each single-class anomalous source.
  - `test/combined/{rgb,xyz,gt,gt_geom}/` for every multi-class anomalous source.
- IDs are zero-padded sequential **per subfolder** (`000001.png`, `000002.png`, …).
- `gt` = SAM2 union mask. `gt_geom` = depth-threshold union mask.
- RGB/XYZ for test subfolders is the **full source image** (no cropping) — only train/val/test_good are 256×256 patches.

### Step 5.1: Write failing tests

- [ ] Create `tests/dataset_3d/test_build_mvtec3d.py`:

```python
"""Tests for build_mvtec3d.

End-to-end on a tiny subset only: 4 source IDs and a synthetic stub of
.cache so the test runs in ~1 s without invoking SAM2.
"""
from pathlib import Path
import numpy as np
import pytest
import tifffile
from PIL import Image

from scripts.dataset_3d import common, build_mvtec3d


def _seed_cache(cfg, ids):
    """Populate .cache/ with placeholder masks + xyz + good patches for `ids`."""
    cache = Path(cfg["cache_dir"])
    (cache / "masks" / "sam2").mkdir(parents=True, exist_ok=True)
    (cache / "masks" / "depth").mkdir(parents=True, exist_ok=True)
    (cache / "xyz").mkdir(parents=True, exist_ok=True)
    (cache / "good_patches" / "train").mkdir(parents=True, exist_ok=True)
    (cache / "good_patches" / "val").mkdir(parents=True, exist_ok=True)
    (cache / "good_patches" / "test_good").mkdir(parents=True, exist_ok=True)
    for sid in ids:
        depth = common.read_depth_tif(cfg, sid)
        h, w = depth.shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        # mark a single pixel so the "≥1 nonzero" invariant holds for tests
        m[h // 2, w // 2] = 255
        Image.fromarray(m).save(cache / "masks" / "sam2" / f"{sid}_full.png")
        Image.fromarray(m).save(cache / "masks" / "depth" / f"{sid}_full.png")
        # per-class mask for whichever classes appear
        for b in common.read_labels(cfg, sid):
            Image.fromarray(m).save(cache / "masks" / "sam2" / f"{sid}_cls{b.cls}.png")
            Image.fromarray(m).save(cache / "masks" / "depth" / f"{sid}_cls{b.cls}.png")
        # xyz
        xyz = np.zeros((h, w, 3), dtype=np.float32)
        tifffile.imwrite(cache / "xyz" / f"{sid}.tiff", xyz)
        # one fake good patch from this source in the train split
        patch = np.ones((256, 256), dtype=np.float32)
        tifffile.imwrite(cache / "good_patches" / "train" / f"{sid}_0_0.tif", patch)


def test_build_creates_expected_skeleton(tmp_path, config_path):
    cfg = common.load_config(config_path)
    cfg["cache_dir"] = str(tmp_path / "cache")
    cfg["output_root"] = str(tmp_path / "out")
    ids = ["1", "2", "3", "4"]  # all 4 are single-class anomalous
    _seed_cache(cfg, ids)
    build_mvtec3d.run(cfg, source_ids=ids)

    root = Path(cfg["output_root"]) / "MVTec3D_Weld" / "weld"
    assert (root / "train" / "good" / "rgb").is_dir()
    assert (root / "train" / "good" / "xyz").is_dir()
    assert (root / "train" / "good" / "gt").is_dir()
    assert (root / "test" / "good").is_dir()
    # at least one test/<class> subfolder must exist
    test_subdirs = [p for p in (root / "test").iterdir() if p.is_dir()]
    assert any(p.name != "good" for p in test_subdirs)
    # zero-padded ids
    rgbs = list((root / "train" / "good" / "rgb").glob("*.png"))
    assert rgbs
    assert rgbs[0].stem.isdigit() and len(rgbs[0].stem) == 6
    # gt for train is all-zero
    gt = np.array(Image.open(rgbs[0].with_suffix(".png").parent.parent / "gt" / rgbs[0].name))
    # ... actually walk back to find the matching gt
    gt_path = root / "train" / "good" / "gt" / rgbs[0].name
    gt = np.array(Image.open(gt_path))
    assert (gt == 0).all()
```

### Step 5.2: Run tests — expect failure

- [ ] Run:

```
pytest tests/dataset_3d/test_build_mvtec3d.py -v
```

- Expected: 1 fail (module missing).

### Step 5.3: Implement `scripts/dataset_3d/build_mvtec3d.py`

- [ ] Create `scripts/dataset_3d/build_mvtec3d.py`:

```python
"""Step 5 of the pipeline: materialize Dataset_3D/MVTec3D_Weld/.

Consumes .cache/good_patches/, .cache/masks/sam2/, .cache/masks/depth/,
.cache/xyz/, and Train_Data/Phong/ to produce an MVTec 3D-AD–shaped tree.

Usage:
    python -m scripts.dataset_3d.build_mvtec3d
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tifffile
from PIL import Image

from scripts.dataset_3d import common

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cfg" / "dataset_3d.yaml"


def _zero_pad(i: int) -> str:
    return f"{i:06d}"


def _ensure_dirs(root: Path, layers: Iterable[str]) -> None:
    for layer in layers:
        (root / layer).mkdir(parents=True, exist_ok=True)


def _save_patch_triplet(
    cfg: dict, patch_tif: Path, out_root: Path, idx: int
) -> None:
    """Crop the source Phong + xyz to the patch window, save with gt=0."""
    stem = patch_tif.stem  # "<src>_<u>_<v>"
    src_id, u_str, v_str = stem.rsplit("_", 2)
    u, v = int(u_str), int(v_str)
    patch_size = cfg["good_harvest"]["patch_size"]

    phong = np.array(Image.open(common.resolve_modality_image(cfg, "phong", src_id)).convert("RGB"))
    rgb_crop = phong[v : v + patch_size, u : u + patch_size]
    Image.fromarray(rgb_crop).save(out_root / "rgb" / f"{_zero_pad(idx)}.png")

    xyz_full = tifffile.imread(str(Path(cfg["cache_dir"]) / "xyz" / f"{src_id}.tiff"))
    xyz_crop = xyz_full[v : v + patch_size, u : u + patch_size]
    tifffile.imwrite(out_root / "xyz" / f"{_zero_pad(idx)}.tiff", xyz_crop)

    gt = np.zeros((patch_size, patch_size), dtype=np.uint8)
    Image.fromarray(gt).save(out_root / "gt" / f"{_zero_pad(idx)}.png")


def _save_test_full(
    cfg: dict, src_id: str, out_root: Path, idx: int, write_geom: bool = True
) -> None:
    """Copy full-size Phong + xyz + gt + gt_geom for one anomalous source."""
    phong = common.resolve_modality_image(cfg, "phong", src_id)
    shutil.copyfile(phong, out_root / "rgb" / f"{_zero_pad(idx)}.png")

    xyz_src = Path(cfg["cache_dir"]) / "xyz" / f"{src_id}.tiff"
    shutil.copyfile(xyz_src, out_root / "xyz" / f"{_zero_pad(idx)}.tiff")

    sam_full = Path(cfg["cache_dir"]) / "masks" / "sam2" / f"{src_id}_full.png"
    shutil.copyfile(sam_full, out_root / "gt" / f"{_zero_pad(idx)}.png")

    if write_geom:
        depth_full = Path(cfg["cache_dir"]) / "masks" / "depth" / f"{src_id}_full.png"
        shutil.copyfile(depth_full, out_root / "gt_geom" / f"{_zero_pad(idx)}.png")


def _save_test_good(cfg: dict, patch_tif: Path, out_root: Path, idx: int) -> None:
    """Held-out good test patches: identical to train/good triplet writer."""
    _save_patch_triplet(cfg, patch_tif, out_root, idx)


def _classify_source(cfg: dict, src_id: str) -> str | None:
    """Return one of class names from cfg (single-class images) or 'combined'
    (multi-class) or None (native good)."""
    boxes = common.read_labels(cfg, src_id)
    if not boxes:
        return None
    classes = {b.cls for b in boxes}
    if len(classes) == 1:
        return cfg["classes"][next(iter(classes))]
    return "combined"


def run(cfg: dict, source_ids: Iterable[str] | None = None) -> None:
    out_root = Path(cfg["output_root"]) / "MVTec3D_Weld" / "weld"
    if source_ids is None:
        source_ids = list(common.iter_source_ids(cfg))
    else:
        source_ids = list(source_ids)

    # train/val/test_good split-good triplets
    for split_dir, split_name in (
        (out_root / "train" / "good", "train"),
        (out_root / "validation" / "good", "val"),
        (out_root / "test" / "good", "test_good"),
    ):
        _ensure_dirs(split_dir, ("rgb", "xyz", "gt"))
        src_split = Path(cfg["cache_dir"]) / "good_patches" / split_name
        patches = sorted(src_split.glob("*.tif"))
        for i, p in enumerate(patches, start=1):
            _save_patch_triplet(cfg, p, split_dir, i)
        print(f"[build_mvtec3d] {split_name}: {len(patches)} samples")

    # test/<class>/ and test/combined/
    by_class: dict[str, List[str]] = {}
    for sid in source_ids:
        bucket = _classify_source(cfg, sid)
        if bucket is None:
            continue
        by_class.setdefault(bucket, []).append(sid)

    for bucket, ids in by_class.items():
        sub = out_root / "test" / bucket
        _ensure_dirs(sub, ("rgb", "xyz", "gt", "gt_geom"))
        for i, sid in enumerate(sorted(ids, key=int), start=1):
            _save_test_full(cfg, sid, sub, i, write_geom=True)
        print(f"[build_mvtec3d] test/{bucket}: {len(ids)} samples")

    print(f"[build_mvtec3d] done → {out_root}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    _main()
```

### Step 5.4: Run tests — expect pass

- [ ] Run:

```
pytest tests/dataset_3d/test_build_mvtec3d.py -v
```

- Expected: 1 passed.

### Step 5.5: Defer full smoke-run to Task 10

- The full build depends on SAM2 masks, which are deferred. Do not run on the full dataset in this task — Task 10 covers that.

---

## Task 6: Build Eyecandies_Weld

**Files:**
- Create: `scripts/dataset_3d/build_eyecandies.py`
- Create: `tests/dataset_3d/test_build_eyecandies.py`

**Behavior (spec §4.2):**
- Same skeleton as MVTec3D_Weld except each sample is replicated across 6 modality folders: `lut, phong, diffuse, specular, normal, depth, gt` (and `gt_geom` for test/non-good).
- `lut/phong/diffuse/specular/normal/` are PNG (.png copied from `Train_Data/<Modality>/{train,val,test}/images/<id>.png`).
- `depth/` is single-channel float32 .tiff (raw depth from `Crop_Data/Tif/`, NOT XYZ).
- For train/val/test_good patches: crop each modality the same way.
- For test/<class> and test/combined: full-size copies.
- Skip a modality folder for a sample if the source file is missing (log a warning).

### Step 6.1: Write failing tests

- [ ] Create `tests/dataset_3d/test_build_eyecandies.py`:

```python
"""Tests for build_eyecandies — modality propagation."""
from pathlib import Path
import numpy as np
import pytest
import tifffile
from PIL import Image

from scripts.dataset_3d import common, build_eyecandies


def _seed_eyecandies_cache(cfg, ids):
    cache = Path(cfg["cache_dir"])
    (cache / "masks" / "sam2").mkdir(parents=True, exist_ok=True)
    (cache / "masks" / "depth").mkdir(parents=True, exist_ok=True)
    (cache / "good_patches" / "train").mkdir(parents=True, exist_ok=True)
    (cache / "good_patches" / "val").mkdir(parents=True, exist_ok=True)
    (cache / "good_patches" / "test_good").mkdir(parents=True, exist_ok=True)
    for sid in ids:
        depth = common.read_depth_tif(cfg, sid)
        h, w = depth.shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        m[h // 2, w // 2] = 255
        Image.fromarray(m).save(cache / "masks" / "sam2" / f"{sid}_full.png")
        Image.fromarray(m).save(cache / "masks" / "depth" / f"{sid}_full.png")
        for b in common.read_labels(cfg, sid):
            Image.fromarray(m).save(cache / "masks" / "sam2" / f"{sid}_cls{b.cls}.png")
            Image.fromarray(m).save(cache / "masks" / "depth" / f"{sid}_cls{b.cls}.png")
        patch = np.ones((256, 256), dtype=np.float32)
        tifffile.imwrite(cache / "good_patches" / "train" / f"{sid}_0_0.tif", patch)


def test_build_creates_six_modality_folders(tmp_path, config_path):
    cfg = common.load_config(config_path)
    cfg["cache_dir"] = str(tmp_path / "cache")
    cfg["output_root"] = str(tmp_path / "out")
    ids = ["1", "2", "3", "4"]
    _seed_eyecandies_cache(cfg, ids)
    build_eyecandies.run(cfg, source_ids=ids)

    root = Path(cfg["output_root"]) / "Eyecandies_Weld" / "weld"
    for mod in ("lut", "phong", "diffuse", "specular", "normal", "depth", "gt"):
        assert (root / "train" / "good" / mod).is_dir(), f"missing train/good/{mod}"
    # test subfolders should additionally have gt_geom
    test_subdirs = [p for p in (root / "test").iterdir() if p.is_dir() and p.name != "good"]
    assert test_subdirs, "expected at least one test/<class> subfolder"
    sample_sub = test_subdirs[0]
    assert (sample_sub / "gt_geom").is_dir()
    # depth must be a .tiff
    depth_files = list((sample_sub / "depth").glob("*.tiff"))
    assert depth_files
    arr = tifffile.imread(str(depth_files[0]))
    assert arr.dtype == np.float32
```

### Step 6.2: Run tests — expect failure

- [ ] Run:

```
pytest tests/dataset_3d/test_build_eyecandies.py -v
```

- Expected: 1 fail (module missing).

### Step 6.3: Implement `scripts/dataset_3d/build_eyecandies.py`

- [ ] Create `scripts/dataset_3d/build_eyecandies.py`:

```python
"""Step 6 of the pipeline: materialize Dataset_3D/Eyecandies_Weld/.

Same skeleton as MVTec3D_Weld, but each sample is duplicated across the
six modality folders (lut, phong, diffuse, specular, normal, depth) plus
gt (and gt_geom for test/non-good). 'depth' is the raw .tif (single-
channel float32), NOT the XYZ point cloud.

Usage:
    python -m scripts.dataset_3d.build_eyecandies
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tifffile
from PIL import Image

from scripts.dataset_3d import common

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cfg" / "dataset_3d.yaml"

PNG_MODALITIES = ("lut", "phong", "diffuse", "specular", "normal")


def _zero_pad(i: int) -> str:
    return f"{i:06d}"


def _ensure_dirs(root: Path, layers: Iterable[str]) -> None:
    for layer in layers:
        (root / layer).mkdir(parents=True, exist_ok=True)


def _save_patch_eyecandies(
    cfg: dict, patch_tif: Path, out_root: Path, idx: int
) -> None:
    stem = patch_tif.stem
    src_id, u_str, v_str = stem.rsplit("_", 2)
    u, v = int(u_str), int(v_str)
    ps = cfg["good_harvest"]["patch_size"]

    for mod in PNG_MODALITIES:
        try:
            img_path = common.resolve_modality_image(cfg, mod, src_id)
        except FileNotFoundError:
            print(f"[build_eyecandies] WARN: {mod} missing for {src_id}")
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        crop = img[v : v + ps, u : u + ps]
        Image.fromarray(crop).save(out_root / mod / f"{_zero_pad(idx)}.png")

    # depth: crop the raw source .tif (not the XYZ tiff)
    depth = common.read_depth_tif(cfg, src_id)
    depth_crop = depth[v : v + ps, u : u + ps]
    tifffile.imwrite(out_root / "depth" / f"{_zero_pad(idx)}.tiff", depth_crop)

    gt = np.zeros((ps, ps), dtype=np.uint8)
    Image.fromarray(gt).save(out_root / "gt" / f"{_zero_pad(idx)}.png")


def _save_test_full_eyecandies(
    cfg: dict, src_id: str, out_root: Path, idx: int
) -> None:
    for mod in PNG_MODALITIES:
        try:
            img_path = common.resolve_modality_image(cfg, mod, src_id)
        except FileNotFoundError:
            print(f"[build_eyecandies] WARN: {mod} missing for {src_id}")
            continue
        shutil.copyfile(img_path, out_root / mod / f"{_zero_pad(idx)}.png")

    depth_src = Path(cfg["source"]["tif_dir"]) / f"{src_id}.tif"
    out_depth = out_root / "depth" / f"{_zero_pad(idx)}.tiff"
    arr = tifffile.imread(str(depth_src))
    tifffile.imwrite(str(out_depth), arr.astype(np.float32))

    sam_full = Path(cfg["cache_dir"]) / "masks" / "sam2" / f"{src_id}_full.png"
    shutil.copyfile(sam_full, out_root / "gt" / f"{_zero_pad(idx)}.png")

    depth_full = Path(cfg["cache_dir"]) / "masks" / "depth" / f"{src_id}_full.png"
    shutil.copyfile(depth_full, out_root / "gt_geom" / f"{_zero_pad(idx)}.png")


def _classify_source(cfg: dict, src_id: str) -> str | None:
    boxes = common.read_labels(cfg, src_id)
    if not boxes:
        return None
    classes = {b.cls for b in boxes}
    if len(classes) == 1:
        return cfg["classes"][next(iter(classes))]
    return "combined"


def run(cfg: dict, source_ids: Iterable[str] | None = None) -> None:
    out_root = Path(cfg["output_root"]) / "Eyecandies_Weld" / "weld"
    if source_ids is None:
        source_ids = list(common.iter_source_ids(cfg))
    else:
        source_ids = list(source_ids)

    train_layers = list(PNG_MODALITIES) + ["depth", "gt"]
    for split_dir, split_name in (
        (out_root / "train" / "good", "train"),
        (out_root / "validation" / "good", "val"),
        (out_root / "test" / "good", "test_good"),
    ):
        _ensure_dirs(split_dir, train_layers)
        src_split = Path(cfg["cache_dir"]) / "good_patches" / split_name
        patches = sorted(src_split.glob("*.tif"))
        for i, p in enumerate(patches, start=1):
            _save_patch_eyecandies(cfg, p, split_dir, i)
        print(f"[build_eyecandies] {split_name}: {len(patches)} samples")

    by_class: dict[str, List[str]] = {}
    for sid in source_ids:
        bucket = _classify_source(cfg, sid)
        if bucket is None:
            continue
        by_class.setdefault(bucket, []).append(sid)

    test_layers = list(PNG_MODALITIES) + ["depth", "gt", "gt_geom"]
    for bucket, ids in by_class.items():
        sub = out_root / "test" / bucket
        _ensure_dirs(sub, test_layers)
        for i, sid in enumerate(sorted(ids, key=int), start=1):
            _save_test_full_eyecandies(cfg, sid, sub, i)
        print(f"[build_eyecandies] test/{bucket}: {len(ids)} samples")

    print(f"[build_eyecandies] done → {out_root}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    _main()
```

### Step 6.4: Run tests — expect pass

- [ ] Run:

```
pytest tests/dataset_3d/test_build_eyecandies.py -v
```

- Expected: 1 passed.

---

## Task 7: Build Real3D_Weld

**Files:**
- Create: `scripts/dataset_3d/build_real3d.py`
- Create: `tests/dataset_3d/test_build_real3d.py`

**Behavior (spec §4.3):**
- `train/` — 4 prototype PCDs from the first 4 good-patch files (sorted, then deterministic seed-42 shuffled order) of `.cache/good_patches/train/`. Each saved as ASCII PCD at `train/tmpl_{0..3}.pcd`. The XYZ for these prototype patches is computed by unprojecting their 256×256 patch using the same `pixel_size_mm_*` config (origin at patch center).
- `test/` — for each of the 541 anomalous source images, write an ASCII PCD with valid points only (invalid pixels dropped, NOT zeroed). Names: `0001.pcd, …, 0541.pcd`.
- `ground_truth/<id>.txt` — one binary label per line, in the same order as points in the corresponding `.pcd`. Label = 1 iff that pixel was inside *any* defect bbox **AND** inside that defect's SAM2 mask, else 0.
- ASCII PCD header is exact (see code).

### Step 7.1: Write failing tests

- [ ] Create `tests/dataset_3d/test_build_real3d.py`:

```python
"""Tests for build_real3d."""
from pathlib import Path
import numpy as np
import pytest
import tifffile
from PIL import Image

from scripts.dataset_3d import common, build_real3d


def test_write_ascii_pcd_roundtrip(tmp_path):
    pts = np.array(
        [[0.0, 0.0, 1.0], [0.016, 0.0, 1.5], [0.0, 0.016, 2.0]], dtype=np.float32
    )
    out = tmp_path / "x.pcd"
    build_real3d.write_ascii_pcd(out, pts)
    text = out.read_text(encoding="ascii").splitlines()
    assert text[0].startswith("# .PCD")
    assert "POINTS 3" in text
    assert text[-3:] == ["0.000000 0.000000 1.000000",
                         "0.016000 0.000000 1.500000",
                         "0.000000 0.016000 2.000000"]


def _seed_real3d_cache(cfg, anom_ids):
    cache = Path(cfg["cache_dir"])
    (cache / "masks" / "sam2").mkdir(parents=True, exist_ok=True)
    (cache / "good_patches" / "train").mkdir(parents=True, exist_ok=True)
    for sid in anom_ids:
        depth = common.read_depth_tif(cfg, sid)
        h, w = depth.shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        for b in common.read_labels(cfg, sid):
            x1, y1, x2, y2 = b.to_pixel_xyxy(img_w=w, img_h=h)
            m[y1:y2, x1:x2] = 255
            Image.fromarray(m).save(cache / "masks" / "sam2" / f"{sid}_cls{b.cls}.png")
        Image.fromarray(m).save(cache / "masks" / "sam2" / f"{sid}_full.png")
    # 5 good train patches so templates can be selected
    for i in range(5):
        patch = np.ones((256, 256), dtype=np.float32)
        tifffile.imwrite(cache / "good_patches" / "train" / f"src_{i}_0.tif", patch)


def test_build_writes_templates_and_test_pcds(tmp_path, config_path):
    cfg = common.load_config(config_path)
    cfg["cache_dir"] = str(tmp_path / "cache")
    cfg["output_root"] = str(tmp_path / "out")
    ids = ["1", "2"]
    _seed_real3d_cache(cfg, ids)
    build_real3d.run(cfg, source_ids=ids)
    root = Path(cfg["output_root"]) / "Real3D_Weld" / "weld"
    assert (root / "train" / "tmpl_0.pcd").exists()
    assert (root / "train" / "tmpl_3.pcd").exists()
    assert (root / "test" / "0001.pcd").exists()
    assert (root / "test" / "0002.pcd").exists()
    # per-point labels line count matches POINTS in matching pcd
    pcd_text = (root / "test" / "0001.pcd").read_text(encoding="ascii").splitlines()
    n_pts = int([l for l in pcd_text if l.startswith("POINTS ")][0].split()[1])
    labels = (root / "ground_truth" / "0001.txt").read_text(encoding="ascii").strip().splitlines()
    assert len(labels) == n_pts
    assert all(l in ("0", "1") for l in labels)
```

### Step 7.2: Run tests — expect failure

- [ ] Run:

```
pytest tests/dataset_3d/test_build_real3d.py -v
```

- Expected: 2 fails (module missing).

### Step 7.3: Implement `scripts/dataset_3d/build_real3d.py`

- [ ] Create `scripts/dataset_3d/build_real3d.py`:

```python
"""Step 7 of the pipeline: materialize Dataset_3D/Real3D_Weld/.

Templates = 4 prototype good patches (first-after-shuffle of the train
good-patch pool). Test = per-source point clouds with per-point binary
labels derived from SAM2 masks ∩ defect bboxes.

Usage:
    python -m scripts.dataset_3d.build_real3d
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tifffile
from PIL import Image

from scripts.dataset_3d import common

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cfg" / "dataset_3d.yaml"


def write_ascii_pcd(path: Path, points: np.ndarray) -> None:
    """Write an ASCII PCD v0.7 file with x,y,z fields.

    points: shape (N, 3) float32 or float64.
    """
    assert points.ndim == 2 and points.shape[1] == 3
    n = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(header)
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _patch_to_pcd(cfg: dict, patch_tif: Path, out_path: Path) -> None:
    """Unproject a 256x256 good patch to an ASCII PCD."""
    depth = tifffile.imread(str(patch_tif)).astype(np.float32)
    cam = cfg["camera"]
    ps = depth.shape[0]
    u = np.arange(ps, dtype=np.float32)[None, :]
    v = np.arange(ps, dtype=np.float32)[:, None]
    x = (u - ps / 2.0) * cam["pixel_size_mm_x"]
    y = (v - ps / 2.0) * cam["pixel_size_mm_y"]
    z = depth * cam["depth_scale_mm"]
    valid = depth > cam["invalid_threshold"]
    pts = np.stack([np.broadcast_to(x, depth.shape)[valid],
                    np.broadcast_to(y, depth.shape)[valid],
                    z[valid]], axis=-1).astype(np.float32)
    write_ascii_pcd(out_path, pts)


def _source_to_pcd_and_labels(
    cfg: dict, src_id: str, pcd_path: Path, label_path: Path
) -> None:
    depth = common.read_depth_tif(cfg, src_id)
    h, w = depth.shape[:2]
    cam = cfg["camera"]
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    x = (u - w / 2.0) * cam["pixel_size_mm_x"]
    y = (v - h / 2.0) * cam["pixel_size_mm_y"]
    z = depth * cam["depth_scale_mm"]

    valid = depth > cam["invalid_threshold"]
    xs = np.broadcast_to(x, depth.shape)[valid]
    ys = np.broadcast_to(y, depth.shape)[valid]
    zs = z[valid]
    pts = np.stack([xs, ys, zs], axis=-1).astype(np.float32)

    # Per-point labels: 1 iff inside any defect bbox AND inside SAM2 mask for
    # that defect's class.
    sam_mask = np.zeros((h, w), dtype=np.uint8)  # accumulated full-image mask
    boxes = common.read_labels(cfg, src_id)
    for b in boxes:
        sam_cls_path = (
            Path(cfg["cache_dir"]) / "masks" / "sam2" / f"{src_id}_cls{b.cls}.png"
        )
        if not sam_cls_path.exists():
            continue
        m = np.array(Image.open(sam_cls_path))
        x1, y1, x2, y2 = b.to_pixel_xyxy(img_w=w, img_h=h)
        # restrict to this bbox AND this class's SAM2 mask
        bbox_region = np.zeros_like(m, dtype=bool)
        bbox_region[y1:y2, x1:x2] = True
        sam_mask = np.maximum(sam_mask, ((m > 0) & bbox_region).astype(np.uint8) * 255)

    labels_full = (sam_mask > 0).astype(np.uint8)
    labels = labels_full[valid]

    write_ascii_pcd(pcd_path, pts)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="ascii", newline="\n") as f:
        for lbl in labels.tolist():
            f.write(f"{lbl}\n")


def _select_template_patches(cfg: dict) -> List[Path]:
    """Reproduce the harvest_good shuffle: take first n_templates patches
    after a deterministic shuffle of the sorted train-good filenames.
    """
    train_dir = Path(cfg["cache_dir"]) / "good_patches" / "train"
    candidates = sorted(train_dir.glob("*.tif"))
    rng = random.Random(cfg["good_harvest"]["shuffle_seed"])
    order = list(range(len(candidates)))
    rng.shuffle(order)
    n = cfg["real3d"]["n_templates"]
    return [candidates[i] for i in order[:n]]


def run(cfg: dict, source_ids: Iterable[str] | None = None) -> None:
    out_root = Path(cfg["output_root"]) / "Real3D_Weld" / "weld"
    train_dir = out_root / "train"
    test_dir = out_root / "test"
    gt_dir = out_root / "ground_truth"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Templates
    templates = _select_template_patches(cfg)
    for i, p in enumerate(templates):
        _patch_to_pcd(cfg, p, train_dir / f"tmpl_{i}.pcd")
    print(f"[build_real3d] wrote {len(templates)} templates → {train_dir}")

    # Test PCDs + labels (only anomalous sources)
    if source_ids is None:
        source_ids = list(common.iter_source_ids(cfg))
    else:
        source_ids = list(source_ids)
    anomalous = [sid for sid in source_ids if common.read_labels(cfg, sid)]
    anomalous = sorted(anomalous, key=int)
    for i, sid in enumerate(anomalous, start=1):
        name = f"{i:04d}"
        _source_to_pcd_and_labels(
            cfg, sid, test_dir / f"{name}.pcd", gt_dir / f"{name}.txt"
        )
    print(f"[build_real3d] wrote {len(anomalous)} test point clouds → {test_dir}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    _main()
```

### Step 7.4: Run tests — expect pass

- [ ] Run:

```
pytest tests/dataset_3d/test_build_real3d.py -v
```

- Expected: 2 passed.

---

## Task 8: Verification script

**Files:**
- Create: `scripts/dataset_3d/verify_dataset.py`
- Create: `tests/dataset_3d/test_verify_dataset.py`

**Behavior (spec §7):**
- For each of MVTec3D_Weld and Eyecandies_Weld:
  - Every `rgb`/`xyz`/`<modality>` subfolder has exactly one file per id.
  - Every `gt/<id>.png` has the same H×W as its companion RGB/xyz.
  - Every test/non-good `gt/<id>.png` has ≥1 white pixel.
  - Every train/good and test/good `gt/<id>.png` is all-zero.
  - XYZ tiffs are float32, 3-channel, no NaN.
- For Real3D_Weld:
  - Every `test/<id>.pcd` has a matching `ground_truth/<id>.txt` with line count == POINTS field.
- Reports per-class test counts at the end. Returns non-zero exit code on first invariant violation.

### Step 8.1: Write failing tests

- [ ] Create `tests/dataset_3d/test_verify_dataset.py`:

```python
"""Tests for verify_dataset CLI."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from PIL import Image

from scripts.dataset_3d import verify_dataset


def _make_good_triplet(root, idx):
    for layer in ("rgb", "gt"):
        (root / layer).mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(
        root / "rgb" / f"{idx:06d}.png"
    )
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(
        root / "gt" / f"{idx:06d}.png"
    )


def test_count_consistency_passes_when_aligned(tmp_path):
    sub = tmp_path / "train" / "good"
    for i in range(1, 4):
        _make_good_triplet(sub, i)
    # We exercise the lower-level check directly.
    issues = verify_dataset.check_layer_count_alignment(sub, layers=("rgb", "gt"))
    assert issues == []


def test_count_consistency_fails_on_mismatch(tmp_path):
    sub = tmp_path / "train" / "good"
    (sub / "rgb").mkdir(parents=True)
    (sub / "gt").mkdir(parents=True)
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(sub / "rgb" / "000001.png")
    # No gt file for 000001 → mismatch
    issues = verify_dataset.check_layer_count_alignment(sub, layers=("rgb", "gt"))
    assert issues


def test_anomalous_gt_must_have_white_pixel(tmp_path):
    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    # All-zero mask in a non-good subfolder → violation
    Image.fromarray(np.zeros((10, 10), dtype=np.uint8)).save(gt_dir / "000001.png")
    issues = verify_dataset.check_anomalous_gt_nonzero(gt_dir.parent)
    assert issues
    # Mask with one white pixel → no violation
    m = np.zeros((10, 10), dtype=np.uint8)
    m[0, 0] = 255
    Image.fromarray(m).save(gt_dir / "000001.png")
    issues = verify_dataset.check_anomalous_gt_nonzero(gt_dir.parent)
    assert not issues
```

### Step 8.2: Run tests — expect failure

- [ ] Run:

```
pytest tests/dataset_3d/test_verify_dataset.py -v
```

- Expected: 3 fails (module missing).

### Step 8.3: Implement `scripts/dataset_3d/verify_dataset.py`

- [ ] Create `scripts/dataset_3d/verify_dataset.py`:

```python
"""Step 8 of the pipeline: verify invariants on the three output datasets.

Exits 0 if every invariant holds, 1 otherwise. Prints per-class test
counts for the dataset card.

Usage:
    python -m scripts.dataset_3d.verify_dataset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import tifffile
from PIL import Image

from scripts.dataset_3d import common

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "cfg" / "dataset_3d.yaml"


def _file_stems(d: Path, suffixes: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for s in suffixes:
        out.update(p.stem for p in d.glob(f"*{s}"))
    return out


def check_layer_count_alignment(sub: Path, layers: Iterable[str]) -> List[str]:
    """Each layer subdirectory of `sub` should have the same set of ids."""
    issues: List[str] = []
    stems_by_layer: dict[str, set[str]] = {}
    for layer in layers:
        d = sub / layer
        if not d.is_dir():
            issues.append(f"missing layer dir: {d}")
            continue
        stems_by_layer[layer] = _file_stems(d, (".png", ".tif", ".tiff"))
    if not stems_by_layer:
        return issues
    ref = next(iter(stems_by_layer.values()))
    for layer, stems in stems_by_layer.items():
        missing = ref - stems
        extra = stems - ref
        if missing:
            issues.append(f"{sub}/{layer} missing ids: {sorted(missing)[:5]}")
        if extra:
            issues.append(f"{sub}/{layer} extra ids: {sorted(extra)[:5]}")
    return issues


def check_anomalous_gt_nonzero(sub: Path) -> List[str]:
    """Every gt/<id>.png in a non-good subfolder must have ≥1 nonzero pixel."""
    issues: List[str] = []
    gt_dir = sub / "gt"
    if not gt_dir.is_dir():
        return issues
    for p in sorted(gt_dir.glob("*.png")):
        arr = np.array(Image.open(p))
        if (arr > 0).sum() == 0:
            issues.append(f"{p} has no nonzero pixel (anomalous folder)")
    return issues


def check_good_gt_zero(sub: Path) -> List[str]:
    issues: List[str] = []
    gt_dir = sub / "gt"
    if not gt_dir.is_dir():
        return issues
    for p in sorted(gt_dir.glob("*.png")):
        arr = np.array(Image.open(p))
        if (arr > 0).any():
            issues.append(f"{p} has nonzero pixel (good folder)")
    return issues


def check_xyz_validity(sub: Path) -> List[str]:
    issues: List[str] = []
    xyz_dir = sub / "xyz"
    if not xyz_dir.is_dir():
        return issues
    for p in sorted(xyz_dir.glob("*.tiff")):
        arr = tifffile.imread(str(p))
        if arr.dtype != np.float32:
            issues.append(f"{p} dtype {arr.dtype} != float32")
        if arr.ndim != 3 or arr.shape[-1] != 3:
            issues.append(f"{p} shape {arr.shape} != (H, W, 3)")
        if np.isnan(arr).any():
            issues.append(f"{p} contains NaN")
    return issues


def check_real3d(out_root: Path) -> List[str]:
    issues: List[str] = []
    root = out_root / "Real3D_Weld" / "weld"
    test_dir = root / "test"
    gt_dir = root / "ground_truth"
    if not test_dir.is_dir() or not gt_dir.is_dir():
        issues.append(f"Real3D_Weld layout incomplete under {root}")
        return issues
    for pcd in sorted(test_dir.glob("*.pcd")):
        gt = gt_dir / f"{pcd.stem}.txt"
        if not gt.exists():
            issues.append(f"missing {gt}")
            continue
        text = pcd.read_text(encoding="ascii").splitlines()
        pts_line = [l for l in text if l.startswith("POINTS ")]
        if not pts_line:
            issues.append(f"{pcd} missing POINTS header")
            continue
        n_pts = int(pts_line[0].split()[1])
        n_labels = sum(1 for l in gt.read_text(encoding="ascii").splitlines() if l.strip())
        if n_pts != n_labels:
            issues.append(f"{pcd} POINTS={n_pts} but {gt} has {n_labels} labels")
    return issues


def _per_class_counts(out_root: Path, dataset_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    root = out_root / dataset_name / "weld" / "test"
    if not root.is_dir():
        return counts
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and sub.name != "good":
            counts[sub.name] = len(list((sub / "rgb").glob("*.png")) or list((sub / "phong").glob("*.png")))
    return counts


def verify_mvtec_style(out_root: Path, name: str, layers_train: tuple, layers_test: tuple) -> List[str]:
    issues: List[str] = []
    weld = out_root / name / "weld"
    if not weld.is_dir():
        issues.append(f"{weld} missing")
        return issues
    for split in ("train", "validation"):
        good = weld / split / "good"
        if good.is_dir():
            issues += check_layer_count_alignment(good, layers_train)
            issues += check_good_gt_zero(good)
            if "xyz" in layers_train:
                issues += check_xyz_validity(good)
    test_good = weld / "test" / "good"
    if test_good.is_dir():
        issues += check_layer_count_alignment(test_good, layers_train)
        issues += check_good_gt_zero(test_good)
        if "xyz" in layers_train:
            issues += check_xyz_validity(test_good)
    for sub in (weld / "test").iterdir():
        if not sub.is_dir() or sub.name == "good":
            continue
        issues += check_layer_count_alignment(sub, layers_test)
        issues += check_anomalous_gt_nonzero(sub)
        if "xyz" in layers_test:
            issues += check_xyz_validity(sub)
    return issues


def run(cfg: dict) -> int:
    out_root = Path(cfg["output_root"])
    issues: List[str] = []

    issues += verify_mvtec_style(
        out_root, "MVTec3D_Weld",
        layers_train=("rgb", "xyz", "gt"),
        layers_test=("rgb", "xyz", "gt", "gt_geom"),
    )
    issues += verify_mvtec_style(
        out_root, "Eyecandies_Weld",
        layers_train=("lut", "phong", "diffuse", "specular", "normal", "depth", "gt"),
        layers_test=("lut", "phong", "diffuse", "specular", "normal", "depth", "gt", "gt_geom"),
    )
    issues += check_real3d(out_root)

    if issues:
        print("[verify_dataset] FAILED")
        for x in issues[:50]:
            print(f"  - {x}")
        if len(issues) > 50:
            print(f"  ... and {len(issues) - 50} more")
        return 1

    print("[verify_dataset] OK")
    for name in ("MVTec3D_Weld", "Eyecandies_Weld"):
        counts = _per_class_counts(out_root, name)
        print(f"  {name} per-class test counts: {counts}")
    return 0


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    sys.exit(run(cfg))


if __name__ == "__main__":
    _main()
```

### Step 8.4: Run tests — expect pass

- [ ] Run:

```
pytest tests/dataset_3d/test_verify_dataset.py -v
```

- Expected: 3 passed.

---

## Task 9: Documentation

**Files:**
- Create: `docs/dataset_3d.md`
- Create: `docs/dataset_3d.zh-CN.md`

### Step 9.1: Write `docs/dataset_3d.md`

- [ ] Create `docs/dataset_3d.md` with this content:

```markdown
# LUT-AD 3D Anomaly-Detection Variants

This page describes three sibling 3D-AD-style datasets derived from the
LUT-AD source data (`Crop_Data/Tif/*.tif` + `Crop_Data/label/*.txt`):

- `Dataset_3D/MVTec3D_Weld/` — MVTec 3D-AD–style layout (RGB + organized XYZ point cloud + pixel GT).
- `Dataset_3D/Eyecandies_Weld/` — Eyecandies–style multi-modality layout (5 photometric renders + raw depth + GT).
- `Dataset_3D/Real3D_Weld/` — Real3D-AD–style point-cloud layout (4 prototype templates + per-source test PCDs with per-point binary labels).

## Generation

Generation is fully driven by `cfg/dataset_3d.yaml` and the eight scripts
under `scripts/dataset_3d/`:

```bash
python -m scripts.dataset_3d.harvest_good
python -m scripts.dataset_3d.segment_sam2          # ~1h on a 12GB consumer GPU
python -m scripts.dataset_3d.segment_depth_threshold
python -m scripts.dataset_3d.unproject_xyz
python -m scripts.dataset_3d.build_mvtec3d
python -m scripts.dataset_3d.build_eyecandies
python -m scripts.dataset_3d.build_real3d
python -m scripts.dataset_3d.verify_dataset
```

Outputs land under `Dataset_3D/` (gitignored). Intermediate caches live
under `.cache/` (also gitignored).

## Layout — MVTec3D_Weld

```
Dataset_3D/MVTec3D_Weld/weld/
  train/good/{rgb,xyz,gt}/
  validation/good/{rgb,xyz,gt}/
  test/good/{rgb,xyz,gt}/
  test/{pseudo_soldering,pinhole,pit,burst,fish_scale_welding,bump,combined}/{rgb,xyz,gt,gt_geom}/
```

`rgb/` is a Phong render, `xyz/` is the float32 `H×W×3` organized point
cloud (X = `(u - W/2) · 0.016 mm`, Y = `(v - H/2) · 0.016 mm`, Z = raw
depth in mm; invalid pixels = `(0,0,0)`). `gt` = SAM2 mask; `gt_geom` =
depth-threshold baseline mask.

train/val/test_good samples are 256×256 patches harvested from regions
of the source images with zero IoU against any defect bbox (and at most
5% invalid pixels). test/<class> and test/combined samples are full
source images.

## Layout — Eyecandies_Weld

```
Dataset_3D/Eyecandies_Weld/weld/
  train/good/{lut,phong,diffuse,specular,normal,depth,gt}/
  validation/good/  (same)
  test/{good,pseudo_soldering,pinhole,pit,burst,fish_scale_welding,bump,combined}/{lut,phong,diffuse,specular,normal,depth,gt,gt_geom}/
```

PNG modalities mirror existing renders. `depth/` is a single-channel
float32 `.tiff` (raw depth, NOT XYZ). The dataset is "Eyecandies-
inspired" — we ship the five existing photometric renders rather than
re-rendering under six light directions.

## Layout — Real3D_Weld

```
Dataset_3D/Real3D_Weld/weld/
  train/tmpl_{0..3}.pcd
  test/{0001..0541}.pcd
  ground_truth/{0001..0541}.txt
```

Templates are the first 4 entries of the seed-42 shuffled
`.cache/good_patches/train/` pool. Test PCDs include only valid points
(invalid depth pixels are dropped, not zeroed). Per-point labels are 1
iff the underlying pixel is inside any defect bbox AND inside that
defect's SAM2 mask.

**Known limitation:** Real3D-AD baselines assume canonical object
templates from the same instance. Welds are per-instance unique
geometry. Templates here are *prototype patches*, not full-object
instances. Rigid-body-registration baselines will require adaptation.

## Scale factors

| Axis | Value |
|---|---|
| `pixel_size_mm_x` | 0.016 |
| `pixel_size_mm_y` | 0.016 |
| `depth_scale_mm` | 1.0 |
| `invalid_threshold` | -1.0e30 |

## Caveats

- **SAM2 quality:** Spot-checked on 20 random source images. Falls back
  to the bbox rectangle if SAM2 cannot find an interior segment (rare).
  Pair `gt` (SAM2) with `gt_geom` (depth-threshold) when evaluating
  pixel-AUROC sensitivity to mask choice.
- **Class balance:** Inherited from the source set —
  `burst: 100, bump: 97, pseudo_soldering: 212, pinhole: 217, pit: 228, fish_scale_welding: 174`.
- **Combined subfolder:** 271 source images contain ≥2 defect classes
  and live under `test/combined/`. The single-class folders
  (`pseudo_soldering/`, etc.) hold only images whose defect set is a
  single class.

## Reproducibility

Every script is deterministic given `cfg/dataset_3d.yaml`. The single
source of randomness is `good_harvest.shuffle_seed = 42`, which controls
the 70/15/15 split, the per-source random subsample, and the Real3D
template selection.
```

### Step 9.2: Write `docs/dataset_3d.zh-CN.md`

- [ ] Create `docs/dataset_3d.zh-CN.md` with this content:

```markdown
# LUT-AD 3D 异常检测变体

本页描述基于 LUT-AD 原始数据（`Crop_Data/Tif/*.tif` + `Crop_Data/label/*.txt`）
派生的三个 3D 异常检测风格的姊妹数据集：

- `Dataset_3D/MVTec3D_Weld/` — MVTec 3D-AD 风格布局（RGB + 有序 XYZ 点云 + 像素 GT）。
- `Dataset_3D/Eyecandies_Weld/` — Eyecandies 风格多模态布局（5 种光度渲染 + 原始深度 + GT）。
- `Dataset_3D/Real3D_Weld/` — Real3D-AD 风格点云布局（4 个原型模板 + 每个测试源的 PCD 与逐点二值标签）。

## 生成流程

完全由 `cfg/dataset_3d.yaml` 与 `scripts/dataset_3d/` 下的 8 个脚本驱动：

```bash
python -m scripts.dataset_3d.harvest_good
python -m scripts.dataset_3d.segment_sam2           # 12GB 消费级 GPU 上约 1 小时
python -m scripts.dataset_3d.segment_depth_threshold
python -m scripts.dataset_3d.unproject_xyz
python -m scripts.dataset_3d.build_mvtec3d
python -m scripts.dataset_3d.build_eyecandies
python -m scripts.dataset_3d.build_real3d
python -m scripts.dataset_3d.verify_dataset
```

输出位于 `Dataset_3D/`（已 gitignore），中间缓存位于 `.cache/`（同样 gitignore）。

## 坐标尺度

| 轴 | 值 |
|---|---|
| `pixel_size_mm_x` | 0.016 |
| `pixel_size_mm_y` | 0.016 |
| `depth_scale_mm` | 1.0 |
| `invalid_threshold` | -1.0e30 |

## 已知限制

- **SAM2 质量：** 在 20 张随机源图上抽检；找不到内部分割时回退为 bbox 矩形（少见）。
  评估对 mask 选择的敏感度时，请同时报告 `gt`（SAM2）与 `gt_geom`（深度阈值）。
- **类别不均衡：** 继承自原始数据 —
  `burst: 100, bump: 97, pseudo_soldering: 212, pinhole: 217, pit: 228, fish_scale_welding: 174`。
- **combined 子目录：** 271 张源图包含 ≥2 个缺陷类别，统一归入 `test/combined/`。
- **Real3D 模板：** 焊缝几何形态因实例而异，模板是"原型 patch"而非整体对象实例；
  刚体配准 baseline 需要适配。

## 可复现性

每个脚本在 `cfg/dataset_3d.yaml` 给定时均为确定性流程，唯一的随机源是
`good_harvest.shuffle_seed = 42`，它同时控制 70/15/15 划分、每源随机
子采样以及 Real3D 模板选择。
```

### Step 9.3: Update README cross-references (optional but recommended)

- [ ] Skim `README.md` and `README.zh-CN.md`. If they have a "Datasets" or "Variants" section, append a one-line link to `docs/dataset_3d.md` / `docs/dataset_3d.zh-CN.md`. If neither exists in a natural position, skip — the docs are still discoverable by file path.

---

## Task 10: Execute the full pipeline + final verification

**Files:** none (this task only runs the pipeline).

**Note:** This task has a long-running step (~1 h SAM2 pass). Do not
dispatch as a normal subagent task — execute interactively or as a
long-running terminal command after Tasks 0–9 are all green.

### Step 10.1: Confirm prerequisites

- [ ] Confirm `pip show sam2` works (or installation succeeded in Task 3.5).
- [ ] Confirm `nvidia-smi` shows ≥10 GB free GPU memory.

### Step 10.2: Run the pipeline in order

- [ ] Run each step from the repo root. Skip any whose cache directory is already populated unless you pass `--force`:

```
python -m scripts.dataset_3d.harvest_good
python -m scripts.dataset_3d.segment_depth_threshold
python -m scripts.dataset_3d.segment_sam2
python -m scripts.dataset_3d.unproject_xyz
python -m scripts.dataset_3d.build_mvtec3d
python -m scripts.dataset_3d.build_eyecandies
python -m scripts.dataset_3d.build_real3d
python -m scripts.dataset_3d.verify_dataset
```

- Expected for `verify_dataset`: `[verify_dataset] OK` followed by per-class test counts. Non-zero exit ⇒ inspect the printed first 50 issues and fix the corresponding builder before re-running.

### Step 10.3: Spot-check 5 random SAM2 masks

- [ ] Pick 5 random source ids and visually inspect `.cache/masks/sam2/<id>_full.png` against `Train_Data/Phong/*/images/<id>.png`. If quality is unacceptable (most masks empty or fill the entire bbox), document in the dataset card and consider re-running the SAM2 pass with `multimask_output=True` and a heuristic score-based mask pick.

### Step 10.4: Record actual SAM2 runtime in the dataset card

- [ ] Edit the `~1h on a 12GB consumer GPU` line in `docs/dataset_3d.md` and `docs/dataset_3d.zh-CN.md` to reflect the actual measured wall-clock time of the SAM2 pass.

---

## Notes for the implementing engineer

- **Working directory** for all `python -m …` and `pytest …` commands is `F:/dataset/LUT_AD_DataSet/` (the repo root).
- **No git commits.** This repository is not a git repo. Use pytest passes + a clean `verify_dataset` run as the integration checkpoint between tasks.
- **Idempotency.** Every script can be re-run; existing output files are silently overwritten. To force a full rebuild after a config change, delete `.cache/` and re-run from Task 10.2.
- **Windows file copies.** `shutil.copyfile` is used everywhere (not `os.symlink`) because Windows symlink creation requires elevated privileges. The trade-off is ~5× more disk usage; with 549 source images this is < 20 GB and acceptable for the deliverable.
- **SAM2 is the only heavyweight step.** All other tasks complete in seconds to a few minutes. Run those first and confirm the layout/invariants are happy on the depth-threshold masks alone before kicking off the long SAM2 pass.
