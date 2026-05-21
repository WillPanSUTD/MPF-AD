# PointAD+ Phase 1 — Infrastructure & Baseline Reproduction

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a reproducible PointAD baseline on 4 datasets (MVTec3D-AD, Eyecandies, Real3D-AD, our welds) and record the welds zero-shot number as the gap Phase 2+3 will close.

**Architecture:** Fork upstream PointAD into `external/PointAD/`, vendor the 3 standard datasets under `external/datasets/`, write per-dataset adapters under `src/datasets/` that emit PointAD's expected JSON manifest, then run upstream `train.sh`+`test.sh` for the baselines.

**Tech Stack:** Python 3.13 (existing env), PyTorch 2.6 + CUDA 12.4 (already installed), CLIP-ViT-L (downloaded by upstream `train.sh`), `tifffile`, `Pillow`, `numpy`, `polars` (already installed), `pytest`.

**Spec:** `docs/superpowers/specs/2026-05-17-pointad-plus-design.md` — Phase 1 is §3.

**Repo state:** Not a git repo. Skip every "commit" step. Pytest passes + integration runs are the checkpoint between tasks.

---

## File structure

```
F:/dataset/LUT_AD_DataSet/
├── external/
│   ├── PointAD/                          # Task 1: clone, pin SHA in PINNED_COMMIT.txt
│   │   └── PINNED_COMMIT.txt             # (records the pinned commit)
│   └── datasets/                          # Tasks 4–7
│       ├── mvtec_3d_anomaly_detection/   # Task 4 (manual download form)
│       ├── eyecandies/                   # Task 5
│       ├── real3d_ad/                    # Task 6 (unpack existing zips)
│       └── welds_3d/                     # Task 7 (symlink to Dataset_3D/MVTec3D_Weld/)
├── src/
│   └── pointad_plus/
│       ├── __init__.py                   # Task 8
│       ├── manifest.py                   # Task 9 — shared manifest builder
│       ├── adapter_mvtec3d.py            # Task 10
│       ├── adapter_eyecandies.py         # Task 11
│       ├── adapter_real3d.py             # Task 12
│       ├── adapter_welds.py              # Task 13
│       └── welds_colored_pcd.py          # Task 14 — XYZ+Phong → colored .ply
├── tests/
│   └── pointad_plus/
│       ├── __init__.py                   # Task 8
│       ├── conftest.py                   # Task 8
│       ├── test_manifest.py              # Task 9
│       ├── test_adapter_mvtec3d.py       # Task 10
│       ├── test_adapter_eyecandies.py    # Task 11
│       ├── test_adapter_real3d.py        # Task 12
│       ├── test_adapter_welds.py         # Task 13
│       └── test_welds_colored_pcd.py     # Task 14
├── scripts/
│   └── pointad_plus/
│       ├── reproduce_baseline.sh         # Task 15 — wraps PointAD train+test
│       └── welds_zero_shot.sh            # Task 16
└── results/
    ├── baseline/
    │   ├── mvtec3d_ad/                   # Task 15 output
    │   ├── eyecandies/
    │   └── real3d_ad/
    └── welds_zero_shot/                  # Task 16 output
```

---

## Task 0: Fork PointAD and document its expected data format

**Files:**
- Create: `external/PointAD/` (cloned)
- Create: `external/PointAD/PINNED_COMMIT.txt`
- Create: `docs/pointad_format_notes.md` — notes on PointAD's expected data layout

- [ ] **Step 0.1: Clone PointAD**

```bash
cd F:/dataset/LUT_AD_DataSet
mkdir -p external
git clone https://github.com/zqhang/PointAD.git external/PointAD
```

- [ ] **Step 0.2: Pin commit SHA**

```bash
cd F:/dataset/LUT_AD_DataSet/external/PointAD
git rev-parse HEAD > PINNED_COMMIT.txt
git log -1 --oneline >> PINNED_COMMIT.txt
```

- [ ] **Step 0.3: Map PointAD's expected data layout**

Inspect these files and capture exact paths + formats:
- `external/PointAD/dataset/` (or wherever it expects data)
- `external/PointAD/train.sh` and any `train.py` — what CLI args it expects, what dataset root it reads
- `external/PointAD/test.sh` / `test.py` — same
- Any JSON / TXT manifest generator (e.g. `mvtec_3d_anomaly_mvtect_3d_ad_whole.py` per the README extraction in REF_PAPER_SUMMARY.md)
- The expected on-disk layout for each of MVTec3D-AD, Eyecandies, Real3D-AD

Document findings in `docs/pointad_format_notes.md` with these sections:

```markdown
# PointAD data-format notes

## Pinned commit
[SHA from PINNED_COMMIT.txt]

## Dataset root and discovery
[Where train.sh reads the dataset from — env var? CLI arg? config file?]

## MVTec3D-AD expected layout
[exact directory tree PointAD expects]

## Eyecandies expected layout
[same]

## Real3D-AD expected layout
[same]

## Manifest format (if used)
[exact JSON schema or TXT format, with one minimal example]

## Train.sh entrypoint
[what command, what flags, what config files]

## Test.sh entrypoint
[same]

## CLIP checkpoint expectations
[where it downloads / loads CLIP-ViT-L from]
```

- [ ] **Step 0.4: Skim PointAD's model file**

Identify the file(s) that house the "plug-and-play RGB" branch. Note the file path + the function/class name we will need to patch in Phase 2. Append to `pointad_format_notes.md` under "## Phase 2 insertion point".

- [ ] **Step 0.5: Verify imports work**

```bash
cd F:/dataset/LUT_AD_DataSet
python -c "import sys; sys.path.insert(0, 'external/PointAD'); from PointAD.model import PointAD" 2>&1 | tail -5
```

(Module path may differ — adjust based on what Step 0.3 found.) Expected: no `ModuleNotFoundError`. If imports fail, note missing dependencies in `pointad_format_notes.md` under "## Install gaps" and install with `pip install <pkg>` before proceeding.

---

## Task 1: Download MVTec3D-AD

**Files:**
- Create: `external/datasets/mvtec_3d_anomaly_detection/` (populated)

**Note for the implementer:** MVTec's website requires a registration form and email confirmation for academic download. The agent cannot complete the form unattended. If the download URL is paywalled or form-gated, ask the controller for the dataset URL or pre-downloaded location and proceed with extraction only.

- [ ] **Step 1.1: Attempt direct download**

Try the official mirror first:

```bash
mkdir -p F:/dataset/LUT_AD_DataSet/external/datasets/mvtec_3d_anomaly_detection
cd F:/dataset/LUT_AD_DataSet/external/datasets/mvtec_3d_anomaly_detection
curl -L --fail --connect-timeout 30 -o mvtec_3d.tar.xz \
  "https://www.mydrive.ch/shares/45920/dd1eb345346df066c63b5c95676b6c0a/download/428824485-1643285385/mvtec_3d_anomaly_detection.tar.xz"
```

If the download URL is wrong or returns HTML (form page), abort and report BLOCKED with the actual URL needed.

- [ ] **Step 1.2: Extract**

```bash
cd F:/dataset/LUT_AD_DataSet/external/datasets/mvtec_3d_anomaly_detection
tar -xJf mvtec_3d.tar.xz
ls   # should show 10 category directories: bagel, cable_gland, carrot, cookie, dowel, foam, peach, potato, rope, tire
```

- [ ] **Step 1.3: Sanity check**

```bash
find F:/dataset/LUT_AD_DataSet/external/datasets/mvtec_3d_anomaly_detection -name '*.png' | wc -l
find F:/dataset/LUT_AD_DataSet/external/datasets/mvtec_3d_anomaly_detection -name '*.tiff' | wc -l
ls F:/dataset/LUT_AD_DataSet/external/datasets/mvtec_3d_anomaly_detection/bagel/train/good/rgb/ | head -5
```

Expected: 10 category dirs, thousands of `.png` + `.tiff` files, `bagel/train/good/rgb/` non-empty.

---

## Task 2: Download Eyecandies

**Files:**
- Create: `external/datasets/eyecandies/` (populated)

**Note for the implementer:** Eyecandies is hosted on a Google Drive / HuggingFace mirror; the URL may have changed. The Eyecandies GitHub README (https://github.com/eyecan-ai/eyecandies) has the canonical download instruction.

- [ ] **Step 2.1: Fetch download instructions**

```bash
curl -s https://raw.githubusercontent.com/eyecan-ai/eyecandies/main/README.md | head -100
```

Read the README to determine the current download URL or instructions.

- [ ] **Step 2.2: Download per the README**

Execute the documented download commands. Save into `external/datasets/eyecandies/`.

If the download requires a Google Drive form or browser auth, abort and report BLOCKED with the URL/path needed.

- [ ] **Step 2.3: Sanity check**

```bash
ls F:/dataset/LUT_AD_DataSet/external/datasets/eyecandies/
```

Expected: 10 category directories (Candy Cane, Chocolate Cookie, Chocolate Praline, Confetto, Gummy Bear, Hazelnut Truffle, Licorice Sandwich, Lollipop, Marshmallow, Peppermint Candy).

```bash
find F:/dataset/LUT_AD_DataSet/external/datasets/eyecandies/CandyCane -maxdepth 4 -name '*.png' | head -10
```

Expected: paths under `train/data/`, multiple `image_*.png` files per scene (one per light direction).

---

## Task 3: Unpack Real3D-AD

**Files:**
- Create: `external/datasets/real3d_ad/` (populated)

- [ ] **Step 3.1: Unpack PCD archive**

```bash
mkdir -p F:/dataset/LUT_AD_DataSet/external/datasets/real3d_ad
cd F:/dataset/LUT_AD_DataSet/external/datasets/real3d_ad
unzip -q F:/dataset/Real3D-AD-PCD.zip -d .
ls
```

Expected: 12 category directories (airplane, candybar, car, chicken, diamond, duck, fish, gemstone, seahorse, shell, starfish, toffees), each containing `train/`, `test/`, `gt/`.

- [ ] **Step 3.2: Unpack PLY archive (companion format)**

```bash
unzip -q F:/dataset/Real3D-AD-PLY.zip -d ply/
ls ply/
```

The PLY directory holds the same 12 categories in `.ply` mesh form — kept as an alternative source.

- [ ] **Step 3.3: Sanity check**

```bash
ls F:/dataset/LUT_AD_DataSet/external/datasets/real3d_ad/airplane/train/ | head -5
ls F:/dataset/LUT_AD_DataSet/external/datasets/real3d_ad/airplane/test/ | head -5
ls F:/dataset/LUT_AD_DataSet/external/datasets/real3d_ad/airplane/gt/ | head -5
```

Expected: 4 template `.pcd` files in `train/`, several test `.pcd` files in `test/`, matching label `.txt` files in `gt/`.

---

## Task 4: Symlink welds_3d

**Files:**
- Create: `external/datasets/welds_3d/` (symlink or copy)

- [ ] **Step 4.1: Create symlink**

Windows note: symlink creation may require admin shell. Fallback to copy if needed.

```bash
cd F:/dataset/LUT_AD_DataSet/external/datasets
# Try symlink first
ln -s ../../Dataset_3D/MVTec3D_Weld welds_3d 2>&1 || \
  cp -r ../../Dataset_3D/MVTec3D_Weld welds_3d
```

- [ ] **Step 4.2: Sanity check**

```bash
ls F:/dataset/LUT_AD_DataSet/external/datasets/welds_3d/weld/
```

Expected: `train/`, `validation/`, `test/` dirs.

---

## Task 5: Test scaffolding for adapter package

**Files:**
- Create: `src/pointad_plus/__init__.py` (empty)
- Create: `tests/pointad_plus/__init__.py` (empty)
- Create: `tests/pointad_plus/conftest.py`

- [ ] **Step 5.1: Create empty init files**

```bash
mkdir -p F:/dataset/LUT_AD_DataSet/src/pointad_plus
mkdir -p F:/dataset/LUT_AD_DataSet/tests/pointad_plus
touch F:/dataset/LUT_AD_DataSet/src/pointad_plus/__init__.py
touch F:/dataset/LUT_AD_DataSet/tests/pointad_plus/__init__.py
```

- [ ] **Step 5.2: Write `tests/pointad_plus/conftest.py`**

```python
"""Shared fixtures for pointad_plus tests."""
from pathlib import Path
import pytest

REPO_ROOT = Path("F:/dataset/LUT_AD_DataSet")


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def dataset_root() -> Path:
    return REPO_ROOT / "external" / "datasets"


@pytest.fixture
def welds_root(dataset_root) -> Path:
    return dataset_root / "welds_3d" / "weld"
```

---

## Task 6: Shared manifest builder

**Files:**
- Create: `src/pointad_plus/manifest.py`
- Create: `tests/pointad_plus/test_manifest.py`

PointAD's exact manifest schema is locked in Task 0 (`docs/pointad_format_notes.md` → "## Manifest format"). The schema referenced below MUST match those notes — if Task 0 found a different schema, update the schema constant in `manifest.py` and the test assertions, then proceed.

- [ ] **Step 6.1: Write failing tests**

`tests/pointad_plus/test_manifest.py`:

```python
"""Tests for the shared manifest builder."""
import json
from pathlib import Path
import pytest

from src.pointad_plus import manifest


def test_sample_keys_minimum():
    """Every emitted sample must contain at least the PointAD-required keys."""
    sample = manifest.make_sample(
        sample_id="s1",
        rgb_paths=["/x/rgb/s1.png"],
        pcd_path="/x/xyz/s1.tiff",
        gt_mask_path=None,
        class_name="weld",
        anomaly=False,
    )
    for k in ("id", "rgb", "pcd", "class", "anomaly"):
        assert k in sample


def test_rgb_is_list():
    """The `rgb` field is always a list (length 1 for single-RGB, M for multi-photo)."""
    s = manifest.make_sample(
        sample_id="s1",
        rgb_paths=["/x/a.png", "/x/b.png", "/x/c.png"],
        pcd_path="/x/xyz/s1.tiff",
        gt_mask_path=None,
        class_name="weld",
        anomaly=False,
    )
    assert isinstance(s["rgb"], list)
    assert len(s["rgb"]) == 3


def test_gt_mask_optional_for_good():
    """Good samples may have gt_mask=None (PointAD treats missing GT as all-zero)."""
    s = manifest.make_sample(
        sample_id="s1",
        rgb_paths=["/x/a.png"],
        pcd_path="/x/xyz/s1.tiff",
        gt_mask_path=None,
        class_name="weld",
        anomaly=False,
    )
    assert s["gt_mask"] is None


def test_write_manifest_round_trip(tmp_path):
    """Manifest round-trips through JSON."""
    samples_train = [manifest.make_sample("s1", ["/a.png"], "/x.tiff", None, "weld", False)]
    samples_test  = [manifest.make_sample("s2", ["/b.png"], "/y.tiff", "/g.png", "weld", True)]
    out = tmp_path / "manifest.json"
    manifest.write_manifest(out, train=samples_train, test=samples_test)
    loaded = json.loads(out.read_text())
    assert loaded["train"] == samples_train
    assert loaded["test"] == samples_test
```

- [ ] **Step 6.2: Run tests — expect failure**

```bash
cd F:/dataset/LUT_AD_DataSet
python -m pytest tests/pointad_plus/test_manifest.py -v
```

Expected: ImportError on `src.pointad_plus.manifest`.

- [ ] **Step 6.3: Implement `src/pointad_plus/manifest.py`**

```python
"""Build PointAD-compatible JSON manifests.

Schema (locked in docs/pointad_format_notes.md):
  {
    "train": [<sample>...],
    "test":  [<sample>...]
  }

Sample:
  {
    "id":       str,
    "rgb":      [path, ...],   # list, even if length 1
    "pcd":      path,
    "gt_mask":  path | null,
    "class":    str,
    "anomaly":  bool
  }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


def make_sample(
    sample_id: str,
    rgb_paths: List[str],
    pcd_path: str,
    gt_mask_path: Optional[str],
    class_name: str,
    anomaly: bool,
) -> dict:
    return {
        "id": sample_id,
        "rgb": list(rgb_paths),
        "pcd": pcd_path,
        "gt_mask": gt_mask_path,
        "class": class_name,
        "anomaly": anomaly,
    }


def write_manifest(out_path: Path | str, train: List[dict], test: List[dict]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train": train, "test": test}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(in_path: Path | str) -> dict:
    return json.loads(Path(in_path).read_text(encoding="utf-8"))
```

- [ ] **Step 6.4: Run tests — expect pass**

```bash
python -m pytest tests/pointad_plus/test_manifest.py -v
```

Expected: 4 passed.

---

## Task 7: MVTec3D-AD adapter

**Files:**
- Create: `src/pointad_plus/adapter_mvtec3d.py`
- Create: `tests/pointad_plus/test_adapter_mvtec3d.py`

- [ ] **Step 7.1: Write failing tests**

```python
"""Tests for the MVTec3D-AD adapter."""
from pathlib import Path
import pytest

from src.pointad_plus import adapter_mvtec3d


def test_collect_train_samples_returns_good_only(dataset_root):
    cats = adapter_mvtec3d.CATEGORIES
    assert "bagel" in cats
    samples = adapter_mvtec3d.collect_train_samples(
        dataset_root / "mvtec_3d_anomaly_detection", category="bagel"
    )
    assert len(samples) > 0
    assert all(s["anomaly"] is False for s in samples)
    assert all(s["gt_mask"] is None for s in samples)


def test_collect_test_samples_has_good_and_anomalous(dataset_root):
    samples = adapter_mvtec3d.collect_test_samples(
        dataset_root / "mvtec_3d_anomaly_detection", category="bagel"
    )
    good = [s for s in samples if not s["anomaly"]]
    anom = [s for s in samples if s["anomaly"]]
    assert len(good) > 0
    assert len(anom) > 0
    # Every anomalous test sample has a GT mask path
    assert all(s["gt_mask"] is not None for s in anom)


def test_emit_manifest_per_category(tmp_path, dataset_root):
    out = tmp_path / "mvtec3d_bagel.json"
    adapter_mvtec3d.emit_category_manifest(
        dataset_root / "mvtec_3d_anomaly_detection",
        category="bagel",
        out_path=out,
    )
    assert out.exists()
```

- [ ] **Step 7.2: Run — expect failure**

```bash
python -m pytest tests/pointad_plus/test_adapter_mvtec3d.py -v
```

- [ ] **Step 7.3: Implement `src/pointad_plus/adapter_mvtec3d.py`**

```python
"""MVTec 3D-AD → PointAD manifest adapter.

Layout assumed (from official release, also documented in
docs/pointad_format_notes.md):
  <root>/<category>/
    train/good/{rgb,xyz}/*
    test/good/{rgb,xyz}/*
    test/<defect_type>/{rgb,xyz,gt}/*
    validation/good/{rgb,xyz}/*
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.pointad_plus.manifest import make_sample, write_manifest

CATEGORIES = (
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
)


def _collect_split(root: Path, category: str, split: str) -> List[dict]:
    """Walk <root>/<category>/<split>/ and emit one sample per stem."""
    cat_dir = root / category / split
    if not cat_dir.is_dir():
        return []
    samples: List[dict] = []
    for defect_dir in sorted(cat_dir.iterdir()):
        if not defect_dir.is_dir():
            continue
        is_good = defect_dir.name == "good"
        rgb_dir = defect_dir / "rgb"
        xyz_dir = defect_dir / "xyz"
        gt_dir = defect_dir / "gt"
        for rgb in sorted(rgb_dir.glob("*.png")):
            stem = rgb.stem
            xyz = xyz_dir / f"{stem}.tiff"
            if not xyz.exists():
                continue
            gt = gt_dir / f"{stem}.png" if (not is_good and gt_dir.is_dir()) else None
            if gt is not None and not gt.exists():
                gt = None
            samples.append(
                make_sample(
                    sample_id=f"{category}_{split}_{defect_dir.name}_{stem}",
                    rgb_paths=[str(rgb)],
                    pcd_path=str(xyz),
                    gt_mask_path=str(gt) if gt else None,
                    class_name=category,
                    anomaly=not is_good,
                )
            )
    return samples


def collect_train_samples(root: Path, category: str) -> List[dict]:
    return _collect_split(root, category, "train")


def collect_test_samples(root: Path, category: str) -> List[dict]:
    return _collect_split(root, category, "test")


def emit_category_manifest(root: Path, category: str, out_path: Path) -> None:
    train = collect_train_samples(root, category)
    test = collect_test_samples(root, category)
    write_manifest(out_path, train=train, test=test)


def emit_all_manifests(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        emit_category_manifest(root, cat, out_dir / f"{cat}.json")
```

- [ ] **Step 7.4: Run — expect pass**

```bash
python -m pytest tests/pointad_plus/test_adapter_mvtec3d.py -v
```

Expected: 3 passed.

- [ ] **Step 7.5: Smoke-run on real data**

```bash
cd F:/dataset/LUT_AD_DataSet
python -c "from pathlib import Path; from src.pointad_plus.adapter_mvtec3d import emit_all_manifests; emit_all_manifests(Path('external/datasets/mvtec_3d_anomaly_detection'), Path('external/datasets/mvtec_3d_anomaly_detection/manifests'))"
ls external/datasets/mvtec_3d_anomaly_detection/manifests/
```

Expected: 10 JSON files, one per category. Each non-empty.

---

## Task 8: Eyecandies adapter

**Files:**
- Create: `src/pointad_plus/adapter_eyecandies.py`
- Create: `tests/pointad_plus/test_adapter_eyecandies.py`

The Eyecandies layout is documented in their GitHub README — Task 0 should have captured the exact tree. Task 2's sanity check produced an output that confirms it; mirror that tree below.

- [ ] **Step 8.1: Write failing tests**

```python
"""Tests for the Eyecandies adapter."""
from pathlib import Path
import pytest

from src.pointad_plus import adapter_eyecandies


def test_categories_are_10(dataset_root):
    assert len(adapter_eyecandies.CATEGORIES) == 10


def test_collect_test_samples_has_6_lights(dataset_root):
    """Each Eyecandies sample exposes 6 light directions in the `rgb` list."""
    root = dataset_root / "eyecandies"
    samples = adapter_eyecandies.collect_test_samples(root, category=adapter_eyecandies.CATEGORIES[0])
    assert len(samples) > 0
    assert all(len(s["rgb"]) == 6 for s in samples)
```

- [ ] **Step 8.2: Run — expect failure**

- [ ] **Step 8.3: Implement adapter (skeleton — fill in directory walking per Task 0 notes)**

The exact subdirectory names (`train/data/`, `image_0.png`–`image_5.png`, `depth.tiff`, `normals.tiff`, `mask.png`) come from Task 0's `pointad_format_notes.md`. If notes say differently, defer to the notes.

```python
"""Eyecandies → PointAD manifest adapter (6-light multi-photometric)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.pointad_plus.manifest import make_sample, write_manifest

CATEGORIES = (
    "CandyCane", "ChocolateCookie", "ChocolatePraline", "Confetto",
    "GummyBear", "HazelnutTruffle", "LicoriceSandwich", "Lollipop",
    "Marshmallow", "PeppermintCandy",
)
# Confirm exact names from `ls external/datasets/eyecandies/` in Task 2.

LIGHTS = ("image_0", "image_1", "image_2", "image_3", "image_4", "image_5")


def _scene_rgb_paths(scene_dir: Path) -> List[str]:
    return [str(scene_dir / f"{light}.png") for light in LIGHTS]


def _collect_split(root: Path, category: str, split: str) -> List[dict]:
    cat_dir = root / category / split / "data"
    if not cat_dir.is_dir():
        return []
    samples: List[dict] = []
    for scene_dir in sorted(cat_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        rgbs = _scene_rgb_paths(scene_dir)
        depth = scene_dir / "depth.tiff"
        gt = scene_dir / "mask.png"
        anomaly = gt.exists() and gt.stat().st_size > 0
        samples.append(make_sample(
            sample_id=f"{category}_{split}_{scene_dir.name}",
            rgb_paths=rgbs,
            pcd_path=str(depth),
            gt_mask_path=str(gt) if (anomaly and gt.exists()) else None,
            class_name=category,
            anomaly=anomaly,
        ))
    return samples


def collect_train_samples(root: Path, category: str) -> List[dict]:
    return _collect_split(root, category, "train")


def collect_test_samples(root: Path, category: str) -> List[dict]:
    return _collect_split(root, category, "test")


def emit_category_manifest(root: Path, category: str, out_path: Path) -> None:
    write_manifest(
        out_path,
        train=collect_train_samples(root, category),
        test=collect_test_samples(root, category),
    )


def emit_all_manifests(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        emit_category_manifest(root, cat, out_dir / f"{cat}.json")
```

- [ ] **Step 8.4: Run — expect pass**

```bash
python -m pytest tests/pointad_plus/test_adapter_eyecandies.py -v
```

- [ ] **Step 8.5: Smoke-run**

```bash
python -c "from pathlib import Path; from src.pointad_plus.adapter_eyecandies import emit_all_manifests; emit_all_manifests(Path('external/datasets/eyecandies'), Path('external/datasets/eyecandies/manifests'))"
ls external/datasets/eyecandies/manifests/
```

Expected: 10 JSON files.

---

## Task 9: Real3D-AD adapter

**Files:**
- Create: `src/pointad_plus/adapter_real3d.py`
- Create: `tests/pointad_plus/test_adapter_real3d.py`

Real3D-AD layout per Task 3:
```
<root>/<category>/
  train/*.pcd         # ≈4 prototype templates per category
  test/*.pcd          # anomalous test PCDs
  gt/*.txt            # per-point binary labels matching test PCDs by stem
```

- [ ] **Step 9.1: Write failing tests**

```python
"""Tests for the Real3D-AD adapter."""
from pathlib import Path
import pytest

from src.pointad_plus import adapter_real3d


def test_categories_are_12(dataset_root):
    assert len(adapter_real3d.CATEGORIES) == 12


def test_train_samples_are_templates(dataset_root):
    samples = adapter_real3d.collect_train_samples(
        dataset_root / "real3d_ad", category="airplane"
    )
    assert 1 <= len(samples) <= 6
    assert all(s["anomaly"] is False for s in samples)
    assert all(s["pcd"].endswith(".pcd") for s in samples)


def test_test_samples_have_gt(dataset_root):
    samples = adapter_real3d.collect_test_samples(
        dataset_root / "real3d_ad", category="airplane"
    )
    assert len(samples) > 0
    assert all(s["anomaly"] is True for s in samples)
    assert all(s["gt_mask"] is not None for s in samples)
```

- [ ] **Step 9.2: Run — expect failure**

- [ ] **Step 9.3: Implement**

```python
"""Real3D-AD → PointAD manifest adapter."""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.pointad_plus.manifest import make_sample, write_manifest

CATEGORIES = (
    "airplane", "candybar", "car", "chicken", "diamond", "duck",
    "fish", "gemstone", "seahorse", "shell", "starfish", "toffees",
)


def collect_train_samples(root: Path, category: str) -> List[dict]:
    train_dir = root / category / "train"
    if not train_dir.is_dir():
        return []
    out: List[dict] = []
    for pcd in sorted(train_dir.glob("*.pcd")):
        out.append(make_sample(
            sample_id=f"{category}_train_{pcd.stem}",
            rgb_paths=[],  # no RGB modality in Real3D-AD
            pcd_path=str(pcd),
            gt_mask_path=None,
            class_name=category,
            anomaly=False,
        ))
    return out


def collect_test_samples(root: Path, category: str) -> List[dict]:
    test_dir = root / category / "test"
    gt_dir = root / category / "gt"
    if not test_dir.is_dir():
        return []
    out: List[dict] = []
    for pcd in sorted(test_dir.glob("*.pcd")):
        gt = gt_dir / f"{pcd.stem}.txt"
        out.append(make_sample(
            sample_id=f"{category}_test_{pcd.stem}",
            rgb_paths=[],
            pcd_path=str(pcd),
            gt_mask_path=str(gt) if gt.exists() else None,
            class_name=category,
            anomaly=True,
        ))
    return out


def emit_category_manifest(root: Path, category: str, out_path: Path) -> None:
    write_manifest(
        out_path,
        train=collect_train_samples(root, category),
        test=collect_test_samples(root, category),
    )


def emit_all_manifests(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in CATEGORIES:
        emit_category_manifest(root, cat, out_dir / f"{cat}.json")
```

- [ ] **Step 9.4: Run — expect pass**

- [ ] **Step 9.5: Smoke-run**

```bash
python -c "from pathlib import Path; from src.pointad_plus.adapter_real3d import emit_all_manifests; emit_all_manifests(Path('external/datasets/real3d_ad'), Path('external/datasets/real3d_ad/manifests'))"
ls external/datasets/real3d_ad/manifests/
```

Expected: 12 JSON files.

---

## Task 10: Welds adapter — single Phong (Phase 1 scope)

**Files:**
- Create: `src/pointad_plus/adapter_welds.py`
- Create: `tests/pointad_plus/test_adapter_welds.py`

Welds layout per `Dataset_3D/MVTec3D_Weld/weld/`:
```
weld/
  train/good/{rgb,xyz,gt}/<000001>.{png,tiff,png}
  validation/good/{rgb,xyz,gt}/...
  test/good/{rgb,xyz,gt}/...
  test/{pseudo_soldering,pinhole,pit,burst,fish_scale_welding,bump,combined}/{rgb,xyz,gt,gt_geom}/...
```

Phase 1 uses single-Phong RGB (paths in `rgb/`) only. Multi-photometric requires Task 11 (colored PCD generation) + a separate manifest, deferred to Phase 2.

- [ ] **Step 10.1: Write failing tests**

```python
"""Tests for the welds adapter."""
from pathlib import Path
import pytest

from src.pointad_plus import adapter_welds


def test_train_samples_collected(welds_root):
    samples = adapter_welds.collect_train_samples(welds_root)
    assert len(samples) > 100  # we have 714 train good patches
    assert all(s["anomaly"] is False for s in samples)
    assert all(len(s["rgb"]) == 1 for s in samples)  # Phase 1: single Phong only


def test_test_samples_per_defect_class(welds_root):
    samples = adapter_welds.collect_test_samples(welds_root)
    classes = {s["defect_class"] for s in samples if s["anomaly"]}
    expected = {
        "pseudo_soldering", "pinhole", "pit", "burst",
        "fish_scale_welding", "bump", "combined",
    }
    assert expected.issubset(classes)


def test_test_good_samples_present(welds_root):
    samples = adapter_welds.collect_test_samples(welds_root)
    good = [s for s in samples if not s["anomaly"]]
    assert len(good) > 50  # we have 153 test_good patches
```

- [ ] **Step 10.2: Run — expect failure**

- [ ] **Step 10.3: Implement**

```python
"""Welds → PointAD manifest adapter (Phase 1: single-Phong RGB only)."""
from __future__ import annotations

from pathlib import Path
from typing import List

from src.pointad_plus.manifest import make_sample, write_manifest

CATEGORY = "weld"
DEFECT_CLASSES = (
    "pseudo_soldering", "pinhole", "pit", "burst",
    "fish_scale_welding", "bump", "combined",
)


def _collect_subfolder(weld_root: Path, split: str, defect_class: str, anomaly: bool) -> List[dict]:
    sub = weld_root / split / defect_class
    rgb_dir = sub / "rgb"
    xyz_dir = sub / "xyz"
    gt_dir = sub / "gt"
    if not rgb_dir.is_dir():
        return []
    out: List[dict] = []
    for rgb in sorted(rgb_dir.glob("*.png")):
        stem = rgb.stem
        xyz = xyz_dir / f"{stem}.tiff"
        if not xyz.exists():
            continue
        gt = gt_dir / f"{stem}.png" if anomaly else None
        s = make_sample(
            sample_id=f"{split}_{defect_class}_{stem}",
            rgb_paths=[str(rgb)],
            pcd_path=str(xyz),
            gt_mask_path=str(gt) if (gt and gt.exists()) else None,
            class_name=CATEGORY,
            anomaly=anomaly,
        )
        s["defect_class"] = defect_class  # extra field for per-class breakdown
        out.append(s)
    return out


def collect_train_samples(weld_root: Path) -> List[dict]:
    return _collect_subfolder(weld_root, "train", "good", anomaly=False)


def collect_val_samples(weld_root: Path) -> List[dict]:
    return _collect_subfolder(weld_root, "validation", "good", anomaly=False)


def collect_test_samples(weld_root: Path) -> List[dict]:
    out: List[dict] = _collect_subfolder(weld_root, "test", "good", anomaly=False)
    for dc in DEFECT_CLASSES:
        out.extend(_collect_subfolder(weld_root, "test", dc, anomaly=True))
    return out


def emit_manifest(weld_root: Path, out_path: Path) -> None:
    write_manifest(
        out_path,
        train=collect_train_samples(weld_root),
        test=collect_test_samples(weld_root),
    )
```

- [ ] **Step 10.4: Run — expect pass**

- [ ] **Step 10.5: Smoke-run**

```bash
python -c "from pathlib import Path; from src.pointad_plus.adapter_welds import emit_manifest; emit_manifest(Path('external/datasets/welds_3d/weld'), Path('external/datasets/welds_3d/manifests/weld.json'))"
python -c "import json; m = json.load(open('F:/dataset/LUT_AD_DataSet/external/datasets/welds_3d/manifests/weld.json')); print('train', len(m['train']), 'test', len(m['test']))"
```

Expected output: `train 714 test ~694` (153 test_good + 541 anomalous).

---

## Task 11: Welds colored-PCD generator

**Files:**
- Create: `src/pointad_plus/welds_colored_pcd.py`
- Create: `tests/pointad_plus/test_welds_colored_pcd.py`

PointAD expects colored point clouds (xyz + per-point RGB). Our welds dataset has organized XYZ tiff (`xyz/<id>.tiff`) + Phong PNG (`rgb/<id>.png`) at the same H×W. Generate colored PCDs by lifting RGB pixels onto XYZ points.

- [ ] **Step 11.1: Write failing test**

```python
"""Tests for welds_colored_pcd."""
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
import pytest

from src.pointad_plus import welds_colored_pcd as wcp


def test_build_colored_pcd_creates_pcd(tmp_path):
    # Synthesize a 4×4 XYZ tiff and an RGB PNG
    xyz = np.array([[[u * 0.016, v * 0.016, 1.0 + 0.01 * (u + v)] for u in range(4)] for v in range(4)], dtype=np.float32)
    rgb = (np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    xyz_path = tmp_path / "x.tiff"
    rgb_path = tmp_path / "x.png"
    tifffile.imwrite(str(xyz_path), xyz)
    Image.fromarray(rgb).save(rgb_path)

    out = tmp_path / "x.pcd"
    wcp.build_colored_pcd(xyz_path=xyz_path, rgb_path=rgb_path, out_path=out)
    assert out.exists()
    text = out.read_text(encoding="ascii").splitlines()
    assert text[0].startswith("# .PCD")
    # FIELDS must include x y z rgb
    fields_line = [l for l in text if l.startswith("FIELDS")][0]
    assert "rgb" in fields_line.lower() or "r g b" in fields_line.lower()
```

- [ ] **Step 11.2: Run — expect failure**

- [ ] **Step 11.3: Implement**

```python
"""Welds organized XYZ + Phong PNG → ASCII colored PCD.

Drops invalid (0,0,0) XYZ pixels per the MVTec3D-AD convention.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def _pack_rgb_uint32(rgb_u8: np.ndarray) -> np.ndarray:
    """Pack per-point (R, G, B) uint8 → uint32 for PointAD/PCL's rgb field."""
    r = rgb_u8[..., 0].astype(np.uint32)
    g = rgb_u8[..., 1].astype(np.uint32)
    b = rgb_u8[..., 2].astype(np.uint32)
    return (r << 16) | (g << 8) | b


def build_colored_pcd(xyz_path: Path, rgb_path: Path, out_path: Path) -> None:
    xyz = tifffile.imread(str(xyz_path)).astype(np.float32)  # H, W, 3
    rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    if xyz.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"shape mismatch xyz {xyz.shape[:2]} vs rgb {rgb.shape[:2]} "
            f"for {xyz_path}/{rgb_path}"
        )
    pts = xyz.reshape(-1, 3)
    cols = rgb.reshape(-1, 3)
    valid = ~np.all(pts == 0.0, axis=1)  # MVTec3D-AD invalid = (0,0,0)
    pts = pts[valid]
    cols = cols[valid]
    rgb_packed = _pack_rgb_uint32(cols)

    n = pts.shape[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F U\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA ascii\n"
    )
    with open(out_path, "w", encoding="ascii", newline="\n") as f:
        f.write(header)
        for (x, y, z), c in zip(pts.tolist(), rgb_packed.tolist()):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {c}\n")
```

- [ ] **Step 11.4: Run — expect pass**

- [ ] **Step 11.5: Batch-generate for welds**

```bash
python - <<'PY'
from pathlib import Path
from src.pointad_plus.welds_colored_pcd import build_colored_pcd
weld_root = Path("F:/dataset/LUT_AD_DataSet/external/datasets/welds_3d/weld")
out_root = Path("F:/dataset/LUT_AD_DataSet/external/datasets/welds_3d/weld_colored_pcd")
n = 0
for split in ("train", "validation", "test"):
    for sub in (weld_root / split).iterdir():
        if not sub.is_dir():
            continue
        rgb_dir = sub / "rgb"
        xyz_dir = sub / "xyz"
        out_dir = out_root / split / sub.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for rgb in sorted(rgb_dir.glob("*.png")):
            xyz = xyz_dir / f"{rgb.stem}.tiff"
            if xyz.exists():
                build_colored_pcd(xyz, rgb, out_dir / f"{rgb.stem}.pcd")
                n += 1
print("colored PCDs written:", n)
PY
```

Expected: prints "colored PCDs written: N" where N ≈ 1561 (714 + 153 + 153 train/val/test_good + 541 anomalous test).

---

## Task 12: Reproduce PointAD baseline on MVTec3D-AD

**Files:**
- Create: `scripts/pointad_plus/reproduce_baseline.sh`
- Create: `results/baseline/mvtec3d_ad/`

- [ ] **Step 12.1: Write the wrapper script**

`scripts/pointad_plus/reproduce_baseline.sh`:

```bash
#!/usr/bin/env bash
# Wraps upstream PointAD train.sh + test.sh for our paths.
set -euo pipefail
DATASET="${1:?usage: reproduce_baseline.sh <mvtec3d|eyecandies|real3d>}"
case "$DATASET" in
  mvtec3d)
    DATA_ROOT="F:/dataset/LUT_AD_DataSet/external/datasets/mvtec_3d_anomaly_detection"
    ;;
  eyecandies)
    DATA_ROOT="F:/dataset/LUT_AD_DataSet/external/datasets/eyecandies"
    ;;
  real3d)
    DATA_ROOT="F:/dataset/LUT_AD_DataSet/external/datasets/real3d_ad"
    ;;
  *)
    echo "unknown dataset: $DATASET" >&2; exit 1 ;;
esac
OUT_ROOT="F:/dataset/LUT_AD_DataSet/results/baseline/$DATASET"
mkdir -p "$OUT_ROOT"

cd F:/dataset/LUT_AD_DataSet/external/PointAD

# CRITICAL: exact CLI args come from Task 0's pointad_format_notes.md.
# The two lines below are PLACEHOLDERS that the implementer MUST replace
# with whatever upstream train.sh actually expects.
bash train.sh --data_root "$DATA_ROOT" --output "$OUT_ROOT/train" 2>&1 | tee "$OUT_ROOT/train.log"
bash test.sh  --data_root "$DATA_ROOT" --ckpt "$OUT_ROOT/train" --output "$OUT_ROOT/test" 2>&1 | tee "$OUT_ROOT/test.log"
```

- [ ] **Step 12.2: Reconcile script with upstream CLI**

Open `external/PointAD/train.sh` and `test.sh`. Replace the two placeholder lines in `reproduce_baseline.sh` with the actual CLI signature (often these are `bash train.sh <category>` per-category and the script loops over categories internally — check Task 0 notes).

- [ ] **Step 12.3: Make executable**

```bash
chmod +x F:/dataset/LUT_AD_DataSet/scripts/pointad_plus/reproduce_baseline.sh
```

- [ ] **Step 12.4: Run on MVTec3D-AD (LONG — ~5 GPU-h on RTX 4090)**

```bash
cd F:/dataset/LUT_AD_DataSet
bash scripts/pointad_plus/reproduce_baseline.sh mvtec3d
```

This is a long-running step. Use `run_in_background: true` and monitor the log.

- [ ] **Step 12.5: Extract baseline numbers**

After the run completes, find the per-category P-AUROC / AUPRO / I-AUROC in `results/baseline/mvtec3d_ad/test/*.json` (or equivalent — exact file location from Task 0 notes). Compute the dataset-averaged P-AUROC, AUPRO, I-AUROC.

Write a summary to `results/baseline/mvtec3d_ad/SUMMARY.md`:

```markdown
# PointAD baseline — MVTec3D-AD

Run date: [today]
Pinned commit: [from PINNED_COMMIT.txt]
GPU: RTX 4090

| Category | P-AUROC | AUPRO | I-AUROC |
|---|---|---|---|
| bagel | ... | ... | ... |
...
| **mean** | ... | ... | ... |

**Published numbers (PointAD paper):** P-AUROC=..., AUPRO=..., I-AUROC=...
**Gap to published:** ±... P-AUROC (target: ±1%)
```

- [ ] **Step 12.6: Phase 1 gate check (MVTec3D-AD)**

If averaged P-AUROC is within ±1% of published, mark this task done. Otherwise log the gap, re-check config / pretrained checkpoint, optionally file an issue on `zqhang/PointAD`.

---

## Task 13: Reproduce baseline on Eyecandies

- [ ] **Step 13.1: Run**

```bash
cd F:/dataset/LUT_AD_DataSet
bash scripts/pointad_plus/reproduce_baseline.sh eyecandies
```

- [ ] **Step 13.2: Summary**

Write `results/baseline/eyecandies/SUMMARY.md` mirroring Task 12.5.

- [ ] **Step 13.3: Gate check** — same as Task 12.6.

---

## Task 14: Reproduce baseline on Real3D-AD

- [ ] **Step 14.1: Run**

```bash
cd F:/dataset/LUT_AD_DataSet
bash scripts/pointad_plus/reproduce_baseline.sh real3d
```

- [ ] **Step 14.2: Summary** at `results/baseline/real3d_ad/SUMMARY.md`.

- [ ] **Step 14.3: Gate check** — same as Task 12.6.

---

## Task 15: Welds zero-shot baseline

**Files:**
- Create: `scripts/pointad_plus/welds_zero_shot.sh`
- Create: `results/welds_zero_shot/SUMMARY.md`

The welds eval reuses the MVTec3D-AD-trained checkpoint from Task 12 (PointAD's zero-shot protocol — auxiliary categories train on one dataset, eval transfers to another).

- [ ] **Step 15.1: Write the script**

`scripts/pointad_plus/welds_zero_shot.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
CKPT_SOURCE="${1:?usage: welds_zero_shot.sh <mvtec3d|eyecandies|real3d>}"

CKPT_PATH="F:/dataset/LUT_AD_DataSet/results/baseline/$CKPT_SOURCE/train"
DATA_ROOT="F:/dataset/LUT_AD_DataSet/external/datasets/welds_3d"
OUT_ROOT="F:/dataset/LUT_AD_DataSet/results/welds_zero_shot/from_$CKPT_SOURCE"
mkdir -p "$OUT_ROOT"

cd F:/dataset/LUT_AD_DataSet/external/PointAD
bash test.sh --data_root "$DATA_ROOT" --ckpt "$CKPT_PATH" --output "$OUT_ROOT" \
    2>&1 | tee "$OUT_ROOT/test.log"
```

Same caveat as Task 12.2: replace the `test.sh` invocation with whatever signature upstream actually expects.

- [ ] **Step 15.2: Make executable + run**

```bash
chmod +x F:/dataset/LUT_AD_DataSet/scripts/pointad_plus/welds_zero_shot.sh
bash F:/dataset/LUT_AD_DataSet/scripts/pointad_plus/welds_zero_shot.sh mvtec3d
```

- [ ] **Step 15.3: Extract welds zero-shot numbers**

Per-defect-class P-AUROC + image-level AUROC. Aggregate to single-dataset numbers.

Write `results/welds_zero_shot/SUMMARY.md`:

```markdown
# Welds zero-shot — PointAD baseline (no fusion, no sliding window)

Source checkpoint: MVTec3D-AD aux categories
Run date: [today]

| Test class | P-AUROC | AUPRO | I-AUROC | count |
|---|---|---|---|---|
| pseudo_soldering | ... | ... | ... | 206 |
| pinhole | ... | ... | ... | 11 |
| pit | ... | ... | ... | 10 |
| burst | ... | ... | ... | 3 |
| fish_scale_welding | ... | ... | ... | 21 |
| bump | ... | ... | ... | 19 |
| combined | ... | ... | ... | 271 |
| **all anomalous** | ... | ... | ... | 541 |

Notes:
- PointAD's resize/center-crop applied (no sliding window — Phase 3 will add it).
- Single-Phong RGB only (no multi-photometric fusion — Phase 2 will add it).
- This is the baseline gap that PointAD+ must close in Phases 2+3.
```

- [ ] **Step 15.4: Phase 1 success-gate close-out**

Confirm:
1. MVTec3D-AD baseline within ±1% of published averaged P-AUROC.
2. Eyecandies baseline within ±1%.
3. Real3D-AD baseline within ±1%.
4. Welds zero-shot number recorded (any value).

If all four hold, Phase 1 is complete. Otherwise resolve gaps before opening the Phase 2 plan.

---

## Notes for the implementing engineer

- **Working directory** is `F:/dataset/LUT_AD_DataSet`.
- **Not a git repo.** No commits. Pytest passes + log files are the checkpoint.
- **Long-running tasks** (12, 13, 14, 15): use `run_in_background` and monitor the log via `Read` or `Monitor`. Don't sleep-poll.
- **Dataset download blockers**: if MVTec3D-AD or Eyecandies requires a form/auth that an agent can't navigate, escalate as BLOCKED with the exact URL the user needs to visit. The user can pre-download and place into the expected path.
- **PointAD CLI signature**: Task 0's `pointad_format_notes.md` is the source of truth. Tasks 12–15 reference placeholders that MUST be replaced with the actual signature before running.
- **Pytest path**: All tests live under `tests/pointad_plus/`. Run with `cd F:/dataset/LUT_AD_DataSet && python -m pytest tests/pointad_plus/ -v`.
- **`src.pointad_plus.*` import path** assumes the repo root is on `sys.path`. If pytest can't find the module, add a `conftest.py` at the repo root that prepends `repo_root` to `sys.path`, or invoke pytest as `python -m pytest` (which adds the cwd to `sys.path`).
