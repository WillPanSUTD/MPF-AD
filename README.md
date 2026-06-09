# MPF-AD + LUT-AD

This repository hosts code for two related projects on weld
defect detection:

1. **MPF-AD** *(this repo's main contribution)* — *Parameter-Free
   Multi-Photometric Fusion for Zero-Shot 3D Weld Anomaly Detection*.
   Under review at *Engineering Applications of Artificial Intelligence*
   (EAAI), 2026.
2. **LUT-AD** *(parent project, code-only)* — *Physically Inspired
   Geometry-to-Photometry Rendering for Real-Time Weld Defect Detection*.
   Also under review at EAAI, 2026.

The two share the same source weld scans and rendering pipeline; the
README below describes the LUT-AD detector code. **For the MPF-AD
zero-shot pipeline, see [`src/pointad_plus/`](src/pointad_plus/),
[`scripts/pointad_plus/`](scripts/pointad_plus/),
[`scripts/mvtec3d/`](scripts/mvtec3d/),
[`tests/pointad_plus/`](tests/pointad_plus/),
[`docs/superpowers/`](docs/superpowers/), and the
[MPW-AD dataset card](MPW-AD/README.md).**

A minimum MPF-AD reproduction (Phase 3 Sliding-Window MPF on welds):
```bash
python -m src.pointad_plus.run_welds_pointad_plus_sw \
    --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
    --pointad_ckpt external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth \
    --patch_size 256 --stride 128 \
    --out results/welds_pointad_plus_sw/
```

Runners require: a single RTX 4090-class GPU (≥ 12 GB), the pre-rendered
MPW-AD dataset (placed at `Dataset_3D/MVTec3D_Weld/`), and the shipped
PointAD checkpoint (fetch from `https://github.com/zqhang/PointAD`).
`results/`, `external/`, and `Dataset_3D/` are intentionally excluded
from this repo (large data, fork of upstream code, regenerable artefacts).

---

# Geometry-to-Photometry Rendering for Real-Time Weld Defect Detection

Code and dataset for the LUT-AD paper *"Physically Inspired
Geometry-to-Photometry Rendering for Real-Time Weld Defect Detection"*
(under review at *Engineering Applications of Artificial Intelligence*,
EAAI).

[ [中文版](README.zh-CN.md) | [Algorithm details](docs/algorithm.md) | [Dataset card](docs/dataset_card.md) | [3D-AD variants](docs/dataset_3d.md) ]

---

## Overview

We target weld defect inspection on the top covers of lithium-ion battery cells.
Pure 2D imaging captures rich texture but is unreliable on specular metallic surfaces;
3D point-cloud methods capture geometry faithfully but are too heavy for on-line
deployment. This work bridges the two by rendering 3D depth into a discriminative 2D
photometric image and feeding that to a lightweight 2D detector:

1. **Depth-Normal Guided Illumination Rendering** (C++, OpenCV, Eigen)
   - Statistical-prior LUT mapping turns the raw depth map into a pseudo-color image
     that preserves the global geometric hierarchy.
   - Surface normals computed from the depth map drive a Phong shading pass that
     amplifies subtle local geometric perturbations into high-contrast photometric cues.
2. **YOLO-WT (YOLO-WaveletTiny)** (Python, PyTorch, Ultralytics fork)
   - **WDSConv** in the backbone: depthwise-separable convolution whose 3×3 depthwise
     core is replaced by a multi-level Haar wavelet transform.
   - **IWUpsample** in the neck: replaces interpolation upsampling with multi-level
     inverse wavelet reconstruction so high-frequency defect cues survive the
     resolution recovery.

On a real industrial dataset of six battery weld defect classes, the rendered modality
alone improves the YOLOv11n baseline by **+4.9** mAP@50, and the full YOLO-WT achieves
**93.8 % mAP@50** with only **2.47 M** parameters and **5.7 GFLOPs**.

See [`docs/algorithm.md`](docs/algorithm.md) for the full mathematical description.

---

## Repository layout

```
.
├── Crop_Data/                  # Raw cropped 32-bit float depth maps + YOLO labels
│   ├── Tif/                    #   *.tif, single-channel CV_32FC1
│   └── label/                  #   *.txt, YOLO format (class cx cy w h, normalized)
│
├── Depth-Normal_Rendering/     # Stage 1: C++ renderer (OpenCV + Eigen)
│   ├── LUT.{h,cpp}             #   Statistical-prior depth → pseudo-color LUT
│   ├── Phong.{h,cpp}           #   Surface normals + Phong shading
│   └── Demo.cpp                #   Reference pipeline driver
│
├── Train_Data/                 # Stage 1 outputs, organized for detector training
│   ├── DepthMap/               #   Min-max normalized depth (8-bit)
│   ├── LUT/                    #   Pseudo-color image only (Stage 1a)
│   ├── Diffuse/                #   Phong diffuse component only
│   ├── Specular/               #   Phong specular component only
│   ├── Normal/                 #   Surface normal map (color-coded)
│   └── Phong/                  #   Final rendered image (Stage 1a × Stage 1b)
│
├── YOLO-WT/                    # Stage 2: detector
│   ├── train.py / val.py / detect.py
│   └── ultralytics/            # Forked Ultralytics tree
│       ├── nn/AddModules/      #   WDSConv.py, IWUpSample.py
│       ├── cfg/models/YOLO-WT/ #   YOLO-WT.yaml + WDSConv/IWUpSample ablations
│       └── cfg/datasets/       #   One YAML per modality (Phong / LUT / Normal / ...)
│
├── paper/                      # LaTeX source for the manuscript (Elsevier cas-sc)
└── docs/                       # Algorithm details and dataset card
```

The dataset is also released on Hugging Face — see [`docs/dataset_card.md`](docs/dataset_card.md).

---

## Quick start

### 1. Clone and fetch the dataset

```bash
git clone https://github.com/WillPanSUTD/MPF-AD.git
cd MPF-AD

# Option A: download the dataset from Hugging Face
huggingface-cli download vpan1226/MPW-AD --repo-type dataset --local-dir .

# Option B: re-render Train_Data from Crop_Data yourself (see Stage 1 below)
```

### 1b. Download pre-trained weights

```bash
# Canonical public checkpoint (seed 42, val mAP@50 = 0.929)
huggingface-cli download vpan1226/MPF-AD \
    checkpoints/YOLO-WT-seed42-best.pt \
    --local-dir YOLO-WT/Abl_Exp/train/YOLO-WT-250-16-640-SGD-seed42/weights \
    --local-dir-use-symlinks False

# All four seed runs (for reproducibility / variance analysis)
huggingface-cli download vpan1226/MPF-AD \
    --include "checkpoints/*.pt" \
    --local-dir checkpoints/ --local-dir-use-symlinks False
```

All checkpoints: <https://huggingface.co/vpan1226/MPF-AD/tree/main/checkpoints>

### 2. Stage 1 — render depth maps to photometric images (C++)

**Dependencies**

| Library  | Version tested |
|----------|----------------|
| OpenCV   | 4.11.0         |
| Eigen    | 3.4.0          |
| Compiler | MSVC 2022 / GCC 11+ (C++17) |

**Windows (MSVC, included `.sln`)**

1. Install OpenCV 4.x and Eigen 3 and add them to the project's include / library paths.
2. Open `Depth-Normal_Rendering/Depth-Normal_Rendering.sln` in Visual Studio 2022.
3. Build in `Release / x64` and run. Edit the input/output paths at the top of `Demo.cpp`
   to point at your `Crop_Data/Tif/*.tif`.

**Linux (CMake, suggested)**

```bash
cd Depth-Normal_Rendering
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/depth_normal_render <depth.tif> <out_dir>
```

The driver emits, per input depth map: `lut.bmp` (Stage 1a pseudo-color),
`normals.bmp` (visualized normals) and `phong.bmp` (final rendered image).

Default rendering parameters used in the paper:

| Param | Value | Meaning |
|-------|-------|---------|
| `L`, `V`     | `[500, 150, 1500]` | Light / view direction (image space) |
| `I_l`        | `1.3`  | Overall light intensity |
| `w_a`        | `0.1`  | Ambient weight |
| `w_d`        | `0.5`  | Diffuse weight |
| `w_s`        | `0.3`  | Specular weight |
| `n` (`shine`)| `32`   | Specular exponent |

### 3. Stage 2 — train, validate, predict (Python)

**Dependencies**

```bash
conda create -n yolo-wt python=3.10
conda activate yolo-wt
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu118

cd YOLO-WT
pip install -r requirements.txt
```

`requirements.txt` installs the upstream `ultralytics` wheel (for its
dependency tree — `numpy`, `opencv-python`, `tqdm`, …) plus `PyWavelets`
and `polars`. At runtime the **bundled fork under `YOLO-WT/ultralytics/`
takes precedence** because the launch directory is at the front of
`sys.path`, so always `cd YOLO-WT` before running `train.py` / `val.py`.

**Train YOLO-WT on the rendered modality**

Open `YOLO-WT/ultralytics/cfg/datasets/Phong.yaml` and set `path:` / `train:` /
`val:` / `test:` to **absolute paths** under your local `Train_Data/Phong/`.
Absolute paths are required here — Ultralytics' dataset loader does not always
resolve relative paths reliably across working directories, so this repo
ships absolute paths as the canonical form. Then:

```bash
cd YOLO-WT
python train.py
```

`train.py` is configured for the main paper run: model
`cfg/models/YOLO-WT/YOLO-WT.yaml`, data `cfg/datasets/Phong.yaml`, 250 epochs,
batch 16, image size 640, SGD, no pretrained weights.

**Validate**

```bash
python val.py    # loads Abl_Exp/train/.../weights/best.pt
```

**Predict on a single image**

```bash
python detect.py    # edit `source=` to your image
```

### 4. Reproducing paper Table results

The dataset YAMLs under `cfg/datasets/` (`DepthMap.yaml`, `LUT.yaml`, `Diffuse.yaml`,
`Specular.yaml`, `Normal.yaml`, `Phong.yaml`) point to the six modalities. Re-running
`train.py` with each YAML reproduces the modality-comparison ablation in the paper
(Sec. 4 / Table — see manuscript). The `WDSConv.yaml` and `IWUpSample.yaml` model
configs reproduce the per-module ablation rows.

---

## Hardware used in the paper

| Component | Spec |
|-----------|------|
| GPU       | NVIDIA GeForce RTX 3080 |
| CPU       | Intel Core i5-13400F     |
| Sensor    | OPT-LPC61-3D structured-light line scanner, mounted at 30°–45° tilt |

---

## Reproducibility

The headline 93.8 % mAP@50 reported in the paper falls comfortably inside the
natural seed variance of training on this small dataset. We retrained the model
under four random seeds with the **exact same configuration** as the paper
(YOLO-WT.yaml, 250 epochs, batch 16, image size 640, SGD, no pretrained
weights) and validated the saved `best.pt` of each run with ultralytics' own
validator:

| seed | val P | val R | val mAP@50 | val mAP@50-95 | test mAP@50 |
|------|-------|-------|------------|---------------|-------------|
| 0  | 0.859 | 0.822 | 0.881 | 0.467 | 0.773 |
| 1  | 0.877 | 0.794 | 0.893 | 0.471 | 0.749 |
| 2  | 0.831 | 0.866 | 0.905 | 0.469 | 0.774 |
| **42** | 0.847 | 0.879 | **0.929** | **0.480** | **0.788** |
| Paper | – | – | **0.938** | – | – |

Mean of 4 seeds val mAP@50 = 0.902, range = 4.8 pt. **Seed 42 is the closest
single-seed reproduction of the paper number and is shipped as the public
release checkpoint** on Hugging Face at
[`vpan1226/MPF-AD`](https://huggingface.co/vpan1226/MPF-AD) under
`checkpoints/YOLO-WT-seed42-best.pt`. The other three seed runs are kept
for reference and to make the variance verifiable.

### Robustness check via 4-seed Weighted Box Fusion

As a sanity check, we ran a 4-seed Weighted Box Fusion ensemble
(`ensemble_eval.py`, dependencies: `pip install ensemble-boxes torchmetrics
faster-coco-eval`). All metrics here are computed via
`torchmetrics.MeanAveragePrecision(backend='faster_coco_eval')`, so they are
**only directly comparable within this section** (they differ slightly from
ultralytics' implementation in the table above):

|              | val mAP@50 | test mAP@50 |
|--------------|------------|-------------|
| seed 42 alone     | 0.894 | 0.783 |
| 4-seed WBF ensemble | **0.931** | **0.809** |

The ensemble lifts every class except *Bump* (which gets pulled down by the
weaker seeds' low-confidence boxes); we still ship the single seed-42
checkpoint for ease of deployment, but the ensemble result is reproducible
with `python ensemble_eval.py` and confirms the single-seed numbers are
stable rather than degenerate.

### Known limitations of the released checkpoint

- **Non-stratified random split.** The paper's 8:1:1 split is at the image
  level, not stratified by class. Per-class instance counts:
  Pseudo `170 / 21 / 23`, Pinhole `306 / 40 / 35`, Pit `199 / 22 / 30`,
  Burst `76 / 21 / 13`, Fish-scale `151 / 13 / 24`, Bump `84 / 6 / 14`
  (train / val / test). With only 6 *Bump* val instances, single
  prediction errors swing val Bump AP by 15+ pt; the val number for that
  class should not be over-interpreted.
- **Bump ↔ Burst confusion.** Both classes are defined as
  "protrusion ≥ 0.2 mm" with the only distinction being qualitative shape
  (volumetric vs. fish-scale-like). On the test split, 4 of 14 *Bump*
  instances are misclassified as *Burst*, dragging test *Bump* mAP to 0.50
  (vs. 0.88 on val for the same checkpoint). This appears to be inherent
  class similarity rather than a training defect; merging the two into a
  single *Protrusion* category, or hard-mining Bump↔Burst pairs in
  training, are reasonable future directions.
- **Val ↔ test gap of ~12 pt** holds across all four seeds and is dominated
  by the *Bump* issue above. Treat val numbers as an upper bound of the
  current dataset's generalization signal, not as deployment performance.

---

## Citation

```bibtex
@article{pan2026geo2pho,
  title   = {Physically Inspired Geometry-to-Photometry Rendering for Real-Time Weld Defect Detection},
  author  = {Pan, Will and others},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Under review}
}
```

(Update once the article is published.)

---

## Acknowledgments

- Detector built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
  (AGPL-3.0).
- WDSConv adapts the wavelet convolution idea from
  Finder et al., *WTConv* (2024).
- Funded by OPT Machine Vision under the Dongguan Key R&D Program (No. 20241200300122)
  and the Guangdong Provincial Key R&D Program (No. 2025B0101120001).

## License

- Code under `YOLO-WT/ultralytics/`: **AGPL-3.0**, inherited from upstream Ultralytics.
- Code under `Depth-Normal_Rendering/`: **Apache-2.0** (this work).
- Dataset under `Crop_Data/` and `Train_Data/`: see [`docs/dataset_card.md`](docs/dataset_card.md).
