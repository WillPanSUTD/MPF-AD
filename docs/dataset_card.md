---
license: cc-by-nc-4.0
task_categories:
- object-detection
language:
- en
tags:
- weld-defect
- industrial-inspection
- depth-map
- structured-light
- lithium-ion-battery
- yolo
size_categories:
- n<1K
pretty_name: Battery-Cap Weld Defect Depth Dataset (LUT-AD)
---

# Battery-Cap Weld Defect Depth Dataset (LUT-AD)

A real-industrial dataset of laser-welded battery top covers, captured as
high-resolution **structured-light depth maps** and annotated with bounding
boxes for **six defect classes**. The dataset accompanies the paper
*"Physically Inspired Geometry-to-Photometry Rendering for Real-Time Weld
Defect Detection"* (accepted at *Engineering Applications of Artificial
Intelligence*, EAAI, 2026). Code and documentation:
<https://github.com/WillPanSUTD/MPF-AD>.

[ [中文版](dataset_card.zh-CN.md) ]

---

## What's inside

The release contains two layers of data — the raw cropped depth maps you can
re-render yourself, and the pre-rendered images we used to train every
modality reported in the paper.

### `Crop_Data/` — raw depth + labels

| Path                | Format | Content |
|---------------------|--------|---------|
| `Crop_Data/Tif/`    | `.tif`, single-channel CV_32FC1 | Cropped 640×640 depth map per defect instance. Sentinel value `< -1e30` marks invalid pixels. |
| `Crop_Data/label/`  | `.txt`, YOLO format             | One row per defect: `class_id cx cy w h`, all four geometry values normalized to `[0, 1]`. |

This is the canonical input to the renderer in `Depth-Normal_Rendering/` —
running Stage 1 on `Crop_Data/Tif/*.tif` reproduces every modality below.

### `Train_Data/` — pre-rendered images, organized for YOLO training

Six modalities, each split 8:1:1 into train / val / test, with identical YOLO
labels copied across modalities (so you can swap modalities without
re-annotating):

| Modality   | Description |
|------------|-------------|
| `DepthMap/`  | Min-max normalized 8-bit grayscale depth — the trivial 2-D baseline. |
| `LUT/`       | Stage 1a output: pseudo-color image from the statistical-prior LUT. |
| `Normal/`    | Color-coded surface normal map (each component mapped from `[-1, 1]` to `[0, 255]`). |
| `Diffuse/`   | Phong diffuse component only ($I_d$ modulation of LUT). |
| `Specular/`  | Phong specular component only ($I_s$ modulation of LUT). |
| `Phong/`     | The full proposed rendered image, $C' = (I_a + I_d + I_s) \cdot C$. **This is the modality we report 93.8 % mAP@50 on.** |

Each modality's directory looks like:

```
Train_Data/<Modality>/
├── classes.txt
├── train/
│   ├── images/   # *.png, 640×640
│   └── labels/   # *.txt, YOLO format
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

---

## Class definitions

Class ids match `classes.txt` and `cfg/datasets/Phong.yaml` in the code repo.

| id | Class                | Definition |
|----|----------------------|------------|
| 0  | Pseudo soldering     | Intersecting planes form a near right angle and the weld seam plane is flat; weld height ≤ 0.1 mm. |
| 1  | Pinhole              | Small pinhole-like depressions along weld lines, depth ≥ 0.2 mm and length ≥ 0.2 mm. |
| 2  | Pit                  | Concave area along weld lines, depression depth ≥ 0.2 mm and length ≥ 0.2 mm. |
| 3  | Burst                | Weld seam with obvious substantial protrusion ≥ 0.2 mm, length ≥ 0.2 mm. |
| 4  | Fish-scale welding   | Raised banded area with longitudinal fish-scale-like protrusions ≥ 0.2 mm, length ≥ 0.2 mm. |
| 5  | Bump                 | Protrusion along weld lines ≥ 0.2 mm and length ≥ 0.2 mm. |

> **Note on class 0.** Earlier internal versions of `classes.txt` and the
> dataset YAMLs spelled class 0 as `Inveracious solding` (a typo). The
> public release standardises on `Pseudo soldering`; if you encounter the
> old string in legacy checkpoints, it refers to the same class.

### Per-class instance count (whole dataset)

| Class id | Class | Instances |
|---------:|-------|-----------|
| 0 | Pseudo soldering   | 214 |
| 1 | Pinhole            | 381 |
| 2 | Pit                | 251 |
| 3 | Burst              | 110 |
| 4 | Fish-scale welding | 188 |
| 5 | Bump               | 104 |
| **Total** |               | **1248** |

There is a non-trivial class imbalance, especially Pinhole vs. Bump, which
reflects the natural distribution observed on the production line.

### Split sizes (images)

| Split | Images |
|-------|-------:|
| train |    429 |
| val   |     52 |
| test  |     60 |
| **Total** | **541** |

---

## Acquisition setup

- **Sensor**: OPT-LPC61-3D structured-light line scanner, calibrated, mounted
  with a 30°–45° tilt above the part to capture the weld profile.
- **Stage**: precision XY stage that translates the battery top cover under
  the scanner.
- **Origin**: every defect sample is taken from real production-line
  inspection data — there are no synthetic or artificially induced defects.
- **Curation**: each scan was manually verified for sensor noise and dropout
  before inclusion.

---

## Usage

### Loading with the Hugging Face datasets library

```python
from datasets import load_dataset

ds = load_dataset("<hf-handle>/lut-ad-weld-defect", split="train")
sample = ds[0]
# sample = {"image": <PIL.Image>, "label_path": <path-to-yolo-txt>, ...}
```

### Direct download

```bash
huggingface-cli download <hf-handle>/lut-ad-weld-defect \
    --repo-type dataset --local-dir ./LUT_AD_DataSet
```

### Re-rendering Train_Data from Crop_Data

The exact rendering parameters used to produce `Train_Data/Phong/` are
documented in [`docs/algorithm.md`](algorithm.md). Build the C++ tool in
`Depth-Normal_Rendering/` and run it on `Crop_Data/Tif/*.tif` to reproduce
every modality.

---

## License

Released under **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)**. Commercial use requires a separate written agreement with
the authors and the funding sponsor.

When using this dataset, please cite the paper (see below) and retain the
attribution to OPT Machine Vision in any public artifact derived from it.

## Citation

```bibtex
@article{cao2026geo2pho,
  title   = {Physically Inspired Geometry-to-Photometry Rendering for Real-Time Weld Defect Detection},
  author  = {Cao, Ling and Qiu, Jiajun and Zhang, Yunzhi and Feng, Daquan and Pan, Wei},
  journal = {Engineering Applications of Artificial Intelligence},
  year    = {2026},
  note    = {Accepted, in press. Preprint: SSRN, doi:10.2139/ssrn.6946138}
}
```

## Acknowledgments

Data collection and curation were funded by OPT Machine Vision under the
Dongguan Key Research and Development Program (No. 20241200300122) and the
Guangdong Provincial Key Research and Development Program
(No. 2025B0101120001).
