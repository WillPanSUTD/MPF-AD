# 3D Anomaly-Detection Extension to LUT-AD

**Date:** 2026-05-16
**Status:** Draft — pending user review
**Scope:** Convert the existing 2D bbox-detection dataset (549 `.tif` depth maps + YOLO labels under `Crop_Data/`) into three sibling 3D-AD–style datasets, without re-acquiring any data.

---

## 1. Goals

Produce three parallel 3D-anomaly-detection benchmarks that downstream methods can load with their existing `mvtec_3d_anomaly_detection/`, `eyecandies/`, or `real3d_ad/` data loaders. Each sibling output must:

- Mirror its source format's directory layout and file naming exactly enough that existing baseline code can run with a single `--data-root` change.
- Use only `Crop_Data/Tif/*.tif` and `Crop_Data/label/*.txt` as the source of truth (plus already-rendered `Train_Data/{LUT,Phong,Diffuse,Specular,Normal,DepthMap}/`).
- Be reproducible from a single config (`cfg/dataset_3d.yaml`) and a one-shot pipeline (`scripts/dataset_3d/*.py`).

Non-goals:
- No new data acquisition.
- No inpainting-based pseudo-good synthesis.
- No truly multi-light re-rendering for Eyecandies (we reuse existing renders and document the divergence).
- No multi-label folder structure (single-category, folder-based defect-type assignment with a `combined/` overflow).

---

## 2. Source data summary

| Item | Value |
|---|---|
| Source `.tif` count | 549 (640×{512,526,551,640}, float32, mm) |
| Invalid-pixel sentinel | `-FLT_MAX` ≈ `-3.4028e+38` |
| X/Y pixel size | 0.016 mm/px (user-supplied) |
| Z scale | 1.0 — values already in mm (user-confirmed) |
| Source bbox labels | 549 `.txt` (YOLO format) |
| Already-good images (empty label file) | 8 |
| Single-class anomalous images | 270 |
| Multi-class anomalous images | 271 |
| Per-class image appearances | pseudo_soldering:212, pinhole:217, pit:228, burst:100, fish_scale_welding:174, bump:97 |
| Defect classes (index 0–5) | `pseudo_soldering, pinhole, pit, burst, fish_scale_welding, bump` |

---

## 3. Shared preprocessing pipeline

One pass, cached, consumed by all three output builders.

### 3.1 Good-sample harvesting — `scripts/dataset_3d/harvest_good.py`
- For all 549 `.tif`s, slide a `patch_size`×`patch_size` window with `stride`. For the 541 anomalous images, keep windows whose IoU with every defect bbox is 0. For the 8 already-good images (empty label file), keep all windows.
- Skip windows where the fraction of invalid (`-FLT_MAX`) pixels exceeds `max_invalid_fraction` (default 0.05).
- Cap ≤`max_patches_per_source` patches per source image; if more candidates remain, sample uniformly at random with the run seed.
- Default config: `patch_size = 256`, `stride = 128`, `max_patches_per_source = 4`, `max_invalid_fraction = 0.05`.
- Output is uniform-size 256×256 patches across native-good and cropped-good — required by the MVTec3D-style fixed-size loaders.
- Expected yield: ≈1500–2500 patches at defaults.
- Shuffle (seed 42) → 70/15/15 split → `train/good`, `validation/good`, `test/good`.
- Output cache: `.cache/good_patches/{train,val,test_good}/<source_stem>_<u>_<v>.tif`.

### 3.2 Dual mask generation
Two cached masks per source image, both written under `.cache/masks/`:

**SAM2 (primary) — `scripts/dataset_3d/segment_sam2.py`**
- Backbone: SAM2-Hiera-Large (downloaded once to `.cache/sam2/`).
- Prompt: each defect bbox from `Crop_Data/label/<id>.txt` independently; image input is the Phong render for that ID. The script walks `Train_Data/Phong/{train,val,test}/images/` to resolve `<id>.png` (a given source ID lives in exactly one of the three splits).
- Output per source image:
  - `masks_sam2/<id>_full.png` — union of all defect masks.
  - `masks_sam2/<id>_cls{0..5}.png` — per-class union (only emitted for classes present).
- 8-bit single-channel PNG, 255 = defect, 0 = good.

**Depth-threshold (baseline) — `scripts/dataset_3d/segment_depth_threshold.py`**
- Inside each bbox, compute local median (5×5 dilated bbox neighborhood) of valid depth, then mark pixels where `|depth − median| > k · σ` (`k = 2.5` default, σ = MAD-based robust std).
- Invalid (`-FLT_MAX`) pixels excluded.
- Output mirrors SAM2 structure under `masks_depth/`.

### 3.3 Organized XYZ unprojection — `scripts/dataset_3d/unproject_xyz.py`
For each .tif, produce an `H×W×3` float32 .tiff matching the MVTec 3D-AD convention:
- `X[u,v] = (u - W/2) · 0.016` mm
- `Y[u,v] = (v - H/2) · 0.016` mm
- `Z[u,v] = depth_tif[u,v]` mm (already in mm)
- Invalid pixels (`depth ≤ -1e30`): write `(0, 0, 0)` per MVTec 3D-AD convention.
- Output cache: `.cache/xyz/<id>.tiff`.

---

## 4. Output layouts

### 4.1 `Dataset_3D/MVTec3D_Weld/`
```
MVTec3D_Weld/weld/
  train/good/
    rgb/000001.png            # Phong render
    xyz/000001.tiff           # HxWx3 organized point cloud
    gt/000001.png             # all-zero (good)
  validation/good/
    rgb/, xyz/, gt/
  test/
    good/{rgb,xyz,gt}/        # held-out good
    pseudo_soldering/{rgb,xyz,gt,gt_geom}/
    pinhole/{rgb,xyz,gt,gt_geom}/
    pit/{rgb,xyz,gt,gt_geom}/
    burst/{rgb,xyz,gt,gt_geom}/
    fish_scale_welding/{rgb,xyz,gt,gt_geom}/
    bump/{rgb,xyz,gt,gt_geom}/
    combined/{rgb,xyz,gt,gt_geom}/    # 271 multi-class images
```
- File IDs are zero-padded sequential per subfolder (`000001.png`, …).
- `gt` = SAM2 mask; `gt_geom` = depth-threshold mask (test only).
- Built by `scripts/dataset_3d/build_mvtec3d.py` from the three caches above.

### 4.2 `Dataset_3D/Eyecandies_Weld/`
Same skeleton as MVTec3D_Weld, but each sample exposes 6 modality folders mirroring our existing renders:
```
Eyecandies_Weld/weld/
  train/good/
    lut/, phong/, diffuse/, specular/, normal/, depth/, gt/
  validation/good/  (same)
  test/
    good/, pseudo_soldering/, pinhole/, pit/, burst/, fish_scale_welding/, bump/, combined/
      lut/, phong/, diffuse/, specular/, normal/, depth/, gt/, gt_geom/
```
- `lut/`, `phong/`, `diffuse/`, `specular/`, `normal/` are PNG; `depth/` is single-channel float32 .tiff (raw depth, NOT XYZ — `xyz/` belongs to MVTec3D format).
- Sample IDs are aligned across modality folders (`000001.png` in `phong/` and `000001.tiff` in `depth/` come from the same source).
- Built by `scripts/dataset_3d/build_eyecandies.py`, which symlinks (or copies, on Windows) from `Train_Data/<modality>/` and the mask cache.
- Documented as "Eyecandies-inspired": we do not re-render under six light directions; we ship the five existing photometric renders (LUT, Phong, Diffuse, Specular, Normal) plus depth.

### 4.3 `Dataset_3D/Real3D_Weld/`
```
Real3D_Weld/weld/
  train/
    tmpl_0.pcd, tmpl_1.pcd, tmpl_2.pcd, tmpl_3.pcd     # 4 good prototype patches
  test/
    0001.pcd, 0002.pcd, ..., 0541.pcd                  # all anomalous source .tifs
  ground_truth/
    0001.txt, 0002.txt, ..., 0541.txt                  # per-point binary labels
```
- Templates = the first 4 patches in `train/good` after the deterministic seed-42 shuffle (i.e., a reproducible subset of the same good-patch pool the other formats use).
- Each test `.pcd` = unprojected point cloud of one anomalous .tif (`X`, `Y`, `Z` in mm, invalid pixels dropped, not zeroed).
- Per-point label `i` = `1` if pixel `(u_i, v_i)` is inside any defect bbox **and** inside that defect's SAM2 mask; else `0`.
- Built by `scripts/dataset_3d/build_real3d.py`.
- Caveat noted in dataset card: weld geometry is per-instance unique. Templates are *prototype patches*, not full-object instances. Rigid-body-registration baselines (PointNet++-based AD) will require adaptation.

---

## 5. Repo additions

```
cfg/dataset_3d.yaml                       # scale factors, patch size, stride, k, SAM2 path
scripts/dataset_3d/
  harvest_good.py
  segment_sam2.py
  segment_depth_threshold.py
  unproject_xyz.py
  build_mvtec3d.py
  build_eyecandies.py
  build_real3d.py
  verify_dataset.py                       # invariant checks across the three outputs
docs/dataset_3d.md                        # English dataset documentation
docs/dataset_3d.zh-CN.md                  # Chinese dataset documentation
Dataset_3D/                               # generated, gitignored
.cache/                                   # generated, gitignored
```

`cfg/dataset_3d.yaml` initial content:
```yaml
source:
  tif_dir: F:/dataset/LUT_AD_DataSet/Crop_Data/Tif
  label_dir: F:/dataset/LUT_AD_DataSet/Crop_Data/label
  # Each modality root contains train/, val/, test/ subdirs with images/ inside.
  # Scripts resolve `<id>.png` by walking all three splits.
  modality_roots:
    lut: F:/dataset/LUT_AD_DataSet/Train_Data/LUT
    phong: F:/dataset/LUT_AD_DataSet/Train_Data/Phong
    diffuse: F:/dataset/LUT_AD_DataSet/Train_Data/Diffuse
    specular: F:/dataset/LUT_AD_DataSet/Train_Data/Specular
    normal: F:/dataset/LUT_AD_DataSet/Train_Data/Normal
    depth: F:/dataset/LUT_AD_DataSet/Train_Data/DepthMap

camera:
  pixel_size_mm_x: 0.016
  pixel_size_mm_y: 0.016
  depth_scale_mm: 1.0
  invalid_threshold: -1.0e30                # depth values <= this are treated as invalid

good_harvest:
  patch_size: 256
  stride: 128
  max_patches_per_source: 4
  max_invalid_fraction: 0.05
  split_ratios: [0.70, 0.15, 0.15]          # train/val/test_good
  shuffle_seed: 42

sam2:
  checkpoint: .cache/sam2/sam2_hiera_large.pt
  config: sam2_hiera_l.yaml
  device: cuda:0

depth_threshold:
  k: 2.5                                    # |x - median| > k * sigma
  neighborhood: 5                           # bbox dilation in px before sigma estimate

real3d:
  n_templates: 4
  template_selection: first_after_shuffle
  pcd_format: ascii                          # 'ascii' | 'binary'

# Index = class id in Crop_Data/label/*.txt and YOLO labels.
classes:
  0: pseudo_soldering
  1: pinhole
  2: pit
  3: burst
  4: fish_scale_welding
  5: bump
```

---

## 6. Build order and idempotency

```
1. harvest_good.py            -> .cache/good_patches/
2. segment_sam2.py            -> .cache/masks/sam2/         (GPU pass — order of an hour on a 12GB consumer GPU; report actual runtime in dataset card after the build)
3. segment_depth_threshold.py -> .cache/masks/depth/
4. unproject_xyz.py           -> .cache/xyz/
5. build_mvtec3d.py           -> Dataset_3D/MVTec3D_Weld/
6. build_eyecandies.py        -> Dataset_3D/Eyecandies_Weld/
7. build_real3d.py            -> Dataset_3D/Real3D_Weld/
8. verify_dataset.py          -> prints per-format counts, fails on invariant violation
```

Each step checks for its output cache and skips already-completed work. Caches are content-addressed by source filename (no implicit invalidation on config changes — use `--force` to rebuild).

---

## 7. Verification invariants (`verify_dataset.py`)

For each of the three outputs:
- Every `rgb/`/`xyz/`/`<modality>/` directory has exactly one file per sample ID in that subfolder.
- Every `gt/` mask matches its source modality's H×W.
- For any non-good subfolder, at least one pixel in `gt/<id>.png` is 255.
- For `train/good`, `validation/good`, `test/good`: every `gt/` mask is all zeros.
- XYZ tiffs are float32, 3-channel, with no NaN.
- Real3D `ground_truth/<id>.txt` line count matches `.pcd` point count.
- Per-class test counts logged for the dataset card.

---

## 8. Documentation

- `docs/dataset_3d.md` / `docs/dataset_3d.zh-CN.md`: describe layout, generation procedure, scale factors, divergences from each source format, known caveats (Real3D template fit, "combined" subfolder origin, SAM2-vs-depth mask comparison).
- Update top-level `README.md` / `README.zh-CN.md` with a "3D-AD variants" section pointing to `docs/dataset_3d.md`, and update the existing HuggingFace dataset card to mention the three new variants alongside the bbox-detection variant.
- Append a usage example for each format in the dataset card.

---

## 9. Open questions (none blocking)

- SAM2 checkpoint: confirm OK to download `sam2_hiera_large.pt` (~900 MB) from Meta's public release. *Assumption: yes.*
- GPU availability for the ~1h SAM2 pass: assumed local 4070 is free when build is run.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| SAM2 produces poor masks on welds (out-of-distribution photometry) | Ship dual masks; depth-threshold acts as fallback baseline. Spot-check 20 random masks visually after the SAM2 pass; if quality is unacceptable, fall back to bbox rectangles + explicit caveat in dataset card. |
| Cropped good patches leak anomalous boundary context | `max_patches_per_source = 4` + IoU=0 check; patches with even partial bbox overlap are dropped. |
| Real3D-AD baselines assume canonical object templates | Documented limitation. Real3D output is positioned as point-cloud format reuse, not as a faithful Real3D-AD-style benchmark. |
| Per-class test counts unbalanced (`bump` and `burst` underrepresented at 97/100 images) | Inherited from source; reported in dataset card. No rebalancing. |

---

## 11. Out of scope (deferred)

- True multi-light Eyecandies-style re-rendering (would require re-running `Depth-Normal_Rendering/` with N light positions).
- Synthetic anomaly augmentation (`SimpleNet`-style).
- Volumetric (mesh / TSDF) ground truth.
- Cross-validation splits beyond the 70/15/15 good-patch split.
