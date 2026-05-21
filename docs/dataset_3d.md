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
- **Class balance:** Inherited from the source set. Total per-class image
  *appearances* (counting an image once per defect class it contains):
  `pseudo_soldering: 212, pinhole: 217, pit: 228, burst: 100, fish_scale_welding: 174, bump: 97`.
  Of the 541 anomalous source images, 270 are single-class and 271 are
  multi-class. The single-class images split across `test/<class>/`
  folders as actually produced by the build pipeline:
  `pseudo_soldering: 206, pinhole: 11, pit: 10, burst: 3, fish_scale_welding: 21, bump: 19`,
  and the 271 multi-class images all live under `test/combined/`. **Most
  defect-class images are multi-class**, so the per-`test/<class>/`
  counts are heavily skewed toward `combined/` and `pseudo_soldering/`.
- **Native-good sources skipped:** 8 source `.tif`s have empty label
  files but no rendered Phong/LUT/Diffuse/Specular/Normal modalities
  (renders are only produced for the 541 anomalous images). These 8
  sources are dropped at harvest time. The good-patch pool comes from
  the 541 anomalous images only.
- **Combined subfolder:** 271 source images contain ≥2 defect classes
  and live under `test/combined/`. The single-class folders
  (`pseudo_soldering/`, etc.) hold only images whose defect set is a
  single class.

## Reproducibility

Every script is deterministic given `cfg/dataset_3d.yaml`. The single
source of randomness is `good_harvest.shuffle_seed = 42`, which controls
the 70/15/15 split, the per-source random subsample, and the Real3D
template selection.
