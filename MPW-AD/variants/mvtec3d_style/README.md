# MPW-AD: MVTec 3D-AD-style variant

This variant ships the same 549 weld scans in MVTec 3D-AD's directory
convention, so existing PointAD / M3DM / CFM / BTF / EasyNet /
Shape-Guided pipelines can use MPW-AD with no code change.

## Layout

```
data/weld/
  train/good/{rgb,xyz,gt}/<000001..000714>.{png,tiff,png}
  validation/good/{rgb,xyz,gt}/<000001..000153>.*
  test/good/{rgb,xyz,gt}/<000001..000153>.*
  test/<defect>/{rgb,xyz,gt,gt_geom}/<000001..>.*
    defect ∈ {pseudo_soldering, pinhole, pit, burst,
              fish_scale_welding, bump, combined}
```

- `rgb/` is the **Phong** rendering (single-channel photometric path).
- `xyz/` is the float32 H×W×3 organised point cloud (`X = (u-W/2)·0.016 mm`,
  `Y = (v-H/2)·0.016 mm`, `Z =` raw depth in mm; invalid pixels = `(0,0,0)`).
- `gt/` is the SAM-2 mask (used for evaluation).
- `gt_geom/` is the depth-threshold mask (provided for sensitivity analysis).

Multi-photometric variants of `rgb/` (LUT / Diffuse / Specular / Normal)
live under the **Eyecandies-style variant** (see `../eyecandies_style/`).

## Loader hint

Set the dataset root to `variants/mvtec3d_style/data/` and use the
standard MVTec 3D-AD loader; the only category name is `weld`.
