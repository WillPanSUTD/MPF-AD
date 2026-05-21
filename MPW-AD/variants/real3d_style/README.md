# MPW-AD: Real3D-AD-style variant

Point-cloud variant of MPW-AD, matching the Real3D-AD layout so
point-only methods (Reg3D-AD, Group3AD, PO3AD, R3D-AD, 3DKeyAD,
Shape-Guided, BTF) can use MPW-AD without re-formatting.

## Layout

```
data/weld/
  train/tmpl_{0..3}.pcd       — 4 prototype "good" templates
  test/{0001..0541}.pcd       — 541 anomalous test point clouds
  ground_truth/{0001..0541}.txt — per-point binary labels (0=good, 1=defect)
```

## Caveats

- Welds are per-instance unique. The 4 templates are *prototype patches*
  (256×256 clean crops harvested from defect-free regions and unprojected
  to PCD), NOT canonical full-object instances. Rigid-body-registration
  Real3D-AD baselines (Reg3D-AD-style) will need to adapt to this.
- Test PCDs are dropped at invalid pixels (depth ≤ -1e30), so per-point
  counts vary per test image.
- Labels are derived from the SAM-2 mask intersected with the defect
  bounding box for that defect class.

## Loader hint

Use the Real3D-AD loader with `dataset_root = variants/real3d_style/data/`
and `category = weld`.
