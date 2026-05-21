# MPW-AD: Multi-Photometric Weld Anomaly Detection Benchmark

**Version:** 1.0
**Released with:** MPF-AD (Multi-Photometric Fusion for Anomaly Detection), submitted to Engineering Applications of Artificial Intelligence, 2026.
**License:** CC-BY-4.0
**Contact:** [author email]

## Overview

MPW-AD is the first publicly-available multi-photometric 3D anomaly
detection benchmark targeting continuous welded-surface inspection.
It contains 549 high-resolution depth scans of laser-welded specimens
(640x{512, 526, 551, 640}, mm-scale precision, 0.016 mm/px in X-Y,
mm-scale in Z) with 541 anomalous scans annotated for six industrial
defect classes -- pseudo_soldering, pinhole, pit, burst,
fish_scale_welding, bump -- plus the multi-class "combined" subset.

Each scan is shipped with FIVE deterministic photometric renderings
derived from the organised 3D scan, NOT real RGB photographs:
- **LUT** -- false-colour visualisation of raw depth values
- **Phong** -- single-light Phong shading
- **Diffuse** -- Lambertian diffuse component only
- **Specular** -- Phong specular component only
- **Normal** -- surface-normal-map shading

These photometric renderings let downstream methods exploit the same
multi-light cues that Eyecandies provides for natural objects, but on
true industrial weld geometry.

## Dataset variants

To support multiple existing 3D-AD evaluation pipelines without
ad-hoc re-formatting, MPW-AD ships THREE format variants of the same
underlying 549 scans:

| Variant | Path | Compatible with |
|---|---|---|
| MVTec3D-AD-style | `variants/mvtec3d_style/` | PointAD, M3DM, CFM, EasyNet, Shape-Guided, BTF (organised XYZ + RGB + GT) |
| Eyecandies-style | `variants/eyecandies_style/` | AFRD, MPF-AD's multi-photometric pipelines (5 photometric channels + depth + GT, native resolution) |
| Real3D-AD-style | `variants/real3d_style/` | Reg3D-AD, Group3AD, PO3AD, R3D-AD, 3DKeyAD (point cloud + per-point binary labels) |

## Splits (from MPF-AD's evaluation protocol)

- 714 train_good 256x256 patches (no defect, from clean regions of anomalous source scans)
- 153 val_good 256x256 patches (held out for fusion-block training, unused by parameter-free MPF-AD)
- 153 test_good 256x256 patches
- 541 anomalous test images (native resolution, full source scans)

Total test set: 694 samples (153 normal + 541 anomalous).

## File layout (MVTec3D-AD-style example)

```
variants/mvtec3d_style/weld/
  train/good/{rgb,xyz,gt}/*.{png,tiff,png}
  validation/good/{rgb,xyz,gt}/*
  test/good/{rgb,xyz,gt}/*
  test/{pseudo_soldering,pinhole,pit,burst,fish_scale_welding,bump,combined}/{rgb,xyz,gt,gt_geom}/*
```

`gt/` uses SAM2 masks (primary); `gt_geom/` uses depth-threshold masks (geometric baseline; provided for sensitivity analysis).

## Statistics

See `statistics.json` for per-class counts, image sizes, and split sizes.

## Reproducibility

To regenerate the photometric renderings from raw `.tif` scans, see
`scripts/` and the upstream rendering pipeline at
`F:/dataset/LUT_AD_DataSet/Depth-Normal_Rendering/` (C++).

## Citation

If you use MPW-AD, please cite (BibTeX in `citation.bib`):

```bibtex
@article{anonymous2026mpfad,
  title={MPF-AD: Parameter-Free Multi-Photometric Fusion for Zero-Shot 3D Weld Anomaly Detection},
  author={Anonymous},
  journal={Engineering Applications of Artificial Intelligence},
  year={2026},
  note={Under review}
}
```

## License

CC-BY-4.0 (data, masks, and renderings). The upstream rendering tool
is licensed separately under the Depth-Normal_Rendering/LICENSE.

## Acknowledgements

MPW-AD reuses the source `.tif` depth scans from LUT-AD (Pan et al.),
extended with deterministic photometric renderings and 3D-AD-format
adapters by the MPF-AD authors.
