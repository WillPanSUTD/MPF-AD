# MPW-AD: Eyecandies-style variant

Multi-photometric variant of MPW-AD. Each sample carries all five
deterministic photometric renderings plus depth and pixel mask, at
native resolution. This is the format consumed by MPF-AD's multi-
photometric pipelines.

## Layout

```
data/weld/
  train/good/{lut, phong, diffuse, specular, normal, depth, gt}/<id>.*
  validation/good/  (same)
  test/{good, pseudo_soldering, pinhole, pit, burst,
        fish_scale_welding, bump, combined}/
        {lut, phong, diffuse, specular, normal, depth, gt, gt_geom}/<id>.*
```

- `lut/`, `phong/`, `diffuse/`, `specular/`, `normal/` are PNG (8-bit
  sRGB; engineered photometric channels).
- `depth/` is single-channel float32 `.tiff` (raw depth, not the XYZ tiff).
- `gt/` and `gt_geom/` are 8-bit binary masks (255 = defect).
- All five photometric renderings share the same H×W per sample, and
  are deterministic functions of the organised XYZ — they are NOT
  separate physical light-direction captures.

## Loader hint

For methods written against Eyecandies' six-light convention, treat
{lut, phong, diffuse, specular, normal} as five of the six light
directions; the sixth slot is unfilled.
