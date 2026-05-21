# MPW-AD Datasheet

Following the Datasheets for Datasets template (Gebru et al., 2021).

## 1. Motivation

**For what purpose was the dataset created?**
MPW-AD targets zero-shot 3D anomaly detection on continuous welded
surfaces, an industrial setting that existing object-centric 3D-AD
benchmarks (MVTec 3D-AD, Eyecandies, Real3D-AD) do not address.
Welds are per-instance geometrically unique, lack canonical templates,
and exhibit defect modes (pseudo soldering, fish-scale welding) whose
signature is photometric rather than purely geometric. MPW-AD is the
first publicly-available multi-photometric weld AD benchmark.

## 2. Composition

**What do the instances represent?** Each instance is a 3D scan of a
laser-welded specimen. There are 549 raw `.tif` depth maps (640 px
along the weld axis, 512–640 px across) at 0.016 mm/px X–Y resolution
and mm-scale Z. 541 scans contain at least one of six defect classes
(pseudo_soldering, pinhole, pit, burst, fish_scale_welding, bump);
8 scans are defect-free; 271 anomalous scans contain ≥ 2 defect classes
("combined" subset). Per-pixel anomaly masks are provided.

**Annotation modalities.** Each scan ships with five deterministic
photometric renderings (LUT, Phong, Diffuse, Specular, Normal) derived
from the organised XYZ point cloud — they are engineered shading outputs,
not real RGB photographs. Pixel-level anomaly masks were generated with
SAM 2 prompted by manual bounding boxes; an alternative depth-threshold
mask is provided as a geometric baseline.

## 3. Collection process

**How was the data acquired?** Raw depth scans were captured on a
production laser-welding line using a structured-light line-scanner at
0.016 mm/px lateral resolution. Acquisition conditions (camera angle,
exposure, line speed) were held constant across the 549 scans.
The five photometric channels are NOT collected separately; they are
rendered offline from the same XYZ tensor using a deterministic Phong
shader (see `Depth-Normal_Rendering/`).

## 4. Preprocessing / labeling

Pixel-level masks were generated via two parallel paths: (a) SAM 2
prompted by manually-drawn bounding boxes around each defect — these
are the primary `gt/` masks used for evaluation; (b) depth-threshold
masks (`gt_geom/`) derived from a per-region MAD-based outlier test,
provided as a fully-automatic geometric baseline. Train/val/test_good
patches are produced by sliding a 256×256 window over defect-free
regions of anomalous scans with IoU=0 against every defect bbox and
≤5 % invalid-pixel fraction, then a seed-42 70/15/15 split.

## 5. Uses

MPW-AD is designed for zero-shot 3D anomaly detection benchmarking
on continuous-surface industrial inspection. It can also be used for:
- multi-photometric fusion research (multi-light AD beyond Eyecandies);
- per-instance unique-geometry AD (no canonical template required);
- weld-quality classification, defect segmentation, and surface
  reconstruction studies;
- domain-adaptation studies that move from object-centric AD to
  industrial inspection.

## 6. Distribution

MPW-AD v1.0 is distributed as a single archive containing the three
format variants plus README, DATASHEET, statistics, citation, and
verification scripts. License: CC-BY-4.0. Mirror sites and DOI will be
listed in the project repository once accepted for publication.

## 7. Maintenance

Maintained by the MPF-AD authors. Bug reports, additional weld
samples, and additional defect classes are welcome via the project
repository. A v1.1 release with the 8 native-normal scans re-rendered
in all five photometric channels is planned once the upstream C++
rendering pipeline is finalised.
