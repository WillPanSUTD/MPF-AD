"""Build a size-matched MPW-AD test set: every test sample becomes a 256x256 window.

Normal test samples are already 256x256 crops. For each anomalous test sample
we take the 256x256 window centred on the bounding box of its ground-truth
mask (clamped to the image), and apply the same window to every modality
(XYZ, rgb/Phong, the five photometric renderings, depth, gt, gt_geom).
Train/validation splits and normal test samples are linked unchanged.

Usage: python scripts/pointad_plus/make_size_matched_dataset.py <src Dataset_3D> <dst Dataset_3D>
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

WIN = 256
SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])


def junction(src: Path, dst: Path) -> None:
    """Link an unchanged directory into the new tree (junction on Windows, symlink elsewhere)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, capture_output=True)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=True)


def window(gt_path: Path, h: int, w: int) -> tuple[int, int]:
    m = np.asarray(Image.open(gt_path).convert("L")) > 0
    if m.any():
        ys, xs = np.nonzero(m)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    else:
        cy, cx = h // 2, w // 2
    y0 = int(np.clip(cy - WIN // 2, 0, h - WIN))
    x0 = int(np.clip(cx - WIN // 2, 0, w - WIN))
    return y0, x0


def crop_file(src: Path, dst: Path, y0: int, x0: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix in (".tif", ".tiff"):
        a = tifffile.imread(src)
        tifffile.imwrite(dst, np.ascontiguousarray(a[y0:y0 + WIN, x0:x0 + WIN]))
    else:
        Image.open(src).crop((x0, y0, x0 + WIN, y0 + WIN)).save(dst)


log = {}
for variant in ["MVTec3D_Weld", "Eyecandies_Weld"]:
    vs, vd = SRC / variant / "weld", DST / variant / "weld"
    for split in ["train", "validation"]:
        junction(vs / split, vd / split)
    junction(vs / "test" / "good", vd / "test" / "good")
    for cls_dir in sorted((vs / "test").iterdir()):
        if cls_dir.name == "good":
            continue
        gt_dir = cls_dir / "gt"
        for gt_file in sorted(gt_dir.iterdir()):
            stem = gt_file.stem
            # Window is defined once per sample (from the MVTec3D_Weld GT) and reused for both variants.
            key = f"{cls_dir.name}/{stem}"
            if key not in log:
                h, w = np.asarray(Image.open(gt_file)).shape[:2]
                log[key] = {"hw": [h, w], "yx": list(window(gt_file, h, w))}
            y0, x0 = log[key]["yx"]
            for mod_dir in cls_dir.iterdir():
                for f in mod_dir.glob(stem + ".*"):
                    crop_file(f, vd / "test" / cls_dir.name / mod_dir.name / f.name, y0, x0)
    print("done", variant)

(DST / "crop_windows.json").write_text(json.dumps(log, indent=1))
print("samples cropped:", len(log))
