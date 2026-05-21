"""Multi-photometric welds adapter for PointAD+.

Walks Dataset_3D/Eyecandies_Weld/weld/{train,validation,test}/<defect>/<modality>/
and emits a single all_meta.json that lists, per sample, the paths to all 5
photometric modalities (lut, phong, diffuse, specular, normal). The
`multi_photo` field is new — it does not exist in stock PointAD manifests.
The downstream test runner consumes this field instead of `d2_img_path`.

Usage:
    python -m src.pointad_plus.multi_photo_welds_adapter \
        --source Dataset_3D/Eyecandies_Weld/weld \
        --out external/datasets/welds_pointad_mp/weld/all_meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

DEFECT_CLASSES = (
    "pseudo_soldering", "pinhole", "pit", "burst",
    "fish_scale_welding", "bump", "combined",
)
PHOTO_MODALITIES = ("lut", "phong", "diffuse", "specular", "normal")


def _samples_in_subfolder(weld_root: Path, split: str, defect: str, anomaly: bool) -> List[dict]:
    sub = weld_root / split / defect
    phong_dir = sub / "phong"
    if not phong_dir.is_dir():
        return []
    out: List[dict] = []
    for png in sorted(phong_dir.glob("*.png")):
        stem = png.stem
        sample = {
            "id": f"{split}_{defect}_{stem}",
            "multi_photo": {m: str(sub / m / f"{stem}.png") for m in PHOTO_MODALITIES},
            "gt": str(sub / "gt" / f"{stem}.png") if anomaly else None,
            "cls_name": "weld",
            "specie_name": defect,
            "anomaly": anomaly,
        }
        # sanity: each photometric path must exist
        if not all(Path(p).exists() for p in sample["multi_photo"].values()):
            continue
        out.append(sample)
    return out


def emit_manifest(source_root: Path, out_path: Path) -> None:
    train = _samples_in_subfolder(source_root, "train", "good", anomaly=False)
    # fold validation/good into train for fusion-block training (more data)
    train += _samples_in_subfolder(source_root, "validation", "good", anomaly=False)
    test = _samples_in_subfolder(source_root, "test", "good", anomaly=False)
    for dc in DEFECT_CLASSES:
        test += _samples_in_subfolder(source_root, "test", dc, anomaly=True)

    payload = {"train": {"weld": train}, "test": {"weld": test}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    emit_manifest(args.source, args.out)
    print(f"[multi_photo_welds_adapter] wrote {args.out}")


if __name__ == "__main__":
    _main()
