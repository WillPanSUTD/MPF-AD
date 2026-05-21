"""Recompute statistics.json from on-disk file counts.

Walks the three variants, counts per-class samples and image sizes,
and writes a fresh statistics.json. Use this after adding new scans
or re-rendering.

Usage:
    python scripts/compute_statistics.py [--out statistics.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]

DEFECTS = ("pseudo_soldering", "pinhole", "pit", "burst",
           "fish_scale_welding", "bump", "combined")


def count_files(p: Path, suffixes=(".png", ".tiff", ".pcd", ".txt")) -> int:
    if not p.is_dir():
        return 0
    return sum(1 for x in p.iterdir() if x.is_file() and x.suffix in suffixes)


def compute() -> dict:
    mvtec_root = HERE / "variants" / "mvtec3d_style" / "data" / "weld"
    real_root = HERE / "variants" / "real3d_style" / "data" / "weld"

    splits = {
        "train_good": count_files(mvtec_root / "train" / "good" / "rgb"),
        "val_good": count_files(mvtec_root / "validation" / "good" / "rgb"),
        "test_good": count_files(mvtec_root / "test" / "good" / "rgb"),
    }
    per_class = {}
    test_anom_total = 0
    for d in DEFECTS:
        n = count_files(mvtec_root / "test" / d / "rgb")
        per_class[d] = {"single_class_test_n": n}
        test_anom_total += n
    splits["test_anomalous"] = test_anom_total

    real3d_counts = {
        "train_templates": count_files(real_root / "train"),
        "test_pcd": count_files(real_root / "test"),
        "ground_truth": count_files(real_root / "ground_truth"),
    }

    return {
        "name": "MPW-AD",
        "long_name": "Multi-Photometric Weld Anomaly Detection",
        "version": "1.0.0",
        "license": "CC-BY-4.0",
        "splits": splits,
        "per_class": per_class,
        "real3d_style_counts": real3d_counts,
        "photometric_modalities": ["lut", "phong", "diffuse", "specular", "normal"],
        "pixel_size_mm": {"xy": 0.016, "z": "mm-scale"},
        "matched_image_protocol_total": splits["test_good"] + splits["test_anomalous"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "statistics.json")
    args = ap.parse_args()
    payload = compute()
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[compute_statistics] wrote {args.out}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
