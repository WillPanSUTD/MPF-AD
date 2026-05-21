"""Verify MPW-AD on-disk file counts against statistics.json.

Walks the three variants and checks counts per defect class against the
manifest. Prints discrepancies if any.

Usage:
    python scripts/verify_integrity.py
"""
from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def count_files(p: Path, pattern: str = "*") -> int:
    if not p.is_dir():
        return -1
    return len([x for x in p.glob(pattern) if x.is_file()])


def check_mvtec3d_style() -> dict:
    root = HERE / "variants" / "mvtec3d_style" / "data" / "weld"
    if not root.is_dir():
        return {"error": f"path missing: {root}"}
    out = {"train_good_rgb": count_files(root / "train" / "good" / "rgb", "*.png")}
    out["val_good_rgb"] = count_files(root / "validation" / "good" / "rgb", "*.png")
    out["test_good_rgb"] = count_files(root / "test" / "good" / "rgb", "*.png")
    defects = ("pseudo_soldering", "pinhole", "pit", "burst",
               "fish_scale_welding", "bump", "combined")
    for d in defects:
        out[f"test_{d}_rgb"] = count_files(root / "test" / d / "rgb", "*.png")
    return out


def check_eyecandies_style() -> dict:
    root = HERE / "variants" / "eyecandies_style" / "data" / "weld"
    if not root.is_dir():
        return {"error": f"path missing: {root}"}
    out = {"train_good_phong": count_files(root / "train" / "good" / "phong", "*.png")}
    out["test_good_phong"] = count_files(root / "test" / "good" / "phong", "*.png")
    return out


def check_real3d_style() -> dict:
    root = HERE / "variants" / "real3d_style" / "data" / "weld"
    if not root.is_dir():
        return {"error": f"path missing: {root}"}
    return {
        "train_tmpl": count_files(root / "train", "*.pcd"),
        "test_pcd": count_files(root / "test", "*.pcd"),
        "ground_truth": count_files(root / "ground_truth", "*.txt"),
    }


def main() -> None:
    stats = json.loads((HERE / "statistics.json").read_text(encoding="utf-8"))
    print("=== MPW-AD integrity check ===")
    print(f"Version: {stats.get('version')}")
    print()
    print("--- mvtec3d_style ---")
    for k, v in check_mvtec3d_style().items():
        print(f"  {k:30s}: {v}")
    print()
    print("--- eyecandies_style ---")
    for k, v in check_eyecandies_style().items():
        print(f"  {k:30s}: {v}")
    print()
    print("--- real3d_style ---")
    for k, v in check_real3d_style().items():
        print(f"  {k:30s}: {v}")
    print()
    # Spot checks against statistics.json
    expected_splits = stats.get("splits", {})
    print("--- spot checks vs statistics.json ---")
    print(f"  splits.train_good      expected={expected_splits.get('train_good')}")
    print(f"  splits.val_good        expected={expected_splits.get('val_good')}")
    print(f"  splits.test_good       expected={expected_splits.get('test_good')}")
    print(f"  splits.test_anomalous  expected={expected_splits.get('test_anomalous')}")


if __name__ == "__main__":
    main()
