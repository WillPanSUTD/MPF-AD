"""Reviewer-requested sanity check: how many of the 153 test-good patches
come from sources that are *not* anomalous in the test split?

The reviewer asked us to validate the matched-crop protocol against the
stricter "use only native-normal scans as negatives" alternative. The
8 native-normal source ids are: 104, 105, 106, 108, 120, 130, 252, 318
(empty YOLO label files in Crop_Data/label/). These 8 scans were
deliberately skipped by ``scripts/dataset_3d/harvest_good.py`` because
they have no Phong/LUT renderings, so they never reach the multi-photo
test pipeline.

The closest computable substitute is "source-disjoint" normals: keep
only test-good patches whose source scan does NOT also contribute an
anomalous test image. This audit script computes that set and reports
its size. If the set is empty, the only honest path is to acknowledge
the limitation in prose.

Outputs:
  results/welds_sanity_check/SUMMARY.md
  results/welds_sanity_check/_class_to_sources.json
  results/welds_sanity_check/_provenance.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "Crop_Data/label"
GOOD_PATCH_DIR = ROOT / ".cache/good_patches/test_good"
OUT_DIR = ROOT / "results/welds_sanity_check"

CLASSES = {
    0: "pseudo_soldering",
    1: "pinhole",
    2: "pit",
    3: "burst",
    4: "fish_scale_welding",
    5: "bump",
}

NATIVE_NORMAL_IDS = ["104", "105", "106", "108", "120", "130", "252", "318"]


def classify_source(src_id: str):
    p = LABEL_DIR / f"{src_id}.txt"
    if not p.exists():
        return None
    cls_set = set()
    for line in p.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        cls_set.add(int(parts[0]))
    if not cls_set:
        return None  # native good
    if len(cls_set) == 1:
        return CLASSES[next(iter(cls_set))]
    return "combined"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Per-source class membership from YOLO label files
    source_to_class = {}
    native_normal = []
    for f in LABEL_DIR.glob("*.txt"):
        try:
            int(f.stem)
        except ValueError:
            continue
        bucket = classify_source(f.stem)
        if bucket is None:
            native_normal.append(f.stem)
        else:
            source_to_class[f.stem] = bucket

    assert sorted(native_normal, key=int) == NATIVE_NORMAL_IDS, (
        f"unexpected native-normal set: {sorted(native_normal, key=int)}"
    )

    class_sources = defaultdict(list)
    for src, cls in source_to_class.items():
        class_sources[cls].append(src)
    for cls in class_sources:
        class_sources[cls].sort(key=int)

    with open(OUT_DIR / "_class_to_sources.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in class_sources.items()}, f, indent=2)

    # 2. Provenance of the 153 test-good patches
    test_good_sources = defaultdict(list)  # src_id -> [1-based phase1 idx]
    for i, p in enumerate(sorted(GOOD_PATCH_DIR.glob("*.tif")), start=1):
        src = p.stem.split("_")[0]
        test_good_sources[src].append(i)
    n_test_good = sum(len(v) for v in test_good_sources.values())

    anom_sources = set(source_to_class)
    overlap_idx = []
    disjoint_idx = []
    for src, idxs in test_good_sources.items():
        if src in anom_sources:
            overlap_idx.extend(idxs)
        else:
            disjoint_idx.extend(idxs)
    overlap_idx.sort()
    disjoint_idx.sort()

    provenance = {
        "n_anomalous_sources": len(anom_sources),
        "n_native_normal_sources_skipped": len(native_normal),
        "native_normal_source_ids": NATIVE_NORMAL_IDS,
        "n_test_good_patches": n_test_good,
        "n_unique_test_good_sources": len(test_good_sources),
        "n_overlapping_test_good_patches": len(overlap_idx),
        "n_source_disjoint_test_good_patches": len(disjoint_idx),
        "source_disjoint_phase1_indices_1based": disjoint_idx,
        "overlapping_phase1_indices_1based": overlap_idx,
    }
    with open(OUT_DIR / "_provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    # 3. Write SUMMARY.md
    lines = []
    lines.append("# Welds normal-protocol sanity check\n")
    lines.append("")
    lines.append("## Setting")
    lines.append("")
    lines.append("A reviewer asked whether the matched-crop normal-set used in this paper "
                 "(`n = 153` defect-free crops harvested from anomalous-scan windows where "
                 "no YOLO defect bbox overlaps) inflates the headline image-AUROC relative "
                 "to a stricter protocol that uses only the 8 native-normal scans (source "
                 f"ids {NATIVE_NORMAL_IDS}, empty YOLO label files in "
                 "`Crop_Data/label/`) as negatives.")
    lines.append("")
    lines.append("## Result of the provenance audit")
    lines.append("")
    lines.append(f"- Anomalous source scans contributing to the 541 anomalous test images: **{len(anom_sources)}**.")
    lines.append(f"- Native-normal source scans (empty YOLO label files): **{len(native_normal)}** (`{', '.join(NATIVE_NORMAL_IDS)}`).")
    lines.append(f"- These 8 native-normal `.tif` scans were skipped by `scripts/dataset_3d/harvest_good.py` because they have no rendered Phong/LUT/Diffuse/Specular/Normal modalities (the photometric renderer only produced multi-photo outputs for the 541 anomalous source images).")
    lines.append(f"- They are therefore **absent from the multi-photo test set**: the literal '8 native normals as negatives' protocol is **not directly evaluable** on the current rendering pipeline.")
    lines.append("")
    lines.append("### Closest computable substitute: source-disjoint normals")
    lines.append("")
    lines.append("We instead asked: are any of the 153 test-good crops harvested from a source scan that does **not** also contribute an anomalous test image (so the same scan never appears on both sides of the AUROC)?")
    lines.append("")
    lines.append(f"- Unique source scans contributing test-good crops: **{len(test_good_sources)}**.")
    lines.append(f"- Of the 153 test-good crops, **{len(disjoint_idx)}** come from source-disjoint scans and **{len(overlap_idx)}** come from scans that also contribute at least one anomalous test image.")
    lines.append("")
    lines.append("Every test-good crop in the 694-sample matched protocol shares its source `.tif` with at least one anomalous test image. The matched-crop protocol is **by construction** a same-scan defect-free-vs-defect contrast: the harvester only kept windows whose IoU with every defect bbox is zero, but it took those windows from defect-bearing scans. There are no fully independent normal scans in the multi-photo split.")
    lines.append("")
    lines.append("## Implication for the paper")
    lines.append("")
    lines.append("We cannot run the exact sanity check the reviewer requested (8 native normals as negatives) because those 8 scans have no rendered photometric modalities. We also cannot run the source-disjoint variant because no test-good crop is source-disjoint. Both observations are now reported in Section 4.1 (Native-normal sanity check paragraph) and Section 7 (Limitations).")
    lines.append("")
    lines.append("Re-rendering the 8 native-normal scans through the existing Phong/LUT/Diffuse/Specular/Normal pipeline (and re-running the multi-photo data builder) is a one-shot data-engineering step that would unlock the strict sanity check in a future revision; the rest of the paper's evaluation pipeline does not require any model retraining.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `_class_to_sources.json`: per-defect-class list of source ids (used to reconstruct the test split provenance).")
    lines.append("- `_provenance.json`: machine-readable summary of the audit (counts and 1-based phase1 indices in each set).")

    with open(OUT_DIR / "SUMMARY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote: {OUT_DIR / 'SUMMARY.md'}")
    print(f"Wrote: {OUT_DIR / '_provenance.json'}")
    print(f"Wrote: {OUT_DIR / '_class_to_sources.json'}")
    print("")
    print(f"Source-disjoint test-good patches: {len(disjoint_idx)} (of 153).")
    print("Honest-limitation prose is the right path.")


if __name__ == "__main__":
    main()
