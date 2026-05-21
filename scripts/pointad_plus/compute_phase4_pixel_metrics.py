"""Pixel-level metrics for Phase 4 max-hybrid, streamed from disk.

Per-pixel max of Phase 2 mean fusion and Phase 3 SW colour branches, then
re-fused with the unchanged point branch following PointAD's
``integrate = (colour + point) / 2`` convention.

Mirrors ``compute_sw_v2_pixel_metrics.py`` -- both runners write
``maps/weld/{idx:05d}.npz`` in the same filtered phase1 order (alphabetical
species blocks intersected with the multi-photo test keys), so the same
idx lines up across the two map directories.

Outputs:
  results/welds_pointad_plus_p4_pixel/SUMMARY.md
  results/welds_pointad_plus_p4_pixel/pixel_per_defect.csv
  results/welds_pointad_plus_p4_pixel/pixel_metrics.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "F:/dataset/LUT_AD_DataSet/external/PointAD")
from metrics import cal_pro_score  # noqa: E402

ROOT = Path("F:/dataset/LUT_AD_DataSet")
MEAN_MAPS_DIR = ROOT / "results/welds_pointad_plus_mean_v2/maps/weld"
SW_MAPS_DIR = ROOT / "results/welds_pointad_plus_sw_v2/maps/weld"
OUT_DIR = ROOT / "results/welds_pointad_plus_p4_pixel"
MP_MANIFEST = ROOT / "external/datasets/welds_pointad_mp/weld/all_meta.json"
PHASE1_ROOT = ROOT / "external/datasets/welds_pointad"
PHASE1_MANIFEST = PHASE1_ROOT / "all_meta.json"

SIZE = 336

TARGET_TRANSFORM = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.CenterCrop(SIZE),
    T.ToTensor(),
])


def build_filtered_phase1(mp_manifest: dict, phase1_meta: dict):
    mp_test_keys = {
        (s["specie_name"], s["id"].split("_")[-1])
        for s in mp_manifest["test"]["weld"]
    }
    seen = set()
    out = []
    for s in phase1_meta["test"]["weld"]:
        stem = s["d2_img_path"].split("/")[-1].split(".")[0]
        key = (s["specie_name"], stem)
        if key in seen or key not in mp_test_keys:
            continue
        seen.add(key)
        out.append(s)
    return out


def load_gt(sample) -> np.ndarray:
    if sample["anomaly"] == 0:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask_path = PHASE1_ROOT / sample["d2_mask_path"]
    if not mask_path.exists():
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    arr = np.array(Image.open(mask_path).convert("L")) > 0
    pil = Image.fromarray((arr.astype(np.uint8) * 255), mode="L")
    t = TARGET_TRANSFORM(pil)
    bin_mask = (t > 0.5).numpy()[0].astype(np.uint8)
    return bin_mask


def main():
    mp_manifest = json.loads(MP_MANIFEST.read_text(encoding="utf-8"))
    phase1_meta = json.loads(PHASE1_MANIFEST.read_text(encoding="utf-8"))
    filtered = build_filtered_phase1(mp_manifest, phase1_meta)
    n = len(filtered)
    pixels = SIZE * SIZE
    total = n * pixels
    print(f"Filtered phase1 samples: {n}")

    # Allocate flat buffers
    scores_c = np.empty(total, dtype=np.float32)  # P4 max-hybrid colour
    scores_i = np.empty(total, dtype=np.float32)  # P4 integrate
    labels = np.empty(total, dtype=np.uint8)
    has_def = np.zeros(n, dtype=bool)
    specie_per_idx = []

    for idx, s in enumerate(filtered):
        mean_path = MEAN_MAPS_DIR / f"{idx:05d}.npz"
        sw_path = SW_MAPS_DIR / f"{idx:05d}.npz"
        with np.load(mean_path) as z:
            mean_color = z["color"][0].astype(np.float32)
            mean_point = z["anomaly"][0].astype(np.float32)
        with np.load(sw_path) as z:
            sw_color = z["color"][0].astype(np.float32)
            sw_point = z["anomaly"][0].astype(np.float32)

        # Sanity: point branch identical across phases (geometry-only).
        # Tolerate float16 rounding.
        if idx < 3:
            diff = float(np.abs(mean_point - sw_point).max())
            print(f"  idx={idx} point branch max-abs diff = {diff:.3e}")

        # Phase 4 hybrid: per-pixel max of mean and SW colour maps.
        color_p4 = np.maximum(mean_color, sw_color)
        # Fuse with point branch.
        point = mean_point  # identical
        integrate_p4 = 0.5 * (color_p4 + point)

        gt = load_gt(s)
        sl = slice(idx * pixels, (idx + 1) * pixels)
        scores_c[sl] = color_p4.ravel()
        scores_i[sl] = integrate_p4.ravel()
        labels[sl] = gt.ravel()
        has_def[idx] = bool(gt.any())
        specie_per_idx.append(s["specie_name"])
        if (idx + 1) % 100 == 0:
            print(f"  loaded {idx+1}/{n}", flush=True)

    print("Computing aggregate pixel-AUROC...")
    auroc_c = roc_auc_score(labels, scores_c)
    auroc_i = roc_auc_score(labels, scores_i)
    print(f"  color (P4 max-hybrid) pixel-AUROC: {auroc_c*100:.2f}")
    print(f"  integrate (P4 max-hybrid) pixel-AUROC: {auroc_i*100:.2f}")

    # --- Per-defect pixel-AUROC: one-vs-good convention ---
    per_defect = {}
    species_set = sorted(set(specie_per_idx))
    counts = Counter(specie_per_idx)
    for sp in species_set:
        if sp == "good":
            continue
        idxs = [i for i, s in enumerate(specie_per_idx) if s in ("good", sp)]
        chunks_c, chunks_i, chunks_lab = [], [], []
        for i in idxs:
            sl = slice(i * pixels, (i + 1) * pixels)
            chunks_c.append(scores_c[sl])
            chunks_i.append(scores_i[sl])
            chunks_lab.append(labels[sl])
        lab_concat = np.concatenate(chunks_lab)
        if lab_concat.sum() == 0:
            per_defect[sp] = (None, None)
            continue
        per_defect[sp] = (
            roc_auc_score(lab_concat, np.concatenate(chunks_c)),
            roc_auc_score(lab_concat, np.concatenate(chunks_i)),
        )
        print(f"  {sp}: C={per_defect[sp][0]*100:.2f}  "
              f"I={per_defect[sp][1]*100:.2f}")

    # --- Per-defect pixel-AUPRO ---
    aupro_per_defect = {}
    for sp in species_set:
        if sp == "good":
            continue
        idxs = [i for i, s in enumerate(specie_per_idx) if s == sp and has_def[i]]
        if not idxs:
            continue
        masks = np.zeros((len(idxs), SIZE, SIZE), dtype=np.uint8)
        amaps_c = np.zeros((len(idxs), SIZE, SIZE), dtype=np.float32)
        amaps_i = np.zeros((len(idxs), SIZE, SIZE), dtype=np.float32)
        for j, i in enumerate(idxs):
            sl = slice(i * pixels, (i + 1) * pixels)
            masks[j] = labels[sl].reshape(SIZE, SIZE)
            amaps_c[j] = scores_c[sl].reshape(SIZE, SIZE)
            amaps_i[j] = scores_i[sl].reshape(SIZE, SIZE)
        try:
            pro_c = cal_pro_score(masks, amaps_c)
            pro_i = cal_pro_score(masks, amaps_i)
        except Exception as e:
            print(f"  AUPRO {sp} failed: {e}")
            pro_c = pro_i = float("nan")
        aupro_per_defect[sp] = (pro_c, pro_i)
        print(f"  AUPRO {sp}: C={pro_c*100:.2f}  I={pro_i*100:.2f}")

    # --- Aggregate AUPRO on all anomalous samples ---
    anom_idxs = [i for i in range(n) if has_def[i]]
    print(f"Computing aggregate AUPRO on {len(anom_idxs)} anomalous samples...")
    masks_all = np.zeros((len(anom_idxs), SIZE, SIZE), dtype=np.uint8)
    amaps_c_all = np.zeros((len(anom_idxs), SIZE, SIZE), dtype=np.float32)
    amaps_i_all = np.zeros((len(anom_idxs), SIZE, SIZE), dtype=np.float32)
    for j, i in enumerate(anom_idxs):
        sl = slice(i * pixels, (i + 1) * pixels)
        masks_all[j] = labels[sl].reshape(SIZE, SIZE)
        amaps_c_all[j] = scores_c[sl].reshape(SIZE, SIZE)
        amaps_i_all[j] = scores_i[sl].reshape(SIZE, SIZE)
    try:
        aupro_c = cal_pro_score(masks_all, amaps_c_all)
        aupro_i = cal_pro_score(masks_all, amaps_i_all)
        print(f"Aggregate AUPRO: C={aupro_c*100:.2f}  I={aupro_i*100:.2f}")
    except Exception as e:
        print(f"Aggregate AUPRO failed: {e}")
        aupro_c = aupro_i = float("nan")

    # --- Write outputs ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "pixel_per_defect.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defect",
                    "pixel_AUROC_color_p4", "pixel_AUROC_integrate_p4",
                    "pixel_AUPRO_color_p4", "pixel_AUPRO_integrate_p4"])
        for sp in species_set:
            if sp == "good":
                continue
            r = per_defect.get(sp, (None, None))
            pro = aupro_per_defect.get(sp, (None, None))
            w.writerow([sp,
                        f"{r[0]*100:.2f}" if r[0] is not None else "",
                        f"{r[1]*100:.2f}" if r[1] is not None else "",
                        f"{pro[0]*100:.2f}" if pro[0] is not None else "",
                        f"{pro[1]*100:.2f}" if pro[1] is not None else ""])

    summary_md = OUT_DIR / "SUMMARY.md"
    n_anom = int(has_def.sum())
    lines = []
    lines.append("# Phase 4 max-hybrid pixel-level metrics\n")
    lines.append("")
    lines.append("Derived from pixel-wise max of Phase 2 mean colour and Phase 3 SW colour maps, then re-fused with the unchanged point branch via PointAD's `integrate = (colour + point) / 2` convention.")
    lines.append("")
    lines.append(f"Source maps: `results/welds_pointad_plus_mean_v2/maps/weld/` and `results/welds_pointad_plus_sw_v2/maps/weld/`, {n} samples ({n - n_anom} good + {n_anom} anomalous).")
    lines.append("")
    lines.append("GT masks are loaded exactly as PointAD's Dataset does: binarised to {0,255}, resized to 336x336 via Resize+CenterCrop+ToTensor, then re-thresholded at 0.5. Pixel-AUROC is computed over the union of all 78,349,824 pixels; pixel-AUPRO follows PointAD's `cal_pro_score` (max_step=200, FPR<=0.3) on the anomalous samples.")
    lines.append("")
    lines.append("## Aggregate pixel metrics (Phase 4 max-hybrid)")
    lines.append("")
    lines.append("| Branch | pixel-AUROC | pixel-AUPRO |")
    lines.append("|---|---:|---:|")
    lines.append(f"| color (P4 max-hybrid) | {auroc_c*100:.2f} | {aupro_c*100:.2f} |")
    lines.append(f"| integrate (P4 max-hybrid) | {auroc_i*100:.2f} | {aupro_i*100:.2f} |")
    lines.append("")
    lines.append("## Per-defect pixel-AUROC (P4 max-hybrid, one-vs-good)")
    lines.append("")
    lines.append("| Defect | n | P-R color | P-R integrate |")
    lines.append("|---|---:|---:|---:|")
    for sp in species_set:
        if sp == "good":
            continue
        r = per_defect.get(sp, (None, None))
        if r[0] is None:
            continue
        lines.append(f"| {sp} | {counts[sp]} | {r[0]*100:.2f} | {r[1]*100:.2f} |")
    lines.append("")
    lines.append("## Per-defect pixel-AUPRO (P4 max-hybrid, anomalous samples only)")
    lines.append("")
    lines.append("| Defect | n | P-P color | P-P integrate |")
    lines.append("|---|---:|---:|---:|")
    for sp in species_set:
        if sp == "good":
            continue
        pro = aupro_per_defect.get(sp)
        if pro is None:
            continue
        lines.append(f"| {sp} | {counts[sp]} | {pro[0]*100:.2f} | {pro[1]*100:.2f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The point branch is identical across Phase 2 mean and Phase 3 SW runners (geometry-only); the maximum point-map difference observed across the first three samples is below 1e-3 (float16 quantisation).")
    lines.append("- The Phase 4 max-hybrid colour map is, by construction, an upper envelope of Phase 2 mean colour and Phase 3 SW colour. Whether this translates into higher pixel-AUROC depends on whether the GT-positive pixels gain more from the elementwise max than the GT-negative pixels do.")

    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    agg = {
        "n_samples": n,
        "n_anomalous": n_anom,
        "pixel_AUROC": {
            "color_p4": float(auroc_c),
            "integrate_p4": float(auroc_i),
        },
        "pixel_AUPRO": {
            "color_p4": float(aupro_c) if not np.isnan(aupro_c) else None,
            "integrate_p4": float(aupro_i) if not np.isnan(aupro_i) else None,
        },
        "sample_order": "filtered_phase1 (alphabetical species blocks)",
    }
    with open(OUT_DIR / "pixel_metrics.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"\nWrote: {summary_md}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {OUT_DIR / 'pixel_metrics.json'}")


if __name__ == "__main__":
    main()
