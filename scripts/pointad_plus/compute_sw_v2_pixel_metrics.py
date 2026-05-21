"""Pixel-level metrics for Phase 3 SW v2 (welds), streamed from disk.

This recomputes the pixel-AUROC / pixel-AUPRO that the in-runner harness
would have reported, but from the streamed per-sample .npz dumps so we
don't need to keep all 694 maps resident.

CRITICAL ORDERING NOTE:
  The npz dumps were written in the order of the runner's *filtered
  phase1 manifest*, NOT the order of the multi-photo manifest
  (`welds_pointad_mp/.../all_meta.json`). The filtered order is the
  Phase 1 manifest order (alphabetical species: bump, burst, combined,
  fish_scale_welding, good, pinhole, pit, pseudo_soldering) intersected
  with the multi-photo test keys. We rebuild that order here exactly as
  `_build_filtered_phase1_manifest` does in the runner.

Outputs:
  results/welds_pointad_plus_sw_v2/SUMMARY.md
  results/welds_pointad_plus_sw_v2/pixel_per_defect.csv
  results/welds_pointad_plus_sw_v2/pixel_metrics.json
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

# Import PointAD's cal_pro_score
sys.path.insert(0, "F:/dataset/LUT_AD_DataSet/external/PointAD")
from metrics import cal_pro_score  # noqa: E402

ROOT = Path("F:/dataset/LUT_AD_DataSet")
MAPS_DIR = ROOT / "results/welds_pointad_plus_sw_v2/maps/weld"
OUT_DIR = ROOT / "results/welds_pointad_plus_sw_v2"
MP_MANIFEST = ROOT / "external/datasets/welds_pointad_mp/weld/all_meta.json"
PHASE1_ROOT = ROOT / "external/datasets/welds_pointad"
PHASE1_MANIFEST = PHASE1_ROOT / "all_meta.json"

SIZE = 336

# GT transform exactly matches PointAD external/PointAD/utils.py:get_transform
TARGET_TRANSFORM = T.Compose([
    T.Resize((SIZE, SIZE)),
    T.CenterCrop(SIZE),
    T.ToTensor(),
])


def build_filtered_phase1(mp_manifest: dict, phase1_meta: dict):
    """Reproduce _build_filtered_phase1_manifest from the SW runner.

    Returns the same ordered list of phase1 samples the runner iterated."""
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
    """Load the binary 336x336 GT exactly like PointAD's Dataset does
    for an anomalous mvtec_pc_3d_rgb sample (dataset.py L200-213)."""
    if sample["anomaly"] == 0:
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask_path = PHASE1_ROOT / sample["d2_mask_path"]
    if not mask_path.exists():
        return np.zeros((SIZE, SIZE), dtype=np.uint8)
    # PointAD: np.array(Image.open(...).convert('L')) > 0 -> uint8*255 -> L
    arr = np.array(Image.open(mask_path).convert("L")) > 0
    pil = Image.fromarray((arr.astype(np.uint8) * 255), mode="L")
    t = TARGET_TRANSFORM(pil)              # tensor (1, 336, 336) in [0,1]
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

    # Order: filtered species order
    species_seen = []
    for s in filtered:
        sp = s["specie_name"]
        if not species_seen or species_seen[-1] != sp:
            species_seen.append(sp)
    print(f"Species block order: {species_seen}")

    # --- Pass 1: stream into flat buffers ---
    scores_p = np.empty(total, dtype=np.float32)
    scores_c = np.empty(total, dtype=np.float32)
    scores_i = np.empty(total, dtype=np.float32)
    labels = np.empty(total, dtype=np.uint8)
    has_def = np.zeros(n, dtype=bool)
    specie_per_idx = []

    for idx, s in enumerate(filtered):
        npz_path = MAPS_DIR / f"{idx:05d}.npz"
        with np.load(npz_path) as z:
            point = z["anomaly"][0].astype(np.float32)
            color = z["color"][0].astype(np.float32)
            integrate = z["integrate"][0].astype(np.float32)
        gt = load_gt(s)
        sl = slice(idx * pixels, (idx + 1) * pixels)
        scores_p[sl] = point.ravel()
        scores_c[sl] = color.ravel()
        scores_i[sl] = integrate.ravel()
        labels[sl] = gt.ravel()
        has_def[idx] = bool(gt.any())
        specie_per_idx.append(s["specie_name"])
        if (idx + 1) % 100 == 0:
            print(f"  loaded {idx+1}/{n}", flush=True)

    print("Computing pixel-AUROC...")
    auroc_p = roc_auc_score(labels, scores_p)
    auroc_c = roc_auc_score(labels, scores_c)
    auroc_i = roc_auc_score(labels, scores_i)
    print(f"  point pixel-AUROC: {auroc_p*100:.2f}")
    print(f"  color pixel-AUROC: {auroc_c*100:.2f}")
    print(f"  integ pixel-AUROC: {auroc_i*100:.2f}")

    # --- Per-defect pixel-AUROC: one-vs-good convention for diagnostic only ---
    # NOTE: pixel-AUROC at per-defect granularity is dominated by per-defect
    # anomalous pixels; we include 'good' samples as negatives (their pixels are
    # all background) for stable AUROC values when defect-class is anomalous.
    per_defect = {}
    species_set = sorted(set(specie_per_idx))
    counts = Counter(specie_per_idx)
    for sp in species_set:
        if sp == "good":
            continue
        idxs = [i for i, s in enumerate(specie_per_idx) if s in ("good", sp)]
        chunks_p, chunks_c, chunks_i, chunks_lab = [], [], [], []
        for i in idxs:
            sl = slice(i * pixels, (i + 1) * pixels)
            chunks_p.append(scores_p[sl])
            chunks_c.append(scores_c[sl])
            chunks_i.append(scores_i[sl])
            chunks_lab.append(labels[sl])
        lab_concat = np.concatenate(chunks_lab)
        if lab_concat.sum() == 0:
            per_defect[sp] = (None, None, None)
            continue
        per_defect[sp] = (
            roc_auc_score(lab_concat, np.concatenate(chunks_p)),
            roc_auc_score(lab_concat, np.concatenate(chunks_c)),
            roc_auc_score(lab_concat, np.concatenate(chunks_i)),
        )
        print(f"  {sp}: P={per_defect[sp][0]*100:.2f}  "
              f"C={per_defect[sp][1]*100:.2f}  "
              f"I={per_defect[sp][2]*100:.2f}")

    # --- Per-defect pixel-AUPRO (anomalous samples only) ---
    aupro_per_defect = {}
    for sp in species_set:
        if sp == "good":
            continue
        idxs = [i for i, s in enumerate(specie_per_idx) if s == sp and has_def[i]]
        if not idxs:
            continue
        masks = np.zeros((len(idxs), SIZE, SIZE), dtype=np.uint8)
        amaps_p = np.zeros((len(idxs), SIZE, SIZE), dtype=np.float32)
        amaps_c = np.zeros((len(idxs), SIZE, SIZE), dtype=np.float32)
        amaps_i = np.zeros((len(idxs), SIZE, SIZE), dtype=np.float32)
        for j, i in enumerate(idxs):
            sl = slice(i * pixels, (i + 1) * pixels)
            masks[j] = labels[sl].reshape(SIZE, SIZE)
            amaps_p[j] = scores_p[sl].reshape(SIZE, SIZE)
            amaps_c[j] = scores_c[sl].reshape(SIZE, SIZE)
            amaps_i[j] = scores_i[sl].reshape(SIZE, SIZE)
        try:
            pro_p = cal_pro_score(masks, amaps_p)
            pro_c = cal_pro_score(masks, amaps_c)
            pro_i = cal_pro_score(masks, amaps_i)
        except Exception as e:
            print(f"  AUPRO {sp} failed: {e}")
            pro_p = pro_c = pro_i = float("nan")
        aupro_per_defect[sp] = (pro_p, pro_c, pro_i)
        print(f"  AUPRO {sp}: P={pro_p*100:.2f}  C={pro_c*100:.2f}  I={pro_i*100:.2f}")

    # --- Aggregate AUPRO: anomalous samples only ---
    anom_idxs = [i for i in range(n) if has_def[i]]
    print(f"Computing aggregate AUPRO on {len(anom_idxs)} anomalous samples...")
    masks_all = np.zeros((len(anom_idxs), SIZE, SIZE), dtype=np.uint8)
    amaps_p_all = np.zeros((len(anom_idxs), SIZE, SIZE), dtype=np.float32)
    amaps_c_all = np.zeros((len(anom_idxs), SIZE, SIZE), dtype=np.float32)
    amaps_i_all = np.zeros((len(anom_idxs), SIZE, SIZE), dtype=np.float32)
    for j, i in enumerate(anom_idxs):
        sl = slice(i * pixels, (i + 1) * pixels)
        masks_all[j] = labels[sl].reshape(SIZE, SIZE)
        amaps_p_all[j] = scores_p[sl].reshape(SIZE, SIZE)
        amaps_c_all[j] = scores_c[sl].reshape(SIZE, SIZE)
        amaps_i_all[j] = scores_i[sl].reshape(SIZE, SIZE)
    try:
        aupro_p = cal_pro_score(masks_all, amaps_p_all)
        aupro_c = cal_pro_score(masks_all, amaps_c_all)
        aupro_i = cal_pro_score(masks_all, amaps_i_all)
        print(f"Aggregate AUPRO: P={aupro_p*100:.2f}  "
              f"C={aupro_c*100:.2f}  I={aupro_i*100:.2f}")
    except Exception as e:
        print(f"Aggregate AUPRO failed: {e}")
        aupro_p = aupro_c = aupro_i = float("nan")

    # --- Write outputs ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "pixel_per_defect.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["defect",
                    "pixel_AUROC_point", "pixel_AUROC_color", "pixel_AUROC_integrate",
                    "pixel_AUPRO_point", "pixel_AUPRO_color", "pixel_AUPRO_integrate"])
        for sp in species_set:
            if sp == "good":
                continue
            r = per_defect.get(sp, (None, None, None))
            pro = aupro_per_defect.get(sp, (None, None, None))
            w.writerow([sp,
                        f"{r[0]*100:.2f}" if r[0] is not None else "",
                        f"{r[1]*100:.2f}" if r[1] is not None else "",
                        f"{r[2]*100:.2f}" if r[2] is not None else "",
                        f"{pro[0]*100:.2f}" if pro[0] is not None else "",
                        f"{pro[1]*100:.2f}" if pro[1] is not None else "",
                        f"{pro[2]*100:.2f}" if pro[2] is not None else ""])

    summary_md = OUT_DIR / "SUMMARY.md"
    n_anom = int(has_def.sum())
    lines = []
    lines.append("# Welds Phase 3 SW v2 -- pixel-level eval")
    lines.append("")
    lines.append(f"Source: per-pixel anomaly maps from "
                 f"`results/welds_pointad_plus_sw_v2/maps/weld/`, "
                 f"{n} samples ({n - n_anom} good + {n_anom} anomalous).")
    lines.append("")
    lines.append("GT masks are loaded exactly as PointAD's Dataset does: "
                 "binarised to {0,255}, resized to 336x336 via "
                 "`Resize+CenterCrop+ToTensor`, then re-thresholded at 0.5. "
                 "Pixel-AUROC is computed over the union of all "
                 f"{n * pixels:,} pixels; pixel-AUPRO follows PointAD's "
                 "`cal_pro_score` (max_step=200, FPR<=0.3) on the "
                 f"{n_anom} anomalous samples to stay within memory.")
    lines.append("")
    lines.append("Sample ordering: this script rebuilds the runner's "
                 "filtered phase1 order (alphabetical species blocks: "
                 "bump, burst, combined, fish_scale_welding, good, "
                 "pinhole, pit, pseudo_soldering) so the npz index maps "
                 "onto the same sample the runner saw. The mp-manifest "
                 "(frequency-ordered) order is NOT what the runner used.")
    lines.append("")
    lines.append("## Aggregate pixel metrics (Phase 3 SW v2)")
    lines.append("")
    lines.append("| Branch | pixel-AUROC | pixel-AUPRO |")
    lines.append("|---|---:|---:|")
    lines.append(f"| point (geometry-only) | "
                 f"{auroc_p*100:.2f} | {aupro_p*100:.2f} |")
    lines.append(f"| color (sliding window) | "
                 f"{auroc_c*100:.2f} | {aupro_c*100:.2f} |")
    lines.append(f"| integrate ((color + point) / 2) | "
                 f"{auroc_i*100:.2f} | {aupro_i*100:.2f} |")
    lines.append("")
    lines.append("Cross-check: the in-runner harness reported "
                 "point 93.5 / 69.9, color 91.1 / 60.6, "
                 "integrate 93.7 / 70.7 before the post-loop crash; this "
                 "table should match those within rounding (it uses the "
                 "same maps and the same GT pipeline).")
    lines.append("")
    lines.append("## Per-defect pixel-AUROC (one-vs-good)")
    lines.append("")
    lines.append("| Defect | n | P-R point | P-R color | P-R integrate |")
    lines.append("|---|---:|---:|---:|---:|")
    for sp in species_set:
        if sp == "good":
            continue
        r = per_defect.get(sp, (None, None, None))
        if r[0] is None:
            continue
        lines.append(f"| {sp} | {counts[sp]} | "
                     f"{r[0]*100:.2f} | {r[1]*100:.2f} | {r[2]*100:.2f} |")
    lines.append("")
    lines.append("## Per-defect pixel-AUPRO (anomalous samples only)")
    lines.append("")
    lines.append("| Defect | n | P-P point | P-P color | P-P integrate |")
    lines.append("|---|---:|---:|---:|---:|")
    for sp in species_set:
        if sp == "good":
            continue
        pro = aupro_per_defect.get(sp)
        if pro is None:
            continue
        lines.append(f"| {sp} | {counts[sp]} | "
                     f"{pro[0]*100:.2f} | {pro[1]*100:.2f} | {pro[2]*100:.2f} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The integrate branch is the SW color branch averaged "
                 "with the unchanged point branch and lightly sigma-smoothed "
                 "(sigma=4), consistent with PointAD's "
                 "`(color + point) / 2` convention.")
    lines.append("- Pixel-level metrics for Phase 1 / Phase 2 / Phase 4 "
                 "max-hybrid are not computable from the saved "
                 "`raw_results.pkl` files because those runners never "
                 "persisted per-pixel maps; only image-level scalars "
                 "survived. The Phase 3 SW row above bounds the Phase 4 "
                 "max-hybrid below in the colour branch (max is monotonic), "
                 "and from above only if the P2 mean colour branch is "
                 "uniformly weaker -- the per-defect SW-vs-mean trade in "
                 "the v1 summary shows this is not always the case.")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    agg = {
        "n_samples": n,
        "n_anomalous": n_anom,
        "pixel_AUROC": {
            "point": auroc_p,
            "color": auroc_c,
            "integrate": auroc_i,
        },
        "pixel_AUPRO": {
            "point": aupro_p,
            "color": aupro_c,
            "integrate": aupro_i,
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
