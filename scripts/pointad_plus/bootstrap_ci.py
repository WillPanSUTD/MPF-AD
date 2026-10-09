"""Bootstrap 95% CIs for per-class and aggregate image-AUROC on the welds 694-sample
matched protocol, for the four phases P1/P2/P3/P4 (max-hybrid).

Inputs:
  results/welds_zero_shot/raw_results.pkl              (P1, 847 -> filtered to 694)
  results/welds_pointad_plus/ablations/mean_fusion/raw_results.pkl  (P2 mean, 694)
  results/welds_pointad_plus_sw/raw_results.pkl        (P3 SW, 694)

P4 hybrid color = max(P2_color, P3_color);  P4 integrate = (P4_color + point) / 2.
The point branch is shared between P2 mean and P3 SW (verified).

Outputs:
  results/welds_pointad_plus_hybrid/bootstrap_ci.csv
  prints a Markdown summary to stdout.

Method:
  For each (phase, defect or aggregate), draw n_boot=1000 paired resamples
  with replacement of the relevant subset (defect-class anomalous + all 153 good
  for per-class; full 694 for aggregate). Compute AUROC per resample; report
  mean and 2.5 / 97.5 percentiles. Random seed 0 (fixed) for reproducibility.
"""
from __future__ import annotations
import csv
import json
import pickle
from pathlib import Path
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
P1_RAW = REPO / "results" / "welds_zero_shot" / "raw_results.pkl"
P2_MEAN_RAW = REPO / "results" / "welds_pointad_plus" / "ablations" / "mean_fusion" / "raw_results.pkl"
P3_SW_RAW = REPO / "results" / "welds_pointad_plus_sw" / "raw_results.pkl"
P1_META = REPO / "external" / "datasets" / "welds_pointad" / "all_meta.json"
P2_META = REPO / "external" / "datasets" / "welds_pointad_mp" / "weld" / "all_meta.json"

OUT_CSV = REPO / "results" / "welds_pointad_plus_hybrid" / "bootstrap_ci.csv"

DEFECT_ORDER = [
    "pseudo_soldering", "pinhole", "pit", "burst",
    "fish_scale_welding", "bump", "combined",
]

N_BOOT = 1000
SEED = 0


def build_species_filter():
    p1_meta = json.loads(P1_META.read_text())
    p2_meta = json.loads(P2_META.read_text())
    mp_keys = set()
    for s in p2_meta["test"]["weld"]:
        stem = s["id"].split("_")[-1]
        mp_keys.add((s["specie_name"], stem))
    seen = set()
    species_filt = []
    keep_idx = []
    for i, s in enumerate(p1_meta["test"]["weld"]):
        stem = s["d2_img_path"].split("/")[-1].split(".")[0]
        key = (s["specie_name"], stem)
        if key in seen or key not in mp_keys:
            continue
        seen.add(key)
        species_filt.append(s["specie_name"])
        keep_idx.append(i)
    assert len(species_filt) == 694, len(species_filt)
    return species_filt, keep_idx


def load_694(path: Path):
    raw = pickle.loads(path.read_bytes())
    w = raw["weld"]
    return {
        "gt_sp": np.asarray(w["gt_sp"], dtype=int),
        "pr_sp": np.asarray(w["pr_sp"], dtype=float),
        "color_pr_sp": np.asarray(w["color_pr_sp"], dtype=float),
        "integrate_pr_sp": np.asarray(w["integrate_pr_sp"], dtype=float),
    }


def load_p1_694(species_filt, keep_idx):
    raw = pickle.loads(P1_RAW.read_bytes())
    w = raw["weld"]
    sel = lambda L: np.asarray([L[i] for i in keep_idx], dtype=float)
    return {
        "gt_sp": np.asarray([w["gt_sp"][i] for i in keep_idx], dtype=int),
        "pr_sp": sel(w["pr_sp"]),
        "color_pr_sp": sel(w["color_pr_sp"]),
        "integrate_pr_sp": sel(w["integrate_pr_sp"]),
    }


def bootstrap_auroc(y_true, y_score, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap; resample indices then compute AUROC.
    Returns (mean, lo, hi, point).
    Skips resamples that produce a single-class label vector.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = roc_auc_score(y_true, y_score)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():
            continue
        aucs.append(roc_auc_score(yt, y_score[idx]))
    aucs = np.asarray(aucs)
    if len(aucs) == 0:
        return None, None, None, float(point)
    return float(aucs.mean()), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), float(point)


def main():
    species_filt, keep_idx = build_species_filter()
    species = np.asarray(species_filt)
    good_mask = species == "good"
    n_good = int(good_mask.sum())

    p1 = load_p1_694(species_filt, keep_idx)
    p2 = load_694(P2_MEAN_RAW)
    p3 = load_694(P3_SW_RAW)

    # Verify gt orderings consistent
    assert np.array_equal(p2["gt_sp"], p3["gt_sp"])
    # point branch identical across P2 mean and P3 SW
    assert np.allclose(p2["pr_sp"], p3["pr_sp"])

    point_branch = p2["pr_sp"]
    p4_color = np.maximum(p2["color_pr_sp"], p3["color_pr_sp"])
    p4_integrate = 0.5 * (p4_color + point_branch)

    phases = OrderedDict([
        ("P1", p1["integrate_pr_sp"]),
        ("P2_mean", p2["integrate_pr_sp"]),
        ("P3_SW", p3["integrate_pr_sp"]),
        ("P4_max_hybrid", p4_integrate),
    ])
    color_phases = OrderedDict([
        ("P1_color", p1["color_pr_sp"]),
        ("P2_mean_color", p2["color_pr_sp"]),
        ("P3_SW_color", p3["color_pr_sp"]),
        ("P4_max_color", p4_color),
    ])
    gt = p2["gt_sp"]

    rows = []
    rows.append(["phase", "defect", "n_anomalous", "n_good", "point_estimate", "boot_mean", "ci_lo_2p5", "ci_hi_97p5", "ci_halfwidth"])

    print(f"# Bootstrap 95% CIs (n_boot={N_BOOT}, seed={SEED}) -- integrate branch\n")

    # Aggregate first (all 694)
    for phase, scores in phases.items():
        m, lo, hi, pt = bootstrap_auroc(gt, scores)
        hw = (hi - lo) / 2 if (lo is not None and hi is not None) else None
        rows.append([phase, "AGGREGATE", int((gt == 1).sum()), n_good,
                     f"{pt*100:.2f}",
                     f"{m*100:.2f}" if m is not None else "",
                     f"{lo*100:.2f}" if lo is not None else "",
                     f"{hi*100:.2f}" if hi is not None else "",
                     f"{hw*100:.2f}" if hw is not None else ""])
        print(f"  {phase:18s} AGGREGATE n={len(gt):4d}  point={pt*100:6.2f}  CI=[{lo*100:6.2f}, {hi*100:6.2f}]  hw={hw*100:5.2f}")

    print()
    # Per-defect (each defect vs all good)
    for defect in DEFECT_ORDER:
        dmask = species == defect
        n_def = int(dmask.sum())
        if n_def == 0:
            continue
        sel = good_mask | dmask
        y_true = (species[sel] == defect).astype(int)
        for phase, scores in phases.items():
            y_score = scores[sel]
            m, lo, hi, pt = bootstrap_auroc(y_true, y_score)
            hw = (hi - lo) / 2 if (lo is not None and hi is not None) else None
            rows.append([phase, defect, n_def, n_good,
                         f"{pt*100:.2f}",
                         f"{m*100:.2f}" if m is not None else "",
                         f"{lo*100:.2f}" if lo is not None else "",
                         f"{hi*100:.2f}" if hi is not None else "",
                         f"{hw*100:.2f}" if hw is not None else ""])
            print(f"  {phase:18s} {defect:20s} n={n_def:3d}  point={pt*100:6.2f}  CI=[{lo*100:6.2f}, {hi*100:6.2f}]  hw={hw*100:5.2f}")
        print()

    # Also write a colour-branch section for ablation table
    for phase, scores in color_phases.items():
        m, lo, hi, pt = bootstrap_auroc(gt, scores)
        hw = (hi - lo) / 2 if (lo is not None and hi is not None) else None
        rows.append([phase, "AGGREGATE", int((gt == 1).sum()), n_good,
                     f"{pt*100:.2f}",
                     f"{m*100:.2f}" if m is not None else "",
                     f"{lo*100:.2f}" if lo is not None else "",
                     f"{hi*100:.2f}" if hi is not None else "",
                     f"{hw*100:.2f}" if hw is not None else ""])
        for defect in DEFECT_ORDER:
            dmask = species == defect
            n_def = int(dmask.sum())
            if n_def == 0:
                continue
            sel = good_mask | dmask
            y_true = (species[sel] == defect).astype(int)
            y_score = scores[sel]
            m, lo, hi, pt = bootstrap_auroc(y_true, y_score)
            hw = (hi - lo) / 2 if (lo is not None and hi is not None) else None
            rows.append([phase, defect, n_def, n_good,
                         f"{pt*100:.2f}",
                         f"{m*100:.2f}" if m is not None else "",
                         f"{lo*100:.2f}" if lo is not None else "",
                         f"{hi*100:.2f}" if hi is not None else "",
                         f"{hw*100:.2f}" if hw is not None else ""])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
