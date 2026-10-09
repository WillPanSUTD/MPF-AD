"""Calibration analysis for the per-sample max gate (Task D / Revision F).

Produces two figures and a small table:

  figures/F14_score_calibration.{pdf,png}   -- 2-panel histogram of P2 mean color
                                                vs P3 SW color, normal vs anomalous.
  figures/F15_gate_selection.{pdf,png}      -- bar chart of fraction of samples per
                                                defect class where P3 SW color > P2 mean
                                                color (i.e. where max-hybrid picks SW).
  stdout: gate-selection fractions per defect class + good.

Inputs from welds 694-sample matched protocol:
  results/welds_pointad_plus/ablations/mean_fusion/raw_results.pkl
  results/welds_pointad_plus_sw/raw_results.pkl
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
P2_RAW = REPO / "results" / "welds_pointad_plus" / "ablations" / "mean_fusion" / "raw_results.pkl"
P3_RAW = REPO / "results" / "welds_pointad_plus_sw" / "raw_results.pkl"
P2_META = REPO / "external" / "datasets" / "welds_pointad_mp" / "weld" / "all_meta.json"
FIG_DIR = REPO / "results" / "welds_pointad_plus_hybrid" / "figures"

DEFECT_ORDER = [
    "pseudo_soldering", "pinhole", "pit", "burst",
    "fish_scale_welding", "bump", "combined",
]


def load_species():
    """Species labels aligned with the raw_results.pkl sample order.

    The runners iterate PointAD's Dataset over the P1 manifest and keep the
    (specie, stem) pairs present in the multi-photo manifest, so the pickle
    order follows the P1 manifest, not the multi-photo manifest. Reuse the
    shared loader that reconstructs exactly that order.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compute_ablation_per_defect import load_phase2_with_species
    species = load_phase2_with_species(P2_RAW)["species"]
    assert len(species) == 694
    return np.asarray(species)


def main():
    species = load_species()
    p2 = pickle.loads(P2_RAW.read_bytes())["weld"]
    p3 = pickle.loads(P3_RAW.read_bytes())["weld"]
    gt = np.asarray(p2["gt_sp"], dtype=int)
    color_p2 = np.asarray(p2["color_pr_sp"], dtype=float)
    color_p3 = np.asarray(p3["color_pr_sp"], dtype=float)

    # --- Panel F14: histogram of color scores, normal vs anomalous ----------
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    nb = 30
    for ax, scores, title in [
        (axes[0], color_p2, "Global MPF (P2), photometric score"),
        (axes[1], color_p3, "Sliding-Window MPF (P3), photometric score"),
    ]:
        s_norm = scores[gt == 0]
        s_anom = scores[gt == 1]
        ax.hist(s_norm, bins=nb, alpha=0.6, color="#4C78A8", label=f"Normal (n={len(s_norm)})", density=True)
        ax.hist(s_anom, bins=nb, alpha=0.6, color="#E45756", label=f"Anomalous (n={len(s_anom)})", density=True)
        ax.set_xlabel("Image-level photometric score")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Density")
    plt.suptitle("Per-sample photometric-score distributions (MPW-AD, 694 samples)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_pdf = FIG_DIR / "F14_score_calibration.pdf"
    out_png = FIG_DIR / "F14_score_calibration.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_pdf}")
    # Score-scale stats
    print(f"\nScore-scale stats (raw range / mean / std):")
    for name, s in [("P2_mean_color", color_p2), ("P3_SW_color", color_p3)]:
        print(f"  {name:18s}  min={s.min():.3f}  max={s.max():.3f}  mean={s.mean():.3f}  std={s.std():.3f}")

    # --- Panel F15: gate selection frequency per defect class --------------
    classes = ["good"] + DEFECT_ORDER
    pick_sw_frac = []
    counts = []
    for cls in classes:
        mask = species == cls
        n = int(mask.sum())
        if n == 0:
            pick_sw_frac.append(np.nan)
            counts.append(0)
            continue
        frac = float((color_p3[mask] > color_p2[mask]).sum()) / n
        pick_sw_frac.append(frac)
        counts.append(n)

    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    xs = np.arange(len(classes))
    bars = ax.bar(xs, [f * 100 for f in pick_sw_frac], color="#54A24B", edgecolor="black", alpha=0.85)
    for x, f, n in zip(xs, pick_sw_frac, counts):
        ax.text(x, f * 100 + 1.5, f"n={n}", ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=8)
    ax.set_ylabel("% samples where P3 > P2")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Max-Gated MPF: fraction of samples taking the P3 score", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_pdf = FIG_DIR / "F15_gate_selection.pdf"
    out_png = FIG_DIR / "F15_gate_selection.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_pdf}")

    print("\nGate-selection fraction per class (fraction of samples where SW > Mean colour score):")
    for cls, f, n in zip(classes, pick_sw_frac, counts):
        print(f"  {cls:22s}  n={n:4d}  pick_SW={f*100:5.1f}%")


if __name__ == "__main__":
    main()
