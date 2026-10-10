"""Paired bootstrap CIs for AUROC differences and triage operating points (unified integrate).

Uses the authors' own raw_results.pkl files (trusted). All scores use the unified
integrate definition s = (color_pr_sp + pr_sp) / 2.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_ablation_per_defect import load_phase1_subset_694, load_phase2_with_species  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
N_BOOT, SEED = 1000, 0

p1 = load_phase1_subset_694()
p2 = load_phase2_with_species(REPO / "results/welds_pointad_plus/ablations/mean_fusion/raw_results.pkl")
p3 = load_phase2_with_species(REPO / "results/welds_pointad_plus_sw/raw_results.pkl")
assert p1["species"] == p2["species"] == p3["species"]
species = np.asarray(p2["species"])
y = np.asarray(p2["gt_sp"], int)


def uni(d, color=None):
    c = np.asarray(d["color_pr_sp"], float) if color is None else color
    return 0.5 * (c + np.asarray(d["pr_sp"], float))


S = {"P1": uni(p1), "P2": uni(p2), "P3": uni(p3),
     "P4": uni(p2, np.maximum(np.asarray(p2["color_pr_sp"], float), np.asarray(p3["color_pr_sp"], float)))}


def paired_ci(a, b, mask):
    """Bootstrap CI of AUROC(a) - AUROC(b) on samples in mask (good + one class), resampled jointly."""
    idx_all = np.nonzero(mask)[0]
    yy = y[idx_all]
    rng = np.random.default_rng(SEED)
    base = roc_auc_score(yy, S[a][idx_all]) - roc_auc_score(yy, S[b][idx_all])
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(idx_all), len(idx_all))
        if yy[idx].min() == yy[idx].max():
            continue
        sel = idx_all[idx]
        diffs.append(roc_auc_score(y[sel], S[a][sel]) - roc_auc_score(y[sel], S[b][sel]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return 100 * base, 100 * lo, 100 * hi


print("# Paired bootstrap 95% CI of image-AUROC differences (points), n_boot=1000, seed=0")
classes = ["AGGREGATE", "pseudo_soldering", "pinhole", "pit", "burst", "fish_scale_welding", "bump", "combined"]
for a, b in [("P4", "P1"), ("P4", "P2"), ("P2", "P1"), ("P3", "P2")]:
    for cls in classes:
        mask = np.ones_like(y, bool) if cls == "AGGREGATE" else (species == "good") | (species == cls)
        d, lo, hi = paired_ci(a, b, mask)
        sig = "*" if lo > 0 or hi < 0 else " "
        print(f"  {a}-{b}  {cls:20s} {d:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}] {sig}")

print("\n# Triage operating points (aggregate, 694 samples)")
print(f"  {'run':4s} {'FPR@TPR95':>10s} {'FPR@TPR99':>10s} {'TPR@FPR1':>9s} {'TPR@FPR5':>9s}")
for name, s in S.items():
    fpr, tpr, _ = roc_curve(y, s)
    fpr_at = lambda t: 100 * fpr[np.searchsorted(tpr, t, side="left")]
    tpr_at = lambda f: 100 * tpr[np.searchsorted(fpr, f, side="right") - 1]
    print(f"  {name:4s} {fpr_at(0.95):10.1f} {fpr_at(0.99):10.1f} {tpr_at(0.01):9.1f} {tpr_at(0.05):9.1f}")
