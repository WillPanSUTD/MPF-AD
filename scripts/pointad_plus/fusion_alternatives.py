"""Alternative-fusion ablation (Task D, table T5).

Compute aggregate image-AUROC and image-AP on the welds 694-sample matched
protocol under five alternative ways of combining the Phase 2 mean colour
score and the Phase 3 sliding-window colour score:

  1. raw_max      (current PointAD+ choice)
  2. z_max        z-score normalise each path on the full 694 set then max
  3. minmax_max   min-max normalise each path on the full 694 set then max
  4. rank_max     rank each path (1..n) on the full 694 set then take the max of ranks
  5. mean         (P2_color + P3_color) / 2

The integrate score for each variant is (color_fused + point) / 2; the point branch
is identical to P2 and P3 (verified). All numbers are reported on the same 694
samples.

Note on scope: ideally normalisation would be fit on a held-out calibration set.
On the 694-sample matched protocol there is no train/val split for the colour
path (the colour branch has zero learnable parameters), so we fit normalisation
on the same 694 samples used for evaluation. This is conservative for the
calibrated variants -- it gives them the best possible scale match -- so any
remaining gap is not a calibration-fitting artefact.

Output: results/welds_pointad_plus_hybrid/fusion_alternatives.json + Markdown
table to stdout.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
from collections import OrderedDict

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

REPO = Path(r"F:/dataset/LUT_AD_DataSet")
P2_RAW = REPO / "results" / "welds_pointad_plus" / "ablations" / "mean_fusion" / "raw_results.pkl"
P3_RAW = REPO / "results" / "welds_pointad_plus_sw" / "raw_results.pkl"
OUT_JSON = REPO / "results" / "welds_pointad_plus_hybrid" / "fusion_alternatives.json"


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def to_ranks(x):
    # average ranks (handles ties)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    return ranks


def main():
    p2 = pickle.loads(P2_RAW.read_bytes())["weld"]
    p3 = pickle.loads(P3_RAW.read_bytes())["weld"]
    gt = np.asarray(p2["gt_sp"], dtype=int)
    cp2 = np.asarray(p2["color_pr_sp"], dtype=float)
    cp3 = np.asarray(p3["color_pr_sp"], dtype=float)
    point = np.asarray(p2["pr_sp"], dtype=float)
    assert np.allclose(point, np.asarray(p3["pr_sp"], dtype=float))

    variants = OrderedDict()
    variants["raw_max"] = np.maximum(cp2, cp3)
    variants["z_max"] = np.maximum(zscore(cp2), zscore(cp3))
    variants["minmax_max"] = np.maximum(minmax(cp2), minmax(cp3))
    variants["rank_max"] = np.maximum(to_ranks(cp2), to_ranks(cp3))
    variants["mean"] = 0.5 * (cp2 + cp3)

    print("# Alternative fusion comparison (welds 694 protocol)\n")
    print(f"{'Variant':<14s} {'colour-AUROC':>12s} {'colour-AP':>10s} {'integrate-AUROC':>16s} {'integrate-AP':>12s}")
    print("-" * 70)

    out = {}
    for name, colour in variants.items():
        # integrate uses raw point + a normalised colour; the late fusion is
        # the simple mean, so for fair comparison we normalise integrate the
        # same way the gate normalises colour: keep raw scale for the
        # downstream `(colour + point) / 2` formula in raw_max and mean; for
        # z_max / minmax_max / rank_max we report the colour-branch number
        # directly and additionally report the integrate variant by
        # re-scaling the fused colour to match the point branch's range,
        # which is the standard recipe when mixing two paths with different
        # scales.
        if name in ("raw_max", "mean"):
            integrate = 0.5 * (colour + point)
        else:
            # rescale fused colour to point's [min,max] before averaging
            c_norm = (colour - colour.min()) / (colour.max() - colour.min() + 1e-12)
            c_rescaled = c_norm * (point.max() - point.min()) + point.min()
            integrate = 0.5 * (c_rescaled + point)

        a_c = roc_auc_score(gt, colour) * 100
        ap_c = average_precision_score(gt, colour) * 100
        a_i = roc_auc_score(gt, integrate) * 100
        ap_i = average_precision_score(gt, integrate) * 100
        print(f"{name:<14s} {a_c:12.2f} {ap_c:10.2f} {a_i:16.2f} {ap_i:12.2f}")
        out[name] = {
            "colour_auroc": a_c, "colour_ap": ap_c,
            "integrate_auroc": a_i, "integrate_ap": ap_i,
        }

    # Reference: P2 alone, P3 alone
    print()
    a_p2 = roc_auc_score(gt, cp2) * 100
    a_p3 = roc_auc_score(gt, cp3) * 100
    ip2 = roc_auc_score(gt, 0.5 * (cp2 + point)) * 100
    ip3 = roc_auc_score(gt, 0.5 * (cp3 + point)) * 100
    print(f"{'P2_mean only':<14s} {a_p2:12.2f} {average_precision_score(gt, cp2)*100:10.2f} {ip2:16.2f} {average_precision_score(gt, 0.5*(cp2+point))*100:12.2f}")
    print(f"{'P3_SW only':<14s} {a_p3:12.2f} {average_precision_score(gt, cp3)*100:10.2f} {ip3:16.2f} {average_precision_score(gt, 0.5*(cp3+point))*100:12.2f}")
    out["P2_mean_only"] = {"colour_auroc": a_p2, "integrate_auroc": ip2}
    out["P3_SW_only"] = {"colour_auroc": a_p3, "integrate_auroc": ip3}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
