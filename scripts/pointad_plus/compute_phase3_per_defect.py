"""
Compute per-defect image-AUROC for Phase 3 (sliding-window) vs Phase 1 and
Phase 2 (mean fusion). Task P3-C.

Loads four raw_results pickles:
 - Phase 1 baseline    : results/welds_zero_shot/raw_results.pkl (847 samples)
 - Phase 2 trained     : results/welds_pointad_plus/raw_results.pkl (694 samples)
 - Phase 2 mean fusion : results/welds_pointad_plus/ablations/mean_fusion/raw_results.pkl (694)
 - Phase 3 SW          : results/welds_pointad_plus_sw/raw_results.pkl (694)
"""
from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
P1_RAW = REPO_ROOT / "results" / "welds_zero_shot" / "raw_results.pkl"
P2_RAW = REPO_ROOT / "results" / "welds_pointad_plus" / "raw_results.pkl"
P2_MEAN_RAW = REPO_ROOT / "results" / "welds_pointad_plus" / "ablations" / "mean_fusion" / "raw_results.pkl"
P3_SW_RAW = REPO_ROOT / "results" / "welds_pointad_plus_sw" / "raw_results.pkl"

P1_META = REPO_ROOT / "external" / "datasets" / "welds_pointad" / "all_meta.json"
P2_META = REPO_ROOT / "external" / "datasets" / "welds_pointad_mp" / "weld" / "all_meta.json"


def _auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return None
    return float(roc_auc_score(y_true, y_score))


def _ap(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return None
    return float(average_precision_score(y_true, y_score))


def build_species_filter():
    p1_meta = json.loads(P1_META.read_text())
    p2_meta = json.loads(P2_META.read_text())
    mp_test_keys = set()
    for s in p2_meta["test"]["weld"]:
        stem = s["id"].split("_")[-1]
        mp_test_keys.add((s["specie_name"], stem))
    assert len(mp_test_keys) == 694
    seen = set()
    species_filt = []
    keep_idx = []
    for i, s in enumerate(p1_meta["test"]["weld"]):
        stem = s["d2_img_path"].split("/")[-1].split(".")[0]
        key = (s["specie_name"], stem)
        if key in seen:
            continue
        if key not in mp_test_keys:
            continue
        seen.add(key)
        species_filt.append(s["specie_name"])
        keep_idx.append(i)
    assert len(species_filt) == 694
    return species_filt, keep_idx


def load_phase1_subset_694(species_filt, keep_idx):
    p1_raw = pickle.loads(P1_RAW.read_bytes())
    gt_sp = p1_raw["weld"]["gt_sp"]
    pr_sp = p1_raw["weld"]["pr_sp"]
    color_pr = p1_raw["weld"]["color_pr_sp"]
    int_pr = p1_raw["weld"]["integrate_pr_sp"]
    assert len(gt_sp) == 847
    return {
        "gt_sp": [gt_sp[i] for i in keep_idx],
        "pr_sp": [pr_sp[i] for i in keep_idx],
        "color_pr_sp": [color_pr[i] for i in keep_idx],
        "integrate_pr_sp": [int_pr[i] for i in keep_idx],
        "species": species_filt,
    }


def load_phase_n(raw_path: Path, species_filt):
    raw = pickle.loads(raw_path.read_bytes())
    pr_sp = raw["weld"]["pr_sp"]
    color_pr = raw["weld"]["color_pr_sp"]
    int_pr = raw["weld"]["integrate_pr_sp"]
    gt_sp = raw["weld"]["gt_sp"]
    assert len(pr_sp) == 694, f"{raw_path}: expected 694, got {len(pr_sp)}"
    return {
        "gt_sp": gt_sp,
        "pr_sp": pr_sp,
        "color_pr_sp": color_pr,
        "integrate_pr_sp": int_pr,
        "species": species_filt,
    }


DEFECT_ORDER = [
    "pseudo_soldering", "pinhole", "pit", "burst",
    "fish_scale_welding", "bump", "combined",
]


def per_defect_table(data, score_key):
    species = np.array(data["species"])
    scores = np.array(data[score_key])
    good_mask = species == "good"
    good_scores = scores[good_mask]
    n_good = int(good_mask.sum())
    rows = OrderedDict()
    for sp in DEFECT_ORDER:
        sp_mask = species == sp
        n = int(sp_mask.sum())
        if n == 0:
            rows[sp] = (n, None)
            continue
        y_true = np.concatenate([np.zeros(n_good), np.ones(n)])
        y_score = np.concatenate([good_scores, scores[sp_mask]])
        rows[sp] = (n, _auroc(y_true, y_score))
    return rows, n_good


def aggregate_metrics(data, score_key):
    y_true = np.array(data["gt_sp"])
    y_score = np.array(data[score_key])
    return _auroc(y_true, y_score), _ap(y_true, y_score)


def fpct(x):
    return f"{x*100:.2f}" if x is not None else "n/a"


def fd(x):
    return f"{x:+.2f}" if x is not None else "n/a"


def main():
    print("Building 694-sample filter ...")
    species_filt, keep_idx = build_species_filter()

    p1 = load_phase1_subset_694(species_filt, keep_idx)
    p2_trained = load_phase_n(P2_RAW, species_filt)
    p2_mean = load_phase_n(P2_MEAN_RAW, species_filt)
    p3_sw = load_phase_n(P3_SW_RAW, species_filt)

    # Sanity check: gt_sp should match across the 694-set variants
    assert p2_trained["gt_sp"] == p2_mean["gt_sp"] == p3_sw["gt_sp"]
    n_good = int((np.array(species_filt) == "good").sum())
    print(f"694-set: {n_good} good + {694 - n_good} anomalous")

    # Verify p3 point branch is identical to p2 (sliding window only changes color)
    same_point = np.allclose(p3_sw["pr_sp"], p2_mean["pr_sp"])
    print(f"P3 point branch == P2 mean point branch: {same_point}")
    diff_color = float(np.mean(np.abs(np.array(p3_sw["color_pr_sp"]) - np.array(p2_mean["color_pr_sp"]))))
    print(f"P3 color branch mean |diff| vs P2 mean color: {diff_color:.4f}")

    print("\n=== AGGREGATE METRICS (image-level) ===")
    variants = [
        ("Phase 1 (694 subset)", p1),
        ("Phase 2 trained fusion", p2_trained),
        ("Phase 2 mean fusion", p2_mean),
        ("Phase 3 SW (mean+SW)", p3_sw),
    ]
    for name, data in variants:
        for mod in ("pr_sp", "color_pr_sp", "integrate_pr_sp"):
            auroc, ap = aggregate_metrics(data, mod)
            print(f"  {name:28s} | {mod:18s} | iAUROC={fpct(auroc):>6s} | iAP={fpct(ap):>6s}")
        print()

    print("=== PER-DEFECT image-AUROC (one-vs-good) ===")

    variant_keys = [("phase1", p1), ("p2_trained", p2_trained), ("p2_mean", p2_mean), ("p3_sw", p3_sw)]

    for mod_label, mod_key in [
        ("POINT  (pr_sp)", "pr_sp"),
        ("COLOR  (color_pr_sp)", "color_pr_sp"),
        ("INTEGRATE (integrate_pr_sp)", "integrate_pr_sp"),
    ]:
        print(f"\n--- {mod_label} ---")
        results_by_variant = {}
        for vname, vdata in variant_keys:
            rows, _ = per_defect_table(vdata, mod_key)
            results_by_variant[vname] = rows

        defects = list(results_by_variant["phase1"].keys())
        print(f"{'Defect':22s} {'n':>5s} {'P1':>8s} {'P2_tr':>8s} {'P2_mn':>8s} {'P3_SW':>8s} "
              f"{'D_SW-P2mn':>10s} {'D_SW-P1':>10s}")
        for d in defects:
            n, v_p1 = results_by_variant["phase1"][d]
            _, v_p2t = results_by_variant["p2_trained"][d]
            _, v_p2m = results_by_variant["p2_mean"][d]
            _, v_p3 = results_by_variant["p3_sw"][d]
            d_sw_p2m = (v_p3 - v_p2m) * 100 if (v_p3 is not None and v_p2m is not None) else None
            d_sw_p1 = (v_p3 - v_p1) * 100 if (v_p3 is not None and v_p1 is not None) else None
            print(f"{d:22s} {n:5d} {fpct(v_p1):>8s} {fpct(v_p2t):>8s} {fpct(v_p2m):>8s} {fpct(v_p3):>8s} "
                  f"{fd(d_sw_p2m):>10s} {fd(d_sw_p1):>10s}")

    print("\ndone.")


if __name__ == "__main__":
    main()
