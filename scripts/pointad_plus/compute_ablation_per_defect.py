"""
Compute per-defect image-AUROC for the mean-fusion ablation (Task P2-G).

Loads three raw_results pickles:
 - Phase 1 baseline   : results/welds_zero_shot/raw_results.pkl (847 samples)
 - Phase 2 trained    : results/welds_pointad_plus/raw_results.pkl (694 samples)
 - Phase 2 mean fusion: results/welds_pointad_plus/ablations/mean_fusion/raw_results.pkl

Phase 1 needs to be filtered to the 694-sample subset (dedupe by (specie, stem))
using the same logic as src/pointad_plus/run_welds_pointad_plus.py.

For each variant, prints per-defect image-AUROC across the three modalities
(pr_sp / color_pr_sp / integrate_pr_sp).
"""
from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
P1_RAW = REPO_ROOT / "results" / "welds_zero_shot" / "raw_results.pkl"
P2_RAW = REPO_ROOT / "results" / "welds_pointad_plus" / "raw_results.pkl"
P2_MEAN_RAW = REPO_ROOT / "results" / "welds_pointad_plus" / "ablations" / "mean_fusion" / "raw_results.pkl"

P1_META = REPO_ROOT / "external" / "datasets" / "welds_pointad" / "all_meta.json"
P2_META = REPO_ROOT / "external" / "datasets" / "welds_pointad_mp" / "weld" / "all_meta.json"


def _auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return None
    return float(roc_auc_score(y_true, y_score))


def _ap(y_true, y_score):
    from sklearn.metrics import average_precision_score
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return None
    return float(average_precision_score(y_true, y_score))


def load_phase1_subset_694():
    """Replicate run_welds_pointad_plus filter: dedupe by (specie, stem) using
    Phase 2 multi-photo manifest test keys. Returns a dict aligned with
    raw_results pickle:
        {"gt_sp", "pr_sp", "color_pr_sp", "integrate_pr_sp", "species"}
    The lists are in dataloader (filtered Phase 1 manifest) iteration order.
    """
    p1_meta = json.loads(P1_META.read_text())
    p2_meta = json.loads(P2_META.read_text())

    # Build mp_test_keys: (specie_name, stem)
    mp_test_keys = set()
    for s in p2_meta["test"]["weld"]:
        stem = s["id"].split("_")[-1]
        mp_test_keys.add((s["specie_name"], stem))
    assert len(mp_test_keys) == 694, f"expected 694, got {len(mp_test_keys)}"

    # Phase 1 raw pickle has its scores in the order PointAD's dataset iterated
    # the *original* phase 1 all_meta.json — which is the order test["weld"]
    # appears flattened. Verify by walking meta in same order as summarize.
    p1_raw = pickle.loads(P1_RAW.read_bytes())
    gt_sp = p1_raw["weld"]["gt_sp"]
    pr_sp = p1_raw["weld"]["pr_sp"]
    color_pr = p1_raw["weld"]["color_pr_sp"]
    int_pr = p1_raw["weld"]["integrate_pr_sp"]
    assert len(gt_sp) == 847, f"phase1 pickle has {len(gt_sp)} samples, expected 847"

    # Walk phase 1 manifest in PointAD's iteration order (flat over test classes).
    # PointAD's Dataset class flattens by class then sample order; here we have
    # a single 'weld' class, so order = test['weld'] list.
    species_p1 = [s["specie_name"] for s in p1_meta["test"]["weld"]]
    stems_p1 = [s["d2_img_path"].split("/")[-1].split(".")[0]
                for s in p1_meta["test"]["weld"]]
    assert len(species_p1) == len(gt_sp), (
        f"order mismatch p1: meta {len(species_p1)} vs pickle {len(gt_sp)}"
    )

    # Apply the same filter: dedupe by (specie, stem), drop if not in mp_test_keys.
    seen = set()
    keep_idx = []
    for i, (sp, st) in enumerate(zip(species_p1, stems_p1)):
        key = (sp, st)
        if key in seen:
            continue
        if key not in mp_test_keys:
            continue
        seen.add(key)
        keep_idx.append(i)
    assert len(keep_idx) == 694, f"phase1 filtered set has {len(keep_idx)}, expected 694"

    out = {
        "gt_sp": [gt_sp[i] for i in keep_idx],
        "pr_sp": [pr_sp[i] for i in keep_idx],
        "color_pr_sp": [color_pr[i] for i in keep_idx],
        "integrate_pr_sp": [int_pr[i] for i in keep_idx],
        "species": [species_p1[i] for i in keep_idx],
    }
    return out


def load_phase2_with_species(raw_path: Path):
    """Phase 2 pickles list 694 samples in the same filter order as
    load_phase1_subset_694 -> 'species' (because run_welds_pointad_plus iterates
    that same filtered manifest). We thus recover the per-sample defect label
    by re-running the same filter and zipping it onto the pickle scores.
    """
    p1_meta = json.loads(P1_META.read_text())
    p2_meta = json.loads(P2_META.read_text())
    mp_test_keys = set()
    for s in p2_meta["test"]["weld"]:
        stem = s["id"].split("_")[-1]
        mp_test_keys.add((s["specie_name"], stem))
    seen = set()
    species_filt = []
    for s in p1_meta["test"]["weld"]:
        stem = s["d2_img_path"].split("/")[-1].split(".")[0]
        key = (s["specie_name"], stem)
        if key in seen:
            continue
        if key not in mp_test_keys:
            continue
        seen.add(key)
        species_filt.append(s["specie_name"])
    assert len(species_filt) == 694

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


def per_defect_table(data, score_key):
    """Return OrderedDict {defect_name: image_auroc_pct}.

    For each anomalous defect class, compute one-vs-good AUROC using only
    (good, that-defect) samples.
    """
    species = np.array(data["species"])
    scores = np.array(data[score_key])
    good_mask = species == "good"
    good_scores = scores[good_mask]
    n_good = int(good_mask.sum())

    defects_order = [
        "pseudo_soldering", "pinhole", "pit", "burst",
        "fish_scale_welding", "bump", "combined",
    ]
    rows = OrderedDict()
    for sp in defects_order:
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
    """Image AUROC + AP using ALL gt_sp as truth (matches PointAD aggregate)."""
    y_true = np.array(data["gt_sp"])
    y_score = np.array(data[score_key])
    return _auroc(y_true, y_score), _ap(y_true, y_score)


def fmt_pct(v):
    return f"{v*100:.2f}" if v is not None else "n/a"


def main():
    print("=" * 80)
    print("Loading data ...")
    p1 = load_phase1_subset_694()
    p2_trained = load_phase2_with_species(P2_RAW)
    p2_mean = load_phase2_with_species(P2_MEAN_RAW)

    # Sanity: species lists should match between p2 variants.
    assert p2_trained["species"] == p2_mean["species"], "species order mismatch"
    assert p2_trained["species"] == p1["species"], "p1 vs p2 species mismatch"

    n_good = int((np.array(p1["species"]) == "good").sum())
    print(f"694-set: {n_good} good + {694 - n_good} anomalous")

    print("=" * 80)
    print("AGGREGATE METRICS (one-row-per-variant, integrate modality)")
    print("=" * 80)
    for name, data in [
        ("Phase 1 (694 subset)", p1),
        ("Phase 2 trained fusion", p2_trained),
        ("Phase 2 mean fusion", p2_mean),
    ]:
        for mod in ("pr_sp", "color_pr_sp", "integrate_pr_sp"):
            auroc, ap = aggregate_metrics(data, mod)
            print(f"  {name:30s} | {mod:18s} | iAUROC={fmt_pct(auroc):>6s} | iAP={fmt_pct(ap):>6s}")
        print()

    print("=" * 80)
    print("PER-DEFECT image-AUROC (one-vs-good)")
    print("=" * 80)

    variant_data = [
        ("phase1", p1),
        ("trained", p2_trained),
        ("mean", p2_mean),
    ]

    for mod_label, mod_key in [
        ("POINT  (pr_sp)", "pr_sp"),
        ("COLOR  (color_pr_sp)", "color_pr_sp"),
        ("INTEGRATE (integrate_pr_sp)", "integrate_pr_sp"),
    ]:
        print(f"\n--- {mod_label} ---")
        results_by_variant = {}
        for vname, vdata in variant_data:
            rows, n_good = per_defect_table(vdata, mod_key)
            results_by_variant[vname] = rows

        # Header
        defects = list(results_by_variant["phase1"].keys())
        print(f"{'Defect':22s} {'n':>5s} {'Phase1':>8s} {'Trained':>8s} {'Mean':>8s} "
              f"{'TrainedDelta':>13s} {'MeanDelta':>10s} {'Mean-Trained':>13s}")
        for d in defects:
            n_p1, v_p1 = results_by_variant["phase1"][d]
            n_tr, v_tr = results_by_variant["trained"][d]
            n_mn, v_mn = results_by_variant["mean"][d]
            assert n_p1 == n_tr == n_mn, f"{d}: count mismatch {n_p1}/{n_tr}/{n_mn}"
            d_tr = (v_tr - v_p1) * 100 if (v_tr is not None and v_p1 is not None) else None
            d_mn = (v_mn - v_p1) * 100 if (v_mn is not None and v_p1 is not None) else None
            d_mt = (v_mn - v_tr) * 100 if (v_mn is not None and v_tr is not None) else None
            def fpct(x):
                return f"{x*100:.2f}" if x is not None else "n/a"
            def fd(x):
                return f"{x:+.2f}" if x is not None else "n/a"
            print(f"{d:22s} {n_p1:5d} {fpct(v_p1):>8s} {fpct(v_tr):>8s} {fpct(v_mn):>8s} "
                  f"{fd(d_tr):>13s} {fd(d_mn):>10s} {fd(d_mt):>13s}")

    print("=" * 80)
    print("done.")


if __name__ == "__main__":
    main()
