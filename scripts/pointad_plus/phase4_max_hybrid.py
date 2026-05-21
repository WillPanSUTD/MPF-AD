"""
Phase 4 quick-spike: per-sample max / mean hybrid of Phase 2 mean fusion and
Phase 3 sliding-window color branches.

Pure analysis on existing pickle files; no GPU runs.

Hybrid definitions (color branch only; point branch is identical across phases):
  1. max_hybrid    : color = max(color_P2_mean, color_P3_SW)
  2. mean_hybrid   : color = (color_P2_mean + color_P3_SW) / 2
  3. oracle_per_defect : for each defect class pick whichever of {P2_mean, P3_SW}
                         has higher image-AUROC on that class (cheating upper
                         bound; per-sample assignment uses the GT defect class).

Recombination: integrate = (color_hybrid + point) / 2 (matches PointAD convention).

Outputs:
  - prints per-defect comparison table + aggregate metrics
  - writes results/welds_pointad_plus_hybrid/SUMMARY.md
"""
from __future__ import annotations

import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

REPO_ROOT = Path(r"F:/dataset/LUT_AD_DataSet")
P1_RAW = REPO_ROOT / "results" / "welds_zero_shot" / "raw_results.pkl"
P2_MEAN_RAW = REPO_ROOT / "results" / "welds_pointad_plus" / "ablations" / "mean_fusion" / "raw_results.pkl"
P3_SW_RAW = REPO_ROOT / "results" / "welds_pointad_plus_sw" / "raw_results.pkl"

P1_META = REPO_ROOT / "external" / "datasets" / "welds_pointad" / "all_meta.json"
P2_META = REPO_ROOT / "external" / "datasets" / "welds_pointad_mp" / "weld" / "all_meta.json"

OUT_DIR = REPO_ROOT / "results" / "welds_pointad_plus_hybrid"
OUT_SUMMARY = OUT_DIR / "SUMMARY.md"

DEFECT_ORDER = [
    "pseudo_soldering", "pinhole", "pit", "burst",
    "fish_scale_welding", "bump", "combined",
]


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


def fpct(x):
    return f"{x*100:.2f}" if x is not None else "n/a"


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


def per_defect_auroc(species, gt_sp, scores):
    """Per-defect image-AUROC (one defect class vs all good)."""
    species = np.asarray(species)
    gt = np.asarray(gt_sp)
    s = np.asarray(scores)
    good_mask = species == "good"
    good_scores = s[good_mask]
    n_good = int(good_mask.sum())
    rows = OrderedDict()
    for sp in DEFECT_ORDER:
        sp_mask = species == sp
        n = int(sp_mask.sum())
        if n == 0:
            rows[sp] = (n, None)
            continue
        y_true = np.concatenate([np.zeros(n_good), np.ones(n)])
        y_score = np.concatenate([good_scores, s[sp_mask]])
        rows[sp] = (n, _auroc(y_true, y_score))
    return rows, n_good


def aggregate_metrics(gt_sp, scores):
    return _auroc(gt_sp, scores), _ap(gt_sp, scores)


def main():
    print("Building 694-sample filter ...")
    species_filt, keep_idx = build_species_filter()
    species_arr = np.array(species_filt)

    p1 = load_phase1_subset_694(species_filt, keep_idx)
    p2_mean = load_phase_n(P2_MEAN_RAW, species_filt)
    p3_sw = load_phase_n(P3_SW_RAW, species_filt)

    # Verify the point branch is shared between P2 mean and P3 SW
    point_p2 = np.array(p2_mean["pr_sp"], dtype=float)
    point_p3 = np.array(p3_sw["pr_sp"], dtype=float)
    same_point = np.allclose(point_p2, point_p3)
    print(f"P3 point branch == P2 mean point branch: {same_point}")
    assert same_point, "Point branches differ; cannot reuse single point for hybrids."
    point = point_p2  # 694
    gt = np.array(p2_mean["gt_sp"], dtype=int)

    color_p2 = np.array(p2_mean["color_pr_sp"], dtype=float)
    color_p3 = np.array(p3_sw["color_pr_sp"], dtype=float)

    # ---- Hybrids ----
    color_max = np.maximum(color_p2, color_p3)
    color_mean = 0.5 * (color_p2 + color_p3)

    int_max = 0.5 * (color_max + point)
    int_mean = 0.5 * (color_mean + point)

    # ---- Oracle per-defect: pick P2 or P3 *color score* depending on which gives
    #      the better per-class image-AUROC. For each sample, the *defect class*
    #      decides which color score to use (and good samples take the mean of
    #      both good-score distributions in proportion to picks they'd appear in;
    #      simpler: pool both good-score arrays per pick). We give the oracle the
    #      strongest fair version: for each defect, build a separate one-vs-good
    #      comparison using whichever color (P2 or P3) gives the higher AUROC on
    #      that class; aggregate AUROC over the union just uses the per-sample
    #      assignment good->best-of-both per-sample (max), anomalous->best-color
    #      for its class. Document the choice.
    # ----
    # Determine, per defect class, which variant has higher color image-AUROC.
    def color_auroc_for_class(sp_name, color_scores):
        m_good = species_arr == "good"
        m_sp = species_arr == sp_name
        if not m_sp.any():
            return None
        y_true = np.concatenate([np.zeros(m_good.sum()), np.ones(m_sp.sum())])
        y_score = np.concatenate([color_scores[m_good], color_scores[m_sp]])
        return _auroc(y_true, y_score)

    oracle_pick = {}
    for sp in DEFECT_ORDER:
        a2 = color_auroc_for_class(sp, color_p2)
        a3 = color_auroc_for_class(sp, color_p3)
        if a2 is None and a3 is None:
            pick = "p2_mean"
        elif a2 is None:
            pick = "p3_sw"
        elif a3 is None:
            pick = "p2_mean"
        else:
            pick = "p3_sw" if a3 >= a2 else "p2_mean"
        oracle_pick[sp] = pick

    # Build per-sample oracle color: for anomalous samples, use the variant
    # picked for that defect class. For good samples (used as negatives for
    # every class), use max(P2, P3) so we don't penalize the oracle on its
    # own negatives.
    color_oracle = color_max.copy()
    for i, sp in enumerate(species_arr):
        if sp == "good":
            continue
        if sp in oracle_pick:
            color_oracle[i] = color_p3[i] if oracle_pick[sp] == "p3_sw" else color_p2[i]
    int_oracle = 0.5 * (color_oracle + point)

    # ---- Per-defect tables (image-AUROC on integrate score, one-vs-good) ----
    variants_int = [
        ("P1", np.array(p1["integrate_pr_sp"], dtype=float)),
        ("P2_mean", np.array(p2_mean["integrate_pr_sp"], dtype=float)),
        ("P3_SW", np.array(p3_sw["integrate_pr_sp"], dtype=float)),
        ("max", int_max),
        ("mean_hybrid", int_mean),
        ("oracle", int_oracle),
    ]
    # All variants use the same gt_sp ordering
    rows_by_variant = OrderedDict()
    for name, scores in variants_int:
        rows, n_good = per_defect_auroc(species_filt, gt, scores)
        rows_by_variant[name] = rows

    # ---- Aggregate metrics (all 694) ----
    print("\n=== AGGREGATE (694 samples) — INTEGRATE branch ===")
    agg = OrderedDict()
    for name, scores in variants_int:
        a, p = aggregate_metrics(gt, scores)
        agg[name] = (a, p)
        print(f"  {name:14s} iAUROC={fpct(a):>6s} iAP={fpct(p):>6s}")

    # Also report color-only aggregate, since the hybrid acts on color
    print("\n=== AGGREGATE (694) — COLOR branch only ===")
    color_variants = [
        ("P1", np.array(p1["color_pr_sp"], dtype=float)),
        ("P2_mean", color_p2),
        ("P3_SW", color_p3),
        ("max", color_max),
        ("mean_hybrid", color_mean),
        ("oracle", color_oracle),
    ]
    color_agg = OrderedDict()
    for name, scores in color_variants:
        a, p = aggregate_metrics(gt, scores)
        color_agg[name] = (a, p)
        print(f"  {name:14s} iAUROC={fpct(a):>6s} iAP={fpct(p):>6s}")

    # ---- Per-defect table print ----
    print("\n=== PER-DEFECT IMAGE-AUROC (integrate, one-vs-good) ===")
    header = f"{'Defect':22s} {'n':>4s} {'P1':>7s} {'P2_mn':>7s} {'P3_SW':>7s} {'max':>7s} {'mean_h':>7s} {'oracle':>7s}"
    print(header)
    print("-" * len(header))
    for d in DEFECT_ORDER:
        n = rows_by_variant["P1"][d][0]
        cells = [fpct(rows_by_variant[v][d][1]) for v in ("P1", "P2_mean", "P3_SW", "max", "mean_hybrid", "oracle")]
        print(f"{d:22s} {n:4d} " + " ".join(f"{c:>7s}" for c in cells))

    print("\nOracle picks per class (which color branch wins):")
    for d in DEFECT_ORDER:
        print(f"  {d:22s} -> {oracle_pick[d]}")

    # ---- Write SUMMARY.md ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase 4 quick-spike: max / mean hybrid of P2 mean fusion + P3 sliding-window\n")
    lines.append("All numbers are image-AUROC (%) on the 694-sample welds test set, ")
    lines.append("computed with `sklearn.metrics.roc_auc_score`. Hybrid acts on the ")
    lines.append("**color** branch only; the point branch is identical across P2 mean ")
    lines.append("and P3 SW (verified `np.allclose`). `integrate = (color + point) / 2`.\n\n")

    lines.append("## Per-defect image-AUROC (integrate, one-vs-good)\n\n")
    lines.append("| Defect | n | P1 | P2 mean | P3 SW | **max** | mean_hybrid | oracle |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for d in DEFECT_ORDER:
        n = rows_by_variant["P1"][d][0]
        cells = [fpct(rows_by_variant[v][d][1]) for v in ("P1", "P2_mean", "P3_SW", "max", "mean_hybrid", "oracle")]
        # Bold the max column
        cells[3] = f"**{cells[3]}**"
        lines.append(f"| {d} | {n} | " + " | ".join(cells) + " |\n")

    lines.append("\n## Aggregate (all 694 samples)\n\n")
    lines.append("### Integrate branch ((color + point) / 2)\n\n")
    lines.append("| Method | image-AUROC | image-AP |\n|---|---:|---:|\n")
    for name in ("P1", "P2_mean", "P3_SW", "max", "mean_hybrid", "oracle"):
        a, p = agg[name]
        lines.append(f"| {name} | {fpct(a)} | {fpct(p)} |\n")

    lines.append("\n### Color branch only (the part that actually changes)\n\n")
    lines.append("| Method | image-AUROC | image-AP |\n|---|---:|---:|\n")
    for name in ("P1", "P2_mean", "P3_SW", "max", "mean_hybrid", "oracle"):
        a, p = color_agg[name]
        lines.append(f"| {name} | {fpct(a)} | {fpct(p)} |\n")

    lines.append("\n## Oracle picks per defect class\n\n")
    lines.append("(Oracle = for each defect class, pick whichever of `{P2_mean, P3_SW}` ")
    lines.append("scores higher on that class's color-AUROC; good samples use `max`.)\n\n")
    lines.append("| Defect | Oracle pick |\n|---|---|\n")
    for d in DEFECT_ORDER:
        lines.append(f"| {d} | {oracle_pick[d]} |\n")

    # ---- Verdict ----
    max_agg_int = agg["max"][0]
    p2_agg_int = agg["P2_mean"][0]
    p3_agg_int = agg["P3_SW"][0]
    mean_agg_int = agg["mean_hybrid"][0]
    oracle_agg_int = agg["oracle"][0]

    delta_max_vs_p2 = (max_agg_int - p2_agg_int) * 100
    delta_max_vs_p3 = (max_agg_int - p3_agg_int) * 100
    delta_oracle_vs_max = (oracle_agg_int - max_agg_int) * 100

    # Count per-defect wins
    def wins_for(name):
        w = 0
        ties = 0
        for d in DEFECT_ORDER:
            ref_p2 = rows_by_variant["P2_mean"][d][1]
            ref_p3 = rows_by_variant["P3_SW"][d][1]
            val = rows_by_variant[name][d][1]
            if val is None or ref_p2 is None or ref_p3 is None:
                continue
            best_pure = max(ref_p2, ref_p3)
            if val > best_pure + 1e-9:
                w += 1
            elif abs(val - best_pure) <= 1e-9:
                ties += 1
        return w, ties

    max_wins, max_ties = wins_for("max")

    lines.append("\n## Verdict\n\n")
    verdict_lines = []
    verdict_lines.append(
        f"On aggregate integrate image-AUROC the **max-hybrid** scores {fpct(max_agg_int)}% "
        f"vs P2 mean {fpct(p2_agg_int)}% (Δ {delta_max_vs_p2:+.2f} pp) and P3 SW {fpct(p3_agg_int)}% "
        f"(Δ {delta_max_vs_p3:+.2f} pp); the mean-hybrid lands at {fpct(mean_agg_int)}%, "
        f"and the per-defect oracle reaches {fpct(oracle_agg_int)}% (Δ vs max {delta_oracle_vs_max:+.2f} pp). "
    )
    if max_agg_int >= max(p2_agg_int, p3_agg_int):
        verdict_lines.append("The max-hybrid is at least tied with both pure variants in aggregate. ")
    else:
        verdict_lines.append("The max-hybrid does not beat the best pure variant in aggregate. ")
    verdict_lines.append(
        f"Per-defect it strictly beats `max(P2_mean, P3_SW)` on {max_wins} of {len(DEFECT_ORDER)} classes "
        f"({max_ties} ties). "
    )
    if max_agg_int >= p2_agg_int and max_agg_int >= p3_agg_int and max_wins >= 1:
        verdict_lines.append(
            "**Recommendation:** promote the max-hybrid to the headline method — it carries the per-defect "
            "wins of P2 mean fusion *and* P3 sliding-window without needing a class oracle."
        )
    elif p2_agg_int >= max(max_agg_int, p3_agg_int):
        verdict_lines.append(
            "**Recommendation:** keep P2 mean fusion as the headline — the max-hybrid does not provide a "
            "consistent aggregate gain over it."
        )
    else:
        verdict_lines.append(
            "**Recommendation:** report both — neither variant dominates uniformly and the oracle gap "
            "shows class-conditional routing is the real lift."
        )
    lines.append("".join(verdict_lines) + "\n")

    OUT_SUMMARY.write_text("".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
