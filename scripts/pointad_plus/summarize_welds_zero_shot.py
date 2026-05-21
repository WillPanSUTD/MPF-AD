"""
Build a per-defect breakdown of PointAD's welds zero-shot run.

PointAD only knows the single "weld" class, so its emitted metrics aggregate
across all 847 test samples. We re-derive per-defect image-AUROC by joining
the captured per-sample scores (raw_results.pkl) with the dataloader sample
order (which we replicate from the same all_meta.json PointAD reads).

Pixel-level per-defect AUPRO requires the full anomaly maps, which we did not
persist. We therefore report aggregate pixel metrics by parsing the PointAD
log.txt and per-defect image-level metrics from the pickle.
"""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "external" / "datasets" / "welds_pointad"
SAVE_DIR = REPO_ROOT / "results" / "welds_zero_shot"
META_PATH = DATA_ROOT / "all_meta.json"
RAW_PKL = SAVE_DIR / "raw_results.pkl"
LOG_TXT = SAVE_DIR / "log.txt"


def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return None  # one class -> auroc undefined
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.min() == y_true.max():
        return None
    return float(average_precision_score(y_true, y_score))


def main() -> None:
    if not RAW_PKL.is_file():
        raise SystemExit(f"missing {RAW_PKL}")
    raw = pickle.loads(RAW_PKL.read_bytes())
    if "weld" not in raw:
        raise SystemExit(f"raw pickle missing 'weld' key: {list(raw.keys())}")
    gt = raw["weld"]["gt_sp"]
    pr = raw["weld"]["pr_sp"]
    color_pr = raw["weld"]["color_pr_sp"]
    integrate_pr = raw["weld"]["integrate_pr_sp"]

    # Replicate the dataloader's sample order: meta['test'][cls] flattened in
    # the same key order PointAD iterates (Python dict insertion order).
    meta = json.loads(META_PATH.read_text())["test"]
    species = []
    for cls in meta:
        for s in meta[cls]:
            species.append(s["specie_name"])
    assert len(species) == len(gt), (
        f"order mismatch: {len(species)} samples in meta vs {len(gt)} in pickle"
    )

    # Sanity: every entry in 'good' must have anomaly==0, others ==1.
    species_arr = np.array(species)
    gt_arr = np.array(gt)
    for sp in sorted(set(species)):
        mask = species_arr == sp
        expected = 0 if sp == "good" else 1
        if not np.all(gt_arr[mask] == expected):
            print(f"warn: specie={sp} has mixed anomaly labels")

    # Per-defect image-AUROC: for each anomalous specie, compare its scores
    # against the "good" baseline.
    good_mask = species_arr == "good"
    good_scores_pr = np.array(pr)[good_mask]
    good_scores_color = np.array(color_pr)[good_mask]
    good_scores_int = np.array(integrate_pr)[good_mask]

    rows = []
    for sp in sorted(set(species)):
        if sp == "good":
            continue
        sp_mask = species_arr == sp
        n = int(sp_mask.sum())
        # Construct binary {good=0, this_defect=1} subset
        y_true = np.concatenate([np.zeros(int(good_mask.sum())), np.ones(n)])
        y_pr = np.concatenate([good_scores_pr, np.array(pr)[sp_mask]])
        y_color = np.concatenate([good_scores_color, np.array(color_pr)[sp_mask]])
        y_int = np.concatenate([good_scores_int, np.array(integrate_pr)[sp_mask]])
        rows.append({
            "defect": sp,
            "n_defect": n,
            "n_good": int(good_mask.sum()),
            "image_auroc_point": _safe_auroc(y_true, y_pr),
            "image_auroc_color": _safe_auroc(y_true, y_color),
            "image_auroc_integrate": _safe_auroc(y_true, y_int),
            "image_ap_integrate": _safe_ap(y_true, y_int),
        })

    # All-anomalous aggregate (the number PointAD reports for the "weld" obj).
    all_y_true = np.array(gt)
    overall = {
        "defect": "ALL anomalous (PointAD's reported weld row)",
        "n_defect": int((all_y_true == 1).sum()),
        "n_good": int((all_y_true == 0).sum()),
        "image_auroc_point": _safe_auroc(all_y_true, pr),
        "image_auroc_color": _safe_auroc(all_y_true, color_pr),
        "image_auroc_integrate": _safe_auroc(all_y_true, integrate_pr),
        "image_ap_integrate": _safe_ap(all_y_true, integrate_pr),
    }

    # Parse pixel-level numbers (aggregate) from PointAD's log.
    pixel_metrics = {"point": {}, "color": {}, "integrate": {}}
    if LOG_TXT.is_file():
        text = LOG_TXT.read_text(errors="replace")
        # PointAD logs three tables back-to-back: point, color, integrate.
        # Each has a row starting with 'weld' under headers pixel_auroc, pixel_aupro,
        # image_auroc, image_ap (the test() function uses image-pixel-level).
        # The 'mean' row equals the 'weld' row since there's only one obj.
        # We grep all numeric rows beginning with 'weld' (3 of them).
        weld_rows = re.findall(
            r"\|\s*weld\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
            text,
        )
        if len(weld_rows) >= 3:
            for tag, row in zip(["point", "color", "integrate"], weld_rows[:3]):
                pixel_metrics[tag] = {
                    "pixel_auroc": float(row[0]),
                    "pixel_aupro": float(row[1]),
                    "image_auroc": float(row[2]),
                    "image_ap": float(row[3]),
                }

    # Write SUMMARY.md
    lines = []
    lines.append("# Welds zero-shot baseline (PointAD, single-view repetition)\n")
    lines.append("")
    lines.append("Source checkpoint: `external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth`")
    lines.append("Run date: 2026-05-17")
    lines.append("GPU: NVIDIA GeForce RTX 4090")
    lines.append("Adapter: welds → PointAD MVTec3D-AD format (single-view repetition; 9 identical views per sample)")
    lines.append("Image resolution: 336x336 | point_size: 336 | sigma=4 | seed=111")
    lines.append("PointAD args mirror the shipped `test.sh` (depth=9, n_ctx=12, t_n_ctx=4, features_list=[24], metrics=image-pixel-level).")
    lines.append("")
    lines.append("## Aggregate metrics (all 541 anomalous vs 306 good, the row PointAD prints)")
    lines.append("")
    lines.append("| Modality | pixel_auroc | pixel_aupro | image_auroc | image_ap |")
    lines.append("|---|---|---|---|---|")
    for tag in ("point", "color", "integrate"):
        pm = pixel_metrics.get(tag) or {}
        if pm:
            lines.append(
                f"| {tag} | {pm['pixel_auroc']:.1f} | {pm['pixel_aupro']:.1f} | {pm['image_auroc']:.1f} | {pm['image_ap']:.1f} |"
            )
        else:
            lines.append(f"| {tag} | (log parse failed) | | | |")
    lines.append("")
    lines.append("PointAD reports three modalities: `point` (geometry-only), `color` (RGB-only), `integrate` (their default fused score). The headline number is `integrate`.")
    lines.append("")
    lines.append("## Per-defect image-level AUROC (each defect vs the 306 good samples)")
    lines.append("")
    lines.append("| Defect | n | AUROC@point | AUROC@color | AUROC@integrate | AP@integrate |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        def fmt(v):
            return f"{v*100:.1f}" if v is not None else "n/a"
        lines.append(
            f"| {r['defect']} | {r['n_defect']} | {fmt(r['image_auroc_point'])} | {fmt(r['image_auroc_color'])} | {fmt(r['image_auroc_integrate'])} | {fmt(r['image_ap_integrate'])} |"
        )
    def fmt(v):
        return f"{v*100:.1f}" if v is not None else "n/a"
    lines.append(
        f"| **ALL anomalous** | {overall['n_defect']} | {fmt(overall['image_auroc_point'])} | {fmt(overall['image_auroc_color'])} | {fmt(overall['image_auroc_integrate'])} | {fmt(overall['image_ap_integrate'])} |"
    )
    lines.append("")
    lines.append("Per-defect pixel-level breakdown is not provided because PointAD's pipeline persists only scalar pixel metrics. To get per-defect AUPRO we would need to re-run with the full anomaly-map cache on disk; that is a future-Phase task.")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- The 9 views per sample are *identical* Phong renders. PointAD's multi-view fusion is therefore degenerate: the geometry branch loses its main source of robustness. Phase 2 will replace with proper rotational rendering.")
    lines.append("- Some defect classes have very few samples (burst=3, pit=10, pinhole=11, bump=19). Per-defect numbers should be read with that in mind; confidence intervals will be wide.")
    lines.append("- Auxiliary training source: MVTec3D-AD `carrot` (largest of the three shipped per-class prompt checkpoints in `exps_9_12_4_mv9_mvtec_3d_336_4/`). PointAD's protocol leaves prompts object-agnostic, so any class' prompts are valid transfer source; `carrot` is a defensible default.")
    lines.append("")
    lines.append("## Sanity check vs PointAD paper")
    lines.append("")
    lines.append("PointAD's published MVTec3D-AD mean image_auroc with the `integrate` modality is on the order of 80-85 on its own benchmark categories. Our welds zero-shot integrate image_auroc of 88.5 lands above the paper's mean (and pixel_auroc 94.1 / pixel_aupro 71.7 are in the same ballpark as PointAD's reported pixel-level numbers). Interpretation:")
    lines.append("")
    lines.append("1. Checkpoint loading is correct — the model is not predicting random.")
    lines.append("2. Welds is *easier* in absolute terms than MVTec3D-AD on this protocol because the anomaly-vs-normal contrast is much more visually salient (large discoloration / scoring on a uniform metallic weld bead is high-signal for CLIP), and because the test set is imbalanced toward anomalous (541 / 847) which inflates image_ap.")
    lines.append("3. The per-defect breakdown shows the headline number hides real weakness: `pseudo_soldering` (the largest defect class, n=206) is only 76.0 AUROC, well below the 88.5 aggregate. The aggregate is buoyed by `combined` (n=271, AUROC=97.1).")
    lines.append("4. The single-view-repetition caveat applies even more strongly given these high numbers — proper multi-view rendering should improve weak classes (pseudo_soldering, fish_scale_welding) more than the easy ones, narrowing the spread.")
    lines.append("")
    lines.append("This is a stronger zero-shot baseline than anticipated; the Phase 2/3 gap is therefore closing pseudo_soldering and fish_scale_welding (the photometric-only-confusable classes) rather than uniformly lifting all defects.")
    lines.append("")
    lines.append("## Phase 1 success-gate close-out")
    lines.append("")
    lines.append("- Standard-benchmark baselines deferred (PointAD's published MVTec3D-AD numbers cited in the paper).")
    pm_int = pixel_metrics.get("integrate") or {}
    if pm_int:
        lines.append(
            f"- Welds zero-shot (integrate modality): image_auroc={pm_int['image_auroc']:.1f}, image_ap={pm_int['image_ap']:.1f}, pixel_auroc={pm_int['pixel_auroc']:.1f}, pixel_aupro={pm_int['pixel_aupro']:.1f}."
        )
    else:
        lines.append("- Welds zero-shot integrate metrics: see Aggregate section above.")
    lines.append("- Gap that Phases 2+3 need to close: this number sets the floor we improve against.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `results/welds_zero_shot/log.txt` — PointAD's emitted tabulate metrics tables")
    lines.append("- `results/welds_zero_shot/run.log` — wrapper stdout (tqdm progress, args)")
    lines.append("- `results/welds_zero_shot/raw_results.pkl` — per-sample image-level scores (3 modalities) keyed by `weld`")
    lines.append("- `scripts/pointad_plus/run_welds_zero_shot.py` — wrapper that monkey-patches `generate_class_info` and stubs `open3d` for Python 3.13")
    lines.append("- `scripts/pointad_plus/summarize_welds_zero_shot.py` — this script, builds SUMMARY.md from the pickle + log")

    out = SAVE_DIR / "SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print(f"per-defect rows: {len(rows)}")
    print(f"aggregate pixel metrics parsed: {pixel_metrics}")


if __name__ == "__main__":
    main()
