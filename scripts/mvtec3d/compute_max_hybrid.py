"""Phase 4 max-hybrid post-processing for MVTec3D-AD.

Reads raw_results.pkl from each aux directory, computes per-sample
max(color_baseline, color_sw) anomaly maps and image-level scores, recombines
with point branch (integrate = (max_color + point) / 2), and writes per-aux
metrics plus the SUMMARY.md aggregate.

Usage:
    python compute_max_hybrid.py \
        --root results/mvtec3d_benchmark \
        --aux carrot cookie dowel
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from skimage import measure
from sklearn.metrics import auc


SIGMA = 4


def cal_pro_score(masks: np.ndarray, amaps: np.ndarray,
                  max_step: int = 200, expect_fpr: float = 0.3) -> float:
    """PointAD's cal_pro_score (vendored)."""
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    if fprs.max() == fprs.min():
        return float("nan")
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return float(pro_auc)


def four_metrics(gt_sp: list, pr_sp: list, masks: np.ndarray, maps: np.ndarray) -> dict:
    """O-R, O-A, P-R, P-P."""
    gt_sp_arr = np.array(gt_sp, dtype=np.int32)
    pr_sp_arr = np.array(pr_sp, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            o_r = float(roc_auc_score(gt_sp_arr, pr_sp_arr))
        except Exception:
            o_r = float("nan")
        try:
            o_a = float(average_precision_score(gt_sp_arr, pr_sp_arr))
        except Exception:
            o_a = float("nan")

    masks_u8 = (masks > 0.5).astype(np.uint8)
    if masks_u8.ndim == 4:
        masks_u8 = masks_u8.squeeze(1)
    if maps.ndim == 4:
        maps = maps.squeeze(1)
    try:
        p_r = float(roc_auc_score(masks_u8.ravel(), maps.ravel()))
    except Exception:
        p_r = float("nan")
    try:
        if (masks_u8 > 0).any():
            p_p = cal_pro_score(masks_u8, maps)
        else:
            p_p = float("nan")
    except Exception:
        p_p = float("nan")
    return {"O-R": o_r, "O-A": o_a, "P-R": p_r, "P-P": p_p}


def compute_max_hybrid_for_obj(d: dict) -> dict:
    """Given the per-obj dict from raw_results.pkl, compute hybrid maps and scores."""
    color_b = np.asarray(d["color_anomaly_maps_baseline"])   # (N, IS, IS) or (N, 1, IS, IS)
    color_s = np.asarray(d["color_anomaly_maps_sw"])
    if color_b.ndim == 4 and color_b.shape[1] == 1:
        color_b = color_b.squeeze(1)
    if color_s.ndim == 4 and color_s.shape[1] == 1:
        color_s = color_s.squeeze(1)
    color_max = np.maximum(color_b, color_s)                  # element-wise per-sample

    point = np.asarray(d["anomaly_maps"])
    if point.ndim == 4 and point.shape[1] == 1:
        point = point.squeeze(1)
    integrate_max = (color_max + point) / 2.0
    # per-sample gaussian smoothing for parity with baseline pipeline
    integrate_max = np.stack([gaussian_filter(m, sigma=SIGMA) for m in integrate_max], axis=0)

    # Image-level scores: take the per-sample max(color_baseline_sp, color_sw_sp)
    cb_sp = np.asarray(d["color_pr_sp_baseline"], dtype=np.float64)
    cs_sp = np.asarray(d["color_pr_sp_sw"], dtype=np.float64)
    color_sp_max = np.maximum(cb_sp, cs_sp).tolist()

    # Integrate image-level score: (max_map.max() + (point_text+color_text)/2)
    # For simplicity recompute from the smoothed integrate_max map's max +
    # average of the two integrate image-level scores' text-prob component.
    # Actually pattern matches: integrate_sp = integrate_map.max() + 0.5*(p_text+c_text)
    # Since we don't have the raw text probs split, derive from existing
    # integrate scores: integrate_sp - integrate_map.max() = text_part. Use that.
    ib_sp = np.asarray(d["integrate_pr_sp_baseline"], dtype=np.float64)
    # Use sample-wise integrate_max map maxes:
    integrate_sp_max = []
    for i in range(integrate_max.shape[0]):
        integrate_sp_max.append(float(integrate_max[i].max()) + 0.5 * (
            (cb_sp[i] - color_b[i].max()) + (cs_sp[i] - color_s[i].max())  # color_text_avg
        ) + 0.5 * 0.0)  # we don't have point_text, but it cancels: see below
    # Actually a cleaner alternative — average baseline/SW integrate sp values:
    # this gives a stable max-hybrid integrate score per sample.
    integrate_sp_max = []
    is_sp_baseline = ib_sp
    is_sp_sw = np.asarray(d["integrate_pr_sp_sw"], dtype=np.float64)
    # max-hybrid integrate sp: take per-sample max of the two integrate sp.
    integrate_sp_max = np.maximum(is_sp_baseline, is_sp_sw).tolist()

    return {
        "color_anomaly_maps_max": color_max,
        "integrate_anomaly_maps_max": integrate_max.astype(np.float32),
        "color_pr_sp_max": color_sp_max,
        "integrate_pr_sp_max": integrate_sp_max,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--aux", nargs="+", required=True)
    args = ap.parse_args()

    all_metrics: dict = {}
    for aux in args.aux:
        pkl = args.root / aux / "raw_results.pkl"
        if not pkl.is_file():
            print(f"[skip] {pkl} not found")
            continue
        with open(pkl, "rb") as f:
            res = pickle.load(f)

        per_aux: dict = {}
        for obj, d in res.items():
            extra = compute_max_hybrid_for_obj(d)

            # Compute metrics for: baseline, sw, max-hybrid.
            m_baseline_point = four_metrics(
                d["gt_sp"], d["pr_sp"],
                d["imgs_masks"], np.asarray(d["anomaly_maps"]),
            )
            m_baseline_color = four_metrics(
                d["gt_sp"], d["color_pr_sp_baseline"],
                d["imgs_masks"], np.asarray(d["color_anomaly_maps_baseline"]),
            )
            m_baseline_int = four_metrics(
                d["gt_sp"], d["integrate_pr_sp_baseline"],
                d["imgs_masks"], np.asarray(d["integrate_anomaly_maps_baseline"]),
            )
            m_sw_color = four_metrics(
                d["gt_sp"], d["color_pr_sp_sw"],
                d["imgs_masks"], np.asarray(d["color_anomaly_maps_sw"]),
            )
            m_sw_int = four_metrics(
                d["gt_sp"], d["integrate_pr_sp_sw"],
                d["imgs_masks"], np.asarray(d["integrate_anomaly_maps_sw"]),
            )
            m_max_color = four_metrics(
                d["gt_sp"], extra["color_pr_sp_max"],
                d["imgs_masks"], extra["color_anomaly_maps_max"],
            )
            m_max_int = four_metrics(
                d["gt_sp"], extra["integrate_pr_sp_max"],
                d["imgs_masks"], extra["integrate_anomaly_maps_max"],
            )
            per_aux[obj] = {
                "baseline_point": m_baseline_point,
                "baseline_color": m_baseline_color,
                "baseline_integrate": m_baseline_int,
                "sw_color": m_sw_color,
                "sw_integrate": m_sw_int,
                "max_color": m_max_color,
                "max_integrate": m_max_int,
            }
        all_metrics[aux] = per_aux

        # Save per-aux summary table.
        with open(args.root / aux / "metrics_max_hybrid.md", "w") as f:
            f.write(f"# {aux} per-category metrics\n\n")
            for obj, mm in per_aux.items():
                f.write(f"\n## {obj}\n")
                f.write("| Branch | O-R | O-A | P-R | P-P |\n|---|---|---|---|---|\n")
                for tag, m in mm.items():
                    f.write(f"| {tag} | {m['O-R']*100:.1f} | {m['O-A']*100:.1f} | "
                            f"{m['P-R']*100:.1f} | {m['P-P']*100:.1f} |\n")

    # Now aggregate across aux × cat for the headline numbers.
    # PointAD's table reports the integrate (point+color) metric.
    # Headline = avg over (3 aux × 9 test cats) of integrate metrics.

    def avg(d_list: list[dict]) -> dict:
        keys = ["O-R", "O-A", "P-R", "P-P"]
        out = {}
        for k in keys:
            vals = [d[k] for d in d_list if not (isinstance(d[k], float) and (d[k] != d[k]))]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out

    headline = {"baseline": [], "max": []}
    per_aux_headline: dict = {}
    for aux, per_aux in all_metrics.items():
        per_aux_headline[aux] = {
            "baseline_point": [], "baseline_color": [], "baseline_integrate": [],
            "sw_color": [], "sw_integrate": [],
            "max_color": [], "max_integrate": [],
        }
        for obj, mm in per_aux.items():
            for k in per_aux_headline[aux]:
                per_aux_headline[aux][k].append(mm[k])
            headline["baseline"].append(mm["baseline_integrate"])
            headline["max"].append(mm["max_integrate"])

    print("\n=== Per-aux integrate averages ===")
    for aux, d in per_aux_headline.items():
        print(f"\n[{aux}]")
        for tag, lst in d.items():
            m = avg(lst)
            print(f"  {tag:22s}: O-R {m['O-R']*100:5.1f}  O-A {m['O-A']*100:5.1f}  "
                  f"P-R {m['P-R']*100:5.1f}  P-P {m['P-P']*100:5.1f}")

    print("\n=== Headline (averaged across all aux × cat) ===")
    for tag in ("baseline", "max"):
        m = avg(headline[tag])
        print(f"  {tag:10s}: O-R {m['O-R']*100:5.1f}  O-A {m['O-A']*100:5.1f}  "
              f"P-R {m['P-R']*100:5.1f}  P-P {m['P-P']*100:5.1f}")

    # Persist JSON for SUMMARY assembly.
    import json as _json
    out_json = args.root / "aggregate_metrics.json"
    serializable = {
        aux: {
            obj: {tag: {k: v for k, v in mm.items()} for tag, mm in branches.items()}
            for obj, branches in per_aux.items()
        }
        for aux, per_aux in all_metrics.items()
    }
    with open(out_json, "w") as f:
        _json.dump({
            "per_aux_per_obj": serializable,
            "per_aux_avg": {
                aux: {tag: avg(lst) for tag, lst in d.items()}
                for aux, d in per_aux_headline.items()
            },
            "headline": {tag: avg(headline[tag]) for tag in headline},
        }, f, indent=2)
    print(f"\n[wrote] {out_json}")


if __name__ == "__main__":
    main()
