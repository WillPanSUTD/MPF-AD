"""Write the final MVTec3D-AD benchmark SUMMARY.md from aggregate_metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v: float) -> str:
    if isinstance(v, float) and v != v:
        return "nan"
    return f"{v * 100:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--run_date", type=str, required=True)
    args = ap.parse_args()

    agg = json.loads((args.root / "aggregate_metrics.json").read_text())
    headline = agg["headline"]
    per_aux_avg = agg["per_aux_avg"]
    per_aux_per_obj = agg["per_aux_per_obj"]

    lines: list[str] = []
    lines.append("# MVTec3D-AD one-vs-rest standard benchmark")
    lines.append("")
    lines.append("Protocol: Align3D-AD §4.1 (3 aux × 9 test categories, averaged).")
    lines.append(f"Run date: {args.run_date}")
    lines.append("")
    lines.append("## Headline (averaged across 3 aux × 9 test categories)")
    lines.append("")
    lines.append("| Model | O-R | O-A | P-R | P-P |")
    lines.append("|---|---|---|---|---|")
    b = headline["baseline"]
    m = headline["max"]
    lines.append(f"| PointAD (shipped, our rerun) | {fmt(b['O-R'])} | "
                 f"{fmt(b['O-A'])} | {fmt(b['P-R'])} | {fmt(b['P-P'])} |")
    lines.append("| PointAD (Align3D-AD paper) | 82.0 | 94.2 | 95.5 | 84.4 |")
    lines.append("| Align3D-AD (Align3D-AD paper) | 83.0 | 94.3 | 95.9 | 86.5 |")
    lines.append(f"| **PointAD+ max-hybrid (ours)** | **{fmt(m['O-R'])}** | "
                 f"**{fmt(m['O-A'])}** | **{fmt(m['P-R'])}** | **{fmt(m['P-P'])}** |")
    lines.append("")
    lines.append("## Per-aux breakdown")
    lines.append("")
    lines.append("Integrate (point + color) metric — averaged across the 9 non-aux test categories.")
    lines.append("")
    lines.append("| Aux | Method | O-R | O-A | P-R | P-P |")
    lines.append("|---|---|---|---|---|---|")
    for aux in sorted(per_aux_avg.keys()):
        d = per_aux_avg[aux]
        bi = d["baseline_integrate"]
        mi = d["max_integrate"]
        lines.append(f"| {aux.capitalize()} | PointAD baseline | "
                     f"{fmt(bi['O-R'])} | {fmt(bi['O-A'])} | {fmt(bi['P-R'])} | {fmt(bi['P-P'])} |")
        lines.append(f"| {aux.capitalize()} | PointAD+ max-hybrid | "
                     f"{fmt(mi['O-R'])} | {fmt(mi['O-A'])} | {fmt(mi['P-R'])} | {fmt(mi['P-P'])} |")
    lines.append("")
    lines.append("## Per-aux all-branch (sanity, integrate-only)")
    lines.append("")
    lines.append("| Aux | baseline_point | baseline_color | baseline_int | sw_color | sw_int | max_color | max_int |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for aux in sorted(per_aux_avg.keys()):
        d = per_aux_avg[aux]
        def short(tag):
            mt = d[tag]
            return f"{fmt(mt['O-R'])}/{fmt(mt['O-A'])}/{fmt(mt['P-R'])}/{fmt(mt['P-P'])}"
        lines.append(f"| {aux} | {short('baseline_point')} | {short('baseline_color')} | "
                     f"{short('baseline_integrate')} | {short('sw_color')} | "
                     f"{short('sw_integrate')} | {short('max_color')} | {short('max_integrate')} |")
    lines.append("")
    lines.append("## Per-category detail (integrate metrics)")
    lines.append("")
    for aux in sorted(per_aux_per_obj.keys()):
        lines.append(f"### Aux = {aux}")
        lines.append("")
        lines.append("| Category | base-OR | base-OA | base-PR | base-PP | max-OR | max-OA | max-PR | max-PP |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for obj in sorted(per_aux_per_obj[aux].keys()):
            bi = per_aux_per_obj[aux][obj]["baseline_integrate"]
            mi = per_aux_per_obj[aux][obj]["max_integrate"]
            lines.append(f"| {obj} | {fmt(bi['O-R'])} | {fmt(bi['O-A'])} | "
                         f"{fmt(bi['P-R'])} | {fmt(bi['P-P'])} | "
                         f"{fmt(mi['O-R'])} | {fmt(mi['O-A'])} | "
                         f"{fmt(mi['P-R'])} | {fmt(mi['P-P'])} |")
        lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- MVTec3D-AD has M=1 (single RGB per sample), so Phase 2 multi-photometric")
    lines.append("  mean fusion **degenerates** to the PointAD baseline; only Phase 3 sliding")
    lines.append("  window contributes any lift from PointAD+ on this benchmark.")
    lines.append("- We use the shipped per-category PointAD checkpoints (epoch_15.pth) from the")
    lines.append("  Align3D-AD repo, identical to what they used.")
    lines.append("- Mean-fusion variant of MultiPhotometricFusion is used with `init=mean` (no")
    lines.append("  training — Phase 2 mean-fusion finding from the welds work).")
    lines.append("- Sliding window: patch_size=256, stride=128, on native 800x800 / 600x800 RGB.")
    lines.append("- Headline metrics are averaged over (3 aux) × (9 test categories per aux).")

    out = args.root / "SUMMARY.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
