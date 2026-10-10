"""Summarize the rendering-subset ablation (size-matched protocol).

Reads results/modality_ablation/ablation.pkl written by run_welds_modality_ablation
(this project's own output). Reports photometric and integrate image-AUROC per subset,
aggregate and per class, plus paired-bootstrap CIs vs the all-five Global MPF.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
with open(REPO / "results/modality_ablation/ablation.pkl", "rb") as fh:
    a = pickle.load(fh)
y = np.asarray(a["gt"], int)
sp = np.asarray(a["species"])
pt = np.asarray(a["point"])
CLASSES = ["pseudo_soldering", "pinhole", "pit", "burst", "fish_scale_welding", "bump", "combined"]
SHORT = ["pseudo", "pinh", "pit", "burst", "fish", "bump", "comb"]
ORDER = ["all"] + [f"only_{m}" for m in ("lut", "phong", "diffuse", "specular", "normal")] + \
        [f"drop_{m}" for m in ("lut", "phong", "diffuse", "specular", "normal")]


def auc(s, m=None):
    m = np.ones_like(y, bool) if m is None else m
    return 100 * roc_auc_score(y[m], s[m])


def paired(sa, sb, m=None, n_boot=1000, seed=0):
    idx_all = np.nonzero(np.ones_like(y, bool) if m is None else m)[0]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = idx_all[rng.integers(0, len(idx_all), len(idx_all))]
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    return 100 * np.percentile(diffs, 2.5), 100 * np.percentile(diffs, 97.5)


for variant in ("ln", "raw"):
    print(f"\n=== fusion = {variant} (photometric O-R | integrate O-R) ===")
    print(f"{'subset':16s} {'agg':>6s} {'int':>6s} " + " ".join(f"{s:>6s}" for s in SHORT))
    for name in ORDER:
        s = np.asarray(a[f"{variant}:{name}"])
        integ = 0.5 * (s + pt)
        per = [auc(s, (sp == "good") | (sp == c)) for c in CLASSES]
        print(f"{name:16s} {auc(s):6.2f} {auc(integ):6.2f} " + " ".join(f"{v:6.2f}" for v in per))

print("\npoint branch only:", f"{auc(pt):.2f}", " pseudo:", f"{auc(pt, (sp == 'good') | (sp == 'pseudo_soldering')):.2f}")

base = np.asarray(a["ln:all"])
print("\n=== paired CI (LN fusion, photometric): subset - all ===")
for name in ORDER[1:]:
    s = np.asarray(a[f"ln:{name}"])
    for label, m in (("agg", None), ("pseudo", (sp == "good") | (sp == "pseudo_soldering"))):
        lo, hi = paired(s, base, m)
        d = auc(s, m) - auc(base, m)
        print(f"  {name:16s} {label:6s} {d:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]{' *' if lo > 0 or hi < 0 else ''}")
