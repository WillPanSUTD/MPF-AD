# PointAD+ Phase 2 — Multi-Photometric Fusion Module

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a cross-attention fusion block over 5 photometric CLIP encodes (LUT/Phong/Diffuse/Specular/Normal) using self-supervised modality-masking on welds train/good, then integrate at test-time to replace PointAD's single-RGB late fusion. Beat the Phase 1 baseline specifically on `pseudo_soldering` (image-AUROC 76) and `fish_scale_welding` (85).

**Architecture (locked):**

```
5 photometric channels per sample
       │
       ▼
[CLIP-ViT-L (frozen)] × 5 (one forward per modality)
       │
       ▼  M=5 token maps T_i ∈ ℝ^(B × N × D)
[+ learned modality embeddings]
       │
       ▼  T ∈ ℝ^(B × 5 × N × D)
[Cross-attention over modality axis per token] (~2M params)
       │
       ▼  fused_color ∈ ℝ^(B × N × D)
[PointAD prompt head, frozen]  → color_anomaly_map (B × H × W)
       │                            │
       ▼ (combined as before)        ▼
point_anomaly_map ← integrate ←─── fused_color_anomaly_map (replaces single-RGB branch)
```

**Phase 1 deltas folded in:**
- No editing inside `external/PointAD/` — wrapper / parallel runner pattern, same as Phase 1 Task H.
- Fusion module lives in `src/pointad_plus/`, NOT in `external/PointAD/src/models/` as the original spec sketched.
- Training data = `Dataset_3D/Eyecandies_Weld/weld/train/good/` (the sibling dataset variant we already built, which exposes all 5 PNG modalities). Welds remains "zero-shot" for anomaly detection (anomaly prompts come from shipped MVTec3D-AD-trained weights); only the fusion block sees welds during training (clean patches only, no anomaly labels).
- Phase 1 baseline numbers are in `results/welds_zero_shot/SUMMARY.md`; treat as the comparison baseline.

**Tech stack:** Python 3.13, PyTorch 2.6+cu124, CLIP-ViT-L (already placed), shipped PointAD weights at `external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth` (Phase 1 chose carrot; we keep that).

**Spec:** `docs/superpowers/specs/2026-05-17-pointad-plus-design.md` §4.

**Repo state:** Not a git repo. No commits. Pytest passes + result file existence = integration gate.

---

## File structure

```
src/pointad_plus/
├── multi_photo_welds_adapter.py     # P2-A — adapter pulling 5 modalities
├── photometric_fusion.py             # P2-B — the cross-attention module
├── ssl_modality_masking.py           # P2-C — self-supervised training loop
└── run_welds_pointad_plus.py         # P2-E — parallel test runner

tests/pointad_plus/
├── test_multi_photo_welds_adapter.py # P2-A
├── test_photometric_fusion.py        # P2-B
└── test_ssl_modality_masking.py      # P2-C

scripts/pointad_plus/
├── train_fusion.sh                   # P2-D — wraps the SSL training
└── eval_welds_pointad_plus.sh        # P2-F + P2-G — wraps the test runner

results/
└── welds_pointad_plus/
    ├── fusion_block.pt               # trained weights
    ├── SUMMARY.md                    # final per-defect table
    └── ablations/{zero_init,mean_fusion,trained}/SUMMARY.md
```

---

## Task P2-A: Multi-photometric welds adapter

**Files:**
- Create: `src/pointad_plus/multi_photo_welds_adapter.py`
- Create: `tests/pointad_plus/test_multi_photo_welds_adapter.py`
- Output: `external/datasets/welds_pointad_mp/weld/all_meta.json` + multi-photo paths per sample

**Behavior:** Walk `Dataset_3D/Eyecandies_Weld/weld/{train,validation,test}/...` and emit a JSON manifest where each sample records all 5 photometric modality paths (lut, phong, diffuse, specular, normal) plus the existing pcd / gt paths. Reuse hardlinks where the file already exists; only the manifest is new.

- [ ] **P2-A.1: Write the failing test**

`tests/pointad_plus/test_multi_photo_welds_adapter.py`:

```python
"""Tests for the multi-photometric welds adapter."""
from pathlib import Path
import json
import pytest

from src.pointad_plus import multi_photo_welds_adapter as adp


def test_eyecandies_weld_root_exists():
    p = Path("F:/dataset/LUT_AD_DataSet/Dataset_3D/Eyecandies_Weld/weld")
    assert p.is_dir()
    # confirms the 5 modality subdirs exist on at least one sample
    sub = next((p / "train" / "good").iterdir())  # first modality subdir
    assert sub.name in {"lut", "phong", "diffuse", "specular", "normal", "depth", "gt"}


def test_sample_has_five_modality_paths(tmp_path):
    out = tmp_path / "all_meta.json"
    adp.emit_manifest(
        source_root=Path("F:/dataset/LUT_AD_DataSet/Dataset_3D/Eyecandies_Weld/weld"),
        out_path=out,
    )
    meta = json.loads(out.read_text())
    assert "train" in meta and "test" in meta
    train_weld = meta["train"]["weld"]
    assert len(train_weld) > 100
    s = train_weld[0]
    for k in ("lut", "phong", "diffuse", "specular", "normal"):
        assert k in s["multi_photo"]
        assert Path(s["multi_photo"][k]).exists()


def test_test_split_has_anomalous(tmp_path):
    out = tmp_path / "all_meta.json"
    adp.emit_manifest(
        source_root=Path("F:/dataset/LUT_AD_DataSet/Dataset_3D/Eyecandies_Weld/weld"),
        out_path=out,
    )
    meta = json.loads(out.read_text())
    test_weld = meta["test"]["weld"]
    anom = [s for s in test_weld if s["anomaly"]]
    assert len(anom) == 541  # matches Phase 1
    classes = {s["specie_name"] for s in anom}
    assert {"pseudo_soldering", "pinhole", "pit", "burst",
            "fish_scale_welding", "bump", "combined"}.issubset(classes)
```

- [ ] **P2-A.2: Confirm failure**

```
cd F:/dataset/LUT_AD_DataSet
python -m pytest tests/pointad_plus/test_multi_photo_welds_adapter.py -v
```

Expected: ImportError on `multi_photo_welds_adapter`.

- [ ] **P2-A.3: Implement**

`src/pointad_plus/multi_photo_welds_adapter.py`:

```python
"""Multi-photometric welds adapter for PointAD+.

Walks Dataset_3D/Eyecandies_Weld/weld/{train,validation,test}/<defect>/<modality>/
and emits a single all_meta.json that lists, per sample, the paths to all 5
photometric modalities (lut, phong, diffuse, specular, normal). The
`multi_photo` field is new — it does not exist in stock PointAD manifests.
The downstream test runner consumes this field instead of `d2_img_path`.

Usage:
    python -m src.pointad_plus.multi_photo_welds_adapter \
        --source Dataset_3D/Eyecandies_Weld/weld \
        --out external/datasets/welds_pointad_mp/weld/all_meta.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

DEFECT_CLASSES = (
    "pseudo_soldering", "pinhole", "pit", "burst",
    "fish_scale_welding", "bump", "combined",
)
PHOTO_MODALITIES = ("lut", "phong", "diffuse", "specular", "normal")


def _samples_in_subfolder(weld_root: Path, split: str, defect: str, anomaly: bool) -> List[dict]:
    sub = weld_root / split / defect
    phong_dir = sub / "phong"
    if not phong_dir.is_dir():
        return []
    out: List[dict] = []
    for png in sorted(phong_dir.glob("*.png")):
        stem = png.stem
        sample = {
            "id": f"{split}_{defect}_{stem}",
            "multi_photo": {m: str(sub / m / f"{stem}.png") for m in PHOTO_MODALITIES},
            "gt": str(sub / "gt" / f"{stem}.png") if anomaly else None,
            "cls_name": "weld",
            "specie_name": defect,
            "anomaly": anomaly,
        }
        # sanity: each photometric path must exist
        if not all(Path(p).exists() for p in sample["multi_photo"].values()):
            continue
        out.append(sample)
    return out


def emit_manifest(source_root: Path, out_path: Path) -> None:
    train = _samples_in_subfolder(source_root, "train", "good", anomaly=False)
    # fold validation/good into train for fusion-block training (more data)
    train += _samples_in_subfolder(source_root, "validation", "good", anomaly=False)
    test = _samples_in_subfolder(source_root, "test", "good", anomaly=False)
    for dc in DEFECT_CLASSES:
        test += _samples_in_subfolder(source_root, "test", dc, anomaly=True)

    payload = {"train": {"weld": train}, "test": {"weld": test}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    emit_manifest(args.source, args.out)
    print(f"[multi_photo_welds_adapter] wrote {args.out}")


if __name__ == "__main__":
    _main()
```

- [ ] **P2-A.4: Run tests**

```
python -m pytest tests/pointad_plus/test_multi_photo_welds_adapter.py -v
```

Expected: 3 passed.

- [ ] **P2-A.5: Smoke-run on full data**

```bash
python -m src.pointad_plus.multi_photo_welds_adapter \
  --source F:/dataset/LUT_AD_DataSet/Dataset_3D/Eyecandies_Weld/weld \
  --out F:/dataset/LUT_AD_DataSet/external/datasets/welds_pointad_mp/weld/all_meta.json
python -c "import json; m = json.load(open('F:/dataset/LUT_AD_DataSet/external/datasets/welds_pointad_mp/weld/all_meta.json')); print('train weld:', len(m['train']['weld']), 'test weld:', len(m['test']['weld']))"
```

Expected output: `train weld: 867 test weld: 694` (714 train_good + 153 val_good = 867 train; 153 test_good + 541 anomalous = 694 test).

---

## Task P2-B: MultiPhotometricFusion module

**Files:**
- Create: `src/pointad_plus/photometric_fusion.py`
- Create: `tests/pointad_plus/test_photometric_fusion.py`

- [ ] **P2-B.1: Write failing tests**

`tests/pointad_plus/test_photometric_fusion.py`:

```python
"""Tests for the multi-photometric fusion block."""
import torch
import pytest

from src.pointad_plus.photometric_fusion import MultiPhotometricFusion


def test_forward_shape_single_modality():
    """M=1 degenerate path: output matches input single token map."""
    mod = MultiPhotometricFusion(clip_dim=64, num_heads=4)
    tokens = [torch.randn(2, 16, 64)]
    out = mod(tokens)
    assert out.shape == (2, 16, 64)


def test_forward_shape_multi_modality():
    mod = MultiPhotometricFusion(clip_dim=64, num_heads=4)
    tokens = [torch.randn(2, 16, 64) for _ in range(5)]
    out = mod(tokens)
    assert out.shape == (2, 16, 64)


def test_mean_init_approximates_mean_fusion():
    """When initialized to identity attention, output ≈ mean of inputs."""
    torch.manual_seed(0)
    mod = MultiPhotometricFusion(clip_dim=64, num_heads=4, init="mean")
    tokens = [torch.randn(1, 8, 64) for _ in range(5)]
    out = mod(tokens)
    target = torch.stack(tokens).mean(dim=0)
    cos = torch.nn.functional.cosine_similarity(
        out.reshape(-1, 64), target.reshape(-1, 64)
    ).mean()
    # Cosine similarity should be close to 1 at mean init
    assert cos > 0.95


def test_param_count_under_5M():
    """Sanity: fusion block is small (~2M params per spec)."""
    mod = MultiPhotometricFusion(clip_dim=1024, num_heads=8)
    n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    assert n < 5_000_000
```

- [ ] **P2-B.2: Confirm failure**

- [ ] **P2-B.3: Implement**

`src/pointad_plus/photometric_fusion.py`:

```python
"""Cross-attention fusion of per-modality CLIP token maps.

Drop-in replacement for PointAD's single-RGB plug-and-play branch. Takes
M token maps (one per photometric modality, all coming from the same CLIP
backbone), adds learned modality-type embeddings, and applies a single
multi-head cross-attention block along the modality axis per token
position. Output: a single fused token map with the same shape as any
input.

The cross-attention uses Query = each token's own modality, Key/Value =
all modalities for that token, then averages across modalities to produce
the fused token. With `init="mean"`, weights are initialised so the
attention output is approximately the mean of inputs — useful as a
zero-shot starting point before any training.
"""
from __future__ import annotations

from typing import List, Literal

import torch
from torch import nn

MAX_MODALITIES = 6  # lut, phong, diffuse, specular, normal, depth (reserve slot)


class MultiPhotometricFusion(nn.Module):
    def __init__(
        self,
        clip_dim: int = 1024,
        num_heads: int = 8,
        num_modalities_max: int = MAX_MODALITIES,
        init: Literal["mean", "random"] = "mean",
    ) -> None:
        super().__init__()
        self.clip_dim = clip_dim
        self.modality_embed = nn.Embedding(num_modalities_max, clip_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=clip_dim, num_heads=num_heads, batch_first=True
        )
        self.ln = nn.LayerNorm(clip_dim)
        if init == "mean":
            nn.init.zeros_(self.modality_embed.weight)
            # MultiheadAttention has out_proj.bias initialized to 0 by default;
            # zero out out_proj.weight too so output ≈ mean-pooled values.
            nn.init.zeros_(self.attn.out_proj.weight)

    def forward(self, channels: List[torch.Tensor]) -> torch.Tensor:
        """
        channels: list of M tensors, each (B, N, D).
        returns:  (B, N, D) fused token map.
        """
        if len(channels) == 1:
            return channels[0]
        B, N, D = channels[0].shape
        M = len(channels)
        # Stack: (B, M, N, D)
        stacked = torch.stack(channels, dim=1)
        # Add modality embedding
        ids = torch.arange(M, device=stacked.device)
        emb = self.modality_embed(ids).view(1, M, 1, D)  # (1, M, 1, D)
        stacked = stacked + emb
        # Reshape for per-token attention: (B*N, M, D)
        permuted = stacked.permute(0, 2, 1, 3).contiguous()  # (B, N, M, D)
        flat = permuted.view(B * N, M, D)
        # Self-attention along modality axis
        attended, _ = self.attn(flat, flat, flat)  # (B*N, M, D)
        # Mean-pool across modalities
        pooled = attended.mean(dim=1)  # (B*N, D)
        fused = pooled.view(B, N, D)
        # Residual + LN (identity-friendly when out_proj is zero-inited)
        anchor = torch.stack(channels, dim=0).mean(dim=0)  # (B, N, D) mean of inputs
        return self.ln(anchor + fused)
```

- [ ] **P2-B.4: Run tests — expect pass**

```
python -m pytest tests/pointad_plus/test_photometric_fusion.py -v
```

Expected: 4 passed. If `test_mean_init_approximates_mean_fusion` fails, adjust the residual / init logic so the mean-init output approximates the mean of inputs.

---

## Task P2-C: Self-supervised modality-masking training script

**Files:**
- Create: `src/pointad_plus/ssl_modality_masking.py`
- Create: `tests/pointad_plus/test_ssl_modality_masking.py`

**Training objective:** for each batch of welds train/good patches:
1. Encode all 5 modalities through CLIP-ViT-L (frozen) → 5 token maps.
2. Forward through `MultiPhotometricFusion` with ALL 5 modalities → `fused_full` (B, N, D). Detach as target.
3. Drop one random modality (M=4) → forward → `fused_dropped` (B, N, D).
4. Loss: `||fused_dropped - fused_full||_2^2` mean over tokens.

Cosine-distance loss is also acceptable and sometimes more stable; choose during implementation.

This makes the fusion block learn to produce consistent fused tokens even when one modality is missing — equivalent to learning a robust cross-modal aggregation.

- [ ] **P2-C.1: Write failing test**

```python
"""Tests for the SSL modality-masking trainer."""
import torch
import pytest

from src.pointad_plus.photometric_fusion import MultiPhotometricFusion
from src.pointad_plus.ssl_modality_masking import masking_loss


def test_masking_loss_returns_scalar():
    mod = MultiPhotometricFusion(clip_dim=32, num_heads=4)
    channels = [torch.randn(2, 8, 32) for _ in range(5)]
    loss = masking_loss(mod, channels, dropout_seed=0)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_masking_loss_is_zero_when_all_inputs_equal():
    """If all 5 modalities are identical, dropping one shouldn't change the fused output."""
    mod = MultiPhotometricFusion(clip_dim=32, num_heads=4, init="mean")
    same = torch.randn(2, 8, 32)
    channels = [same.clone() for _ in range(5)]
    loss = masking_loss(mod, channels, dropout_seed=0)
    assert loss.item() < 1e-4
```

- [ ] **P2-C.2: Implement**

`src/pointad_plus/ssl_modality_masking.py`:

```python
"""Self-supervised modality-masking training for the photometric fusion block.

For each batch:
  full     = fusion(M=5 modalities)
  dropped  = fusion(M=4, one random modality removed)
  loss     = ||dropped - full.detach()||_2^2  (mean over tokens)

Inputs are precomputed CLIP-ViT-L token maps (frozen backbone). The fusion
block sees 5 token-map tensors and learns to be invariant to which 4 of 5
modalities are present.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.pointad_plus.photometric_fusion import MultiPhotometricFusion


def masking_loss(
    model: MultiPhotometricFusion,
    channels: List[torch.Tensor],
    dropout_seed: int,
) -> torch.Tensor:
    """L2 between fusion-with-all-5 (target) and fusion-with-4-of-5 (prediction)."""
    rng = random.Random(dropout_seed)
    drop_idx = rng.randrange(len(channels))
    with torch.no_grad():
        fused_full = model(channels).detach()
    remaining = [c for i, c in enumerate(channels) if i != drop_idx]
    fused_dropped = model(remaining)
    return F.mse_loss(fused_dropped, fused_full)


class CLIPTokenDataset(Dataset):
    """Reads precomputed CLIP token maps from disk.

    Expects directory layout:
      <root>/<sample_id>/lut.pt
      <root>/<sample_id>/phong.pt
      ...
    Each .pt is a (N, D) tensor (single-sample CLIP token map).
    """
    def __init__(self, root: Path):
        self.root = Path(root)
        self.sample_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        self.modalities = ("lut", "phong", "diffuse", "specular", "normal")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        d = self.sample_dirs[idx]
        return [torch.load(d / f"{m}.pt") for m in self.modalities]


def precompute_clip_tokens(
    manifest_json: Path,
    out_root: Path,
    clip_model,  # passed in from the wrapper
    clip_preprocess,
    device: str = "cuda:0",
) -> None:
    """One-time pass: encode each modality of each train sample through CLIP,
    save token map to disk. Trades disk for repeated CLIP forwards during
    training.

    Wrapper is responsible for loading PointAD's CLIP backbone (frozen).
    """
    import json
    from PIL import Image
    meta = json.loads(Path(manifest_json).read_text())
    train_samples = meta["train"]["weld"]
    out_root.mkdir(parents=True, exist_ok=True)
    for s in train_samples:
        d = out_root / s["id"]
        if d.exists() and len(list(d.glob("*.pt"))) == 5:
            continue
        d.mkdir(exist_ok=True)
        for m, path in s["multi_photo"].items():
            img = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                # NOTE: this depends on PointAD's CLIP wrapper; we want the
                # per-patch token map, not the pooled output. The wrapper
                # script (Task P2-E groundwork) handles the actual extraction
                # and writes (N, D) tensors here.
                tokens = clip_model.encode_image_with_tokens(img).cpu()  # placeholder API
            torch.save(tokens.squeeze(0), d / f"{m}.pt")


def train(
    token_root: Path,
    out_ckpt: Path,
    clip_dim: int = 1024,
    num_heads: int = 8,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda:0",
) -> None:
    ds = CLIPTokenDataset(token_root)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda x: [torch.stack([s[i] for s in x]) for i in range(5)])
    model = MultiPhotometricFusion(clip_dim=clip_dim, num_heads=num_heads,
                                    init="mean").to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    step = 0
    for epoch in range(epochs):
        for batch in dl:
            batch = [b.to(device) for b in batch]
            loss = masking_loss(model, batch, dropout_seed=step)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step % 20 == 0:
                print(f"[ssl] epoch={epoch} step={step} loss={loss.item():.6f}")
            step += 1
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    print(f"[ssl] saved fusion checkpoint to {out_ckpt}")


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True,
                    help="Directory of precomputed CLIP token maps (per-sample subdirs)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    train(args.tokens, args.out, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    _main()
```

- [ ] **P2-C.3: Run unit tests**

```
python -m pytest tests/pointad_plus/test_ssl_modality_masking.py -v
```

Expected: 2 passed.

---

## Task P2-D: Train the fusion block (long-running)

**Two-stage process:**

1. **Precompute CLIP tokens** for all 867 welds train/good patches × 5 modalities = 4335 CLIP forwards. This is the expensive part (~10 min on RTX 4090).
2. **Train fusion** on cached tokens (~5–10 min for 20 epochs of 867 samples).

- [ ] **P2-D.1: Wrapper script (precompute + train)**

`scripts/pointad_plus/train_fusion.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
cd F:/dataset/LUT_AD_DataSet

# Step 1: precompute CLIP tokens
python -m scripts.pointad_plus.precompute_clip_tokens \
  --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
  --out external/datasets/welds_pointad_mp/clip_tokens/

# Step 2: train fusion
python -m src.pointad_plus.ssl_modality_masking \
  --tokens external/datasets/welds_pointad_mp/clip_tokens/ \
  --out results/welds_pointad_plus/fusion_block.pt \
  --epochs 20 --batch_size 16
```

- [ ] **P2-D.2: Write the precompute helper**

`scripts/pointad_plus/precompute_clip_tokens.py`:

```python
"""Walk the manifest, encode each sample × modality through PointAD's CLIP,
save (N, D) token maps per sample under <out>/<sample_id>/<modality>.pt.

Uses the SAME CLIP wrapper PointAD uses, loaded via the shipped weights at
the hardcoded cache path. Re-uses the monkey-patches from
scripts/pointad_plus/run_welds_zero_shot.py (Phase 1) — import and reuse,
don't duplicate.
"""
# Implementation: copy the import-and-CLIP-setup block from
# scripts/pointad_plus/run_welds_zero_shot.py, then iterate the manifest's
# train samples, encode each modality, save token tensor.
```

(Full implementation deferred to the subagent — they'll lift the setup from `run_welds_zero_shot.py`.)

- [ ] **P2-D.3: Run training (background)**

```bash
bash scripts/pointad_plus/train_fusion.sh 2>&1 | tee results/welds_pointad_plus/train.log
```

Monitor loss curve — expect monotonic decrease to ~1e-3 or lower. If loss plateaus immediately or explodes, investigate.

- [ ] **P2-D.4: Verify output**

`results/welds_pointad_plus/fusion_block.pt` exists, ~ 8 MB (~2M params × 4 bytes). Final training-loss line in the log < 0.01 (MSE on normalized token maps).

---

## Task P2-E: Parallel test runner with fusion at inference

**File:** `src/pointad_plus/run_welds_pointad_plus.py`

This is the integration that produces the headline number. It mirrors Phase 1's `scripts/pointad_plus/run_welds_zero_shot.py` but with the following changes:

1. Reads the multi-photometric manifest (`welds_pointad_mp/weld/all_meta.json`).
2. For each test sample: loads all 5 photometric PNGs (lut/phong/diffuse/specular/normal), CLIP-encodes each, runs through the trained fusion block to produce `fused_color_tokens`.
3. Replaces the `color_text_probs = text_probs[-b:]` color branch in PointAD's test loop with the fused-color-token output going through the same prompt-head projection.
4. Late fusion `(color_anomaly_map + anomaly_map) / 2` unchanged — the difference is just that `color_anomaly_map` now comes from fused multi-photometric tokens instead of single Phong.
5. Saves per-defect-class anomaly scores to `results/welds_pointad_plus/raw_results.pkl`.

- [ ] **P2-E.1: Read Phase 1 runner**

Read `scripts/pointad_plus/run_welds_zero_shot.py` and `external/PointAD/test.py:127-168` to confirm the exact tensor shapes at each step. The runner must reproduce them with the multi-photometric variant.

- [ ] **P2-E.2: Implement the runner**

(Full code in Phase 1's `run_welds_zero_shot.py` as the starting template; the change is:
- Replace single-image load with 5-image load
- Replace single CLIP encode with 5 CLIP encodes
- Cross-attend via the loaded fusion block
- Use fused tokens in place of single-color tokens for the rest of the pipeline)

- [ ] **P2-E.3: Sanity test**

Run on a tiny subset (5 samples) first to verify shape correctness and that the late-fusion line executes without error. Then proceed to full eval.

---

## Task P2-F: Full evaluation on welds + per-defect breakdown

- [ ] **P2-F.1: Run full eval**

```bash
python -m src.pointad_plus.run_welds_pointad_plus \
  --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
  --fusion_ckpt results/welds_pointad_plus/fusion_block.pt \
  --pointad_ckpt external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth \
  --out results/welds_pointad_plus/
```

Background run. Expect ~15-30 min for 694 test samples on RTX 4090.

- [ ] **P2-F.2: Write SUMMARY**

`results/welds_pointad_plus/SUMMARY.md`:

```markdown
# Welds zero-shot — PointAD+ (multi-photometric fusion)

Fusion checkpoint: `results/welds_pointad_plus/fusion_block.pt`
Source PointAD checkpoint: `exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth`
Adapter: multi-photometric (5 channels: lut/phong/diffuse/specular/normal)

## Aggregate

| Modality | pixel-AUROC | pixel-AUPRO | image-AUROC | image-AP |
|---|---|---|---|---|
| point only | … | … | … | … |
| fused color (ours) | … | … | … | … |
| **integrate** | … | … | … | … |

## Per-defect (integrate)

| Defect | n | image-AUROC | vs Phase 1 |
|---|---|---|---|
| pseudo_soldering | 206 | … | Δ from 76.0 |
| fish_scale_welding | 21 | … | Δ from 85.3 |
| (all others) | … | … | … |

## Phase 2 success-gate check
- Spec target: ≥3 absolute P-AUROC improvement on welds where multi-photometric input exists. ✅ / ❌ ?
- Spec target: ±0.5 P-AUROC on Phase 1 baseline metric (no regression on aggregate). ✅ / ❌ ?
```

---

## Task P2-G: Ablation runs (cheap; one per variant)

Three ablation variants of the fusion block, each evaluated with the same runner from P2-E:

1. **Zero-init / mean-fusion (no training):** load `MultiPhotometricFusion` initialized with `init="mean"` and DO NOT load any trained weights. Saves to `results/welds_pointad_plus/ablations/mean_fusion/SUMMARY.md`.
2. **Trained:** the headline run from P2-F. Saves to `.../ablations/trained/SUMMARY.md` (or just reuse P2-F output).
3. **Single-modality (Phong only):** Phase 1 result (already in `results/welds_zero_shot/SUMMARY.md`). Copy into the ablations dir for a uniform comparison.

- [ ] **P2-G.1: Mean-fusion eval**

```bash
python -m src.pointad_plus.run_welds_pointad_plus \
  --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
  --fusion_ckpt "" \
  --fusion_init mean \
  --pointad_ckpt external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth \
  --out results/welds_pointad_plus/ablations/mean_fusion/
```

- [ ] **P2-G.2: Ablation table**

Combine all three into a single ablation row in `results/welds_pointad_plus/SUMMARY.md`:

```markdown
## Ablation: fusion variant

| Variant | pixel-AUROC | image-AUROC | pseudo_soldering AUROC |
|---|---|---|---|
| Single Phong (Phase 1) | 94.1 | 88.5 | 76.0 |
| Mean fusion (no training) | … | … | … |
| Trained fusion (ours) | … | … | … |
```

---

## Phase 2 success gates

Per the spec §4.5:
1. Trained fusion (P2-F) beats Phase 1 baseline by **≥3 absolute P-AUROC** on welds aggregate.
2. **Specifically:** pseudo_soldering image-AUROC improves from 76.0.
3. No regression worse than -0.5 P-AUROC anywhere.

If gate 2 fails (pseudo_soldering doesn't improve), Phase 2 is incomplete and we re-design the training objective (e.g., add a per-modality importance loss, or weight the masking by the missing modality's expected contribution).

If gate 1 fails but gate 2 succeeds, document and move on; the paper can use per-defect numbers as the lead, not the aggregate.

---

## Notes for the implementing engineer

- **Working directory:** `F:/dataset/LUT_AD_DataSet`.
- **No git commits.** Use pytest passes + result-file existence as the integration gate.
- **Don't edit anything inside `external/PointAD/`.** All integration is via wrapper scripts in `scripts/pointad_plus/` and `src/pointad_plus/`.
- **Phase 1 deliverables already on disk** — re-use them:
  - `scripts/pointad_plus/run_welds_zero_shot.py` (Phase 1 runner — copy the CLIP setup block)
  - `external/datasets/welds_pointad/weld/all_meta.json` (Phase 1 manifest, single-Phong)
  - `results/welds_zero_shot/SUMMARY.md` (Phase 1 numbers)
- **PointAD's CLIP wrapper is `AnomalyCLIP_lib.model_load`.** It returns the full token map via specific layer hooks. Look at how Phase 1's runner calls it for the right invocation.
- **The token-map dimensionality** comes from CLIP-ViT-L: 336/14 = 24 patches per side, so N = 24×24 = 576 tokens + 1 CLS = 577. Confirm exact shape by inspecting Phase 1's runner output.
- **Memory budget:** 5 CLIP forwards per sample × batch ≈ 5× the Phase 1 memory pressure. On RTX 4090 (24 GB), batch size 4–8 should fit comfortably. If OOM, drop batch size or use gradient checkpointing on the fusion block.
