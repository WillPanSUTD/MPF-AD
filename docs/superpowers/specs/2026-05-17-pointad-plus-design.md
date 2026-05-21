# PointAD+ — Multi-Photometric Zero-Shot 3D Anomaly Detection on Continuous Surfaces

**Date:** 2026-05-17
**Status:** Draft — pending user review
**Authors of this spec:** willpan + Claude
**Related work tracked at:** `idea-stage/{REF_PAPER_SUMMARY,LIT_SURVEY,COMPETITOR_ANALYSIS}.md`
**Memory:** `[[project-pointad-followup-angle]]`, `[[project-3d-ad-extension-complete]]`

---

## 1. Goal and contributions

A follow-up paper to PointAD (NeurIPS 2024, arXiv 2410.00320). Three sub-contributions ship together:

1. **Multi-Photometric Fusion (MPF):** A drop-in CLIP-feature-level fusion module that takes 5 deterministically-rendered photometric channels (LUT, Phong, Diffuse, Specular, Normal) and produces a single token map for PointAD's prompt-learning head via cross-attention. Replaces PointAD's single plug-and-play RGB pathway.
2. **Template-Free Sliding-Window Inference (SW):** An inference path that handles continuous-surface scans (welds, free-form castings) without resizing or assuming a canonical object instance. PointAD's stock inference resizes/center-crops; ours preserves native resolution by patch-level inference + overlap-mean aggregation.
3. **Welds 3D-AD Benchmark + Cross-Dataset Eval:** First published zero-shot 3D-AD result on welds, using our `Dataset_3D/MVTec3D_Weld/` (built in `[[project-3d-ad-extension-complete]]`) plus a cross-dataset matrix against MVTec3D-AD, Eyecandies, Real3D-AD.

The lane has been verified clear of overlap with the four closest competitors (PointAD+ arXiv 2509.03277, MVP-PCLIP, GS-CLIP, AFRD). See `idea-stage/COMPETITOR_ANALYSIS.md`.

---

## 2. Architecture overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Input: organized depth (H×W float32) + M photometric renders            │
│  (M=5 for welds, 6 for Eyecandies, 1 for MVTec3D-AD / Real3D-AD)         │
│  1:1 pixel ↔ point correspondence via organized XYZ tiff                 │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │  (Phase 3 only) sliding 256×256, stride 128
┌────────────────────────────▼─────────────────────────────────────────────┐
│  Multi-Photometric CLIP Fusion (MPF — NEW, Phase 2)                      │
│  • Per-modality CLIP-ViT-L encode, frozen backbone, M forward passes     │
│  • Cross-attention over the M token sequences (learnable, ~2M params)    │
│  • Output: fused N × D map (N = CLIP patch tokens) for the prompt head   │
│  • Degenerates to identity when M == 1 (no regression on single-RGB)     │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│  PointAD's hybrid prompt head (UNCHANGED from upstream)                  │
│  • Learnable object + anomaly prompts                                    │
│  • 3D/2D joint loss on auxiliary categories                              │
│  • Per-token anomaly scores → pixel + point maps                         │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│  Inference aggregation (Phase 3 only)                                    │
│  • Per-patch anomaly map → image-level via overlap-mean (50% overlap)    │
│  • Sample-level score = max over per-pixel scores (PointAD convention)   │
└──────────────────────────────────────────────────────────────────────────┘
```

The base CLIP-ViT-L backbone stays frozen as in PointAD. New learnable parameters are confined to:
- The cross-attention block inside MPF (~2M parameters).
- PointAD's existing learnable prompts and lightweight projection heads (unchanged from upstream).

---

## 3. Phase 1 — Infrastructure and baseline reproduction

### 3.1 Fork PointAD

`git clone https://github.com/zqhang/PointAD.git external/PointAD/` at a pinned commit (record SHA in repo README). Never push back. All Phase 2/3 modifications live in `external/PointAD/src/` as branch-local edits.

### 3.2 Datasets

| Dataset | Location | Source | Notes |
|---|---|---|---|
| MVTec3D-AD | `external/datasets/mvtec_3d_anomaly_detection/` | MVTec official download (~9 GB) | 10 categories, organized XYZ + RGB |
| Eyecandies | `external/datasets/eyecandies/` | eyecan-ai official (~5 GB) | 10 categories, 6 lights + depth + normal |
| Real3D-AD | `external/datasets/real3d_ad/` | unpack `F:/dataset/Real3D-AD-{PLY,PCD}.zip` | 12 categories, 4 templates + per-point labels |
| Welds 3D | `external/datasets/welds_3d/` | symlink or copy from `Dataset_3D/MVTec3D_Weld/` | 1 category "weld", 6 defect-class subfolders + combined |

### 3.3 Dataset adapters

Under `src/datasets/`, one adapter per dataset. Each emits PointAD's expected JSON manifest:

```json
{
  "train": [{"id": "...", "image": "...", "pcd": "...", "rgb": [...], "class": "..."}],
  "test":  [{"id": "...", "image": "...", "pcd": "...", "rgb": [...], "gt_mask": "...", "anomaly": true|false}]
}
```

For multi-photometric datasets (Eyecandies, welds), the `rgb` field is a list of paths to N channels rather than a single path; PointAD's loader needs a corresponding patch to handle the list case (deferred to Phase 2 — Phase 1 uses single Phong only for welds).

The welds adapter additionally generates colored PCDs (XYZ + Phong RGB per point) by combining `Dataset_3D/MVTec3D_Weld/{rgb,xyz}/` per sample, since PointAD expects colored point clouds.

### 3.4 Baseline reproduction

Run upstream `train.sh` + `test.sh` on MVTec3D-AD, Eyecandies, Real3D-AD with PointAD's default config and pretrained CLIP-ViT-L checkpoint. Record P-AUROC, AUPRO, I-AUROC. Match published numbers within ±1% per dataset (averaged across categories).

If the match fails on any dataset, log the gap, re-check config and checkpoint, then if still stuck file a GitHub Issue on `zqhang/PointAD`. Do not advance to Phase 2 until baselines match (or the gap is reproducible and documented).

### 3.5 Welds zero-shot baseline

Load checkpoint trained on MVTec3D-AD auxiliary categories → eval on welds via the Phase 3.3 adapter (single Phong channel only, no sliding window). This is the "PointAD baseline on welds" number — expected to be poor; it establishes the gap that Phase 2 + 3 will close.

### Phase 1 success gate
- Baselines match published numbers within ±1% averaged P-AUROC per standard dataset.
- A reproducible welds-zero-shot number exists (regardless of value).

---

## 4. Phase 2 — Multi-Photometric Fusion module

### 4.1 Notation

- `M` = number of photometric modalities for this sample (1 for MVTec3D-AD / Real3D-AD; 5 for welds; 6 for Eyecandies).
- `N` = CLIP patch-token count (fixed by the CLIP-ViT-L vision encoder; 256 for 224×224 input).
- `D` = CLIP embedding dim (1024 for ViT-L/14).
- `B` = batch.

### 4.2 Module interface

```python
class MultiPhotometricFusion(nn.Module):
    """Drop-in replacement for PointAD's single-RGB plug-and-play branch."""

    def __init__(self, clip_dim: int = 1024, num_heads: int = 8,
                 num_modalities_max: int = 6):
        # cross-attention block + learned modality-type embedding bank

    def forward(self, channels: list[Tensor]) -> Tensor:
        # channels[i] : (B, 3, H, W) — one photometric render, M = len(channels)
        # returns      : (B, N, D) fused token map
        # degenerates to identity (single-pass CLIP) when M == 1
```

### 4.3 Forward pass

1. Each `channels[i]` is encoded by the frozen CLIP-ViT-L vision encoder → token map `T_i ∈ ℝ^(B × N × D)`.
2. Add a learned modality-type embedding to each `T_i` (one of 6 reserved slots: lut/phong/diffuse/specular/normal/depth).
3. Stack along a new modality axis: `T = stack([T_0, ..., T_{M-1}], dim=1) ∈ ℝ^(B × M × N × D)`.
4. Cross-attention applied **along the modality axis per token position**: for each token `n`, attend over the `M` modalities (queries = keys = values = `T[:, :, n, :]`). Pool the `M` attended modality vectors back to one per token (mean or first-modality query, decided empirically). Output: `T_fused ∈ ℝ^(B × N × D)`.
5. Return `T_fused` — same shape as PointAD's single-image branch output.

### 4.4 Integration

- New file `external/PointAD/src/models/photometric_fusion.py` for the module.
- Patch `external/PointAD/src/models/pointad_model.py` so the visual pathway dispatches to `MultiPhotometricFusion` when the sample's `rgb` field is a list of paths.
- Backwards-compatible: single-RGB datasets (MVTec3D-AD, Real3D-AD) take the M=1 path; the cross-attention is a no-op pass-through (single-query attention over a single key).

### 4.5 Training

Same recipe as PointAD upstream:
- Frozen CLIP backbone, train only prompts + fusion block + projection heads.
- Auxiliary-category protocol (PointAD's default split).
- Loss: PointAD's existing binary cross-entropy on anomaly tokens, no new terms.
- Optimizer/schedule: copy upstream.
- Per-dataset configs in `external/PointAD/configs/`.

### 4.6 Ablations (table rows in Section 6)

1. **Baseline:** PointAD as-is.
2. **+ MPF (ours, headline):** cross-attention fusion over N channels.
3. **MPF-Lite:** linear channel-mixer at the CLIP *input* (single CLIP forward pass).
4. **No-learning fusion:** average N per-modality anomaly maps at the output level.
5. **Single modality:** drop all but Phong, run baseline — control for "the gain comes from extra training, not from fusion".

### Phase 2 success gate
- Ablation #2 beats #1 by ≥3 absolute P-AUROC points on Eyecandies (where Eyecandies' 6 lights provide multi-photometric input).
- On MVTec3D-AD and Real3D-AD, #2 stays within ±0.5 P-AUROC of #1 (the M=1 degenerate path).

---

## 5. Phase 3 — Template-free sliding-window inference + welds eval

### 5.1 Sliding-window inference

```python
def sliding_window_score(image, model, patch_size=256, stride=128):
    H, W = image.shape[-2:]
    aggregated = zeros((H, W))
    coverage = zeros((H, W))
    for v in range(0, H - patch_size + 1, stride):
        for u in range(0, W - patch_size + 1, stride):
            patch_anomaly = model(image[..., v:v+patch_size, u:u+patch_size])
            aggregated[v:v+patch_size, u:u+patch_size] += patch_anomaly
            coverage[v:v+patch_size, u:u+patch_size] += 1
    return aggregated / clip(coverage, 1, None)
```

Image-level score = max over per-pixel scores (PointAD convention).

Pixels not covered by any patch (small strips at the bottom/right edge when image dims aren't divisible by stride) get score 0 (no anomaly), since they have no valid signal.

### 5.2 Integration

- New file `external/PointAD/src/inference/sliding_window.py`.
- Add `--continuous_surface` flag to `test.sh`. When enabled, replaces PointAD's stock resize/center-crop with the sliding-window path.
- When disabled, behavior is unchanged (used for MVTec3D-AD / Real3D-AD standard eval where input size is already CLIP-compatible).

### 5.3 Welds zero-shot eval protocol

- Each test image: the full source `.tif` (640×{512,526,551,640}, varies per source).
- Sliding window with `patch_size=256`, `stride=128`. For 640×551 image: 4 rows × 3 cols = 12 patches.
- Aggregate per-patch maps; image-level score from aggregated map.
- Pixel-level GT: SAM2 mask from `Dataset_3D/MVTec3D_Weld/test/<class>/gt/<id>.png` (depth-threshold mask `gt_geom/` reported as a sensitivity-analysis supplementary).
- Three training sources tested: MVTec3D-AD aux, Eyecandies aux, Real3D-AD aux.

### 5.4 Metrics
- **Primary:** P-AUROC (pixel-level) and AUPRO (pixel-level).
- **Secondary:** I-AUROC (image-level).
- **Per-class breakdown** for welds: P-AUROC per defect class (`pseudo_soldering`, `pinhole`, `pit`, `burst`, `fish_scale_welding`, `bump`, `combined`).

### Phase 3 success gate
- PointAD+ (Phase 2 fusion + Phase 3 sliding window) beats PointAD baseline by ≥5 absolute P-AUROC points on welds in at least 2 of 3 train-aux scenarios.

---

## 6. Experimental matrix

### 6.1 Headline table — main result

| Method | MVTec3D-AD | Eyecandies | Real3D-AD | Welds (ZS from MVTec3D) |
|---|---|---|---|---|
| PointAD (baseline, ours rerun) | ⬜ | ⬜ | ⬜ | ⬜ |
| PointAD + SW only | — | — | — | ⬜ |
| PointAD + MPF only | — | ⬜ | — | ⬜ |
| **PointAD+ (full, MPF + SW)** | ⬜ | ⬜ | ⬜ | ⬜ |
| MPF-Lite (channel-mixer) | — | ⬜ | — | ⬜ |
| Phong-only (single modality) | — | ⬜ | — | ⬜ |
| Late-score-fusion (no learned) | — | ⬜ | — | ⬜ |

`⬜` = expected number. `—` = N/A by construction (e.g., MPF needs multi-photometric input).

### 6.2 Welds cross-source table — generalization

| Train aux on | PointAD baseline | PointAD+ (ours) |
|---|---|---|
| MVTec3D-AD | ⬜ | ⬜ |
| Eyecandies | ⬜ | ⬜ |
| Real3D-AD | ⬜ | ⬜ |

### 6.3 Per-class welds (supplementary)
P-AUROC per `test/<class>/` subfolder. 7 rows × 2 columns (baseline vs ours).

### 6.4 Compute budget
- ~36 train+eval runs total.
- PointAD's auxiliary-category training is fast (≈2 h per dataset per config on RTX 4090 per upstream reports).
- Total: ~80 GPU-hours ≈ 3–4 days of solid runtime. Fits the "minimum viable" budget.
- Phase 1 baseline reproduction: ~10 runs, ~20 GPU-h.
- Phase 2 ablations: ~10 runs, ~20 GPU-h.
- Phase 3 welds eval: ~16 runs (including cross-source), ~40 GPU-h.

---

## 7. Out of scope (deferred)

- **True multi-light Eyecandies-style re-rendering** of welds (use existing renders only).
- **Few-shot or full-shot training** (zero-shot only).
- **Loss function modifications** (no new loss terms — keeps the contribution focused on the fusion module).
- **View-adaptive rendering / camera-pose search.**
- **CLIP-free distillation** (DINOv2/Dinomaly2 path tracked in related work, not implemented).
- **Real3D-AD coordinate-frame fix** (templates patch-local vs test global) — documented limitation, our welds Real3D PCDs ship as-is per `[[project-3d-ad-extension-complete]]`.
- **Paper-writing pipeline.** Spec covers experiments; paper drafting (intro / related work / figures / tables generation) is a separate later effort using the existing `paper-writing` skill workflow.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Baseline reproduction off by >1% on standard datasets | Pin upstream commit, copy exact configs, file Issue on `zqhang/PointAD` if persistent. Block Phase 2 until matched or gap is documented and reproducible. |
| Fusion module doesn't help on single-RGB datasets | Expected and documented as a no-regression target, not a primary metric. |
| Welds zero-shot baseline is *already* good (>0.85 P-AUROC), removing room for improvement | Run baseline first in Phase 1; if so, pivot to a harder subset (small-defect classes only, or `combined/` only). |
| 5× CLIP forward passes blow up 12 GB GPU memory on the 4090 | Use gradient checkpointing on the CLIP backbone; or compute the M forwards sequentially with `torch.no_grad` (CLIP is frozen). |
| Sliding-window seam artifacts at patch boundaries | Stride=128 (50% overlap) + overlap-mean aggregation strongly attenuates. Report a robustness ablation with `patch_size=384` if seams appear. |
| 3-day GPU schedule slips | Cut ablations #4 (late-score) and #5 (single-modality), keeping the cross-attention vs channel-mixer comparison. Saves ~25% compute. |
| Welds dataset has only 541 anomalous images — concern about per-class stat power | Per-class P-AUROC reported as supplementary; primary metric is aggregated across all anomalous classes. Use bootstrap CIs in the table. |
| PointAD's licence restricts our redistribution | Read the upstream LICENSE before forking; ship our modifications as a patch series + a clean `src/pointad_plus/` for our novel modules, with separate licence terms. |

---

## 9. Repo layout (proposed)

```
F:/dataset/LUT_AD_DataSet/
├── external/
│   ├── PointAD/                              # forked upstream, commit pinned
│   │   ├── src/
│   │   │   ├── models/
│   │   │   │   ├── photometric_fusion.py    # NEW (Phase 2)
│   │   │   │   └── pointad_model.py         # PATCHED
│   │   │   └── inference/
│   │   │       └── sliding_window.py        # NEW (Phase 3)
│   │   ├── train.sh
│   │   └── test.sh                          # PATCHED (--continuous_surface)
│   └── datasets/
│       ├── mvtec_3d_anomaly_detection/
│       ├── eyecandies/
│       ├── real3d_ad/
│       └── welds_3d/                        # symlink → Dataset_3D/MVTec3D_Weld/
├── src/
│   └── datasets/                             # adapters (JSON manifest emitters)
│       ├── adapter_mvtec3d.py
│       ├── adapter_eyecandies.py
│       ├── adapter_real3d.py
│       └── adapter_welds.py
├── scripts/
│   └── pointad_plus/
│       ├── download_datasets.sh
│       ├── run_baseline.sh
│       └── run_ablations.sh
└── results/
    ├── baseline/                              # raw PointAD numbers per dataset
    └── pointad_plus/                          # our method + ablations
```

---

## 10. Open questions (none blocking spec approval, decided during implementation)

- Pin specific PointAD commit SHA — pick during Phase 1 day 1.
- Exact CLIP-ViT-L checkpoint (HuggingFace vs OpenAI release) — copy whatever PointAD's `train.sh` uses.
- Whether the modality-type embeddings need a learned scale per dataset — likely no, decide empirically in Phase 2.
- Whether to keep PointAD's auxiliary-category split or merge categories for our cross-source training — follow upstream by default.
