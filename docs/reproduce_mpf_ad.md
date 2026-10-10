# Reproducing MPF-AD on MPW-AD

This guide reproduces the MPW-AD image-level results of MPF-AD (P1–P4) from a
fresh clone, using only public resources. It was verified end to end on
Windows 11, Python 3.12.7, PyTorch 2.5.1 + CUDA 12.4 and a single RTX 3070
(8 GB). Wall-clock times below are from that machine.

MPW-AD has two test protocols (both 694 samples: 153 normal + 541 anomalous):

- **Original protocol.** Normal samples are 256×256 crops, anomalous samples are
  full scans (~430–640 px). Image size alone separates the two classes, which
  inflates every method; use it only to reproduce the shortcut analysis.
- **Size-matched protocol (primary).** Every anomalous sample is cropped to the
  256×256 window centred on its ground-truth bounding box (same window for XYZ,
  renderings and masks), so all test samples have the same size.

Expected image-AUROC / AP, integrate score `s_int = (s_photo + s_point) / 2`:

| Config | Description | Size-matched | Original |
|---|---|---|---|
| P1 | PointAD, single Phong rendering | 81.82 / 93.21 | 88.60 / 96.26 |
| P2 | Global MPF (mean of 5 rendering token maps) | 83.42 / 94.29 | 90.01 / 96.96 |
| P3 | Sliding-Window MPF (256×256, stride 128) | 83.37 / 94.27 | 88.77 / 95.40 |
| P4 | Max-Gated MPF (per-sample max of P2/P3 photometric scores) | 83.42 / 94.29 | 91.62 / 97.40 |

## 1. Environment

```bash
git clone https://github.com/WillPanSUTD/MPF-AD.git
cd MPF-AD
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-mpfad.txt
```

`open3d` is not needed: the runners install a stub for it (it is only used by
PointAD's Real3D-AD branch).

## 2. PointAD (upstream, unmodified) and its checkpoint

MPF-AD imports PointAD's CLIP, prompt learner, dataset loader and metrics
without modifying them. Clone the exact commit used in the paper into
`external/PointAD/`; the prompt-learner checkpoint ships with the repository.

```bash
git clone https://github.com/zqhang/PointAD.git external/PointAD
git -C external/PointAD checkout 19e2c82a169c52a53133acbc73badb24d5ef91d2
ls external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth
```

On first use PointAD downloads OpenAI's `ViT-L-14-336px.pt` (934 MB) into a
cache path hard-coded upstream (`/remote-home/iot_zhouqihang/root/.cache/clip`).
On Windows this becomes `<drive of the repo>:\remote-home\...`. That is harmless;
to avoid the download, place the file there yourself.

## 3. Data (MPW-AD, ~5.9 GB)

The dataset is gated on Hugging Face: request access on
<https://huggingface.co/datasets/vpan1226/MPW-AD>, then `hf auth login`.

```bash
hf download vpan1226/MPW-AD --repo-type dataset --include "Dataset_3D/*" --local-dir .
python -m pytest tests/pointad_plus -q          # 15 passed
```

## 4. Original protocol

### 4.1 Convert to PointAD's input format (~1 min)

```bash
# P1 input: single Phong rendering replicated into PointAD's 9 view slots (694 test samples)
python -m src.pointad_plus.welds_to_pointad
# P2/P3 input: manifest listing the 5 renderings per sample (694 test samples)
python -m src.pointad_plus.multi_photo_welds_adapter \
    --source Dataset_3D/Eyecandies_Weld/weld \
    --out external/datasets/welds_pointad_mp/weld/all_meta.json
```

### 4.2 Inference

```bash
CKPT=external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth
MANIFEST=external/datasets/welds_pointad_mp/weld/all_meta.json

# P1 — PointAD baseline (~10 min)
python -m scripts.pointad_plus.run_welds_zero_shot

# P2 — Global MPF, parameter-free mean fusion (~24 min)
python -m src.pointad_plus.run_welds_pointad_plus --manifest $MANIFEST \
    --fusion_init mean --pointad_ckpt $CKPT \
    --out results/welds_pointad_plus/ablations/mean_fusion

# P3 — Sliding-Window MPF (~62 min)
python -m src.pointad_plus.run_welds_pointad_plus_sw --manifest $MANIFEST \
    --pointad_ckpt $CKPT --patch_size 256 --stride 128 \
    --out results/welds_pointad_plus_sw
```

Each run writes `results/<run>/raw_results.pkl` (per-sample scores) and `log.txt`
(PointAD's image- and pixel-level metrics).

### 4.3 P4 and the paper tables (seconds, CPU only)

```bash
python scripts/pointad_plus/bootstrap_ci_unified.py        # P1–P4, unified integrate, per class + 95% CIs (Table 1)
python scripts/pointad_plus/paired_and_triage.py           # paired-difference CIs, triage operating points
python scripts/pointad_plus/phase4_max_hybrid.py           # oracle and mean-hybrid references (Table 3)
python scripts/pointad_plus/calibration_analysis.py        # gate-selection rates, score histograms (Figs. F14/F15)
python scripts/pointad_plus/fusion_alternatives.py         # alternative fusion rules (Table 5)
```

`phase4_max_hybrid.py` and `bootstrap_ci.py` print PointAD's native integrate
score for P1–P3 (average the maps, then max); the paper uses the unified
`(s_photo + s_point) / 2` for every configuration, as printed by
`bootstrap_ci_unified.py` and `paired_and_triage.py`.

## 5. Size-matched protocol (primary results)

Build a second repository root whose `Dataset_3D` holds the cropped test set
(train/validation and normal test samples are linked, not copied), then run the
same pipeline there.

```bash
python scripts/pointad_plus/make_size_matched_dataset.py Dataset_3D ../MPF-AD-sm/Dataset_3D
cp -r src scripts tests ../MPF-AD-sm/
mkdir -p ../MPF-AD-sm/external && ln -s "$PWD/external/PointAD" ../MPF-AD-sm/external/PointAD
cd ../MPF-AD-sm
# then steps 4.1 and 4.2 unchanged (P1 ~10 min, P2 ~26 min, P3 ~26 min), and 4.3 for the tables
```

On Windows, create the PointAD link with `mklink /J`. Under this protocol P3 sees
one window per sample, so P3 ≈ P2 and P4 = P2 exactly.

### 5.1 Rendering-subset ablation (~15 min)

Encodes the five renderings once per sample and scores the photometric branch
for all five, each rendering alone and each leave-one-out subset (raw mean and
LayerNorm of the mean; `ln:all` reproduces P2 exactly).

```bash
python -m src.pointad_plus.run_welds_modality_ablation --manifest $MANIFEST \
    --pointad_ckpt $CKPT --out results/modality_ablation
python scripts/pointad_plus/summarize_modality_ablation.py
```

Expected (photometric / integrate image-AUROC): all five 82.21 / 83.42,
Specular only 90.26 / 88.45, LUT only 68.21 / 84.23 (27.29 on pseudo_soldering).

## Expected deviation

A reproduction on different hardware matches the original per-sample scores to
about 1e-4 (mean absolute difference), and aggregate metrics to within ±0.02.
This comes from GPU floating-point nondeterminism, not from randomness in the
method (MPF-AD has no trainable parameters).

## Not covered here

- The SSL-trained fusion baseline (`scripts/pointad_plus/train_fusion.sh`; trained on
  the 714 + 153 normal crops of MPW-AD).
- MVTec3D-AD results (`scripts/mvtec3d/`), which need the MVTec3D-AD dataset in
  PointAD's pre-rendered format.
- Pixel-level metrics of P3/P4 (`compute_phase4_pixel_metrics.py`,
  `compute_sw_v2_pixel_metrics.py`), which need the per-pixel maps saved by
  the P2/P3 runners.
