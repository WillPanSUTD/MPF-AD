# Reproducing MPF-AD on MPW-AD

This guide reproduces the MPW-AD image-level results of MPF-AD (P1–P4) from a
fresh clone, using only public resources. It was verified end to end on
Windows 11, Python 3.12.7, PyTorch 2.5.1 + CUDA 12.4 and a single RTX 3070
(8 GB). Wall-clock times below are from that machine.

| Config | Description | image-AUROC / AP (integrate) |
|---|---|---|
| P1 | PointAD, single Phong rendering | 88.51 / 96.17 |
| P2 | Global MPF (mean of 5 photometric token maps) | 89.82 / 96.84 |
| P3 | Sliding-Window MPF (256×256, stride 128) | 88.74 / 95.46 |
| P4 | Max-Gated MPF (per-sample max of P2/P3 color scores) | 91.62 / 97.40 |

All four rows are on the 694-sample matched test set (153 normal + 541 anomalous).
P1 is run on the full 847-sample set and filtered to the 694 subset by the analysis scripts.

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

## 4. Convert to PointAD's input format (~1 min)

```bash
# P1 input: single Phong rendering replicated into PointAD's 9 view slots (847 test samples)
python -m src.pointad_plus.welds_to_pointad
# P2/P3 input: manifest listing the 5 photometric renderings per sample (694 test samples)
python -m src.pointad_plus.multi_photo_welds_adapter \
    --source Dataset_3D/Eyecandies_Weld/weld \
    --out external/datasets/welds_pointad_mp/weld/all_meta.json
```

## 5. Inference

```bash
CKPT=external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth
MANIFEST=external/datasets/welds_pointad_mp/weld/all_meta.json

# P1 — PointAD baseline (~11 min)
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

Each run writes `results/<run>/raw_results.pkl` (per-sample scores) and `log.txt`.

## 6. P4 and the paper tables (seconds, CPU only)

```bash
python scripts/pointad_plus/phase4_max_hybrid.py           # P1–P4 aggregate + per-defect (Table 1), oracle
python scripts/pointad_plus/bootstrap_ci.py                # 95% bootstrap CIs (Table 1)
python scripts/pointad_plus/compute_ablation_per_defect.py # point/color/integrate per defect (Table 2)
python scripts/pointad_plus/fusion_alternatives.py         # alternative fusion rules (Table 5)
```

## Expected deviation

A reproduction on different hardware matches the original per-sample scores to
about 1e-4 (mean absolute difference), and aggregate metrics to within ±0.02.
This comes from GPU floating-point nondeterminism, not from randomness in the
method (MPF-AD has no trainable parameters).

## Not covered here

- The SSL-trained fusion baseline (`scripts/pointad_plus/train_fusion.sh`).
- MVTec3D-AD results (`scripts/mvtec3d/`), which need the MVTec3D-AD dataset in
  PointAD's pre-rendered format.
- Pixel-level metrics (`compute_phase4_pixel_metrics.py`,
  `compute_sw_v2_pixel_metrics.py`), which need the per-pixel maps saved by
  the P2/P3 runners.
