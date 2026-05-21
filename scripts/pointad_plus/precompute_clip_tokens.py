"""Precompute CLIP-ViT-L per-patch token maps for the welds multi-photometric
manifest.

Walks the train.weld list in
`external/datasets/welds_pointad_mp/weld/all_meta.json` and, for each sample,
encodes each of its 5 photometric modalities (lut, phong, diffuse, specular,
normal) through PointAD's CLIP-ViT-L backbone (frozen, with the same DPAM_layer
patch the Phase 1 runner uses). Saves the per-patch token map to disk as
`<out>/<sample_id>/<modality>.pt`, where each tensor has shape (N, D) with
N = 577 (CLS + 24x24 patches) and D = 768 (ViT-L's projected dim).

This is the one-shot cache the SSL training step (P2-D.3) consumes via
`src/pointad_plus/ssl_modality_masking.py`. CLIP forwards are the expensive
part of P2-D; once cached, fusion training is fast.

Idempotent: if a sample directory already contains all 5 .pt files, it is
skipped.

Usage (called from `scripts/pointad_plus/train_fusion.sh`):
    python -m scripts.pointad_plus.precompute_clip_tokens \
      --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
      --out external/datasets/welds_pointad_mp/clip_tokens/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POINTAD_DIR = REPO_ROOT / "external" / "PointAD"

PHOTO_MODALITIES = ("lut", "phong", "diffuse", "specular", "normal")


def _patch_pointad_imports() -> None:
    """Same monkey-patches Phase 1's `run_welds_zero_shot.py` uses.

    Adds external/PointAD to sys.path and stubs out `open3d` (which PointAD's
    dataset.py imports unconditionally even though we never touch the
    real_pc_3d_rgb branch here).
    """
    p = str(POINTAD_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    if "open3d" not in sys.modules:
        open3d_stub = types.ModuleType("open3d")
        open3d_stub.io = types.SimpleNamespace(
            read_point_cloud=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("open3d stub: real_pc_3d_rgb branch is not supported")
            )
        )
        sys.modules["open3d"] = open3d_stub


def _load_clip(device: str = "cuda"):
    """Load PointAD's CLIP-ViT-L backbone with the same parameters Phase 1
    uses (matches shipped checkpoint weights)."""
    import AnomalyCLIP_lib  # noqa: E402

    AnomalyCLIP_parameters = {
        "Prompt_length": 12,
        "learnabel_text_embedding_depth": 9,
        "learnabel_text_embedding_length": 4,
    }
    model, preprocess = AnomalyCLIP_lib.load(
        "ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters
    )
    model.eval()
    # Phase 1 / PointAD's test.py applies DAPM_replace before encoding.
    model.visual.DAPM_replace(DPAM_layer=20)
    return model, preprocess


def precompute(manifest_path: Path, out_root: Path, device: str = "cuda") -> dict:
    """Encode and cache CLIP token maps for every (sample × modality) pair in
    the manifest's train.weld split. Returns a tally dict."""
    import torch
    from PIL import Image

    _patch_pointad_imports()
    model, preprocess = _load_clip(device=device)

    meta = json.loads(Path(manifest_path).read_text())
    train_samples = meta["train"]["weld"]
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    n_samples = len(train_samples)
    tally = {"samples": n_samples, "encoded": 0, "skipped": 0, "errors": 0}

    t0 = time.time()
    for i, s in enumerate(train_samples):
        sample_id = s["id"]
        sample_dir = out_root / sample_id
        existing = list(sample_dir.glob("*.pt")) if sample_dir.is_dir() else []
        if len(existing) >= len(PHOTO_MODALITIES):
            tally["skipped"] += 1
            continue
        sample_dir.mkdir(parents=True, exist_ok=True)

        for m in PHOTO_MODALITIES:
            out_pt = sample_dir / f"{m}.pt"
            if out_pt.is_file():
                continue
            img_path = s["multi_photo"][m]
            try:
                img = Image.open(img_path).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, patch_features = model.encode_image(x, [24], DPAM_layer=20)
                # patch_features is a list of length 1; element shape (1, N, D)
                tokens = patch_features[0].squeeze(0).cpu()  # (N, D)
                torch.save(tokens, out_pt)
            except Exception as e:  # noqa: BLE001
                tally["errors"] += 1
                print(f"[precompute] ERROR {sample_id}/{m}: {e}", flush=True)
                continue

        tally["encoded"] += 1
        if (i + 1) % 50 == 0 or i + 1 == n_samples:
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (n_samples - i - 1) / max(rate, 1e-6)
            print(
                f"[precompute] {i + 1}/{n_samples} samples "
                f"(skipped={tally['skipped']}, errors={tally['errors']}) "
                f"rate={rate:.2f}/s eta={eta:.0f}s",
                flush=True,
            )

    return tally


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tally = precompute(args.manifest, args.out, device=args.device)
    n_dirs_with_all_5 = 0
    for d in args.out.iterdir():
        if d.is_dir() and len(list(d.glob("*.pt"))) == len(PHOTO_MODALITIES):
            n_dirs_with_all_5 += 1
    print(
        f"[precompute] DONE: samples={tally['samples']} "
        f"encoded={tally['encoded']} skipped={tally['skipped']} "
        f"errors={tally['errors']} dirs_with_all_5={n_dirs_with_all_5} "
        f"-> {n_dirs_with_all_5 * 5} (sample × modality) tokens cached",
        flush=True,
    )


if __name__ == "__main__":
    main()
