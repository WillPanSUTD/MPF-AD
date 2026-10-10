"""
Rendering-subset ablation for Global MPF (P2).

Same data loading, CLIP/prompt setup and point branch as
run_welds_pointad_plus.py (Global MPF). For every test sample the five
photometric renderings (lut / phong / diffuse / specular / normal) are encoded
once; the photometric branch is then scored for 11 rendering subsets
(all five, each rendering alone, each leave-one-out) with two fusions:
raw token mean ("raw:<subset>") and LayerNorm of the mean ("ln:<subset>",
identical to the Global MPF module with mean init; "ln:all" reproduces P2).

Output: <out>/ablation.pkl with keys gt, species, stem, point and
"<raw|ln>:<subset>" (photometric image scores). Summarize with
scripts/pointad_plus/summarize_modality_ablation.py.

Usage:
  python -m src.pointad_plus.run_welds_modality_ablation \
    --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
    --pointad_ckpt external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth \
    --out results/modality_ablation
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POINTAD_DIR = REPO_ROOT / "external" / "PointAD"

PHOTO_MODALITIES = ("lut", "phong", "diffuse", "specular", "normal")
N_VIEWS = 9                  # matches Phase 1 adapter
POINT_SIZE_DEFAULT = 336
IMAGE_SIZE_DEFAULT = 336


# ----------------------------------------------------------------------------
# PointAD bootstrap (monkey-patch generate_class_info + stub open3d).
# ----------------------------------------------------------------------------

def _patch_pointad_imports() -> None:
    p = str(POINTAD_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    if "open3d" not in sys.modules:
        stub = types.ModuleType("open3d")
        stub.io = types.SimpleNamespace(
            read_point_cloud=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("open3d stub: real_pc_3d_rgb branch is not supported")
            )
        )
        sys.modules["open3d"] = stub


def _patch_generate_class_info() -> None:
    """Same patch as Phase 1: PointAD's Dataset keys off class_info to
    figure out which obj_list to expose. We expose a single ``weld`` class."""
    import dataset as pointad_dataset  # noqa: E402

    _orig = pointad_dataset.generate_class_info

    def _patched(dataset_name: str):
        if dataset_name == "mvtec_pc_3d_rgb":
            return ["weld"], {"weld": 0}
        return _orig(dataset_name)

    pointad_dataset.generate_class_info = _patched
    obj_list, _ = pointad_dataset.generate_class_info("mvtec_pc_3d_rgb")
    assert obj_list == ["weld"], f"patch failed: {obj_list}"


# ----------------------------------------------------------------------------
# Manifest helpers
# ----------------------------------------------------------------------------

def _build_multi_photo_lookup(mp_manifest: dict) -> dict:
    """Index Phase 2 manifest by (specie, stem) -> sample dict.

    Used to look up the 5 photometric PNG paths for each Phase 1 dataloader
    sample at inference time.
    """
    lookup: dict[tuple[str, str], dict] = {}
    for split in ("train", "test"):
        if split not in mp_manifest:
            continue
        for s in mp_manifest[split]["weld"]:
            stem = s["id"].split("_")[-1]
            lookup[(s["specie_name"], stem)] = s
    return lookup


def _build_filtered_phase1_manifest(
    phase1_meta: dict,
    phase1_data_root: Path,
    mp_test_keys: set[tuple[str, str]],
    out_dir: Path,
    limit: int | None = None,
) -> Path:
    """Write a Phase-1-format manifest containing only the test samples whose
    (specie, stem) is in mp_test_keys. Deduplicates by (specie, stem).

    Rewrites relative paths (e.g. ``weld/test/good/rgb/000001.png``) to
    absolute paths anchored at ``phase1_data_root`` so the manifest can live
    in a temp directory and still resolve.

    Returns the path to the written ``all_meta.json``.
    """
    # Dedup: Phase 1's test.weld has duplicate entries for val/good folded
    # into test/good. Keep one entry per (specie, stem).
    seen: set[tuple[str, str]] = set()
    out_samples = []
    path_keys_rel = ("d2_img_path", "d2_mask_path", "d3_pc")
    dir_keys_rel = ("d2_render_img_path", "d2_render_gt_path", "d2_corrdinate")
    for s in phase1_meta["test"]["weld"]:
        stem = s["d2_img_path"].split("/")[-1].split(".")[0]
        key = (s["specie_name"], stem)
        if key in seen:
            continue
        if key not in mp_test_keys:
            continue
        seen.add(key)
        # Absolutize any non-absolute paths.
        new_s = dict(s)
        for k in path_keys_rel + dir_keys_rel:
            v = new_s.get(k)
            if not v:
                continue
            p = Path(v)
            if not p.is_absolute():
                new_s[k] = str((phase1_data_root / v).as_posix())
        out_samples.append(new_s)
    if limit is not None:
        out_samples = out_samples[:limit]
    payload = {
        "train": {"weld": []},
        "test": {"weld": out_samples},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_meta = out_dir / "all_meta.json"
    out_meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_meta


# ----------------------------------------------------------------------------
# Core runner
# ----------------------------------------------------------------------------

def _setup_seed(seed: int) -> None:
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(
    multi_photo_manifest: Path,
    fusion_ckpt: Path | None,
    pointad_ckpt: Path,
    out_dir: Path,
    fusion_init: str = "trained",
    image_size: int = IMAGE_SIZE_DEFAULT,
    point_size: int = POINT_SIZE_DEFAULT,
    depth: int = 9,
    n_ctx: int = 12,
    t_n_ctx: int = 4,
    features_list: list[int] | None = None,
    sigma: int = 4,
    seed: int = 111,
    train_class: str = "carrot",
    metrics_mode: str = "image-pixel-level",
    limit: int | None = None,
    device: str = "cuda",
    phase1_data_root: Path | None = None,
) -> dict:
    """End-to-end Phase 2 evaluation. Returns the captured results dict
    (Phase 1 schema) so callers can re-summarize without reopening the pickle.
    """
    if features_list is None:
        features_list = [24]
    if phase1_data_root is None:
        phase1_data_root = REPO_ROOT / "external" / "datasets" / "welds_pointad"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _patch_pointad_imports()
    _patch_generate_class_info()

    # Late imports so the patches above are visible.
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm

    import AnomalyCLIP_lib  # noqa: E402
    from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
    from dataset import Dataset as PointADDataset  # noqa: E402
    from logger import get_logger  # noqa: E402
    from metrics import image_level_metrics, pixel_level_metrics  # noqa: E402
    from utils import get_transform  # noqa: E402

    from src.pointad_plus.photometric_fusion import MultiPhotometricFusion

    _setup_seed(seed)

    # ------------------------------------------------------------------
    # 1. Build the filtered Phase 1 manifest covering only the Phase 2
    #    test samples. PointAD's Dataset will iterate this manifest, giving
    #    us the geometry pieces (d2_render_img, d2_3d_cor, non_zero_index,
    #    img_mask). We never use its `img` field; the color branch comes
    #    from our multi-photo CLIP encodes.
    # ------------------------------------------------------------------
    mp_manifest = json.loads(Path(multi_photo_manifest).read_text())
    mp_test_keys = {
        (s["specie_name"], s["id"].split("_")[-1])
        for s in mp_manifest["test"]["weld"]
    }
    mp_lookup = _build_multi_photo_lookup(mp_manifest)

    # Read Phase 1 manifest (top-level, has absolute render-path entries) and
    # write a filtered copy with ALL paths absolutized into a temp workdir.
    # PointAD's Dataset class then resolves relative joins to the same paths
    # without us mirroring the on-disk tree.
    phase1_top_meta = phase1_data_root / "all_meta.json"
    if not phase1_top_meta.is_file():
        # Fall back to nested manifest.
        phase1_top_meta = phase1_data_root / "weld" / "all_meta.json"
    phase1_meta = json.loads(phase1_top_meta.read_text())
    workdir = Path(tempfile.mkdtemp(prefix="pointad_plus_phase1_"))
    try:
        filtered_meta_path = _build_filtered_phase1_manifest(
            phase1_meta, phase1_data_root, mp_test_keys, workdir, limit=limit,
        )

        # ------------------------------------------------------------------
        # 2. Load PointAD CLIP + prompts (mirrors test.py:64-105).
        # ------------------------------------------------------------------
        AnomalyCLIP_parameters = {
            "Prompt_length": n_ctx,
            "learnabel_text_embedding_depth": depth,
            "learnabel_text_embedding_length": t_n_ctx,
        }
        model, _ = AnomalyCLIP_lib.load(
            "ViT-L/14@336px", device=device,
            design_details=AnomalyCLIP_parameters,
        )
        model.eval()

        prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
        ckpt = torch.load(str(pointad_ckpt), map_location="cpu")
        prompt_learner.load_state_dict(ckpt["prompt_learner"])
        prompt_learner.to(device)
        model.to(device)
        model.visual.DAPM_replace(DPAM_layer=20)

        prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
        text_features = model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text
        ).float()
        text_features = torch.stack(
            torch.chunk(text_features, dim=0, chunks=2), dim=1
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ------------------------------------------------------------------
        # 3. Load fusion block.
        # ------------------------------------------------------------------
        # CLIP-ViT-L cached token dim = 768 (post-projection). Phase 2-B
        # trained with clip_dim=768.
        fusion = MultiPhotometricFusion(clip_dim=768, num_heads=8, init="mean").to(device)
        if fusion_init == "mean":
            print(f"[run_welds_pointad_plus] using mean-init fusion (no checkpoint)")
        else:
            assert fusion_ckpt is not None and Path(fusion_ckpt).is_file(), \
                f"missing fusion checkpoint: {fusion_ckpt}"
            sd = torch.load(str(fusion_ckpt), map_location=device)
            fusion.load_state_dict(sd)
            print(f"[run_welds_pointad_plus] loaded fusion block from {fusion_ckpt}")
        fusion.eval()

        # ------------------------------------------------------------------
        # 4. PointAD Dataset over filtered Phase 1 manifest.
        # ------------------------------------------------------------------
        ds_args = types.SimpleNamespace(
            image_size=image_size, point_size=point_size,
        )
        preprocess, target_transform, target_transform_pc = get_transform(ds_args)
        test_data = PointADDataset(
            root=str(workdir), dataset_name="mvtec_pc_3d_rgb",
            transform=preprocess, target_transform=target_transform,
            target_transform_pc=target_transform_pc,
            mode="test", is_all=True, point_size=point_size,
        )
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        obj_list = test_data.obj_list

        results: dict = {}
        for obj in obj_list:
            results[obj] = {
                "gt_sp": [], "pr_sp": [], "color_pr_sp": [], "integrate_pr_sp": [],
                "imgs_masks": [], "anomaly_maps": [],
                "color_anomaly_maps": [], "integrate_anomaly_maps": [],
            }

        from test import back_to_3d  # imported after patches  # noqa: E402

        # ------------------------------------------------------------------
        # 5. Inference loop. Reproduces test.py:109-176 with the color branch
        # replaced by multi-photometric fusion.
        # ------------------------------------------------------------------
        from collections import defaultdict as _dd
        _MODS = ("lut", "phong", "diffuse", "specular", "normal")
        SUBSETS = {"all": tuple(range(5))}
        SUBSETS.update({f"only_{m}": (i,) for i, m in enumerate(_MODS)})
        SUBSETS.update({f"drop_{m}": tuple(j for j in range(5) if j != i) for i, m in enumerate(_MODS)})
        abl = _dd(list)
        per_sample_log = []  # list of dicts for sanity prints / smoke test
        for idx, items in enumerate(tqdm(test_dataloader, desc="phase2_eval")):
            cls_name = items["cls_name"]
            gt_mask = items["img_mask"]
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name[0]]["imgs_masks"].append(gt_mask)
            results[cls_name[0]]["gt_sp"].extend(items["anomaly"].detach().cpu())

            render_image = items["d2_render_img"].to(device)
            b, nv, c, h, w = render_image.shape
            render_image = render_image.reshape(-1, c, h, w)
            d2_3d_cor = items["d2_3d_cor"].to(device)
            non_zero_index = items["non_zero_index"].to(device)
            non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)

            # Look up the multi-photo paths for this sample by (specie, stem).
            img_path = items["img_path"][0]
            stem = Path(img_path).stem
            specie = items.get("specie_name", [None])[0] if "specie_name" in items else None
            if specie is None:
                # PointAD Dataset doesn't expose specie_name; recover from the
                # filtered manifest in same order (we iterate single-batch and
                # the filtered manifest preserves Phase 1 order). Fallback:
                # parse from img_path.
                specie = Path(img_path).parents[1].name  # .../<specie>/rgb/<stem>.png
            mp_entry = mp_lookup.get((specie, stem))
            if mp_entry is None:
                raise RuntimeError(
                    f"missing multi-photo entry for sample (specie={specie}, stem={stem})"
                )

            # Load 5 photometric PNGs through CLIP preprocess.
            mp_imgs = []
            for m in PHOTO_MODALITIES:
                img = Image.open(mp_entry["multi_photo"][m]).convert("RGB")
                mp_imgs.append(preprocess(img))
            # (5, 3, H, W) -> stack and forward through CLIP once for all 5.
            mp_batch = torch.stack(mp_imgs, dim=0).to(device)  # (5, 3, H, W)

            with torch.no_grad():
                # 5a. Single CLIP forward over the 5 photometric images.
                color_image_features, color_patch_features = model.encode_image(
                    mp_batch, features_list, DPAM_layer=20
                )
                # color_patch_features is a list of len 1; each (5, 577, 768)
                # color_image_features is (5, 768) — projected CLS tokens.
                color_patch = color_patch_features[0]  # (5, N, D)
                color_patch = color_patch.float()
                D = color_patch.shape[-1]
                # Split (5, N, D) -> list of 5 tensors of (1, N, D) for fusion.
                token_list = [color_patch[i].unsqueeze(0) for i in range(color_patch.shape[0])]

                # ---- point branch: computed once per sample (unchanged from P2) ----
                render_features, render_patch_features = model.encode_image(
                    render_image, features_list, DPAM_layer=20
                )
                render_features_norm = render_features / render_features.norm(dim=-1, keepdim=True)
                point_text_probs = render_features_norm.unsqueeze(1) @ text_features.permute(0, 2, 1)
                point_text_probs = (point_text_probs / 0.07).softmax(-1)[:, 0, 1]
                point_text_probs = torch.stack(torch.chunk(point_text_probs, nv, dim=0), dim=1).mean(1)
                render_patch = render_patch_features[0]
                render_patch = render_patch / render_patch.norm(dim=-1, keepdim=True)
                render_sim, _ = AnomalyCLIP_lib.compute_similarity(render_patch, text_features[0])
                render_sim_map = AnomalyCLIP_lib.get_similarity_map(render_sim[:, 1:, :], image_size)
                d3_similarity_map = back_to_3d(render_sim_map, d2_3d_cor, non_zero_index)
                anomaly_map = torch.stack([
                    torch.from_numpy(gaussian_filter(i, sigma=sigma))
                    for i in d3_similarity_map[..., 1].detach().cpu()
                ], dim=0)
                point_score = float(0.5 * anomaly_map.max() + 0.5 * point_text_probs.detach().cpu())
                abl["gt"].append(float(items["anomaly"].item()))
                abl["species"].append(specie)
                abl["stem"].append(stem)
                abl["point"].append(point_score)

                # ---- photometric branch for every rendering subset ----
                stack = torch.stack(token_list, dim=0)  # (5, 1, N, D)
                for subset_name, idxs in SUBSETS.items():
                    mean_tok = stack[list(idxs)].mean(dim=0)  # (1, N, D)
                    for variant, tok in (("raw", mean_tok), ("ln", F.layer_norm(mean_tok, (D,)))):
                        cls_tok = tok[:, 0, :]
                        cls_tok = cls_tok / cls_tok.norm(dim=-1, keepdim=True)
                        ctp = cls_tok.unsqueeze(1) @ text_features.permute(0, 2, 1)
                        ctp = (ctp / 0.07).softmax(-1)[:, 0, 1]
                        patch = tok / tok.norm(dim=-1, keepdim=True)
                        csim, _ = AnomalyCLIP_lib.compute_similarity(patch, text_features[0])
                        cmap = AnomalyCLIP_lib.get_similarity_map(csim[:, 1:, :], image_size)
                        cmap = (cmap[..., 1] + 1 - cmap[..., 0]) / 2.0
                        cmap = torch.stack([
                            torch.from_numpy(gaussian_filter(i, sigma=sigma)) for i in cmap.detach().cpu()
                        ], dim=0)
                        abl[f"{variant}:{subset_name}"].append(float(cmap.max() + ctp.detach().cpu()))

        out_pkl = out_dir / "ablation.pkl"
        with open(out_pkl, "wb") as f:
            pickle.dump(dict(abl), f)
        print(f"[modality_ablation] wrote {out_pkl} ({len(abl['gt'])} samples)", flush=True)
        return {"ablation_pkl": str(out_pkl)}
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rendering-subset ablation for Global MPF")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--pointad_ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(multi_photo_manifest=args.manifest, fusion_ckpt=None, pointad_ckpt=args.pointad_ckpt,
        out_dir=args.out, fusion_init="mean", limit=args.limit)


if __name__ == "__main__":
    main()
