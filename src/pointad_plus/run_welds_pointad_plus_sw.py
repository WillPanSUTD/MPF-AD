"""
PointAD+ Phase 3 test runner: sliding-window mean-fusion inference at NATIVE
photometric resolution.

This is a fork of Phase 2's ``run_welds_pointad_plus.py`` that eliminates the
336x336 input-resize bottleneck. Instead of resizing the whole 5-modal image
stack down to 336x336 once, we slide 256x256 windows over the native
resolution, run CLIP on each patch (the CLIP backbone itself still resizes
the 256 patch to 336 internally — that's expected and benign), and aggregate
the per-patch color anomaly maps via overlap-mean to obtain a full-image
anomaly map at native resolution.

Mean fusion (Phase 2 finding: parameter-free `T1..T5 -> mean(stack)` beats
the SSL-trained cross-attention block) replaces the cross-attention fusion.
We do NOT load a fusion checkpoint; we just stack-and-mean the 5 patch token
maps.

Architecture (per sample):

  5 photometric PNGs at native resolution
            |
            v
  [patch_coords(H, W, 256, 128)]  -> list of (u, v)
            |
            v (per patch)
  [Crop 5 modalities to 256x256 window]  -> (5, 3, 256, 256) PIL/tensor
            |
            v
  [PointAD CLIP encode_image, features_list=[24], DPAM_layer=20]
            |
            v
  Per-modality token map (5, 577, 768)
            |
            v
  [stack(...).mean(0)]  -> (1, 577, 768) fused tokens
            |
            v
  [compute_similarity + get_similarity_map(..., 256)]  -> (1, 256, 256, 2)
            |
            v
  color_map = (sim[...,1] + 1 - sim[...,0]) / 2  -> (1, 256, 256)
            |
            v (collect across patches)
  [aggregate_overlap_mean(maps, coords, H, W, 256)]  -> (H, W) color map
            |
            v
  gaussian_filter(sigma)

Point branch: UNCHANGED from Phase 1/2. PointAD's render-multi-view branch
operates on the rendered images (from the Phase 1 manifest), not the
photometric channels, so sliding window does not apply to it. We resize the
point-branch anomaly map (at image_size=336) up to native resolution before
late fusion.

Late fusion: integrate = (sliding_aggregated_color + point_native) / 2.

Image-level score = max over per-pixel scores of the aggregated map +
text_prob contribution (same scheme as Phase 2 runner).

We deliberately do NOT modify external/PointAD/. The runner replicates the
relevant snippets of PointAD's test() inline so that we have a clean
substitution point for the sliding-window color branch.

CLI:
  python -m src.pointad_plus.run_welds_pointad_plus_sw \
    --manifest external/datasets/welds_pointad_mp/weld/all_meta.json \
    --pointad_ckpt external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth \
    --patch_size 256 --stride 128 \
    --out results/welds_pointad_plus_sw/
    [--limit N]            # smoke-test on first N test samples
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
PATCH_SIZE_DEFAULT = 256
STRIDE_DEFAULT = 128


# ----------------------------------------------------------------------------
# PointAD bootstrap (monkey-patch generate_class_info + stub open3d).
# (Copied verbatim from Phase 2 runner so monkey-patching applies identically.)
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
    """Same patch as Phase 1/2: tell PointAD's Dataset class there's a single
    'weld' class for the 'mvtec_pc_3d_rgb' dataset."""
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
# Manifest helpers (forked verbatim from Phase 2 runner).
# ----------------------------------------------------------------------------

def _build_multi_photo_lookup(mp_manifest: dict) -> dict:
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
    pointad_ckpt: Path,
    out_dir: Path,
    patch_size: int = PATCH_SIZE_DEFAULT,
    stride: int = STRIDE_DEFAULT,
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
    """End-to-end Phase 3 sliding-window evaluation."""
    if features_list is None:
        features_list = [24]
    if phase1_data_root is None:
        phase1_data_root = REPO_ROOT / "external" / "datasets" / "welds_pointad"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _patch_pointad_imports()
    _patch_generate_class_info()

    # Late imports so patches above are visible.
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

    from src.pointad_plus.sliding_window import patch_coords, aggregate_overlap_mean

    _setup_seed(seed)

    # ------------------------------------------------------------------
    # 1. Filtered Phase 1 manifest (unchanged from Phase 2).
    # ------------------------------------------------------------------
    mp_manifest = json.loads(Path(multi_photo_manifest).read_text())
    mp_test_keys = {
        (s["specie_name"], s["id"].split("_")[-1])
        for s in mp_manifest["test"]["weld"]
    }
    mp_lookup = _build_multi_photo_lookup(mp_manifest)

    phase1_top_meta = phase1_data_root / "all_meta.json"
    if not phase1_top_meta.is_file():
        phase1_top_meta = phase1_data_root / "weld" / "all_meta.json"
    phase1_meta = json.loads(phase1_top_meta.read_text())
    workdir = Path(tempfile.mkdtemp(prefix="pointad_plus_phase3_"))
    try:
        filtered_meta_path = _build_filtered_phase1_manifest(
            phase1_meta, phase1_data_root, mp_test_keys, workdir, limit=limit,
        )

        # ------------------------------------------------------------------
        # 2. CLIP + prompt-learner setup (unchanged from Phase 2).
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
        # 3. PointAD Dataset for point-branch geometry (unchanged).
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

        from test import back_to_3d  # noqa: E402

        # ------------------------------------------------------------------
        # 4. Inference loop.
        # ------------------------------------------------------------------
        per_sample_log = []
        for idx, items in enumerate(tqdm(test_dataloader, desc="phase3_sw_eval")):
            cls_name = items["cls_name"]
            gt_mask = items["img_mask"]                       # (1, 1, image_size, image_size)
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name[0]]["imgs_masks"].append(gt_mask)
            results[cls_name[0]]["gt_sp"].extend(items["anomaly"].detach().cpu())

            render_image = items["d2_render_img"].to(device)
            b, nv, c, h, w = render_image.shape
            render_image = render_image.reshape(-1, c, h, w)
            d2_3d_cor = items["d2_3d_cor"].to(device)
            non_zero_index = items["non_zero_index"].to(device)
            non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)

            # Resolve multi-photo entry by (specie, stem).
            img_path = items["img_path"][0]
            stem = Path(img_path).stem
            specie = items.get("specie_name", [None])[0] if "specie_name" in items else None
            if specie is None:
                specie = Path(img_path).parents[1].name
            mp_entry = mp_lookup.get((specie, stem))
            if mp_entry is None:
                raise RuntimeError(
                    f"missing multi-photo entry for (specie={specie}, stem={stem})"
                )

            # Open the 5 native-res photometric PNGs.
            mp_pil = [
                Image.open(mp_entry["multi_photo"][m]).convert("RGB")
                for m in PHOTO_MODALITIES
            ]
            native_w, native_h = mp_pil[0].size  # PIL: (W, H)

            # ----- Determine patch grid -----
            if native_h < patch_size or native_w < patch_size:
                # Should not happen for our welds dataset (good test patches
                # are exactly 256x256, anomalous test images are >= 432x485),
                # but guard anyway.
                raise RuntimeError(
                    f"image too small for patch_size={patch_size}: "
                    f"{native_h}x{native_w} (sample {specie}/{stem})"
                )
            coords = patch_coords(native_h, native_w, patch_size=patch_size, stride=stride)

            # ----- Per-patch color anomaly maps -----
            patch_color_maps: list[np.ndarray] = []     # each (patch_size, patch_size)
            patch_text_probs: list[float] = []          # fused-CLS text_prob per patch
            with torch.no_grad():
                for (u, v) in coords:
                    # Crop each modality to the (patch_size x patch_size) window.
                    # PIL.Image.crop expects (left, upper, right, lower) = (u, v, u+P, v+P).
                    crops = [
                        m.crop((u, v, u + patch_size, v + patch_size))
                        for m in mp_pil
                    ]
                    # preprocess(): Resize(image_size, image_size)+CenterCrop+ToTensor+Normalize.
                    # We rely on this to upsample 256 -> 336 for CLIP. CLIP backbone
                    # itself wraps to 336x336 (see plan §critical details).
                    mp_batch = torch.stack([preprocess(c) for c in crops], dim=0).to(device)  # (5, 3, IS, IS)

                    color_image_features, color_patch_features = model.encode_image(
                        mp_batch, features_list, DPAM_layer=20
                    )
                    # color_patch_features[0]: (5, N, D), color_image_features: (5, D_proj)
                    color_patch = color_patch_features[0].float()  # (5, N, D)
                    # Mean-fuse across the 5 modalities (Phase 2 finding):
                    fused_tokens = color_patch.mean(dim=0, keepdim=True)  # (1, N, D)

                    # Fused CLS -> text_prob.
                    fused_cls = fused_tokens[:, 0, :]
                    fused_cls_norm = fused_cls / fused_cls.norm(dim=-1, keepdim=True)
                    cl_probs = fused_cls_norm.unsqueeze(1) @ text_features.permute(0, 2, 1)
                    cl_probs = (cl_probs / 0.07).softmax(-1)
                    cl_probs = cl_probs[:, 0, 1]                 # (1,)
                    patch_text_probs.append(float(cl_probs.item()))

                    # Fused patch tokens -> similarity_map at patch_size.
                    fused_patch_norm = fused_tokens / fused_tokens.norm(dim=-1, keepdim=True)
                    color_sim, _ = AnomalyCLIP_lib.compute_similarity(
                        fused_patch_norm, text_features[0]
                    )
                    color_sim_map = AnomalyCLIP_lib.get_similarity_map(
                        color_sim[:, 1:, :], patch_size
                    )                                            # (1, P, P, 2)
                    color_map_patch = (
                        color_sim_map[..., 1] + 1 - color_sim_map[..., 0]
                    ) / 2.0                                      # (1, P, P)
                    patch_color_maps.append(
                        color_map_patch[0].detach().cpu().numpy().astype(np.float32)
                    )

                # ----- Aggregate to native resolution -----
                color_native = aggregate_overlap_mean(
                    patch_color_maps, coords, native_h, native_w, patch_size=patch_size,
                )                                                 # (H, W), float32 np

                # ----- Point branch (unchanged from Phase 2) -----
                render_features, render_patch_features = model.encode_image(
                    render_image, features_list, DPAM_layer=20
                )
                # text_prob for point branch (mean over nv views).
                render_features_norm = render_features / render_features.norm(
                    dim=-1, keepdim=True
                )
                point_text_probs = render_features_norm.unsqueeze(1) @ text_features.permute(0, 2, 1)
                point_text_probs = (point_text_probs / 0.07).softmax(-1)
                point_text_probs = point_text_probs[:, 0, 1]
                point_text_probs = torch.chunk(point_text_probs, nv, dim=0)
                point_text_probs = torch.stack(point_text_probs, dim=1).mean(1)  # (1,)

                # Patch-level similarity map for the point branch (at image_size).
                render_patch = render_patch_features[0]
                render_patch = render_patch / render_patch.norm(dim=-1, keepdim=True)
                render_sim, _ = AnomalyCLIP_lib.compute_similarity(
                    render_patch, text_features[0]
                )
                render_sim_map = AnomalyCLIP_lib.get_similarity_map(
                    render_sim[:, 1:, :], image_size
                )                                                # (nv, IS, IS, 2)
                d3_similarity_map = back_to_3d(render_sim_map, d2_3d_cor, non_zero_index)
                anomaly_map_pt = d3_similarity_map[..., 1]       # (1, IS_pc, IS_pc) at point_size grid

                # Apply Phase 2's per-channel gaussian smoothing.
                anomaly_map_pt = torch.stack([
                    torch.from_numpy(gaussian_filter(i, sigma=sigma))
                    for i in anomaly_map_pt.detach().cpu()
                ], dim=0)                                        # (1, IS, IS)
                # Resize point-branch anomaly map to native resolution for late fusion.
                anomaly_map_pt_native = torch.nn.functional.interpolate(
                    anomaly_map_pt.unsqueeze(1).float(),
                    size=(native_h, native_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)                                     # (1, H, W)

                # Smooth the sliding-window color map and tensorize for combination.
                color_native_smoothed = gaussian_filter(color_native, sigma=sigma)
                color_anomaly_map_native = torch.from_numpy(
                    color_native_smoothed[None, ...].astype(np.float32)
                )                                                # (1, H, W)

                integrate_anomaly_map_native = (
                    color_anomaly_map_native + anomaly_map_pt_native
                ) / 2.0
                integrate_smoothed = torch.from_numpy(
                    gaussian_filter(integrate_anomaly_map_native[0].numpy(), sigma=sigma)
                )[None, ...]                                     # (1, H, W)

                # Per-sample scores (mirror Phase 2 / PointAD test.py).
                # We need scalars matching the existing pickle schema; max over
                # the native-res aggregated map is the principled choice.
                pt_cpu = point_text_probs.detach().cpu()
                cl_cpu = torch.tensor([float(np.mean(patch_text_probs))])
                results[cls_name[0]]["pr_sp"].extend(
                    0.5 * anomaly_map_pt.max() + 0.5 * pt_cpu
                )
                results[cls_name[0]]["color_pr_sp"].extend(
                    color_anomaly_map_native.max() + cl_cpu
                )
                results[cls_name[0]]["integrate_pr_sp"].extend(
                    integrate_smoothed.max() + (cl_cpu + pt_cpu) / 2
                )

                # ----- For pixel-level metrics: we need anomaly maps at the
                #   same resolution as `imgs_masks` (which is image_size x
                #   image_size from PointAD's Dataset target_transform).
                #   Downsample the native-res aggregated map back to
                #   image_size for parity with Phase 1/2 pixel metrics, so
                #   downstream summary code can compare like-to-like.
                color_for_metrics = torch.nn.functional.interpolate(
                    color_anomaly_map_native.unsqueeze(1),
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)                                     # (1, IS, IS)
                integrate_for_metrics = torch.nn.functional.interpolate(
                    integrate_smoothed.unsqueeze(1),
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)                                     # (1, IS, IS)
                # Keep CPU float16 copies for the in-memory metric pass
                # (downsampled to 336x336 = ~225 KB per map).
                am_cpu = anomaly_map_pt.detach().cpu().to(torch.float16)
                co_cpu = color_for_metrics.detach().cpu().to(torch.float16)
                in_cpu = integrate_for_metrics.detach().cpu().to(torch.float16)
                results[cls_name[0]]["anomaly_maps"].append(am_cpu)
                results[cls_name[0]]["color_anomaly_maps"].append(co_cpu)
                results[cls_name[0]]["integrate_anomaly_maps"].append(in_cpu)

                # ALSO stream to disk immediately to avoid memory accumulation
                # crashes mid-eval. Per-sample .npz under <out>/maps/<cls>/<idx>.npz
                maps_dir = out_dir / "maps" / cls_name[0]
                maps_dir.mkdir(parents=True, exist_ok=True)
                import numpy as _np
                _np.savez_compressed(
                    maps_dir / f"{idx:05d}.npz",
                    anomaly=am_cpu.numpy(),
                    color=co_cpu.numpy(),
                    integrate=in_cpu.numpy(),
                )

                # Smoke-time per-sample log.
                if limit is not None or idx < 5:
                    has_nan = (
                        bool(torch.isnan(anomaly_map_pt).any())
                        or bool(torch.isnan(color_anomaly_map_native).any())
                        or bool(torch.isnan(integrate_smoothed).any())
                    )
                    per_sample_log.append({
                        "idx": idx,
                        "specie": specie,
                        "stem": stem,
                        "native_hw": (native_h, native_w),
                        "n_patches": len(coords),
                        "color_native_shape": tuple(color_anomaly_map_native.shape),
                        "color_native_max": float(color_anomaly_map_native.max()),
                        "integrate_native_max": float(integrate_smoothed.max()),
                        "point_text_probs": float(pt_cpu.item()),
                        "color_text_probs_mean": float(cl_cpu.item()),
                        "has_nan": has_nan,
                    })

            # Close PIL handles for this sample.
            for m in mp_pil:
                m.close()

        # ------------------------------------------------------------------
        # 5. Aggregate metrics (mirror Phase 2 / test.py post-loop).
        # ------------------------------------------------------------------
        import torch as _torch
        for obj in obj_list:
            results[obj]["imgs_masks"] = _torch.cat(results[obj]["imgs_masks"])
            results[obj]["anomaly_maps"] = _torch.cat(
                results[obj]["anomaly_maps"]
            ).detach().cpu().numpy()
            results[obj]["color_anomaly_maps"] = _torch.cat(
                results[obj]["color_anomaly_maps"]
            ).detach().cpu().numpy()
            results[obj]["integrate_anomaly_maps"] = _torch.cat(
                results[obj]["integrate_anomaly_maps"]
            ).detach().cpu().numpy()

        logger = get_logger(str(out_dir))
        table_rows = {"point": [], "color": [], "integrate": []}
        for obj in obj_list:
            obj_results = {obj: results[obj]}
            try:
                point_iauroc = image_level_metrics(obj_results, obj, "image-auroc", modality="pr_sp")
                point_iap = image_level_metrics(obj_results, obj, "image-ap", modality="pr_sp")
                color_iauroc = image_level_metrics(obj_results, obj, "image-auroc", modality="color_pr_sp")
                color_iap = image_level_metrics(obj_results, obj, "image-ap", modality="color_pr_sp")
                int_iauroc = image_level_metrics(obj_results, obj, "image-auroc", modality="integrate_pr_sp")
                int_iap = image_level_metrics(obj_results, obj, "image-ap", modality="integrate_pr_sp")
            except ValueError as e:
                logger.warning(f"image-level skipped for {obj}: {e}")
                point_iauroc = color_iauroc = int_iauroc = float("nan")
                point_iap = color_iap = int_iap = float("nan")
            try:
                point_pauroc = pixel_level_metrics(obj_results, obj, "pixel-auroc", modality="anomaly_maps")
                point_paupro = pixel_level_metrics(obj_results, obj, "pixel-aupro", modality="anomaly_maps")
                color_pauroc = pixel_level_metrics(obj_results, obj, "pixel-auroc", modality="color_anomaly_maps")
                color_paupro = pixel_level_metrics(obj_results, obj, "pixel-aupro", modality="color_anomaly_maps")
                int_pauroc = pixel_level_metrics(obj_results, obj, "pixel-auroc", modality="integrate_anomaly_maps")
                int_paupro = pixel_level_metrics(obj_results, obj, "pixel-aupro", modality="integrate_anomaly_maps")
            except ValueError as e:
                logger.warning(f"pixel-level skipped for {obj}: {e}")
                point_pauroc = point_paupro = color_pauroc = color_paupro = int_pauroc = int_paupro = float("nan")

            for tag, row in (
                ("point", [obj, point_pauroc, point_paupro, point_iauroc, point_iap]),
                ("color", [obj, color_pauroc, color_paupro, color_iauroc, color_iap]),
                ("integrate", [obj, int_pauroc, int_paupro, int_iauroc, int_iap]),
            ):
                table_rows[tag].append(row)

        from tabulate import tabulate as _tabulate
        for tag, rows in table_rows.items():
            disp_rows = [
                [r[0]] + [f"{v * 100:.1f}" if not (isinstance(v, float) and v != v) else "nan"
                          for v in r[1:]]
                for r in rows
            ]
            logger.info("\n%s\n%s", tag,
                _tabulate(disp_rows, headers=[
                    "objects", "pixel_auroc", "pixel_aupro", "image_auroc", "image_ap"
                ], tablefmt="pipe"))

        # ------------------------------------------------------------------
        # 6. Persist raw per-sample image-level scores AND per-pixel maps.
        # Per-pixel maps stored at 336x336 resolution (PointAD's pixel-metric
        # space). Both branches (baseline + sw) are needed to compute the
        # Phase 4 max-hybrid at pixel level downstream.
        # ------------------------------------------------------------------
        def _stack_maps(map_list):
            # The metric block above may already have stacked these lists into
            # arrays/tensors, so test for emptiness explicitly (`not arr` raises).
            if map_list is None or len(map_list) == 0:
                return None
            import numpy as _np
            stacked = []
            for m in map_list:
                arr = m.detach().cpu().numpy() if hasattr(m, "detach") else _np.asarray(m)
                # Reduce to (H, W) - take first channel/slice if needed
                while arr.ndim > 2:
                    arr = arr[0]
                stacked.append(arr.astype(_np.float32))
            return _np.stack(stacked, axis=0)

        slim = {}
        for obj, d in results.items():
            slim[obj] = {
                "gt_sp": [float(x) for x in d["gt_sp"]],
                "pr_sp": [float(x) for x in d["pr_sp"]],
                "color_pr_sp": [float(x) for x in d["color_pr_sp"]],
                "integrate_pr_sp": [float(x) for x in d["integrate_pr_sp"]],
                "anomaly_maps": _stack_maps(d.get("anomaly_maps", [])),
                "color_anomaly_maps": _stack_maps(d.get("color_anomaly_maps", [])),
                "integrate_anomaly_maps": _stack_maps(d.get("integrate_anomaly_maps", [])),
                "imgs_masks": _stack_maps(d.get("imgs_masks", [])),
            }
        raw_pkl = out_dir / "raw_results.pkl"
        with open(raw_pkl, "wb") as f:
            pickle.dump(slim, f)
        print(f"[run_welds_pointad_plus_sw] wrote {raw_pkl}", flush=True)

        if per_sample_log:
            print("[smoke] per-sample sanity log (first {} samples):".format(
                len(per_sample_log)))
            for row in per_sample_log:
                print(f"  {row}")

        return {"results": results, "raw_pkl": str(raw_pkl), "per_sample_log": per_sample_log}
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="PointAD+ Phase 3 sliding-window runner")
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Phase 2 multi-photo manifest (welds_pointad_mp/.../all_meta.json)")
    ap.add_argument("--pointad_ckpt", type=Path, required=True,
                    help="PointAD prompt-learner checkpoint (.pth from exps_9_12_4_mv9_*).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory (writes raw_results.pkl + log.txt).")
    ap.add_argument("--patch_size", type=int, default=PATCH_SIZE_DEFAULT,
                    help="Sliding-window patch size (default 256).")
    ap.add_argument("--stride", type=int, default=STRIDE_DEFAULT,
                    help="Sliding-window stride (default 128).")
    ap.add_argument("--phase1_data_root", type=Path, default=None,
                    help="Phase 1 PointAD-format dataset root (defaults to "
                         "external/datasets/welds_pointad).")
    ap.add_argument("--image_size", type=int, default=IMAGE_SIZE_DEFAULT)
    ap.add_argument("--point_size", type=int, default=POINT_SIZE_DEFAULT)
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--n_ctx", type=int, default=12)
    ap.add_argument("--t_n_ctx", type=int, default=4)
    ap.add_argument("--features_list", type=int, nargs="+", default=[24])
    ap.add_argument("--sigma", type=int, default=4)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--train_class", type=str, default="carrot")
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke test: only run on the first N test samples.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run(
        multi_photo_manifest=args.manifest,
        pointad_ckpt=args.pointad_ckpt,
        out_dir=args.out,
        patch_size=args.patch_size,
        stride=args.stride,
        image_size=args.image_size,
        point_size=args.point_size,
        depth=args.depth,
        n_ctx=args.n_ctx,
        t_n_ctx=args.t_n_ctx,
        features_list=args.features_list,
        sigma=args.sigma,
        seed=args.seed,
        train_class=args.train_class,
        limit=args.limit,
        device=args.device,
        phase1_data_root=args.phase1_data_root,
    )


if __name__ == "__main__":
    main()
