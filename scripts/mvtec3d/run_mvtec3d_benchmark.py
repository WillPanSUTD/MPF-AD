"""MVTec3D-AD one-vs-rest standard benchmark runner.

For a given aux category (carrot / cookie / dowel), this runs both:

  (1) the PointAD baseline color branch (single 800x800 RGB resized to 336 once);
  (2) the PointAD+ Phase 3 sliding-window color branch at native resolution
      (256 patches, stride 128, mean over patches in the overlap regions).

MVTec3D-AD has M=1 photometric channel per sample, so Phase 2 multi-photometric
mean fusion degenerates to identity (mean of one tensor). Only Phase 3 sliding
window provides any lift over baseline.

We re-implement the PointAD test() inner loop inline so both branches share the
same forward pass through the prompt-learner and DPAM-replaced ViT; the SW path
just adds an extra per-patch CLIP call on the native RGB. This is simpler than
running test.py twice and keeps the random state identical between branches.

CLI:
  python run_mvtec3d_benchmark.py \
    --data_root F:/dataset/LUT_AD_DataSet/dataset/mvtec3d \
    --meta_file F:/dataset/LUT_AD_DataSet/dataset/mvtec3d/all_meta_remapped.json \
    --aux carrot \
    --pointad_ckpt F:/dataset/LUT_AD_DataSet/external/PointAD/exps_9_12_4_mv9_mvtec_3d_336_4/carrot/epoch_15.pth \
    --out F:/dataset/LUT_AD_DataSet/results/mvtec3d_benchmark/carrot \
    [--limit N]

Saves out_dir/raw_results.pkl with per-category arrays needed for Phase 4
max-hybrid recomputation.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POINTAD_DIR = REPO_ROOT / "external" / "PointAD"

PATCH_SIZE_DEFAULT = 256
STRIDE_DEFAULT = 128
IMAGE_SIZE = 336
POINT_SIZE = 336


# ---------------------------------------------------------------------------
# PointAD bootstrap
# ---------------------------------------------------------------------------

def _patch_pointad_imports() -> None:
    p = str(POINTAD_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    if "open3d" not in sys.modules:
        stub = types.ModuleType("open3d")
        stub.io = types.SimpleNamespace(
            read_point_cloud=lambda *a, **k: (_ for _ in ()).throw(
                RuntimeError("open3d stub: real_pc_3d_rgb branch not supported")
            )
        )
        sys.modules["open3d"] = stub


def _setup_seed(seed: int) -> None:
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_meta_workdir(
    meta_file: Path, aux: str, limit: int | None,
) -> Path:
    """Materialize an all_meta.json with only test samples (good + anomalous)
    from the 9 non-aux categories, into a fresh workdir."""
    src_meta = json.loads(Path(meta_file).read_text())
    out_meta = {"train": {}, "test": {}}
    for cls, samples in src_meta["test"].items():
        if cls == aux:
            continue
        if limit is not None:
            samples = samples[:limit]
        out_meta["test"][cls] = samples
    workdir = Path(tempfile.mkdtemp(prefix=f"mvtec3d_bench_{aux}_"))
    (workdir / "all_meta.json").write_text(json.dumps(out_meta), encoding="utf-8")
    return workdir


def run(
    data_root: Path,
    meta_file: Path,
    aux: str,
    pointad_ckpt: Path,
    out_dir: Path,
    patch_size: int = PATCH_SIZE_DEFAULT,
    stride: int = STRIDE_DEFAULT,
    image_size: int = IMAGE_SIZE,
    point_size: int = POINT_SIZE,
    depth: int = 9,
    n_ctx: int = 12,
    t_n_ctx: int = 4,
    features_list: list[int] | None = None,
    sigma: int = 4,
    seed: int = 111,
    limit: int | None = None,
    device: str = "cuda",
) -> dict:
    if features_list is None:
        features_list = [24]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _patch_pointad_imports()

    import torch
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from tqdm import tqdm

    import AnomalyCLIP_lib  # noqa: E402
    from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
    from dataset import Dataset as PointADDataset  # noqa: E402
    from utils import get_transform  # noqa: E402
    from test import back_to_3d  # noqa: E402

    # Late import to keep src/ on path.
    sys.path.insert(0, str(REPO_ROOT))
    from src.pointad_plus.sliding_window import patch_coords, aggregate_overlap_mean  # noqa: E402

    _setup_seed(seed)

    workdir = _build_meta_workdir(meta_file, aux, limit)
    try:
        AnomalyCLIP_parameters = {
            "Prompt_length": n_ctx,
            "learnabel_text_embedding_depth": depth,
            "learnabel_text_embedding_length": t_n_ctx,
        }
        print(f"[{aux}] loading CLIP...", flush=True)
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
            prompts, tokenized_prompts, compound_prompts_text,
        ).float()
        text_features = torch.stack(
            torch.chunk(text_features, dim=0, chunks=2), dim=1,
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

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
        obj_list_all = test_data.obj_list  # 10 categories from generate_class_info
        obj_list = [c for c in obj_list_all if c != aux]

        # results buckets for baseline + sw
        # baseline = PointAD test.py reproduction
        # sw       = sliding-window color branch (point branch is the same)
        keys = ("gt_sp", "imgs_masks",
                "pr_sp",  # point image-level score (shared)
                "anomaly_maps",  # point pixel-level map (shared)
                "color_pr_sp_baseline", "color_anomaly_maps_baseline",
                "integrate_pr_sp_baseline", "integrate_anomaly_maps_baseline",
                "color_pr_sp_sw", "color_anomaly_maps_sw",
                "integrate_pr_sp_sw", "integrate_anomaly_maps_sw")
        results: dict = {obj: {k: [] for k in keys} for obj in obj_list}

        n_total = len(test_data)
        print(f"[{aux}] {n_total} test samples across {len(obj_list)} categories", flush=True)

        for idx, items in enumerate(tqdm(test_dataloader, desc=f"{aux}_eval")):
            cls_name = items["cls_name"][0]
            if cls_name == aux:  # safety, should be filtered
                continue

            image = items["img"].to(device)  # (1, 3, IS, IS) — single resized RGB
            gt_mask = items["img_mask"]
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            results[cls_name]["imgs_masks"].append(gt_mask)
            results[cls_name]["gt_sp"].extend(items["anomaly"].detach().cpu())

            render_image = items["d2_render_img"].to(device)
            b, nv, c, h, w = render_image.shape
            render_image = render_image.reshape(-1, c, h, w)
            d2_3d_cor = items["d2_3d_cor"].to(device)
            non_zero_index = items["non_zero_index"].to(device)
            non_zero_index = non_zero_index.unsqueeze(1).repeat(1, nv, 1, 1)

            img_path = items["img_path"][0]
            native_pil = Image.open(img_path).convert("RGB")
            native_w, native_h = native_pil.size

            with torch.no_grad():
                # ------------------------------------------------------------
                # (A) PointAD baseline: render views + single resized RGB.
                #     Reproduce test.py inner loop exactly.
                # ------------------------------------------------------------
                whole_image = torch.cat([render_image, image], dim=0)  # (nv+1, 3, IS, IS)
                image_features, patch_features = model.encode_image(
                    whole_image, features_list, DPAM_layer=20,
                )
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_probs = image_features.unsqueeze(1) @ text_features.permute(0, 2, 1)
                text_probs = (text_probs / 0.07).softmax(-1)
                text_probs = text_probs[:, 0, 1]
                # last `b` are color, first nv*b are render
                color_text_probs = text_probs[-b:]
                rt_probs = text_probs[:-b]
                rt_probs = torch.chunk(rt_probs, nv, dim=0)
                rt_probs = torch.stack(rt_probs, dim=1).mean(1)

                patch_feature = patch_features[0]
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(
                    patch_feature, text_features[0],
                )
                similarity_map = AnomalyCLIP_lib.get_similarity_map(
                    similarity[:, 1:, :], image_size,
                )
                color_similarity_map = similarity_map[-b:]            # (b, IS, IS, 2)
                render_similarity_map = similarity_map[:-b]           # (nv*b, IS, IS, 2)

                d3_similarity_map = back_to_3d(render_similarity_map, d2_3d_cor, non_zero_index)
                anomaly_map = d3_similarity_map[..., 1]               # (b, IS, IS)
                color_map_baseline = (
                    color_similarity_map[..., 1] + 1 - color_similarity_map[..., 0]
                ) / 2.0                                                # (b, IS, IS)

                anomaly_map = torch.stack([
                    torch.from_numpy(gaussian_filter(i, sigma=sigma))
                    for i in anomaly_map.detach().cpu()
                ], dim=0)
                color_anomaly_map_baseline = torch.stack([
                    torch.from_numpy(gaussian_filter(i, sigma=sigma))
                    for i in color_map_baseline.detach().cpu()
                ], dim=0)
                integrate_anomaly_map_baseline = (color_anomaly_map_baseline + anomaly_map) / 2
                integrate_anomaly_map_baseline = torch.stack([
                    torch.from_numpy(gaussian_filter(i, sigma=sigma))
                    for i in integrate_anomaly_map_baseline.detach().cpu()
                ], dim=0)

                # Image-level scores (shared format with PointAD test.py)
                point_sp = 0.5 * anomaly_map.max() + 0.5 * rt_probs.detach().cpu()
                color_sp_baseline = color_anomaly_map_baseline.max() + color_text_probs.detach().cpu()
                integrate_sp_baseline = (integrate_anomaly_map_baseline.max()
                                         + (color_text_probs + rt_probs) / 2).detach().cpu()

                results[cls_name]["pr_sp"].extend(point_sp)
                results[cls_name]["anomaly_maps"].append(anomaly_map)
                results[cls_name]["color_pr_sp_baseline"].extend(color_sp_baseline)
                results[cls_name]["color_anomaly_maps_baseline"].append(color_anomaly_map_baseline)
                results[cls_name]["integrate_pr_sp_baseline"].extend(integrate_sp_baseline)
                results[cls_name]["integrate_anomaly_maps_baseline"].append(integrate_anomaly_map_baseline)

                # ------------------------------------------------------------
                # (B) Sliding-window color branch.
                # M=1: each "modality" is just the single native RGB.
                # ------------------------------------------------------------
                if native_h < patch_size or native_w < patch_size:
                    # Native too small for SW; fall back to baseline color map.
                    color_native = color_map_baseline[0].detach().cpu().numpy().astype(np.float32)
                    color_native_resized = color_native  # already at IS — but we'll resize for parity
                else:
                    coords = patch_coords(native_h, native_w,
                                          patch_size=patch_size, stride=stride)
                    patch_color_maps = []
                    patch_text_probs = []
                    for (u, v) in coords:
                        crop = native_pil.crop((u, v, u + patch_size, v + patch_size))
                        crop_t = preprocess(crop).unsqueeze(0).to(device)  # (1, 3, IS, IS)
                        ci_feat, cp_feat = model.encode_image(
                            crop_t, features_list, DPAM_layer=20,
                        )
                        # CLS -> text_prob
                        ci_feat_n = ci_feat / ci_feat.norm(dim=-1, keepdim=True)
                        tp = ci_feat_n.unsqueeze(1) @ text_features.permute(0, 2, 1)
                        tp = (tp / 0.07).softmax(-1)
                        tp = tp[:, 0, 1]
                        patch_text_probs.append(float(tp.item()))

                        # patch tokens -> sim map at patch_size
                        cp = cp_feat[0]
                        cp = cp / cp.norm(dim=-1, keepdim=True)
                        cs, _ = AnomalyCLIP_lib.compute_similarity(cp, text_features[0])
                        csm = AnomalyCLIP_lib.get_similarity_map(cs[:, 1:, :], patch_size)
                        cmp_patch = (csm[..., 1] + 1 - csm[..., 0]) / 2.0  # (1, P, P)
                        patch_color_maps.append(
                            cmp_patch[0].detach().cpu().numpy().astype(np.float32)
                        )

                    color_native = aggregate_overlap_mean(
                        patch_color_maps, coords, native_h, native_w,
                        patch_size=patch_size,
                    )  # (H, W) float32

                color_native_smoothed = gaussian_filter(color_native, sigma=sigma)
                color_anomaly_native = torch.from_numpy(
                    color_native_smoothed[None, ...].astype(np.float32)
                )  # (1, H, W)

                # Resize sw color map to IS for pixel-level metrics parity.
                color_anomaly_map_sw = torch.nn.functional.interpolate(
                    color_anomaly_native.unsqueeze(1),
                    size=(image_size, image_size),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)  # (1, IS, IS)

                integrate_anomaly_map_sw = (color_anomaly_map_sw + anomaly_map) / 2
                integrate_anomaly_map_sw = torch.stack([
                    torch.from_numpy(gaussian_filter(i, sigma=sigma))
                    for i in integrate_anomaly_map_sw.detach().cpu()
                ], dim=0)

                # Image-level scores for SW branch.
                # color_text_probs_sw = mean over per-patch text_probs (1-D scalar).
                if native_h < patch_size or native_w < patch_size:
                    color_text_probs_sw = color_text_probs.detach().cpu()
                else:
                    color_text_probs_sw = torch.tensor([float(np.mean(patch_text_probs))])

                color_sp_sw = color_anomaly_native.max() + color_text_probs_sw
                integrate_sp_sw = (integrate_anomaly_map_sw.max()
                                   + (color_text_probs_sw + rt_probs.detach().cpu()) / 2)

                results[cls_name]["color_pr_sp_sw"].extend(color_sp_sw)
                results[cls_name]["color_anomaly_maps_sw"].append(color_anomaly_map_sw)
                results[cls_name]["integrate_pr_sp_sw"].extend(integrate_sp_sw)
                results[cls_name]["integrate_anomaly_maps_sw"].append(integrate_anomaly_map_sw)

            native_pil.close()

            if idx < 3:
                # NaN check on first 3 samples for smoke confidence.
                anyNaN = any(
                    bool(torch.isnan(t).any()) for t in [
                        anomaly_map, color_anomaly_map_baseline,
                        color_anomaly_map_sw, integrate_anomaly_map_baseline,
                        integrate_anomaly_map_sw,
                    ]
                )
                print(f"  smoke idx={idx} cls={cls_name} native={native_h}x{native_w} "
                      f"point_sp={float(point_sp.item()):.4f} "
                      f"color_sp_base={float(color_sp_baseline.item()):.4f} "
                      f"color_sp_sw={float(color_sp_sw.item()):.4f} "
                      f"NaN={anyNaN}",
                      flush=True)

        # ---------------- Cat per category ----------------
        for obj in obj_list:
            results[obj]["imgs_masks"] = torch.cat(results[obj]["imgs_masks"]) \
                .detach().cpu().numpy()
            for k in ("anomaly_maps", "color_anomaly_maps_baseline",
                      "integrate_anomaly_maps_baseline",
                      "color_anomaly_maps_sw", "integrate_anomaly_maps_sw"):
                results[obj][k] = torch.cat(results[obj][k]).detach().cpu().numpy()
            for k in ("gt_sp", "pr_sp",
                      "color_pr_sp_baseline", "integrate_pr_sp_baseline",
                      "color_pr_sp_sw", "integrate_pr_sp_sw"):
                results[obj][k] = [float(x) for x in results[obj][k]]

        raw_pkl = out_dir / "raw_results.pkl"
        with open(raw_pkl, "wb") as f:
            pickle.dump(results, f)
        print(f"[{aux}] wrote {raw_pkl}", flush=True)

        # Compute per-category metrics inline so we have a quick log.
        from metrics import image_level_metrics, pixel_level_metrics
        log_path = out_dir / "metrics.log"
        with open(log_path, "w") as f:
            f.write(f"# MVTec3D-AD benchmark: aux={aux}, ckpt={pointad_ckpt}\n\n")
            for branch_tag, color_sp_key, int_sp_key, color_map_key, int_map_key in [
                ("baseline", "color_pr_sp_baseline", "integrate_pr_sp_baseline",
                 "color_anomaly_maps_baseline", "integrate_anomaly_maps_baseline"),
                ("sw", "color_pr_sp_sw", "integrate_pr_sp_sw",
                 "color_anomaly_maps_sw", "integrate_anomaly_maps_sw"),
            ]:
                f.write(f"\n## Branch: {branch_tag}\n")
                f.write("| obj | point-OR | point-OA | color-OR | color-OA | int-OR | int-OA | "
                        "point-PR | point-PP | color-PR | color-PP | int-PR | int-PP |\n")
                f.write("|---" * 13 + "|\n")
                # Build per-obj wrapper dicts that match PointAD metric signatures.
                for obj in obj_list:
                    wrap = {obj: {
                        "gt_sp": results[obj]["gt_sp"],
                        "pr_sp": results[obj]["pr_sp"],
                        "color_pr_sp": results[obj][color_sp_key],
                        "integrate_pr_sp": results[obj][int_sp_key],
                        "imgs_masks": results[obj]["imgs_masks"],
                        "anomaly_maps": results[obj]["anomaly_maps"],
                        "color_anomaly_maps": results[obj][color_map_key],
                        "integrate_anomaly_maps": results[obj][int_map_key],
                    }}
                    try:
                        p_or = image_level_metrics(wrap, obj, "image-auroc", "pr_sp")
                        p_oa = image_level_metrics(wrap, obj, "image-ap", "pr_sp")
                        c_or = image_level_metrics(wrap, obj, "image-auroc", "color_pr_sp")
                        c_oa = image_level_metrics(wrap, obj, "image-ap", "color_pr_sp")
                        i_or = image_level_metrics(wrap, obj, "image-auroc", "integrate_pr_sp")
                        i_oa = image_level_metrics(wrap, obj, "image-ap", "integrate_pr_sp")
                    except Exception as e:
                        p_or = p_oa = c_or = c_oa = i_or = i_oa = float("nan")
                        print(f"warn obj={obj} branch={branch_tag} image-level failed: {e}")
                    try:
                        p_pr = pixel_level_metrics(wrap, obj, "pixel-auroc", "anomaly_maps")
                        p_pp = pixel_level_metrics(wrap, obj, "pixel-aupro", "anomaly_maps")
                        c_pr = pixel_level_metrics(wrap, obj, "pixel-auroc", "color_anomaly_maps")
                        c_pp = pixel_level_metrics(wrap, obj, "pixel-aupro", "color_anomaly_maps")
                        i_pr = pixel_level_metrics(wrap, obj, "pixel-auroc", "integrate_anomaly_maps")
                        i_pp = pixel_level_metrics(wrap, obj, "pixel-aupro", "integrate_anomaly_maps")
                    except Exception as e:
                        p_pr = p_pp = c_pr = c_pp = i_pr = i_pp = float("nan")
                        print(f"warn obj={obj} branch={branch_tag} pixel-level failed: {e}")
                    f.write(f"| {obj} | {p_or*100:.1f} | {p_oa*100:.1f} | "
                            f"{c_or*100:.1f} | {c_oa*100:.1f} | "
                            f"{i_or*100:.1f} | {i_oa*100:.1f} | "
                            f"{p_pr*100:.1f} | {p_pp*100:.1f} | "
                            f"{c_pr*100:.1f} | {c_pp*100:.1f} | "
                            f"{i_pr*100:.1f} | {i_pp*100:.1f} |\n")
        print(f"[{aux}] wrote {log_path}", flush=True)

        return {"raw_pkl": str(raw_pkl), "log": str(log_path)}

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--meta_file", type=Path, required=True)
    ap.add_argument("--aux", type=str, required=True,
                    choices=["carrot", "cookie", "dowel"])
    ap.add_argument("--pointad_ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--patch_size", type=int, default=PATCH_SIZE_DEFAULT)
    ap.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    run(
        data_root=args.data_root,
        meta_file=args.meta_file,
        aux=args.aux,
        pointad_ckpt=args.pointad_ckpt,
        out_dir=args.out,
        patch_size=args.patch_size,
        stride=args.stride,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
