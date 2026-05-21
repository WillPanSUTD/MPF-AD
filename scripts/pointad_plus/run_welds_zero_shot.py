"""
Run PointAD's zero-shot test on welds data using a shipped MVTec3D-AD checkpoint.

This wrapper monkey-patches `dataset.generate_class_info` (without modifying
external/PointAD/) so that PointAD's `Dataset` loader, which is hardcoded to
expect MVTec3D-AD class names, instead accepts our single "weld" class.

We invoke `test.test(args)` in-process to keep the patch active across imports.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POINTAD_DIR = REPO_ROOT / "external" / "PointAD"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PointAD zero-shot welds test wrapper",
    )
    parser.add_argument(
        "--checkpoint_class",
        default="carrot",
        choices=["carrot", "cookie", "dowel"],
        help="Which shipped MVTec3D-AD per-class checkpoint to use",
    )
    parser.add_argument(
        "--data_path",
        default=str(REPO_ROOT / "external" / "datasets" / "welds_pointad"),
        help="Welds adapter root (contains all_meta.json and weld/ subdir)",
    )
    parser.add_argument(
        "--save_path",
        default=str(REPO_ROOT / "results" / "welds_zero_shot"),
        help="Where PointAD's logger and metrics will be written",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=336,
        help="Image size used by PointAD test (must match shipped weights -> 336)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=9,
        help="Prompt depth (must match shipped weights)",
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=12,
        help="Prompt visual context length (must match shipped weights)",
    )
    parser.add_argument(
        "--t_n_ctx",
        type=int,
        default=4,
        help="Prompt text context length (must match shipped weights)",
    )
    parser.add_argument(
        "--features_list",
        type=int,
        nargs="+",
        default=[24],
        help="CLIP layer features to use (test.sh uses [24])",
    )
    parser.add_argument(
        "--sigma",
        type=int,
        default=4,
        help="Gaussian smoothing sigma for anomaly map",
    )
    parser.add_argument(
        "--point_size",
        type=int,
        default=336,
        help="Per-view resolution for organized point cloud (must match adapter -> 336)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=111,
        help="Random seed",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="image-pixel-level",
        choices=["image-level", "pixel-level", "image-pixel-level"],
    )
    cli = parser.parse_args()

    # Make external/PointAD importable.
    sys.path.insert(0, str(POINTAD_DIR))

    # open3d is imported at the top of external/PointAD/dataset.py but only
    # used in the real_pc_3d_rgb branch (which we do not use). Python 3.13
    # has no open3d wheel, so install a stub before importing dataset.
    if "open3d" not in sys.modules:
        open3d_stub = types.ModuleType("open3d")
        open3d_stub.io = types.SimpleNamespace(
            read_point_cloud=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("open3d stub: real_pc_3d_rgb branch is not supported")
            )
        )
        sys.modules["open3d"] = open3d_stub

    # Step 1: import dataset and monkey-patch generate_class_info.
    import dataset as pointad_dataset  # noqa: E402

    def patched_generate_class_info(dataset_name: str):
        # PointAD's `Dataset.__getitem__` keys off the dataset_name string for
        # data loader branching. We keep the loader name as "mvtec_pc_3d_rgb"
        # (which selects the organized-PC + RGB branch the welds adapter
        # writes), but we override the class list to our single "weld" class.
        if dataset_name == "mvtec_pc_3d_rgb":
            obj_list = ["weld"]
            return obj_list, {"weld": 0}
        # Fall back to the original behaviour for unrelated dataset names.
        return _original_generate_class_info(dataset_name)

    _original_generate_class_info = pointad_dataset.generate_class_info
    pointad_dataset.generate_class_info = patched_generate_class_info

    # Sanity check before importing test (test.py does `from dataset import Dataset`)
    obj_list, _ = pointad_dataset.generate_class_info("mvtec_pc_3d_rgb")
    assert obj_list == ["weld"], f"patch failed: {obj_list}"

    # Step 2: import PointAD's test module and invoke its test() function.
    import test as pointad_test  # noqa: E402

    # Hook the metrics functions to capture the raw per-sample arrays before
    # PointAD reduces them to scalars. We use this to compute per-defect
    # breakdown (PointAD only knows about the single "weld" class).
    import pickle

    captured: dict = {"results": None, "save_path": None}

    _orig_image_level = pointad_test.image_level_metrics
    _orig_pixel_level = pointad_test.pixel_level_metrics

    def _capture_then_call_image(results, obj, metric, modality="pr_sp"):
        if captured["results"] is None:
            captured["results"] = results
        return _orig_image_level(results, obj, metric, modality)

    def _capture_then_call_pixel(results, obj, metric, modality="anomaly_maps"):
        if captured["results"] is None:
            captured["results"] = results
        return _orig_pixel_level(results, obj, metric, modality)

    pointad_test.image_level_metrics = _capture_then_call_image
    pointad_test.pixel_level_metrics = _capture_then_call_pixel

    ckpt_path = (
        POINTAD_DIR
        / "exps_9_12_4_mv9_mvtec_3d_336_4"
        / cli.checkpoint_class
        / "epoch_15.pth"
    )
    assert ckpt_path.is_file(), f"missing checkpoint: {ckpt_path}"

    save_path = Path(cli.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    args = types.SimpleNamespace(
        data_path=cli.data_path,
        save_path=str(save_path),
        checkpoint_path=str(ckpt_path),
        dataset="mvtec_pc_3d_rgb",
        features_list=cli.features_list,
        image_size=cli.image_size,
        depth=cli.depth,
        n_ctx=cli.n_ctx,
        t_n_ctx=cli.t_n_ctx,
        feature_map_layer=[0, 1, 2, 3],
        metrics=cli.metrics,
        seed=cli.seed,
        sigma=cli.sigma,
        # train_class is used by test.py only to exclude that class from the
        # eval object loop. Since our object list is ["weld"], any non-"weld"
        # value preserves welds in the eval set.
        train_class=cli.checkpoint_class,
        point_size=cli.point_size,
    )

    print("[wrapper] args:", args, flush=True)
    pointad_test.setup_seed(args.seed)
    pointad_test.test(args)

    # Persist the captured raw arrays for post-hoc per-defect breakdown.
    if captured["results"] is not None:
        out_pkl = save_path / "raw_results.pkl"
        # results[obj] contains: gt_sp (list), pr_sp (list), color_pr_sp,
        # integrate_pr_sp, imgs_masks (tensor), anomaly_maps (np.ndarray),
        # color_anomaly_maps, integrate_anomaly_maps.
        # We only need image-level scalars + a per-sample summary for the
        # per-defect breakdown (the pixel maps are large; we keep image-level
        # only here to keep the pickle small).
        slim = {}
        for obj, d in captured["results"].items():
            slim[obj] = {
                "gt_sp": [float(x) for x in d["gt_sp"]],
                "pr_sp": [float(x) for x in d["pr_sp"]],
                "color_pr_sp": [float(x) for x in d["color_pr_sp"]],
                "integrate_pr_sp": [float(x) for x in d["integrate_pr_sp"]],
            }
        with open(out_pkl, "wb") as f:
            pickle.dump(slim, f)
        print(f"[wrapper] wrote raw image-level scores to {out_pkl}", flush=True)


if __name__ == "__main__":
    main()
