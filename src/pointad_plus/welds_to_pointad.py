"""Convert Dataset_3D/MVTec3D_Weld/ into PointAD's pre-rendered input format.

Strategy: single-view repetition. PointAD expects ~9 views per sample; we
have 1 Phong render per patch. We replicate each view 9 times. Lossy but
sufficient to get a baseline zero-shot welds number with shipped weights.
Phase 2 replaces this with proper rendering.

This adapter mirrors the on-disk layout produced by
``external/PointAD/multi_view/multiview_mvtec.py`` and the manifest schema
emitted by
``external/PointAD/generate_dataset_json/mvtec_3d_anomaly_mvtect_3d_ad_whole.py``
so the resulting tree can be fed directly to ``test.py`` with
``--dataset mvtec_pc_3d_rgb`` and ``cls_name="weld"``.

Key facts confirmed by reading ``external/PointAD/dataset.py:102-225``:

- ``d2_corrdinate`` (note the upstream typo) is a *directory* of ``.npy``
  files. The loader iterates ``sorted(os.listdir(d2_corrdinate))`` and
  treats ``idx == 0`` (alphabetically first) as a special non-zero-index
  template - a 1-D long array of flat pixel indices ``i in [0, P*P)``
  with ``P = point_size = 336`` where the organized XYZ has geometry.
  All other files are per-view correspondence tensors of shape
  ``(N_nonzero, 3)`` with columns ``[u_x, u_y, is_seen]`` in 336x336
  pixel space.
- ``d3_pc`` is *referenced in the manifest but never opened* in the
  ``mvtec_pc_3d_rgb`` branch of ``__getitem__``. We still point it at the
  xyz tiff so the manifest is honest (and to keep parity with the
  upstream generator).
- ``d2_mask_path`` is a single PNG. The loader does
  ``Image.open(...).convert('L')`` then thresholds at 0. Our SAM2 gts
  fit that contract.
- ``d2_render_img_path`` and ``d2_render_gt_path`` are directories of
  per-view PNGs. PointAD doesn't care how many - it just iterates
  ``sorted(os.listdir(...))``. Upstream ships 9 views (5 x_rot + 4 y_rot).
  We replicate the single Phong render 9 times as ``view_0..view_8.png``.

For the single-view (identity) correspondence, every non-zero pixel
``i`` is visible and projects back to itself, so its 2D coordinate at
336x336 is ``(u, v) = (i % 336, i // 336)`` with ``is_seen = 1``.

To keep disk usage and I/O down, we use NTFS hardlinks (``os.link``) for
the replicated 2d_rendering / 2d_gt files and for the per-view npys
(they are identical across views). The d2_img_path / d3_pc / d2_mask_path
files are hardlinked into the destination tree as well so we don't
duplicate the entire MVTec3D_Weld folder. Hardlinks on the same Windows
volume don't need admin privileges.

Usage::

    python -m src.pointad_plus.welds_to_pointad

Outputs::

    external/datasets/welds_pointad/weld/all_meta.json
    external/datasets/welds_pointad/weld/<phase>/<specie>/{rgb,xyz,gt,
        2d_rendering/<id>/, 2d_gt/<id>/, 2d_3d_cor/<id>/}/...
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "Dataset_3D" / "MVTec3D_Weld" / "weld"
DST_ROOT = REPO_ROOT / "external" / "datasets" / "welds_pointad" / "weld"

# PointAD pads/resizes everything to 336x336 via ``point_size``.
POINT_SIZE = 336
# Number of views to replicate. Matches the 5 x_rot + 4 y_rot scheme in
# external/PointAD/multi_view/multiview_mvtec.py.
N_VIEWS = 9

# Source -> destination phase mapping. PointAD's split is only
# {train, test}; our `validation` good patches get folded into the test
# good bucket so they're still scored (they're zero-shot anyway).
SOURCE_PHASE_TO_DST: dict[str, str] = {
    "train": "train",
    "validation": "test",  # fold val/good into test/good for the zero-shot eval
    "test": "test",
}

CLS_NAME = "weld"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _replicate_file(src: Path, dst: Path) -> None:
    """Link src -> dst (hardlink if possible, else copy).

    Removes any pre-existing dst so re-runs are idempotent.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        # Cross-volume or permission issue - fall back to copy.
        shutil.copy2(src, dst)


def _save_npy(dst: Path, arr: np.ndarray) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, arr)


def _link_npy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _resize_organized_pc(pc: np.ndarray, size: int) -> np.ndarray:
    """Resize an organized HxWx3 XYZ tiff to size x size via nearest-neighbour.

    Nearest preserves the (0,0,0) "no-geometry" sentinels (any anti-aliased
    interpolation would smear them).
    """
    h, w = pc.shape[:2]
    if (h, w) == (size, size):
        return pc.astype(np.float32, copy=False)
    # Index map using nearest-neighbour.
    ys = (np.linspace(0, h - 1, size)).round().astype(np.int64)
    xs = (np.linspace(0, w - 1, size)).round().astype(np.int64)
    return pc[ys[:, None], xs[None, :], :].astype(np.float32, copy=False)


def _compute_nonzero_and_cor(xyz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build the non_zero_index template and identity correspondence.

    Returns:
        non_zero_index: 1-D int64 array of flat indices i in [0, P*P) where
            the resized organized XYZ has geometry (z != 0).
        cor: shape (N_nonzero, 3) int64 array with [u_x, u_y, is_seen=1].
    """
    pc = tifffile.imread(str(xyz_path))
    if pc.ndim != 3 or pc.shape[2] != 3:
        raise ValueError(f"{xyz_path} is not an HxWx3 organized XYZ tiff (got {pc.shape})")
    pc = _resize_organized_pc(pc, POINT_SIZE)
    flat = pc.reshape(-1, 3)
    # Match multiview_mvtec.py: nonzero on the Z channel.
    nz = np.nonzero(flat[:, 2])[0].astype(np.int64)
    # 2D pixel coordinate of each nonzero index at 336x336.
    u_x = (nz % POINT_SIZE).astype(np.int64)
    u_y = (nz // POINT_SIZE).astype(np.int64)
    visibility = np.ones_like(nz, dtype=np.int64)
    cor = np.stack([u_x, u_y, visibility], axis=1)
    return nz, cor


# ---------------------------------------------------------------------------
# Per-sample builder
# ---------------------------------------------------------------------------


def _process_sample(
    src_specie_dir: Path,
    dst_specie_dir: Path,
    file_id: str,
    is_abnormal: bool,
) -> dict:
    """Build all PointAD-format artefacts for one sample. Return manifest entry."""
    # Source files
    src_rgb = src_specie_dir / "rgb" / f"{file_id}.png"
    src_xyz = src_specie_dir / "xyz" / f"{file_id}.tiff"
    src_gt = src_specie_dir / "gt" / f"{file_id}.png"

    # Destination tree
    dst_rgb_dir = dst_specie_dir / "rgb"
    dst_xyz_dir = dst_specie_dir / "xyz"
    dst_gt_dir = dst_specie_dir / "gt"
    dst_render_dir = dst_specie_dir / "2d_rendering" / file_id
    dst_gt_view_dir = dst_specie_dir / "2d_gt" / file_id
    dst_cor_dir = dst_specie_dir / "2d_3d_cor" / file_id

    dst_rgb = dst_rgb_dir / f"{file_id}.png"
    dst_xyz = dst_xyz_dir / f"{file_id}.tiff"
    dst_gt = dst_gt_dir / f"{file_id}.png"

    # 1) Single-file copies (hardlink) for d2_img_path / d3_pc / d2_mask_path.
    _replicate_file(src_rgb, dst_rgb)
    _replicate_file(src_xyz, dst_xyz)
    if src_gt.exists():
        _replicate_file(src_gt, dst_gt)

    # 2) Replicate the rendered RGB and 2D GT N_VIEWS times.
    dst_render_dir.mkdir(parents=True, exist_ok=True)
    dst_gt_view_dir.mkdir(parents=True, exist_ok=True)
    for v in range(N_VIEWS):
        _replicate_file(src_rgb, dst_render_dir / f"view_{v}.png")
        # If no GT exists (e.g. train/good has no defects), synthesise an
        # all-zero mask at source resolution so the loader's resize stays
        # consistent. Otherwise replicate the SAM2 mask.
        if src_gt.exists():
            _replicate_file(src_gt, dst_gt_view_dir / f"view_{v}_gt.png")
        else:
            # Build a zero PNG matching the rgb size.
            zero_png = dst_gt_view_dir / f"view_{v}_gt.png"
            if not zero_png.exists():
                with Image.open(src_rgb) as im:
                    w, h = im.size
                Image.new("L", (w, h), 0).save(zero_png)

    # 3) Per-view correspondence + non_zero_index template.
    nz, cor = _compute_nonzero_and_cor(src_xyz)
    dst_cor_dir.mkdir(parents=True, exist_ok=True)
    # Loader sorts alphabetically and uses idx==0 as the non_zero_index
    # template. "nonzero_indices.npy" < "view_*.npy" alphabetically, so
    # it lands at idx 0 - identical to the upstream MVTec3D layout.
    nz_path = dst_cor_dir / "nonzero_indices.npy"
    _save_npy(nz_path, nz)
    # Save one canonical view, then hardlink the rest.
    cor_view0 = dst_cor_dir / "view_0_cor.npy"
    _save_npy(cor_view0, cor)
    for v in range(1, N_VIEWS):
        _link_npy(cor_view0, dst_cor_dir / f"view_{v}_cor.npy")

    # 4) Build manifest entry. Paths in the manifest are stored as
    #    `<root>/...` style strings - the loader joins them back via
    #    `os.path.join(self.root, img_path)`. To stay consistent with the
    #    upstream generator we store paths *relative to* DST_ROOT.parent
    #    (the `<DATASET_ROOT>` that `--data_path` will point at), i.e.
    #    rooted at `welds_pointad/`. That way `--data_path
    #    external/datasets/welds_pointad` works directly.
    dataset_root = DST_ROOT.parent  # .../welds_pointad
    rel = lambda p: p.relative_to(dataset_root).as_posix()  # noqa: E731

    specie_name = src_specie_dir.name  # "good" or defect class
    return {
        "d2_img_path": rel(dst_rgb),
        "d2_mask_path": rel(dst_gt) if src_gt.exists() else "",
        "d3_pc": rel(dst_xyz),
        "cls_name": CLS_NAME,
        "specie_name": specie_name,
        "anomaly": 1 if is_abnormal else 0,
        "d2_render_img_path": rel(dst_render_dir),
        "d2_render_gt_path": rel(dst_gt_view_dir),
        "d2_corrdinate": rel(dst_cor_dir),  # note: upstream typo preserved
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_dataset() -> dict:
    """Walk SRC_ROOT and emit DST_ROOT in PointAD format. Returns the manifest."""
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_ROOT}")
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    # info[phase][cls_name] = [sample, ...]
    info: dict[str, dict[str, list]] = {"train": defaultdict(list), "test": defaultdict(list)}

    total_normal = 0
    total_anomaly = 0

    for src_phase, dst_phase in SOURCE_PHASE_TO_DST.items():
        phase_dir = SRC_ROOT / src_phase
        if not phase_dir.exists():
            print(f"  [skip] {phase_dir} does not exist")
            continue
        species = sorted(p.name for p in phase_dir.iterdir() if p.is_dir())
        for specie in species:
            src_specie_dir = phase_dir / specie
            rgb_dir = src_specie_dir / "rgb"
            if not rgb_dir.exists():
                continue
            is_abnormal = specie != "good"
            # In src, all phases follow `<phase>/<specie>/...`. In dst we
            # land at `weld/<dst_phase>/<specie>/...`.
            dst_specie_dir = DST_ROOT / dst_phase / specie
            file_ids = sorted(p.stem for p in rgb_dir.glob("*.png"))
            print(
                f"  {src_phase}/{specie:>22s} -> {dst_phase}/{specie:<22s}  "
                f"({len(file_ids)} samples)"
            )
            for file_id in file_ids:
                entry = _process_sample(
                    src_specie_dir=src_specie_dir,
                    dst_specie_dir=dst_specie_dir,
                    file_id=file_id,
                    is_abnormal=is_abnormal,
                )
                info[dst_phase][CLS_NAME].append(entry)
                if is_abnormal:
                    total_anomaly += 1
                else:
                    total_normal += 1

    # Convert defaultdicts to plain dicts for clean JSON.
    info = {phase: dict(cls_map) for phase, cls_map in info.items()}

    meta_path = DST_ROOT / "all_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        f.write("\n")

    # Also write the class-specific meta. PointAD's loader uses this when
    # is_all=False / mode='test' with --train_dataset_name=weld, but the
    # default path (is_all=True) reads all_meta.json. Emit both to be safe.
    cls_meta_path = DST_ROOT / f"{CLS_NAME}_meta.json"
    with open(cls_meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)
        f.write("\n")

    # PointAD's loader reads `<data_path>/all_meta.json` and lists the three
    # per-view directories below with os.listdir() without joining them to
    # the dataset root, so the manifest at <data_path> must carry absolute
    # paths for exactly these keys.
    dataset_root = DST_ROOT.parent
    dir_keys = ("d2_corrdinate", "d2_render_gt_path", "d2_render_img_path")
    root_info = {
        phase: {cls: [{k: (str(dataset_root / v) if k in dir_keys and v else v) for k, v in s.items()}
                      for s in samples]
                for cls, samples in cls_map.items()}
        for phase, cls_map in info.items()
    }
    root_meta_path = dataset_root / "all_meta.json"
    with open(root_meta_path, "w", encoding="utf-8") as f:
        json.dump(root_info, f, indent=4)
        f.write("\n")

    print()
    print(f"Wrote manifest: {meta_path}")
    print(f"Wrote manifest: {cls_meta_path}")
    print(f"Wrote manifest: {root_meta_path} (absolute per-view dirs, read by PointAD)")
    print(f"  total normal samples: {total_normal}")
    print(f"  total anomalous samples: {total_anomaly}")
    print(f"  train cls samples: {len(info.get('train', {}).get(CLS_NAME, []))}")
    print(f"  test cls samples:  {len(info.get('test', {}).get(CLS_NAME, []))}")
    return info


if __name__ == "__main__":
    build_dataset()
