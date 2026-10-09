"""Upload YOLO-WT checkpoints to HuggingFace MPF-AD model repo, forcing IPv4."""
import socket, pathlib

_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig(host, port, family, type, proto, flags)
    v4 = [r for r in res if r[0] == socket.AF_INET]
    return v4 if v4 else res
socket.getaddrinfo = _ipv4

from huggingface_hub import HfApi

REPO_ID = "vpan1226/MPF-AD"
BASE = pathlib.Path(__file__).resolve().parents[1] / "YOLO-WT"

# (src_path_relative_to_BASE, dest_path_in_repo, description)
WEIGHTS = [
    # 4-seed sweep — seed 42 is the canonical public release
    ("Abl_Exp/train/YOLO-WT-250-16-640-SGD-seed42/weights/best.pt",
     "checkpoints/YOLO-WT-seed42-best.pt",
     "Canonical public checkpoint (seed 42, val mAP@50=0.929)"),
    ("Abl_Exp/train/YOLO-WT-250-16-640-SGD/weights/best.pt",
     "checkpoints/YOLO-WT-seed0-best.pt",
     "Seed 0 (val mAP@50=0.881)"),
    ("Abl_Exp/train/YOLO-WT-250-16-640-SGD-seed1/weights/best.pt",
     "checkpoints/YOLO-WT-seed1-best.pt",
     "Seed 1 (val mAP@50=0.893)"),
    ("Abl_Exp/train/YOLO-WT-250-16-640-SGD-seed2/weights/best.pt",
     "checkpoints/YOLO-WT-seed2-best.pt",
     "Seed 2 (val mAP@50=0.905)"),
    # Component ablations
    ("ultralytics/Abl_Exp/train/WDSConv-250-16-640-SGD/weights/best.pt",
     "checkpoints/ablation/WDSConv-only-best.pt",
     "Ablation: WDSConv only (no IWUpSample)"),
    ("ultralytics/Abl_Exp/train/IWUpSample-250-16-640-SGD/weights/best.pt",
     "checkpoints/ablation/IWUpSample-only-best.pt",
     "Ablation: IWUpSample only (no WDSConv)"),
]

api = HfApi()
for src_rel, dest, desc in WEIGHTS:
    src = BASE / src_rel
    if not src.exists():
        print(f"SKIP (not found): {src_rel}")
        continue
    size_mb = src.stat().st_size / 1_048_576
    print(f"Uploading {dest} ({size_mb:.1f} MB) — {desc}")
    api.upload_file(
        path_or_fileobj=str(src),
        path_in_repo=dest,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Add {dest}",
    )
    print(f"  Done.")

print("\nAll weights uploaded.")
print(f"Model repo: https://huggingface.co/{REPO_ID}")
