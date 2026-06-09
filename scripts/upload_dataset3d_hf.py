"""Upload Dataset_3D/MVTec3D_Weld to HuggingFace MPW-AD repo.

Uses a temporary directory junction so upload_large_folder sees the
correct Dataset_3D/ path prefix without copying 3.4 GB of data.
Forces IPv4 to avoid IPv6 connection reset on this host.
"""
import socket, os, subprocess, pathlib

_orig = socket.getaddrinfo
def _ipv4(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig(host, port, family, type, proto, flags)
    v4 = [r for r in res if r[0] == socket.AF_INET]
    return v4 if v4 else res
socket.getaddrinfo = _ipv4

from huggingface_hub import HfApi

REPO_ID = "vpan1226/MPW-AD"
SRC = r"F:\dataset\LUT_AD_DataSet\Dataset_3D\MVTec3D_Weld"
TMP = r"F:\tmp_hf_upload"
JUNCTION = os.path.join(TMP, "Dataset_3D", "MVTec3D_Weld")

# Set up temporary directory with junction so upload lands at Dataset_3D/MVTec3D_Weld/
os.makedirs(os.path.join(TMP, "Dataset_3D"), exist_ok=True)
if not os.path.exists(JUNCTION):
    ret = subprocess.run(
        ["cmd", "/c", "mklink", "/J", JUNCTION, SRC],
        capture_output=True, text=True
    )
    print(ret.stdout.strip() or ret.stderr.strip())
else:
    print(f"Junction already exists: {JUNCTION}")

api = HfApi()
print(f"Uploading Dataset_3D/MVTec3D_Weld (3.4 GB) via upload_large_folder ...")
api.upload_large_folder(
    folder_path=TMP,
    repo_id=REPO_ID,
    repo_type="dataset",
)
print("Done.")
print(f"Dataset: https://huggingface.co/datasets/{REPO_ID}")
