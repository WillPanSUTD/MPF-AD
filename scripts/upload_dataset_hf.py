"""Upload MPW-AD dataset files to HuggingFace, forcing IPv4 to avoid IPv6 reset issues."""
import socket
import os

# Monkey-patch: force IPv4 only to avoid IPv6 connection reset on HF
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    results = _orig_getaddrinfo(host, port, family, type, proto, flags)
    ipv4 = [r for r in results if r[0] == socket.AF_INET]
    return ipv4 if ipv4 else results
socket.getaddrinfo = _ipv4_only

from huggingface_hub import HfApi, CommitOperationAdd
import pathlib

REPO_ID = "vpan1226/MPW-AD"
REPO_TYPE = "dataset"
BASE = pathlib.Path(__file__).resolve().parents[1]

api = HfApi()

# Upload folder-by-folder to give progress feedback
for folder in ["Train_Data", "Crop_Data", "Depth-Normal_Rendering"]:
    src = BASE / folder
    if not src.exists():
        print(f"SKIP {folder} (not found)")
        continue
    print(f"Uploading {folder} ...")
    api.upload_folder(
        folder_path=str(src),
        path_in_repo=folder,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=f"Add {folder}",
    )
    print(f"  Done: {folder}")

# Upload dataset card README from MPW-AD/README.md (add HF YAML header)
readme_src = BASE / "MPW-AD" / "README.md"
readme_content = readme_src.read_text(encoding="utf-8")
if not readme_content.startswith("---"):
    yaml_header = (
        "---\n"
        "license: cc-by-4.0\n"
        "task_categories:\n"
        "  - image-classification\n"
        "  - object-detection\n"
        "tags:\n"
        "  - anomaly-detection\n"
        "  - industrial\n"
        "  - welding\n"
        "  - 3d\n"
        "  - photometric\n"
        "pretty_name: MPW-AD\n"
        "size_categories:\n"
        "  - 1K<n<10K\n"
        "---\n\n"
    )
    readme_content = yaml_header + readme_content

import tempfile, textwrap
with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
    f.write(readme_content)
    tmp_path = f.name

print("Uploading dataset card README.md ...")
api.upload_file(
    path_or_fileobj=tmp_path,
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type=REPO_TYPE,
    commit_message="Add dataset card",
)
os.unlink(tmp_path)
print("All uploads complete.")
print(f"Dataset: https://huggingface.co/datasets/{REPO_ID}")
