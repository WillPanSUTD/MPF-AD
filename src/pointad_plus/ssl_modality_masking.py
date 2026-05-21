"""Self-supervised modality-masking training for the photometric fusion block.

For each batch:
  full     = fusion(M=5 modalities)
  dropped  = fusion(M=4, one random modality removed)
  loss     = ||dropped - full.detach()||_2^2  (mean over tokens)

Inputs are precomputed CLIP-ViT-L token maps (frozen backbone). The fusion
block sees 5 token-map tensors and learns to be invariant to which 4 of 5
modalities are present.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from src.pointad_plus.photometric_fusion import MultiPhotometricFusion


def masking_loss(
    model: MultiPhotometricFusion,
    channels: List[torch.Tensor],
    dropout_seed: int,
) -> torch.Tensor:
    """L2 between fusion-with-all-5 (target) and fusion-with-4-of-5 (prediction)."""
    rng = random.Random(dropout_seed)
    drop_idx = rng.randrange(len(channels))
    with torch.no_grad():
        fused_full = model(channels).detach()
    remaining = [c for i, c in enumerate(channels) if i != drop_idx]
    fused_dropped = model(remaining)
    return F.mse_loss(fused_dropped, fused_full)


class CLIPTokenDataset(Dataset):
    """Reads precomputed CLIP token maps from disk.

    Expects directory layout:
      <root>/<sample_id>/lut.pt
      <root>/<sample_id>/phong.pt
      ...
    Each .pt is a (N, D) tensor (single-sample CLIP token map).
    """
    def __init__(self, root: Path):
        self.root = Path(root)
        self.sample_dirs = sorted(d for d in self.root.iterdir() if d.is_dir())
        self.modalities = ("lut", "phong", "diffuse", "specular", "normal")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> List[torch.Tensor]:
        d = self.sample_dirs[idx]
        return [torch.load(d / f"{m}.pt") for m in self.modalities]


def precompute_clip_tokens(
    manifest_json: Path,
    out_root: Path,
    clip_model,  # passed in from the wrapper
    clip_preprocess,
    device: str = "cuda:0",
) -> None:
    """One-time pass: encode each modality of each train sample through CLIP,
    save token map to disk. Trades disk for repeated CLIP forwards during
    training.

    Wrapper is responsible for loading PointAD's CLIP backbone (frozen).
    """
    import json
    from PIL import Image
    meta = json.loads(Path(manifest_json).read_text())
    train_samples = meta["train"]["weld"]
    out_root.mkdir(parents=True, exist_ok=True)
    for s in train_samples:
        d = out_root / s["id"]
        if d.exists() and len(list(d.glob("*.pt"))) == 5:
            continue
        d.mkdir(exist_ok=True)
        for m, path in s["multi_photo"].items():
            img = clip_preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                # NOTE: this depends on PointAD's CLIP wrapper; we want the
                # per-patch token map, not the pooled output. The wrapper
                # script (Task P2-E groundwork) handles the actual extraction
                # and writes (N, D) tensors here.
                tokens = clip_model.encode_image_with_tokens(img).cpu()  # placeholder API
            torch.save(tokens.squeeze(0), d / f"{m}.pt")


def train(
    token_root: Path,
    out_ckpt: Path,
    clip_dim: int | None = None,
    num_heads: int = 8,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda:0",
) -> None:
    ds = CLIPTokenDataset(token_root)
    if len(ds) == 0:
        raise RuntimeError(f"[ssl] no sample dirs under {token_root}")
    # Auto-detect CLIP token dim from the first cached tensor if not given.
    if clip_dim is None:
        probe = ds[0][0]
        clip_dim = int(probe.shape[-1])
        print(f"[ssl] auto-detected clip_dim={clip_dim} from {token_root}")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    collate_fn=lambda x: [torch.stack([s[i] for s in x]) for i in range(5)])
    model = MultiPhotometricFusion(clip_dim=clip_dim, num_heads=num_heads,
                                    init="mean").to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ssl] fusion params: {n_params:,} ({n_params * 4 / 1024 / 1024:.2f} MB fp32)")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    step = 0
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch in dl:
            batch = [b.to(device) for b in batch]
            loss = masking_loss(model, batch, dropout_seed=step)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
            if step % 20 == 0:
                print(f"[ssl] epoch={epoch} step={step} loss={loss.item():.6f}", flush=True)
            step += 1
        ep_mean = sum(epoch_losses) / max(len(epoch_losses), 1)
        losses.append(ep_mean)
        print(f"[ssl] epoch={epoch} mean_loss={ep_mean:.6f}", flush=True)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_ckpt)
    print(f"[ssl] saved fusion checkpoint to {out_ckpt}", flush=True)
    print(f"[ssl] per-epoch mean losses: {['%.6f' % v for v in losses]}", flush=True)


def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True,
                    help="Directory of precomputed CLIP token maps (per-sample subdirs)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip_dim", type=int, default=None,
                    help="Override CLIP feature dim; auto-detected from cache if omitted")
    args = ap.parse_args()
    train(args.tokens, args.out, epochs=args.epochs,
          batch_size=args.batch_size, lr=args.lr, clip_dim=args.clip_dim)


if __name__ == "__main__":
    _main()
