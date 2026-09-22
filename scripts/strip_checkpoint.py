"""Export step: remove the DINOv3 teacher and the distill channels from a trained checkpoint."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    before = sum(p.numel() for p in model.parameters())
    if hasattr(model, "teacher"):  # already-stripped checkpoints have no teacher to remove
        model.strip_distillation()
    after = sum(p.numel() for p in model.parameters())
    print(f"params: {before:,} -> {after:,}")
    torch.save({"model": model}, args.out)


if __name__ == "__main__":
    main()
