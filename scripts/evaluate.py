"""Wrap ultralytics' native DetectionValidator (per-class AP, correct letterboxed decode).

YOLO-DINO's own eval.py hand-rolls decode with a naive square resize (no letterbox, so boxes
are subtly wrong) and has a class-order bug scoring base-first labels against numeric-id
order; DetectionValidator gives per-class AP directly and is correct by construction.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ultralytics.models.yolo.detect import DetectionValidator  # noqa: E402

from leaf_dinov3.data import resolve_dataset_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--split", default="val")
    p.add_argument(
        "--base-classes",
        type=int,
        default=None,
        help="number of base classes; if set, also report base vs. novel (index >= this) mAP",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float()
    if hasattr(model, "teacher"):  # already-stripped checkpoints have no teacher to remove
        model.strip_distillation()
    model.eval()

    validator = DetectionValidator(
        args={"data": resolve_dataset_yaml(args.data), "imgsz": args.imgsz, "batch": args.batch, "split": args.split}
    )
    validator(model=model)
    metric = validator.metrics.box

    if args.base_classes is not None:
        class_idx = np.asarray(metric.ap_class_index)
        ap = metric.ap
        base_mask = class_idx < args.base_classes
        if base_mask.any():
            print(f"base  mAP50-95 ({int(base_mask.sum())} classes): {ap[base_mask].mean():.4f}")
        if (~base_mask).any():
            print(f"novel mAP50-95 ({int((~base_mask).sum())} classes): {ap[~base_mask].mean():.4f}")
    print(f"all   mAP50-95: {metric.map:.4f}  mAP50: {metric.map50:.4f}")


if __name__ == "__main__":
    main()
