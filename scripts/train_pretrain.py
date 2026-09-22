"""Stage-1 pretrain: whole dataset, all classes, distillation on."""

import argparse
import os
import sys
from pathlib import Path

if "--offline" in sys.argv:
    os.environ["YOLO_OFFLINE"] = "1"  # ultralytics.utils.ONLINE is import-time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaf_dinov3.data import resolve_dataset_yaml  # noqa: E402
from leaf_dinov3.trainer import DistillTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--variant", choices=["d2", "d3"], default="d2")
    p.add_argument("--teacher-variant", default=None)
    p.add_argument("--teacher-weights", default=None)
    p.add_argument("--teacher-input-size", type=int, default=None)
    p.add_argument("--attn-grid", type=int, default=20)
    p.add_argument("--attn-num-heads", type=int, default=8)
    p.add_argument("--lambda-distill", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--fraction", type=float, default=None, help="dataset fraction/count; >1 = image count")
    p.add_argument("--lr0", type=float, default=1e-2)
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--cos-lr", action="store_true")
    p.add_argument("--device", default="0")
    p.add_argument("--smoke", action="store_true", help="local 4GB-GPU profile: tiny model/teacher/batch/epochs")
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        model, teacher_variant, imgsz, batch, epochs, fraction = "yolo26n-leaf.yaml", "dinov3_vits16", 320, 2, 1, [8, 8]
    else:
        model, teacher_variant, imgsz, batch, epochs, fraction = "yolo26s-leaf.yaml", "dinov3_vitb16", 640, 64, 300, 1.0
    model = args.model or model
    teacher_variant = args.teacher_variant or teacher_variant
    imgsz = args.imgsz or imgsz
    batch = args.batch or batch
    epochs = args.epochs or epochs
    fraction = args.fraction if args.fraction is not None else fraction

    trainer = DistillTrainer(
        overrides={
            "model": model,
            "data": resolve_dataset_yaml(args.data),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "lr0": args.lr0,
            "optimizer": args.optimizer,
            "cos_lr": args.cos_lr,
            "device": args.device,
            "fraction": fraction,
        },
        variant=args.variant,
        teacher_variant=teacher_variant,
        teacher_weights=args.teacher_weights,
        teacher_input_size=args.teacher_input_size,
        lambda_distill=args.lambda_distill,
        attn_grid=args.attn_grid,
        attn_num_heads=args.attn_num_heads,
    )
    trainer.train()


if __name__ == "__main__":
    main()
