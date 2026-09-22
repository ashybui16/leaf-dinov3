"""Stage-2 finetune: expand a stage-1 checkpoint to a dataset with novel classes appended."""

import argparse
import os
import sys
from pathlib import Path

if "--offline" in sys.argv:
    os.environ["YOLO_OFFLINE"] = "1"  # ultralytics.utils.ONLINE is import-time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaf_dinov3.data import check_class_order, resolve_dataset_yaml  # noqa: E402
from leaf_dinov3.trainer import DistillFineTuneTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="stage-1 pretrain checkpoint (.pt)")
    p.add_argument("--data", required=True, help="finetune dataset yaml; novel classes appended at the tail")
    p.add_argument("--pretrain-data", required=True, help="stage-1 dataset yaml, for class-order validation")
    p.add_argument("--lambda-distill", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--fraction", type=float, default=None, help="dataset fraction/count; >1 = image count")
    p.add_argument("--lr0", type=float, default=1e-3)
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--cos-lr", action="store_true")
    p.add_argument("--device", default="0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_class_order(args.pretrain_data, args.data)

    batch = args.batch or (2 if args.smoke else 64)
    imgsz = args.imgsz or (320 if args.smoke else 640)
    fraction = args.fraction if args.fraction is not None else ([8, 8] if args.smoke else 1.0)

    trainer = DistillFineTuneTrainer(
        overrides={
            "model": args.weights,
            "data": resolve_dataset_yaml(args.data),
            "epochs": args.epochs,
            "batch": batch,
            "imgsz": imgsz,
            "lr0": args.lr0,
            "optimizer": args.optimizer,
            "cos_lr": args.cos_lr,
            "device": args.device,
            "fraction": fraction,
        },
        lambda_distill=args.lambda_distill,
    )
    trainer.train()


if __name__ == "__main__":
    main()
