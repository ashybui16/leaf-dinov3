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

    # yolo train hyperparameters (default=None: falls back to ultralytics' own default)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--warmup-epochs", type=float, default=None)
    p.add_argument("--warmup-momentum", type=float, default=None)
    p.add_argument("--warmup-bias-lr", type=float, default=None)
    p.add_argument("--box", type=float, default=None, help="box loss gain")
    p.add_argument("--cls", type=float, default=None, help="cls loss gain")
    p.add_argument("--dfl", type=float, default=None, help="dfl loss gain")
    p.add_argument("--nbs", type=int, default=None, help="nominal batch size")

    # yolo train run control
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--close-mosaic", type=int, default=None)
    p.add_argument("--save-period", type=int, default=None)
    p.add_argument("--multi-scale", type=float, default=None)
    p.add_argument("--time", type=float, default=None, help="max train hours (overrides epochs)")
    p.add_argument("--freeze", type=int, default=None, help="freeze first N layers")
    p.add_argument("--cache", type=str, default=None, choices=["ram", "disk", "true", "false"])
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--val", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--exist-ok", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--single-cls", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--rect", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None)

    # yolo train augmentation
    p.add_argument("--hsv-h", type=float, default=None)
    p.add_argument("--hsv-s", type=float, default=None)
    p.add_argument("--hsv-v", type=float, default=None)
    p.add_argument("--degrees", type=float, default=None)
    p.add_argument("--translate", type=float, default=None)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--shear", type=float, default=None)
    p.add_argument("--perspective", type=float, default=None)
    p.add_argument("--flipud", type=float, default=None)
    p.add_argument("--fliplr", type=float, default=None)
    p.add_argument("--bgr", type=float, default=None)
    p.add_argument("--mosaic", type=float, default=None)
    p.add_argument("--mixup", type=float, default=None)
    p.add_argument("--cutmix", type=float, default=None)
    p.add_argument("--copy-paste", type=float, default=None)

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

    overrides = {
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
    }
    cache = {"true": True, "false": False}.get(args.cache, args.cache)
    extra = {
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "nbs": args.nbs,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "close_mosaic": args.close_mosaic,
        "save_period": args.save_period,
        "multi_scale": args.multi_scale,
        "time": args.time,
        "freeze": args.freeze,
        "cache": cache,
        "project": args.project,
        "name": args.name,
        "val": args.val,
        "exist_ok": args.exist_ok,
        "verbose": args.verbose,
        "deterministic": args.deterministic,
        "single_cls": args.single_cls,
        "rect": args.rect,
        "resume": args.resume,
        "amp": args.amp,
        "plots": args.plots,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "bgr": args.bgr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
        "copy_paste": args.copy_paste,
    }
    overrides.update({k: v for k, v in extra.items() if v is not None})

    trainer = DistillTrainer(
        overrides=overrides,
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
