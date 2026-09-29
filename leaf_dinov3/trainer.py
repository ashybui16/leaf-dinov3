"""Trainers that build a `DistillDetectionModel` instead of the stock `DetectionModel`.

Distillation settings are explicit constructor kwargs, never `overrides` keys: `get_cfg`
(via `check_dict_alignment`) raises `SyntaxError` on any key it doesn't recognize.
"""

from __future__ import annotations

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, RANK

from .model import DistillDetectionModel


def _infer_learnable_resize_student(head) -> bool:
    """Read `learnable_resize_student` back off a loaded head, the same way `attn_num_heads` is below.

    Only learned resamplers (convs/`CARAFE`) have params; `Identity`, `nn.AvgPool2d` and
    `nn.Upsample` have none. If every level already sits at the teacher's stride, the flag can't
    be observed and also doesn't matter, so `False` (the model's own default) is returned.
    """
    return any(True for r in head.cls_resample for _ in r.parameters())


class DistillTrainer(DetectionTrainer):
    """`DetectionTrainer` that builds a `DistillDetectionModel`."""

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict | None = None,
        _callbacks=None,
        *,
        variant: str = "d2",
        teacher_variant: str = "dinov3_vitb16",
        teacher_weights: str | None = None,
        teacher_input_size: int | None = None,
        lambda_distill: float = 1.0,
        attn_num_heads: int = 8,
        learnable_resize_student: bool = False,
    ):
        self.variant = variant
        self.teacher_variant = teacher_variant
        self.teacher_weights = teacher_weights
        self.teacher_input_size = teacher_input_size
        self.lambda_distill = lambda_distill
        self.attn_num_heads = attn_num_heads
        self.learnable_resize_student = learnable_resize_student
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def _build_model(self, cfg, nc: int | None, verbose: bool) -> DistillDetectionModel:
        return DistillDetectionModel(
            cfg,
            nc=nc,
            ch=self.data["channels"],
            verbose=verbose,
            variant=self.variant,
            teacher_variant=self.teacher_variant,
            teacher_weights=self.teacher_weights,
            teacher_input_size=self.teacher_input_size,
            lambda_distill=self.lambda_distill,
            attn_num_heads=self.attn_num_heads,
            learnable_resize_student=self.learnable_resize_student,
        )

    def get_model(self, cfg=None, weights=None, verbose: bool = True) -> DistillDetectionModel:
        """Build a `DistillDetectionModel` for the training dataset's class count."""
        model = self.set_model_names_for_load(
            self._build_model(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model


class DistillFineTuneTrainer(DistillTrainer):
    """Loads a stage-1 checkpoint at its own class count, then expands to the finetune dataset.

    `model=<stage-1 .pt>` is required: `BaseTrainer.setup_model` unpickles the
    `DistillDetectionModel` (student only, see `DistillDetectionModel.__getstate__`) and hands it
    to `get_model` as `weights`. No teacher/d3 args are needed: `variant`/`teacher_variant`/
    `teacher_weights`/`teacher_input_size`/
    `attn_num_heads`/`learnable_resize_student` are all read back off the checkpoint itself, so a
    freshly-built model can never end up architecturally mismatched with the checkpoint it's
    about to load. This matters beyond a state_dict shape check for `attn_num_heads`:
    `SelfAttentionMap`'s q/k `nn.Linear` shapes don't depend on it, so a wrong value would
    silently load fine and then scramble the q/k head-reshape at runtime instead of erroring.
    `learnable_resize_student` has the same failure mode: `cls_resample`/`box_resample` hold a
    param-free pool/bilinear resize on one side or a real resampler (with params) on the other, so a mismatched
    guess wouldn't error either -- it'd just leave real resampler weights randomly initialized
    instead of loaded.

    The teacher is rebuilt from the same `.pth` stage 1 used (`teacher_weights`, or the default
    file in `dinov3/weights/` when that is None), so it must still be loadable at finetune time;
    a relative `teacher_weights` path resolves against the finetune's working directory.
    """

    def get_model(self, cfg=None, weights=None, verbose: bool = True) -> DistillDetectionModel:
        if weights is None:
            raise ValueError("DistillFineTuneTrainer requires `model=<stage-1 checkpoint>.pt`")
        head = weights.model[-1]
        attn_num_heads = head.self_attn[0].num_heads if getattr(head, "self_attn", None) is not None else self.attn_num_heads
        learnable_resize_student = _infer_learnable_resize_student(head)
        model = DistillDetectionModel(
            cfg,
            nc=None,  # keep the checkpoint's own nc
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            variant=weights.variant,
            teacher_variant=weights.teacher_variant,
            teacher_weights=weights.teacher_weights,
            teacher_input_size=weights.teacher_input_size,
            lambda_distill=self.lambda_distill,
            attn_num_heads=attn_num_heads,
            learnable_resize_student=learnable_resize_student,
        )
        model.load(weights)
        model.expand_num_classes(self.data["nc"])
        model.lambda_distill = self.lambda_distill
        return model
