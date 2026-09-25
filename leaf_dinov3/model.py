"""DetectionModel subclass that hot-swaps the Detect head for a distillation-widened one.

No `parse_model` registration is needed: build the stock model via `super().__init__(cfg)`,
then replace `self.model[-1]` with the custom head, hand-copying the attributes `parse_model`
stamps on every layer (`i`, `f`, `type`, `np`) plus `stride`, then call `bias_init()`. This
sidesteps needing to touch the protected `my-ultralytics/` package at all.
"""

from __future__ import annotations

import torch
from ultralytics.nn.tasks import DetectionModel

from .heads import DistillDetectD2, DistillDetectD3
from .losses import attention_distill_loss, embedding_distill_loss
from .teacher import DinoV3Teacher

_HEAD_CLASSES = {"d2": DistillDetectD2, "d3": DistillDetectD3}


class DistillDetectionModel(DetectionModel):
    """A YOLO `DetectionModel` whose head also predicts DINOv3-distillation targets."""

    def __init__(
        self,
        cfg: str | dict = "yolo26n-leaf.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        variant: str = "d2",
        teacher_variant: str = "dinov3_vitb16",
        teacher_weights: str | None = None,
        teacher_input_size: int | None = None,
        teacher_pretrained: bool = True,
        lambda_distill: float = 1.0,
        attn_num_heads: int = 8,
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if variant not in _HEAD_CLASSES:
            raise ValueError(f"Unknown distillation variant {variant!r}; available: {sorted(_HEAD_CLASSES)}")
        self.variant = variant
        self.lambda_distill = lambda_distill

        old = self.model[-1]
        ch_levels = tuple(seq[0].conv.in_channels for seq in old.cv2)  # infers 3 or 4 levels
        end2end = bool(self.yaml.get("end2end", False))

        self.teacher = DinoV3Teacher(
            variant=teacher_variant,
            input_size=teacher_input_size,
            weights=teacher_weights,
            need_attention=(variant == "d3"),
            pretrained=teacher_pretrained,
        )

        extra = {"attn_num_heads": attn_num_heads} if variant == "d3" else {}
        new = _HEAD_CLASSES[variant](
            nc=old.nc, reg_max=old.reg_max, end2end=end2end, ch=ch_levels, distill_dim=self.teacher.embed_dim, **extra
        )
        new.stride = old.stride
        new.i, new.f, new.type = old.i, old.f, old.type
        device = next(self.parameters()).device
        new = new.to(device)
        new.bias_init()
        new.np = sum(p.numel() for p in new.parameters())
        self.model[-1] = new
        self.teacher = self.teacher.to(device)
        self.criterion = None

    def loss(self, batch: dict, preds=None) -> tuple[torch.Tensor, dict]:
        """Detection loss plus a distillation term, appended as an extra loss-vector element."""
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.forward(batch["img"])
        det_loss, loss_items = self.criterion(preds, batch)
        distill = self._distill_loss(batch, preds) * self.lambda_distill
        loss = torch.cat([det_loss, distill.view(1)])
        loss_items = {**loss_items, "distill_loss": distill.detach()}
        return loss, loss_items

    def _distill_loss(self, batch: dict, preds) -> torch.Tensor:
        """Zero unless training and the one2many head carries `dino_embed`."""
        p = preds["one2many"] if isinstance(preds, dict) and "one2many" in preds else preds
        if not self.training or not isinstance(p, dict) or "dino_embed" not in p:
            return torch.zeros((), device=batch["img"].device)
        with torch.no_grad():
            teacher_out = self.teacher(batch["img"])
        loss = embedding_distill_loss(p["dino_embed"], teacher_out["patch_tokens"])
        if self.variant == "d3":
            loss = loss + embedding_distill_loss(p["dino_box_embed"], teacher_out["patch_tokens"])
            teacher_grid = teacher_out["patch_tokens"].shape[-1]
            loss = loss + attention_distill_loss(p["dino_attn"], teacher_out["attn_map"], teacher_grid)
        return loss

    def expand_num_classes(self, new_nc: int) -> None:
        """Grow the head to `new_nc` classes, preserving base-class weights and the distill tail.

        Existing names are kept as-is; only the newly added indices get a numeric placeholder,
        so a direct (non-trainer) caller doesn't lose real class names it had already set.
        """
        self.model[-1].expand_num_classes(new_nc)
        self.yaml["nc"] = new_nc
        old_names = getattr(self, "names", {})
        self.names = {i: old_names.get(i, str(i)) for i in range(new_nc)}
        self.criterion = None  # v8DetectionLoss caches nc/no at construction

    def strip_distillation(self) -> None:
        """Remove the teacher and shrink the head back to stock shapes, for export/inference."""
        self.model[-1].strip_distillation()
        del self.teacher
        self.criterion = None
