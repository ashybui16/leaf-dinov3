"""Detect-head variants that append DINOv3-distillation channels to the one2many branch."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.nn.modules.head import Detect


def _extend_last_conv(branch: nn.Sequential, extra: int) -> None:
    """Widen a 1x1 `nn.Conv2d` tail by `extra` output channels, preserving pretrained rows."""
    old = branch[-1]
    assert isinstance(old, nn.Conv2d) and old.kernel_size == (1, 1)
    new = nn.Conv2d(old.in_channels, old.out_channels + extra, kernel_size=1)
    new.weight.data[: old.out_channels] = old.weight.data
    new.bias.data[: old.out_channels] = old.bias.data
    branch[-1] = new


class CARAFE(nn.Module):
    """Content-aware reassembly upsampler (CARAFE): each output pixel is a softmax-weighted sum
    of a `k_up x k_up` input neighbourhood, with the kernel predicted from the content."""

    def __init__(self, dim: int, scale: int = 2, k_up: int = 5, k_enc: int = 3, mid: int = 64):
        super().__init__()
        self.scale = scale
        self.k_up = k_up
        self.comp = nn.Conv2d(dim, mid, kernel_size=1)
        self.enc = nn.Conv2d(mid, k_up**2 * scale**2, kernel_size=k_enc, padding=k_enc // 2)
        nn.init.normal_(self.enc.weight, std=0.001)
        nn.init.zeros_(self.enc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, dim, H, W)` -> `(B, dim, scale*H, scale*W)`."""
        b, c, h, w = x.shape
        s, k = self.scale, self.k_up
        kernels = self.enc(self.comp(x)).reshape(b, k * k, s * s, h, w).softmax(dim=1)
        patches = F.unfold(x, k, padding=k // 2).reshape(b, c, k * k, h, w)
        out = torch.einsum("bckhw,bkshw->bcshw", patches, kernels).reshape(b, c * s * s, h, w)
        return F.pixel_shuffle(out, s)


def _make_resampler(dim: int, stride: int, target_stride: int, learnable: bool = False) -> nn.Module:
    """Map a stride-`stride` embedding onto the teacher's stride-`target_stride` grid.

    If `learnable` is False, it has no params: finer levels are average-pooled (each output is
    the mean of exactly the student cells under one teacher patch), coarser levels bilinearly
    upsampled. Otherwise finer levels get a chain of linear 3x3 stride-2 convs, coarser levels a
    `CARAFE`.
    """
    if stride == target_stride:
        return nn.Identity()
    if not learnable:
        if stride < target_stride:
            return nn.AvgPool2d(target_stride // stride)
        return nn.Upsample(scale_factor=stride / target_stride, mode="bilinear", align_corners=False)
    if stride < target_stride:
        n = int(math.log2(target_stride // stride))
        assert stride * 2**n == target_stride, f"stride {stride} vs teacher {target_stride}: not a power of 2"
        return nn.Sequential(*(nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1) for _ in range(n)))
    scale = stride // target_stride
    assert target_stride * scale == stride and scale & (scale - 1) == 0, (
        f"stride {stride} vs teacher {target_stride}: not a power of 2"
    )
    return CARAFE(dim, scale)


class DistillDetectD2(Detect):
    """Detect head widened with per-level DINOv3 patch-embedding targets on the cls branch.

    Each level's embedding is resampled onto the teacher's patch grid by a student-side
    resampler (`cls_resample`): param-free pooling/bilinear, or learned if `learnable_resize_student`.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        distill_dim: int = 768,
        *,
        strides: tuple,
        target_stride: int = 16,
        learnable_resize_student: bool = False,
    ):
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        self.distill_dim = distill_dim
        for seq in self.cv3:
            _extend_last_conv(seq, distill_dim)
        self.cls_resample = nn.ModuleList(
            _make_resampler(distill_dim, s, target_stride, learnable_resize_student) for s in strides
        )

    def forward_head(
        self, x: list[torch.Tensor], box_head: nn.Module = None, cls_head: nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Mirrors `Detect.forward_head`, but the cls conv now emits `nc + distill_dim` channels."""
        if box_head is None or cls_head is None:
            return {}  # fused-inference contract, preserved
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        cls_raw = [cls_head[i](x[i]) for i in range(self.nl)]
        scores = torch.cat([c[:, : self.nc].reshape(bs, self.nc, -1) for c in cls_raw], dim=-1)
        out = {"boxes": boxes, "scores": scores, "feats": x}
        if self.training and cls_head is self.cv3 and self.distill_dim:  # train-time one2many only; o2o convs stay narrow
            out["dino_embed"] = [self.cls_resample[i](c[:, self.nc :]) for i, c in enumerate(cls_raw)]  # teacher grid
        return out

    def strip_distillation(self) -> None:
        """Shrink `cv3` back to stock width and drop the distill tail and its resamplers."""
        for seq in self.cv3:
            old = seq[-1]
            new = nn.Conv2d(old.in_channels, self.nc, kernel_size=1)
            new.weight.data[:] = old.weight.data[: self.nc]
            new.bias.data[:] = old.bias.data[: self.nc]
            seq[-1] = new
        del self.cls_resample
        self.distill_dim = 0
        self.np = sum(p.numel() for p in self.parameters())

    def expand_num_classes(self, new_nc: int) -> None:
        """Grow `nc`, moving the distill tail on `cv3` to the new tail position.

        New class rows get the same rare-object bias prior `bias_init()` gives every class
        (`log(5 / nc / (640 / stride) ** 2)`); left at the default near-zero conv init, novel
        classes that never appear in any label would predict ~0.5 at every anchor instead of
        ~0.01, inflating BCE loss by orders of magnitude once summed over anchors and divided
        by the (small) foreground-normalized `target_scores_sum`.
        """
        assert new_nc >= self.nc
        old_nc = self.nc
        for i, seq in enumerate(self.cv3):
            old = seq[-1]
            new = nn.Conv2d(old.in_channels, new_nc + self.distill_dim, kernel_size=1)
            new.weight.data[:old_nc] = old.weight.data[:old_nc]
            new.weight.data[new_nc:] = old.weight.data[old_nc:]
            new.bias.data[:old_nc] = old.bias.data[:old_nc]
            new.bias.data[new_nc:] = old.bias.data[old_nc:]
            if new_nc > old_nc:
                new.bias.data[old_nc:new_nc] = math.log(5 / new_nc / (640 / self.stride[i]) ** 2)
            seq[-1] = new
        if getattr(self, "one2one_cv3", None) is not None:
            for i, seq in enumerate(self.one2one_cv3):
                old = seq[-1]  # stock width: exactly old_nc channels, no distill tail
                new = nn.Conv2d(old.in_channels, new_nc, kernel_size=1)
                new.weight.data[:old_nc] = old.weight.data
                new.bias.data[:old_nc] = old.bias.data
                if new_nc > old_nc:
                    new.bias.data[old_nc:new_nc] = math.log(5 / new_nc / (640 / self.stride[i]) ** 2)
                seq[-1] = new
        self.nc = new_nc
        self.no = new_nc + self.reg_max * 4
        self.np = sum(p.numel() for p in self.parameters())


class SelfAttentionMap(nn.Module):
    """q/k-only self-attention: no `v`, no output projection, just a head-averaged affinity map."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: `(B, N, dim)` -> head-averaged attention `(B, N, N)`."""
        b, n, c = x.shape
        h = self.num_heads
        q = self.q(x).reshape(b, n, h, c // h).transpose(1, 2)
        k = self.k(x).reshape(b, n, h, c // h).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        return attn.mean(dim=1)


class DistillDetectD3(DistillDetectD2):
    """D2 plus a box-branch embedding target and a per-level self-attention-map target.

    The box-tail embeddings get their own resamplers (`box_resample`) onto the teacher grid, and
    each level's attention runs on those resampled embeddings, so every level's attention map
    has the teacher's own shape.
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
        distill_dim: int = 768,
        attn_num_heads: int = 8,
        *,
        strides: tuple,
        target_stride: int = 16,
        learnable_resize_student: bool = False,
    ):
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
            distill_dim=distill_dim,
            strides=strides,
            target_stride=target_stride,
            learnable_resize_student=learnable_resize_student,
        )
        for seq in self.cv2:
            _extend_last_conv(seq, distill_dim)
        self.box_resample = nn.ModuleList(
            _make_resampler(distill_dim, s, target_stride, learnable_resize_student) for s in strides
        )
        self.self_attn = nn.ModuleList(SelfAttentionMap(distill_dim, attn_num_heads) for _ in range(self.nl))

    def forward_head(
        self, x: list[torch.Tensor], box_head: nn.Module = None, cls_head: nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Mirrors `DistillDetectD2.forward_head`, but the box conv now also emits an embed tail."""
        if box_head is None or cls_head is None:
            return {}  # fused-inference contract, preserved
        bs = x[0].shape[0]
        n_box = 4 * self.reg_max
        box_raw = [box_head[i](x[i]) for i in range(self.nl)]
        boxes = torch.cat([b[:, :n_box].reshape(bs, n_box, -1) for b in box_raw], dim=-1)
        cls_raw = [cls_head[i](x[i]) for i in range(self.nl)]
        scores = torch.cat([c[:, : self.nc].reshape(bs, self.nc, -1) for c in cls_raw], dim=-1)
        out = {"boxes": boxes, "scores": scores, "feats": x}
        if self.training and cls_head is self.cv3 and self.distill_dim:  # train-time one2many only; o2o convs stay narrow
            out["dino_embed"] = [self.cls_resample[i](c[:, self.nc :]) for i, c in enumerate(cls_raw)]  # teacher grid
        if self.training and box_head is self.cv2 and self.distill_dim:  # train-time one2many only; o2o convs stay narrow
            box_embeds = [self.box_resample[i](b[:, n_box:]) for i, b in enumerate(box_raw)]  # teacher grid
            out["dino_box_embed"] = box_embeds
            tokens = [e.flatten(2).transpose(1, 2) for e in box_embeds]  # (bs, H_i*W_i, distill_dim)
            out["dino_attn"] = [self.self_attn[i](t) for i, t in enumerate(tokens)]
        return out

    def bias_init(self) -> None:
        """Like `Detect.bias_init`, but the one2many box line is sliced to `[:4*reg_max]`.

        Stock writes `a[-1].bias.data[:] = 2.0` unrestricted, which would clobber the box
        distill tail we appended to `cv2`.
        """
        n_box = 4 * self.reg_max
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:n_box] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)  # cls
        if getattr(self, "one2one_cv2", None) is not None:  # narrow throughout; unaffected
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)  # cls

    def strip_distillation(self) -> None:
        """Shrink `cv2` and `cv3` back to stock width and drop the self-attention and resamplers."""
        super().strip_distillation()
        n_box = 4 * self.reg_max
        for seq in self.cv2:
            old = seq[-1]
            new = nn.Conv2d(old.in_channels, n_box, kernel_size=1)
            new.weight.data[:] = old.weight.data[:n_box]
            new.bias.data[:] = old.bias.data[:n_box]
            seq[-1] = new
        del self.self_attn, self.box_resample
        self.np = sum(p.numel() for p in self.parameters())
