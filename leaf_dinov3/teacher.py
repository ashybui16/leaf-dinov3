"""Frozen DINOv3 teacher wrapper.

`_ensure_dinov3_on_syspath` runs at import time (not in `__init__`) because checkpoints saved
before `DistillDetectionModel.__getstate__` excluded the teacher still pickle a `DinoV3Teacher`:
a process that only *loads* one of those must still be able to resolve `dinov3.*` submodules
even though it never calls the constructor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DINOV3_DIR = REPO_ROOT / "dinov3"


def _ensure_dinov3_on_syspath() -> None:
    path = str(DINOV3_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)  # position 0: resolves `dinov3` to dinov3/dinov3/, not the outer dir


_ensure_dinov3_on_syspath()

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_DEFAULT_WEIGHTS = {
    "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}


def _load_backbone(variant: str, weights: str):
    """Build a DINOv3 backbone via `dinov3.hub.backbones` directly, bypassing `hubconf.py`.

    `torch.hub.load(..., source="local")` execs the whole `hubconf.py`, which eagerly imports
    every entrypoint (segmentors, detectors, dinotxt, classifiers) and pulls in dependencies
    (e.g. `torchmetrics`) not installed in `mu` and irrelevant to the backbone we need.
    """
    from dinov3.hub import backbones

    return getattr(backbones, variant)(weights=weights)


class _RopeAwareAttention:
    """Picklable replacement for `SelfAttention.compute_attention` that also stashes weights.

    A plain closure can't be pickled ("Can't pickle local object"), which broke mid-training
    checkpoint saves whenever `need_attention=True` (d3) back when the teacher was pickled into
    checkpoints; it still lets old teacher-carrying checkpoints unpickle. Storing the state on a module-level callable instead of a
    closure keeps the class trivially picklable; the only reference cycle is `attn <-> self`,
    which pickle's object memoization handles fine.

    The stashed map is a separate per-head softmax over patch keys only (the first `n_prefix`
    CLS/storage tokens dropped), head-averaged; the real output still uses full attention.
    """

    def __init__(self, attn: nn.Module, n_prefix: int) -> None:
        self.attn = attn
        self.n_prefix = n_prefix

    def __call__(self, qkv: torch.Tensor, attn_bias=None, rope=None) -> torch.Tensor:
        attn = self.attn
        B, N, _ = qkv.shape
        C = attn.qkv.in_features
        h = attn.num_heads
        q, k, v = (t.transpose(1, 2) for t in qkv.reshape(B, N, 3, h, C // h).unbind(2))
        if rope is not None:
            q, k = attn.apply_rope(q, k, rope)
        logits = q @ k.transpose(-2, -1) * attn.scale
        w = logits.softmax(dim=-1)
        n = self.n_prefix
        attn.last_attn = logits[..., n:, n:].softmax(dim=-1).mean(dim=1)  # patch-only, head-avg
        return (w @ v).transpose(1, 2).reshape(B, N, C)


class DinoV3Teacher(nn.Module):
    """Frozen DINOv3 backbone that produces patch-token (and optionally attention) targets."""

    def __init__(
        self,
        variant: str = "dinov3_vitb16",
        input_size: int | None = None,
        weights: str | Path | None = None,
        need_attention: bool = False,
    ) -> None:
        super().__init__()
        if weights is None:
            weights = DINOV3_DIR / "weights" / _DEFAULT_WEIGHTS[variant]
        self.model = _load_backbone(variant, str(weights))

        self.variant = variant
        self.input_size = input_size
        self.need_attention = need_attention
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.num_heads = self.model.num_heads
        self.n_storage_tokens = self.model.n_storage_tokens

        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

        if need_attention:
            self._wrap_last_block_attention()

        self.eval()
        self.requires_grad_(False)

    def train(self, mode: bool = True) -> "DinoV3Teacher":
        """Frozen: ultralytics' `model.train()` must not unfreeze the teacher."""
        return super().train(False)

    def _wrap_last_block_attention(self) -> None:
        """Replace `compute_attention` on the last block so softmax weights are stashed.

        RoPE is applied *inside* `SelfAttention.compute_attention`, so a qkv-forward-hook would
        miss the rotation and compute wrong attention. `_RopeAwareAttention` mirrors the
        original SDPA math exactly (same scale, no mask, no dropout) and additionally stores
        the head-averaged patch-only softmax weights on the wrapped `attn` module.
        """
        attn = self.model.blocks[-1].attn
        attn.compute_attention = _RopeAwareAttention(attn, self.n_storage_tokens + 1)

    @property
    def _attn(self) -> torch.Tensor:
        """The last block's head-averaged patch-only attention from the most recent forward pass."""
        return self.model.blocks[-1].attn.last_attn

    @property
    def grid_size(self) -> int | None:
        """Patch grid side length for a fixed `input_size`, or None when it tracks the input."""
        return None if self.input_size is None else self.input_size // self.patch_size

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the frozen teacher on a `[0, 1]` image batch (ultralytics' `batch["img"]`)."""
        if self.input_size is None:
            h, w = images.shape[-2:]
            assert h % self.patch_size == 0 and w % self.patch_size == 0, (
                f"input size ({h}, {w}) must be divisible by patch_size={self.patch_size}"
            )
            x = images
        else:
            assert self.input_size % self.patch_size == 0
            x = F.interpolate(images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)

        x = (x - self._mean) / self._std
        patch_tokens = self.model.get_intermediate_layers(x, n=1, reshape=True, norm=True)[0]
        out = {"patch_tokens": patch_tokens}
        if self.need_attention:
            out["attn_map"] = self._attn
        return out
