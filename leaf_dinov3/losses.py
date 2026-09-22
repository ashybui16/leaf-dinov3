"""Distillation loss terms."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_channels(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """L2-normalize `x` along `dim`."""
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-6)


def embedding_distill_loss(
    student_embeds: list[torch.Tensor], teacher_tokens: torch.Tensor
) -> torch.Tensor:
    """Per-level SmoothL1 between L2-normalized student embeddings and resized teacher tokens.

    `student_embeds[i]` is `(B, embed_dim, H_i, W_i)`; `teacher_tokens` is the single teacher
    grid `(B, embed_dim, H_t, W_t)`, bilinearly resized down to each level (there is no learned
    projection: the student channel count is forced to equal `embed_dim`).
    """
    losses = []
    for embed in student_embeds:
        h, w = embed.shape[-2:]
        teacher = F.interpolate(teacher_tokens, size=(h, w), mode="bilinear", align_corners=False)
        losses.append(F.smooth_l1_loss(normalize_channels(embed), normalize_channels(teacher)))
    return torch.stack(losses).mean()


def resize_attention_map(attn: torch.Tensor, in_grid: int, out_grid: int) -> torch.Tensor:
    """Resize an affinity matrix `(B, N_in, N_in)` to `(B, N_out, N_out)`.

    The query grid and the key grid are interpolated independently: each flattened dim is
    reshaped back to its own `in_grid x in_grid` spatial grid, bilinearly resized, and
    reflattened. Applied once, under `no_grad`, to bring the teacher's native-resolution
    attention down to `attn_grid`.
    """
    b, n_in, _ = attn.shape
    out_grid_sq = out_grid * out_grid
    x = attn.reshape(b * n_in, 1, in_grid, in_grid)
    x = F.interpolate(x, size=(out_grid, out_grid), mode="bilinear", align_corners=False)
    x = x.reshape(b, n_in, out_grid_sq)  # (B, Q_in, K_out)
    x = x.transpose(1, 2).reshape(b * out_grid_sq, 1, in_grid, in_grid)  # (B*K_out, 1, Q_in grid)
    x = F.interpolate(x, size=(out_grid, out_grid), mode="bilinear", align_corners=False)
    x = x.reshape(b, out_grid_sq, out_grid_sq)  # (B, K_out, Q_out)
    return x.transpose(1, 2)  # (B, Q_out, K_out)


def attention_distill_loss(student_attns: list[torch.Tensor], teacher_attn: torch.Tensor) -> torch.Tensor:
    """Per-level SmoothL1 between student attention maps and the resized teacher attention map.

    Every student level is pooled to `attn_grid` and the teacher is resized to the same grid,
    so no per-level shape bookkeeping is needed here.
    """
    losses = [F.smooth_l1_loss(attn, teacher_attn) for attn in student_attns]
    return torch.stack(losses).mean()
