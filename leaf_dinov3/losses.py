"""Distillation loss terms."""

from __future__ import annotations

import math

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
    """Resize an affinity matrix `(B, N_in, N_in)` to `(B, N_out, N_out)`, rows renormalized.

    The query grid and the key grid are interpolated independently: each flattened dim is
    reshaped back to its own `in_grid x in_grid` spatial grid, bilinearly resized, and
    reflattened. Bilinear resizing scales row sums by ~`(out_grid / in_grid) ** 2`, so each row
    is renormalized to sum to 1, matching the student's softmax rows.
    """
    b, n_in, _ = attn.shape
    out_grid_sq = out_grid * out_grid
    x = attn.reshape(b * n_in, 1, in_grid, in_grid)
    x = F.interpolate(x, size=(out_grid, out_grid), mode="bilinear", align_corners=False)
    x = x.reshape(b, n_in, out_grid_sq)  # (B, Q_in, K_out)
    x = x.transpose(1, 2).reshape(b * out_grid_sq, 1, in_grid, in_grid)  # (B*K_out, 1, Q_in grid)
    x = F.interpolate(x, size=(out_grid, out_grid), mode="bilinear", align_corners=False)
    x = x.reshape(b, out_grid_sq, out_grid_sq).transpose(1, 2)  # (B, Q_out, K_out)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def attention_distill_loss(
    student_attns: list[torch.Tensor], teacher_attn: torch.Tensor, teacher_grid: int
) -> torch.Tensor:
    """Per-level SmoothL1 between student attention maps and the teacher map resized to each level.

    `student_attns[i]` is `(B, N_i, N_i)` over a square `sqrt(N_i)` grid; `teacher_attn` is
    `(B, teacher_grid**2, teacher_grid**2)`. Teacher resizes are cached by grid, so levels
    sharing a grid (P2 resized to P3) share one resize.
    """
    resized = {}
    losses = []
    for attn in student_attns:
        grid = math.isqrt(attn.shape[-1])
        if grid not in resized:
            with torch.no_grad():
                resized[grid] = resize_attention_map(teacher_attn, teacher_grid, grid)
        losses.append(F.smooth_l1_loss(attn, resized[grid]))
    return torch.stack(losses).mean()
