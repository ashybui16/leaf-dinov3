"""Distillation loss terms."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def embedding_distill_loss(student_embeds: list[torch.Tensor], teacher_tokens: torch.Tensor) -> torch.Tensor:
    """Per-level cosine distance `1 - cos(s, t)` between channel-centered student and teacher.

    `student_embeds[i]` is `(B, embed_dim, H_i, W_i)`; `teacher_tokens` is the single teacher
    grid `(B, embed_dim, H_t, W_t)` (there is no learned projection: the student channel count
    is forced to equal `embed_dim`). The student is already on the teacher grid, so the bilinear
    resize of the teacher is a no-op unless `teacher_input_size` differs from the student's.

    Each side has its per-channel mean over (batch, H, W) subtracted first: a few massive DINOv3
    channels are near-constant, and uncentered, predicting that shared mean token everywhere
    already scores 1 - cos ~ 0.33 with no spatial information at all.
    """
    losses = []
    for embed in student_embeds:
        h, w = embed.shape[-2:]
        teacher = F.interpolate(teacher_tokens, size=(h, w), mode="bilinear", align_corners=False).float()
        embed = embed.float()
        s = embed - embed.mean(dim=(0, 2, 3), keepdim=True)
        t = teacher - teacher.mean(dim=(0, 2, 3), keepdim=True)
        losses.append(1 - F.cosine_similarity(s, t, dim=1).mean())
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
    """Per-level row-wise KL `sum(t * (log t - log s))` between teacher and student attention.

    Same gradients as the cross-entropy `-sum(t * log s)`, minus the teacher's row entropy (a
    constant), so the logged value bottoms out at 0 instead of ~6 nats.

    `student_attns[i]` is `(B, N_i, N_i)` over a square `sqrt(N_i)` grid; `teacher_attn` is
    `(B, teacher_grid**2, teacher_grid**2)`. Teacher resizes are cached by grid, so levels
    sharing a grid share one resize (a no-op when the grids already match).
    """
    resized = {}
    losses = []
    for attn in student_attns:
        grid = math.isqrt(attn.shape[-1])
        if grid not in resized:
            with torch.no_grad():
                resized[grid] = resize_attention_map(teacher_attn, teacher_grid, grid)
        t = resized[grid]
        log_s = attn.float().clamp_min(1e-8).log()
        losses.append((t * (t.clamp_min(1e-12).log() - log_s)).sum(dim=-1).mean())
    return torch.stack(losses).mean()
