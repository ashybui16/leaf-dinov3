"""Verification step 5: loss unit tests, for both d2 and d3."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaf_dinov3.losses import (  # noqa: E402
    attention_distill_loss,
    embedding_distill_loss,
    normalize_channels,
    resize_attention_map,
)


def test_zero_loss_on_identical_inputs():
    teacher = torch.rand(2, 16, 40, 40)
    student = [teacher.clone()]  # same spatial size, no resize needed
    loss = embedding_distill_loss(student, teacher)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_gradient_flows_to_student():
    teacher = torch.rand(2, 16, 40, 40)
    student = torch.rand(2, 16, 20, 20, requires_grad=True)
    loss = embedding_distill_loss([student], teacher)
    loss.backward()
    assert student.grad is not None
    assert torch.any(student.grad != 0)


def test_normalize_channels_unit_norm():
    x = torch.rand(2, 16, 5, 5) * 10 + 0.1
    n = normalize_channels(x)
    norms = n.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def _random_attention(bs: int, grid: int) -> torch.Tensor:
    n = grid * grid
    logits = torch.rand(bs, n, n)
    return logits.softmax(dim=-1)


def test_resize_attention_map_identity_preserves_row_sums():
    attn = _random_attention(2, 5)
    resized = resize_attention_map(attn, in_grid=5, out_grid=5)
    assert torch.allclose(resized, attn, atol=1e-5)
    assert torch.allclose(resized.sum(dim=-1), attn.sum(dim=-1), atol=1e-4)


def test_resize_attention_map_downsamples_shape():
    attn = _random_attention(2, 8)
    resized = resize_attention_map(attn, in_grid=8, out_grid=4)
    assert resized.shape == (2, 16, 16)


def test_resize_attention_map_upsamples_with_unit_row_sums():
    attn = _random_attention(2, 4)
    resized = resize_attention_map(attn, in_grid=4, out_grid=8)
    assert resized.shape == (2, 64, 64)
    assert torch.allclose(resized.sum(dim=-1), torch.ones(2, 64), atol=1e-5)


def test_attention_distill_loss_zero_on_identical_inputs():
    teacher = _random_attention(2, 4)
    loss = attention_distill_loss([teacher.clone(), teacher.clone()], teacher, teacher_grid=4)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_attention_distill_loss_gradient_flows_to_student():
    teacher = _random_attention(2, 4)
    student = torch.rand(2, 16, 16, requires_grad=True)
    loss = attention_distill_loss([student], teacher, teacher_grid=4)
    loss.backward()
    assert student.grad is not None
    assert torch.any(student.grad != 0)


def test_attention_distill_loss_per_level_grids():
    teacher = _random_attention(2, 4)
    students = [_random_attention(2, 8), _random_attention(2, 4), _random_attention(2, 2)]
    loss = attention_distill_loss(students, teacher, teacher_grid=4)
    assert torch.isfinite(loss)
    same = [resize_attention_map(teacher, 4, g) for g in (8, 4, 2)]
    assert torch.allclose(attention_distill_loss(same, teacher, teacher_grid=4), torch.zeros(()), atol=1e-6)


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
