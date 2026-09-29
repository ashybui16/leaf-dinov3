"""Verification step 5: loss unit tests, for both d2 and d3."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaf_dinov3.losses import (  # noqa: E402
    attention_distill_loss,
    embedding_distill_loss,
    resize_attention_map,
)


def test_embedding_loss_zero_at_teacher_direction():
    """Cosine distance: 0 for any positive rescaling of the teacher, 2 for its negation."""
    teacher = torch.randn(2, 16, 40, 40)
    assert torch.allclose(embedding_distill_loss([teacher * 3.0], teacher), torch.zeros(()), atol=1e-6)
    assert torch.allclose(embedding_distill_loss([-teacher], teacher), torch.full((), 2.0), atol=1e-6)


def test_embedding_loss_ignores_shared_mean_token():
    """Centering: predicting the teacher's near-constant mean token everywhere earns ~nothing."""
    torch.manual_seed(0)
    teacher = torch.randn(2, 64, 10, 10)
    teacher[:, :4] += 20.0  # a few massive, near-constant channels, like DINOv3's
    mean_token = teacher.mean(dim=(0, 2, 3), keepdim=True).expand_as(teacher)
    loss = embedding_distill_loss([mean_token + 1e-3 * torch.randn_like(teacher)], teacher)
    assert loss > 0.9


def test_gradient_flows_to_student():
    teacher = torch.rand(2, 16, 40, 40)
    student = torch.rand(2, 16, 20, 20, requires_grad=True)
    loss = embedding_distill_loss([student], teacher)
    loss.backward()
    assert student.grad is not None
    assert torch.any(student.grad != 0)


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


def test_attention_kl_zero_on_identical_inputs():
    teacher = _random_attention(2, 4)
    loss = attention_distill_loss([teacher.clone(), teacher.clone()], teacher, teacher_grid=4)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-6)


def test_attention_kl_direction_is_teacher_to_student():
    """KL(t || s), not KL(s || t): the teacher is the target distribution."""
    teacher, student = _random_attention(2, 4) ** 4, _random_attention(2, 4)  # **4: sharper, unnormalized
    teacher = teacher / teacher.sum(dim=-1, keepdim=True)
    expected = (teacher * (teacher.log() - student.log())).sum(dim=-1).mean()
    swapped = (student * (student.log() - teacher.log())).sum(dim=-1).mean()
    loss = attention_distill_loss([student], teacher, teacher_grid=4)
    assert torch.allclose(loss, expected, atol=1e-5)
    assert not torch.allclose(loss, swapped, atol=1e-3)


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
