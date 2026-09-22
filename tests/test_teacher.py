"""Verification steps 1-3: import/load on py3.10, shapes+freezing, RoPE attention correctness."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaf_dinov3.teacher import DinoV3Teacher  # noqa: E402


def test_import_and_load():
    """Step 1: DINOv3 imports and loads on Python 3.10 despite setup.py declaring >=3.11."""
    t = DinoV3Teacher("dinov3_vits16")
    assert (t.embed_dim, t.patch_size, t.n_storage_tokens) == (384, 16, 4)


def test_shapes_and_freezing():
    """Step 2: teacher shapes and hard freezing."""
    t = DinoV3Teacher("dinov3_vits16")
    x = torch.rand(2, 3, 320, 320)
    out = t(x)
    assert out["patch_tokens"].shape == (2, 384, 20, 20)

    t.train(True)
    assert t.training is False
    for p in t.parameters():
        assert not p.requires_grad


def test_attention_matches_unwrapped():
    """Step 3 (load-bearing): wrapped compute_attention must reproduce the unwrapped block's output.

    A qkv-forward-hook implementation would miss RoPE (applied inside compute_attention) and
    pass every shape test while computing wrong attention; this is the only check that catches it.
    """
    torch.manual_seed(0)
    t_ref = DinoV3Teacher("dinov3_vits16", need_attention=False)
    t_wrapped = DinoV3Teacher("dinov3_vits16", need_attention=True)
    t_wrapped.load_state_dict(t_ref.state_dict())

    x = torch.rand(2, 3, 320, 320)
    ref_out = t_ref(x)["patch_tokens"]
    wrapped_out = t_wrapped(x)["patch_tokens"]

    assert torch.allclose(ref_out, wrapped_out, atol=1e-4)
    assert t_wrapped._attn is not None

    attn = t_wrapped._attn
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    n = t_wrapped.n_storage_tokens + 1
    patch_only_sums = attn[:, n:, n:].sum(dim=-1)
    assert (patch_only_sums <= 1.0 + 1e-4).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
