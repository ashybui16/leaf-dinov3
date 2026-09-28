"""Verification step 4: head unit tests, for both d2 and d3."""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaf_dinov3.heads import DistillDetectD2, DistillDetectD3, DySample  # noqa: E402

DISTILL_DIM = 16
ATTN_HEADS = 2


def _feats(bs: int, ch: tuple, base: int = 40):
    """Fake FPN feature maps: level i has `ch[i]` channels at spatial size `base // 2**i`."""
    return [torch.rand(bs, c, base // (2**i), base // (2**i)) for i, c in enumerate(ch)]


@pytest.mark.parametrize("ch", [(256, 512, 1024), (64, 128, 256, 512)], ids=["3lvl-yolov8m", "4lvl-leaf"])
@pytest.mark.parametrize("end2end", [False, True], ids=["e2e-off", "e2e-on"])
def test_forward_shapes(ch, end2end):
    nc, reg_max, bs = 10, 16, 2
    strides = [8 * 2**i for i in range(len(ch))]
    head = DistillDetectD2(
        nc=nc, reg_max=reg_max, end2end=end2end, ch=ch, distill_dim=DISTILL_DIM, strides=strides, resize_student=True
    )
    head.stride = torch.tensor(strides)
    head.train()
    x = _feats(bs, ch)
    preds = head(x)

    o2m = preds["one2many"] if end2end else preds
    n_anchors = sum((40 // 2**i) ** 2 for i in range(len(ch)))
    assert o2m["boxes"].shape == (bs, 4 * reg_max, n_anchors)
    assert o2m["scores"].shape == (bs, nc, n_anchors)
    assert "dino_embed" in o2m
    assert len(o2m["dino_embed"]) == len(ch)
    g = 40 * strides[0] // 16  # teacher grid (patch stride 16)
    for embed in o2m["dino_embed"]:
        assert embed.shape == (bs, DISTILL_DIM, g, g)

    if end2end:
        o2o = preds["one2one"]
        assert "dino_embed" not in o2o
        assert o2o["boxes"].shape == (bs, 4 * reg_max, n_anchors)
        assert o2o["scores"].shape == (bs, nc, n_anchors)


@pytest.mark.parametrize("ch", [(256, 512, 1024), (64, 128, 256, 512)], ids=["3lvl-yolov8m", "4lvl-leaf"])
def test_forward_shapes_no_resize(ch):
    nc, reg_max, bs = 10, 16, 2
    strides = [8 * 2**i for i in range(len(ch))]
    head = DistillDetectD2(
        nc=nc, reg_max=reg_max, ch=ch, distill_dim=DISTILL_DIM, strides=strides, resize_student=False
    )
    head.stride = torch.tensor(strides)
    head.train()
    x = _feats(bs, ch)
    preds = head(x)
    for i, embed in enumerate(preds["dino_embed"]):
        native = 40 // 2**i
        assert embed.shape == (bs, DISTILL_DIM, native, native)


def test_bias_init_does_not_clobber_cls_slice():
    """Regression: cls bias write is sliced to [:nc], so the distill tail keeps its own init."""
    ch = (64, 128, 256)
    strides = [8, 16, 32]
    head = DistillDetectD2(nc=10, reg_max=16, end2end=False, ch=ch, distill_dim=DISTILL_DIM, strides=strides)
    head.stride = torch.tensor(strides)
    before = [seq[-1].bias.data[10:].clone() for seq in head.cv3]
    head.bias_init()
    after = [seq[-1].bias.data[10:] for seq in head.cv3]
    for b, a in zip(before, after):
        assert torch.equal(b, a)
    for seq in head.cv3:
        assert seq[-1].bias.data[:10].allclose(seq[-1].bias.data[0].expand(10))


def test_strip_distillation_restores_stock_shapes():
    ch = (64, 128, 256)
    nc, reg_max = 10, 16
    strides = [8, 16, 32]
    head = DistillDetectD2(nc=nc, reg_max=reg_max, end2end=True, ch=ch, distill_dim=DISTILL_DIM, strides=strides)
    head.stride = torch.tensor(strides)
    head.bias_init()
    head.strip_distillation()

    for seq in head.cv3:
        assert seq[-1].out_channels == nc
    assert head.distill_dim == 0
    assert not hasattr(head, "cls_resample")

    head.eval()
    x = _feats(2, ch)
    with torch.no_grad():
        preds = head.forward_head(x, **head.one2many)
    assert "dino_embed" not in preds
    assert preds["scores"].shape[1] == nc


def test_expand_num_classes_preserves_weights_and_shifts_tail():
    ch = (64, 128, 256)
    nc, reg_max, new_nc = 10, 16, 12
    strides = [8, 16, 32]
    head = DistillDetectD2(nc=nc, reg_max=reg_max, end2end=True, ch=ch, distill_dim=DISTILL_DIM, strides=strides)
    head.stride = torch.tensor(strides)
    head.bias_init()

    old_cls_w = [seq[-1].weight.data[:nc].clone() for seq in head.cv3]
    old_tail_w = [seq[-1].weight.data[nc:].clone() for seq in head.cv3]
    old_o2o_w = [seq[-1].weight.data.clone() for seq in head.one2one_cv3]

    head.expand_num_classes(new_nc)

    assert head.nc == new_nc
    assert head.no == new_nc + 4 * reg_max
    for i, seq in enumerate(head.cv3):
        assert seq[-1].out_channels == new_nc + DISTILL_DIM
        assert torch.equal(seq[-1].weight.data[:nc], old_cls_w[i])
        assert torch.equal(seq[-1].weight.data[new_nc:], old_tail_w[i])
    for i, seq in enumerate(head.one2one_cv3):
        assert seq[-1].out_channels == new_nc
        assert torch.equal(seq[-1].weight.data[:nc], old_o2o_w[i])


def test_expand_num_classes_biases_new_rows_with_rare_object_prior():
    """Regression: new class rows must not be left at ~0 bias, or BCE blows up on backgrounds."""
    ch = (64, 128, 256)
    nc, reg_max, new_nc = 10, 16, 12
    strides = [8, 16, 32]
    head = DistillDetectD2(nc=nc, reg_max=reg_max, end2end=True, ch=ch, distill_dim=DISTILL_DIM, strides=strides)
    head.stride = torch.tensor(strides)
    head.bias_init()
    head.expand_num_classes(new_nc)

    for i, seq in enumerate(head.cv3):
        expected = math.log(5 / new_nc / (640 / head.stride[i]) ** 2)
        assert seq[-1].bias.data[nc:new_nc].allclose(torch.full((new_nc - nc,), expected))
    for i, seq in enumerate(head.one2one_cv3):
        expected = math.log(5 / new_nc / (640 / head.stride[i]) ** 2)
        assert seq[-1].bias.data[nc:new_nc].allclose(torch.full((new_nc - nc,), expected))


@pytest.mark.parametrize("ch", [(256, 512, 1024), (64, 128, 256, 512)], ids=["3lvl-yolov8m", "4lvl-leaf"])
@pytest.mark.parametrize("end2end", [False, True], ids=["e2e-off", "e2e-on"])
def test_d3_forward_shapes(ch, end2end):
    nc, reg_max, bs = 10, 16, 2
    p2 = len(ch) == 4
    strides = [(4 if p2 else 8) * 2**i for i in range(len(ch))]
    head = DistillDetectD3(
        nc=nc,
        reg_max=reg_max,
        end2end=end2end,
        ch=ch,
        distill_dim=DISTILL_DIM,
        attn_num_heads=ATTN_HEADS,
        strides=strides,
        resize_student=True,
    )
    head.stride = torch.tensor(strides)
    head.train()
    x = _feats(bs, ch)
    preds = head(x)

    o2m = preds["one2many"] if end2end else preds
    n_anchors = sum((40 // 2**i) ** 2 for i in range(len(ch)))
    assert o2m["boxes"].shape == (bs, 4 * reg_max, n_anchors)
    assert o2m["scores"].shape == (bs, nc, n_anchors)
    assert "dino_embed" in o2m
    assert "dino_box_embed" in o2m
    assert "dino_attn" in o2m
    assert len(o2m["dino_box_embed"]) == len(ch)
    assert len(o2m["dino_attn"]) == len(ch)
    g = 40 * strides[0] // 16  # teacher grid (patch stride 16)
    for embed in o2m["dino_embed"] + o2m["dino_box_embed"]:
        assert embed.shape == (bs, DISTILL_DIM, g, g)
    for attn in o2m["dino_attn"]:
        assert attn.shape == (bs, g * g, g * g)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)

    if end2end:
        o2o = preds["one2one"]
        assert "dino_box_embed" not in o2o
        assert "dino_attn" not in o2o
        assert o2o["boxes"].shape == (bs, 4 * reg_max, n_anchors)


def test_d3_bias_init_does_not_clobber_box_distill_tail():
    """Regression: box bias write is sliced to [:4*reg_max], so the box distill tail is untouched."""
    ch = (64, 128, 256)
    reg_max = 16
    strides = [8, 16, 32]
    head = DistillDetectD3(nc=10, reg_max=reg_max, end2end=False, ch=ch, distill_dim=DISTILL_DIM, strides=strides)
    head.stride = torch.tensor(strides)
    before = [seq[-1].bias.data[4 * reg_max :].clone() for seq in head.cv2]
    head.bias_init()
    for seq, b in zip(head.cv2, before):
        assert torch.equal(seq[-1].bias.data[4 * reg_max :], b)
        assert seq[-1].bias.data[: 4 * reg_max].allclose(torch.full((4 * reg_max,), 2.0))
    # cls side (inherited from D2) still correctly sliced too
    for seq in head.cv3:
        assert seq[-1].bias.data[:10].allclose(seq[-1].bias.data[0].expand(10))


def test_d3_eval_skips_distill_outputs():
    """Distill outputs only feed the train-time loss, so eval (validation) must not compute them."""
    ch = (64, 128, 256)
    strides = [8, 16, 32]
    head = DistillDetectD3(
        nc=10, ch=ch, distill_dim=DISTILL_DIM, attn_num_heads=ATTN_HEADS, strides=strides, resize_student=True
    )
    head.stride = torch.tensor(strides)
    head.eval()
    with torch.no_grad():
        preds = head.forward_head(_feats(2, ch), **head.one2many)
    assert "dino_embed" not in preds
    assert "dino_box_embed" not in preds
    assert "dino_attn" not in preds


def test_d3_strip_distillation_restores_stock_shapes():
    ch = (64, 128, 256)
    nc, reg_max = 10, 16
    strides = [8, 16, 32]
    head = DistillDetectD3(nc=nc, reg_max=reg_max, end2end=True, ch=ch, distill_dim=DISTILL_DIM, strides=strides)
    head.stride = torch.tensor(strides)
    head.bias_init()
    head.strip_distillation()

    for seq in head.cv2:
        assert seq[-1].out_channels == 4 * reg_max
    for seq in head.cv3:
        assert seq[-1].out_channels == nc
    assert head.distill_dim == 0
    assert not hasattr(head, "self_attn")
    assert not hasattr(head, "box_resample")

    head.eval()
    x = _feats(2, ch)
    with torch.no_grad():
        preds = head.forward_head(x, **head.one2many)
    assert "dino_embed" not in preds
    assert "dino_box_embed" not in preds
    assert "dino_attn" not in preds
    assert preds["boxes"].shape[1] == 4 * reg_max


def test_d3_expand_num_classes_leaves_box_branch_untouched():
    """cv2/self_attn don't depend on nc, so D3 can reuse D2's expand_num_classes unchanged."""
    ch = (64, 128, 256)
    nc, reg_max, new_nc = 10, 16, 12
    strides = [8, 16, 32]
    head = DistillDetectD3(nc=nc, reg_max=reg_max, end2end=True, ch=ch, distill_dim=DISTILL_DIM, strides=strides)
    head.stride = torch.tensor(strides)
    head.bias_init()

    old_cv2_w = [seq[-1].weight.data.clone() for seq in head.cv2]
    old_self_attn_q_w = [m.q.weight.data.clone() for m in head.self_attn]

    head.expand_num_classes(new_nc)

    assert head.nc == new_nc
    for i, seq in enumerate(head.cv2):
        assert torch.equal(seq[-1].weight.data, old_cv2_w[i])
    for i, m in enumerate(head.self_attn):
        assert torch.equal(m.q.weight.data, old_self_attn_q_w[i])
    for seq in head.cv3:
        assert seq[-1].out_channels == new_nc + DISTILL_DIM


@pytest.mark.parametrize("ch", [(256, 512, 1024), (64, 128, 256, 512)], ids=["3lvl-yolov8m", "4lvl-leaf"])
def test_d3_forward_shapes_no_resize(ch):
    nc, reg_max, bs = 10, 16, 2
    p2 = len(ch) == 4
    strides = [(4 if p2 else 8) * 2**i for i in range(len(ch))]
    head = DistillDetectD3(
        nc=nc,
        reg_max=reg_max,
        ch=ch,
        distill_dim=DISTILL_DIM,
        attn_num_heads=ATTN_HEADS,
        strides=strides,
        resize_student=False,
    )
    head.stride = torch.tensor(strides)
    head.train()
    x = _feats(bs, ch)
    preds = head(x)
    native_grids = [40 // 2**i for i in range(len(ch))]
    for embed, g in zip(preds["dino_embed"] + preds["dino_box_embed"], native_grids * 2):
        assert embed.shape[-2:] == (g, g)
    # attention grids match native resolution, except P2 (level 0) is bilinearly resized to P3's
    # (level 1's) grid first, so its attention matrix stays a manageable size
    attn_grids = [native_grids[1] if p2 else native_grids[0], *native_grids[1:]]
    for attn, g in zip(preds["dino_attn"], attn_grids):
        assert attn.shape == (bs, g * g, g * g)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    # remaining levels differ: confirms N_i tracks each level's own native grid, not a shared teacher grid
    n_per_level = [a.shape[-1] for a in preds["dino_attn"]]
    assert len(set(n_per_level)) == len(ch) - (1 if p2 else 0)


def test_dysample_upsamples_and_learns():
    up = DySample(DISTILL_DIM, scale=2)
    x = torch.rand(2, DISTILL_DIM, 5, 5)
    out = up(x)
    assert out.shape == (2, DISTILL_DIM, 10, 10)
    out.sum().backward()
    assert up.offset.weight.grad is not None and torch.any(up.offset.weight.grad != 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
