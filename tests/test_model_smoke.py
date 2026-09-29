"""Verification step 6: synthetic overfit smoke test, no dataset needed.

Run for all four combinations of {yolo26n-leaf, yolov8m} x {d2, d3} -- this is what catches
the end2end/E2ELoss nesting and the 4-level path.
"""

import copy
import io
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics.utils import DEFAULT_CFG  # noqa: E402

from leaf_dinov3.model import DistillDetectionModel  # noqa: E402

IMGSZ = 320
NC = 10


def _fake_batch(bs: int, imgsz: int, nc: int, boxes_per_image: int = 2):
    """Targets must be grouped contiguously by image index, matching real dataloader collation."""
    img = torch.rand(bs, 3, imgsz, imgsz)
    batch_idx = torch.repeat_interleave(torch.arange(bs), boxes_per_image).float()
    n_boxes = bs * boxes_per_image
    cls = torch.randint(0, nc, (n_boxes, 1)).float()
    bboxes = torch.rand(n_boxes, 4) * 0.5 + 0.25  # keep boxes away from the image edge
    return {"img": img, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}


@pytest.mark.parametrize("learnable_resize_student", [True, False], ids=["learnable", "fixed"])
@pytest.mark.parametrize("variant", ["d2", "d3"])
@pytest.mark.parametrize("cfg", ["yolo26n-leaf.yaml", "yolov8m.yaml"])
def test_overfit_smoke(cfg, variant, learnable_resize_student):
    torch.manual_seed(0)
    model = DistillDetectionModel(
        cfg,
        ch=3,
        nc=NC,
        verbose=False,
        variant=variant,
        teacher_variant="dinov3_vits16",
        lambda_distill=1.0,
        learnable_resize_student=learnable_resize_student,
    )
    model.args = DEFAULT_CFG
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    losses = []
    batch = _fake_batch(2, IMGSZ, NC)  # reused every step: this is an overfit test, not a generalization one
    for _ in range(5):
        loss, loss_items = model.loss(batch)
        assert torch.isfinite(loss).all()
        assert "distill_loss" in loss_items
        total = loss.sum()
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())

    # min, not last: raw SGD on yolov8m occasionally spikes on one step, with or without distillation
    assert min(losses[1:]) < losses[0]

    model.strip_distillation()
    assert not hasattr(model, "teacher")
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, IMGSZ, IMGSZ))
    assert isinstance(out, tuple)


def test_checkpoint_excludes_teacher():
    """Pickles (checkpoints) and deepcopies (EMA) drop the frozen teacher but keep its config."""
    model = DistillDetectionModel("yolov8m.yaml", ch=3, nc=NC, verbose=False, variant="d3", teacher_variant="dinov3_vits16")
    buf = io.BytesIO()
    torch.save({"ema": copy.deepcopy(model).half()}, buf)
    buf.seek(0)
    ckpt = torch.load(buf, weights_only=False)["ema"]

    assert hasattr(model, "teacher")  # the live model keeps it
    assert not hasattr(ckpt, "teacher")
    assert not any(k.startswith("teacher.") for k in ckpt.state_dict())
    assert buf.getbuffer().nbytes < sum(p.numel() for p in model.model.parameters()) * 2 * 1.1  # fp16 student only
    assert (ckpt.teacher_variant, ckpt.teacher_weights, ckpt.teacher_input_size) == ("dinov3_vits16", None, None)

    fresh = DistillDetectionModel("yolov8m.yaml", ch=3, nc=NC, verbose=False, variant="d3", teacher_variant="dinov3_vits16")
    fresh.load(ckpt, verbose=False)
    for k, v in model.model.state_dict().items():
        assert torch.allclose(fresh.model.state_dict()[k].float(), v.half().float()), k


def test_strict_load_from_teacherless_copy_keeps_teacher():
    """ultralytics' NaN recovery strict-loads last.pt's (teacher-less) EMA state into the live model."""
    model = DistillDetectionModel("yolov8m.yaml", ch=3, nc=NC, verbose=False, variant="d2", teacher_variant="dinov3_vits16")
    teacher_before = {k: v.clone() for k, v in model.teacher.state_dict().items()}
    ema = copy.deepcopy(model)
    model.load_state_dict(ema.state_dict())  # strict=True
    for k, v in model.teacher.state_dict().items():
        assert torch.equal(v, teacher_before[k]), k
    result = ema.load_state_dict(model.state_dict(), strict=False)  # teacher-less side: only teacher keys extra
    assert not result.missing_keys
    assert result.unexpected_keys and all(k.startswith("teacher.") for k in result.unexpected_keys)


def test_distill_term_is_batch_scaled_like_detection_loss():
    """ultralytics' det loss is `per-sample * batch_size`; the distill element must match that scale."""
    torch.manual_seed(0)
    model = DistillDetectionModel("yolov8m.yaml", ch=3, nc=NC, verbose=False, variant="d2", teacher_variant="dinov3_vits16")
    model.args = DEFAULT_CFG
    model.train()
    bs = 3
    loss, items = model.loss(_fake_batch(bs, IMGSZ, NC))
    assert torch.allclose(loss[-1], items["distill_loss"] * bs)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
