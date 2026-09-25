"""Verification step 6: synthetic overfit smoke test, no dataset needed.

Run for all four combinations of {yolo26n-leaf, yolov8m} x {d2, d3} -- this is what catches
the end2end/E2ELoss nesting and the 4-level path.
"""

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


@pytest.mark.parametrize("variant", ["d2", "d3"])
@pytest.mark.parametrize("cfg", ["yolo26n-leaf.yaml", "yolov8m.yaml"])
def test_overfit_smoke(cfg, variant):
    torch.manual_seed(0)
    model = DistillDetectionModel(
        cfg,
        ch=3,
        nc=NC,
        verbose=False,
        variant=variant,
        teacher_variant="dinov3_vits16",
        lambda_distill=1.0,
    )
    model.args = DEFAULT_CFG
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    losses = []
    for _ in range(5):
        batch = _fake_batch(2, IMGSZ, NC)
        loss, loss_items = model.loss(batch)
        assert torch.isfinite(loss).all()
        assert "distill_loss" in loss_items
        total = loss.sum()
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(total.item())

    assert losses[-1] < losses[0]

    model.strip_distillation()
    assert not hasattr(model, "teacher")
    model.eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, IMGSZ, IMGSZ))
    assert isinstance(out, tuple)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
