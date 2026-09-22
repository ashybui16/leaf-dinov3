import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from leaf_dinov3.data import REPO_ROOT, check_class_order, resolve_dataset_yaml  # noqa: E402


def _write_yaml(tmp_path: Path, name: str, names) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump({"names": names}))
    return p


def test_matching_prefix_passes(tmp_path):
    pretrain = _write_yaml(tmp_path, "pretrain.yaml", {0: "a", 1: "b"})
    finetune = _write_yaml(tmp_path, "finetune.yaml", {0: "a", 1: "b", 2: "novel"})
    check_class_order(pretrain, finetune)  # should not raise


def test_reordered_prefix_fails(tmp_path):
    pretrain = _write_yaml(tmp_path, "pretrain.yaml", {0: "a", 1: "b"})
    finetune = _write_yaml(tmp_path, "finetune.yaml", {0: "b", 1: "a", 2: "novel"})
    with pytest.raises(ValueError):
        check_class_order(pretrain, finetune)


def test_fewer_finetune_classes_fails(tmp_path):
    pretrain = _write_yaml(tmp_path, "pretrain.yaml", {0: "a", 1: "b"})
    finetune = _write_yaml(tmp_path, "finetune.yaml", {0: "a"})
    with pytest.raises(ValueError):
        check_class_order(pretrain, finetune)


def test_resolve_dataset_yaml_anchors_relative_path_to_repo_root(tmp_path):
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(yaml.safe_dump({"path": "SomeDataset", "train": "images/train", "names": {0: "a"}}))
    resolved = resolve_dataset_yaml(cfg)
    assert resolved != str(cfg)
    with open(resolved) as f:
        data = yaml.safe_load(f)
    assert data["path"] == str((REPO_ROOT / "SomeDataset").resolve())
    assert data["train"] == "images/train"  # untouched


def test_resolve_dataset_yaml_leaves_absolute_path_untouched(tmp_path):
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(yaml.safe_dump({"path": "/already/absolute", "names": {0: "a"}}))
    resolved = resolve_dataset_yaml(cfg)
    assert resolved == str(cfg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
