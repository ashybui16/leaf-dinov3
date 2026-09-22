"""Dataset-YAML helpers: portable path resolution and the pretrain -> finetune class-order check."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_dataset_yaml(yaml_path: str | Path) -> str:
    """Rewrite a dataset YAML's relative `path:` to an absolute path anchored at this repo's
    root, and return the path to the resolved copy (or the original, if `path:` is already
    absolute).

    Two problems, one fix: `~/.config/Ultralytics/settings.json` can set `datasets_dir` to an
    unrelated project, so ultralytics would resolve any *relative* `path:` there instead of
    here; and a hardcoded *absolute* `path:` checked into `cfg/*.yaml` would only work on the
    machine it was written on. A relative `path:` in the checked-in YAML, resolved here
    against this repo's own location rather than ultralytics' global settings, is portable
    across clones and immune to `datasets_dir`.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    raw_path = Path(data.get("path", "."))
    if raw_path.is_absolute():
        return str(yaml_path)
    data["path"] = str((REPO_ROOT / raw_path).resolve())
    resolved_path = yaml_path.with_name(f".resolved_{yaml_path.name}")
    with open(resolved_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(resolved_path)


def _load_names(yaml_path: str | Path) -> dict[int, str]:
    with open(yaml_path) as f:
        names = yaml.safe_load(f)["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    return {int(k): v for k, v in names.items()}


def check_class_order(pretrain_yaml: str | Path, finetune_yaml: str | Path) -> None:
    """Assert the finetune dataset's first N names match the pretrain dataset's, in order.

    `DistillDetectionModel.expand_num_classes` grows `nc` in place, so index `k` must mean the
    same class at `nc=N` and at `nc=N+M`: novel classes may only be appended at the tail.
    """
    pretrain_names = _load_names(pretrain_yaml)
    finetune_names = _load_names(finetune_yaml)
    n = len(pretrain_names)
    if len(finetune_names) < n:
        raise ValueError(
            f"finetune dataset has fewer classes ({len(finetune_names)}) than the pretrain dataset ({n})"
        )
    for i in range(n):
        if finetune_names[i] != pretrain_names[i]:
            raise ValueError(
                f"class order mismatch at index {i}: pretrain={pretrain_names[i]!r} finetune={finetune_names[i]!r}"
            )
