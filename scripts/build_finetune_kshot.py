"""Build class-balanced k-shot fine-tune datasets from a base + novel dataset pair.

Each input dataset is described by a standard repo dataset YAML (`path`, `train`, `val`,
`names` -- the same shape as `cfg/coco.yaml`/`cfg/visdrone.yaml`). The base dataset's classes
keep their ids; the novel dataset's classes are appended at the tail (offset by the base
dataset's class count), matching the pretrain -> finetune class-order contract checked by
`leaf_dinov3.data.check_class_order`.

For each k in --k-values, selects a training set where every class (base classes, then novel
classes) has close to k boxes, using an independent per-class greedy sampling pass per k
(TFA/DeFRCN-style seed generation). The val set (full base val + full novel val) is
materialized once and shared, unsampled, across all k values. Images/labels are copied (not
symlinked) into `finetune_kshot/`, so it's self-contained and portable (e.g. into a Docker
volume mount) instead of depending on the source datasets' absolute host paths.
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class DatasetSpec:
    tag: str
    images_dir: dict[str, Path]  # split -> dir
    labels_dir: dict[str, Path]  # split -> dir
    class_offset: int
    names: dict[int, str]  # already offset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-yaml", default="cfg/coco.yaml", help="dataset yaml whose classes keep their ids")
    p.add_argument("--novel-yaml", default="cfg/hotspot_ship.yaml", help="dataset yaml appended at the tail")
    p.add_argument("--base-tag", default=None, help="output filename prefix for the base dataset (default: its yaml's stem)")
    p.add_argument("--novel-tag", default=None, help="output filename prefix for the novel dataset (default: its yaml's stem)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-values", default="1,3,5,10,30")
    p.add_argument("--out-dir", default="finetune_kshot")
    p.add_argument("--cfg-dir", default="cfg")
    return p.parse_args()


def images_to_labels_dir(images_dir: Path) -> Path:
    """Mirror ultralytics' own `images` -> `labels` path convention, so both dataset-yaml
    shapes in this repo work (`images/train2017` and `train/images` alike)."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            return Path(*parts[:i], "labels", *parts[i + 1 :])
    raise ValueError(f"no 'images' path component in {images_dir}; can't infer its labels dir")


def load_dataset(yaml_path: Path, tag: str | None, class_offset: int) -> DatasetSpec:
    """Load a dataset yaml. Assumes contiguous 0-based class ids, since `class_offset` is
    computed by the caller as another dataset's class count."""
    data = yaml.safe_load(yaml_path.read_text())
    root = Path(data["path"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    train_images = root / data["train"]
    val_images = root / data["val"]
    names = {int(k): v for k, v in data["names"].items()}
    return DatasetSpec(
        tag=tag or yaml_path.stem,
        images_dir={"train": train_images, "val": val_images},
        labels_dir={"train": images_to_labels_dir(train_images), "val": images_to_labels_dir(val_images)},
        class_offset=class_offset,
        names={class_offset + i: n for i, n in names.items()},
    )


def index_labels(
    ds: DatasetSpec,
    all_counts: dict[tuple[str, str], Counter],
    class_index: dict[int, list[tuple[str, str]]],
    stem_to_file: dict[str, dict[str, str]],
) -> None:
    images_dir, labels_dir = ds.images_dir["train"], ds.labels_dir["train"]
    files: dict[str, str] = {}
    for p in images_dir.iterdir():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stem in files:
            raise ValueError(f"duplicate stem {p.stem!r} in {images_dir}: {files[p.stem]} vs {p.name}")
        files[p.stem] = p.name
    stem_to_file[ds.tag] = files

    for label_path in labels_dir.glob("*.txt"):
        stem = label_path.stem
        if stem not in files:
            print(f"WARNING: skipping orphaned label {label_path} (no matching image)")
            continue
        counts: Counter = Counter()
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0]) + ds.class_offset
            counts[cls] += 1
        key = (ds.tag, stem)
        all_counts[key] = counts
        for cls in counts:
            class_index.setdefault(cls, []).append(key)


def select_kshot(
    class_index: dict[int, list[tuple[str, str]]],
    all_counts: dict[tuple[str, str], Counter],
    k: int,
    num_classes: int,
    seed: int,
) -> tuple[set[tuple[str, str]], Counter]:
    selected: set[tuple[str, str]] = set()
    running: Counter = Counter()

    for c in range(num_classes):
        pool = [key for key in class_index.get(c, []) if key not in selected]
        rng = random.Random(f"{seed}:{c}")
        rng.shuffle(pool)

        while running[c] < k and pool:
            fit = next((key for key in pool if running[c] + all_counts[key][c] <= k), None)
            if fit is None:
                fit = min(pool, key=lambda key: all_counts[key][c])
            pool.remove(fit)
            selected.add(fit)
            for cls, n in all_counts[fit].items():
                running[cls] += n

    return selected, running


def report_counts(k: int, running: Counter, names: dict[int, str], selected: set[tuple[str, str]]) -> None:
    counts_by_tag = Counter(tag for tag, _ in selected)
    tally = ", ".join(f"{n} {tag}" for tag, n in counts_by_tag.items())
    print(f"--- k={k}: {len(selected)} images selected ({tally}) ---")
    for c in sorted(names):
        cnt = running.get(c, 0)
        print(f"  {c:3d} {names[c]:<15s} count={cnt:4d}  overshoot={cnt - k:+d}")


def _remap_lines(lines: list[str], offset: int) -> list[str]:
    remapped = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        parts[0] = str(int(parts[0]) + offset)
        remapped.append(" ".join(parts))
    return remapped


def _write_label(src_label: Path, dst_label: Path, offset: int) -> None:
    lines = src_label.read_text().splitlines() if src_label.exists() else []
    remapped = _remap_lines(lines, offset)
    dst_label.write_text("\n".join(remapped) + ("\n" if remapped else ""))


def materialize_train(
    selected: set[tuple[str, str]],
    stem_to_file: dict[str, dict[str, str]],
    datasets: dict[str, DatasetSpec],
    out_dir: Path,
) -> None:
    images_out = out_dir / "images" / "train"
    labels_out = out_dir / "labels" / "train"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    for tag, stem in selected:
        ds = datasets[tag]
        fname = stem_to_file[tag][stem]
        shutil.copy2(ds.images_dir["train"] / fname, images_out / f"{tag}_{fname}")
        _write_label(ds.labels_dir["train"] / f"{stem}.txt", labels_out / f"{tag}_{stem}.txt", ds.class_offset)


def materialize_val(datasets: dict[str, DatasetSpec], out_dir: Path) -> None:
    images_out = out_dir / "images" / "val"
    labels_out = out_dir / "labels" / "val"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    for tag, ds in datasets.items():
        images_dir, labels_dir = ds.images_dir["val"], ds.labels_dir["val"]
        images = sorted(f.name for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        for fname in images:
            shutil.copy2(images_dir / fname, images_out / f"{tag}_{fname}")
            stem = Path(fname).stem
            _write_label(labels_dir / f"{stem}.txt", labels_out / f"{tag}_{stem}.txt", ds.class_offset)
        n_lbl = sum(1 for _ in labels_out.glob(f"{tag}_*.txt"))
        print(f"--- val: {len(images)} {tag} images, {n_lbl} {tag} label files ---")


def write_yaml(k: int, cfg_dir: Path, out_dir_name: str, names: dict[int, str]) -> None:
    data = {
        "path": out_dir_name,
        "train": f"k{k}/images/train",
        "val": "val/images/val",
        "names": names,
    }
    (cfg_dir / f"finetune_k{k}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    args = parse_args()
    k_values = [int(k) for k in args.k_values.split(",")]
    cfg_dir = REPO_ROOT / args.cfg_dir
    out_root = REPO_ROOT / args.out_dir

    base = load_dataset(REPO_ROOT / args.base_yaml, args.base_tag, class_offset=0)
    novel = load_dataset(REPO_ROOT / args.novel_yaml, args.novel_tag, class_offset=len(base.names))
    if base.tag == novel.tag:
        raise ValueError(f"--base-tag and --novel-tag both resolved to {base.tag!r}; pass distinct tags")
    datasets = {base.tag: base, novel.tag: novel}
    names = {**base.names, **novel.names}
    num_classes = len(names)

    if out_root.exists():
        shutil.rmtree(out_root)

    all_counts: dict[tuple[str, str], Counter] = {}
    class_index: dict[int, list[tuple[str, str]]] = {}
    stem_to_file: dict[str, dict[str, str]] = {}
    for ds in datasets.values():
        index_labels(ds, all_counts, class_index, stem_to_file)

    materialize_val(datasets, out_root / "val")

    for k in k_values:
        selected, running = select_kshot(class_index, all_counts, k, num_classes, args.seed)
        report_counts(k, running, names, selected)
        materialize_train(selected, stem_to_file, datasets, out_root / f"k{k}")
        write_yaml(k, cfg_dir, args.out_dir, names)

    print(f"--- done: wrote cfg/finetune_k{{{','.join(str(k) for k in k_values)}}}.yaml ---")


if __name__ == "__main__":
    main()
