"""Extract the novel-class images/labels from a finetune dataset built by `build_finetune_kshot.py`.

Reads a `finetune_k{k}.yaml` and copies every train and val image that has at least one
novel-class box (class id >= the base dataset's class count) into a new dataset directory,
keeping only the novel-class boxes. Two versions are written, each with its own dir + yaml:

- `<out-dir>`: novel classes only -- ids shifted down by the base class count so they start at
  0, and `names` lists only the novel classes.
- `<out-dir>_full`: same images, but original finetune ids and the full base + novel `names`,
  so it's directly usable with a model fine-tuned on the original yaml.

With --novel-yaml, also writes a `test` split: every image in the novel dataset's original val
set that wasn't sampled into the finetune val set (needs --val-size <N> at build time, otherwise
nothing is left over).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from build_finetune_kshot import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--finetune-yaml", required=True, help="finetune_k{k}.yaml written by build_finetune_kshot.py")
    p.add_argument("--base-yaml", required=True, help="base dataset yaml used to build it (gives the base class count)")
    p.add_argument("--novel-yaml", default=None, help="novel dataset yaml used to build it; adds a test split of its unused val images")
    p.add_argument("--novel-tag", default=None, help="--novel-tag passed to build_finetune_kshot.py (default: --novel-yaml's stem)")
    p.add_argument("--out-dir", default=None, help="output directory of the novel-only version; the full-class one gets a '_full' suffix (default: '<finetune dir>/<yaml stem>_novel')")
    return p.parse_args()


def labels_dir_for(images_dir: Path) -> Path:
    # build_finetune_kshot.py always writes '<split dir>/images/<split>' next to '<split dir>/labels/<split>'
    return images_dir.parent.parent / "labels" / images_dir.name


def extract_split(
    images_dir: Path,
    labels_dir: Path,
    src_offset: int,
    num_base: int,
    shift: int,
    out_dir: Path,
    split: str,
    exclude: set[str] = frozenset(),
    keep_empty: bool = False,
) -> None:
    """Copy images with novel boxes and their novel-only labels. A source id maps to the finetune
    id `id + src_offset` (novel iff >= num_base), which is written as `finetune id + shift`."""
    images_out = out_dir / "images" / split
    labels_out = out_dir / "labels" / split
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    n_images = n_boxes = 0
    for img in sorted(images_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS or img.name in exclude:
            continue
        label = labels_dir / f"{img.stem}.txt"
        lines = label.read_text().splitlines() if label.exists() else []
        novel = []
        for line in lines:
            parts = line.split()
            if parts and int(parts[0]) + src_offset >= num_base:
                parts[0] = str(int(parts[0]) + src_offset + shift)
                novel.append(" ".join(parts))
        if not novel and not keep_empty:
            continue
        shutil.copy2(img, images_out / img.name)
        (labels_out / label.name).write_text("\n".join(novel) + ("\n" if novel else ""))
        n_images += 1
        n_boxes += len(novel)
    print(f"--- {out_dir.name} {split}: {n_images} images, {n_boxes} novel boxes ---")


def write_yaml(out_dir: Path, names: dict[int, str], with_test: bool) -> None:
    out_yaml = out_dir.with_suffix(".yaml")
    path = out_dir.relative_to(REPO_ROOT) if out_dir.is_relative_to(REPO_ROOT) else out_dir
    data = {"path": str(path), "train": "images/train", "val": "images/val"}
    if with_test:
        data["test"] = "images/test"
    out_yaml.write_text(yaml.safe_dump({**data, "names": names}, sort_keys=False))
    print(f"--- wrote {out_yaml.relative_to(REPO_ROOT) if out_yaml.is_relative_to(REPO_ROOT) else out_yaml} ---")


def main() -> None:
    args = parse_args()
    finetune_yaml = REPO_ROOT / args.finetune_yaml
    data = yaml.safe_load(finetune_yaml.read_text())
    num_base = len(yaml.safe_load((REPO_ROOT / args.base_yaml).read_text())["names"])
    full_names = {int(k): v for k, v in data["names"].items()}
    novel_names = {k - num_base: v for k, v in full_names.items() if k >= num_base}

    root = Path(data["path"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    novel = load_dataset(REPO_ROOT / args.novel_yaml, args.novel_tag, class_offset=0) if args.novel_yaml else None

    out_dir = REPO_ROOT / (args.out_dir or finetune_yaml.parent / f"{finetune_yaml.stem}_novel")
    versions = [(out_dir, -num_base, novel_names), (out_dir.with_name(f"{out_dir.name}_full"), 0, full_names)]
    for version_dir, shift, names in versions:
        if version_dir.exists():
            shutil.rmtree(version_dir)
        for split in ("train", "val"):
            images_dir = root / data[split]
            extract_split(images_dir, labels_dir_for(images_dir), 0, num_base, shift, version_dir, split)
        if novel:
            # finetune val images are named '<novel tag>_<original name>'
            prefix = f"{novel.tag}_"
            used = {p.name[len(prefix) :] for p in (root / data["val"]).iterdir() if p.name.startswith(prefix)}
            extract_split(
                novel.images_dir["val"], novel.labels_dir["val"], num_base, num_base, shift,
                version_dir, "test", exclude=used, keep_empty=True,
            )
        write_yaml(version_dir, names, with_test=novel is not None)


if __name__ == "__main__":
    main()
