"""Build class-balanced k-shot fine-tune datasets from hotspot-ship + COCO.

For each k in --k-values, selects a training set where every class (COCO's 80 base
classes plus hot-spot/ship, appended at ids 80/81) has close to k boxes, using an
independent per-class greedy sampling pass per k (TFA/DeFRCN-style seed generation).
The val set (full COCO val2017 + full hotspot-ship val) is materialized once and
shared, unsampled, across all k values.
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-values", default="1,3,5,10,30")
    p.add_argument("--out-dir", default="finetune_kshot")
    p.add_argument("--coco-root", default="COCO")
    p.add_argument("--hotspot-root", default="hotspot-ship")
    p.add_argument("--cfg-dir", default="cfg")
    return p.parse_args()


def index_labels(
    images_dir: Path,
    labels_dir: Path,
    source: str,
    class_offset: int,
    all_counts: dict[tuple[str, str], Counter],
    class_index: dict[int, list[tuple[str, str]]],
    stem_to_file: dict[str, dict[str, str]],
) -> None:
    files: dict[str, str] = {}
    for p in images_dir.iterdir():
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stem in files:
            raise ValueError(f"duplicate stem {p.stem!r} in {images_dir}: {files[p.stem]} vs {p.name}")
        files[p.stem] = p.name
    stem_to_file[source] = files

    for label_path in labels_dir.glob("*.txt"):
        stem = label_path.stem
        if stem not in files:
            print(f"WARNING: skipping orphaned label {label_path} (no matching image)")
            continue
        counts: Counter = Counter()
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            cls = int(line.split()[0]) + class_offset
            counts[cls] += 1
        key = (source, stem)
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
    n_coco = sum(1 for s, _ in selected if s == "coco")
    n_hs = sum(1 for s, _ in selected if s == "hs")
    print(f"--- k={k}: {len(selected)} images selected ({n_coco} coco, {n_hs} hs) ---")
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


def materialize_train(
    selected: set[tuple[str, str]],
    stem_to_file: dict[str, dict[str, str]],
    coco_images_dir: Path,
    coco_labels_dir: Path,
    hs_images_dir: Path,
    hs_labels_dir: Path,
    out_dir: Path,
    novel_offset: int,
) -> None:
    images_out = out_dir / "images" / "train"
    labels_out = out_dir / "labels" / "train"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    for source, stem in selected:
        prefix = "coco" if source == "coco" else "hs"
        img_dir = coco_images_dir if source == "coco" else hs_images_dir
        lbl_dir = coco_labels_dir if source == "coco" else hs_labels_dir
        fname = stem_to_file[source][stem]
        (images_out / f"{prefix}_{fname}").symlink_to((img_dir / fname).resolve())

        src_label = lbl_dir / f"{stem}.txt"
        dst_label = labels_out / f"{prefix}_{stem}.txt"
        if source == "coco":
            dst_label.symlink_to(src_label.resolve())
        else:
            lines = src_label.read_text().splitlines() if src_label.exists() else []
            remapped = _remap_lines(lines, novel_offset)
            dst_label.write_text("\n".join(remapped) + ("\n" if remapped else ""))


def materialize_val(
    coco_images_dir: Path,
    coco_labels_dir: Path,
    hs_images_dir: Path,
    hs_labels_dir: Path,
    out_dir: Path,
    novel_offset: int,
) -> None:
    images_out = out_dir / "images" / "val"
    labels_out = out_dir / "labels" / "val"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_out.mkdir(parents=True)
    labels_out.mkdir(parents=True)

    coco_images = [f for f in os.listdir(coco_images_dir) if Path(f).suffix.lower() in IMAGE_EXTS]
    hs_images = [f for f in os.listdir(hs_images_dir) if Path(f).suffix.lower() in IMAGE_EXTS]

    for fname in coco_images:
        (images_out / f"coco_{fname}").symlink_to((coco_images_dir / fname).resolve())
        stem = Path(fname).stem
        lbl = coco_labels_dir / f"{stem}.txt"
        if lbl.exists():
            (labels_out / f"coco_{stem}.txt").symlink_to(lbl.resolve())

    for fname in hs_images:
        (images_out / f"hs_{fname}").symlink_to((hs_images_dir / fname).resolve())
        stem = Path(fname).stem
        lbl = hs_labels_dir / f"{stem}.txt"
        lines = lbl.read_text().splitlines() if lbl.exists() else []
        remapped = _remap_lines(lines, novel_offset)
        (labels_out / f"hs_{stem}.txt").write_text("\n".join(remapped) + ("\n" if remapped else ""))

    n_coco_lbl = sum(1 for _ in labels_out.glob("coco_*.txt"))
    n_hs_lbl = sum(1 for _ in labels_out.glob("hs_*.txt"))
    print(
        f"--- val: {len(coco_images)} coco + {len(hs_images)} hs images, "
        f"{n_coco_lbl} coco + {n_hs_lbl} hs label files ---"
    )


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

    coco_root = REPO_ROOT / args.coco_root
    hs_root = REPO_ROOT / args.hotspot_root
    cfg_dir = REPO_ROOT / args.cfg_dir
    out_root = REPO_ROOT / args.out_dir

    coco_names = yaml.safe_load((cfg_dir / "coco.yaml").read_text())["names"]
    coco_names = {int(i): n for i, n in coco_names.items()}
    novel_offset = len(coco_names)
    names = {**coco_names, novel_offset: "hot-spot", novel_offset + 1: "ship"}
    num_classes = len(names)

    if out_root.exists():
        shutil.rmtree(out_root)

    all_counts: dict[tuple[str, str], Counter] = {}
    class_index: dict[int, list[tuple[str, str]]] = {}
    stem_to_file: dict[str, dict[str, str]] = {}

    index_labels(
        coco_root / "images" / "train2017",
        coco_root / "labels" / "train2017",
        "coco",
        0,
        all_counts,
        class_index,
        stem_to_file,
    )
    index_labels(
        hs_root / "images" / "train",
        hs_root / "labels" / "train",
        "hs",
        novel_offset,
        all_counts,
        class_index,
        stem_to_file,
    )

    materialize_val(
        coco_root / "images" / "val2017",
        coco_root / "labels" / "val2017",
        hs_root / "images" / "val",
        hs_root / "labels" / "val",
        out_root / "val",
        novel_offset,
    )

    for k in k_values:
        selected, running = select_kshot(class_index, all_counts, k, num_classes, args.seed)
        report_counts(k, running, names, selected)
        materialize_train(
            selected,
            stem_to_file,
            coco_root / "images" / "train2017",
            coco_root / "labels" / "train2017",
            hs_root / "images" / "train",
            hs_root / "labels" / "train",
            out_root / f"k{k}",
            novel_offset,
        )
        write_yaml(k, cfg_dir, args.out_dir, names)

    print(f"--- done: wrote cfg/finetune_k{{{','.join(str(k) for k in k_values)}}}.yaml ---")


if __name__ == "__main__":
    main()
