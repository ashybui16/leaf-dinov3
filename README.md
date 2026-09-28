# LEAF-DINOv3

DINOv3 &rarr; LEAF-YOLO / YOLOv8 knowledge distillation, ported from *"Improving YOLOv8 for
Fast Few-Shot Object Detection by DINOv2 Distillation"* (Fourret et al.) with two
substitutions: **DINOv3** instead of DINOv2, and the **`my-ultralytics` fork** (carrying the
LEAF-YOLO modules) instead of upstream Ultralytics.

This is **knowledge distillation, not feature fusion**. A frozen DINOv3 teacher supplies
regression targets for extra output channels appended to the detector's head, during training
only. Those channels are stripped before inference (`strip_checkpoint.py`), so the deployed
model is bit-identical to stock LEAF-YOLO/YOLOv8 at the same FPS and parameter count. The
teacher never runs at inference, and there is no projection into the backbone, no feature
concat, no cross-attention.

Deviation from the paper: pretraining runs on the **whole dataset** (all 80 COCO or all 10
VisDrone classes), not a 60-class base split. The two-stage structure (pretrain &rarr;
finetune) is kept so a future custom dataset with novel classes can be added on top via
`expand_num_classes`.

## Variants

| | |
|---|---|
| **d2** | Cls-branch embeddings only. Widens `cv3`'s last conv by `embed_dim`; the extra channels regress the teacher's patch tokens per level. |
| **d3** | d2, plus box-branch embeddings (`cv2` widened the same way, with its own resample path) and a self-attention-map term (`SelfAttentionMap`) matching the teacher's own (RoPE-aware) last-block attention. |

### Resize target: student or teacher (`resize_student`)

Student and teacher patch grids rarely match resolution (e.g. a student level at stride 8 vs.
a stride-16 teacher patch size), so one side has to be resized before the embedding/attention
losses can compare them. `resize_student` (constructor kwarg on `DistillDetectionModel`,
`--resize-student`/`--no-resize-student` on `train_pretrain.py`) picks which side:

- **`resize_student=False` (default).** Student embeddings stay at each level's own native FPN
  stride; the teacher target is resized down/up to match, per level, entirely inside
  `losses.py` (`embedding_distill_loss` derives its target size from the student tensor's own
  shape, `attention_distill_loss`/`resize_attention_map` do the same for the attention affinity
  matrix). Nothing learned sits between the student features and the loss.
- **`resize_student=True`.** Each level gets its own learned resampler (`_make_resampler` in
  `heads.py`) mapping it onto the teacher's patch grid before the loss ever sees it: a chain of
  `Conv2d(dim, dim, kernel_size=3, stride=2)` for finer levels, a `DySample` (learned
  point-sampling upsampler) for coarser ones. The teacher target is never passed through
  anything learned in this mode either — only the student side gains parameters.

Either way the teacher side may still get resized (by `losses.py`, unconditionally, to match
whatever shape the student embed already has) — `resize_student` only controls whether the
*student* side also carries a learned resampler. For stage-2 finetuning, `resize_student` is
read back off the stage-1 checkpoint (see below), never re-specified.

Distillation only ever trains through the **one2many** branch (see `leaf_dinov3/heads.py`
docstrings / the design notes below for why) — `yolov8m` has no one2one branch and is
unaffected; `yolo26*-leaf` keeps a stock-width, stock-behavior one2one branch automatically.

## Design constraints (see `CLAUDE.md`)

- `dinov3/` and `my-ultralytics/` are treated as **protected, vendored dependencies**: nothing
  inside either is created, edited, renamed, moved, or deleted by this project's code. (Editing
  either yourself is of course fine — this is a design choice for `leaf_dinov3/`'s own code,
  not a restriction on the repo. `CLAUDE.md` encodes it as an instruction for AI coding agents
  working in this repo.) Everything here hot-swaps the detection head at runtime
  (`DistillDetectionModel.__init__` in `leaf_dinov3/model.py`) instead of registering new
  modules in `parse_model`, specifically so it never needs to touch either vendored tree.

## Layout

```
leaf_dinov3/
  teacher.py   DinoV3Teacher — frozen backbone, patch tokens (+ RoPE-aware attention for d3)
  heads.py     DistillDetectD2 / DistillDetectD3, SelfAttentionMap, _extend_last_conv
  losses.py    normalize_channels, embedding_distill_loss, resize_attention_map, attention_distill_loss
  model.py     DistillDetectionModel — hot-swaps the head, adds the distill loss term
  trainer.py   DistillTrainer, DistillFineTuneTrainer
  data.py      resolve_dataset_yaml, check_class_order (pretrain/finetune class-order contract)
VisDrone/visdrone.yaml, COCO/coco.yaml, hotspot-ship/hotspot_ship.yaml
  repo-relative dataset YAMLs, each next to its data -- gitignored, not in a fresh clone
  (see "Dataset setup" below)
scripts/
  train_pretrain.py   train_finetune.py   strip_checkpoint.py   evaluate.py
tests/
  test_teacher.py  test_heads.py  test_losses.py  test_model_smoke.py  test_data.py
```

## Environment setup

`.gitignore` deliberately excludes `dinov3/`, `my-ultralytics/`, and the dataset directories,
so a fresh clone of this repo needs each of them set up separately, in this order, before
anything here will run. None of this assumes you already have a Python environment for this
project — the steps below create one from scratch.

**1. Create and activate a Python 3.10+ environment.** Conda example (any env manager and any
env name work identically — `mu` below is just this repo's own convention, referenced in
`CLAUDE.md` and reused in every command in this README):

```bash
conda create -n mu python=3.10 -y
conda activate mu
```

**2. Install torch matching your hardware, before anything else.** `requirements.txt` pins a
minimum version but not a specific build, so installing it first would grab whatever wheel
`pip` defaults to for your platform (often a CPU-only one on Linux). Pick your platform's
command from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), e.g.
for CUDA 11.8 (what this project was developed against):

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

CPU-only machines (or anything where the default PyPI wheel is already right for you) can skip
this and let step 4 install torch normally.

**3. Clone the two dependencies this repo builds on top of, into the repo root**, then install
the second one editable so `import ultralytics` resolves to it:

```bash
git clone https://github.com/facebookresearch/dinov3.git
git clone <your-leaf-yolo-capable-ultralytics-fork-url> my-ultralytics
pip install -e my-ultralytics
```

`my-ultralytics/` needs to be a fork carrying the LEAF-YOLO modules (`GhostConv`, `LEAF`,
`MGC`, `SPPRFEM`, `CoordBlock`, `LEAFT`, `ELAN`, etc.) — ask whoever maintains this project for
that fork's URL. If you don't have access to it: plain upstream `ultralytics` (`pip install
ultralytics`) still works for every script here as long as you pass `--model` a stock YOLO
YAML (e.g. `yolov8m.yaml`) instead of a `yolo26*-leaf.yaml` one, which needs the fork's modules
to even parse.

Then download DINOv3's pretrained backbone weights into `dinov3/weights/`, named exactly as
released (`teacher.py`'s `_DEFAULT_WEIGHTS` looks them up by variant name):
- `dinov3_vits16_pretrain_lvd1689m-08c60483.pth`
- `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` (optional, ViT-L/16)

Pass `--teacher-weights <path>` (scripts) or `weights=<path>` (`DinoV3Teacher`) to use a
different location or a variant outside this list.

**4. Install this repo's own dependencies:**

```bash
pip install -r requirements.txt
```

**5. Get your dataset(s) in place** — see "Dataset setup" below.

**6. Verify the setup:**

```bash
python -m pytest tests/ -v
```
(Or `conda run -n mu python -m pytest tests/ -v` from outside the activated env, as every
command below does — pick whichever fits your workflow.)

### Dataset setup

`VisDrone/visdrone.yaml` and `COCO/coco.yaml` use a **repo-root-relative** `path:` (e.g. `path:
VisDrone`), resolved to an absolute path by `leaf_dinov3.data.resolve_dataset_yaml` (called by
every script in `scripts/`) before ultralytics ever sees it. This sidesteps two portability
problems at once: a *relative* `path:` handed straight to ultralytics resolves against
`~/.config/Ultralytics/settings.json`'s `datasets_dir`, which may point at an unrelated project
on your machine; and a hardcoded *absolute* `path:` in the YAML would only work on
the machine it was written on, breaking for every other clone of this repo.

The dataset folders, YAMLs included, are gitignored, so a fresh clone has none of them:
provide your own YAMLs in this shape. By default this expects your data at:
- `<repo_root>/VisDrone/{train,val,test}/images` + matching `labels/` (YOLO-txt format)
- `<repo_root>/COCO/images/{train2017,val2017}` + matching `labels/` (YOLO-txt format)

If your data lives elsewhere, edit `path:` in `VisDrone/visdrone.yaml` / `COCO/coco.yaml` to another
repo-relative path, or to an absolute path (left untouched by the resolver either way). For a
dataset that isn't COCO or VisDrone, write your own dataset YAML following the same shape
(`path`, `train`, `val`, `names`) — see the [Ultralytics dataset YAML
docs](https://docs.ultralytics.com/datasets/detect/) for the general format.

If you've regenerated or edited labels in place, delete the matching `*/labels.cache` first —
cache validity is keyed on label path + total size, not content or mtime.

For stage-2 fine-tuning with novel classes, put a YOLO-txt `train`/`val` split at
`<repo_root>/hotspot-ship/images/{train,val}` + matching `labels/`, then run
`conda run -n mu python scripts/build_finetune_kshot.py --base-yaml COCO/coco.yaml
--novel-yaml hotspot-ship/hotspot_ship.yaml` to generate `coco_hotspot_ship_kshot/`
(gitignored, copied from `COCO/` + `hotspot-ship/` — self-contained, not symlinked, so it's
portable to e.g. a Docker volume mount) with its `finetune_k{1,3,5,10,30}.yaml` files inside.
These give class-balanced k-shot training sets (COCO's 80 base classes + `hot-spot`/`ship` all
sampled to ~k boxes each) with a fixed, unsampled val set (full COCO val2017 + full hotspot-ship val) — re-run the script after any change to `hotspot-ship/` or
`COCO/` to regenerate.

`build_finetune_kshot.py` isn't hardcoded to COCO + hotspot-ship: `--base-yaml`/`--novel-yaml`
(both required) each take any dataset yaml of this repo's
standard shape (`path`, `train`, `val`, `names`) — the novel dataset's classes are appended
after the base dataset's, matching `check_class_order`'s tail-append contract.

## Usage

**Stage 1 — pretrain** (whole dataset, distillation on):

```bash
conda run -n mu python scripts/train_pretrain.py \
  --data COCO/coco.yaml --model yolo26s-leaf.yaml --variant d2 \
  --teacher-variant dinov3_vitb16 --imgsz 640 --batch 64 --epochs 300
```

Key flags: `--variant {d2,d3}`, `--teacher-variant`, `--teacher-weights` (local `.pth`,
defaults to the matching file in `dinov3/weights/`), `--teacher-input-size` (defaults to
`imgsz`, so the teacher grid tracks the student's), `--resize-student`/`--no-resize-student`
(default off — student embeddings stay at native stride and the teacher target is resized to
match in `losses.py`; pass `--resize-student` to instead learn a per-level resampler that maps
the student side onto the teacher's patch grid, see "Resize target" above), `--attn-num-heads`
(d3 only — the box-branch attention runs on each level's own embeddings, whichever grid they end
up on depending on `--resize-student`, matching the teacher's patch-only attention shape resized
to fit), `--lambda-distill`, `--fraction`
(dataset subset; `>1` = image count, or `[train, val]` list), `--offline` (sets `YOLO_OFFLINE=1`, must precede any network dependent import),
and `--smoke` — a fast, low-VRAM profile (`yolo26n-leaf.yaml`, `dinov3_vits16`, imgsz 320,
batch 2, epochs 1, `fraction=[8, 8]`) for quick local/CI-style sanity checks on any machine,
including CPU-only, vs. the full-scale defaults (`yolo26s-leaf.yaml`, `dinov3_vitb16`). All
ultralytics train flags (`--epochs`, `--batch`, `--lr0`, `--optimizer`, ...) default to
ultralytics' own `default.yaml` values unless passed.

**Stage 2 — finetune** (expand to a dataset with novel classes appended at the tail):

```bash
conda run -n mu python scripts/train_finetune.py \
  --weights runs/detect/train/weights/best.pt \
  --data coco_hotspot_ship_kshot/finetune_k10.yaml --pretrain-data COCO/coco.yaml \
  --lr0 1e-3 --epochs 100
```

No teacher/variant flags: `DistillFineTuneTrainer` reads `variant`, `teacher_variant`,
`teacher_input_size`, `attn_num_heads`, and `resize_student` straight off the unpickled stage-1
checkpoint, so the rebuilt scaffold can never end up architecturally mismatched with the
checkpoint it's about to load. `--data`'s first N class names must match `--pretrain-data`
exactly, in order (`check_class_order`, run before training starts) — novel classes go
strictly at the tail, since `expand_num_classes` preserves existing class weight rows and
moves the distill tail to the new end position by index.

**Export** (strip the teacher and the distill channels for deployment):

```bash
conda run -n mu python scripts/strip_checkpoint.py --weights best.pt --out stripped.pt
```

Idempotent — running it again on an already-stripped checkpoint is a no-op, not an error.
Removes the frozen teacher (~327 MB of ViT-B rides along in every checkpoint otherwise) and
shrinks the head back to stock shapes; parameter count afterwards matches a stock model built
from the same YAML exactly.

**Evaluate:**

```bash
conda run -n mu python scripts/evaluate.py --weights best.pt --data VisDrone/visdrone.yaml \
  --split val --base-classes 10   # optional: report base vs. novel mAP separately
```

Wraps ultralytics' native `DetectionValidator` (correct letterboxed decode, per-class AP)
rather than hand-rolling a pycocotools loop; works on both stripped and unstripped
checkpoints (auto-strips if a `teacher` is present).

## Testing

```bash
conda run -n mu python -m pytest tests/ -v
```

45 tests: teacher shape/freeze/RoPE-attention-correctness checks, head forward-shape /
`bias_init` / `strip_distillation` / `expand_num_classes` unit tests (3-level and 4-level,
`end2end` on and off, d2 and d3, and both `resize_student` settings), loss-function unit tests
(zero-loss, gradient-flow, attention-resize row-sum preservation), dataset-YAML path resolution /
class-order checks, and a synthetic overfit smoke test across all eight
`{yolo26n-leaf, yolov8m} x {d2, d3} x {resize_student on, off}` combinations.

## Known gotchas (found the hard way — see git history / commit messages)

- **`dinov3/hubconf.py` is not imported.** `torch.hub.load(..., source="local")` execs the
  whole file, which eagerly imports segmentor/detector/dinotxt entrypoints we never use and
  pulls in `torchmetrics` (not installed in `mu`). `teacher.py` imports
  `dinov3.hub.backbones` directly instead — the backbone path only needs `torch` and `numpy`.
- **Attention extraction wraps `compute_attention`, not a qkv hook.** RoPE is applied *inside*
  `SelfAttention.compute_attention` in DINOv3; a qkv-forward-hook would silently compute wrong
  attention (passes every shape test, fails a numerical comparison against the unwrapped
  block). The replacement is a module-level callable class, not a closure — closures aren't
  picklable, and ultralytics periodically pickles the whole model (teacher included) at every
  checkpoint save, so a closure-based version trained fine for one epoch and then crashed
  right before saving `last.pt`.
- **`lambda_distill` needs to be tuned up from 1.0 for a real training signal.**
  `embedding_distill_loss` L2-normalizes both sides before `smooth_l1_loss`, so its natural
  scale is roughly `1/embed_dim` regardless of resolution or batch size — tiny next to
  `box_loss`/`cls_loss` (O(1-5)). This doesn't affect correctness (verified the loss converges
  cleanly to ~0 given a properly scaled gradient), but a smoke run at the default lambda won't
  show a visible downward trend the way the other loss terms might. Scale `lambda_distill` up
  (order-of-magnitude starting point: `~embed_dim`) for it to meaningfully influence training.
- **`expand_num_classes` must bias-init new class rows**, not leave them at ~0. Left
  uninitialized, a class that never appears in any label predicts ~50% at every anchor instead
  of ~1%, inflating `cls_loss` by ~100x once summed over anchors and divided by the small
  foreground-normalized `target_scores_sum`. Fixed in `heads.py`; regression-tested.

## Status

d2 and d3 are both fully implemented and verified locally (unit tests + real 1-epoch VisDrone
runs, both variants, both `{yolo26n-leaf, yolov8m}`, pretrain and finetune, strip round-trip).
Full-scale pretraining (118k COCO images vs. 6.5k VisDrone) needs a large-memory GPU and is the
only remaining step, starting with `--variant d2` as the baseline before spending a run on d3.
