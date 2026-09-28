# LEAF-DINOv3 -- mirrors README.md's "Environment setup".
#
# Base image already carries torch==2.4.1 built against CUDA 11.8, matching the exact build
# README.md recommends -- no separate torch install needed.
#
# Only code and DINOv3's pretrained weights (dinov3/weights/*.pth) are baked into the image;
# datasets (COCO/, VisDrone/, hotspot-ship/, *_kshot/) and my-ultralytics/weights/
# (YOLO weights, not DINO's) are excluded -- see .dockerignore. Mount datasets at run time, e.g.:
#   docker run --gpus all -v $(pwd)/COCO:/workspace/COCO -it leaf-dinov3
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# HOME=/tmp: writable HOME for any `--user` UID (torch hub / ultralytics / matplotlib caches go under $HOME)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

# libgl1/libglib2.0-0: required by opencv-python (pulled in via my-ultralytics' dependencies)
# git/vim: dev-container conveniences
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# This repo's own dependencies (torch already satisfied by the base image)
COPY requirements.txt .
RUN pip install -r requirements.txt

# my-ultralytics fork (LEAF-YOLO modules), installed editable so `import ultralytics` resolves
# to it -- must be copied in before this, since pip needs the source tree present to install it.
COPY my-ultralytics/ my-ultralytics/
RUN pip install -e my-ultralytics

CMD ["bash"]
