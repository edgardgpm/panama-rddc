"""
pipeline/train.py

Trains a YOLO11n model on the prepared RDD2022ES dataset.
Expects pipeline/prepare.py (step 1) to have been run first.
"""

import os
from ultralytics import YOLO

import filehandler as fh


def _ensure_pretrained():
    """Download yolo11n.pt into the project folder if not already present."""
    if not os.path.exists(fh.PRETRAINED_WEIGHTS):
        print("Downloading yolo11n.pt into project folder ...")
        from ultralytics.utils.downloads import download
        download(fh.PRETRAINED_URL, dir=fh.PROJECT_DIR)
    else:
        print(f"Pretrained weights found: {fh.PRETRAINED_WEIGHTS}")


def run():
    """Train YOLO11n on the prepared dataset."""

    if not fh.check_dataset_ready():
        raise RuntimeError(
            f"Dataset not ready — {fh.DATA_YAML} not found.\n"
            "Run step 1 (prepare) first."
        )

    _ensure_pretrained()

    print(f"Using dataset : {fh.DATA_YAML}")
    print(f"Output dir    : {fh.RUNS_DIR}\n")

    model = YOLO(fh.PRETRAINED_WEIGHTS)

    model.train(
        data=fh.DATA_YAML,
        epochs=5,
        imgsz=320,  # 320 instead of 640 — 4x fewer pixels per image, biggest speed win on CPU
        batch=16,   # larger batch amortizes per-step overhead, faster than batch=8 on CPU
        workers=0,  # 0 = load data in the main process; faster on Windows because
                    # higher values spawn separate processes with heavy startup cost.
        project=fh.RUNS_DIR,
        name="train",
    )

    print("\nTraining complete.")
    print(f"Weights saved to: {os.path.join(fh.RUNS_DIR, 'train', 'weights')}")