"""
pipeline/predict.py

Runs inference with the trained YOLO11n model.
Expects pipeline/train.py (step 2) to have been run first.

When called via main.py the source defaults to dataset_yaml/images/val/.
Pass --source to override from the command line.
"""

import os
from ultralytics import YOLO

import filehandler as fh

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def run(
    source=None,
    weights=None,
    conf=0.25,
    iou=0.45,
    imgsz=640,
    save_txt=False,
    save_conf=False,
):
    """
    Run YOLO inference.

    Parameters
    ----------
    source    : path to image, video, or folder. Defaults to val image dir.
    weights   : path to .pt file. Defaults to best.pt from the last training run.
    conf      : confidence threshold.
    iou       : IoU threshold for NMS.
    imgsz     : inference image size.
    save_txt  : save predictions as .txt label files.
    save_conf : include confidence scores in .txt files.
    """

    weights = weights or fh.TRAINED_WEIGHTS
    source  = source  or fh.VAL_IMAGE_DIR

    # Guard: trained weights must exist
    if not fh.check_weights_ready() and weights == fh.TRAINED_WEIGHTS:
        raise RuntimeError(
            f"Trained weights not found: {fh.TRAINED_WEIGHTS}\n"
            "Run step 2 (train) first."
        )

    # Guard: source path must exist (webcam '0' is exempt)
    if not os.path.exists(str(source)) and str(source) != "0":
        raise FileNotFoundError(
            f"Source not found: {source}\n"
            "Pass a valid image path, folder, video, or '0' for webcam."
        )

    # Guard: if source is a folder it must contain at least one image
    if os.path.isdir(str(source)):
        available = [
            f for f in os.listdir(str(source))
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]
        if not available:
            raise RuntimeError(
                f"Source folder contains no recognised images: {source}\n"
                "Run step 1 (prepare) first, or pass --source with a non-empty folder."
            )
        print(f"Found {len(available)} image(s) in source folder.")

    print(f"Weights : {weights}")
    print(f"Source  : {source}")
    print(f"Conf    : {conf}   IoU: {iou}   imgsz: {imgsz}\n")

    model   = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=True,
        save_txt=save_txt,
        save_conf=save_conf,
        project=fh.RUNS_DIR,
        name="predict",
    )

    total = sum(len(r.boxes) for r in results)
    print(f"\nProcessed {len(results)} image(s) — {total} total detection(s).")
    print(f"Results saved to: {fh.PREDICT_RUN_DIR}")