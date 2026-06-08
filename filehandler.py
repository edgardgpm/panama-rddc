"""
filehandler.py

Single source of truth for every path in the project.
All other modules import from here — nothing hardcodes its own paths.

Project layout managed by this module:

    AI_RDDPanama/
        yolo11n.pt              ← pretrained weights (downloaded on first train)
        dataset_yaml/
            images/
                train/
                val/
            labels/
                train/
                val/
            data.yaml
        runs/
            detect/
                train/
                    weights/
                        best.pt
                        last.pt
                predict/
        pipeline/
            __init__.py
            prepare.py
            train.py
            predict.py
        filehandler.py
        main.py
"""

import os

# -----------------------------------------
# Root — everything hangs off this
# -----------------------------------------

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------
# Dataset
# -----------------------------------------

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_yaml")
IMAGE_DIR   = os.path.join(DATASET_DIR, "images")
LABEL_DIR   = os.path.join(DATASET_DIR, "labels")
DATA_YAML   = os.path.join(DATASET_DIR, "data.yaml")

TRAIN_IMAGE_DIR = os.path.join(IMAGE_DIR, "train")
VAL_IMAGE_DIR   = os.path.join(IMAGE_DIR, "val")
TRAIN_LABEL_DIR = os.path.join(LABEL_DIR, "train")
VAL_LABEL_DIR   = os.path.join(LABEL_DIR, "val")

# -----------------------------------------
# Model weights
# -----------------------------------------

PRETRAINED_WEIGHTS = os.path.join(PROJECT_DIR, "yolo11n.pt")
PRETRAINED_URL     = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

RUNS_DIR        = os.path.join(PROJECT_DIR, "runs", "detect")
TRAIN_RUN_DIR   = os.path.join(RUNS_DIR, "train")
TRAINED_WEIGHTS = os.path.join(TRAIN_RUN_DIR, "weights", "best.pt")
PREDICT_RUN_DIR = os.path.join(RUNS_DIR, "predict")

# -----------------------------------------
# Dataset split ratio
# -----------------------------------------

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# -----------------------------------------
# Setup — call once at startup to create
# every folder the pipeline will write into
# -----------------------------------------

def setup_folders():
    """Create the full project folder tree (no-op if already exists)."""
    folders = [
        TRAIN_IMAGE_DIR,
        VAL_IMAGE_DIR,
        TRAIN_LABEL_DIR,
        VAL_LABEL_DIR,
        RUNS_DIR,
        TRAIN_RUN_DIR,
        PREDICT_RUN_DIR,
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def check_dataset_ready():
    """Return True if the dataset has been prepared (data.yaml exists)."""
    return os.path.exists(DATA_YAML)

def check_weights_ready():
    """Return True if trained weights exist."""
    return os.path.exists(TRAINED_WEIGHTS)

def summary():
    """Print a quick status overview of the project state."""
    print("\n--- Project status ---")
    print(f"  Project dir      : {PROJECT_DIR}")
    print(f"  Pretrained model : {'OK' if os.path.exists(PRETRAINED_WEIGHTS) else 'not downloaded'}")
    print(f"  Dataset ready    : {'OK' if check_dataset_ready() else 'run step 1'}")
    print(f"  Trained weights  : {'OK' if check_weights_ready() else 'run step 2'}")
    print("----------------------\n")
