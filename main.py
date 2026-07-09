"""
main.py
 
Orchestrates the full road-defect detection pipeline.
 
Usage
-----
  Run all steps:
    python main.py
 
  Run specific steps (space-separated):
    python main.py --steps 1          # download + prepare dataset
    python main.py --steps 2          # train model
    python main.py --steps 3          # predict on val set
    python main.py --steps 4          # convert HEIC images to JPG
    python main.py --steps 1 2        # prepare then train
    python main.py --steps 4 3        # convert HEIC then predict on them
 
  Prediction overrides (only used when step 3 is included):
    python main.py --steps 3 --source path/to/image.jpg
    python main.py --steps 3 --weights path/to/custom.pt
    python main.py --steps 3 --conf 0.4 --iou 0.5
    python main.py --steps 3 --save-txt --save-conf
 
  HEIC conversion overrides (only used when step 4 is included):
    python main.py --steps 4 --heic-source path/to/heic/folder
    python main.py --steps 4 --heic-source path/to/folder --heic-dest path/to/output/
 
Steps
-----
  1 — Download RDD2022ES from KaggleHub and format into dataset_yaml/
  2 — Train YOLO11n on the prepared dataset (5 epochs, CPU-friendly)
  3 — Run inference on dataset_yaml/images/val/ (or --source override)
  4 — Convert HEIC/HEIF images to .jpg (output: dataset_yaml/images/test/)
"""
 
import argparse
import sys
 
import filehandler as fh
from pipeline import prepare, train, predict, convert
 
 
STEP_LABELS = {
    1: "Prepare dataset",
    2: "Train model",
    3: "Predict",
    4: "Convert HEIC to JPG",
}
 
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="RDD2022ES road-defect detection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2, 4, 3],
        metavar="N",
        help="Step(s) to run: 1=prepare, 2=train, 3=predict, 4=convert HEIC (default: 1 2 3)",
    )
 
    # Predict overrides
    predict_group = parser.add_argument_group("predict options (step 3)")
    predict_group.add_argument("--source",    default=None,  help="Image/video/folder for inference")
    predict_group.add_argument("--weights",   default=None,  help="Path to .pt weights file")
    predict_group.add_argument("--conf",      type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    predict_group.add_argument("--iou",       type=float, default=0.45, help="IoU NMS threshold (default: 0.45)")
    predict_group.add_argument("--imgsz",     type=int,   default=640,  help="Inference image size (default: 640)")
    predict_group.add_argument("--save-txt",  action="store_true", help="Save prediction labels as .txt")
    predict_group.add_argument("--save-conf", action="store_true", help="Include confidence in .txt labels")
 
    # HEIC conversion overrides
    heic_group = parser.add_argument_group("convert options (step 4)")
    heic_group.add_argument("--heic-source", default=None, help="Folder containing .heic/.heif images")
    heic_group.add_argument("--heic-dest",   default=None, help="Output folder for .jpg files (default: dataset_yaml/images/test/)")
 
    return parser.parse_args()
 
 
def run_step(step, args):
    print(f"\n{'='*50}")
    print(f"  Step {step}: {STEP_LABELS[step]}")
    print(f"{'='*50}\n")
 
    if step == 1:
        if fh.check_dataset_ready():
            print(f"Dataset already prepared ({fh.DATA_YAML} exists) — skipping.")
            print("Delete dataset_yaml/ and re-run if you want a fresh download.")
            return
        prepare.run()
    elif step == 2:
        train.run()
    elif step == 3:
        predict.run(
            source=args.source,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
        )
    elif step == 4:
        convert.run(
            source=args.heic_source,
            dest=args.heic_dest,
        )
 
 
def main():
    args  = parse_args()
    steps = sorted(set(args.steps), key=lambda s: [1, 2, 4, 3].index(s) if s in [1, 2, 4, 3] else s)  # deduplicate and enforce order
 
    print("\nRDD2022ES Road Defect Detection Pipeline")
    print(f"Running steps: {steps}")
    fh.summary()
    fh.setup_folders()
 
    for step in steps:
        try:
            run_step(step, args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"\n[ERROR] Step {step} failed: {exc}", file=sys.stderr)
            sys.exit(1)
 
    print("\nAll requested steps completed successfully.")
    fh.summary()
 
 
if __name__ == "__main__":
    main()