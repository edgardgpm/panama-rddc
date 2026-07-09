"""
pipeline/convert.py
 
Converts HEIC/HEIF images to .jpg for use with YOLO.
 
Workflow:
  1. Drop your HEIC files into dataset_yaml/images/convert/
  2. Run step 4 — converted JPGs appear in dataset_yaml/images/test/
  3. Run step 3 — predict on dataset_yaml/images/test/
 
Usage via main.py:
    python main.py --steps 4                         # uses convert/ → test/ defaults
    python main.py --steps 4 3                       # convert then predict
    python main.py --steps 4 --heic-source path/to/heic/folder
    python main.py --steps 4 --heic-source path/to/folder --heic-dest path/to/output/
 
Requirements:
    pip install pillow pillow-heif
"""
 
import os
from PIL import Image
import pillow_heif
 
# Register HEIF opener with Pillow so Image.open() handles .heic files
pillow_heif.register_heif_opener()
 
HEIC_EXTENSIONS = {".heic", ".heif"}
 
 
def run(source=None, dest=None):
    """
    Convert all HEIC/HEIF images in source to .jpg in dest.
 
    Parameters
    ----------
    source : folder containing .heic/.heif files
             Default: dataset_yaml/images/convert/  ← drop your files here
    dest   : folder to write .jpg files into
             Default: dataset_yaml/images/test/     ← picked up automatically by step 3
    """
 
    import filehandler as fh
 
    source = source or fh.CONVERT_IMAGE_DIR
    dest   = dest   or fh.TEST_IMAGE_DIR
 
    print(f"HEIC source : {source}")
    print(f"JPG dest    : {dest}")
 
    if not os.path.isdir(source):
        raise FileNotFoundError(
            f"Source folder not found: {source}\n"
            f"Drop your HEIC images into that folder and re-run step 4."
        )
 
    os.makedirs(dest, exist_ok=True)
 
    heic_files = [
        f for f in os.listdir(source)
        if os.path.splitext(f)[1].lower() in HEIC_EXTENSIONS
    ]
 
    if not heic_files:
        raise RuntimeError(
            f"No .heic or .heif files found in: {source}\n"
            f"Drop your HEIC images into that folder and re-run step 4."
        )
 
    print(f"Found {len(heic_files)} HEIC image(s) — converting ...\n")
 
    converted = 0
    failed    = []
 
    for filename in heic_files:
        src_path = os.path.join(source, filename)
        dst_name = os.path.splitext(filename)[0] + ".jpg"
        dst_path = os.path.join(dest, dst_name)
 
        try:
            img = Image.open(src_path)
            img.convert("RGB").save(dst_path, "JPEG", quality=95)
            converted += 1
            print(f"  OK  {filename} -> {dst_name}")
        except Exception as e:
            failed.append(filename)
            print(f"  FAIL  {filename}: {e}")
 
    print(f"\nConverted : {converted}/{len(heic_files)}")
    if failed:
        print(f"Failed    : {len(failed)}")
        for f in failed:
            print(f"  - {f}")
 
    print(f"\nJPG images ready in: {dest}")
    print("Run step 3 to predict on them.")