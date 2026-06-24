"""
export_fivek_to_hdrnet_raw_resize_output.py

Goal:
- Export FiveK image pairs in HDRNet-PyTorch folder format.
- Input remains original .dng.
- Output remains .tif.
- No preprocessing on input.
- Output is resized only if its size does not match the input DNG size.
"""

from pathlib import Path
import json
import shutil

import rawpy
import numpy as np
from PIL import Image


# ============================================================
# Paths
# ============================================================

MANIFEST_PATH = Path(r"E:/henrique/data/MITAboveFiveK/data_split_manifest.json")
OUT_ROOT = Path(r"E:/henrique/data/hdrnet_fivek")


# ============================================================
# Dataset size control
# Use None to export all available images from a split.
# ============================================================

TRAIN = 15
VAL = 5
TEST = 3


# ============================================================
# Export options
# ============================================================

OVERWRITE_OUTPUT_FOLDER = True
RENAME_WITH_INDEX = True


# ============================================================
# Helpers
# ============================================================

def limit_items(items, n_images):
    if n_images is None:
        return items
    return items[:n_images]


def get_hdrnet_postprocessed_dng_size(dng_path: Path):
    """
    Return the exact image size that HDRNet-PyTorch will obtain
    when loading the DNG with --hdr.

    Returns:
        (width, height)
    """
    with rawpy.imread(str(dng_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=False,
            no_auto_bright=True,
            output_bps=16,
        )

    h, w = rgb.shape[:2]
    return w, h


def get_output_name(item, idx, input_path, output_path):
    if RENAME_WITH_INDEX:
        stem = f"{idx:04d}"
    else:
        stem = item.get("basename", input_path.stem)

    input_name = stem + input_path.suffix.lower()

    # Keep TIFF output
    output_suffix = output_path.suffix.lower()
    if output_suffix not in [".tif", ".tiff"]:
        output_suffix = ".tif"

    output_name = stem + output_suffix

    return input_name, output_name


def prepare_output_tiff(tiff_path: Path, dst_output: Path, target_size):
    """
    Load target TIFF, convert to RGB, resize if needed, and save as TIFF.

    target_size is (width, height), matching PIL convention.
    """
    with Image.open(tiff_path) as img:
        img = img.convert("RGB")

        if img.size != target_size:
            print(f"  resizing output from {img.size} to {target_size}")
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        else:
            print(f"  output already has correct size: {target_size}")

        img.save(dst_output)


def copy_pair(item, target_split: str, idx: int):
    dng_path = Path(item["dng"])
    tiff_path = Path(item["tiff16_c"])

    if not dng_path.exists():
        raise FileNotFoundError(f"Missing DNG file: {dng_path}")

    if not tiff_path.exists():
        raise FileNotFoundError(f"Missing Expert C TIFF file: {tiff_path}")

    input_dir = OUT_ROOT / target_split / "input"
    output_dir = OUT_ROOT / target_split / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_name, output_name = get_output_name(item, idx, dng_path, tiff_path)

    dst_input = input_dir / input_name
    dst_output = output_dir / output_name

    # Copy DNG unchanged
    shutil.copy2(dng_path, dst_input)

    # Resize TIFF to the exact size produced by HDRNet's rawpy.postprocess
    target_size = get_hdrnet_postprocessed_dng_size(dng_path)

    prepare_output_tiff(
        tiff_path=tiff_path,
        dst_output=dst_output,
        target_size=target_size,
    )

    print(f"Saved {target_split} pair {idx:04d}: {item.get('basename', dng_path.stem)}")
    print(f"  input : {dst_input}")
    print(f"  output: {dst_output}")
    print(f"  HDRNet input size: {target_size}")


def export_items(items, target_split: str, n_images):
    items = limit_items(items, n_images)

    print("\n" + "=" * 80)
    print(f"Exporting {target_split}: {len(items)} image pairs")
    print("=" * 80)

    for idx, item in enumerate(items):
        copy_pair(item, target_split, idx)


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if OUT_ROOT.exists() and OVERWRITE_OUTPUT_FOLDER:
        print(f"Removing old folder: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    export_items(manifest["train"], "train", TRAIN)
    export_items(manifest["val"], "eval", VAL)
    export_items(manifest["test"], "test", TEST)

    print("\nDone.")
    print(f"HDRNet dataset saved at: {OUT_ROOT}")


if __name__ == "__main__":
    main()