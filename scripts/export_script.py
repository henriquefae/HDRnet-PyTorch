"""
export_ready_fivek_to_hdrnet.py

Goal:
- Copy already-prepared FiveK input/output images into HDRNet-PyTorch folder format.

Input:
    E:/fivekdataset/input/total_srgb

Output / target:
    E:/fivekdataset/Expertc_resize/total

Exported HDRNet format:
    E:/henrique/data/hdrnet_fivek/
    ├── train/input
    ├── train/output
    ├── eval/input
    ├── eval/output
    ├── test/input
    └── test/output

No image modification is performed.
"""

from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import tifffile


# ============================================================
# Paths
# ============================================================

INPUT_ROOT = Path(r"E:/fivekdataset/input/total_srgb")
OUTPUT_ROOT = Path(r"E:/fivekdataset/Expertc_resize/total")

OUT_ROOT = Path(r"E:/henrique/data/hdrnet_fivek")


# ============================================================
# Dataset size control
# Use None to export all remaining images.
# ============================================================

TRAIN = 4500
VAL = 100
TEST = 400


# ============================================================
# Image conversion
# ============================================================

SAVE_AS_UINT8 = False

def convert_to_uint8_rgb(src_path: Path):
    """
    Convert image to true uint8 RGB.

    Cases:
    - uint8  -> keep as [0, 255]
    - uint16 -> divide by 65535 and convert to [0, 255]
    - float [0, 1] -> multiply by 255
    - float [0, 255] -> clip to [0, 255]
    """
    arr = tifffile.imread(str(src_path))

    original_dtype = arr.dtype
    arr = arr.astype(np.float32)

    if original_dtype == np.uint8:
        print(f"Image {src_path} is already uint8. No conversion needed.")
        pass

    elif original_dtype == np.uint16:
        print(f"Image {src_path} is uint16. Converting to uint8 by dividing by 65535 and multiplying by 255.")
        arr = arr / 65535.0 * 255.0

    else:
        raise ValueError(f"Unsupported image dtype: {original_dtype}. Only uint8 and uint16 are supported.")

    arr = arr.clip(0, 255).astype(np.uint8)

    return arr


def save_as_uint8_image(src_path: Path, dst_path: Path):
    """
    Save source image as true 8-bit RGB image.

    This makes the exported dataset compatible with repositories that do:
        image.astype(np.float32) / 255.0
    """
    arr_uint8 = convert_to_uint8_rgb(src_path)
    img = Image.fromarray(arr_uint8, mode="RGB")
    img.save(dst_path)


# ============================================================
# Dataset utilities
# ============================================================

def list_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = [
        p for p in folder.iterdir() if p.is_file()
    ]

    return sorted(images)


def build_pairs(input_images, output_images):
    """
    Match input/output images by filename stem.
    """
    output_by_stem = {p.stem: p for p in output_images}

    pairs = []

    for input_path in input_images:
        stem = input_path.stem

        if stem not in output_by_stem:
            print(f"Warning: no output found for input {input_path.name}")
            continue

        pairs.append((input_path, output_by_stem[stem]))

    return pairs


def split_pairs(pairs):
    if TRAIN is None or VAL is None or TEST is None:
        raise ValueError("Please set TRAIN, VAL, and TEST as integers.")

    total_needed = TRAIN + VAL + TEST

    if len(pairs) < total_needed:
        raise ValueError(
            f"Not enough pairs. Requested {total_needed}, but found {len(pairs)}."
        )

    train_pairs = pairs[:TRAIN]
    val_pairs = pairs[TRAIN:TRAIN + VAL]
    test_pairs = pairs[TRAIN + VAL:TRAIN + VAL + TEST]

    return train_pairs, val_pairs, test_pairs


def copy_pairs(pairs, split_name: str):
    input_dir = OUT_ROOT / split_name / "input"
    output_dir = OUT_ROOT / split_name / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Exporting {split_name}: {len(pairs)} image pairs")
    print("=" * 80)

    for idx, (input_path, output_path) in enumerate(pairs):
        # Keep original extension, but rename consistently.
        input_name = f"{idx:04d}{input_path.suffix.lower()}"
        output_name = f"{idx:04d}{output_path.suffix.lower()}"

        dst_input = input_dir / input_name
        dst_output = output_dir / output_name

        if SAVE_AS_UINT8:
            save_as_uint8_image(input_path, dst_input)
            save_as_uint8_image(output_path, dst_output)
        else:
            shutil.copy2(input_path, dst_input)
            shutil.copy2(output_path, dst_output)
            
        print(f"Saved {split_name} pair {idx:04d}:")
        print(f"  input : {input_path.name} -> {dst_input.name}")
        print(f"  output: {output_path.name} -> {dst_output.name}")


# ============================================================
# Main
# ============================================================

def main():
    input_images = list_images(INPUT_ROOT)
    output_images = list_images(OUTPUT_ROOT)

    print(f"Found input images: {len(input_images)}")
    print(f"Found output images: {len(output_images)}")

    pairs = build_pairs(input_images, output_images)

    print(f"Found valid input/output pairs: {len(pairs)}")

    train_pairs, val_pairs, test_pairs = split_pairs(pairs)

    if OUT_ROOT.exists():
        print(f"Removing old folder: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs, "eval")
    copy_pairs(test_pairs, "test")

    print("\nDone.")
    print(f"HDRNet dataset saved at: {OUT_ROOT}")


if __name__ == "__main__":
    main()