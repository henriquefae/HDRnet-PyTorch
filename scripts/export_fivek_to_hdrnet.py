"""
export_fivek_to_hdrnet.py

Goal:
- Read:
    E:/henrique/data/MITAboveFiveK/data_split_manifest.json

- Export FiveK image pairs in HDRNet-PyTorch folder format:

    E:/henrique/data/hdrnet_fivek/
    ├── train/input
    ├── train/output
    ├── eval/input
    ├── eval/output
    ├── test/input
    └── test/output

Main controls:
1. Choose how many images to export with TRAIN, VAL, TEST.
2. Choose whether to apply quality changes.
3. Always resize target image to match input image size.
"""

from pathlib import Path
import json
import shutil
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

TRAIN = 2
VAL = 1
TEST = 1


# ============================================================
# Quality / preprocessing control
# ============================================================

APPLY_QUALITY_CHANGES = False

# Used only if APPLY_QUALITY_CHANGES = True
IMG_SIZE_LONG_EDGE = 512

# "jpg" is smaller but lossy.
# "png" is larger but lossless.
SAVE_FORMAT = "jpg"

# Used only if SAVE_FORMAT = "jpg"
JPEG_QUALITY = 95


# ============================================================
# Image processing functions
# ============================================================

def resize_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def convert_dng_to_rgb(dng_path: Path) -> Image.Image:
    """
    DNG must be converted to RGB before HDRNet can use it.

    Observation:
    This conversion already changes the original RAW data,
    because we are creating a displayable RGB image.
    """
    import rawpy

    with rawpy.imread(str(dng_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=8,
            no_auto_bright=False,
        )

    return Image.fromarray(rgb).convert("RGB")


def apply_quality_changes(input_img: Image.Image) -> Image.Image:
    """
    Optional quality / size change applied only to the input image.

    The target image will later be resized to match the input size,
    independently of this setting.
    """
    if APPLY_QUALITY_CHANGES:
        input_img = resize_long_edge(input_img, IMG_SIZE_LONG_EDGE)

    return input_img


def get_extension() -> str:
    if SAVE_FORMAT.lower() == "jpg" or SAVE_FORMAT.lower() == "jpeg":
        return "jpg"
    elif SAVE_FORMAT.lower() == "png":
        return "png"
    else:
        raise ValueError(f"Unsupported SAVE_FORMAT: {SAVE_FORMAT}")


def save_image(img: Image.Image, path: Path):
    if SAVE_FORMAT.lower() in ["jpg", "jpeg"]:
        img.save(path, quality=JPEG_QUALITY)
    elif SAVE_FORMAT.lower() == "png":
        img.save(path)
    else:
        raise ValueError(f"Unsupported SAVE_FORMAT: {SAVE_FORMAT}")


def save_pair(input_img: Image.Image, target_img: Image.Image, split: str, idx: int):
    # Optional quality changes on the input
    input_img = apply_quality_changes(input_img)

    # Always resize target to match input size.
    # This avoids HDRNet error:
    # assert input.shape == output.shape
    target_img = target_img.resize(input_img.size, Image.Resampling.LANCZOS)

    input_dir = OUT_ROOT / split / "input"
    output_dir = OUT_ROOT / split / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = get_extension()
    name = f"{idx:04d}.{ext}"

    save_image(input_img, input_dir / name)
    save_image(target_img, output_dir / name)


# ============================================================
# Export logic
# ============================================================

def limit_items(items, n_images):
    """
    If n_images is None, return all items.
    Otherwise, return only the first n_images.
    """
    if n_images is None:
        return items

    return items[:n_images]


def export_items(items, target_split: str, n_images):
    items = limit_items(items, n_images)

    print("\n" + "=" * 80)
    print(f"Exporting {target_split}: {len(items)} image pairs")
    print("=" * 80)

    for idx, item in enumerate(items):
        dng_path = Path(item["dng"])
        tiff_path = Path(item["tiff16_c"])

        if not dng_path.exists():
            raise FileNotFoundError(f"Missing DNG file: {dng_path}")

        if not tiff_path.exists():
            raise FileNotFoundError(f"Missing Expert C TIFF file: {tiff_path}")

        input_img = convert_dng_to_rgb(dng_path)
        target_img = Image.open(tiff_path).convert("RGB")

        save_pair(input_img, target_img, target_split, idx)

        print(f"Saved {target_split} pair {idx:04d}: {item['basename']}")


def main():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if OUT_ROOT.exists():
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