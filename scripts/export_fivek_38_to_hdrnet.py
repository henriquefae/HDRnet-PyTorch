"""
export_fivek_38_to_hdrnet.py

Goal:
- Read:
    E:/henrique/data/MITAboveFiveK/small_30_5_3_manifest.json
- Convert DNG input images to RGB JPG
- Convert Expert C TIFF images to RGB JPG
- Export in HDRNet-PyTorch folder format:

    E:/henrique/data/hdrnet_fivek_38/
    ├── train/input
    ├── train/output
    ├── eval/input
    ├── eval/output
    ├── test/input
    └── test/output

This is based on your working debug exporter, but uses the manifest instead of
MITAboveFiveK(split="train"), because the full train split is not downloaded.
"""

from pathlib import Path
import json
import shutil
from PIL import Image

MANIFEST_PATH = Path(r"E:/henrique/data/MITAboveFiveK/small_30_5_3_manifest.json")
OUT_ROOT = Path(r"E:/henrique/data/hdrnet_fivek_38")

IMG_SIZE_LONG_EDGE = 512


def resize_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    w, h = img.size
    scale = long_edge / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def convert_dng_to_rgb(dng_path: Path) -> Image.Image:
    import rawpy

    with rawpy.imread(str(dng_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=8,
            no_auto_bright=False,
        )

    return Image.fromarray(rgb).convert("RGB")


def save_pair(input_img: Image.Image, target_img: Image.Image, split: str, idx: int):
    # Resize input first
    input_img = resize_long_edge(input_img, IMG_SIZE_LONG_EDGE)

    # Force target to exactly the same size as input.
    # This avoids: assert input.shape == output.shape
    target_img = target_img.resize(input_img.size, Image.Resampling.LANCZOS)

    input_dir = OUT_ROOT / split / "input"
    output_dir = OUT_ROOT / split / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = f"{idx:04d}.jpg"

    input_img.save(input_dir / name, quality=95)
    target_img.save(output_dir / name, quality=95)


def export_items(items, target_split: str):
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
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}\n"
            "Run download_fivek_38.py first."
        )

    if OUT_ROOT.exists():
        print(f"Removing old folder: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    export_items(manifest["train"], "train")
    export_items(manifest["val"], "eval")
    export_items(manifest["test"], "test")

    print("\nDone.")
    print(f"HDRNet dataset saved at: {OUT_ROOT}")


if __name__ == "__main__":
    main()
