"""
check_hdrnet_pairs.py

Checks that every HDRNet input/output pair exists and has the same image shape
as seen by HDRNet-PyTorch during training.

Important:
- For .dng input files, do NOT use PIL and do NOT use raw.sizes.iwidth/iheight.
- Use rawpy.postprocess(...) with the same parameters as datasets.py.
- This checks the effective array shape used during training.
"""

from pathlib import Path
from PIL import Image
import rawpy
import numpy as np


ROOT = Path(r"E:/henrique/data/hdrnet_fivek")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".dng"}


def list_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    return sorted(
        [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def get_hdrnet_shape(path: Path):
    """
    Return image shape as (H, W, C), matching what HDRNet-PyTorch sees.

    For DNG:
        use rawpy.postprocess, because this is what --hdr training uses.

    For normal images:
        use PIL and force RGB, because output should be RGB too.
    """
    suffix = path.suffix.lower()

    if suffix == ".dng":
        with rawpy.imread(str(path)) as raw:
            img = raw.postprocess(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
            )

        return img.shape  # (H, W, C)

    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img)

    return arr.shape  # (H, W, C)


def check_split(split: str):
    print(f"\nChecking {split}...")

    input_dir = ROOT / split / "input"
    output_dir = ROOT / split / "output"

    input_paths = list_images(input_dir)
    output_paths = list_images(output_dir)

    print(f"input count : {len(input_paths)}")
    print(f"output count: {len(output_paths)}")

    assert len(input_paths) == len(output_paths), (
        f"{split}/input has {len(input_paths)} images, "
        f"while {split}/output has {len(output_paths)} images"
    )

    output_by_stem = {p.stem: p for p in output_paths}

    for input_path in input_paths:
        stem = input_path.stem

        assert stem in output_by_stem, (
            f"Missing output pair for input file: {input_path.name}"
        )

        output_path = output_by_stem[stem]

        input_shape = get_hdrnet_shape(input_path)
        output_shape = get_hdrnet_shape(output_path)

        print(
            f"{input_path.name} <-> {output_path.name}: "
            f"{input_shape} / {output_shape}"
        )

        assert input_shape == output_shape, (
            f"Shape mismatch for pair {input_path.name} / {output_path.name}: "
            f"{input_shape} vs {output_shape}"
        )


def main():
    for split in ["train", "eval", "test"]:
        check_split(split)

    print("\nAll pairs exist and have matching HDRNet shapes.")


if __name__ == "__main__":
    main()