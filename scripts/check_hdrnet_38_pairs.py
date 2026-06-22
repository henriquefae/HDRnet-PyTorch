"""
check_hdrnet_38_pairs.py

Checks that every HDRNet input/output pair exists and has the same size.
"""

from pathlib import Path
from PIL import Image

ROOT = Path(r"E:/henrique/data/hdrnet_fivek_38")

EXPECTED_COUNTS = {
    "train": 30,
    "eval": 5,
    "test": 3,
}


def check_split(split: str, expected_count: int):
    print(f"\nChecking {split}...")

    input_dir = ROOT / split / "input"
    output_dir = ROOT / split / "output"

    input_paths = sorted(input_dir.glob("*.jpg"))
    output_paths = sorted(output_dir.glob("*.jpg"))

    print(f"input count : {len(input_paths)}")
    print(f"output count: {len(output_paths)}")

    assert len(input_paths) == expected_count, (
        f"{split}/input has {len(input_paths)} images, expected {expected_count}"
    )
    assert len(output_paths) == expected_count, (
        f"{split}/output has {len(output_paths)} images, expected {expected_count}"
    )

    for input_path in input_paths:
        output_path = output_dir / input_path.name

        assert output_path.exists(), f"Missing output file: {output_path}"

        input_img = Image.open(input_path)
        output_img = Image.open(output_path)

        print(input_path.name, input_img.size, output_img.size)

        assert input_img.size == output_img.size, f"Size mismatch: {input_path.name}"


def main():
    for split, expected_count in EXPECTED_COUNTS.items():
        check_split(split, expected_count)

    print("\nAll pairs exist and have matching sizes.")


if __name__ == "__main__":
    main()
