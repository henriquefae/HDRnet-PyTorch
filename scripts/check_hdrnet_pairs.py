from pathlib import Path
from PIL import Image

ROOT = Path(r"E:/henrique/data/hdrnet_fivek_debug")

for split in ["train", "eval"]:
    print(f"\nChecking {split}...")

    input_dir = ROOT / split / "input"
    output_dir = ROOT / split / "output"

    for input_path in sorted(input_dir.glob("*.jpg")):
        output_path = output_dir / input_path.name

        input_img = Image.open(input_path)
        output_img = Image.open(output_path)

        print(input_path.name, input_img.size, output_img.size)

        assert input_img.size == output_img.size, f"Mismatch: {input_path.name}"

print("\nAll pairs have matching sizes.")