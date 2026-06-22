from pathlib import Path
import sys
import shutil

from PIL import Image
from torch.utils.data import DataLoader

# Path to the cloned FiveK dataset repo
FIVEK_REPO = Path(r"E:/henrique/mit-adobe-fivek-dataset")
sys.path.append(str(FIVEK_REPO))

from dataset.fivek import MITAboveFiveK


FIVEK_ROOT = r"E:/henrique/data"
OUT_ROOT = Path(r"E:/henrique/data/hdrnet_fivek_debug")

EXPERT = "c"
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


def find_input_file(item) -> Path:
    return Path(item["files"]["dng"])


def find_expert_file(item, expert: str) -> Path:
    return Path(item["files"]["tiff16"][expert])


def save_pair(input_img: Image.Image, target_img: Image.Image, split: str, idx: int):
    # Resize input first
    input_img = resize_long_edge(input_img, IMG_SIZE_LONG_EDGE)

    # Force target to have exactly the same size as input
    target_img = target_img.resize(input_img.size, Image.Resampling.LANCZOS)

    input_dir = OUT_ROOT / split / "input"
    output_dir = OUT_ROOT / split / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = f"{idx:04d}.jpg"

    input_img.save(input_dir / name, quality=95)
    target_img.save(output_dir / name, quality=95)


def main():
    if OUT_ROOT.exists():
        print(f"Removing old folder: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    dataset = MITAboveFiveK(
        root=FIVEK_ROOT,
        split="debug",
        download=False,
        experts=[EXPERT],
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
    )

    items = list(loader)

    print(f"Loaded {len(items)} debug items.")

    # Use 7 images for train, 2 for eval
    train_items = items[:7]
    eval_items = items[7:9]

    for i, item in enumerate(train_items):
        dng_path = find_input_file(item)
        expert_path = find_expert_file(item, EXPERT)

        input_img = convert_dng_to_rgb(dng_path)
        target_img = Image.open(expert_path).convert("RGB")

        save_pair(input_img, target_img, "train", i)
        print(f"Saved train pair {i}: {dng_path.name}")

    for i, item in enumerate(eval_items):
        dng_path = find_input_file(item)
        expert_path = find_expert_file(item, EXPERT)

        input_img = convert_dng_to_rgb(dng_path)
        target_img = Image.open(expert_path).convert("RGB")

        save_pair(input_img, target_img, "eval", i)
        print(f"Saved eval pair {i}: {dng_path.name}")

    print("\nDone.")
    print(f"HDRNet debug dataset saved at: {OUT_ROOT}")


if __name__ == "__main__":
    main()