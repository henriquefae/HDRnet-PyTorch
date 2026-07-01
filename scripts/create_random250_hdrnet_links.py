from pathlib import Path
import os
import shutil
import random


# -----------------------------
# Paths
# -----------------------------
INPUT_ROOT = Path(r"E:\fivekdataset\input\total_srgb")
OUTPUT_ROOT = Path(r"E:\fivekdataset\Expertc_resize\total")
DEST_ROOT = Path(r"E:\henrique\data\hdrnet_random250_fivek_dataset")

EXTENSION = ".tif"
EVAL_COUNT = 250
RANDOM_SEED = 42  # change if you want a different random split


# -----------------------------
# Destination folders
# -----------------------------
TRAIN_INPUT = DEST_ROOT / "train" / "input"
TRAIN_OUTPUT = DEST_ROOT / "train" / "output"
EVAL_INPUT = DEST_ROOT / "eval" / "input"
EVAL_OUTPUT = DEST_ROOT / "eval" / "output"


def make_dirs():
    for folder in [TRAIN_INPUT, TRAIN_OUTPUT, EVAL_INPUT, EVAL_OUTPUT]:
        folder.mkdir(parents=True, exist_ok=True)


def create_hardlink_or_copy(src: Path, dst: Path):
    """
    Create a hard link from src to dst.
    If hard linking fails, fall back to copying.
    """
    if dst.exists():
        return

    try:
        os.link(src, dst)
    except OSError as e:
        print(f"Warning: hard link failed for {dst.name}. Falling back to copy.")
        print(f"Reason: {e}")
        shutil.copy2(src, dst)


def link_split(files, input_dst: Path, output_dst: Path, start_index: int = 1):
    """
    Create renamed links for paired input/output files.
    Assumes input and output original filenames are identical.
    """
    for i, input_file in enumerate(files, start=start_index):
        original_name = input_file.name
        output_file = OUTPUT_ROOT / original_name

        if not output_file.exists():
            raise FileNotFoundError(
                f"Missing output image for input file:\n"
                f"Input:  {input_file}\n"
                f"Output: {output_file}"
            )

        new_name = f"{i:04d}{EXTENSION}"

        input_link = input_dst / new_name
        output_link = output_dst / new_name

        create_hardlink_or_copy(input_file, input_link)
        create_hardlink_or_copy(output_file, output_link)


def main():
    make_dirs()

    input_files = sorted(INPUT_ROOT.glob(f"*{EXTENSION}"))

    if len(input_files) == 0:
        raise RuntimeError(f"No {EXTENSION} files found in {INPUT_ROOT}")

    if len(input_files) < EVAL_COUNT:
        raise RuntimeError(
            f"Not enough images to create eval split. "
            f"Found {len(input_files)}, but EVAL_COUNT={EVAL_COUNT}."
        )

    # Reproducible random split
    random.seed(RANDOM_SEED)
    eval_files = set(random.sample(input_files, EVAL_COUNT))
    train_files = [f for f in input_files if f not in eval_files]

    # Sort both splits so numbering is deterministic
    train_files = sorted(train_files, key=lambda x: x.name)
    eval_files = sorted(eval_files, key=lambda x: x.name)

    print(f"Found total input images: {len(input_files)}")
    print(f"Train images: {len(train_files)}")
    print(f"Eval images:  {len(eval_files)}")
    print(f"Random seed:  {RANDOM_SEED}")

    # Train: 0001.tif, 0002.tif, ...
    link_split(train_files, TRAIN_INPUT, TRAIN_OUTPUT, start_index=1)

    # Eval: 0001.tif, 0002.tif, ...
    link_split(eval_files, EVAL_INPUT, EVAL_OUTPUT, start_index=1)

    print("\nDone.")
    print(f"Dataset created at: {DEST_ROOT}")


if __name__ == "__main__":
    main()