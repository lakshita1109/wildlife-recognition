"""
prepare_dataset.py - turn a folder of images (one sub-folder per animal) into the
train / validation / test layout used by train.py.

Example (Windows):
    python prepare_dataset.py --source C:\\Users\\you\\Downloads\\animals --output dataset

What it does:
  * Finds the class folders (steps into wrapper folders such as animals\\animals automatically).
  * Skips unreadable/corrupted images.
  * Removes exact duplicate files (same content hash) so the same picture can not end up
    in both train and test. Near-duplicates (resized/edited copies) can not be detected.
  * Splits every class 70% / 15% / 15% with a fixed random seed.
  * Renames Italian Animals-10 folder names (cane -> dog, ...) to English.
  * Copies files (your raw download is left untouched).
"""

import argparse
import hashlib
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Animals-10 (Kaggle) uses Italian folder names
ANIMALS10_TRANSLATION = {
    "cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
    "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep",
    "ragno": "spider", "scoiattolo": "squirrel",
}


def list_images(folder):
    return sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


def find_class_root(source):
    """Descend through wrapper folders until we reach the folder that holds the class folders."""
    current = source
    for _ in range(5):
        subdirs = [d for d in current.iterdir() if d.is_dir()]
        if not subdirs:
            break
        if list_images(current) or len(subdirs) > 1:
            break
        current = subdirs[0]
    return current


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:  # verify() leaves the file unusable, reopen to decode
            img.load()
        return True
    except Exception:  # noqa: BLE001
        return False


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_counts(n, train_ratio, val_ratio):
    """Return (n_train, n_val, n_test). Tiny classes keep at least one val/test image."""
    if n < 5:
        return n, 0, 0
    n_val = max(1, round(n * val_ratio))
    n_test = max(1, n - round(n * train_ratio) - n_val)
    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def prepare(source, output, train_ratio, val_ratio, seed, clean, min_images):
    source, output = Path(source), Path(output)
    if not source.is_dir():
        print(f"ERROR: source folder not found: {source}")
        return 1

    root = find_class_root(source)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and list_images(d))
    if not class_dirs:
        print(f"ERROR: no class folders with images found in {root}")
        print("Each animal needs its own folder, e.g. animals/cat/*.jpg, animals/dog/*.jpg")
        return 1

    if output.resolve() == source.resolve():
        print("ERROR: source and output must be different folders.")
        return 1

    splits = ("train", "validation", "test")
    if clean:
        for s in splits:
            if (output / s).exists():
                shutil.rmtree(output / s)
    elif any((output / s).exists() and any((output / s).iterdir()) for s in splits):
        print(f"ERROR: {output} already contains train/validation/test data.")
        print("Re-run with --clean to replace it.")
        return 1

    rng = random.Random(seed)
    seen_hashes = {}
    totals = {s: 0 for s in splits}
    skipped_bad = skipped_dupes = skipped_small = 0
    used_classes = []

    print(f"Reading classes from: {root}\n")
    for class_dir in class_dirs:
        name = class_dir.name.strip().lower().replace(" ", "_")
        name = ANIMALS10_TRANSLATION.get(name, name)

        unique = []
        for img_path in list_images(class_dir):
            if not is_valid_image(img_path):
                skipped_bad += 1
                continue
            digest = file_hash(img_path)
            if digest in seen_hashes:
                skipped_dupes += 1
                continue
            seen_hashes[digest] = img_path
            unique.append(img_path)

        if len(unique) < min_images:
            print(f"  skipping '{name}': only {len(unique)} usable images (min {min_images})")
            skipped_small += 1
            continue

        rng.shuffle(unique)
        n_train, n_val, _ = split_counts(len(unique), train_ratio, val_ratio)
        parts = {
            "train": unique[:n_train],
            "validation": unique[n_train:n_train + n_val],
            "test": unique[n_train + n_val:],
        }
        for split, files in parts.items():
            if not files:
                continue
            dest = output / split / name
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(files):
                # prefix with index so files from different folders never collide
                shutil.copy2(src, dest / f"{name}_{i:05d}{src.suffix.lower()}")
            totals[split] += len(files)
        used_classes.append(name)
        print(f"  {name:<20} train {len(parts['train']):>4}  val {len(parts['validation']):>4}  "
              f"test {len(parts['test']):>4}")

    print("\nDone.")
    print(f"  Classes: {len(used_classes)}")
    print(f"  Images:  train {totals['train']}, validation {totals['validation']}, "
          f"test {totals['test']}")
    print(f"  Skipped: {skipped_bad} unreadable, {skipped_dupes} exact duplicates, "
          f"{skipped_small} classes with too few images")
    print(f"  Output:  {output.resolve()}")
    print("\nNext step:  python train.py")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Split an image-folder dataset for training.")
    parser.add_argument("--source", required=True,
                        help="folder that contains one sub-folder per animal")
    parser.add_argument("--output", default="dataset", help="output folder (default: dataset)")
    parser.add_argument("--train", type=float, default=0.70, help="train ratio (default 0.70)")
    parser.add_argument("--val", type=float, default=0.15, help="validation ratio (default 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-images", type=int, default=10,
                        help="skip classes with fewer usable images than this")
    parser.add_argument("--clean", action="store_true",
                        help="delete existing train/validation/test folders in the output first")
    args = parser.parse_args()

    if not (0 < args.train < 1 and 0 < args.val < 1 and args.train + args.val < 1):
        print("ERROR: --train and --val must be between 0 and 1 and add up to less than 1.")
        sys.exit(1)
    sys.exit(prepare(args.source, args.output, args.train, args.val, args.seed,
                     args.clean, args.min_images))


if __name__ == "__main__":
    main()
