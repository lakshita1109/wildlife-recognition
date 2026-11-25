"""
prepare_oxford_pets.py - turn the Oxford-IIIT Pet dataset into the train/validation/test
folders used by train.py.

Dataset: 37 cat and dog breeds, ~7,400 images, https://www.robots.ox.ac.uk/~vgg/data/pets/

Option A - automatic download (about 800 MB, needs internet):
    python prepare_oxford_pets.py --download --raw D:\\pets_raw --output dataset --clean

Option B - manual download: get images.tar.gz from the page above, extract it so that a folder
contains the .jpg files (e.g. D:\\pets_raw\\images\\Abyssinian_1.jpg), then:
    python prepare_oxford_pets.py --source D:\\pets_raw --output dataset --clean

Modes:
    --mode breed    37 classes (default), e.g. pug, siamese, beagle
    --mode species  2 classes: cat, dog

Also adds basic entries for the new classes to data/animal_info.json so the info card works.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from prepare_dataset import file_hash, is_valid_image, split_counts

INFO_PATH = Path(__file__).resolve().parent / "data" / "animal_info.json"


def download(raw_dir):
    try:
        from torchvision.datasets import OxfordIIITPet
        OxfordIIITPet(root=str(raw_dir), split="trainval", download=True)
        OxfordIIITPet(root=str(raw_dir), split="test", download=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        print("Download images.tar.gz manually from https://www.robots.ox.ac.uk/~vgg/data/pets/,")
        print("extract it, and use --source instead of --download.")
        sys.exit(1)


def label_for(path, mode):
    """Oxford file names look like 'Egyptian_Mau_145.jpg'. Cats start with a capital letter."""
    breed = path.stem.rsplit("_", 1)[0]
    if mode == "species":
        return "cat" if breed[0].isupper() else "dog"
    return breed.lower()


def add_info(labels, mode):
    """Add simple info entries for new labels without touching existing ones."""
    try:
        info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        info = {}
    added = 0
    for name, is_cat in labels.items():
        if name in info:
            continue
        pet = "cat" if is_cat else "dog"
        pretty = name.replace("_", " ").title()
        info[name] = {
            "category": "Mammal (domestic " + pet + ")",
            "habitat": "Homes and domestic environments",
            "diet": "Carnivore" if is_cat else "Omnivore (mostly meat)",
            "facts": [f"The {pretty} is a recognised {pet} breed in the Oxford-IIIT Pet dataset."],
        }
        added += 1
    if added:
        INFO_PATH.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return added


def main():
    parser = argparse.ArgumentParser(description="Prepare the Oxford-IIIT Pet dataset.")
    parser.add_argument("--source", help="folder containing the extracted Oxford pet .jpg images")
    parser.add_argument("--download", action="store_true", help="download via torchvision")
    parser.add_argument("--raw", default="dataset/raw", help="download location (with --download)")
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--mode", choices=["breed", "species"], default="breed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.download:
        download(Path(args.raw))
        source = Path(args.raw)
    elif args.source:
        source = Path(args.source)
    else:
        print("Use --download or --source <folder>. See the top of this file for examples.")
        sys.exit(1)

    images = sorted(p for p in source.rglob("*.jpg") if "_" in p.stem)
    if not images:
        print(f"ERROR: no Oxford pet .jpg images found under {source}")
        sys.exit(1)

    output = Path(args.output)
    splits = ("train", "validation", "test")
    if args.clean:
        for s in splits:
            shutil.rmtree(output / s, ignore_errors=True)
    elif any((output / s).exists() and any((output / s).iterdir()) for s in splits):
        print(f"ERROR: {output} already has data. Re-run with --clean to replace it.")
        sys.exit(1)

    groups, seen, bad, dupes = {}, set(), 0, 0
    for img in images:
        if not is_valid_image(img):
            bad += 1
            continue
        digest = file_hash(img)
        if digest in seen:
            dupes += 1
            continue
        seen.add(digest)
        groups.setdefault(label_for(img, args.mode), []).append(img)

    rng = random.Random(args.seed)
    totals = {s: 0 for s in splits}
    for name, files in sorted(groups.items()):
        rng.shuffle(files)
        n_train, n_val, _ = split_counts(len(files), 0.70, 0.15)
        parts = {"train": files[:n_train], "validation": files[n_train:n_train + n_val],
                 "test": files[n_train + n_val:]}
        for split, items in parts.items():
            dest = output / split / name
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(items):
                shutil.copy2(src, dest / f"{name}_{i:05d}.jpg")
            totals[split] += len(items)

    cat_flags = {}
    for name, files in groups.items():
        cat_flags[name] = (name == "cat") if args.mode == "species" else files[0].stem[0].isupper()
    added = add_info(cat_flags, args.mode)

    print(f"Classes: {len(groups)}  ({args.mode} mode)")
    print(f"Images:  train {totals['train']}, validation {totals['validation']}, test {totals['test']}")
    print(f"Skipped: {bad} unreadable, {dupes} exact duplicates")
    print(f"Added {added} entries to data/animal_info.json")
    print("\nNext step:  python train.py")


if __name__ == "__main__":
    main()