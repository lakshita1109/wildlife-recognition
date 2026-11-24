# Dataset folder

Training images go here. They are not included in the project (too large) and are ignored by Git.

## Required layout

```
dataset/
├── train/
│   ├── cat/        (many .jpg / .png images of cats)
│   ├── dog/
│   └── ...         (one folder per animal)
├── validation/
│   ├── cat/
│   └── ...         (same class folders as train)
└── test/           (optional, same class folders)
    ├── cat/
    └── ...
```

Class names are read from the folder names in `train/`, so every animal needs its own folder, and
`train/` and `validation/` must contain exactly the same folders.

## Easiest way to get this layout

1. Download a dataset from Kaggle (free account needed):
   * **Animal Image Dataset (90 Different Animals)** – recommended: 90 classes, ~60 images each.
     https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals
   * **Animals-10** – 10 classes, ~26,000 images (cat, dog, horse, cow, sheep, elephant, butterfly,
     chicken, spider, squirrel). https://www.kaggle.com/datasets/alessiocorrado99/animals10
2. Extract the zip into `dataset/raw/` (so you have e.g. `dataset/raw/animals/animals/antelope/...`).
3. From the project folder run:

```
python prepare_dataset.py --source dataset\raw --output dataset --clean
```

This copies the images into `train/`, `validation/` and `test/` (70% / 15% / 15%), skips unreadable
images and removes exact duplicate files so the same picture is not in two splits.

Optional automatic download (needs a Kaggle API token): `python download_dataset.py --dataset animals90`

## Notes

* Animals-90 does **not** contain giraffe, camel, peacock, buffalo, frog, cheetah or monkey.
  Add your own folders (at least ~50 images each) to `dataset/raw/` before running
  `prepare_dataset.py` if you need them.
* More images per class and cleaner photos give better accuracy.
* Check the dataset's licence on Kaggle if you plan to publish anything.
