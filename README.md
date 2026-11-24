# Wildlife AI – Animal Image Classification & Recognition System

A web application where you upload an animal photo and a deep-learning model (MobileNetV3, PyTorch)
identifies the animal, shows the top 3 predictions with confidence bars, and displays an information card
(category, habitat, diet, facts).

It runs on a normal Windows laptop **without a GPU** and has two modes:

| Mode | When it is used | What it can recognise |
|------|-----------------|-----------------------|
| **Pretrained** | Default. No training needed. | 43 animal groups taken from the 1,000 ImageNet classes |
| **Custom trained** | Automatically, once you run `train.py` | Whatever classes are in *your* dataset (e.g. 90 with Animals-90) |

---

## 1. Features

* Real CNN inference (no hardcoded or random predictions)
* Drag-and-drop upload, preview, loading animation, confidence bars
* Top-3 predictions and an animal information card from `data/animal_info.json`
* "Unable to confidently identify this animal" message for low-confidence results
* Validation of file type, file size, corrupted and tiny/huge images
* Full training pipeline: augmentation, transfer learning, fine-tuning, graphs, test evaluation
* Dataset preparation script with train/validation/test split and duplicate removal
* Model loaded once at startup; uploads are processed in memory (nothing is stored)

## 2. AI model details

* **Model:** MobileNetV3-Large (torchvision), small and fast on CPU (~5 million parameters).
* **Framework:** PyTorch (used everywhere in the project).
* **Preprocessing:** resize (short side 232) → centre crop 224×224 → normalise with ImageNet mean/std.
  The same preprocessing is used for training validation, testing and the web app.
* **Pretrained mode:** the ImageNet model predicts 1,000 classes. Related classes are merged into animal
  labels by adding their probabilities (e.g. all dog breeds → *Dog*, tabby/tiger cat/Siamese → *Cat*).
  Non-animal photos usually spread their probability over non-animal classes, so the animal score stays low
  and the app says it is unsure.
* **Custom mode:** `train.py` replaces the last layer with one sized for your classes.
  Phase 1 trains only that layer; Phase 2 fine-tunes the last 4 blocks with a low learning rate.
  The best epoch by validation accuracy is saved.
* **Confidence thresholds:** 35% (pretrained) / 50% (custom), set in `model/model_config.py`.

## 3. Dataset details

Recommended: **Animal Image Dataset (90 Different Animals)** on Kaggle – 90 classes, about 60 images each.
Alternative: **Animals-10** – 10 classes, ~26,000 images.
See `dataset/README.md`. Kaggle needs a free account, so downloading is manual (or use the optional helper).

## 4. Supported animals

**Pretrained mode (43):** bear, bee, buffalo, butterfly, camel, cat, cheetah, chicken, cow, crocodile, dog,
duck, eagle, elephant, fish, flamingo, fox, frog, gorilla, hippopotamus, horse, hyena, leopard, lion, lizard,
monkey, ostrich, owl, panda, parrot, peacock, penguin, pig, rabbit, sheep, snake, spider, swan, tiger, toucan,
turtle, wolf, zebra.

**Not recognised in pretrained mode** (ImageNet has no such class): giraffe, rhinoceros, kangaroo, deer, goat,
donkey, crow, pigeon, sparrow. Train a custom model to get several of these.

**Custom mode with Animals-90** (class list as published for that dataset; the exact classes are whatever
folders you download, and the About section always shows the real list): antelope, badger, bat, bear, bee, beetle, bison, boar, butterfly, cat,
caterpillar, chimpanzee, cockroach, cow, coyote, crab, crow, deer, dog, dolphin, donkey, dragonfly, duck,
eagle, elephant, flamingo, fly, fox, goat, goldfish, goose, gorilla, grasshopper, hamster, hare, hedgehog,
hippopotamus, hornbill, horse, hummingbird, hyena, jellyfish, kangaroo, koala, ladybug, leopard, lion, lizard,
lobster, mosquito, moth, mouse, octopus, okapi, orangutan, otter, owl, ox, oyster, panda, parrot, pelican-type
birds, penguin, pig, pigeon, porcupine, possum, raccoon, rat, reindeer, rhinoceros, sandpiper, seahorse, seal,
shark, sheep, snake, sparrow, squid, squirrel, starfish, swan, tiger, turkey, turtle, whale, wombat, woodpecker,
wolf, zebra.

**Not in Animals-90 either:** giraffe, camel, peacock, buffalo, frog, cheetah, monkey. Add your own image
folders for these (50+ images each) to get them.

The app's About section lists exactly what the *currently loaded* model supports.

## 5. Project structure

```
Wildlife-AI/
├── app.py                 Flask backend (serves the site, POST /predict)
├── train.py               Transfer-learning training script
├── predict.py             Classify one image from the command line
├── prepare_dataset.py     Split a raw dataset into train / validation / test
├── download_dataset.py    Optional Kaggle download helper
├── requirements.txt
├── .gitignore
├── README.md
├── model/
│   ├── __init__.py
│   ├── model_config.py    Paths, thresholds, ImageNet -> animal mapping
│   └── model_utils.py     Model building, preprocessing, AnimalClassifier
├── dataset/               Training images (README.md explains the layout)
├── trained_models/        Output of train.py (model, class names)
├── templates/index.html   Website
├── static/css/style.css
├── static/js/script.js
├── static/images/
├── data/animal_info.json  Animal facts
├── uploads/               (unused placeholder – uploads are not saved)
└── outputs/               Training graphs and history
```

## 6. Prerequisites

* Windows 10/11
* **Python 3.10 – 3.13 (64-bit)** from https://www.python.org/downloads/ – tick **"Add python.exe to PATH"**
  during installation.
* About 3 GB free disk space (PyTorch is large) and an internet connection for the first run
  (it downloads the ~22 MB pretrained weights once).

Check Python in Command Prompt:

```
python --version
```

## 7. Installation

Open **Command Prompt** (or PowerShell) and go to the project folder:

```
cd path\to\Wildlife-AI
```

Create and activate a virtual environment:

```
python -m venv venv
venv\Scripts\activate
```

(PowerShell: if activation is blocked run `Set-ExecutionPolicy -Scope Process Bypass` first.)

Install the dependencies:

```
pip install -r requirements.txt
```

The default `torch` package from PyPI on Windows is the CPU build, which is all you need.
(Optional: NVIDIA GPU owners can install a CUDA build from https://pytorch.org to train faster;
`train.py` uses the GPU automatically when available. The web app always runs fine on CPU.)

## 8. Run the app (pretrained mode – no training needed)

```
python app.py
```

Open **http://127.0.0.1:5000** in your browser. The first start downloads the pretrained weights, so you need
internet that one time. The badge on the home page shows which mode is active.

## 9. Dataset download and preparation (for custom mode)

1. Download **Animal Image Dataset (90 Different Animals)** from Kaggle (link in `dataset/README.md`).
2. Extract the zip into `dataset\raw\`.
3. Create the train/validation/test folders:

```
python prepare_dataset.py --source dataset\raw --output dataset --clean
```

Optional automatic download instead of steps 1–2 (needs `pip install kaggle` and a Kaggle API token in
`C:\Users\<you>\.kaggle\kaggle.json`):

```
python download_dataset.py --dataset animals90
```

## 10. Train the model

```
python train.py
```

Useful options:

```
python train.py --epochs 4 --finetune-epochs 2        (faster, lower accuracy)
python train.py --epochs 8 --finetune-epochs 6        (slower, higher accuracy)
python train.py --batch-size 16                       (if you run out of memory)
```

On a CPU, expect roughly 2–6 minutes per epoch for ~4,000 images (fine-tuning epochs are slower). Training
prints accuracy and loss each epoch and, at the end, saves:

* `trained_models/animal_model.pth`, `class_names.json`, `model_info.json`
* `outputs/training_history.png` (accuracy and loss graphs) and `training_history.json`
* test accuracy and weakest classes if `dataset/test/` exists

**Restart the app** (`Ctrl+C`, then `python app.py`). It switches to custom mode automatically. To force a mode:

```
set WILDLIFE_MODE=pretrained
python app.py
```

(values: `auto`, `pretrained`, `custom`). If the custom files are missing or damaged, `auto` falls back to
pretrained mode and shows a notice.

## 11. Using the website

1. Click **Upload an Image** or go to **Animal Recognition**.
2. Drag a photo onto the box or click **Browse Image** (JPG, PNG, WEBP or BMP, max 8 MB).
3. Click **Identify Animal**.
4. Read the predicted animal, confidence, top-3 bars and the information card.

You can also classify from the command line:

```
python predict.py path\to\photo.jpg
```

## 12. API

`POST /predict` – multipart form, field name `image`.

```json
{
  "success": true,
  "identified": true,
  "predicted_animal": "Elephant",
  "confidence": 96.5,
  "top_predictions": [
    {"animal": "Elephant", "confidence": 96.5},
    {"animal": "Hippopotamus", "confidence": 1.1},
    {"animal": "Rhinoceros", "confidence": 0.8}
  ],
  "animal_info": {"name": "Elephant", "category": "Mammal", "habitat": "...", "diet": "...", "facts": ["..."]},
  "message": null,
  "model": {"mode": "pretrained", "name": "MobileNetV3-Large"}
}
```

Low confidence → `identified: false`, `predicted_animal: null`, and `message` = "Unable to confidently identify
this animal. Please upload a clearer image." Errors return `{"success": false, "error": "..."}` with status
400 (bad file), 413 (too large), 500 or 503 (model problem).

Other endpoints: `GET /api/status` (active model, supported animals), `GET /health`.

## 13. Testing

Use clear photos with one animal in the frame. Expected results:

| Image | Pretrained mode | Custom mode (Animals-90) |
|-------|-----------------|--------------------------|
| Cat | Cat, usually >80% | Cat |
| Dog | Dog, usually >80% (all breeds are merged) | Dog |
| Cow | Cow, often 40–90% (cattle are matched as "ox"); can be lower for unusual angles | Cow or Ox (the dataset has both, so these two can be confused) |
| Tiger | Tiger, usually >80% | Tiger |
| Elephant | Elephant, usually >80% | Elephant |
| Car / building | Should show "Unable to confidently identify" | Often *not* rejected: it picks the closest animal. Check whether the confidence is low |

Also try: a giraffe photo (pretrained: not supported, expect a low-confidence result or a wrong guess), a
`.gif` file or text file renamed to `.jpg` (error message), and an image above 8 MB (size error).
These results depend on the photo, so treat the percentages as rough expectations.

## 14. Troubleshooting

| Problem | Fix |
|---------|-----|
| `python` is not recognised | Reinstall Python and tick "Add to PATH", or use `py` instead of `python` |
| `venv\Scripts\activate` blocked in PowerShell | `Set-ExecutionPolicy -Scope Process Bypass` or use Command Prompt |
| "Could not load the pretrained model" | You need internet once to download the weights; then restart |
| `ModuleNotFoundError: torch` | Activate the venv, then `pip install -r requirements.txt` |
| Port 5000 already in use | `set PORT=5001` then `python app.py` and open port 5001 |
| Training says "No usable dataset found" | Run `prepare_dataset.py` first (section 9) |
| Training runs out of memory or is very slow | `--batch-size 16`, fewer epochs, or fewer images per class |
| App still uses pretrained mode after training | Restart `python app.py`; check `trained_models/` has both `.pth` and `class_names.json` |
| Page shows "Server unreachable" | `python app.py` is not running |

## 15. Limitations

* A model only recognises what it was trained on; it cannot name animals outside its class list.
* Pretrained mode has no giraffe, rhinoceros, kangaroo, deer, goat, donkey, crow, pigeon or sparrow.
* Related breeds/species are merged (all dogs → Dog), so it does not identify breeds or exact species.
* A custom model always picks the closest known class, so non-animal images may get a confident-looking
  wrong answer. Pretrained mode rejects non-animals better, but not perfectly.
* Photos with several animals, tiny animals, heavy blur or unusual angles reduce accuracy.
* Accuracy with ~60 images per class is moderate; more and cleaner data helps.
* The animal facts are written by hand for a college project; verify before relying on them.
