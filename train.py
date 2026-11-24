"""
train.py - transfer-learning training pipeline for Wildlife AI (PyTorch).

Expected dataset layout (create it with prepare_dataset.py):

    dataset/
      train/<class_name>/*.jpg
      validation/<class_name>/*.jpg
      test/<class_name>/*.jpg        (optional)

Phase 1 trains only the new classification layer (backbone frozen).
Phase 2 (optional) fine-tunes the last few backbone blocks with a low learning rate.

Outputs (in trained_models/ and outputs/):
    animal_model.pth, class_names.json, model_info.json
    training_history.png, training_history.json

Run:
    python train.py
    python train.py --epochs 5 --finetune-epochs 3 --batch-size 32
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets  # noqa: E402

from model import model_config as cfg  # noqa: E402
from model.model_utils import build_model, get_eval_transform, get_train_transform  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_warned_files = set()


def safe_loader(path):
    """Load an image as RGB; a corrupted file becomes a gray image instead of crashing training."""
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    except Exception:  # noqa: BLE001
        if path not in _warned_files:
            _warned_files.add(path)
            print(f"  [warning] could not read image, using blank placeholder: {path}")
        return Image.new("RGB", (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), (128, 128, 128))


def find_split_dir(root, names):
    for name in names:
        p = root / name
        if p.is_dir():
            return p
    return None


def has_images(folder):
    return folder is not None and any(
        f.suffix.lower() in IMAGE_EXTENSIONS for f in folder.rglob("*") if f.is_file()
    )


def print_dataset_help():
    print(
        "\nNo usable dataset found.\n"
        "Expected folders: dataset/train/<class>/..., dataset/validation/<class>/...\n\n"
        "To create it:\n"
        "  1. Download a dataset (see dataset/README.md), e.g. 'Animals-90' from Kaggle.\n"
        "     Optional helper:  python download_dataset.py\n"
        "  2. Arrange and split it:\n"
        "     python prepare_dataset.py --source path\\to\\raw\\animals --output dataset\n"
        "  3. Run  python train.py  again.\n"
    )


def run_epoch(model, loader, criterion, optimizer, device, train, frozen_blocks=0):
    if train:
        model.train()
        # Keep frozen backbone blocks in eval mode so their BatchNorm statistics stay unchanged
        for block in model.features[:frozen_blocks]:
            block.eval()
    else:
        model.eval()

    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            count += images.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def train_phase(name, model, train_loader, val_loader, criterion, optimizer, device,
                epochs, history, best, frozen_blocks):
    for epoch in range(1, epochs + 1):
        start = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device,
                                    True, frozen_blocks)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer, device, False)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["phase"].append(name)
        marker = ""
        if va_acc > best["acc"]:
            best["acc"] = va_acc
            best["state"] = copy.deepcopy(model.state_dict())
            marker = "  <- best so far"
        print(f"[{name}] epoch {epoch}/{epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc * 100:.2f}%  |  "
              f"val loss {va_loss:.4f} acc {va_acc * 100:.2f}%  "
              f"({time.time() - start:.0f}s){marker}")


def evaluate_test(model, test_dir, class_names, device, batch_size, workers):
    test_ds = datasets.ImageFolder(test_dir, transform=get_eval_transform(), loader=safe_loader)
    if test_ds.classes != class_names:
        # Only evaluate on classes that exist in training; remap labels by name
        print("  [warning] test classes differ from training classes; remapping by name.")
        test_ds.samples = [(p, class_names.index(test_ds.classes[t]))
                           for p, t in test_ds.samples if test_ds.classes[t] in class_names]
        test_ds.targets = [t for _, t in test_ds.samples]
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    model.eval()
    per_class_total = [0] * len(class_names)
    per_class_correct = [0] * len(class_names)
    loss_sum, correct, count = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * images.size(0)
            preds = outputs.argmax(1)
            for t, p in zip(labels.tolist(), preds.tolist()):
                per_class_total[t] += 1
                per_class_correct[t] += int(t == p)
            correct += (preds == labels).sum().item()
            count += images.size(0)

    acc = correct / max(count, 1)
    print(f"\nTest accuracy: {acc * 100:.2f}%  ({correct}/{count} images)  "
          f"loss {loss_sum / max(count, 1):.4f}")
    weakest = sorted(
        [(per_class_correct[i] / per_class_total[i], class_names[i], per_class_total[i])
         for i in range(len(class_names)) if per_class_total[i] > 0]
    )[:5]
    print("Weakest classes:", ", ".join(f"{n} ({a * 100:.0f}%, n={t})" for a, n, t in weakest))
    return acc, count


def save_plots(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [a * 100 for a in history["train_acc"]], label="Training")
    axes[0].plot(epochs, [a * 100 for a in history["val_acc"]], label="Validation")
    axes[1].plot(epochs, history["train_loss"], label="Training")
    axes[1].plot(epochs, history["val_loss"], label="Validation")
    for ax, title, ylabel in ((axes[0], "Accuracy", "Accuracy (%)"), (axes[1], "Loss", "Loss")):
        # mark where fine-tuning starts
        if "finetune" in history["phase"]:
            ax.axvline(history["phase"].index("finetune") + 0.5, color="gray",
                       linestyle="--", label="Fine-tuning starts")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the Wildlife AI animal classifier.")
    parser.add_argument("--data-dir", default=str(cfg.DATASET_DIR), help="dataset folder")
    parser.add_argument("--epochs", type=int, default=6, help="epochs for the classifier head")
    parser.add_argument("--finetune-epochs", type=int, default=4,
                        help="extra epochs fine-tuning the last backbone blocks (0 to skip)")
    parser.add_argument("--unfreeze-blocks", type=int, default=4,
                        help="how many final backbone blocks to unfreeze when fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for the head")
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0,
                        help="data loading processes (0 is safest on Windows)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="DEBUG ONLY: random weights, no download (accuracy will be poor)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  (PyTorch {torch.__version__})")

    root = Path(args.data_dir)
    train_dir = find_split_dir(root, ["train"])
    val_dir = find_split_dir(root, ["validation", "val", "valid"])
    test_dir = find_split_dir(root, ["test"])
    if not has_images(train_dir) or not has_images(val_dir):
        print_dataset_help()
        sys.exit(1)

    train_ds = datasets.ImageFolder(train_dir, transform=get_train_transform(), loader=safe_loader)
    val_ds = datasets.ImageFolder(val_dir, transform=get_eval_transform(), loader=safe_loader)
    class_names = train_ds.classes
    if val_ds.classes != class_names:
        missing = sorted(set(class_names) ^ set(val_ds.classes))
        print("ERROR: train and validation folders contain different classes:", missing)
        sys.exit(1)
    print(f"Classes: {len(class_names)}   train images: {len(train_ds)}   "
          f"validation images: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)

    try:
        model = build_model(num_classes=len(class_names), pretrained=not args.no_pretrained)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not load pretrained weights (internet needed once): {exc}")
        sys.exit(1)
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "phase": []}
    best = {"acc": -1.0, "state": None}
    n_blocks = len(model.features)

    # ---- Phase 1: train the classifier head only ----
    for p in model.features.parameters():
        p.requires_grad = False
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    print("\nPhase 1: training the classification layer (backbone frozen)")
    train_phase("head", model, train_loader, val_loader, criterion, optimizer, device,
                args.epochs, history, best, frozen_blocks=n_blocks)

    # ---- Phase 2: fine-tune the last backbone blocks ----
    if args.finetune_epochs > 0:
        if best["state"] is not None:
            model.load_state_dict(best["state"])
        frozen = max(n_blocks - args.unfreeze_blocks, 0)
        for block in model.features[frozen:]:
            for p in block.parameters():
                p.requires_grad = True
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=args.finetune_lr, weight_decay=1e-4)
        print(f"\nPhase 2: fine-tuning the last {n_blocks - frozen} backbone blocks")
        train_phase("finetune", model, train_loader, val_loader, criterion, optimizer, device,
                    args.finetune_epochs, history, best, frozen_blocks=frozen)

    # ---- Save best model ----
    model.load_state_dict(best["state"])
    model.to("cpu")
    cfg.TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.MODEL_PATH)
    cfg.CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    info = {
        "model": cfg.MODEL_NAME,
        "num_classes": len(class_names),
        "image_size": cfg.IMAGE_SIZE,
        "best_validation_accuracy": round(best["acc"] * 100, 2),
        "epochs_head": args.epochs,
        "epochs_finetune": args.finetune_epochs,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "class_to_idx": train_ds.class_to_idx,
    }

    print(f"\nBest validation accuracy: {best['acc'] * 100:.2f}%")
    if has_images(test_dir):
        acc, n = evaluate_test(model, test_dir, class_names, torch.device("cpu"),
                               args.batch_size, args.workers)
        info["test_accuracy"] = round(acc * 100, 2)
        info["test_images"] = n
    else:
        print("No test folder found - skipping test evaluation.")

    cfg.MODEL_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    (cfg.OUTPUTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2),
                                                           encoding="utf-8")
    save_plots(history, cfg.OUTPUTS_DIR / "training_history.png")

    print(f"\nSaved model:   {cfg.MODEL_PATH}")
    print(f"Saved classes: {cfg.CLASS_NAMES_PATH}")
    print(f"Saved graphs:  {cfg.OUTPUTS_DIR / 'training_history.png'}")
    print("Restart the web app (python app.py) - it will use the custom model automatically.")


if __name__ == "__main__":
    main()
