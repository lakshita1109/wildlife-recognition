"""
Model helpers shared by app.py, predict.py and train.py.

Framework: PyTorch / torchvision (MobileNetV3-Large, transfer learning).

Two inference modes:
  * "pretrained": ImageNet-pretrained MobileNetV3. Probabilities of related
    ImageNet classes are summed into broad animal labels (see model_config).
  * "custom": MobileNetV3 fine-tuned by train.py on the user's dataset.
    Used automatically when trained_models/ contains a valid model.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

from model import model_config as cfg

logger = logging.getLogger("wildlife_ai")

Image.MAX_IMAGE_PIXELS = cfg.MAX_IMAGE_PIXELS * 2  # PIL warns at this, we reject below


class ModelLoadError(Exception):
    """The model could not be loaded."""


class ImageError(Exception):
    """The uploaded image is invalid, corrupted or too large."""


# ----------------------------------------------------------------------------
# Image handling
# ----------------------------------------------------------------------------
def open_image(source):
    """Open an image from a path or file-like object and return an RGB PIL image.

    Raises ImageError for corrupted, unreadable, tiny or huge images.
    """
    try:
        img = Image.open(source)
        if img.width * img.height > cfg.MAX_IMAGE_PIXELS:
            raise ImageError(
                f"Image is too large ({img.width}x{img.height} pixels). "
                "Please upload a smaller image."
            )
        img.load()  # forces full decode, so truncated/corrupted files fail here
    except ImageError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError, SyntaxError) as exc:
        raise ImageError("The file is not a valid or readable image.") from exc

    img = ImageOps.exif_transpose(img)  # respect phone camera rotation

    if min(img.size) < cfg.MIN_IMAGE_SIDE:
        raise ImageError(
            f"Image is too small (minimum {cfg.MIN_IMAGE_SIDE}x{cfg.MIN_IMAGE_SIDE} pixels)."
        )

    # Flatten transparency onto white so PNGs with alpha are handled sensibly
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return img.convert("RGB")


def get_eval_transform():
    """Deterministic preprocessing used at inference AND for validation/testing."""
    return transforms.Compose([
        transforms.Resize(cfg.RESIZE_SIZE),
        transforms.CenterCrop(cfg.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
    ])


def get_train_transform():
    """Training preprocessing with data augmentation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
    ])


# ----------------------------------------------------------------------------
# Model construction
# ----------------------------------------------------------------------------
def build_model(num_classes=None, pretrained=True):
    """MobileNetV3-Large; optionally replace the final layer with `num_classes` outputs."""
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    net = models.mobilenet_v3_large(weights=weights)
    if num_classes is not None:
        in_features = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_features, num_classes)
    return net


def display_name(label):
    return cfg.DISPLAY_NAMES.get(label, label.replace("_", " ").replace("-", " ").title())


# ----------------------------------------------------------------------------
# Classifier used by the app
# ----------------------------------------------------------------------------
class AnimalClassifier:
    """Loads a model once and classifies PIL images.

    mode: "auto"       -> custom model if available and valid, otherwise pretrained
          "pretrained" -> always ImageNet-pretrained
          "custom"     -> only the custom model (fails if it is missing)
    """

    def __init__(self, mode="auto"):
        mode = (mode or "auto").lower()
        if mode not in ("auto", "pretrained", "custom"):
            raise ValueError("mode must be 'auto', 'pretrained' or 'custom'")
        self.requested_mode = mode
        self.mode = None                # actual mode after load(): "pretrained" / "custom"
        self.model = None
        self.class_names = []           # custom mode: folder names; pretrained: animal labels
        self.notice = None              # e.g. why we fell back to pretrained
        self.model_info = {}
        self.threshold = cfg.CONFIDENCE_THRESHOLD_PRETRAINED
        self._transform = get_eval_transform()
        self._group_indices = {}        # pretrained mode: label -> LongTensor of ImageNet indices
        self.device = torch.device("cpu")

    # -- loading -------------------------------------------------------------
    @property
    def ready(self):
        return self.model is not None

    def load(self):
        """Load the model (call once at startup). Raises ModelLoadError on failure."""
        if self.requested_mode in ("auto", "custom"):
            try:
                self._load_custom()
                return
            except ModelLoadError as exc:
                if self.requested_mode == "custom":
                    raise
                # auto: fall back, but remember why (None if there was simply no custom model)
                self.notice = str(exc) if cfg.MODEL_PATH.exists() else None
                if self.notice:
                    logger.warning("Custom model unusable, falling back to pretrained: %s", exc)
        self._load_pretrained()

    def _load_custom(self):
        if not cfg.MODEL_PATH.exists() or not cfg.CLASS_NAMES_PATH.exists():
            raise ModelLoadError(
                "No custom model found. Run 'python train.py' to create one "
                f"(expected {cfg.MODEL_PATH.name} and {cfg.CLASS_NAMES_PATH.name} "
                "in trained_models/)."
            )
        try:
            class_names = json.loads(cfg.CLASS_NAMES_PATH.read_text(encoding="utf-8"))
            if not isinstance(class_names, list) or len(class_names) < 2:
                raise ValueError("class_names.json must be a list with at least 2 classes")
            net = build_model(num_classes=len(class_names), pretrained=False)
            state = torch.load(cfg.MODEL_PATH, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
        except Exception as exc:  # noqa: BLE001 - report any failure uniformly
            raise ModelLoadError(f"Could not load the custom model: {exc}") from exc

        net.eval()
        self.model = net
        self.class_names = class_names
        self.mode = "custom"
        self.threshold = cfg.CONFIDENCE_THRESHOLD_CUSTOM
        if cfg.MODEL_INFO_PATH.exists():
            try:
                self.model_info = json.loads(cfg.MODEL_INFO_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.model_info = {}
        logger.info("Loaded custom model with %d classes", len(class_names))

    def _load_pretrained(self):
        try:
            net = build_model(pretrained=True)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "Could not load the pretrained model. The ImageNet weights (about 22 MB) "
                "are downloaded once on first use, so an internet connection is needed. "
                f"Details: {exc}"
            ) from exc
        net.eval()

        categories = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2.meta["categories"]
        name_to_index = {name: i for i, name in enumerate(categories)}
        groups = {}
        for label, entries in cfg.IMAGENET_ANIMAL_GROUPS.items():
            indices = []
            for entry in entries:
                if isinstance(entry, tuple):
                    indices.extend(range(entry[0], entry[1] + 1))
                elif entry in name_to_index:
                    indices.append(name_to_index[entry])
                else:
                    raise ModelLoadError(f"ImageNet class '{entry}' (label '{label}') not found.")
            groups[label] = torch.tensor(sorted(set(indices)), dtype=torch.long)

        self.model = net
        self._group_indices = groups
        self.class_names = sorted(groups)
        self.mode = "pretrained"
        self.threshold = cfg.CONFIDENCE_THRESHOLD_PRETRAINED
        self.model_info = {"model": cfg.MODEL_NAME, "weights": "ImageNet-1k (torchvision)"}
        logger.info("Loaded pretrained model with %d animal labels", len(groups))

    # -- inference -----------------------------------------------------------
    @torch.inference_mode()
    def predict(self, image):
        """Classify a PIL RGB image. Returns a dict with the top predictions."""
        if not self.ready:
            raise ModelLoadError("Model is not loaded.")

        batch = self._transform(image).unsqueeze(0)
        probs = torch.softmax(self.model(batch)[0], dim=0)

        if self.mode == "custom":
            scores = {name: float(probs[i]) for i, name in enumerate(self.class_names)}
        else:
            scores = {label: float(probs[idx].sum()) for label, idx in self._group_indices.items()}

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: cfg.TOP_K]
        top = [
            {"label": label, "animal": display_name(label), "confidence": round(p * 100, 2)}
            for label, p in ranked
        ]
        return {
            "top_predictions": top,
            "identified": top[0]["confidence"] >= self.threshold,
        }

    def supported_animals(self):
        return [display_name(c) for c in self.class_names]
