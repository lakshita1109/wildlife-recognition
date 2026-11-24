"""
Central configuration for Wildlife AI.

Everything that other files need to agree on (paths, image size, thresholds,
the ImageNet -> animal mapping used by pretrained mode) lives here.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
TRAINED_DIR = BASE_DIR / "trained_models"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"

ANIMAL_INFO_PATH = DATA_DIR / "animal_info.json"

# Files written by train.py and read by the app (custom-trained mode)
MODEL_PATH = TRAINED_DIR / "animal_model.pth"
CLASS_NAMES_PATH = TRAINED_DIR / "class_names.json"
MODEL_INFO_PATH = TRAINED_DIR / "model_info.json"

# ----------------------------------------------------------------------------
# Model / preprocessing (used by BOTH training and inference)
# ----------------------------------------------------------------------------
MODEL_NAME = "MobileNetV3-Large"
RESIZE_SIZE = 232          # resize the short side to this ...
IMAGE_SIZE = 224           # ... then center-crop to this
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ----------------------------------------------------------------------------
# Prediction behaviour
# ----------------------------------------------------------------------------
TOP_K = 3
# Minimum confidence (in %) for the top prediction to be reported as an answer.
CONFIDENCE_THRESHOLD_PRETRAINED = 35.0
CONFIDENCE_THRESHOLD_CUSTOM = 50.0
LOW_CONFIDENCE_MESSAGE = (
    "Unable to confidently identify this animal. Please upload a clearer image."
)

# ----------------------------------------------------------------------------
# Upload limits
# ----------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
MAX_UPLOAD_MB = 8
MAX_IMAGE_PIXELS = 40_000_000   # reject absurdly large images (decompression bombs)
MIN_IMAGE_SIDE = 32             # reject tiny images

# ----------------------------------------------------------------------------
# Display names for labels that are not simply label.title()
# ----------------------------------------------------------------------------
DISPLAY_NAMES = {
    "ladybugs": "Ladybug",
    "pelecaniformes": "Pelican (Pelecaniformes)",
    "boar": "Wild Boar",
    "ox": "Ox",
    "hippopotamus": "Hippopotamus",
    "goldfish": "Goldfish",
    "sandpiper": "Sandpiper",
}

# ----------------------------------------------------------------------------
# PRETRAINED MODE: mapping from ImageNet-1k classes to animal labels.
#
# The pretrained MobileNetV3 knows 1000 ImageNet classes. Many of them are
# animals, often very specific (120 dog breeds, 5 cat types, ...). We merge
# them into broad animal labels by SUMMING the softmax probabilities of all
# ImageNet classes that belong to one label. Entries are either a class name
# (exactly as in torchvision's ImageNet category list) or an inclusive
# (first_index, last_index) range.
#
# Animals that have NO ImageNet class (so pretrained mode can not recognise
# them): giraffe, rhinoceros, kangaroo, deer, goat, donkey, crow, pigeon,
# sparrow. Train a custom model (train.py) to get several of these.
# ----------------------------------------------------------------------------
IMAGENET_ANIMAL_GROUPS = {
    # Domestic animals
    "cat": [(281, 285)],
    "dog": [(151, 268)],
    "cow": ["ox"],
    "buffalo": ["water buffalo"],
    "horse": ["sorrel"],
    "sheep": ["ram"],
    "pig": ["hog"],
    "rabbit": ["wood rabbit", "Angora"],
    # Wild mammals
    "lion": ["lion"],
    "tiger": ["tiger"],
    "leopard": ["leopard", "snow leopard"],
    "cheetah": ["cheetah"],
    "elephant": ["Indian elephant", "African elephant"],
    "bear": ["brown bear", "American black bear", "ice bear", "sloth bear"],
    "wolf": ["timber wolf", "white wolf", "red wolf"],
    "fox": ["red fox", "kit fox", "Arctic fox", "grey fox"],
    "monkey": [
        "guenon", "patas", "baboon", "macaque", "langur", "colobus",
        "proboscis monkey", "marmoset", "capuchin", "howler monkey", "titi",
        "spider monkey", "squirrel monkey",
    ],
    "gorilla": ["gorilla"],
    "zebra": ["zebra"],
    "panda": ["giant panda"],
    "hippopotamus": ["hippopotamus"],
    "hyena": ["hyena"],
    "camel": ["Arabian camel"],
    # Birds
    "eagle": ["bald eagle"],
    "parrot": ["African grey", "macaw", "sulphur-crested cockatoo", "lorikeet"],
    "owl": ["great grey owl"],
    "peacock": ["peacock"],
    "flamingo": ["flamingo"],
    "penguin": ["king penguin"],
    "duck": ["drake", "red-breasted merganser"],
    "chicken": ["cock", "hen"],
    "ostrich": ["ostrich"],
    "swan": ["black swan"],
    "toucan": ["toucan"],
    # Reptiles, amphibians, fish, insects, arachnids
    "snake": [(52, 68)],
    "turtle": ["loggerhead", "leatherback turtle", "mud turtle", "terrapin", "box turtle"],
    "crocodile": ["African crocodile", "American alligator"],
    "lizard": [
        "banded gecko", "common iguana", "American chameleon", "whiptail",
        "agama", "frilled lizard", "alligator lizard", "Gila monster",
        "green lizard", "African chameleon", "Komodo dragon",
    ],
    "frog": ["bullfrog", "tree frog", "tailed frog"],
    "fish": [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "stingray", "barracouta", "eel", "coho", "rock beauty",
        "anemone fish", "sturgeon", "gar", "lionfish", "puffer",
    ],
    "butterfly": ["admiral", "ringlet", "monarch", "cabbage butterfly",
                  "sulphur butterfly", "lycaenid"],
    "bee": ["bee"],
    "spider": [(72, 77)],
}
