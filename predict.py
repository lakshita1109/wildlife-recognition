"""
predict.py - classify one image from the command line.

Usage:
    python predict.py path\\to\\image.jpg
    python predict.py path\\to\\image.jpg --mode pretrained
    python predict.py path\\to\\image.jpg --mode custom
"""

import argparse
import json
import sys

from model import model_config as cfg
from model.model_utils import AnimalClassifier, ImageError, ModelLoadError, open_image


def main():
    parser = argparse.ArgumentParser(description="Identify the animal in an image.")
    parser.add_argument("image", help="path to a jpg/png/webp/bmp image")
    parser.add_argument("--mode", choices=["auto", "pretrained", "custom"], default="auto",
                        help="auto = custom model if trained, otherwise pretrained")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    args = parser.parse_args()

    classifier = AnimalClassifier(mode=args.mode)
    try:
        classifier.load()
        image = open_image(args.image)
    except (ModelLoadError, ImageError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    result = classifier.predict(image)
    result["mode"] = classifier.mode

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Model mode: {classifier.mode}")
    if result["identified"]:
        top = result["top_predictions"][0]
        print(f"\nPredicted animal: {top['animal']} ({top['confidence']:.1f}%)")
    else:
        print(f"\n{cfg.LOW_CONFIDENCE_MESSAGE}")
    print("\nTop predictions:")
    for p in result["top_predictions"]:
        print(f"  {p['animal']:<28} {p['confidence']:6.2f}%")


if __name__ == "__main__":
    main()
