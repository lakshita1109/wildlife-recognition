"""
app.py - Flask backend for Wildlife AI.

Run:
    python app.py
Then open http://127.0.0.1:5000

Environment variables (optional):
    WILDLIFE_MODE = auto | pretrained | custom     (default: auto)
    PORT          = 5000
"""

import json
import logging
import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from model import model_config as cfg
from model.model_utils import AnimalClassifier, ImageError, ModelLoadError, open_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("wildlife_ai")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_UPLOAD_MB * 1024 * 1024

# ----------------------------------------------------------------------------
# Load everything ONCE at startup
# ----------------------------------------------------------------------------
try:
    ANIMAL_INFO = json.loads(cfg.ANIMAL_INFO_PATH.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    logger.warning("Could not read animal info file: %s", exc)
    ANIMAL_INFO = {}

classifier = AnimalClassifier(mode=os.environ.get("WILDLIFE_MODE", "auto"))
model_error = None


def load_model():
    """Load the model; remember the error text if it fails so the UI can show it."""
    global model_error
    try:
        classifier.load()
        model_error = None
    except ModelLoadError as exc:
        model_error = str(exc)
        logger.error("Model loading failed: %s", exc)


load_model()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def error_response(message, status):
    return jsonify({"success": False, "error": message}), status


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in cfg.ALLOWED_EXTENSIONS


def model_summary():
    return {
        "ready": classifier.ready,
        "mode": classifier.mode,
        "name": cfg.MODEL_NAME,
        "num_classes": len(classifier.class_names),
        "notice": classifier.notice,
        "error": model_error,
        "validation_accuracy": classifier.model_info.get("best_validation_accuracy"),
        "test_accuracy": classifier.model_info.get("test_accuracy"),
    }


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", max_mb=cfg.MAX_UPLOAD_MB,
                           extensions=sorted(cfg.ALLOWED_EXTENSIONS))


@app.route("/api/status")
def status():
    data = model_summary()
    data["supported_animals"] = classifier.supported_animals() if classifier.ready else []
    data["max_upload_mb"] = cfg.MAX_UPLOAD_MB
    return jsonify(data)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_ready": classifier.ready})


@app.route("/predict", methods=["POST"])
def predict():
    # Try to recover if the model failed to load earlier (e.g. no internet at first start)
    if not classifier.ready:
        load_model()
    if not classifier.ready:
        return error_response(model_error or "The AI model is not available.", 503)

    if "image" not in request.files:
        return error_response("No image was uploaded. Please select an image.", 400)
    file = request.files["image"]
    if not file or file.filename == "":
        return error_response("No image selected. Please choose an image file.", 400)
    if not allowed_file(file.filename):
        allowed = ", ".join(e.upper() for e in sorted(cfg.ALLOWED_EXTENSIONS))
        return error_response(f"Unsupported file type. Please upload one of: {allowed}.", 400)

    try:
        image = open_image(file.stream)   # decoded from memory, nothing is saved to disk
    except ImageError as exc:
        return error_response(str(exc), 400)

    try:
        result = classifier.predict(image)
    except Exception:  # noqa: BLE001
        logger.exception("Inference failed")
        return error_response("The AI model failed while analysing the image.", 500)

    top = [{"animal": p["animal"], "confidence": p["confidence"]}
           for p in result["top_predictions"]]
    response = {
        "success": True,
        "identified": result["identified"],
        "predicted_animal": None,
        "confidence": top[0]["confidence"],
        "top_predictions": top,
        "animal_info": None,
        "message": None,
        "model": {"mode": classifier.mode, "name": cfg.MODEL_NAME},
    }

    if result["identified"]:
        best = result["top_predictions"][0]
        response["predicted_animal"] = best["animal"]
        info = ANIMAL_INFO.get(best["label"])
        if info:
            response["animal_info"] = {"name": best["animal"], **info}
        else:
            response["message"] = "No detailed information is available for this animal yet."
    else:
        response["message"] = cfg.LOW_CONFIDENCE_MESSAGE

    return jsonify(response)


# ----------------------------------------------------------------------------
# Error handlers (always JSON for API routes)
# ----------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_):
    return error_response(
        f"The image is too large. The maximum upload size is {cfg.MAX_UPLOAD_MB} MB.", 413)


@app.errorhandler(Exception)
def unexpected(exc):
    if isinstance(exc, HTTPException):
        return error_response(exc.description, exc.code)
    logger.exception("Unhandled error")
    return error_response("Something went wrong on the server.", 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    print(f"\nWildlife AI is running: http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
