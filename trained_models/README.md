# Trained models folder

`python train.py` writes these files here:

| File | Purpose |
|------|---------|
| `animal_model.pth` | Trained MobileNetV3 weights |
| `class_names.json` | List of animal classes, in the order the model outputs them |
| `model_info.json`  | Accuracy, training date and class mapping |

When `animal_model.pth` and `class_names.json` both exist, the web app starts in **custom trained mode**
automatically. If they are missing (or broken), it falls back to **pretrained mode**.

To go back to pretrained mode, delete or move these files, or start the app with
`set WILDLIFE_MODE=pretrained` (Command Prompt) before `python app.py`.

These files are ignored by Git because they can be large.
