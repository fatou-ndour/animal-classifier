"""Application web Flask : identifier un chat, un chien ou un oiseau sur une photo.

Lancer avec :  python app.py   puis ouvrir http://127.0.0.1:5000
(le modèle du notebook Kaggle doit être copié dans le dossier models/)
"""
import json
import logging
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
# Le premier fichier trouvé est utilisé (modèle du notebook Kaggle, ou de train.py)
MODEL_CANDIDATES = [
    MODELS_DIR / "CatDogBird_model.keras",
    MODELS_DIR / "CatDogBird_model.h5",
    MODELS_DIR / "animal_model.keras",
]
LABELS_PATH = MODELS_DIR / "labels.json"

IMG_SIZE = (224, 224)                 # taille d'image attendue par le modèle
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD_MB = 5
CONFIDENCE_THRESHOLD = 0.60           # en dessous : « Je ne suis pas sûr »

# Noms d'affichage des classes (les dossiers de data/ sont en anglais)
DISPLAY = {
    "cat": {"name": "Chat", "emoji": "🐱"},
    "dog": {"name": "Chien", "emoji": "🐶"},
    "bird": {"name": "Oiseau", "emoji": "🐦"},
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

model = None
class_names = ["bird", "cat", "dog"]  # remplacé par labels.json au chargement


def load_model():
    """Charge le modèle entraîné et l'ordre des classes, une seule fois au démarrage."""
    global model, class_names
    model_path = next((p for p in MODEL_CANDIDATES if p.exists()), None)
    if model_path is None:
        log.warning("Aucun modèle trouvé dans %s. Copiez-y CatDogBird_model.keras.", MODELS_DIR)
        return
    import tensorflow as tf  # importé ici pour un démarrage rapide si le modèle manque

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # version de TensorFlow différente de celle de l'entraînement
        log.error(
            "Impossible de charger %s avec TensorFlow %s : %s\n"
            "Installez la même version de TensorFlow que dans le notebook Kaggle.",
            model_path.name, tf.__version__, exc,
        )
        return
    if LABELS_PATH.exists():
        class_names = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    log.info("Modèle %s chargé. Classes : %s", model_path.name, class_names)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare_image(file_storage) -> np.ndarray:
    """Ouvre l'image envoyée et la met au format du modèle : (1, 224, 224, 3)."""
    image = Image.open(file_storage.stream).convert("RGB").resize(IMG_SIZE)
    return np.expand_dims(np.asarray(image, dtype="float32"), axis=0)


def error(message: str, status: int):
    return jsonify({"error": message}), status


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if model is None:
        return error("Le modèle n'est pas chargé. Voir le README (dossier models/)", 503)

    file = request.files.get("image")
    if file is None or file.filename == "":
        return error("Aucune image reçue.", 400)
    if not allowed_file(file.filename):
        return error("Format non pris en charge. Utilisez JPG, PNG ou WEBP.", 400)

    try:
        batch = prepare_image(file)
    except (UnidentifiedImageError, OSError):
        return error("Ce fichier n'est pas une image valide.", 400)

    probabilities = model.predict(batch, verbose=0)[0]
    results = sorted(
        (
            {
                **DISPLAY.get(label, {"name": label.capitalize(), "emoji": "🐾"}),
                "probability": round(float(p) * 100, 1),
            }
            for label, p in zip(class_names, probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return jsonify(
        {
            "best": results[0],
            "is_confident": float(np.max(probabilities)) >= CONFIDENCE_THRESHOLD,
            "results": results,
        }
    )


@app.errorhandler(413)
def too_large(_):
    return error(f"Image trop lourde (maximum {MAX_UPLOAD_MB} Mo).", 413)


load_model()

if __name__ == "__main__":
    app.run(debug=True)
