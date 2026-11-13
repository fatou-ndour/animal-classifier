"""
Entraînement du modèle de reconnaissance Chat / Chien / Oiseau.

Principe : on réutilise MobileNetV2 (un réseau de neurones convolutif déjà
entraîné sur ImageNet) et on entraîne seulement une petite "tête" de
classification pour nos 3 classes. C'est le "transfer learning" : il donne
de bons résultats avec peu d'images et s'entraîne vite, même sans GPU.

Structure attendue des données :
    data/train/cat/*.jpg   data/val/cat/*.jpg
    data/train/dog/*.jpg   data/val/dog/*.jpg
    data/train/bird/*.jpg  data/val/bird/*.jpg

Lancement :
    python train.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # génère les graphiques dans des fichiers, sans fenêtre
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
SEED = 42


def load_datasets():
    """Charge les images depuis les dossiers train/ et val/."""
    common = dict(
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train", shuffle=True, **common
    )
    # shuffle=False : on garde l'ordre pour pouvoir calculer la matrice de confusion
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val", shuffle=False, **common
    )
    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune), class_names


def build_model(num_classes):
    """MobileNetV2 (gelé) + augmentation de données + tête de classification."""
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # on ne ré-entraîne pas le réseau de base

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)  # actif uniquement pendant l'entraînement
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(x)  # pixels 0-255 -> [-1, 1]
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    """Sauvegarde les courbes d'apprentissage (précision et perte)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(
        axes, ["accuracy", "loss"], ["Précision", "Perte (loss)"]
    ):
        ax.plot(history.history[metric], label="entraînement")
        ax.plot(history.history[f"val_{metric}"], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Époque")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "courbes_apprentissage.png", dpi=150)
    plt.close(fig)


def evaluate(model, val_ds, class_names):
    """Affiche le rapport de classification et sauvegarde la matrice de confusion."""
    y_true = np.concatenate([np.argmax(y, axis=1) for _, y in val_ds])
    y_pred = np.argmax(model.predict(val_ds, verbose=0), axis=1)

    print("\n=== Rapport de classification (jeu de validation) ===")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Matrice de confusion")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "matrice_confusion.png", dpi=150)
    plt.close(fig)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    train_ds, val_ds, class_names = load_datasets()
    print(f"Classes détectées : {class_names}")

    model = build_model(len(class_names))
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ],
    )

    plot_history(history)
    evaluate(model, val_ds, class_names)

    model.save(MODEL_DIR / "animal_model.keras")
    # L'ordre des classes doit être identique dans l'application web
    (MODEL_DIR / "labels.json").write_text(
        json.dumps(class_names, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nModèle sauvegardé dans {MODEL_DIR}")


if __name__ == "__main__":
    main()
