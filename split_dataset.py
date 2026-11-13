"""
Répartit automatiquement des images en jeu d'entraînement (80 %) et de validation (20 %).

Mettez vos images brutes dans un dossier de ce type :
    raw_images/cat/...   raw_images/dog/...   raw_images/bird/...

Puis lancez :
    python split_dataset.py raw_images
Les images sont COPIÉES dans data/train/<classe>/ et data/val/<classe>/.
"""
import random
import shutil
import sys
from pathlib import Path

CLASSES = ["cat", "dog", "bird"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VAL_RATIO = 0.2
SEED = 42


def main(source):
    source = Path(source)
    base = Path(__file__).resolve().parent / "data"
    random.seed(SEED)

    for name in CLASSES:
        folder = source / name
        if not folder.is_dir():
            sys.exit(f"Dossier introuvable : {folder}")
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTENSIONS)
        random.shuffle(images)
        n_val = int(len(images) * VAL_RATIO)
        splits = {"val": images[:n_val], "train": images[n_val:]}

        for split, files in splits.items():
            target = base / split / name
            target.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, target / f.name)
        print(f"{name:5s} : {len(splits['train'])} train / {len(splits['val'])} val")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage : python split_dataset.py <dossier_images_brutes>")
    main(sys.argv[1])
