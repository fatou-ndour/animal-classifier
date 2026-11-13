# Chat, chien ou oiseau ? — Reconnaissance d'animaux avec Flask

Application web qui identifie un animal (**chat**, **chien** ou **oiseau**) à partir
d'une photo. Un réseau de neurones convolutif (CNN) fait la prédiction ; Flask
sert l'interface et l'API.

Projet inspiré de [egrizq/pizza_or_pasta](https://github.com/egrizq/pizza_or_pasta).

## Fonctionnement

1. L'utilisateur dépose une image dans le navigateur.
2. Le navigateur l'envoie à la route `/predict` (Flask).
3. Flask redimensionne l'image en 224×224 px et la donne au modèle.
4. Le modèle renvoie une probabilité par classe ; la page affiche le résultat.
   Si la probabilité maximale est inférieure à 60 %, l'application répond
   « Je ne suis pas sûr » plutôt que de se tromper avec assurance.

### Le modèle

- **Architecture** : MobileNetV2 (CNN pré-entraîné sur ImageNet) + une couche de
  classification à 3 sorties (softmax). C'est du *transfer learning*.
- **Augmentation de données** : retournement horizontal, rotation, zoom.
- **Entraînement** : Adam, `categorical_crossentropy`, arrêt anticipé (*early stopping*).

### Résultats

> À compléter après l'entraînement (voir `docs/`).

| Métrique                     | Valeur |
|------------------------------|--------|
| Précision sur l'entraînement | …      |
| Précision sur la validation  | …      |

Courbes d'apprentissage : `docs/courbes_apprentissage.png`
Matrice de confusion : `docs/matrice_confusion.png`

## Structure du projet

```
animal_classifier/
├── app.py                 # Serveur Flask (routes / et /predict)
├── train.py               # Entraînement du modèle
├── split_dataset.py       # Découpe des images en train / val
├── tests/test_app.py      # Tests automatiques de l'API Flask
├── requirements.txt       # Dépendances Python
├── models/                # animal_model.keras + labels.json (créés par train.py)
├── data/
│   ├── train/{cat,dog,bird}/
│   └── val/{cat,dog,bird}/
├── templates/index.html   # Page web
├── static/css/style.css   # Style
├── static/js/main.js      # Envoi de l'image et affichage du résultat
├── notebook/              # (optionnel) exploration dans Jupyter
└── docs/                  # Graphiques et captures d'écran
```

## Installation

Prérequis : Python 3.10 à 3.12.

```bash
python -m venv .venv
# Windows :  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### Option A : modèle entraîné dans le notebook Kaggle (TP2)

1. Exécutez `tp2-cat-dog-bird-complet.ipynb` sur Kaggle (EfficientNetB0).
2. Téléchargez `saved_model/` (`CatDogBird_model.keras` et `labels.json`).
3. Copiez ces deux fichiers dans le dossier `models/`.
4. Installez la **même version de TensorFlow** que sur Kaggle (voir `tf.__version__` dans le notebook).
5. Lancez `python app.py`.

### Option B : entraîner en local avec `train.py`

#### 1. Préparer les données

Rassemblez quelques centaines d'images par classe (idéalement 300 ou plus),
rangées ainsi :

```
raw_images/cat/   raw_images/dog/   raw_images/bird/
```

Puis :

```bash
python split_dataset.py raw_images
```

#### 2. Entraîner le modèle

```bash
python train.py
```

#### 3. Lancer l'application

```bash
python app.py
```

Ouvrir http://127.0.0.1:5000 dans le navigateur.

### 4. Lancer les tests (optionnel)

```bash
pytest
```

## Limites et pistes d'amélioration

- Le modèle ne connaît que 3 classes : une photo de cheval sera quand même
  classée comme chat, chien ou oiseau (le seuil de confiance limite ce problème
  sans le supprimer). On pourrait ajouter une classe « autre ».
- Les performances dépendent de la qualité et de la variété des images
  d'entraînement.
- Pistes : fine-tuning des dernières couches de MobileNetV2, plus de classes,
  déploiement en ligne (Render, Hugging Face Spaces).

## Auteur

Prénom Nom — Cours, établissement, année
