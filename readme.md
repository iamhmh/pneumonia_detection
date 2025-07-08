# Zoidberg 2.0

## Description
Projet d'analyse d'images médicales utilisant le machine learning et le deep learning pour la détection automatique de la pneumonie à partir de radiographies pulmonaires.

## Structure du Projet
```
.
├── dataset/               
│   ├── train/                 # Images d'entraînement
│   ├── val/                   # Images de validation
│   └── test/                  # Images de test
├── processed/                 # Données prétraitées
├── notebooks/             
│   ├── 01_EDA.ipynb           # Analyse exploratoire des données
│   └── 02_Modeling.ipynb      # Développement des modèles
│   └── 03_DeepLearning.ipynb  # Développement des modèles
├── reports/              
│   └── eda/                   # Graphiques d'analyse exploratoire des données
│   └── modeling/              # Graphiques d'anlyses des eprformances des models
├── src/                   
│   └── app.py                 # Page Web de test
│   └── predict.py             # Récupération du modèle, traitement de l'image et prédiction
│   └── preprocessing.py       # Scripts de prétraitement
└── models/                    # Modèles entraînés
└── templates/                 # Point d'entrée web 
└── uploads/                   # Suavegarde locale des images (web)
```

## Installation

### Prérequis
- Python 3.13+
- pip ou conda

### Configuration de l'environnement

```bash
# Création de l'environnement virtuel
python -m venv .venv

# Activation de l'environnement (macOS/Linux)
source .venv/bin/activate

# Activation de l'environnement (windows)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1

# Installation des dépendances
pip install -r requirements.txt
```

## Dataset
Le dataset est organisé en trois parties :
- Train : 1341 images normales, 3875 images de pneumonie
- Validation : 8 images par classe
- Test : 234 images normales, 390 images de pneumonie

## Utilisation

### Analyse Exploratoire des Données
```bash
jupyter lab notebooks/01_EDA.ipynb
```

### Entraînement des Modèles de Machine Learning
```bash
jupyter lab notebooks/02_Modeling.ipynb
```

### Entraînement des Modèles de Deep Learning
```bash
jupyter lab notebooks/03_DeepLearning.ipynb
```

## Modèles pré-entraînés

En raison de leur taille, les modèles pré-entraînés ne sont pas inclus dans ce dépôt GitHub. Vous pouvez :

- Les générer en exécutant les notebooks `02_Modeling.ipynb` et `03_DeepLearning.ipynb`

Placez les fichiers téléchargés dans le dossier `models/` :
- `best_classical_model.pkl` - Modèle de machine learning classique
- `resnet18_pneumonia.pth` - Modèle de deep learning ResNet18

## Technologies Utilisées
- **Data Processing**: NumPy, Pandas
- **Image Processing**: OpenCV, Pillow, Albumentations
- **Machine Learning**: Scikit-learn, PyTorch
- **Visualisation**: Matplotlib, Seaborn, Plotly
- **Deep Learning**: PyTorch, Torchvision

## Versions des Dépendances
Voir le fichier `requirements.txt` pour la liste complète des dépendances et leurs versions.

## Author
[**Hichem GOUIA**](https://github.com/iamhmh)
[**Melvyn DENIS**](https://github.com/MelvynDenisEpitech)
[**Mathéo SERRIER**](https://github.com/matheoSerrier)