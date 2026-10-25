# Projet 2 : Détection d'Émotions Faciales (Noir & Blanc)

Ce projet vise à développer un modèle de réseau de neurones convolutifs (CNN) capable de classifier automatiquement les expressions faciales à partir d'images en niveaux de gris (noir et blanc). Il s'agit d'une tâche de classification binaire visant à distinguer une expression positive (*Sourire / Joie*) d'une expression neutre ou négative (*Tristesse*).

---

## 📌 Objectif du Projet

- **Problématique** : Déterminer si un visage exprime la joie (`happy`) ou la tristesse (`sad`).
- **Type de tâche** : Classification binaire d'images.
- **Dataset de base** : FER-2013 (Kaggle), filtré pour ne conserver que les deux classes d'intérêt.

---

## 📊 Données & Prétraitement

- **Source des données** : Dataset **FER-2013** téléchargé via la bibliothèque `kagglehub`.
- **Format des images** :
  - **Dimensions** : 48 × 48 pixels.
  - **Mode de couleur** : Niveaux de gris (1 canal / `grayscale`).
- **Filtrage des classes** :
  - `happy` (Sourire / Joie)
  - `sad` (Neutre / Tristesse)
- **Répartition** :
  - **Entraînement (`train`)** : 12 045 images appartenant à 2 classes.
  - **Test (`test`)** : 3 021 images appartenant à 2 classes.
- **Augmentation & Normalisation** :
  - Rescaling des valeurs de pixels dans l'intervalle `[0, 1]`.
  - Augmentation de données modérée (`RandomFlip` horizontal, `RandomRotation`, `RandomZoom`).

---

## 🏗️ Architecture du Modèle (CNN)

Le modèle s'appuie sur une architecture séquentielle Keras (CNN profond) comprenant :

1. **Couche d'augmentation de données & Rescaling**
2. **Blocs de Convolution (Conv2D + BatchNormalization + MaxPooling2D)** :
   - Bloc 1 : 32 & 64 filtres (3x3)
   - Bloc 2 : 64 & 64 filtres (3x3)
   - Bloc 3 : 128 & 128 filtres (3x3)
   - Bloc 4 : 256 & 256 filtres (3x3) + `Dropout(0.2)`
3. **Classificateur Dense (Fully Connected)** :
   - `Flatten` (Vecteur de 2 304 valeurs)
   - Couches Denses successives (256, 128, 64, 16 neurones) avec `BatchNormalization` et `Dropout(0.5)`
   - **Couche de sortie** : 1 neurone avec activation `sigmoid` (classification binaire).

- **Optimiseur** : Adam
- **Fonction de perte** : Binary Crossentropy
- **Nombre total de paramètres** : 1 837 473 (~7.01 Mo)

---

## 📈 Résultats et Performances

L'entraînement a été configuré avec un callback `EarlyStopping` (patience de 5 époques sur `val_loss`).

| Ensemble | Perte (Loss) | Précision (Accuracy) |
| :--- | :---: | :---: |
| **Entraînement (Train)** | **0.2094** | **91.56%** |
| **Test (Validation)** | **0.2421** | **90.10%** |

---

## 🛠️ Préréquis & Installation

### 1. Environnement Python
Assurez-vous d'avoir Python 3.8+ installé.

### 2. Installation des dépendances
Installez les packages nécessaires avec la commande suivante :

```bash
pip install kagglehub pandas numpy scikit-learn tensorflow keras seaborn matplotlib statsmodels xgboost pillow