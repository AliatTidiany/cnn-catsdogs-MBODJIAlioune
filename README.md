# 🐱🐶 CNN “From Scratch” vs Transfer Learning — Cats vs Dogs Classification

## 🎯 Objectif du projet

Ce projet a pour objectif de **comparer deux approches de classification d’images (chats vs chiens)** à l’aide de réseaux de neurones convolutionnels (CNN) :

1. **Modèle A – From Scratch** : entraînement complet d’un CNN conçu manuellement (au moins 3 blocs convolutionnels).  
2. **Modèle B – Transfer Learning** : utilisation d’un modèle pré-entraîné (ResNet18, MobileNetV2, ou EfficientNet) avec adaptation des couches finales.  

L’étude met en évidence l’impact du **transfert d’apprentissage** sur :
- la **vitesse de convergence**,  
- la **précision finale**,  
- et la **robustesse du modèle**.

---

## ⚙️ Environnement

### 🔧 Installation via `pip`
```bash
git clone https://github.com/<username>/cnn-catsdogs-AliouneMbodji.git
cd cnn-catsdogs-AliouneMbodji
pip install -r requirements.txt
```

### 📦 Librairies principales
- Python 3.10+
- PyTorch / torchvision
- NumPy / pandas
- Matplotlib / seaborn
- scikit-learn
- tqdm

---

## 📂 Organisation du dépôt

```
cnn-catsdogs-AliouneMbodji/
├─ data/                # dossier local contenant le dataset (non versionné)
├─ notebook.ipynb       # notebook principal (expériences A et B)
├─ requirements.txt     # dépendances Python
├─ README.md            # ce fichier
├─ .gitignore
└─ LICENSE (optionnel)
```

### `.gitignore` minimal
```
data/
*.pt
*.pth
runs/
checkpoints/
```

---

## 🧠 Jeu de données

Le jeu de données **Cats vs Dogs** est disponible publiquement sur Kaggle :  
👉 [https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)

### 📁 Organisation attendue :
```
data/
├─ train/
│  ├─ cat/
│  ├─ dog/
└─ test/
```

Téléchargez et extrayez le dataset dans le dossier `data/` avant l’exécution.

---

## 🚀 Commandes & Entraînement

### 🔹 Expérience A – CNN From Scratch
Architecture personnalisée (3 blocs Conv2D + BatchNorm + Dropout + MaxPool).  
Optimiseur : `Adam` ou `SGD`.  
Régularisation : **Dropout** et **Batch Normalization**.

Exemple d’exécution dans le notebook :
```python
python notebook.ipynb
# ou exécuter toutes les cellules de la section "Expérience A"
```

### 🔹 Expérience B – Transfer Learning
Base utilisée : **ResNet18** (pré-entraînée sur ImageNet).  
Dernières couches remplacées par un classifieur binaire (`cat` / `dog`).  
Deux configurations testées :
- **Frozen base** (feature extractor)
- **Fine-tuning complet** (poids mis à jour)

---

## 📊 Suivi des métriques

Les métriques suivantes sont calculées et tracées à chaque époque :
- **Accuracy**
- **Loss**
- **Précision**
- **Recall**

Les résultats sont visualisés sous forme de courbes :
- Courbes `train` / `validation` pour Loss & Accuracy
- Matrice de confusion sur le test final
- Exemples d’erreurs typiques (mauvaise prédiction commentée)

---

## 🧮 Optimisation et régularisation

- **Dropout** : évite le surapprentissage sur les couches denses finales.  
- **Batch Normalization** : stabilise l’entraînement après chaque couche convolutionnelle.  
- **Optimiseurs testés** : `SGD` et `Adam`.  
- **Scheduler** : `StepLR` testé pour ajuster le taux d’apprentissage.  
- **Data augmentation** : rotations, flips horizontaux et recadrages pour améliorer la généralisation.  

---

## 💾 Persistance du modèle

Le **meilleur modèle** est sauvegardé automatiquement :
```
checkpoints/best_model_from_scratch.pth
checkpoints/best_model_transfer.pth
```

Pour recharger et évaluer :
```python
model.load_state_dict(torch.load("checkpoints/best_model_transfer.pth"))
model.eval()
```

⚠️ Les fichiers `.pth` ne sont **pas inclus dans le dépôt**.

---

## 📈 Résultats comparatifs

| Modèle                  | Accuracy (val) | Précision | Recall | Époques | Optimiseur |
|--------------------------|----------------|------------|---------|----------|-------------|
| CNN From Scratch         | 84.2 %         | 83.7 %     | 82.5 %  | 25       | Adam        |
| Transfer Learning (ResNet18) | **96.1 %** | **95.8 %** | **96.4 %** | 10 | SGD (fine-tuning) |

### 🔍 Analyse
- Le modèle **from scratch** converge lentement et nécessite plus d’époques.  
- Le **transfert learning** atteint de meilleures performances en moins de temps grâce à la réutilisation de caractéristiques pré-apprises.  
- La **généralisation** sur des images inédites est également meilleure avec ResNet18.  

---

## ⚠️ Limites & Pistes d’amélioration

- Dataset relativement simple (binaire) → à étendre à plusieurs classes animales.  
- Entraînement limité (peu d’époques) par contrainte GPU.  
- Possible d’améliorer via :
  - Fine-tuning plus profond
  - Scheduler plus sophistiqué (CosineAnnealingLR)
  - Mixup / Cutout pour l’augmentation des données
  - Regularization L2

---

## 🧩 Reproductibilité

- **Seed fixé** pour NumPy, Torch et CUDA.  
- Entraînement réalisé sur **GPU NVIDIA (Google Colab)**.  
- Tous les hyperparamètres sont indiqués dans le notebook.  

---

##  Auteur

👤 **Alioune MBODJI**    
 Master 1 Intelligence Artificielle – Dakar Institute of Technology  
 Projet – Octobre 2025  

