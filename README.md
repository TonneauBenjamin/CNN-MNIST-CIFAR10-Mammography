# 🧠 De la Reconnaissance de Chiffres à la Détection du Cancer du Sein

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> Pipeline progressif de Machine Learning et Deep Learning : implémentation de modèles linéaires, de réseaux de neurones (MLP) et de réseaux convolutifs (CNN) codés **from scratch en NumPy**, puis transition vers **PyTorch** pour la classification d'images complexes et l'analyse de mammographies médicales (CBIS-DDSM).

---

## 🎯 Objectifs & Démarche Pédagogique

Ce projet illustre une montée en compétence progressive et rigoureuse en Computer Vision et Deep Learning :
1. **Comprendre les fondations mathématiques :** Coder la rétropropagation du gradient, la descente de gradient et les couches denses *from scratch* avec NumPy sans framework de haut niveau.
2. **Visualiser les mécanismes de convolution :** Implémentation manuelle du produit de convolution, des filtres (Sobel, flou gaussien, détection de contours) et du pooling.
3. **Mesurer les limites des architectures simples :** Comparer un modèle linéaire, un MLP et un CNN sur des jeux de données de complexité croissante.
4. **Application clinique réelle :** Application des architectures CNN à la détection de masses suspectes et à la classification bénin / malin sur le dataset de mammographie **CBIS-DDSM**.

---

## 📊 Résultats & Performances Comparatives

| Jeu de données | Modèle | Implémentation | Erreur Train | Erreur Test | Précision Test (Accuracy) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **MNIST** (28×28 niveaux de gris) | Modèle Linéaire | NumPy (scratch) | 11.15% | 10.62% | **89.38%** |
| **MNIST** (28×28 niveaux de gris) | MLP (Dense + Sigmoïde) | NumPy (scratch) | 0.25% | 2.12% | **97.88%** |
| **CIFAR-10** (32×32 couleur RGB) | Modèle Linéaire | NumPy (scratch) | 80.17% | 79.96% | **20.04%** |
| **CIFAR-10** (32×32 couleur RGB) | MLP | NumPy (scratch) | 44.67% | 53.67% | **46.33%** |
| **CIFAR-10** (32×32 couleur RGB) | **CNN Convolutif** | **PyTorch** | — | **26.39%** | **73.61%** |
| **CBIS-DDSM** (Mammographies) | CNN / Transfer Learning | PyTorch | — | — | **Classification bénin/malin** |

> 💡 **Constat clé :** Sur MNIST, un simple MLP atteint ~98% de précision. En revanche, sur CIFAR-10 (images couleur avec variabilité de pose et d'arrière-plan), le MLP plafonne à 46%, tandis que le **CNN PyTorch** bondit à **73.6%** grâce à l'invariance par translation et aux filtres spatiaux.

---

## 📁 Structure du Projet

```text
├── dataset/
│   └── mass_case_description_train_set.csv  # Métadonnées et annotations CBIS-DDSM
├── src/
│   ├── models_numpy.py                      # Modèle Linéaire & MLP codés from scratch (NumPy)
│   ├── mnist.py                             # Pipeline de chargement et entraînement MNIST
│   ├── convolution.py                       # Opérations de convolution et visualisation des filtres
│   ├── cifar10.py                           # Entraînement MLP & Linéaire sur CIFAR-10
│   ├── cnn_torch.py                         # Architecture ConvNet en PyTorch
│   ├── mass_dataset.py                      # DataLoader et pré-traitement des mammographies
│   ├── comparaison_activations.py           # Analyse des fonctions d'activation (ReLU, Sigmoïde, Tanh)
│   └── utils.py                             # Fonctions d'évaluation et de traçage des courbes
├── images/                                  # Sorties visuelles générées (filtres, prédictions)
├── main.py                                  # Script principal orchestrant l'ensemble des expériences
├── requirements.txt                         # Dépendances du projet
└── execution_output.txt                     # Traces d'exécution complètes et métriques
```

---

## 🛠️ Installation & Utilisation

### Prérequis
- Python 3.10+
- PyTorch & Torchvision
- NumPy, Matplotlib, Pandas

### Installation
```bash
# Cloner le dépôt
git clone https://github.com/TonneauBenjamin/CNN-MNIST-CIFAR10-Mammography.git
cd CNN-MNIST-CIFAR10-Mammography

# Installer les dépendances
pip install -r requirements.txt
```

### Lancement des expériences
```bash
# Exécuter l'ensemble du pipeline (MNIST, filtres de convolution, CIFAR-10, CBIS-DDSM)
python main.py
```

---

## 🔬 Détails des Composants

1. **NumPy from Scratch :**
   - Implémentation vectorisée des forward passes et backward passes.
   - Fonctions de perte : Cross-Entropy, MSE.
   - Optimisation : Descente de gradient stochastique (SGD) avec régularisation.
2. **Convolution & Feature Maps :**
   - Visualisation pas-à-pas de l'application de masques de convolution sur des images réelles.
   - Extraction de bordures horizontales/verticales et filtrage spatial.
3. **Mammographie Médicale (CBIS-DDSM) :**
   - Normalisation du contraste, gestion des zones d'intérêt (ROI) et catégorisation des calcifications et masses tissulaires.

---

## 👤 Auteur

**Benjamin Tonneau** — Étudiant en Data Engineering @ EFREI Paris  
- 💼 [LinkedIn](https://linkedin.com/in/benjamin-tonneau)
- 🐙 [GitHub](https://github.com/TonneauBenjamin)
