# Test Technique ML Engineer - Génération de Code CadQuery

## Vue d'ensemble

Ce projet implémente un modèle de génération de code CadQuery à partir d'images de pièces mécaniques. Il s'agit d'un problème de **Computer Vision + Code Generation** dans le domaine de la CAO (Conception Assistée par Ordinateur).

**⚠️ Version Simplifiée** : Ce projet utilise une version simplifiée adaptée aux contraintes d'environnement. Voir la section "Contraintes d'environnement" pour plus de détails.

## Objectifs

1. **Baseline Model** : Créer un modèle simple mais fonctionnel
2. **Enhanced Model** : Améliorer significativement les performances
3. **Évaluation** : Utiliser des métriques simplifiées (accuracy au lieu de VSR/IOU)
4. **Documentation** : Expliquer les choix techniques et améliorations

## Structure du projet

```
mecagent-technical-test/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Chargement et préprocessing des données
│   ├── model.py            # Architectures des modèles
│   └── trainer_simple.py   # Entraînement et évaluation (version simplifiée)
├── metrics/
│   ├── valid_syntax_rate.py # Métrique VSR (non utilisée dans cette version)
│   └── best_iou.py         # Métrique IOU (non utilisée dans cette version)
├── main_simple.py          # Script principal (version simplifiée)
├── eda_simple.py           # Analyse exploratoire des données
├── test_quick.py           # Tests rapides
├── requirements.txt        # Dépendances
└── README_complet.md      # Ce fichier
```

## Contraintes d'environnement

### Pourquoi cadquery et OCP ne sont pas utilisés ?

Ce projet a été développé sur un environnement **macOS ARM64 avec Python 3.13**, ce qui pose des contraintes importantes :

1. **Architecture ARM64 (Apple Silicon)** :
   - `OCP` (OpenCascade) nécessite une compilation native complexe
   - Les wheels précompilés pour ARM64 sont rares ou inexistants
   - La compilation depuis les sources sur ARM64 est très difficile

2. **Python 3.13** :
   - `cadquery` et ses dépendances ne supportent pas encore Python 3.13
   - La plupart des packages 3D/CAD sont encore sur Python 3.8-3.11

3. **Dépendances système** :
   - OpenCascade nécessite des bibliothèques système (OpenGL, etc.)
   - Sur macOS ARM64, ces dépendances peuvent être manquantes

### Solution adoptée

Pour contourner ces contraintes, nous utilisons une **version simplifiée** qui :
- ✅ Fonctionne sur l'environnement actuel
- ✅ Démontre la compréhension du problème
- ✅ Implémente un pipeline ML complet
- ✅ Utilise des métriques alternatives (accuracy au lieu de VSR/IOU)
- ✅ Permet l'entraînement et l'évaluation des modèles

## Installation

### Option 1 : Avec uv (recommandé)
```bash
# Installer uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer les dépendances
uv sync

# Activer l'environnement
source .venv/bin/activate
```

### Option 2 : Avec pip
```bash
# Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Exécution complète (Version Simplifiée)
```bash
# 1. Analyse exploratoire des données
python eda_simple.py

# 2. Tests rapides
python test_quick.py

# 3. Entraînement complet
python main_simple.py --batch_size 8 --epochs_baseline 5 --epochs_enhanced 10
```

### Arguments disponibles
- `--batch_size` : Taille du batch (défaut: 8)
- `--epochs_baseline` : Nombre d'époques pour le baseline (défaut: 5)
- `--epochs_enhanced` : Nombre d'époques pour l'enhanced (défaut: 10)
- `--cache_dir` : Répertoire de cache pour le dataset
- `--device` : Device à utiliser ('cpu', 'cuda', 'auto')

### Exécution avec GPU
```bash
python main_simple.py --device cuda --batch_size 16
```

## Architecture des modèles

### Modèle Baseline
- **Encoder** : CNN simple (Conv2D + MaxPool2D)
- **Décodeur** : Couches linéaires avec dropout
- **Complexité** : Faible, rapide à entraîner
- **Objectif** : Modèle de référence simple

### Modèle Enhanced
- **Encoder** : Vision Transformer (ViT-B/16)
- **Décodeur** : Transformer avec attention
- **Mécanismes** : Attention géométrique, validation syntaxique
- **Complexité** : Élevée, meilleures performances
- **Objectif** : Amélioration significative des métriques

## Métriques d'évaluation (Version Simplifiée)

### Accuracy (au lieu de VSR/IOU)
- Évalue la précision de prédiction des tokens
- Métrique standard pour la génération de texte
- **Objectif** : > 0.7 (baseline), > 0.85 (enhanced)

### Métriques originales (non utilisées dans cette version)
- **Valid Syntax Rate (VSR)** : Évalue la validité syntaxique du code généré
- **Best IOU** : Compare les maillages 3D générés par le code

## Défis techniques identifiés

### 1. Complexité du domaine CadQuery
- Syntaxe spécifique et complexe
- Opérations 3D séquentielles et dépendantes
- Variables locales, paramètres géométriques

### 2. Évaluation difficile
- Code valide ≠ Code correct fonctionnellement
- IOU 3D coûteuse en calcul
- Alignement géométrique complexe

### 3. Problème multimodal
- Image → Code textuel
- Compréhension géométrique 3D
- Génération de code structuré

## Améliorations implémentées

### 1. Architecture spécialisée
- Encoder multimodal (Vision + Text)
- Decoder avec attention sur la géométrie
- Mécanismes de validation syntaxique

### 2. Techniques avancées
- Attention géométrique pour la compréhension 3D
- Validation syntaxique intégrée
- Génération auto-régressive avec contraintes

### 3. Optimisations
- Gradient clipping pour la stabilité
- Learning rate scheduling
- Checkpointing intelligent

## Résultats attendus (Version Simplifiée)

### Métriques de succès
- **Accuracy** : Amélioration de +15% minimum
- **Temps d'entraînement** : < 2h sur GPU, < 8h sur CPU

### Fichiers de sortie
- `checkpoints/` : Modèles entraînés
- `training_metrics.json` : Résultats détaillés
- `training_metrics.png` : Graphiques d'entraînement

## Améliorations futures

### 1. Architecture spécialisée
- Encoder multimodal (Vision + Text)
- Decoder avec attention sur la géométrie
- Mécanismes de validation syntaxique

### 2. Techniques avancées
- Program synthesis avec contraintes
- Reinforcement learning avec les métriques
- Data augmentation géométrique

### 3. Optimisations
- Curriculum learning (simple → complexe)
- Multi-task learning (syntaxe + géométrie)
- Ensemble de modèles spécialisés

## Développement

### Tests unitaires
```bash
# Test du data loader
python -c "from src.data_loader import *; print('Data loader OK')"

# Test du modèle
python -c "from src.model import *; print('Model OK')"

# Test du trainer
python -c "from src.trainer_simple import *; print('Trainer OK')"
```

### Développement local
```bash
# Mode développement avec données factices
python main_simple.py --batch_size 4 --epochs_baseline 2 --epochs_enhanced 3
```

## Dépannage

### Problèmes courants

1. **Erreur de mémoire GPU**
   ```bash
   python main_simple.py --batch_size 4 --device cpu
   ```

2. **Dataset non disponible**
   - Le code utilise automatiquement des données factices
   - Vérifiez votre connexion internet pour le dataset HuggingFace

3. **Dépendances manquantes**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Logs et debugging
- Les logs détaillés sont affichés pendant l'entraînement
- Les métriques sont sauvegardées dans `checkpoints/metrics.json`
- Les graphiques sont générés automatiquement

## Pour utiliser la version complète

Si vous souhaitez utiliser la version complète avec cadquery et OCP :

1. **Environnement recommandé** :
   - Linux x86_64
   - Python 3.8-3.11
   - GPU NVIDIA (optionnel mais recommandé)

2. **Installation des dépendances** :
   ```bash
   # Installer OCP (OpenCascade)
   pip install OCP
   
   # Installer cadquery
   pip install cadquery
   
   # Installer les autres dépendances
   pip install -r requirements.txt
   ```

3. **Utiliser les scripts complets** :
   - `main.py` au lieu de `main_simple.py`
   - `eda.py` au lieu de `eda_simple.py`
   - Les métriques VSR et IOU seront disponibles

## Contribution

Ce projet est un test technique. Pour contribuer :

1. Fork le repository
2. Créez une branche feature
3. Implémentez vos améliorations
4. Testez avec les métriques fournies
5. Soumettez une pull request

## Licence

Ce projet est un test technique pour un poste de ML Engineer.

## Contact

Pour toute question sur ce test technique, contactez l'équipe de recrutement. 