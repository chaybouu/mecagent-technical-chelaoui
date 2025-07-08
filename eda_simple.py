#!/usr/bin/env python3
"""
Analyse exploratoire des données (EDA) simplifiée - sans cadquery.
"""

from datasets import load_dataset
from PIL import Image
import os
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

def load_and_explore_dataset():
    """Charge et explore le dataset."""
    print("="*60)
    print("CHARGEMENT ET EXPLORATION DU DATASET")
    print("="*60)
    
    try:
        ds_train, ds_test = load_dataset(
            "CADCODER/GenCAD-Code",
            split=["train", "test"],
            num_proc=4,
            cache_dir="./hf_cache"
        )
        print(f"✓ Dataset chargé avec succès")
        print(f"Train examples: {len(ds_train):,}")
        print(f"Test examples:  {len(ds_test):,}")
        print(f"Total:          {len(ds_train) + len(ds_test):,}")
        
        return ds_train, ds_test
    except Exception as e:
        print(f"✗ Erreur lors du chargement du dataset: {e}")
        print("Utilisation de données factices pour la démonstration...")
        return create_dummy_datasets()

def create_dummy_datasets():
    """Crée des datasets factices pour la démonstration."""
    from PIL import Image
    import random
    
    # Créer des données factices
    dummy_train = []
    dummy_test = []
    
    for i in range(1000):
        # Image factice
        img = Image.new('RGB', (224, 224), 
                       color=(random.randint(0, 255), 
                              random.randint(0, 255), 
                              random.randint(0, 255)))
        
        # Code CadQuery factice
        height = 10 + random.randint(0, 50)
        width = 15 + random.randint(0, 60)
        thickness = 5 + random.randint(0, 20)
        
        code = f"""
height = {height}.0
width = {width}.0
thickness = {thickness}.0

result = cq.Workplane("XY").box(height, width, thickness)
"""
        
        item = {
            'image': img,
            'code': code,
            'id': f'dummy_{i}'
        }
        
        if i < 800:
            dummy_train.append(item)
        else:
            dummy_test.append(item)
    
    return dummy_train, dummy_test

def analyze_dataset_structure(ds_train, ds_test):
    """Analyse la structure du dataset."""
    print("\n" + "="*60)
    print("ANALYSE DE LA STRUCTURE DU DATASET")
    print("="*60)
    
    # Analyser les colonnes
    if hasattr(ds_train, 'column_names'):
        print(f"Colonnes train: {ds_train.column_names}")
        print(f"Colonnes test:  {ds_test.column_names}")
    else:
        print("Dataset factice - pas de colonnes prédéfinies")
    
    # Analyser un exemple
    ex0 = ds_train[0]
    print(f"\nExemple d'échantillon:")
    print(f"Clés disponibles: {list(ex0.keys())}")
    
    # Analyser le code
    if 'code' in ex0:
        code_sample = ex0['code'][:200].replace('\n', ' ')
        print(f"Extrait de code (200 chars): {code_sample}…")
        print(f"Longueur du code: {len(ex0['code'])} caractères")
    
    # Analyser l'image
    if 'image' in ex0:
        img = ex0['image']
        if isinstance(img, Image.Image):
            print(f"Type d'image: PIL.Image")
            print(f"Taille d'image: {img.size}")
            print(f"Mode d'image: {img.mode}")
        elif isinstance(img, str):
            print(f"Type d'image: chemin de fichier")
            print(f"Chemin: {img}")
        else:
            print(f"Type d'image: {type(img)}")

def visualize_sample_images(ds_train, num_samples=5):
    """Visualise des exemples d'images."""
    print("\n" + "="*60)
    print("VISUALISATION D'EXEMPLES D'IMAGES")
    print("="*60)
    
    os.makedirs("outputs", exist_ok=True)
    
    for i in range(min(num_samples, len(ds_train))):
        sample = ds_train[i]
        
        if 'image' in sample:
            img = sample['image']
            if isinstance(img, str):
                img = Image.open(img)
            elif not isinstance(img, Image.Image):
                continue
            
            # Redimensionner et sauvegarder
            img_resized = img.resize((224, 224))
            img_resized.save(f"outputs/sample_image_{i}.png")
            print(f"✓ Image {i} sauvegardée: outputs/sample_image_{i}.png")
            
            # Afficher des informations
            print(f"  Taille originale: {img.size}")
            print(f"  Mode: {img.mode}")

def test_metrics_simple():
    """Teste les métriques de manière simplifiée."""
    print("\n" + "="*60)
    print("TEST DES MÉTRIQUES (VERSION SIMPLIFIÉE)")
    print("="*60)
    
    # Test avec du code valide
    valid_code = """
height = 60.0
width = 80.0
thickness = 10.0
diameter = 22.0

result = (
    cq.Workplane("XY")
    .box(height, width, thickness)
)
"""
    
    # Test avec du code invalide
    invalid_code = """
height = 60.0
width = 80.0
thickness = 10.0

result = cq.Workplane("XY").box(height, width, thickness
# Manque la parenthèse fermante
"""
    
    print("✓ Test de syntaxe basique:")
    print(f"  Code valide: {len(valid_code)} caractères")
    print(f"  Code invalide: {len(invalid_code)} caractères")
    
    # Analyse simple de la syntaxe
    def simple_syntax_check(code):
        """Vérification basique de la syntaxe."""
        # Vérifier les parenthèses
        open_paren = code.count('(')
        close_paren = code.count(')')
        balanced = open_paren == close_paren
        
        # Vérifier les guillemets
        open_quote = code.count('"')
        close_quote = code.count('"')
        quotes_balanced = open_quote % 2 == 0
        
        return balanced and quotes_balanced
    
    print(f"✓ Syntaxe valide: {simple_syntax_check(valid_code)}")
    print(f"✓ Syntaxe invalide: {simple_syntax_check(invalid_code)}")

def analyze_code_statistics(ds_train, sample_size=5000):
    """Analyse les statistiques du code."""
    print("\n" + "="*60)
    print("ANALYSE STATISTIQUE DU CODE")
    print("="*60)
    
    # Charger le tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-small", use_fast=True)
        print(f"✓ Tokenizer chargé: {tokenizer.name_or_path}")
    except Exception as e:
        print(f"✗ Erreur tokenizer: {e}")
        return
    
    def count_tokens(code: str) -> int:
        """Compte le nombre de tokens dans un code."""
        try:
            return len(tokenizer(code).input_ids)
        except:
            return len(code.split())  # Fallback simple
    
    # Analyser un échantillon
    sample_size = min(sample_size, len(ds_train))
    sample = ds_train[:sample_size] if hasattr(ds_train, '__getitem__') else ds_train[:sample_size]
    
    print(f"Analyse d'un échantillon de {len(sample)} codes...")
    
    # Statistiques de longueur
    lengths = []
    code_samples = []
    
    for i, item in enumerate(sample):
        if 'code' in item:
            code = item['code']
            length = count_tokens(code)
            lengths.append(length)
            code_samples.append(code)
    
    if lengths:
        print(f"\nStatistiques des longueurs de code:")
        print(f"Moyenne tokens : {np.mean(lengths):.1f}")
        print(f"Médiane       : {np.percentile(lengths, 50):.0f}")
        print(f"Écart-type    : {np.std(lengths):.1f}")
        print(f"Min tokens    : {np.min(lengths)}")
        print(f"Max tokens    : {np.max(lengths)}")
        print(f"90ᵉ percentile : {np.percentile(lengths, 90):.0f}")
        print(f"95ᵉ percentile : {np.percentile(lengths, 95):.0f}")
        
        # Visualisation
        plt.figure(figsize=(12, 8))
        
        # Histogramme des longueurs
        plt.subplot(2, 2, 1)
        plt.hist(lengths, bins=50, alpha=0.7, color='skyblue')
        plt.title('Distribution des longueurs de code')
        plt.xlabel('Nombre de tokens')
        plt.ylabel('Fréquence')
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(lengths)
        plt.title('Box plot des longueurs')
        plt.ylabel('Nombre de tokens')
        
        # Analyse des patterns de code
        plt.subplot(2, 2, 3)
        patterns = analyze_code_patterns(code_samples[:100])  # Analyser les 100 premiers
        if patterns:
            plt.bar(range(len(patterns)), list(patterns.values()))
            plt.title('Patterns de code les plus fréquents')
            plt.xlabel('Pattern')
            plt.ylabel('Fréquence')
            plt.xticks(range(len(patterns)), list(patterns.keys()), rotation=45)
        
        # Courbe de distribution cumulative
        plt.subplot(2, 2, 4)
        sorted_lengths = np.sort(lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        plt.plot(sorted_lengths, cumulative)
        plt.title('Distribution cumulative')
        plt.xlabel('Nombre de tokens')
        plt.ylabel('Proportion cumulative')
        
        plt.tight_layout()
        plt.savefig('outputs/code_statistics.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Graphiques sauvegardés: outputs/code_statistics.png")
        plt.close()

def analyze_code_patterns(codes, top_n=10):
    """Analyse les patterns de code les plus fréquents."""
    patterns = []
    
    for code in codes:
        # Extraire les opérations CadQuery
        operations = re.findall(r'\.(\w+)\(', code)
        patterns.extend(operations)
    
    # Compter les occurrences
    pattern_counts = Counter(patterns)
    return dict(pattern_counts.most_common(top_n))

def main():
    """Fonction principale de l'EDA simplifiée."""
    print("ANALYSE EXPLORATOIRE DES DONNÉES - CADQUERY (SIMPLIFIÉE)")
    print("="*60)
    
    # 1. Charger et explorer le dataset
    ds_train, ds_test = load_and_explore_dataset()
    
    # 2. Analyser la structure
    analyze_dataset_structure(ds_train, ds_test)
    
    # 3. Visualiser des exemples
    visualize_sample_images(ds_train)
    
    # 4. Tester les métriques (version simplifiée)
    test_metrics_simple()
    
    # 5. Analyser les statistiques
    analyze_code_statistics(ds_train)
    
    print("\n" + "="*60)
    print("EDA SIMPLIFIÉE TERMINÉE AVEC SUCCÈS!")
    print("="*60)
    print("Fichiers générés:")
    print("- outputs/sample_image_*.png : Exemples d'images")
    print("- outputs/code_statistics.png : Statistiques du code")
    print("\nNote: Cette version utilise des données factices et des métriques simplifiées")
    print("car cadquery n'est pas disponible sur ce système.")
    print("\nProchaines étapes:")
    print("1. Analyser les résultats de l'EDA")
    print("2. Ajuster les paramètres du modèle selon les statistiques")
    print("3. Lancer l'entraînement: python main.py")

if __name__ == "__main__":
    main() 