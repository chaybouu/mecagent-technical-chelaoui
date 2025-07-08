#!/usr/bin/env python3
"""
Script de test rapide pour vérifier le fonctionnement du projet.
"""

import sys
import os
import torch
import numpy as np

# Ajouter le répertoire src au path
sys.path.append('src')

def test_imports():
    """Teste les imports des modules."""
    print("Test des imports...")
    
    try:
        from src.data_loader import CadQueryDataset, get_tokenizer, create_dummy_dataset
        print("✓ data_loader importé avec succès")
    except Exception as e:
        print(f"✗ Erreur import data_loader: {e}")
        return False
    
    try:
        from src.model import create_model, BaselineModel, EnhancedModel
        print("✓ model importé avec succès")
    except Exception as e:
        print(f"✗ Erreur import model: {e}")
        return False
    
    try:
        from src.trainer import create_trainer, CadQueryTrainer
        print("✓ trainer importé avec succès")
    except Exception as e:
        print(f"✗ Erreur import trainer: {e}")
        return False
    
    try:
        from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
        from metrics.best_iou import get_iou_best
        print("✓ metrics importées avec succès")
    except Exception as e:
        print(f"✗ Erreur import metrics: {e}")
        return False
    
    return True

def test_data_loader():
    """Teste le data loader."""
    print("\nTest du data loader...")
    
    try:
        from src.data_loader import create_dummy_dataset, get_tokenizer, CadQueryDataset
        
        # Créer des données factices
        dummy_data = create_dummy_dataset()
        print(f"✓ Dataset factice créé: {len(dummy_data)} échantillons")
        
        # Charger le tokenizer
        tokenizer = get_tokenizer()
        print(f"✓ Tokenizer chargé: vocab_size={tokenizer.vocab_size}")
        
        # Créer le dataset
        dataset = CadQueryDataset(dummy_data, tokenizer)
        print(f"✓ Dataset créé: {len(dataset)} échantillons")
        
        # Test d'un échantillon
        sample = dataset[0]
        print(f"✓ Sample testé: image_shape={sample['image'].shape}, input_ids_shape={sample['input_ids'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur data loader: {e}")
        return False

def test_model():
    """Teste les modèles."""
    print("\nTest des modèles...")
    
    try:
        from src.model import create_model
        
        # Test du modèle baseline
        baseline_model = create_model("baseline", vocab_size=1000)
        dummy_images = torch.randn(2, 3, 224, 224)
        output = baseline_model(dummy_images)
        print(f"✓ Modèle baseline testé: output_shape={output.shape}")
        
        # Test du modèle enhanced
        enhanced_model = create_model("enhanced", vocab_size=1000)
        output = enhanced_model(dummy_images)
        print(f"✓ Modèle enhanced testé: output_shape={output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur modèle: {e}")
        return False

def test_trainer():
    """Teste le trainer."""
    print("\nTest du trainer...")
    
    try:
        from src.trainer import create_trainer
        from src.data_loader import create_dummy_dataset, get_tokenizer
        from src.model import create_model
        from torch.utils.data import DataLoader
        
        # Créer des données factices
        dummy_data = create_dummy_dataset()
        tokenizer = get_tokenizer()
        model = create_model("baseline", vocab_size=1000)
        
        # Créer des DataLoaders factices
        train_loader = DataLoader(dummy_data[:80], batch_size=4)
        val_loader = DataLoader(dummy_data[80:], batch_size=4)
        
        # Créer le trainer
        trainer = create_trainer(
            model, train_loader, val_loader, tokenizer,
            device='cpu', learning_rate=1e-4
        )
        print("✓ Trainer créé avec succès")
        
        return True
    except Exception as e:
        print(f"✗ Erreur trainer: {e}")
        return False

def test_metrics():
    """Teste les métriques."""
    print("\nTest des métriques...")
    
    try:
        from metrics.valid_syntax_rate import evaluate_syntax_rate_simple
        from metrics.best_iou import get_iou_best
        
        # Test VSR
        test_codes = {
            "test1": "result = cq.Workplane('XY').box(10, 10, 10)",
            "test2": "result = cq.Workplane('XY').cylinder(5, 10)"
        }
        vsr = evaluate_syntax_rate_simple(test_codes)
        print(f"✓ VSR testé: {vsr}")
        
        # Test IOU
        code1 = "result = cq.Workplane('XY').box(10, 10, 10)"
        code2 = "result = cq.Workplane('XY').box(10, 10, 10)"
        iou = get_iou_best(code1, code2)
        print(f"✓ IOU testé: {iou}")
        
        return True
    except Exception as e:
        print(f"✗ Erreur métriques: {e}")
        return False

def test_environment():
    """Teste l'environnement."""
    print("\nTest de l'environnement...")
    
    # Test PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
    
    # Test NumPy
    print(f"✓ NumPy version: {np.__version__}")
    
    # Test des seeds
    torch.manual_seed(42)
    np.random.seed(42)
    print("✓ Seeds configurés")
    
    return True

def main():
    """Fonction principale de test."""
    print("="*50)
    print("TEST RAPIDE DU PROJET CADQUERY")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Data Loader", test_data_loader),
        ("Modèles", test_model),
        ("Trainer", test_trainer),
        ("Métriques", test_metrics),
        ("Environnement", test_environment)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ Erreur inattendue dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé
    print("\n" + "="*50)
    print("RÉSUMÉ DES TESTS")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<15} {status}")
        if success:
            passed += 1
    
    print(f"\nTests passés: {passed}/{total}")
    
    if passed == total:
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("Le projet est prêt pour l'entraînement.")
        print("\nPour commencer l'entraînement:")
        print("python main.py --batch_size 4 --epochs_baseline 2 --epochs_enhanced 3")
    else:
        print("⚠️  CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez les dépendances et l'installation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 