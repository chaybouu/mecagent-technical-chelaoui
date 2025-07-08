"""
Module de chargement et préprocessing des données pour le modèle CadQuery.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import os

class CadQueryDataset(Dataset):
    """Dataset pour les paires Image/Code CadQuery."""
    
    def __init__(self, dataset, tokenizer, max_length=512, image_size=224):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Charger et préprocesser l'image
        if 'image' in item:
            image = item['image']
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert('RGB')
        else:
            # Image par défaut si manquante
            image = Image.new('RGB', (self.image_size, self.image_size), color='gray')
        
        # Redimensionner l'image
        image = image.resize((self.image_size, self.image_size))
        
        # Convertir en tensor
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Tokeniser le code
        code = item.get('code', '')
        if not code:
            code = "# Empty code"
            
        # Tokenisation avec padding et truncation
        encoding = self.tokenizer(
            code,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image_tensor,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'code': code
        }

def load_cadquery_dataset(cache_dir=None, split=['train', 'test']):
    """Charge le dataset CADCODER/GenCAD-Code."""
    try:
        dataset = load_dataset(
            "CADCODER/GenCAD-Code", 
            num_proc=4, 
            split=split, 
            cache_dir=cache_dir
        )
        return dataset
    except Exception as e:
        print(f"Erreur lors du chargement du dataset: {e}")
        # Dataset factice pour les tests
        return create_dummy_dataset()

def create_dummy_dataset(num_samples=100):
    """Crée un dataset factice pour les tests."""
    dummy_data = []
    for i in range(num_samples):
        dummy_data.append({
            'image': Image.new('RGB', (224, 224), color=(i % 255, (i*2) % 255, (i*3) % 255)),
            'code': f"""
height = {10 + i % 20}.0
width = {15 + i % 25}.0
thickness = {5 + i % 10}.0

result = cq.Workplane("XY").box(height, width, thickness)
"""
        })
    return dummy_data

def create_data_loaders(dataset, tokenizer, batch_size=8, train_split=0.8):
    """Crée les DataLoaders pour l'entraînement et la validation."""
    
    # Diviser le dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader

def get_tokenizer(model_name="microsoft/codebert-base"):
    """Retourne le tokenizer approprié pour le code CadQuery."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ajouter des tokens spéciaux pour CadQuery si nécessaire
        special_tokens = {
            'additional_special_tokens': [
                '<cadquery>', '</cadquery>',
                '<workplane>', '</workplane>',
                '<operation>', '</operation>'
            ]
        }
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer
    except Exception as e:
        print(f"Erreur lors du chargement du tokenizer: {e}")
        # Tokenizer de base
        return AutoTokenizer.from_pretrained("gpt2")

if __name__ == "__main__":
    # Test du module
    print("Test du module data_loader...")
    
    # Charger le dataset
    dataset = load_cadquery_dataset()
    print(f"Dataset chargé: {len(dataset)} échantillons")
    
    # Charger le tokenizer
    tokenizer = get_tokenizer()
    print("Tokenizer chargé")
    
    # Créer le dataset
    cadquery_dataset = CadQueryDataset(dataset, tokenizer)
    print(f"Dataset créé: {len(cadquery_dataset)} échantillons")
    
    # Test d'un échantillon
    sample = cadquery_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Code: {sample['code'][:100]}...") 