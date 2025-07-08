"""
Module d'entraînement simplifié pour le modèle de génération de code CadQuery.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime

class CadQueryTrainerSimple:
    """Trainer simplifié pour le modèle de génération de code CadQuery."""
    
    def __init__(self, model, train_loader, val_loader, tokenizer, 
                 device='cpu', learning_rate=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimiseur et loss
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        
        # Métriques
        self.train_losses = []
        self.val_losses = []
        self.accuracy_scores = []
        
        # Checkpointing
        self.best_accuracy = 0.0
        
    def train_epoch(self):
        """Entraîne le modèle pendant une époque."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Entraînement")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            target_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'forward'):
                # Modèle personnalisé
                logits = self.model(images, target_ids)
            else:
                # Modèle standard
                logits = self.model(images)
            
            # Calculer la loss
            if logits.dim() == 3:  # [batch, seq_len, vocab_size]
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1)
                )
            else:
                # Cas du modèle baseline
                loss = self.criterion(logits, target_ids[:, 0])
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Mettre à jour la barre de progression
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self):
        """Valide le modèle pendant une époque."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(self.device)
                target_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    logits = self.model(images, target_ids)
                else:
                    logits = self.model(images)
                
                # Calculer la loss
                if logits.dim() == 3:
                    loss = self.criterion(
                        logits.view(-1, logits.size(-1)),
                        target_ids.view(-1)
                    )
                    
                    # Calculer l'accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == target_ids).sum().item()
                    total_predictions += target_ids.numel()
                else:
                    loss = self.criterion(logits, target_ids[:, 0])
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        return total_loss / num_batches, accuracy
    
    def generate_code(self, images, max_length=512):
        """Génère du code CadQuery à partir d'images."""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                # Génération auto-régressive
                logits = self.model(images, max_length=max_length)
                if logits.dim() == 3:
                    tokens = torch.argmax(logits, dim=-1)
                else:
                    tokens = torch.argmax(logits, dim=-1)
            else:
                # Modèle baseline - génération simple
                logits = self.model(images)
                tokens = torch.argmax(logits, dim=-1)
            
            # Décoder les tokens
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            
            generated_code = self.tokenizer.decode(
                tokens[0], skip_special_tokens=True
            )
            
            return generated_code
    
    def train(self, num_epochs=10, save_dir='checkpoints'):
        """Entraîne le modèle complet."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Début de l'entraînement sur {num_epochs} époques...")
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        
        for epoch in range(num_epochs):
            print(f"\nÉpoque {epoch + 1}/{num_epochs}")
            
            # Entraînement
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, accuracy = self.validate_epoch()
            self.val_losses.append(val_loss)
            self.accuracy_scores.append(accuracy)
            
            # Affichage des résultats
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            
            # Sauvegarde du meilleur modèle
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                try:
                    self.save_checkpoint(f"{save_dir}/best_model.pth", epoch)
                    print(f"Nouveau meilleur accuracy: {accuracy:.4f}")
                except Exception as e:
                    print(f"⚠️  Erreur lors de la sauvegarde du meilleur modèle: {e}")
            
            # Sauvegarde régulière
            if (epoch + 1) % 5 == 0:
                try:
                    self.save_checkpoint(f"{save_dir}/checkpoint_epoch_{epoch+1}.pth", epoch)
                except Exception as e:
                    print(f"⚠️  Erreur lors de la sauvegarde régulière: {e}")
        
        # Sauvegarde finale (avec gestion d'erreur)
        try:
            self.save_checkpoint(f"{save_dir}/final_model.pth", num_epochs)
        except Exception as e:
            print(f"⚠️  Erreur lors de la sauvegarde finale: {e}")
        
        # Sauvegarde des métriques
        try:
            self.save_metrics(save_dir)
        except Exception as e:
            print(f"⚠️  Erreur lors de la sauvegarde des métriques: {e}")
        
        print(f"\nEntraînement terminé!")
        print(f"Meilleur accuracy: {self.best_accuracy:.4f}")
    
    def save_checkpoint(self, path, epoch):
        """Sauvegarde un checkpoint du modèle."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'accuracy_scores': self.accuracy_scores,
            'best_accuracy': self.best_accuracy
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """Charge un checkpoint du modèle."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.accuracy_scores = checkpoint['accuracy_scores']
        self.best_accuracy = checkpoint['best_accuracy']
        return checkpoint['epoch']
    
    def save_metrics(self, save_dir):
        """Sauvegarde les métriques d'entraînement."""
        metrics = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'accuracy_scores': self.accuracy_scores,
            'best_accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"{save_dir}/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Créer les graphiques
        self.plot_metrics(save_dir)
    
    def plot_metrics(self, save_dir):
        """Crée des graphiques des métriques d'entraînement."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss pendant l\'entraînement')
        axes[0, 0].set_xlabel('Époque')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.accuracy_scores, label='Accuracy', color='green')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Époque')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Comparaison
        axes[1, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[1, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[1, 0].set_title('Train vs Val Loss')
        axes[1, 0].set_xlabel('Époque')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        
        # Accuracy vs Loss
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(self.val_losses, label='Val Loss', color='red')
        ax2.plot(self.accuracy_scores, label='Accuracy', color='green')
        axes[1, 1].set_title('Accuracy vs Loss')
        axes[1, 1].set_xlabel('Époque')
        axes[1, 1].set_ylabel('Loss', color='red')
        ax2.set_ylabel('Accuracy', color='green')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/training_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_trainer_simple(model, train_loader, val_loader, tokenizer, **kwargs):
    """Factory pour créer un trainer simplifié."""
    return CadQueryTrainerSimple(model, train_loader, val_loader, tokenizer, **kwargs) 