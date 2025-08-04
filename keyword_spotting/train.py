# Training script for BC-ResNet keyword spotting
# Based on "Broadcasted Residual Learning for Efficient Keyword Spotting" (Kim et al., 2021)

import os
import argparse
import numpy as np
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from bcresnet import create_bcresnet
from data_utils import (
    SpeechCommandsDataset, 
    AudioPreprocessor, 
    Padding, 
    prepare_speech_commands_dataset,
    LABEL_DICT
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class KeywordSpottingTrainer:
    """
    Trainer class for BC-ResNet keyword spotting model.
    Implements the training pipeline from the original paper.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
        print(f"Using device: {self.device}")
        
        # Load data
        self.dataset_paths = self._prepare_data()
        self._load_datasets()
        
        # Load model
        self.model = self._load_model()
        
        # Setup preprocessors
        self._setup_preprocessors()
        
        print(f"Training setup complete:")
        print(f"  Model: BC-ResNet-{args.tau}")
        print(f"  Dataset: Speech Commands v{args.version}")
        print(f"  Train samples: {len(self.train_dataset)}")
        print(f"  Valid samples: {len(self.valid_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")

    def _prepare_data(self) -> Dict[str, str]:
        """Prepare Speech Commands dataset"""
        print("Preparing Speech Commands dataset...")
        return prepare_speech_commands_dataset(
            data_dir=self.args.data_dir,
            version=self.args.version,
            download=self.args.download
        )

    def _load_datasets(self):
        """Load train/valid/test datasets"""
        transform = Padding()
        
        self.train_dataset = SpeechCommandsDataset(
            self.dataset_paths["train"], 
            self.args.version, 
            transform=transform
        )
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=self.args.num_workers,
            drop_last=False
        )
        
        self.valid_dataset = SpeechCommandsDataset(
            self.dataset_paths["valid"], 
            self.args.version, 
            transform=transform
        )
        self.valid_loader = DataLoader(
            self.valid_dataset, 
            batch_size=self.args.batch_size, 
            num_workers=self.args.num_workers
        )
        
        self.test_dataset = SpeechCommandsDataset(
            self.dataset_paths["test"], 
            self.args.version, 
            transform=transform
        )
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.args.batch_size, 
            num_workers=self.args.num_workers
        )

    def _load_model(self):
        """Load BC-ResNet model"""
        print(f"Creating BC-ResNet-{self.args.tau} model...")
        model = create_bcresnet(tau=self.args.tau, num_classes=len(LABEL_DICT))
        model = model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model

    def _setup_preprocessors(self):
        """Setup audio preprocessors for train and test"""
        # SpecAugment is enabled for models with tau >= 1.5
        specaugment = self.args.tau >= 1.5
        
        # Frequency masking parameters based on model size
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}
        freq_mask_para = frequency_masking_para.get(self.args.tau, 7)
        
        # Training preprocessor with augmentation
        self.train_preprocessor = AudioPreprocessor(
            noise_dir=self.dataset_paths["noise"],
            device=self.device,
            specaug=specaugment,
            frequency_masking_para=freq_mask_para,
        )
        
        # Test preprocessor without augmentation
        self.test_preprocessor = AudioPreprocessor(
            noise_dir=self.dataset_paths["noise"],
            device=self.device,
            specaug=False
        )
        
        print(f"Preprocessors setup - SpecAugment: {specaugment}, Freq mask: {freq_mask_para}")

    def train_epoch(self, epoch: int, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Preprocess inputs
            inputs = self.train_preprocessor(inputs, labels, augment=True)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Update scheduler
        if scheduler:
            scheduler.step()
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

    def evaluate(self, data_loader, preprocessor, augment: bool = False) -> float:
        """Evaluate model on given dataset"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Preprocess inputs
                inputs = preprocessor(inputs, labels, augment=augment, is_train=False)
                
                # Forward pass
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy

    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Setup optimizer and scheduler
        optimizer = SGD(
            self.model.parameters(), 
            lr=self.args.lr, 
            momentum=0.9, 
            weight_decay=1e-3
        )
        
        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        
        best_valid_acc = 0.0
        best_model_path = os.path.join(self.args.save_dir, f"best_bcresnet_{self.args.tau}_v{self.args.version}.pth")
        
        for epoch in range(self.args.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(epoch, optimizer, scheduler)
            
            # Validate
            valid_acc = self.evaluate(self.valid_loader, self.test_preprocessor, augment=False)
            
            print(f"Epoch {epoch+1}/{self.args.epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Valid Acc: {valid_acc:.2f}%")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                os.makedirs(self.args.save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_acc': valid_acc,
                    'args': self.args
                }, best_model_path)
                print(f"  New best model saved! Valid Acc: {valid_acc:.2f}%")
            
            print("-" * 60)
        
        # Load best model for final evaluation
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Final test evaluation
        test_acc = self.evaluate(self.test_loader, self.test_preprocessor, augment=False)
        print(f"\nFinal Results:")
        print(f"  Best Valid Acc: {best_valid_acc:.2f}%")
        print(f"  Test Acc: {test_acc:.2f}%")
        
        return test_acc

    def test_only(self, model_path: str):
        """Test pre-trained model"""
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_acc = self.evaluate(self.test_loader, self.test_preprocessor, augment=False)
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        return test_acc


def main():
    parser = argparse.ArgumentParser(description="BC-ResNet Keyword Spotting Training")
    
    # Dataset arguments
    parser.add_argument("--data_dir", default="./data", help="Data directory")
    parser.add_argument("--version", type=int, default=2, choices=[1, 2], 
                       help="Speech Commands version (1 or 2)")
    parser.add_argument("--download", action="store_true", 
                       help="Download dataset if not present")
    
    # Model arguments
    parser.add_argument("--tau", type=float, default=1.0, 
                       choices=[1, 1.5, 2, 3, 6, 8],
                       help="Model scale factor")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=200, 
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, 
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, 
                       help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4, 
                       help="Number of data loading workers")
    
    # System arguments
    parser.add_argument("--gpu", type=int, default=0, 
                       help="GPU device ID (-1 for CPU)")
    parser.add_argument("--save_dir", default="./checkpoints", 
                       help="Directory to save models")
    
    # Test arguments
    parser.add_argument("--test_only", action="store_true",
                       help="Only test pre-trained model")
    parser.add_argument("--model_path", type=str, 
                       help="Path to pre-trained model for testing")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = KeywordSpottingTrainer(args)
    
    if args.test_only:
        if not args.model_path:
            raise ValueError("--model_path required for test_only mode")
        trainer.test_only(args.model_path)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
