"""
Complete CNN Training System for Emergency Sound Detection
Includes: Data splitting, training, validation, testing, and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import time

# ============================================================
# 1. DATASET CLASS
# ============================================================

class EmergencySoundDataset(Dataset):
    """PyTorch Dataset for emergency sounds"""
    
    def __init__(self, features, labels):
        """
        Args:
            features: numpy array of mel-spectrograms [N, H, W]
            labels: numpy array of class labels [N]
        """
        self.features = torch.FloatTensor(features).unsqueeze(1)  # Add channel dimension
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================
# 2. CNN MODEL ARCHITECTURE
# ============================================================

class EmergencySoundCNN(nn.Module):
    """
    CNN Architecture for Emergency Sound Classification
    Input: Mel-spectrogram (1, 128, ~260)
    Output: Class probabilities (4 classes)
    """
    
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(EmergencySoundCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.2)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.3)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(dropout_rate * 0.5)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # Convolutional blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)
        
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        
        return x


# ============================================================
# 3. TRAINING FUNCTIONS
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store predictions
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


# ============================================================
# 4. MAIN TRAINER CLASS
# ============================================================

class EmergencySoundTrainer:
    """Complete training system"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 class_names, device, save_dir='models'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train(self, num_epochs=50, lr=0.001, weight_decay=1e-4):
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization strength
        """
        print("="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning Rate: {lr}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples: {len(self.val_loader.dataset)}")
        print(f"  Test samples: {len(self.test_loader.dataset)}")
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6
        )
        
        # Early stopping
        patience = 15
        patience_counter = 0
        
        print("\n" + "="*60)
        print("Training Progress")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = train_epoch(
                self.model, self.train_loader, criterion, optimizer, self.device
            )
            
            # Validate
            val_loss, val_acc, _, _ = validate(
                self.model, self.val_loader, criterion, self.device
            )
            
            # Update scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                
                self.best_model_path = self.save_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_names': self.class_names,
                }, self.best_model_path)
                
                print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  Early stopping triggered (no improvement for {patience} epochs)")
                break
        
        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"Total time: {training_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best model saved to: {self.best_model_path}")
        
        # Save training history
        self.save_training_history()
        
        # Plot training curves
        self.plot_training_curves()
    
    def test(self):
        """Test the model on test set"""
        print("\n" + "="*60)
        print("🧪 TESTING MODEL")
        print("="*60)
        
        # Load best model
        if self.best_model_path and self.best_model_path.exists():
            checkpoint = torch.load(self.best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded best model from epoch {checkpoint['epoch']+1}")
        else:
            print("⚠️  Using current model weights (no saved best model found)")
        
        # Test
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc, all_preds, all_labels = validate(
            self.model, self.test_loader, criterion, self.device
        )
        
        print(f"\n📊 Test Results:")
        print(f"   Loss: {test_loss:.4f}")
        print(f"   Accuracy: {test_acc:.2f}%")
        
        # Confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds)
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        print("-" * 60)
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        # Save report
        report_dict = classification_report(
            all_labels, 
            all_preds, 
            target_names=self.class_names,
            output_dict=True
        )
        
        with open(self.save_dir / 'test_report.json', 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\n✓ Test report saved to: {self.save_dir / 'test_report.json'}")
        
        return test_acc, all_preds, all_labels
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Training curves saved to: {self.save_dir / 'training_curves.png'}")
        plt.close()
    
    def plot_confusion_matrix(self, true_labels, pred_labels):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, pred_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {self.save_dir / 'confusion_matrix.png'}")
        plt.close()
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to: {history_path}")


# ============================================================
# 5. MAIN EXECUTION
# ============================================================

def main():
    """Main training pipeline"""
    print("="*60)
    print("🎯 EMERGENCY SOUND DETECTION - CNN TRAINING")
    print("="*60)
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    DATA_PATH = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\processed_data\processed_data.npz"
    SAVE_DIR = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models"
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Train/Val/Test split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ============================================================
    # LOAD DATA
    # ============================================================
    
    print("\n📂 Loading processed data...")
    data = np.load(DATA_PATH)
    features = data['features']
    labels = data['labels']
    class_names = data['class_names'].tolist()
    
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Labels shape: {labels.shape}")
    print(f"✓ Classes: {class_names}")
    
    # ============================================================
    # CREATE DATASET AND SPLIT
    # ============================================================
    
    print("\n📊 Splitting dataset...")
    dataset = EmergencySoundDataset(features, labels)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Train: {len(train_dataset)} samples ({TRAIN_RATIO*100:.0f}%)")
    print(f"✓ Val:   {len(val_dataset)} samples ({VAL_RATIO*100:.0f}%)")
    print(f"✓ Test:  {len(test_dataset)} samples ({TEST_RATIO*100:.0f}%)")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # ============================================================
    # INITIALIZE MODEL
    # ============================================================
    
    print("\n🏗️  Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    model = EmergencySoundCNN(num_classes=len(class_names), dropout_rate=0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # ============================================================
    # TRAIN MODEL
    # ============================================================
    
    trainer = EmergencySoundTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        save_dir=SAVE_DIR
    )
    
    # Train
    trainer.train(num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Test
    trainer.test()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📂 All outputs saved to: {SAVE_DIR}")
    print("\n📋 Files created:")
    print("   • best_model.pth - Trained model weights")
    print("   • training_history.json - Training metrics")
    print("   • training_curves.png - Loss/accuracy plots")
    print("   • confusion_matrix.png - Confusion matrix")
    print("   • test_report.json - Detailed test results")
    print("\n🎉 Your model is ready to use!")


if __name__ == "__main__":
    main()
