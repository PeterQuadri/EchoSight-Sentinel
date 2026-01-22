"""
Check what's inside your trained model
Run this first to see what classes your model was trained with
"""

import torch
from pathlib import Path

def check_model(model_path):
    """Check model information"""
    print("="*70)
    print("🔍 MODEL INFORMATION CHECKER")
    print("="*70)
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"\n❌ Model not found: {model_path}")
        return
    
    print(f"\n📂 Model path: {model_path}")
    
    # Load checkpoint
    print("\n📦 Loading checkpoint...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    print("✅ Checkpoint loaded!")
    
    # Check what's in the checkpoint
    print("\n" + "="*70)
    print("📋 CHECKPOINT CONTENTS")
    print("="*70)
    
    for key in checkpoint.keys():
        print(f"  ✓ {key}")
    
    # Extract information
    print("\n" + "="*70)
    print("📊 MODEL DETAILS")
    print("="*70)
    
    # Number of classes
    if 'model_state_dict' in checkpoint:
        fc3_bias_shape = checkpoint['model_state_dict']['fc3.bias'].shape
        num_classes = fc3_bias_shape[0]
        print(f"\n🎯 Number of classes: {num_classes}")
    
    # Class names
    if 'class_names' in checkpoint:
        class_names = checkpoint['class_names']
        print(f"\n📝 Class names found in checkpoint:")
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")
    else:
        print(f"\n⚠️  No class names stored in checkpoint")
        print(f"   The model has {num_classes} output classes but names weren't saved")
        print(f"\n💡 Common configurations:")
        if num_classes == 4:
            print("   Likely: ['background', 'glass_breaking', 'gun_shots', 'screams']")
        elif num_classes == 5:
            print("   Likely: ['background', 'glass_breaking', 'gun_shots', 'screams', 'forced_entry']")
    
    # Training info
    if 'epoch' in checkpoint:
        print(f"\n📈 Training info:")
        print(f"   Epoch: {checkpoint['epoch'] + 1}")
        if 'val_acc' in checkpoint:
            print(f"   Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        if 'val_loss' in checkpoint:
            print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
    
    # Model architecture info
    print("\n" + "="*70)
    print("🏗️  MODEL ARCHITECTURE")
    print("="*70)
    
    state_dict = checkpoint['model_state_dict']
    
    # Count layers
    conv_layers = [k for k in state_dict.keys() if 'conv' in k and 'weight' in k]
    fc_layers = [k for k in state_dict.keys() if 'fc' in k and 'weight' in k]
    
    print(f"\n  Convolutional layers: {len(conv_layers)}")
    print(f"  Fully connected layers: {len(fc_layers)}")
    
    # Total parameters
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"  Total parameters: {total_params:,}")
    
    # Input/Output shapes
    print(f"\n  Input: 1 channel (grayscale spectrogram)")
    print(f"  Output: {num_classes} classes")
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    
    # Recommendations
    print("\n💡 WHAT TO DO NEXT:")
    print("\n1. If class names are missing:")
    print("   - The updated test scripts will auto-detect the number of classes")
    print("   - But you should verify the class order matches your training data")
    
    print("\n2. To use the model:")
    print("   - Run: python test_on_files.py")
    print("   - The script will automatically detect the correct number of classes")
    
    print("\n3. If you see unexpected number of classes:")
    print("   - Check your training data preprocessing")
    print("   - Make sure all class folders were included")
    
    print("\n" + "="*70)
    
    return checkpoint


def main():
    """Main function"""
    MODEL_PATH = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models\best_model.pth"
    
    # Allow custom path
    import sys
    if len(sys.argv) > 1:
        MODEL_PATH = sys.argv[1]
    
    check_model(MODEL_PATH)


if __name__ == "__main__":
    main()
