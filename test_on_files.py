"""
Test trained model on individual audio files
Use this before real-time detection to verify model works
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
from pathlib import Path
import sys

# Import model architecture
class EmergencySoundCNN(torch.nn.Module):
    """Must match training architecture"""
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(EmergencySoundCNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2, 2)
        self.dropout1 = torch.nn.Dropout2d(0.1)
        
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2, 2)
        self.dropout2 = torch.nn.Dropout2d(0.2)
        
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(2, 2)
        self.dropout3 = torch.nn.Dropout2d(0.2)
        
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.relu4 = torch.nn.ReLU()
        self.pool4 = torch.nn.MaxPool2d(2, 2)
        self.dropout4 = torch.nn.Dropout2d(0.3)
        
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = torch.nn.Linear(256, 128)
        self.bn_fc1 = torch.nn.BatchNorm1d(128)
        self.relu_fc1 = torch.nn.ReLU()
        self.dropout_fc1 = torch.nn.Dropout(dropout_rate)
        
        self.fc2 = torch.nn.Linear(128, 64)
        self.bn_fc2 = torch.nn.BatchNorm1d(64)
        self.relu_fc2 = torch.nn.ReLU()
        self.dropout_fc2 = torch.nn.Dropout(dropout_rate * 0.5)
        
        self.fc3 = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout_fc1(self.relu_fc1(self.bn_fc1(self.fc1(x))))
        x = self.dropout_fc2(self.relu_fc2(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        return x


def load_model(model_path, num_classes, device):
    """Load trained model"""
    model = EmergencySoundCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_audio(audio_path, sr=22050, duration=3):
    """Preprocess audio file"""
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Pad or trim
    n_samples = sr * duration
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    else:
        audio = audio[:n_samples]
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return audio


def extract_features(audio, sr=22050, n_mels=128):
    """Extract mel-spectrogram"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmax=8000,
        hop_length=512
    )
    
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    features = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
    return features


def predict_audio_file(model, audio_path, class_names, device):
    """Predict class for audio file"""
    # Preprocess
    audio = preprocess_audio(audio_path)
    
    # Extract features
    features = extract_features(audio).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(features)
        probabilities = F.softmax(outputs, dim=1)
    
    # Get results
    all_probs = probabilities.cpu().numpy()[0]
    predicted_idx = np.argmax(all_probs)
    confidence = all_probs[predicted_idx]
    
    return predicted_idx, class_names[predicted_idx], confidence, all_probs


def test_single_file(model_path, audio_path, class_names):
    """Test model on a single audio file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("🧪 TESTING MODEL ON AUDIO FILE")
    print("="*70)
    print(f"\n📂 Model: {model_path}")
    print(f"🎵 Audio: {audio_path}")
    print(f"🖥️  Device: {device}")
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model(model_path, len(class_names), device)
    print("✅ Model loaded!")
    
    # Predict
    print("\n🔍 Analyzing audio...")
    idx, predicted_class, confidence, all_probs = predict_audio_file(
        model, audio_path, class_names, device
    )
    
    # Display results
    print("\n" + "="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70)
    print(f"\n🎯 Predicted Class: {predicted_class.upper()}")
    print(f"📈 Confidence: {confidence*100:.2f}%")
    print(f"\n📋 All Probabilities:")
    for name, prob in zip(class_names, all_probs):
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"   {name:15s} [{bar}] {prob*100:5.2f}%")
    
    print("\n" + "="*70)
    
    return predicted_class, confidence


def test_directory(model_path, directory, class_names):
    """Test model on all audio files in directory"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("🧪 TESTING MODEL ON DIRECTORY")
    print("="*70)
    print(f"\n📂 Model: {model_path}")
    print(f"📁 Directory: {directory}")
    
    # Load model
    print("\n📦 Loading model...")
    model = load_model(model_path, len(class_names), device)
    print("✅ Model loaded!")
    
    # Get all audio files
    directory = Path(directory)
    audio_files = list(directory.glob("*.wav")) + list(directory.glob("*.mp3"))
    
    if not audio_files:
        print(f"\n❌ No audio files found in {directory}")
        return
    
    print(f"\n🎵 Found {len(audio_files)} audio files")
    print("\n" + "="*70)
    
    results = []
    
    for audio_file in audio_files:
        print(f"\n📄 {audio_file.name}")
        
        try:
            idx, predicted_class, confidence, all_probs = predict_audio_file(
                model, audio_file, class_names, device
            )
            
            print(f"   Prediction: {predicted_class} ({confidence*100:.2f}%)")
            
            results.append({
                'file': audio_file.name,
                'predicted': predicted_class,
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    for result in results:
        status = "✅" if result['confidence'] > 0.7 else "⚠️"
        print(f"{status} {result['file']:40s} → {result['predicted']:15s} ({result['confidence']*100:.1f}%)")
    
    print("="*70)


def main():
    """Main function"""
    MODEL_PATH = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models\best_model.pth"
    
    # Load class names from the saved model checkpoint
    print("📦 Loading model to check class names...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    if 'class_names' in checkpoint:
        CLASS_NAMES = checkpoint['class_names']
        print(f"✅ Found {len(CLASS_NAMES)} classes in model: {CLASS_NAMES}")
    else:
        # Fallback - try to detect from model shape
        num_classes = checkpoint['model_state_dict']['fc3.bias'].shape[0]
        print(f"⚠️  Class names not in checkpoint. Detected {num_classes} classes.")
        
        # Make educated guess based on common patterns
        if num_classes == 4:
            CLASS_NAMES = ['background', 'glass_breaking', 'gun_shots', 'screams']
        elif num_classes == 5:
            CLASS_NAMES = ['background', 'glass_breaking', 'gun_shots', 'screams', 'forced_entry']
        else:
            CLASS_NAMES = [f'class_{i}' for i in range(num_classes)]
        
        print(f"   Using guessed class names: {CLASS_NAMES}")
        print(f"   ⚠️  These might be wrong! Check your training script.")
    
    print("="*70)
    print("🎯 EMERGENCY SOUND DETECTION - FILE TESTING")
    print("="*70)
    
    print("\nOptions:")
    print("  1. Test single audio file")
    print("  2. Test all files in a directory")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        audio_path = input("Enter path to audio file: ").strip()
        
        if not Path(audio_path).exists():
            print(f"\n❌ File not found: {audio_path}")
            return
        
        test_single_file(MODEL_PATH, audio_path, CLASS_NAMES)
        
    elif choice == '2':
        directory = input("Enter path to directory: ").strip()
        
        if not Path(directory).exists():
            print(f"\n❌ Directory not found: {directory}")
            return
        
        test_directory(MODEL_PATH, directory, CLASS_NAMES)
    
    else:
        print("\n❌ Invalid choice")


if __name__ == "__main__":
    main()