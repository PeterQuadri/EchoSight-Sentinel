"""
Data Augmentation for Emergency Sound Detection
This will help expand your small datasets, especially glass breaking
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random

class AudioAugmenter:
    def __init__(self, sr=22050):
        self.sr = sr
    
    def time_stretch(self, audio, rate):
        """Change speed of audio without changing pitch"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, n_steps):
        """Change pitch of audio"""
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def add_noise(self, audio, noise_factor):
        """Add random noise to audio"""
        noise = np.random.randn(len(audio))
        augmented = audio + noise_factor * noise
        # Normalize to prevent clipping
        augmented = augmented / np.max(np.abs(augmented))
        return augmented
    
    def change_volume(self, audio, factor):
        """Change volume of audio"""
        augmented = audio * factor
        # Prevent clipping
        augmented = np.clip(augmented, -1.0, 1.0)
        return augmented
    
    def time_shift(self, audio, shift_max):
        """Shift audio in time"""
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(audio, shift)
    
    def add_background_noise(self, audio, background_audio, snr_db):
        """Add background noise at specific SNR (Signal-to-Noise Ratio)"""
        # Calculate signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(background_audio ** 2)
        
        # Calculate required noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        required_noise_power = signal_power / snr_linear
        
        # Scale noise
        scaling_factor = np.sqrt(required_noise_power / noise_power)
        scaled_noise = background_audio * scaling_factor
        
        # Mix signal and noise
        augmented = audio + scaled_noise[:len(audio)]
        
        # Normalize
        augmented = augmented / np.max(np.abs(augmented))
        return augmented
    
    def augment_single_file(self, input_path, output_dir, num_augmentations=5):
        """
        Create multiple augmented versions of a single audio file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original audio
        audio, sr = librosa.load(input_path, sr=self.sr)
        
        input_path = Path(input_path)
        base_name = input_path.stem
        
        augmented_files = []
        
        # Save original
        original_path = output_dir / f"{base_name}_original.wav"
        sf.write(original_path, audio, sr)
        augmented_files.append(original_path)
        
        # Generate augmentations
        for i in range(num_augmentations):
            aug_audio = audio.copy()
            aug_name = f"{base_name}_aug{i+1}"
            
            # Randomly apply augmentations
            augmentations_applied = []
            
            # 1. Time stretch (80% to 120% speed)
            if random.random() > 0.5:
                rate = random.uniform(0.8, 1.2)
                aug_audio = self.time_stretch(aug_audio, rate)
                augmentations_applied.append(f"stretch{rate:.2f}")
            
            # 2. Pitch shift (±2 semitones)
            if random.random() > 0.5:
                n_steps = random.uniform(-2, 2)
                aug_audio = self.pitch_shift(aug_audio, n_steps)
                augmentations_applied.append(f"pitch{n_steps:.1f}")
            
            # 3. Add noise (light)
            if random.random() > 0.5:
                noise_factor = random.uniform(0.001, 0.005)
                aug_audio = self.add_noise(aug_audio, noise_factor)
                augmentations_applied.append("noise")
            
            # 4. Volume change (70% to 130%)
            if random.random() > 0.5:
                volume_factor = random.uniform(0.7, 1.3)
                aug_audio = self.change_volume(aug_audio, volume_factor)
                augmentations_applied.append(f"vol{volume_factor:.2f}")
            
            # 5. Time shift
            if random.random() > 0.3:
                shift_max = int(self.sr * 0.2)  # Max 0.2 second shift
                aug_audio = self.time_shift(aug_audio, shift_max)
                augmentations_applied.append("shift")
            
            # Save augmented file
            aug_filename = f"{aug_name}_{'_'.join(augmentations_applied[:2])}.wav"
            aug_path = output_dir / aug_filename
            sf.write(aug_path, aug_audio, sr)
            augmented_files.append(aug_path)
        
        return augmented_files
    
    def augment_dataset(self, input_dir, output_dir, num_augmentations_per_file=5, target_count=None):
        """
        Augment entire dataset
        
        Args:
            input_dir: Directory containing original audio files
            output_dir: Directory to save augmented files
            num_augmentations_per_file: Number of augmented versions per original file
            target_count: Target number of samples (will calculate augmentations needed)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all audio files
        audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3"))
        
        if len(audio_files) == 0:
            print(f"❌ No audio files found in {input_dir}")
            return
        
        print(f"\n📁 Processing: {input_dir.name}")
        print(f"   Original files: {len(audio_files)}")
        
        # Calculate augmentations needed if target specified
        if target_count:
            total_files_needed = target_count - len(audio_files)
            if total_files_needed > 0:
                num_augmentations_per_file = int(np.ceil(total_files_needed / len(audio_files)))
                print(f"   Target: {target_count} files")
                print(f"   Will create {num_augmentations_per_file} augmentations per file")
            else:
                print(f"   Already have {len(audio_files)} files (target: {target_count})")
                print(f"   Creating {num_augmentations_per_file} augmentations anyway for diversity")
        
        # Process each file
        total_generated = 0
        
        for audio_file in tqdm(audio_files, desc="Augmenting files"):
            try:
                augmented = self.augment_single_file(
                    audio_file, 
                    output_dir, 
                    num_augmentations=num_augmentations_per_file
                )
                total_generated += len(augmented)
            except Exception as e:
                print(f"\n⚠️  Error processing {audio_file.name}: {e}")
        
        print(f"   ✅ Generated {total_generated} total files")
        print(f"   📂 Saved to: {output_dir.absolute()}")
        
        return total_generated


def augment_all_classes(base_dir=None, output_dir=None):
    """
    Augment all classes in your dataset
    """
    print("="*60)
    print("🎵 DATA AUGMENTATION FOR EMERGENCY SOUNDS")
    print("="*60)
    
    augmenter = AudioAugmenter(sr=22050)
    
    # ============================================================
    # 🔧 CONFIGURE YOUR PATHS HERE (if not provided)
    # ============================================================
    if base_dir is None:
        base_dir = Path(r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\emergency_sound_dataset")
    else:
        base_dir = Path(base_dir)
    
    if output_dir is None:
        output_dir = Path(r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\augmented_dataset")
    else:
        output_dir = Path(output_dir)
    # ============================================================
    
    # Define augmentation strategy for each class
    augmentation_config = {
    'glass_breaking': {'target': 300, 'priority': 'HIGH'},
    'gun_shots': {'target': 400, 'priority': 'LOW'},   # Matches your folder
    'screams': {'target': 600, 'priority': 'LOW'},     # Matches your folder
    'background': {'target': 500, 'priority': 'HIGH'}
}

    
    results = {}
    
    for class_name, config in augmentation_config.items():
        input_class_dir = base_dir / class_name
        output_class_dir = output_dir / class_name
        
        if not input_class_dir.exists():
            print(f"\n⚠️  {class_name} directory not found, skipping...")
            continue
        
        # Count original files
        original_count = len(list(input_class_dir.glob("*.wav")) + 
                              list(input_class_dir.glob("*.mp3")))
        
        if original_count == 0:
            print(f"\n⚠️  No files in {class_name}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"📊 Class: {class_name.upper()}")
        print(f"   Priority: {config['priority']}")
        print(f"   Current: {original_count} files")
        print(f"   Target: {config['target']} files")
        
        if config['priority'] == 'HIGH' or original_count < config['target']:
            total = augmenter.augment_dataset(
                input_class_dir,
                output_class_dir,
                target_count=config['target']
            )
            results[class_name] = total
        else:
            print(f"   ✓ Sufficient data, creating minimal augmentations for diversity")
            total = augmenter.augment_dataset(
                input_class_dir,
                output_class_dir,
                num_augmentations_per_file=2
            )
            results[class_name] = total
    
    # Print summary
    print("\n" + "="*60)
    print("📊 AUGMENTATION SUMMARY")
    print("="*60)
    
    for class_name, count in results.items():
        print(f"  {class_name}: {count} total files")
    
    print("\n✅ Augmentation complete!")
    print(f"📂 Augmented dataset location: {output_dir.absolute()}")
    print("\n💡 Next step: Use the augmented_dataset folder for training")


if __name__ == "__main__":
    print("🎯 This script will augment your dataset to improve training")
    print("\n⚙️  Configuration:")
    print("   - Glass breaking: Will expand to ~300 samples")
    print("   - Gun shots: Will create some augmentations for diversity")
    print("   - Screams: Will create some augmentations for diversity")
    print("   - Background: Will expand to ~500 samples")

    response = input("\n▶️  Start augmentation? (y/n): ").lower()

    if response == 'y':
        augment_all_classes(
            base_dir=r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\emergency_sound_dataset",
            output_dir=r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\augmented_dataset"
        )
    else:
        print("\n❌ Augmentation cancelled")
