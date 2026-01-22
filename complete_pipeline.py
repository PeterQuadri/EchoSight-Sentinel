"""
Complete Data Pipeline: Preprocessing → Feature Extraction → Augmentation
This script does everything you need before training!
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import pickle
import json

class EmergencySoundPipeline:
    def __init__(self, input_dir, output_dir, sr=22050, duration=3, n_mels=128):
        """
        Complete pipeline for emergency sound data
        
        Args:
            input_dir: Path to your organized dataset
            output_dir: Path to save processed data
            sr: Sample rate (22050 Hz is standard)
            duration: Audio duration in seconds
            n_mels: Number of mel frequency bands
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_samples = sr * duration
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*60)
        print("🎵 EMERGENCY SOUND DATA PIPELINE")
        print("="*60)
        print(f"\n⚙️  Configuration:")
        print(f"   Sample Rate: {sr} Hz")
        print(f"   Duration: {duration} seconds")
        print(f"   Mel Bands: {n_mels}")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
    
    # ================================================================
    # STEP 1: PREPROCESSING
    # ================================================================
    
    def preprocess_audio(self, audio_path):
        """
        Preprocess a single audio file
        Steps:
        1. Load audio
        2. Resample to target sample rate
        3. Convert to mono if stereo
        4. Trim/pad to fixed length
        5. Normalize amplitude
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            
            # Trim silence from beginning and end
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Pad or trim to fixed length
            if len(audio) < self.n_samples:
                # Pad with zeros
                audio = np.pad(audio, (0, self.n_samples - len(audio)), mode='constant')
            else:
                # Trim to length
                audio = audio[:self.n_samples]
            
            # Normalize to [-1, 1]
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            print(f"\n⚠️  Error preprocessing {audio_path.name}: {e}")
            return None
    
    # ================================================================
    # STEP 2: FEATURE EXTRACTION
    # ================================================================
    
    def extract_features(self, audio):
        """
        Extract features from preprocessed audio
        We use mel-spectrograms - they work best for audio classification
        """
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            fmax=8000,
            hop_length=512
        )
        
        # Convert to decibels (log scale)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_db
    
    # ================================================================
    # STEP 3: DATA AUGMENTATION
    # ================================================================
    
    def augment_audio(self, audio, augmentation_type):
        """
        Apply augmentation to audio
        
        Augmentation types:
        - time_stretch: Change speed
        - pitch_shift: Change pitch
        - add_noise: Add random noise
        - time_shift: Shift in time
        - volume_change: Change volume
        """
        aug_audio = audio.copy()
        
        if augmentation_type == 'time_stretch':
            rate = np.random.uniform(0.85, 1.15)
            aug_audio = librosa.effects.time_stretch(aug_audio, rate=rate)
            # Ensure correct length
            if len(aug_audio) < self.n_samples:
                aug_audio = np.pad(aug_audio, (0, self.n_samples - len(aug_audio)))
            else:
                aug_audio = aug_audio[:self.n_samples]
        
        elif augmentation_type == 'pitch_shift':
            n_steps = np.random.uniform(-2, 2)
            aug_audio = librosa.effects.pitch_shift(aug_audio, sr=self.sr, n_steps=n_steps)
        
        elif augmentation_type == 'add_noise':
            noise_factor = np.random.uniform(0.001, 0.005)
            noise = np.random.randn(len(aug_audio)) * noise_factor
            aug_audio = aug_audio + noise
            aug_audio = np.clip(aug_audio, -1, 1)
        
        elif augmentation_type == 'time_shift':
            shift = np.random.randint(-self.sr // 4, self.sr // 4)
            aug_audio = np.roll(aug_audio, shift)
        
        elif augmentation_type == 'volume_change':
            factor = np.random.uniform(0.7, 1.3)
            aug_audio = aug_audio * factor
            aug_audio = np.clip(aug_audio, -1, 1)
        
        return aug_audio
    
    def decide_augmentation_count(self, class_name, current_count):
        """
        Decide how many augmentations to create based on current count
        """
        targets = {
            'glass_breaking': 300,
            'gun_shots': 400,
            'screams': 500,
            'background': 400
        }
        
        target = targets.get(class_name, 300)
        
        if current_count >= target:
            # Create minimal augmentations for diversity
            return 2
        else:
            # Calculate augmentations needed per file
            augs_per_file = max(1, (target - current_count) // current_count)
            return min(augs_per_file, 10)  # Cap at 10 to avoid too much duplication
    
    # ================================================================
    # MAIN PIPELINE
    # ================================================================
    
    def process_dataset(self, apply_augmentation=True):
        """
        Process entire dataset:
        1. Preprocess all audio files
        2. Extract features
        3. Apply augmentation (if enabled)
        4. Save processed data
        """
        print("\n" + "="*60)
        print("🚀 STARTING PIPELINE")
        print("="*60)
        
        # Get all class directories
        class_dirs = [d for d in self.input_dir.iterdir() if d.is_dir()]
        
        if not class_dirs:
            print(f"\n❌ No class directories found in {self.input_dir}")
            return
        
        all_data = []
        all_labels = []
        class_names = []
        stats = {}
        
        for class_idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            class_names.append(class_name)
            
            print(f"\n{'='*60}")
            print(f"📁 Processing Class: {class_name.upper()}")
            print(f"{'='*60}")
            
            # Get audio files
            audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
            
            if not audio_files:
                print(f"⚠️  No audio files found in {class_name}")
                continue
            
            print(f"   Found {len(audio_files)} original files")
            
            # Decide augmentation count
            if apply_augmentation:
                aug_count = self.decide_augmentation_count(class_name, len(audio_files))
                print(f"   Will create {aug_count} augmentations per file")
                total_expected = len(audio_files) * (1 + aug_count)
                print(f"   Expected total: {total_expected} files")
            else:
                aug_count = 0
                print(f"   Augmentation: DISABLED")
            
            processed_count = 0
            
            # Process each file
            for audio_file in tqdm(audio_files, desc=f"Processing {class_name}"):
                # 1. Preprocess
                audio = self.preprocess_audio(audio_file)
                if audio is None:
                    continue
                
                # 2. Extract features
                features = self.extract_features(audio)
                
                # 3. Save original
                all_data.append(features)
                all_labels.append(class_idx)
                processed_count += 1
                
                # 4. Apply augmentations
                if apply_augmentation and aug_count > 0:
                    aug_types = ['time_stretch', 'pitch_shift', 'add_noise', 'time_shift', 'volume_change']
                    
                    for i in range(aug_count):
                        # Randomly select augmentation type
                        aug_type = np.random.choice(aug_types)
                        
                        # Apply augmentation
                        aug_audio = self.augment_audio(audio, aug_type)
                        
                        # Extract features from augmented audio
                        aug_features = self.extract_features(aug_audio)
                        
                        # Save
                        all_data.append(aug_features)
                        all_labels.append(class_idx)
                        processed_count += 1
            
            stats[class_name] = processed_count
            print(f"   ✅ Processed {processed_count} total samples")
        
        # Convert to numpy arrays
        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        
        # Save processed data
        print("\n" + "="*60)
        print("💾 SAVING PROCESSED DATA")
        print("="*60)
        
        output_file = self.output_dir / "processed_data.npz"
        np.savez_compressed(
            output_file,
            features=all_data,
            labels=all_labels,
            class_names=class_names
        )
        
        print(f"\n✅ Saved processed data to: {output_file}")
        print(f"   Features shape: {all_data.shape}")
        print(f"   Labels shape: {all_labels.shape}")
        
        # Save metadata
        metadata = {
            'class_names': class_names,
            'num_classes': len(class_names),
            'sample_rate': self.sr,
            'duration': self.duration,
            'n_mels': self.n_mels,
            'stats': stats
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Metadata saved to: {metadata_file}")
        
        # Print final statistics
        print("\n" + "="*60)
        print("📊 FINAL DATASET STATISTICS")
        print("="*60)
        
        for class_name, count in stats.items():
            percentage = (count / len(all_labels)) * 100
            print(f"   {class_name:20s}: {count:5d} samples ({percentage:5.1f}%)")
        
        print(f"\n   {'TOTAL':20s}: {len(all_labels):5d} samples")
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETE!")
        print("="*60)
        print("\n🎉 Your data is ready for training!")
        print(f"📂 Location: {output_file}")
        print("\n💡 Next step: Run the training script")
        
        return all_data, all_labels, class_names


# ================================================================
# MAIN EXECUTION
# ================================================================

def main():
    """
    Main function - Configure and run the pipeline
    """
    print("="*60)
    print("🎯 EMERGENCY SOUND DETECTION - DATA PIPELINE")
    print("="*60)
    
    # ============================================================
    # 🔧 CONFIGURE YOUR PATHS HERE
    # ============================================================
    
    INPUT_DIR = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\emergency_sound_dataset"
    OUTPUT_DIR = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\processed_data"
    
    # ============================================================
    
    # Check if input directory exists
    if not Path(INPUT_DIR).exists():
        print(f"\n❌ Input directory not found: {INPUT_DIR}")
        print("   Please check the path and try again.")
        return
    
    # Ask about augmentation
    print("\n" + "="*60)
    print("⚙️  CONFIGURATION")
    print("="*60)
    
    print("\n❓ Data Augmentation Options:")
    print("   1. Enable augmentation (RECOMMENDED)")
    print("      - Expands small classes (glass_breaking)")
    print("      - Improves model robustness")
    print("      - Takes longer (~15-20 minutes)")
    print("\n   2. Disable augmentation")
    print("      - Faster processing (~5 minutes)")
    print("      - May result in poor performance on small classes")
    
    choice = input("\nEnable data augmentation? (y/n): ").lower()
    apply_augmentation = (choice == 'y')
    
    if not apply_augmentation:
        print("\n⚠️  WARNING: Augmentation disabled!")
        print("   Your model may perform poorly on classes with few samples.")
        confirm = input("   Are you sure? (y/n): ").lower()
        if confirm != 'y':
            apply_augmentation = True
            print("   ✓ Augmentation re-enabled")
    
    # Initialize pipeline
    pipeline = EmergencySoundPipeline(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        sr=22050,
        duration=3,
        n_mels=128
    )
    
    # Run pipeline
    try:
        pipeline.process_dataset(apply_augmentation=apply_augmentation)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
