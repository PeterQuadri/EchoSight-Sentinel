"""
Dataset Organizer for Emergency Sound Detection
Filters and organizes audio files from multiple datasets into a unified structure
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from collections import defaultdict

class DatasetOrganizer:
    def __init__(self, output_dir="organized_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define your target classes
        self.target_classes = {
            'glass_breaking': self.output_dir / 'glass_breaking',
            'scream': self.output_dir / 'scream',
            'gunshot': self.output_dir / 'gunshot',
            'forced_entry': self.output_dir / 'forced_entry',
            'background': self.output_dir / 'background'
        }
        
        # Create directories
        for class_dir in self.target_classes.values():
            class_dir.mkdir(exist_ok=True)
        
        self.stats = defaultdict(int)
    
    def organize_esc50(self, esc50_path):
        """
        Organize ESC-50 dataset
        Extract only glass breaking samples
        """
        print("\n📁 Processing ESC-50...")
        esc50_path = Path(esc50_path)
        
        # Read metadata
        metadata_file = esc50_path / "meta" / "esc50.csv"
        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return
        
        df = pd.read_csv(metadata_file)
        
        # Filter for glass breaking (target = 37 in ESC-50)
        # Check the actual target number in your esc50.csv file!
        glass_breaking_samples = df[df['category'] == 'glass_breaking']
        
        audio_dir = esc50_path / "audio"
        
        for idx, row in glass_breaking_samples.iterrows():
            src_file = audio_dir / row['filename']
            if src_file.exists():
                dst_file = self.target_classes['glass_breaking'] / row['filename']
                shutil.copy2(src_file, dst_file)
                self.stats['glass_breaking'] += 1
                print(f"  ✓ Copied: {row['filename']}")
            else:
                print(f"  ⚠️ File not found: {src_file}")
        
        print(f"✅ ESC-50: Copied {self.stats['glass_breaking']} glass breaking samples")
    
    def organize_urbansound8k(self, urbansound_path):
        """
        Organize UrbanSound8K dataset
        Extract only gunshot samples (class_id = 6)
        """
        print("\n📁 Processing UrbanSound8K...")
        urbansound_path = Path(urbansound_path)
        
        # Read metadata
        metadata_file = urbansound_path / "metadata" / "UrbanSound8K.csv"
        if not metadata_file.exists():
            print(f"❌ Metadata file not found: {metadata_file}")
            return
        
        df = pd.read_csv(metadata_file)
        
        # Filter for gunshots (classID = 6)
        gunshot_samples = df[df['classID'] == 6]
        
        for idx, row in gunshot_samples.iterrows():
            # Files are organized in fold directories
            fold_dir = urbansound_path / "audio" / f"fold{row['fold']}"
            src_file = fold_dir / row['slice_file_name']
            
            if src_file.exists():
                # Rename to include fold info to avoid duplicates
                new_name = f"fold{row['fold']}_{row['slice_file_name']}"
                dst_file = self.target_classes['gunshot'] / new_name
                shutil.copy2(src_file, dst_file)
                self.stats['gunshot'] += 1
                if self.stats['gunshot'] % 50 == 0:
                    print(f"  ✓ Processed {self.stats['gunshot']} gunshot samples...")
            else:
                print(f"  ⚠️ File not found: {src_file}")
        
        print(f"✅ UrbanSound8K: Copied {self.stats['gunshot']} gunshot samples")
    
    def organize_nonspeech7k(self, nonspeech_path):
        """
        Organize Nonspeech7k dataset
        Extract only scream samples
        """
        print("\n📁 Processing Nonspeech7k...")
        nonspeech_path = Path(nonspeech_path)
        
        # Assuming structure: nonspeech7k/scream/*.wav
        scream_dir = nonspeech_path / "scream"
        
        if not scream_dir.exists():
            # Try alternative structure
            scream_dir = nonspeech_path / "screaming"
        
        if not scream_dir.exists():
            print(f"❌ Scream directory not found in {nonspeech_path}")
            print("   Please check the dataset structure")
            return
        
        scream_files = list(scream_dir.glob("*.wav")) + list(scream_dir.glob("*.mp3"))
        
        for src_file in scream_files:
            dst_file = self.target_classes['scream'] / src_file.name
            shutil.copy2(src_file, dst_file)
            self.stats['scream'] += 1
        
        print(f"✅ Nonspeech7k: Copied {self.stats['scream']} scream samples")
    
    def add_background_noise(self, esc50_path=None, custom_paths=None):
        """
        Add background noise samples from various sources
        """
        print("\n📁 Adding background noise samples...")
        
        if esc50_path:
            esc50_path = Path(esc50_path)
            metadata_file = esc50_path / "meta" / "esc50.csv"
            
            if metadata_file.exists():
                df = pd.read_csv(metadata_file)
                
                # Select neutral background sounds
                background_categories = [
                    'rain', 'wind', 'crickets', 'footsteps', 
                    'keyboard_typing', 'mouse_click', 'breathing'
                ]
                
                background_samples = df[df['category'].isin(background_categories)]
                audio_dir = esc50_path / "audio"
                
                for idx, row in background_samples.iterrows():
                    src_file = audio_dir / row['filename']
                    if src_file.exists():
                        dst_file = self.target_classes['background'] / row['filename']
                        shutil.copy2(src_file, dst_file)
                        self.stats['background'] += 1
        
        # Add custom background noise if provided
        if custom_paths:
            for path in custom_paths:
                path = Path(path)
                if path.is_dir():
                    for audio_file in path.glob("*.wav"):
                        dst_file = self.target_classes['background'] / audio_file.name
                        shutil.copy2(audio_file, dst_file)
                        self.stats['background'] += 1
        
        print(f"✅ Added {self.stats['background']} background noise samples")
    
    def print_summary(self):
        """Print organization summary"""
        print("\n" + "="*60)
        print("📊 DATASET ORGANIZATION SUMMARY")
        print("="*60)
        
        total = 0
        for class_name, count in self.stats.items():
            print(f"  {class_name.replace('_', ' ').title()}: {count} samples")
            total += count
        
        print(f"\n  Total: {total} samples")
        print(f"  Output directory: {self.output_dir.absolute()}")
        print("="*60)
        
        # Check for class imbalance
        if self.stats:
            min_samples = min(self.stats.values())
            max_samples = max(self.stats.values())
            
            if max_samples > min_samples * 3:
                print("\n⚠️  WARNING: Significant class imbalance detected!")
                print("   Consider data augmentation for underrepresented classes")
    
    def verify_audio_files(self):
        """Verify all copied audio files are valid"""
        print("\n🔍 Verifying audio files...")
        
        try:
            import librosa
            
            for class_name, class_dir in self.target_classes.items():
                audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
                
                corrupted = 0
                for audio_file in audio_files[:5]:  # Check first 5 files
                    try:
                        _, sr = librosa.load(audio_file, sr=None, duration=1)
                    except Exception as e:
                        print(f"  ❌ Corrupted file: {audio_file}")
                        corrupted += 1
                
                if corrupted == 0:
                    print(f"  ✓ {class_name}: Sample files OK")
        
        except ImportError:
            print("  ⚠️ Librosa not installed. Skipping verification.")
            print("  Install with: pip install librosa")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize organizer
    organizer = DatasetOrganizer(output_dir="emergency_sound_dataset")
    
    # Organize datasets (uncomment and modify paths as needed)
    
    # 1. Process ESC-50 for glass breaking
    organizer.organize_esc50("path/to/ESC-50-master")
    
    # 2. Process UrbanSound8K for gunshots
    organizer.organize_urbansound8k("path/to/UrbanSound8K")
    
    # 3. Process Nonspeech7k for screams
    organizer.organize_nonspeech7k("path/to/nonspeech7k")
    
    # 4. Add background noise
    organizer.add_background_noise(
        esc50_path="path/to/ESC-50-master",
        custom_paths=["path/to/custom/background/sounds"]
    )
    
    # 5. Print summary
    organizer.print_summary()
    
    # 6. Verify files
    organizer.verify_audio_files()
    
    print("\n✨ Dataset organization complete!")
    print(f"📂 Your organized dataset is in: emergency_sound_dataset/")
    print("\nNext steps:")
    print("  1. Review the organized files")
    print("  2. Add custom recordings for forced_entry class")
    print("  3. Apply data augmentation if needed")
    print("  4. Start training your model!")