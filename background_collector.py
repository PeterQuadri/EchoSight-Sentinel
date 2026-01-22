"""
Collect background/normal sounds from ESC-50 dataset
These are essential for training - without them, your model will have constant false alarms!
"""

import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import random
def collect_background_sounds_from_esc50(esc50_path, output_dir):
    """
    Extract suitable background sounds from ESC-50 dataset
    """
    print("="*60)
    print(" COLLECTING BACKGROUND SOUNDS FROM ESC-50")
    print("="*60)
    
    esc50_path = Path(esc50_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
      
    
    # Read metadata
    metadata_file = esc50_path / "meta" / "esc50.csv"
    
    if not metadata_file.exists():
        print(f" Metadata not found: {metadata_file}")
        print("   Make sure ESC-50 is properly extracted")
        return 0
    
    df = pd.read_csv(metadata_file)
    
    # Define background/normal sound categories
    # These are everyday sounds that should NOT trigger alarms
    background_categories = [
        # Human non-emergency sounds
        'breathing',
        'footsteps',
        'laughing',
        'coughing',
        'sneezing',
        'brushing_teeth',
        'drinking_sipping',
        
        # Household sounds
        # 'door_wood_knock',     <-- REMOVED: Mimics gunshots
        # 'door_wood_creaks',    <-- REMOVED: Mimics gunshots
        # 'mouse_click',         <-- REMOVED: Mimics gunshots
        # 'keyboard_typing',     <-- REMOVED: Mimics gunshots
        # 'clock_tick',          <-- REMOVED: Mimics gunshots
        # 'clock_alarm',         <-- REMOVED: Mimics gunshots
        # 'can_opening',         <-- REMOVED: Mimics gunshots
        'washing_machine',
        'vacuum_cleaner',
        
        # Nature sounds
        'rain',
        'wind',
        'crickets',
        'chirping_birds',
        'sea_waves',
        'crackling_fire',
        'pouring_water',
        # 'water_drops',
        
        # Urban sounds (non-emergency)
        'car_horn',
        'engine',
        'train',
        'airplane',
        'church_bells',
        
        # Animal sounds
        'dog',
        'cat',
        'cow',
        'rooster',
        'frog',
        'hen',
        'crow'
    ]
    
    print(f"\n🔍 Looking for these categories:")
    for cat in background_categories[:10]:
        print(f"   • {cat}")
    print(f"   ... and {len(background_categories) - 10} more")
    
    # Filter for background sounds
    background_samples = df[df['category'].isin(background_categories)]
    
    print(f"\n📊 Found {len(background_samples)} background sound samples")
    
    audio_dir = esc50_path / "audio"
    copied_count = 0
    
    # Copy files
    for idx, row in tqdm(background_samples.iterrows(), 
                         total=len(background_samples),
                         desc="Copying files"):
        src_file = audio_dir / row['filename']
        
        if src_file.exists():
            # Add category prefix to filename for organization
            new_filename = f"{row['category']}_{row['filename']}"
            dst_file = output_dir / new_filename
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        else:
            print(f"\n⚠️  File not found: {src_file}")
    
    print(f"\n✅ Copied {copied_count} background sound files")
    print(f"📂 Location: {output_dir.absolute()}")
    
    return copied_count


def collect_from_urbansound8k(urbansound_path, output_dir):
    """
    Extract non-emergency sounds from UrbanSound8K
    """
    print("\n" + "="*60)
    print("📦 COLLECTING BACKGROUND SOUNDS FROM URBANSOUND8K")
    print("="*60)
    
    urbansound_path = Path(urbansound_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read metadata
    metadata_file = urbansound_path / "metadata" / "UrbanSound8K.csv"
    
    if not metadata_file.exists():
        print(f"❌ Metadata not found: {metadata_file}")
        return 0
    
    df = pd.read_csv(metadata_file)
    
    # UrbanSound8K class IDs (we want everything EXCEPT gunshots)
    # 0 = air_conditioner, 1 = car_horn, 2 = children_playing, 3 = dog_bark
    # 4 = drilling, 5 = engine_idling, 6 = gun_shot (EXCLUDE THIS)
    # 7 = jackhammer, 8 = siren, 9 = street_music
    
    # Background sounds (not emergency)
    background_class_ids = [0, 1, 2, 3, 5, 9]  # Exclude 6 (gunshot), 8 (siren), 4 (drilling), 7 (jackhammer)
    
    class_names = {
        0: 'air_conditioner',
        1: 'car_horn',
        2: 'children_playing',
        3: 'dog_bark',
        # 4: 'drilling',      <--- REMOVED
        5: 'engine_idling',
        # 7: 'jackhammer',    <--- REMOVED
        9: 'street_music'
    }
    
    print(f"\n🔍 Collecting these urban sounds:")
    for class_id, name in class_names.items():
        print(f"   • {name}")
    
    # Filter for background sounds
    background_samples = df[df['classID'].isin(background_class_ids)]
    
    print(f"\n📊 Found {len(background_samples)} background sound samples")
    
    # Limit to avoid overwhelming the dataset
    max_samples = 200
    if len(background_samples) > max_samples:
        background_samples = background_samples.sample(n=max_samples, random_state=42)
        print(f"   (Randomly selecting {max_samples} samples for balance)")
    
    copied_count = 0
    
    # Copy files
    for idx, row in tqdm(background_samples.iterrows(), 
                         total=len(background_samples),
                         desc="Copying files"):
        fold_dir = urbansound_path / "audio" / f"fold{row['fold']}"
        src_file = fold_dir / row['slice_file_name']
        
        if src_file.exists():
            # Add class name prefix
            class_name = class_names.get(row['classID'], 'unknown')
            new_filename = f"urban_{class_name}_{row['slice_file_name']}"
            dst_file = output_dir / new_filename
            shutil.copy2(src_file, dst_file)
            copied_count += 1
    
    print(f"\n✅ Copied {copied_count} background sound files")
    print(f"📂 Location: {output_dir.absolute()}")
    
    return copied_count


def main():
    """Collect all background sounds"""
    print("="*60)
    print("🎯 BACKGROUND SOUND COLLECTION")
    print("="*60)
    
    print("\n⚠️  IMPORTANT: Background sounds are ESSENTIAL!")
    print("   Without them, your model will:")
    print("   • Have constant false alarms")
    print("   • Classify any sound as an emergency")
    print("   • Be unusable in real-world scenarios")
    
    # ============================================================
    # 🔧 CONFIGURE YOUR PATHS HERE
    # ============================================================
    
    # Path to ESC-50 dataset folder
    # Change this to match your actual path
    esc50_path = Path(r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\datasets\ESC-50-master")
    
    # Path to UrbanSound8K dataset folder
    # Change this to match your actual path  
    urbansound_path = Path(r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\datasets\UrbanSound8k")
    
    # Output directory for background sounds
    output_dir = Path(r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\emergency_sound_dataset\background")
    
    # ============================================================
    
    total_collected = 0
    
    # Collect from ESC-50
    if output_dir.exists():
        print(f"\n🧹 Clearing old background sounds in: {output_dir}")
        for f in output_dir.glob("*.wav"):
            try:
                os.remove(f)
            except:
                pass
        print("✅ Directory cleaned")

    if esc50_path.exists():
        count = collect_background_sounds_from_esc50(esc50_path, output_dir)
        total_collected += count
    else:
        print(f"\n⚠️  ESC-50 not found at: {esc50_path}")
        print(f"   Please check the path and try again")
    
    # Collect from UrbanSound8K
    if urbansound_path.exists():
        count = collect_from_urbansound8k(urbansound_path, output_dir)
        total_collected += count
    else:
        print(f"\n⚠️  UrbanSound8K not found at: {urbansound_path}")
        print(f"   Please check the path and try again")
    
    all_files = list(output_dir.glob("*.wav"))
    if len(all_files) > 600:
        print(f"\n⚙️  Reducing dataset size to 600 samples for balance...")
        keep_files = set(random.sample(all_files, 600))
        for f in all_files:
            if f not in keep_files:
                os.remove(f)
        print(f"✅ Reduced background sounds to 600 samples")

    # Summary
    print("\n" + "="*60)
    print("📊 COLLECTION SUMMARY")
    print("="*60)
    print(f"   Total background sounds collected: {total_collected}")
    print(f"   Location: {output_dir.absolute()}")
    
    if total_collected >= 300:
        print("\n✅ Excellent! You have enough background sounds")
    elif total_collected >= 200:
        print("\n⚠️  You have some background sounds, but more would be better")
        print("   Consider using data augmentation to reach 300+")
    else:
        print("\n❌ Not enough background sounds!")
        print("   You need at least 200-300 for good training")
    
    print("="*60)


if __name__ == "__main__":
    main()