"""
Automated Dataset Download and Setup Script
Run this in Python to download and organize all datasets
"""

import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

class DatasetDownloader:
    def __init__(self, base_dir="datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        print(f"📁 Base directory: {self.base_dir.absolute()}")
    
    def run_command(self, command, description):
        """Run shell command safely"""
        print(f"\n🔄 {description}...")
        try:
            # Use shell=True for Windows compatibility
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True,
                cwd=self.base_dir
            )
            
            if result.returncode == 0:
                print(f"✅ {description} - Success!")
                return True
            else:
                print(f"❌ {description} - Failed!")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def download_file(self, url, destination, description):
        """Download file with progress"""
        print(f"\n⬇️  Downloading {description}...")
        try:
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100 / total_size, 100)
                print(f"\r  Progress: {percent:.1f}%", end='')
            
            urllib.request.urlretrieve(url, destination, progress_hook)
            print(f"\n✅ Downloaded: {destination}")
            return True
        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            return False
    
    def extract_zip(self, zip_path, extract_to, description):
        """Extract zip file"""
        print(f"\n📦 Extracting {description}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"✅ Extracted to: {extract_to}")
            
            # Remove zip file to save space
            os.remove(zip_path)
            print(f"🗑️  Removed zip file to save space")
            return True
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def download_esc50(self):
        """Download ESC-50 dataset"""
        print("\n" + "="*60)
        print("📥 DOWNLOADING ESC-50 DATASET")
        print("="*60)
        
        url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
        zip_path = self.base_dir / "ESC-50-master.zip"
        
        if (self.base_dir / "ESC-50-master").exists():
            print("✓ ESC-50 already exists. Skipping download.")
            return True
        
        if self.download_file(url, zip_path, "ESC-50"):
            return self.extract_zip(zip_path, self.base_dir, "ESC-50")
        return False
    
    def download_nonspeech7k(self):
        """Download Nonspeech7k dataset"""
        print("\n" + "="*60)
        print("📥 DOWNLOADING NONSPEECH7K DATASET")
        print("="*60)
        
        if (self.base_dir / "nonspeech7k").exists():
            print("✓ Nonspeech7k already exists. Skipping download.")
            return True
        
        # Check if git is installed
        git_check = subprocess.run(
            "git --version", 
            shell=True, 
            capture_output=True
        )
        
        if git_check.returncode != 0:
            print("❌ Git is not installed!")
            print("   Please install Git from: https://git-scm.com/downloads")
            print("   Or download manually from: https://github.com/0xnurl/nonspeech7k")
            return False
        
        command = "git clone https://github.com/0xnurl/nonspeech7k.git"
        return self.run_command(command, "Cloning Nonspeech7k")
    
    def check_urbansound8k(self):
        """Check if UrbanSound8K exists"""
        print("\n" + "="*60)
        print("📥 CHECKING URBANSOUND8K DATASET")
        print("="*60)
        
        urbansound_path = self.base_dir / "UrbanSound8K"
        
        if urbansound_path.exists():
            print("✅ UrbanSound8K found!")
            
            # Check if it has the expected structure
            audio_dir = urbansound_path / "audio"
            metadata_file = urbansound_path / "metadata" / "UrbanSound8K.csv"
            
            if audio_dir.exists() and metadata_file.exists():
                print("✅ Dataset structure looks correct")
                return True
            else:
                print("⚠️  Dataset found but structure seems incomplete")
                print(f"   Audio dir exists: {audio_dir.exists()}")
                print(f"   Metadata exists: {metadata_file.exists()}")
                return False
        else:
            print("❌ UrbanSound8K not found!")
            print("\n📋 Manual Download Required:")
            print("   1. Go to: https://www.kaggle.com/datasets/chrisfilo/urbansound8k")
            print("   2. Click 'Download' (requires Kaggle account)")
            print(f"   3. Extract the zip file to: {urbansound_path}")
            print("   4. Run this script again")
            return False
    
    def install_dependencies(self):
        """Install required Python packages"""
        print("\n" + "="*60)
        print("📦 INSTALLING PYTHON DEPENDENCIES")
        print("="*60)
        
        packages = [
            "librosa",
            "soundfile",
            "numpy",
            "pandas",
            "matplotlib",
            "scikit-learn",
            "tqdm"
        ]
        
        print("\nPackages to install:")
        for pkg in packages:
            print(f"  - {pkg}")
        
        response = input("\nInstall these packages? (y/n): ").lower()
        
        if response == 'y':
            command = f"pip install {' '.join(packages)}"
            return self.run_command(command, "Installing Python packages")
        else:
            print("⏭️  Skipped package installation")
            return True
    
    def create_directory_structure(self):
        """Create organized directory structure"""
        print("\n" + "="*60)
        print("📁 CREATING DIRECTORY STRUCTURE")
        print("="*60)
        
        organized_dir = Path("emergency_sound_dataset")
        
        classes = ['glass_breaking', 'scream', 'gunshot', 'forced_entry', 'background']
        
        for class_name in classes:
            class_dir = organized_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created: {class_dir}")
        
        print("\n✅ Directory structure ready!")
        return True
    
    def print_summary(self, results):
        """Print download summary"""
        print("\n" + "="*60)
        print("📊 DOWNLOAD SUMMARY")
        print("="*60)
        
        for dataset, status in results.items():
            status_icon = "✅" if status else "❌"
            print(f"  {status_icon} {dataset}")
        
        all_success = all(results.values())
        
        if all_success:
            print("\n🎉 All datasets are ready!")
            print("\n📋 Next Steps:")
            print("  1. Run the dataset organizer script to filter files")
            print("  2. Add custom 'forced_entry' sound recordings")
            print("  3. Start training your model!")
        else:
            print("\n⚠️  Some datasets are missing. Please download them manually.")
            print("    Check the instructions above.")


def main():
    """Main execution function"""
    print("="*60)
    print("🚀 EMERGENCY SOUND DETECTION - DATASET SETUP")
    print("="*60)
    
    downloader = DatasetDownloader(base_dir="datasets")
    
    results = {}
    
    # 1. Download ESC-50
    results['ESC-50'] = downloader.download_esc50()
    
    # 2. Download Nonspeech7k
    results['Nonspeech7k'] = downloader.download_nonspeech7k()
    
    # 3. Check UrbanSound8K (manual download required)
    results['UrbanSound8K'] = downloader.check_urbansound8k()
    
    # 4. Create directory structure
    downloader.create_directory_structure()
    
    # 5. Optional: Install dependencies
    print("\n")
    install = input("Would you like to install Python dependencies now? (y/n): ").lower()
    if install == 'y':
        downloader.install_dependencies()
    
    # 6. Print summary
    downloader.print_summary(results)
    
    print("\n" + "="*60)
    print("✨ Setup Complete!")
    print("="*60)


if __name__ == "__main__":
    main()