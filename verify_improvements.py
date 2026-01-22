import torch
import numpy as np
import sys
from pathlib import Path

# Add project dir to path
sys.path.append(str(Path(__file__).parent))

from realtime_detection_system import RealTimeDetector

def test_logic():
    # Mock config
    config = {
        'sample_rate': 22050,
        'duration': 3,
        'chunk_size': 1024,
        'n_mels': 128,
        'confidence_threshold': 0.85,
        'temporal_window': 2,
        'energy_threshold': 0.005,
        'cooldown_seconds': 5
    }
    
    class_names = ['background', 'forced_entry', 'glass_breaking', 'gun_shots', 'screams']
    model_path = r"D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models\best_model.pth"
    
    try:
        detector = RealTimeDetector(model_path, class_names, config)
        
        # 1. Test Silence (should be caught by energy threshold)
        print("\nTest 1: Silence")
        silence = np.zeros(22050 * 3)
        idx, name, conf, probs = detector.predict(silence)
        print(f"Result: {name} (Conf: {conf:.4f})")
        if name == 'background' and conf == 1.0:
            print("✅ Silence correctly classified as background via energy threshold.")
        else:
            print("❌ Silence failed energy threshold check.")
            
        # 2. Test Low Noise (should be caught by energy threshold or normalization fix)
        print("\nTest 2: Very low noise")
        low_noise = np.random.normal(0, 0.001, 22050 * 3)
        idx, name, conf, probs = detector.predict(low_noise)
        print(f"Result: {name} (Conf: {conf:.4f})")
        if name == 'background':
            print("✅ Low noise correctly classified as background.")
        else:
            print(f"⚠️ Low noise classified as {name}. This might still be a false positive.")

        # 3. Test Temporal Filter
        print("\nTest 3: Temporal Filter Logic")
        # Reset history
        detector.detection_history.clear()
        
        # Simulation: first detection is high confidence gunshot
        is_emerg, cls, conf = detector.temporal_filter(3, 0.9)
        print(f"First detection (Gunshot, 0.9): is_emergency={is_emerg}")
        
        # Second detection is also high confidence gunshot
        is_emerg, cls, conf = detector.temporal_filter(3, 0.95)
        print(f"Second detection (Gunshot, 0.95): is_emergency={is_emerg}")
        
        if is_emerg:
            print("✅ Temporal filter correctly triggered on 2 consecutive detections.")
        else:
            print("❌ Temporal filter failed to trigger on 2 consecutive detections.")

    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_logic()
