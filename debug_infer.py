import os
import torch
import numpy as np
import librosa
import torch.nn.functional as F
from realtime_detection_system import EmergencySoundCNN

MODEL = r'D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\models\best_model.pth'
CLIP = r'D:\DOCUMENTS\RAIN\AIML\Second_semester\Project\detection_logs\debug_clips\clip_20251223_141537_gun_shots_86.wav'
import glob

# If a specific clip path doesn't exist, try to find the latest saved debug clip anywhere under the workspace
if not os.path.exists(CLIP):
    candidates = glob.glob(r'D:\DOCUMENTS\RAIN\AIML\Second_semester\**\detection_logs\debug_clips\*.wav', recursive=True)
    candidates = sorted(candidates)
    if candidates:
        CLIP = candidates[-1]

print('clip exists:', os.path.exists(CLIP))
cp = torch.load(MODEL, map_location='cpu')
class_names = cp.get('class_names')
print('class names:', class_names)
model = EmergencySoundCNN(num_classes=len(class_names))
model.load_state_dict(cp['model_state_dict'])
model.eval()

sr = 22050
# load audio and preprocess like realtime
audio, _ = librosa.load(CLIP, sr=sr, duration=3.0)
# pad/trim
n_samples = sr * 3
if len(audio) < n_samples:
    audio = np.pad(audio, (0, n_samples - len(audio)))
else:
    audio = audio[:n_samples]
if np.max(np.abs(audio)) > 0:
    audio = audio / np.max(np.abs(audio))

mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000, hop_length=512)
mel_db = librosa.power_to_db(mel, ref=np.max)
mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
feat = torch.FloatTensor(mel_db).unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    out = model(feat)
    probs = F.softmax(out, dim=1).cpu().numpy()[0]

print('probs:')
for name, p in zip(class_names, probs):
    print(f'  {name}: {p:.6f} ({p*100:.2f}%)')
