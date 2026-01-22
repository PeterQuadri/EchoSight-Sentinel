import glob, os
import numpy as np
import librosa

candidates = glob.glob(r'D:\DOCUMENTS\RAIN\AIML\Second_semester\**\detection_logs\debug_clips\*.wav', recursive=True)
if not candidates:
    print('No debug clips found')
    raise SystemExit(1)
clip = sorted(candidates)[-1]
print('Using clip:', clip)

sr = 22050
audio, _ = librosa.load(clip, sr=sr)
print('duration (s):', len(audio)/sr)
print('samples:', len(audio))
print('dtype:', audio.dtype)
print('min,max:', audio.min(), audio.max())
rms = np.sqrt(np.mean(audio**2))
print('RMS:', rms)
print('mean abs:', np.mean(np.abs(audio)))
print('zero crossings:', librosa.feature.zero_crossing_rate(audio, frame_length=2048, hop_length=512).mean())
centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).mean()
bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr).mean()
print('spectral centroid (Hz):', centroid)
print('spectral bandwidth (Hz):', bandwidth)

# print first 200 samples
print('first 200 samples:', audio[:200])

# compute mel spectrogram summary
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, hop_length=512)
mel_db = librosa.power_to_db(mel, ref=np.max)
print('mel_db min/max/mean:', mel_db.min(), mel_db.max(), mel_db.mean())
# top mel bins by mean energy
bin_means = mel_db.mean(axis=1)
top_bins = np.argsort(bin_means)[-10:][::-1]
print('top mel bins:', top_bins)
print('top bin means:', bin_means[top_bins])

# print a small snip of mel_db
print('mel_db slice (first 5 bins, first 10 frames):')
print(mel_db[:5, :10])
