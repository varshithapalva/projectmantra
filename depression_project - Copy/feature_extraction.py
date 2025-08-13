import numpy as np
import librosa

def extract_all_features(path):
    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        if y.size == 0:
            raise ValueError("Empty audio file")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        rms = np.mean(librosa.feature.rms(y=y))
        return np.hstack([mfcc_mean, zcr, rms])
    except Exception as e:
        print(f"❌ Feature extraction error for {path}: {e}")
        return None
