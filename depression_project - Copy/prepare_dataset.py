import os
import pandas as pd
import numpy as np
from feature_extraction import extract_all_features

AUDIO_ROOT = r"depress4class/data"
LABELS_CSV = "dataset/labels.csv"

def goldberg_to_class(score):
    if score <= 9: return 0
    elif score <= 17: return 1
    elif score <= 24: return 2
    else: return 3

print(f"📄 Reading labels from: {LABELS_CSV}")
if not os.path.exists(LABELS_CSV):
    print("❌ ERROR: labels.csv not found! Exiting.")
    exit(1)

df = pd.read_csv(LABELS_CSV)
print(f"✅ Loaded {len(df)} entries from labels.csv")

features_list, labels_list = [], []
processed = 0
failed = 0

for idx, row in df.iterrows():
    wav_path = os.path.normpath(os.path.join(AUDIO_ROOT, row["filename"]))
    if not os.path.exists(wav_path):
        print(f"⚠ Missing: {wav_path}")
        failed += 1
        continue
    feats = extract_all_features(wav_path)
    if feats is None or not hasattr(feats, "__len__"):
        print(f"❌ Feature extraction failed for: {wav_path}")
        failed += 1
        continue
    features_list.append(feats)
    labels_list.append(goldberg_to_class(row["goldberg_score"]))
    processed += 1
    if processed <= 10 or processed % 500 == 0:
        print(f"Processed {processed}: {wav_path} → {len(feats)} features")

print(f"\n📊 Summary: processed={processed}, failed={failed}")

if processed > 0:
    X_arr = np.array(features_list)
    y_arr = np.array(labels_list)
    os.makedirs("dataset", exist_ok=True)
    np.savez("dataset/features_labels.npz", X=X_arr, y=y_arr)
    print(f"✅ Saved dataset/features_labels.npz → X={X_arr.shape}, y={y_arr.shape}")
else:
    print("❌ No features extracted – check AUDIO_ROOT and feature_extraction.py.")
