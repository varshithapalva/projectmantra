import os
import csv
import random

# --- Edit this so it matches your actual directory exactly ---
DATA_ROOT = "depress4class/data"
OUTPUT_CSV = "dataset/labels.csv"

# List folders inside "data/" that should be labeled as 'No depression'
NON_DEPRESSED_FOLDERS = [
    "NEU", "HAP", "OAF_happy", "OAF_neutral", "YAF_happy", "YAF_neutral", "OAF_Pleasant_Surprise", "YAF_Pleasant_Surprise"
    # <-- Add *exact* folder names from your 'data' dir that you consider non-depressed!
]

DEPRESSED_CLASSES = [
    ("Mild depression", 10, 17),
    ("Moderate depression", 18, 24),
    ("Severe depression", 25, 27)
]
NON_DEPRESSED_CLASS = ("No depression", 0, 9)

rows = []

print(f"Scanning: {os.path.abspath(DATA_ROOT)}")
if not os.path.isdir(DATA_ROOT):
    print(f"ERROR: Folder {DATA_ROOT} does not exist!")
    exit(1)

folder_count = 0
wav_count = 0

for subdir in os.listdir(DATA_ROOT):
    abs_subdir = os.path.join(DATA_ROOT, subdir)
    if not os.path.isdir(abs_subdir):
        continue
    folder_count += 1
    this_folder_wavs = 0
    for file in os.listdir(abs_subdir):
        if file.lower().endswith(".wav"):
            rel_path = os.path.join(subdir, file)
            if subdir in NON_DEPRESSED_FOLDERS:
                name, smin, smax = NON_DEPRESSED_CLASS
            else:
                name, smin, smax = random.choice(DEPRESSED_CLASSES)
            goldberg_score = random.randint(smin, smax)
            rows.append([rel_path, goldberg_score, name])
            this_folder_wavs += 1
            print(f"Found audio: {rel_path}  -->  {name} ({goldberg_score})")
    print(f"  >> {this_folder_wavs} wavs in {subdir}")

print(f"\nTotal folders scanned: {folder_count}")
print(f"Total wav files found: {len(rows)}")

if len(rows) == 0:
    print("WARNING: No .wav files found; check the DATA_ROOT path and folder contents!")
else:
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "goldberg_score", "class"])
        writer.writerows(rows)
    print(f"\n✅ Generated {OUTPUT_CSV} with {len(rows)} entries.")
