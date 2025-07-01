# scripts/a4_phase1_active_learning_loop.py
import os
import shutil
import csv
from a3_phase1_active_learning_round import active_learning_round
from config import LABELS_FILE, TEMP_LABELS_FILE

def initialize_temp_labels():
    if not os.path.exists(TEMP_LABELS_FILE):
        shutil.copyfile(LABELS_FILE, TEMP_LABELS_FILE)
        print(f"Created temp_labels.csv from master labels => {TEMP_LABELS_FILE}")
    else:
        print(f"Reusing existing temp_labels.csv => {TEMP_LABELS_FILE}")

def append_temp_labels(new_labels_file):
    if not os.path.exists(new_labels_file):
        print(f"No new labels file => {new_labels_file}")
        return
    with open(new_labels_file, "r") as nf:
        rd = csv.DictReader(nf)
        new_rows = list(rd)
    if not new_rows:
        print("No candidate labels to append.")
        return
    # Append using the master CSV column order: id, lat, lon, tile, label
    with open(TEMP_LABELS_FILE, "a", newline="") as tf:
        wri = csv.writer(tf)
        for r in new_rows:
            wri.writerow([r["id"], r["lat"], r["lon"], r["tile"], r["label"]])
    print(f"Appended {len(new_rows)} new labels => {TEMP_LABELS_FILE}")

def active_learning_loop():
    try:
        nr = int(input("How many AL rounds? => "))
    except ValueError:
        print("Invalid => default=1")
        nr = 1

    print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest")
    models = ["ResNet", "SVM", "RandomForest"]
    ch = input("=> ").strip()
    if ch in ["1", "2", "3"]:
        mchoice = models[int(ch) - 1]
    else:
        print("Invalid => default=RandomForest")
        mchoice = "RandomForest"

    initialize_temp_labels()
    for r in range(1, nr + 1):
        newfile = active_learning_round(r, TEMP_LABELS_FILE, mchoice)
        if r < nr and newfile:
            append_temp_labels(newfile)
    print("AL loop done. Final model => last round folder.")

if __name__ == "__main__":
    active_learning_loop()
