# scripts/a4_phase1_active_learning_loop.py
import os
import shutil
import csv
import logging
from a3_phase1_active_learning_round import active_learning_round
from checkpoint_utils import resume_from_checkpoint
from config import LABELS_FILE, TEMP_LABELS_FILE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def initialize_temp_labels():
    if not os.path.exists(TEMP_LABELS_FILE):
        shutil.copyfile(LABELS_FILE, TEMP_LABELS_FILE)
        logging.info(
            f"Created temp_labels.csv from master labels => {TEMP_LABELS_FILE}"
        )
    else:
        logging.info(f"Reusing existing temp_labels.csv => {TEMP_LABELS_FILE}")

def append_temp_labels(new_labels_file):
    if not os.path.exists(new_labels_file):
        logging.warning(f"No new labels file => {new_labels_file}")
        return
    with open(new_labels_file, "r") as nf:
        rd = csv.DictReader(nf)
        new_rows = list(rd)
    if not new_rows:
        logging.info("No candidate labels to append.")
        return
    with open(TEMP_LABELS_FILE, "a", newline="") as tf:
        wri = csv.writer(tf)
        for r in new_rows:
            # Fix: use "lat" and "lon" instead of "candidate_x" and "candidate_y"
            wri.writerow([r["id"], r["tile"], r["lat"], r["lon"], r["label"]])
    logging.info(f"Appended {len(new_rows)} new labels => {TEMP_LABELS_FILE}")

def active_learning_loop():
    ckpt = resume_from_checkpoint()
    start_round = ckpt["round"] + 1 if ckpt else 1
    if ckpt:
        logging.info(f"Resuming from round {ckpt['round']}")
    try:
        nr = int(input("How many AL rounds? => "))
    except ValueError:
        logging.warning("Invalid => default=1")
        nr = 1

    print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest, 4=Ensemble")
    models = ["ResNet", "SVM", "RandomForest", "Ensemble"]
    ch = input("=> ").strip()
    if ch in ["1", "2", "3", "4"]:
        mchoice = models[int(ch) - 1]
    else:
        logging.warning("Invalid => default=RandomForest")
        mchoice = "RandomForest"

    if not ckpt:
        initialize_temp_labels()

    for r in range(start_round, start_round + nr):
        newfile = active_learning_round(r, TEMP_LABELS_FILE, mchoice)
        if r < start_round + nr - 1 and newfile:
            append_temp_labels(newfile)
    logging.info("AL loop done. Final model => last round folder.")

    if __name__ == "__main__":
        active_learning_loop()
