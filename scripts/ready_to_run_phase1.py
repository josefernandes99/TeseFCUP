# scripts/ready_to_run_phase1.py
from a0_setup_check import setup_check
from a1_phase1_data_download import download_data
from a2_phase1_initial_labeling import (
    initial_labeling,
    check_label_requirements,
    load_labels,
)
from a4_phase1_active_learning_loop import active_learning_loop
from a6_phase1_postprocessing import postprocessing
from config import RAW_DATA_DIR
import glob
import os

def main():
    print("=== Starting PythonProject Pipeline ===\n")

    setup_check()

    if not glob.glob(os.path.join(RAW_DATA_DIR, "*.tif")):
        download_data()
    else:
        print("Raw data already present; skipping download.")

    ok, _, _ = check_label_requirements()
    if not ok:
        initial_labeling()
    else:
        total = len(load_labels())
        print(f"Found {total} labels meeting requirements; skipping initial labeling.")

    active_learning_loop()
    postprocessing()

    print("\n=== Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    main()
