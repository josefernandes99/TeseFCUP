# scripts/a0_setup_check.py
import ee
import os
import glob
import shutil
import certifi
from config import (
    ROUNDS_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    TEMP_LABELS_FILE,
    CANDIDATE_KML,
    GRID_KML_DIR,
)

# Override problematic certificate environment variables.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()
os.environ.pop("REQUESTS_CA_BUNDLE", None)

def cleanup_previous_runs():
    """Remove outputs from any prior pipeline executions."""
    # Delete round folders like round_1, round_2, etc.
    round_folders = glob.glob(os.path.join(ROUNDS_DIR, "round*"))
    for rf in round_folders:
        try:
            shutil.rmtree(rf)
            print(f"Deleted previous round folder => {rf}")
        except Exception as e:
            print(f"Failed to delete {rf}: {e}")

    # Remove temporary labels file
    if os.path.exists(TEMP_LABELS_FILE):
        try:
            os.remove(TEMP_LABELS_FILE)
            print(f"Deleted => {TEMP_LABELS_FILE}")
        except Exception as e:
            print(f"Failed to delete {TEMP_LABELS_FILE}: {e}")

    # Remove final predictions CSV
    final_preds = os.path.join(DATA_DIR, "final_predictions.csv")
    if os.path.exists(final_preds):
        try:
            os.remove(final_preds)
            print(f"Deleted => {final_preds}")
        except Exception as e:
            print(f"Failed to delete {final_preds}: {e}")

    # Remove final predictions CSV
    final_summary = os.path.join("C:/Users/Vitorino/PycharmProjects/PythonProject/data/phase1", "final_summary.txt")
    if os.path.exists(final_summary):
        try:
            os.remove(final_summary)
            print(f"Deleted => {final_summary}")
        except Exception as e:
            print(f"Failed to delete {final_summary}: {e}")

    # Remove overlay tiles produced by previous postprocessing
    overlays = glob.glob(os.path.join(RAW_DATA_DIR, "*_overlay.tif"))
    for o in overlays:
        try:
            os.remove(o)
            print(f"Deleted overlay => {o}")
        except Exception as e:
            print(f"Failed to delete {o}: {e}")

    # Delete candidate patch KML if present
    cand_kml_path = CANDIDATE_KML
    if os.path.exists(cand_kml_path):
        try:
            os.remove(cand_kml_path)
            print(f"Deleted => {cand_kml_path}")
        except Exception as e:
            print(f"Failed to delete {cand_kml_path}: {e}")
    if os.path.exists(GRID_KML_DIR):
        for f in glob.glob(os.path.join(GRID_KML_DIR, "*.kml")):
            try:
                os.remove(f)
                print(f"Deleted => {f}")
            except Exception as e:
                print(f"Failed to delete {f}: {e}")

def setup_check():
    cleanup_previous_runs()
    print("Initializing Google Earth Engine API...")
    try:
        ee.Initialize(project="earthenginecapeverde")
        print("Earth Engine API initialized successfully!")
    except Exception as e:
        print("ERROR: Failed to initialize Earth Engine API. Check credentials & internet.")
        print("Details:", e)
        exit(1)

if __name__ == "__main__":
    setup_check()
