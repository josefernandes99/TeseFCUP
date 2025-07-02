# scripts/a6_phase3_postprocessing.py

import os
import glob
import csv
import numpy as np
import rasterio
from joblib import load
from config import RAW_DATA_DIR, ROUNDS_DIR, DATA_DIR, SIEVE_MIN_SIZE

# Note: small-patch filtering via rasterio.sieve is now performed during each
# active learning round. The final postprocessing step simply runs the last
# model and saves geotiffs without additional sieving.

def get_final_model_path():
    round_folders = glob.glob(os.path.join(ROUNDS_DIR, "round_*"))
    if not round_folders:
        print("No round folders ⇒ cannot load final model.")
        return None
    rounds = []
    for f in round_folders:
        base = os.path.basename(f)
        try:
            num = int(base.split("_")[1])
            rounds.append((num, f))
        except Exception:
            continue
    if not rounds:
        print("No valid round folders ⇒ no final model.")
        return None
    rounds.sort(key=lambda x: x[0])
    final_r = rounds[-1][1]
    model_path = os.path.join(final_r, f"model_round_{rounds[-1][0]}.pkl")
    if not os.path.exists(model_path):
        print("Model file not found ⇒", model_path)
        return None
    return model_path

def classify_tile(tile_path, model):
    """
    1) Reads the multi-band tile.
    2) Predicts class (0/1) for every pixel.
    Returns the single‐band prediction array and original profile.
    """
    with rasterio.open(tile_path) as src:
        img = src.read()                 # shape: (bands, height, width)
        b, h, w = img.shape
        # reshape to (n_pixels, bands)
        X = img.reshape(b, -1).T        # (h*w, bands)
        # predict all at once
        preds = model.predict(X)        # array of length h*w, values ∈ {0,1}
        pimg = preds.reshape(h, w).astype(np.uint8)

        return pimg, src.profile

def save_geotiff(path_out, data, profile):
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    with rasterio.open(path_out, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"Saved geotiff ⇒ {path_out}")

def postprocessing():
    print("Starting postprocessing…")
    mp = get_final_model_path()
    if mp is None:
        return
    model = load(mp)
    tile_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    if not tile_files:
        print("No tiles found in raw data directory.")
        return

    summary = []
    for tfile in tile_files:
        print(f"Classifying & sieving ⇒ {tfile}")
        cleaned, prof = classify_tile(tfile, model)
        pct_agri = (cleaned.sum() / cleaned.size) * 100
        summary.append([os.path.basename(tfile), f"{pct_agri:.2f}"])

        out_path = tfile.replace(".tif", "_overlay.tif")
        save_geotiff(out_path, cleaned, prof)

    # write out summary CSV
    csv_path = os.path.join(DATA_DIR, "final_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "pct_agricultural"])
        writer.writerows(summary)
    print(f"Postprocessing complete ⇒ {csv_path}")

if __name__ == "__main__":
    postprocessing()
