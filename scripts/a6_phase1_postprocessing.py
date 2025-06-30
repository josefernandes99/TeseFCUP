# scripts/a6_phase1_postprocessing.py

import os
import glob
import csv
import logging
import numpy as np
import rasterio
from rasterio.features import sieve
from rasterio import windows
from joblib import load
from rasterio import windows
from joblib import load
from multiprocessing import Pool, cpu_count
from memory_utils import monitor_memory_usage
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
import config
from config import RAW_DATA_DIR, ROUNDS_DIR, DATA_DIR

def get_final_model_path():
    round_folders = glob.glob(os.path.join(ROUNDS_DIR, "round_*"))
    if not round_folders:
        logging.error("No round folders ⇒ cannot load final model.")
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
        logging.error("No valid round folders ⇒ no final model.")
        return None
    rounds.sort(key=lambda x: x[0])
    final_r = rounds[-1][1]
    model_path = os.path.join(final_r, f"model_round_{rounds[-1][0]}.pkl")
    if not os.path.exists(model_path):
        logging.error(f"Model file not found ⇒ {model_path}")
        return None
    return model_path

def classify_tile(tile_path, model):
    """
    1) Reads the multi-band tile.
    2) Predicts class (0/1) for every pixel.
    3) Applies a connected‐component sieve filter to drop
       all agricultural patches < config.SIEVE_MIN_SIZE pixels.
    """
    with rasterio.open(tile_path) as src:
        img = src.read()                 # shape: (bands, height, width)
        b, h, w = img.shape
        # reshape to (n_pixels, bands)
        X = img.reshape(b, -1).T        # (h*w, bands)
        # predict all at once
        preds = model.predict(X)        # array of length h*w, values ∈ {0,1}
        pimg = preds.reshape(h, w).astype(np.uint8)

        # --- SIEVE FILTER ---
        # remove any connected region of 1’s smaller than config.SIEVE_MIN_SIZE
        sieved = sieve(
            pimg,
            size_threshold=config.SIEVE_MIN_SIZE,
            connectivity=8
        )

        return sieved, src.profile

def classify_tile_windowed(tile_path, model, window_size=1024):
    """Classify a tile using sliding windows to limit memory usage."""
    with rasterio.open(tile_path) as src:
        height, width = src.height, src.width
        profile = src.profile
        preds_img = np.zeros((height, width), dtype=np.uint8)

        for row in range(0, height, window_size):
            for col in range(0, width, window_size):
                win = windows.Window(
                    col_off=col,
                    row_off=row,
                    width=min(window_size, width - col),
                    height=min(window_size, height - row),
                )
                chunk = src.read(window=win)
                b, h, w = chunk.shape
                X = chunk.reshape(b, -1).T
                preds = model.predict(X)
                preds_img[row : row + h, col : col + w] = preds.reshape(h, w)

        cleaned = sieve(preds_img, size_threshold=config.SIEVE_MIN_SIZE, connectivity=8)
        return cleaned.astype(np.uint8), profile

def save_geotiff(path_out, data, profile):
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    with rasterio.open(path_out, "w", **profile) as dst:
        dst.write(data, 1)
    logging.info(f"Saved geotiff ⇒ {path_out}")

def _init_pool(model_path):
    """Initializer for worker processes to load the model once per process."""
    global _MODEL
    _MODEL = load(model_path)


def _process_tile(tfile):
    """Classify, sieve and save a single tile."""
    logging.info(f"Classifying & sieving ⇒ {tfile}")
    cleaned, prof = classify_tile_windowed(tfile, _MODEL)
    pct_agri = (cleaned.sum() / cleaned.size) * 100
    out_path = tfile.replace(".tif", "_overlay.tif")
    save_geotiff(out_path, cleaned, prof)
    monitor_memory_usage()
    return [os.path.basename(tfile), f"{pct_agri:.2f}"]

def postprocessing():
    logging.info("Starting postprocessing with Rasterio sieve…")
    mp = get_final_model_path()
    if mp is None:
        return
    tile_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    if not tile_files:
        logging.error("No tiles found in raw data directory.")
        return

    processes = min(4, cpu_count())
    with Pool(processes=processes, initializer=_init_pool, initargs=(mp,)) as pool, \
            Progress(BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as prog:
        task = prog.add_task("Processing tiles", total=len(tile_files))
        summary = []
        for res in pool.imap_unordered(_process_tile, tile_files):
            summary.append(res)
            prog.update(task, advance=1)

    # write out summary CSV
    csv_path = os.path.join(DATA_DIR, "final_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "pct_agricultural"])
        writer.writerows(summary)
    logging.info(f"Postprocessing complete ⇒ {csv_path}")

if __name__ == "__main__":
    postprocessing()
