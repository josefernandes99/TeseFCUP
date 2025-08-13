# scripts/a6_phase3_postprocessing.py

import os
import glob
import csv
import numpy as np
import rasterio
from joblib import load, Parallel, delayed
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from memory_watcher import free_unused_memory
from config import RAW_DATA_DIR, ROUNDS_DIR, DATA_DIR, MIN_AGRI_PROB, CANDIDATE_PROB_LOWER, SIEVE_MIN_SIZE
from features import load_cached_features, feature_cache_path
from scipy.ndimage import binary_closing, binary_fill_holes, label as ndlabel
from rasterio.features import sieve

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
    final_num, final_r = rounds[-1]
    # Models may be stored within subdirectories (e.g., "manual" or grid-search
    # combo names). Search recursively for the expected model file inside the
    # last round folder.
    pattern = os.path.join(final_r, "**", f"model_round_{final_num}.pkl")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        print("Model file not found ⇒", os.path.join(final_r, f"model_round_{final_num}.pkl"))
        return None
    return matches[0]

def classify_tile(tile_path, model):
    """
    1) Reads the multi-band tile.
    2) Predicts class (0/1) for every pixel.
    Returns the single‐band prediction array and original profile.
    """
    with rasterio.open(tile_path) as src:
        cache_path = feature_cache_path(tile_path)
        img, _, _ = load_cached_features(cache_path, arr=src.read().astype(np.float32))
        b, h, w = img.shape
        X = img.reshape(b, -1).T        # (h*w, bands)
        probs = model.predict_proba(X)[:,1].reshape(h, w)
        crop = probs >= MIN_AGRI_PROB
        uncertain = (probs >= CANDIDATE_PROB_LOWER) & (probs < MIN_AGRI_PROB)
        mask = crop | uncertain
        mask = binary_fill_holes(binary_closing(mask))
        lbl, num = ndlabel(mask)
        for i in range(1, num+1):
            comp = (lbl==i)
            if not np.any(crop[comp]):
                mask[comp] = False
        if SIEVE_MIN_SIZE > 0:
            mask = sieve(mask.astype("uint8"), size=SIEVE_MIN_SIZE, connectivity=8).astype(bool)
        cleaned = (mask & (probs >= MIN_AGRI_PROB)).astype(np.uint8)
        return cleaned, src.profile

def save_geotiff(path_out, data, profile):
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    with rasterio.open(path_out, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"Saved geotiff ⇒ {path_out}")

def process_tile(tfile, model):
    """Classify one tile and save overlay GeoTIFF."""
    print(f"Classifying ⇒ {tfile}")
    cleaned, prof = classify_tile(tfile, model)
    n_pix = cleaned.size
    n_agri = int(cleaned.sum())
    pct_agri = (n_agri / n_pix) * 100 if n_pix else 0.0
    out_path = tfile.replace(".tif", "_overlay.tif")
    save_geotiff(out_path, cleaned, prof)
    free_unused_memory()
    return os.path.basename(tfile), pct_agri, n_pix, n_agri

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
    total_pixels = 0
    total_agri_pixels = 0
    pct_list = []

    with Progress(
        "[bold cyan]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Classifying tiles", total=len(tile_files))

        def wrapped(tp):
            res = process_tile(tp, model)
            progress.update(task, advance=1)
            return res

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(wrapped)(tp) for tp in tile_files
        )

    for tile, pct_agri, n_pix, n_agri in results:
        summary.append([
            tile,
            f"{pct_agri:.2f}",
            n_pix,
            n_agri,
        ])
        pct_list.append(pct_agri)
        total_pixels += n_pix
        total_agri_pixels += n_agri

    # write out per-tile CSV
    csv_path = os.path.join(DATA_DIR, "final_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "pct_agricultural", "pixels_total", "pixels_agricultural"])
        writer.writerows(summary)

    # global statistics
    if pct_list:
        overall_pct = (total_agri_pixels / total_pixels) * 100 if total_pixels else 0
        avg_pct = float(np.mean(pct_list))
        std_pct = float(np.std(pct_list))
        min_pct = float(np.min(pct_list))
        max_pct = float(np.max(pct_list))
    else:
        overall_pct = avg_pct = std_pct = min_pct = max_pct = 0.0

    stats_path = os.path.join(DATA_DIR, "final_summary.txt")
    with open(stats_path, "w") as sf:
        sf.write(f"Tiles processed: {len(tile_files)}\n")
        sf.write(f"Total pixels: {total_pixels}\n")
        sf.write(f"Total agricultural pixels: {total_agri_pixels}\n")
        sf.write(f"Overall agricultural %: {overall_pct:.2f}\n")
        sf.write(f"Average tile %: {avg_pct:.2f}\n")
        sf.write(f"Std dev tile %: {std_pct:.2f}\n")
        sf.write(f"Min tile %: {min_pct:.2f}\n")
        sf.write(f"Max tile %: {max_pct:.2f}\n")

    print(f"Postprocessing complete ⇒ {csv_path}")
    print(f"Summary statistics ⇒ {stats_path}")

if __name__ == "__main__":
    postprocessing()
