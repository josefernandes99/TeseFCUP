from __future__ import annotations

"""Helpers shared between active learning rounds and evaluation to avoid
circular imports."""

import os
from pathlib import Path
from typing import Iterable, Dict, Tuple

import numpy as np
import rasterio
from pyproj import Transformer
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn

from config import RAW_DATA_DIR
from features import load_cached_features, TEXTURE_BANDS, feature_cache_path

# tile cache: tile name -> (features array, transform, CRS)
_tile_cache: Dict[str, Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]] = {}

__all__ = ["preload_tiles", "extract_features_from_label"]


def preload_tiles(tile_list: Iterable[str]) -> None:
    """Pre-compute derived features for a list of tiles with a single progress bar."""
    global _tile_cache
    tiles = list(tile_list)
    if not tiles:
        return

    heights = []
    for t in tiles:
        with rasterio.open(os.path.join(RAW_DATA_DIR, t)) as src:
            heights.append(src.height)
    total_rows = sum(heights) * len(TEXTURE_BANDS)

    with Progress(
        "[bold cyan]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as prog:
        task = prog.add_task(
            f"Processing GLCM textures 0/{len(tiles)}", total=total_rows
        )
        completed = 0
        for t in tiles:
            tile_path = os.path.join(RAW_DATA_DIR, t)
            with rasterio.open(tile_path) as src:
                cache_path = feature_cache_path(tile_path)
                arr, _, used = load_cached_features(
                    cache_path, arr=src.read().astype(np.float32), prog=prog, task=task
                )
                _tile_cache[t] = (arr, src.transform, src.crs)
            completed += 1
            desc = (
                "Loaded GLCM textures" if used else "Caching GLCM textures"
            )
            prog.update(
                task, description=f"{desc} {completed}/{len(tiles)}"
            )
        prog.update(
            task, description=f"Loaded GLCM textures {completed}/{len(tiles)}"
        )


def extract_features_from_label(row: Dict[str, str]):
    """Read augmented pixel features at the label coordinate."""
    lat, lon = float(row["lat"]), float(row["lon"])
    tile = row["tile"]
    tif_path = os.path.join(RAW_DATA_DIR, tile)
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Tile file not found: {tif_path}")

    if tile not in _tile_cache:
        with rasterio.open(tif_path) as src:
            cache_path = feature_cache_path(tif_path)
            arr, _, _ = load_cached_features(cache_path, arr=src.read().astype(np.float32))
            _tile_cache[tile] = (arr, src.transform, src.crs)

    arr, transform, crs = _tile_cache[tile]
    x, y = lon, lat
    if crs and not crs.is_geographic:
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
    row_i, col_i = rasterio.transform.rowcol(transform, x, y)
    if (0 <= row_i < arr.shape[1]) and (0 <= col_i < arr.shape[2]):
        return arr[:, row_i, col_i].astype(float).tolist()
    return None
