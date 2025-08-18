from __future__ import annotations

"""Helpers shared between active learning rounds and evaluation to avoid
circular imports."""

import os
from typing import Dict, Tuple

import numpy as np
import rasterio
from pyproj import Transformer

from config import RAW_DATA_DIR
from features import add_derived_features

# tile cache: tile name -> (features array, transform, CRS)
_tile_cache: Dict[str, Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]] = {}

__all__ = ["extract_features_from_label"]


def extract_features_from_label(row: Dict[str, str]):
    """Read augmented pixel features at the label coordinate."""
    lat, lon = float(row["lat"]), float(row["lon"])
    tile = row["tile"]
    tif_path = os.path.join(RAW_DATA_DIR, tile)
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Tile file not found: {tif_path}")

    if tile not in _tile_cache:
        with rasterio.open(tif_path) as src:
            raw = src.read().astype(np.float32)
            arr, _ = add_derived_features(raw)
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
