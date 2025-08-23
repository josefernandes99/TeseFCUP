from __future__ import annotations

"""Helpers shared between active learning rounds and evaluation to avoid
circular imports."""

import os
from typing import Dict, Tuple, Optional

import numpy as np
import rasterio
from pyproj import Transformer
from affine import Affine

from config import RAW_DATA_DIR, FEATURE_CACHE_DIR, FEATURE_CACHE_ENABLED
from features import add_derived_features

# tile cache: tile name -> (features array, transform, CRS)
_tile_cache: Dict[str, Tuple[np.ndarray, Affine, rasterio.crs.CRS]] = {}

__all__ = [
    "extract_features_from_label",
    "get_tile_features",
    "pixel_key",
    "snap_to_pixel_center",
]


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


def get_tile_features(tile_name: str) -> Optional[Tuple[np.ndarray, Affine, rasterio.crs.CRS]]:
    """Return cached per-tile features, transform and CRS; load if needed."""
    tif_path = os.path.join(RAW_DATA_DIR, tile_name)
    if not os.path.exists(tif_path):
        return None
    if tile_name not in _tile_cache:
        # Try disk cache first
        cache_path = os.path.join(FEATURE_CACHE_DIR, f"{os.path.splitext(tile_name)[0]}.npz")
        if FEATURE_CACHE_ENABLED and os.path.exists(cache_path):
            try:
                data = np.load(cache_path, allow_pickle=True)
                arr = data["arr"]
                transform = Affine(*data["transform"]) if "transform" in data else None
                crs_wkt = data["crs_wkt"].item() if "crs_wkt" in data else None
                crs = rasterio.crs.CRS.from_wkt(wkt=str(crs_wkt)) if crs_wkt else None
                _tile_cache[tile_name] = (arr, transform, crs)
                return _tile_cache[tile_name]
            except Exception:
                pass
        with rasterio.open(tif_path) as src:
            raw = src.read().astype(np.float32)
            arr, _ = add_derived_features(raw)
            _tile_cache[tile_name] = (arr, src.transform, src.crs)
            # Persist disk cache
            if FEATURE_CACHE_ENABLED:
                os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
                try:
                    np.savez_compressed(
                        cache_path,
                        arr=arr,
                        transform=np.array(src.transform)[:6],
                        crs_wkt=np.array(src.crs.to_wkt() if src.crs else ""),
                    )
                except Exception:
                    pass
    return _tile_cache[tile_name]


def pixel_key(tile: str, row: int, col: int) -> str:
    return f"{tile}:{row}:{col}"


def snap_to_pixel_center(tile: str, lat: float, lon: float):
    """Snap arbitrary WGS84 coordinates to the exact pixel center for a tile.

    Returns (snapped_lat, snapped_lon, row, col) or None if outside tile.
    """
    tif_path = os.path.join(RAW_DATA_DIR, tile)
    if not os.path.exists(tif_path):
        return None
    with rasterio.open(tif_path) as src:
        x, y = lon, lat
        if src.crs and not src.crs.is_geographic:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
        r, c = rasterio.transform.rowcol(src.transform, x, y)
        if r < 0 or c < 0 or r >= src.height or c >= src.width:
            return None
        cx, cy = rasterio.transform.xy(src.transform, r, c, offset="center")
        if src.crs and not src.crs.is_geographic:
            to_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            cx, cy = to_ll.transform(cx, cy)
        return float(cy), float(cx), int(r), int(c)
