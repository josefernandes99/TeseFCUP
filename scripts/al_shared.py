from __future__ import annotations

"""Helpers shared between active learning rounds and evaluation to avoid
circular imports."""

import os
from typing import Dict, Tuple, Optional
from collections import OrderedDict

import numpy as np
import rasterio
from pyproj import Transformer

from config import RAW_DATA_DIR, FEATURE_CACHE_DIR, FEATURE_CACHE_ENABLED, SKIPPED_PIXELS_FILE
from features import add_derived_features, current_feature_names

# tile cache (LRU): tile name -> (features array, transform, CRS)
_tile_cache: "OrderedDict[str, Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]]" = OrderedDict()

def _cache_put(tile: str, value: Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]):
    """Insert into LRU cache with max size bound from config."""
    from config import FEATURE_CACHE_MAX_TILES_IN_MEMORY
    _tile_cache[tile] = value
    _tile_cache.move_to_end(tile)
    try:
        max_items = max(1, int(FEATURE_CACHE_MAX_TILES_IN_MEMORY))
    except Exception:
        max_items = 2
    while len(_tile_cache) > max_items:
        try:
            _tile_cache.popitem(last=False)
        except Exception:
            break

__all__ = [
    "extract_features_from_label",
    "get_tile_features",
    "pixel_key",
    "snap_to_pixel_center",
    "load_skipped_set",
    "record_skipped_pixel",
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
            _cache_put(tile, (arr, src.transform, src.crs))

    arr, transform, crs = _tile_cache[tile]
    x, y = lon, lat
    if crs and not crs.is_geographic:
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
    row_i, col_i = rasterio.transform.rowcol(transform, x, y)
    if (0 <= row_i < arr.shape[1]) and (0 <= col_i < arr.shape[2]):
        return arr[:, row_i, col_i].astype(float).tolist()
    return None


def get_tile_features(tile_name: str) -> Optional[Tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]]:
    """Return per-tile features, transform, and CRS; robust to cache races.

    Never assumes the cache entry exists after computation; always returns the
    freshly computed tuple even if an eviction happens concurrently.
    """
    tif_path = os.path.join(RAW_DATA_DIR, tile_name)
    if not os.path.exists(tif_path):
        return None
    # Fast path: in-memory cache hit
    if tile_name in _tile_cache:
        try:
            return _tile_cache[tile_name]
        except KeyError:
            # Rare race: fall through to recompute
            pass
    # Prepare disk cache path
    cache_path = os.path.join(FEATURE_CACHE_DIR, f"{os.path.splitext(tile_name)[0]}.npz")
    # Try disk cache first
    if FEATURE_CACHE_ENABLED and os.path.exists(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=True)
            arr = data["arr"]
            transform = rasterio.Affine(*data["transform"]) if "transform" in data else None
            crs_wkt = data["crs_wkt"].item() if "crs_wkt" in data else None
            crs = rasterio.crs.CRS.from_wkt(wkt=crs_wkt) if crs_wkt else None
            # Validate channel count vs current config
            try:
                exp = len(current_feature_names())
            except Exception:
                exp = None
            if exp is not None and hasattr(arr, 'shape') and arr.ndim == 3 and arr.shape[0] != exp:
                raise ValueError("stale_feature_cache")
            value = (arr, transform, crs)
            _cache_put(tile_name, value)
            return value
        except Exception:
            # Cache missing or stale; compute from raw
            pass
    # Compute from raw
    with rasterio.open(tif_path) as src:
        raw = src.read().astype(np.float32)
        arr, _ = add_derived_features(raw)
        value = (arr, src.transform, src.crs)
        _cache_put(tile_name, value)
        # Persist disk cache best-effort
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
        return value


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


def load_skipped_set():
    """Load skipped pixels as a set of keys tile:row:col."""
    s = set()
    try:
        import csv as _csv
        if os.path.exists(SKIPPED_PIXELS_FILE):
            with open(SKIPPED_PIXELS_FILE, newline='') as f:
                for r in _csv.DictReader(f):
                    t = r.get('tile')
                    try:
                        key = f"{t}:{int(r.get('row'))}:{int(r.get('col'))}"
                        s.add(key)
                    except Exception:
                        continue
    except Exception:
        pass
    return s


def record_skipped_pixel(tile: str, row: int, col: int, lat: float, lon: float, source: str = ""):
    """Append a skipped pixel record to labels/phase1/skipped.csv.

    Columns: tile,row,col,lat,lon,source
    """
    import csv as _csv
    os.makedirs(os.path.dirname(SKIPPED_PIXELS_FILE), exist_ok=True)
    write_header = not os.path.exists(SKIPPED_PIXELS_FILE)
    with open(SKIPPED_PIXELS_FILE, 'a', newline='') as f:
        w = _csv.writer(f)
        if write_header:
            w.writerow(["tile","row","col","lat","lon","source"])
        w.writerow([tile, int(row), int(col), f"{float(lat):.7f}", f"{float(lon):.7f}", source or ""]) 
