import os
from typing import List, Tuple, Optional
import numpy as np
import rasterio
import config as cfg
from config import BANDS, INDICES
from scipy.ndimage import uniform_filter

# Names of features after augmenting aspect with sine and cosine (legacy single-season baseline)
BASE_FEATURE_NAMES = BANDS + [i for i in INDICES if i != "ASPECT"] + ["ASPECT_SIN", "ASPECT_COS"]


def _infer_season_count(raw_channels: int) -> int:
    """
    Infer number of seasons from total raw channels before aspect augmentation.
    Exporter layout: per season => len(BANDS) + 5 indices, then ELEVATION, SLOPE, ASPECT.
    """
    per_season = len(BANDS) + 5  # NDVI, EVI, EVI2, NBR, NDMI
    if raw_channels < 3 or (raw_channels - 3) % per_season != 0:
        raise ValueError(f"Cannot infer seasons from channel count={raw_channels} (per_season={per_season}).")
    n_seasons = (raw_channels - 3) // per_season
    if n_seasons <= 0:
        raise ValueError("Invalid season count inferred (<=0).")
    return int(n_seasons)


def _raw_feature_names_for_channels(raw_channels: int) -> List[str]:
    """
    Construct raw band names (before aspect augmentation) for a given channel count.
    Order matches scripts/a1_phase1_data_download.py export: for each season, BANDS then indices; then ELEVATION,SLOPE,ASPECT.
    """
    n_seasons = _infer_season_count(raw_channels)
    names: List[str] = []
    for s in range(1, n_seasons + 1):
        names.extend([f"{b}_s{s}" for b in BANDS])
        names.extend([f"{idx}_s{s}" for idx in ["NDVI", "EVI", "EVI2", "NBR", "NDMI"]])
    names.extend(["ELEVATION", "SLOPE", "ASPECT"])
    return names


def _augmented_feature_names_for_channels(raw_channels: int) -> List[str]:
    """
    Names after replacing ASPECT with ASPECT_SIN/ASPECT_COS placed after SLOPE.
    """
    raw_names = _raw_feature_names_for_channels(raw_channels)
    names = [n for n in raw_names if n != "ASPECT"]
    names.extend(["ASPECT_SIN", "ASPECT_COS"])
    return names


def _append_textures(arr_aug, names):
    """Optionally append light-weight texture features (local mean/std) on NDVI.

    Uses a square window of size cfg.TEXTURE_WINDOW_SIZE.
    Currently disabled for multi-season stacks (no-op when NDVI not present)."""
    if "NDVI" not in names:
        return arr_aug, names
    k = max(int(getattr(cfg, "TEXTURE_WINDOW_SIZE", 5)), 1)
    ndvi = arr_aug[names.index("NDVI")]  # (H, W)
    mean = uniform_filter(ndvi, size=k, mode="nearest")
    mean_sq = uniform_filter(ndvi**2, size=k, mode="nearest")
    var = np.clip(mean_sq - mean**2, 0.0, None)
    std = np.sqrt(var)
    arr_out = np.concatenate([arr_aug, mean[None].astype(np.float32), std[None].astype(np.float32)], axis=0)
    names_out = list(names) + [f"NDVI_mean{ k }", f"NDVI_std{ k }"]
    return arr_out, names_out


def add_derived_features(arr: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Augment raw spectral/indice bands with aspect sine and cosine, aligned to exported stack.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (C, H, W) containing the stacked seasons and terrain
        as exported by a1_phase1_data_download.py (before augmentation).

    Returns
    -------
    arr_aug : np.ndarray
        Augmented feature stack with ASPECT replaced by its sine and cosine.
    names : list[str]
        Names corresponding to the augmented feature stack.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected (C,H,W) array, got shape={arr.shape}")
    C, H, W = arr.shape
    raw_names = _raw_feature_names_for_channels(C)
    # ASPECT is expected at the end of the raw export order
    try:
        aspect_pos = raw_names.index("ASPECT")
    except ValueError:
        aspect_pos = C - 1
    aspect = arr[aspect_pos].astype(np.float32)
    arr_no_aspect = np.delete(arr, aspect_pos, axis=0)
    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)
    arr_aug = np.concatenate([
        arr_no_aspect,
        aspect_sin[None],
        aspect_cos[None],
    ], axis=0)
    names = _augmented_feature_names_for_channels(C)
    # Feature set toggles: for now, temporal/textures/full are not active.
    fs = str(getattr(cfg, "FEATURE_SET", "base")).lower()
    if "textures" in fs:
        # Not applied for multi-season stacks in current pipeline
        pass
    return arr_aug.astype(np.float32), names


def current_feature_names() -> List[str]:
    """
    Return feature names that align to the actual stacked raster schema on disk.

    Robust: open any tile in RAW_DATA_DIR, read channel count, and compute
    names accordingly (ASPECT replaced by ASPECT_SIN/COS).
    """
    try:
        tiles = [p for p in os.listdir(cfg.RAW_DATA_DIR) if p.lower().endswith(".tif")]
        if not tiles:
            return list(BASE_FEATURE_NAMES)
        tif = os.path.join(cfg.RAW_DATA_DIR, tiles[0])
        with rasterio.open(tif) as src:
            C = src.count
        return _augmented_feature_names_for_channels(C)
    except Exception:
        return list(BASE_FEATURE_NAMES)


def get_reporting_ndvi_index(names: List[str]) -> Optional[int]:
    """
    Choose a representative NDVI band index for reporting (CSV/KML).
    Preference order: NDVI_s2 (middle season), then any NDVI_s*, then 'NDVI'.
    """
    ndvi_seasonals = [(i, n) for i, n in enumerate(names) if n.startswith("NDVI_s")]
    if ndvi_seasonals:
        try:
            parsed = [(i, int(n.split("_s")[-1])) for i, n in ndvi_seasonals]
            parsed.sort(key=lambda x: x[1])
            mid = parsed[len(parsed)//2][0]
            return mid
        except Exception:
            return ndvi_seasonals[len(ndvi_seasonals)//2][0]
    try:
        return names.index("NDVI")
    except ValueError:
        return None
