import numpy as np
import config as cfg
from config import BANDS, INDICES, TIMESTAMPS
from scipy.ndimage import uniform_filter

# Season-aware base names; derived from config to cover all seasons
def _season_feature_names():
    seasons = list(range(1, len(TIMESTAMPS) + 1))
    names = []
    terrain = list(getattr(cfg, "TERRAIN_BANDS", ["ELEVATION", "SLOPE", "ASPECT"]))
    excluded = set(x.upper() for x in (getattr(cfg, "EXCLUDED_BANDS", []) or []))
    for s in seasons:
        # Spectral bands except excluded
        names += [f"{b}_s{s}" for b in BANDS if b.upper() not in excluded]
        # Indices (always included)
        for idx in INDICES:
            names.append(f"{idx}_s{s}")
        # DEM-derived terrain per season (static across time, replicated per season)
        names += [f"{tb}_s{s}" for tb in terrain]
    return names

def _base_feature_names():
    # Base names without any terrain/aspect transforms
    return _season_feature_names()

def _append_textures(arr_aug, names):
    """Append NDVI texture features.

    - Per-season NDVI local mean: NDVI_s{t}_localmean{k} for each season where NDVI_s{t} exists.
    - Aggregated NDVI_MEAN local mean/std across seasons: NDVI_MEAN_localmean{k}, NDVI_MEAN_localstd{k}.
    """
    k = max(int(getattr(cfg, "TEXTURE_WINDOW_SIZE", 5)), 1)
    names_out = list(names)
    arr_out = arr_aug

    # 1) Per-season NDVI local mean (adds +S features for S seasons)
    seasons = list(range(1, len(TIMESTAMPS) + 1))
    for s in seasons:
        ndvi_name = f"NDVI_s{s}"
        if ndvi_name in names_out:
            nd = arr_out[names_out.index(ndvi_name)].astype(np.float32)
            mean = uniform_filter(nd, size=k, mode="nearest")
            arr_out = np.concatenate([arr_out, mean[None].astype(np.float32)], axis=0)
            names_out.append(f"NDVI_s{s}_localmean{k}")

    # 2) Aggregated NDVI mean/std across seasons
    # Use exact season NDVI names only (avoid matching derived names)
    ndvi_season_names = [f"NDVI_s{s}" for s in seasons if f"NDVI_s{s}" in names_out]
    if ndvi_season_names:
        ndvi_stack = np.stack([arr_out[names_out.index(n)] for n in ndvi_season_names], axis=0)
        ndvi_ref = ndvi_stack.mean(axis=0).astype(np.float32)
        mean = uniform_filter(ndvi_ref, size=k, mode="nearest")
        mean_sq = uniform_filter(ndvi_ref**2, size=k, mode="nearest")
        var = np.clip(mean_sq - mean**2, 0.0, None)
        std = np.sqrt(var)
        arr_out = np.concatenate([arr_out, mean[None].astype(np.float32), std[None].astype(np.float32)], axis=0)
        names_out += [f"NDVI_MEAN_localmean{k}", f"NDVI_MEAN_localstd{k}"]

    return arr_out, names_out


def add_derived_features(arr):
    """Augment raw spectral/indice bands with optional light textures.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (bands, H, W) containing the spectral bands and indices
        defined by ``BANDS`` + ``INDICES`` across seasons.

    Returns
    -------
    arr_aug : np.ndarray
        Augmented feature stack with optional local texture features appended.
    names : list[str]
        Names corresponding to the augmented feature stack.
    """
    # Build raw per-channel names to match export order
    seasons = list(range(1, len(TIMESTAMPS) + 1))
    terrain = list(getattr(cfg, "TERRAIN_BANDS", ["ELEVATION", "SLOPE", "ASPECT"]))
    names_raw = []
    for s in seasons:
        names_raw += [f"{b}_s{s}" for b in BANDS]
        names_raw += [f"{idx}_s{s}" for idx in INDICES]
        names_raw += [f"{tb}_s{s}" for tb in terrain]

    # If array shape doesn't match expected naming, fall back to pass-through
    if arr.shape[0] != len(names_raw):
        arr_aug = arr.astype(np.float32)
        names = _base_feature_names()
        arr_aug, names = _append_textures(arr_aug, names)
        return arr_aug.astype(np.float32), names

    # Apply EXCLUDED_BANDS filtering on spectral bands only
    excluded = set(x.upper() for x in (getattr(cfg, "EXCLUDED_BANDS", []) or []))
    keep_idx = []
    for i, nm in enumerate(names_raw):
        base = nm.split('_s')[0].upper()
        if base in [b.upper() for b in BANDS] and base in excluded:
            continue
        keep_idx.append(i)
    arr_f = arr[keep_idx].astype(np.float32)
    names_f = [names_raw[i] for i in keep_idx]

    # Append textures (per-season NDVI + aggregated NDVI_MEAN)
    arr_aug, names = _append_textures(arr_f, names_f)
    # Note: temporal features require multi-timestamp stacks and are not inferred here.
    return arr_aug.astype(np.float32), names

def current_feature_names():
    """Return expected feature names given current config (approximate)."""
    names = _base_feature_names()
    k = max(int(getattr(cfg, "TEXTURE_WINDOW_SIZE", 5)), 1)
    # Per-season NDVI local mean
    for s in range(1, len(TIMESTAMPS) + 1):
        nd = f"NDVI_s{s}"
        if nd in names:
            names.append(f"NDVI_s{s}_localmean{k}")
    # Aggregated NDVI mean/std
    names += [f"NDVI_MEAN_localmean{k}", f"NDVI_MEAN_localstd{k}"]
    return names
