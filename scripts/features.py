import numpy as np
import config as cfg
from config import BANDS, INDICES, TIMESTAMPS
from scipy.ndimage import uniform_filter

# Season-aware base names; derived from config to cover all seasons
def _season_feature_names():
    seasons = list(range(1, len(TIMESTAMPS) + 1))
    names = []
    terrain = list(getattr(cfg, "TERRAIN_BANDS", ["ELEVATION", "SLOPE", "ASPECT"]))
    for s in seasons:
        names += [f"{b}_s{s}" for b in BANDS]
        for idx in INDICES:
            names.append(f"{idx}_s{s}")
        # DEM-derived terrain per season (static across time, replicated per season)
        names += [f"{tb}_s{s}" for tb in terrain]
    return names

def _base_feature_names():
    # Base names without any terrain/aspect transforms
    return _season_feature_names()

def _append_textures(arr_aug, names):
    """Optionally append light-weight texture features (local mean/std) on NDVI.

    Uses a square window of size cfg.TEXTURE_WINDOW_SIZE.
    """
    # Prefer seasonal NDVI; fallback to plain NDVI if present
    ndvi_names = [n for n in names if n.upper().startswith("NDVI_S") or n.upper() == "NDVI"]
    if not ndvi_names:
        return arr_aug, names
    k = max(int(getattr(cfg, "TEXTURE_WINDOW_SIZE", 5)), 1)
    if any(n.upper().startswith("NDVI_S") for n in ndvi_names):
        ndvi_stack = np.stack([arr_aug[names.index(n)] for n in ndvi_names if n.upper().startswith("NDVI_S")], axis=0)
        ndvi_ref = ndvi_stack.mean(axis=0)
        ndvi_tag = "NDVI_MEAN"
    else:
        ndvi_ref = arr_aug[names.index("NDVI")]
        ndvi_tag = "NDVI"
    ndvi = ndvi_ref.astype(np.float32)
    mean = uniform_filter(ndvi, size=k, mode="nearest")
    mean_sq = uniform_filter(ndvi**2, size=k, mode="nearest")
    var = np.clip(mean_sq - mean**2, 0.0, None)
    std = np.sqrt(var)
    arr_out = np.concatenate([arr_aug, mean[None].astype(np.float32), std[None].astype(np.float32)], axis=0)
    names_out = list(names) + [f"{ndvi_tag}_localmean{k}", f"{ndvi_tag}_localstd{k}"]
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
    # Pass-through raw stack (now includes terrain replicated per season), then add textures
    arr_aug = arr
    names = _base_feature_names()
    # Always use all available derived features
    arr_aug, names = _append_textures(arr_aug, names)
    # Note: temporal features require multi-timestamp stacks and are not inferred here.
    return arr_aug.astype(np.float32), names

def current_feature_names():
    """Return expected feature names given current config (approximate)."""
    names = _base_feature_names()
    k = max(int(getattr(cfg, "TEXTURE_WINDOW_SIZE", 5)), 1)
    names += [f"NDVI_MEAN_localmean{k}", f"NDVI_MEAN_localstd{k}"]
    return names
