import numpy as np
import config as cfg
from config import BANDS, INDICES
from scipy.ndimage import uniform_filter

# Names of features after augmenting aspect with sine and cosine
BASE_FEATURE_NAMES = BANDS + [i for i in INDICES if i != "ASPECT"] + ["ASPECT_SIN", "ASPECT_COS"]

def _append_textures(arr_aug, names):
    """Optionally append light-weight texture features (local mean/std) on NDVI.

    Uses a square window of size cfg.TEXTURE_WINDOW_SIZE.
    """
    if "NDVI" not in names:
        return arr_aug, names
    k = max(int(getattr(cfg, "TEXTURE_WINDOW_SIZE", 5)), 1)
    ndvi = arr_aug[names.index("NDVI")]  # (H, W)
    # compute local mean and std via uniform_filter
    mean = uniform_filter(ndvi, size=k, mode="nearest")
    mean_sq = uniform_filter(ndvi**2, size=k, mode="nearest")
    var = np.clip(mean_sq - mean**2, 0.0, None)
    std = np.sqrt(var)
    arr_out = np.concatenate([arr_aug, mean[None].astype(np.float32), std[None].astype(np.float32)], axis=0)
    names_out = list(names) + [f"NDVI_mean{ k }", f"NDVI_std{ k }"]
    return arr_out, names_out


def add_derived_features(arr):
    """Augment raw spectral/indice bands with aspect sine and cosine.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (bands, H, W) containing the spectral bands and indices
        defined by ``BANDS`` + ``INDICES``.

    Returns
    -------
    arr_aug : np.ndarray
        Augmented feature stack with aspect replaced by its sine and cosine.
    names : list[str]
        Names corresponding to the augmented feature stack.
    """
    band_order = BANDS + INDICES
    aspect_idx = band_order.index("ASPECT")
    aspect = arr[aspect_idx]
    arr_no_aspect = np.delete(arr, aspect_idx, axis=0)
    aspect_rad = np.deg2rad(aspect).astype(np.float32)
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)
    arr_aug = np.concatenate([
        arr_no_aspect,
        aspect_sin[None],
        aspect_cos[None],
    ], axis=0)
    names = list(BASE_FEATURE_NAMES)
    # Feature set toggles
    fs = str(getattr(cfg, "FEATURE_SET", "base")).lower()
    if "textures" in fs:
        arr_aug, names = _append_textures(arr_aug, names)
    # Note: temporal features require multi-timestamp stacks and are not inferred here.
    return arr_aug.astype(np.float32), names

def current_feature_names():
    """Return expected feature names given current config (approximate)."""
    names = list(BASE_FEATURE_NAMES)
    fs = str(getattr(cfg, "FEATURE_SET", "base")).lower()
    if "textures" in fs:
        k = max(int(getattr(cfg, "TEXTURE_WINDOW_SIZE", 5)), 1)
        names += [f"NDVI_mean{ k }", f"NDVI_std{ k }"]
    return names
