import numpy as np

from config import BANDS, INDICES

# Names of features after augmenting aspect with sine and cosine
BASE_FEATURE_NAMES = BANDS + [i for i in INDICES if i != "ASPECT"] + ["ASPECT_SIN", "ASPECT_COS"]
EXPECTED_FEATURE_NAMES = BASE_FEATURE_NAMES
EXPECTED_FEATURE_COUNT = len(EXPECTED_FEATURE_NAMES)


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
    return arr_aug.astype(np.float32), list(BASE_FEATURE_NAMES)
