import os
from typing import List, Tuple, Optional
import numpy as np
import rasterio
import config as cfg
from config import BANDS, INDICES
from scipy.ndimage import uniform_filter
from skimage.filters import rank
from skimage.morphology import square

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


def _safe_quantize(img: np.ndarray, levels: int = 32):
    """Quantize float image to [0, levels-1] uint8 robustly using 1–99 percentiles.

    Handles NaNs by treating them as the local mean during rank operations.
    """
    if not np.isfinite(img).any():
        mn, mx = 0.0, 1.0
    else:
        p1 = float(np.nanpercentile(img, 1))
        p99 = float(np.nanpercentile(img, 99))
        if not np.isfinite(p1):
            p1 = float(np.nanmin(img)) if np.isfinite(np.nanmin(img)) else 0.0
        if not np.isfinite(p99):
            p99 = float(np.nanmax(img)) if np.isfinite(np.nanmax(img)) else 1.0
        if p99 <= p1:
            p99 = p1 + 1e-6
        mn, mx = p1, p99
    x = np.clip(img, mn, mx)
    q = np.round((x - mn) / (mx - mn) * (levels - 1)).astype(np.uint8)
    # NaNs -> 0 for rank filters; they will be handled gracefully
    q[~np.isfinite(img)] = 0
    return q


def _append_neighbor_pooling(arr: np.ndarray, names: list[str]):
    """Append local mean/std for selected bands/indices and windows.

    Channels to pool and window sizes come from config. If a requested band
    has seasonal variants (e.g., NDVI_s1..s3), prefer the middle season.
    """
    if not getattr(cfg, "NEIGHBOR_POOLING_ENABLED", False):
        return arr, names
    bands = list(getattr(cfg, "NEIGHBOR_POOLING_BANDS", []))
    windows = [int(k) for k in getattr(cfg, "NEIGHBOR_POOLING_WINDOWS", []) if int(k) >= 1]
    if not bands or not windows:
        return arr, names
    # Map logical band name to channel index in 'names'
    def pick_channel(bname: str) -> int | None:
        # Prefer seasonal middle (s2) when available
        cand = [i for i, n in enumerate(names) if n.lower().startswith(bname.lower() + "_s2")]
        if cand:
            return cand[0]
        # any seasonal
        any_s = [i for i, n in enumerate(names) if n.lower().startswith(bname.lower() + "_s")]
        if any_s:
            return any_s[len(any_s)//2]
        # plain name
        try:
            return names.index(bname)
        except ValueError:
            return None

    out = [arr]
    out_names = list(names)
    _, H, W = arr.shape
    for b in bands:
        idx = pick_channel(b)
        if idx is None:
            continue
        img = arr[idx]
        for k in windows:
            k = max(int(k), 1)
            mean = uniform_filter(img, size=k, mode="nearest")
            mean_sq = uniform_filter(img * img, size=k, mode="nearest")
            var = np.clip(mean_sq - mean * mean, 0.0, None)
            std = np.sqrt(var, dtype=np.float32)
            out.append(mean[None].astype(np.float32))
            out.append(std[None].astype(np.float32))
            out_names.append(f"{b}_mean{k}")
            out_names.append(f"{b}_std{k}")
    return np.concatenate(out, axis=0), out_names


def _append_textures(arr: np.ndarray, names: list[str]):
    """Append lightweight texture features (entropy/contrast/homogeneity proxies).

    Uses rank entropy (fast, robust) and local std/mean‑abs‑deviation proxies for
    contrast/homogeneity. Applied to selected bands and window sizes. Skips very
    large tiles to avoid excessive compute.
    """
    if not getattr(cfg, "TEXTURE_GLCM_ENABLED", False):
        return arr, names
    bands = list(getattr(cfg, "TEXTURE_GLCM_BANDS", []))
    windows = [int(k) for k in getattr(cfg, "TEXTURE_GLCM_WINDOWS", []) if int(k) >= 1]
    if not bands or not windows:
        return arr, names
    C, H, W = arr.shape

    def pick_channel(bname: str) -> int | None:
        cand = [i for i, n in enumerate(names) if n.lower().startswith(bname.lower() + "_s2")]
        if cand:
            return cand[0]
        any_s = [i for i, n in enumerate(names) if n.lower().startswith(bname.lower() + "_s")]
        if any_s:
            return any_s[len(any_s)//2]
        try:
            return names.index(bname)
        except ValueError:
            return None

    out = [arr]
    out_names = list(names)
    for b in bands:
        idx = pick_channel(b)
        if idx is None:
            continue
        img = arr[idx].astype(np.float32)
        q = _safe_quantize(img, levels=int(getattr(cfg, "TEXTURE_GLCM_LEVELS", 32)))
        for k in windows:
            se = square(int(k))
            # entropy (bits) via rank entropy on quantized image
            try:
                ent = rank.entropy(q, selem=se)  # returns uint8 scaled [0,8]
                ent = (ent.astype(np.float32) / 8.0)
            except Exception:
                # fallback to variance‑based proxy if rank fails
                mean = uniform_filter(img, size=k, mode="nearest")
                mean_sq = uniform_filter(img * img, size=k, mode="nearest")
                var = np.clip(mean_sq - mean * mean, 0.0, None)
                ent = np.log1p(var).astype(np.float32)
            # contrast proxy: local std
            mean = uniform_filter(img, size=k, mode="nearest")
            mean_sq = uniform_filter(img * img, size=k, mode="nearest")
            var = np.clip(mean_sq - mean * mean, 0.0, None)
            std = np.sqrt(var, dtype=np.float32)
            # homogeneity proxy: 1 / (1 + mean absolute deviation)
            mad = uniform_filter(np.abs(img - mean), size=k, mode="nearest")
            homo = (1.0 / (1.0 + mad)).astype(np.float32)
            out.extend([ent[None], std[None], homo[None]])
            out_names.extend([f"{b}_tex_entropy{k}", f"{b}_tex_contrast{k}", f"{b}_tex_homogeneity{k}"])
    return np.concatenate(out, axis=0), out_names


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

    # Temporal summaries across seasons for key indices
    if getattr(cfg, "TEMPORAL_FEATURES_ENABLED", False):
        indices = list(getattr(cfg, "TEMPORAL_INDICES", []))
        # Build per‑index stacks [S,H,W] using names ending with _s*
        for idx_name in indices:
            # Collect seasonal channels in order
            chan_ids = [(i, n) for i, n in enumerate(names) if n.startswith(f"{idx_name}_s")]
            if not chan_ids:
                continue
            # sort by season number
            try:
                chan_ids.sort(key=lambda t: int(t[1].split("_s")[-1]))
            except Exception:
                chan_ids.sort(key=lambda t: t[1])
            stack = np.stack([arr_aug[i] for i, _ in chan_ids], axis=0)  # [S,H,W]
            tmin = np.nanmin(stack, axis=0).astype(np.float32)
            tmax = np.nanmax(stack, axis=0).astype(np.float32)
            tmean = np.nanmean(stack, axis=0).astype(np.float32)
            tstd = np.nanstd(stack, axis=0).astype(np.float32)
            trng = (tmax - tmin).astype(np.float32)
            tdiff = (stack[-1] - stack[0]).astype(np.float32)
            arr_aug = np.concatenate([arr_aug,
                                      tmin[None], tmax[None], tmean[None], tstd[None], trng[None], tdiff[None]], axis=0)
            names.extend([f"{idx_name}_tmin", f"{idx_name}_tmax", f"{idx_name}_tmean",
                          f"{idx_name}_tstd", f"{idx_name}_trange", f"{idx_name}_tdiff_last_first"])

    # Neighbor pooling (mean/std over windows for selected channels)
    arr_aug, names = _append_neighbor_pooling(arr_aug, names)

    # Texture features (entropy/contrast/homogeneity proxies)
    arr_aug, names = _append_textures(arr_aug, names)

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
