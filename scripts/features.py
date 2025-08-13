import os
import glob
import csv
import hashlib
import itertools
import json
import tempfile
import zlib
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import rasterio
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeRemainingColumn
from numba import njit, prange
from filelock import FileLock

from config import RAW_DATA_DIR, BANDS, INDICES, BASE_DIR

# Texture feature names used outside Numba-compiled functions.  Keep a plain
# Python tuple and a separate constant for the feature count so that the Numba
# kernels don't depend on Python objects that would otherwise be treated as
# "reflected" lists, which are unsupported at compile time.
GLCM_PROPS = (
    "contrast",
    "dissimilarity",
    "idm",
    "entropy",
    "inertia",
    "shade",
    "sum_avg",
)
NUM_GLCM_PROPS = len(GLCM_PROPS)

# Texture bands for which GLCM metrics are computed.  The SWIR bands are
# referred to by their common "SWIR1"/"SWIR2" names to match the derived feature
# names appended later.
TEXTURE_BANDS = ["NDVI", "EVI", "NBR", "SWIR1", "SWIR2"]

# Expected feature names after deriving textures and aspect sine/cosine.
# This is used to validate cached feature stacks and to know the total
# feature count without needing to read a tile first.
BASE_FEATURE_NAMES = (
    BANDS + [i for i in INDICES if i != "ASPECT"] + ["ASPECT_SIN", "ASPECT_COS"]
)
EXPECTED_FEATURE_NAMES = (
    BASE_FEATURE_NAMES
    + [f"{band}_{prop}" for band in TEXTURE_BANDS for prop in GLCM_PROPS]
)
EXPECTED_FEATURE_COUNT = len(EXPECTED_FEATURE_NAMES)

CACHE_ROOT = Path(BASE_DIR) / "data" / "cache"
GLCM_CACHE_DIR = CACHE_ROOT / "glcm"
NORMALIZER_DIR = CACHE_ROOT / "normalizers"
GLCM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZER_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZER_FILE = NORMALIZER_DIR / "normalizers.csv"

# Version stamp for cached feature files. Increment to invalidate existing
# caches when the feature layout changes.
CACHE_VERSION = 1


def _save_npz_atomic(path: str | Path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # create a writable temp file in the same directory
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False, mode="wb"
    ) as f:
        tmp_name = f.name
        # write using the open file handle so fsync is valid on Windows
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())

    # atomic replace
    os.replace(tmp_name, path)


def _meta_path(npz_path):
    return Path(str(npz_path) + ".meta.json")


def _write_meta(npz_path, arr):
    meta = {
        "version": CACHE_VERSION,
        "bands": int(arr.shape[0]),
        "dtype": str(arr.dtype),
    }
    _meta_path(npz_path).write_text(json.dumps(meta), encoding="utf-8")


def _meta_valid(npz_path):
    mp = _meta_path(npz_path)
    if not mp.exists():
        return False
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return False
    if meta.get("version") != CACHE_VERSION:
        return False
    if meta.get("bands") != EXPECTED_FEATURE_COUNT:
        return False
    if meta.get("dtype") != "float32":
        return False
    return True


def _tif_fingerprint(tifs):
    """Generate a fingerprint string from tile paths and mtimes."""
    parts = []
    for tp in sorted(tifs):
        parts.append(f"{os.path.basename(tp)}:{int(os.path.getmtime(tp))}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()


def _list_tifs():
    return [
        tp
        for tp in glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
        if not tp.endswith("_overlay.tif")
    ]


def feature_cache_path(tile_path: str | Path, norm_scheme: str = "global") -> Path:
    """Return deterministic cache path for a tile."""
    tile_id = Path(tile_path).stem
    key = f"{tile_id}__glcm5x5__norm{norm_scheme}__v{CACHE_VERSION}.npz"
    return GLCM_CACHE_DIR / key


@njit
def _glcm_patch_features(patch, levels):
    glcm = np.zeros((levels, levels), dtype=np.float32)
    h, w = patch.shape
    for i in range(h):
        for j in range(w - 1):
            a = patch[i, j]
            b = patch[i, j + 1]
            glcm[a, b] += 1.0
            glcm[b, a] += 1.0
    glcm /= glcm.sum()

    contrast = 0.0
    dissimilarity = 0.0
    idm = 0.0
    entropy = 0.0
    inertia = 0.0
    mu_i = 0.0
    mu_j = 0.0
    for i in range(levels):
        for j in range(levels):
            p = glcm[i, j]
            diff = i - j
            contrast += diff * diff * p
            dissimilarity += abs(diff) * p
            idm += p / (1.0 + diff * diff)
            entropy -= p * np.log(p + 1e-10)
            inertia += diff * diff * p
            mu_i += i * p
            mu_j += j * p

    shade = 0.0
    for i in range(levels):
        for j in range(levels):
            p = glcm[i, j]
            shade += ((i - mu_i) + (j - mu_j)) ** 3 * p

    p_sum = np.zeros(2 * levels - 1, dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            p_sum[i + j] += glcm[i, j]
    sum_avg = 0.0
    for k in range(p_sum.shape[0]):
        sum_avg += (k + 2) * p_sum[k]

    return np.array(
        [contrast, dissimilarity, idm, entropy, inertia, shade, sum_avg],
        dtype=np.float32,
    )


@njit(parallel=True)
def _compute_glcm_stack_nb(scaled, window, levels):
    pad = window // 2
    H, W = scaled.shape
    padded = np.empty((H + 2 * pad, W + 2 * pad), dtype=np.uint8)
    padded[pad:pad + H, pad:pad + W] = scaled
    for i in range(pad):
        padded[i, pad:pad + W] = scaled[0]
        padded[H + pad + i, pad:pad + W] = scaled[-1]
    for j in range(pad):
        padded[:, j] = padded[:, pad]
        padded[:, W + pad + j] = padded[:, W + pad - 1]

    # Allocate the output volume using the constant feature count rather than
    # referencing the Python tuple above so that Numba sees a simple integer
    # and avoids compiling a reflected Python object.
    feats = np.zeros((NUM_GLCM_PROPS, H, W), dtype=np.float32)
    for i in prange(H):
        for j in range(W):
            patch = padded[i : i + window, j : j + window]
            feats[:, i, j] = _glcm_patch_features(patch, levels)
    return feats


def compute_glcm_stack(band, window=5, levels=32, prog=None, task=None):
    vmin = np.nanmin(band)
    vmax = np.nanmax(band)
    if np.isclose(vmax, vmin):
        scaled = np.zeros_like(band, dtype=np.uint8)
    else:
        scaled = np.round((band - vmin) / (vmax - vmin + 1e-6) * (levels - 1)).astype("uint8")
    feats = _compute_glcm_stack_nb(scaled, window, levels)
    if prog is not None and task is not None:
        prog.update(task, advance=band.shape[0])
    return feats


def add_derived_features(arr, prog=None, task=None):
    band_order = BANDS + INDICES
    aspect_idx = band_order.index("ASPECT")
    aspect = arr[aspect_idx]
    arr_no_aspect = np.delete(arr, aspect_idx, axis=0)
    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)
    arr_aug = np.concatenate([arr_no_aspect, aspect_sin[None], aspect_cos[None]], axis=0)
    names = BANDS + [i for i in INDICES if i != "ASPECT"] + ["ASPECT_SIN", "ASPECT_COS"]

    # Map texture band names to the index of the source layer from which the
    # textures will be computed.  The SWIR bands use their spectral band names
    # (B11/B12) as the source but are exported as SWIR1/SWIR2 in the feature
    # stack to remain human-readable and stable.
    band_indices = {
        "NDVI": names.index("NDVI"),
        "EVI": names.index("EVI"),
        "NBR": names.index("NBR"),
        "SWIR1": names.index("B11"),
        "SWIR2": names.index("B12"),
    }

    def _run(p, t):
        nonlocal arr_aug, names
        for bname in TEXTURE_BANDS:
            idx = band_indices[bname]
            tex = compute_glcm_stack(arr_aug[idx], prog=p, task=t)
            arr_aug = np.concatenate([arr_aug, tex], axis=0)
            names.extend([f"{bname}_{prop}" for prop in GLCM_PROPS])
        return arr_aug, names
    if prog is not None and task is not None:
        return _run(prog, task)
    return _run(None, None)


def load_cached_features(cache_path, arr=None, prog=None, task=None):
    """Load derived features from cache or compute and save them.

    Returns
    -------
    arr_aug : np.ndarray
        Augmented feature stack.
    names : list[str]
        Feature names corresponding to ``arr_aug``.
    used_cache : bool
        ``True`` if features were loaded from an existing cache file.
    """
    cache_path = Path(cache_path)
    lock = FileLock(str(cache_path) + ".lock")
    with lock:
        if cache_path.exists() and _meta_valid(cache_path):
            try:
                with np.load(cache_path) as data:
                    arr_aug = data["arr"]
                    names = data["names"].tolist()
            except (BadZipFile, zlib.error, OSError, ValueError, KeyError):
                arr_aug = None
            else:
                if (
                    arr_aug.shape[0] == EXPECTED_FEATURE_COUNT
                    and names == list(EXPECTED_FEATURE_NAMES)
                ):
                    if prog is not None and task is not None:
                        prog.update(task, advance=arr_aug.shape[1] * len(TEXTURE_BANDS))
                    return arr_aug, names, True
            for p in [cache_path, _meta_path(cache_path)]:
                try:
                    os.remove(p)
                except OSError:
                    pass

        if arr is None:
            raise ValueError("arr must be provided when cache is missing")
        arr_aug, names = add_derived_features(arr, prog=prog, task=task)
        _save_npz_atomic(cache_path, arr=arr_aug, names=np.array(names))
        _write_meta(cache_path, arr_aug)
        return arr_aug, names, False


def compute_normalizers():
    tifs = _list_tifs()
    fingerprint = _tif_fingerprint(tifs)
    heights = []
    for tp in tifs:
        with rasterio.open(tp) as src:
            heights.append(src.height)
    total_rows = sum(heights) * len(TEXTURE_BANDS)
    sums = sqs = counts = None
    feat_names = None
    with Progress(
        "[bold cyan]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as prog:
        task = prog.add_task(
            f"Scanning tiles for global normalizers 0/{len(tifs)}", total=total_rows
        )
        completed = 0
        for tp in tifs:
            prog.update(
                task,
                description=f"Scanning tiles for global normalizers {completed}/{len(tifs)}",
            )
            with rasterio.open(tp) as src:
                cache_path = feature_cache_path(tp)
                arr_aug, names, _ = load_cached_features(
                    cache_path, arr=src.read().astype(np.float32), prog=prog, task=task
                )
            n_feat = arr_aug.shape[0]
            flat = arr_aug.reshape(n_feat, -1)
            if sums is None:
                feat_names = names
                sums = np.zeros(n_feat, dtype=np.float64)
                sqs = np.zeros(n_feat, dtype=np.float64)
                counts = np.zeros(n_feat, dtype=np.int64)
            elif n_feat != sums.shape[0]:
                raise ValueError(
                    f"Feature count mismatch: expected {sums.shape[0]}, got {n_feat}"
                )
            valid = np.isfinite(flat)
            counts += valid.sum(axis=1)
            flat[~valid] = 0
            sums += flat.sum(axis=1)
            sqs += (flat ** 2).sum(axis=1)
            completed += 1
        prog.update(
            task,
            description=f"Scanning tiles for global normalizers {completed}/{len(tifs)}",
        )
    means = sums / counts
    vars_ = sqs / counts - means ** 2
    stds = np.sqrt(np.clip(vars_, 0, None))
    with open(NORMALIZER_FILE, "w", newline="") as f:
        f.write(f"# fingerprint,{fingerprint}\n")
        writer = csv.writer(f)
        writer.writerow(["feature", "mean", "std"])
        for n, m, s in zip(feat_names, means, stds):
            writer.writerow([n, m, s])
    return feat_names, means, stds


def load_normalizers():
    with open(NORMALIZER_FILE, "r") as f:
        first = f.readline().strip()
        if first.startswith("# fingerprint"):
            fingerprint = first.split(",", 1)[1]
            reader = csv.DictReader(f)
        else:
            fingerprint = None
            reader = csv.DictReader(itertools.chain([first], f))
        names, means, stds = [], [], []
        for row in reader:
            names.append(row["feature"])
            means.append(float(row["mean"]))
            stds.append(float(row["std"]))
    return fingerprint, names, np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def get_normalizers():
    """Return global feature means and stds, recomputing if stale."""
    if not os.path.exists(NORMALIZER_FILE):
        # No statistics cached yet; compute from scratch.
        return compute_normalizers()

    fingerprint, names, means, stds = load_normalizers()
    # If the saved stats don't match the currently expected feature layout,
    # recompute them to stay in sync with whatever feature set is produced by
    # ``load_cached_features``.
    if (
        len(names) != EXPECTED_FEATURE_COUNT
        or names != list(EXPECTED_FEATURE_NAMES)
    ):
        return compute_normalizers()
    current_fp = _tif_fingerprint(_list_tifs())
    if fingerprint != current_fp:
        return compute_normalizers()
    return names, means, stds
