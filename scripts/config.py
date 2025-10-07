# scripts/config.py
import glob
import os
from typing import Iterable, List, Optional

# --------------------------
# PATHS
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "phase1")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
ROUNDS_DIR = os.path.join(DATA_DIR, "rounds")
LABELS_DIR = os.path.join(BASE_DIR, "labels", "phase1")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint.txt")

for folder in [RAW_DATA_DIR, ROUNDS_DIR, LABELS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

SELECTED_ISLAND: Optional[str] = None


def _normalize_island(name: Optional[str]) -> Optional[str]:
    if not isinstance(name, str):
        return None
    trimmed = name.strip()
    return trimmed.lower() if trimmed else None


def extract_island_name(tile_name: str) -> Optional[str]:
    """Return the island prefix from a tile filename."""
    if not tile_name:
        return None
    name = os.path.basename(tile_name)
    if name.lower().endswith(".tif"):
        name = name[:-4]
    if "_tile" not in name:
        return None
    return name.split("_tile", 1)[0] or None


def discover_islands(raw_dir: Optional[str] = None) -> List[str]:
    """Scan the raw directory and list unique island prefixes."""
    raw_dir = raw_dir or RAW_DATA_DIR
    islands = set()
    try:
        for entry in os.listdir(raw_dir):
            if not entry.lower().endswith(".tif"):
                continue
            island = extract_island_name(entry)
            if island:
                islands.add(island)
    except FileNotFoundError:
        return []
    return sorted(islands)


def set_selected_island(name: Optional[str]) -> None:
    """Store the island filter to be applied across the pipeline."""
    global SELECTED_ISLAND
    SELECTED_ISLAND = name.strip() if isinstance(name, str) and name.strip() else None


def get_selected_island() -> Optional[str]:
    return SELECTED_ISLAND


def tile_matches_island(tile_name: str, island: Optional[str] = None) -> bool:
    target = _normalize_island(island or SELECTED_ISLAND)
    if target is None:
        return True
    tile_island = extract_island_name(tile_name)
    return _normalize_island(tile_island) == target if tile_island else False


def filter_paths_by_island(paths: Iterable[str], island: Optional[str] = None) -> List[str]:
    target = _normalize_island(island or SELECTED_ISLAND)
    if target is None:
        return list(paths)
    filtered = []
    for path in paths:
        name = os.path.basename(path)
        if tile_matches_island(name, target):
            filtered.append(path)
    return filtered


def list_raw_tiles(
    pattern: str = "*.tif",
    raw_dir: Optional[str] = None,
    island: Optional[str] = None,
    base_only: bool = True,
) -> List[str]:
    """Return raw tile paths respecting the selected island filter."""
    raw_dir = raw_dir or RAW_DATA_DIR
    matches = glob.glob(os.path.join(raw_dir, pattern))
    matches = filter_paths_by_island(matches, island=island)
    if base_only:
        base = []
        for path in matches:
            name = os.path.basename(path)
            if "_tile" not in name or "overlay" in name.lower():
                continue
            base.append(path)
        matches = base
    return sorted(matches)


def filter_label_rows(rows: Iterable[dict], island: Optional[str] = None) -> List[dict]:
    """Keep only label rows that belong to the chosen island."""
    target = _normalize_island(island or SELECTED_ISLAND)
    if target is None:
        return list(rows)
    filtered = []
    for row in rows:
        tile = row.get("tile") if isinstance(row, dict) else None
        if tile and tile_matches_island(tile, target):
            filtered.append(row)
    return filtered


MASTER_LABELS_FILE = os.path.join(LABELS_DIR, "labels.csv")
TRAINING_LABELS_FILE = os.path.join(LABELS_DIR, "trainingLabels.csv")
TESTING_LABELS_FILE = os.path.join(LABELS_DIR, "testingLabels.csv")
LABELS_FILE = os.path.join(LABELS_DIR, "labels.csv")
TEMP_LABELS_FILE = os.path.join(LABELS_DIR, "temp_labels.csv")
TRAINING_LABELS_KML = os.path.join(LABELS_DIR, "trainingLabels.kml")
TESTING_LABELS_KML = os.path.join(LABELS_DIR, "testingLabels.kml")
LABELS_KML = TRAINING_LABELS_KML
CANDIDATE_KML = os.path.join(LABELS_DIR, "candidate_patch.kml")
GRID_KML_DIR = os.path.join(LABELS_DIR, "grids")
# Persistent informative pixel sets
# Split control: independent toggles for Highscore vs ProbableAgri
HIGHSCORE_LIST_ENABLED = False       # default ON
PROBABLE_AGRI_LIST_ENABLED = False  # default OFF
HIGHSCORE_FILE = os.path.join(LABELS_DIR, "highscore.csv")
PROBABLE_AGRI_FILE = os.path.join(LABELS_DIR, "probableAgri.csv")
HIGHSCORE_KML_GLOBAL = os.path.join(LABELS_DIR, "highscore_top.kml")
PROBABLE_AGRI_KML_GLOBAL = os.path.join(LABELS_DIR, "probableAgri_top.kml")
# Backward-compatible alias (deprecated): treated as "any list enabled"
PERSISTENT_LISTS_ENABLED = HIGHSCORE_LIST_ENABLED or PROBABLE_AGRI_LIST_ENABLED
FINAL_LABELS_FILE = os.path.join(LABELS_DIR, "finalLabels.csv")
SKIPPED_PIXELS_FILE = os.path.join(LABELS_DIR, "skipped.csv")
if not os.path.exists(GRID_KML_DIR):
    os.makedirs(GRID_KML_DIR, exist_ok=True)

# Path to the Google Earth "My Places" KML. Can be overridden via the
# ``MYPLACES_KML`` environment variable.
if os.name == "nt":
    _default_myplaces = os.path.join(os.path.expanduser("~"),
                                     "AppData", "LocalLow",
                                     "Google", "GoogleEarth",
                                     "myplaces.kml")
else:
    _default_myplaces = os.path.join(os.path.expanduser("~"),
                                     ".googleearth", "myplaces.kml")
MYPLACES_KML = os.environ.get("MYPLACES_KML", _default_myplaces)

# --------------------------
# DATA DOWNLOAD CONFIGURATION (GEE)
# --------------------------
ROI_COORDS = []
TIMESTAMPS = [
    ("2024-01-01", "2024-04-30"),
    ("2024-05-01", "2024-08-31"),
    ("2024-09-01", "2024-12-31"),
]
CLOUDY_PIXEL_PERCENTAGE = 100
BANDS = [
    # Sentinel-2 SR bands to include (exclude B10: cirrus)
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
]
# Derived indices computed per season in export
# Keep existing ones and add new ones verified for coverage in GEE checks
INDICES = [
    "NDVI", "EVI", "EVI2", "NBR", "NDMI",    # existing
    "NDRE", "GNDVI", "OSAVI", "NDWI", "BSI"  # new
]

# Google Cloud Storage export target for Earth Engine jobs
GCS_BUCKET = os.environ.get("EE_GCS_BUCKET", "fcup-thesis-2025")
GCS_PATH_PREFIX = os.environ.get("EE_GCS_PATH_PREFIX", "2024")

# --------------------------
# INITIAL LABELING CONFIG
# --------------------------
MIN_AGRI_COUNT = 10
MIN_AGRI_RATIO = 0
MAX_AGRI_RATIO = 1
DUPLICATE_TOLERANCE = 0.0001

# --------------------------
# ACTIVE LEARNING CONFIG
# --------------------------
SUPPORTED_MODELS = ["SVM", "RandomForest", "Ensemble"]  # Available model backends
ENSEMBLE_EXPORT_BASE_MODELS = False  # Skip per-base inference/exports when using ensemble
SVM_PARAMS = {  # C≈0.5–10; gamma: "scale"/"auto"; cache_size in MB
    "C": 3.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
    "cache_size": 2048,
}
RF_PARAMS = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1, "class_weight": "balanced"}  # trees≈200–400; depth≈8–14

# Ensemble (stacking) configuration
ENSEMBLE_BASE_MODELS = ("SVM", "RandomForest")
ENSEMBLE_STACKING_FOLDS = 3          # Out-of-fold predictions to train the stacking head
ENSEMBLE_LOGREG_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 200,
}
ENSEMBLE_RUNTIME_STATS_ENABLED = True  # Aggregate base vs ensemble contributions during inference

# Auto-tuning (warm-start mini grid)
AUTO_TUNE_HISTORY_FILE = os.path.join(BASE_DIR, "auto_tuning_history.json")
AUTO_TUNE_MAX_COMBOS = 6  # cap per model per round to control runtime
AUTO_TUNE_SVM_SEEDS = [
    {
        "SVM_PARAMS": {"C": 1.0, "gamma": "scale", "class_weight": "balanced"},
        "MIN_AGRI_PROB": 0.33,
        "SIEVE_MIN_SIZE": 5,
    },
    {
        "SVM_PARAMS": {"C": 3.0, "gamma": "scale", "class_weight": "balanced"},
        "MIN_AGRI_PROB": 0.35,
        "SIEVE_MIN_SIZE": 5,
    },
    {
        "SVM_PARAMS": {"C": 5.0, "gamma": "auto", "class_weight": "balanced"},
        "MIN_AGRI_PROB": 0.37,
        "SIEVE_MIN_SIZE": 5,
    },
]
AUTO_TUNE_RF_SEEDS = [
    {
        "RF_PARAMS": {"n_estimators": 250, "max_depth": 10, "min_samples_leaf": 1, "class_weight": "balanced"},
        "MIN_AGRI_PROB": 0.34,
        "SIEVE_MIN_SIZE": 5,
    },
    {
        "RF_PARAMS": {"n_estimators": 350, "max_depth": 12, "min_samples_leaf": 2, "class_weight": "balanced"},
        "MIN_AGRI_PROB": 0.35,
        "SIEVE_MIN_SIZE": 5,
    },
    {
        "RF_PARAMS": {"n_estimators": 450, "max_depth": 14, "min_samples_leaf": 2, "class_weight": "balanced"},
        "MIN_AGRI_PROB": 0.36,
        "SIEVE_MIN_SIZE": 7,
    },
]
AUTO_TUNE_ENSEMBLE_SEEDS = [
    {
        "SVM_PARAMS": {"C": 3.0, "gamma": "scale", "class_weight": "balanced"},
        "RF_PARAMS": {"n_estimators": 350, "max_depth": 12, "min_samples_leaf": 2, "class_weight": "balanced"},
        "MIN_AGRI_PROB": 0.35,
        "SIEVE_MIN_SIZE": 5,
    },
    {
        "SVM_PARAMS": {"C": 5.0, "gamma": "auto", "class_weight": "balanced"},
        "RF_PARAMS": {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 2, "class_weight": "balanced"},
        "MIN_AGRI_PROB": 0.36,
        "SIEVE_MIN_SIZE": 5,
    },
]
AUTO_TUNE_SVM_C_FACTORS = [0.5, 1.5]
AUTO_TUNE_SVM_NUMERIC_GAMMA_FACTORS = [0.5, 2.0]
AUTO_TUNE_THRESHOLD_DELTA = 0.02
AUTO_TUNE_SIEVE_STEPS = [0, 2]
AUTO_TUNE_RF_ESTIMATOR_STEP = 100
AUTO_TUNE_RF_DEPTH_STEP = 2
AUTO_TUNE_RF_LEAF_OPTIONS = [1, 2, 3]

# Splits
TRAIN_FRACTION = 0.7  # 0.6–0.8 typical; must satisfy TRAIN+VAL+TEST ≤ 1.0
VAL_FRACTION = 0.3    # 0.2–0.4 typical; used for evaluation in a3
TEST_FRACTION = 0.0   # Often 0 in phase-1; enable only if needed
SPLIT_SEED_MODE = "random"  # "random" | "fixed". Fixed yields repeatable splits
SPLIT_RANDOM_SEED = 42      # Used when SPLIT_SEED_MODE == "fixed"

# Cross-validation
CV_FOLDS = 5               # 3–5 typical; more folds = more compute
CV_AUTO_REDUCE = True      # Reduce folds to ≥2 per minority class when data is small

# Calibration
CALIBRATION_METHOD = "isotonic"  # "sigmoid" fast; "isotonic" needs more data
CALIBRATION_FOLDS = 3           # 3–5 typical

# Candidate selection
NUM_CANDIDATES_PER_ROUND = 50         # 20–50 typical (depends on label capacity)
CANDIDATE_PROB_LOWER = 0.28              # Must be ≤ MIN_AGRI_PROB; defines lower bound of candidate band
CANDIDATE_DBSCAN_EPS_KM = 0.8           # 0.5–3.0 km typical; spatial diversity
CANDIDATE_NEGATIVE_QUOTA = 0.40         # 0 disables; use 20–40% when chasing hard negatives
NEG_LIKE_PROB_RANGE = (0.28, 0.36)      # Base/fallback; dynamically adjusted from MIN_AGRI_PROB via NEG_LIKE_PROB_DELTA
NEG_LIKE_PROB_DELTA = 0.05             # Effective prob range = (MIN_AGRI_PROB - DELTA, MIN_AGRI_PROB)
NEG_LIKE_NDVI_RANGE = (0.10, 0.60)        # Agri-like NDVI window (tune per region)
NEG_LIKE_NDVI_RELATIVE = True         # If True, use NDVI percentiles from current predictions
NEG_LIKE_NDVI_PERC_RANGE = (0.6, 0.9)  # Percentile window when RELATIVE=True (e.g., 60th–90th)

# Composite ranking weights
UNCERTAINTY_BAND_DELTA = 0.05  # Treat |p-MIN_AGRI_PROB| < delta as uncertain for persistence weighting
HIGHSCORE_TOP_K = 10000        # 0 or less => no size limit; otherwise keep top-K informative pixels
HIGHSCORE_COMPONENT_WEIGHTS = {"uncertainty": 0.5, "representativeness": 0.3, "consistency": 0.2}  # sum≈1
PROBABLE_AGRI_TOP_K = 10000    # 0 or less => no size limit; otherwise keep top-K probable-agri pixels
PROBABLE_AGRI_COMPONENT_WEIGHTS = {"confidence": 0.7, "representativeness": 0.3}
HIGHSCORE_KML_TOP_PIXELS = 10000       # Cap per-pixel KML to avoid huge files (0 disables cap)
PROBABLE_AGRI_KML_TOP_PIXELS = 10000   # Cap per-pixel KML to avoid huge files (0 disables cap)

# Model training
MIN_AGRI_PROB = 0.35      # Decision threshold (Orange ≥ this). Ensure CANDIDATE_PROB_LOWER ≤ this
AUTO_USE_BEST_THRESHOLD = True  # If True, skip manual threshold prompt and adopt round best-th for outputs
AUTO_CANDIDATE_PROB_MARGIN = 0.10   # Offset added to the round threshold when deriving the candidate floor
AUTO_CANDIDATE_PROB_MIN = 0.10      # Lower bound for the candidate floor under auto threshold mode
AUTO_NEG_LIKE_DELTA_MIN = 0.05      # Minimum probability band for negative-like picks under auto mode
# Inference performance
INFER_CHUNKING_ENABLED = True          # Improves stability on large tiles; no effect on results
INFER_MAX_PIXELS_PER_BATCH = 1_000_000     # Smaller batches to reduce peak RAM
FEATURE_CACHE_ENABLED = True           # Cache derived features per tile to disk
FEATURE_CACHE_DIR = os.path.join(DATA_DIR, "cache")
FEATURE_CACHE_MAX_TILES_IN_MEMORY = 8    # Keep at most 2 tiles in memory to cap usage

# Feature selection & importance
FEATURE_SET = "base"  # one of: base, temporal_only, textures_only, temporal_textures, full
TEXTURE_WINDOW_SIZE = 7
RUN_PERMUTATION_IMPORTANCE = True
PERMUTATION_IMPORTANCE_JOBS = 1   # Use 1 to avoid heavy parallel forks on Windows/WSL
REPEATED_VALIDATION_REPEATS = 10   # >1 enables repeated validation with mean/std aggregation

# --------------------------
# POSTPROCESSING CONFIG
# --------------------------
SIEVE_MIN_SIZE = 5
SIEVE_KEEP_PROB = 0.80            # Must be > MIN_AGRI_PROB. Red (very certain) ≥ this; Orange ∈ [MIN_AGRI_PROB, SIEVE_KEEP_PROB)
SIEVE_KEEP_MODE = "pixel"      # "component" keeps whole component if any pixel ≥ SIEVE_KEEP_PROB; else "pixel" keeps only high-prob pixels
SIEVE_USE_KEEP_PROB = False         # If True, apply SIEVE_KEEP_PROB in component/pixel rules; else fall back to MIN_AGRI_PROB

# Final sweep (thresholds, sieve, morphology)
FINAL_SWEEP_ENABLED = False
FINAL_THRESHOLDS = [0.28, 0.33, 0.38]  # Sweep around MIN_AGRI_PROB
FINAL_SIEVE_SIZES = [0, 5, 10]         # Include a wider sieve range for robustness
FINAL_ROUND_ENABLED = False  # Skip final grid search/inference when False
# Final-round combos only sweep threshold (th) and sieve (s); morphology disabled
FINAL_MORPH_OPEN = False
FINAL_MORPH_KERNEL_SIZES = [3]

# KML output limits (to keep files usable in Google Earth)
KML_MAX_POINTS = 0  # Deprecated cap (kept for backward compat; not used)

# --------------------------
# LABEL NOTES OPTIONS
# --------------------------
NOTE_OPTIONS = [
    "Agricultural Certain",
    "Agricultural Doubtful",
    "Open Field / Tree",
    "Building / Man Made",
    "Water Bodies",
    "Other",
]

# --------------------------
# PERFORMANCE / TUNING TOGGLES
# --------------------------
# Inference batch tuning per model (overrides are clamped by INFER_MAX_PIXELS_PER_BATCH)
AUTO_BATCH_TUNING_ENABLED = True
INFER_BATCH_OVERRIDE_SVM = 1_000_000
INFER_BATCH_OVERRIDE_RANDOMFOREST = 400_000
INFER_TILE_THREADS = 16         # Fewer concurrent tiles to prevent RAM spikes

# Refresh/per-tile metrics optimization
REFRESH_CHUNK_ROWS = 600_000        # Rows per chunk when computing per-tile metrics
REFRESH_TILE_THREADS = 2            # Limit parallelism during refresh steps
REFRESH_KD_WORKERS = 4              # Lower KDTree workers to reduce memory pressure
GZIP_COMPRESSLEVEL = 0              # 1–3 is fast; higher compresses more but is slower

# Polygonisation tuning
POLYGONIZE_WORKERS = 1              # Process workers for tile polygonisation (1 => sequential)

# ANN/hnswlib removed: representativeness uses exact sklearn NN only.

# --------------------------
# CSV/VECTORIZATION & I/O TUNING
# --------------------------
# Chunk size for scanning predictions.csv during candidate selection
PREDICTIONS_CSV_CHUNK_ROWS = 500_000

# Optional binary sidecars (.npy) to accelerate k-way merges
BINARY_SIDECARS_ENABLED = True

# --------------------------
# BLAS/NUMERICAL THREADS (ENV)
# --------------------------
# Set default BLAS/OpenMP thread caps for NumPy/Scikit
BLAS_NUM_THREADS = 8
OMP_NUM_THREADS = BLAS_NUM_THREADS
MKL_NUM_THREADS = BLAS_NUM_THREADS
OPENBLAS_NUM_THREADS = BLAS_NUM_THREADS
NUMEXPR_NUM_THREADS = BLAS_NUM_THREADS

# --------------------------
# GDAL / Rasterio tuning
# --------------------------
GDAL_CACHEMAX_MB = 512
GDAL_NUM_THREADS = 'ALL_CPUS'

# --------------------------
# TERRAIN / DEM FEATURES
# --------------------------
# Always include per-season copies of DEM‐derived terrain attributes to keep
# the exported stack season-aligned with spectral inputs.
TERRAIN_BANDS = ["ELEVATION", "SLOPE", "ASPECT"]

# --------------------------
# FEATURE EXCLUSION (OPTIONAL)
# --------------------------
# To temporarily exclude certain spectral bands from features (without
# re-exporting GeoTIFFs), list their base names here (e.g., ["B9", "B1"]).

# --------------------------
# OUTPUT TOGGLES
# --------------------------
# Emit best-threshold advisory statistics and KML alongside normal-threshold outputs
BEST_THRESHOLD_OUTPUTS_ENABLED = True
# This affects training/inference and candidate selection features only; the
# raw tiles remain unchanged.
EXCLUDED_BANDS = ["B1", "B9"]

# Compact console summary of persistent lists (Highscore) at startup
HIGHSCORE_SUMMARY_ENABLED = True

# Assisted labeling spatial diversity (DBSCAN haversine over lat/lon)
ASSISTED_SPATIAL_DIVERSITY_ENABLED = True
# Use same default distance as candidate selection; can override if needed
ASSISTED_DIVERSITY_EPS_KM = CANDIDATE_DBSCAN_EPS_KM

# --------------------------
# MEMORY & PROCESS POOL BATCHING
# --------------------------
MEMORY_WATCHER_ENABLED = True
MEMORY_WATCHER_THRESHOLD_PERCENT = 70   # Free Python memory earlier under load
MEMORY_WATCHER_INTERVAL_SEC = 5         # Seconds between checks

# Batch size for process pools (number of tiles per pool cycle) to limit peak RSS
REFRESH_PROCESS_POOL_BATCH = 20
INFER_BLOCK_SIZE = 512                  # Smaller blocks => lower transient memory

# --------------------------
# DEDUP (PERSISTENT LISTS) CONFIG
# --------------------------
# Chunk size (rows) for external sorts during dedup
DEDUP_CHUNK_ROWS = 1_000_000 # Smaller chunks to keep dedup RAM bounded
DEDUP_SORT_WORKERS = 12
