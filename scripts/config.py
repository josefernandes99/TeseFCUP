# scripts/config.py
import os

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

LABELS_FILE = os.path.join(LABELS_DIR, "labels.csv")
TEMP_LABELS_FILE = os.path.join(LABELS_DIR, "temp_labels.csv")
LABELS_KML = os.path.join(LABELS_DIR, "labels.kml")
CANDIDATE_KML = os.path.join(LABELS_DIR, "candidate_patch.kml")
GRID_KML_DIR = os.path.join(LABELS_DIR, "grids")
# Persistent informative pixel sets
# Toggle: enable or disable generation/use of Highscore and ProbableAgri lists
PERSISTENT_LISTS_ENABLED = True  # Set to False to disable all related processes
HIGHSCORE_FILE = os.path.join(LABELS_DIR, "highscore.csv")
PROBABLE_AGRI_FILE = os.path.join(LABELS_DIR, "probableAgri.csv")
HIGHSCORE_KML_GLOBAL = os.path.join(LABELS_DIR, "highscore_top.kml")
PROBABLE_AGRI_KML_GLOBAL = os.path.join(LABELS_DIR, "probableAgri_top.kml")
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
    ("2019-01-01", "2019-04-30"),
    ("2019-05-01", "2019-08-31"),
    ("2019-09-01", "2019-12-31"),
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
SUPPORTED_MODELS = ["ResNet", "SVM", "RandomForest", "XGBoost"]  # Available model backends
SVM_PARAMS = {  # C≈0.5–10; gamma: "scale"/"auto"; cache_size in MB
    "C": 3.0,
    "kernel": "rbf",
    "gamma": "scale",
    "class_weight": "balanced",
    "cache_size": 2048,
}
RF_PARAMS = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1, "class_weight": "balanced"}  # trees≈200–400; depth≈8–14
XGB_PARAMS = {  # Tuned for speed+accuracy; can be grid-searched
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}

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
NUM_CANDIDATES_PER_ROUND = 25          # 20–50 typical (depends on label capacity)
CANDIDATE_PROB_LOWER = 0.30              # Must be ≤ MIN_AGRI_PROB; defines lower bound of candidate band
CANDIDATE_DBSCAN_EPS_KM = 1.5           # 0.5–3.0 km typical; spatial diversity
CANDIDATE_NEGATIVE_QUOTA = 0.30         # 0 disables; use 10–30% of candidates when enabled
NEG_LIKE_PROB_RANGE = (0.35, 0.4)      # Base/fallback; dynamically adjusted from MIN_AGRI_PROB via NEG_LIKE_PROB_DELTA
NEG_LIKE_PROB_DELTA = 0.05             # Effective prob range = (MIN_AGRI_PROB - DELTA, MIN_AGRI_PROB)
NEG_LIKE_NDVI_RANGE = (0.15, 0.45)        # Agri-like NDVI window (tune per region)
NEG_LIKE_NDVI_RELATIVE = True         # If True, use NDVI percentiles from current predictions
NEG_LIKE_NDVI_PERC_RANGE = (0.6, 0.9)  # Percentile window when RELATIVE=True (e.g., 60th–90th)

# Composite ranking weights
UNCERTAINTY_BAND_DELTA = 0.05  # Treat |p-MIN_AGRI_PROB| < delta as uncertain for persistence weighting
HIGHSCORE_TOP_K = 0            # 0 or less => no size limit; otherwise keep top-K informative pixels
HIGHSCORE_COMPONENT_WEIGHTS = {"uncertainty": 0.5, "representativeness": 0.3, "consistency": 0.2}  # sum≈1
PROBABLE_AGRI_TOP_K = 0        # 0 or less => no size limit; otherwise keep top-K probable-agri pixels
PROBABLE_AGRI_COMPONENT_WEIGHTS = {"confidence": 0.7, "representativeness": 0.3}
HIGHSCORE_KML_TOP_PIXELS = 50000        # Cap per-pixel KML to avoid huge files (0 disables cap)
PROBABLE_AGRI_KML_TOP_PIXELS = 50000    # Cap per-pixel KML to avoid huge files (0 disables cap)

# Model training
RESNET_EPOCHS = 10       # 5–20 typical
RESNET_LR = 0.001        # 1e-4–3e-3 typical
BATCH_SIZE = 32          # Tune to memory
MIN_AGRI_PROB = 0.35      # Decision threshold (Orange ≥ this). Ensure CANDIDATE_PROB_LOWER ≤ this

# Inference performance
INFER_CHUNKING_ENABLED = True          # Improves stability on large tiles; no effect on results
INFER_MAX_PIXELS_PER_BATCH = 1_500_000   # 100k–500k typical; adjust to RAM
FEATURE_CACHE_ENABLED = True           # Cache derived features per tile to disk
FEATURE_CACHE_DIR = os.path.join(DATA_DIR, "cache")
FEATURE_CACHE_MAX_TILES_IN_MEMORY = 10   # LRU bound to avoid RAM blow-outs

# Feature selection & importance
FEATURE_SET = "base"  # one of: base, temporal_only, textures_only, temporal_textures, full
TEXTURE_WINDOW_SIZE = 7
RUN_PERMUTATION_IMPORTANCE = True
REPEATED_VALIDATION_REPEATS = 5   # >1 enables repeated validation with mean/std aggregation

# --------------------------
# POSTPROCESSING CONFIG
# --------------------------
SIEVE_MIN_SIZE = 10
SIEVE_KEEP_PROB = 0.85            # Must be > MIN_AGRI_PROB. Red (very certain) ≥ this; Orange ∈ [MIN_AGRI_PROB, SIEVE_KEEP_PROB)
SIEVE_KEEP_MODE = "pixel"      # "component" keeps whole component if any pixel ≥ SIEVE_KEEP_PROB; else "pixel" keeps only high-prob pixels
SIEVE_USE_KEEP_PROB = True         # If True, apply SIEVE_KEEP_PROB in component/pixel rules; else fall back to MIN_AGRI_PROB

# Final sweep (thresholds, sieve, morphology)
FINAL_SWEEP_ENABLED = True
FINAL_THRESHOLDS = [0.30, 0.35, 0.40]  # Sweep around MIN_AGRI_PROB
FINAL_SIEVE_SIZES = [0, 5, 10]         # Include a wider sieve range for robustness
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
INFER_BATCH_OVERRIDE_SVM = 1_500_000
INFER_BATCH_OVERRIDE_RANDOMFOREST = 400_000
INFER_BATCH_OVERRIDE_XGBOOST = 500_000
INFER_BATCH_OVERRIDE_RESNET = 200_000
INFER_TILE_THREADS = 10        # Number of tiles processed in parallel during inference

# Refresh/per-tile metrics optimization
REFRESH_CHUNK_ROWS = 300_000        # Rows per chunk when computing per-tile metrics
REFRESH_TILE_THREADS = 2            # Number of tiles processed in parallel (threads)
REFRESH_KD_WORKERS = 6              # cKDTree internal workers per query (parallel in C)
GZIP_COMPRESSLEVEL = 1              # 1–3 is fast; higher compresses more but is slower

# ANN/hnswlib removed: representativeness uses exact sklearn NN only.

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
# This affects training/inference and candidate selection features only; the
# raw tiles remain unchanged.
EXCLUDED_BANDS = ["B1", "B9"]
