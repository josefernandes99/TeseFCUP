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
HIGHSCORE_FILE = os.path.join(LABELS_DIR, "highscore.csv")
PROBABLE_AGRI_FILE = os.path.join(LABELS_DIR, "probableAgri.csv")
HIGHSCORE_KML_GLOBAL = os.path.join(LABELS_DIR, "highscore_top.kml")
PROBABLE_AGRI_KML_GLOBAL = os.path.join(LABELS_DIR, "probableAgri_top.kml")
FINAL_LABELS_FILE = os.path.join(LABELS_DIR, "finalLabels.csv")
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
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
INDICES = ["NDVI", "EVI", "EVI2", "NBR", "NDMI", "ELEVATION", "SLOPE", "ASPECT"]

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
SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}  # C≈0.5–10; gamma: "scale"/"auto"
RF_PARAMS = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1, "class_weight": "balanced"}  # trees≈200–400; depth≈8–14
# XGBoost (GPU‑friendly). tree_method: 'hist' (CPU) or 'gpu_hist' (GPU); predictor auto‑selects.
# scale_pos_weight will be computed dynamically per round (neg/pos) if XGB_USE_AUTO_SPW is True.
XGB_USE_GPU = True
XGB_USE_AUTO_SPW = True
XGB_PARAMS = {
    "n_estimators": 400,       # 300–600 typical
    "learning_rate": 0.1,     # 0.05–0.15 typical
    "max_depth": 8,           # 6–10 typical; controls interaction complexity
    "min_child_weight": 4,    # regularization; larger => simpler trees
    "subsample": 0.8,         # row sampling per tree
    "colsample_bytree": 0.8,  # feature sampling per tree
    "reg_lambda": 2.0,        # L2 regularization (lambda)
    "reg_alpha": 0.0,         # L1 regularization (alpha)
    "tree_method": "gpu_hist" if True else "hist",  # auto‑overridden by XGB_USE_GPU
    "predictor": "auto",
    "random_state": 42,
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
VAL_REPEATS = 1            # Number of repeated validation splits for reporting (>=1). Seeds recorded.

# Calibration
CALIBRATION_METHOD = "sigmoid"  # "sigmoid" fast; "isotonic" needs more data
CALIBRATION_FOLDS = 3           # 3–5 typical

# Candidate selection
NUM_CANDIDATES_PER_ROUND = 25          # 20–50 typical (depends on label capacity)
CANDIDATE_PROB_LOWER = 0.3              # Must be ≤ MIN_AGRI_PROB; defines lower bound of candidate band
CANDIDATE_DBSCAN_EPS_KM = 1.5           # 0.5–3.0 km typical; spatial diversity
CANDIDATE_NEGATIVE_QUOTA = 0            # 0 disables; use 10–30% of candidates when enabled
NEG_LIKE_PROB_RANGE = (0.3, 0.4)      # Should be just below MIN_AGRI_PROB
NEG_LIKE_NDVI_RANGE = (0.2, 0.5)        # Agri-like NDVI window (tune per region)

# Composite ranking weights
UNCERTAINTY_BAND_DELTA = 0.05  # Treat |p-MIN_AGRI_PROB| < delta as uncertain for persistence weighting
HIGHSCORE_TOP_K = 200          # 100–500 typical; persistent top informative pixels
HIGHSCORE_COMPONENT_WEIGHTS = {"uncertainty": 0.5, "representativeness": 0.3, "consistency": 0.2}  # sum≈1
PROBABLE_AGRI_TOP_K = 200      # 100–500 typical; persistent top probable-agri pixels
PROBABLE_AGRI_COMPONENT_WEIGHTS = {"confidence": 0.7, "representativeness": 0.3}

# Model training
RESNET_EPOCHS = 10       # 5–20 typical
RESNET_LR = 0.001        # 1e-4–3e-3 typical
BATCH_SIZE = 32          # Tune to memory
MIN_AGRI_PROB = 0.4      # Decision threshold (Orange ≥ this). Ensure CANDIDATE_PROB_LOWER ≤ this

# Inference performance
INFER_CHUNKING_ENABLED = True          # Improves stability on large tiles; no effect on results
INFER_MAX_PIXELS_PER_BATCH = 200_000   # 100k–500k typical; adjust to RAM
FEATURE_CACHE_ENABLED = True           # Cache derived features per tile to disk
FEATURE_CACHE_DIR = os.path.join(DATA_DIR, "cache")
FEATURE_CACHE_MAX_GB = 4               # Planned soft cap (eviction not enforced yet)

# Feature selection & importance
FEATURE_SET = "base"  # one of: base, temporal_only, textures_only, temporal_textures, full

# --- Temporal features (multi‑time summaries) ---
# Enable to add across‑season stats for selected indices: min/max/mean/std/range and (last-first) delta.
TEMPORAL_FEATURES_ENABLED = True
TEMPORAL_INDICES = ["NDVI", "NDMI", "EVI"]  # indices to summarize across seasons

# --- Neighbor pooling (spatial context for tabular models) ---
# Adds local mean/std for selected bands/indices using square windows (in pixels).
NEIGHBOR_POOLING_ENABLED = True
NEIGHBOR_POOLING_BANDS = ["NDVI", "B8", "B4"]  # choose stable, informative channels
NEIGHBOR_POOLING_WINDOWS = [7, 11]               # window sizes (odd ints)

# --- Texture features (lightweight, GLCM‑like) ---
# Adds local entropy (rank‑entropy) + local contrast/std + homogeneity proxies for selected bands.
# Designed for robustness and speed; quantizes values to 8‑bit internally. Safe on large tiles.
TEXTURE_GLCM_ENABLED = True
TEXTURE_GLCM_BANDS = ["NDVI", "B8"]
TEXTURE_GLCM_WINDOWS = [5, 9]
TEXTURE_GLCM_LEVELS = 32           # quantization levels for entropy

RUN_PERMUTATION_IMPORTANCE = True

# --- Patch CNN refiner (targeted spatial context) ---
# Trains a tiny CNN on labeled patches and blends its score with the base model
# for the top K% most‑uncertain pixels during inference. Enable to add extra
# texture/shape cues without heavy compute.
PATCH_CNN_ENABLED = False           # Off by default; turn on to refine uncertain pixels
PATCH_CNN_WINDOW = 9                # Patch size (odd int), e.g., 9=>9x9
PATCH_CNN_TOP_UNCERTAIN_FRAC = 0.10 # Fraction of pixels refined per tile (0.05–0.20 typical)
PATCH_CNN_BLEND_ALPHA = 0.5         # Blend weight: alpha*CNN + (1-alpha)*base
PATCH_CNN_MAX_PATCHES_PER_CLASS = 2000  # Training cap per class to limit compute
PATCH_CNN_EPOCHS = 5
PATCH_CNN_LR = 1e-3
PATCH_CNN_BATCH = 64

# --------------------------
# POSTPROCESSING CONFIG
# --------------------------
SIEVE_MIN_SIZE = 5
SIEVE_KEEP_PROB = 0.85            # Must be > MIN_AGRI_PROB. Red (very certain) ≥ this; Orange ∈ [MIN_AGRI_PROB, SIEVE_KEEP_PROB)
SIEVE_KEEP_MODE = "component"      # "component" keeps whole component if any pixel ≥ SIEVE_KEEP_PROB; else "pixel" keeps only high-prob pixels

# Final sweep (thresholds, sieve, morphology)
FINAL_SWEEP_ENABLED = True
FINAL_THRESHOLDS = [0.35, 0.4, 0.45, 0.5]  # Sweep around MIN_AGRI_PROB
FINAL_SIEVE_SIZES = [0, 5, 10, 20]         # Include a wider sieve range for robustness
FINAL_MORPH_OPEN = False
FINAL_MORPH_KERNEL_SIZES = [3]

# --------------------------
# PERSISTENT STORES (PARQUET)
# --------------------------
# Global, run‑to‑run persistent stores to accumulate information.
# These are optional and require pyarrow (+pandas). The pipeline runs without them.
PERSISTENT_PARQUET_ENABLED = True
PERSISTENT_PARQUET_DIR = LABELS_DIR  # Folder to store parquet datasets
GLOBAL_PREDICTIONS_PARQUET = os.path.join(PERSISTENT_PARQUET_DIR, "global_predictions.parquet")
HIGHSCORE_PARQUET = os.path.join(PERSISTENT_PARQUET_DIR, "highscore_store.parquet")
PROBABLE_AGRI_PARQUET = os.path.join(PERSISTENT_PARQUET_DIR, "probableAgri_store.parquet")

# Aggregator outputs (ranked global lists produced from parquet history)
HIGHSCORE_RANKED_PARQUET = os.path.join(PERSISTENT_PARQUET_DIR, "highscore_ranked.parquet")
PROBABLE_AGRI_RANKED_PARQUET = os.path.join(PERSISTENT_PARQUET_DIR, "probableAgri_ranked.parquet")

# Aggregation settings
# Committee scoring uses (uncertainty, disagreement, consistency) for highscore,
# and (confidence minus disagreement penalty) for probable-agri. Recency weight
# emphasizes newer rounds: weight = exp(alpha * (round / max_round)).
HIGHSCORE_AGGR_WEIGHTS = {"uncertainty": 0.5, "disagreement": 0.4, "consistency": 0.1}
PROBABLE_AGRI_AGGR_WEIGHTS = {"confidence": 0.8, "disagreement_penalty": 0.2}
AGGR_RECENCY_ALPHA = 0.1

# Global prediction KMLs
PREDICTION_KMLS_DIR = os.path.join(PERSISTENT_PARQUET_DIR, "prediction_kmls")

# Concatenation progress (predictions chunk merge)
CONCAT_PROGRESS_BLOCK_ROWS = 200_000  # Progress update step during chunk concatenation

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
