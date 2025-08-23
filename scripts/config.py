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
SUPPORTED_MODELS = ["ResNet", "SVM", "RandomForest"]  # Available model backends
SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"}  # C≈0.5–10; gamma: "scale"/"auto"
RF_PARAMS = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 1, "class_weight": "balanced"}  # trees≈200–400; depth≈8–14

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
TEXTURE_WINDOW_SIZE = 5
RUN_PERMUTATION_IMPORTANCE = True

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
