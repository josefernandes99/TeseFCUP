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
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint.txt")

for folder in [RAW_DATA_DIR, ROUNDS_DIR, LABELS_DIR, MODELS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

LABELS_FILE = os.path.join(LABELS_DIR, "labels.csv")
TEMP_LABELS_FILE = os.path.join(LABELS_DIR, "temp_labels.csv")
LABELS_KML = os.path.join(LABELS_DIR, "labels.kml")
CANDIDATE_KML = os.path.join(LABELS_DIR, "candidate_patch.kml")
GRID_KML_DIR = os.path.join(LABELS_DIR, "grids")
# Optional evaluation labels used for model assessment
EVALUATE_FILE = os.path.join(LABELS_DIR, "evaluate.csv")
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
SUPPORTED_MODELS = ["ResNet", "SVM", "RandomForest"]
SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
RF_PARAMS = {"n_estimators": 100, "max_depth": 8}
NUM_CANDIDATES_PER_ROUND = 25
CANDIDATE_PROB_LOWER = 0.4
CANDIDATE_PROB_UPPER = 0.6
RESNET_EPOCHS = 10
RESNET_LR = 0.001
BATCH_SIZE = 32
MIN_AGRI_PROB = 0.5
NUM_RANDOM_PICKS_PER_TILE = 5

# --------------------------
# POSTPROCESSING CONFIG
# --------------------------
SIEVE_MIN_SIZE = 5

# --------------------------
# LABEL NOTES OPTIONS
# --------------------------
NOTE_OPTIONS = [
    "Agricultural Crop Center",
    "Agricultural Crop Border",
    "Agricultural Crop Doubtful",
    "Open Field, No Vegetation",
    "Open Field, Vegetation",
    "Tree/Bush",
    "Building / Man Made",
    "Water Bodies",
    "Other",
]
