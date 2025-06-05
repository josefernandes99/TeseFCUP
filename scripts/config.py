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

for folder in [RAW_DATA_DIR, ROUNDS_DIR, LABELS_DIR, MODELS_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

LABELS_FILE = os.path.join(LABELS_DIR, "labels.csv")
TEMP_LABELS_FILE = os.path.join(LABELS_DIR, "temp_labels.csv")
CANDIDATE_KML = os.path.join(LABELS_DIR, "candidate_patch.kml")

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
MIN_AGRI_RATIO = 0.20
MAX_AGRI_RATIO = 0.30
DUPLICATE_TOLERANCE = 0.0001

# --------------------------
# ACTIVE LEARNING CONFIG
# --------------------------
SUPPORTED_MODELS = ["ResNet", "SVM", "RandomForest"]
SVM_PARAMS = {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
RF_PARAMS = {"n_estimators": 100, "max_depth": 8}
NUM_CANDIDATES_PER_ROUND = 5
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
SIEVE_MIN_SIZE = 50
