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
SUPPORTED_MODELS = ["ResNet", "SVM", "RandomForest", "Ensemble"]
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

# --------------------------
# ADAPTIVE PARAMETERS
# --------------------------
import numpy as np
from scipy.ndimage import label


def estimate_field_sizes(predictions):
    """Estimate sizes of predicted agricultural fields."""
    tiles = {}
    for tile, r, c, lat, lon, prob in predictions:
        tiles.setdefault(tile, []).append((r, c, prob))
    sizes = []
    for _, items in tiles.items():
        rows = [r for r, _c, _p in items]
        cols = [c for _r, c, _p in items]
        if not rows or not cols:
            continue
        h, w = max(rows) + 1, max(cols) + 1
        mask = np.zeros((h, w), dtype=np.uint8)
        for r, c, p in items:
            if p >= MIN_AGRI_PROB:
                mask[r, c] = 1
        labeled, _ = label(mask)
        counts = np.bincount(labeled.ravel())
        sizes.extend(counts[1:])
    return sizes


def get_prediction_uncertainties(predictions):
    """Return absolute distance from 0.5 for each probability."""
    return [abs(p[5] - 0.5) for p in predictions]


def auto_tune_parameters(training_data):
    """Automatically adjust parameters based on data characteristics."""
    field_sizes = estimate_field_sizes(training_data)
    if field_sizes:
        optimal_sieve = int(np.percentile(field_sizes, 10))
    else:
        optimal_sieve = SIEVE_MIN_SIZE

    uncertainties = get_prediction_uncertainties(training_data)
    if uncertainties:
        lower_thresh = float(np.percentile(uncertainties, 40))
        upper_thresh = float(np.percentile(uncertainties, 60))
    else:
        lower_thresh = CANDIDATE_PROB_LOWER
        upper_thresh = CANDIDATE_PROB_UPPER

    return {
        "SIEVE_MIN_SIZE": optimal_sieve,
        "CANDIDATE_PROB_LOWER": lower_thresh,
        "CANDIDATE_PROB_UPPER": upper_thresh,
    }


def auto_tune_model_hyperparameters(X, y):
    """Grid search for best SVM and RF hyperparameters."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    svm_pipe = make_pipeline(StandardScaler(), SVC(probability=True))
    svm_grid = {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", "auto"]}
    svm_search = GridSearchCV(svm_pipe, svm_grid, cv=3, scoring="f1", n_jobs=-1)
    svm_search.fit(X, y)
    best_svm = {
        "C": svm_search.best_params_["svc__C"],
        "gamma": svm_search.best_params_["svc__gamma"],
        "kernel": "rbf",
    }

    rf = RandomForestClassifier()
    rf_grid = {"n_estimators": [100, 200], "max_depth": [6, 8, 10]}
    rf_search = GridSearchCV(rf, rf_grid, cv=3, scoring="f1", n_jobs=-1)
    rf_search.fit(X, y)
    best_rf = {
        "n_estimators": rf_search.best_params_["n_estimators"],
        "max_depth": rf_search.best_params_["max_depth"],
    }

    return {"SVM_PARAMS": best_svm, "RF_PARAMS": best_rf}