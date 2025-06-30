#!/usr/bin/env python3
# scripts/a3_phase1_active_learning_round.py

import os
import logging
from multiprocessing import cpu_count
from a2_phase1_initial_labeling import extract_features_with_cache
from temporal_features import add_temporal_features
from memory_utils import monitor_memory_usage
from checkpoint_utils import save_checkpoint

# -----------------------------------------------------------------------------
# 1) Speed‐ups: thread‐tune BLAS/OpenMP to use all CPU cores
# -----------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = str(cpu_count())
os.environ["MKL_NUM_THREADS"] = str(cpu_count())

import csv
import glob
import random
import time
import datetime

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, GeometryCollection
from shapely.ops import unary_union
from geo_utils import wgs_to_tile, tile_to_wgs
import torch
import torch.nn as nn
from joblib import dump
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

import config
from config import (
    RAW_DATA_DIR,
    ROUNDS_DIR,
    TEMP_LABELS_FILE,
    NUM_CANDIDATES_PER_ROUND,
    MIN_AGRI_PROB,
    SVM_PARAMS,
    RF_PARAMS,
    RESNET_EPOCHS,
    RESNET_LR,
    BATCH_SIZE,
    INDICES,   # ["NDVI","EVI","EVI2"]
    TIMESTAMPS,
)


# -----------------------------------------------------------------------------
# 2) Feature‐extraction helper (unchanged—reads all bands including indices)
# -----------------------------------------------------------------------------
def extract_features_from_label(row):
    lat, lon = float(row["lat"]), float(row["lon"])
    tile_path = os.path.join(RAW_DATA_DIR, row["tile"])
    if os.path.isdir(tile_path):
        tifs = glob.glob(os.path.join(tile_path, "*.tif"))
        if not tifs:
            raise FileNotFoundError(f"No .tif in {tile_path}")
        tile_path = tifs[0]
    if not os.path.exists(tile_path):
        cand = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
        if not cand:
            raise FileNotFoundError(f"No tile files under {RAW_DATA_DIR}")
        tile_path = cand[0]
        logging.warning(f"Tile '{row['tile']}' not found; using {tile_path}")
    with rasterio.open(tile_path) as src:
        x, y = wgs_to_tile(lat, lon)
        for vals in src.sample([(x, y)]):
            return list(vals.astype(float))
    return None


# -----------------------------------------------------------------------------
# 3) Model wrappers & training, now with full‐feature statistics and scaling
# -----------------------------------------------------------------------------
class SklearnWrapper:
    def __init__(self, clf, feat_means, feat_std, scaler=None):
        self.clf = clf
        self.feat_means = feat_means
        self.feat_std = feat_std
        self.scaler = scaler  # for SVM

    def predict_proba(self, X):
        # 1) impute missing with feature means
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X = X.copy()
            X[inds] = np.take(self.feat_means, inds[1])
        # 2) if an sklearn scaler is provided (SVM), use it
        if self.scaler is not None:
            Xs = self.scaler.transform(X)
        else:
            # for RF, do z‐score scaling manually (optional)
            Xs = (X - self.feat_means) / (self.feat_std + 1e-6)
        return self.clf.predict_proba(Xs)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


class TabularResNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.fc_in     = nn.Linear(input_dim, hidden_dim)
        self.bn_in     = nn.BatchNorm1d(hidden_dim)
        self.block1_fc = nn.Linear(hidden_dim, hidden_dim)
        self.block1_bn = nn.BatchNorm1d(hidden_dim)
        self.block2_fc = nn.Linear(hidden_dim, hidden_dim)
        self.block2_bn = nn.BatchNorm1d(hidden_dim)
        self.out       = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x  = torch.relu(self.bn_in(self.fc_in(x)))
        r1 = x
        b1 = torch.relu(self.block1_bn(self.block1_fc(x))); x = b1 + r1
        r2 = x
        b2 = torch.relu(self.block2_bn(self.block2_fc(x))); x = b2 + r2
        return self.out(x)


class PytorchResNetWrapper:
    def __init__(self, scripted_net, feat_means, feat_std):
        self.net = scripted_net.eval()
        self.feat_means = feat_means
        self.feat_std = feat_std

    def predict_proba(self, X):
        # impute
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X = X.copy()
            X[inds] = np.take(self.feat_means, inds[1])
        # z‐score scale
        Xs = (X - self.feat_means) / (self.feat_std + 1e-6)
        # forward
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        xt = torch.from_numpy(Xs.astype(np.float32)).to(device)
        with torch.no_grad():
            logits = self.net(xt)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

class EnsembleWrapper:
    """Combine multiple model wrappers using soft voting."""

    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.models]
        stacked = np.stack(probas, axis=0)
        return stacked.mean(axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

def train_resnet(net, x_t, y_t):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).train()
    ds     = torch.utils.data.TensorDataset(x_t, y_t)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt    = torch.optim.Adam(net.parameters(), lr=RESNET_LR)
    crit   = nn.CrossEntropyLoss()
    for ep in range(RESNET_EPOCHS):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = net(bx)
            loss   = crit(logits, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep+1) % 2 == 0:
            logging.info(
                f"ResNet epoch {ep + 1}/{RESNET_EPOCHS}, loss={total_loss / len(loader):.4f}"
            )
    net.eval()


def train_model(choice, X, y):
    """
    Compute full‐feature stats (min, max, mean, var),
    then train SVM / RF / ResNet using z‐score or StandardScaler
    so RBF distances are meaningful.
    """
    # 1) compute & impute training‐set means
    feat_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X[inds] = np.take(feat_means, inds[1])

    # 2) compute full statistics
    feat_min  = np.min(X, axis=0)
    feat_max  = np.max(X, axis=0)
    feat_mean = feat_means
    feat_var  = np.var(X, axis=0)
    feat_std  = np.sqrt(feat_var)

    # (optional debug)
    logging.debug("Feature statistics:")
    for i, name in enumerate(INDICES + []):  # if you want to label features, extend with other band names
        if i < 5:  # only print first few for brevity
            logging.debug(
                f"  [{i}] min={feat_min[i]:.3f}, max={feat_max[i]:.3f}, "
                f"mean={feat_mean[i]:.3f}, var={feat_var[i]:.3f}"
            )
    # you can remove or expand this as needed

    c = choice.lower()
    if c == "svm":
        # StandardScaler for true zero‐mean/unit‐variance
        scaler = StandardScaler().fit(X)
        Xs     = scaler.transform(X)
        clf    = SVC(probability=True, gamma="auto", **SVM_PARAMS)
        clf.fit(Xs, y)
        return SklearnWrapper(clf, feat_means, feat_std, scaler)

    elif c == "randomforest":
        rf = RandomForestClassifier(n_jobs=-1, **RF_PARAMS)
        rf.fit(X, y)
        # we will z‐score scale at inference for consistency
        return SklearnWrapper(rf, feat_means, feat_std, scaler=None)

    elif c == "resnet":
        net = TabularResNet(input_dim=X.shape[1])
        train_resnet(net, torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
        scripted = torch.jit.script(net)
        return PytorchResNetWrapper(scripted, feat_means, feat_std)


    elif c == "ensemble":
        svm = train_model("svm", X, y)
        rf = train_model("randomforest", X, y)
        return EnsembleWrapper([svm, rf])

    else:
        raise ValueError(f"Unknown model choice: {choice}")

def evaluate_models_cross_validation(X, y, models):
    """Evaluate models using 5-fold CV and return best along with metrics."""
    metrics = {}
    for name, model in models.items():
        f1 = cross_val_score(model, X, y, cv=5, scoring="f1")
        prec = cross_val_score(model, X, y, cv=5, scoring="precision")
        rec = cross_val_score(model, X, y, cv=5, scoring="recall")
        metrics[name] = {
            "f1": float(np.mean(f1)),
            "precision": float(np.mean(prec)),
            "recall": float(np.mean(rec)),
        }
        logging.info(
            f"{name} CV - F1={metrics[name]['f1']:.3f}, "
            f"P={metrics[name]['precision']:.3f}, R={metrics[name]['recall']:.3f}"
        )

    best_name, best_vals = max(metrics.items(), key=lambda kv: kv[1]["f1"])
    return best_name, best_vals, metrics


def diverse_uncertainty_sampling(probs, features, n_candidates=5):
    """Select uncertain samples while ensuring feature diversity."""
    uncertain_indices = [
        i
        for i, p in enumerate(probs)
        if config.CANDIDATE_PROB_LOWER <= p <= config.CANDIDATE_PROB_UPPER
    ]

    if len(uncertain_indices) > n_candidates * 3:
        uncertain_features = features[uncertain_indices]
        kmeans = KMeans(n_clusters=n_candidates, n_init="auto")
        clusters = kmeans.fit_predict(uncertain_features)

        selected = []
        for cluster_id in range(n_candidates):
            cluster_mask = clusters == cluster_id
            if np.any(cluster_mask):
                cluster_indices = np.where(cluster_mask)[0]
                # pick most uncertain from this cluster
                subset = [uncertain_indices[idx] for idx in cluster_indices]
                best_idx = subset[np.argmin(np.abs(probs[subset] - 0.5))]
                selected.append(best_idx)
        return selected

    return uncertain_indices[:n_candidates]

# -----------------------------------------------------------------------------
# 4) Fast batch inference + per‐pixel geometry
# -----------------------------------------------------------------------------
def get_pixel_corners(src, r, c):
    tl = src.xy(r,   c)
    tr = src.xy(r,   c+1)
    br = src.xy(r+1, c+1)
    bl = src.xy(r+1, c)
    corners = [tl, tr, br, bl, tl]
    out = []
    for x, y in corners:
        lat, lon = tile_to_wgs(x, y)
        out.append([lon, lat])
    return out

def predict_entire_tile(tile_path, model):
    """Return per-pixel predictions and their feature vectors."""
    tile_name = os.path.basename(tile_path)
    with rasterio.open(tile_path) as src:
        arr   = src.read().astype(np.float32)       # (bands, H, W)
        b, H, W = arr.shape
        periods = len(TIMESTAMPS)
        extra = 3  # elevation, slope, aspect
        bands_per_period = (b - extra) // periods
        stack = arr[: periods * bands_per_period].reshape(periods, bands_per_period, H, W)
        temp_feats = add_temporal_features(stack).reshape(-1, H * W)
        base_feats = arr.reshape(b, -1)
        X = np.concatenate([base_feats, temp_feats], axis=0).T
        probs = model.predict_proba(X)[:, 1]        # bulk proba
        rows  = np.repeat(np.arange(H), W)
        cols  = np.tile(  np.arange(W), H)

        results = []
        features = []
        for idx, (r, c) in enumerate(zip(rows, cols)):
            p = float(probs[idx])
            corners = get_pixel_corners(src, r, c)
            mean_lat = sum(y for x,y in corners) / len(corners)
            mean_lon = sum(x for x,y in corners) / len(corners)
            results.append([tile_name, r, c, mean_lat, mean_lon, p])
            features.append(X[idx])
    monitor_memory_usage()
    return results, np.array(features)


# -----------------------------------------------------------------------------
# 5) CSV + polygonized KML exporters
# -----------------------------------------------------------------------------
def save_predictions(round_folder, preds):
    path = os.path.join(round_folder, "predictions.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tile",
            "row_idx",
            "col_idx",
            "center_lat",
            "center_lon",
            "predicted_prob",
        ])
        w.writerows(preds)
    logging.info(f"Predictions written to {path}")

def _write_polygon(f, poly, indent="      "):
    coord_list = []
    for x, y in poly.exterior.coords:
        lat, lon = tile_to_wgs(x, y)
        coord_list.append(f"{lon},{lat},0")
    f.write(f"{indent}<outerBoundaryIs>\n")
    f.write(f"{indent}  <LinearRing>\n")
    f.write(f"{indent}    <coordinates>{' '.join(coord_list)}</coordinates>\n")
    f.write(f"{indent}  </LinearRing>\n")
    f.write(f"{indent}</outerBoundaryIs>\n")
    for interior in poly.interiors:
        ilist = []
        for x, y in interior.coords:
            lat, lon = tile_to_wgs(x, y)
            ilist.append(f"{lon},{lat},0")
        f.write(f"{indent}<innerBoundaryIs>\n")
        f.write(f"{indent}  <LinearRing>\n")
        f.write(f"{indent}    <coordinates>{' '.join(ilist)}</coordinates>\n")
        f.write(f"{indent}  </LinearRing>\n")
        f.write(f"{indent}</innerBoundaryIs>\n")


def _write_geom(f, geom, style_id):
    if geom.is_empty:
        return 0
    polys = []
    if geom.geom_type == "Polygon":
        polys = [geom]
    elif geom.geom_type == "MultiPolygon":
        polys = list(geom.geoms)
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            if g.geom_type == "Polygon":
                polys.append(g)
            elif g.geom_type == "MultiPolygon":
                polys.extend(list(g.geoms))
    if not polys:
        return 0

    f.write("    <Placemark>\n")
    f.write(f"      <styleUrl>#{style_id}</styleUrl>\n")
    if len(polys) == 1:
        f.write("      <Polygon>\n")
        _write_polygon(f, polys[0], "        ")
        f.write("      </Polygon>\n")
    else:
        f.write("      <MultiGeometry>\n")
        for p in polys:
            f.write("        <Polygon>\n")
            _write_polygon(f, p, "          ")
            f.write("        </Polygon>\n")
        f.write("      </MultiGeometry>\n")
    f.write("    </Placemark>\n")
    return len(polys)

def save_agricultural_polygons_kml(round_folder, model, round_num):
    kml_path = os.path.join(round_folder, f"agricultural_patches_round_{round_num}.kml")
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    agri_geom = GeometryCollection()
    non_agri_geom = GeometryCollection()

    for tp in tifs:
        with rasterio.open(tp) as src:
            arr = src.read().astype(np.float32)
            b, H, W = arr.shape
            X = arr.reshape(b, -1).T
            probs = model.predict_proba(X)[:, 1].reshape(H, W)
            mask = (probs >= MIN_AGRI_PROB).astype("uint8")

            for geom, val in shapes(mask, transform=src.transform):
                poly = shape(geom)
                if val == 1:
                    agri_geom = agri_geom.union(poly)
                else:
                    non_agri_geom = non_agri_geom.union(poly)

    with open(kml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n')
        f.write('    <Style id="agriStyle">\n')
        f.write('      <LineStyle><color>ff0000ff</color><width>1</width></LineStyle>\n')
        f.write('      <PolyStyle><color>400000ff</color><outline>1</outline></PolyStyle>\n')
        f.write('    </Style>\n')
        f.write('    <Style id="nonAgriStyle">\n')
        f.write('      <LineStyle><color>ff00ffff</color><width>1</width></LineStyle>\n')
        f.write('      <PolyStyle><color>4000ff00</color><outline>1</outline></PolyStyle>\n')
        f.write('    </Style>\n')

        total_polys = 0
        total_polys += _write_geom(f, agri_geom, "agriStyle")
        total_polys += _write_geom(f, non_agri_geom, "nonAgriStyle")

        f.write("  </Document>\n</kml>\n")

    if total_polys == 0:
        logging.warning(
            f"No polygons saved (all probs < {MIN_AGRI_PROB})"
        )
    else:
        logging.info(f"{total_polys} polygons saved to {kml_path}")

# -----------------------------------------------------------------------------
# 6) Candidate‐patch KML (unchanged)
# -----------------------------------------------------------------------------
def generate_candidate_kml(tile_name, row_idx, col_idx, outpath):
    """Generate a KML polygon for a 3x3 patch centered at the given pixel."""
    tile_path = os.path.join(RAW_DATA_DIR, tile_name)
    if os.path.isdir(tile_path):
        tifs = glob.glob(os.path.join(tile_path, "*.tif"))
        if not tifs:
            logging.warning(f"Missing tile {tile_name}; skipping candidate KML.")
            return
        tile_path = tifs[0]
    if not os.path.exists(tile_path):
        logging.warning(f"Missing tile {tile_name}; skipping candidate KML.")
        return
    with rasterio.open(tile_path) as src:
        r0 = max(row_idx - 1, 0)
        c0 = max(col_idx - 1, 0)
        r1 = min(row_idx + 2, src.height)
        c1 = min(col_idx + 2, src.width)
        ul = src.xy(r0, c0)
        ur = src.xy(r0, c1)
        lr = src.xy(r1, c1)
        ll = src.xy(r1, c0)
        corners = [ul, ur, lr, ll, ul]
        corners = [(tile_to_wgs(x, y)[1], tile_to_wgs(x, y)[0]) for x, y in corners]

    kml = Element('kml', xmlns="http://www.opengis.net/kml/2.2")
    doc = SubElement(kml, 'Document')
    style = SubElement(doc, 'Style', id="candidateStyle")
    ln = SubElement(style,'LineStyle'); SubElement(ln,'color').text="ff000000"; SubElement(ln,'width').text="2"
    ps = SubElement(style,'PolyStyle'); SubElement(ps,'fill').text="0"; SubElement(ps,'outline').text="1"

    pm = SubElement(doc, 'Placemark')
    SubElement(pm, 'styleUrl').text="#candidateStyle"
    SubElement(pm, 'name').text=f"Candidate {tile_name} r={row_idx},c={col_idx}"
    poly = SubElement(pm, 'Polygon')
    ob   = SubElement(poly, 'outerBoundaryIs')
    ring = SubElement(ob, 'LinearRing')
    coords_str = " ".join(f"{x},{y},0" for x,y in corners)
    SubElement(ring, 'coordinates').text = coords_str

    rough  = tostring(kml, encoding="utf-8")
    pretty = parseString(rough).toprettyxml(indent="  ", encoding="utf-8")
    with open(outpath, "wb") as f:
        f.write(pretty)
    logging.info(f"Candidate KML => {outpath}")

# -----------------------------------------------------------------------------
# 7) Active‐Learning orchestration (unchanged aside from new train_model)
# -----------------------------------------------------------------------------
def active_learning_round(round_num, labels_file, model_choice):
    logging.info(f"\n=== Starting Active Learning Round {round_num} ===")
    rnd_dir = os.path.join(ROUNDS_DIR, f"round_{round_num}")
    os.makedirs(rnd_dir, exist_ok=True)

    # load & featurize
    rows = list(csv.DictReader(open(labels_file)))
    if len(rows) <= 1:
        logging.warning("Not enough labels; aborting.")
        return None
    X, y = [], []
    for r in rows:
        tile_path = os.path.join(RAW_DATA_DIR, r["tile"])
        if os.path.isdir(tile_path):
            tifs = glob.glob(os.path.join(tile_path, "*.tif"))
            if not tifs:
                logging.warning(f"No .tif in {tile_path}; skipping label {r['id']}")
                continue
            tile_path = tifs[0]
        if not os.path.exists(tile_path):
            logging.warning(f"No .tif in {tile_path}; skipping label {r['id']}")
            continue
        feats = extract_features_with_cache(tile_path, float(r["lat"]), float(r["lon"]))
        if feats is not None:
            X.append(feats)
            y.append(1 if r["label"].lower()=="agricultural" else 0)
        monitor_memory_usage()
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    tuned_hp = config.auto_tune_model_hyperparameters(X, y)
    config.SVM_PARAMS.update(tuned_hp["SVM_PARAMS"])
    config.RF_PARAMS.update(tuned_hp["RF_PARAMS"])
    logging.info(
        f"Hyperparameters tuned => SVM {config.SVM_PARAMS}, RF {config.RF_PARAMS}"
    )

    models_dict = {
        "SVM": make_pipeline(StandardScaler(), SVC(probability=True, gamma="auto", **SVM_PARAMS)),
        "RandomForest": RandomForestClassifier(n_jobs=-1, **RF_PARAMS),
    }
    models_dict["Ensemble"] = VotingClassifier(
        estimators=[
            ("svm", make_pipeline(StandardScaler(), SVC(probability=True, gamma="auto", **SVM_PARAMS))),
            ("rf", RandomForestClassifier(n_jobs=-1, **RF_PARAMS)),
        ],
        voting="soft",
    )
    best_name, best_vals, all_metrics = evaluate_models_cross_validation(X, y, models_dict)
    if best_name:
        model_choice = best_name
        logging.info(f"Selected {model_choice} via CV with F1={best_vals['f1']:.3f}")
        metrics_path = os.path.join(rnd_dir, "cv_metrics.csv")
        with open(metrics_path, "w", newline="") as mf:
            writer = csv.writer(mf)
            writer.writerow(["model", "f1", "precision", "recall"])
            for m, vals in all_metrics.items():
                writer.writerow([m, vals["f1"], vals["precision"], vals["recall"]])
        logging.info(f"CV metrics written to {metrics_path}")

    # train & save
    logging.info(f"Training data shape: {X.shape}, model: {model_choice}")
    model = train_model(model_choice, X, y)
    mp = os.path.join(rnd_dir, f"model_round_{round_num}.pkl")
    dump(model, mp)
    logging.info(f"Model saved to {mp}")

    # inference + timing
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    preds = []
    feats = []
    start = time.time()
    with Progress("[bold cyan]{task.description}", BarColumn(), TaskProgressColumn(),
                  TimeElapsedColumn(), TimeRemainingColumn()) as prog:
        task = prog.add_task("Running inference", total=len(tifs))
        for tp in tifs:
            p, f = predict_entire_tile(tp, model)
            preds.extend(p)
            feats.append(f)
            monitor_memory_usage()
            prog.update(task, advance=1)
    feats = np.concatenate(feats, axis=0) if feats else np.empty((0, X.shape[1]))
    logging.info(f"Total pixels inferred: {len(preds)}")
    logging.info(
        f"Inference completed in {str(datetime.timedelta(seconds=int(time.time() - start)))}"
    )

    # auto-tune config parameters based on predictions
    tuned = config.auto_tune_parameters(preds)
    config.SIEVE_MIN_SIZE = tuned["SIEVE_MIN_SIZE"]
    config.CANDIDATE_PROB_LOWER = tuned["CANDIDATE_PROB_LOWER"]
    config.CANDIDATE_PROB_UPPER = tuned["CANDIDATE_PROB_UPPER"]
    logging.info(
        f"Auto-tuned: sieve={config.SIEVE_MIN_SIZE}, "
        f"lower={config.CANDIDATE_PROB_LOWER:.3f}, upper={config.CANDIDATE_PROB_UPPER:.3f}"
    )

    # outputs
    save_predictions(rnd_dir, preds)
    save_agricultural_polygons_kml(rnd_dir, model, round_num)

    # candidate selection
    probs_arr = np.array([p[5] for p in preds], dtype=float)
    feats_arr = feats
    cand_idx = diverse_uncertainty_sampling(
        probs_arr, feats_arr, n_candidates=NUM_CANDIDATES_PER_ROUND
    )
    cands = [preds[i] for i in cand_idx]
    logging.info(f"{len(cands)} candidate patches selected")

    tmp = os.path.join(rnd_dir, "temp_labels.csv")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","tile","lat","lon","label"])
    for t, r, c, la, lo, p in cands:
        sub = os.path.join(ROUNDS_DIR, f"round_{t}")
        if not os.path.isdir(sub):
            sub = os.path.join(ROUNDS_DIR, "round_temp_candidates")
            os.makedirs(sub, exist_ok=True)
        kmlp = os.path.join(sub, "candidate_patch.kml")
        generate_candidate_kml(t, r, c, kmlp)
        logging.info(f"Candidate: {t} r={r},c={c}, p={p:.3f}")
        ui = input("Label? (1=Agri,2=NonAgri,3=Skip): ").strip()
        if ui=="3":
            logging.info("Skipped.")
            continue
        lab = "Agricultural" if ui == "1" else "Non-Agricultural" if ui == "2" else None
        if lab:
            eid = f"AL_{round_num}_{int(random.random() * 1e6)}"
            with open(tmp, "a", newline="") as f2:
                csv.writer(f2).writerow([eid, t, la, lo, lab])
            logging.info("Label saved.")
    save_checkpoint(round_num, model, len(X))
    logging.info(f"Round {round_num} complete; labels at {tmp}")
    return tmp

if __name__ == "__main__":
    active_learning_round(1, TEMP_LABELS_FILE, "SVM")
