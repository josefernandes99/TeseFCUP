#!/usr/bin/env python3
# scripts/a3_phase1_active_learning_round.py

import os
from multiprocessing import cpu_count

# -----------------------------------------------------------------------------
# 1) Speed‐ups: thread‐tune BLAS/OpenMP to use all CPU cores
# -----------------------------------------------------------------------------
# Threading caps to avoid OpenBLAS/OpenMP warnings and oversubscription
# Tune to your 16C/32T machine: allow up to 16 threads for math libs
_N_THREADS = str(min(16, max(1, cpu_count())))
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _N_THREADS)

import csv
import glob
import random
import time
import datetime
import json
from pyproj import Transformer

import numpy as np
import rasterio
from rasterio.features import shapes, sieve
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union, transform as shp_transform
import torch
import torch.nn as nn
from joblib import dump
from memory_watcher import free_unused_memory
from progress_utils import new_progress
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

from config import (
    RAW_DATA_DIR,
    ROUNDS_DIR,
    TEMP_LABELS_FILE,
    RESNET_EPOCHS,
    RESNET_LR,
    BATCH_SIZE,
    NOTE_OPTIONS,
)
import config as cfg
from evaluation import evaluate_model
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from splits import stratified_train_val_test_indices
from features import current_feature_names
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial import cKDTree

# Note: grid KML generation is performed once at pipeline start.
from features import add_derived_features
from al_shared import extract_features_from_label, get_tile_features



# -----------------------------------------------------------------------------
# 2) Feature-extraction helper (unchanged - reads all bands including indices)
# -----------------------------------------------------------------------------
def prompt_note():
    """Prompt user to select one of the predefined note options."""
    print("notes options:")
    for idx, opt in enumerate(NOTE_OPTIONS, 1):
        print(f" {idx}. {opt}")
    choice = input("Select note [1-9]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(NOTE_OPTIONS):
        return NOTE_OPTIONS[int(choice) - 1]
    print("Invalid choice; using 'Other'.")
    return NOTE_OPTIONS[-1]

# -----------------------------------------------------------------------------
# 3) Model wrappers & training, now with full‐feature statistics and scaling
# -----------------------------------------------------------------------------
class SklearnWrapper:
    def __init__(self, clf, feat_means, feat_std):
        self.clf = clf
        self.feat_means = feat_means
        self.feat_std = feat_std
        self.kind = 'svm' if 'SVC' in str(type(clf)) else ('randomforest' if 'Forest' in str(type(clf)) else ('xgboost' if 'xgboost' in str(type(clf)).lower() else 'sklearn'))

    def predict_proba(self, X):
        # 1) impute missing with feature means
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X = X.copy()
            X[inds] = np.take(self.feat_means, inds[1])
        # 2) z‐score scale with global stats
        Xs = (X - self.feat_means) / (self.feat_std + 1e-6)
        return self.clf.predict_proba(Xs)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    # Needed so sklearn.permutation_importance accepts this wrapper
    def fit(self, X, y):  # no-op
        return self


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
        self.kind = 'resnet'

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

    # Needed so sklearn.permutation_importance accepts this wrapper
    def fit(self, X, y):  # no-op
        return self


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
            print(f"ResNet epoch {ep+1}/{RESNET_EPOCHS}, loss={total_loss/len(loader):.4f}")
    net.eval()


def train_model(choice, X, y):
    """Train model using per-feature statistics from the training data."""
    feat_means = np.nanmean(X, axis=0).astype(np.float32)
    feat_std = np.nanstd(X, axis=0).astype(np.float32)
    feat_std[feat_std == 0] = 1.0
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X = X.copy()
        X[inds] = np.take(feat_means, inds[1])
    Xs = (X - feat_means) / feat_std

    c = choice.lower()
    if c == "svm":
        params = cfg.SVM_PARAMS.copy()
        base = SVC(probability=True, **{k: v for k, v in params.items() if k != 'class_weight'})
        clf = CalibratedClassifierCV(base, method=cfg.CALIBRATION_METHOD, cv=cfg.CALIBRATION_FOLDS)
        clf.fit(Xs, y)
        w = SklearnWrapper(clf, feat_means, feat_std)
        w.kind = 'svm'
        return w

    elif c == "randomforest":
        rf = RandomForestClassifier(n_jobs=-1, **cfg.RF_PARAMS)
        rf.fit(Xs, y)
        w = SklearnWrapper(rf, feat_means, feat_std)
        w.kind = 'randomforest'
        return w

    elif c == "resnet":
        net = TabularResNet(input_dim=Xs.shape[1])
        train_resnet(net, torch.from_numpy(Xs.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
        scripted = torch.jit.script(net)
        return PytorchResNetWrapper(scripted, feat_means, feat_std)

    elif c in ("xgboost", "xgb"):
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("XGBoost not installed. Please install xgboost to use this model.") from e
        params = cfg.XGB_PARAMS.copy()
        xgb_clf = xgb.XGBClassifier(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            n_jobs=-1,
            tree_method="hist",
            random_state=None if cfg.SPLIT_SEED_MODE == "random" else int(cfg.SPLIT_RANDOM_SEED),
        )
        xgb_clf.fit(Xs, y)
        w = SklearnWrapper(xgb_clf, feat_means, feat_std)
        w.kind = 'xgboost'
        return w

    else:
        raise ValueError(f"Unknown model choice: {choice}")


# -----------------------------------------------------------------------------
# 4) Fast batch inference + per‐pixel geometry
# -----------------------------------------------------------------------------
def get_pixel_corners(src, r, c):
    """Return corner coordinates for a pixel as (lon, lat) pairs in WGS84."""
    tl = src.xy(r,   c)
    tr = src.xy(r,   c+1)
    br = src.xy(r+1, c+1)
    bl = src.xy(r+1, c)
    corners = [tl, tr, br, bl, tl]
    if src.crs and not src.crs.is_geographic:
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        corners = [transformer.transform(x, y) for x, y in corners]
    return [[lon, lat] for lon, lat in corners]

def _ndvi_summary_from_names(arr, names):
    ndvi_idxs = [i for i, n in enumerate(names) if n.upper().startswith("NDVI")]
    if not ndvi_idxs:
        return None
    # Use mean NDVI across seasons when multiple are present
    ndvi_stack = arr[ndvi_idxs]
    return ndvi_stack.mean(axis=0).reshape(-1).astype(np.float32)


def predict_entire_tile(tile_path, model, progress=None, task_id=None):
    """Run inference on a tile and optionally update a progress bar."""
    tile_name = os.path.basename(tile_path)
    with rasterio.open(tile_path) as src:
        # Reuse cached per-tile features if available to avoid recomputation
        tile = os.path.basename(tile_path)
        tf = get_tile_features(tile)
        if tf is not None:
            arr, _, _ = tf
            # get_tile_features returns derived features already; infer names approximately
            names = current_feature_names()
        else:
            raw = src.read().astype(np.float32)
            arr, names = add_derived_features(raw)
        b, H, W = arr.shape
        Xflat  = arr.reshape(b, -1).T                # (H*W, bands)
        rows = np.repeat(np.arange(H, dtype=np.int32), W)
        cols = np.tile(np.arange(W, dtype=np.int32), H)
        # Chunked inference to limit memory
        def _effective_bs():
            base = int(getattr(cfg, 'INFER_MAX_PIXELS_PER_BATCH', 400_000))
            if not bool(getattr(cfg, 'AUTO_BATCH_TUNING_ENABLED', True)):
                return base
            kind = (getattr(model, 'kind', '') or '').lower()
            if kind == 'svm':
                return min(base, int(getattr(cfg, 'INFER_BATCH_OVERRIDE_SVM', 200_000)))
            if kind == 'resnet':
                return min(base, int(getattr(cfg, 'INFER_BATCH_OVERRIDE_RESNET', 200_000)))
            if kind == 'xgboost':
                return min(max(base, int(getattr(cfg, 'INFER_BATCH_OVERRIDE_XGBOOST', 500_000))), base)
            if kind == 'randomforest':
                return min(max(base, int(getattr(cfg, 'INFER_BATCH_OVERRIDE_RANDOMFOREST', 400_000))), base)
            return base

        if cfg.INFER_CHUNKING_ENABLED:
            probs = np.empty((Xflat.shape[0],), dtype=np.float32)
            bs = _effective_bs()
            for i in range(0, Xflat.shape[0], bs):
                probs[i:i+bs] = model.predict_proba(Xflat[i:i+bs])[:, 1].astype(np.float32)
        else:
            probs = model.predict_proba(Xflat)[:, 1].astype(np.float32)

        # vectorized center coordinate computation
        xs, ys = rasterio.transform.xy(src.transform, rows.tolist(), cols.tolist(), offset="center")
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if src.crs and not src.crs.is_geographic:
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            xs, ys = transformer.transform(xs, ys)

        ndvi_vec = _ndvi_summary_from_names(arr, names)
        if ndvi_vec is None:
            # Gracefully fallback to zeros if NDVI not present
            ndvi_vec = np.zeros_like(probs, dtype=np.float32)
        ndvi_vals = ndvi_vec
        results = [
            [tile_name, int(r), int(c), float(lat), float(lon), float(p), float(ndvi)]
            for r, c, lat, lon, p, ndvi in zip(rows, cols, ys, xs, probs, ndvi_vals)
        ]

        if progress is not None and task_id is not None:
            # bulk update instead of per-pixel loop
            progress.update(task_id, advance=len(results))

    return results


# -----------------------------------------------------------------------------
# 5) CSV + polygonized KML exporters
# -----------------------------------------------------------------------------
def save_predictions(round_folder, preds):
    path = os.path.join(round_folder, "predictions.csv")
    from progress_utils import new_progress as _npb
    with open(path, "w", newline="") as f, _npb() as _prog:
        w = csv.writer(f)
        w.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
        total = len(preds)
        task = _prog.add_task("Write predictions.csv", total=total or None)
        batch = 50_000
        for i in range(0, total, batch):
            chunk = preds[i:i+batch]
            w.writerows(chunk)
            _prog.update(task, advance=len(chunk))
    print(f"Predictions written to {path}")
    return path


def _write_tile_predictions_csv(round_folder, tile_name, rows):
    """Write per-tile predictions to a temporary CSV and return its path."""
    tmp_dir = os.path.join(round_folder, "_tile_preds")
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, f"{os.path.splitext(tile_name)[0]}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
        w.writerows(rows)
    return path


def _merge_tile_prediction_csvs(round_folder, merged_path=None):
    """Concatenate per-tile CSVs into a single predictions.csv and remove shards."""
    tmp_dir = os.path.join(round_folder, "_tile_preds")
    if merged_path is None:
        merged_path = os.path.join(round_folder, "predictions.csv")
    files = sorted([p for p in glob.glob(os.path.join(tmp_dir, "*.csv"))])
    if not files:
        return None
    # Count total rows for progress (minus headers)
    total_rows = 0
    for fp in files:
        try:
            with open(fp, newline="") as f:
                # subtract header
                total_rows += max(0, sum(1 for _ in f) - 1)
        except Exception:
            pass
    from progress_utils import new_progress as _npb
    with open(merged_path, "w", newline="") as out, _npb() as _prog:
        w = csv.writer(out)
        w.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
        processed = 0
        task = _prog.add_task("Merge predictions", total=total_rows or None)
        batch = 50_000
        for fp in files:
            with open(fp, newline="") as f:
                r = csv.reader(f)
                header = next(r, None)
                for row in r:
                    w.writerow(row)
                    processed += 1
                    if total_rows and (processed % batch == 0):
                        _prog.update(task, completed=min(processed, total_rows))
        if total_rows:
            _prog.update(task, completed=total_rows)
    # cleanup shards
    for fp in files:
        try:
            os.remove(fp)
        except Exception:
            pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass
    print(f"Predictions written to {merged_path}")
    return merged_path


def _load_predictions_by_tile(csv_path):
    """Return mapping: tile -> (row_indices, col_indices, probs) with progress."""
    pred_map = {}
    import os as _os
    from progress_utils import new_progress as _npb
    file_size = _os.path.getsize(csv_path) if _os.path.exists(csv_path) else 0
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f, _npb() as _prog:
        task = _prog.add_task("Scan predictions.csv for KML", total=file_size or None)
        processed = 0
        header = f.readline()
        processed += len(header.encode('utf-8', 'replace'))
        if file_size:
            _prog.update(task, completed=processed)
        # Determine column indices from header
        hdr = next(csv.reader([header])) if header else []
        try:
            col_tile = hdr.index('tile')
            col_r = hdr.index('row_idx')
            col_c = hdr.index('col_idx')
            col_p = hdr.index('predicted_prob')
        except Exception:
            col_tile = col_r = col_c = col_p = None
        for line in f:
            processed += len(line.encode('utf-8', 'replace'))
            try:
                row = next(csv.reader([line]))
                if col_tile is None or len(row) <= max(col_tile, col_r, col_c, col_p):
                    continue
                tile = row[col_tile]
                ri = int(row[col_r]); ci = int(row[col_c])
                prob = float(row[col_p])
            except Exception:
                continue
            if tile not in pred_map:
                pred_map[tile] = ([], [], [])
            pred_map[tile][0].append(ri)
            pred_map[tile][1].append(ci)
            pred_map[tile][2].append(prob)
            if file_size and processed % (1024*1024) < 1000:  # roughly per MB
                _prog.update(task, completed=min(processed, file_size))
        if file_size:
            _prog.update(task, completed=file_size)
    for t, (rs, cs, ps) in pred_map.items():
        pred_map[t] = (
            np.array(rs, dtype=np.int32),
            np.array(cs, dtype=np.int32),
            np.array(ps, dtype=np.float32),
        )
    return pred_map


def save_agricultural_polygons_kml(round_folder, round_num, pred_csv=None, preds=None):
    """Polygonize predictions (single blue style over MIN_AGRI_PROB).

    Accepts either a path to predictions.csv or an in-memory list of
    [tile,row,col,lat,lon,prob,ndvi] rows via `preds`.
    """
    pred_map = {}
    if preds is not None:
        for tile, r, c, _la, _lo, p, _ndvi in preds:
            if tile not in pred_map:
                pred_map[tile] = ([], [], [])
            pred_map[tile][0].append(int(r))
            pred_map[tile][1].append(int(c))
            pred_map[tile][2].append(float(p))
        for t, (rs, cs, ps) in pred_map.items():
            pred_map[t] = (
                np.array(rs, dtype=np.int32),
                np.array(cs, dtype=np.int32),
                np.array(ps, dtype=np.float32),
            )
    else:
        pred_csv = pred_csv or os.path.join(round_folder, "predictions.csv")
        if not os.path.exists(pred_csv):
            print(f"Missing predictions file => {pred_csv}")
            return
        pred_map = _load_predictions_by_tile(pred_csv)
    kml_path = os.path.join(round_folder, f"agricultural_patches_round_{round_num}.kml")
    agri_polys = []

    with new_progress() as prog:
        task = prog.add_task("Polygonizing tiles", total=len(pred_map))

        for tile, (rows, cols, probs) in pred_map.items():
            tif = os.path.join(RAW_DATA_DIR, tile)
            if not os.path.exists(tif):
                print(f"WARNING: Missing tile for predictions => {tile}")
                prog.update(task, advance=1)
                continue
            with rasterio.open(tif) as src:
                H, W = src.height, src.width
                arr_probs = np.zeros((H, W), dtype=np.float32)
                arr_probs[rows, cols] = probs
                agri_mask = arr_probs >= cfg.MIN_AGRI_PROB
                from scipy.ndimage import binary_closing, binary_fill_holes
                agri_mask = binary_fill_holes(binary_closing(agri_mask))
                if cfg.SIEVE_MIN_SIZE > 0:
                    agri_mask = sieve(agri_mask.astype("uint8"), size=cfg.SIEVE_MIN_SIZE, connectivity=8).astype(bool)
                transformer = None
                if src.crs and not src.crs.is_geographic:
                    transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                # collect polygons
                for geom, val in shapes(agri_mask.astype("uint8"), mask=agri_mask, transform=src.transform):
                    if val != 1:
                        continue
                    poly = shape(geom)
                    if transformer:
                        poly = shp_transform(transformer.transform, poly)
                    agri_polys.append(poly)
            prog.update(task, advance=1)

    if not agri_polys:
        print(f"WARNING: No polygons (all probs < {cfg.MIN_AGRI_PROB})")
        return
    # Merge for compactness
    def ensure_list(merged):
        if isinstance(merged, Polygon):
            return [merged]
        elif isinstance(merged, MultiPolygon):
            return list(merged.geoms)
        else:
            return []
    agri_list = ensure_list(unary_union(agri_polys)) if agri_polys else []

    # Build KML with one blue style
    doc = Element('Document')
    style_ag = SubElement(doc, 'Style', id='agri')
    # Color set to web hex #55ffff -> ABGR aabbggrr: AA + BB(ff) + GG(ff) + RR(55)
    # Line: opaque; Fill: semi-transparent
    ln = SubElement(style_ag, 'LineStyle'); SubElement(ln, 'color').text = 'ffffff55'; SubElement(ln, 'width').text = '1'
    ps = SubElement(style_ag, 'PolyStyle'); SubElement(ps, 'color').text = '40ffff55'; SubElement(ps, 'outline').text = '1'

    total_polys = 0
    for p in agri_list:
        coords = list(p.exterior.coords)
        coord_str = " ".join(f"{lon},{lat},0" for lon, lat in coords)
        pm = SubElement(doc, 'Placemark')
        SubElement(pm, 'styleUrl').text = '#agri'
        poly_el = SubElement(pm, 'Polygon')
        ob = SubElement(poly_el, 'outerBoundaryIs')
        ring = SubElement(ob, 'LinearRing')
        SubElement(ring, 'coordinates').text = coord_str
        total_polys += 1

    if kml_path:
        kml = Element('kml'); kml.set('xmlns','http://www.opengis.net/kml/2.2')
        d2 = SubElement(kml, 'Document')
        for el in list(doc):
            d2.append(el)
        xml = parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
        with open(kml_path, 'wb') as f:
            f.write(xml)
        print(f"{total_polys} agricultural polygons saved to {kml_path}")

# -----------------------------------------------------------------------------
# 6) Candidate‐patch KML (unchanged)
# -----------------------------------------------------------------------------
def generate_candidate_kml(tile_name, row_idx, col_idx, outpath=None):
    tif_path = os.path.join(RAW_DATA_DIR, tile_name)
    if not os.path.exists(tif_path):
        print(f"WARNING: missing tile {tile_name}; skipping candidate KML.")
        return
    with rasterio.open(tif_path) as src:
        corners = get_pixel_corners(src, row_idx, col_idx)
        # approximate center and range for LookAt (2x zoom compared to bbox diagonal)
        lons = [p[0] for p in corners]
        lats = [p[1] for p in corners]
        lon_c = sum(lons) / len(lons)
        lat_c = sum(lats) / len(lats)
        # crude diagonal distance in meters (equirectangular approx)
        try:
            import math
            R = 6371000.0
            lon1, lat1 = math.radians(min(lons)), math.radians(min(lats))
            lon2, lat2 = math.radians(max(lons)), math.radians(max(lats))
            x = (lon2 - lon1) * math.cos(0.5 * (lat1 + lat2))
            y = (lat2 - lat1)
            diag_m = R * math.sqrt(x*x + y*y)
            look_range = max(400.0, diag_m / 4.0)
        except Exception:
            look_range = 400.0

    doc = Element('Document')
    style = SubElement(doc, 'Style', id="candidateStyle")
    ln = SubElement(style,'LineStyle'); SubElement(ln,'color').text="ff000000"; SubElement(ln,'width').text="2"
    ps = SubElement(style,'PolyStyle'); SubElement(ps,'fill').text="0"; SubElement(ps,'outline').text="1"
    # Add a LookAt to zoom in on the candidate
    look = SubElement(doc, 'LookAt')
    SubElement(look, 'longitude').text = f"{lon_c}"
    SubElement(look, 'latitude').text = f"{lat_c}"
    SubElement(look, 'range').text = f"{look_range:.2f}"
    SubElement(look, 'tilt').text = "0"
    SubElement(look, 'heading').text = "0"

    pm = SubElement(doc, 'Placemark')
    SubElement(pm, 'styleUrl').text="#candidateStyle"
    SubElement(pm, 'name').text=f"Candidate {tile_name} r={row_idx},c={col_idx}"
    poly = SubElement(pm, 'Polygon')
    ob   = SubElement(poly, 'outerBoundaryIs')
    ring = SubElement(ob, 'LinearRing')
    coords_str = " ".join(f"{x},{y},0" for x,y in corners)
    SubElement(ring, 'coordinates').text = coords_str

    if outpath:
        kml = Element('kml', xmlns="http://www.opengis.net/kml/2.2")
        d = SubElement(kml, 'Document')
        for el in list(doc):
            d.append(el)
        xml = parseString(tostring(kml, encoding="utf-8")).toprettyxml(indent="  ", encoding="utf-8")
        with open(outpath, "wb") as f:
            f.write(xml)
        print(f"Candidate KML => {outpath}")


# -----------------------------------------------------------------------------
# 7) Active‐Learning orchestration (unchanged aside from new train_model)
# -----------------------------------------------------------------------------
def active_learning_round(
    round_num,
    labels_file,
    model_choice,
    request_labels=True,
    out_dir=None,
    save_preds=True,
    return_metrics=False,
    top_n_predictions=None,
    return_predictions=False,
):
    """Run one active learning round.

    Parameters
    ----------
    round_num : int
        Current round number (1-indexed).
    labels_file : str
        CSV with existing labels used for training.
    model_choice : str
        Which model to train ("ResNet", "SVM", or "RandomForest").
    request_labels : bool, optional
        If False, skip the candidate selection/labeling step. This is used for
        the final round so the user isn't prompted for more labels.
    out_dir : str, optional
        Directory to write round outputs to. If None, defaults to
        ``ROUNDS_DIR/round_<round_num>``.
    save_preds : bool, optional
        If False, don't write predictions.csv.
    return_metrics : bool, optional
        If True, return the evaluation metrics instead of the candidate label
        file. Metrics are also returned whenever ``save_preds`` is False or
        ``request_labels`` is False.
    top_n_predictions : int or None, optional
        If given, only the ``top_n_predictions`` most uncertain predictions
        (by |p-0.5|) are written to predictions.csv.
    """
    print(f"\n=== Starting Active Learning Round {round_num} ===")
    rnd_dir = out_dir or os.path.join(ROUNDS_DIR, f"round_{round_num}")
    os.makedirs(rnd_dir, exist_ok=True)

    # load & featurize
    rows = list(csv.DictReader(open(labels_file)))
    if len(rows) <= 1:
        print("Not enough labels; aborting.")
        return None
    X, y = [], []
    from progress_utils import new_progress as _npb
    with _npb() as _prog:
        t = _prog.add_task("Extract training features", total=len(rows))
        for idx, r in enumerate(rows):
            feats = extract_features_from_label(r)
            if feats is not None:
                X.append(feats)
                y.append(1 if r["label"].lower()=="agricultural" else 0)
            # advance per row for accurate ETA
            _prog.update(t, advance=1)
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    # train & save
    print(f"Training data shape: {X.shape}, model: {model_choice}")
    model = train_model(model_choice, X, y)
    mp = os.path.join(rnd_dir, f"model_round_{round_num}.pkl")
    dump(model, mp)
    # keep X,y for representativeness computations
    free_unused_memory()

    # inference + timing (ignore overlay/final-sweep artifacts in RAW_DATA_DIR)
    all_tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    tifs = [tp for tp in all_tifs if ("_overlay" not in os.path.basename(tp) and "_th" not in os.path.basename(tp))]
    start = time.time()

    with new_progress() as prog:
        task = prog.add_task("Running inference", total=len(tifs))

        # Stream predictions to per-tile CSVs to control memory usage
        def run_tile(tp):
            tile_preds = predict_entire_tile(tp, model)
            tile_name = os.path.basename(tp)
            _ = _write_tile_predictions_csv(rnd_dir, tile_name, tile_preds)
            prog.update(task, advance=1)
            return tile_name

        # Threaded tile-level inference to overlap I/O and compute
        results = []
        tile_threads = int(getattr(cfg, 'INFER_TILE_THREADS', 4))
        with ThreadPoolExecutor(max_workers=tile_threads) as ex:
            futs = [ex.submit(run_tile, tp) for tp in tifs]
            for fut in as_completed(futs):
                try:
                    results.append(fut.result())
                except Exception:
                    pass

    # Merge per-tile CSVs and avoid keeping a giant list in memory
    pred_csv_path = _merge_tile_prediction_csvs(rnd_dir)
    # Count rows efficiently
    total_rows = 0
    try:
        with open(pred_csv_path) as f:
            total_rows = sum(1 for _ in f) - 1
    except Exception:
        pass
    print(f"Total pixels inferred: {total_rows}")
    print(f"Inference completed in {str(datetime.timedelta(seconds=int(time.time() - start)))}")

    # Optionally keep only top-N predictions by uncertainty
    preds = None
    if top_n_predictions is not None and pred_csv_path:
        # Stream top-N by |p-0.5|
        import heapq
        heap = []
        with open(pred_csv_path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                p = float(row["predicted_prob"]) if row.get("predicted_prob") not in (None, "") else 0.5
                score = -abs(p - 0.5)
                item = [row.get("tile"), int(row.get("row_idx")), int(row.get("col_idx")),
                        float(row.get("center_lat")), float(row.get("center_lon")), p,
                        float(row.get("ndvi") or 0.0)]
                if len(heap) < top_n_predictions:
                    heapq.heappush(heap, (score, item))
                else:
                    if score > heap[0][0]:
                        heapq.heapreplace(heap, (score, item))
        preds = [it for _s, it in sorted(heap, key=lambda x: x[0], reverse=True)]

    # outputs
    if save_preds:
        if pred_csv_path is None and preds is not None:
            pred_csv_path = save_predictions(rnd_dir, preds)
        save_agricultural_polygons_kml(rnd_dir, round_num, pred_csv=pred_csv_path)
    else:
        print("Using temporary predictions.csv for downstream steps (will delete).")
        # Use the merged CSV for downstream polygonization without keeping it permanently
        save_agricultural_polygons_kml(rnd_dir, round_num, pred_csv=pred_csv_path)

    # Evaluate against optional evaluation set
    stats_dir = os.path.join(rnd_dir, "statistics")
    os.makedirs(stats_dir, exist_ok=True)
    metrics = evaluate_model(model, out_dir=stats_dir)
    if metrics is not None:
        from rich.table import Table
        from rich.console import Console
        tbl = Table(title="Evaluation Metrics")
        tbl.add_column("Metric")
        tbl.add_column("Value", justify="right")
        for k, v in metrics.items():
            tbl.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        Console().print(tbl)
    # snapshot config used
    try:
        snap = {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}
        with open(os.path.join(stats_dir, "config_snapshot.json"), "w") as jf:
            json.dump(snap, jf, indent=2)
    except Exception as e:
        print(f"Config snapshot failed: {e}")
    # Feature importance (permutation) on validation split if enabled
    try:
        if cfg.RUN_PERMUTATION_IMPORTANCE and len(np.unique(y)) > 1:
            tr_idx, va_idx, _ = stratified_train_val_test_indices(
                y, cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.TEST_FRACTION,
                cfg.SPLIT_RANDOM_SEED if cfg.SPLIT_SEED_MODE == "fixed" else None,
            )
            if va_idx.size > 0:
                scorer = get_scorer('f1')
                pi = permutation_importance(model, X[va_idx], y[va_idx], scoring=scorer, n_repeats=5, n_jobs=-1, random_state=0)
                importances = pi.importances_mean
                order = np.argsort(importances)[::-1]
                exp_names = current_feature_names()
                if len(exp_names) == importances.size:
                    names = exp_names
                else:
                    # Fallback: align as much as possible so names appear
                    names = [exp_names[i] if i < len(exp_names) else f"f{i}" for i in range(importances.size)]
                # write text
                with open(os.path.join(stats_dir, 'feature_importance.txt'), 'w') as f:
                    for idx in order:
                        f.write(f"{names[idx]}\t{importances[idx]:.6f}\n")
                # simple bar plot
                try:
                    import matplotlib
                    matplotlib.use('Agg', force=True)
                    import matplotlib.pyplot as plt
                    topk = min(25, len(order))
                    plt.figure(figsize=(8.0, max(3.0, topk*0.3)))
                    plt.barh(range(topk), importances[order][:topk][::-1])
                    plt.yticks(range(topk), [names[i] for i in order][:topk][::-1], fontsize=7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(stats_dir, 'feature_importance.png'), dpi=180)
                    plt.close()
                except Exception as e:
                    print(f"Feature importance plot failed: {e}")
    except Exception as e:
        print(f"Permutation importance skipped: {e}")

    # Update persistent informative lists via the standalone script. This keeps
    # the round runner robust even if refresh encounters native-lib issues.
    pred_csv_arg = os.path.join(rnd_dir, "predictions.csv")
    try:
        script = os.path.join(os.path.dirname(__file__), "refresh_lists.py")
        print("Refreshing persistent lists via refresh_lists.py ...")
        subprocess.run([sys.executable, script, pred_csv_arg], check=True)
    except Exception as e:
        print(f"Persistent list update error: {e}")

    if return_metrics or not request_labels:
        print(f"Round {round_num} complete (no candidate labeling).")
        if return_predictions:
            return {"metrics": metrics, "pred_csv": os.path.join(rnd_dir, "predictions.csv")}
        # Safe to delete predictions.csv if not preserving and not returning its path
        try:
            if not save_preds and pred_csv_path and os.path.exists(pred_csv_path):
                os.remove(pred_csv_path)
        except Exception:
            pass
        return metrics

    if preds is not None:
        tmp = candidate_selection_from_predictions(
            preds,
            rnd_dir,
            round_num,
            train_rows=rows,
            X_train=X,
            y_train=y,
        )
    else:
        tmp = candidate_selection_from_csv(
            os.path.join(rnd_dir, "predictions.csv"),
            rnd_dir,
            round_num,
            train_rows=rows,
            X_train=X,
            y_train=y,
        )
    # After candidate selection, remove temporary predictions.csv if it was only used internally
    try:
        if not save_preds and pred_csv_path and os.path.exists(pred_csv_path):
            os.remove(pred_csv_path)
    except Exception:
        pass
    print(f"Round {round_num} complete; labels at {tmp}")
    return tmp


def _update_persistent_lists(preds, train_rows, X_train, y_train, round_num, round_dir, pred_csv=None):
    """Update global and per-round highscore/probableAgri CSVs and KMLs.

    preds: list of [tile,row,col,lat,lon,prob,ndvi]
    train_rows: label rows used for training (for feature-space representativeness)
    X_train, y_train: feature matrix and labels (scaled later inside model)
    """
    import csv as _csv
    # build set of master labels to exclude (labels.csv only; temp labels allowed to remain)
    labeled_keys = set()
    try:
        import config as _cfg
        from splits import load_labels as _load_labels
        master_rows = _load_labels(_cfg.LABELS_FILE) if os.path.exists(_cfg.LABELS_FILE) else []
        for r in master_rows:
            try:
                tile = r.get("tile"); lat = float(r.get("lat")); lon = float(r.get("lon"))
            except Exception:
                continue
            labeled_keys.add(f"{tile}:{lat:.7f}:{lon:.7f}")
    except Exception:
        pass

    # compute uncertainty (entropy)
    entropy = None
    # choose a pool for highscore: top by uncertainty (streaming to avoid RAM blow-up)
    pool_size = max(cfg.HIGHSCORE_TOP_K * 10, cfg.HIGHSCORE_TOP_K)
    if not pool_size or pool_size <= 0:
        # default pool when unlimited: keep the queue bounded to a large, but finite size
        pool_size = 50000
    print("Updating persistent lists: selecting high-entropy pool...")
    if preds is not None:
        probs = np.array([p[5] for p in preds], dtype=np.float32)
        # Use margin (|p-0.5|) as a monotonic proxy for entropy to save log calls
        margin = np.abs(probs - 0.5)
        pool_idx = np.argsort(margin)[:pool_size]
        source_iter = ((i, preds[i]) for i in pool_idx)
    else:
        import heapq
        heap = []
        # maintain top pool_size by entropy with a progress bar over file bytes
        file_size = os.path.getsize(pred_csv) if os.path.exists(pred_csv) else 0
        from progress_utils import new_progress as _npb
        with open(pred_csv, newline="") as f, _npb() as _prog:
            task = _prog.add_task("Scan predictions (entropy)", total=file_size or None)
            r = csv.reader(f)
            header = next(r, None)
            # Expected header: tile,row_idx,col_idx,center_lat,center_lon,predicted_prob,ndvi
            # Use fixed indices for speed
            IDX_TILE, IDX_R, IDX_C, IDX_LAT, IDX_LON, IDX_P, IDX_NDVI = 0,1,2,3,4,5,6
            last_tell = 0
            for i, row in enumerate(r):
                try:
                    p = float(row[IDX_P])
                except Exception:
                    p = 0.5
                # Use negative margin as a key: larger = closer to 0.5 => more uncertain
                key = -abs(p - 0.5)
                try:
                    item = [row[IDX_TILE], int(row[IDX_R]), int(row[IDX_C]),
                            float(row[IDX_LAT]), float(row[IDX_LON]), p,
                            float(row[IDX_NDVI]) if len(row) > 6 and row[IDX_NDVI] != '' else 0.0]
                except Exception:
                    # Skip badly-formed lines
                    continue
                if len(heap) < pool_size:
                    heapq.heappush(heap, (key, i, item))
                else:
                    if key > heap[0][0]:
                        heapq.heapreplace(heap, (key, i, item))
                # progress by file bytes read (best-effort)
                try:
                    cur = f.tell()
                    if cur > last_tell:
                        _prog.update(task, completed=cur)
                        last_tell = cur
                except Exception:
                    pass
        # largest uncertainty (closest to 0.5) first
        source_iter = ((i, it) for _e, i, it in sorted(heap, key=lambda t: t[0], reverse=True))

    # feature-space representativeness: distance to nearest labeled in standardized space
    # derive means/std from X_train
    feat_means = np.nanmean(X_train, axis=0).astype(np.float32)
    feat_std = np.nanstd(X_train, axis=0).astype(np.float32)
    feat_std[feat_std == 0] = 1.0
    # build list of candidate feature vectors
    def feat_at(tile, r, c):
        tf = get_tile_features(tile)
        if tf is None:
            return None
        arr, _, _ = tf
        if r < 0 or c < 0 or r >= arr.shape[1] or c >= arr.shape[2]:
            return None
        return arr[:, r, c].astype(np.float32)

    cand_feats = []
    cand_meta = []
    # Collect candidate metadata first (skip labeled), then fetch features in tile batches
    print("Gathering candidate features per tile...")
    tmp_meta = []
    for i, row in source_iter:
        tile, r, c, la, lo, p, ndvi = row
        if f"{tile}:{la:.7f}:{lo:.7f}" in labeled_keys:
            continue
        tmp_meta.append((i, tile, r, c, la, lo, p, ndvi))
    if not tmp_meta:
        return
    # Group by tile to avoid repeated loads and slice features in batches
    from collections import defaultdict as _dd
    by_tile = _dd(list)
    for idx, (i, tile, r, c, la, lo, p, ndvi) in enumerate(tmp_meta):
        by_tile[tile].append((idx, r, c))
    # Determine feature dimension from first available tile
    feat_dim = None
    cand_feats = [None] * len(tmp_meta)
    from progress_utils import new_progress as _npb
    with _npb() as _prog:
        tfeat = _prog.add_task("Load candidate features", total=len(tmp_meta))
        loaded = 0
        for tile, items in by_tile.items():
            tf = get_tile_features(tile)
            if tf is None:
                loaded += len(items)
                _prog.update(tfeat, advance=len(items))
                continue
            arr, _, _ = tf
            if feat_dim is None:
                feat_dim = int(arr.shape[0])
            # Vectorized gather for this tile
            if items:
                rr = np.array([r for _, r, _ in items], dtype=np.int32)
                cc = np.array([c for _, _, c in items], dtype=np.int32)
                # Advanced indexing to fetch all at once: (bands, n)
                feats_tile = arr[:, rr, cc].transpose(1, 0).astype(np.float32)
                for (idx, _r, _c), fv in zip(items, feats_tile):
                    cand_feats[idx] = fv
            loaded += len(items)
            _prog.update(tfeat, advance=len(items))
    # Filter out any entries that failed to load features
    for meta, f in zip(tmp_meta, cand_feats):
        if f is not None:
            cand_meta.append(meta)
    cand_feats = [f for f in cand_feats if f is not None]
    if not cand_feats:
        return
    CF = np.vstack(cand_feats)
    # standardize both candidate and train features with train stats
    Xs = (X_train - feat_means) / (feat_std + 1e-6)
    CFs = (CF - feat_means) / (feat_std + 1e-6)
    # compute L2 distance to nearest labeled using an exact NN index (faster, same result)
    print("Computing nearest-label distances...")
    try:
        from sklearn.neighbors import NearestNeighbors as _NN
        nn = _NN(n_neighbors=1, algorithm='auto', metric='euclidean')
        nn.fit(Xs)
        core_dist = nn.kneighbors(CFs, n_neighbors=1, return_distance=True)[0].reshape(-1)
    except Exception:
        # Fallback to brute-force if sklearn is unavailable
        core_dist = np.sqrt(((CFs[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    # normalize components 0-1 (rank-based)
    def rank_norm(v):
        r = np.argsort(np.argsort(v))
        return r.astype(np.float32) / max(len(v) - 1, 1)
    # Compute entropy per-candidate directly from their probabilities
    ent_values = np.array([
        -m[6]*np.log(m[6] + 1e-9) - (1 - m[6])*np.log(1 - m[6] + 1e-9)
        for m in cand_meta
    ], dtype=np.float32)
    U = rank_norm(ent_values)
    R = rank_norm(core_dist)
    # consistency over rounds (simple: if near threshold this round, treat as consistent=1 else 0; can be extended via global counts)
    C = (np.abs(np.array([m[6] for m in cand_meta]) - cfg.MIN_AGRI_PROB) < cfg.UNCERTAINTY_BAND_DELTA).astype(np.float32)
    # composite
    wu = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("uncertainty", 0.5)
    wr = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("representativeness", 0.3)
    wc = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("consistency", 0.2)
    score = wu * U + wr * R + wc * C
    order = np.argsort(score)[::-1]

    # merge with existing global highscore and update counts/round indices
    os.makedirs(os.path.dirname(cfg.HIGHSCORE_FILE), exist_ok=True)
    existing = {}
    if os.path.exists(cfg.HIGHSCORE_FILE):
        try:
            with open(cfg.HIGHSCORE_FILE) as f:
                for r in _csv.DictReader(f):
                    key = f"{r.get('tile')}:{r.get('row')}:{r.get('col')}"
                    existing[key] = r
        except Exception:
            existing = {}
    updated = {}
    # mark top-K from this round
    # Select indices to include from this round (no limit when HIGHSCORE_TOP_K <= 0)
    if getattr(cfg, 'HIGHSCORE_TOP_K', 0) and cfg.HIGHSCORE_TOP_K > 0:
        selected_idx = order[: cfg.HIGHSCORE_TOP_K]
    else:
        selected_idx = order
    for k in selected_idx:
        i, tile, r, c, la, lo, p, ndvi = cand_meta[k]
        key = f"{tile}:{r}:{c}"
        prev = existing.get(key, {})
        times = int(prev.get('times_selected', 0)) + 1
        first_r = int(prev.get('first_round', round_num))
        updated[key] = {
            "tile": tile,
            "row": r,
            "col": c,
            "lat": la,
            "lon": lo,
            "prob": p,
            "entropy": float(ent_values[k]),
            "ndvi": ndvi,
            "score": float(score[k]),
            "times_selected": times,
            "first_round": first_r,
            "last_round": round_num,
        }
    # carry forward non-selected existing entries (to allow persistence across runs), then prune to TOP_K by score
    for key, r in existing.items():
        if key not in updated:
            updated[key] = r
    # drop any that are now in master labels (labels.csv only)
    updated = {k: v for k, v in updated.items() if f"{v.get('tile')}:{float(v.get('lat')):.7f}:{float(v.get('lon')):.7f}" not in labeled_keys}
    # sort by score desc, keep top K
    def score_of(v):
        try:
            return float(v.get('score', 0))
        except Exception:
            return 0.0
    top_items = sorted(updated.values(), key=score_of, reverse=True)
    if getattr(cfg, 'HIGHSCORE_TOP_K', 0) and cfg.HIGHSCORE_TOP_K > 0:
        top_items = top_items[: cfg.HIGHSCORE_TOP_K]
    # persist CSV and KML
    if top_items:
        with open(cfg.HIGHSCORE_FILE, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(top_items[0].keys()))
            w.writeheader(); w.writerows(top_items)
        # Polygonized KML with opacity by score
        try:
            topk = int(getattr(cfg, 'HIGHSCORE_KML_TOP_PIXELS', 50000))
        except Exception:
            topk = 50000
        _write_ranked_pixel_kml(top_items, cfg.HIGHSCORE_KML_GLOBAL, weight_key='score', top_k=topk)

    # probableAgri: top positives by confidence × representativeness
    pos_meta = []
    pos_feats = []
    if preds is None and pred_csv is not None:
        print("Selecting top positive pool from predictions.csv for probableAgri...")
        import heapq as _hq
        pool = []
        pool_size = max(cfg.PROBABLE_AGRI_TOP_K * 10, cfg.PROBABLE_AGRI_TOP_K)
        if not pool_size or pool_size <= 0:
            pool_size = 100000
        with open(pred_csv, newline="") as _f:
            _r = csv.reader(_f); _ = next(_r, None)
            for i, row in enumerate(_r):
                try:
                    p = float(row[5])
                except Exception:
                    continue
                if p < cfg.MIN_AGRI_PROB:
                    continue
                try:
                    item = [row[0], int(row[1]), int(row[2]), float(row[3]), float(row[4]), p,
                            float(row[6]) if len(row) > 6 and row[6] != '' else 0.0]
                except Exception:
                    continue
                key = p  # higher prob preferred for pool
                if len(pool) < pool_size:
                    _hq.heappush(pool, (key, i, item))
                else:
                    if key > pool[0][0]:
                        _hq.heapreplace(pool, (key, i, item))
        # gather features per tile in batches
        ordered = [it for _k, _i, it in sorted(pool, key=lambda t: t[0], reverse=True)]
        # filter out already-labeled
        filtered = [it for it in ordered if f"{it[0]}:{it[3]:.7f}:{it[4]:.7f}" not in labeled_keys]
        # group by tile
        from collections import defaultdict as _dd
        by_tile2 = _dd(list)
        for idx, rec in enumerate(filtered):
            by_tile2[rec[0]].append((idx, rec[1], rec[2]))
        feats_tmp = [None] * len(filtered)
        for tile, items in by_tile2.items():
            tf = get_tile_features(tile)
            if tf is None:
                continue
            arr, _, _ = tf
            rr = np.array([r for _, r, _ in items], dtype=np.int32)
            cc = np.array([c for _, _, c in items], dtype=np.int32)
            sel = arr[:, rr, cc].transpose(1, 0).astype(np.float32)
            for (idx, _r, _c), fv in zip(items, sel):
                feats_tmp[idx] = fv
        for rec, fv in zip(filtered, feats_tmp):
            if fv is None:
                continue
            tile, r, c, la, lo, p, ndvi = rec
            pos_meta.append((0, tile, r, c, la, lo, p, ndvi))
            pos_feats.append(fv)
    else:
        probs_all = np.array([p[5] for p in preds])
        idx_pos = np.where(probs_all >= cfg.MIN_AGRI_PROB)[0]
        for i in idx_pos:
            tile, r, c, la, lo, p, ndvi = preds[i]
            if f"{tile}:{la:.7f}:{lo:.7f}" in labeled_keys:
                continue
            f = feat_at(tile, r, c)
            if f is None:
                continue
            pos_meta.append((i, tile, r, c, la, lo, p, ndvi))
            pos_feats.append(f)
    if pos_meta:
        print("Ranking probable-agri positives...")
        PF = np.vstack(pos_feats)
        valid = ~np.isnan(PF).any(axis=1)
        PF = PF[valid]
        pos_meta = [pm for pm, ok in zip(pos_meta, valid) if ok]
        # Compute representativeness distances for positives using training set stats
        if X_train is not None:
            try:
                has_train2 = np.asarray(X_train).size > 0
            except Exception:
                has_train2 = False
        else:
            has_train2 = False
        if has_train2:
            feat_means2 = np.nanmean(X_train, axis=0).astype(np.float32)
            feat_std2 = np.nanstd(X_train, axis=0).astype(np.float32)
            feat_std2[feat_std2 == 0] = 1.0
            PFs = (PF - feat_means2) / (feat_std2 + 1e-6)
            try:
                from scipy.spatial import cKDTree as _KD
                kd2 = _KD((X_train - feat_means2) / (feat_std2 + 1e-6))
                dpos, _ = kd2.query(PFs, k=1, workers=int(getattr(cfg, 'REFRESH_KD_WORKERS', 1)))
                dpos = dpos.astype(np.float32, copy=False)
            except Exception:
                dpos = np.sqrt(((PFs[:, None, :] - ((X_train - feat_means2) / (feat_std2 + 1e-6))[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        else:
            PFs = PF
            dpos = np.zeros((PFs.shape[0],), dtype=np.float32)
        P = rank_norm(np.array([m[6] for m in pos_meta]))
        Rpos = rank_norm(dpos)
        wconf = cfg.PROBABLE_AGRI_COMPONENT_WEIGHTS.get("confidence", 0.7)
        wrep = cfg.PROBABLE_AGRI_COMPONENT_WEIGHTS.get("representativeness", 0.3)
        ps = wconf * P + wrep * Rpos
        orderp = np.argsort(ps)[::-1]
        pa_rows = []
        for k in orderp:
            i, tile, r, c, la, lo, p, ndvi = pos_meta[k]
            pa_rows.append({
                "tile": tile,
                "row": r,
                "col": c,
                "lat": la,
                "lon": lo,
                "prob": p,
                "ndvi": ndvi,
                "score": float(ps[k]),
                "round": round_num,
            })
        # merge with existing probableAgri persistently, prune to TOP_K by score
        existing_pa = {}
        if os.path.exists(cfg.PROBABLE_AGRI_FILE):
            try:
                with open(cfg.PROBABLE_AGRI_FILE) as f:
                    for r in _csv.DictReader(f):
                        key = f"{r.get('tile')}:{r.get('row')}:{r.get('col')}"
                        existing_pa[key] = r
            except Exception:
                existing_pa = {}
        for r in pa_rows:
            key = f"{r['tile']}:{r['row']}:{r['col']}"
            existing_pa[key] = r
        # drop labeled (master labels only)
        existing_pa = {k: v for k, v in existing_pa.items() if f"{v.get('tile')}:{float(v.get('lat')):.7f}:{float(v.get('lon')):.7f}" not in labeled_keys}
        # Sort by probability descending (most probable agri first); no limit if TOP_K <= 0
        top_pa = sorted(existing_pa.values(), key=lambda v: float(v.get('prob', 0)), reverse=True)
        if getattr(cfg, 'PROBABLE_AGRI_TOP_K', 0) and cfg.PROBABLE_AGRI_TOP_K > 0:
            top_pa = top_pa[: cfg.PROBABLE_AGRI_TOP_K]
        if top_pa:
            with open(cfg.PROBABLE_AGRI_FILE, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(top_pa[0].keys()))
                w.writeheader(); w.writerows(top_pa)
            # Polygonized KML with opacity by probability
            try:
                topk_pa = int(getattr(cfg, 'PROBABLE_AGRI_KML_TOP_PIXELS', 50000))
            except Exception:
                topk_pa = 50000
            _write_ranked_pixel_kml(top_pa, cfg.PROBABLE_AGRI_KML_GLOBAL, weight_key='prob', top_k=topk_pa)
    print("Persistent lists updated.")


def _write_points_kml(rows, out_path, placemark_prefix="#"):
    """Write KML with hidden icon and red pixel perimeter polygon.

    rows: list of dicts with fields tile,row,col,lat,lon,score
    """
    kml = Element('kml'); kml.set('xmlns', 'http://www.opengis.net/kml/2.2')
    doc = SubElement(kml, 'Document')
    style = SubElement(doc, 'Style', id='pt')
    ic = SubElement(style, 'IconStyle'); SubElement(ic, 'scale').text = '0'
    ln = SubElement(style, 'LineStyle'); SubElement(ln, 'color').text = 'ff0000ff'; SubElement(ln, 'width').text = '2'
    ps = SubElement(style, 'PolyStyle'); SubElement(ps, 'fill').text = '0'; SubElement(ps, 'outline').text = '1'
    count = 0
    for rank, r in enumerate(rows, 1):
        pm = SubElement(doc, 'Placemark')
        # Safe score formatting: accept both numeric and string fields, fallback to prob or 0.0
        raw_score = r.get('score', None)
        if raw_score is None:
            raw_score = r.get('prob', 0.0)
        try:
            sc = float(raw_score)
        except Exception:
            sc = 0.0
        SubElement(pm, 'name').text = f"{placemark_prefix}{rank} {r['tile']} score={sc:.3f}"
        SubElement(pm, 'styleUrl').text = '#pt'
        # compute exact pixel perimeter from tile geotransform
        tif = os.path.join(RAW_DATA_DIR, r['tile'])
        try:
            with rasterio.open(tif) as src:
                corners = get_pixel_corners(src, int(r['row']), int(r['col']))
        except Exception:
            continue
        poly = SubElement(pm, 'Polygon')
        ob = SubElement(poly, 'outerBoundaryIs')
        ring = SubElement(ob, 'LinearRing')
        SubElement(ring, 'coordinates').text = ' '.join(f"{x},{y},0" for x,y in corners)
        count += 1
    xml = parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
    with open(out_path, 'wb') as f:
        f.write(xml)


def _write_weighted_polygons_kml(rows, out_path, weight_key='score'):
    """Polygonize per-tile selected pixels and write KML with opacity
    proportional to weight (e.g., score or prob). Uses per-placemark inline
    style so each polygon can have its own alpha.

    rows: list of dicts with fields: tile,row,col,lat,lon,<weight_key>
    """
    from collections import defaultdict
    import numpy as _np
    # Group rows by tile
    by_tile = defaultdict(list)
    w_vals = []
    for r in rows:
        try:
            tile = r.get('tile')
            rr = int(r.get('row'))
            cc = int(r.get('col'))
            w = float(r.get(weight_key, r.get('prob', 0.0)))
        except Exception:
            continue
        by_tile[tile].append((rr, cc, w))
        w_vals.append(w)
    if not by_tile:
        return
    w_min = float(min(w_vals)) if w_vals else 0.0
    w_max = float(max(w_vals)) if w_vals else 1.0
    if w_max <= w_min:
        w_max = w_min + 1.0
    # Build KML doc
    kml = Element('kml'); kml.set('xmlns','http://www.opengis.net/kml/2.2')
    doc = SubElement(kml, 'Document')
    # Static base style (outline)
    base_style = SubElement(doc, 'Style', id='polybase')
    ln = SubElement(base_style, 'LineStyle'); SubElement(ln, 'color').text = 'ff0000ff'; SubElement(ln, 'width').text = '1'
    # Iterate tiles (keep tiles hot)
    from scipy.ndimage import label as _label
    for tile, items in by_tile.items():
        tif = os.path.join(RAW_DATA_DIR, tile)
        if not os.path.exists(tif):
            continue
        with rasterio.open(tif) as src:
            H, W = src.height, src.width
            mask = _np.zeros((H, W), dtype=_np.uint8)
            vals = _np.zeros((H, W), dtype=_np.float32)
            for r, c, w in items:
                if 0 <= r < H and 0 <= c < W:
                    mask[r, c] = 1
                    vals[r, c] = w
            if not mask.any():
                continue
            lbl, ncomp = _label(mask)
            if ncomp == 0:
                continue
            # Average weight per component
            sums = _np.bincount(lbl.reshape(-1), weights=vals.reshape(-1), minlength=ncomp+1)
            counts = _np.bincount(lbl.reshape(-1), minlength=ncomp+1)
            means = _np.divide(sums, _np.maximum(counts, 1), out=_np.zeros_like(sums), where=counts>0)
            # shapes over labeled image yields one polygon per component with value=label id
            transformer = None
            if src.crs and not src.crs.is_geographic:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            for geom, val in shapes(lbl.astype('int32'), mask=(lbl>0), transform=src.transform):
                lab = int(val)
                if lab <= 0 or lab >= means.size:
                    continue
                poly = shape(geom)
                if transformer:
                    poly = shp_transform(transformer.transform, poly)
                wmean = float(means[lab])
                # Normalize to [0,1]
                alpha = (wmean - w_min) / (w_max - w_min)
                alpha = float(_np.clip(alpha, 0.0, 1.0))
                # Alpha to ABGR hex (00=transparent, ff=opaque)
                a = max(0, min(255, int(round(alpha * 255))))
                ahex = f"{a:02x}"
                # Red fill: ABGR = aabbggrr; blue=00, green=00, red=ff
                color = ahex + '0000ff'
                def _emit_polygon(pgeom):
                    pm = SubElement(doc, 'Placemark')
                    st = SubElement(pm, 'Style')
                    ln = SubElement(st, 'LineStyle'); SubElement(ln, 'color').text = 'ff0000ff'; SubElement(ln, 'width').text = '1'
                    ps = SubElement(st, 'PolyStyle'); SubElement(ps, 'color').text = color; SubElement(ps, 'outline').text = '1'
                    poly_el = SubElement(pm, 'Polygon')
                    ob = SubElement(poly_el, 'outerBoundaryIs')
                    ring = SubElement(ob, 'LinearRing')
                    coords = list(pgeom.exterior.coords)
                    SubElement(ring, 'coordinates').text = ' '.join(f"{lon},{lat},0" for lon, lat in coords)
                if isinstance(poly, Polygon):
                    _emit_polygon(poly)
                elif isinstance(poly, MultiPolygon):
                    for p2 in poly.geoms:
                        _emit_polygon(p2)
    xml = parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
    with open(out_path, 'wb') as f:
        f.write(xml)

def _write_ranked_pixel_kml(rows, out_path, weight_key='score', top_k=0):
    """Write KML with exactly 4 placemarks: Red, Orange, Yellow, Baby Blue.

    - Groups pixels by quartiles of weight and merges contiguous pixels into
      polygons per group using raster polygonization.
    - Produces a single Placemark per group with MultiGeometry containing all
      polygons across tiles.
    - top_k caps the number of pixels considered before grouping.
    """
    import numpy as _np
    from collections import defaultdict
    # Extract (tile,row,col,weight)
    items = []
    for r in rows:
        try:
            tile = r.get('tile'); rr = int(r.get('row')); cc = int(r.get('col'))
            w = float(r.get(weight_key, r.get('prob', 0.0)))
        except Exception:
            continue
        if not tile:
            continue
        items.append((tile, rr, cc, w))
    if not items:
        return
    # Sort by weight desc and cap
    items.sort(key=lambda t: t[3], reverse=True)
    if top_k and top_k > 0:
        items = items[:top_k]
    weights = _np.array([t[3] for t in items], dtype=float)
    if weights.size == 0:
        return
    q25, q50, q75 = _np.percentile(weights, [25, 50, 75])
    # Group per quartile index: 3=red (top), 2=orange, 1=yellow, 0=blue
    def _bin_idx(w):
        if w >= q75: return 3
        if w >= q50: return 2
        if w >= q25: return 1
        return 0
    groups = [defaultdict(list) for _ in range(4)]  # per group: tile -> [(r,c),...]
    for tile, rr, cc, w in items:
        groups[_bin_idx(w)][tile].append((rr, cc))

    # KML doc with 4 placemarks
    kml = Element('kml'); kml.set('xmlns','http://www.opengis.net/kml/2.2')
    doc = SubElement(kml, 'Document')
    # Colors (ABGR) with fixed alpha
    def _abgr(r,g,b,a=0x66):
        return f"{a:02x}{b:02x}{g:02x}{r:02x}"
    styles = [
        ("Baby Blue (Bottom 25%)", _abgr(173,216,230), _abgr(173,216,230,0xff)),
        ("Yellow (50–75%)",       _abgr(255,255,0),   _abgr(255,255,0,0xff)),
        ("Orange (25–50%)",       _abgr(255,165,0),   _abgr(255,165,0,0xff)),
        ("Red (Top 25%)",         _abgr(255,0,0),     _abgr(255,0,0,0xff)),
    ]
    for gi, (name, fill_col, line_col) in enumerate(styles):
        pm = SubElement(doc, 'Placemark')
        SubElement(pm, 'name').text = name
        st = SubElement(pm, 'Style')
        ln = SubElement(st, 'LineStyle'); SubElement(ln, 'color').text = line_col; SubElement(ln, 'width').text = '1'
        ps = SubElement(st, 'PolyStyle'); SubElement(ps, 'color').text = fill_col; SubElement(ps, 'outline').text = '1'
        mg = SubElement(pm, 'MultiGeometry')
        # For each tile, polygonize mask of this group
        for tile, pts in groups[gi].items():
            tif = os.path.join(RAW_DATA_DIR, tile)
            if not os.path.exists(tif) or not pts:
                continue
            try:
                with rasterio.open(tif) as src:
                    H, W = src.height, src.width
                    import numpy as _np
                    mask = _np.zeros((H, W), dtype=_np.uint8)
                    for r, c in pts:
                        if 0 <= r < H and 0 <= c < W:
                            mask[r, c] = 1
                    if not mask.any():
                        continue
                    transformer = None
                    if src.crs and not src.crs.is_geographic:
                        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                    for geom, val in shapes(mask.astype('int32'), mask=(mask>0), transform=src.transform):
                        g = shape(geom)
                        if transformer:
                            g = shp_transform(transformer.transform, g)
                        def emit_polygon(pgeom):
                            poly_el = SubElement(mg, 'Polygon')
                            ob = SubElement(poly_el, 'outerBoundaryIs')
                            ring = SubElement(ob, 'LinearRing')
                            coords = list(pgeom.exterior.coords)
                            SubElement(ring, 'coordinates').text = ' '.join(f"{lon},{lat},0" for lon, lat in coords)
                        if isinstance(g, Polygon):
                            emit_polygon(g)
                        elif isinstance(g, MultiPolygon):
                            for p2 in g.geoms:
                                emit_polygon(p2)
            except Exception:
                continue
    xml = parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
    with open(out_path, 'wb') as f:
        f.write(xml)

def _split_predictions_to_shards(pred_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    handles = {}
    writers = {}
    counts = {}
    size = os.path.getsize(pred_csv) if os.path.exists(pred_csv) else 0
    from progress_utils import new_progress as _npb
    with open(pred_csv, 'r', encoding='utf-8', errors='replace') as f, _npb() as _prog:
        task = _prog.add_task("Split predictions into shards", total=size or None)
        processed = 0
        # header
        header = f.readline()
        processed += len(header.encode('utf-8', 'replace'))
        if size:
            _prog.update(task, completed=processed)
        for line in f:
            processed += len(line.encode('utf-8', 'replace'))
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            tile, r, c, la, lo, p = parts[:6]
            nd = parts[6] if len(parts) > 6 else ''
            if tile not in handles:
                tp = os.path.join(out_dir, f"{os.path.splitext(tile)[0]}.csv.gz")
                fh = _open_text_auto(tp, 'wt')
                handles[tile] = fh
                w = csv.writer(fh); writers[tile] = w
                w.writerow(['row','col','lat','lon','prob','ndvi'])
                counts[tile] = 0
            writers[tile].writerow([r, c, la, lo, p, nd])
            counts[tile] += 1
            if size and processed % (1024*1024) < 1000:  # update roughly each MB
                _prog.update(task, completed=min(processed, size))
        if size:
            _prog.update(task, completed=size)
    for fh in handles.values():
        try: fh.close()
        except Exception: pass
    return counts

def refresh_global_lists_full(pred_csv, round_dir, round_num, train_rows, X_train, y_train):
    print("Refreshing global highscore and probableAgri from all pixels...")
    # master labels only
    labeled_keys = set()
    for r in train_rows:
        try:
            tile = r["tile"]; lat = float(r["lat"]); lon = float(r["lon"])
        except Exception:
            continue
        labeled_keys.add(f"{tile}:{lat:.7f}:{lon:.7f}")

    # Feature stats and NN index (fallback when no training features available)
    has_train = (X_train is not None)
    try:
        has_train = has_train and (np.asarray(X_train).size > 0)
    except Exception:
        has_train = False
    if has_train:
        feat_means = np.nanmean(X_train, axis=0).astype(np.float32)
        feat_std = np.nanstd(X_train, axis=0).astype(np.float32)
        feat_std[feat_std == 0] = 1.0
        Xs = (X_train - feat_means) / (feat_std + 1e-6)
        # Build one cKDTree index over training features (exact 1-NN)
        kd = cKDTree(Xs)
    else:
        feat_means = None
        feat_std = None
        Xs = None
        kd = None
        # Descriptive log about fallback
        n_rows = len(train_rows) if train_rows is not None else 0
        try:
            xshape = getattr(np.asarray(X_train), 'shape', None)
        except Exception:
            xshape = None
        print("Representativeness distances disabled (R=0):\n"
              f"- Reason: no usable training features. train_rows={n_rows}, X_train shape={xshape}.\n"
              "- Highscore uses Uncertainty (U) + Consistency (C).\n"
              "- ProbableAgri uses Probability only.")
    # Query helper using cKDTree (or zeros when no training features)
    def _query_dists(Q):
        if kd is None:
            try:
                n = int(Q.shape[0])
            except Exception:
                n = len(Q) if hasattr(Q, '__len__') else 0
            return np.zeros((n,), dtype=np.float32)
        d, _ = kd.query(Q, k=1, workers=int(getattr(cfg, 'REFRESH_KD_WORKERS', 1)))
        return d.astype(np.float32, copy=False)

    tmp_root = os.path.join(round_dir, '_global_refresh')
    shards_dir = os.path.join(tmp_root, 'shards')
    metrics_dir = os.path.join(tmp_root, 'metrics')
    dists_dir = os.path.join(tmp_root, 'dists')
    ranks_dir = os.path.join(tmp_root, 'ranks')
    scored_dir = os.path.join(tmp_root, 'scored')
    pos_dir = os.path.join(tmp_root, 'pos')
    for d in [shards_dir, metrics_dir, dists_dir, ranks_dir, scored_dir, pos_dir]:
        os.makedirs(d, exist_ok=True)

    # 1) split predictions into per-tile shards
    _ = _split_predictions_to_shards(pred_csv, shards_dir)

    # 2) per-tile metrics and positives
    from progress_utils import new_progress as _npb
    with _npb() as prog_metrics:
        task_metrics = prog_metrics.add_task("Per-tile metrics", total=0)
        prog_metrics.update(task_metrics, total=len(_list_csvs(shards_dir)))

        def per_tile(tile_file):
            base = os.path.basename(tile_file)
            if base.endswith('.csv.gz'):
                base_core = base[:-7]  # strip .csv.gz
            elif base.endswith('.csv'):
                base_core = base[:-4]
            else:
                base_core = os.path.splitext(base)[0]
            tile_name = base_core + '.tif'

            # Prepare writers and accumulators
            mp = os.path.join(metrics_dir, base_core + '_metrics.csv.gz')
            dists_parts = []
            pos_parts = []
            pos_count = 0
            total_rows = 0

            # Load features for this tile once if needed
            arr = None
            if has_train:
                tf = get_tile_features(tile_name)
                if tf is None:
                    return (tile_name, 0, 0)
                arr, _, _ = tf

            # Chunked scan of shard rows (no metrics CSV; only distance parts and pos parts)
            chunk_r = []
            chunk_c = []
            chunk_la = []
            chunk_lo = []
            chunk_p = []
            chunk_nd = []

            def flush_chunk(part_idx):
                nonlocal pos_count, total_rows
                if not chunk_r:
                    return part_idx
                rr = np.asarray(chunk_r, dtype=np.int32)
                cc = np.asarray(chunk_c, dtype=np.int32)
                la = np.asarray(chunk_la, dtype=np.float32)
                lo = np.asarray(chunk_lo, dtype=np.float32)
                pr = np.asarray(chunk_p, dtype=np.float32)
                nd = np.asarray(chunk_nd, dtype=np.float32)
                total_rows += rr.size
                # distances
                if has_train:
                    feats = arr[:, rr, cc].transpose(1, 0).astype(np.float32)
                    valid = ~np.isnan(feats).any(axis=1)
                    if not np.all(valid):
                        rr, cc, la, lo, pr, nd, feats = (
                            rr[valid], cc[valid], la[valid], lo[valid], pr[valid], nd[valid], feats[valid]
                        )
                    feats_s = (feats - feat_means) / (feat_std + 1e-6)
                    dists = _query_dists(feats_s)
                else:
                    dists = np.zeros((rr.shape[0],), dtype=np.float32)
                # Write sorted dists as a part file for this chunk
                order = np.argsort(dists)
                dp_part = os.path.join(dists_dir, f"{base_core}_dists.part{part_idx}.csv.gz")
                with _open_text_auto(dp_part, 'wt') as fd:
                    w = csv.writer(fd)
                    w.writerow(['tile','row','col','dist'])
                    for j in order:
                        w.writerow([tile_name, int(rr[j]), int(cc[j]), float(dists[j])])
                dists_parts.append(dp_part)
                # Positives sorted by prob desc for this chunk
                idx_pos = np.where(pr >= cfg.MIN_AGRI_PROB)[0]
                if idx_pos.size:
                    pos_count += int(idx_pos.size)
                    orderp = idx_pos[np.argsort(pr[idx_pos])[::-1]]
                    pp_part = os.path.join(pos_dir, f"{base_core}_pos.part{part_idx}.csv.gz")
                    with _open_text_auto(pp_part, 'wt') as fp:
                        w = csv.writer(fp)
                        w.writerow(['tile','row','col','lat','lon','prob','ndvi'])
                        for j in orderp:
                            w.writerow([tile_name, int(rr[j]), int(cc[j]), float(la[j]), float(lo[j]), float(pr[j]), float(nd[j])])
                    pos_parts.append(pp_part)
                # reset chunk
                chunk_r.clear(); chunk_c.clear(); chunk_la.clear(); chunk_lo.clear(); chunk_p.clear(); chunk_nd.clear()
                return part_idx + 1

            part_idx = 0
            CHUNK = int(getattr(cfg, 'REFRESH_CHUNK_ROWS', 200000))
            with _open_text_auto(tile_file, 'rt') as f:
                rd = csv.DictReader(f)
                for r in rd:
                    try:
                        ri = int(r['row']); ci = int(r['col'])
                        la = float(r['lat']); lo = float(r['lon'])
                        p = float(r['prob']); ndv = float(r['ndvi'] or 0.0)
                    except Exception:
                        continue
                    # skip master labels
                    key = f"{tile_name}:{la:.7f}:{lo:.7f}"
                    if key in labeled_keys:
                        continue
                    chunk_r.append(ri); chunk_c.append(ci); chunk_la.append(la); chunk_lo.append(lo); chunk_p.append(p); chunk_nd.append(ndv)
                    if len(chunk_r) >= CHUNK:
                        part_idx = flush_chunk(part_idx)
                # flush remainder
                part_idx = flush_chunk(part_idx)

            # Merge parts for dists (ascending)
            if dists_parts:
                import heapq as _hq
                dp_final = os.path.join(dists_dir, base_core + '_dists.csv.gz')
                with _open_text_auto(dp_final, 'wt') as out:
                    w = csv.writer(out)
                    w.writerow(['tile','row','col','dist'])
                    readers = []
                    for pp in dists_parts:
                        f = _open_text_auto(pp, 'rt'); r = csv.reader(f); next(r, None)
                        readers.append((pp, f, r))
                    heap = []
                    for idx, (_pp, f, r) in enumerate(readers):
                        row = next(r, None)
                        if not row:
                            continue
                        try:
                            d = float(row[3]); rr = int(row[1]); cc = int(row[2]); ti = row[0]
                        except Exception:
                            continue
                        _hq.heappush(heap, (d, ti, rr, cc, idx, row))
                    while heap:
                        d, ti, rr, cc, idx, row = _hq.heappop(heap)
                        w.writerow([ti, rr, cc, d])
                        _pp, f, r = readers[idx]
                        row2 = next(r, None)
                        if row2:
                            try:
                                d2 = float(row2[3]); rr2 = int(row2[1]); cc2 = int(row2[2]); ti2 = row2[0]
                            except Exception:
                                row2 = None
                            if row2:
                                _hq.heappush(heap, (d2, ti2, rr2, cc2, idx, row2))
                    for _pp, f, _r in readers:
                        try: f.close()
                        except Exception: pass
                # cleanup part files
                for pp in dists_parts:
                    try: os.remove(pp)
                    except Exception: pass

            # Merge parts for positives (descending by prob)
            if pos_parts:
                import heapq as _hq
                pp_final = os.path.join(pos_dir, base_core + '_pos.csv.gz')
                with _open_text_auto(pp_final, 'wt') as out:
                    w = csv.writer(out)
                    w.writerow(['tile','row','col','lat','lon','prob','ndvi'])
                    readers = []
                    for pp in pos_parts:
                        f = _open_text_auto(pp, 'rt'); r = csv.reader(f); next(r, None)
                        readers.append((pp, f, r))
                    heap = []
                    for idx, (_pp, f, r) in enumerate(readers):
                        row = next(r, None)
                        if not row:
                            continue
                        try:
                            pr = float(row[5])
                        except Exception:
                            continue
                        _hq.heappush(heap, (-pr, idx, row))
                    while heap:
                        neg, idx, row = _hq.heappop(heap)
                        w.writerow(row)
                        _pp, f, r = readers[idx]
                        row2 = next(r, None)
                        if row2:
                            try:
                                pr2 = float(row2[5])
                            except Exception:
                                row2 = None
                            if row2:
                                _hq.heappush(heap, (-pr2, idx, row2))
                    for _pp, f, _r in readers:
                        try: f.close()
                        except Exception: pass
                for pp in pos_parts:
                    try: os.remove(pp)
                    except Exception: pass

            # Done with this tile
            return (tile_name, total_rows, pos_count)

        tile_files = _list_csvs(shards_dir)
        # Process tiles in a small thread pool; cKDTree handles parallel query via workers.
        res = []
        max_workers = int(getattr(cfg, 'REFRESH_TILE_THREADS', 2))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(per_tile, tp): tp for tp in tile_files}
            for fut in as_completed(futs):
                try:
                    res.append(fut.result())
                except Exception:
                    # If a tile fails, continue with others; it'll just have zero rows
                    pass
                # Update progress when any tile finishes
                try:
                    prog_metrics.update(task_metrics, advance=1)
                except Exception:
                    pass
    n_total = sum(n for _t,n,_p in res)
    pos_total = sum(p for _t,_n,p in res)
    if n_total <= 0:
        print("No unlabeled pixels found for global lists update.")
        return

    # 3) global rank for distances via k-way merge
    dist_files = _list_csvs(dists_dir, suffix='_dists.csv')
    # open rank writers per tile
    rank_writers = {}
    rank_files = {}
    def _tile_base_from_dist_path(path):
        name = os.path.basename(path)
        if name.endswith('.csv.gz'):
            name = name[:-7]
        elif name.endswith('.csv'):
            name = name[:-4]
        # strip trailing suffix "_dists"
        if name.endswith('_dists'):
            name = name[:-6]
        return name
    for dp in dist_files:
        tile_base = _tile_base_from_dist_path(dp)
        rp = os.path.join(ranks_dir, tile_base + '_rank.csv')
        fh = open(rp,'w',newline='')
        rank_files[tile_base] = fh
        w = csv.writer(fh)
        # Seed known aliases for this tile base
        for alias in {tile_base, tile_base + '.tif', os.path.splitext(tile_base)[0]}:
            rank_writers[alias] = w
        w.writerow(['row','col','rank'])
    # init heap from each file
    import heapq
    heap = []
    readers = []
    for dp in dist_files:
        f = _open_text_auto(dp, 'rt'); r = csv.reader(f); next(r, None)
        readers.append((dp,f,r))
        # Advance until a non-empty row is found
        while True:
            try:
                row = next(r)
            except StopIteration:
                row = None
            if not row:
                if row is None:
                    break
                else:
                    continue
            try:
                tile = row[0]; rr=int(row[1]); cc=int(row[2]); dist=float(row[3])
            except Exception:
                continue
            heapq.heappush(heap, (dist, tile, rr, cc, len(readers)-1))
            break
    i = 0
    denom = max(1, n_total-1)
    with _npb() as _prog:
        t_rank = _prog.add_task("Global distance rank", total=n_total)
        while heap:
            dist, tile, rr, cc, idx = heapq.heappop(heap)
            rank = i/denom
            base = os.path.splitext(tile)[0]
            writer = rank_writers.get(base) or rank_writers.get(tile) or rank_writers.get(base + '.tif')
            if writer is None:
                # Lazily create a rank writer for this tile base to avoid key errors
                rp = os.path.join(ranks_dir, base + '_rank.csv')
                fh = open(rp, 'w', newline='')
                w = csv.writer(fh)
                w.writerow(['row','col','rank'])
                rank_files[base] = fh
                # Register common aliases
                for alias in {base, base + '.tif', os.path.splitext(base)[0], tile}:
                    rank_writers[alias] = w
                writer = w
            writer.writerow([rr, cc, f"{rank:.6f}"])
            i += 1
            if i % 50000 == 0:
                _prog.update(t_rank, completed=i)
            # pull next from same reader
            dp, f, r = readers[idx]
            while True:
                try:
                    row = next(r)
                except StopIteration:
                    row = None
                if not row:
                    if row is None:
                        break
                    else:
                        continue
                try:
                    tile2 = row[0]; rr2=int(row[1]); cc2=int(row[2]); dist2=float(row[3])
                except Exception:
                    continue
                heapq.heappush(heap, (dist2, tile2, rr2, cc2, idx))
                break
        _prog.update(t_rank, completed=n_total)
    # close rank files
    for _, f, _ in readers:
        try: f.close()
        except Exception: pass
    for f in rank_files.values():
        try: f.close()
        except Exception: pass

    # 4) per-tile score and sort
    wu = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("uncertainty", 0.5)
    wr = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("representativeness", 0.3)
    wc = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("consistency", 0.2)

    def per_tile_score(tile_base):
        # rank map for this tile
        rp = os.path.join(ranks_dir, tile_base + '_rank.csv')
        # original shard (source of lat/lon/prob/ndvi)
        sp_csv = os.path.join(shards_dir, tile_base + '.csv')
        sp_gz  = sp_csv + '.gz'
        sp = sp_gz if os.path.exists(sp_gz) else sp_csv
        if not (os.path.exists(sp) and os.path.exists(rp)):
            return None
        # load ranks into dict
        ranks = {}
        with open(rp) as f:
            rd = csv.DictReader(f)
            for r in rd:
                ranks[(int(r['row']), int(r['col']))] = float(r['rank'])
        rows = []
        with _open_text_auto(sp, 'rt') as f:
            rd = csv.DictReader(f)
            for r in rd:
                try:
                    rr = int(r['row']); cc = int(r['col'])
                    la = float(r['lat']); lo = float(r['lon'])
                    pr = float(r['prob']); nd = float(r['ndvi'] or 0.0)
                except Exception:
                    continue
                R = ranks.get((rr, cc), 0.0)
                U = float(max(0.0, min(1.0, 1.0 - 2.0*abs(pr - 0.5))))
                C = 1.0 if abs(pr - cfg.MIN_AGRI_PROB) < cfg.UNCERTAINTY_BAND_DELTA else 0.0
                score = wu*U + wr*R + wc*C
                tile_name = tile_base + '.tif'
                rows.append((score, (tile_name, rr, cc, la, lo, pr, nd)))
        if not rows:
            return None
        rows.sort(key=lambda t: t[0], reverse=True)
        outp = os.path.join(scored_dir, tile_base + '_scored.csv.gz')
        with _open_text_auto(outp,'wt') as f:
            w = csv.writer(f)
            w.writerow(['tile','row','col','lat','lon','prob','ndvi','score'])
            for sc, tpl in rows:
                tile, rr, cc, la, lo, pr, nd = tpl
                w.writerow([tile, rr, cc, la, lo, pr, nd, f"{sc:.6f}"])
        return outp

    def _tile_base_from_shard_path(path):
        name = os.path.basename(path)
        if name.endswith('.csv.gz'):
            return name[:-7]
        if name.endswith('.csv'):
            return name[:-4]
        return os.path.splitext(name)[0]
    tile_bases = [_tile_base_from_shard_path(tp) for tp in tile_files]
    # progress for per-tile scoring
    _ctx_score = _npb()
    prog_score = _ctx_score.__enter__()
    task_score = prog_score.add_task("Per-tile scoring", total=len(tile_bases))
    def _per_tile_score_wrap(tb):
        res = per_tile_score(tb)
        prog_score.update(task_score, advance=1)
        return res
    # Parallelize scoring across tiles (I/O bound)
    scored_files = []
    max_workers = max(2, int(getattr(cfg, 'REFRESH_TILE_THREADS', 3)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_per_tile_score_wrap, tb) for tb in tile_bases]
        for fut in as_completed(futs):
            try:
                scored_files.append(fut.result())
            except Exception:
                pass
    try:
        _ctx_score.__exit__(None, None, None)
    except Exception:
        pass
    scored_files = [p for p in scored_files if p]

    # 5) k-way merge scored files by score desc to global highscore.csv
    def kmerge_desc(files, out_path, total=None):
        import heapq
        hs = []
        readers = []
        for fp in files:
            f = _open_text_auto(fp, 'rt'); r = csv.reader(f); next(r, None)
            readers.append((fp,f,r))
            # Advance until a non-empty row is found
            while True:
                try:
                    row = next(r)
                except StopIteration:
                    row = None
                if not row:
                    if row is None:
                        break
                    else:
                        continue
                try:
                    sc = float(row[7])
                except Exception:
                    continue
                heapq.heappush(hs, (-sc, row, len(readers)-1))
                break
        from progress_utils import new_progress as _npb2
        with open(out_path,'w',newline='') as out, _npb2() as _prog2:
            w = csv.writer(out); w.writerow(['tile','row','col','lat','lon','prob','ndvi','score'])
            t_hs = _prog2.add_task("Global highscore merge", total=total or 0)
            count = 0
            while hs:
                neg, row, idx = heapq.heappop(hs)
                w.writerow(row)
                count += 1
                if total:
                    if count % 50000 == 0:
                        _prog2.update(t_hs, completed=count)
                fp, f, r = readers[idx]
                # Pull next non-empty row
                while True:
                    try:
                        row2 = next(r)
                    except StopIteration:
                        row2 = None
                    if not row2:
                        if row2 is None:
                            break
                        else:
                            continue
                    try:
                        sc2 = float(row2[7])
                    except Exception:
                        continue
                    heapq.heappush(hs, (-sc2, row2, idx))
                    break
            if total:
                _prog2.update(t_hs, completed=total)
        for _, f, _ in readers:
            try: f.close()
            except Exception: pass

    hs_out = cfg.HIGHSCORE_FILE
    os.makedirs(os.path.dirname(hs_out), exist_ok=True)
    kmerge_desc(scored_files, hs_out, total=n_total)
    # global KML limited
    try:
        with open(hs_out) as f:
            rd = csv.DictReader(f)
            rows = list(rd)
        try:
            topk = int(getattr(cfg, 'HIGHSCORE_KML_TOP_PIXELS', 50000))
        except Exception:
            topk = 50000
        _write_ranked_pixel_kml(rows, cfg.HIGHSCORE_KML_GLOBAL, weight_key='score', top_k=topk)
    except Exception as e:
        print(f"Highscore KML failed: {e}")

    # 6) k-way merge positive shards by prob desc to global probableAgri.csv
    pos_files = _list_csvs(pos_dir, suffix='_pos.csv')
    def kmerge_pos(files, out_path, total=None):
        import heapq
        hs = []
        readers = []
        for fp in files:
            f = _open_text_auto(fp, 'rt'); r = csv.reader(f); next(r, None)
            readers.append((fp,f,r))
            # Advance until a non-empty row is found
            while True:
                try:
                    row = next(r)
                except StopIteration:
                    row = None
                if not row:
                    if row is None:
                        break
                    else:
                        continue
                try:
                    pr = float(row[5])
                except Exception:
                    continue
                heapq.heappush(hs, (-pr, row, len(readers)-1))
                break
        from progress_utils import new_progress as _npb3
        with open(out_path,'w',newline='') as out, _npb3() as _prog3:
            w = csv.writer(out); w.writerow(['tile','row','col','lat','lon','prob','ndvi'])
            t_pa = _prog3.add_task("Global probableAgri merge", total=total or 0)
            count = 0
            while hs:
                neg, row, idx = heapq.heappop(hs)
                w.writerow(row)
                count += 1
                if total and count % 50000 == 0:
                    _prog3.update(t_pa, completed=count)
                fp, f, r = readers[idx]
                # Pull next non-empty row
                while True:
                    try:
                        row2 = next(r)
                    except StopIteration:
                        row2 = None
                    if not row2:
                        if row2 is None:
                            break
                        else:
                            continue
                    try:
                        pr2 = float(row2[5])
                    except Exception:
                        continue
                    heapq.heappush(hs, (-pr2, row2, idx))
                    break
            if total:
                _prog3.update(t_pa, completed=total)
        for _, f, _ in readers:
            try: f.close()
            except Exception: pass

    pa_out = cfg.PROBABLE_AGRI_FILE
    kmerge_pos(pos_files, pa_out, total=pos_total)
    try:
        with open(pa_out) as f:
            rd = csv.DictReader(f)
            rows = list(rd)
        try:
            topk_pa = int(getattr(cfg, 'PROBABLE_AGRI_KML_TOP_PIXELS', 50000))
        except Exception:
            topk_pa = 50000
        _write_ranked_pixel_kml(rows, cfg.PROBABLE_AGRI_KML_GLOBAL, weight_key='prob', top_k=topk_pa)
    except Exception as e:
        print(f"ProbableAgri KML failed: {e}")

def candidate_selection_from_csv(pred_csv, round_dir, round_num, train_rows=None, X_train=None, y_train=None):
    """Load predictions from CSV and prompt the user to label candidates."""
    if not os.path.exists(pred_csv):
        print(f"Missing predictions CSV => {pred_csv}")
        return None
    # Stream to build a top-M uncertainty pool instead of loading everything
    print("Building uncertainty/negative pools for candidate selection...")
    import heapq
    pool_size = cfg.NUM_CANDIDATES_PER_ROUND * 10
    target_neg = int(cfg.NUM_CANDIDATES_PER_ROUND * max(0.0, float(getattr(cfg, 'CANDIDATE_NEGATIVE_QUOTA', 0))))
    pool_neg = max(target_neg * 10, target_neg) if target_neg > 0 else 0
    heap = []      # uncertainty pool: min-heap by negative margin (closer to 0.5)
    heap_neg = []  # negative-like pool: min-heap by prob (closer to MIN_AGRI_PROB from below)
    # dynamic negative-like prob range
    def _neg_prob_range():
        delta = float(getattr(cfg, 'NEG_LIKE_PROB_DELTA', 0.05))
        lo = max(0.0, cfg.MIN_AGRI_PROB - delta)
        hi = cfg.MIN_AGRI_PROB
        # also respect configured fallback range if provided and delta absent
        try:
            base = getattr(cfg, 'NEG_LIKE_PROB_RANGE', (lo, hi))
            # ensure the upper bound is MIN_AGRI_PROB and lower bound is not above it
            lo = min(lo, base[0]) if base else lo
            hi = cfg.MIN_AGRI_PROB
        except Exception:
            pass
        return lo, hi
    nlo, nhi = _neg_prob_range()
    # ndvi range (absolute or relative percentiles)
    ndvi_abs = getattr(cfg, 'NEG_LIKE_NDVI_RANGE', (None, None))
    ndvi_rel = bool(getattr(cfg, 'NEG_LIKE_NDVI_RELATIVE', False))
    ndvi_pr = getattr(cfg, 'NEG_LIKE_NDVI_PERC_RANGE', (0.6, 0.9))
    ndvi_vals_for_pr = []
    file_size = os.path.getsize(pred_csv) if os.path.exists(pred_csv) else 0
    from progress_utils import new_progress as _npb
    with open(pred_csv, "r", encoding='utf-8', errors='replace') as pf, _npb() as _prog:
        task = _prog.add_task("Scan predictions.csv", total=file_size or None)
        processed = 0
        header = pf.readline()
        processed += len(header.encode('utf-8','replace'))
        if file_size:
            _prog.update(task, completed=processed)
        for line in pf:
            processed += len(line.encode('utf-8','replace'))
            row = line.strip().split(',')
            if len(row) < 6:
                continue
            try:
                p = float(row[5])
            except Exception:
                p = 0.5
            # Parse common fields
            try:
                item = [row[0], int(row[1]), int(row[2]),
                        float(row[3]), float(row[4]), p,
                        float(row[6]) if len(row) > 6 and row[6] != '' else 0.0]
            except Exception:
                continue
            # Fill uncertainty pool (respecting CANDIDATE_PROB_LOWER)
            if p >= cfg.CANDIDATE_PROB_LOWER:
                key = -abs(p - 0.5)
                if len(heap) < pool_size:
                    heapq.heappush(heap, (key, item))
                else:
                    if key > heap[0][0]:
                        heapq.heapreplace(heap, (key, item))
            # Fill negative-like pool regardless of CANDIDATE_PROB_LOWER
            if pool_neg > 0:
                nd = item[6]
                if ndvi_rel:
                    ndvi_vals_for_pr.append(nd)
                is_neg_prob = (nlo <= p < nhi)
                is_neg_ndvi = True
                if isinstance(ndvi_abs, (list, tuple)) and ndvi_abs[0] is not None and ndvi_abs[1] is not None:
                    is_neg_ndvi = (ndvi_abs[0] <= nd <= ndvi_abs[1])
                if is_neg_prob and is_neg_ndvi:
                    keyn = p  # closer to threshold (higher p) preferred
                    if len(heap_neg) < pool_neg:
                        heapq.heappush(heap_neg, (keyn, item))
                    else:
                        if keyn > heap_neg[0][0]:
                            heapq.heapreplace(heap_neg, (keyn, item))
            if file_size and processed % (1024*1024) < 1000:
                _prog.update(task, completed=min(processed, file_size))
        if file_size:
            _prog.update(task, completed=file_size)
    # If NDVI relative mode, refine negative pool by NDVI percentiles
    if ndvi_rel and pool_neg > 0 and ndvi_vals_for_pr:
        import numpy as _np
        lo_p, hi_p = ndvi_pr
        lo_v = float(_np.quantile(_np.array(ndvi_vals_for_pr, dtype=float), lo_p))
        hi_v = float(_np.quantile(_np.array(ndvi_vals_for_pr, dtype=float), hi_p))
        heap_neg = [(k, it) for (k, it) in heap_neg if lo_v <= it[6] <= hi_v]
    preds_unc = [it for _k, it in sorted(heap, key=lambda t: t[0], reverse=True)]
    preds_neg = [it for _k, it in sorted(heap_neg, key=lambda t: t[0], reverse=True)] if pool_neg > 0 else []
    # Union (dedupe by tile,row,col)
    seen = set()
    preds_pool = []
    for it in preds_neg + preds_unc:
        key = (it[0], it[1], it[2])
        if key in seen:
            continue
        seen.add(key)
        preds_pool.append(it)
    return candidate_selection_from_predictions(preds_pool, round_dir, round_num, train_rows, X_train, y_train)


def candidate_selection_from_predictions(preds, round_dir, round_num, train_rows=None, X_train=None, y_train=None):
    """Prompt the user to label candidates from in-memory predictions."""
    from collections import defaultdict

    # derive dynamic negative-like ranges
    def _neg_prob_range():
        delta = float(getattr(cfg, 'NEG_LIKE_PROB_DELTA', 0.05))
        lo = max(0.0, cfg.MIN_AGRI_PROB - delta)
        hi = cfg.MIN_AGRI_PROB
        return (lo, hi)
    nlo, nhi = _neg_prob_range()
    ndvi_abs = getattr(cfg, 'NEG_LIKE_NDVI_RANGE', (None, None))
    def _is_negative_like(rec):
        p = rec[5]; nd = rec[6]
        if not (nlo <= p < nhi):
            return False
        if isinstance(ndvi_abs, (list, tuple)) and ndvi_abs[0] is not None and ndvi_abs[1] is not None:
            return (ndvi_abs[0] <= nd <= ndvi_abs[1])
        return True

    def select_candidates_entropy(predictions):
        import numpy as np
        from sklearn.cluster import DBSCAN
        # vectors
        print("Selecting candidates by entropy + spatial diversity...")
        probs = np.array([p[5] for p in predictions])
        lats = np.array([p[3] for p in predictions])
        lons = np.array([p[4] for p in predictions])
        entropy = -probs*np.log(probs + 1e-9) - (1 - probs)*np.log(1 - probs + 1e-9)
        # candidate band
        mask_band = probs >= cfg.CANDIDATE_PROB_LOWER
        idx_all = np.where(mask_band)[0]
        if idx_all.size == 0:
            return []
        pool_idx = idx_all[np.argsort(entropy[idx_all])[::-1][: cfg.NUM_CANDIDATES_PER_ROUND * 10]]
        # haversine DBSCAN
        print("Running DBSCAN clustering on pool...")
        earth_km = 6371.0088
        eps_rad = cfg.CANDIDATE_DBSCAN_EPS_KM / earth_km
        coords_rad = np.vstack([np.deg2rad(lats[pool_idx]), np.deg2rad(lons[pool_idx])]).T
        clustering = DBSCAN(eps=eps_rad, min_samples=1, metric='haversine').fit(coords_rad)
        labels = clustering.labels_ if hasattr(clustering, 'labels_') else np.zeros(coords_rad.shape[0], dtype=int)
        from collections import defaultdict
        # bucket by tile and cluster
        buckets = defaultdict(list)
        for ii, cid in zip(pool_idx, labels):
            buckets[(predictions[ii][0], cid)].append(ii)
        # sort within buckets by entropy desc
        for k in buckets:
            buckets[k].sort(key=lambda i: entropy[i], reverse=True)
        picks = []
        keys = list(buckets.keys())
        while len(picks) < cfg.NUM_CANDIDATES_PER_ROUND and any(len(v) > 0 for v in buckets.values()):
            for k in keys:
                if buckets[k]:
                    picks.append(buckets[k].pop(0))
                    if len(picks) >= cfg.NUM_CANDIDATES_PER_ROUND:
                        break
        print(f"Entropy/DBSCAN selection complete: {len(picks)} picks")
        return [predictions[i] for i in picks]

    # Negative-like quota picks (optional)
    target_neg = int(cfg.NUM_CANDIDATES_PER_ROUND * max(0.0, float(getattr(cfg, 'CANDIDATE_NEGATIVE_QUOTA', 0))))
    neg_picks = []
    if target_neg > 0:
        neg_pool = [p for p in preds if _is_negative_like(p)]
        # sort by p desc (closer to threshold), then tile-balanced round robin
        neg_pool.sort(key=lambda r: r[5], reverse=True)
        by_tile_neg = defaultdict(list)
        for e in neg_pool:
            by_tile_neg[e[0]].append(e)
        tiles = list(by_tile_neg.keys())
        # round-robin until target_neg or pool exhausted
        idx = 0
        while len(neg_picks) < target_neg and tiles:
            t = tiles[idx % len(tiles)]
            if by_tile_neg[t]:
                neg_picks.append(by_tile_neg[t].pop(0))
                idx += 1
            else:
                tiles.pop(idx % len(tiles) if tiles else 0)
    # Main uncertainty picks for the remaining slots
    main_target = cfg.NUM_CANDIDATES_PER_ROUND - len(neg_picks)
    cands_main = select_candidates_entropy(preds)
    # de-dup and limit
    sel = []
    seen = set((e[0], e[1], e[2]) for e in neg_picks)
    for e in cands_main:
        k = (e[0], e[1], e[2])
        if k in seen:
            continue
        sel.append(e)
        if len(sel) >= main_target:
            break
    cands = neg_picks + sel
    if len(cands) < cfg.NUM_CANDIDATES_PER_ROUND:
        unc = [p for p in preds if cfg.CANDIDATE_PROB_LOWER <= p[5] <= cfg.MIN_AGRI_PROB]
        if len(unc) < cfg.NUM_CANDIDATES_PER_ROUND:
            preds.sort(key=lambda r: abs(r[5] - 0.5))
            cands = preds[:cfg.NUM_CANDIDATES_PER_ROUND]
        else:
            by_tile = defaultdict(list)
            for entry in unc:
                by_tile[entry[0]].append(entry)
            tiles = list(by_tile.keys())
            random.shuffle(tiles)
            per_tile = cfg.NUM_CANDIDATES_PER_ROUND // len(tiles)
            remainder = cfg.NUM_CANDIDATES_PER_ROUND % len(tiles)
            cands = []
            leftovers = []
            for i, tile in enumerate(tiles):
                random.shuffle(by_tile[tile])
                target = per_tile + (1 if i < remainder else 0)
                selected = by_tile[tile][:target]
                cands.extend(selected)
                leftovers.extend(by_tile[tile][len(selected):])
            random.shuffle(leftovers)
            while len(cands) < cfg.NUM_CANDIDATES_PER_ROUND and leftovers:
                cands.append(leftovers.pop())

    print(f"{len(cands)} candidate patches selected")

    tmp = os.path.join(round_dir, "temporary", "temp_labels.csv")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "lat", "lon", "tile", "label", "notes",
            "method", "prob", "entropy", "margin", "cluster_id", "cluster_size",
            "cluster_rank", "tile_pick_order", "local_density_k", "dist_to_threshold",
            "ndvi", "nearest_label_dist", "nearest_label_class", "reason"
        ])

    # Use a single KML filename so that each candidate overwrites the previous
    kmlp = os.path.join(round_dir, "temp_candidate.kml")

    # Precompute feature stats from training data if provided
    feat_means = feat_std = None
    Xs = None
    if X_train is not None and y_train is not None and len(X_train):
        feat_means = np.nanmean(X_train, axis=0).astype(np.float32)
        feat_std = np.nanstd(X_train, axis=0).astype(np.float32)
        feat_std[feat_std == 0] = 1.0
        Xs = (X_train - feat_means) / (feat_std + 1e-6)

    # Compute cluster info among selected candidates
    try:
        import numpy as _np
        from sklearn.cluster import DBSCAN as _DB
        coords = _np.array([[x[3], x[4]] for x in cands], dtype=float)
        # haversine clustering on selected cands
        earth_km = 6371.0088
        eps_rad = cfg.CANDIDATE_DBSCAN_EPS_KM / earth_km
        coords_rad = _np.deg2rad(coords)
        cl = _DB(eps=eps_rad, min_samples=1, metric='haversine').fit(coords_rad)
        cids = cl.labels_.tolist() if hasattr(cl, 'labels_') else [0] * len(coords)
        # cluster sizes
        from collections import Counter
        csize = Counter(cids)
        # cluster rank by entropy desc
        ent_all = [-x[5]*_np.log(x[5]+1e-9) - (1-x[5])*_np.log(1-x[5]+1e-9) for x in cands]
        crank = {}
        for cid in set(cids):
            idxs = [i for i, cc in enumerate(cids) if cc == cid]
            idxs.sort(key=lambda i: ent_all[i], reverse=True)
            for rank, i in enumerate(idxs, 1):
                crank[i] = rank
    except Exception:
        cids = [""] * len(cands)
        csize = {}
        crank = {}

    tile_pick_counter = {}
    for idx, (t, r, c, la, lo, p, ndvi) in enumerate(cands):
        generate_candidate_kml(t, r, c, kmlp)
        ent = -p*np.log(p + 1e-9) - (1-p)*np.log(1-p + 1e-9)
        dist_th = abs(p - cfg.MIN_AGRI_PROB)
        # nearest labeled feature-space distance
        nld, nlc = "", ""
        if Xs is not None:
            tf = get_tile_features(t)
            if tf is not None:
                arr, _, _ = tf
                if 0 <= r < arr.shape[1] and 0 <= c < arr.shape[2]:
                    f = arr[:, r, c].astype(np.float32)
                    if not np.isnan(f).any():
                        fs = (f - feat_means) / (feat_std + 1e-6)
                        d = np.sqrt(((Xs - fs[None, :]) ** 2).sum(axis=1))
                        j = int(np.argmin(d))
                        nld = float(d[j])
                        nlc = int(y_train[j])
        # local density within eps on same tile
        kcount = 0
        for (t2, r2, c2, la2, lo2, p2, _) in cands:
            if t2 != t:
                continue
            dlat = np.deg2rad(la2 - la); dlon = np.deg2rad(lo2 - lo)
            a = np.sin(dlat/2)**2 + np.cos(np.deg2rad(la))*np.cos(np.deg2rad(la2))*np.sin(dlon/2)**2
            if 2*6371.0088*np.arcsin(np.sqrt(a)) <= cfg.CANDIDATE_DBSCAN_EPS_KM:
                kcount += 1
        tile_pick_counter[t] = tile_pick_counter.get(t, 0) + 1
        reason = f"uncertain H={ent:.2f}, dthr={dist_th:.2f}, ndvi={ndvi:.2f}, dens_k={kcount}, nld={nld}"
        print(f"Candidate: {t} r={r},c={c}, p={p:.3f}, ndvi={ndvi:.3f} :: {reason}")
        ui = input("Label? (1=Agri,2=NonAgri,3=Skip): ").strip()
        if ui == "3":
            print("Skipped.")
            continue
        lab = "Agricultural" if ui == "1" else "Non-Agricultural" if ui == "2" else None
        if lab:
            note = prompt_note()
            eid = f"AL_{round_num}_{int(random.random()*1e6)}"
            with open(tmp, "a", newline="") as f2:
                csv.writer(f2).writerow([
                    eid, la, lo, t, lab, note,
                    "entropy_dbscan_tile_balanced",
                    f"{p:.6f}", f"{ent:.6f}", f"{abs(p-0.5):.6f}", cids[idx], csize.get(cids[idx], ""),
                    crank.get(idx, ""), tile_pick_counter[t], kcount, f"{dist_th:.6f}", f"{ndvi:.6f}", nld if nld != "" else "", nlc if nlc != "" else "", reason
                ])
            print("Label saved.")
    return tmp


if __name__ == "__main__":
    active_learning_round(1, TEMP_LABELS_FILE, "SVM")
def _open_text_auto(path, mode='rt'):
    """Open plain or gzip text files with correct newline handling.

    - Uses newline='' for text write/append to avoid blank rows on Windows.
    - For gzip writes, uses cfg.GZIP_COMPRESSLEVEL (lower = faster) to speed IO.
    """
    import gzip
    text_mode = 't' in mode
    write_mode = ('w' in mode) or ('a' in mode) or ('x' in mode)
    newline = '' if (text_mode and write_mode) else None
    if path.endswith('.gz'):
        if write_mode:
            level = int(getattr(cfg, 'GZIP_COMPRESSLEVEL', 3))
            return gzip.open(path, mode, newline=newline, compresslevel=level)
        return gzip.open(path, mode, newline=newline)
    return open(path, mode, newline=newline)

def _list_csvs(folder, suffix=None):
    files = []
    for name in os.listdir(folder):
        if name.endswith('.csv') or name.endswith('.csv.gz'):
            if suffix is None or suffix in name:
                files.append(os.path.join(folder, name))
    return sorted(files)
