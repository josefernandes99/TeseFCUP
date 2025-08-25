#!/usr/bin/env python3
# scripts/a3_phase1_active_learning_round.py

import os
from multiprocessing import cpu_count

# -----------------------------------------------------------------------------
# 1) Speed‐ups: thread‐tune BLAS/OpenMP to use all CPU cores
# -----------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = str(cpu_count())
os.environ["MKL_NUM_THREADS"] = str(cpu_count())

import csv
import glob
import random
from collections import defaultdict
import time
import datetime
import json
from operator import itemgetter
from pyproj import Transformer

import numpy as np
import rasterio
from rasterio.features import shapes, sieve
from shapely.geometry import shape, Polygon, MultiPolygon
from shapely.ops import unary_union, transform as shp_transform
import torch
import torch.nn as nn
from joblib import dump, Parallel, delayed
from memory_watcher import free_unused_memory
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
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
from evaluation import evaluate_model_repeated as evaluate_model
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer
from splits import stratified_train_val_test_indices
from features import current_feature_names, get_reporting_ndvi_index

from a2_phase1_initial_labeling import generate_grids_for_all_tiles
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

class CombinedModelWithPatch:
    """Wrap a base probabilistic model and an optional tiny patch CNN refiner.

    If a patch net is present and PATCH_CNN_ENABLED is True, refine the most
    uncertain pixels in a tile by blending base prob with CNN prob.
    """
    def __init__(self, base_wrapper, patch_net_scripted=None, patch_channels=None, patch_win=9, blend_alpha=0.5):
        self.base = base_wrapper
        self.patch_net = patch_net_scripted  # torch.jit.ScriptModule or None
        self.patch_channels = patch_channels or []
        self.patch_win = int(patch_win)
        self.blend_alpha = float(blend_alpha)

    def predict_proba(self, X):
        return self.base.predict_proba(X)

    def predict(self, X):
        return self.base.predict(X)

    def fit(self, X, y):
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
        base_wrapper = SklearnWrapper(clf, feat_means, feat_std)
        # Optional: tiny patch CNN refiner
        if getattr(cfg, 'PATCH_CNN_ENABLED', False):
            patch = _train_patch_cnn()
            return CombinedModelWithPatch(
                base_wrapper,
                patch_net_scripted=patch.get('scripted'),
                patch_channels=patch.get('channels', []),
                patch_win=int(patch.get('win', 9)),
                blend_alpha=float(getattr(cfg, 'PATCH_CNN_BLEND_ALPHA', 0.5)),
            )
        return base_wrapper

    elif c == "randomforest":
        rf = RandomForestClassifier(n_jobs=-1, **cfg.RF_PARAMS)
        rf.fit(Xs, y)
        return SklearnWrapper(rf, feat_means, feat_std)

    elif c == "resnet":
        net = TabularResNet(input_dim=Xs.shape[1])
        train_resnet(net, torch.from_numpy(Xs.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
        scripted = torch.jit.script(net)
        return PytorchResNetWrapper(scripted, feat_means, feat_std)

    elif c == "xgboost":
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("XGBoost model requested but xgboost package is not installed.")
        params = cfg.XGB_PARAMS.copy()
        # Override tree_method based on GPU flag
        params['tree_method'] = 'gpu_hist' if getattr(cfg, 'XGB_USE_GPU', False) else 'hist'
        # Scale pos weight auto if requested
        if getattr(cfg, 'XGB_USE_AUTO_SPW', True):
            pos = max(int((y == 1).sum()), 1)
            neg = max(int((y == 0).sum()), 1)
            params['scale_pos_weight'] = float(neg) / float(pos)
        # Fit on standardized features (keeps wrapper interface consistent)
        clf = xgb.XGBClassifier(**params)
        clf.fit(Xs, y, verbose=False)
        return SklearnWrapper(clf, feat_means, feat_std)

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

def predict_entire_tile(tile_path, model, progress=None, task_id=None):
    """Run inference on a tile and optionally update a progress bar."""
    tile_name = os.path.basename(tile_path)
    with rasterio.open(tile_path) as src:
        raw = src.read().astype(np.float32)
        arr, names = add_derived_features(raw)
        b, H, W = arr.shape
        Xflat  = arr.reshape(b, -1).T                # (H*W, bands)
        rows = np.repeat(np.arange(H, dtype=np.int32), W)
        cols = np.tile(np.arange(W, dtype=np.int32), H)
        # Chunked inference to limit memory
        if cfg.INFER_CHUNKING_ENABLED:
            probs = np.empty((Xflat.shape[0],), dtype=np.float32)
            bs = int(cfg.INFER_MAX_PIXELS_PER_BATCH)
            for i in range(0, Xflat.shape[0], bs):
                probs[i:i+bs] = model.predict_proba(Xflat[i:i+bs])[:, 1].astype(np.float32)
        else:
            probs = model.predict_proba(Xflat)[:, 1].astype(np.float32)

        # Optional patch-CNN refine on most-uncertain pixels (blend)
        try:
            if hasattr(model, 'patch_net') and model.patch_net is not None and getattr(cfg, 'PATCH_CNN_ENABLED', False):
                th = float(getattr(cfg, 'MIN_AGRI_PROB', 0.5))
                frac = float(getattr(cfg, 'PATCH_CNN_TOP_UNCERTAIN_FRAC', 0.1))
                k = int(max(1, frac * probs.size))
                # pick closest to threshold
                idx = np.argsort(np.abs(probs - th))[:k]
                # build patches for selected indices
                win = int(getattr(cfg, 'PATCH_CNN_WINDOW', getattr(model, 'patch_win', 9)))
                rad = win // 2
                chans = model.patch_channels if getattr(model, 'patch_channels', None) else _select_patch_channels(names)
                # ensure channels are valid
                chans = [ci for ci in chans if 0 <= ci < b]
                if chans and win > 0:
                    # prepare patches tensor [k, C, H, W]
                    ph = np.zeros((k, len(chans), win, win), dtype=np.float32)
                    rr = rows[idx]; cc = cols[idx]
                    # pad array to simplify edge handling
                    pad = ((0,0),(rad,rad),(rad,rad))
                    arrp = np.pad(arr, pad_width=pad, mode='edge')
                    for j,(r0,c0) in enumerate(zip(rr,cc)):
                        rs = r0; cs = c0
                        ph[j] = arrp[chans, rs:rs+win, cs:cs+win]
                    # run CNN
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    xt = torch.from_numpy(ph).to(device)
                    with torch.no_grad():
                        logits = model.patch_net(xt)
                        p2 = torch.softmax(logits, dim=1)[:,1].cpu().numpy().astype(np.float32)
                    alpha = float(getattr(cfg, 'PATCH_CNN_BLEND_ALPHA', 0.5))
                    probs[idx] = alpha * p2 + (1.0 - alpha) * probs[idx]
        except Exception:
            # Refinement is optional; ignore any errors to keep pipeline robust
            pass

        # vectorized center coordinate computation
        xs, ys = rasterio.transform.xy(src.transform, rows.tolist(), cols.tolist(), offset="center")
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if src.crs and not src.crs.is_geographic:
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            xs, ys = transformer.transform(xs, ys)

        ndvi_idx = get_reporting_ndvi_index(names)
        if ndvi_idx is None:
            raise RuntimeError("NDVI feature missing (no NDVI_s* or NDVI); aborting as required.")
        ndvi_vals = arr[ndvi_idx].reshape(-1).astype(np.float32)
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
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
        w.writerows(preds)
    print(f"Predictions written to {path}")


# -------------------------
# Patch CNN helpers
# -------------------------
def _select_patch_channels(names):
    """Choose a small set of channels for patch CNN (robust defaults).

    Preference: NDVI_s2, B8_s2, B4_s2, and optionally NDMI_s2 if present.
    Returns list of channel indices.
    """
    prefs = ["NDVI_s2", "B8_s2", "B4_s2", "NDMI_s2", "NDVI", "B8", "B4", "NDMI"]
    idxs = []
    for p in prefs:
        try:
            idxs.append(names.index(p))
        except ValueError:
            # try seasonal fallback
            pref = p.split("_s")[0]
            cand = [i for i, n in enumerate(names) if n.startswith(pref + "_s")]
            if cand:
                idxs.append(cand[len(cand)//2])
    # unique preserve order
    seen = set(); out = []
    for i in idxs:
        if i not in seen:
            seen.add(i); out.append(i)
    return out[:4]


def _train_patch_cnn():
    """Train a tiny CNN on labeled patches if enabled; return dict with scripted net.

    Uses labels.csv + temp_labels.csv for supervision. Keeps compute limited by
    sampling up to PATCH_CNN_MAX_PATCHES per class.
    """
    if not getattr(cfg, 'PATCH_CNN_ENABLED', False):
        return {"scripted": None, "channels": [], "win": int(getattr(cfg, 'PATCH_CNN_WINDOW', 9))}
    try:
        import csv as _csv
        from splits import load_labels as _load
        from al_shared import get_tile_features, snap_to_pixel_center
    except Exception:
        return {"scripted": None, "channels": [], "win": int(getattr(cfg, 'PATCH_CNN_WINDOW', 9))}

    rows = []
    if os.path.exists(cfg.LABELS_FILE):
        rows.extend(_load(cfg.LABELS_FILE))
    if os.path.exists(cfg.TEMP_LABELS_FILE):
        rows.extend(_load(cfg.TEMP_LABELS_FILE))
    if not rows:
        return {"scripted": None, "channels": [], "win": int(getattr(cfg, 'PATCH_CNN_WINDOW', 9))}

    # Collect patches
    win = int(getattr(cfg, 'PATCH_CNN_WINDOW', 9)); rad = win // 2
    Xp, Yp = [], []
    # Determine channels once from the first available tile
    chans = None
    for r in rows:
        tile = r.get('tile'); lab = r.get('label', '').lower()
        if tile is None or lab == '':
            continue
        snap = snap_to_pixel_center(tile, float(r.get('lat', 'nan')), float(r.get('lon', 'nan')))
        if not snap:
            continue
        _, _, ri, ci = snap
        tf = get_tile_features(tile)
        if tf is None:
            continue
        arr, _, _ = tf
        if chans is None:
            # derive names by reading any tile via features.current_feature_names
            from features import current_feature_names
            names = current_feature_names()
            chans = _select_patch_channels(names)
        # bounds with padding via np.pad for simplicity
        pad = ((0,0),(rad,rad),(rad,rad))
        arrp = np.pad(arr, pad_width=pad, mode='edge')
        rs, cs = ri, ci
        if rs < 0 or cs < 0 or rs >= arr.shape[1] or cs >= arr.shape[2]:
            continue
        patch = arrp[chans, rs:rs+win, cs:cs+win].astype(np.float32)
        Xp.append(patch)
        Yp.append(1 if lab.startswith('agri') else 0)
    if not Xp:
        return {"scripted": None, "channels": [], "win": win}

    # Downsample to cap patches per class
    max_per = int(getattr(cfg, 'PATCH_CNN_MAX_PATCHES_PER_CLASS', 2000))
    import numpy as _np
    Xp = _np.stack(Xp, axis=0)
    Yp = _np.array(Yp, dtype=_np.int64)
    pos_idx = _np.where(Yp == 1)[0]; neg_idx = _np.where(Yp == 0)[0]
    def _sample(idx):
        if idx.size <= max_per: return idx
        _np.random.seed(0); _np.random.shuffle(idx); return idx[:max_per]
    keep = _np.concatenate([_sample(pos_idx), _sample(neg_idx)])
    Xp = Xp[keep]; Yp = Yp[keep]

    # Define tiny CNN
    class TinyPatchCNN(nn.Module):
        def __init__(self, in_ch, win):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(32)
            self.head = nn.Linear(32 * win * win, 2)
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = x.reshape(x.size(0), -1)
            return self.head(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = TinyPatchCNN(in_ch=Xp.shape[1], win=win).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=float(getattr(cfg, 'PATCH_CNN_LR', 1e-3)))
    crit = nn.CrossEntropyLoss()
    bs = int(getattr(cfg, 'PATCH_CNN_BATCH', 64))
    epochs = int(getattr(cfg, 'PATCH_CNN_EPOCHS', 5))
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xp), torch.from_numpy(Yp))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    net.train()
    for ep in range(epochs):
        tot = 0.0
        for bx, by in loader:
            bx = bx.to(device); by = by.to(device)
            opt.zero_grad(); logits = net(bx); loss = crit(logits, by); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % 2 == 0:
            print(f"PatchCNN epoch {ep+1}/{epochs}, loss={tot/len(loader):.4f}")
    net.eval()
    scripted = torch.jit.script(net)
    return {"scripted": scripted, "channels": chans, "win": win}


def _load_predictions_by_tile(csv_path):
    """Return mapping: tile -> (row_indices, col_indices, probs)."""
    pred_map = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile = row["tile"]
            ri = int(row["row_idx"])
            ci = int(row["col_idx"])
            prob = float(row["predicted_prob"])
            if tile not in pred_map:
                pred_map[tile] = ([], [], [])
            pred_map[tile][0].append(ri)
            pred_map[tile][1].append(ci)
            pred_map[tile][2].append(prob)
    for t, (rs, cs, ps) in pred_map.items():
        pred_map[t] = (
            np.array(rs, dtype=np.int32),
            np.array(cs, dtype=np.int32),
            np.array(ps, dtype=np.float32),
        )
    return pred_map


def save_agricultural_polygons_kml(round_folder, round_num, pred_csv=None):
    """Polygonize cached predictions without re-running model inference."""
    pred_csv = pred_csv or os.path.join(round_folder, "predictions.csv")
    if not os.path.exists(pred_csv):
        print(f"Missing predictions file => {pred_csv}")
        return

    pred_map = _load_predictions_by_tile(pred_csv)
    kml_path = os.path.join(round_folder, f"agricultural_patches_round_{round_num}.kml")
    red_polys = []
    orange_polys = []

    # Helper to polygonize a single tile
    def _poly_from_tile(tile, rows, cols, probs):
        tif = os.path.join(RAW_DATA_DIR, tile)
        if not os.path.exists(tif):
            return ([], [])
        reds, oranges = [], []
        with rasterio.open(tif) as src:
            H, W = src.height, src.width
            arr_probs = np.zeros((H, W), dtype=np.float32)
            arr_probs[rows, cols] = probs
            red_mask = arr_probs >= cfg.SIEVE_KEEP_PROB
            orange_mask = (arr_probs >= cfg.MIN_AGRI_PROB) & (arr_probs < cfg.SIEVE_KEEP_PROB)
            from scipy.ndimage import binary_closing, binary_fill_holes
            red_mask = binary_fill_holes(binary_closing(red_mask))
            orange_mask = binary_fill_holes(binary_closing(orange_mask))
            if cfg.SIEVE_MIN_SIZE > 0:
                red_mask = sieve(red_mask.astype("uint8"), size=cfg.SIEVE_MIN_SIZE, connectivity=8).astype(bool)
                orange_mask = sieve(orange_mask.astype("uint8"), size=cfg.SIEVE_MIN_SIZE, connectivity=8).astype(bool)
            orange_mask &= ~red_mask
            transformer = None
            if src.crs and not src.crs.is_geographic:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            for geom, val in shapes(red_mask.astype("uint8"), mask=red_mask, transform=src.transform):
                if val != 1:
                    continue
                poly = shape(geom)
                if transformer:
                    poly = shp_transform(transformer.transform, poly)
                reds.append(poly)
            for geom, val in shapes(orange_mask.astype("uint8"), mask=orange_mask, transform=src.transform):
                if val != 1:
                    continue
                poly = shape(geom)
                if transformer:
                    poly = shp_transform(transformer.transform, poly)
                oranges.append(poly)
        return (reds, oranges)

    # Parallelize per-tile polygonization
    from joblib import Parallel, delayed
    items = list(pred_map.items())
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_poly_from_tile)(tile, rc[0], rc[1], rc[2]) for tile, rc in items
    )
    for reds, oranges in results:
        if reds:
            red_polys.extend(reds)
        if oranges:
            orange_polys.extend(oranges)

    if not red_polys and not orange_polys:
        print(f"WARNING: No polygons (all probs < {cfg.MIN_AGRI_PROB})")
        return
    # Merge within each class for compactness
    def ensure_list(merged):
        if isinstance(merged, Polygon):
            return [merged]
        elif isinstance(merged, MultiPolygon):
            return list(merged.geoms)
        else:
            return []
    red_list = ensure_list(unary_union(red_polys)) if red_polys else []
    orange_list = ensure_list(unary_union(orange_polys)) if orange_polys else []

    # Build KML with two styles: borderline (orange) and confident (red)
    doc = Element('Document')
    style_or = SubElement(doc, 'Style', id='borderline')
    # Requested cyan-like color #55ffff: aabbggrr => alpha ff, bb ff, gg ff, rr 55 => ffffff55
    ln1 = SubElement(style_or, 'LineStyle'); SubElement(ln1, 'color').text = 'ffffff55'; SubElement(ln1, 'width').text = '1'
    # 25% opacity fill
    ps1 = SubElement(style_or, 'PolyStyle'); SubElement(ps1, 'color').text = '40ffff55'; SubElement(ps1, 'outline').text = '1'
    style_rd = SubElement(doc, 'Style', id='confident')
    ln2 = SubElement(style_rd, 'LineStyle'); SubElement(ln2, 'color').text = 'ff0000ff'; SubElement(ln2, 'width').text = '1'
    ps2 = SubElement(style_rd, 'PolyStyle'); SubElement(ps2, 'color').text = '400000ff'; SubElement(ps2, 'outline').text = '1'

    total_polys = 0
    # orange polygons
    for p in orange_list:
        coords = list(p.exterior.coords)
        coord_str = " ".join(f"{lon},{lat},0" for lon, lat in coords)
        pm = SubElement(doc, 'Placemark')
        SubElement(pm, 'styleUrl').text = '#borderline'
        poly_el = SubElement(pm, 'Polygon')
        ob = SubElement(poly_el, 'outerBoundaryIs')
        ring = SubElement(ob, 'LinearRing')
        SubElement(ring, 'coordinates').text = coord_str
        total_polys += 1
    # red polygons
    for p in red_list:
        coords = list(p.exterior.coords)
        coord_str = " ".join(f"{lon},{lat},0" for lon, lat in coords)
        pm = SubElement(doc, 'Placemark')
        SubElement(pm, 'styleUrl').text = '#confident'
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
            look_range = max(10.0, diag_m / 4.0)  # 2x zoom = halve the range twice relative to diagonal
        except Exception:
            look_range = 100.0

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
    progress_enabled=True,
    parallel_tiles=True,
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
    for r in rows:
        feats = extract_features_from_label(r)
        if feats is not None:
            X.append(feats)
            y.append(1 if r["label"].lower()=="agricultural" else 0)
    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

    # train & save
    print(f"Training data shape: {X.shape}, model: {model_choice}")
    model = train_model(model_choice, X, y)
    mp = os.path.join(rnd_dir, f"model_round_{round_num}.pkl")
    dump(model, mp)
    print(f"Model saved to {mp}")
    # keep X,y for representativeness computations
    free_unused_memory()

    # Decide whether we need per-tile inference (skip for metrics-only grid-search runs)
    do_infer_tiles = save_preds or request_labels
    total_pixels = 0
    start = time.time()
    if do_infer_tiles:
        # inference + timing (ignore overlay/final-sweep artifacts in RAW_DATA_DIR)
        all_tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
        tifs = [tp for tp in all_tifs if ("_overlay" not in os.path.basename(tp) and "_th" not in os.path.basename(tp))]
        pred_path = os.path.join(rnd_dir, "predictions.csv")
        if save_preds and top_n_predictions is None and parallel_tiles:
            # Parallel inference: each tile writes to its own chunk file; then concatenate
            chunk_dir = os.path.join(rnd_dir, "temporary", "pred_chunks")
            os.makedirs(chunk_dir, exist_ok=True)
            from joblib import Parallel, delayed
            def _infer_and_write(tp, _round=round_num, _model=model_choice):
                rows = predict_entire_tile(tp, model)
                base = os.path.basename(tp)
                out_csv = os.path.join(chunk_dir, f"{base}.csv")
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
                    w.writerows(rows)
                # Optional: write a parquet chunk for global predictions
                if getattr(cfg, 'PERSISTENT_PARQUET_ENABLED', False):
                    try:
                        import pyarrow as pa, pyarrow.parquet as pq
                        cols = list(zip(*rows)) if rows else []
                        if rows and len(cols) == 7:
                            probs = [r[5] for r in rows]
                            ent = [float(-(p*np.log(p + 1e-9) + (1-p)*np.log(1-p + 1e-9))) for p in probs]
                            tbl = pa.table({
                                'tile': [r[0] for r in rows],
                                'row_idx': [r[1] for r in rows],
                                'col_idx': [r[2] for r in rows],
                                'center_lat': [r[3] for r in rows],
                                'center_lon': [r[4] for r in rows],
                                'predicted_prob': [r[5] for r in rows],
                                'ndvi': [r[6] for r in rows],
                                'entropy': ent,
                                'round': [_round] * len(rows),
                                'model': [_model] * len(rows),
                            })
                            out_parq = os.path.join(chunk_dir, f"{base}.parquet")
                            pq.write_table(tbl, out_parq)
                    except Exception:
                        pass
                return len(rows)
            counts = Parallel(n_jobs=-1, prefer="threads")(delayed(_infer_and_write)(tp) for tp in tifs)
            total_pixels = int(sum(counts))
            # Concatenate into the final predictions.csv
            blk = int(getattr(cfg, 'CONCAT_PROGRESS_BLOCK_ROWS', 200_000))
            from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
            print(f"Concatenating {len(tifs)} chunks (~{total_pixels} rows) ...")
            with open(pred_path, "w", newline="") as outf:
                wout = csv.writer(outf)
                wout.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
                with Progress("[bold cyan]Merging predictions", BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as p:
                    task = p.add_task("merge", total=total_pixels)
                    written = 0
                    for tp in tifs:
                        part = os.path.join(chunk_dir, f"{os.path.basename(tp)}.csv")
                        try:
                            with open(part, newline="") as inf:
                                rd = csv.reader(inf)
                                next(rd, None)
                                buf = []
                                for row in rd:
                                    buf.append(row)
                                    if len(buf) >= blk:
                                        wout.writerows(buf); written += len(buf); buf.clear(); p.update(task, advance=blk)
                                if buf:
                                    wout.writerows(buf); written += len(buf); p.update(task, advance=len(buf))
                        except Exception:
                            continue
            # Append parquet chunks to global store
            if getattr(cfg, 'PERSISTENT_PARQUET_ENABLED', False):
                try:
                    import pyarrow as pa, pyarrow.parquet as pq
                    gp_path = getattr(cfg, 'GLOBAL_PREDICTIONS_PARQUET', None)
                    if gp_path:
                        # Create/append dataset by combining all tile parquets
                        schema = None
                        writer = None
                        os.makedirs(os.path.dirname(gp_path), exist_ok=True)
                        with Progress("[bold cyan]Updating global predictions", BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn()) as p2:
                            task2 = p2.add_task("append", total=len(tifs))
                            for tp in tifs:
                                pfile = os.path.join(chunk_dir, f"{os.path.basename(tp)}.parquet")
                                if not os.path.exists(pfile):
                                    p2.update(task2, advance=1); continue
                                tbl = pq.read_table(pfile)
                                if writer is None:
                                    writer = pq.ParquetWriter(gp_path, tbl.schema)
                                writer.write_table(tbl)
                                p2.update(task2, advance=1)
                            if writer is not None:
                                writer.close()
                        # Mirror into highscore/probableAgri stores (same rows, different consumers)
                        try:
                            hs_path = getattr(cfg, 'HIGHSCORE_PARQUET', None)
                            pa_path = getattr(cfg, 'PROBABLE_AGRI_PARQUET', None)
                            if hs_path:
                                # Append same table; downstream scoring reads entropy/prob fields
                                writer = None
                                for tp in tifs:
                                    pfile = os.path.join(chunk_dir, f"{os.path.basename(tp)}.parquet")
                                    if not os.path.exists(pfile):
                                        continue
                                    tbl = pq.read_table(pfile)
                                    if writer is None:
                                        writer = pq.ParquetWriter(hs_path, tbl.schema)
                                    writer.write_table(tbl)
                                if writer is not None:
                                    writer.close()
                            if pa_path:
                                writer = None
                                for tp in tifs:
                                    pfile = os.path.join(chunk_dir, f"{os.path.basename(tp)}.parquet")
                                    if not os.path.exists(pfile):
                                        continue
                                    tbl = pq.read_table(pfile)
                                    if writer is None:
                                        writer = pq.ParquetWriter(pa_path, tbl.schema)
                                    writer.write_table(tbl)
                                if writer is not None:
                                    writer.close()
                        except Exception:
                            pass
                except Exception:
                    pass
            # Run global aggregation from parquet (committee over history + KMLs)
            if getattr(cfg, 'PERSISTENT_PARQUET_ENABLED', False):
                try:
                    from persistent_aggregator import rebuild_global_rankings, build_global_prediction_kmls
                    rebuild_global_rankings(); build_global_prediction_kmls()
                except Exception as e:
                    print(f"Global aggregation failed: {e}")
            # Cleanup chunk files
            try:
                for tp in tifs:
                    p1 = os.path.join(chunk_dir, f"{os.path.basename(tp)}.csv")
                    if os.path.exists(p1):
                        os.remove(p1)
                    p2 = os.path.join(chunk_dir, f"{os.path.basename(tp)}.parquet")
                    if os.path.exists(p2):
                        os.remove(p2)
            except Exception:
                pass
        else:
            # Sequential streaming with optional top-N heap
            writer = None
            if save_preds and top_n_predictions is None:
                wf = open(pred_path, "w", newline="")
                writer = csv.writer(wf)
                writer.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
            import heapq as _heapq
            top_heap = []
            heap_cap = int(top_n_predictions) if (top_n_predictions is not None and top_n_predictions > 0) else 0
            idx_counter = 0
            _use_prog = bool(progress_enabled)
            if _use_prog:
                prog_ctx = Progress(
                    "[bold cyan]{task.description}",
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                )
                prog_ctx.__enter__()
                task = prog_ctx.add_task("Running inference", total=len(tifs))
            else:
                prog_ctx = None
                task = None
            try:
                # Prepare parquet chunk dir for optional persistent store
                chunk_dir = os.path.join(rnd_dir, "temporary", "pred_chunks")
                if getattr(cfg, 'PERSISTENT_PARQUET_ENABLED', False):
                    os.makedirs(chunk_dir, exist_ok=True)
                for tp in tifs:
                    tile_preds = predict_entire_tile(tp, model)
                    total_pixels += len(tile_preds)
                    if save_preds:
                        if top_n_predictions is None:
                            writer.writerows(tile_preds)  # type: ignore[union-attr]
                        else:
                            for row in tile_preds:
                                p = float(row[5])
                                unc = 0.5 - abs(p - 0.5)
                                if len(top_heap) < heap_cap:
                                    _heapq.heappush(top_heap, (unc, idx_counter, row))
                                else:
                                    if heap_cap > 0 and unc > top_heap[0][0]:
                                        _heapq.heapreplace(top_heap, (unc, idx_counter, row))
                                idx_counter += 1
                        # Optional parquet per-tile
                        if getattr(cfg, 'PERSISTENT_PARQUET_ENABLED', False):
                            try:
                                import pyarrow as pa, pyarrow.parquet as pq
                                base = os.path.basename(tp)
                                probs = [r[5] for r in tile_preds]
                                ent = [float(-(p*np.log(p + 1e-9) + (1-p)*np.log(1-p + 1e-9))) for p in probs]
                                tbl = pa.table({
                                    'tile': [r[0] for r in tile_preds],
                                    'row_idx': [r[1] for r in tile_preds],
                                    'col_idx': [r[2] for r in tile_preds],
                                    'center_lat': [r[3] for r in tile_preds],
                                    'center_lon': [r[4] for r in tile_preds],
                                    'predicted_prob': probs,
                                    'ndvi': [r[6] for r in tile_preds],
                                    'entropy': ent,
                                    'round': [round_num] * len(tile_preds),
                                    'model': [model_choice] * len(tile_preds),
                                })
                                out_parq = os.path.join(chunk_dir, f"{base}.parquet")
                                pq.write_table(tbl, out_parq)
                            except Exception:
                                pass
                    if prog_ctx is not None and task is not None:
                        prog_ctx.update(task, advance=1)
            finally:
                if writer is not None:
                    try:
                        wf.flush(); wf.close()  # type: ignore[name-defined]
                    except Exception:
                        pass
                if prog_ctx is not None:
                    try:
                        prog_ctx.__exit__(None, None, None)
                    except Exception:
                        pass
            # If top-N requested, write the selected rows now
            if save_preds and top_n_predictions is not None and heap_cap > 0:
                with open(pred_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
                    for (_, __, row) in sorted(top_heap, key=lambda t: t[0], reverse=True):
                        w.writerow(row)
            # Append parquet chunks to global store (sequential branch)
            if getattr(cfg, 'PERSISTENT_PARQUET_ENABLED', False):
                try:
                    import pyarrow as pa, pyarrow.parquet as pq
                    gp_path = getattr(cfg, 'GLOBAL_PREDICTIONS_PARQUET', None)
                    if gp_path:
                        writer = None
                        os.makedirs(os.path.dirname(gp_path), exist_ok=True)
                        files = [f for f in os.listdir(chunk_dir)] if os.path.exists(chunk_dir) else []
                        for fn in files:
                            if not fn.endswith('.parquet'):
                                continue
                            pfile = os.path.join(chunk_dir, fn)
                            tbl = pq.read_table(pfile)
                            if writer is None:
                                writer = pq.ParquetWriter(gp_path, tbl.schema)
                            writer.write_table(tbl)
                        if writer is not None:
                            writer.close()
                        # Mirror into highscore/probableAgri
                        hs_path = getattr(cfg, 'HIGHSCORE_PARQUET', None)
                        pa_path = getattr(cfg, 'PROBABLE_AGRI_PARQUET', None)
                        if hs_path:
                            writer = None
                            for fn in files:
                                if not fn.endswith('.parquet'):
                                    continue
                                pfile = os.path.join(chunk_dir, fn)
                                tbl = pq.read_table(pfile)
                                if writer is None:
                                    writer = pq.ParquetWriter(hs_path, tbl.schema)
                                writer.write_table(tbl)
                            if writer is not None:
                                writer.close()
                        if pa_path:
                            writer = None
                            for fn in files:
                                if not fn.endswith('.parquet'):
                                    continue
                                pfile = os.path.join(chunk_dir, fn)
                                tbl = pq.read_table(pfile)
                                if writer is None:
                                    writer = pq.ParquetWriter(pa_path, tbl.schema)
                                writer.write_table(tbl)
                            if writer is not None:
                                writer.close()
                    # After updating parquet, run global aggregation + KMLs
                    from persistent_aggregator import rebuild_global_rankings, build_global_prediction_kmls
                    try:
                        rebuild_global_rankings(); build_global_prediction_kmls()
                    except Exception as e:
                        print(f"Global aggregation failed: {e}")
                    # cleanup parquet chunks
                    try:
                        if os.path.exists(chunk_dir):
                            for fn in os.listdir(chunk_dir):
                                if fn.endswith('.parquet'):
                                    os.remove(os.path.join(chunk_dir, fn))
                    except Exception:
                        pass
                except Exception:
                    pass
        print(f"Total pixels inferred: {total_pixels}")
        print(f"Inference completed in {str(datetime.timedelta(seconds=int(time.time() - start)))}")

    # old parallel tile inference removed in favor of streaming per-tile writes

    # Optionally keep only top-N predictions by uncertainty
    # (Handled during streaming above.)

    # outputs
    if save_preds:
        print(f"Predictions written to {os.path.join(rnd_dir, 'predictions.csv')}")
        save_agricultural_polygons_kml(rnd_dir, round_num)
    else:
        print("Skipping predictions.csv generation and polygonization.")

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
                if len(exp_names) != importances.size:
                    raise RuntimeError(f"Feature importance naming mismatch: expected {importances.size} features, have {len(exp_names)} names.")
                names = exp_names
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
                    plt.figure(figsize=(8, max(3, topk*0.3)))
                    plt.barh(range(topk), importances[order][:topk][::-1])
                    plt.yticks(range(topk), [names[i] for i in order][:topk][::-1], fontsize=7)
                    plt.tight_layout()
                    plt.savefig(os.path.join(stats_dir, 'feature_importance.png'), dpi=180)
                    plt.close()
                except Exception as e:
                    print(f"Feature importance plot failed: {e}")
                    raise
    except Exception as e:
        # Halt pipeline as requested when FI naming fails or other errors occur
        raise

    # update persistent informative lists (highscore and probableAgri) whenever predictions are saved
    # so that assisted labeling lists are always available across modes
    if save_preds:
        try:
            pcsv = os.path.join(rnd_dir, "predictions.csv")
            _update_persistent_lists_from_csv(pcsv, rows, X, y, round_num, rnd_dir)
        except Exception as e:
            print(f"Persistent list update error: {e}")

    if return_metrics or not save_preds or not request_labels:
        print(f"Round {round_num} complete (no candidate labeling).")
        # Delete heavy predictions once no longer needed
        try:
            pcsv = os.path.join(rnd_dir, "predictions.csv")
            if os.path.exists(pcsv):
                os.remove(pcsv)
        except Exception:
            pass
        return metrics

    tmp = candidate_selection_from_csv(
        os.path.join(rnd_dir, "predictions.csv"),
        rnd_dir,
        round_num,
        train_rows=rows,
        X_train=X,
        y_train=y,
    )
    print(f"Round {round_num} complete; labels at {tmp}")
    # After candidate selection consumed predictions.csv, delete it to save space
    try:
        pcsv = os.path.join(rnd_dir, "predictions.csv")
        if os.path.exists(pcsv):
            os.remove(pcsv)
    except Exception:
        pass
    return tmp


def _update_persistent_lists(preds, train_rows, X_train, y_train, round_num, round_dir):
    """Update global and per-round highscore/probableAgri CSVs and KMLs.

    preds: list of [tile,row,col,lat,lon,prob,ndvi]
    train_rows: label rows used for training (for feature-space representativeness)
    X_train, y_train: feature matrix and labels (scaled later inside model)
    """
    import csv as _csv
    # build set of master labels to exclude
    labeled_keys = set()
    for r in train_rows:
        try:
            tile = r["tile"]; lat = float(r["lat"]); lon = float(r["lon"])
        except Exception:
            continue
        # snap to pixel centers is assumed upstream; use string key
        labeled_keys.add(f"{tile}:{lat:.7f}:{lon:.7f}")

    # compute uncertainty (entropy)
    probs = np.array([p[5] for p in preds])
    entropy = -probs*np.log(probs + 1e-9) - (1 - probs)*np.log(1 - probs + 1e-9)

    # choose a pool for highscore: top by entropy
    pool_size = max(cfg.HIGHSCORE_TOP_K * 10, cfg.HIGHSCORE_TOP_K)
    pool_idx = np.argsort(entropy)[::-1][:pool_size]

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
    for i in pool_idx:
        tile, r, c, la, lo, p, ndvi = preds[i]
        # skip if in master labels
        if f"{tile}:{la:.7f}:{lo:.7f}" in labeled_keys:
            continue
        f = feat_at(tile, r, c)
        if f is None:
            continue
        cand_feats.append(f)
        cand_meta.append((i, tile, r, c, la, lo, p, ndvi))
    if not cand_feats:
        return
    CF = np.vstack(cand_feats)
    # standardize both candidate and train features with train stats
    Xs = (X_train - feat_means) / (feat_std + 1e-6)
    CFs = (CF - feat_means) / (feat_std + 1e-6)
    # compute L2 distances to nearest labeled (brute-force for now)
    # shape (Ncand, Ntrain) -> take min
    dists = np.sqrt(((CFs[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2))
    core_dist = dists.min(axis=1)
    # normalize components 0-1 (rank-based)
    def rank_norm(v):
        r = np.argsort(np.argsort(v))
        return r.astype(np.float32) / max(len(v) - 1, 1)
    U = rank_norm(np.array([entropy[m[0]] for m in cand_meta]))
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
    selected_idx = order[: cfg.HIGHSCORE_TOP_K]
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
            "entropy": float(entropy[i]),
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
    # drop any that are now in master labels
    updated = {k: v for k, v in updated.items() if f"{v.get('tile')}:{float(v.get('lat')):.7f}:{float(v.get('lon')):.7f}" not in labeled_keys}
    # sort by score desc, keep top K
    def score_of(v):
        try:
            return float(v.get('score', 0))
        except Exception:
            return 0.0
    top_items = sorted(updated.values(), key=score_of, reverse=True)[: cfg.HIGHSCORE_TOP_K]
    # persist CSV and KML
    if top_items:
        with open(cfg.HIGHSCORE_FILE, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(top_items[0].keys()))
            w.writeheader(); w.writerows(top_items)
        _write_points_kml(top_items, cfg.HIGHSCORE_KML_GLOBAL, placemark_prefix="#")
        round_hs_kml = os.path.join(round_dir, f"highscore_top.kml")
        _write_points_kml(top_items, round_hs_kml, placemark_prefix=f"#r{round_num}")

    # probableAgri: top positives by confidence × representativeness
    probs_all = np.array([p[5] for p in preds])
    idx_pos = np.where(probs_all >= cfg.MIN_AGRI_PROB)[0]
    pos_meta = []
    pos_feats = []
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
        PF = np.vstack(pos_feats)
        PFs = (PF - feat_means) / (feat_std + 1e-6)
        dpos = np.sqrt(((PFs[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        P = rank_norm(np.array([m[6] for m in pos_meta]))
        Rpos = rank_norm(dpos)
        wconf = cfg.PROBABLE_AGRI_COMPONENT_WEIGHTS.get("confidence", 0.7)
        wrep = cfg.PROBABLE_AGRI_COMPONENT_WEIGHTS.get("representativeness", 0.3)
        ps = wconf * P + wrep * Rpos
        orderp = np.argsort(ps)[::-1]
        pa_rows = []
        for k in orderp[: cfg.PROBABLE_AGRI_TOP_K]:
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
        # drop labeled
        existing_pa = {k: v for k, v in existing_pa.items() if f"{v.get('tile')}:{float(v.get('lat')):.7f}:{float(v.get('lon')):.7f}" not in labeled_keys}
        top_pa = sorted(existing_pa.values(), key=lambda v: float(v.get('score', v.get('prob', 0))), reverse=True)[: cfg.PROBABLE_AGRI_TOP_K]
        if top_pa:
            with open(cfg.PROBABLE_AGRI_FILE, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(top_pa[0].keys()))
                w.writeheader(); w.writerows(top_pa)
            _write_points_kml(top_pa, cfg.PROBABLE_AGRI_KML_GLOBAL, placemark_prefix="PA")
            round_pa_kml = os.path.join(round_dir, f"probableAgri_top.kml")
            _write_points_kml(top_pa, round_pa_kml, placemark_prefix=f"PA_r{round_num}")
            
def _update_persistent_lists_from_csv(pred_csv, train_rows, X_train, y_train, round_num, round_dir):
    """Lightweight adapter: load predictions.csv into the native preds list format and reuse logic.
    
    preds row format expected by _update_persistent_lists:
      [tile:str, row:int, col:int, lat:float, lon:float, prob:float, ndvi:float]
    """
    if not os.path.exists(pred_csv):
        return
    preds = []
    with open(pred_csv, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            try:
                preds.append([
                    r["tile"],
                    int(r["row_idx"]),
                    int(r["col_idx"]),
                    float(r["center_lat"]),
                    float(r["center_lon"]),
                    float(r["predicted_prob"]),
                    float(r.get("ndvi", "nan")) if r.get("ndvi") not in (None, "",) else float("nan"),
                ])
            except Exception:
                continue
    if preds:
        _update_persistent_lists(preds, train_rows, X_train, y_train, round_num, round_dir)


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
    for rank, r in enumerate(rows, 1):
        pm = SubElement(doc, 'Placemark')
        SubElement(pm, 'name').text = f"{placemark_prefix}{rank} {r['tile']} score={r.get('score', 0):.3f}"
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
    xml = parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
    with open(out_path, 'wb') as f:
        f.write(xml)

def candidate_selection_from_csv(pred_csv, round_dir, round_num, train_rows=None, X_train=None, y_train=None):
    """Load predictions from CSV and prompt the user to label candidates."""
    if not os.path.exists(pred_csv):
        print(f"Missing predictions CSV => {pred_csv}")
        return None
    preds = []
    with open(pred_csv, "r") as pf:
        rd = csv.reader(pf)
        next(rd, None)
        for row in rd:
            ndvi_val = float(row[6]) if len(row) > 6 else 0.0
            preds.append([row[0], int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5]), ndvi_val])

    def select_candidates_entropy(predictions):
        import numpy as np
        from sklearn.cluster import DBSCAN
        # vectors
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
        return [predictions[i] for i in picks]

    cands = select_candidates_entropy(preds)
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
    labeled_total = 0
    labeled_agri = 0
    labeled_non = 0
    skipped = 0
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
        print(f"\rCandidate {idx+1}/{len(cands)}: {t} r={r},c={c}, p={p:.3f}, ndvi={ndvi:.3f} :: {reason}", end="", flush=True)
        ui = input("Label? (1=Agri,2=NonAgri,3=Skip): ").strip()
        if ui == "3":
            print("\rSkipped.", end="", flush=True)
            skipped += 1
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
            labeled_total += 1
            if lab.lower().startswith("agri"):
                labeled_agri += 1
            else:
                labeled_non += 1
            print("\rLabel saved.", end="", flush=True)
    print(f"\nSession summary (AL round {round_num}): labeled={labeled_total} (agri={labeled_agri}, non={labeled_non}), skipped={skipped}")
    return tmp


if __name__ == "__main__":
    active_learning_round(1, TEMP_LABELS_FILE, "SVM")
