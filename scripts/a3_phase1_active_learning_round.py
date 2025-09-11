#!/usr/bin/env python3
# scripts/a3_phase1_active_learning_round.py

import os
from multiprocessing import cpu_count

# -----------------------------------------------------------------------------
# 1) Speed‐ups: thread‐tune BLAS/OpenMP to use all CPU cores
# -----------------------------------------------------------------------------
# Threading caps to avoid OpenBLAS/OpenMP warnings and oversubscription
# Choose conservative defaults and let user/env override explicitly.
_DEFAULT_THREADS = str(max(1, min(4, (cpu_count() or 1))))
# Do not override if already set in environment
os.environ.setdefault("OMP_NUM_THREADS", _DEFAULT_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _DEFAULT_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _DEFAULT_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _DEFAULT_THREADS)

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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from scipy.spatial import cKDTree

# Note: grid KML generation is performed once at pipeline start.
from features import add_derived_features
from al_shared import extract_features_from_label, get_tile_features, load_skipped_set, record_skipped_pixel



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
    # Use a single-process DataLoader to avoid worker spawn issues on WSL/Windows
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
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


# Cap PyTorch internal thread pools to avoid oversubscription
try:
    _torch_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    if _torch_threads > 0:
        try:
            torch.set_num_threads(_torch_threads)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
except Exception:
    pass


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
        # Try GPU first (gpu_hist + gpu_predictor). If it fails, fall back to CPU.
        base_kwargs = dict(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            objective="binary:logistic",
            n_jobs=-1,
            random_state=None if cfg.SPLIT_SEED_MODE == "random" else int(cfg.SPLIT_RANDOM_SEED),
        )
        tried_gpu = False
        try:
            xgb_clf = xgb.XGBClassifier(
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                **base_kwargs,
            )
            tried_gpu = True
            xgb_clf.fit(Xs, y)
        except Exception as _gpu_err:
            # Fallback: CPU histogram
            try:
                xgb_clf = xgb.XGBClassifier(
                    tree_method="hist",
                    predictor="auto",
                    **base_kwargs,
                )
                xgb_clf.fit(Xs, y)
                if tried_gpu:
                    print("XGBoost GPU unavailable; fell back to CPU (hist).")
            except Exception as _cpu_err:
                raise RuntimeError(f"XGBoost training failed (GPU then CPU). GPU err={_gpu_err}; CPU err={_cpu_err}")
        w = SklearnWrapper(xgb_clf, feat_means, feat_std)
        w.kind = 'xgboost'
        return w

    else:
        raise ValueError(f"Unknown model choice: {choice}")


# -----------------------------------------------------------------------------
# 4) Fast batch inference + per‐pixel geometry
# -----------------------------------------------------------------------------
def get_pixel_corners(src, r, c):
    """Return corner coordinates for a single pixel as (lon, lat) pairs in WGS84.

    Uses pixel-corner offsets to avoid center-based misalignment.
    """
    # Compute pixel corners in dataset CRS
    tl = src.xy(r, c, offset='ul')
    tr = src.xy(r, c, offset='ur')
    br = src.xy(r, c, offset='lr')
    bl = src.xy(r, c, offset='ll')
    corners = [tl, tr, br, bl, tl]
    # Reproject to WGS84 if necessary
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


def _kml_polygons_from_tilefile(tile_csv_path, raw_data_dir, min_agri_prob, sieve_size):
    """Worker function for parallel KML polygonization.

    Reads a per-tile predictions shard (prefers .npy sidecar) and returns a list
    of polygon rings as lists of (lon, lat) tuples.
    """
    try:
        import numpy as _np
        base = os.path.basename(tile_csv_path)
        core = base[:-4] if base.endswith('.csv') else os.path.splitext(base)[0]
        tile_name = core + '.tif'
        tif = os.path.join(raw_data_dir, tile_name)
        if not os.path.exists(tif):
            return []
        # Load predictions
        npy = tile_csv_path[:-4] + '.npy' if tile_csv_path.endswith('.csv') else os.path.splitext(tile_csv_path)[0] + '.npy'
        if os.path.exists(npy):
            arr = _np.load(npy, mmap_mode='r')
            if arr.size == 0:
                return []
            rr = arr[:, 0].astype(_np.int32)
            cc = arr[:, 1].astype(_np.int32)
            pr = arr[:, 2].astype(_np.float32)
        else:
            rr = []; cc = []; pr = []
            with open(tile_csv_path, newline='') as f:
                rd = csv.DictReader(f)
                flds = [x.strip().lower() for x in (rd.fieldnames or [])]
                has_tilepred = all(k in flds for k in ['row_idx','col_idx','predicted_prob'])
                has_shard = all(k in flds for k in ['row','col','prob'])
                for r in rd:
                    try:
                        if has_tilepred:
                            rr.append(int(r['row_idx'])); cc.append(int(r['col_idx'])); pr.append(float(r['predicted_prob']))
                        elif has_shard:
                            rr.append(int(r['row'])); cc.append(int(r['col'])); pr.append(float(r['prob']))
                    except Exception:
                        continue
            rr = _np.asarray(rr, dtype=_np.int32); cc = _np.asarray(cc, dtype=_np.int32); pr = _np.asarray(pr, dtype=_np.float32)
            if rr.size == 0:
                return []
        import rasterio
        from rasterio.features import sieve as _sieve, shapes as _shapes
        from scipy.ndimage import binary_closing, binary_fill_holes
        from pyproj import Transformer as _Transformer
        from shapely.geometry import shape as _shape
        from shapely.ops import transform as _shp_transform
        with rasterio.open(tif) as src:
            H, W = src.height, src.width
            mask = _np.zeros((H, W), dtype=_np.uint8)
            idx = pr >= float(min_agri_prob)
            if not _np.any(idx):
                return []
            rrs = rr[idx]; ccs = cc[idx]
            valid = (rrs >= 0) & (rrs < H) & (ccs >= 0) & (ccs < W)
            rrs = rrs[valid]; ccs = ccs[valid]
            if rrs.size == 0:
                return []
            mask[rrs, ccs] = 1
            mask = binary_fill_holes(binary_closing(mask.astype(bool))).astype(_np.uint8)
            if int(sieve_size) > 0:
                try:
                    import warnings
                    from rasterio.errors import NotGeoreferencedWarning
                except Exception:
                    class NotGeoreferencedWarning(Warning):
                        pass
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", NotGeoreferencedWarning)
                    mask = _sieve(mask, size=int(sieve_size), connectivity=8).astype(_np.uint8)
            rings = []
            transformer = None
            if src.crs and not src.crs.is_geographic:
                transformer = _Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            for geom, val in _shapes(mask, mask=(mask>0), transform=src.transform):
                if not val:
                    continue
                coords_img = geom['coordinates'][0]
                if transformer:
                    ring = [transformer.transform(x, y) for x, y in coords_img]
                else:
                    ring = [(x, y) for x, y in coords_img]
                rings.append(ring)
            return rings
    except Exception:
        return []


def _per_tile_metrics_worker(tile_file,
                             has_train,
                             feat_means,
                             feat_std,
                             Xs_train,
                             labeled_keys_list,
                             hs_on,
                             pa_on,
                             min_agri_prob,
                             kd_workers,
                             chunk_rows,
                             dists_dir,
                             pos_dir):
    """Process one tile shard: compute per-chunk distances/positives, then merge to final.

    Returns (tile_name, total_rows, pos_count).
    """
    import numpy as _np
    import csv as _csv
    from scipy.spatial import cKDTree as _cKDTree
    base = os.path.basename(tile_file)
    if base.endswith('.csv.gz'):
        base_core = base[:-7]
    elif base.endswith('.csv'):
        base_core = base[:-4]
    else:
        base_core = os.path.splitext(base)[0]
    tile_name = base_core + '.tif'
    labeled_keys = set(labeled_keys_list or [])

    # Training KD tree per process (training set is small in this project)
    kd = None
    if hs_on and has_train and Xs_train is not None and getattr(Xs_train, 'size', 0) > 0:
        try:
            kd = _cKDTree(_np.asarray(Xs_train, dtype=_np.float32))
        except Exception:
            kd = None

    # Load features for this tile once if needed
    arr = None
    if has_train:
        tf = get_tile_features(tile_name)
        if tf is None:
            return (tile_name, 0, 0)
        arr, _, _ = tf

    def _query_dists(Q):
        if kd is None:
            return _np.zeros((Q.shape[0],), dtype=_np.float32)
        try:
            d, _ = kd.query(Q, k=1, workers=int(kd_workers))
            return d.astype(_np.float32, copy=False)
        except Exception:
            try:
                d, _ = kd.query(Q, k=1)
                return d.astype(_np.float32, copy=False)
            except Exception:
                # last resort: zeros
                return _np.zeros((Q.shape[0],), dtype=_np.float32)

    dists_parts = []
    pos_parts = []
    pos_count = 0
    total_rows = 0

    def flush_chunk(rr, cc, la, lo, pr, nd, part_idx):
        nonlocal pos_count, total_rows
        if rr.size == 0:
            return part_idx
        total_rows += rr.size
        # Improve locality
        try:
            order_loc = _np.lexsort((cc, rr))
            rr, cc, la, lo, pr, nd = rr[order_loc], cc[order_loc], la[order_loc], lo[order_loc], pr[order_loc], nd[order_loc]
        except Exception:
            pass
        # distances (Highscore only)
        if hs_on:
            if has_train and arr is not None:
                feats = arr[:, rr, cc].transpose(1, 0).astype(_np.float32)
                valid = ~_np.isnan(feats).any(axis=1)
                if not _np.all(valid):
                    rr, cc, la, lo, pr, nd, feats = (rr[valid], cc[valid], la[valid], lo[valid], pr[valid], nd[valid], feats[valid])
                feats_s = (feats - feat_means) / (feat_std + 1e-6)
                dists = _query_dists(feats_s)
            else:
                dists = _np.zeros((rr.shape[0],), dtype=_np.float32)
            order = _np.argsort(dists)
            dp_part = os.path.join(dists_dir, f"{base_core}_dists.part{part_idx}.csv.gz")
            with _open_text_auto(dp_part, 'wt') as fd:
                w = _csv.writer(fd)
                w.writerow(['tile','row','col','dist'])
                for j in order:
                    w.writerow([tile_name, int(rr[j]), int(cc[j]), float(dists[j])])
            try:
                arr_side = _np.vstack([
                    rr[order].astype(_np.int32),
                    cc[order].astype(_np.int32),
                    dists[order].astype(_np.float32)
                ]).T
                npy_part = dp_part[:-7] + '.npy'
                _np.save(npy_part, arr_side)
            except Exception:
                pass
            dists_parts.append(dp_part)
        # positives (ProbableAgri only)
        if pa_on:
            idx_pos = _np.where(pr >= float(min_agri_prob))[0]
            if idx_pos.size:
                pos_count += int(idx_pos.size)
                orderp = idx_pos[_np.argsort(pr[idx_pos])[::-1]]
                pp_part = os.path.join(pos_dir, f"{base_core}_pos.part{part_idx}.csv.gz")
                with _open_text_auto(pp_part, 'wt') as fp:
                    w = _csv.writer(fp)
                    w.writerow(['tile','row','col','lat','lon','prob','ndvi'])
                    for j in orderp:
                        w.writerow([tile_name, int(rr[j]), int(cc[j]), float(la[j]), float(lo[j]), float(pr[j]), float(nd[j])])
                try:
                    arrp = _np.vstack([
                        rr[orderp].astype(_np.int32),
                        cc[orderp].astype(_np.int32),
                        la[orderp].astype(_np.float32),
                        lo[orderp].astype(_np.float32),
                        pr[orderp].astype(_np.float32),
                        nd[orderp].astype(_np.float32)
                    ]).T
                    npy_pp = pp_part[:-7] + '.npy'
                    _np.save(npy_pp, arrp)
                except Exception:
                    pass
                pos_parts.append(pp_part)
        return part_idx + 1

    # Stream read shard
    CHUNK = int(chunk_rows)
    chunk_r, chunk_c, chunk_la, chunk_lo, chunk_p, chunk_nd = [], [], [], [], [], []
    part_idx = 0
    with _open_text_auto(tile_file, 'rt') as f:
        rd = _csv.DictReader(f)
        flds = [x.strip().lower() for x in (rd.fieldnames or [])]
        has_shard = all(k in flds for k in ['row','col','lat','lon','prob'])
        has_tilepred = all(k in flds for k in ['row_idx','col_idx','center_lat','center_lon','predicted_prob'])
        for r in rd:
            try:
                if has_shard:
                    ri = int(r['row']); ci = int(r['col'])
                    la = float(r['lat']); lo = float(r['lon'])
                    p = float(r['prob']); ndv = float(r.get('ndvi') or 0.0)
                elif has_tilepred:
                    ri = int(r['row_idx']); ci = int(r['col_idx'])
                    la = float(r['center_lat']); lo = float(r['center_lon'])
                    p = float(r['predicted_prob']); ndv = float(r.get('ndvi') or 0.0)
                else:
                    continue
            except Exception:
                continue
            key = f"{tile_name}:{la:.7f}:{lo:.7f}"
            if key in labeled_keys:
                continue
            chunk_r.append(ri); chunk_c.append(ci); chunk_la.append(la); chunk_lo.append(lo); chunk_p.append(p); chunk_nd.append(ndv)
            if len(chunk_r) >= CHUNK:
                rr = _np.asarray(chunk_r, dtype=_np.int32)
                cc = _np.asarray(chunk_c, dtype=_np.int32)
                laa = _np.asarray(chunk_la, dtype=_np.float32)
                loo = _np.asarray(chunk_lo, dtype=_np.float32)
                pr = _np.asarray(chunk_p, dtype=_np.float32)
                nd = _np.asarray(chunk_nd, dtype=_np.float32)
                part_idx = flush_chunk(rr, cc, laa, loo, pr, nd, part_idx)
                chunk_r.clear(); chunk_c.clear(); chunk_la.clear(); chunk_lo.clear(); chunk_p.clear(); chunk_nd.clear()
        if chunk_r:
            rr = _np.asarray(chunk_r, dtype=_np.int32)
            cc = _np.asarray(chunk_c, dtype=_np.int32)
            laa = _np.asarray(chunk_la, dtype=_np.float32)
            loo = _np.asarray(chunk_lo, dtype=_np.float32)
            pr = _np.asarray(chunk_p, dtype=_np.float32)
            nd = _np.asarray(chunk_nd, dtype=_np.float32)
            part_idx = flush_chunk(rr, cc, laa, loo, pr, nd, part_idx)

    # Merge parts for dists
    if dists_parts:
        import heapq as _hq
        dp_final = os.path.join(dists_dir, base_core + '_dists.csv.gz')
        with _open_text_auto(dp_final, 'wt') as out:
            w = _csv.writer(out)
            w.writerow(['tile','row','col','dist'])
            sources = []
            for pp in dists_parts:
                npy_part = pp[:-7] + '.npy'
                if os.path.exists(npy_part):
                    try:
                        arr = _np.load(npy_part)
                        sources.append({'type': 'npy', 'data': arr, 'pos': 0})
                        continue
                    except Exception:
                        pass
                f = _open_text_auto(pp, 'rt'); r = _csv.reader(f); next(r, None)
                sources.append({'type': 'csv', 'file': f, 'reader': r})
            heap = []
            for i, src in enumerate(sources):
                if src['type'] == 'npy':
                    arr = src['data']
                    if arr.shape[0] == 0:
                        continue
                    rr, cc, d = int(arr[0,0]), int(arr[0,1]), float(arr[0,2])
                    src['pos'] = 1
                    _hq.heappush(heap, (d, tile_name, rr, cc, i))
                else:
                    row = next(src['reader'], None)
                    if not row:
                        continue
                    try:
                        d = float(row[3]); rr = int(row[1]); cc = int(row[2])
                    except Exception:
                        continue
                    _hq.heappush(heap, (d, tile_name, rr, cc, i))
            while heap:
                d, ti, rr, cc, i = _hq.heappop(heap)
                w.writerow([ti, rr, cc, d])
                src = sources[i]
                if src['type'] == 'npy':
                    posi = src.get('pos', 0)
                    arr = src['data']
                    if posi < arr.shape[0]:
                        rr2, cc2, d2 = int(arr[posi,0]), int(arr[posi,1]), float(arr[posi,2])
                        src['pos'] = posi + 1
                        _hq.heappush(heap, (d2, tile_name, rr2, cc2, i))
                else:
                    row2 = next(src['reader'], None)
                    if row2:
                        try:
                            d2 = float(row2[3]); rr2 = int(row2[1]); cc2 = int(row2[2])
                        except Exception:
                            row2 = None
                        if row2:
                            _hq.heappush(heap, (d2, tile_name, rr2, cc2, i))
            for src in sources:
                if src['type'] == 'csv':
                    try: src['file'].close()
                    except Exception: pass
        # cleanup parts
        for pp in dists_parts:
            try: os.remove(pp)
            except Exception: pass
            npy_part = pp[:-7] + '.npy'
            if os.path.exists(npy_part):
                try: os.remove(npy_part)
                except Exception: pass

    # Merge parts for positives
    if pos_parts:
        import heapq as _hq
        pp_final = os.path.join(pos_dir, base_core + '_pos.csv.gz')
        with _open_text_auto(pp_final, 'wt') as out:
            w = _csv.writer(out)
            w.writerow(['tile','row','col','lat','lon','prob','ndvi'])
            sources = []
            for pp in pos_parts:
                npy_part = pp[:-7] + '.npy'
                if os.path.exists(npy_part):
                    try:
                        arr = _np.load(npy_part)
                        sources.append({'type':'npy','data':arr,'pos':0})
                        continue
                    except Exception:
                        pass
                f = _open_text_auto(pp, 'rt'); r = _csv.reader(f); next(r, None)
                sources.append({'type':'csv','file':f,'reader':r})
            heap = []
            for i, src in enumerate(sources):
                if src['type'] == 'npy':
                    arr = src['data']
                    if arr.shape[0] == 0:
                        continue
                    rr, cc = int(arr[0,0]), int(arr[0,1])
                    la, lo = float(arr[0,2]), float(arr[0,3])
                    pr, nd = float(arr[0,4]), float(arr[0,5])
                    src['pos'] = 1
                    _hq.heappush(heap, (-pr, i, (rr, cc, la, lo, pr, nd)))
                else:
                    row = next(src['reader'], None)
                    if not row:
                        continue
                    try:
                        rr = int(row[1]); cc = int(row[2])
                        la = float(row[3]); lo = float(row[4])
                        pr = float(row[5]); nd = float(row[6]) if len(row) > 6 and row[6] != '' else 0.0
                    except Exception:
                        continue
                    _hq.heappush(heap, (-pr, i, (rr, cc, la, lo, pr, nd)))
            while heap:
                neg, i, tpl = _hq.heappop(heap)
                rr, cc, la, lo, pr, nd = tpl
                w.writerow([tile_name, rr, cc, la, lo, pr, nd])
                src = sources[i]
                if src['type'] == 'npy':
                    posi = src.get('pos', 0)
                    arr = src['data']
                    if posi < arr.shape[0]:
                        rr2, cc2 = int(arr[posi,0]), int(arr[posi,1])
                        la2, lo2 = float(arr[posi,2]), float(arr[posi,3])
                        pr2, nd2 = float(arr[posi,4]), float(arr[posi,5])
                        src['pos'] = posi + 1
                        _hq.heappush(heap, (-pr2, i, (rr2, cc2, la2, lo2, pr2, nd2)))
                else:
                    row2 = next(src['reader'], None)
                    if row2:
                        try:
                            rr2 = int(row2[1]); cc2 = int(row2[2])
                            la2 = float(row2[3]); lo2 = float(row2[4])
                            pr2 = float(row2[5]); nd2 = float(row2[6]) if len(row2) > 6 and row2[6] != '' else 0.0
                        except Exception:
                            row2 = None
                        if row2:
                            _hq.heappush(heap, (-pr2, i, (rr2, cc2, la2, lo2, pr2, nd2)))
            for src in sources:
                if src['type'] == 'csv':
                    try: src['file'].close()
                    except Exception: pass
        # cleanup
        for pp in pos_parts:
            try: os.remove(pp)
            except Exception: pass
            npy_part = pp[:-7] + '.npy'
            if os.path.exists(npy_part):
                try: os.remove(npy_part)
                except Exception: pass

    return (tile_name, total_rows, pos_count)


def _per_tile_score_worker(tile_base,
                           ranks_dir,
                           shards_dir,
                           round_dir,
                           scored_dir,
                           min_agri_prob,
                           wu, wr, wc,
                           uncertainty_delta):
    import csv as _csv
    import numpy as _np
    rp = os.path.join(ranks_dir, tile_base + '_rank.csv')
    sp_csv = os.path.join(shards_dir, tile_base + '.csv')
    sp_gz = sp_csv + '.gz'
    sp = sp_gz if os.path.exists(sp_gz) else sp_csv
    if not os.path.exists(sp):
        alt = os.path.join(round_dir, '_tile_preds', tile_base + '.csv')
        if os.path.exists(alt):
            sp = alt
    if not (os.path.exists(sp) and os.path.exists(rp)):
        return None
    ranks = {}
    with open(rp) as f:
        rd = _csv.DictReader(f)
        for r in rd:
            ranks[(int(r['row']), int(r['col']))] = float(r['rank'])
    rows = []
    with _open_text_auto(sp, 'rt') as f:
        rd = _csv.DictReader(f)
        flds = [x.strip().lower() for x in (rd.fieldnames or [])]
        has_shard = all(k in flds for k in ['row','col','lat','lon','prob'])
        has_tilepred = all(k in flds for k in ['row_idx','col_idx','center_lat','center_lon','predicted_prob'])
        for r in rd:
            try:
                if has_shard:
                    rr = int(r['row']); cc = int(r['col'])
                    la = float(r['lat']); lo = float(r['lon'])
                    pr = float(r['prob']); nd = float(r.get('ndvi') or 0.0)
                elif has_tilepred:
                    rr = int(r['row_idx']); cc = int(r['col_idx'])
                    la = float(r['center_lat']); lo = float(r['center_lon'])
                    pr = float(r['predicted_prob']); nd = float(r.get('ndvi') or 0.0)
                else:
                    continue
            except Exception:
                continue
            R = ranks.get((rr, cc), 0.0)
            U = float(max(0.0, min(1.0, 1.0 - 2.0*abs(pr - 0.5))))
            C = 1.0 if abs(pr - float(min_agri_prob)) < float(uncertainty_delta) else 0.0
            score = wu*U + wr*R + wc*C
            tile_name = tile_base + '.tif'
            rows.append((score, (tile_name, rr, cc, la, lo, pr, nd)))
    if not rows:
        return None
    rows.sort(key=lambda t: t[0], reverse=True)
    outp = os.path.join(scored_dir, tile_base + '_scored.csv.gz')
    # optional .npy sidecar
    try:
        arr = _np.zeros((len(rows), 7), dtype=_np.float32)
        for i, (sc, tpl) in enumerate(rows):
            _, rr, cc, la, lo, pr, nd = tpl
            arr[i, :] = [float(rr), float(cc), float(la), float(lo), float(pr), float(nd), float(sc)]
        _np.save(os.path.join(scored_dir, tile_base + '_scored.npy'), arr)
    except Exception:
        pass
    with _open_text_auto(outp, 'wt') as f:
        w = _csv.writer(f)
        w.writerow(['tile','row','col','lat','lon','prob','ndvi','score'])
        for sc, tpl in rows:
            tile, rr, cc, la, lo, pr, nd = tpl
            w.writerow([tile, rr, cc, la, lo, pr, nd, f"{sc:.6f}"])
    return outp


def _merge_scored_files(files, out_path, total=None):
    """Single-process k-way merge of scored files (desc by score).

    Each input file is a CSV (optionally with .npy sidecar). Writes merged CSV.
    """
    import csv as _csv
    import numpy as _np
    import heapq
    use_sidecar = bool(getattr(cfg, 'BINARY_SIDECARS_ENABLED', False))
    hs = []
    sources = []  # ('npy', arr, pos, tile) or ('csv', file, reader)
    for fp in files:
        base = os.path.basename(fp)
        if base.endswith('.csv.gz'):
            npy = fp[:-7] + '.npy'
            core = base[:-7]
        elif base.endswith('.csv'):
            npy = fp[:-4] + '.npy'
            core = base[:-4]
        else:
            npy = os.path.splitext(fp)[0] + '.npy'
            core = os.path.splitext(base)[0]
        core2 = core.replace('_scored', '')
        tile_name = core2 + '.tif'
        if use_sidecar and os.path.exists(npy):
            try:
                arr = _np.load(npy, mmap_mode='r')
                if arr.shape[0] > 0:
                    rr, cc, la, lo, pr, nd, sc = [arr[0, i] for i in range(7)]
                    sources.append(('npy', arr, 1, tile_name))
                    heapq.heappush(hs, (-float(sc), (tile_name, int(rr), int(cc), float(la), float(lo), float(pr), float(nd), float(sc)), len(sources)-1))
                    continue
            except Exception:
                pass
        f = _open_text_auto(fp, 'rt'); r = _csv.reader(f); next(r, None)
        sources.append(('csv', f, r))
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
            heapq.heappush(hs, (-sc, row, len(sources)-1))
            break
    from progress_utils import new_progress as _npb2
    # Optional global sidecar (.npy) for highscore: (row,col,lat,lon,prob,ndvi,score)
    mm = None
    written = 0
    side_path = None
    if total and total > 0 and use_sidecar:
        try:
            side_path = out_path[:-4] + '.npy' if out_path.endswith('.csv') else os.path.splitext(out_path)[0] + '.npy'
            mm = _np.memmap(side_path, dtype=_np.float32, mode='w+', shape=(int(total), 7))
        except Exception:
            mm = None
            side_path = None
    with open(out_path,'w',newline='') as out, _npb2() as _prog2:
        w = _csv.writer(out); w.writerow(['tile','row','col','lat','lon','prob','ndvi','score'])
        t_hs = _prog2.add_task("Global highscore merge", total=total or 0)
        count = 0
        while hs:
            neg, row, idx = heapq.heappop(hs)
            if isinstance(row, (list, tuple)) and isinstance(row[0], str) and len(row) == 8:
                w.writerow(row)
                # write sidecar row if enabled
                if mm is not None:
                    try:
                        _, rr, cc, la, lo, pr, nd, sc = row
                        mm[written, 0] = float(rr)
                        mm[written, 1] = float(cc)
                        mm[written, 2] = float(la)
                        mm[written, 3] = float(lo)
                        mm[written, 4] = float(pr)
                        mm[written, 5] = float(nd)
                        mm[written, 6] = float(sc)
                        written += 1
                    except Exception:
                        pass
            else:
                w.writerow(row)
                if mm is not None:
                    try:
                        # row is a CSV row list: tile,row,col,lat,lon,prob,ndvi,score
                        rr = float(row[1]); cc = float(row[2]); la = float(row[3]); lo = float(row[4])
                        pr = float(row[5]); nd = float(row[6]); sc = float(row[7])
                        mm[written, 0] = rr; mm[written, 1] = cc; mm[written, 2] = la; mm[written, 3] = lo
                        mm[written, 4] = pr; mm[written, 5] = nd; mm[written, 6] = sc
                        written += 1
                    except Exception:
                        pass
            count += 1
            if total and count % 50000 == 0:
                _prog2.update(t_hs, completed=count)
            src = sources[idx]
            if src[0] == 'npy':
                arr, pos, tile_name = src[1], src[2], src[3]
                if pos < arr.shape[0]:
                    rr, cc, la, lo, pr, nd, sc2 = [arr[pos, i] for i in range(7)]
                    sources[idx] = ('npy', arr, pos+1, tile_name)
                    heapq.heappush(hs, (-float(sc2), (tile_name, int(rr), int(cc), float(la), float(lo), float(pr), float(nd), float(sc2)), idx))
            else:
                f, r = src[1], src[2]
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
            # finalize the progress bar to 100%. Use the active progress
            # instance from the context (not the context manager itself).
            _prog2.update(t_hs, completed=total)
    for src in sources:
        if src[0] == 'csv':
            try: src[1].close()
            except Exception: pass
    # finalize sidecar
    if mm is not None:
        try:
            mm.flush(); del mm
            # if fewer rows written than total, we could leave trailing zeros; acceptable for K indexing
        except Exception:
            # remove incomplete sidecar
            try:
                if side_path and os.path.exists(side_path):
                    os.remove(side_path)
            except Exception:
                pass


def _merge_pos_files(files, out_path, total=None):
    """Single-process k-way merge of probable-agri files (desc by prob)."""
    import csv as _csv
    import numpy as _np
    import heapq
    use_sidecar = bool(getattr(cfg, 'BINARY_SIDECARS_ENABLED', False))
    hs = []
    sources = []
    for fp in files:
        base = os.path.basename(fp)
        if base.endswith('.csv.gz'):
            npy = fp[:-7] + '.npy'
            core = base[:-7]
        elif base.endswith('.csv'):
            npy = fp[:-4] + '.npy'
            core = base[:-4]
        else:
            npy = os.path.splitext(fp)[0] + '.npy'
            core = os.path.splitext(base)[0]
        core2 = core.replace('_pos', '')
        tile_name = core2 + '.tif'
        if use_sidecar and os.path.exists(npy):
            try:
                arr = _np.load(npy, mmap_mode='r')
                if arr.shape[0] > 0:
                    rr, cc = int(arr[0,0]), int(arr[0,1])
                    la, lo = float(arr[0,2]), float(arr[0,3])
                    pr, nd = float(arr[0,4]), float(arr[0,5])
                    sources.append(('npy', arr, 1, tile_name))
                    heapq.heappush(hs, (-pr, (tile_name, rr, cc, la, lo, pr, nd), len(sources)-1))
                    continue
            except Exception:
                pass
        f = _open_text_auto(fp, 'rt'); r = _csv.reader(f); next(r, None)
        sources.append(('csv', f, r))
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
            heapq.heappush(hs, (-pr, row, len(sources)-1))
            break
    from progress_utils import new_progress as _npb3
    with open(out_path,'w',newline='') as out, _npb3() as _prog3:
        w = _csv.writer(out); w.writerow(['tile','row','col','lat','lon','prob','ndvi'])
        t_pa = _prog3.add_task("Global probableAgri merge", total=total or 0)
        count = 0
        while hs:
            neg, row, idx = heapq.heappop(hs)
            if isinstance(row, (list, tuple)) and isinstance(row[0], str) and len(row) >= 6:
                w.writerow(row)
            else:
                w.writerow(row)
            count += 1
            if total and count % 50000 == 0:
                _prog3.update(t_pa, completed=count)
            src = sources[idx]
            if src[0] == 'npy':
                arr, pos, tile_name = src[1], src[2], src[3]
                if pos < arr.shape[0]:
                    rr, cc = int(arr[pos,0]), int(arr[pos,1])
                    la, lo = float(arr[pos,2]), float(arr[pos,3])
                    pr, nd = float(arr[pos,4]), float(arr[pos,5])
                    sources[idx] = ('npy', arr, pos+1, tile_name)
                    heapq.heappush(hs, (-pr, (tile_name, rr, cc, la, lo, pr, nd), idx))
            else:
                f, r = src[1], src[2]
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
            _npb3().update(t_pa, completed=total)
    for src in sources:
        if src[0] == 'csv':
            try: src[1].close()
            except Exception: pass


def predict_entire_tile(tile_path, model, progress=None, task_id=None):
    """Run inference on a tile and stream rows to avoid big in-memory lists.

    Returns an iterator of rows: [tile, row, col, lat, lon, prob, ndvi].
    """
    tile_name = os.path.basename(tile_path)
    with rasterio.open(tile_path) as src:
        # Reuse cached per-tile features if available to avoid recomputation
        tile = os.path.basename(tile_path)
        tf = get_tile_features(tile)
        if tf is not None:
            arr, _, _ = tf
            names = current_feature_names()
        else:
            raw = src.read().astype(np.float32)
            arr, names = add_derived_features(raw)
        ch, H, W = arr.shape
        # Inference in spatial blocks to improve locality and smooth CPU
        block = int(getattr(cfg, 'INFER_BLOCK_SIZE', 512))

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

        bs = _effective_bs()
        a, bA, c0 = src.transform.a, src.transform.b, src.transform.c
        d, e, f0 = src.transform.d, src.transform.e, src.transform.f
        # center offset: add half a pixel
        ca = 0.5 * a + 0.5 * bA
        ce = 0.5 * d + 0.5 * e
        ndvi_vec = _ndvi_summary_from_names(arr, names)
        if ndvi_vec is None:
            ndvi_vec = np.zeros((H * W,), dtype=np.float32)

        def _row_iter():
            for r0 in range(0, H, block):
                r1 = min(H, r0 + block)
                for c_start in range(0, W, block):
                    c1 = min(W, c_start + block)
                    # Flatten features for this block
                    sub = arr[:, r0:r1, c_start:c1].reshape(ch, -1).T  # (pixels, bands)
                    # Predict in chunks inside the block
                    probs_blk = np.empty((sub.shape[0],), dtype=np.float32)
                    if cfg.INFER_CHUNKING_ENABLED:
                        for i in range(0, sub.shape[0], bs):
                            probs_blk[i:i + bs] = model.predict_proba(sub[i:i + bs])[:, 1].astype(np.float32)
                    else:
                        probs_blk[:] = model.predict_proba(sub)[:, 1].astype(np.float32)
                    # Compute row/col indices for this block
                    hB, wB = (r1 - r0), (c1 - c_start)
                    rows_blk = np.repeat(np.arange(r0, r1, dtype=np.int32), wB)
                    cols_blk = np.tile(np.arange(c_start, c1, dtype=np.int32), hB)
                    # Vectorized center coordinates via affine
                    xs = a * cols_blk + bA * rows_blk + c0 + ca
                    ys = d * cols_blk + e * rows_blk + f0 + ce
                    if src.crs and not src.crs.is_geographic:
                        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                        xs, ys = transformer.transform(xs, ys)
                    # NDVI for block from global vector
                    ndvi_blk = ndvi_vec.reshape(H, W)[r0:r1, c_start:c1].reshape(-1)
                    for r_i, c_i, lat, lon, p, nv in zip(rows_blk, cols_blk, ys, xs, probs_blk, ndvi_blk):
                        yield [tile_name, int(r_i), int(c_i), float(lat), float(lon), float(p), float(nv)]
            if progress is not None and task_id is not None:
                # Not tracking exact count here; streaming avoids peak RAM.
                try:
                    progress.update(task_id)
                except Exception:
                    pass

        return _row_iter()


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
    from collections.abc import Sequence
    streaming = not isinstance(rows, (list, tuple))
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob","ndvi"])
        w.writerows(rows)
    # Also write a binary sidecar for faster downstream consumption
    try:
        import numpy as _np
        # Only build sidecar when rows is a materialized sequence we can re-iterate
        if (not streaming) and rows:
            arr = _np.asarray([[r[1], r[2], r[5], r[6]] for r in rows], dtype=_np.float32)
            # first two columns were ints; cast back on load as needed
            side = os.path.join(tmp_dir, f"{os.path.splitext(tile_name)[0]}.npy")
            _np.save(side, arr)
    except Exception:
        pass
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
    # Keep per-tile shards to allow downstream steps to reuse them directly
    # (avoids re-splitting the merged predictions.csv during metrics refresh).
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
    Now prefers per-tile shards in `_tile_preds/` and parallelizes polygonization.
    """
    kml_path = os.path.join(round_folder, f"agricultural_patches_round_{round_num}.kml")

    # Prefer per-tile shards directory
    tile_dir = os.path.join(round_folder, '_tile_preds')
    rings_all = []
    if os.path.isdir(tile_dir):
        files = sorted([p for p in glob.glob(os.path.join(tile_dir, '*.csv'))])
        if files:
            with new_progress() as prog:
                task = prog.add_task("Polygonizing tiles", total=len(files))
                max_workers = max(1, (cpu_count() or 4) - 2)
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = {ex.submit(_kml_polygons_from_tilefile, fp, RAW_DATA_DIR, float(cfg.MIN_AGRI_PROB), int(getattr(cfg, 'SIEVE_MIN_SIZE', 0))): fp for fp in files}
                    for fut in as_completed(futs):
                        try:
                            rings_all.extend(fut.result() or [])
                        except Exception:
                            pass
                        prog.update(task, advance=1)
    else:
        # Fallback to merged predictions.csv path (sequential)
        pred_csv = pred_csv or os.path.join(round_folder, "predictions.csv")
        if not os.path.exists(pred_csv):
            print(f"Missing predictions file => {pred_csv}")
            return
        pred_map = _load_predictions_by_tile(pred_csv)
        with new_progress() as prog:
            task = prog.add_task("Polygonizing tiles", total=len(pred_map))
            for tile, (rows, cols, probs) in pred_map.items():
                tif = os.path.join(RAW_DATA_DIR, tile)
                if not os.path.exists(tif):
                    prog.update(task, advance=1)
                    continue
                with rasterio.open(tif) as src:
                    H, W = src.height, src.width
                    mask = np.zeros((H, W), dtype=np.uint8)
                    mask[rows, cols] = (probs >= float(cfg.MIN_AGRI_PROB)).astype(np.uint8)
                    from scipy.ndimage import binary_closing, binary_fill_holes
                    mask = binary_fill_holes(binary_closing(mask.astype(bool))).astype(np.uint8)
                    if getattr(cfg, 'SIEVE_MIN_SIZE', 0) > 0:
                        mask = sieve(mask, size=int(cfg.SIEVE_MIN_SIZE), connectivity=8).astype(np.uint8)
                    transformer = None
                    if src.crs and not src.crs.is_geographic:
                        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                    for geom, val in shapes(mask, mask=(mask>0), transform=src.transform):
                        if not val:
                            continue
                        coords_img = geom['coordinates'][0]
                        if transformer:
                            ring = [transformer.transform(x, y) for x, y in coords_img]
                        else:
                            ring = [(x, y) for x, y in coords_img]
                        rings_all.append(ring)
                prog.update(task, advance=1)

    if not rings_all:
        print(f"WARNING: No polygons (all probs < {cfg.MIN_AGRI_PROB})")
        return
    # Build KML with one blue style, streaming polygons (skip union for performance)
    doc = Element('Document')
    style_ag = SubElement(doc, 'Style', id='agri')
    ln = SubElement(style_ag, 'LineStyle'); SubElement(ln, 'color').text = 'ffffff55'; SubElement(ln, 'width').text = '1'
    ps = SubElement(style_ag, 'PolyStyle'); SubElement(ps, 'color').text = '40ffff55'; SubElement(ps, 'outline').text = '1'
    total_polys = 0
    for coords in rings_all:
        pm = SubElement(doc, 'Placemark')
        SubElement(pm, 'styleUrl').text = '#agri'
        poly_el = SubElement(pm, 'Polygon')
        ob = SubElement(poly_el, 'outerBoundaryIs')
        ring = SubElement(ob, 'LinearRing')
        SubElement(ring, 'coordinates').text = ' '.join(f"{lon},{lat},0" for lon, lat in coords)
        total_polys += 1

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
        # Update aggregate rounds metrics chart (best-effort)
        try:
            agg_script = os.path.join(os.path.dirname(__file__), 'plot_round_metrics.py')
            if os.path.exists(agg_script):
                subprocess.run([sys.executable, agg_script], check=False)
        except Exception:
            pass
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

    # Update persistent informative lists via the standalone script (optional).
    pred_csv_arg = os.path.join(rnd_dir, "predictions.csv")
    if getattr(cfg, 'HIGHSCORE_LIST_ENABLED', True) or getattr(cfg, 'PROBABLE_AGRI_LIST_ENABLED', False):
        try:
            script = os.path.join(os.path.dirname(__file__), "refresh_lists.py")
            print("Refreshing persistent lists via refresh_lists.py ...")
            subprocess.run([sys.executable, script, pred_csv_arg], check=True)
        except Exception as e:
            print(f"Persistent list update error: {e}")
    else:
        print("Persistent lists disabled in config; skipping refresh of Highscore/ProbableAgri.")

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
                try:
                    dpos, _ = kd2.query(PFs, k=1, workers=int(getattr(cfg, 'REFRESH_KD_WORKERS', 1)))
                except Exception:
                    try:
                        dpos, _ = kd2.query(PFs, k=1, workers=1)
                    except Exception:
                        dpos, _ = kd2.query(PFs, k=1)
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
    # Cleanup heavy per-round caches to save disk space (safe after lists/KML)
    try:
        for sub in ['_global_refresh', '_tile_preds']:
            p = os.path.join(round_dir, sub)
            if os.path.isdir(p):
                import shutil as _shutil
                _shutil.rmtree(p, ignore_errors=True)
                print(f"Removed cache folder => {p}")
    except Exception as _e:
        print(f"Round cache cleanup warning: {_e}")


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
                    # Build a compact windowed mask around the points to limit memory use
                    rs = [r for r, _ in pts if 0 <= r < H]
                    cs = [c for _, c in pts if 0 <= c < W]
                    if not rs or not cs:
                        continue
                    r0, r1 = max(0, min(rs)), min(H-1, max(rs))
                    c0, c1 = max(0, min(cs)), min(W-1, max(cs))
                    h_win = int(r1 - r0 + 1)
                    w_win = int(c1 - c0 + 1)
                    if h_win <= 0 or w_win <= 0:
                        continue
                    mask = _np.zeros((h_win, w_win), dtype=_np.uint8)
                    for r, c in pts:
                        if r0 <= r <= r1 and c0 <= c <= c1:
                            mask[r - r0, c - c0] = 1
                    if not mask.any():
                        continue
                    transformer = None
                    if src.crs and not src.crs.is_geographic:
                        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                    # Adjust transform for the window
                    try:
                        from affine import Affine as _Affine
                        window_transform = src.transform * _Affine.translation(c0, r0)
                    except Exception:
                        window_transform = src.transform
                    for geom, val in shapes(mask.astype('int32'), mask=(mask>0), transform=window_transform):
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
    hs_on = bool(getattr(cfg, 'HIGHSCORE_LIST_ENABLED', True))
    pa_on = bool(getattr(cfg, 'PROBABLE_AGRI_LIST_ENABLED', False))
    if not (hs_on or pa_on):
        print("Both persistent lists disabled; nothing to refresh.")
        return
    if hs_on and pa_on:
        print("Refreshing global Highscore and ProbableAgri from all pixels...")
    elif hs_on:
        print("Refreshing global Highscore list from all pixels...")
    else:
        print("Refreshing global ProbableAgri list from all pixels...")
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
        """Robust 1-NN distance query with safe fallbacks.

        Tries parallel workers from config; on failure, falls back to
        workers=1, then to default (no workers arg), and finally to a
        brute-force NumPy distance if KDTree query still errors.
        """
        if kd is None:
            try:
                n = int(Q.shape[0])
            except Exception:
                n = len(Q) if hasattr(Q, '__len__') else 0
            return np.zeros((n,), dtype=np.float32)
        try:
            d, _ = kd.query(Q, k=1, workers=int(getattr(cfg, 'REFRESH_KD_WORKERS', 1)))
            return d.astype(np.float32, copy=False)
        except Exception:
            try:
                d, _ = kd.query(Q, k=1, workers=1)
                return d.astype(np.float32, copy=False)
            except Exception:
                try:
                    d, _ = kd.query(Q, k=1)
                    return d.astype(np.float32, copy=False)
                except Exception:
                    try:
                        Q = np.asarray(Q, dtype=np.float32)
                        Xs_np = np.asarray(Xs, dtype=np.float32)
                        return np.sqrt(((Q[:, None, :] - Xs_np[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
                    except Exception:
                        # Last-resort: zeros (neutral representativeness)
                        return np.zeros((getattr(Q, 'shape', [0])[0] if hasattr(Q, 'shape') else len(Q)), dtype=np.float32)

    tmp_root = os.path.join(round_dir, '_global_refresh')
    shards_dir = os.path.join(tmp_root, 'shards')
    metrics_dir = os.path.join(tmp_root, 'metrics')
    dists_dir = os.path.join(tmp_root, 'dists')
    ranks_dir = os.path.join(tmp_root, 'ranks')
    scored_dir = os.path.join(tmp_root, 'scored')
    pos_dir = os.path.join(tmp_root, 'pos')
    for d in [shards_dir, metrics_dir, dists_dir, ranks_dir, scored_dir, pos_dir]:
        os.makedirs(d, exist_ok=True)

    # 1) Use existing per-tile shards if available to avoid a full split of the
    #    merged predictions.csv. Fallback to splitting when shards are missing.
    tile_preds_dir = os.path.join(round_dir, '_tile_preds')
    use_existing_shards = os.path.isdir(tile_preds_dir) and any(
        name.endswith('.csv') for name in os.listdir(tile_preds_dir)
    )
    if use_existing_shards:
        # We'll read shards directly from _tile_preds/*.csv (header differs but
        # is handled downstream). No need to populate tmp shards_dir.
        tile_files = _list_csvs(tile_preds_dir)
    else:
        _ = _split_predictions_to_shards(pred_csv, shards_dir)
        tile_files = _list_csvs(shards_dir)

    # 2) per-tile metrics and positives
    from progress_utils import new_progress as _npb
    with _npb() as prog_metrics:
        task_metrics = prog_metrics.add_task("Per-tile metrics", total=0)
        prog_metrics.update(task_metrics, total=len(tile_files))

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
                # Improve memory locality: process in row-major order before feature gather
                if rr.size:
                    try:
                        order_loc = np.lexsort((cc, rr))
                        rr, cc, la, lo, pr, nd = (
                            rr[order_loc], cc[order_loc], la[order_loc], lo[order_loc], pr[order_loc], nd[order_loc]
                        )
                    except Exception:
                        pass
                # distances (Highscore only)
                if hs_on:
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
                    # Sidecar .npy for faster merge (row,col,dist)
                    try:
                        arr_side = np.vstack([
                            rr[order].astype(np.int32),
                            cc[order].astype(np.int32),
                            dists[order].astype(np.float32)
                        ]).T
                        npy_part = dp_part[:-7] + '.npy' if dp_part.endswith('.csv.gz') else dp_part[:-4] + '.npy'
                        np.save(npy_part, arr_side)
                    except Exception:
                        pass
                    dists_parts.append(dp_part)
                # Positives sorted by prob desc for this chunk
                if pa_on:
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
                        # Sidecar .npy (row,col,lat,lon,prob,ndvi) sorted by prob desc
                        try:
                            arrp = np.vstack([
                                rr[orderp].astype(np.int32),
                                cc[orderp].astype(np.int32),
                                la[orderp].astype(np.float32),
                                lo[orderp].astype(np.float32),
                                pr[orderp].astype(np.float32),
                                nd[orderp].astype(np.float32)
                            ]).T
                            npy_pp = pp_part[:-7] + '.npy' if pp_part.endswith('.csv.gz') else pp_part[:-4] + '.npy'
                            np.save(npy_pp, arrp)
                        except Exception:
                            pass
                        pos_parts.append(pp_part)
                # reset chunk
                chunk_r.clear(); chunk_c.clear(); chunk_la.clear(); chunk_lo.clear(); chunk_p.clear(); chunk_nd.clear()
                return part_idx + 1

            part_idx = 0
            CHUNK = int(getattr(cfg, 'REFRESH_CHUNK_ROWS', 200000))
            with _open_text_auto(tile_file, 'rt') as f:
                rd = csv.DictReader(f)
                flds = [x.strip().lower() for x in (rd.fieldnames or [])]
                # Support both shard header formats:
                #  - shards: row,col,lat,lon,prob,ndvi
                #  - tile-preds: tile,row_idx,col_idx,center_lat,center_lon,predicted_prob,ndvi
                has_shard = all(k in flds for k in ['row','col','lat','lon','prob'])
                has_tilepred = all(k in flds for k in ['row_idx','col_idx','center_lat','center_lon','predicted_prob'])
                for r in rd:
                    try:
                        if has_shard:
                            ri = int(r['row']); ci = int(r['col'])
                            la = float(r['lat']); lo = float(r['lon'])
                            p = float(r['prob']); ndv = float(r.get('ndvi') or 0.0)
                        elif has_tilepred:
                            ri = int(r['row_idx']); ci = int(r['col_idx'])
                            la = float(r['center_lat']); lo = float(r['center_lon'])
                            p = float(r['predicted_prob']); ndv = float(r.get('ndvi') or 0.0)
                        else:
                            continue
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

            # Merge parts for dists (ascending) with optional .npy sidecars
            if dists_parts:
                import heapq as _hq
                import numpy as _np
                dp_final = os.path.join(dists_dir, base_core + '_dists.csv.gz')
                # Optional sidecar for final dists
                use_sidecar = bool(getattr(cfg, 'BINARY_SIDECARS_ENABLED', False))
                if use_sidecar and total_rows > 0:
                    try:
                        dp_side = os.path.join(dists_dir, base_core + '_dists.npy')
                        # 3 columns: row, col, dist (float32)
                        mm = _np.memmap(dp_side, dtype=_np.float32, mode='w+', shape=(int(total_rows), 3))
                        write_idx = 0
                    except Exception:
                        mm = None; write_idx = 0; use_sidecar = False
                else:
                    mm = None; write_idx = 0
                with _open_text_auto(dp_final, 'wt') as out:
                    w = csv.writer(out)
                    w.writerow(['tile','row','col','dist'])
                    sources = []
                    for pp in dists_parts:
                        npy_part = pp[:-7] + '.npy' if pp.endswith('.csv.gz') else pp[:-4] + '.npy'
                        if os.path.exists(npy_part):
                            try:
                                arr = _np.load(npy_part)
                                sources.append({'type': 'npy', 'data': arr, 'pos': 0})
                                continue
                            except Exception:
                                pass
                        f = _open_text_auto(pp, 'rt'); r = csv.reader(f); next(r, None)
                        sources.append({'type': 'csv', 'file': f, 'reader': r})
                    heap = []
                    for i, src in enumerate(sources):
                        if src['type'] == 'npy':
                            arr = src['data']
                            if arr.shape[0] == 0:
                                continue
                            rr, cc, d = int(arr[0,0]), int(arr[0,1]), float(arr[0,2])
                            src['pos'] = 1
                            _hq.heappush(heap, (d, tile_name, rr, cc, i))
                        else:
                            row = next(src['reader'], None)
                            if not row:
                                continue
                            try:
                                d = float(row[3]); rr = int(row[1]); cc = int(row[2])
                            except Exception:
                                continue
                            _hq.heappush(heap, (d, tile_name, rr, cc, i))
                    while heap:
                        d, ti, rr, cc, i = _hq.heappop(heap)
                        w.writerow([ti, rr, cc, d])
                        if mm is not None:
                            try:
                                mm[write_idx, 0] = float(rr)
                                mm[write_idx, 1] = float(cc)
                                mm[write_idx, 2] = float(d)
                                write_idx += 1
                            except Exception:
                                pass
                        src = sources[i]
                        if src['type'] == 'npy':
                            posi = src.get('pos', 0)
                            arr = src['data']
                            if posi < arr.shape[0]:
                                rr2, cc2, d2 = int(arr[posi,0]), int(arr[posi,1]), float(arr[posi,2])
                                src['pos'] = posi + 1
                                _hq.heappush(heap, (d2, tile_name, rr2, cc2, i))
                        else:
                            row2 = next(src['reader'], None)
                            if row2:
                                try:
                                    d2 = float(row2[3]); rr2 = int(row2[1]); cc2 = int(row2[2])
                                except Exception:
                                    row2 = None
                                if row2:
                                    _hq.heappush(heap, (d2, tile_name, rr2, cc2, i))
                    for src in sources:
                        if src['type'] == 'csv':
                            try: src['file'].close()
                            except Exception: pass
                if mm is not None:
                    try:
                        mm.flush(); del mm
                    except Exception:
                        pass
                # cleanup part files (CSV and NPY)
                for pp in dists_parts:
                    try: os.remove(pp)
                    except Exception: pass
                    npy_part = pp[:-7] + '.npy' if pp.endswith('.csv.gz') else pp[:-4] + '.npy'
                    if os.path.exists(npy_part):
                        try: os.remove(npy_part)
                        except Exception: pass

            # Merge parts for positives (descending by prob) with optional .npy sidecars
            if pos_parts:
                import heapq as _hq
                import numpy as _np
                pp_final = os.path.join(pos_dir, base_core + '_pos.csv.gz')
                # Optional sidecar for final positives
                use_sidecar2 = bool(getattr(cfg, 'BINARY_SIDECARS_ENABLED', False))
                if use_sidecar2 and pos_count > 0:
                    try:
                        pp_side = os.path.join(pos_dir, base_core + '_pos.npy')
                        mm2 = _np.memmap(pp_side, dtype=_np.float32, mode='w+', shape=(int(pos_count), 6))
                        write_idx2 = 0
                    except Exception:
                        mm2 = None; write_idx2 = 0; use_sidecar2 = False
                else:
                    mm2 = None; write_idx2 = 0
                with _open_text_auto(pp_final, 'wt') as out:
                    w = csv.writer(out)
                    w.writerow(['tile','row','col','lat','lon','prob','ndvi'])
                    sources = []
                    for pp in pos_parts:
                        npy_part = pp[:-7] + '.npy' if pp.endswith('.csv.gz') else pp[:-4] + '.npy'
                        if os.path.exists(npy_part):
                            try:
                                arr = _np.load(npy_part)
                                sources.append({'type':'npy','data':arr,'pos':0})
                                continue
                            except Exception:
                                pass
                        f = _open_text_auto(pp, 'rt'); r = csv.reader(f); next(r, None)
                        sources.append({'type':'csv','file':f,'reader':r})
                    heap = []
                    for i, src in enumerate(sources):
                        if src['type'] == 'npy':
                            arr = src['data']
                            if arr.shape[0] == 0:
                                continue
                            rr, cc = int(arr[0,0]), int(arr[0,1])
                            la, lo = float(arr[0,2]), float(arr[0,3])
                            pr, nd = float(arr[0,4]), float(arr[0,5])
                            src['pos'] = 1
                            _hq.heappush(heap, (-pr, i, (rr, cc, la, lo, pr, nd)))
                        else:
                            row = next(src['reader'], None)
                            if not row:
                                continue
                            try:
                                rr = int(row[1]); cc = int(row[2])
                                la = float(row[3]); lo = float(row[4])
                                pr = float(row[5]); nd = float(row[6]) if len(row) > 6 and row[6] != '' else 0.0
                            except Exception:
                                continue
                            _hq.heappush(heap, (-pr, i, (rr, cc, la, lo, pr, nd)))
                    while heap:
                        neg, i, tpl = _hq.heappop(heap)
                        rr, cc, la, lo, pr, nd = tpl
                        w.writerow([tile_name, rr, cc, la, lo, pr, nd])
                        if mm2 is not None:
                            try:
                                mm2[write_idx2, 0] = float(rr)
                                mm2[write_idx2, 1] = float(cc)
                                mm2[write_idx2, 2] = float(la)
                                mm2[write_idx2, 3] = float(lo)
                                mm2[write_idx2, 4] = float(pr)
                                mm2[write_idx2, 5] = float(nd)
                                write_idx2 += 1
                            except Exception:
                                pass
                        src = sources[i]
                        if src['type'] == 'npy':
                            posi = src.get('pos', 0)
                            arr = src['data']
                            if posi < arr.shape[0]:
                                rr2, cc2 = int(arr[posi,0]), int(arr[posi,1])
                                la2, lo2 = float(arr[posi,2]), float(arr[posi,3])
                                pr2, nd2 = float(arr[posi,4]), float(arr[posi,5])
                                src['pos'] = posi + 1
                                _hq.heappush(heap, (-pr2, i, (rr2, cc2, la2, lo2, pr2, nd2)))
                        else:
                            row2 = next(src['reader'], None)
                            if row2:
                                try:
                                    rr2 = int(row2[1]); cc2 = int(row2[2])
                                    la2 = float(row2[3]); lo2 = float(row2[4])
                                    pr2 = float(row2[5]); nd2 = float(row2[6]) if len(row2) > 6 and row2[6] != '' else 0.0
                                except Exception:
                                    row2 = None
                                if row2:
                                    _hq.heappush(heap, (-pr2, i, (rr2, cc2, la2, lo2, pr2, nd2)))
                    for src in sources:
                        if src['type'] == 'csv':
                            try: src['file'].close()
                            except Exception: pass
                if mm2 is not None:
                    try:
                        mm2.flush(); del mm2
                    except Exception:
                        pass
                for pp in pos_parts:
                    try: os.remove(pp)
                    except Exception: pass
                    npy_part = pp[:-7] + '.npy' if pp.endswith('.csv.gz') else pp[:-4] + '.npy'
                    if os.path.exists(npy_part):
                        try: os.remove(npy_part)
                        except Exception: pass

            # Done with this tile
            return (tile_name, total_rows, pos_count)

        # tile_files already resolved above from either _tile_preds or shards_dir
        # Process tiles in a process pool to parallelize Python-bound work
        res = []
        max_workers = int(getattr(cfg, 'REFRESH_TILE_THREADS', 2))
        has_train_flag = bool(has_train)
        feat_means_local = feat_means if has_train_flag else None
        feat_std_local = feat_std if has_train_flag else None
        Xs_local = Xs if has_train_flag else None
        labeled_list = list(labeled_keys)
        kd_workers = int(getattr(cfg, 'REFRESH_KD_WORKERS', 1))
        chunk_rows = int(getattr(cfg, 'REFRESH_CHUNK_ROWS', 200000))
        min_prob = float(cfg.MIN_AGRI_PROB)
        # Batch process to limit peak memory
        pool_batch = int(getattr(cfg, 'REFRESH_PROCESS_POOL_BATCH', 20))
        for i0 in range(0, len(tile_files), max(1, pool_batch)):
            batch = tile_files[i0:i0 + max(1, pool_batch)]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = {
                    ex.submit(
                        _per_tile_metrics_worker,
                        tp,
                        has_train_flag,
                        feat_means_local,
                        feat_std_local,
                        Xs_local,
                        labeled_list,
                        bool(hs_on),
                        bool(pa_on),
                        min_prob,
                        kd_workers,
                        chunk_rows,
                        dists_dir,
                        pos_dir,
                    ): tp for tp in batch
                }
                for fut in as_completed(futs):
                    try:
                        res.append(fut.result())
                    except Exception:
                        pass
                    try:
                        prog_metrics.update(task_metrics, advance=1)
                    except Exception:
                        pass
            # free memory between batches
            try:
                free_unused_memory()
            except Exception:
                pass
    n_total = sum(n for _t,n,_p in res)
    pos_total = sum(p for _t,_n,p in res)
    if n_total <= 0:
        print("No unlabeled pixels found for global lists update.")
        return

    # 3) global rank for distances via k-way merge (Highscore only)
    dist_files = _list_csvs(dists_dir, suffix='_dists.csv') if hs_on else []
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
    use_sidecar_global = bool(getattr(cfg, 'BINARY_SIDECARS_ENABLED', False))
    for dp in dist_files:
        if use_sidecar_global:
            npy = dp[:-7] + '.npy' if dp.endswith('.csv.gz') else dp[:-4] + '.npy'
        else:
            npy = None
        if npy and os.path.exists(npy):
            try:
                arr = np.load(npy, mmap_mode='r')
                # seed first row
                if arr.shape[0] > 0:
                    dist = float(arr[0, 2]); rr = int(arr[0, 0]); cc = int(arr[0, 1])
                    readers.append((dp, ('npy', arr, 1)))
                    base = os.path.basename(dp)
                    if base.endswith('.csv.gz'):
                        core = base[:-7]
                    elif base.endswith('.csv'):
                        core = base[:-4]
                    else:
                        core = os.path.splitext(base)[0]
                    if core.endswith('_dists'):
                        core = core[:-6]
                    tile_name = core + '.tif'
                    heapq.heappush(heap, (dist, tile_name, rr, cc, len(readers)-1))
                    continue
            except Exception:
                pass
        # Fallback to CSV reader
        f = _open_text_auto(dp, 'rt'); r = csv.reader(f); next(r, None)
        readers.append((dp, ('csv', f, r)))
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
    if hs_on and dist_files:
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
                dp, info = readers[idx]
                typ = info[0]
                if typ == 'npy':
                    arr, pos = info[1], info[2]
                    if pos < arr.shape[0]:
                        dist2 = float(arr[pos, 2]); rr2 = int(arr[pos, 0]); cc2 = int(arr[pos, 1])
                        readers[idx] = (dp, ('npy', arr, pos+1))
                        heapq.heappush(heap, (dist2, tile, rr2, cc2, idx))
                else:
                    f, r = info[1], info[2]
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
    for _, info in readers:
        if info[0] == 'csv':
            f = info[1]
            try: f.close()
            except Exception: pass
    for f in rank_files.values():
        try: f.close()
        except Exception: pass

    # 4) per-tile score and sort (Highscore only)
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
        # If no temp shard exists (we used _tile_preds directly), read from there
        if not os.path.exists(sp):
            alt = os.path.join(round_dir, '_tile_preds', tile_base + '.csv')
            if os.path.exists(alt):
                sp = alt
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
            flds = [x.strip().lower() for x in (rd.fieldnames or [])]
            has_shard = all(k in flds for k in ['row','col','lat','lon','prob'])
            has_tilepred = all(k in flds for k in ['row_idx','col_idx','center_lat','center_lon','predicted_prob'])
            for r in rd:
                try:
                    if has_shard:
                        rr = int(r['row']); cc = int(r['col'])
                        la = float(r['lat']); lo = float(r['lon'])
                        pr = float(r['prob']); nd = float(r.get('ndvi') or 0.0)
                    elif has_tilepred:
                        rr = int(r['row_idx']); cc = int(r['col_idx'])
                        la = float(r['center_lat']); lo = float(r['center_lon'])
                        pr = float(r['predicted_prob']); nd = float(r.get('ndvi') or 0.0)
                    else:
                        continue
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
        use_sidecar3 = bool(getattr(cfg, 'BINARY_SIDECARS_ENABLED', False))
        if use_sidecar3 and rows:
            try:
                import numpy as _np
                # Pre-assemble sidecar array (row,col,lat,lon,prob,ndvi,score)
                arr = _np.zeros((len(rows), 7), dtype=_np.float32)
                for i, (sc, tpl) in enumerate(rows):
                    _, rr, cc, la, lo, pr, nd = tpl
                    arr[i, :] = [float(rr), float(cc), float(la), float(lo), float(pr), float(nd), float(sc)]
                npy_scored = os.path.join(scored_dir, tile_base + '_scored.npy')
                _np.save(npy_scored, arr)
            except Exception:
                pass
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
    scored_files = []
    if hs_on:
        # progress for per-tile scoring
        _ctx_score = _npb()
        prog_score = _ctx_score.__enter__()
        task_score = prog_score.add_task("Per-tile scoring", total=len(tile_bases))
        # Parallelize scoring across tiles using processes
        max_workers = max(2, int(getattr(cfg, 'REFRESH_TILE_THREADS', 3)))
        wu = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("uncertainty", 0.5)
        wr = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("representativeness", 0.3)
        wc = cfg.HIGHSCORE_COMPONENT_WEIGHTS.get("consistency", 0.2)
        unc_delta = float(getattr(cfg, 'UNCERTAINTY_BAND_DELTA', 0.05))
        pool_batch = int(getattr(cfg, 'REFRESH_PROCESS_POOL_BATCH', 20))
        for i0 in range(0, len(tile_bases), max(1, pool_batch)):
            batch = tile_bases[i0:i0 + max(1, pool_batch)]
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = [
                    ex.submit(
                        _per_tile_score_worker,
                        tb,
                        ranks_dir,
                        shards_dir,
                        round_dir,
                        scored_dir,
                        float(cfg.MIN_AGRI_PROB),
                        float(wu), float(wr), float(wc),
                        unc_delta,
                    ) for tb in batch
                ]
                for fut in as_completed(futs):
                    try:
                        pth = fut.result()
                        if pth:
                            scored_files.append(pth)
                    except Exception:
                        pass
                    prog_score.update(task_score, advance=1)
            try:
                free_unused_memory()
            except Exception:
                pass
        try:
            _ctx_score.__exit__(None, None, None)
        except Exception:
            pass

    # 5) k-way merge scored files by score desc to global highscore.csv
    def kmerge_desc(files, out_path, total=None):
        import heapq
        use_sidecar = bool(getattr(cfg, 'BINARY_SIDECARS_ENABLED', False))
        hs = []
        sources = []  # ('npy', arr, pos, tile) or ('csv', file, reader)
        for fp in files:
            base = os.path.basename(fp)
            if base.endswith('.csv.gz'):
                npy = fp[:-7] + '.npy'
                core = base[:-7]
            elif base.endswith('.csv'):
                npy = fp[:-4] + '.npy'
                core = base[:-4]
            else:
                npy = os.path.splitext(fp)[0] + '.npy'
                core = os.path.splitext(base)[0]
            # derive tile name from core by removing trailing suffixes
            core2 = core.replace('_scored', '')
            tile_name = core2 + '.tif'
            if use_sidecar and os.path.exists(npy):
                try:
                    arr = np.load(npy, mmap_mode='r')
                    if arr.shape[0] > 0:
                        rr, cc, la, lo, pr, nd, sc = [arr[0, i] for i in range(7)]
                        sources.append(('npy', arr, 1, tile_name))
                        heapq.heappush(hs, (-float(sc), (tile_name, int(rr), int(cc), float(la), float(lo), float(pr), float(nd), float(sc)), len(sources)-1))
                        continue
                except Exception:
                    pass
            # fallback to CSV
            f = _open_text_auto(fp, 'rt'); r = csv.reader(f); next(r, None)
            sources.append(('csv', f, r))
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
                heapq.heappush(hs, (-sc, row, len(sources)-1))
                break
        from progress_utils import new_progress as _npb2
        with open(out_path,'w',newline='') as out, _npb2() as _prog2:
            w = csv.writer(out); w.writerow(['tile','row','col','lat','lon','prob','ndvi','score'])
            t_hs = _prog2.add_task("Global highscore merge", total=total or 0)
            count = 0
            while hs:
                neg, row, idx = heapq.heappop(hs)
                if isinstance(row, (list, tuple)) and isinstance(row[0], str) and len(row) == 8:
                    # row from npy path already assembled
                    w.writerow(row)
                else:
                    w.writerow(row)
                count += 1
                if total and count % 50000 == 0:
                    _prog2.update(t_hs, completed=count)
                src = sources[idx]
                if src[0] == 'npy':
                    arr, pos, tile_name = src[1], src[2], src[3]
                    if pos < arr.shape[0]:
                        rr, cc, la, lo, pr, nd, sc2 = [arr[pos, i] for i in range(7)]
                        sources[idx] = ('npy', arr, pos+1, tile_name)
                        heapq.heappush(hs, (-float(sc2), (tile_name, int(rr), int(cc), float(la), float(lo), float(pr), float(nd), float(sc2)), idx))
                else:
                    f, r = src[1], src[2]
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
        # close csv files
        for src in sources:
            if src[0] == 'csv':
                try: src[1].close()
                except Exception: pass

    if hs_on:
        hs_out = cfg.HIGHSCORE_FILE
        os.makedirs(os.path.dirname(hs_out), exist_ok=True)
        # Two-level parallel merge for highscore
        group_max = max(2, int(getattr(cfg, 'REFRESH_TILE_THREADS', 3)))
        if len(scored_files) > group_max:
            tmp_merge_dir = os.path.join(scored_dir, '_merge_tmp')
            os.makedirs(tmp_merge_dir, exist_ok=True)
            groups = [[] for _ in range(group_max)]
            for idx, fp in enumerate(scored_files):
                groups[idx % group_max].append(fp)
            inter_files = []
            with ProcessPoolExecutor(max_workers=group_max) as ex:
                futs = {}
                for gi, g in enumerate(groups):
                    if not g:
                        continue
                    outg = os.path.join(tmp_merge_dir, f'group_{gi}_scored.csv')
                    futs[ex.submit(_merge_scored_files, g, outg, None)] = outg
                for fut in as_completed(futs):
                    try:
                        inter_files.append(futs[fut])
                    except Exception:
                        pass
            _merge_scored_files(inter_files, hs_out, total=n_total)
            for p in inter_files:
                try: os.remove(p)
                except Exception: pass
        else:
            kmerge_desc(scored_files, hs_out, total=n_total)
        # global KML limited (stream only top-K rows to avoid large RAM)
        try:
            try:
                topk = int(getattr(cfg, 'HIGHSCORE_KML_TOP_PIXELS', 50000))
            except Exception:
                topk = 50000
            rows_small = []
            with open(hs_out) as f:
                rd = csv.DictReader(f)
                if topk and topk > 0:
                    for i, r in enumerate(rd):
                        rows_small.append(r)
                        if len(rows_small) >= topk:
                            break
                else:
                    # no cap; still stream to avoid list(rd)
                    for r in rd:
                        rows_small.append(r)
            _write_ranked_pixel_kml(rows_small, cfg.HIGHSCORE_KML_GLOBAL, weight_key='score', top_k=topk)
        except Exception as e:
            print(f"Highscore KML failed: {e}")

    # 6) k-way merge positive shards by prob desc to global probableAgri.csv
    pos_files = _list_csvs(pos_dir, suffix='_pos.csv')
    def kmerge_pos(files, out_path, total=None):
        import heapq
        use_sidecar = bool(getattr(cfg, 'BINARY_SIDECARS_ENABLED', False))
        hs = []
        sources = []  # ('npy', arr, pos, tile) or ('csv', file, reader)
        for fp in files:
            base = os.path.basename(fp)
            if base.endswith('.csv.gz'):
                npy = fp[:-7] + '.npy'
                core = base[:-7]
            elif base.endswith('.csv'):
                npy = fp[:-4] + '.npy'
                core = base[:-4]
            else:
                npy = os.path.splitext(fp)[0] + '.npy'
                core = os.path.splitext(base)[0]
            core2 = core.replace('_pos', '')
            tile_name = core2 + '.tif'
            if use_sidecar and os.path.exists(npy):
                try:
                    arr = np.load(npy, mmap_mode='r')
                    if arr.shape[0] > 0:
                        rr, cc = int(arr[0,0]), int(arr[0,1])
                        la, lo = float(arr[0,2]), float(arr[0,3])
                        pr, nd = float(arr[0,4]), float(arr[0,5])
                        sources.append(('npy', arr, 1, tile_name))
                        heapq.heappush(hs, (-pr, (tile_name, rr, cc, la, lo, pr, nd), len(sources)-1))
                        continue
                except Exception:
                    pass
            f = _open_text_auto(fp, 'rt'); r = csv.reader(f); next(r, None)
            sources.append(('csv', f, r))
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
                heapq.heappush(hs, (-pr, row, len(sources)-1))
                break
        from progress_utils import new_progress as _npb3
        with open(out_path,'w',newline='') as out, _npb3() as _prog3:
            w = csv.writer(out); w.writerow(['tile','row','col','lat','lon','prob','ndvi'])
            t_pa = _prog3.add_task("Global probableAgri merge", total=total or 0)
            count = 0
            while hs:
                neg, row, idx = heapq.heappop(hs)
                if isinstance(row, (list, tuple)) and isinstance(row[0], str):
                    w.writerow(row)
                else:
                    w.writerow(row)
                count += 1
                if total and count % 50000 == 0:
                    _prog3.update(t_pa, completed=count)
                src = sources[idx]
                if src[0] == 'npy':
                    arr, pos, tile_name = src[1], src[2], src[3]
                    if pos < arr.shape[0]:
                        rr, cc = int(arr[pos,0]), int(arr[pos,1])
                        la, lo = float(arr[pos,2]), float(arr[pos,3])
                        pr2, nd2 = float(arr[pos,4]), float(arr[pos,5])
                        sources[idx] = ('npy', arr, pos+1, tile_name)
                        heapq.heappush(hs, (-pr2, (tile_name, rr, cc, la, lo, pr2, nd2), idx))
                else:
                    f, r = src[1], src[2]
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
        for src in sources:
            if src[0] == 'csv':
                try: src[1].close()
                except Exception: pass

    if pa_on:
        pa_out = cfg.PROBABLE_AGRI_FILE
        # Two-level parallel merge for probable agri
        group_max = max(2, int(getattr(cfg, 'REFRESH_TILE_THREADS', 3)))
        if len(pos_files) > group_max:
            tmp_merge_dir = os.path.join(pos_dir, '_merge_tmp')
            os.makedirs(tmp_merge_dir, exist_ok=True)
            groups = [[] for _ in range(group_max)]
            for idx, fp in enumerate(pos_files):
                groups[idx % group_max].append(fp)
            inter_files = []
            with ProcessPoolExecutor(max_workers=group_max) as ex:
                futs = {}
                for gi, g in enumerate(groups):
                    if not g:
                        continue
                    outg = os.path.join(tmp_merge_dir, f'group_{gi}_pos.csv')
                    futs[ex.submit(_merge_pos_files, g, outg, None)] = outg
                for fut in as_completed(futs):
                    try:
                        inter_files.append(futs[fut])
                    except Exception:
                        pass
            _merge_pos_files(inter_files, pa_out, total=pos_total)
            for p in inter_files:
                try: os.remove(p)
                except Exception: pass
        else:
            kmerge_pos(pos_files, pa_out, total=pos_total)
        try:
            try:
                topk_pa = int(getattr(cfg, 'PROBABLE_AGRI_KML_TOP_PIXELS', 50000))
            except Exception:
                topk_pa = 50000
            rows_small = []
            with open(pa_out) as f:
                rd = csv.DictReader(f)
                if topk_pa and topk_pa > 0:
                    for i, r in enumerate(rd):
                        rows_small.append(r)
                        if len(rows_small) >= topk_pa:
                            break
                else:
                    for r in rd:
                        rows_small.append(r)
            _write_ranked_pixel_kml(rows_small, cfg.PROBABLE_AGRI_KML_GLOBAL, weight_key='prob', top_k=topk_pa)
        except Exception as e:
            print(f"ProbableAgri KML failed: {e}")

    # Final cleanup: remove heavy per-round caches no longer needed after refresh
    # - `_global_refresh/` holds temporary shards/metrics for the refresh process
    # - `_tile_preds/` holds per-tile prediction shards (used as a speed-up). After
    #   lists/KML are generated, these can be safely removed to save disk space;
    #   future refreshes will fall back to re-splitting `predictions.csv` if needed.
    try:
        import shutil as _shutil
        for sub in ['_global_refresh', '_tile_preds']:
            p = os.path.join(round_dir, sub)
            if os.path.isdir(p):
                _shutil.rmtree(p, ignore_errors=True)
                print(f"Removed cache folder => {p}")
    except Exception as _e:
        print(f"Round cache cleanup warning: {_e}")

def candidate_selection_from_csv(pred_csv, round_dir, round_num, train_rows=None, X_train=None, y_train=None):
    """Load predictions from CSV and prompt the user to label candidates."""
    if not os.path.exists(pred_csv):
        print(f"Missing predictions CSV => {pred_csv}")
        return None
    # Stream to build a top-M uncertainty pool instead of loading everything
    print("Building uncertainty/negative pools for candidate selection...")
    import heapq
    skipped = load_skipped_set()
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
    import numpy as _np
    CHUNK = int(getattr(cfg, 'PREDICTIONS_CSV_CHUNK_ROWS', 500_000))
    with open(pred_csv, "r", encoding='utf-8', errors='replace') as pf, _npb() as _prog:
        task = _prog.add_task("Scan predictions.csv", total=file_size or None)
        processed = 0
        header = pf.readline()
        processed += len(header.encode('utf-8','replace'))
        if file_size:
            _prog.update(task, completed=processed)
        # Determine header indices
        hdr = next(csv.reader([header])) if header else []
        try:
            IDX_TILE = hdr.index('tile'); IDX_R = hdr.index('row_idx'); IDX_C = hdr.index('col_idx')
            IDX_LAT = hdr.index('center_lat'); IDX_LON = hdr.index('center_lon'); IDX_P = hdr.index('predicted_prob')
            IDX_ND = hdr.index('ndvi') if 'ndvi' in hdr else None
        except Exception:
            IDX_TILE, IDX_R, IDX_C, IDX_LAT, IDX_LON, IDX_P, IDX_ND = 0,1,2,3,4,5,6

        # Buffers for chunk
        buf_tile = []
        buf_r = []
        buf_c = []
        buf_la = []
        buf_lo = []
        buf_p = []
        buf_nd = []

        def flush_chunk():
            nonlocal buf_tile, buf_r, buf_c, buf_la, buf_lo, buf_p, buf_nd
            if not buf_p:
                return
            # Convert to arrays
            T = buf_tile
            R = _np.asarray(buf_r, dtype=_np.int32)
            Cc = _np.asarray(buf_c, dtype=_np.int32)
            La = _np.asarray(buf_la, dtype=_np.float32)
            Lo = _np.asarray(buf_lo, dtype=_np.float32)
            P = _np.asarray(buf_p, dtype=_np.float32)
            ND = _np.asarray(buf_nd, dtype=_np.float32) if buf_nd else _np.zeros_like(P)

            # Uncertainty pool: consider only p >= CANDIDATE_PROB_LOWER
            mask_unc = (P >= float(cfg.CANDIDATE_PROB_LOWER))
            if mask_unc.any():
                keys = -_np.abs(P[mask_unc] - 0.5)
                # Keep top-K from this chunk only (K = pool_size)
                k = int(pool_size)
                if k > 0 and keys.size > k:
                    idx_local = _np.argpartition(keys, -k)[-k:]
                else:
                    idx_local = _np.arange(keys.size)
                # Map back to original indices
                sel_idx = _np.flatnonzero(mask_unc)[idx_local]
                for j in sel_idx:
                    # skip globally skipped pixels
                    if skipped and f"{T[j]}:{int(R[j])}:{int(Cc[j])}" in skipped:
                        continue
                    item = [T[j], int(R[j]), int(Cc[j]), float(La[j]), float(Lo[j]), float(P[j]), float(ND[j])]
                    key = -abs(item[5] - 0.5)
                    if len(heap) < pool_size:
                        heapq.heappush(heap, (key, item))
                    else:
                        if key > heap[0][0]:
                            heapq.heapreplace(heap, (key, item))

            # Negative-like pool (prob window + NDVI range)
            if pool_neg > 0:
                if ndvi_rel:
                    # Collect NDVI values for percentile computation later (all rows in chunk)
                    ndvi_vals_for_pr.extend([float(x) for x in ND.tolist()])
                is_neg_prob = (P >= nlo) & (P < nhi)
                is_neg_ndvi = _np.ones_like(is_neg_prob, dtype=bool)
                if isinstance(ndvi_abs, (list, tuple)) and ndvi_abs[0] is not None and ndvi_abs[1] is not None:
                    is_neg_ndvi = (ND >= float(ndvi_abs[0])) & (ND <= float(ndvi_abs[1]))
                mask_neg = is_neg_prob & is_neg_ndvi
                if mask_neg.any():
                    # closer to threshold (higher p) preferred ⇒ key = p
                    keysn = P[mask_neg]
                    k2 = int(pool_neg)
                    if k2 > 0 and keysn.size > k2:
                        idx_local2 = _np.argpartition(keysn, -k2)[-k2:]
                    else:
                        idx_local2 = _np.arange(keysn.size)
                    sel_idx2 = _np.flatnonzero(mask_neg)[idx_local2]
                    for j in sel_idx2:
                        if skipped and f"{T[j]}:{int(R[j])}:{int(Cc[j])}" in skipped:
                            continue
                        item = [T[j], int(R[j]), int(Cc[j]), float(La[j]), float(Lo[j]), float(P[j]), float(ND[j])]
                        keyn = item[5]
                        if len(heap_neg) < pool_neg:
                            heapq.heappush(heap_neg, (keyn, item))
                        else:
                            if keyn > heap_neg[0][0]:
                                heapq.heapreplace(heap_neg, (keyn, item))

            # reset buffers
            buf_tile = []
            buf_r = []
            buf_c = []
            buf_la = []
            buf_lo = []
            buf_p = []
            buf_nd = []

        # Stream lines into chunk buffers
        count = 0
        for line in pf:
            processed += len(line.encode('utf-8','replace'))
            parts = line.rstrip('\n').split(',')
            if len(parts) <= max(IDX_TILE, IDX_R, IDX_C, IDX_LAT, IDX_LON, IDX_P):
                if file_size and processed % (1024*1024) < 1000:
                    _prog.update(task, completed=min(processed, file_size))
                continue
            try:
                t0 = parts[IDX_TILE]
                r0 = int(parts[IDX_R]); c0 = int(parts[IDX_C])
                la0 = float(parts[IDX_LAT]); lo0 = float(parts[IDX_LON])
                p0 = float(parts[IDX_P]) if parts[IDX_P] != '' else 0.5
                nd0 = float(parts[IDX_ND]) if (IDX_ND is not None and len(parts) > IDX_ND and parts[IDX_ND] != '') else 0.0
            except Exception:
                if file_size and processed % (1024*1024) < 1000:
                    _prog.update(task, completed=min(processed, file_size))
                continue
            # buffer
            buf_tile.append(t0); buf_r.append(r0); buf_c.append(c0); buf_la.append(la0); buf_lo.append(lo0); buf_p.append(p0); buf_nd.append(nd0)
            count += 1
            if count >= CHUNK:
                flush_chunk()
                count = 0
            if file_size and processed % (1024*1024) < 1000:
                _prog.update(task, completed=min(processed, file_size))
        # flush remainder
        flush_chunk()
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
    # Filter out globally skipped pixels
    skipped = load_skipped_set()
    if skipped:
        preds = [p for p in preds if f"{p[0]}:{int(p[1])}:{int(p[2])}" not in skipped]

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
    # If initial selection shorter than target, backfill pool by uncertainty
    if len(cands) < cfg.NUM_CANDIDATES_PER_ROUND:
        unc_all = [p for p in preds if p[5] >= cfg.CANDIDATE_PROB_LOWER]
        unc_all.sort(key=lambda r: abs(r[5] - 0.5))  # smallest margin (most uncertain) first
        # append until we have at least target candidates
        seen = set((e[0], e[1], e[2]) for e in cands)
        for e in unc_all:
            k = (e[0], e[1], e[2])
            if k in seen:
                continue
            cands.append(e)
            seen.add(k)
            if len(cands) >= cfg.NUM_CANDIDATES_PER_ROUND * 5:
                break

    print(f"{len(cands)} candidate patches preselected")

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
    # Build an extended stream: preselected list followed by extras by uncertainty
    asked = set()
    def _extra_stream():
        pool = [p for p in preds if p[5] >= cfg.CANDIDATE_PROB_LOWER]
        # sort by highest entropy first
        pool.sort(key=lambda e: (-e[5]*np.log(e[5]+1e-9) - (1-e[5])*np.log(1-e[5]+1e-9)))
        for e in pool:
            k = (e[0], e[1], e[2])
            if k in asked:
                continue
            yield e

    labeled_count = 0
    target = int(cfg.NUM_CANDIDATES_PER_ROUND)
    idx = 0
    extra_iter = None
    while labeled_count < target:
        if idx < len(cands):
            t, r, c, la, lo, p, ndvi = cands[idx]
            idx += 1
        else:
            if extra_iter is None:
                extra_iter = _extra_stream()
            try:
                t, r, c, la, lo, p, ndvi = next(extra_iter)
            except StopIteration:
                print("Exhausted candidate pool before reaching target labels.")
                break
        k = (t, r, c)
        if k in asked:
            continue
        asked.add(k)
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
            try:
                record_skipped_pixel(t, r, c, la, lo, source="al")
            except Exception:
                pass
            continue
        lab = "Agricultural" if ui == "1" else "Non-Agricultural" if ui == "2" else None
        if lab:
            note = prompt_note()
            eid = f"AL_{round_num}_{int(random.random()*1e6)}"
            cur_i = idx - 1
            in_initial = (cur_i < len(cands)) and (cur_i >= 0)
            cid_val = cids[cur_i] if in_initial and cids else ""
            csize_val = csize.get(cid_val, "") if in_initial else ""
            crank_val = crank.get(cur_i, "") if in_initial else ""
            with open(tmp, "a", newline="") as f2:
                csv.writer(f2).writerow([
                    eid, la, lo, t, lab, note,
                    "entropy_dbscan_tile_balanced",
                    f"{p:.6f}", f"{ent:.6f}", f"{abs(p-0.5):.6f}", cid_val, csize_val,
                    crank_val, tile_pick_counter[t], kcount, f"{dist_th:.6f}", f"{ndvi:.6f}", nld if nld != "" else "", nlc if nlc != "" else "", reason
                ])
            print("Label saved.")
            labeled_count += 1
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
