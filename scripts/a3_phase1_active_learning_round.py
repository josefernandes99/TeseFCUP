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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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

from a2_phase1_initial_labeling import generate_grids_for_all_tiles


# -----------------------------------------------------------------------------
# 2) Feature‐extraction helper (unchanged—reads all bands including indices)
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

def extract_features_from_label(row):
    """Read pixel values at the label coordinate.

    Sentinel-2 tiles are typically stored in a projected UTM coordinate
    reference system while label coordinates are assumed to be WGS84
    (latitude/longitude).  The function therefore converts from WGS84 to the
    tile CRS prior to sampling.  If tiles have been reprojected to WGS84 during
    download (user provided flag), the conversion becomes a no-op.
    """
    lat, lon = float(row["lat"]), float(row["lon"])
    tif_path = os.path.join(RAW_DATA_DIR, row["tile"])
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Tile file not found: {tif_path}")
    with rasterio.open(tif_path) as src:
        x, y = lon, lat
        if src.crs and not src.crs.is_geographic:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
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

    c = choice.lower()
    if c == "svm":
        # StandardScaler for true zero-mean/unit-variance
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        base_params = {k: v for k, v in cfg.SVM_PARAMS.items() if k not in ("C", "gamma")}
        params = cfg.SVM_PARAMS.copy()
        clf = SVC(probability=True, **params)
        clf.fit(Xs, y)
        return SklearnWrapper(clf, feat_means, feat_std, scaler)

    elif c == "randomforest":
        rf = RandomForestClassifier(n_jobs=-1, **cfg.RF_PARAMS)
        rf.fit(X, y)
        # we will z‐score scale at inference for consistency
        return SklearnWrapper(rf, feat_means, feat_std, scaler=None)

    elif c == "resnet":
        net = TabularResNet(input_dim=X.shape[1])
        train_resnet(net, torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.int64)))
        scripted = torch.jit.script(net)
        return PytorchResNetWrapper(scripted, feat_means, feat_std)

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
    print(f"Inferring ⇒ {tile_name}")
    with rasterio.open(tile_path) as src:
        arr   = src.read().astype(np.float32)       # (bands, H, W)
        b, H, W = arr.shape
        X     = arr.reshape(b, -1).T                # (H*W, bands)
        probs = model.predict_proba(X)[:, 1]        # bulk proba
        rows = np.repeat(np.arange(H, dtype=np.int32), W)
        cols = np.tile(np.arange(W, dtype=np.int32), H)

        # vectorized center coordinate computation
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        if src.crs and not src.crs.is_geographic:
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            xs, ys = transformer.transform(xs, ys)

        results = [
            [tile_name, int(r), int(c), float(lat), float(lon), float(p)]
            for r, c, lat, lon, p in zip(rows, cols, ys, xs, probs)
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
        w.writerow(["tile","row_idx","col_idx","center_lat","center_lon","predicted_prob"])
        w.writerows(preds)
    print(f"Predictions written to {path}")


def save_agricultural_polygons_kml(round_folder, model, round_num):
    """Export polygons after thresholding and sieving predictions."""
    kml_path = os.path.join(round_folder, f"agricultural_patches_round_{round_num}.kml")
    tifs     = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    polys = []

    for tp in tifs:
        with rasterio.open(tp) as src:
            arr = src.read().astype(np.float32)
            b, H, W = arr.shape
            X = arr.reshape(b, -1).T
            probs = model.predict_proba(X)[:, 1].reshape(H, W)
            crop = probs >= cfg.MIN_AGRI_PROB
            uncertain = (probs >= cfg.CANDIDATE_PROB_LOWER) & (probs < cfg.MIN_AGRI_PROB)
            mask = crop | uncertain
            from scipy.ndimage import binary_closing, binary_fill_holes, label as ndlabel
            mask = binary_fill_holes(binary_closing(mask))
            lbl, num = ndlabel(mask)
            for i in range(1, num+1):
                comp = (lbl==i)
                if not np.any(crop[comp]):
                    mask[comp] = False
            if cfg.SIEVE_MIN_SIZE > 0:
                mask = sieve(mask.astype("uint8"), size=cfg.SIEVE_MIN_SIZE, connectivity=8).astype(bool)
            transformer = None
            if src.crs and not src.crs.is_geographic:
                transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

            for geom, val in shapes(mask.astype("uint8"), mask=mask, transform=src.transform):
                if val != 1:
                    continue
                poly = shape(geom)
                if transformer:
                    poly = shp_transform(transformer.transform, poly)
                polys.append(poly)

    if not polys:
        print(f"⚠️  No polygons (all probs < {cfg.MIN_AGRI_PROB})")
        return

    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        poly_list = [merged]
    elif isinstance(merged, MultiPolygon):
        poly_list = list(merged.geoms)
    else:
        poly_list = []

    doc = Element('Document')
    style = SubElement(doc, 'Style', id='agriStyle')
    ln = SubElement(style, 'LineStyle'); SubElement(ln,'color').text='ff0000ff'; SubElement(ln,'width').text='1'
    ps = SubElement(style, 'PolyStyle'); SubElement(ps,'color').text='400000ff'; SubElement(ps,'outline').text='1'

    total_polys = 0
    for p in poly_list:
        coords = list(p.exterior.coords)
        coord_str = " ".join(f"{lon},{lat},0" for lon, lat in coords)
        pm = SubElement(doc, 'Placemark')
        SubElement(pm, 'styleUrl').text = '#agriStyle'
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

    doc = Element('Document')
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
    """
    print(f"\n=== Starting Active Learning Round {round_num} ===")
    rnd_dir = out_dir or os.path.join(ROUNDS_DIR, f"round_{round_num}")
    os.makedirs(rnd_dir, exist_ok=True)
    generate_grids_for_all_tiles()

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
    del X, y
    free_unused_memory()

    # inference + timing
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    preds = []
    start = time.time()

    def run_tile(tp):
        tile_preds = predict_entire_tile(tp, model)
        prog.update(task, advance=1)
        return tile_preds

    with Progress(
        "[bold cyan]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as prog:
        task = prog.add_task("Running inference", total=len(tifs))

        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(run_tile)(tp) for tp in tifs
        )

    for tile_preds in results:
        preds.extend(tile_preds)
    print(f"Total pixels inferred: {len(preds)}")
    print(f"Inference completed in {str(datetime.timedelta(seconds=int(time.time() - start)))}")

    # outputs
    if save_preds:
        save_predictions(rnd_dir, preds)
    else:
        print("Skipping predictions.csv generation.")
    save_agricultural_polygons_kml(rnd_dir, model, round_num)

    if not request_labels or not save_preds:
        print(f"Round {round_num} complete (no candidate labeling).")
        return None

    tmp = candidate_selection_from_csv(os.path.join(rnd_dir, "predictions.csv"), rnd_dir, round_num)
    print(f"Round {round_num} complete; labels at {tmp}")
    return tmp


def candidate_selection_from_csv(pred_csv, round_dir, round_num):
    """Load predictions from CSV and prompt the user to label candidates."""
    if not os.path.exists(pred_csv):
        print(f"Missing predictions CSV => {pred_csv}")
        return None
    preds = []
    with open(pred_csv, "r") as pf:
        rd = csv.reader(pf)
        next(rd, None)
        for row in rd:
            preds.append([row[0], int(row[1]), int(row[2]), float(row[3]), float(row[4]), float(row[5])])

    def select_candidates_entropy(predictions):
        import numpy as np
        from sklearn.cluster import DBSCAN

        probs = np.array([p[5] for p in predictions])
        lats = np.array([p[3] for p in predictions])
        lons = np.array([p[4] for p in predictions])
        entropy = -probs*np.log(probs + 1e-9) - (1 - probs)*np.log(1 - probs + 1e-9)
        margin = np.abs(probs - 0.5)
        score = entropy - margin
        pool_idx = np.argsort(score)[::-1][:cfg.NUM_CANDIDATES_PER_ROUND * 5]
        if pool_idx.size == 0:
            return []
        coords = np.vstack([lats[pool_idx], lons[pool_idx]]).T
        clustering = DBSCAN(eps=0.01, min_samples=1).fit(coords)
        cands = []
        for lbl in set(clustering.labels_):
            idxs = pool_idx[clustering.labels_ == lbl]
            idxs = idxs[np.argsort(score[idxs])[::-1]]
            for i in range(min(2, len(idxs))):
                cands.append(predictions[int(idxs[i])])
                if len(cands) >= cfg.NUM_CANDIDATES_PER_ROUND:
                    break
            if len(cands) >= cfg.NUM_CANDIDATES_PER_ROUND:
                break
        return cands

    cands = select_candidates_entropy(preds)
    if len(cands) < cfg.NUM_CANDIDATES_PER_ROUND:
        unc = [p for p in preds if cfg.CANDIDATE_PROB_LOWER <= p[5] <= cfg.CANDIDATE_PROB_UPPER]
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

    tmp = os.path.join(round_dir, "temp_labels.csv")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "lat", "lon", "tile", "label", "notes"])
    for idx, (t, r, c, la, lo, p) in enumerate(cands):
        kmlp = os.path.join(round_dir, f"candidate_{idx}.kml")
        generate_candidate_kml(t, r, c, kmlp)
        print(f"Candidate: {t} r={r},c={c}, p={p:.3f}")
        ui = input("Label? (1=Agri,2=NonAgri,3=Skip): ").strip()
        if ui == "3":
            print("Skipped.")
            continue
        lab = "Agricultural" if ui == "1" else "Non-Agricultural" if ui == "2" else None
        if lab:
            note = prompt_note()
            eid = f"AL_{round_num}_{int(random.random()*1e6)}"
            with open(tmp, "a", newline="") as f2:
                csv.writer(f2).writerow([eid, la, lo, t, lab, note])
            print("Label saved.")
    return tmp


if __name__ == "__main__":
    active_learning_round(1, TEMP_LABELS_FILE, "SVM")
