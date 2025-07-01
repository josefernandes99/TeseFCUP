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
import time
import datetime

import numpy as np
import rasterio
from rasterio.features import shapes
import torch
import torch.nn as nn
from joblib import dump
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

from config import (
    RAW_DATA_DIR,
    ROUNDS_DIR,
    TEMP_LABELS_FILE,
    NUM_CANDIDATES_PER_ROUND,
    CANDIDATE_PROB_LOWER,
    CANDIDATE_PROB_UPPER,
    MIN_AGRI_PROB,
    SVM_PARAMS,
    RF_PARAMS,
    RESNET_EPOCHS,
    RESNET_LR,
    BATCH_SIZE,
    INDICES,   # ["NDVI","EVI","EVI2"]
)


# -----------------------------------------------------------------------------
# 2) Feature‐extraction helper (unchanged—reads all bands including indices)
# -----------------------------------------------------------------------------
def extract_features_from_label(row):
    lat, lon = float(row["lat"]), float(row["lon"])
    tif_path = os.path.join(RAW_DATA_DIR, row["tile"])
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Tile file not found: {tif_path}")
    with rasterio.open(tif_path) as src:
        for vals in src.sample([(lon, lat)]):
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

    # (optional debug)
    print("Feature statistics:")
    for i, name in enumerate(INDICES + []):  # if you want to label features, extend with other band names
        if i < 5:  # only print first few for brevity
            print(f"  [{i}] min={feat_min[i]:.3f}, max={feat_max[i]:.3f}, mean={feat_mean[i]:.3f}, var={feat_var[i]:.3f}")
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

    else:
        raise ValueError(f"Unknown model choice: {choice}")


# -----------------------------------------------------------------------------
# 4) Fast batch inference + per‐pixel geometry
# -----------------------------------------------------------------------------
def get_pixel_corners(src, r, c):
    tl = src.xy(r,   c)
    tr = src.xy(r,   c+1)
    br = src.xy(r+1, c+1)
    bl = src.xy(r+1, c)
    return [[tl[0], tl[1]], [tr[0], tr[1]], [br[0], br[1]], [bl[0], bl[1]], [tl[0], tl[1]]]


def predict_entire_tile(tile_path, model):
    tile_name = os.path.basename(tile_path)
    with rasterio.open(tile_path) as src:
        arr   = src.read().astype(np.float32)       # (bands, H, W)
        b, H, W = arr.shape
        X     = arr.reshape(b, -1).T                # (H*W, bands)
        probs = model.predict_proba(X)[:, 1]        # bulk proba
        rows  = np.repeat(np.arange(H), W)
        cols  = np.tile(  np.arange(W), H)

        results = []
        for idx, (r, c) in enumerate(zip(rows, cols)):
            p = float(probs[idx])
            corners = get_pixel_corners(src, r, c)
            mean_lat = sum(y for x,y in corners) / len(corners)
            mean_lon = sum(x for x,y in corners) / len(corners)
            results.append([tile_name, r, c, mean_lat, mean_lon, p])
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
    kml_path = os.path.join(round_folder, f"agricultural_patches_round_{round_num}.kml")
    tifs     = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    total_polys = 0

    with open(kml_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n')
        f.write('    <Style id="agriStyle">\n')
        f.write('      <LineStyle><color>ff0000ff</color><width>1</width></LineStyle>\n')
        f.write('      <PolyStyle><color>400000ff</color><outline>1</outline></PolyStyle>\n')
        f.write('    </Style>\n')

        for tp in tifs:
            with rasterio.open(tp) as src:
                arr   = src.read().astype(np.float32)
                b, H, W = arr.shape
                X     = arr.reshape(b, -1).T
                probs = model.predict_proba(X)[:,1].reshape(H, W)
                mask  = (probs >= MIN_AGRI_PROB).astype("uint8")

                for geom, val in shapes(mask, mask=mask, transform=src.transform):
                    if val != 1: continue
                    coord_str = " ".join(f"{x},{y},0" for x,y in geom["coordinates"][0])
                    f.write("    <Placemark>\n")
                    f.write("      <styleUrl>#agriStyle</styleUrl>\n")
                    f.write("      <Polygon>\n")
                    f.write("        <outerBoundaryIs>\n")
                    f.write("          <LinearRing>\n")
                    f.write(f"            <coordinates>{coord_str}</coordinates>\n")
                    f.write("          </LinearRing>\n")
                    f.write("        </outerBoundaryIs>\n")
                    f.write("      </Polygon>\n")
                    f.write("    </Placemark>\n")
                    total_polys += 1

        f.write("  </Document>\n</kml>\n")

    if total_polys == 0:
        print(f"⚠️  No polygons (all probs < {MIN_AGRI_PROB})")
    else:
        print(f"{total_polys} agricultural polygons saved to {kml_path}")


# -----------------------------------------------------------------------------
# 6) Candidate‐patch KML (unchanged)
# -----------------------------------------------------------------------------
def generate_candidate_kml(tile_name, row_idx, col_idx, outpath):
    tif_path = os.path.join(RAW_DATA_DIR, tile_name)
    if not os.path.exists(tif_path):
        print(f"WARNING: missing tile {tile_name}; skipping candidate KML.")
        return
    with rasterio.open(tif_path) as src:
        corners = get_pixel_corners(src, row_idx, col_idx)

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
    print(f"Candidate KML => {outpath}")


# -----------------------------------------------------------------------------
# 7) Active‐Learning orchestration (unchanged aside from new train_model)
# -----------------------------------------------------------------------------
def active_learning_round(round_num, labels_file, model_choice):
    print(f"\n=== Starting Active Learning Round {round_num} ===")
    rnd_dir = os.path.join(ROUNDS_DIR, f"round_{round_num}")
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

    # inference + timing
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    preds = []
    start = time.time()
    with Progress("[bold cyan]{task.description}", BarColumn(), TaskProgressColumn(),
                  TimeElapsedColumn(), TimeRemainingColumn()) as prog:
        task = prog.add_task("Running inference", total=len(tifs))
        for tp in tifs:
            preds.extend(predict_entire_tile(tp, model))
            prog.update(task, advance=1)
    print(f"Total pixels inferred: {len(preds)}")
    print(f"Inference completed in {str(datetime.timedelta(seconds=int(time.time()-start)))}")

    # outputs
    save_predictions(rnd_dir, preds)
    save_agricultural_polygons_kml(rnd_dir, model, round_num)

    # candidate selection
    unc = [p for p in preds if CANDIDATE_PROB_LOWER <= p[5] <= CANDIDATE_PROB_UPPER]
    if len(unc) < NUM_CANDIDATES_PER_ROUND:
        preds.sort(key=lambda r: abs(r[5]-0.5))
        cands = preds[:NUM_CANDIDATES_PER_ROUND]
    else:
        cands = unc[:NUM_CANDIDATES_PER_ROUND]
    print(f"{len(cands)} candidate patches selected")

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
        print(f"Candidate: {t} r={r},c={c}, p={p:.3f}")
        ui = input("Label? (1=Agri,2=NonAgri,3=Skip): ").strip()
        if ui=="3":
            print("Skipped.")
            continue
        lab = "Agricultural" if ui=="1" else "Non-Agricultural" if ui=="2" else None
        if lab:
            eid = f"AL_{round_num}_{int(random.random()*1e6)}"
            with open(tmp, "a", newline="") as f2:
                csv.writer(f2).writerow([eid, t, la, lo, lab])
            print("Label saved.")
    print(f"Round {round_num} complete; labels at {tmp}")
    return tmp


if __name__ == "__main__":
    active_learning_round(1, TEMP_LABELS_FILE, "SVM")
