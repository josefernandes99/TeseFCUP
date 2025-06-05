# scripts/a2_phase1_initial_labeling.py

import csv
import glob
import math
import os
import random
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
import rasterio
from rasterio.windows import Window

from config import (
    LABELS_FILE, MIN_AGRI_COUNT, MIN_AGRI_RATIO, MAX_AGRI_RATIO,
    DUPLICATE_TOLERANCE, RAW_DATA_DIR, CANDIDATE_KML,
    TEMP_LABELS_FILE, ROUNDS_DIR,
    NUM_RANDOM_PICKS_PER_TILE,
    CANDIDATE_PROB_LOWER, CANDIDATE_PROB_UPPER,
    NUM_CANDIDATES_PER_ROUND
)

# Path to your normalizers.csv produced in a1_phase1_data_download.py
NORMALIZERS_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "normalizers.csv")
# Where we'll write the hybrid-candidates list
CANDIDATES_CSV  = os.path.join(os.path.dirname(LABELS_FILE), "candidate_labels.csv")


def ensure_labels_file():
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    if not os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["id", "lat", "lon", "tile", "label"])
        print("Created new master labels CSV.")


def load_labels(path=LABELS_FILE):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def check_label_requirements():
    labels = load_labels()
    total = len(labels)
    agri  = sum(1 for r in labels if r["label"].lower() == "agricultural")
    ratio = agri / total if total else 0.0
    ok    = (agri >= MIN_AGRI_COUNT) and (MIN_AGRI_RATIO <= ratio <= MAX_AGRI_RATIO)
    return ok, agri, ratio


def duplicate_exists(lat, lon, labels):
    for r in labels:
        try:
            if math.hypot(float(r["lat"]) - lat, float(r["lon"]) - lon) < DUPLICATE_TOLERANCE:
                return True
        except:
            continue
    return False


def get_tile_for_coordinate(_lat, _lon):
    folders = glob.glob(os.path.join(RAW_DATA_DIR, "*"))
    return os.path.basename(random.choice(folders)) if folders else "unknown"


def get_patch_dimensions():
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*", "*.tif"))
    if not tifs:
        return 0.001, 0.001
    with rasterio.open(tifs[0]) as src:
        w = abs(src.transform[0])
        h = abs(src.transform[4])
        return 3 * w, 3 * h


def generate_kml_for_patch(center_lat, center_lon, patch_width, patch_height, out_path=None):
    half_w, half_h = patch_width / 2, patch_height / 2
    coords = [
        [center_lon-half_w, center_lat-half_h],
        [center_lon-half_w, center_lat+half_h],
        [center_lon+half_w, center_lat+half_h],
        [center_lon+half_w, center_lat-half_h],
        [center_lon-half_w, center_lat-half_h]
    ]
    kml = Element('kml'); kml.set("xmlns","http://www.opengis.net/kml/2.2")
    doc = SubElement(kml, 'Document')
    style = SubElement(doc, 'Style', id="patchStyle")
    ln = SubElement(style, 'LineStyle'); SubElement(ln,'color').text="ff0000ff"; SubElement(ln,'width').text="2"
    ps = SubElement(style, 'PolyStyle'); SubElement(ps,'fill').text="0"; SubElement(ps,'outline').text="1"
    pm = SubElement(doc, 'Placemark'); SubElement(pm,'styleUrl').text="#patchStyle"; SubElement(pm,'name').text="Candidate Patch"
    poly = SubElement(pm,'Polygon'); outer = SubElement(poly,'outerBoundaryIs'); linear = SubElement(outer,'LinearRing')
    SubElement(linear,'coordinates').text = " ".join(f"{lon},{lat},0" for lon,lat in coords)

    xml = minidom.parseString(tostring(kml,encoding="utf-8")).toprettyxml(indent="  ", encoding="utf-8")
    dest = out_path or CANDIDATE_KML
    with open(dest, "wb") as f:
        f.write(xml)
    print(f"KML file generated at {dest}")


# ——————— BALANCED SUBSET ———————

def create_balanced_subset():
    labels = load_labels()
    agri = [r for r in labels if r["label"].lower() == "agricultural"]
    non_agri = [r for r in labels if r["label"].lower() != "agricultural"]
    A = len(agri)
    max_non = int((1/MIN_AGRI_RATIO - 1) * A)
    if len(non_agri) <= max_non:
        print("Non-agri already within balance.")
        return labels
    sel_non = random.sample(non_agri, max_non)
    balanced = agri + sel_non
    ratio = len(agri)/len(balanced)
    print(f"Balanced: {len(agri)} agri, {max_non} non → ratio={ratio:.2f}")
    return balanced


# ——————— MANUAL & GLOBAL LABELING ———————

def manual_labeling(num_labels):
    labels = load_labels()
    added = 0
    w, h = get_patch_dimensions()
    for _ in range(num_labels):
        try:
            lat = float(input("Enter latitude: "))
            lon = float(input("Enter longitude: "))
        except ValueError:
            print("Invalid. Skip.")
            continue
        if duplicate_exists(lat, lon, labels):
            print("Duplicate. Skip.")
            continue
        tile = get_tile_for_coordinate(lat, lon)
        lab  = "Agricultural"
        eid  = f"manual_{int(random.random()*1e6)}"
        with open(LABELS_FILE, "a", newline="") as f:
            csv.writer(f).writerow([eid, lat, lon, tile, lab])
        print(f"Added manual label at ({lat},{lon}).")
        generate_kml_for_patch(lat, lon, w, h)
        labels.append({"lat":lat,"lon":lon,"tile":tile,"label":lab})
        added += 1
    return added


def global_sampling_labeling(num_patches):
    # sample uniformly over ROI from config or from first tile
    from config import ROI_COORDS
    if ROI_COORDS:
        roi = ROI_COORDS
    else:
        tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*", "*.tif"))
        if tifs:
            with rasterio.open(tifs[0]) as src:
                b = src.bounds
                roi = [[b.left,b.bottom],[b.left,b.top],[b.right,b.top],[b.right,b.bottom],[b.left,b.bottom]]
        else:
            roi = [[-180,-90],[-180,90],[180,90],[180,-90],[-180,-90]]
    lons = [p[0] for p in roi]; lats = [p[1] for p in roi]
    w, h = get_patch_dimensions()
    added = 0
    for _ in range(num_patches):
        lat = random.uniform(min(lats), max(lats))
        lon = random.uniform(min(lons), max(lons))
        generate_kml_for_patch(lat, lon, w, h)
        print(f"Open KML {CANDIDATE_KML} to view patch.")
        ui = input("Label? (1=Agri,2=Non,3=Skip): ").strip()
        if ui=="3":
            continue
        lab = "Agricultural" if ui=="1" else "Non-Agricultural" if ui=="2" else None
        if not lab:
            print("Invalid. Skip.")
            continue
        tile = get_tile_for_coordinate(lat, lon)
        eid  = f"global_{int(random.random()*1e6)}"
        with open(LABELS_FILE, "a", newline="") as f:
            csv.writer(f).writerow([eid, lat, lon, tile, lab])
        print(f"Added global label at ({lat},{lon}).")
        added += 1
    return added


# ——————— RICH FEATURE ENGINEERING (for a3 later) ———————

def load_normalizers(path=NORMALIZERS_CSV):
    stats = {}
    with open(path) as f:
        rd = csv.DictReader(f)
        for r in rd:
            stats[r["feature"]] = {"mean":float(r["mean"]), "std":float(r["stdDev"]) or 1.0}
    return stats

def compute_extra_indices(b2,b3,b4,b8,b11):
    eps=1e-6
    ndvi  = (b8 - b4)/(b8 + b4 + eps)
    evi   = 2.5*((b8 - b4)/(b8 + 6*b4 -7.5*b2 +1 + eps))
    evi2  = 2.5*((b8 - b4)/(b8 + 2.4*b4 +1 + eps))
    ndwi  = (b3 - b8)/(b3 + b8 + eps)
    nbr_s = (b8 - b11)/(b8 + b11 + eps)
    return ndvi,evi,evi2,ndwi,nbr_s

def zscore_vector(feats, normals):
    out=[]
    for f,val in feats.items():
        m=normals[f]["mean"]; s=normals[f]["std"]
        out.append((val - m)/s)
    return np.array(out, dtype=np.float32)

def extract_features_from_label(row):
    """3×3 window flatten + z-scored extras on center pixel."""
    normals = load_normalizers()
    lat, lon = float(row["lat"]), float(row["lon"])
    tile = os.path.join(RAW_DATA_DIR, row["tile"])
    tifs = glob.glob(os.path.join(tile,"*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif in {tile}")
    with rasterio.open(tifs[0]) as src:
        col_c,row_c = src.index(lon,lat)
        col0 = max(col_c-1,0); row0 = max(row_c-1,0)
        if col0+3>src.width:  col0=src.width-3
        if row0+3>src.height: row0=src.height-3
        win = Window(col0,row0,3,3)
        patch = src.read(window=win).astype(float)  # (bands,3,3)
        flat  = patch.reshape(-1).tolist()          # 5*9
        b2,b3,b4,b8,b11 = patch[:,1,1]
        ndvi,evi,evi2,ndwi,nbr_s = compute_extra_indices(b2,b3,b4,b8,b11)
        extras = {"NDVI":ndvi,"EVI":evi,"EVI2":evi2,"NDWI":ndwi,"NBR_s":nbr_s}
        zs = zscore_vector(extras, normals).tolist()
        return flat + zs


# ——————— HYBRID CANDIDATE SAMPLING ———————

def sample_random_candidates():
    pts=[]
    for tif in glob.glob(os.path.join(RAW_DATA_DIR,"*","*.tif")):
        tile = os.path.basename(os.path.dirname(tif))
        with rasterio.open(tif) as src:
            H,W = src.height, src.width
            total = H*W
            picks = random.sample(range(total), min(NUM_RANDOM_PICKS_PER_TILE, total))
            for idx in picks:
                r,c = divmod(idx, W)
                lon,lat = src.xy(r,c)
                pts.append((tile,lat,lon))
    return pts

def sample_margin_candidates():
    rnds = glob.glob(os.path.join(ROUNDS_DIR,"round_*"))
    if not rnds:
        return []
    rnd = sorted(rnds, key=lambda p:int(p.split("_")[-1]))[-1]
    pcsv = os.path.join(rnd,"predictions.csv")
    if not os.path.exists(pcsv):
        return []
    unc=[]
    with open(pcsv) as f:
        for r in csv.DictReader(f):
            p=float(r["predicted_prob"])
            if CANDIDATE_PROB_LOWER <= p <= CANDIDATE_PROB_UPPER:
                unc.append(r)
    chosen = random.sample(unc, min(len(unc), NUM_CANDIDATES_PER_ROUND))
    return [(r["tile"], float(r["center_lat"]), float(r["center_lon"])) for r in chosen]

def generate_candidate_list():
    normals = load_normalizers()
    seen = set()
    rows = []

    # random
    for tile,lat,lon in sample_random_candidates():
        key = (tile, round(lat,6), round(lon,6))
        if key in seen: continue
        seen.add(key)
        rows.append((f"rand_{len(rows)}", tile, lat, lon))

    # margin
    for tile,lat,lon in sample_margin_candidates():
        key = (tile, round(lat,6), round(lon,6))
        if key in seen: continue
        seen.add(key)
        rows.append((f"marg_{len(rows)}", tile, lat, lon))

    os.makedirs(os.path.dirname(CANDIDATES_CSV), exist_ok=True)
    with open(CANDIDATES_CSV,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["id","tile","lat","lon","label"])
        wpx,hpx = get_patch_dimensions()
        for cid,t,la,lo in rows:
            w.writerow([cid,t,la,lo,""])
            # write per-candidate KML
            kmlp = CANDIDATES_CSV.replace(".csv", f"_{cid}.kml")
            generate_kml_for_patch(la, lo, wpx, hpx, out_path=kmlp)
    print(f"Generated {len(rows)} candidates → {CANDIDATES_CSV}")


# ——————— LABELING FROM CANDIDATES CSV ———————

def label_from_candidates(num_labels):
    if not os.path.exists(CANDIDATES_CSV):
        print("No candidates CSV; run sampling first.")
        return 0

    rows = list(csv.DictReader(open(CANDIDATES_CSV)))
    labeled = 0
    for row in rows:
        if row["label"].strip() != "":
            continue
        if labeled >= num_labels:
            break

        cid, tile, lat, lon = row["id"], row["tile"], float(row["lat"]), float(row["lon"])
        kmlp = CANDIDATES_CSV.replace(".csv", f"_{cid}.kml")
        print(f"Open {kmlp} to view patch.")
        ui = input("Label? (1=Agri,2=Non,3=Skip): ").strip()
        if ui == "3":
            continue
        lab = "Agricultural" if ui=="1" else "Non-Agricultural" if ui=="2" else None
        if not lab:
            print("Invalid choice; skipping.")
            continue

        # append to master
        with open(LABELS_FILE, "a", newline="") as lf:
            csv.writer(lf).writerow([cid, lat, lon, tile, lab])
        row["label"] = lab
        labeled += 1
        print(f"Labeled {cid} → {lab}")

    # rewrite back updated candidates
    with open(CANDIDATES_CSV,"w",newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","tile","lat","lon","label"])
        w.writeheader()
        w.writerows(rows)

    return labeled


# ——————— INITIAL LABELING LOOP ———————

def initial_labeling():
    ensure_labels_file()

    print("\n=== Generating hybrid random+margin candidates ===")
    generate_candidate_list()
    print("Review the KMLs and candidate_labels.csv before labeling.\n")

    while True:
        ok, agri_cnt, ratio = check_label_requirements()
        total = len(load_labels())
        print(f"\nCurrent labels: total={total}, agri={agri_cnt}, ratio={ratio:.2f}")

        if not ok and ratio < MIN_AGRI_RATIO:
            print("Agricultural ratio too low.")
            if input("Create balanced subset? (y/n): ").strip().lower() == "y":
                bal = create_balanced_subset()
                with open(TEMP_LABELS_FILE,"w",newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["id","lat","lon","tile","label"])
                    for r in bal:
                        w.writerow([r["id"], r["lat"], r["lon"], r["tile"], r["label"]])
                print(f"Balanced subset → {TEMP_LABELS_FILE}. Proceed to training.")
                break

        if ok:
            if input("[1] Label more or [2] Train? => ").strip() == "2":
                break
            n = 5
        else:
            print(f"Need ≥{MIN_AGRI_COUNT} agri and ratio in [{MIN_AGRI_RATIO},{MAX_AGRI_RATIO}].")
            try:
                n = int(input("How many to label? "))
            except:
                n = 5

        print("Choose labeling: [1] Manual, [2] From candidates, [3] Global random")
        m = input("=> ").strip()
        if m == "1":
            added = manual_labeling(n)
        elif m == "2":
            added = label_from_candidates(n)
        elif m == "3":
            added = global_sampling_labeling(n)
        else:
            print("Invalid; skipping.")
            added = 0

        print(f"Added {added} labels.")

    print("Initial labeling complete; proceed to AL rounds.")


if __name__ == "__main__":
    initial_labeling()
