# scripts/a2_phase1_initial_labeling.py

import csv
import glob
import math
import os
import random
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement, tostring
import rasterio
from pyproj import Transformer

from config import (
    LABELS_FILE, MIN_AGRI_COUNT, MIN_AGRI_RATIO, MAX_AGRI_RATIO,
    DUPLICATE_TOLERANCE, RAW_DATA_DIR, CANDIDATE_KML,
    TEMP_LABELS_FILE
)

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
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    return os.path.basename(random.choice(files)) if files else "unknown"

def get_patch_dimensions():
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    if not tifs:
        return 0.001, 0.001
    with rasterio.open(tifs[0]) as src:
        w = abs(src.transform[0])
        h = abs(src.transform[4])
        return 3 * w, 3 * h


def generate_kml_for_patch(center_lat, center_lon, patch_width, patch_height, out_path=None):
    """Generate a KML file describing a square patch centred on the given point.

        ``center_lat`` and ``center_lon`` are expected to be in the same coordinate
        reference system as the raw tiles.  If that CRS is projected (e.g. UTM), the
        coordinates are converted to the standard WGS84 latitude/longitude system so
        that the polygon displays correctly in applications like Google Earth.
        """
    half_w, half_h = patch_width / 2, patch_height / 2
    corners = [
        [center_lon - half_w, center_lat - half_h],
        [center_lon - half_w, center_lat + half_h],
        [center_lon + half_w, center_lat + half_h],
        [center_lon + half_w, center_lat - half_h],
        [center_lon - half_w, center_lat - half_h],
    ]

    # Obtain CRS from any available raw tile
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    crs = None
    if tifs:
        try:
            with rasterio.open(tifs[0]) as src:
                crs = src.crs
        except Exception:
            crs = None

    if crs and not crs.is_geographic:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        corners = [transformer.transform(x, y) for x, y in corners]

    kml = Element('kml'); kml.set("xmlns","http://www.opengis.net/kml/2.2")
    doc = SubElement(kml, 'Document')
    style = SubElement(doc, 'Style', id="patchStyle")
    ln = SubElement(style, 'LineStyle'); SubElement(ln,'color').text="ff0000ff"; SubElement(ln,'width').text="2"
    ps = SubElement(style, 'PolyStyle'); SubElement(ps,'fill').text="0"; SubElement(ps,'outline').text="1"
    pm = SubElement(doc, 'Placemark'); SubElement(pm,'styleUrl').text="#patchStyle"; SubElement(pm,'name').text="Candidate Patch"
    poly = SubElement(pm,'Polygon'); outer = SubElement(poly,'outerBoundaryIs'); linear = SubElement(outer,'LinearRing')
    SubElement(linear,'coordinates').text = " ".join(f"{lon},{lat},0" for lon, lat in corners)

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
        tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
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

# ——————— INITIAL LABELING LOOP ———————

def initial_labeling():
    ensure_labels_file()

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

        print("Choose labeling: [1] Manual, [2] Global random")
        m = input("=> ").strip()
        if m == "1":
            added = manual_labeling(n)
        elif m == "2":
            added = global_sampling_labeling(n)
        else:
            print("Invalid; skipping.")
            added = 0

        print(f"Added {added} labels.")

    print("Initial labeling complete; proceed to AL rounds.")


if __name__ == "__main__":
    initial_labeling()
