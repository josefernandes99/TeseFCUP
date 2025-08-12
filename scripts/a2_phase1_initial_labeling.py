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
    LABELS_FILE, LABELS_KML, MIN_AGRI_COUNT, MIN_AGRI_RATIO, MAX_AGRI_RATIO,
    DUPLICATE_TOLERANCE, RAW_DATA_DIR, CANDIDATE_KML, GRID_KML_DIR,
    TEMP_LABELS_FILE, ROI_COORDS, EVALUATE_FILE, NOTE_OPTIONS
)

def ensure_labels_file():
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    if not os.path.exists(LABELS_FILE):
        # ``notes`` column added so users can attach free form comments to any
        # label.  Downstream code simply ignores the column if present.
        with open(LABELS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["id", "lat", "lon", "tile", "label", "notes"])
        print("Created new master labels CSV.")
    export_labels_kml()


def ensure_evaluate_file():
    """Create evaluation CSV with notes column if it does not yet exist."""
    os.makedirs(os.path.dirname(EVALUATE_FILE), exist_ok=True)
    if not os.path.exists(EVALUATE_FILE):
        with open(EVALUATE_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["id", "lat", "lon", "tile", "label", "notes"])
        print("Created new evaluation CSV.")


def load_labels(path=LABELS_FILE):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def export_labels_kml(path=LABELS_FILE, out_path=LABELS_KML):
    """Export all labels to a KML with point markers for Google Earth."""
    labels = load_labels(path)
    doc = minidom.Document()
    kml = doc.createElement("kml")
    kml.setAttribute("xmlns", "http://www.opengis.net/kml/2.2")
    doc.appendChild(kml)
    d = doc.createElement("Document")
    kml.appendChild(d)
    # Define styles for colored pins without visible labels
    style_agri = doc.createElement("Style"); style_agri.setAttribute("id", "agri")
    icon_agri = doc.createElement("IconStyle")
    color_agri = doc.createElement("color")
    color_agri.appendChild(doc.createTextNode("ff00ff00"))  # green
    icon_agri.appendChild(color_agri)
    label_agri = doc.createElement("LabelStyle")
    scale_agri = doc.createElement("scale"); scale_agri.appendChild(doc.createTextNode("0"))
    label_agri.appendChild(scale_agri)
    style_agri.appendChild(icon_agri); style_agri.appendChild(label_agri)
    d.appendChild(style_agri)

    style_non = doc.createElement("Style"); style_non.setAttribute("id", "nonagri")
    icon_non = doc.createElement("IconStyle")
    color_non = doc.createElement("color")
    color_non.appendChild(doc.createTextNode("ff0000ff"))  # red
    icon_non.appendChild(color_non)
    label_non = doc.createElement("LabelStyle")
    scale_non = doc.createElement("scale"); scale_non.appendChild(doc.createTextNode("0"))
    label_non.appendChild(scale_non)
    style_non.appendChild(icon_non); style_non.appendChild(label_non)
    d.appendChild(style_non)

    for r in labels:
        pm = doc.createElement("Placemark")
        # Add the note as the placemark name so it is available when clicking the
        # pin, but keep the on-map label hidden via LabelStyle scale=0.
        name_el = doc.createElement("name")
        name_el.appendChild(doc.createTextNode(r.get("notes", "")))
        pm.appendChild(name_el)

        style = doc.createElement("styleUrl")
        if r.get("label", "").lower() == "agricultural":
            style.appendChild(doc.createTextNode("#agri"))
        else:
            style.appendChild(doc.createTextNode("#nonagri"))
        pm.appendChild(style)

        pt = doc.createElement("Point")
        coords = doc.createElement("coordinates")
        coords.appendChild(doc.createTextNode(f"{r.get('lon')},{r.get('lat')},0"))
        pt.appendChild(coords)
        pm.appendChild(pt)
        d.appendChild(pm)
    with open(out_path, "w") as f:
        f.write(doc.toprettyxml(indent="  "))
    print(f"Exported {len(labels)} labels to KML => {out_path}")


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


def prompt_note():
    """Prompt user to choose a predefined note option."""
    print("notes options:")
    for idx, opt in enumerate(NOTE_OPTIONS, 1):
        print(f" {idx}. {opt}")
    choice = input("Select note [1-9]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(NOTE_OPTIONS):
        return NOTE_OPTIONS[int(choice) - 1]
    print("Invalid choice; using 'Other'.")
    return NOTE_OPTIONS[-1]


def get_tile_for_coordinate(_lat, _lon):
    """Return the tile filename that contains the given WGS84 coordinate.

    If no tile covers the point, ``None`` is returned.  This replaces the
    previous random selection which could associate labels with the wrong
    tile once multiple images are present.
    """
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    for fp in files:
        try:
            with rasterio.open(fp) as src:
                x, y = _lon, _lat
                if src.crs and not src.crs.is_geographic:
                    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                    x, y = transformer.transform(_lon, _lat)
                b = src.bounds
                if b.left <= x <= b.right and b.bottom <= y <= b.top:
                    return os.path.basename(fp)
        except Exception:
            continue
    return None

def get_patch_dimensions():
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    if not tifs:
        return 0.001, 0.001
    with rasterio.open(tifs[0]) as src:
        w = abs(src.transform[0])
        h = abs(src.transform[4])
        return 3 * w, 3 * h


def compute_roi_bbox():
    """Return overall ROI bounding box covering all available tiles."""
    if ROI_COORDS:
        return ROI_COORDS

    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    if not tifs:
        return [[-180, -90], [-180, 90], [180, 90], [180, -90], [-180, -90]]

    min_lon = min_lat = float("inf")
    max_lon = max_lat = float("-inf")

    for fp in tifs:
        try:
            with rasterio.open(fp) as src:
                b = src.bounds
                if src.crs and not src.crs.is_geographic:
                    transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                    left, bottom = transformer.transform(b.left, b.bottom)
                    right, top = transformer.transform(b.right, b.top)
                else:
                    left, bottom, right, top = b.left, b.bottom, b.right, b.top

                min_lon = min(min_lon, left)
                min_lat = min(min_lat, bottom)
                max_lon = max(max_lon, right)
                max_lat = max(max_lat, top)
        except Exception:
            continue

    return [
        [min_lon, min_lat],
        [min_lon, max_lat],
        [max_lon, max_lat],
        [max_lon, min_lat],
        [min_lon, min_lat],
    ]


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


def generate_grid_kml(tile_path, patch_width, patch_height, out_path):
    """Create a grid overlay KML for the given tile.

    ``patch_width`` and ``patch_height`` are in the same CRS units as the tile
    itself (typically metres for UTM tiles).  The resulting file is written to
    ``out_path``.
    """
    with rasterio.open(tile_path) as src:
        res_x, res_y = abs(src.transform[0]), abs(src.transform[4])
        width, height = src.width, src.height
        patch_px = max(int(round(patch_width / res_x)), 1)
        patch_py = max(int(round(patch_height / res_y)), 1)

        transformer = None
        if src.crs and not src.crs.is_geographic:
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)

        def to_lonlat(x, y):
            if transformer:
                x, y = transformer.transform(x, y)
            return f"{x},{y},0"

        kml = Element('kml'); kml.set('xmlns', 'http://www.opengis.net/kml/2.2')
        doc = SubElement(kml, 'Document')
        sp = SubElement(doc, 'Style', id='patchGrid')
        ln1 = SubElement(sp, 'LineStyle'); SubElement(ln1, 'color').text = 'ff000000'; SubElement(ln1, 'width').text = '2'
        si = SubElement(doc, 'Style', id='pixelGrid')
        ln2 = SubElement(si, 'LineStyle'); SubElement(ln2, 'color').text = 'ff000000'; SubElement(ln2, 'width').text = '1'

        def add_line(p0, p1, use_patch):
            pm = SubElement(doc, 'Placemark')
            SubElement(pm, 'styleUrl').text = '#patchGrid' if use_patch else '#pixelGrid'
            ls = SubElement(pm, 'LineString')
            SubElement(ls, 'coordinates').text = f"{to_lonlat(*p0)} {to_lonlat(*p1)}"

        for c in range(0, width + 1):
            p0 = src.xy(0, c, offset="ul")
            p1 = src.xy(height, c, offset="ul")
            add_line(p0, p1, c % patch_px == 0)

        for r in range(0, height + 1):
            p0 = src.xy(r, 0, offset="ul")
            p1 = src.xy(r, width, offset="ul")
            add_line(p0, p1, r % patch_py == 0)

    xml = minidom.parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
    with open(out_path, 'wb') as f:
        f.write(xml)
    print(f"Grid KML generated => {out_path}")


def generate_grids_for_all_tiles():
    """Ensure a grid KML exists for every raw tile."""
    patch_w, patch_h = get_patch_dimensions()
    tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    if not tifs:
        print(f"No raw tiles in {RAW_DATA_DIR}; skipping grid creation.")
        return
    for tp in tifs:
        name = os.path.splitext(os.path.basename(tp))[0]
        out = os.path.join(GRID_KML_DIR, f"{name}_grid.kml")
        if os.path.exists(out):
            continue
        try:
            generate_grid_kml(tp, patch_w, patch_h, out)
        except Exception as e:
            print(f"Failed to make grid for {tp}: {e}")


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
        if not tile:
            print("No tile for coordinate; skipping.")
            continue
        print("Label? (1=Agri,2=Non,3=Skip)")
        ui = input("=> ").strip()
        if ui == "3":
            continue
        lab = "Agricultural" if ui == "1" else "Non-Agricultural" if ui == "2" else None
        if not lab:
            print("Invalid label. Skip.")
            continue
        eid  = f"manual_{int(random.random()*1e6)}"
        note = prompt_note()
        with open(LABELS_FILE, "a", newline="") as f:
            csv.writer(f).writerow([eid, lat, lon, tile, lab, note])
        print(f"Added manual label at ({lat},{lon}).")
        labels.append({"lat":lat,"lon":lon,"tile":tile,"label":lab,"notes":note})
        export_labels_kml()
        # convert to tile CRS for patch display
        with rasterio.open(os.path.join(RAW_DATA_DIR, tile)) as src:
            x, y = lon, lat
            if src.crs and not src.crs.is_geographic:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
        generate_kml_for_patch(y, x, w, h)
        added += 1
    export_labels_kml()
    return added


def global_sampling_labeling(num_patches):
    """Sample random coordinates across the ROI covering all tiles."""
    roi = compute_roi_bbox()
    lons = [p[0] for p in roi]; lats = [p[1] for p in roi]
    w, h = get_patch_dimensions()
    added = 0
    for _ in range(num_patches):
        lat = random.uniform(min(lats), max(lats))
        lon = random.uniform(min(lons), max(lons))
        tile = get_tile_for_coordinate(lat, lon)
        if not tile:
            print("No tile for coordinate; skipping.")
            continue
        # convert to tile CRS for visualization
        with rasterio.open(os.path.join(RAW_DATA_DIR, tile)) as src:
            x, y = lon, lat
            if src.crs and not src.crs.is_geographic:
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(lon, lat)
        generate_kml_for_patch(y, x, w, h)
        print(f"Open KML {CANDIDATE_KML} to view patch.")
        ui = input("Label? (1=Agri,2=Non,3=Skip): ").strip()
        if ui=="3":
            continue
        lab = "Agricultural" if ui=="1" else "Non-Agricultural" if ui=="2" else None
        if not lab:
            print("Invalid. Skip.")
            continue
        eid  = f"global_{int(random.random()*1e6)}"
        note = prompt_note()
        with open(LABELS_FILE, "a", newline="") as f:
            csv.writer(f).writerow([eid, lat, lon, tile, lab, note])
        print(f"Added global label at ({lat},{lon}).")
        export_labels_kml()
        added += 1
    export_labels_kml()
    return added


def create_evaluation_labels():
    """Prompt the user to manually create an evaluation set of 100 labels
    (25 agricultural / 75 non-agricultural).  Duplicates with existing labels
    are not allowed.  Labels are stored in ``evaluate.csv``.
    """
    ensure_evaluate_file()
    eval_labels = load_labels(EVALUATE_FILE)
    master_labels = load_labels(LABELS_FILE)
    existing = eval_labels + master_labels
    agri = sum(1 for r in eval_labels if r["label"].lower() == "agricultural")
    non  = sum(1 for r in eval_labels if r["label"].lower() != "agricultural")
    target_agri, target_non = 25, 75
    while agri < target_agri or non < target_non:
        try:
            lat = float(input("Enter latitude: "))
            lon = float(input("Enter longitude: "))
        except ValueError:
            print("Invalid. Skip.")
            continue
        if duplicate_exists(lat, lon, existing):
            print("Duplicate label. Skip.")
            continue
        tile = get_tile_for_coordinate(lat, lon)
        if not tile:
            print("No tile for coordinate; skipping.")
            continue
        print("Label? (1=Agri,2=Non,3=Skip)")
        ui = input("=> ").strip()
        if ui == "3":
            continue
        lab = "Agricultural" if ui == "1" else "Non-Agricultural" if ui == "2" else None
        if lab is None:
            print("Invalid label; skipping.")
            continue
        # enforce ratio
        if lab == "Agricultural" and agri >= target_agri:
            print("Already have required agricultural samples; choose non-agricultural.")
            continue
        if lab != "Agricultural" and non >= target_non:
            print("Already have required non-agricultural samples; choose agricultural.")
            continue
        note = prompt_note()
        eid = f"eval_{int(random.random()*1e6)}"
        with open(EVALUATE_FILE, "a", newline="") as f:
            csv.writer(f).writerow([eid, lat, lon, tile, lab, note])
        existing.append({"lat": lat, "lon": lon})
        if lab == "Agricultural":
            agri += 1
        else:
            non += 1
        print(f"Evaluation label added. Totals → Agri:{agri}/25 Non:{non}/75")
    print(f"Evaluation labeling complete → {EVALUATE_FILE}")

# ——————— INITIAL LABELING LOOP ———————

def initial_labeling():
    ensure_labels_file()
    generate_grids_for_all_tiles()

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
                    # keep notes column for compatibility
                    w.writerow(["id","lat","lon","tile","label","notes"])
                    for r in bal:
                        w.writerow([r.get("id"), r.get("lat"), r.get("lon"), r.get("tile"), r.get("label"), r.get("notes", "")])
                print(f"Balanced subset → {TEMP_LABELS_FILE}. Proceed to training.")
                break

        if ok:
            choice = input("[1] Label more, [2] Train, [3] Create Evaluation Labels? => ").strip()
            if choice == "2":
                break
            if choice == "3":
                create_evaluation_labels()
                continue
            try:
                n = int(input("How many to label? "))
            except Exception:
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
