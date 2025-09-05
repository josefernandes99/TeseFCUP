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
    TEMP_LABELS_FILE, ROI_COORDS, NOTE_OPTIONS,
    HIGHSCORE_FILE, PROBABLE_AGRI_FILE, PERSISTENT_LISTS_ENABLED
)
from al_shared import snap_to_pixel_center, load_skipped_set, record_skipped_pixel

def ensure_labels_file():
    os.makedirs(os.path.dirname(LABELS_FILE), exist_ok=True)
    if not os.path.exists(LABELS_FILE):
        # ``notes`` column added so users can attach free form comments to any
        # label.  Downstream code simply ignores the column if present.
        with open(LABELS_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["id", "lat", "lon", "tile", "label", "notes"])
        print("Created new master labels CSV.")
    export_labels_kml()


# Evaluation CSV/KML removed in favor of stratified split over labels + temp_labels

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


# export_evaluate_kml removed


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
        # Use 5x5 pixel patches to reduce KML complexity
        return 5 * w, 5 * h


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


def _get_pixel_corners(src, row, col):
    """Return exact pixel corner coordinates in WGS84 for a given row/col."""
    tl = src.xy(row, col, offset='ul')
    tr = src.xy(row, col, offset='ur')
    br = src.xy(row, col, offset='lr')
    bl = src.xy(row, col, offset='ll')
    corners = [tl, tr, br, bl, tl]
    if src.crs and not src.crs.is_geographic:
        to_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        corners = [to_ll.transform(x, y) for x, y in corners]
    return corners


def generate_kml_for_pixel(tile, row, col, out_path=None):
    """Generate a KML outlining exactly one pixel (no 3x3 patch)."""
    tif = os.path.join(RAW_DATA_DIR, tile)
    if not os.path.exists(tif):
        print(f"Missing tile for candidate KML => {tile}")
        return
    with rasterio.open(tif) as src:
        corners = _get_pixel_corners(src, int(row), int(col))
    kml = Element('kml'); kml.set("xmlns","http://www.opengis.net/kml/2.2")
    doc = SubElement(kml, 'Document')
    style = SubElement(doc, 'Style', id="pixelStyle")
    ln = SubElement(style, 'LineStyle'); SubElement(ln,'color').text="ff0000ff"; SubElement(ln,'width').text="2"
    ps = SubElement(style, 'PolyStyle'); SubElement(ps,'fill').text="0"; SubElement(ps,'outline').text="1"
    pm = SubElement(doc, 'Placemark'); SubElement(pm,'styleUrl').text="#pixelStyle"; SubElement(pm,'name').text="Candidate Pixel"
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
        # 15% transparent black (aabbggrr): 26 alpha, 000000 black
        ln1 = SubElement(sp, 'LineStyle'); SubElement(ln1, 'color').text = '26000000'; SubElement(ln1, 'width').text = '2'

        def add_line(p0, p1):
            pm = SubElement(doc, 'Placemark')
            SubElement(pm, 'styleUrl').text = '#patchGrid'
            ls = SubElement(pm, 'LineString')
            SubElement(ls, 'coordinates').text = f"{to_lonlat(*p0)} {to_lonlat(*p1)}"

        # Draw only patch boundaries (every patch_px / patch_py)
        for c in range(0, width + 1, patch_px):
            p0 = src.xy(0, c, offset="ul")
            p1 = src.xy(height, c, offset="ul")
            add_line(p0, p1)

        for r in range(0, height + 1, patch_py):
            p0 = src.xy(r, 0, offset="ul")
            p1 = src.xy(r, width, offset="ul")
            add_line(p0, p1)

    xml = minidom.parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
    with open(out_path, 'wb') as f:
        f.write(xml)
    # Do not spam per-tile logs; a single summary will be printed by
    # generate_grids_for_all_tiles() after attempting all tiles.


def generate_grids_for_all_tiles():
    """Ensure a grid KML exists for every raw tile."""
    patch_w, patch_h = get_patch_dimensions()
    # Only consider original raw tiles; ignore any generated overlays or final-sweep artifacts
    all_tifs = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    tifs = [tp for tp in all_tifs if ("_overlay" not in os.path.basename(tp) and "_th" not in os.path.basename(tp))]
    if not tifs:
        print(f"No raw tiles in {RAW_DATA_DIR}; skipping grid creation.")
        return
    failed = []
    created = 0
    for tp in tifs:
        name = os.path.splitext(os.path.basename(tp))[0]
        out = os.path.join(GRID_KML_DIR, f"{name}_grid.kml")
        if os.path.exists(out):
            continue
        try:
            generate_grid_kml(tp, patch_w, patch_h, out)
            created += 1
        except Exception as e:
            failed.append((tp, str(e)))
    if failed:
        print("error on grid kml generation")
        for tp, err in failed:
            print(f" - {os.path.basename(tp)}: {err}")
    else:
        print("all grid kml generated")


# ----- BALANCED SUBSET -----

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


# ----- MANUAL & GLOBAL LABELING -----

def manual_labeling(num_labels):
    labels = load_labels()
    added = 0
    w, h = get_patch_dimensions()
    to_remove = []
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
            # record skip if possible
            try:
                snapped = snap_to_pixel_center(tile, lat, lon)
                if snapped:
                    la_s, lo_s, r_s, c_s = snapped
                    record_skipped_pixel(tile, r_s, c_s, la_s, lo_s, source="manual")
            except Exception:
                pass
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
        try:
            # collect for batch removal from persistent lists
            from al_shared import snap_to_pixel_center as _snap
            snapped = _snap(tile, lat, lon)
            if snapped:
                la_s, lo_s, r_s, c_s = snapped
                to_remove.append((tile, r_s, c_s, la_s, lo_s))
        except Exception:
            pass
        # Show exact candidate pixel KML
        try:
            from al_shared import snap_to_pixel_center as _snap
            snapped = _snap(tile, lat, lon)
            if snapped:
                la_s, lo_s, r_s, c_s = snapped
                generate_kml_for_pixel(tile, r_s, c_s)
            else:
                # Fallback: keep old patch behavior centered on point
                with rasterio.open(os.path.join(RAW_DATA_DIR, tile)) as src:
                    x, y = lon, lat
                    if src.crs and not src.crs.is_geographic:
                        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                        x, y = transformer.transform(lon, lat)
                generate_kml_for_patch(y, x, w, h)
        except Exception:
            pass
        added += 1
    if added:
        if to_remove:
            _batch_remove_pixels_from_lists(to_remove)
        _snap_labels_only()
        export_labels_kml()
    return added


def global_sampling_labeling(num_patches):
    """Sample random coordinates across the ROI covering all tiles."""
    roi = compute_roi_bbox()
    lons = [p[0] for p in roi]; lats = [p[1] for p in roi]
    w, h = get_patch_dimensions()
    added = 0
    to_remove = []
    session_seen = set()  # avoid duplicate pixels within this batch
    skipped = load_skipped_set()
    for _ in range(num_patches):
        lat = random.uniform(min(lats), max(lats))
        lon = random.uniform(min(lons), max(lons))
        tile = get_tile_for_coordinate(lat, lon)
        if not tile:
            print("No tile for coordinate; skipping.")
            continue
        # Show exact candidate pixel KML prior to labeling
        try:
            from al_shared import snap_to_pixel_center as _snap
            r_s = c_s = la_s = lo_s = None
            snapped = _snap(tile, lat, lon)
            if snapped:
                la_s, lo_s, r_s, c_s = snapped
                key = f"{tile}:{r_s}:{c_s}"
                if key in session_seen:
                    continue
                if key in skipped:
                    continue
                session_seen.add(key)
                generate_kml_for_pixel(tile, r_s, c_s)
            else:
                with rasterio.open(os.path.join(RAW_DATA_DIR, tile)) as src:
                    x, y = lon, lat
                    if src.crs and not src.crs.is_geographic:
                        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                        x, y = transformer.transform(lon, lat)
                generate_kml_for_patch(y, x, w, h)
        except Exception:
            pass
        print(f"Open KML {CANDIDATE_KML} to view patch.")
        ui = input("Label? (1=Agri,2=Non,3=Skip): ").strip()
        if ui=="3":
            try:
                if r_s is not None and c_s is not None and la_s is not None and lo_s is not None:
                    record_skipped_pixel(tile, r_s, c_s, la_s, lo_s, source="global")
                else:
                    snapped2 = snap_to_pixel_center(tile, lat, lon)
                    if snapped2:
                        la_s2, lo_s2, rs2, cs2 = snapped2
                        record_skipped_pixel(tile, rs2, cs2, la_s2, lo_s2, source="global")
            except Exception:
                pass
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
        try:
            from al_shared import snap_to_pixel_center as _snap
            snapped = _snap(tile, lat, lon)
            if snapped:
                la_s, lo_s, r_s, c_s = snapped
                to_remove.append((tile, r_s, c_s, la_s, lo_s))
        except Exception:
            pass
        added += 1
    if added:
        if to_remove:
            _batch_remove_pixels_from_lists(to_remove)
        _snap_labels_only()
        export_labels_kml()
    return added


# create_evaluation_labels removed (validation uses stratified split)

# ----- INITIAL LABELING LOOP -----

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
            choice = input("[1] Label more, [2] Train, [3] Review Highscore, [4] Review ProbableAgri => ").strip()
            if choice == "2":
                break
            if choice == "3":
                if not PERSISTENT_LISTS_ENABLED:
                    print("Persistent lists are disabled in config; Highscore review unavailable.")
                    continue
                try:
                    n = int(input("How many highscore entries? ").strip())
                except Exception:
                    n = 5
                added = assisted_labeling_from_list(HIGHSCORE_FILE, n, list_name="Highscore")
                print(f"Added {added} labels.")
                continue
            if choice == "4":
                if not PERSISTENT_LISTS_ENABLED:
                    print("Persistent lists are disabled in config; ProbableAgri review unavailable.")
                    continue
                try:
                    n = int(input("How many probableAgri entries? ").strip())
                except Exception:
                    n = 5
                added = assisted_labeling_from_list(PROBABLE_AGRI_FILE, n, list_name="ProbableAgri")
                print(f"Added {added} labels.")
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

        print("Choose labeling: [1] Manual, [2] Global random, [3] Highscore assisted, [4] ProbableAgri assisted")
        m = input("=> ").strip()
        if m == "1":
            added = manual_labeling(n)
        elif m == "2":
            added = global_sampling_labeling(n)
        elif m == "3":
            if not PERSISTENT_LISTS_ENABLED:
                print("Persistent lists are disabled in config; Highscore assisted labeling unavailable.")
                added = 0
            else:
                added = assisted_labeling_from_list(HIGHSCORE_FILE, n, list_name="Highscore")
        elif m == "4":
            if not PERSISTENT_LISTS_ENABLED:
                print("Persistent lists are disabled in config; ProbableAgri assisted labeling unavailable.")
                added = 0
            else:
                added = assisted_labeling_from_list(PROBABLE_AGRI_FILE, n, list_name="ProbableAgri")
        else:
            print("Invalid; skipping.")
            added = 0

        print(f"Added {added} labels.")

    print("Initial labeling complete; proceed to AL rounds.")


if __name__ == "__main__":
    initial_labeling()


def _remove_pixel_from_lists(tile: str, row: int, col: int, lat: float | None = None, lon: float | None = None):
    """Remove pixel from highscore/probableAgri/temp labels if present."""
    import csv as _csv
    # persistent lists (by row/col)
    if PERSISTENT_LISTS_ENABLED:
        for path in [HIGHSCORE_FILE, PROBABLE_AGRI_FILE]:
            if not os.path.exists(path):
                continue
            with open(path) as f:
                rows = list(_csv.DictReader(f))
            keep = [r for r in rows if not (r.get('tile') == tile and str(r.get('row')) == str(row) and str(r.get('col')) == str(col))]
            if len(keep) != len(rows):
                with open(path, 'w', newline='') as f:
                    w = _csv.DictWriter(f, fieldnames=list(keep[0].keys()) if keep else rows[0].keys())
                    w.writeheader(); w.writerows(keep)
    # global temp labels (by tile+lat+lon)
    from config import TEMP_LABELS_FILE as _TL
    if os.path.exists(_TL) and lat is not None and lon is not None:
        with open(_TL) as f:
            rows = list(_csv.DictReader(f))
        keep = [r for r in rows if not (r.get('tile') == tile and str(r.get('lat')) == f"{lat}" and str(r.get('lon')) == f"{lon}")]
        if len(keep) != len(rows):
            with open(_TL, 'w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=list(keep[0].keys()) if keep else rows[0].keys())
                w.writeheader(); w.writerows(keep)


def _snap_labels_only():
    """Snap labels.csv rows to exact pixel centers and fill row/col with progress.

    - Uses al_shared.snap_to_pixel_center(tile, lat, lon)
    - Updates lat/lon to 7 decimals and writes row/col columns.
    - Preserves other fields.
    """
    import csv as _csv
    from progress_utils import new_progress as _npb
    if not os.path.exists(LABELS_FILE):
        return
    with open(LABELS_FILE, newline='') as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return
    out = []
    from al_shared import snap_to_pixel_center as _snap
    with _npb() as _prog:
        task = _prog.add_task("Snap labels to pixel centers", total=len(rows))
        for r in rows:
            tile = r.get('tile')
            try:
                la = float(r.get('lat')); lo = float(r.get('lon'))
            except Exception:
                out.append(r); _prog.update(task, advance=1); continue
            snapped = None
            try:
                snapped = _snap(tile, la, lo) if tile else None
            except Exception:
                snapped = None
            if snapped:
                sla, slo, rowi, coli = snapped
                r['lat'] = f"{sla:.7f}"; r['lon'] = f"{slo:.7f}"
                r['row'] = int(rowi); r['col'] = int(coli)
            out.append(r)
            _prog.update(task, advance=1)
    # Write back; include row/col in header
    fields = list(out[0].keys())
    if 'row' not in fields:
        fields.append('row')
    if 'col' not in fields:
        fields.append('col')
    with open(LABELS_FILE, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)


def _batch_remove_pixels_from_lists(pixels):
    """Batch remove many pixels from persistent lists in one pass per file.

    pixels: list of (tile, row, col, lat, lon)
    """
    import csv as _csv
    from progress_utils import new_progress as _npb
    # Build key sets
    keys_rowcol = set()
    keys_latlon = set()
    for t, r, c, la, lo in pixels:
        try:
            keys_rowcol.add(f"{t}:{int(r)}:{int(c)}")
        except Exception:
            pass
        try:
            keys_latlon.add(f"{t}:{float(la):.7f}:{float(lo):.7f}")
        except Exception:
            pass

    def _filter_file(path, title):
        if not os.path.exists(path):
            return 0, 0
        try:
            size = os.path.getsize(path)
        except Exception:
            size = None
        kept = []
        removed = 0
        with open(path, newline='') as f, _npb() as _prog:
            task = _prog.add_task(f"{title}: {os.path.basename(path)}", total=size or None)
            rd = _csv.DictReader(f)
            last_tell = 0
            for r in rd:
                t = r.get('tile')
                rc_key = None
                ll_key = None
                try:
                    rc_key = f"{t}:{int(r.get('row'))}:{int(r.get('col'))}"
                except Exception:
                    rc_key = None
                try:
                    ll_key = f"{t}:{float(r.get('lat')):.7f}:{float(r.get('lon')):.7f}"
                except Exception:
                    ll_key = None
                if (rc_key and rc_key in keys_rowcol) or (ll_key and ll_key in keys_latlon):
                    removed += 1
                else:
                    kept.append(r)
                try:
                    cur = f.tell()
                    if size and cur > last_tell:
                        _prog.update(task, completed=min(cur, size))
                        last_tell = cur
                except Exception:
                    pass
            if size:
                _prog.update(task, completed=size)
        if removed > 0:
            with open(path, 'w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=list(kept[0].keys()) if kept else ['tile','row','col','lat','lon','prob','ndvi'])
                w.writeheader(); w.writerows(kept)
        return removed, len(kept)

    # Persistent lists (honor toggle)
    if PERSISTENT_LISTS_ENABLED:
        for path in [HIGHSCORE_FILE, PROBABLE_AGRI_FILE]:
            _filter_file(path, title="Batch remove from list")
    # Temp labels: remove by lat/lon if available
    from config import TEMP_LABELS_FILE as _TL
    if os.path.exists(_TL) and keys_latlon:
        _filter_file(_TL, title="Batch remove from temp labels")


def assisted_labeling_from_list(csv_path: str, max_count: int, list_name: str = "Assisted") -> int:
    """Stream candidates from a persistent list CSV and prompt labeling.

    Removes each labeled entry from the persistent files to avoid relabeling.
    """
    if not os.path.exists(csv_path):
        print(f"No {list_name} file at {csv_path}")
        return 0
    with open(csv_path) as f:
        import csv as _csv
        rows = list(_csv.DictReader(f))
    if not rows:
        print(f"{list_name} list empty.")
        return 0
    # Build a unique selection up to max_count
    uniq = []
    seen = set()
    skipped = load_skipped_set()
    for r in rows:
        t = r.get('tile'); rr = r.get('row'); cc = r.get('col')
        try:
            key = f"{t}:{int(rr)}:{int(cc)}"
        except Exception:
            continue
        if key in seen or (skipped and key in skipped):
            continue
        seen.add(key)
        uniq.append(r)
        if len(uniq) >= max_count:
            break

    added = 0
    w, h = get_patch_dimensions()
    to_remove = []
    for r in uniq:
        tile = r.get('tile')
        try:
            la = float(r.get('lat')); lo = float(r.get('lon'))
            row = int(r.get('row')); col = int(r.get('col'))
        except Exception:
            continue
        # Show exact candidate pixel KML for list-based review
        try:
            generate_kml_for_pixel(tile, row, col)
        except Exception:
            continue
        print(f"Open KML {CANDIDATE_KML} to view candidate from {list_name}.")
        ui = input("Label? (1=Agri,2=Non,3=Skip): ").strip()
        if ui == "3":
            # record skip directly using provided row/col and lat/lon from the list
            try:
                record_skipped_pixel(tile, row, col, la, lo, source=list_name)
            except Exception:
                pass
            continue
        lab = "Agricultural" if ui == "1" else "Non-Agricultural" if ui == "2" else None
        if not lab:
            print("Invalid choice.")
            continue
        note = prompt_note()
        eid = f"{list_name}_{int(random.random()*1e6)}"
        with open(LABELS_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([eid, la, lo, tile, lab, note])
        to_remove.append((tile, row, col, la, lo))
        print(f"Labeled from {list_name}: {tile} r={row},c={col}")
        added += 1
    if added:
        if to_remove:
            _batch_remove_pixels_from_lists(to_remove)
        _snap_labels_only()
        export_labels_kml()
    return added
