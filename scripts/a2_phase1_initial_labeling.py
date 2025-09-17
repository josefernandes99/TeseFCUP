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
    HIGHSCORE_FILE, PROBABLE_AGRI_FILE,
    HIGHSCORE_LIST_ENABLED, PROBABLE_AGRI_LIST_ENABLED,
)
from al_shared import snap_to_pixel_center, load_skipped_set, record_skipped_pixel
import config as cfg

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

# ---- Unified proximity duplicate handling for assisted reviews ----
def _build_label_index():
    """Build quick lookup for existing labels from master + temp.

    Returns:
      (rowcol_keys: set[str], by_tile_latlon: dict[str, list[tuple[float,float]]])
    """
    import csv as _csv
    rows = []
    # master
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, newline='') as f:
            rows += list(_csv.DictReader(f))
    # temp
    if os.path.exists(TEMP_LABELS_FILE):
        with open(TEMP_LABELS_FILE, newline='') as f:
            rows += list(_csv.DictReader(f))
    rowcol = set()
    by_tile = {}
    for r in rows:
        t = r.get('tile') or ''
        try:
            key = f"{t}:{int(r.get('row'))}:{int(r.get('col'))}"
            rowcol.add(key)
        except Exception:
            pass
        try:
            la = float(r.get('lat')); lo = float(r.get('lon'))
            by_tile.setdefault(t, []).append((la, lo))
        except Exception:
            pass
    return rowcol, by_tile

def _is_duplicate_candidate(tile: str,
                            lat: float,
                            lon: float,
                            row: int | None,
                            col: int | None,
                            label_index: tuple[set[str], dict[str, list[tuple[float,float]]]]
                            ) -> bool:
    """Return True if candidate is already represented by existing labels.

    Rules (fast → robust):
    - If (tile,row,col) exists in labels, treat as duplicate.
    - Else, if any existing label on same tile is within DUPLICATE_TOLERANCE (deg), duplicate.
    - Else, snap to pixel center and re-check row/col if missing.
    """
    rowcol_keys, by_tile = label_index
    # Exact same pixel (preferred)
    if row is not None and col is not None:
        if f"{tile}:{int(row)}:{int(col)}" in rowcol_keys:
            return True
    else:
        # try snapping to get row/col
        try:
            sn = snap_to_pixel_center(tile, float(lat), float(lon))
        except Exception:
            sn = None
        if sn:
            la_s, lo_s, r_s, c_s = sn
            if f"{tile}:{int(r_s)}:{int(c_s)}" in rowcol_keys:
                return True
    # Proximity in degrees (legacy tolerance; consistent across flows)
    try:
        pts = by_tile.get(tile, [])
        for la0, lo0 in pts:
            if math.hypot(float(la0) - float(lat), float(lo0) - float(lon)) < DUPLICATE_TOLERANCE:
                return True
    except Exception:
        pass
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
    labels_all = load_labels()
    w, h = get_patch_dimensions()
    to_remove = []
    for _ in range(num_labels):
        try:
            lat = float(input("Enter latitude: "))
            lon = float(input("Enter longitude: "))
        except ValueError:
            print("Invalid. Skip.")
            continue
        tile = get_tile_for_coordinate(lat, lon)
        if not tile:
            print("No tile for coordinate; skipping.")
            continue
        # Snap to pixel center first, then deduplicate using snapped coordinates
        try:
            snapped = snap_to_pixel_center(tile, lat, lon)
        except Exception:
            snapped = None
        if snapped:
            la_s, lo_s, r_s, c_s = snapped
            lat, lon = float(la_s), float(lo_s)
        if duplicate_exists(lat, lon, labels):
            print("Duplicate (snapped). Skip.")
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
            if 'snapped' in locals() and snapped:
                to_remove.append((tile, int(r_s), int(c_s), lat, lon))
            else:
                sn2 = snap_to_pixel_center(tile, lat, lon)
                if sn2:
                    la2, lo2, rr2, cc2 = sn2
                    to_remove.append((tile, int(rr2), int(cc2), la2, lo2))
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
        # Snap coordinates (again) to ensure we record exact pixel center
        if r_s is not None and c_s is not None and la_s is not None and lo_s is not None:
            lat_write, lon_write = float(la_s), float(lo_s)
        else:
            sn3 = snap_to_pixel_center(tile, lat, lon)
            if sn3:
                la3, lo3, rs3, cs3 = sn3
                lat_write, lon_write = float(la3), float(lo3)
            else:
                lat_write, lon_write = lat, lon
        # Dedup using snapped lat/lon
        if duplicate_exists(lat_write, lon_write, labels_all):
            print("Duplicate (snapped). Skip.")
            continue
        eid  = f"global_{int(random.random()*1e6)}"
        note = prompt_note()
        with open(LABELS_FILE, "a", newline="") as f:
            csv.writer(f).writerow([eid, lat_write, lon_write, tile, lab, note])
        print(f"Added global label at ({lat_write},{lon_write}).")
        try:
            if r_s is not None and c_s is not None and la_s is not None and lo_s is not None:
                to_remove.append((tile, int(r_s), int(c_s), la_s, lo_s))
            else:
                sn4 = snap_to_pixel_center(tile, lat_write, lon_write)
                if sn4:
                    la4, lo4, rr4, cc4 = sn4
                    to_remove.append((tile, int(rr4), int(cc4), la4, lo4))
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
            choice = input("[1] Label more, [2] Train, [3] Review Highscore, [4] Review ProbableAgri, [5] Review HardNeg (HN) => ").strip()
            if choice == "2":
                break
            if choice == "3":
                if not HIGHSCORE_LIST_ENABLED:
                    print("Highscore list is disabled in config; review unavailable.")
                    continue
                try:
                    n = int(input("How many highscore entries? ").strip())
                except Exception:
                    n = 5
                added = assisted_labeling_from_list(HIGHSCORE_FILE, n, list_name="Highscore")
                print(f"Added {added} labels.")
                continue
            if choice == "4":
                if not PROBABLE_AGRI_LIST_ENABLED:
                    print("ProbableAgri list is disabled in config; review unavailable.")
                    continue
                try:
                    n = int(input("How many probableAgri entries? ").strip())
                except Exception:
                    n = 5
                added = assisted_labeling_from_list(PROBABLE_AGRI_FILE, n, list_name="ProbableAgri")
                print(f"Added {added} labels.")
                continue
            if choice == "5":
                try:
                    n = int(input("How many hard negatives? ").strip())
                except Exception:
                    n = 5
                added = assisted_labeling_hard_negatives(n)
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

        print("Choose labeling: [1] Manual, [2] Global random, [3] Highscore assisted, [4] ProbableAgri assisted, [5] HardNeg assisted")
        m = input("=> ").strip()
        if m == "1":
            added = manual_labeling(n)
        elif m == "2":
            added = global_sampling_labeling(n)
        elif m == "3":
            if not HIGHSCORE_LIST_ENABLED:
                print("Highscore list is disabled in config; assisted labeling unavailable.")
                added = 0
            else:
                added = assisted_labeling_from_list(HIGHSCORE_FILE, n, list_name="Highscore")
        elif m == "4":
            if not PROBABLE_AGRI_LIST_ENABLED:
                print("ProbableAgri list is disabled in config; assisted labeling unavailable.")
                added = 0
            else:
                added = assisted_labeling_from_list(PROBABLE_AGRI_FILE, n, list_name="ProbableAgri")
        elif m == "5":
            added = assisted_labeling_hard_negatives(n)
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
    if HIGHSCORE_LIST_ENABLED:
        path = HIGHSCORE_FILE
        if os.path.exists(path):
            with open(path) as f:
                rows = list(_csv.DictReader(f))
            keep = [r for r in rows if not (r.get('tile') == tile and str(r.get('row')) == str(row) and str(r.get('col')) == str(col))]
            if len(keep) != len(rows):
                with open(path, 'w', newline='') as f:
                    w = _csv.DictWriter(f, fieldnames=list(keep[0].keys()) if keep else rows[0].keys())
                    w.writeheader(); w.writerows(keep)
    if PROBABLE_AGRI_LIST_ENABLED:
        path = PROBABLE_AGRI_FILE
        if os.path.exists(path):
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

    # Persistent lists (honor per-list toggles)
    if HIGHSCORE_LIST_ENABLED:
        _filter_file(HIGHSCORE_FILE, title="Batch remove from list")
    if PROBABLE_AGRI_LIST_ENABLED:
        _filter_file(PROBABLE_AGRI_FILE, title="Batch remove from list")
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
    # Build unique pool; honour skipped set; keep full pool for diversity selection
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

    # Optional spatial diversity: cluster by haversine and interleave one per cluster
    def _diversify(cands):
        import math
        diversified = []
        if not getattr(cfg, 'ASSISTED_SPATIAL_DIVERSITY_ENABLED', True) or not cands:
            return cands[:max_count]
        try:
            from sklearn.cluster import DBSCAN
            import numpy as _np
            eps_km = float(getattr(cfg, 'ASSISTED_DIVERSITY_EPS_KM', getattr(cfg, 'CANDIDATE_DBSCAN_EPS_KM', 1.5)))
            earth_km = 6371.0088
            lat = []; lon = []
            for r in cands:
                try:
                    lat.append(float(r.get('lat'))); lon.append(float(r.get('lon')))
                except Exception:
                    lat.append(_np.nan); lon.append(_np.nan)
            A = _np.vstack([_np.deg2rad(_np.array(lat, dtype=float)), _np.deg2rad(_np.array(lon, dtype=float))]).T
            # Filter out rows with missing coords
            valid = _np.isfinite(A).all(axis=1)
            idxs = _np.where(valid)[0]
            if idxs.size == 0:
                return cands[:max_count]
            coords = A[idxs]
            cl = DBSCAN(eps=eps_km/earth_km, min_samples=1, metric='haversine').fit(coords)
            labels = cl.labels_ if hasattr(cl, 'labels_') else _np.zeros((idxs.size,), dtype=int)
            # Build per-cluster queues; sort each cluster by best available score/prob desc
            from collections import defaultdict
            buckets = defaultdict(list)
            def _score(r):
                try:
                    return float(r.get('score'))
                except Exception:
                    try: return float(r.get('prob'))
                    except Exception: return float('-inf')
            for ii, cid in zip(idxs.tolist(), labels.tolist()):
                buckets[cid].append(cands[ii])
            for cid in buckets:
                buckets[cid].sort(key=_score, reverse=True)
            # Interleave one-per-cluster
            keys = list(buckets.keys())
            ptr = {cid:0 for cid in keys}
            while len(diversified) < max_count and any(ptr[cid] < len(buckets[cid]) for cid in keys):
                for cid in keys:
                    p = ptr[cid]
                    if p < len(buckets[cid]):
                        diversified.append(buckets[cid][p])
                        ptr[cid] = p + 1
                        if len(diversified) >= max_count:
                            break
            # If some rows lacked coords or we still need more, backfill in original order
            if len(diversified) < max_count:
                seenK = set(id(r) for r in diversified)
                for r in cands:
                    if id(r) in seenK: continue
                    diversified.append(r)
                    if len(diversified) >= max_count:
                        break
            # Debug print
            try:
                print(f"[ASSISTED] {list_name} diversity: clusters={len(keys)}, picked={len(diversified)} (eps_km={eps_km})")
            except Exception:
                pass
            return diversified
        except Exception as _e:
            try: print(f"[ASSISTED] Diversity disabled due to error: {_e}")
            except Exception: pass
            return cands[:max_count]

    uniq = _diversify(uniq)

    # Pre-filter duplicates by proximity and exact pixel (same logic as other assisted flows)
    try:
        label_index = _build_label_index()
        labels_all_pref = load_labels()
        filtered = []
        dropped = 0
        for r in uniq:
            t = r.get('tile')
            try:
                la = float(r.get('lat')); lo = float(r.get('lon'))
            except Exception:
                continue
            try:
                rr = int(r.get('row')); cc = int(r.get('col'))
            except Exception:
                rr = None; cc = None
            # unified pixel key or proximity
            dup = _is_duplicate_candidate(t, la, lo, rr, cc, label_index)
            # legacy proximity guard in case of rounding/snap differences
            if not dup:
                try:
                    if duplicate_exists(la, lo, labels_all_pref):
                        dup = True
                except Exception:
                    pass
            if dup:
                dropped += 1
                continue
            filtered.append(r)
        if dropped:
            print(f"[ASSISTED] {list_name}: pre-filtered {dropped} near-duplicate candidates.")
        uniq = filtered
    except Exception as _e_pf:
        try:
            print(f"[ASSISTED] HardNeg: pre-filter step skipped due to error: {_e_pf}")
        except Exception:
            pass

    # (pre-filter applied above)

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
        # Pre-prompt duplicate guard: never show a duplicate
        try:
            if _is_duplicate_candidate(tile, la, lo, row, col, label_index):
                # secondary guard using legacy proximity against labels snapshot
                try:
                    if duplicate_exists(la, lo, load_labels()):
                        pass
                except Exception:
                    pass
                continue
        except Exception:
            pass
        # Show exact candidate pixel KML for list-based review
        try:
            generate_kml_for_pixel(tile, row, col)
        except Exception:
            continue
        print(f"Open KML {CANDIDATE_KML} to view candidate from {list_name}.")
        print("Label? (1=Agri,2=Non,3=Skip): ")
        while True:
            ui = input("=> ").strip()
            if ui in ("1","2","3"):
                break
            print("Please type 1, 2, or 3.")
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
        # Update in-memory indices so subsequent candidates respect proximity
        try:
            if 'label_index' in locals() and isinstance(label_index, tuple) and len(label_index) == 2:
                label_index[0].add(f"{tile}:{int(row)}:{int(col)}")
                label_index[1].setdefault(tile, []).append((float(la), float(lo)))
        except Exception:
            pass
    if added:
        if to_remove:
            _batch_remove_pixels_from_lists(to_remove)
        _snap_labels_only()
        export_labels_kml()
    return added


def assisted_labeling_hard_negatives(max_count: int) -> int:
    """Stream 'hard negative' candidates from Highscore list and prompt labeling.

    Selection criteria (default):
      - prob in [MIN_AGRI_PROB - NEG_LIKE_PROB_DELTA, MIN_AGRI_PROB)
      - optional NDVI filter via NEG_LIKE_NDVI_RANGE when set (absolute mode)

    Notes:
      - Uses the same interactive loop and KML preview as other assisted flows.
      - De-duplicates against skipped set and enforces uniqueness by (tile,row,col).
    """
    if not HIGHSCORE_LIST_ENABLED:
        print("Highscore list is disabled in config; assisted HN labeling unavailable.")
        return 0
    if not os.path.exists(HIGHSCORE_FILE):
        print(f"No Highscore file at {HIGHSCORE_FILE}")
        return 0
    try:
        import config as cfg
        lo = max(0.0, float(cfg.MIN_AGRI_PROB) - float(getattr(cfg, 'NEG_LIKE_PROB_DELTA', 0.05)))
        hi = float(cfg.MIN_AGRI_PROB)
    except Exception:
        lo, hi = max(0.0, MIN_AGRI_PROB - 0.05), MIN_AGRI_PROB
    ndvi_abs = getattr(cfg, 'NEG_LIKE_NDVI_RANGE', (None, None))
    use_ndvi_abs = isinstance(ndvi_abs, (list, tuple)) and ndvi_abs[0] is not None and ndvi_abs[1] is not None

    # Quick on-screen summary: estimate how many HN candidates meet filters
    try:
        total_candidates = 0
        cap = 1_000_000  # safety cap for huge files
        import csv as _csv
        with open(HIGHSCORE_FILE, newline='') as f:
            rd = _csv.DictReader(f)
            for r in rd:
                try:
                    p = float(r.get('prob'))
                except Exception:
                    continue
                if not (lo <= p < hi):
                    continue
                if use_ndvi_abs:
                    try:
                        ndv = float(r.get('ndvi'))
                    except Exception:
                        ndv = None
                    if ndv is None or not (ndvi_abs[0] <= ndv <= ndvi_abs[1]):
                        continue
                total_candidates += 1
                if total_candidates >= cap:
                    break
        if total_candidates >= cap:
            print(f"HardNeg candidates (prob in [{lo:.2f},{hi:.2f}) + NDVI filter): >= {cap:,}")
        else:
            print(f"HardNeg candidates (prob in [{lo:.2f},{hi:.2f}) + NDVI filter): {total_candidates:,}")
    except Exception as _e:
        print(f"HN pre-scan skipped: {_e}")

    # Build full unique pool (prob/NDVI filters already applied)
    uniq = []
    seen = set()
    skipped = load_skipped_set()
    # Stream read to avoid loading the full CSV
    import csv as _csv
    with open(HIGHSCORE_FILE, newline='') as f:
        rd = _csv.DictReader(f)
        for r in rd:
            t = r.get('tile'); rr = r.get('row'); cc = r.get('col')
            try:
                p = float(r.get('prob'))
            except Exception:
                continue
            if not (lo <= p < hi):
                continue
            if use_ndvi_abs:
                try:
                    ndv = float(r.get('ndvi'))
                except Exception:
                    ndv = None
                if ndv is None or not (ndvi_abs[0] <= ndv <= ndvi_abs[1]):
                    continue
            try:
                key = f"{t}:{int(rr)}:{int(cc)}"
            except Exception:
                continue
            if key in seen or (skipped and key in skipped):
                continue
            seen.add(key)
            uniq.append(r)

    # Apply spatial diversity (round-robin across haversine clusters)
    def _diversify(cands):
        if not getattr(cfg, 'ASSISTED_SPATIAL_DIVERSITY_ENABLED', True) or not cands:
            return cands[:max_count]
        try:
            from sklearn.cluster import DBSCAN
            import numpy as _np
            eps_km = float(getattr(cfg, 'ASSISTED_DIVERSITY_EPS_KM', getattr(cfg, 'CANDIDATE_DBSCAN_EPS_KM', 1.5)))
            earth_km = 6371.0088
            # Prepare arrays
            lat = _np.array([float(r.get('lat')) for r in cands], dtype=float)
            lon = _np.array([float(r.get('lon')) for r in cands], dtype=float)
            coords = _np.vstack([_np.deg2rad(lat), _np.deg2rad(lon)]).T
            cl = DBSCAN(eps=eps_km/earth_km, min_samples=1, metric='haversine').fit(coords)
            labels = cl.labels_ if hasattr(cl, 'labels_') else _np.zeros((coords.shape[0],), dtype=int)
            # Order by prob desc (closer to threshold from below already enforced by selection)
            from collections import defaultdict
            buckets = defaultdict(list)
            for idx, cid in enumerate(labels.tolist()):
                buckets[cid].append(cands[idx])
            for cid in buckets:
                buckets[cid].sort(key=lambda r: float(r.get('prob', 0.0)), reverse=True)
            picks = []
            keys = list(buckets.keys())
            ptr = {cid:0 for cid in keys}
            while len(picks) < max_count and any(ptr[cid] < len(buckets[cid]) for cid in keys):
                for cid in keys:
                    p = ptr[cid]
                    if p < len(buckets[cid]):
                        picks.append(buckets[cid][p])
                        ptr[cid] = p + 1
                        if len(picks) >= max_count:
                            break
            try:
                print(f"[ASSISTED] HardNeg diversity: clusters={len(keys)}, picked={len(picks)} (eps_km={eps_km})")
            except Exception:
                pass
            return picks
        except Exception as _e:
            try: print(f"[ASSISTED] HardNeg diversity disabled due to error: {_e}")
            except Exception: pass
            return cands[:max_count]

    uniq = _diversify(uniq)

    # Pre-filter duplicates against existing labels before prompting (unified logic)
    try:
        label_index = _build_label_index()
        labels_all_pref = load_labels()
        filtered = []
        dropped = 0
        for r in uniq:
            t = r.get('tile')
            try:
                la = float(r.get('lat')); lo = float(r.get('lon'))
            except Exception:
                continue
            try:
                rr = int(r.get('row')); cc = int(r.get('col'))
            except Exception:
                rr = None; cc = None
            dup = _is_duplicate_candidate(t, la, lo, rr, cc, label_index)
            if not dup:
                try:
                    if duplicate_exists(la, lo, labels_all_pref):
                        dup = True
                except Exception:
                    pass
            if dup:
                dropped += 1
                continue
            filtered.append(r)
        if dropped:
            print(f"[ASSISTED] HardNeg: pre-filtered {dropped} near-duplicate candidates.")
        uniq = filtered
    except Exception as _e_pf:
        try:
            print(f"[ASSISTED] HardNeg: pre-filter step skipped due to error: {_e_pf}")
        except Exception:
            pass

    if not uniq:
        print("No hard negative candidates found with current filters.")
        return 0

    added = 0
    w, h = get_patch_dimensions()
    to_remove = []
    # For duplicate checks against existing labels
    labels_all = load_labels()

    for r in uniq:
        tile = r.get('tile')
        try:
            la = float(r.get('lat')); lo = float(r.get('lon'))
            row = int(r.get('row')); col = int(r.get('col'))
        except Exception:
            continue
        # Pre-prompt duplicate guard: never show a duplicate
        try:
            if _is_duplicate_candidate(tile, la, lo, row, col, label_index):
                # also guard using legacy proximity against labels snapshot
                try:
                    if duplicate_exists(la, lo, labels_all):
                        pass
                except Exception:
                    pass
                # Skip silently to avoid unnecessary console spam
                continue
        except Exception:
            pass
        # Show exact candidate pixel KML for review
        try:
            generate_kml_for_pixel(tile, row, col)
        except Exception:
            continue
        print(f"Open KML {CANDIDATE_KML} to view Hard Negative candidate.")
        print("Label? (1=Agri,2=Non,3=Skip): ")
        while True:
            ui = input("=> ").strip()
            if ui in ("1","2","3"):
                break
            print("Please type 1, 2, or 3.")
        if ui == "3":
            try:
                record_skipped_pixel(tile, row, col, la, lo, source="HardNeg")
            except Exception:
                pass
            continue
        lab = "Agricultural" if ui == "1" else "Non-Agricultural" if ui == "2" else None
        if not lab:
            print("Invalid choice.")
            continue
        # Safety guard in case a duplicate slipped past pre-prompt
        try:
            if _is_duplicate_candidate(tile, la, lo, row, col, label_index):
                print("Duplicate (pre-prompt guard). Skip.")
                continue
        except Exception:
            try:
                if duplicate_exists(la, lo, labels_all):
                    print("Duplicate (pre-prompt guard). Skip.")
                    continue
            except Exception:
                pass
        note = prompt_note()
        eid = f"HN_{int(random.random()*1e6)}"
        with open(LABELS_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([eid, la, lo, tile, lab, note])
        to_remove.append((tile, row, col, la, lo))
        print(f"Labeled Hard Negative: {tile} r={row},c={col}")
        added += 1
        # Update in-memory indices so subsequent candidates respect proximity
        try:
            if 'label_index' in locals() and isinstance(label_index, tuple) and len(label_index) == 2:
                label_index[0].add(f"{tile}:{int(row)}:{int(col)}")
                label_index[1].setdefault(tile, []).append((float(la), float(lo)))
        except Exception:
            pass

    if added:
        if to_remove:
            _batch_remove_pixels_from_lists(to_remove)
        _snap_labels_only()
        export_labels_kml()
    return added
