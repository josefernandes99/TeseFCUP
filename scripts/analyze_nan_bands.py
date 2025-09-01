#!/usr/bin/env python3
import os
import json
import glob
from typing import Tuple

import numpy as np
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shp_shape, Polygon, MultiPolygon
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from config import RAW_DATA_DIR, DATA_DIR, BASE_DIR
from progress_utils import new_progress


def _band_type(name: str) -> str:
    n = (name or "").upper()
    if any(k in n for k in ("NDVI", "NBR", "NDMI", "EVI", "EVI2")):
        return "index"
    if n in ("ELEVATION", "SLOPE", "ASPECT"):
        return "terrain"
    if n.startswith("B"):
        return "spectral"
    return "unknown"


def _suggest_for_nans(band_name: str, valid_min: float, valid_max: float, zeros: int, total: int) -> Tuple[str, str]:
    btype = _band_type(band_name)
    # Heuristics:
    # - Spectral bands: masked areas commonly 0; zeros are acceptable fill.
    # - Indices (NDVI/NBR/NDMI/EVI/EVI2): valid range roughly [-1, 1]; NaNs usually due to denom=0 → zeros are a safe neutral fill.
    # - Terrain: outside DEM coverage yields gaps; 0 is a safe fill (sea level, flat aspect/slope).
    if btype == "spectral":
        return ("zero", "Spectral band: masked/empty areas are typically 0; zeros are an appropriate fill for gaps.")
    if btype == "index":
        return ("zero", "Index band: NaNs arise from denominator=0; 0.0 is a neutral, bounded fill within [-1,1].")
    if btype == "terrain":
        if band_name.upper() == "ELEVATION":
            return ("zero", "Elevation gap: 0.0 (sea level) is a reasonable fill for missing DEM cells.")
        if band_name.upper() == "SLOPE":
            return ("zero", "Slope gap: 0.0 (flat) is a reasonable fill for missing DEM cells.")
        if band_name.upper() == "ASPECT":
            return ("zero", "Aspect gap: 0.0° is a conventional neutral angle when aspect is undefined.")
    # Unknown band: if zeros already common, suggest zero; else suggest inspect.
    if zeros > 0 and zeros / max(total, 1) > 0.01:
        return ("zero", "Unknown band: zeros are already common; using 0.0 for gaps is consistent.")
    return ("inspect", "Unknown band type; inspect source or choose a neutral fill consistent with valid range.")


def analyze_tiles(pattern=None, limit=None):
    tiles = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.tif")))
    if pattern:
        tiles = [t for t in tiles if pattern in os.path.basename(t)]
    if limit is not None:
        tiles = tiles[: int(limit)]
    report = []
    # Prepare KML doc for NaN polygons (three classes)
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom.minidom import parseString
    kml = Element('kml'); kml.set('xmlns','http://www.opengis.net/kml/2.2')
    doc = SubElement(kml, 'Document')
    # Styles: elevation (red), slope (blue), aspect (pink/magenta)
    def _add_style(doc, sid, line_color, fill_color):
        st = SubElement(doc, 'Style', id=sid)
        ln = SubElement(st, 'LineStyle'); SubElement(ln, 'color').text = line_color; SubElement(ln, 'width').text = '1'
        ps = SubElement(st, 'PolyStyle'); SubElement(ps, 'color').text = fill_color; SubElement(ps, 'outline').text = '1'
    # KML colors are ABGR (alpha, blue, green, red)
    _add_style(doc, 'nan_elev', 'ff0000ff', '400000ff')   # red
    _add_style(doc, 'nan_slope', 'ffff0000', '40ff0000')  # blue
    _add_style(doc, 'nan_aspect', 'ffff00ff', '40ff00ff') # magenta/pink

    total_polys = 0
    cnt_elev = cnt_slope = cnt_aspect = 0

    with new_progress() as prog:
        t = prog.add_task("Scanning tiles", total=len(tiles))
        for tp in tiles:
            try:
                with rasterio.open(tp) as src:
                    arr = src.read().astype(np.float32)
                    names = list(src.descriptions) if src.descriptions else []
                    H, W = src.height, src.width
                    total = int(H) * int(W)
                    nans = np.isnan(arr).sum(axis=(1, 2)).astype(int)
                    infs = np.isinf(arr).sum(axis=(1, 2)).astype(int)
                    zeros = np.count_nonzero(arr == 0, axis=(1, 2)).astype(int)
                    bands = []
                    for i in range(arr.shape[0]):
                        data = arr[i]
                        mask_valid = np.isfinite(data)
                        valid = data[mask_valid]
                        vmin = float(np.min(valid)) if valid.size else None
                        vmax = float(np.max(valid)) if valid.size else None
                        bname = names[i] if i < len(names) and names[i] else f"band_{i+1}"
                        suggestion, rationale = (None, None)
                        if int(nans[i]) > 0:
                            suggestion, rationale = _suggest_for_nans(bname, vmin if vmin is not None else 0.0,
                                                                      vmax if vmax is not None else 0.0,
                                                                      int(zeros[i]), total)
                        bands.append({
                            "index": i + 1,
                            "name": bname,
                            "type": _band_type(bname),
                            "nans": int(nans[i]),
                            "infs": int(infs[i]),
                            "zeros": int(zeros[i]),
                            "total": total,
                            "valid_min": vmin,
                            "valid_max": vmax,
                            "suggest_fill": suggestion,
                            "rationale": rationale,
                        })
                    # Identify band indices for terrain
                    names_up = [ (n or '').upper() for n in names ]
                    def _find_idx(label, fallback):
                        return names_up.index(label) if label in names_up else fallback
                    # Fallback to last 3 bands if names are missing
                    elev_idx = _find_idx('ELEVATION', arr.shape[0]-3)
                    slope_idx = _find_idx('SLOPE', arr.shape[0]-2)
                    aspect_idx = _find_idx('ASPECT', arr.shape[0]-1)
                    # Masks for NaNs (strict: NaN only, not Inf)
                    elev_nan = np.isnan(arr[elev_idx])
                    slope_nan = np.isnan(arr[slope_idx])
                    aspect_nan = np.isnan(arr[aspect_idx])
                    # Polygonize each class separately into contiguous regions
                    def _emit_regions(mask, style_id):
                        nonlocal total_polys, cnt_elev, cnt_slope, cnt_aspect
                        if not mask.any():
                            return
                        transformer = None
                        if src.crs and not src.crs.is_geographic:
                            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                        for geom, val in rio_shapes(mask.astype('uint8'), mask=mask, transform=src.transform):
                            if val != 1:
                                continue
                            poly = shp_shape(geom)
                            if transformer:
                                poly = shp_transform(transformer.transform, poly)
                            def emit_polygon(p):
                                coords = list(p.exterior.coords)
                                pm = SubElement(doc, 'Placemark')
                                SubElement(pm, 'styleUrl').text = f'#{style_id}'
                                poly_el = SubElement(pm, 'Polygon')
                                ob = SubElement(poly_el, 'outerBoundaryIs')
                                lr = SubElement(ob, 'LinearRing')
                                SubElement(lr, 'coordinates').text = ' '.join(f"{lon},{lat},0" for lon, lat in coords)
                            if isinstance(poly, MultiPolygon):
                                for sub in poly.geoms:
                                    emit_polygon(sub)
                                    total_polys += 1
                                    if style_id=='nan_elev': cnt_elev += 1
                                    elif style_id=='nan_slope': cnt_slope += 1
                                    else: cnt_aspect += 1
                            elif isinstance(poly, Polygon):
                                emit_polygon(poly)
                                total_polys += 1
                                if style_id=='nan_elev': cnt_elev += 1
                                elif style_id=='nan_slope': cnt_slope += 1
                                else: cnt_aspect += 1

                    _emit_regions(elev_nan, 'nan_elev')
                    _emit_regions(slope_nan, 'nan_slope')
                    _emit_regions(aspect_nan, 'nan_aspect')
                report.append({
                    "tile": os.path.basename(tp),
                    "bands": bands,
                })
            except Exception as e:
                report.append({"tile": os.path.basename(tp), "error": str(e)})
            prog.update(t, advance=1)
    out_dir = os.path.join(DATA_DIR, "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "nan_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"NaN/Inf/Zero report => {out_path}")
    # Write KML at project root
    try:
        kml_path = os.path.join(BASE_DIR, 'detected_nan_pixels.kml')
        xml = parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
        with open(kml_path, 'wb') as f:
            f.write(xml)
        print(f"NaN KML polygons => total={total_polys}, elevation={cnt_elev}, slope={cnt_slope}, aspect={cnt_aspect} @ {kml_path}")
    except Exception as e:
        print(f"NaN KML generation failed: {e}")
    # Console summaries
    total_tiles = len([r for r in report if 'bands' in r])
    tiles_with_nans = sum(1 for r in report if any(b.get('nans', 0) > 0 for b in r.get('bands', [])))
    print(f"Tiles analyzed: {total_tiles}; tiles with NaNs in any band: {tiles_with_nans}")
    print("\nPer-tile NaN bands:")
    for r in report:
        if 'bands' not in r:
            continue
        nan_bands = [f"{b['name']}[#{b['index']}] (NaNs={b['nans']})" for b in r['bands'] if b.get('nans', 0) > 0]
        if nan_bands:
            print(f" - {r['tile']}: {', '.join(nan_bands)}")
        else:
            print(f" - {r['tile']}: none")
    # Per-band aggregation across tiles
    agg = {}
    for r in report:
        if 'bands' not in r:
            continue
        for b in r['bands']:
            name = b.get('name') or f"band_{b.get('index','?')}"
            a = agg.setdefault(name, {"type": b.get('type','unknown'), "tiles_with_nans": 0, "total_nans": 0})
            if b.get('nans', 0) > 0:
                a["tiles_with_nans"] += 1
                a["total_nans"] += b.get('nans', 0)
    print("\nPer-band summary across tiles (sorted by tiles_with_nans desc):")
    for name, stats in sorted(agg.items(), key=lambda kv: kv[1]['tiles_with_nans'], reverse=True):
        print(f" - {name} [{stats['type']}]: tiles_with_nans={stats['tiles_with_nans']}, total_nans={stats['total_nans']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=None, help="Substring to filter tile names")
    ap.add_argument("--limit", default=None, help="Max tiles to scan")
    args = ap.parse_args()
    analyze_tiles(pattern=args.pattern, limit=args.limit)
