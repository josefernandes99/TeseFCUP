#!/usr/bin/env python3
"""Recover labels/phase1/highscore.csv from a saved NPY/NPZ or KML.

Usage:
  python scripts/recover_highscore.py [--npy PATH] [--kml PATH] [--out PATH]

Defaults:
  --out defaults to config.HIGHSCORE_FILE
  --kml defaults to config.HIGHSCORE_KML_GLOBAL (if not provided)

Notes:
  - NPY/NPZ is preferred because it can contain probabilities and scores.
  - KML fallback reconstructs pixel positions (tile,row,col,lat,lon) by
    intersecting polygons with each tile; prob/ndvi/score will be empty.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from progress_utils import new_progress
from rich.console import Console

sys.path.append(os.path.dirname(__file__))
import config as cfg  # noqa: E402


console = Console()


def _iter_tiles() -> List[str]:
    from glob import glob
    patt = os.path.join(cfg.RAW_DATA_DIR, "*.tif")
    return sorted([os.path.basename(p) for p in glob(patt)])


def _tile_bounds_wgs84(tile: str) -> Optional[Tuple[float, float, float, float]]:
    import rasterio
    from pyproj import Transformer
    path = os.path.join(cfg.RAW_DATA_DIR, tile)
    if not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            left, bottom, right, top = src.bounds
            if src.crs and not src.crs.is_geographic:
                to_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                x0, y0 = to_ll.transform(left, bottom)
                x1, y1 = to_ll.transform(right, top)
                lon_min, lon_max = (min(x0, x1), max(x0, x1))
                lat_min, lat_max = (min(y0, y1), max(y0, y1))
                return (lon_min, lat_min, lon_max, lat_max)
            return (left, bottom, right, top)
    except Exception:
        return None


def _point_in_tile(tile: str, lat: float, lon: float) -> bool:
    b = _tile_bounds_wgs84(tile)
    if not b:
        return False
    lon_min, lat_min, lon_max, lat_max = b
    return (lon_min <= lon <= lon_max) and (lat_min <= lat <= lat_max)


def _rowcol_from_latlon(tile: str, lat: float, lon: float) -> Optional[Tuple[int, int]]:
    import rasterio
    from pyproj import Transformer
    path = os.path.join(cfg.RAW_DATA_DIR, tile)
    if not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            x, y = lon, lat
            if src.crs and not src.crs.is_geographic:
                to_proj = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = to_proj.transform(lon, lat)
            r, c = rasterio.transform.rowcol(src.transform, x, y)
            if 0 <= r < src.height and 0 <= c < src.width:
                return int(r), int(c)
    except Exception:
        return None
    return None


def _latlon_center(tile: str, row: int, col: int) -> Optional[Tuple[float, float]]:
    import rasterio
    from pyproj import Transformer
    path = os.path.join(cfg.RAW_DATA_DIR, tile)
    if not os.path.exists(path):
        return None
    try:
        with rasterio.open(path) as src:
            cx, cy = rasterio.transform.xy(src.transform, row, col, offset="center")
            if src.crs and not src.crs.is_geographic:
                to_ll = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                cx, cy = to_ll.transform(cx, cy)
            return float(cy), float(cx)
    except Exception:
        return None


def _load_global_npy(npy_path: str) -> List[Dict[str, object]]:
    """Best-effort loader for a global Highscore NPY/NPZ.

    Accepts various layouts:
      - Structured array with named fields
      - 2D object array with 8 columns (tile,row,col,lat,lon,prob,ndvi,score)
      - 2D numeric array with 7 columns (row,col,lat,lon,prob,ndvi,score)
        -> tile inferred from lat/lon by scanning tiles
    """
    rows: List[Dict[str, object]] = []
    arr = None
    if npy_path.lower().endswith(".npz"):
        data = np.load(npy_path, allow_pickle=True)
        # Heuristic: pick the largest array inside
        name = max(data.files, key=lambda k: np.prod(data[k].shape) if hasattr(data[k], "shape") else 0)
        arr = data[name]
    else:
        arr = np.load(npy_path, allow_pickle=True)

    tiles = _iter_tiles()
    # Build bounds with a visible progress bar
    bounds = {}
    with new_progress() as prog:
        t_b = prog.add_task("Scan tile bounds", total=len(tiles))
        for t in tiles:
            bounds[t] = _tile_bounds_wgs84(t)
            prog.update(t_b, advance=1)

    def _infer_tile(lat: float, lon: float) -> Optional[str]:
        for t in tiles:
            b = bounds.get(t)
            if not b:
                continue
            lon_min, lat_min, lon_max, lat_max = b
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                return t
        return None

    def _emit(tile: str, r: int, c: int, lat: float, lon: float,
              prob: Optional[float], ndvi: Optional[float], score: Optional[float]):
        rows.append({
            "tile": tile,
            "row": int(r),
            "col": int(c),
            "lat": f"{float(lat):.7f}",
            "lon": f"{float(lon):.7f}",
            **({"prob": float(prob)} if prob is not None else {}),
            **({"ndvi": float(ndvi)} if ndvi is not None else {}),
            **({"score": float(score)} if score is not None else {}),
        })

    # Case 1: structured array with named fields
    if getattr(arr, "dtype", None) is not None and getattr(arr.dtype, "names", None):
        names = {n.lower(): n for n in arr.dtype.names}
        need = {"tile", "row", "col", "lat", "lon"}
        if not need.issubset(set(k.lower() for k in arr.dtype.names)):
            raise ValueError("Structured NPY missing required fields: tile,row,col,lat,lon")
        with new_progress() as prog:
            t_s = prog.add_task("Read structured NPY records", total=len(arr))
            for rec in arr:
                tile = str(rec[names["tile"]])
                r = int(rec[names["row"]]); c = int(rec[names["col"]])
                lat = float(rec[names["lat"]]); lon = float(rec[names["lon"]])
                prob = float(rec[names["prob"]]) if "prob" in names else None
                ndvi = float(rec[names["ndvi"]]) if "ndvi" in names else None
                score = float(rec[names["score"]]) if "score" in names else None
                _emit(tile, r, c, lat, lon, prob, ndvi, score)
                prog.update(t_s, advance=1)
        return rows

    # Case 2/3: rank-2 array (object or numeric)
    if getattr(arr, "ndim", 0) != 2 or arr.shape[1] not in (7, 8):
        raise ValueError("Unsupported NPY shape: expected Nx7 or Nx8 (got %s)" % (arr.shape,))

    with new_progress() as prog:
        t_r = prog.add_task("Read NPY rows", total=arr.shape[0])
        for i in range(arr.shape[0]):
            rec = arr[i]
            try:
                if arr.shape[1] == 8:
                    tile, r, c, lat, lon, prob, ndvi, score = rec
                    tile = str(tile)
                else:
                    r, c, lat, lon, prob, ndvi, score = rec
                    r, c, lat, lon = int(r), int(c), float(lat), float(lon)
                    tile = _infer_tile(lat, lon)
                    if not tile:
                        prog.update(t_r, advance=1)
                        continue
                r = int(r); c = int(c); lat = float(lat); lon = float(lon)
                prob = float(prob) if prob is not None else None
                ndvi = float(ndvi) if ndvi is not None else None
                score = float(score) if score is not None else None
            except Exception:
                prog.update(t_r, advance=1)
                continue
            _emit(tile, r, c, lat, lon, prob, ndvi, score)
            prog.update(t_r, advance=1)
    return rows


def _load_raw_npy_topk(npy_path: str, top_k: int) -> List[Dict[str, object]]:
    """Fallback for raw float32 dumps without .npy header.

    Assumes per-record layout of 7 or 8 float32 values in little-endian order.
    Preferred 7-float layout: [row, col, lat, lon, prob, ndvi, score].
    """
    import numpy as _np
    import heapq as _hq

    sz = os.path.getsize(npy_path)
    if sz % 4 != 0:
        raise ValueError("File size is not a multiple of 4 bytes; not a float32 dump.")
    n_floats = sz // 4
    rec_len = None
    for cand in (7, 8):
        if n_floats % cand == 0:
            rec_len = cand
            break
    if rec_len is None:
        raise ValueError(f"Raw NPY fallback: float count {n_floats} not divisible by 7 or 8.")

    n_rows = n_floats // rec_len
    console.print(f"[cyan]Raw-NPY fallback:[/cyan] {n_rows:,} rows × {rec_len} floats")
    mm = _np.memmap(npy_path, dtype='<f4', mode='r')
    block_rows = min(2_000_000, n_rows)
    heap: list[tuple[float, tuple]] = []  # (score, (row,col,lat,lon,prob,ndvi))
    score_idx = 6  # both layouts store score at index 6 in practice

    with new_progress() as prog:
        t = prog.add_task("Scan raw NPY blocks", total=n_rows)
        for start in range(0, n_rows, block_rows):
            end = min(n_rows, start + block_rows)
            seg = mm[start*rec_len:end*rec_len]
            blk = seg.reshape(-1, rec_len)
            # score column is index 6. Guard against short rows.
            if blk.shape[1] <= score_idx:
                prog.update(t, advance=(end-start))
                continue
            scores = blk[:, score_idx]
            if blk.shape[0] > top_k:
                idx = _np.argpartition(-scores, top_k-1)[:top_k]
                cand = blk[idx]
                cand_scores = scores[idx]
            else:
                cand = blk
                cand_scores = scores
            for i in range(cand.shape[0]):
                r, c, la, lo, pr, nd, sc = cand[i, 0], cand[i, 1], cand[i, 2], cand[i, 3], cand[i, 4], cand[i, 5], cand_scores[i]
                item = (int(r), int(c), float(la), float(lo), float(pr), float(nd))
                if len(heap) < top_k:
                    _hq.heappush(heap, (float(sc), item))
                else:
                    if sc > heap[0][0]:
                        _hq.heapreplace(heap, (float(sc), item))
            prog.update(t, advance=(end-start))

    # Extract winners sorted by score desc
    winners = [ _hq.heappop(heap) for _ in range(len(heap)) ]
    winners.sort(key=lambda t: t[0], reverse=True)

    # Infer tile names for winners
    tiles = _iter_tiles()
    bounds = {t: _tile_bounds_wgs84(t) for t in tiles}
    def _infer_tile(lat: float, lon: float) -> Optional[str]:
        for tn, b in bounds.items():
            if not b: continue
            xmin, ymin, xmax, ymax = b[0], b[1], b[2], b[3]
            if xmin <= lon <= xmax and ymin <= lat <= ymax:
                return tn
        return None

    rows: List[Dict[str, object]] = []
    with new_progress() as prog:
        t2 = prog.add_task("Infer tiles for winners", total=len(winners))
        for sc, (r, c, la, lo, pr, nd) in winners:
            tile = _infer_tile(la, lo) or ""
            rows.append({
                "tile": tile,
                "row": int(r),
                "col": int(c),
                "lat": f"{float(la):.7f}",
                "lon": f"{float(lo):.7f}",
                "prob": float(pr),
                "ndvi": float(nd),
                "score": float(sc),
            })
            prog.update(t2, advance=1)
    return rows


def _parse_kml_polygons(kml_path: str) -> List[List[Tuple[float, float]]]:
    """Return list of WGS84 polygon rings (lon,lat) from a KML with polygons.

    This assumes the format produced by _write_ranked_pixel_kml or similar.
    """
    import xml.etree.ElementTree as ET

    rings: List[List[Tuple[float, float]]] = []
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
    except Exception as e:
        raise RuntimeError(f"Failed to parse KML: {e}")

    # KML namespace handling
    ns = "{http://www.opengis.net/kml/2.2}"
    for lr in root.iter(f"{ns}LinearRing"):
        coords_el = lr.find(f"{ns}coordinates")
        if coords_el is None or not coords_el.text:
            continue
        pts = []
        for tok in coords_el.text.strip().split():
            try:
                lon_s, lat_s, *_ = tok.split(',')
                pts.append((float(lon_s), float(lat_s)))
            except Exception:
                continue
        if len(pts) >= 4:
            rings.append(pts)
    return rings


def _recover_from_kml(kml_path: str) -> List[Dict[str, object]]:
    """Reconstruct pixel list by intersecting KML polygons with each tile grid.

    Outputs minimal fields: tile,row,col,lat,lon (prob/ndvi/score omitted).
    """
    from shapely.geometry import Polygon
    from shapely.ops import transform as shp_transform
    import rasterio
    from rasterio.features import rasterize
    from pyproj import Transformer

    console.print(f"[cyan]Parsing KML rings from {kml_path}...[/cyan]")
    rings = _parse_kml_polygons(kml_path)
    if not rings:
        return []
    # Build shapely polygons in WGS84 with progress
    polys_wgs = []
    with new_progress() as prog:
        t_p = prog.add_task("Build polygons", total=len(rings))
        for ring in rings:
            try:
                poly = Polygon(ring)
                if not poly.is_valid or poly.is_empty:
                    prog.update(t_p, advance=1)
                    continue
                polys_wgs.append(poly)
            except Exception:
                pass
            prog.update(t_p, advance=1)

    out: List[Dict[str, object]] = []
    tiles = _iter_tiles()
    console.print(f"[cyan]Rasterizing polygons over {len(tiles)} tiles...[/cyan]")
    with new_progress() as prog:
        t_t = prog.add_task("Rasterize tiles", total=len(tiles))
        for tile in tiles:
            tif = os.path.join(cfg.RAW_DATA_DIR, tile)
            try:
                with rasterio.open(tif) as src:
                    # Build polygons in tile CRS
                    to_proj = None
                    if src.crs and not src.crs.is_geographic:
                        to_proj = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                        polys_proj = [shp_transform(to_proj.transform, p) for p in polys_wgs]
                    else:
                        polys_proj = polys_wgs
                    # Rasterize all polygons at once
                    if not polys_proj:
                        prog.update(t_t, advance=1)
                        continue
                    shapes = [(p, 1) for p in polys_proj if not p.is_empty]
                    mask = rasterize(
                        shapes=shapes,
                        out_shape=(src.height, src.width),
                        transform=src.transform,
                        fill=0,
                        dtype="uint8",
                        all_touched=False,
                    )
                    ys, xs = np.nonzero(mask)
                    for r, c in zip(ys.tolist(), xs.tolist()):
                        latlon = _latlon_center(tile, r, c)
                        if not latlon:
                            continue
                        lat, lon = latlon
                        out.append({
                            "tile": tile,
                            "row": int(r),
                            "col": int(c),
                            "lat": f"{float(lat):.7f}",
                            "lon": f"{float(lon):.7f}",
                        })
            except Exception:
                pass
            prog.update(t_t, advance=1)
    return out


def main():
    ap = argparse.ArgumentParser(description="Recover highscore.csv from NPY/KML")
    ap.add_argument("--npy", dest="npy", help="Path to global highscore .npy/.npz", default=None)
    ap.add_argument("--kml", dest="kml", help="Path to highscore_top.kml", default=None)
    ap.add_argument("--out", dest="out", help="Output CSV path (default: config.HIGHSCORE_FILE)", default=None)
    args = ap.parse_args()

    out_path = args.out or cfg.HIGHSCORE_FILE
    npy_path = args.npy
    # If no --npy provided, try default locations under labels dir
    if not npy_path:
        try:
            default_npy = os.path.join(cfg.LABELS_DIR, 'highscore.npy')
            default_npz = os.path.join(cfg.LABELS_DIR, 'highscore.npz')
        except Exception:
            default_npy = os.path.join(os.path.dirname(cfg.HIGHSCORE_FILE), 'highscore.npy')
            default_npz = os.path.join(os.path.dirname(cfg.HIGHSCORE_FILE), 'highscore.npz')
        if os.path.exists(default_npy):
            npy_path = default_npy
            console.print(f"[cyan]Using default NPY:[/cyan] {npy_path}")
        elif os.path.exists(default_npz):
            npy_path = default_npz
            console.print(f"[cyan]Using default NPZ:[/cyan] {npy_path}")
    kml_path = args.kml or (getattr(cfg, "HIGHSCORE_KML_GLOBAL", None) or None)

    rows: List[Dict[str, object]] = []

    # Prefer NPY if provided and exists
    if npy_path and os.path.exists(npy_path):
        console.print(f"[cyan]Loading NPY/NPZ: {npy_path}[/cyan]")
        try:
            rows = _load_global_npy(npy_path)
            if rows:
                console.print(f"[green]Recovered {len(rows):,} rows from NPY[/green]")
        except Exception as e:
            console.print(f"[yellow]NPY structured load failed:[/yellow] {e}")
            # Try raw-float fallback (top-K only)
            try:
                top_k = int(getattr(cfg, 'HIGHSCORE_TOP_K', 10000) or 10000)
                console.print(f"[cyan]Trying raw-NPY fallback (top-{top_k})...[/cyan]")
                rows = _load_raw_npy_topk(npy_path, top_k=top_k)
                console.print(f"[green]Recovered {len(rows):,} rows from raw NPY[/green]")
            except Exception as e2:
                console.print(f"[red]Raw NPY fallback failed:[/red] {e2}")

    # Fallback to KML
    if not rows and kml_path and os.path.exists(kml_path):
        try:
            rows = _recover_from_kml(kml_path)
            if rows:
                console.print(f"[green]Recovered {len(rows):,} rows from KML[/green]")
            else:
                console.print("[yellow]KML contained no recoverable polygons.[/yellow]")
        except Exception as e:
            console.print(f"[red]KML recovery failed:[/red] {e}")

    if not rows:
        console.print("[red]No rows recovered.[/red] Provide a valid --npy or --kml path.")
        sys.exit(2)

    # Deduplicate by (tile,row,col)
    console.print("[cyan]Deduplicating rows by (tile,row,col)...[/cyan]")
    uniq: Dict[str, Dict[str, object]] = {}
    with new_progress() as prog:
        t_d = prog.add_task("Deduplicate", total=len(rows))
        for r in rows:
            try:
                key = f"{r.get('tile')}:{int(r.get('row'))}:{int(r.get('col'))}"
            except Exception:
                prog.update(t_d, advance=1)
                continue
            uniq[key] = r
            prog.update(t_d, advance=1)
    rows = list(uniq.values())

    # Compose fieldnames
    base_fields = ["tile", "row", "col", "lat", "lon"]
    extra = []
    for k in ("prob", "ndvi", "score", "entropy", "times_selected", "first_round", "last_round"):
        if any(k in r for r in rows):
            extra.append(k)
    fields = base_fields + extra

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    console.print(f"[cyan]Writing CSV → {out_path}[/cyan]")
    with new_progress() as prog:
        t_w = prog.add_task("Write CSV", total=len(rows))
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
                prog.update(t_w, advance=1)
    console.print(f"[green]Highscore CSV recovered[/green] → {out_path} ([bold]{len(rows):,}[/bold] rows)")


if __name__ == "__main__":
    main()
