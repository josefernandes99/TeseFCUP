# scripts/a6_phase3_postprocessing.py

import os
import glob
import csv
import numpy as np
import rasterio
from joblib import load, Parallel, delayed
from rich.progress import (
    Progress,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from memory_watcher import free_unused_memory
from config import (
    RAW_DATA_DIR,
    ROUNDS_DIR,
    MIN_AGRI_PROB,
    CANDIDATE_PROB_LOWER,
    SIEVE_MIN_SIZE,
    SIEVE_KEEP_MODE,
    FINAL_SWEEP_ENABLED,
    FINAL_THRESHOLDS,
    FINAL_SIEVE_SIZES,
    FINAL_MORPH_OPEN,
    FINAL_MORPH_KERNEL_SIZES,
)
from features import add_derived_features
from scipy.ndimage import binary_closing, binary_fill_holes, label as ndlabel
from rasterio.features import sieve, shapes
import config as cfg

# Note: small-patch filtering via rasterio.sieve is now performed during each
# active learning round. The final postprocessing step simply runs the last
# model and saves geotiffs without additional sieving.

def get_final_model_path():
    round_folders = glob.glob(os.path.join(ROUNDS_DIR, "round_*"))
    if not round_folders:
        print("No round folders => cannot load final model.")
        return None
    rounds = []
    for f in round_folders:
        base = os.path.basename(f)
        try:
            num = int(base.split("_")[1])
            rounds.append((num, f))
        except Exception:
            continue
    if not rounds:
        print("No valid round folders => no final model.")
        return None
    rounds.sort(key=lambda x: x[0])
    final_num, final_r = rounds[-1]
    # Models may be stored within subdirectories (e.g., "manual" or grid-search
    # combo names). Search recursively for the expected model file inside the
    # last round folder.
    pattern = os.path.join(final_r, "**", f"model_round_{final_num}.pkl")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        print("Model file not found =>", os.path.join(final_r, f"model_round_{final_num}.pkl"))
        return None
    return matches[0]

def classify_tile(tile_path, model, th=None, sieve_size=None, morph_open=False, morph_k=3):
    """
    1) Reads the multi-band tile.
    2) Predicts class (0/1) for every pixel.
    Returns the single‐band prediction array and original profile.
    """
    with rasterio.open(tile_path) as src:
        raw = src.read().astype(np.float32)
        img, _ = add_derived_features(raw)
        b, h, w = img.shape
        X = img.reshape(b, -1).T        # (h*w, bands)
        # Chunked inference using config limits if available
        try:
            import config as cfg
            if getattr(cfg, 'INFER_CHUNKING_ENABLED', False):
                probs_flat = np.empty((X.shape[0],), dtype=np.float32)
                bs = int(getattr(cfg, 'INFER_MAX_PIXELS_PER_BATCH', 200_000))
                for i in range(0, X.shape[0], bs):
                    probs_flat[i:i+bs] = model.predict_proba(X[i:i+bs])[:,1].astype(np.float32)
                probs = probs_flat.reshape(h, w)
            else:
                probs = model.predict_proba(X)[:,1].reshape(h, w)
        except Exception:
            probs = model.predict_proba(X)[:,1].reshape(h, w)
        thr = MIN_AGRI_PROB if th is None else th
        ssz = SIEVE_MIN_SIZE if sieve_size is None else sieve_size
        crop = probs >= thr
        uncertain = (probs >= CANDIDATE_PROB_LOWER) & (probs < thr)
        mask = crop | uncertain
        if morph_open:
            from scipy.ndimage import binary_opening
            mask = binary_opening(mask, structure=np.ones((morph_k, morph_k)))
        mask = binary_fill_holes(binary_closing(mask))
        lbl, num = ndlabel(mask)
        for i in range(1, num+1):
            comp = (lbl==i)
            if SIEVE_KEEP_MODE == 'component':
                # remove components with no high-prob pixel
                if not np.any(crop[comp]):
                    mask[comp] = False
            else:
                # keep only high-prob pixels, drop the rest of the component
                if not np.any(crop[comp]):
                    mask[comp] = False
                else:
                    mask[comp] = False
                    mask[comp & crop] = True
        if ssz > 0:
            mask = sieve(mask.astype("uint8"), size=int(ssz), connectivity=8).astype(bool)
        cleaned = (mask & (probs >= thr)).astype(np.uint8)
        return cleaned, src.profile

def save_geotiff(path_out, data, profile):
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    with rasterio.open(path_out, "w", **profile) as dst:
        dst.write(data, 1)
    # Live update (console) with base filename; full sequential logs preserved
    try:
        base = os.path.basename(path_out)
    except Exception:
        base = path_out
    print(f"\rSaved geotiff => {base}", end="", flush=True)

def process_tile(tfile, model, th=None, sieve_size=None, morph_open=False, morph_k=3, suffix=None, out_dir=None):
    """Classify one tile and save overlay GeoTIFF."""
    # Live update on console; full sequence preserved in logs via tee CR→NL translation
    print(f"\rClassifying => {os.path.basename(tfile)}", end="", flush=True)
    cleaned, prof = classify_tile(tfile, model, th=th, sieve_size=sieve_size, morph_open=morph_open, morph_k=morph_k)
    n_pix = cleaned.size
    n_agri = int(cleaned.sum())
    pct_agri = (n_agri / n_pix) * 100 if n_pix else 0.0
    if suffix is None:
        suffix = "overlay"
    base = os.path.splitext(os.path.basename(tfile))[0]
    if out_dir is None:
        out_dir = os.path.dirname(tfile)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base}_{suffix}.tif")
    save_geotiff(out_path, cleaned, prof)
    free_unused_memory()
    return os.path.basename(tfile), pct_agri, n_pix, n_agri

def postprocessing():
    print("Starting postprocessing...")
    mp = get_final_model_path()
    if mp is None:
        return
    model = load(mp)
    tile_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.tif"))
    if not tile_files:
        print("No tiles found in raw data directory.")
        return
    # Prepare final-round root directory
    final_root = os.path.join(ROUNDS_DIR, "final_round")
    os.makedirs(final_root, exist_ok=True)

    summary = []
    total_pixels = 0
    total_agri_pixels = 0
    pct_list = []

    with Progress(
        "[bold cyan]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        # Either a single pass with base settings, or sweep across combos
        if not FINAL_SWEEP_ENABLED:
            task = progress.add_task("Classifying tiles", total=len(tile_files))
            def wrapped(tp):
                # save overlays under final_round/default/overlays
                overlays_dir = os.path.join(final_root, "default", "overlays")
                res = process_tile(tp, model, out_dir=overlays_dir)
                progress.update(task, advance=1)
                return res
            results = Parallel(n_jobs=-1, prefer="threads")(delayed(wrapped)(tp) for tp in tile_files)
            combo_tag = "default"
            results_map = {combo_tag: results}
        else:
            results_map = {}
            combos = []
            for th in FINAL_THRESHOLDS:
                for sz in FINAL_SIEVE_SIZES:
                    if FINAL_MORPH_OPEN:
                        for mk in FINAL_MORPH_KERNEL_SIZES:
                            combos.append((th, sz, True, mk))
                    else:
                        combos.append((th, sz, False, 3))
            task = progress.add_task("Final sweep", total=len(tile_files)*len(combos))
            for th, sz, mopen, mk in combos:
                tag = f"th{th}_s{sz}" + (f"_m{mk}" if mopen else "")
                def wrapped(tp, th=th, sz=sz, mopen=mopen, mk=mk, tag=tag):
                    overlays_dir = os.path.join(final_root, tag, "overlays")
                    res = process_tile(tp, model, th=th, sieve_size=sz, morph_open=mopen, morph_k=mk, suffix=tag, out_dir=overlays_dir)
                    progress.update(task, advance=1)
                    return res
                res = Parallel(n_jobs=-1, prefer="threads")(delayed(wrapped)(tp) for tp in tile_files)
                results_map[tag] = res

    # Write per-combo summaries into rounds/final_round/<tag>/
    for tag, results in results_map.items():
        combo_dir = os.path.join(final_root, tag if tag else "default")
        os.makedirs(combo_dir, exist_ok=True)
        stats_dir = os.path.join(combo_dir, "statistics")
        os.makedirs(stats_dir, exist_ok=True)
        summary = []
        total_pixels = 0
        total_agri_pixels = 0
        pct_list = []
        for tile, pct_agri, n_pix, n_agri in results:
            summary.append([tile, f"{pct_agri:.2f}", n_pix, n_agri])
            pct_list.append(pct_agri)
            total_pixels += n_pix
            total_agri_pixels += n_agri
        csv_path = os.path.join(combo_dir, "final_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tile", "pct_agricultural", "pixels_total", "pixels_agricultural"])
            writer.writerows(summary)
        # global statistics
        if pct_list:
            overall_pct = (total_agri_pixels / total_pixels) * 100 if total_pixels else 0
            avg_pct = float(np.mean(pct_list))
            std_pct = float(np.std(pct_list))
            min_pct = float(np.min(pct_list))
            max_pct = float(np.max(pct_list))
        else:
            overall_pct = avg_pct = std_pct = min_pct = max_pct = 0.0
        stats_path = os.path.join(combo_dir, "final_summary.txt")
        with open(stats_path, "w") as sf:
            sf.write(f"Tiles processed: {len(tile_files)}\n")
            sf.write(f"Total pixels: {total_pixels}\n")
            sf.write(f"Total agricultural pixels: {total_agri_pixels}\n")
            sf.write(f"Overall agricultural %: {overall_pct:.2f}\n")
            sf.write(f"Average tile %: {avg_pct:.2f}\n")
            sf.write(f"Std dev tile %: {std_pct:.2f}\n")
            sf.write(f"Min tile %: {min_pct:.2f}\n")
            sf.write(f"Max tile %: {max_pct:.2f}\n")
        # Metrics and plots similar to per-round statistics
        # Seed threshold variables before try/finally so linters see them as defined.
        old_th = cfg.MIN_AGRI_PROB
        th_local = old_th
        try:
            from evaluation import evaluate_model_repeated as evaluate_model
            # Determine threshold from tag if present
            if tag and isinstance(tag, str) and tag.startswith("th"):
                try:
                    th_local = float(tag.split("_")[0][2:])
                except Exception:
                    th_local = old_th
            cfg.MIN_AGRI_PROB = th_local
            _ = evaluate_model(model, out_dir=stats_dir)
        finally:
            try:
                cfg.MIN_AGRI_PROB = old_th
            except Exception:
                pass
        # Permutation feature importance with names
        try:
            if getattr(cfg, "RUN_PERMUTATION_IMPORTANCE", False):
                # Build eval set as in evaluation.py
                from al_shared import extract_features_from_label
                from splits import load_labels as _load, stratified_train_val_test_indices as _split
                from sklearn.inspection import permutation_importance as _pi
                from sklearn.metrics import get_scorer as _get_scorer
                from features import current_feature_names as _names
                all_rows = []
                if os.path.exists(cfg.LABELS_FILE):
                    all_rows.extend(_load(cfg.LABELS_FILE))
                if os.path.exists(cfg.TEMP_LABELS_FILE):
                    all_rows.extend(_load(cfg.TEMP_LABELS_FILE))
                # dedup
                dedup, seen = [], set()
                for r in all_rows:
                    k = f"{r.get('tile')}:{r.get('lat')}:{r.get('lon')}"
                    if k in seen: continue
                    seen.add(k); dedup.append(r)
                X, y = [], []
                for r in dedup:
                    f = extract_features_from_label(r)
                    if f is None: continue
                    X.append(f); y.append(1 if r.get("label","").lower()=="agricultural" else 0)
                X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int64)
                if X.size:
                    tr, va, _ = _split(y, cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.TEST_FRACTION,
                                       cfg.SPLIT_RANDOM_SEED if cfg.SPLIT_SEED_MODE=="fixed" else None)
                    if va.size:
                        scorer = _get_scorer('f1')
                        pi = _pi(model, X[va], y[va], scoring=scorer, n_repeats=5, n_jobs=-1, random_state=0)
                        importances = pi.importances_mean
                        order = np.argsort(importances)[::-1]
                        exp_names = _names()
                        if len(exp_names) != importances.size:
                            raise RuntimeError(f"Final FI naming mismatch: expected {importances.size}, have {len(exp_names)}")
                        names = exp_names
                        with open(os.path.join(stats_dir, 'feature_importance.txt'), 'w') as f:
                            for idx in order:
                                f.write(f"{names[idx]}\t{importances[idx]:.6f}\n")
                        try:
                            import matplotlib
                            matplotlib.use('Agg', force=True)
                            import matplotlib.pyplot as plt
                            topk = min(25, len(order))
                            plt.figure(figsize=(8, max(3.0, topk*0.3)))
                            plt.barh(range(topk), importances[order][:topk][::-1])
                            plt.yticks(range(topk), [names[i] for i in order][:topk][::-1], fontsize=7)
                            plt.tight_layout()
                            plt.savefig(os.path.join(stats_dir, 'feature_importance.png'), dpi=180)
                            plt.close()
                        except Exception as e:
                            print(f"Final feature importance plot failed: {e}")
                            raise
        except Exception as e:
            # Halt pipeline on final FI failure as requested
            raise
        # Config snapshot
        try:
            import json as _json
            snap = {
                "MIN_AGRI_PROB": th_local,
                "SIEVE_MIN_SIZE": cfg.SIEVE_MIN_SIZE,
                "FINAL_SWEEP_ENABLED": FINAL_SWEEP_ENABLED,
            }
            with open(os.path.join(combo_dir, "config_snapshot.json"), "w") as jf:
                _json.dump(snap, jf, indent=2)
        except Exception as e:
            print(f"Config snapshot failed: {e}")
        # Basic final KML (agri polygons) for each combo
        try:
            from xml.etree.ElementTree import Element, SubElement, tostring
            from xml.dom.minidom import parseString
            import rasterio
            from pyproj import Transformer
            kml = Element('kml'); kml.set('xmlns','http://www.opengis.net/kml/2.2')
            doc = SubElement(kml, 'Document')
            style = SubElement(doc, 'Style', id='agri')
            ln = SubElement(style, 'LineStyle'); SubElement(ln, 'color').text = 'ff0000ff'; SubElement(ln, 'width').text = '1'
            ps = SubElement(style, 'PolyStyle'); SubElement(ps, 'color').text = '400000ff'; SubElement(ps, 'outline').text = '1'
            for tif in tile_files:
                try:
                    with rasterio.open(tif) as src:
                        cleaned, _ = classify_tile(tif, model, th=th_local, sieve_size=cfg.SIEVE_MIN_SIZE)
                        for geom, val in shapes(cleaned.astype('uint8'), mask=cleaned.astype(bool), transform=src.transform):
                            if val != 1:
                                continue
                            coords_img = geom['coordinates'][0]
                            # transform to lon/lat if needed
                            ring = []
                            if src.crs and not src.crs.is_geographic:
                                transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                                for x, y in coords_img:
                                    lon, lat = transformer.transform(x, y)
                                    ring.append((lon, lat))
                            else:
                                ring = [(x, y) for x, y in coords_img]
                            pm = SubElement(doc, 'Placemark')
                            SubElement(pm, 'styleUrl').text = '#agri'
                            poly = SubElement(pm, 'Polygon')
                            ob = SubElement(poly, 'outerBoundaryIs')
                            lr = SubElement(ob, 'LinearRing')
                            SubElement(lr, 'coordinates').text = ' '.join(f"{lon},{lat},0" for lon, lat in ring)
                except Exception:
                    continue
            kml_path = os.path.join(combo_dir, f"agricultural_patches_final_{tag if tag else 'default'}.kml")
            xml = parseString(tostring(kml, encoding='utf-8')).toprettyxml(indent='  ', encoding='utf-8')
            with open(kml_path, 'wb') as f:
                f.write(xml)
        except Exception as e:
            print(f"Final KML export failed: {e}")
        # Remove heavy CSV once summary and metrics are produced
        try:
            if os.path.exists(csv_path):
                os.remove(csv_path)
        except Exception:
            pass
    # Build overall comparison summary across combinations
    try:
        comp_dir = os.path.join(final_root, "summary")
        os.makedirs(comp_dir, exist_ok=True)
        rows = []
        import json as _json
        for combo in sorted(os.listdir(final_root)):
            combo_path = os.path.join(final_root, combo)
            if not os.path.isdir(combo_path) or combo == "summary":
                continue
            mj = os.path.join(combo_path, "statistics", "metrics.json")
            fs = os.path.join(combo_path, "final_summary.txt")
            met = {}
            if os.path.exists(mj):
                try:
                    with open(mj) as f:
                        met = _json.load(f)
                except Exception:
                    met = {}
            # parse overall % from final_summary.txt
            overall_pct = ""
            try:
                with open(fs) as f:
                    for line in f:
                        if line.lower().startswith("overall agricultural %"):
                            overall_pct = line.strip().split(":")[-1].strip()
                            break
            except Exception:
                pass
            rows.append({
                "combo": combo,
                "macro_f1": met.get("macro_f1", None),
                "f1": met.get("f1", None),
                "accuracy": met.get("accuracy", None),
                "auc": met.get("auc", None),
                "auc_pr": met.get("auc_pr", None),
                "overall_pct": overall_pct,
            })
        # choose best by macro_f1 then auc
        def _score(r):
            mf1 = r.get("macro_f1") or 0.0
            auc = r.get("auc") or 0.0
            return (mf1, auc)
        best = None
        for r in rows:
            if best is None or _score(r) > _score(best):
                best = r
        # write comparison CSV and best summary
        comp_csv = os.path.join(comp_dir, "final_comparison.csv")
        with open(comp_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["combo", "macro_f1", "f1", "accuracy", "auc", "auc_pr", "overall_pct"])
            for r in rows:
                w.writerow([r["combo"], r["macro_f1"], r["f1"], r["accuracy"], r["auc"], r["auc_pr"], r["overall_pct"]])
        if best:
            with open(os.path.join(comp_dir, "best_choice.txt"), "w") as f:
                f.write(f"Best final combo: {best['combo']}\n")
                f.write(f"Criteria: max macro_f1, tie-breaker AUC\n")
                f.write(_json.dumps(best, indent=2))
    except Exception as e:
        print(f"Final comparison summary failed: {e}")
    print("Postprocessing complete => see data/phase1/rounds/final_round/<combo>/")

if __name__ == "__main__":
    postprocessing()
