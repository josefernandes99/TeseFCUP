# scripts/ready_to_run_phase1.py
import glob
import json
import os
import sys
import atexit
import signal

import rasterio
import numpy as np

from a0_setup_check import setup_check
from a1_phase1_data_download import download_data
from a2_phase1_initial_labeling import (
    initial_labeling,
    ensure_labels_file,
)
from a4_phase1_active_learning_loop import active_learning_loop, collect_user_hyperparams
from a6_phase1_postprocessing import postprocessing
from grid_search import run_grid_search
from config import RAW_DATA_DIR, TIMESTAMPS, BANDS, INDICES
import config as cfg
from al_shared import snap_to_pixel_center

STEP_ORDER = [
    "setup_check",
    "download_data",
    "initial_labeling",
    "active_learning_loop",
    "postprocessing",
]

def main():
    print("=== Starting PythonProject Pipeline ===\n")
    # Removed outdated startup note to reduce console noise
    # Set up console log tee (stdout + stderr) to data/phase1/consoleLogs.txt
    try:
        from config import DATA_DIR
        os.makedirs(DATA_DIR, exist_ok=True)
        log_path = os.path.join(DATA_DIR, "consoleLogs.txt")
        log_fh = open(log_path, "w", encoding="utf-8", errors="replace")

        class _Tee:
            def __init__(self, stream, fileobj):
                self._stream = stream
                self._file = fileobj
            def write(self, data):
                try:
                    self._stream.write(data)
                except Exception:
                    # ignore console encoding errors
                    pass
                try:
                    # Convert carriage-return live updates into newlines in the file
                    # so the log preserves every update while console shows a single line refresh.
                    to_file = data.replace('\r', '\n')
                    self._file.write(to_file)
                except Exception:
                    pass
            def flush(self):
                try:
                    self._stream.flush()
                except Exception:
                    pass
                try:
                    self._file.flush()
                except Exception:
                    pass
            def isatty(self):
                try:
                    return bool(getattr(self._stream, 'isatty', lambda: False)())
                except Exception:
                    return False
            @property
            def encoding(self):
                return getattr(self._stream, 'encoding', 'utf-8')
            @property
            def errors(self):
                return getattr(self._stream, 'errors', 'replace')
            def fileno(self):
                if hasattr(self._stream, 'fileno'):
                    try:
                        return self._stream.fileno()
                    except Exception:
                        pass
                raise OSError('fileno not available')

        _orig_out, _orig_err = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, log_fh)
        sys.stderr = _Tee(sys.stderr, log_fh)

        def _cleanup_logs(*_args):
            try:
                sys.stdout.flush(); sys.stderr.flush()
            except Exception:
                pass
            try:
                log_fh.flush(); log_fh.close()
            except Exception:
                pass
            # restore
            try:
                sys.stdout, sys.stderr = _orig_out, _orig_err
            except Exception:
                pass

        atexit.register(_cleanup_logs)
        try:
            def _sig_handler(signum, frame):
                _cleanup_logs()
                try:
                    sys.exit(1)
                except SystemExit:
                    return
            signal.signal(signal.SIGINT, _sig_handler)
            signal.signal(signal.SIGTERM, _sig_handler)
        except Exception:
            pass
    except Exception as e:
        print(f"Log setup failed: {e}")
    ensure_labels_file()
    # evaluation now uses stratified splits from labels; no evaluate.csv required
    # Snap and dedup labels/highscore/probable lists at startup
    try:
        _validate_and_dedup_all()
    except Exception as e:
        print(f"Startup validation error: {e}")
    try:
        # Always start cleanly from the first step, no checkpoint resume
        for step in STEP_ORDER:
            if step == "setup_check":
                setup_check()
            elif step == "download_data":
                if not glob.glob(os.path.join(RAW_DATA_DIR, "*.tif")):
                    download_data()
                else:
                    print("Raw data already present; skipping download.")
                # After a1: verify features/indices presence and readiness
                try:
                    _verify_feature_stack()
                except Exception as e:
                    print(f"Feature stack verification failed: {e}")
            elif step == "initial_labeling":
                initial_labeling()
            elif step == "active_learning_loop":
                mode = input("Hyper-parameter mode: [1] grid search, [2] specify => ").strip()
                if mode == "1":
                    print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest, 4=XGBoost")
                    models = ["ResNet", "SVM", "RandomForest", "XGBoost"]
                    ch = input("=> ").strip()
                    mchoice = models[int(ch) - 1] if ch in ["1", "2", "3", "4"] else "RandomForest"
                    run_grid_search(mchoice)
                    return
                else:
                    print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest, 4=XGBoost")
                    models = ["ResNet", "SVM", "RandomForest", "XGBoost"]
                    ch = input("=> ").strip()
                    if ch in ["1", "2", "3", "4"]:
                        mchoice = models[int(ch) - 1]
                    else:
                        mchoice = "RandomForest"
                    params = collect_user_hyperparams(mchoice)
                    active_learning_loop(
                        1,
                        None,
                        mchoice,
                        checkpoint_cb=None,
                        use_grid_search=False,
                        user_params=params,
                    )
                    # export merged final labels for convenience
                    try:
                        import csv as _csv
                        from config import LABELS_FILE, TEMP_LABELS_FILE, FINAL_LABELS_FILE
                        rows = []
                        if os.path.exists(LABELS_FILE):
                            with open(LABELS_FILE) as f:
                                rows += list(_csv.DictReader(f))
                        if os.path.exists(TEMP_LABELS_FILE):
                            with open(TEMP_LABELS_FILE) as f:
                                rows += list(_csv.DictReader(f))
                        if rows:
                            keys = ["id","lat","lon","tile","label","notes"]
                            with open(FINAL_LABELS_FILE, 'w', newline='') as f:
                                w = _csv.DictWriter(f, fieldnames=keys)
                                w.writeheader()
                                for r in rows:
                                    w.writerow({k: r.get(k, '') for k in keys})
                            print(f"Exported merged labels => {FINAL_LABELS_FILE}")
                    except Exception as e:
                        print(f"Final labels export failed: {e}")
            elif step == "postprocessing":
                postprocessing()
        print("\n=== Pipeline Completed Successfully! ===")
    except Exception as e:
        print(f"Pipeline failed: {e}")

 


def _validate_and_dedup_all():
    import csv as _csv
    from config import LABELS_FILE, TEMP_LABELS_FILE, HIGHSCORE_FILE, PROBABLE_AGRI_FILE
    from progress_utils import new_progress as _npb
    # Helper to load, snap, and dedup by (tile,row,col)
    def load_and_snap(path, title="Snapping"):
        if not os.path.exists(path):
            return []
        # Read rows (stream) and show progress by number of records processed
        with open(path, newline='') as f:
            reader = _csv.DictReader(f)
            # Try to estimate total rows for progress; if it fails, show indeterminate
            try:
                # Peek all rows once to count quickly (file sizes are moderate; if large, still OK)
                rows_all = list(reader)
                total = len(rows_all)
                rows_iter = rows_all
            except Exception:
                rows_iter = list(reader)
                total = None
        # Process with snapping + dedup progress
        out = []
        seen = set()
        with _npb() as _prog:
            task = _prog.add_task(f"{title}: {os.path.basename(path)}", total=total or None)
            processed = 0
            for r in rows_iter:
                tile = r.get('tile')
                try:
                    lat = float(r.get('lat')); lon = float(r.get('lon'))
                except Exception:
                    processed += 1; _prog.update(task, advance=1); continue
                if not tile:
                    processed += 1; _prog.update(task, advance=1); continue
                snapped = snap_to_pixel_center(tile, lat, lon)
                if not snapped:
                    processed += 1; _prog.update(task, advance=1); continue
                slat, slon, row, col = snapped
                key = f"{tile}:{row}:{col}"
                if key in seen:
                    processed += 1; _prog.update(task, advance=1); continue
                seen.add(key)
                r['lat'], r['lon'] = f"{slat:.7f}", f"{slon:.7f}"
                r['row'], r['col'] = row, col
                # normalize notes to new scheme
                note_map = {
                    'Agricultural Crop Center': 'Agricultural Certain',
                    'Agricultural Crop Border': 'Agricultural Certain',
                    'Agricultural Crop Doubtful': 'Agricultural Doubtful',
                    'Open Field, No Vegetation': 'Open Field / Tree',
                    'Open Field, Vegetation': 'Open Field / Tree',
                    'Tree/Bush': 'Open Field / Tree',
                    'Building / Man Made': 'Building / Man Made',
                    'Water Bodies': 'Water Bodies',
                    'Other': 'Other',
                }
                n0 = (r.get('notes') or '').strip()
                r['notes'] = note_map.get(n0, n0)
                out.append(r)
                processed += 1
                _prog.update(task, advance=1)
        return out
    # process labels
    labels = load_and_snap(LABELS_FILE, title="Snap & dedup labels")
    if labels:
        first = next(iter(labels), None)
        fields = list(first.keys()) if first else []
        with open(LABELS_FILE, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(labels)
    # process temp labels if present
    temp_labels = load_and_snap(TEMP_LABELS_FILE, title="Snap & dedup temp_labels")
    if temp_labels:
        first = next(iter(temp_labels), None)
        fields = list(first.keys()) if first else []
        with open(TEMP_LABELS_FILE, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(temp_labels)
    # highscore & probable agri: just snap/dedup; do not delete global files
    # For persistent lists, avoid snapping (already pixel-centered from pipeline).
    # Perform a fast streaming de-dup by (tile,row,col) or fallback to (tile,lat,lon).
    def _fast_dedup(path):
        if not os.path.exists(path):
            return []
        try:
            size = os.path.getsize(path)
        except Exception:
            size = None
        out = []
        seen = set()
        with open(path, newline='') as f, _npb() as _prog:
            task = _prog.add_task(f"Dedup persistent list: {os.path.basename(path)}", total=size or None)
            rd = _csv.DictReader(f)
            last_tell = 0
            for r in rd:
                t = r.get('tile')
                # prefer row/col keys when available
                key = None
                try:
                    if r.get('row') is not None and r.get('col') is not None:
                        key = f"{t}:{int(r.get('row'))}:{int(r.get('col'))}"
                except Exception:
                    key = None
                if key is None:
                    try:
                        la = float(r.get('lat'))
                        lo = float(r.get('lon'))
                        key = f"{t}:{la:.7f}:{lo:.7f}"
                    except Exception:
                        key = None
                if not key:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                out.append(r)
                # progress roughly by file bytes read
                try:
                    cur = f.tell()
                    if size and cur > last_tell:
                        _prog.update(task, completed=min(cur, size))
                        last_tell = cur
                except Exception:
                    pass
            if size:
                _prog.update(task, completed=size)
        return out

    if getattr(cfg, 'PERSISTENT_LISTS_ENABLED', True):
        for path in [HIGHSCORE_FILE, PROBABLE_AGRI_FILE]:
            rows = _fast_dedup(path)
            if rows:
                first = next(iter(rows), None)
                fields = list(first.keys()) if first else []
                with open(path, 'w', newline='') as f:
                    w = _csv.DictWriter(f, fieldnames=fields)
                    w.writeheader(); w.writerows(rows)

def _verify_feature_stack():
    """Verify that exported tiles have all configured bands/indices per season and are usable.

    Checks:
    - Band count matches expected (BANDS + INDICES per season)
    - No NaNs in the stack
    - Each channel has some non-zero data (not entirely missing)
    Logs a confirmation list; warns for any issues.
    """
    tiles = sorted(glob.glob(os.path.join(RAW_DATA_DIR, '*.tif')))
    if not tiles:
        print("Feature stack check: no tiles found under raw/; skipping.")
        return
    # Expected names in the exported stack (no derived textures here)
    exp_names = []
    for s in range(1, len(TIMESTAMPS) + 1):
        exp_names += [f"{b}_s{s}" for b in BANDS]
        exp_names += [f"{idx}_s{s}" for idx in INDICES]
        for tb in getattr(cfg, 'TERRAIN_BANDS', ["ELEVATION", "SLOPE", "ASPECT"]):
            exp_names.append(f"{tb}_s{s}")
    exp_bands = len(exp_names)

    # Scan a small subset of tiles for speed (up to 3)
    sample = tiles[:3]
    ok = True
    from progress_utils import new_progress as _npb
    with _npb() as _prog:
        task = _prog.add_task("Verify feature stack", total=len(sample) or None)
        for tp in sample:
            try:
                with rasterio.open(tp) as src:
                    arr = src.read()
            except Exception as e:
                print(f"[WARN] Could not open tile {tp}: {e}")
                ok = False
                _prog.update(task, advance=1)
                continue
            b, H, W = arr.shape
            if b != exp_bands:
                print(f"[WARN] Band count mismatch for {os.path.basename(tp)}: got {b}, expected {exp_bands}")
                ok = False
            # NaN check
            if np.isnan(arr).any():
                print(f"[WARN] NaNs found in {os.path.basename(tp)}; features not ready to use.")
                ok = False
            # Per-channel all-zero check
            eps = 0.0
            all_zero = []
            for i in range(min(b, exp_bands)):
                ch = arr[i]
                if not np.any(ch != eps):
                    all_zero.append(exp_names[i] if i < len(exp_names) else f"band{i}")
            if all_zero:
                print(f"[WARN] Found {len(all_zero)} empty channels in {os.path.basename(tp)}: {', '.join(all_zero[:10])}{' ...' if len(all_zero)>10 else ''}")
                ok = False
            _prog.update(task, advance=1)
    # Summary
    if ok:
        print("Feature stack check: OK. Confirmed channels:")
        print(", ".join(exp_names))
    else:
        print("Feature stack check: issues detected (see warnings above). Consider re-downloading if persistent.")

if __name__ == "__main__":
    main()
