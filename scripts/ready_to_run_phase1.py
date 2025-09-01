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
from config import RAW_DATA_DIR, CHECKPOINT_FILE, TIMESTAMPS, BANDS, INDICES
from al_shared import snap_to_pixel_center

STEP_ORDER = [
    "setup_check",
    "download_data",
    "initial_labeling",
    "active_learning_loop",
    "postprocessing",
]

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                data = json.load(f)
            step = data.get("step")
            if step in STEP_ORDER:
                return data
        except Exception:
            pass
    return None

def save_checkpoint(step, **extra):
    data = {"step": step}
    data.update(extra)
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)

def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

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
        cp = load_checkpoint()
        if cp:
            step = cp.get("step")
            msg = f"Resume from checkpoint at '{step}'"
            if step == "active_learning_loop" and cp.get("round"):
                msg += f" (round {cp['round']})"
            msg += "? (y/n) => "
            ans = input(msg).strip().lower()
            if not ans.startswith("y"):
                clear_checkpoint()
                cp = None

        start_step = cp["step"] if cp else STEP_ORDER[0]

        step_idx = STEP_ORDER.index(start_step)

        al_start_round = cp.get("round", 1) if cp and start_step == "active_learning_loop" else 1
        al_total_rounds = cp.get("total_rounds") if cp else None
        al_model_choice = cp.get("model_choice") if cp else None
        al_params = cp.get("params") if cp else None

        for step in STEP_ORDER[step_idx:]:
            if step == "setup_check":
                save_checkpoint(step)
                setup_check()
            elif step == "download_data":
                save_checkpoint(step)
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
                save_checkpoint(step)
                initial_labeling()
            elif step == "active_learning_loop":
                save_checkpoint(step)
                mode = input("Hyper-parameter mode: [1] grid search, [2] specify => ").strip()
                if mode == "1":
                    print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest, 4=XGBoost")
                    models = ["ResNet", "SVM", "RandomForest", "XGBoost"]
                    ch = input("=> ").strip()
                    mchoice = models[int(ch) - 1] if ch in ["1", "2", "3", "4"] else "RandomForest"
                    run_grid_search(mchoice)
                    clear_checkpoint()
                    return
                else:
                    def cb(r, nr, mc, params):
                        save_checkpoint(
                            "active_learning_loop",
                            round=r,
                            total_rounds=nr,
                            model_choice=mc,
                            params=params,
                        )
                    mchoice = al_model_choice
                    if mchoice is None:
                        print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest, 4=XGBoost")
                        models = ["ResNet", "SVM", "RandomForest", "XGBoost"]
                        ch = input("=> ").strip()
                        if ch in ["1", "2", "3", "4"]:
                            mchoice = models[int(ch) - 1]
                        else:
                            mchoice = "RandomForest"
                    params = al_params or collect_user_hyperparams(mchoice)
                    active_learning_loop(
                        al_start_round,
                        al_total_rounds,
                        mchoice,
                        checkpoint_cb=cb,
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
                save_checkpoint(step)
                postprocessing()

        clear_checkpoint()
        print("\n=== Pipeline Completed Successfully! ===")
    except Exception as e:
        print(f"Pipeline failed: {e}")

 


def _validate_and_dedup_all():
    import csv as _csv
    from config import LABELS_FILE, TEMP_LABELS_FILE, HIGHSCORE_FILE, PROBABLE_AGRI_FILE
    # Helper to load, snap, and dedup by (tile,row,col)
    def load_and_snap(path):
        if not os.path.exists(path):
            return []
        with open(path) as f:
            rows = list(_csv.DictReader(f))
        out = []
        seen = set()
        for r in rows:
            tile = r.get('tile')
            try:
                lat = float(r.get('lat')); lon = float(r.get('lon'))
            except Exception:
                continue
            if not tile:
                continue
            snapped = snap_to_pixel_center(tile, lat, lon)
            if not snapped:
                continue
            slat, slon, row, col = snapped
            key = f"{tile}:{row}:{col}"
            if key in seen:
                continue
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
        return out
    # process labels
    labels = load_and_snap(LABELS_FILE)
    if labels:
        first = next(iter(labels), None)
        fields = list(first.keys()) if first else []
        with open(LABELS_FILE, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(labels)
    # process temp labels if present
    temp_labels = load_and_snap(TEMP_LABELS_FILE)
    if temp_labels:
        first = next(iter(temp_labels), None)
        fields = list(first.keys()) if first else []
        with open(TEMP_LABELS_FILE, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(temp_labels)
    # highscore & probable agri: just snap/dedup; do not delete global files
    for path in [HIGHSCORE_FILE, PROBABLE_AGRI_FILE]:
        rows = load_and_snap(path)
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
    exp_bands = len(exp_names)

    # Scan a small subset of tiles for speed (up to 3)
    sample = tiles[:3]
    ok = True
    for tp in sample:
        try:
            with rasterio.open(tp) as src:
                arr = src.read()
        except Exception as e:
            print(f"[WARN] Could not open tile {tp}: {e}")
            ok = False
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
        # Use a small epsilon to ignore float rounding
        eps = 0.0
        all_zero = []
        for i in range(min(b, exp_bands)):
            ch = arr[i]
            if not np.any(ch != eps):
                all_zero.append(exp_names[i] if i < len(exp_names) else f"band{i}")
        if all_zero:
            print(f"[WARN] Found {len(all_zero)} empty channels in {os.path.basename(tp)}: {', '.join(all_zero[:10])}{' ...' if len(all_zero)>10 else ''}")
            ok = False
    # Summary
    if ok:
        print("Feature stack check: OK. Confirmed channels:")
        print(", ".join(exp_names))
    else:
        print("Feature stack check: issues detected (see warnings above). Consider re-downloading if persistent.")

if __name__ == "__main__":
    main()
