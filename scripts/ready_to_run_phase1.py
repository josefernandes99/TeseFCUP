# scripts/ready_to_run_phase1.py
import glob
import json
import os
import sys
import atexit
import signal
import shutil
from glob import glob as _glob

from a0_setup_check import setup_check
from a1_phase1_data_download import download_data
from a2_phase1_initial_labeling import (
    initial_labeling,
    ensure_labels_file,
)
from a4_phase1_active_learning_loop import active_learning_loop, collect_user_hyperparams
from a6_phase1_postprocessing import postprocessing
from grid_search import run_grid_search
from config import RAW_DATA_DIR, CHECKPOINT_FILE
from al_shared import snap_to_pixel_center

STEP_ORDER = [
    "setup_check",
    "download_data",
    "initial_labeling",
    "active_learning_loop",
    "postprocessing",
]
def _clean_previous_outputs():
    """Remove artifacts from previous runs to start fresh.
    Deletes:
    - Overlay TIFFs in RAW_DATA_DIR (e.g., *_overlay.tif, *_th*.tif)
    - Everything inside data/phase1/rounds/ (subfolders and files)
    - labels/phase1/finalLabels.csv and labels/phase1/temp_labels.csv
    - Old root-level final_predictions*.csv and final_summary*.txt in data/phase1/
    Note: Does NOT delete the checkpoint file; resume decision governs that.
    """
    try:
        from config import RAW_DATA_DIR as _RAW, ROUNDS_DIR as _RDS, DATA_DIR as _DD, LABELS_DIR as _LD, FINAL_LABELS_FILE as _FL, TEMP_LABELS_FILE as _TL
        # 1) Clean overlays left in raw
        patterns = ["*_overlay.tif", "*_th*.tif"]
        for pat in patterns:
            for fp in _glob(os.path.join(_RAW, pat)):
                try:
                    os.remove(fp)
                    print(f"Deleted overlay => {fp}")
                except Exception as e:
                    print(f"Overlay delete failed ({fp}): {e}")
        # 2) Clean rounds directory (all content)
        if os.path.exists(_RDS):
            for name in os.listdir(_RDS):
                p = os.path.join(_RDS, name)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                    print(f"Deleted previous round artifact => {p}")
                except Exception as e:
                    print(f"Round cleanup failed ({p}): {e}")
        # 3) Remove specific label working files
        for fp in [_FL, _TL]:
            try:
                if os.path.exists(fp):
                    os.remove(fp)
                    print(f"Deleted labels artifact => {fp}")
            except Exception as e:
                print(f"Labels cleanup failed ({fp}): {e}")
        # 4) Remove old final summaries at data root (legacy layout)
        for pat in ["final_predictions*.csv", "final_summary*.txt"]:
            for fp in _glob(os.path.join(_DD, pat)):
                try:
                    os.remove(fp)
                    print(f"Deleted legacy final artifact => {fp}")
                except Exception as e:
                    print(f"Legacy final cleanup failed ({fp}): {e}")
    except Exception as e:
        print(f"Cleanup failed: {e}")

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
        # Register signal handlers; ignore environments where this isn't allowed
        import contextlib
        def _sig_handler(signum, frame):
            _cleanup_logs()
            sys.exit(1)
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, _sig_handler)
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGTERM, _sig_handler)
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
                # User chose not to resume: clean workspace and reset checkpoint for this run
                _clean_previous_outputs()
                clear_checkpoint()
                cp = None
        else:
            # No checkpoint: perform workspace cleanup for a clean run
            _clean_previous_outputs()

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
            elif step == "initial_labeling":
                save_checkpoint(step)
                initial_labeling()
            elif step == "active_learning_loop":
                save_checkpoint(step)
                mode = input("Hyper-parameter mode: [1] grid search, [2] specify => ").strip()
                if mode == "1":
                    print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest")
                    models = ["ResNet", "SVM", "RandomForest"]
                    ch = input("=> ").strip()
                    mchoice = models[int(ch) - 1] if ch in ["1", "2", "3"] else "RandomForest"
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
                        print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest")
                        models = ["ResNet", "SVM", "RandomForest"]
                        ch = input("=> ").strip()
                        if ch in ["1", "2", "3"]:
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
        fields = list(labels[0].keys())
        with open(LABELS_FILE, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(labels)
    # process temp labels if present
    temp_labels = load_and_snap(TEMP_LABELS_FILE)
    if temp_labels:
        fields = list(temp_labels[0].keys())
        with open(TEMP_LABELS_FILE, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(temp_labels)
    # highscore & probable agri: just snap/dedup; do not delete global files
    for path in [HIGHSCORE_FILE, PROBABLE_AGRI_FILE]:
        rows = load_and_snap(path)
        if rows:
            fields = list(rows[0].keys())
            with open(path, 'w', newline='') as f:
                w = _csv.DictWriter(f, fieldnames=fields)
                w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
