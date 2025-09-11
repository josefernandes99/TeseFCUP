# scripts/ready_to_run_phase1.py
import glob
import json
import os
import sys
import atexit
import signal

import rasterio
import numpy as np

from a0_setup_check import setup_check, init_gee
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
from memory_watcher import start_memory_watcher, free_unused_memory
from joblib import load as _joblib_load
from concurrent.futures import ProcessPoolExecutor, as_completed

STEP_ORDER = [
    "setup_check",
    "download_data",
    "initial_labeling",
    "active_learning_loop",
    "postprocessing",
]

def main():
    print("=== Starting PythonProject Pipeline ===\n")
    # Apply environment tuning early (BLAS/GDAL) so downstream libs honor it.
    try:
        import os as _os
        import config as _cfg
        # BLAS/OpenMP threading caps
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            val = str(getattr(_cfg, k, getattr(_cfg, 'BLAS_NUM_THREADS', 8)))
            if val:
                _os.environ[k] = str(val)
        # GDAL tuning for raster IO
        _os.environ['GDAL_CACHEMAX'] = str(int(getattr(_cfg, 'GDAL_CACHEMAX_MB', 512)))
        _os.environ['GDAL_NUM_THREADS'] = str(getattr(_cfg, 'GDAL_NUM_THREADS', 'ALL_CPUS'))
        # Keep a global Rasterio Env open for the run
        try:
            import atexit as _atexit
            import rasterio as _rio
            # Pass proper types to rasterio.Env: GDAL_CACHEMAX expects int
            _GDAL_ENV = _rio.Env(GDAL_CACHEMAX=int(getattr(_cfg, 'GDAL_CACHEMAX_MB', 512)),
                                 NUM_THREADS=str(getattr(_cfg, 'GDAL_NUM_THREADS', 'ALL_CPUS')))
            _GDAL_ENV.__enter__()
            def _close_env():
                try:
                    _GDAL_ENV.__exit__(None, None, None)
                except Exception:
                    pass
            _atexit.register(_close_env)
        except Exception as _e:
            print(f"Rasterio Env setup skipped: {_e}")
    except Exception as _e:
        print(f"Env tuning skipped: {_e}")
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
    # Start memory watcher to keep system headroom stable (avoids IDE JRE OOM)
    _mw = None
    try:
        if getattr(cfg, 'MEMORY_WATCHER_ENABLED', True):
            thr = int(getattr(cfg, 'MEMORY_WATCHER_THRESHOLD_PERCENT', 80))
            ivl = int(getattr(cfg, 'MEMORY_WATCHER_INTERVAL_SEC', 5))
            _mw = start_memory_watcher(threshold_percent=thr, check_interval=ivl)
    except Exception as _e:
        print(f"Memory watcher not started: {_e}")

    # Startup choice: New run vs Load last run
    choice = None
    while choice not in ("1", "2"):
        print("Start mode => [1] New run, [2] Load last run")
        choice = input("=> ").strip()

    ensure_labels_file()
    # evaluation now uses stratified splits from labels; no evaluate.csv required
    # Snap and dedup labels/highscore/probable lists at startup
    try:
        _validate_and_dedup_all()
    except Exception as e:
        print(f"Startup validation error: {e}")
    try:
        # Determine last round info for resume mode
        last_round = None
        last_round_dir = None
        last_combo_dir = None
        resume_mchoice = None
        resume_params = {}
        try:
            rdirs = [d for d in os.listdir(cfg.ROUNDS_DIR) if d.startswith('round_')]
            if rdirs:
                nums = []
                for d in rdirs:
                    try:
                        nums.append((int(d.split('_')[1]), d))
                    except Exception:
                        pass
                if nums:
                    nums.sort()
                    last_round = nums[-1][0]
                    last_round_dir = os.path.join(cfg.ROUNDS_DIR, nums[-1][1])
                    # Find a combo dir with a model file for this round
                    import glob as _glob
                    patt = os.path.join(last_round_dir, "**", f"model_round_{last_round}.pkl")
                    models = _glob.glob(patt, recursive=True)
                    if models:
                        model_path = models[0]
                        last_combo_dir = os.path.dirname(model_path)
                        # Detect model choice from wrapper.kind
                        try:
                            mdl = _joblib_load(model_path)
                            kind = (getattr(mdl, 'kind', '') or '').lower()
                            if kind == 'svm':
                                resume_mchoice = 'SVM'
                            elif kind == 'randomforest':
                                resume_mchoice = 'RandomForest'
                            elif kind == 'xgboost':
                                resume_mchoice = 'XGBoost'
                            elif kind == 'resnet':
                                resume_mchoice = 'ResNet'
                        except Exception:
                            pass
                        # Load config snapshot to recover hyperparameters
                        try:
                            snap = os.path.join(last_combo_dir, 'statistics', 'config_snapshot.json')
                            if os.path.exists(snap):
                                with open(snap, 'r') as jf:
                                    snap_cfg = json.load(jf)
                                # Threshold/sieve
                                if 'MIN_AGRI_PROB' in snap_cfg:
                                    resume_params['MIN_AGRI_PROB'] = float(snap_cfg['MIN_AGRI_PROB'])
                                if 'SIEVE_MIN_SIZE' in snap_cfg:
                                    resume_params['SIEVE_MIN_SIZE'] = int(snap_cfg['SIEVE_MIN_SIZE'])
                                # Model-specific params
                                if resume_mchoice == 'SVM' and 'SVM_PARAMS' in snap_cfg:
                                    resume_params['SVM_PARAMS'] = snap_cfg['SVM_PARAMS']
                                if resume_mchoice == 'RandomForest' and 'RF_PARAMS' in snap_cfg:
                                    resume_params['RF_PARAMS'] = snap_cfg['RF_PARAMS']
                                if resume_mchoice == 'XGBoost' and 'XGB_PARAMS' in snap_cfg:
                                    resume_params['XGB_PARAMS'] = snap_cfg['XGB_PARAMS']
                        except Exception:
                            pass
        except Exception:
            pass

        for step in STEP_ORDER:
            if step == "setup_check":
                if choice == "1":
                    setup_check()
                else:
                    init_gee()
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
                if choice == "1":
                    initial_labeling()
                else:
                    print("Resume mode: skipping initial labeling (using existing labels/temp_labels).")
            elif step == "active_learning_loop":
                # In resume mode, restart from the last round number; else fresh from 1
                mode = input("Hyper-parameter mode: [1] grid search, [2] specify => ").strip()
                if mode == "1":
                    print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest, 4=XGBoost")
                    models = ["ResNet", "SVM", "RandomForest", "XGBoost"]
                    ch = input("=> ").strip()
                    mchoice = models[int(ch) - 1] if ch in ["1", "2", "3", "4"] else "RandomForest"
                    if choice == "2" and last_round and last_round_dir:
                        print(f"Resume mode: restarting round {last_round} from current state.")
                        # Optionally clean only that round folder to ensure fresh outputs
                        try:
                            import shutil as _shutil
                            _shutil.rmtree(last_round_dir)
                            print(f"Deleted last round folder => {last_round_dir}")
                        except Exception as _e:
                            print(f"Could not delete last round folder (continuing): {_e}")
                        run_grid_search(mchoice)
                    else:
                        run_grid_search(mchoice)
                    return
                else:
                    if choice == "2" and resume_mchoice:
                        mchoice = resume_mchoice
                        params = resume_params or {}
                        print(f"Resume mode: using previous model '{mchoice}' with params: {params}")
                    else:
                        print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest, 4=XGBoost")
                        models = ["ResNet", "SVM", "RandomForest", "XGBoost"]
                        ch = input("=> ").strip()
                        if ch in ["1", "2", "3", "4"]:
                            mchoice = models[int(ch) - 1]
                        else:
                            mchoice = "RandomForest"
                        params = collect_user_hyperparams(mchoice)
                    start_r = 1 if choice == "1" or not last_round else last_round
                    if choice == "2" and last_round_dir:
                        print(f"Resume mode: restarting round {start_r} from current state.")
                        try:
                            import shutil as _shutil
                            _shutil.rmtree(last_round_dir)
                            print(f"Deleted last round folder => {last_round_dir}")
                        except Exception as _e:
                            print(f"Could not delete last round folder (continuing): {_e}")
                    active_learning_loop(
                        start_r,
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
    finally:
        # Stop the memory watcher and free memory one last time
        try:
            free_unused_memory()
        except Exception:
            pass
        try:
            if _mw:
                _mw.stop()
        except Exception:
            pass

 


def _validate_and_dedup_all():
    import csv as _csv
    from config import LABELS_FILE, TEMP_LABELS_FILE, HIGHSCORE_FILE, PROBABLE_AGRI_FILE, RAW_DATA_DIR
    from progress_utils import new_progress as _npb
    # Accumulate warnings for tiles that fail to open or snap
    warn_counts = {}
    warn_examples = {}
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
                # Pre-check: missing tile file
                tifp = os.path.join(RAW_DATA_DIR, tile)
                if not os.path.exists(tifp):
                    warn_counts[tile] = warn_counts.get(tile, 0) + 1
                    warn_examples.setdefault(tile, "Missing tile file")
                    processed += 1; _prog.update(task, advance=1); continue
                try:
                    snapped = snap_to_pixel_center(tile, lat, lon)
                except Exception as _e:
                    # Record raster errors for this tile
                    warn_counts[tile] = warn_counts.get(tile, 0) + 1
                    msg = str(_e)
                    if msg:
                        warn_examples.setdefault(tile, msg)
                    else:
                        warn_examples.setdefault(tile, "Raster IO error")
                    snapped = None
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
    # Helpers for parallel chunk sorting
    def _sort_chunk_by_key(input_csv, output_csv):
        import csv as _csv
        with open(input_csv, 'r', newline='') as f:
            rr = _csv.reader(f)
            header = next(rr, None)
            if not header:
                with open(output_csv, 'w', newline='') as out:
                    pass
                return
            # Column indices
            def _idx(name):
                try: return header.index(name)
                except Exception: return None
            i_tile = _idx('tile'); i_row = _idx('row'); i_col = _idx('col')
            i_lat = _idx('lat'); i_lon = _idx('lon')
            items = []
            for row in rr:
                try:
                    t = row[i_tile] if i_tile is not None else ''
                except Exception:
                    continue
                key = None
                if i_row is not None and i_col is not None:
                    try:
                        rr_i = int(row[i_row]); cc_i = int(row[i_col])
                        key = (t, f"{rr_i:09d}", f"{cc_i:09d}")
                    except Exception:
                        key = None
                if key is None and i_lat is not None and i_lon is not None:
                    try:
                        la = float(row[i_lat]); lo = float(row[i_lon])
                        key = (t, f"{la:.7f}", f"{lo:.7f}")
                    except Exception:
                        key = None
                if key is None:
                    continue
                items.append((key, row))
        items.sort(key=lambda t: t[0])
        with open(output_csv, 'w', newline='') as out:
            ww = _csv.writer(out)
            ww.writerow(header)
            for _, row in items:
                ww.writerow(row)

    def _sort_chunk_by_score(input_csv, output_csv):
        import csv as _csv
        with open(input_csv, 'r', newline='') as f:
            rr = _csv.reader(f)
            header = next(rr, None)
            if not header:
                with open(output_csv, 'w', newline='') as out:
                    pass
                return
            def _idx(name):
                try: return header.index(name)
                except Exception: return None
            i_score = _idx('score'); i_prob = _idx('prob')
            i_tile = _idx('tile'); i_row = _idx('row'); i_col = _idx('col')
            i_lat = _idx('lat'); i_lon = _idx('lon')
            def _score_of_row(row):
                try:
                    if i_score is not None:
                        return float(row[i_score])
                except Exception:
                    pass
                try:
                    if i_prob is not None:
                        return float(row[i_prob])
                except Exception:
                    return float('-inf')
            def _tie_of_row(row):
                t = row[i_tile] if i_tile is not None else ''
                if i_row is not None and i_col is not None:
                    try:
                        return (t, int(row[i_row]), int(row[i_col]))
                    except Exception:
                        return (t, row[i_lat] if i_lat is not None else '', row[i_lon] if i_lon is not None else '')
                return (t, row[i_lat] if i_lat is not None else '', row[i_lon] if i_lon is not None else '')
            items = []
            for row in rr:
                items.append((-_score_of_row(row), _tie_of_row(row), row))
        items.sort(key=lambda t: (t[0], t[1]))
        with open(output_csv, 'w', newline='') as out:
            ww = _csv.writer(out)
            ww.writerow(header)
            for _, _, row in items:
                ww.writerow(row)

    def _fast_dedup(path):
        """External sort + unique by key (tile,row,col) or (tile,lat,lon) with bounded RAM.

        Parallel two-sort pipeline to preserve score-descending final order.
        Steps:
        - Phase 1: Read CSV in chunks, write raw chunk CSVs; sort each chunk by key in a ProcessPool.
        - Phase 2: K-way merge key-sorted chunks, select max-score per key (or max-prob), write dedup_tmp.csv.
        - Phase 3: Read dedup_tmp.csv in chunks; sort each chunk by score desc in a ProcessPool; k-way merge to final CSV.
        - Replace original file atomically; cleanup temps.
        """
        if not os.path.exists(path):
            return False
        base = os.path.basename(path)
        root_dir = os.path.dirname(path)
        tmp_dir = os.path.join(root_dir, "_dedup_tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            size = os.path.getsize(path)
        except Exception:
            size = None

        # Phase 1: chunked write + parallel sort by key
        CHUNK_ROWS = int(getattr(cfg, 'DEDUP_CHUNK_ROWS', 200_000))
        raw_chunks = []
        with open(path, 'r', newline='') as f, _npb() as _prog:
            task = _prog.add_task(f"Dedup (phase 1/3 write): {base}", total=size or None)
            rd = _csv.DictReader(f)
            fieldnames = rd.fieldnames or []
            rows_buf = []
            idx = 0
            last_tell = 0
            for r in rd:
                rows_buf.append(r)
                if len(rows_buf) >= CHUNK_ROWS:
                    cpath_raw = os.path.join(tmp_dir, f"{base}.chunk{idx}.raw.csv")
                    with open(cpath_raw, 'w', newline='') as cf:
                        w = _csv.DictWriter(cf, fieldnames=fieldnames)
                        if fieldnames:
                            w.writeheader()
                        w.writerows(rows_buf)
                    raw_chunks.append(cpath_raw)
                    rows_buf.clear(); idx += 1
                try:
                    cur = f.tell()
                    if size and cur > last_tell:
                        _prog.update(task, completed=min(cur, size))
                        last_tell = cur
                except Exception:
                    pass
            if rows_buf:
                cpath_raw = os.path.join(tmp_dir, f"{base}.chunk{idx}.raw.csv")
                with open(cpath_raw, 'w', newline='') as cf:
                    w = _csv.DictWriter(cf, fieldnames=fieldnames)
                    if fieldnames:
                        w.writeheader()
                    w.writerows(rows_buf)
                raw_chunks.append(cpath_raw)
            if size:
                _prog.update(task, completed=size)

        # Parallel sort raw chunks by key (pipeline submission)
        sorted_chunks = []
        if raw_chunks:
            with _npb() as _prog2:
                t2 = _prog2.add_task(f"Dedup (phase 2/3 sort-by-key): {base}", total=len(raw_chunks))
                max_workers = max(2, min(int(getattr(cfg, 'DEDUP_SORT_WORKERS', 4)), os.cpu_count() or 4))
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = {}
                    for rp in raw_chunks:
                        outp = rp.replace('.raw.csv', '.keysorted.csv')
                        futs[ex.submit(_sort_chunk_by_key, rp, outp)] = (rp, outp)
                    for fut in as_completed(futs):
                        try:
                            rp, outp = futs[fut]
                            sorted_chunks.append(outp)
                        except Exception:
                            pass
                        _prog2.update(t2, advance=1)
                # remove raw chunks
                for rp in raw_chunks:
                    try: os.remove(rp)
                    except Exception: pass

        # Phase 2: k-way merge on key to select max score per key
        dedup_tmp = os.path.join(tmp_dir, f"{base}.dedup_tmp.csv")
        import heapq
        def _key_of(r):
            t = r.get('tile') or ''
            try:
                if r.get('row') is not None and r.get('col') is not None:
                    rr = int(r.get('row')); cc = int(r.get('col'))
                    return (t, f"{rr:09d}", f"{cc:09d}")
            except Exception:
                pass
            try:
                la = float(r.get('lat')); lo = float(r.get('lon'))
                return (t, f"{la:.7f}", f"{lo:.7f}")
            except Exception:
                return None
        def _score_of(r):
            try:
                return float(r.get('score'))
            except Exception:
                try: return float(r.get('prob'))
                except Exception: return float('-inf')
        with open(dedup_tmp, 'w', newline='') as out:
            w = _csv.DictWriter(out, fieldnames=fieldnames)
            if fieldnames:
                w.writeheader()
            readers = []
            for sp in sorted_chunks:
                try:
                    cf = open(sp, 'r', newline='')
                    rd2 = _csv.DictReader(cf)
                except Exception:
                    continue
                readers.append((cf, rd2))
            heap = []
            for idx, (cf, rd2) in enumerate(readers):
                try:
                    row = next(rd2)
                except StopIteration:
                    row = None
                if not row:
                    continue
                k = _key_of(row)
                if not k:
                    continue
                heapq.heappush(heap, (k, idx, row))
            last_key = None
            best_row = None
            best_score = float('-inf')
            while heap:
                k, idx, row = heapq.heappop(heap)
                if last_key is None:
                    last_key = k
                    best_row = row
                    best_score = _score_of(row)
                elif k == last_key:
                    sc = _score_of(row)
                    if sc > best_score:
                        best_score = sc
                        best_row = row
                else:
                    # flush best of previous key
                    if best_row is not None:
                        w.writerow(best_row)
                    last_key = k
                    best_row = row
                    best_score = _score_of(row)
                # advance reader
                cf, rd2 = readers[idx]
                try:
                    row2 = next(rd2)
                except StopIteration:
                    row2 = None
                if row2:
                    k2 = _key_of(row2)
                    if k2:
                        heapq.heappush(heap, (k2, idx, row2))
            # flush last
            if best_row is not None:
                w.writerow(best_row)
        # cleanup keysorted chunks
        for sp in sorted_chunks:
            try: os.remove(sp)
            except Exception: pass

        # Phase 3: parallel sort by score desc; k-way merge
        # Chunk dedup_tmp into raw score chunks
        score_raws = []
        with open(dedup_tmp, 'r', newline='') as f, _npb() as _prog3:
            task3 = _prog3.add_task(f"Dedup (phase 3/3 sort-by-score): {base}", total=None)
            rd = _csv.DictReader(f)
            flds = rd.fieldnames or []
            rows_buf = []
            idx = 0
            for r in rd:
                rows_buf.append(r)
                if len(rows_buf) >= CHUNK_ROWS:
                    rp = os.path.join(tmp_dir, f"{base}.scorechunk{idx}.raw.csv")
                    with open(rp, 'w', newline='') as cf:
                        w = _csv.DictWriter(cf, fieldnames=flds)
                        if flds:
                            w.writeheader()
                        w.writerows(rows_buf)
                    score_raws.append(rp)
                    rows_buf.clear(); idx += 1
                    _prog3.update(task3, advance=1)
            if rows_buf:
                rp = os.path.join(tmp_dir, f"{base}.scorechunk{idx}.raw.csv")
                with open(rp, 'w', newline='') as cf:
                    w = _csv.DictWriter(cf, fieldnames=flds)
                    if flds:
                        w.writeheader()
                    w.writerows(rows_buf)
                score_raws.append(rp)
                _prog3.update(task3, advance=1)

        # Parallel sort score chunks
        score_sorted = []
        if score_raws:
            with _npb() as _prog4:
                t4 = _prog4.add_task(f"Dedup (sort-by-score chunks): {base}", total=len(score_raws))
                max_workers = max(2, min(int(getattr(cfg, 'DEDUP_SORT_WORKERS', 4)), os.cpu_count() or 4))
                with ProcessPoolExecutor(max_workers=max_workers) as ex:
                    futs = {}
                    for rp in score_raws:
                        outp = rp.replace('.raw.csv', '.scoresorted.csv')
                        futs[ex.submit(_sort_chunk_by_score, rp, outp)] = (rp, outp)
                    for fut in as_completed(futs):
                        try:
                            rp, outp = futs[fut]
                            score_sorted.append(outp)
                        except Exception:
                            pass
                        _prog4.update(t4, advance=1)
                for rp in score_raws:
                    try: os.remove(rp)
                    except Exception: pass

        # Merge score-sorted chunks (descending) to final CSV
        final_tmp = path + '.tmp'
        import heapq as _hq
        def _score_of_row(r):
            try: return float(r.get('score'))
            except Exception:
                try: return float(r.get('prob'))
                except Exception: return float('-inf')
        with open(final_tmp, 'w', newline='') as out:
            w = _csv.DictWriter(out, fieldnames=fieldnames)
            if fieldnames:
                w.writeheader()
            readers = []
            for sp in score_sorted:
                try:
                    cf = open(sp, 'r', newline='')
                    rd2 = _csv.DictReader(cf)
                except Exception:
                    continue
                readers.append((cf, rd2))
            heap = []
            for idx, (cf, rd2) in enumerate(readers):
                try:
                    row = next(rd2)
                except StopIteration:
                    row = None
                if not row:
                    continue
                sc = _score_of_row(row)
                tieb = (row.get('tile') or '', int(row.get('row') or 0), int(row.get('col') or 0))
                _hq.heappush(heap, (-sc, tieb, idx, row))
            while heap:
                neg_sc, tieb, idx, row = _hq.heappop(heap)
                w.writerow(row)
                cf, rd2 = readers[idx]
                try:
                    row2 = next(rd2)
                except StopIteration:
                    row2 = None
                if row2:
                    sc2 = _score_of_row(row2)
                    tie2 = (row2.get('tile') or '', int(row2.get('row') or 0), int(row2.get('col') or 0))
                    _hq.heappush(heap, (-sc2, tie2, idx, row2))
            for cf, _ in readers:
                try: cf.close()
                except Exception: pass

        # Replace original atomically and cleanup
        try:
            os.replace(final_tmp, path)
        except Exception:
            try: os.remove(final_tmp)
            except Exception: pass
            return False
        try: os.remove(dedup_tmp)
        except Exception: pass
        try: os.rmdir(tmp_dir)
        except Exception: pass
        print(f"Dedup complete: {base} (parallel two-sort)")
        return True

        # If only one chunk, just replace original with sorted unique from that chunk
        if not chunk_paths:
            # No valid rows; truncate file to header only
            with open(path, 'r', newline='') as f:
                rd = _csv.DictReader(f)
                fieldnames = rd.fieldnames or []
            with open(path, 'w', newline='') as out:
                w = _csv.DictWriter(out, fieldnames=fieldnames)
                if fieldnames:
                    w.writeheader()
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass
            return True

        # Phase 2: k-way merge with unique
        tmp_out = path + '.tmp'
        import heapq
        with open(tmp_out, 'w', newline='') as out, _npb() as _prog:
            task = _prog.add_task(f"Dedup (phase 2/2 merge): {base}", total=len(chunk_paths) or None)
            # Open readers
            readers = []
            for cpath in chunk_paths:
                try:
                    cf = open(cpath, 'r', newline='')
                except Exception:
                    continue
                rd = _csv.DictReader(cf)
                readers.append((cf, rd))
            # Prepare heap with first row from each reader
            heap = []
            for idx, (cf, rd) in enumerate(readers):
                try:
                    row = next(rd)
                except StopIteration:
                    row = None
                if not row:
                    continue
                k = _key_of(row)
                if not k:
                    continue
                heapq.heappush(heap, (k, idx, row))
            # Write header
            fieldnames = readers[0][1].fieldnames if readers else []
            w = _csv.DictWriter(out, fieldnames=fieldnames)
            if fieldnames:
                w.writeheader()
            last_key = None
            progressed = 0
            while heap:
                k, idx, row = heapq.heappop(heap)
                if k != last_key:
                    w.writerow(row)
                    last_key = k
                # pull next from the same reader
                cf, rd = readers[idx]
                try:
                    row2 = next(rd)
                except StopIteration:
                    row2 = None
                if row2:
                    k2 = _key_of(row2)
                    if k2:
                        heapq.heappush(heap, (k2, idx, row2))
                # progress by number of readers advanced
                progressed += 1
                if progressed % 50000 == 0:
                    _prog.update(task, advance=0)  # keep spinner alive
            # Close readers
            for cf, _ in readers:
                try: cf.close()
                except Exception: pass

        # Replace original atomically
        try:
            os.replace(tmp_out, path)
        except Exception:
            try:
                os.remove(tmp_out)
            except Exception:
                pass
            return False
        # Cleanup chunks
        for cpath in chunk_paths:
            try: os.remove(cpath)
            except Exception: pass
        try:
            os.rmdir(tmp_dir)
        except Exception:
            pass
        print(f"Dedup complete: {base} (external sort-unique)")
        return True

    # Print warning summary for tiles that failed to snap/open
    if warn_counts:
        print("\n[WARN] Snap warnings summary (tiles with open/snap issues):")
        top = sorted(warn_counts.items(), key=lambda t: t[1], reverse=True)[:10]
        for tile, cnt in top:
            ex = warn_examples.get(tile, "")
            print(f" - {tile}: {cnt} rows skipped. Example error: {ex}")
        rest = len(warn_counts) - len(top)
        if rest > 0:
            print(f" ... and {rest} more tiles with fewer skips.")

    # Dedup enabled persistent lists individually (streaming to avoid large RAM)
    if getattr(cfg, 'HIGHSCORE_LIST_ENABLED', True):
        path = HIGHSCORE_FILE
        _fast_dedup(path)
    if getattr(cfg, 'PROBABLE_AGRI_LIST_ENABLED', False):
        path = PROBABLE_AGRI_FILE
        _fast_dedup(path)

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
