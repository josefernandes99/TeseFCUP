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
from final_grid_search import run_final_grid_search
from config import RAW_DATA_DIR, TIMESTAMPS, BANDS, INDICES
import config as cfg
from al_shared import snap_to_pixel_center
from memory_watcher import start_memory_watcher, free_unused_memory
from joblib import load as _joblib_load
from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from progress_utils import console, print_section


def _prompt_menu(title: str, options: list[tuple[str, str]], prompt: str = "Select option") -> str:
    table = Table.grid(padding=(0, 1))
    table.add_column("Opt", justify="right", style="cyan", no_wrap=True)
    table.add_column("Description", justify="left")
    for key, desc in options:
        table.add_row(key, desc)
    console.print(Panel.fit(table, title=title, border_style="bright_blue"))
    return console.input(f"[bold white]{prompt}: [/bold white]").strip()

# --- Module-scope helpers for dedup ProcessPool (Windows-friendly) ---
def _dedup_sort_chunk_by_key(input_csv, output_csv):
    import csv as _csv
    with open(input_csv, 'r', newline='') as f:
        rr = _csv.reader(f)
        header = next(rr, None)
        if not header:
            with open(output_csv, 'w', newline='') as out:
                pass
            return True
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
    return True


def _dedup_sort_chunk_by_score(input_csv, output_csv):
    import csv as _csv
    with open(input_csv, 'r', newline='') as f:
        rr = _csv.reader(f)
        header = next(rr, None)
        if not header:
            with open(output_csv, 'w', newline='') as out:
                pass
            return True
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
    return True

STEP_ORDER = [
    "setup_check",
    "download_data",
    "initial_labeling",
    "active_learning_loop",
    "postprocessing",
]

def _prompt_island_selection() -> None:
    """Interactive island picker that stores the global selection."""
    while True:
        islands = cfg.discover_islands()
        if islands:
            menu_options = [(str(idx), name) for idx, name in enumerate(islands, 1)]
            menu_options.append(("R", "Rescan"))
            console.print(Panel.fit("Available islands detected in raw/", border_style="green"))
            pick = _prompt_menu("Islands", menu_options, prompt="Island").strip()
            if pick.lower() == "r":
                continue
            if pick.isdigit():
                idx = int(pick)
                if 1 <= idx <= len(islands):
                    cfg.set_selected_island(islands[idx - 1])
                    break
            if pick:
                for name in islands:
                    if name.lower() == pick.lower():
                        cfg.set_selected_island(name)
                        break
                else:
                    console.print("[yellow]Invalid choice. Please pick one of the listed islands.[/yellow]")
                    continue
                break
            console.print("[yellow]Invalid choice. Please pick one of the listed islands.[/yellow]")
        else:
            console.print("[yellow]No tiles found in raw/. Enter the island name you plan to work with.[/yellow]")
            console.print("[dim]Leave blank to continue without a filter; the menu will reuse that later.[/dim]")
            pick = console.input("Island name => ").strip()
            if pick:
                cfg.set_selected_island(pick)
            else:
                cfg.set_selected_island(None)
            break
    sel = cfg.get_selected_island()
    if sel:
        console.print(f"[bold green]Island selected:[/bold green] {sel}")
    else:
        console.print("[yellow]No island filter applied; pipeline will use every tile it finds.[/yellow]")

def main():
    console.rule("PythonProject Pipeline")
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
            console.print(f"[yellow]Rasterio Env setup skipped:[/yellow] {_e}")
    except Exception as _e:
        console.print(f"[yellow]Env tuning skipped:[/yellow] {_e}")
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
        console.print(f"[yellow]Log setup failed:[/yellow] {e}")
    # Start memory watcher to keep system headroom stable (avoids IDE JRE OOM)
    _mw = None
    try:
        if getattr(cfg, 'MEMORY_WATCHER_ENABLED', True):
            thr = int(getattr(cfg, 'MEMORY_WATCHER_THRESHOLD_PERCENT', 80))
            ivl = int(getattr(cfg, 'MEMORY_WATCHER_INTERVAL_SEC', 5))
            _mw = start_memory_watcher(threshold_percent=thr, check_interval=ivl)
    except Exception as _e:
        console.print(f"[yellow]Memory watcher not started:[/yellow] {_e}")

    _prompt_island_selection()

    # Startup choice: New run vs Load last run
    choice = None
    while choice not in ("1", "2"):
        choice = _prompt_menu(
            "Start Mode",
            [("1", "New run"), ("2", "Load last run")],
            prompt="Selection"
        )

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
                            elif kind == 'resnet':
                                print("Found a legacy ResNet model. ResNet support was removed; please retrain with SVM or RandomForest.")
                                resume_mchoice = None
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
                if not cfg.list_raw_tiles():
                    download_data()
                else:
                    console.print("[dim]Raw data already present; skipping download.[/dim]")
                # After a1: verify features/indices presence and readiness
                try:
                    _verify_feature_stack()
                except Exception as e:
                    console.print(f"[yellow]Feature stack verification failed:[/yellow] {e}")
            elif step == "initial_labeling":
                if choice == "1":
                    initial_labeling()
                else:
                    console.print("[dim]Resume mode: skipping initial labeling (using existing labels/temp_labels).[/dim]")
            elif step == "active_learning_loop":
                # In resume mode, restart from the last round number; else fresh from 1
                mode = _prompt_menu(
                    "Hyper-parameter Mode",
                    [("1", "Grid search"), ("2", "Manual specify"), ("3", "Auto tuning")],
                    prompt="Mode"
                )
                if mode == "1":
                    model_choice = _prompt_menu(
                        "Model",
                        [("1", "SVM"), ("2", "RandomForest")],
                        prompt="Model"
                    )
                    models = {"1": "SVM", "2": "RandomForest"}
                    mchoice = models.get(model_choice, "RandomForest")
                    if choice == "2" and last_round and last_round_dir:
                        console.print(f"[dim]Resume mode: restarting round {last_round} from current state.[/dim]")
                        # Optionally clean only that round folder to ensure fresh outputs
                        try:
                            import shutil as _shutil
                            _shutil.rmtree(last_round_dir)
                            console.print(f"[dim]Deleted last round folder => {last_round_dir}[/dim]")
                        except Exception as _e:
                            console.print(f"[yellow]Could not delete last round folder (continuing): {_e}[/yellow]")
                        run_grid_search(mchoice)
                    else:
                        run_grid_search(mchoice)
                    return
                elif mode == "3":
                    if choice == "2" and resume_mchoice:
                        mchoice = resume_mchoice
                        console.print(f"[dim]Resume mode: using previous model '{mchoice}' for auto tuning.[/dim]")
                    else:
                        model_choice = _prompt_menu(
                            "Model",
                            [("1", "SVM"), ("2", "RandomForest"), ("3", "Ensemble")],
                            prompt="Model"
                        )
                        models = {"1": "SVM", "2": "RandomForest", "3": "Ensemble"}
                        mchoice = models.get(model_choice, "RandomForest")
                    start_r = 1 if choice == "1" or not last_round else last_round
                    if choice == "2" and last_round_dir:
                        console.print(f"[dim]Resume mode: restarting round {start_r} from current state.[/dim]")
                        try:
                            import shutil as _shutil
                            _shutil.rmtree(last_round_dir)
                            console.print(f"[dim]Deleted last round folder => {last_round_dir}[/dim]")
                        except Exception as _e:
                            console.print(f"[yellow]Could not delete last round folder (continuing): {_e}[/yellow]")
                    active_learning_loop(
                        start_r,
                        None,
                        mchoice,
                        checkpoint_cb=None,
                        tuning_mode="auto",
                        user_params=None,
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
                            console.print(f"[dim]Exported merged labels => {FINAL_LABELS_FILE}[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Final labels export failed:[/yellow] {e}")
                else:
                    if choice == "2" and resume_mchoice:
                        mchoice = resume_mchoice
                        params = resume_params or {}
                        console.print(f"[dim]Resume mode: using previous model '{mchoice}' with params: {params}[/dim]")
                    else:
                        model_choice = _prompt_menu(
                            "Model",
                            [("1", "SVM"), ("2", "RandomForest")],
                            prompt="Model"
                        )
                        mchoice = "SVM" if model_choice == "1" else "RandomForest"
                        params = collect_user_hyperparams(mchoice)
                    start_r = 1 if choice == "1" or not last_round else last_round
                    if choice == "2" and last_round_dir:
                        console.print(f"[dim]Resume mode: restarting round {start_r} from current state.[/dim]")
                        try:
                            import shutil as _shutil
                            _shutil.rmtree(last_round_dir)
                            console.print(f"[dim]Deleted last round folder => {last_round_dir}[/dim]")
                        except Exception as _e:
                            console.print(f"[yellow]Could not delete last round folder (continuing): {_e}[/yellow]")
                    active_learning_loop(
                        start_r,
                        None,
                        mchoice,
                        checkpoint_cb=None,
                        tuning_mode="manual",
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
                            console.print(f"[dim]Exported merged labels => {FINAL_LABELS_FILE}[/dim]")
                    except Exception as e:
                        console.print(f"[yellow]Final labels export failed:[/yellow] {e}")
            elif step == "postprocessing":
                postprocessing()
                # Final compact grid search as the last step of the pipeline
                if getattr(cfg, 'FINAL_ROUND_ENABLED', False):
                    try:
                        console.print("\n[bold blue]Running final grid search round[/bold blue]")
                        run_final_grid_search(mchoice)
                    except Exception as _e:
                        console.print(f"[yellow]Final grid search skipped:[/yellow] {_e}")
                else:
                    console.print("[dim]Final round disabled via configuration; skipping final grid search.[/dim]")
        console.print("\n[bold green]Pipeline Completed Successfully![/bold green]")
    except Exception as e:
        console.print(f"[red]Pipeline failed:[/red] {e}")
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
    # Sorting helpers now moved to module scope for Windows ProcessPool compatibility

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
        # Normalize keys so every row has stable (tile,row,col) or snapped (lat,lon)
        def _normalize_keys(src_csv: str, dst_csv: str) -> tuple[int, int, int]:
            import csv as _csv
            from al_shared import snap_to_pixel_center as _snap
            from config import RAW_DATA_DIR as _RAW
            total, kept, fixed = 0, 0, 0
            # Build output header
            fields_union = []
            have = set()
            with open(src_csv, 'r', newline='') as f0:
                rd0 = _csv.DictReader(f0)
                if rd0.fieldnames:
                    fields_union = list(rd0.fieldnames)
                    have.update(fields_union)
            for k in ("tile","row","col","lat","lon","prob","ndvi","score"):
                if k not in have:
                    fields_union.append(k); have.add(k)
            with open(src_csv, 'r', newline='') as f, open(dst_csv, 'w', newline='') as out:
                rd = _csv.DictReader(f)
                w = _csv.DictWriter(out, fieldnames=fields_union)
                w.writeheader()
                for r in rd:
                    total += 1
                    tile = r.get('tile') or ''
                    # Try row/col
                    rr = cc = None
                    try:
                        rr = int(r.get('row')) if r.get('row') not in (None, '') else None
                        cc = int(r.get('col')) if r.get('col') not in (None, '') else None
                    except Exception:
                        rr = cc = None
                    # Lat/lon
                    la = lo = None
                    try:
                        la = float(r.get('lat')) if r.get('lat') not in (None, '') else None
                        lo = float(r.get('lon')) if r.get('lon') not in (None, '') else None
                    except Exception:
                        la = lo = None
                    # Fill row/col from lat/lon if needed
                    if (rr is None or cc is None) and tile and la is not None and lo is not None and os.path.exists(os.path.join(_RAW, tile)):
                        try:
                            sla, slo, rri, cci = _snap(tile, la, lo)
                            rr, cc = int(rri), int(cci)
                            la, lo = float(sla), float(slo)
                            fixed += 1
                        except Exception:
                            pass
                    # Write normalized row (non-destructive)
                    r_out = dict(r)
                    if rr is not None:
                        r_out['row'] = rr
                    if cc is not None:
                        r_out['col'] = cc
                    if la is not None:
                        r_out['lat'] = f"{la:.7f}"
                    if lo is not None:
                        r_out['lon'] = f"{lo:.7f}"
                    w.writerow({k: r_out.get(k, '') for k in fields_union})
                    kept += 1
            return total, kept, fixed

        norm_src = os.path.join(tmp_dir, f"{base}.norm.csv")
        total_rows, kept_rows, fixed_rows = _normalize_keys(path, norm_src)
        try:
            size = os.path.getsize(norm_src)
        except Exception:
            size = None

        # Phase 1: chunked write + parallel sort by key
        CHUNK_ROWS = int(getattr(cfg, 'DEDUP_CHUNK_ROWS', 200_000))
        raw_chunks = []
        orig_rows = 0
        with open(norm_src, 'r', newline='') as f, _npb() as _prog:
            task = _prog.add_task(f"Dedup (phase 1/3 write): {base}", total=size or None)
            rd = _csv.DictReader(f)
            fieldnames = rd.fieldnames or []
            rows_buf = []
            idx = 0
            last_tell = 0
            for r in rd:
                rows_buf.append(r)
                orig_rows += 1
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
                print(f"[DEDUP] sort-by-key start: chunks={len(raw_chunks)}, workers={max_workers}")
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as ex:
                        futs = {}
                        for rp in raw_chunks:
                            outp = rp.replace('.raw.csv', '.keysorted.csv')
                            fut = ex.submit(_dedup_sort_chunk_by_key, rp, outp)
                            futs[fut] = (rp, outp)
                        for fut in as_completed(futs):
                            rp, outp = futs[fut]
                            err = fut.exception()
                            if err is not None:
                                print(f"[DEDUP] sort-by-key worker failed for {rp}: {err}")
                            elif os.path.exists(outp):
                                sorted_chunks.append(outp)
                            else:
                                print(f"[DEDUP] sort-by-key missing output for {rp}: {outp}")
                            _prog2.update(t2, advance=1)
                except Exception as e:
                    print(f"[DEDUP] sort-by-key pool failed: {e}; falling back to sequential.")
                    for rp in raw_chunks:
                        outp = rp.replace('.raw.csv', '.keysorted.csv')
                        ok = _dedup_sort_chunk_by_key(rp, outp)
                        if ok and os.path.exists(outp):
                            sorted_chunks.append(outp)
                        _prog2.update(t2, advance=1)
                print(f"[DEDUP] sort-by-key done: produced={len(sorted_chunks)}")
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
                print(f"[DEDUP] sort-by-score start: chunks={len(score_raws)}, workers={max_workers}")
                try:
                    with ProcessPoolExecutor(max_workers=max_workers) as ex:
                        futs = {}
                        for rp in score_raws:
                            outp = rp.replace('.raw.csv', '.scoresorted.csv')
                            fut = ex.submit(_dedup_sort_chunk_by_score, rp, outp)
                            futs[fut] = (rp, outp)
                        for fut in as_completed(futs):
                            rp, outp = futs[fut]
                            err = fut.exception()
                            if err is not None:
                                print(f"[DEDUP] sort-by-score worker failed for {rp}: {err}")
                            elif os.path.exists(outp):
                                score_sorted.append(outp)
                            else:
                                print(f"[DEDUP] sort-by-score missing output for {rp}: {outp}")
                            _prog4.update(t4, advance=1)
                except Exception as e:
                    print(f"[DEDUP] sort-by-score pool failed: {e}; falling back to sequential.")
                    for rp in score_raws:
                        outp = rp.replace('.raw.csv', '.scoresorted.csv')
                        ok = _dedup_sort_chunk_by_score(rp, outp)
                        if ok and os.path.exists(outp):
                            score_sorted.append(outp)
                        _prog4.update(t4, advance=1)
                print(f"[DEDUP] sort-by-score done: produced={len(score_sorted)}")
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
        out_rows = 0
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
                out_rows += 1
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

        # Safety: if dedup would result in an empty file but original had rows,
        # do NOT replace. Keep the original as-is and warn.
        if orig_rows > 0 and out_rows == 0:
            try:
                os.remove(final_tmp)
            except Exception:
                pass
            print(f"[WARN] Dedup skipped for {base}: produced 0 rows from {orig_rows}. Keeping original file.")
            try: os.remove(dedup_tmp)
            except Exception: pass
            try: os.rmdir(tmp_dir)
            except Exception: pass
            return False
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
        print(f"Dedup complete: {base} (parallel two-sort). Total={total_rows}, kept={kept_rows}, fixed_keys={fixed_rows}")
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
        # Optional compact summary for Highscore after dedup/normalisation
        try:
            if getattr(cfg, 'HIGHSCORE_SUMMARY_ENABLED', True) and os.path.exists(path):
                import csv as _csv
                lo = max(0.0, float(getattr(cfg,'MIN_AGRI_PROB',0.35)) - float(getattr(cfg,'NEG_LIKE_PROB_DELTA',0.05)))
                hi = float(getattr(cfg,'MIN_AGRI_PROB',0.35))
                nd_rng = getattr(cfg,'NEG_LIKE_NDVI_RANGE',(None,None))
                use_nd = isinstance(nd_rng,(tuple,list)) and nd_rng[0] is not None and nd_rng[1] is not None
                nd_lo, nd_hi = (nd_rng if use_nd else (None,None))
                total = 0; band = 0; band_nd = 0
                below = right1 = right2 = 0
                cols = []
                with open(path, newline='') as f:
                    rd = _csv.DictReader(f)
                    cols = rd.fieldnames or []
                    for r in rd:
                        total += 1
                        try:
                            p = float(r.get('prob'))
                        except Exception:
                            continue
                        if p < lo: below += 1
                        elif p < hi: band += 1
                        elif p < hi + 0.05: right1 += 1
                        else: right2 += 1
                        if use_nd and lo <= p < hi:
                            try:
                                nd=float(r.get('ndvi'))
                            except Exception:
                                nd=None
                            if nd is not None and nd_lo <= nd <= nd_hi:
                                band_nd += 1
                print(f"Highscore summary => rows={total}; HN_prob_only={band} in [{lo:.2f},{hi:.2f}); ", end='')
                if use_nd:
                    print(f"HN_with_NDVI={band_nd}; ", end='')
                else:
                    print(f"HN_with_NDVI=n/a; ", end='')
                print(f"prob dist: <{lo:.2f}={below}, [{lo:.2f},{hi:.2f})={band}, [{hi:.2f},{hi+0.05:.2f})={right1}, ≥{hi+0.05:.2f}={right2}")
                if cols:
                    need = ['tile','row','col','lat','lon']
                    extra = [c for c in ('prob','ndvi','score') if c in cols]
                    print("Highscore columns:", ','.join(need + extra))
        except Exception as _e_sum:
            print(f"Highscore summary skipped: {_e_sum}")
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
    tiles = cfg.list_raw_tiles()
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
