# scripts/a4_phase1_active_learning_loop.py
import os
import csv
import json
from datetime import datetime
from pathlib import Path

import config as cfg
from a3_phase1_active_learning_round import (
    active_learning_round,
    candidate_selection_from_csv,
    candidate_selection_from_predictions,
)
from grid_search import generate_param_combinations
from config import LABELS_FILE, TRAINING_LABELS_FILE, TEMP_LABELS_FILE
from progress_utils import console


def _auto_history_path():
    return getattr(cfg, 'AUTO_TUNE_HISTORY_FILE', os.path.join(cfg.ROUNDS_DIR, 'auto_tuning_history.json'))


_AUTO_HISTORY_MIRROR: Path | None = None

def _current_island_key():
    getter = getattr(cfg, 'get_selected_island', None)
    if callable(getter):
        island = getter()
    else:
        island = None
    if isinstance(island, str) and island.strip():
        return island.strip().lower()
    return "__global__"


def _load_auto_history():
    path = Path(_auto_history_path())
    history: dict = {}
    try:
        if path.exists():
            with path.open('r', encoding='utf-8') as f:
                history = json.load(f) or {}
    except Exception:
        history = {}

    merged = False
    legacy_root = Path(cfg.BASE_DIR) / 'final_results'
    if legacy_root.exists():
        for legacy in legacy_root.rglob('auto_tuning_history.json'):
            try:
                if legacy.resolve() == path.resolve():
                    continue
            except Exception:
                continue
            try:
                with legacy.open('r', encoding='utf-8') as lf:
                    legacy_hist = json.load(lf) or {}
            except Exception:
                continue
            if legacy_hist:
                _merge_auto_history(history, legacy_hist)
                merged = True
    if merged and history:
        try:
            _save_auto_history(history)
        except Exception:
            pass
    return history


def _save_auto_history(history):
    path = Path(_auto_history_path())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Auto-tuning history save failed:[/yellow] {e}")
        return
    mirror = _AUTO_HISTORY_MIRROR
    if mirror is not None:
        try:
            mirror.parent.mkdir(parents=True, exist_ok=True)
            with mirror.open('w', encoding='utf-8') as mf:
                json.dump(history, mf, indent=2)
        except Exception as e:
            console.print(f"[yellow]Auto-tuning mirror save failed:[/yellow] {e}")


def _clone_params(params):
    import copy
    return copy.deepcopy(params)


def _merge_auto_history(base: dict, extra: dict) -> None:
    if not isinstance(extra, dict):
        return
    for model_key, payload in extra.items():
        if not isinstance(payload, dict):
            continue
        if 'params' in payload and not any(isinstance(v, dict) for v in payload.values() if v is not payload):
            payload = {"__global__": payload}
        dest = base.setdefault(model_key, {})
        if isinstance(dest, dict) and 'params' in dest and not any(isinstance(v, dict) for v in dest.values() if v is not dest):
            dest = base[model_key] = {"__global__": dest}
        for island_key, entry in payload.items():
            dest[island_key] = entry


def _combo_signature(params):
    try:
        return json.dumps(params, sort_keys=True, default=str)
    except Exception:
        return str(params)


def _score_metrics(metrics):
    if not metrics:
        return float('-inf')
    if 'macro_f1' in metrics:
        return metrics.get('macro_f1', 0.0)
    if 'f1' in metrics:
        return metrics.get('f1', 0.0)
    if 'auc' in metrics:
        return metrics.get('auc', 0.0)
    if 'roc_auc' in metrics:
        return metrics.get('roc_auc', 0.0)
    return 0.0
from progress_utils import new_progress

def initialize_temp_labels():
    os.makedirs(os.path.dirname(TEMP_LABELS_FILE), exist_ok=True)

    def _read_csv(path):
        if not os.path.exists(path):
            return [], []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames or []
        return header, rows

    def _resolve_header(header, rows):
        if header:
            return header
        ordered = []
        for row in rows:
            for key in row.keys():
                if key not in ordered:
                    ordered.append(key)
        return ordered or ["id", "lat", "lon", "tile", "label", "notes"]

    def _write_csv(path, header, rows):
        fieldnames = _resolve_header(header, rows)
        with open(path, 'w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    fallback_master = getattr(cfg, 'MASTER_LABELS_FILE', LABELS_FILE)
    source_path = TRAINING_LABELS_FILE if os.path.exists(TRAINING_LABELS_FILE) else fallback_master
    src_header, src_rows = _read_csv(source_path)
    filtered_master = cfg.filter_label_rows(src_rows)

    if not os.path.exists(TEMP_LABELS_FILE):
        _write_csv(TEMP_LABELS_FILE, src_header, filtered_master)
        console.print(f"[dim]Created temp_labels.csv from training labels => {TEMP_LABELS_FILE}[/dim]")
        return

    tmp_header, tmp_rows = _read_csv(TEMP_LABELS_FILE)
    filtered_temp = cfg.filter_label_rows(tmp_rows)
    existing_keys = {
        (row.get('tile'), row.get('lat'), row.get('lon'))
        for row in filtered_temp
    }
    merged_rows = list(filtered_temp)
    for row in filtered_master:
        key = (row.get('tile'), row.get('lat'), row.get('lon'))
        if key not in existing_keys:
            merged_rows.append(dict(row))
    _write_csv(TEMP_LABELS_FILE, tmp_header or src_header, merged_rows)
    console.print(f"[dim]Reusing existing temp_labels.csv => {TEMP_LABELS_FILE}[/dim]")

def append_temp_labels(new_labels_file):
    if not os.path.exists(new_labels_file):
        console.print(f"[yellow]No new labels file => {new_labels_file}[/yellow]")
        return
    with open(new_labels_file, "r") as nf:
        rd = csv.DictReader(nf)
        new_rows = list(rd)
    if not new_rows:
        console.print("[yellow]No candidate labels to append.[/yellow]")
        return
    # Append using the master CSV column order: id, lat, lon, tile, label, notes
    with open(TEMP_LABELS_FILE, "a", newline="") as tf:
        wri = csv.writer(tf)
        for r in new_rows:
            wri.writerow([r.get("id"), r.get("lat"), r.get("lon"), r.get("tile"), r.get("label"), r.get("notes", "")])
    console.print(f"[dim]Appended {len(new_rows)} new labels => {TEMP_LABELS_FILE}[/dim]")


def collect_user_hyperparams(model_choice):
    """Interactively collect hyper-parameters for a fixed run.

    Returns a dictionary containing parameter overrides and a ``NAME`` key
    describing the combination for use in output folder names.
    """
    params = {}
    parts = []
    mc = model_choice.lower()
    if mc == "svm":
        C = input("C [0.1,1,10,100]? => ").strip() or "1"
        gamma = input("gamma [scale,auto,0.01,0.1,1.0]? => ").strip() or "scale"
        cw = input("class_weight [none,balanced]? => ").strip().lower() or "none"
        try:
            gamma_val = float(gamma)
        except ValueError:
            gamma_val = gamma
        cw_val = None if cw == "none" else "balanced"
        params["SVM_PARAMS"] = {"C": float(C), "gamma": gamma_val, "class_weight": cw_val}
        parts.extend([f"C-{C}", f"gamma-{gamma}", f"cw-{cw_val}"])
    elif mc == "randomforest":
        ne = input("n_estimators [100,200,400]? => ").strip() or "100"
        md = input("max_depth [6,8,10,12]? => ").strip() or "8"
        ml = input("min_samples_leaf [1,2,4]? => ").strip() or "1"
        cw = input("class_weight [none,balanced]? => ").strip().lower() or "none"
        cw_val = None if cw == "none" else "balanced"
        params["RF_PARAMS"] = {
            "n_estimators": int(ne),
            "max_depth": int(md),
            "min_samples_leaf": int(ml),
            "class_weight": cw_val,
        }
        parts.extend([f"ne-{ne}", f"md-{md}", f"ml-{ml}", f"cw-{cw_val}"])
    elif mc == "ensemble":
        console.print("[dim]Ensemble stacking uses both SVM and RandomForest with current config parameters.[/dim]")
        console.print("[dim]Override base hyperparameters in config.py if needed before running.[/dim]")
        parts.append("stacking")
    auto_best = bool(getattr(cfg, 'AUTO_USE_BEST_THRESHOLD', False))
    if auto_best:
        console.print("[dim]AUTO_USE_BEST_THRESHOLD is enabled; MIN_AGRI_PROB will follow each round's best threshold.[/dim]")
        parts.append("th-best")
    else:
        th = input("MIN_AGRI_PROB [0.3,0.4,0.5,0.6]? => ").strip() or str(cfg.MIN_AGRI_PROB)
        params["MIN_AGRI_PROB"] = float(th)
        parts.append(f"th-{th}")
    sv = input("SIEVE_MIN_SIZE [0,2,5,10,20]? => ").strip() or str(cfg.SIEVE_MIN_SIZE)
    params["SIEVE_MIN_SIZE"] = int(sv)
    parts.append(f"sieve-{sv}")
    params["NAME"] = "_".join(parts)
    return params


def _auto_seed_combos(model_choice):
    model = model_choice.lower()
    seeds = []
    if model == "svm":
        src = getattr(cfg, 'AUTO_TUNE_SVM_SEEDS', []) or []
    elif model == "randomforest":
        src = getattr(cfg, 'AUTO_TUNE_RF_SEEDS', []) or []
    elif model == "ensemble":
        src = getattr(cfg, 'AUTO_TUNE_ENSEMBLE_SEEDS', []) or []
    else:
        src = []
    for idx, seed in enumerate(src, 1):
        params = _clone_params(seed)
        seeds.append((f"auto_seed_{idx}", params))
    return seeds


def _bounded(value, lower, upper):
    return max(lower, min(upper, value))


def _neighbors_for_svm(best_params):
    if not best_params:
        return []
    combos = []
    base = _clone_params(best_params)
    svm_params = _clone_params(base.get('SVM_PARAMS', {}))
    c_val = svm_params.get('C')
    factors = getattr(cfg, 'AUTO_TUNE_SVM_C_FACTORS', [0.5, 1.5])
    for fac in factors:
        if isinstance(c_val, (int, float)) and c_val > 0:
            new_c = _bounded(c_val * fac, 0.05, 20.0)
            new = _clone_params(base)
            new.setdefault('SVM_PARAMS', {})['C'] = round(new_c, 5)
            combos.append((f"auto_neigh_Cx{fac}", new))

    gamma_val = svm_params.get('gamma')
    gamma_factors = getattr(cfg, 'AUTO_TUNE_SVM_NUMERIC_GAMMA_FACTORS', [0.5, 2.0])
    if isinstance(gamma_val, (int, float, float)) and gamma_val > 0:
        for fac in gamma_factors:
            new_gamma = max(gamma_val * fac, 1e-4)
            new = _clone_params(base)
            new.setdefault('SVM_PARAMS', {})['gamma'] = round(new_gamma, 6)
            combos.append((f"auto_neigh_gammax{fac}", new))
    elif isinstance(gamma_val, str):
        options = {'scale', 'auto'}
        options.discard(gamma_val)
        for opt in options:
            new = _clone_params(base)
            new.setdefault('SVM_PARAMS', {})['gamma'] = opt
            combos.append((f"auto_neigh_gamma-{opt}", new))

    steps = getattr(cfg, 'AUTO_TUNE_SIEVE_STEPS', [0, 2]) or []
    sieve = base.get('SIEVE_MIN_SIZE', cfg.SIEVE_MIN_SIZE)
    for step in steps:
        if step == 0:
            continue
        val = max(0, sieve + step if step > 0 else sieve - abs(step))
        if val != sieve:
            new = _clone_params(base)
            new['SIEVE_MIN_SIZE'] = int(val)
            combos.append((f"auto_neigh_sieve_{val}", new))
    return combos


def _neighbors_for_rf(best_params):
    if not best_params:
        return []
    combos = []
    base = _clone_params(best_params)
    rf_params = _clone_params(base.get('RF_PARAMS', {}))
    est_step = int(getattr(cfg, 'AUTO_TUNE_RF_ESTIMATOR_STEP', 100) or 100)
    if est_step > 0 and 'n_estimators' in rf_params:
        for direction in (-1, 1):
            new_estimators = _bounded(rf_params['n_estimators'] + direction * est_step, 150, 800)
            if new_estimators != rf_params['n_estimators']:
                new = _clone_params(base)
                new.setdefault('RF_PARAMS', {})['n_estimators'] = int(new_estimators)
                combos.append((f"auto_neigh_estimators_{new_estimators}", new))

    depth_step = int(getattr(cfg, 'AUTO_TUNE_RF_DEPTH_STEP', 2) or 0)
    if depth_step > 0 and 'max_depth' in rf_params:
        for direction in (-1, 1):
            new_depth = _bounded(rf_params['max_depth'] + direction * depth_step, 4, 20)
            if new_depth != rf_params['max_depth']:
                new = _clone_params(base)
                new.setdefault('RF_PARAMS', {})['max_depth'] = int(new_depth)
                combos.append((f"auto_neigh_depth_{new_depth}", new))

    leaf_opts = getattr(cfg, 'AUTO_TUNE_RF_LEAF_OPTIONS', [1, 2, 3]) or []
    current_leaf = rf_params.get('min_samples_leaf')
    for opt in leaf_opts:
        if opt != current_leaf:
            new = _clone_params(base)
            new.setdefault('RF_PARAMS', {})['min_samples_leaf'] = int(opt)
            combos.append((f"auto_neigh_leaf_{opt}", new))

    steps = getattr(cfg, 'AUTO_TUNE_SIEVE_STEPS', [0, 2]) or []
    sieve = base.get('SIEVE_MIN_SIZE', cfg.SIEVE_MIN_SIZE)
    for step in steps:
        if step == 0:
            continue
        val = max(0, sieve + step if step > 0 else sieve - abs(step))
        if val != sieve:
            new = _clone_params(base)
            new['SIEVE_MIN_SIZE'] = int(val)
            combos.append((f"auto_neigh_sieve_{val}", new))
    return combos


def _neighbors_for_ensemble(best_params):
    combos = []
    if not best_params:
        return combos
    combos.extend(_neighbors_for_svm(best_params))
    combos.extend(_neighbors_for_rf(best_params))
    return combos


def _build_auto_combos(model_choice, history):
    model_key = model_choice.capitalize()
    model_history = history.get(model_key) or {}

    # Backward compatibility: migrate flat entries to per-island structure on the fly
    if model_history and isinstance(model_history, dict) and 'params' in model_history:
        model_history = {"__global__": model_history}
        history[model_key] = model_history

    island_key = _current_island_key()
    history_entry = model_history.get(island_key) or {}
    if not history_entry and model_history.get("__global__"):
        history_entry = model_history.get("__global__")

    best_params = history_entry.get('params')
    candidates = []
    seen = set()

    for name, params in _auto_seed_combos(model_choice):
        sig = _combo_signature(params)
        if sig in seen:
            continue
        seen.add(sig)
        candidates.append((name, params))

    if best_params:
        sig = _combo_signature(best_params)
        if sig not in seen:
            seen.add(sig)
            candidates.append(("auto_prev_best", _clone_params(best_params)))

        neighbors = []
        if model_choice.lower() == 'svm':
            neighbors = _neighbors_for_svm(best_params)
        elif model_choice.lower() == 'randomforest':
            neighbors = _neighbors_for_rf(best_params)
        elif model_choice.lower() == 'ensemble':
            neighbors = _neighbors_for_ensemble(best_params)

        for name, params in neighbors:
            sig = _combo_signature(params)
            if sig in seen:
                continue
            seen.add(sig)
            candidates.append((name, params))

    max_combos = int(getattr(cfg, 'AUTO_TUNE_MAX_COMBOS', 6) or 6)
    if len(candidates) > max_combos:
        candidates = candidates[:max_combos]
    return candidates

def active_learning_loop(
    start_round=1,
    total_rounds=None,
    model_choice=None,
    checkpoint_cb=None,
    tuning_mode="manual",
    user_params=None,
):
    infinite = False
    if total_rounds is None:
        ans = console.input("How many AL rounds? ('infinite' or x amount) => ").strip().lower()
        if ans in ("infinite", "inf"):
            infinite = True
            nr = 1  # seed first round
        else:
            try:
                nr = int(ans)
            except ValueError:
                console.print("[yellow]Invalid value; defaulting to 1 round.[/yellow]")
                nr = 1
    else:
        if isinstance(total_rounds, str) and total_rounds.lower() in ("infinite", "inf"):
            infinite = True
            nr = 1
        else:
            nr = total_rounds

    if model_choice is None:
        models = {"1": "SVM", "2": "RandomForest", "3": "Ensemble"}
        ch = console.input("Choose model => 1=SVM, 2=RandomForest, 3=Ensemble => ").strip()
        mchoice = models.get(ch, "RandomForest")
        if ch not in models:
            console.print("[yellow]Invalid choice; defaulting to RandomForest.[/yellow]")
    else:
        mchoice = model_choice


    initialize_temp_labels()
    init_params = dict(user_params) if user_params else {}

    if checkpoint_cb:
        checkpoint_cb(start_round, nr, mchoice, init_params)

    tuning_mode = (tuning_mode or "manual").lower()

    ensemble_mode = (mchoice.lower() == "ensemble")
    auto_mode = tuning_mode == "auto"
    run_grid = tuning_mode == "grid" and not ensemble_mode
    if tuning_mode == "grid" and ensemble_mode:
        console.print("[yellow]Grid search is disabled for the Ensemble option; running the stacking configuration directly.[/yellow]")

    auto_history = None
    if run_grid:
        combos = generate_param_combinations(mchoice)
        if not combos:
            combos = [("default", {})]
    elif auto_mode:
        auto_history = _load_auto_history()
        combos = _build_auto_combos(mchoice, auto_history)
        if not combos:
            console.print("[yellow]Auto tuning: no candidate combos generated; falling back to seed 1.[/yellow]")
            seeds = _auto_seed_combos(mchoice)
            combos = seeds[:1] if seeds else [("auto_seed_fallback", {})]
    else:
        default_name = "stacking" if ensemble_mode else "manual"
        name = init_params.pop("NAME", default_name)
        combos = [(name, init_params)]

    cached_xy = {}
    r = start_round
    while True:
        if auto_mode or run_grid:
            combo_names = ", ".join(name for name, _ in combos)
            mode_label = "Auto Tuning" if auto_mode else "Grid Search"
            console.print(f"[{mode_label}] Round {r}: evaluating {len(combos)} combos ({combo_names})", style="cyan")
        if run_grid or auto_mode:
            base_min = cfg.MIN_AGRI_PROB
            base_sieve = cfg.SIEVE_MIN_SIZE
            base_svm = cfg.SVM_PARAMS.copy()
            base_rf = cfg.RF_PARAMS.copy()

            results = []
            label = "Grid search combos" if run_grid else "Auto tuning combos"
            test_root = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", "grid_testing" if run_grid else "auto_testing")
            with new_progress() as prog:
                ptask = prog.add_task(label, total=len(combos))
                for name, params in combos:
                    params = _clone_params(params)
                    combo_dir = os.path.join(test_root, name)

                    old_min = cfg.MIN_AGRI_PROB
                    old_sieve = cfg.SIEVE_MIN_SIZE
                    old_svm = cfg.SVM_PARAMS.copy()
                    old_rf = cfg.RF_PARAMS.copy()

                    cfg.MIN_AGRI_PROB = params.get("MIN_AGRI_PROB", cfg.MIN_AGRI_PROB)
                    cfg.SIEVE_MIN_SIZE = params.get("SIEVE_MIN_SIZE", cfg.SIEVE_MIN_SIZE)
                    if "SVM_PARAMS" in params:
                        cfg.SVM_PARAMS.update(params["SVM_PARAMS"])
                    if "RF_PARAMS" in params:
                        cfg.RF_PARAMS.update(params["RF_PARAMS"])

                    metrics = active_learning_round(
                        r,
                        TEMP_LABELS_FILE,
                        mchoice,
                        request_labels=False,
                        out_dir=combo_dir,
                        save_preds=False,
                        return_metrics=True,
                        skip_inference=True,
                        skip_auto_threshold=True,
                        cached_data=cached_xy,
                    )
                    results.append((name, params, metrics))
                    prog.update(ptask, advance=1)

                    cfg.MIN_AGRI_PROB = old_min
                    cfg.SIEVE_MIN_SIZE = old_sieve
                    cfg.SVM_PARAMS = old_svm
                    cfg.RF_PARAMS = old_rf

            cfg.MIN_AGRI_PROB = base_min
            cfg.SIEVE_MIN_SIZE = base_sieve
            cfg.SVM_PARAMS = base_svm
            cfg.RF_PARAMS = base_rf

            console.print(("Grid" if run_grid else "Auto") + " tuning results:", style="bold blue")
            best_idx = 0
            best_score = float('-inf')
            for idx, (name, params, metrics) in enumerate(results):
                sc = _score_metrics(metrics)
                console.print(f"   • {name}: score={sc:.4f}, metrics={metrics}")
                if sc > best_score:
                    best_idx = idx
                    best_score = sc

            best_name, best_params, best_metrics = results[best_idx]
            console.print(f"Selected combination: {best_name} (score={best_score:.4f})", style="green")

            cfg.MIN_AGRI_PROB = best_params.get("MIN_AGRI_PROB", cfg.MIN_AGRI_PROB)
            cfg.SIEVE_MIN_SIZE = best_params.get("SIEVE_MIN_SIZE", cfg.SIEVE_MIN_SIZE)
            if "SVM_PARAMS" in best_params:
                cfg.SVM_PARAMS.update(best_params["SVM_PARAMS"])
            if "RF_PARAMS" in best_params:
                cfg.RF_PARAMS.update(best_params["RF_PARAMS"])

            out_dir = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", best_name)
            mode_label = "Auto Tuning" if auto_mode else "Grid Search"
            console.print(f"[{mode_label}] Round {r}: running selected combo '{best_name}'", style="bold cyan")

            if auto_mode:
                if auto_history is None:
                    auto_history = _load_auto_history()
                model_key = mchoice.capitalize()
                island_key = _current_island_key()
                model_history = auto_history.get(model_key) or {}
                if model_history and isinstance(model_history, dict) and 'params' in model_history:
                    model_history = {"__global__": model_history}
                else:
                    model_history = dict(model_history)
                model_history[island_key] = {
                    "params": _clone_params(best_params),
                    "score": best_score,
                    "metrics": best_metrics,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                }
                auto_history[model_key] = model_history
                _save_auto_history(auto_history)

            if infinite:
                res = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=False,
                    out_dir=out_dir,
                    save_preds=False,
                    return_predictions=True,
                    cached_data=cached_xy,
                )
                go = console.input("Proceed to candidate labeling for this round? [Y/N] => ").strip().lower()
                if go.startswith("y"):
                    if isinstance(res, dict) and res.get("pred_csv"):
                        tmp = candidate_selection_from_csv(res["pred_csv"], out_dir, r)
                    else:
                        preds = res.get("preds") if isinstance(res, dict) else None
                        tmp = candidate_selection_from_predictions(preds, out_dir, r) if preds is not None else None
                else:
                    tmp = None
            else:
                tmp = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=(r < nr),
                    out_dir=out_dir,
                    save_preds=False,
                    cached_data=cached_xy,
                )
            chosen_params = best_params
        else:
            name, params = combos[0]
            combo_dir = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", name)

            old_min = cfg.MIN_AGRI_PROB
            old_sieve = cfg.SIEVE_MIN_SIZE
            old_svm = cfg.SVM_PARAMS.copy()
            old_rf = cfg.RF_PARAMS.copy()
            # no feature-set backup needed

            cfg.MIN_AGRI_PROB = params.get("MIN_AGRI_PROB", cfg.MIN_AGRI_PROB)
            cfg.SIEVE_MIN_SIZE = params.get("SIEVE_MIN_SIZE", cfg.SIEVE_MIN_SIZE)
            if "SVM_PARAMS" in params:
                cfg.SVM_PARAMS.update(params["SVM_PARAMS"])
            if "RF_PARAMS" in params:
                cfg.RF_PARAMS.update(params["RF_PARAMS"])
            # FEATURE_SET removed; ignore

            if infinite:
                # Train/infer/eval first to produce predictions in-memory; ask ONCE before labeling
                res = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=False,
                    out_dir=combo_dir,
                    save_preds=False,
                    return_predictions=True,
                )
                go = input("Proceed to candidate labeling for this round? [Y/N] => ").strip().lower()
                if go.startswith('y'):
                    if isinstance(res, dict) and res.get("pred_csv"):
                        tmp = candidate_selection_from_csv(res["pred_csv"], combo_dir, r)
                    else:
                        preds = res.get("preds") if isinstance(res, dict) else None
                        tmp = candidate_selection_from_predictions(preds, combo_dir, r) if preds is not None else None
                else:
                    tmp = None
            else:
                tmp = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=(r < nr),
                    out_dir=combo_dir,
                    save_preds=False,
                )

            cfg.MIN_AGRI_PROB = old_min
            cfg.SIEVE_MIN_SIZE = old_sieve
            cfg.SVM_PARAMS = old_svm
            cfg.RF_PARAMS = old_rf
            # no feature-set restore needed
            chosen_params = params

        # Post-round bookkeeping
        if isinstance(tmp, str) and os.path.exists(tmp):
            append_temp_labels(tmp)
        # Infinite mode: no post-round prompt; control via the pre-labeling prompt instead
        if infinite:
            if checkpoint_cb:
                checkpoint_cb(r + 1, "infinite", mchoice, chosen_params)
            # If user declined labeling, consider this the end of the loop.
            # Otherwise, continue to next round automatically.
            if not (isinstance(tmp, str) and os.path.exists(tmp)):
                # No labels appended this round (user likely chose N); exit.
                break
            r += 1
            continue
        else:
            if r < nr:
                if checkpoint_cb:
                    checkpoint_cb(r + 1, nr, mchoice, chosen_params)
                r += 1
                continue
            else:
                break

    console.print("[bold green]AL loop complete. Final model saved in this round folder.[/bold green]")

if __name__ == "__main__":
    active_learning_loop()
