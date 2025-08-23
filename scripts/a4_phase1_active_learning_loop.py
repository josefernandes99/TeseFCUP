# scripts/a4_phase1_active_learning_loop.py
import os
import shutil
import csv

import config as cfg
from a3_phase1_active_learning_round import active_learning_round, candidate_selection_from_csv
from grid_search import generate_param_combinations
from config import LABELS_FILE, TEMP_LABELS_FILE

def initialize_temp_labels():
    if not os.path.exists(TEMP_LABELS_FILE):
        shutil.copyfile(LABELS_FILE, TEMP_LABELS_FILE)
        print(f"Created temp_labels.csv from master labels => {TEMP_LABELS_FILE}")
    else:
        print(f"Reusing existing temp_labels.csv => {TEMP_LABELS_FILE}")

def append_temp_labels(new_labels_file):
    if not os.path.exists(new_labels_file):
        print(f"No new labels file => {new_labels_file}")
        return
    with open(new_labels_file, "r") as nf:
        rd = csv.DictReader(nf)
        new_rows = list(rd)
    if not new_rows:
        print("No candidate labels to append.")
        return
    # Append using the master CSV column order: id, lat, lon, tile, label, notes
    with open(TEMP_LABELS_FILE, "a", newline="") as tf:
        wri = csv.writer(tf)
        for r in new_rows:
            wri.writerow([r.get("id"), r.get("lat"), r.get("lon"), r.get("tile"), r.get("label"), r.get("notes", "")])
    print(f"Appended {len(new_rows)} new labels => {TEMP_LABELS_FILE}")


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
    th = input("MIN_AGRI_PROB [0.3,0.4,0.5,0.6]? => ").strip() or str(cfg.MIN_AGRI_PROB)
    sv = input("SIEVE_MIN_SIZE [0,2,5,10,20]? => ").strip() or str(cfg.SIEVE_MIN_SIZE)
    # Feature set selection
    print("Feature set => 1=base (working), 2=base+temporal (currently unsupported), 3=base+textures (currently unsupported), 4=base+temporal+textures (currently unsupported), 5=full (currently unsupported)")
    fs_map = {"1": "base", "2": "temporal_only", "3": "textures_only", "4": "temporal_textures", "5": "full"}
    fsel = input("=> ").strip()
    fset = fs_map.get(fsel, cfg.FEATURE_SET)
    params["MIN_AGRI_PROB"] = float(th)
    params["SIEVE_MIN_SIZE"] = int(sv)
    params["FEATURE_SET"] = fset
    parts.extend([f"th-{th}", f"sieve-{sv}", f"feats-{fset}"])
    params["NAME"] = "_".join(parts)
    return params

def active_learning_loop(
    start_round=1,
    total_rounds=None,
    model_choice=None,
    checkpoint_cb=None,
    use_grid_search=True,
    user_params=None,
):
    infinite = False
    if total_rounds is None:
        ans = input("How many AL rounds? ('infinite' or x amount) => ").strip().lower()
        if ans in ("infinite", "inf"):
            infinite = True
            nr = 1  # seed first round
        else:
            try:
                nr = int(ans)
            except ValueError:
                print("Invalid => default=1")
                nr = 1
    else:
        if isinstance(total_rounds, str) and total_rounds.lower() in ("infinite", "inf"):
            infinite = True
            nr = 1
        else:
            nr = total_rounds

    if model_choice is None:
        print("Choose model => 1=ResNet, 2=SVM, 3=RandomForest")
        models = ["ResNet", "SVM", "RandomForest"]
        ch = input("=> ").strip()
        if ch in ["1", "2", "3"]:
            mchoice = models[int(ch) - 1]
        else:
            print("Invalid => default=RandomForest")
            mchoice = "RandomForest"
    else:
        mchoice = model_choice


    initialize_temp_labels()
    init_params = dict(user_params) if user_params else {}

    if checkpoint_cb:
        checkpoint_cb(start_round, nr, mchoice, init_params)

    if use_grid_search:
        combos = generate_param_combinations(mchoice)
        if not combos:
            combos = [("default", {})]
    else:
        name = init_params.pop("NAME", "manual")
        combos = [(name, init_params)]

    r = start_round
    while True:
        if use_grid_search:
            base_min = cfg.MIN_AGRI_PROB
            base_sieve = cfg.SIEVE_MIN_SIZE
            base_svm = cfg.SVM_PARAMS.copy()
            base_rf = cfg.RF_PARAMS.copy()
            base_fs = cfg.FEATURE_SET

            results = []
            for name, params in combos:
                combo_dir = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", name)

                # backup current settings
                old_min = cfg.MIN_AGRI_PROB
                old_sieve = cfg.SIEVE_MIN_SIZE
                old_svm = cfg.SVM_PARAMS.copy()
                old_rf = cfg.RF_PARAMS.copy()
                old_fs = cfg.FEATURE_SET

                # apply params
                cfg.MIN_AGRI_PROB = params.get("MIN_AGRI_PROB", cfg.MIN_AGRI_PROB)
                cfg.SIEVE_MIN_SIZE = params.get("SIEVE_MIN_SIZE", cfg.SIEVE_MIN_SIZE)
                if "SVM_PARAMS" in params:
                    cfg.SVM_PARAMS.update(params["SVM_PARAMS"])
                if "RF_PARAMS" in params:
                    cfg.RF_PARAMS.update(params["RF_PARAMS"])
                if "FEATURE_SET" in params:
                    cfg.FEATURE_SET = params["FEATURE_SET"]

                metrics = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=False,
                    out_dir=combo_dir,
                    save_preds=False,
                    return_metrics=True,
                )
                results.append((name, params, metrics))

                # restore
                cfg.MIN_AGRI_PROB = old_min
                cfg.SIEVE_MIN_SIZE = old_sieve
                cfg.SVM_PARAMS = old_svm
                cfg.RF_PARAMS = old_rf
                cfg.FEATURE_SET = old_fs

            # restore to baseline before scoring
            cfg.MIN_AGRI_PROB = base_min
            cfg.SIEVE_MIN_SIZE = base_sieve
            cfg.SVM_PARAMS = base_svm
            cfg.RF_PARAMS = base_rf
            cfg.FEATURE_SET = base_fs
            # FEATURE_SET remains as global default

            def score(metrics):
                return metrics.get("macro_f1", metrics.get("auc", 0.0)) if metrics else 0.0

            print("Grid search results:")
            best_idx = 0
            best_score = -float("inf")
            for idx, (name, params, metrics) in enumerate(results):
                sc = score(metrics)
                print(f" - {name}: score={sc:.4f}, metrics={metrics}")
                if sc > best_score:
                    best_idx = idx
                    best_score = sc

            best_name, best_params, best_metrics = results[best_idx]
            print(f"Selected combination: {best_name} (score={best_score:.4f})")

            # apply best params
            cfg.MIN_AGRI_PROB = best_params.get("MIN_AGRI_PROB", cfg.MIN_AGRI_PROB)
            cfg.SIEVE_MIN_SIZE = best_params.get("SIEVE_MIN_SIZE", cfg.SIEVE_MIN_SIZE)
            if "SVM_PARAMS" in best_params:
                cfg.SVM_PARAMS.update(best_params["SVM_PARAMS"])
            if "RF_PARAMS" in best_params:
                cfg.RF_PARAMS.update(best_params["RF_PARAMS"])
            if "FEATURE_SET" in best_params:
                cfg.FEATURE_SET = best_params["FEATURE_SET"]

            out_dir = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", best_name)
            # In infinite mode: run train/infer/eval first (to produce predictions),
            # then ask ONCE whether to proceed to candidate labeling; no post-round prompt.
            if infinite:
                _ = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=False,
                    out_dir=out_dir,
                    save_preds=True,
                )
                go = input("Proceed to candidate labeling for this round? [Y/N] => ").strip().lower()
                if go.startswith("y"):
                    pred_csv = os.path.join(str(out_dir), "predictions.csv")
                    tmp = candidate_selection_from_csv(pred_csv, out_dir, r)
                else:
                    tmp = None
            else:
                tmp = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=(r < nr),
                    out_dir=out_dir,
                    save_preds=True,
                )
            chosen_params = best_params
        else:
            name, params = combos[0]
            combo_dir = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", name)

            old_min = cfg.MIN_AGRI_PROB
            old_sieve = cfg.SIEVE_MIN_SIZE
            old_svm = cfg.SVM_PARAMS.copy()
            old_rf = cfg.RF_PARAMS.copy()
            old_fs = cfg.FEATURE_SET

            cfg.MIN_AGRI_PROB = params.get("MIN_AGRI_PROB", cfg.MIN_AGRI_PROB)
            cfg.SIEVE_MIN_SIZE = params.get("SIEVE_MIN_SIZE", cfg.SIEVE_MIN_SIZE)
            if "SVM_PARAMS" in params:
                cfg.SVM_PARAMS.update(params["SVM_PARAMS"])
            if "RF_PARAMS" in params:
                cfg.RF_PARAMS.update(params["RF_PARAMS"])
            if "FEATURE_SET" in params:
                cfg.FEATURE_SET = params["FEATURE_SET"]

            if infinite:
                # Train/infer/eval first to produce predictions; ask ONCE before labeling
                _ = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=False,
                    out_dir=combo_dir,
                    save_preds=True,
                )
                go = input("Proceed to candidate labeling for this round? [Y/N] => ").strip().lower()
                if go.startswith('y'):
                    pred_csv = os.path.join(combo_dir, "predictions.csv")
                    tmp = candidate_selection_from_csv(pred_csv, combo_dir, r)
                else:
                    tmp = None
            else:
                tmp = active_learning_round(
                    r,
                    TEMP_LABELS_FILE,
                    mchoice,
                    request_labels=(r < nr),
                    out_dir=combo_dir,
                )

            cfg.MIN_AGRI_PROB = old_min
            cfg.SIEVE_MIN_SIZE = old_sieve
            cfg.SVM_PARAMS = old_svm
            cfg.RF_PARAMS = old_rf
            cfg.FEATURE_SET = old_fs
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

    print("AL loop done. Final model => last round folder.")

if __name__ == "__main__":
    active_learning_loop()
