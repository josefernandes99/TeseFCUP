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
    """Interactively collect hyper-parameters for a fixed run."""
    params = {}
    mc = model_choice.lower()
    if mc == "svm":
        C = input("C [0.1,1,10,100]? => ").strip() or "1"
        gamma = input("gamma [scale,auto,0.01,0.1,1.0]? => ").strip() or "scale"
        try:
            gamma_val = float(gamma)
        except ValueError:
            gamma_val = gamma
        params["SVM_PARAMS"] = {"C": float(C), "gamma": gamma_val}
    elif mc == "randomforest":
        ne = input("n_estimators [100,200,400]? => ").strip() or "100"
        md = input("max_depth [6,8,10,12]? => ").strip() or "8"
        ml = input("min_samples_leaf [1,2,4]? => ").strip() or "1"
        params["RF_PARAMS"] = {"n_estimators": int(ne), "max_depth": int(md), "min_samples_leaf": int(ml)}
    th = input("MIN_AGRI_PROB [0.3,0.4,0.5,0.6]? => ").strip() or str(cfg.MIN_AGRI_PROB)
    sv = input("SIEVE_MIN_SIZE [0,2,5,10,20]? => ").strip() or str(cfg.SIEVE_MIN_SIZE)
    params["MIN_AGRI_PROB"] = float(th)
    params["SIEVE_MIN_SIZE"] = int(sv)
    return params

def active_learning_loop(start_round=1, total_rounds=None, model_choice=None,
                        checkpoint_cb=None, use_grid_search=True, user_params=None):
    if total_rounds is None:
        try:
            nr = int(input("How many AL rounds? => "))
        except ValueError:
            print("Invalid => default=1")
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

    if checkpoint_cb:
        checkpoint_cb(start_round, nr, mchoice)

    if use_grid_search:
        combos = generate_param_combinations(mchoice)
        if not combos:
            combos = [("default", {})]
    else:
        combos = [("manual", user_params or {})]

    for r in range(start_round, nr + 1):
        combo_names = []
        for name, params in combos:
            combo_dir = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", name)

            # backup current settings
            old_min = cfg.MIN_AGRI_PROB
            old_sieve = cfg.SIEVE_MIN_SIZE
            old_svm = cfg.SVM_PARAMS.copy()
            old_rf = cfg.RF_PARAMS.copy()

            # apply params
            cfg.MIN_AGRI_PROB = params.get("MIN_AGRI_PROB", cfg.MIN_AGRI_PROB)
            cfg.SIEVE_MIN_SIZE = params.get("SIEVE_MIN_SIZE", cfg.SIEVE_MIN_SIZE)
            if "SVM_PARAMS" in params:
                cfg.SVM_PARAMS.update(params["SVM_PARAMS"])
            if "RF_PARAMS" in params:
                cfg.RF_PARAMS.update(params["RF_PARAMS"])

            active_learning_round(r, TEMP_LABELS_FILE, mchoice, request_labels=False, out_dir=combo_dir)

            # restore
            cfg.MIN_AGRI_PROB = old_min
            cfg.SIEVE_MIN_SIZE = old_sieve
            cfg.SVM_PARAMS = old_svm
            cfg.RF_PARAMS = old_rf
            combo_names.append(name)

        if use_grid_search and r < nr:
            print("Available combinations:")
            for n in combo_names:
                print(f" - {n}")
            chosen = input("what combination to use? => ").strip()
            if chosen not in combo_names:
                print("Invalid choice; defaulting to first")
                chosen = combo_names[0]
            selected_dir = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", chosen)
            pred_csv = os.path.join(selected_dir, "predictions.csv")
            if os.path.exists(pred_csv):
                newfile = candidate_selection_from_csv(pred_csv, selected_dir, r)
                if newfile:
                    append_temp_labels(newfile)
            else:
                print(f"Missing predictions CSV => {pred_csv}")
            if checkpoint_cb:
                checkpoint_cb(r + 1, nr, mchoice)
        elif not use_grid_search and r < nr:
            combo_name = combos[0][0]
            selected_dir = os.path.join(cfg.ROUNDS_DIR, f"round_{r}", combo_name)
            pred_csv = os.path.join(selected_dir, "predictions.csv")
            if os.path.exists(pred_csv):
                newfile = candidate_selection_from_csv(pred_csv, selected_dir, r)
                if newfile:
                    append_temp_labels(newfile)
            else:
                print(f"Missing predictions CSV => {pred_csv}")
            if checkpoint_cb:
                checkpoint_cb(r + 1, nr, mchoice)

    print("AL loop done. Final model => last round folder.")

if __name__ == "__main__":
    active_learning_loop()
