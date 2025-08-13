"""Simple hyper-parameter grid search utility.

Each combination of parameters is trained using the existing active learning
round code (without requesting additional labels). Results and evaluation
metrics are stored in a dedicated sub-folder inside the rounds directory.
"""

import os
import json
from itertools import product

from config import LABELS_FILE, ROUNDS_DIR
import config as cfg
from a3_phase1_active_learning_round import active_learning_round
from evaluation import evaluate_model


def generate_param_combinations(model_choice):
    """Return a list of (name, params) tuples for the given model."""
    thresholds = [0.5]
    sieves = [0, 5]
    combos = []
    if model_choice.lower() == "svm":
        Cs = [0.1, 1]
        gammas = ["auto", 0.1]
        class_weights = [None, "balanced"]
        for C, g, cw, th, sv in product(Cs, gammas, class_weights, thresholds, sieves):
            name = f"C-{C}_gamma-{g}_cw-{cw}_th-{th}_sieve-{sv}_"
            params = {"C": C, "gamma": g, "class_weight": cw}
            combos.append((name, {
                "SVM_PARAMS": params,
                "MIN_AGRI_PROB": th,
                "SIEVE_MIN_SIZE": sv,
            }))
    elif model_choice.lower() == "randomforest":
        n_estimators = [100, 200, 400]
        depths = [6, 8, 10, 12]
        leaves = [1, 2, 4]
        class_weights = [None, "balanced"]
        for ne, md, ml, cw, th, sv in product(n_estimators, depths, leaves, class_weights, thresholds, sieves):
            name = f"ne-{ne}_md-{md}_ml-{ml}_cw-{cw}_th-{th}_sieve-{sv}"
            combos.append((name, {
                "RF_PARAMS": {
                    "n_estimators": ne,
                    "max_depth": md,
                    "min_samples_leaf": ml,
                    "class_weight": cw,
                },
                "MIN_AGRI_PROB": th,
                "SIEVE_MIN_SIZE": sv,
            }))
    else:
        print("Grid search currently implemented for SVM and RandomForest only.")
    return combos


def run_grid_search(model_choice):
    os.makedirs(ROUNDS_DIR, exist_ok=True)
    combos = generate_param_combinations(model_choice)
    if not combos:
        return

    for idx, (name, params) in enumerate(combos, 1):
        print(f"\n=== Combination {idx}/{len(combos)} ===")
        out_dir = os.path.join(ROUNDS_DIR, f"grid_{model_choice}_{name}")

        # backup current settings
        old_min = cfg.MIN_AGRI_PROB
        old_sieve = cfg.SIEVE_MIN_SIZE
        old_svm = cfg.SVM_PARAMS.copy()
        old_rf = cfg.RF_PARAMS.copy()

        # update globals for this run
        cfg.MIN_AGRI_PROB = params.get("MIN_AGRI_PROB", cfg.MIN_AGRI_PROB)
        cfg.SIEVE_MIN_SIZE = params.get("SIEVE_MIN_SIZE", cfg.SIEVE_MIN_SIZE)
        if "SVM_PARAMS" in params:
            cfg.SVM_PARAMS.update(params["SVM_PARAMS"])
        if "RF_PARAMS" in params:
            cfg.RF_PARAMS.update(params["RF_PARAMS"])

        # always use round 1 for each combination to avoid state bleed
        round_num = 1
        src_dir = os.path.join(ROUNDS_DIR, f"round_{round_num}")
        if os.path.exists(src_dir):
            import shutil
            shutil.rmtree(src_dir)

        # run a training round without requesting extra labels or prediction CSV
        active_learning_round(round_num, LABELS_FILE, model_choice, request_labels=False, save_preds=False)

        if os.path.exists(out_dir):
            import shutil
            shutil.rmtree(out_dir)
        os.rename(src_dir, out_dir)

        # evaluate if possible
        model_path = os.path.join(out_dir, f"model_round_{round_num}.pkl")
        if os.path.exists(model_path):
            from joblib import load
            model = load(model_path)
            metrics = evaluate_model(model)
            with open(os.path.join(out_dir, "metrics.json"), "w") as jf:
                json.dump(metrics or {}, jf, indent=2)

        # restore globals
        cfg.MIN_AGRI_PROB = old_min
        cfg.SIEVE_MIN_SIZE = old_sieve
        cfg.SVM_PARAMS = old_svm
        cfg.RF_PARAMS = old_rf

    print("Grid search complete.")


if __name__ == "__main__":
    run_grid_search("SVM")

