"""Grid search via StratifiedKFold CV over labels.csv (+temp), with summary.

Writes: data/phase1/rounds/grid_<MODEL>_summary/results.json
"""

import os
from multiprocessing import cpu_count
import json
from itertools import product
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score

from config import LABELS_FILE, TEMP_LABELS_FILE, ROUNDS_DIR
import config as cfg
from splits import load_labels, build_feature_matrix
from a3_phase1_active_learning_round import train_model


def _load_union_labels():
    rows = []
    if os.path.exists(LABELS_FILE):
        rows.extend(load_labels(LABELS_FILE))
    if os.path.exists(TEMP_LABELS_FILE):
        rows.extend(load_labels(TEMP_LABELS_FILE))
    # simple de-dup on (tile,lat,lon)
    seen = set(); out = []
    for r in rows:
        k = f"{r.get('tile')}:{r.get('lat')}:{r.get('lon')}"
        if k in seen: continue
        seen.add(k); out.append(r)
    return out


def generate_param_combinations(model_choice):
    combos = []
    if model_choice.lower() == "svm":
        Cs = [0.5, 1, 3, 10]
        gammas = ["scale", "auto"]
        class_weights = ["balanced"]
        for C, g, cw in product(Cs, gammas, class_weights):
            name = f"svm_C-{C}_gamma-{g}_cw-{cw}"
            combos.append((name, {"SVM_PARAMS": {"C": C, "gamma": g, "class_weight": cw}}))
    elif model_choice.lower() == "randomforest":
        n_estimators = [200, 400]
        depths = [8, 12]
        leaves = [1, 2]
        class_weights = ["balanced"]
        for ne, md, ml, cw in product(n_estimators, depths, leaves, class_weights):
            name = f"rf_ne-{ne}_md-{md}_ml-{ml}_cw-{cw}"
            combos.append((name, {"RF_PARAMS": {"n_estimators": ne, "max_depth": md, "min_samples_leaf": ml, "class_weight": cw}}))
    else:
        print("Grid search currently implemented for SVM and RandomForest only.")
    return combos


def run_grid_search(model_choice):
    os.makedirs(ROUNDS_DIR, exist_ok=True)
    rows = _load_union_labels()
    X, y = build_feature_matrix(rows)
    if X.size == 0:
        print("No labeled features to grid-search.")
        return
    # auto-reduce folds
    min_class = min(np.bincount(y)) if len(np.unique(y)) > 1 else 1
    n_splits = max(2, min(cfg.CV_FOLDS, min_class)) if cfg.CV_AUTO_REDUCE else cfg.CV_FOLDS
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=None if cfg.SPLIT_SEED_MODE=="random" else cfg.SPLIT_RANDOM_SEED)

    results = []
    combos = generate_param_combinations(model_choice)
    for name, params in combos:
        # backup & apply
        old_svm = cfg.SVM_PARAMS.copy(); old_rf = cfg.RF_PARAMS.copy()
        if "SVM_PARAMS" in params: cfg.SVM_PARAMS.update(params["SVM_PARAMS"])
        if "RF_PARAMS" in params: cfg.RF_PARAMS.update(params["RF_PARAMS"])
        scores = {"f1": [], "acc": [], "auc": [], "auc_pr": []}
        for train_idx, val_idx in skf.split(X, y):
            Xm, ym = X[train_idx], y[train_idx]
            model = train_model(model_choice, Xm, ym)
            pv = model.predict_proba(X[val_idx])[:,1]
            yv = y[val_idx]
            yhat = (pv >= cfg.MIN_AGRI_PROB).astype(int)
            scores["f1"].append(f1_score(yv, yhat, zero_division=0))
            scores["acc"].append(accuracy_score(yv, yhat))
            try:
                scores["auc"].append(roc_auc_score(yv, pv))
            except Exception:
                scores["auc"].append(0.0)
            try:
                scores["auc_pr"].append(average_precision_score(yv, pv))
            except Exception:
                scores["auc_pr"].append(0.0)
        # restore
        cfg.SVM_PARAMS = old_svm; cfg.RF_PARAMS = old_rf
        res = {k: float(np.mean(v)) for k, v in scores.items()}
        res.update({f"std_{k}": float(np.std(v)) for k, v in scores.items()})
        results.append({"name": name, "params": params, "metrics": res})

    # choose best by F1 then AUC_PR
    def key(m):
        return (m["metrics"].get("f1", 0.0), m["metrics"].get("auc_pr", 0.0))
    best = max(results, key=key)
    out_dir = os.path.join(ROUNDS_DIR, f"grid_{model_choice}_summary")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"best": best, "all": results, "folds": n_splits}, f, indent=2)
    print(f"Grid search complete. Summary => {out_dir}/results.json")


if __name__ == "__main__":
    run_grid_search("SVM")

# Threading caps to avoid OpenBLAS/OpenMP warnings and oversubscription
_N_THREADS = str(min(8, max(1, cpu_count()), 24))
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _N_THREADS)
