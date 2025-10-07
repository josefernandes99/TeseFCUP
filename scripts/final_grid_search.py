#!/usr/bin/env python3
# scripts/final_grid_search.py

import os
import json
import time
from typing import Dict, List, Tuple

import numpy as np

import config as cfg
from splits import load_labels, stratified_train_val_test_indices
from al_shared import extract_features_from_label
from evaluation import compute_best_threshold_weighted
from a3_phase1_active_learning_round import active_learning_round


def _load_xy():
    rows = []
    if os.path.exists(cfg.LABELS_FILE):
        rows.extend(load_labels(cfg.LABELS_FILE))
    if os.path.exists(cfg.TEMP_LABELS_FILE):
        rows.extend(load_labels(cfg.TEMP_LABELS_FILE))
    seen = set(); dedup = []
    for r in rows:
        k = f"{r.get('tile')}:{r.get('lat')}:{r.get('lon')}"
        if k in seen: continue
        seen.add(k); dedup.append(r)
    X, y = [], []
    for r in dedup:
        f = extract_features_from_label(r)
        if f is None: continue
        X.append(f); y.append(1 if r.get('label','').lower()=="agricultural" else 0)
    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _svm_combos() -> List[Tuple[str, Dict]]:
    combos = []
    for C in [1.0, 3.0, 5.0]:
        for gamma in ["scale", "auto"]:
            for cw in [None, "balanced"]:
                name = f"svm_C-{C}_gamma-{gamma}_cw-{cw or 'none'}"
                params = {"SVM_PARAMS": {"C": C, "gamma": gamma, "class_weight": cw}}
                combos.append((name, params))
    return combos


def _rf_combos() -> List[Tuple[str, Dict]]:
    grid = [
        (300, 10, 1),
        (500, 12, 2),
        (400, 14, 1),
        (600, 10, 2),
    ]
    out = []
    for ne, md, ml in grid:
        name = f"rf_ne-{ne}_md-{md}_ml-{ml}"
        out.append((name, {"RF_PARAMS": {"n_estimators": ne, "max_depth": md, "min_samples_leaf": ml, "class_weight": "balanced"}}))
    return out


def _ensemble_combos() -> List[Tuple[str, Dict]]:
    """Single stacking configuration that relies on config-defined base params."""
    return [("stacking", {})]


def _eval_combo(model_choice: str, params: Dict, X, Y) -> Tuple[float, Dict]:
    # Temporarily override config
    backup = {
        "SVM_PARAMS": cfg.SVM_PARAMS.copy(),
        "RF_PARAMS": cfg.RF_PARAMS.copy(),
    }
    try:
        if "SVM_PARAMS" in params: cfg.SVM_PARAMS.update(params["SVM_PARAMS"])  # keep calibration isotonic
        if "RF_PARAMS"  in params: cfg.RF_PARAMS.update(params["RF_PARAMS"])
        # One validation split, then compute best-threshold weighted score
        seed_plot = cfg.SPLIT_RANDOM_SEED if cfg.SPLIT_SEED_MODE == "fixed" else None
        tr, va, _ = stratified_train_val_test_indices(Y, cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.TEST_FRACTION, seed_plot)
        if va.size == 0:
            return -1.0, {"note": "empty validation"}
        from a3_phase1_active_learning_round import train_model
        model = train_model(model_choice, X[tr], Y[tr])
        probs = model.predict_proba(X[va])[:, 1]
        best, best_m = compute_best_threshold_weighted(Y[va], probs)
        return float(best.get("score", 0.0)), {"best": best, "metrics": best_m}
    finally:
        cfg.SVM_PARAMS = backup["SVM_PARAMS"]
        cfg.RF_PARAMS = backup["RF_PARAMS"]


def run_final_grid_search(model_choice: str | None = None):
    if not getattr(cfg, 'FINAL_ROUND_ENABLED', False):
        print('Final round disabled in config; skipping final grid search.')
        return
    if not model_choice:
        print("Final grid search: choose model => 1=SVM 2=RandomForest 3=Ensemble")
        ch = input("=> ").strip()
        model_choice = {"1": "SVM", "2": "RandomForest", "3": "Ensemble"}.get(ch, "RandomForest")
    X, Y = _load_xy()
    if X is None:
        print("No features found; aborting.")
        return
    if model_choice == "SVM":
        combos = _svm_combos()
    elif model_choice == "RandomForest":
        combos = _rf_combos()
    else:
        combos = _ensemble_combos()

    print(f"Evaluating {len(combos)} combos for {model_choice} (train/val only)...")
    scores: List[Tuple[str, float, Dict]] = []
    t0 = time.time()
    for name, params in combos:
        sc, info = _eval_combo(model_choice, params, X, Y)
        print(f" - {name}: score={sc:.4f}")
        scores.append((name, sc, info))
    scores.sort(key=lambda t: t[1], reverse=True)
    best_name, best_score, best_info = scores[0]
    print(f"Selected: {best_name} (score={best_score:.4f})")

    # Apply best params and run one final full-tile inference in a dedicated folder
    out_dir = os.path.join(cfg.ROUNDS_DIR, 'final_round', best_name)
    os.makedirs(out_dir, exist_ok=True)
    # Temporarily set params
    backup = {
        "SVM_PARAMS": cfg.SVM_PARAMS.copy(),
        "RF_PARAMS": cfg.RF_PARAMS.copy(),
    }
    try:
        if model_choice == "SVM":
            for nm, param in _svm_combos():
                if nm == best_name:
                    cfg.SVM_PARAMS.update(param["SVM_PARAMS"])
                    break
        if model_choice == "RandomForest":
            for nm, param in _rf_combos():
                if nm == best_name:
                    cfg.RF_PARAMS.update(param["RF_PARAMS"])
                    break
        # Run without requesting labels and keep predictions ephemeral
        active_learning_round(round_num=0,
                              labels_file=cfg.TEMP_LABELS_FILE if os.path.exists(cfg.TEMP_LABELS_FILE) else cfg.LABELS_FILE,
                              model_choice=model_choice,
                              request_labels=False,
                              out_dir=out_dir,
                              save_preds=False)
        # Store summary
        with open(os.path.join(out_dir, 'final_grid_summary.json'), 'w') as jf:
            json.dump({"best_name": best_name, "best_score": best_score, "best_info": best_info, "duration_min": (time.time()-t0)/60.0}, jf, indent=2)
        print(f"Final grid search completed in {(time.time()-t0)/60.0:.2f} minutes.")
    finally:
        cfg.SVM_PARAMS = backup["SVM_PARAMS"]
        cfg.RF_PARAMS = backup["RF_PARAMS"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["SVM","RandomForest","Ensemble"], default=None)
    args = ap.parse_args()
    run_final_grid_search(args.model)


if __name__ == "__main__":
    main()
