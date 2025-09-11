#!/usr/bin/env python3
# scripts/refresh_lists.py

import os
import sys
import glob
import time
from typing import Optional


import config as cfg
from splits import load_labels
from splits import build_feature_matrix
from a3_phase1_active_learning_round import refresh_global_lists_full
from progress_utils import new_progress


def _find_latest_predictions_csv() -> Optional[str]:
    base = cfg.ROUNDS_DIR
    patt = os.path.join(base, "**", "predictions.csv")
    files = glob.glob(patt, recursive=True)
    if not files:
        return None
    # choose the most recently modified
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _dedup_rows(rows):
    seen = set()
    out = []
    for r in rows:
        k = f"{r.get('tile')}:{r.get('lat')}:{r.get('lon')}"
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def main():
    # Honor per-list toggles: if both disabled, skip updates silently.
    hs_on = bool(getattr(cfg, 'HIGHSCORE_LIST_ENABLED', True))
    pa_on = bool(getattr(cfg, 'PROBABLE_AGRI_LIST_ENABLED', False))
    if not (hs_on or pa_on):
        print("Persistent lists disabled in config; skipping refresh.")
        return
    pred_csv = sys.argv[1] if len(sys.argv) > 1 else None
    if pred_csv is None:
        pred_csv = _find_latest_predictions_csv()
        if not pred_csv:
            print("No predictions.csv found under rounds/. Pass a path explicitly.")
            sys.exit(1)
    if not os.path.exists(pred_csv):
        print(f"Missing predictions.csv => {pred_csv}")
        sys.exit(1)
    round_dir = os.path.dirname(pred_csv)
    print(f"Using predictions: {pred_csv}")

    # Build training rows from both master and temp (for representativeness)
    rows = []
    if os.path.exists(cfg.LABELS_FILE):
        rows.extend(load_labels(cfg.LABELS_FILE))
    if os.path.exists(cfg.TEMP_LABELS_FILE):
        rows.extend(load_labels(cfg.TEMP_LABELS_FILE))
    rows = _dedup_rows(rows)
    print(f"Training rows collected: {len(rows)} (labels + temp_labels)")
    # Graceful fallback: if no labels or features, proceed without representativeness (R=0)
    X = y = []
    if rows:
        try:
            X, y = build_feature_matrix(rows)
        except Exception as e:
            print(f"Feature extraction failed ({e}); representativeness disabled (R=0).")
    if not hasattr(X, 'size') or getattr(X, 'size', 0) == 0:
        print("Fallback engaged: no usable training features.\n"
              "- Highscore ranking = Uncertainty (U) + Consistency (C); Representativeness (R) omitted.\n"
              "- ProbableAgri ranking = Probability only; Representativeness (R) omitted.")
    else:
        print(f"Training feature matrix: X={X.shape}, y={y.shape}")

    start = time.time()
    print("Starting global highscore/probableAgri refresh...")
    refresh_global_lists_full(pred_csv=pred_csv,
                              round_dir=round_dir,
                              round_num=0,
                              train_rows=rows,
                              X_train=X,
                              y_train=y)
    dur = time.time() - start
    print(f"Global lists refreshed in {dur/60.0:.2f} minutes.")

    # Also build/update a persistent feature-importance leaderboard across rounds
    try:
        _update_global_feature_importance()
    except Exception as e:
        print(f"Global feature-importance aggregation failed: {e}")


def _update_global_feature_importance():
    """Aggregate permutation importances across all rounds and write a global CSV.

    Scans: data/phase1/rounds/round_*/**/statistics/feature_importance.txt
           data/phase1/rounds/final_round/*/statistics/feature_importance.txt
    Writes: labels/phase1/features_importance_global.csv
    """
    import glob as _glob
    import csv as _csv
    import numpy as _np
    from collections import defaultdict
    print("Aggregating feature importances across rounds...")
    patt1 = os.path.join(cfg.ROUNDS_DIR, "round_*", "*", "statistics", "feature_importance.txt")
    patt2 = os.path.join(cfg.ROUNDS_DIR, "final_round", "*", "statistics", "feature_importance.txt")
    files = sorted(set(_glob.glob(patt1)) | set(_glob.glob(patt2)))
    if not files:
        print("No feature_importance.txt files found; skipping aggregation.")
        return
    fmap = defaultdict(list)
    with new_progress() as prog:
        t = prog.add_task("Scan FI files", total=len(files))
        for fp in files:
            try:
                with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        if len(parts) < 2:
                            parts = line.split()
                            if len(parts) < 2:
                                continue
                        name = parts[0]
                        try:
                            val = float(parts[1])
                        except Exception:
                            continue
                        fmap[name].append(val)
            except Exception:
                pass
            prog.update(t, advance=1)
    rows = []
    for k, vs in fmap.items():
        arr = _np.array(vs, dtype=float)
        rows.append((k, float(arr.mean()), float(arr.std()), int(arr.size)))
    rows.sort(key=lambda t: t[1], reverse=True)
    out = os.path.join(cfg.LABELS_DIR, 'features_importance_global.csv')
    with open(out, 'w', newline='') as f:
        w = _csv.writer(f)
        w.writerow(['feature', 'mean_importance', 'std_importance', 'count'])
        for name, mean, std, cnt in rows:
            w.writerow([name, f"{mean:.6f}", f"{std:.6f}", cnt])
    print(f"Feature-importance leaderboard => {out}")


if __name__ == "__main__":
    main()
