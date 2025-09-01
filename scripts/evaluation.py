import os
from multiprocessing import cpu_count
import json
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
os.environ.setdefault("MPLBACKEND", "Agg")
# Threading caps to avoid OpenBLAS/OpenMP warnings and oversubscription
_N_THREADS = str(min(8, max(1, cpu_count()), 24))
os.environ["OMP_NUM_THREADS"] = _N_THREADS
os.environ["MKL_NUM_THREADS"] = _N_THREADS
os.environ.setdefault("OPENBLAS_NUM_THREADS", _N_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _N_THREADS)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from al_shared import extract_features_from_label
import config as cfg
from splits import load_labels, stratified_train_val_test_indices

def _single_split_metrics(model, X, y, seed=None):
    tr_idx, va_idx, _ = stratified_train_val_test_indices(
        y, cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.TEST_FRACTION,
        seed if cfg.SPLIT_SEED_MODE == "fixed" else None,
    )
    if va_idx.size == 0:
        return None
    probs = model.predict_proba(X[va_idx])[:, 1]
    preds = (probs >= cfg.MIN_AGRI_PROB).astype(int)
    yv = y[va_idx]
    cm = confusion_matrix(yv, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": precision_score(yv, preds, zero_division=0),
        "recall": recall_score(yv, preds, zero_division=0),
        "f1": f1_score(yv, preds, zero_division=0),
        "macro_f1": f1_score(yv, preds, average="macro", zero_division=0),
        "accuracy": accuracy_score(yv, preds),
        "auc": roc_auc_score(yv, probs),
        "auc_pr": average_precision_score(yv, probs),
        "balanced_accuracy": ((tp/(tp+fn) if (tp+fn)>0 else 0) + (tn/(tn+fp) if (tn+fp)>0 else 0))/2,
        "mcc": matthews_corrcoef(yv, preds) if len(np.unique(yv))>1 and len(np.unique(preds))>1 else 0.0,
        "cohen_kappa": cohen_kappa_score(yv, preds),
        "brier_score": brier_score_loss(yv, probs),
    }
    return metrics


def evaluate_model(model, out_dir=None):
    """Evaluate model on a stratified validation split from labels + temp_labels.

    Parameters
    ----------
    model : sklearn-like estimator
        Trained model implementing ``predict_proba``.
    out_dir : str, optional
        If provided, write a ``metrics.json`` file and diagnostic plots
        (confusion matrix, ROC curve) into this directory.

    Returns
    -------
    dict | None
        Dictionary of scalar metrics or ``None`` if evaluation labels are
        missing.
    """
    rows = []
    if os.path.exists(cfg.LABELS_FILE):
        rows.extend(load_labels(cfg.LABELS_FILE))
    if os.path.exists(cfg.TEMP_LABELS_FILE):
        rows.extend(load_labels(cfg.TEMP_LABELS_FILE))
    if not rows:
        print("No labels found for evaluation.")
        return None
    # de-dup simple: pixel key via (tile, lat, lon) string combo
    seen = set()
    dedup = []
    for r in rows:
        k = f"{r.get('tile')}:{r.get('lat')}:{r.get('lon')}"
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    rows = dedup

    X, y = [], []
    for r in rows:
        feats = extract_features_from_label(r)
        if feats is None:
            continue
        X.append(feats)
        y.append(1 if r.get("label", "").lower() == "agricultural" else 0)
    if not X:
        print("No features extracted for evaluation set.")
        return None
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    # Repeated validation with aggregation
    repeats = max(1, int(getattr(cfg, "REPEATED_VALIDATION_REPEATS", 1)))
    all_metrics = []
    for i in range(repeats):
        seed = (cfg.SPLIT_RANDOM_SEED + i) if cfg.SPLIT_SEED_MODE == "fixed" else None
        m = _single_split_metrics(model, X, y, seed=seed)
        if m is not None:
            all_metrics.append(m)
    if not all_metrics:
        print("Validation split empty; skipping evaluation.")
        return None
    # aggregate: mean/std for numeric metrics
    keys = list(all_metrics[0].keys())
    metrics_mean = {}
    metrics_std = {}
    for k in keys:
        try:
            vals = np.array([float(m[k]) for m in all_metrics], dtype=float)
            metrics_mean[k] = float(np.mean(vals))
            metrics_std[f"std_{k}"] = float(np.std(vals))
        except Exception:
            metrics_mean[k] = all_metrics[0][k]
    metrics = {**metrics_mean, **metrics_std}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # Recompute a canonical split for plots/report (first repeat's seed)
        seed_plot = cfg.SPLIT_RANDOM_SEED if cfg.SPLIT_SEED_MODE == "fixed" else None
        tr_idx, va_idx, _ = stratified_train_val_test_indices(
            y, cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.TEST_FRACTION,
            seed_plot,
        )
        if va_idx.size:
            probs = model.predict_proba(X[va_idx])[:, 1]
            preds = (probs >= cfg.MIN_AGRI_PROB).astype(int)
            yv = y[va_idx]
            cm = confusion_matrix(yv, preds, labels=[0, 1])
            # classification report
            report = classification_report(yv, preds, digits=3)
            with open(os.path.join(out_dir, "classification_report.txt"), "w") as rf:
                rf.write(report)

            # confusion matrix plot
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NonAgri", "Agri"])
            fig, ax = plt.subplots(figsize=(4, 4))
            disp.plot(ax=ax, colorbar=False)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "confusion_matrix.png"))
            plt.close(fig)

            # ROC curve
            roc_disp = RocCurveDisplay.from_predictions(yv, probs)
            roc_disp.figure_.savefig(os.path.join(out_dir, "roc_curve.png"))
            plt.close(roc_disp.figure_)

            # Precision-Recall curve
            pr_disp = PrecisionRecallDisplay.from_predictions(yv, probs)
            pr_disp.figure_.savefig(os.path.join(out_dir, "pr_curve.png"))
            plt.close(pr_disp.figure_)

        # metrics JSON for programmatic use (mean/std when repeats>1)
        with open(os.path.join(out_dir, "metrics.json"), "w") as jf:
            json.dump(metrics, jf, indent=2)
        if repeats > 1:
            with open(os.path.join(out_dir, "metrics_repeated.json"), "w") as jf:
                json.dump({"repeats": repeats, "runs": all_metrics, "aggregate": metrics}, jf, indent=2)
        # also a compact CSV summary for quick diffing
        try:
            import csv as _csv
            with open(os.path.join(out_dir, "metrics_summary.csv"), "w", newline="") as cf:
                w = _csv.writer(cf)
                w.writerow(["metric", "value"])
                for k, v in metrics.items():
                    w.writerow([k, v])
        except Exception as e:
            print(f"metrics_summary.csv failed: {e}")

    return metrics

def evaluate_model_repeated(model, out_dir=None):
    """Evaluate model using repeated stratified validation splits.

    - Respects cfg.SPLIT_SEED_MODE ("fixed" uses SPLIT_RANDOM_SEED + i per repeat; "random" draws new seeds).
    - Uses cfg.VAL_REPEATS (default 1).
    - Writes standard plots for the first split and saves per-repeat metrics and seeds to disk.
    """
    rows = []
    if os.path.exists(cfg.LABELS_FILE):
        rows.extend(load_labels(cfg.LABELS_FILE))
    if os.path.exists(cfg.TEMP_LABELS_FILE):
        rows.extend(load_labels(cfg.TEMP_LABELS_FILE))
    if not rows:
        print("No labels found for evaluation.")
        return None
    seen = set(); dedup = []
    for r in rows:
        k = f"{r.get('tile')}:{r.get('lat')}:{r.get('lon')}"
        if k in seen: continue
        seen.add(k); dedup.append(r)
    rows = dedup

    X, y = [], []
    for r in rows:
        f = extract_features_from_label(r)
        if f is None: continue
        X.append(f)
        y.append(1 if r.get("label", "").lower()=="agricultural" else 0)
    if not X:
        print("No features extracted for evaluation set.")
        return None
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    n_repeats = int(getattr(cfg, "VAL_REPEATS", 1))
    if n_repeats < 1:
        n_repeats = 1
    seeds = []
    if getattr(cfg, "SPLIT_SEED_MODE", "random") == "fixed":
        # Ensure seeds are always within [0, 2**32 - 1]
        base = int(getattr(cfg, "SPLIT_RANDOM_SEED", 42)) & 0xFFFFFFFF
        seeds = [((base + i) & 0xFFFFFFFF) for i in range(n_repeats)]
    else:
        # Draw cryptographically-strong 32-bit seeds (compatible with sklearn/random_state bounds)
        from secrets import randbits
        for _ in range(n_repeats):
            seeds.append(randbits(32))

    all_metrics = []
    first = None
    for ridx, seed in enumerate(seeds):
        tr_idx, va_idx, _ = stratified_train_val_test_indices(
            y, cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.TEST_FRACTION, random_state=seed)
        if va_idx.size == 0:
            continue
        probs = model.predict_proba(X[va_idx])[:,1]
        preds = (probs >= cfg.MIN_AGRI_PROB).astype(int)
        yv = y[va_idx]
        cm = confusion_matrix(yv, preds, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        m = {
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "precision": precision_score(yv, preds, zero_division=0),
            "recall": recall_score(yv, preds, zero_division=0),
            "f1": f1_score(yv, preds, zero_division=0),
            "macro_f1": f1_score(yv, preds, average='macro', zero_division=0),
            "accuracy": accuracy_score(yv, preds),
            "auc": roc_auc_score(yv, probs) if len(np.unique(yv))>1 else 0.0,
            "auc_pr": average_precision_score(yv, probs),
            "balanced_accuracy": (((tp/(tp+fn)) if (tp+fn)>0 else 0) + ((tn/(tn+fp)) if (tn+fp)>0 else 0))/2,
            "mcc": matthews_corrcoef(yv, preds) if len(np.unique(yv))>1 and len(np.unique(preds))>1 else 0.0,
            "cohen_kappa": cohen_kappa_score(yv, preds),
            "brier_score": brier_score_loss(yv, probs),
            "_seed": seed,
        }
        all_metrics.append(m)
        if first is None:
            first = (va_idx, yv, preds, probs, cm)

    if not all_metrics:
        print("Validation split empty; skipping evaluation.")
        return None

    def _mean_std(key):
        vals = [float(mm[key]) for mm in all_metrics if key in mm]
        if not vals: return 0.0, 0.0
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    numeric = ["precision","recall","f1","macro_f1","accuracy","auc","auc_pr","balanced_accuracy","mcc","cohen_kappa","brier_score"]
    summary = {}
    for k in numeric:
        mu, sd = _mean_std(k)
        summary[k] = mu
        summary[f"std_{k}"] = sd
    # confusion counts from first split
    _, yv0, preds0, probs0, cm0 = first
    tn0, fp0, fn0, tp0 = cm0.ravel()
    summary.update({"TP": int(tp0), "FP": int(fp0), "TN": int(tn0), "FN": int(fn0), "repeats": len(all_metrics)})

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # first split report
        rep = classification_report(yv0, preds0, digits=3)
        with open(os.path.join(out_dir, 'classification_report.txt'), 'w') as rf:
            rf.write(rep)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm0, display_labels=["NonAgri","Agri"])
        fig, ax = plt.subplots(figsize=(4,4))
        disp.plot(ax=ax, colorbar=False)
        plt.tight_layout(); fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
        plt.close(fig)
        roc_disp = RocCurveDisplay.from_predictions(yv0, probs0)
        roc_disp.figure_.savefig(os.path.join(out_dir, 'roc_curve.png'))
        plt.close(roc_disp.figure_)
        pr_disp = PrecisionRecallDisplay.from_predictions(yv0, probs0)
        pr_disp.figure_.savefig(os.path.join(out_dir, 'pr_curve.png'))
        plt.close(pr_disp.figure_)
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as jf:
            json.dump(summary, jf, indent=2)
        try:
            import csv as _csv
            with open(os.path.join(out_dir, 'metrics_summary.csv'), 'w', newline='') as cf:
                w = _csv.writer(cf); w.writerow(['metric','value'])
                for k, v in summary.items():
                    w.writerow([k, v])
        except Exception as e:
            print(f"metrics_summary.csv failed: {e}")
        # write repeats and seeds
        try:
            with open(os.path.join(out_dir, 'metrics_repeats.json'), 'w') as jf:
                json.dump(all_metrics, jf, indent=2)
            with open(os.path.join(out_dir, 'split_seeds.json'), 'w') as sf:
                json.dump({'seeds': seeds}, sf, indent=2)
            # store validation split rows for first split
            try:
                import csv as _csv
                va_idx0 = first[0]
                with open(os.path.join(out_dir, 'validation_split_rows.csv'), 'w', newline='') as vf:
                    fieldnames = ['tile','lat','lon','label','id']
                    w = _csv.DictWriter(vf, fieldnames=fieldnames)
                    w.writeheader()
                    for i in va_idx0.tolist():
                        r = rows[i]
                        w.writerow({'tile': r.get('tile',''), 'lat': r.get('lat',''), 'lon': r.get('lon',''), 'label': r.get('label',''), 'id': r.get('id','')})
            except Exception as e:
                print(f"validation_split_rows.csv failed: {e}")
        except Exception as e:
            print(f"repeat metrics write failed: {e}")

    return summary
