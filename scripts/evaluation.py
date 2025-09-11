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
from sklearn.calibration import calibration_curve
os.environ.setdefault("MPLBACKEND", "Agg")
# Threading caps to avoid OpenBLAS/OpenMP warnings and oversubscription
try:
    import config as _cfg_threads
    _thr = str(int(getattr(_cfg_threads, 'BLAS_NUM_THREADS', max(1, min(4, (cpu_count() or 1))))))
    for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(_k, str(getattr(_cfg_threads, _k, _thr)))
except Exception:
    _DEFAULT_THREADS = str(max(1, min(4, (cpu_count() or 1))))
    os.environ.setdefault("OMP_NUM_THREADS", _DEFAULT_THREADS)
    os.environ.setdefault("MKL_NUM_THREADS", _DEFAULT_THREADS)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", _DEFAULT_THREADS)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", _DEFAULT_THREADS)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from al_shared import extract_features_from_label
import config as cfg
from splits import load_labels, stratified_train_val_test_indices


def _plot_threshold_sweep(y_true, probs, out_path):
    import numpy as _np
    import matplotlib.pyplot as _plt
    ths = _np.linspace(0.0, 1.0, 101)
    y_true = _np.asarray(y_true)
    probs = _np.asarray(probs)
    P = []
    R = []
    F1 = []
    ACC = []
    for t in ths:
        pred = (probs >= t).astype(int)
        P.append(precision_score(y_true, pred, zero_division=0))
        R.append(recall_score(y_true, pred, zero_division=0))
        F1.append(f1_score(y_true, pred, zero_division=0))
        ACC.append(accuracy_score(y_true, pred))
    _plt.figure(figsize=(7, 4))
    _plt.plot(ths, P, label="Precision")
    _plt.plot(ths, R, label="Recall")
    _plt.plot(ths, F1, label="F1")
    _plt.plot(ths, ACC, label="Accuracy")
    # mark configured threshold
    try:
        import config as _cfg
        t0 = float(getattr(_cfg, 'MIN_AGRI_PROB', 0.5))
        _plt.axvline(t0, color='k', linestyle='--', alpha=0.5, label=f"th={t0:.2f}")
    except Exception:
        pass
    _plt.xlabel("Threshold")
    _plt.ylabel("Metric value")
    _plt.ylim(0.0, 1.0)
    _plt.grid(True, alpha=0.2)
    _plt.legend(loc='best')
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=180)
    _plt.close()


def _plot_calibration_with_hist(y_true, probs, out_path, n_bins=10):
    import numpy as _np
    import matplotlib.pyplot as _plt
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy='uniform')
    # Create a two-row figure: reliability (top) + histogram (bottom)
    fig = _plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    ax1.plot(mean_pred, frac_pos, marker='o', linestyle='-', label='Model')
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Empirical positive rate")
    ax1.grid(True, alpha=0.2)
    # Histogram (probability distribution)
    ax2.hist(probs, bins=n_bins, range=(0.0, 1.0), color='#6666cc', alpha=0.8)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.grid(True, axis='y', alpha=0.2)
    # Title with Brier score
    try:
        bs = brier_score_loss(y_true, probs)
        fig.suptitle(f"Calibration (Brier={bs:.3f})", y=0.98)
    except Exception:
        pass
    _plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)

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
        try:
            feats = extract_features_from_label(r)
        except Exception:
            # Skip labels whose tiles cannot be read
            continue
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

            # Precision-Recall curve (annotate AP)
            ap = average_precision_score(yv, probs)
            from sklearn.metrics import precision_recall_curve
            prec, rec, _ = precision_recall_curve(yv, probs)
            fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
            ax_pr.plot(rec, prec, label=f"PR curve (AP={ap:.3f})")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_xlim(0.0, 1.0)
            ax_pr.set_ylim(0.0, 1.0)
            ax_pr.grid(True, alpha=0.2)
            ax_pr.legend(loc='best')
            plt.tight_layout()
            fig_pr.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=180)
            plt.close(fig_pr)

            # Threshold sweep curves
            try:
                _plot_threshold_sweep(yv, probs, os.path.join(out_dir, "threshold_sweep.png"))
            except Exception as e:
                print(f"Threshold sweep plot failed: {e}")

            # Calibration + histogram
            try:
                _plot_calibration_with_hist(yv, probs, os.path.join(out_dir, "calibration.png"), n_bins=10)
            except Exception as e:
                print(f"Calibration plot failed: {e}")

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
