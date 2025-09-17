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
    roc_curve,
    precision_recall_curve,
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

def plot_feature_family_importance(importances, feature_names, out_path, normalize=True):
    """Plot contributions grouped by feature family.

    Families: 'red-edge' (B5,B6,B7,B8A), 'spectral' (other B* bands),
              'indices' (NDVI/EVI2/GNDVI/NDWI/NDRE/etc.),
              'terrain' (ELEVATION/SLOPE/ASPECT), 'textures' (NDVI_*_local*).

    importances: 1D array-like of permutation importances (mean).
    feature_names: list of names aligned with importances.
    out_path: output PNG path.
    normalize: when True, scale bars to sum to 1 (percent).
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    imps = _np.asarray(importances, dtype=float)
    names = list(feature_names or [])
    if imps.ndim != 1 or len(names) != imps.size:
        return

    families = ["red-edge", "spectral", "indices", "terrain", "textures"]
    contrib = {k: 0.0 for k in families}

    idx_set = set(str(x).upper() for x in getattr(cfg, "INDICES", []))
    terrain_set = {"ELEVATION", "SLOPE", "ASPECT"}
    red_edge_set = {"B5", "B6", "B7", "B8A"}

    def _family_for(name: str) -> str:
        n = str(name)
        if ("localmean" in n.lower()) or ("localstd" in n.lower()):
            return "textures"
        base = n.split("_s")[0].upper()
        if base in terrain_set:
            return "terrain"
        if base in idx_set:
            return "indices"
        if base in red_edge_set:
            return "red-edge"
        if base.startswith("B"):
            return "spectral"
        # Fallbacks: treat NDVI_MEAN_* textures as textures; else indices
        if n.upper().startswith("NDVI_MEAN_"):
            return "textures"
        return "indices"

    for imp, nm in zip(imps, names):
        fam = _family_for(nm)
        val = float(max(0.0, imp))
        contrib[fam] = contrib.get(fam, 0.0) + val

    vals = [contrib[k] for k in families]
    total = float(sum(vals))
    if normalize and total > 0:
        vals = [v / total for v in vals]

    _plt.figure(figsize=(6.5, 3.6))
    bars = _plt.bar(range(len(families)), vals, color=["#8c564b", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"])  # fixed palette
    _plt.xticks(range(len(families)), families, rotation=0)
    _plt.ylabel("Contribution" + (" (fraction)" if normalize else ""))
    _plt.title("Feature family contributions")
    _plt.ylim(0.0, max(1.0 if normalize else max(vals + [0.0]) * 1.10, max(vals + [0.0]) * 1.05 if not normalize else 1.0))
    if total > 0:
        for rect, v in zip(bars, vals):
            lab = f"{v*100:.1f}%" if normalize else f"{v:.3f}"
            _plt.text(rect.get_x() + rect.get_width()/2.0, rect.get_height() + (0.01 if normalize else (0.01*max(vals+[1]))), lab,
                      ha='center', va='bottom', fontsize=8)
    _plt.tight_layout()
    _plt.savefig(out_path, dpi=180)
    _plt.close()

def _metrics_at_threshold(y_true, probs, th):
    import numpy as _np
    from sklearn.metrics import confusion_matrix
    y_true = _np.asarray(y_true)
    probs = _np.asarray(probs)
    y_pred = (probs >= float(th)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tpr
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "tpr": float(tpr), "fpr": float(fpr), "precision": float(prec), "recall": float(rec)}

def _plot_confusion_normalised(y_true, y_pred, out_path, labels=("Negative","Positive")):
    import numpy as _np
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with _np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cmn = _np.divide(cm, row_sums, where=row_sums != 0)
        cmn = _np.nan_to_num(cmn)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cmn, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_title('Confusion Matrix (normalised by true class)')
    tick_marks = _np.arange(len(labels))
    ax.set_xticks(tick_marks, labels)
    ax.set_yticks(tick_marks, labels)
    ax.set_ylabel('True class')
    ax.set_xlabel('Predicted class')
    fmt = '.2f'
    thresh = cmn.max() / 2.
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, format(cmn[i, j], fmt), ha="center", va="center",
                    color="white" if cmn[i, j] > thresh else "black", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Proportion within true class', rotation=90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def _plot_prob_hist_by_class(y_true, probs, out_path):
    import numpy as _np
    y_true = _np.asarray(y_true)
    probs = _np.asarray(probs)
    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = _np.linspace(0.0, 1.0, 21)
    ax.hist(neg, bins=bins, density=True, alpha=0.5, label='NonAgri', color="#1f77b4")
    ax.hist(pos, bins=bins, density=True, alpha=0.5, label='Agri', color="#d62728")
    ax.set_xlabel('Predicted probability (Agri)')
    ax.set_ylabel('Density')
    ax.set_title('Probability histograms by class')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def _plot_pr_roc_combined(y_true, probs, out_path, th_selected=None, th_best=None):
    import numpy as _np
    from matplotlib.lines import Line2D
    y_true = _np.asarray(y_true)
    probs = _np.asarray(probs)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=False)
    fpr, tpr, _ = roc_curve(y_true, probs)
    ax_roc.plot(fpr, tpr, color="#1f77b4", label="ROC curve")
    ax_roc.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC")
    ax_roc.grid(True, alpha=0.2)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ax_pr.plot(rec, prec, color="#2ca02c", label="PR curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.grid(True, alpha=0.2)
    handles = [Line2D([0], [0], color="#1f77b4"), Line2D([0], [0], color="#2ca02c")]
    labels = ["ROC curve", "PR curve"]
    def _mark(th, color, label):
        m = _metrics_at_threshold(y_true, probs, th)
        ax_roc.scatter([m["fpr"]], [m["tpr"]], color=color, s=40, edgecolor="black", zorder=5)
        ax_roc.annotate(f"th={th:.2f}", (m["fpr"], m["tpr"]), textcoords="offset points", xytext=(5, -12), fontsize=8)
        ax_pr.scatter([m["recall"]], [m["precision"]], color=color, s=40, edgecolor="black", zorder=5)
        ax_pr.annotate(f"th={th:.2f}", (m["recall"], m["precision"]), textcoords="offset points", xytext=(5, -12), fontsize=8)
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markeredgecolor='black'))
        labels.append(label)
    if th_selected is not None:
        _mark(float(th_selected), color="#d62728", label="Selected threshold")
    if th_best is not None:
        _mark(float(th_best), color="#9467bd", label="Best threshold")
    fig.subplots_adjust(right=0.80)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.82, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

def compute_best_threshold_weighted(y_true, probs, weights=None):
    """Return best threshold based on a weighted score of metrics.

    weights: dict with keys 'precision','recall','f1','accuracy'. Defaults to
             precision=0.50, recall=0.30, f1=0.15, accuracy=0.05.
    """
    import numpy as _np
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    if weights is None:
        weights = {"precision": 0.50, "recall": 0.30, "f1": 0.15, "accuracy": 0.05}
    wP = float(weights.get("precision", 0.5))
    wR = float(weights.get("recall", 0.3))
    wF = float(weights.get("f1", 0.15))
    wA = float(weights.get("accuracy", 0.05))
    ths = _np.linspace(0.0, 1.0, 101)
    best = {"threshold": 0.0, "score": -1.0}
    best_metrics = None
    y_true = _np.asarray(y_true)
    probs = _np.asarray(probs)
    for t in ths:
        pred = (probs >= t).astype(int)
        P = precision_score(y_true, pred, zero_division=0)
        R = recall_score(y_true, pred, zero_division=0)
        F1 = f1_score(y_true, pred, zero_division=0)
        ACC = accuracy_score(y_true, pred)
        score = wP * P + wR * R + wF * F1 + wA * ACC
        # tie-breakers: higher recall, then lower threshold
        if (score > best["score"]) or (
            _np.isclose(score, best["score"]) and (
                (best_metrics and R > best_metrics.get("recall", 0.0)) or
                (best_metrics and _np.isclose(R, best_metrics.get("recall", 0.0)) and t < best["threshold"]) or
                (best_metrics is None)
            )
        ):
            best = {"threshold": float(t), "score": float(score)}
            best_metrics = {"precision": float(P), "recall": float(R), "f1": float(F1), "accuracy": float(ACC)}
    return best, best_metrics


def _plot_threshold_sweep(y_true, probs, out_path, best_th: float | None = None):
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
        _plt.axvline(t0, color='k', linestyle='--', alpha=0.7, label=f"selected th={t0:.2f}")
    except Exception:
        pass
    if best_th is not None:
        _plt.axvline(float(best_th), color='#9467bd', linestyle=':', linewidth=2.0, alpha=0.9, label=f"best th={float(best_th):.2f}")
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

            # Normalised confusion matrix with colorbar legend
            _plot_confusion_normalised(yv, preds, os.path.join(out_dir, "confusion_matrix.png"), labels=("NonAgri","Agri"))

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

            # Compute best threshold for annotation and build sweep
            try:
                best, _best_m = compute_best_threshold_weighted(yv, probs)
                th_best = float(best.get("threshold", float(getattr(cfg, 'MIN_AGRI_PROB', 0.5))))
            except Exception:
                th_best = None
            # Threshold sweep curves
            try:
                _plot_threshold_sweep(yv, probs, os.path.join(out_dir, "threshold_sweep.png"), best_th=th_best)
            except Exception as e:
                print(f"Threshold sweep plot failed: {e}")

            # Calibration + histogram
            try:
                _plot_calibration_with_hist(yv, probs, os.path.join(out_dir, "calibration.png"), n_bins=10)
            except Exception as e:
                print(f"Calibration plot failed: {e}")

            # Combined PR+ROC and probability histograms by class
            try:
                _plot_pr_roc_combined(yv, probs, os.path.join(out_dir, "pr_roc_combined.png"),
                                      th_selected=float(getattr(cfg, 'MIN_AGRI_PROB', 0.5)),
                                      th_best=th_best)
            except Exception as e:
                print(f"Combined PR/ROC plot failed: {e}")
            try:
                _plot_prob_hist_by_class(yv, probs, os.path.join(out_dir, "prob_hist_by_class.png"))
            except Exception as e:
                print(f"Probability histogram plot failed: {e}")

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
        _plot_confusion_normalised(yv0, preds0, os.path.join(out_dir, 'confusion_matrix.png'), labels=("NonAgri","Agri"))
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
