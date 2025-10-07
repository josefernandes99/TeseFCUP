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
from sklearn.cluster import DBSCAN
from features import current_feature_names

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
    import math
    import numpy as _np
    from sklearn.metrics import confusion_matrix

    y_true = _np.asarray(y_true)
    probs = _np.asarray(probs)
    y_pred = (probs >= float(th)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = 1.0 - specificity
    fnr = 1.0 - recall

    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    # Negative-class F1 for macro averaging
    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_neg = specificity
    f1_neg = 0.0
    if (precision_neg + recall_neg) > 0:
        f1_neg = 2.0 * precision_neg * recall_neg / (precision_neg + recall_neg)
    macro_f1 = (f1 + f1_neg) / 2.0

    accuracy = (tp + tn) / total if total else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    # Matthews correlation coefficient
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0

    # Cohen's kappa
    if total:
        po = accuracy
        pe = 0.0
        pe += ((tp + fp) / total) * ((tp + fn) / total)
        pe += ((fn + tn) / total) * ((fp + tn) / total)
        kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) else 0.0
    else:
        kappa = 0.0

    brier = (fp + fn) / total if total else 0.0

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa),
        "brier_score": float(brier),
        "tpr": float(recall),
    }

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
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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



def _plot_false_positive_hotspots(lats, lons, y_true, y_pred, out_path):
    import numpy as _np
    import matplotlib.pyplot as _plt

    lats = _np.asarray(lats, dtype=float)
    lons = _np.asarray(lons, dtype=float)
    mask_valid = _np.isfinite(lats) & _np.isfinite(lons)
    if not mask_valid.any():
        return
    lats = lats[mask_valid]
    lons = lons[mask_valid]
    y_true = _np.asarray(y_true)[mask_valid]
    y_pred = _np.asarray(y_pred)[mask_valid]

    mask_fp = (y_pred == 1) & (y_true == 0)
    fig, ax = _plt.subplots(figsize=(6, 5))
    ax.scatter(lons, lats, s=8, color='#d3d3d3', alpha=0.4, label='All samples')
    if mask_fp.any():
        fp_lons = lons[mask_fp]
        fp_lats = lats[mask_fp]
        coords_rad = _np.deg2rad(_np.column_stack((fp_lats, fp_lons)))
        try:
            clustering = DBSCAN(eps=1.0 / 6371.0088, min_samples=3, metric='haversine').fit(coords_rad)
            labels = clustering.labels_
        except Exception:
            labels = _np.full(fp_lats.shape, -1)
        unique = _np.unique(labels)
        colors = _plt.cm.get_cmap('tab10', len(unique) or 1)
        for idx, lab in enumerate(unique):
            sel = labels == lab
            if lab == -1:
                ax.scatter(fp_lons[sel], fp_lats[sel], s=25, facecolors='none', edgecolors='red', linewidths=1.2, label='FP (noise)' if idx == 0 else None)
            else:
                ax.scatter(fp_lons[sel], fp_lats[sel], s=35, color=colors(idx), alpha=0.85, label=f'FP cluster {lab+1}')
    else:
        ax.text(0.5, 0.5, 'No false positives', transform=ax.transAxes, ha='center', va='center', fontsize=12)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('False-positive hotspots (validation set)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)


def _plot_per_island_bars(islands, y_true, y_pred, out_path):
    import numpy as _np
    import matplotlib.pyplot as _plt

    islands = _np.asarray(islands)
    y_true = _np.asarray(y_true)
    y_pred = _np.asarray(y_pred)
    unique = [isl for isl in _np.unique(islands) if isl]
    if not unique:
        islands = _np.where((islands == '') | (islands == None), 'unknown', islands)
        unique = ['unknown']
    metrics = []
    for isl in unique:
        sel = islands == isl
        if sel.sum() == 0:
            continue
        yt = y_true[sel]
        yp = y_pred[sel]
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics.append((isl, precision, recall))
    if not metrics:
        return
    islands_order, precisions, recalls = zip(*metrics)
    import numpy as _np
    idx = _np.arange(len(islands_order))
    width = 0.35
    fig, ax = _plt.subplots(figsize=(max(6, len(islands_order) * 1.4), 5))
    ax.bar(idx - width/2, precisions, width, label='Precision', color='#1f77b4')
    ax.bar(idx + width/2, recalls, width, label='Recall', color='#ff7f0e')
    ax.set_xticks(idx)
    ax.set_xticklabels(islands_order, rotation=30, ha='right')
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title('Per-island precision and recall (validation set)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)


def _plot_feature_response_scatter(X_val, feature_names, probs, y_true, out_path):
    import numpy as _np
    import matplotlib.pyplot as _plt

    if X_val is None or feature_names is None:
        return
    feature_names = list(feature_names)
    lower_names = [name.lower() for name in feature_names]
    wanted = ['ndvi', 'evi', 'evi2', 'ndre', 'gndvi', 'b5', 'b6', 'b7', 'ndwi']
    picked = []
    for candidate in wanted:
        if candidate in lower_names:
            idx = lower_names.index(candidate)
            if idx not in picked:
                picked.append(idx)
        if len(picked) >= 4:
            break
    if not picked:
        return
    cols = len(picked)
    fig, axes = _plt.subplots(1, cols, figsize=(4*cols, 4), sharey=True)
    if cols == 1:
        axes = [axes]
    X_val = _np.asarray(X_val, dtype=float)
    probs = _np.asarray(probs, dtype=float)
    y_true = _np.asarray(y_true)
    for ax, idx in zip(axes, picked):
        feat = X_val[:, idx]
        ax.scatter(feat[y_true == 0], probs[y_true == 0], color='#1f77b4', alpha=0.5, s=16, label='Non-agri')
        ax.scatter(feat[y_true == 1], probs[y_true == 1], color='#ff7f0e', alpha=0.6, s=18, label='Agri')
        ax.set_xlabel(feature_names[idx])
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel('Predicted probability (Agri)')
    axes[0].legend(loc='lower right')
    fig.suptitle('Probability vs feature scatter (validation set)')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)


def _plot_round_metric_trends(out_path, island_label=None):
    import csv as _csv
    import numpy as _np
    import matplotlib.pyplot as _plt

    path = os.path.join(cfg.ROUNDS_DIR, 'rounds_metrics.csv')
    if not os.path.exists(path):
        return
    rounds, precision, recall, f1, auc_pr = [], [], [], [], []
    with open(path, newline='') as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                rounds.append(int(row.get('round', len(rounds) + 1)))
                precision.append(float(row.get('precision', 0)))
                recall.append(float(row.get('recall', 0)))
                f1.append(float(row.get('f1', 0)))
                auc_pr.append(float(row.get('auc_pr', 0)))
            except Exception:
                continue
    if not rounds:
        return
    fig, ax = _plt.subplots(figsize=(6, 4))
    ax.plot(rounds, precision, marker='o', label='Precision')
    ax.plot(rounds, recall, marker='o', label='Recall')
    ax.plot(rounds, f1, marker='o', label='F1')
    ax.plot(rounds, auc_pr, marker='o', label='AUC-PR')
    ax.set_xlabel('Round')
    ax.set_ylabel('Score')
    title = 'Round metrics trend'
    if island_label:
        title += f' ({island_label})'
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)


def _plot_confusion_map(lats, lons, y_true, y_pred, probs, out_path):
    import numpy as _np
    import matplotlib.pyplot as _plt

    lats = _np.asarray(lats, dtype=float)
    lons = _np.asarray(lons, dtype=float)
    probs = _np.asarray(probs, dtype=float)
    mask = _np.isfinite(lats) & _np.isfinite(lons)
    if not mask.any():
        return
    lats = lats[mask]
    lons = lons[mask]
    y_true = _np.asarray(y_true)[mask]
    y_pred = _np.asarray(y_pred)[mask]
    probs = probs[mask]
    outcome = _np.empty_like(y_true, dtype='<U3')
    outcome[(y_true == 1) & (y_pred == 1)] = 'TP'
    outcome[(y_true == 0) & (y_pred == 0)] = 'TN'
    outcome[(y_true == 0) & (y_pred == 1)] = 'FP'
    outcome[(y_true == 1) & (y_pred == 0)] = 'FN'
    colors = {'TP': '#2ca02c', 'TN': '#1f77b4', 'FP': '#d62728', 'FN': '#ff7f0e'}
    fig, ax = _plt.subplots(figsize=(6, 5))
    for label in ['TP', 'FP', 'FN', 'TN']:
        sel = outcome == label
        if sel.any():
            ax.scatter(lons[sel], lats[sel], s=20, color=colors[label], alpha=0.7, label=label)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial confusion map (validation set)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)


def _plot_runtime_summary(csv_path, out_path):
    import csv as _csv
    import numpy as _np
    import matplotlib.pyplot as _plt

    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline='') as f:
        reader = _csv.DictReader(f)
        rows = [row for row in reader]
    if not rows:
        return
    try:
        rows.sort(key=lambda r: int(r.get('round', 0)))
    except Exception:
        pass
    def _to_float(row, key):
        try:
            return float(row.get(key, 0.0))
        except Exception:
            return 0.0
    rounds = [int(row.get('round', idx + 1)) for idx, row in enumerate(rows)]
    training = [_to_float(r, 'training_seconds') for r in rows]
    inference = [_to_float(r, 'inference_seconds') for r in rows]
    evaluation = [_to_float(r, 'evaluation_seconds') for r in rows]
    candidate = [_to_float(r, 'candidate_seconds') for r in rows]
    total = [_to_float(r, 'total_seconds') for r in rows]

    fig, ax = _plt.subplots(figsize=(max(6, len(rounds) * 0.9), 4))
    ax.plot(rounds, training, marker='o', label='Training')
    ax.plot(rounds, inference, marker='o', label='Inference')
    ax.plot(rounds, evaluation, marker='o', label='Evaluation')
    if any(candidate):
        ax.plot(rounds, candidate, marker='o', label='Candidate')
    ax.plot(rounds, total, marker='o', linestyle='--', label='Total')
    ax.set_xlabel('Round')
    ax.set_ylabel('Seconds')
    ax.set_title('Round runtime summary')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)


def _plot_runtime_breakdown(runtime_data, out_path):
    import matplotlib.pyplot as _plt
    stages = [
        ('Training', float(runtime_data.get('training_seconds', 0.0))),
        ('Inference', float(runtime_data.get('inference_seconds', 0.0))),
        ('Evaluation', float(runtime_data.get('evaluation_seconds', 0.0))),
        ('Candidate', float(runtime_data.get('candidate_seconds', 0.0))),
        ('Other', float(runtime_data.get('other_seconds', 0.0))),
    ]
    fig, ax = _plt.subplots(figsize=(6, 3.5))
    labels = [s for s, _ in stages]
    values = [v for _, v in stages]
    ax.barh(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#7f7f7f'])
    ax.set_xlabel('Seconds')
    ax.set_title('Runtime breakdown (per round)')
    ax.grid(True, axis='x', alpha=0.2)
    for i, v in enumerate(values):
        ax.text(v + max(values) * 0.01 if values else 0.01, i, f"{v:.1f}", va='center')
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)


def _plot_pr_roc_combined(y_true, probs, out_path, th_selected=None, th_best=None):
    import numpy as _np
    from matplotlib.lines import Line2D
    y_true = _np.asarray(y_true)
    probs = _np.asarray(probs)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=False)
    fpr, tpr, _ = roc_curve(y_true, probs)
    ax_roc.plot(fpr, tpr, color="#1f77b4", label="ROC curve")
    ax_roc.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC")
    ax_roc.grid(True, alpha=0.2)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    ax_pr.plot(rec, prec, color="#2ca02c", label="PR curve")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall")
    ax_pr.grid(True, alpha=0.2)
    handles = [Line2D([0], [0], color="#1f77b4"), Line2D([0], [0], color="#2ca02c")]
    labels = ["ROC curve", "PR curve"]
    if th_selected is not None and th_best is not None:
        try:
            if abs(float(th_selected) - float(th_best)) < 1e-6:
                th_best = None
        except Exception:
            pass
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

def compute_best_threshold_weighted(y_true, probs, weights=None, thresholds=None, return_sweep=False):
    """Return best threshold based on a weighted score aligned with crop-only focus.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels (0 non-agri, 1 agri).
    probs : array-like
        Predicted probabilities for the positive (agri) class.
    weights : dict, optional
        Custom metric weights overriding the default objective.
    thresholds : sequence, optional
        Explicit thresholds to evaluate; defaults to np.linspace(0,1,101).
    return_sweep : bool, default False
        When True, also return per-threshold metric traces for plotting.
    """
    import numpy as _np
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = _np.asarray(y_true)
    probs = _np.asarray(probs)
    if thresholds is None:
        thresholds = _np.linspace(0.0, 1.0, 101)
    else:
        thresholds = _np.asarray(thresholds, dtype=float)

    # Default weights ordered by importance (precision heavy, penalise false positives)
    default_weights = {
        "precision": 0.28,
        "recall": 0.16,
        "f1": 0.15,
        "macro_f1": 0.06,
        "balanced_accuracy": 0.08,
        "mcc": 0.10,
        "cohen_kappa": 0.05,
        "accuracy": 0.03,
        "one_minus_brier": 0.05,
        "local_pr_area": 0.12,
        "pr_auc": 0.07,
        "fpr_penalty": 0.10,
    }
    if weights:
        default_weights.update(weights)

    ap = float(average_precision_score(y_true, probs)) if y_true.size else 0.0
    if y_true.size and len(_np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, probs))
    else:
        roc_auc = 0.0
    brier_prob = float(_np.mean((probs - y_true) ** 2)) if y_true.size else 0.0

    metrics_grid = {}
    score_series = []
    local_pr_series = []
    best = {"threshold": float(thresholds[0]) if thresholds.size else 0.0, "score": -_np.inf}
    best_metrics = None

    for t in thresholds:
        stats = _metrics_at_threshold(y_true, probs, t)
        if not metrics_grid:
            metrics_grid = {key: [] for key in stats.keys()}
        for key, value in stats.items():
            metrics_grid[key].append(value)

        precision = stats.get("precision", 0.0)
        recall = stats.get("recall", 0.0)
        f1 = stats.get("f1", 0.0)
        macro_f1 = stats.get("macro_f1", 0.0)
        balanced_accuracy = stats.get("balanced_accuracy", 0.0)
        mcc = stats.get("mcc", 0.0)
        kappa = stats.get("cohen_kappa", 0.0)
        accuracy = stats.get("accuracy", 0.0)
        brier = stats.get("brier_score", 0.0)
        fpr = stats.get("fpr", 0.0)

        local_pr_area = precision * recall  # emphasises joint high precision and recall
        one_minus_brier = 1.0 - brier

        score = (
            default_weights["precision"] * precision
            + default_weights["recall"] * recall
            + default_weights["f1"] * f1
            + default_weights["macro_f1"] * macro_f1
            + default_weights["balanced_accuracy"] * balanced_accuracy
            + default_weights["mcc"] * mcc
            + default_weights["cohen_kappa"] * kappa
            + default_weights["accuracy"] * accuracy
            + default_weights["one_minus_brier"] * one_minus_brier
            + default_weights["local_pr_area"] * local_pr_area
            + default_weights["pr_auc"] * ap
            - default_weights["fpr_penalty"] * fpr
        )

        score_series.append(float(score))
        local_pr_series.append(float(local_pr_area))

        # Tie-breakers: higher precision, lower FPR, higher recall, higher threshold (to prefer conservative)
        better = False
        if score > best["score"]:
            better = True
        elif _np.isclose(score, best["score"], atol=1e-6):
            prev = best_metrics or {}
            prec_prev = prev.get("precision", 0.0)
            fpr_prev = prev.get("fpr", 1.0)
            rec_prev = prev.get("recall", 0.0)
            thr_prev = best.get("threshold", 0.0)
            if precision > prec_prev + 1e-6:
                better = True
            elif _np.isclose(precision, prec_prev, atol=1e-6) and fpr < fpr_prev - 1e-6:
                better = True
            elif (
                _np.isclose(precision, prec_prev, atol=1e-6)
                and _np.isclose(fpr, fpr_prev, atol=1e-6)
                and recall > rec_prev + 1e-6
            ):
                better = True
            elif (
                _np.isclose(precision, prec_prev, atol=1e-6)
                and _np.isclose(fpr, fpr_prev, atol=1e-6)
                and _np.isclose(recall, rec_prev, atol=1e-6)
                and t > thr_prev
            ):
                better = True

        if better:
            best = {"threshold": float(t), "score": float(score)}
            best_metrics = {
                **stats,
                "pr_auc": ap,
                "roc_auc": roc_auc,
                "brier_prob": brier_prob,
                "local_pr_area": float(local_pr_area),
                "one_minus_brier": float(one_minus_brier),
                "score": float(score),
            }

    if not best_metrics:
        best_metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "pr_auc": ap,
            "roc_auc": roc_auc,
            "brier_prob": brier_prob,
        }

    if return_sweep:
        sweep_metrics = {k: list(v) for k, v in metrics_grid.items()}
        sweep_metrics["pr_auc"] = [ap] * len(thresholds)
        sweep_metrics["brier_prob"] = [brier_prob] * len(thresholds)
        sweep_metrics["local_pr_area"] = local_pr_series
        sweep_metrics["score"] = score_series
        sweep_metrics["roc_auc"] = [roc_auc] * len(thresholds)
        sweep = {"thresholds": thresholds.tolist(), "metrics": sweep_metrics}
        return best, best_metrics, sweep
    return best, best_metrics


def _plot_threshold_sweep(y_true, probs, out_path, selected_th: float | None = None, sweep=None):
    import numpy as _np
    import matplotlib.pyplot as _plt
    from matplotlib.lines import Line2D

    if sweep is None:
        _, _, sweep = compute_best_threshold_weighted(y_true, probs, return_sweep=True)

    thresholds = _np.asarray(sweep.get("thresholds", []), dtype=float)
    metrics_map = sweep.get("metrics", {})

    plot_specs = [
        ("precision", "Precision", None, True),
        ("recall", "Recall", None, True),
        ("f1", "F1", None, True),
        ("macro_f1", "Macro F1", None, True),
        ("accuracy", "Accuracy", None, True),
        ("balanced_accuracy", "Balanced Accuracy", None, True),
        ("mcc", "MCC", None, True),
        ("cohen_kappa", "Cohen's κ", None, True),
        ("specificity", "Specificity", None, True),
        ("fpr", "False Positive Rate", None, False),
        ("local_pr_area", "Precision×Recall", None, True),
        ("brier_score", "1 − Brier Score", lambda arr: 1.0 - _np.asarray(arr), True),
    ]

    rows, cols = 3, 4
    fig_width = cols * 3.2 + 2.0
    fig, axes = _plt.subplots(rows, cols, figsize=(fig_width, rows * 2.8), sharex=False, sharey=False)
    axes = axes.flatten()

    if selected_th is None:
        try:
            import config as _cfg
            selected_th = float(getattr(_cfg, "MIN_AGRI_PROB", 0.5))
        except Exception:
            selected_th = None

    for ax, (metric_key, label, transform, higher_is_better) in zip(axes, plot_specs):
        series = _np.asarray(metrics_map.get(metric_key, []), dtype=float)
        if transform is not None and series.size:
            series = transform(series)
        if series.size:
            ax.plot(thresholds, series, label=label, color="#1f77b4")
        else:
            ax.plot([], [])
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0.0, 1.0)
        if selected_th is not None:
            ax.axvline(selected_th, color='k', linestyle='--', linewidth=1.5, alpha=0.9)
        if not higher_is_better and series.size:
            ymax = max(series.max() * 1.05, 0.1)
            ax.set_ylim(0.0, min(1.0, ymax))
        ax.set_xlim(0.0, 1.0)
        ax.set_xticks(_np.linspace(0.0, 1.0, 6))
        ax.tick_params(axis='x', labelrotation=0)
    # ensure x tick labels visible on all subplots
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_visible(True)

    for ax in axes[len(plot_specs):]:
        ax.set_visible(False)

    pr_auc = metrics_map.get("pr_auc", [None])[:1][0]
    roc_auc = metrics_map.get("roc_auc", [None])[:1][0]
    brier_prob = metrics_map.get("brier_prob", [None])[:1][0]
    fig.suptitle(
        "PR-AUC={:.3f} | ROC-AUC={:.3f} | Prob. Brier={:.3f}".format(
            pr_auc if pr_auc is not None else float("nan"),
            roc_auc if roc_auc is not None else float("nan"),
            brier_prob if brier_prob is not None else float("nan"),
        ),
        fontsize=12,
    )

    legend_handles = [
        Line2D([0], [0], color="#1f77b4", linewidth=2, label="Metric trace"),
        Line2D([0], [0], color='k', linewidth=1.5, linestyle='--', label="Selected threshold"),
        Line2D([], [], color='none', label="Axes: X=threshold; Y=metric"),
    ]

    fig.tight_layout(rect=(0.02, 0.04, 0.80, 0.95))
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc='center left',
        bbox_to_anchor=(0.83, 0.5),
        frameon=False,
    )
    fig.savefig(out_path, dpi=180)
    _plt.close(fig)


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
    ax1.set_ylabel("Expected positive rate")
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
    stats = _metrics_at_threshold(yv, probs, cfg.MIN_AGRI_PROB)
    cm = np.array([[stats.get("tn", 0), stats.get("fp", 0)], [stats.get("fn", 0), stats.get("tp", 0)]])
    tn, fp, fn, tp = cm.ravel()
    auc_val = roc_auc_score(yv, probs) if len(np.unique(yv)) > 1 else 0.0
    auc_pr_val = average_precision_score(yv, probs)
    brier_prob = brier_score_loss(yv, probs)
    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": stats.get("precision", 0.0),
        "recall": stats.get("recall", 0.0),
        "f1": stats.get("f1", 0.0),
        "macro_f1": stats.get("macro_f1", 0.0),
        "accuracy": stats.get("accuracy", 0.0),
        "balanced_accuracy": stats.get("balanced_accuracy", 0.0),
        "specificity": stats.get("specificity", 0.0),
        "fpr": stats.get("fpr", 0.0),
        "fnr": stats.get("fnr", 0.0),
        "mcc": stats.get("mcc", 0.0),
        "cohen_kappa": stats.get("cohen_kappa", 0.0),
        "brier_score": stats.get("brier_score", 0.0),
        "auc": auc_val,
        "auc_pr": auc_pr_val,
        "pr_auc": auc_pr_val,
        "roc_auc": auc_val,
        "brier_prob": brier_prob,
        "local_pr_area": stats.get("precision", 0.0) * stats.get("recall", 0.0),
        "one_minus_brier": 1.0 - stats.get("brier_score", 0.0),
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
    latitudes, longitudes, tiles = [], [], []
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
        try:
            latitudes.append(float(r.get('lat')))
        except Exception:
            latitudes.append(float('nan'))
        try:
            longitudes.append(float(r.get('lon')))
        except Exception:
            longitudes.append(float('nan'))
        tiles.append(r.get('tile', ''))
    if not X:
        print("No features extracted for evaluation set.")
        return None
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    latitudes = np.array(latitudes, dtype=float)
    longitudes = np.array(longitudes, dtype=float)
    islands_all = np.array([cfg.extract_island_name(t) or '' for t in tiles])
    feature_names_all = None
    try:
        feature_names_all = current_feature_names()
    except Exception:
        feature_names_all = None
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
            lat_val = latitudes[va_idx]
            lon_val = longitudes[va_idx]
            islands_val = islands_all[va_idx]
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
            sweep = None
            try:
                best, _best_m, sweep = compute_best_threshold_weighted(yv, probs, return_sweep=True)
                th_best = float(best.get("threshold", float(getattr(cfg, 'MIN_AGRI_PROB', 0.5))))
            except Exception:
                th_best = None
                sweep = None
            # Threshold sweep curves
            try:
                selected_th_plot = float(getattr(cfg, 'MIN_AGRI_PROB', 0.5))
                _plot_threshold_sweep(
                    yv,
                    probs,
                    os.path.join(out_dir, "threshold_sweep.png"),
                    selected_th=selected_th_plot,
                    sweep=sweep,
                )
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

            try:
                _plot_false_positive_hotspots(lat_val, lon_val, yv, preds, os.path.join(out_dir, "false_positive_hotspots.png"))
            except Exception as e:
                print(f"False-positive hotspot plot failed: {e}")
            try:
                _plot_per_island_bars(islands_val, yv, preds, os.path.join(out_dir, "per_island_precision_recall.png"))
            except Exception as e:
                print(f"Per-island bar plot failed: {e}")
            try:
                _plot_feature_response_scatter(X[va_idx], feature_names_all, probs, yv, os.path.join(out_dir, "feature_response_scatter.png"))
            except Exception as e:
                print(f"Feature response scatter failed: {e}")
            try:
                island_label = getattr(cfg, 'get_selected_island', lambda: None)()
                _plot_round_metric_trends(os.path.join(out_dir, "round_metric_trends.png"), island_label)
            except Exception as e:
                print(f"Round metrics trend plot failed: {e}")
            try:
                _plot_confusion_map(lat_val, lon_val, yv, preds, probs, os.path.join(out_dir, "confusion_map.png"))
            except Exception as e:
                print(f"Spatial confusion plot failed: {e}")

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
