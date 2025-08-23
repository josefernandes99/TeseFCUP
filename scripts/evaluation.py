import os
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
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from al_shared import extract_features_from_label
import config as cfg
from splits import load_labels, stratified_train_val_test_indices

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
    # stratified split indices
    tr_idx, va_idx, _ = stratified_train_val_test_indices(y, cfg.TRAIN_FRACTION, cfg.VAL_FRACTION, cfg.TEST_FRACTION,
                                                          cfg.SPLIT_RANDOM_SEED if cfg.SPLIT_SEED_MODE == "fixed" else None)
    if va_idx.size == 0:
        print("Validation split empty; skipping evaluation.")
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
        "balanced_accuracy": ( (tp/(tp+fn) if (tp+fn)>0 else 0) + (tn/(tn+fp) if (tn+fp)>0 else 0) )/2,
        "mcc": matthews_corrcoef(yv, preds) if len(np.unique(yv))>1 and len(np.unique(preds))>1 else 0.0,
        "cohen_kappa": cohen_kappa_score(yv, preds),
        "brier_score": brier_score_loss(yv, probs),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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

        # metrics JSON for programmatic use
        with open(os.path.join(out_dir, "metrics.json"), "w") as jf:
            json.dump(metrics, jf, indent=2)
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
