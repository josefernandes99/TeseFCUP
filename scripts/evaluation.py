import os
import csv
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay,
)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from al_shared import extract_features_from_label, preload_tiles
from config import EVALUATE_FILE, MIN_AGRI_PROB

def evaluate_model(model, eval_file=EVALUATE_FILE, out_dir=None):
    """Evaluate ``model`` against the labels in ``evaluate.csv``.

    Parameters
    ----------
    model : sklearn-like estimator
        Trained model implementing ``predict_proba``.
    eval_file : str
        Path to ``evaluate.csv`` with columns ``tile,lat,lon,label``.
    out_dir : str, optional
        If provided, write a ``metrics.json`` file and diagnostic plots
        (confusion matrix, ROC curve) into this directory.

    Returns
    -------
    dict | None
        Dictionary of scalar metrics or ``None`` if evaluation labels are
        missing.
    """
    if not os.path.exists(eval_file):
        print("Evaluation file missing →", eval_file)
        return None
    rows = list(csv.DictReader(open(eval_file)))
    if not rows:
        print("No evaluation labels found.")
        return None
    tiles = sorted({r["tile"] for r in rows})
    preload_tiles(tiles)
    X, y = [], []
    for r in rows:
        feats = extract_features_from_label(r)
        if feats is None:
            continue
        X.append(feats)
        y.append(1 if r["label"].lower() == "agricultural" else 0)
    if not X:
        print("No features extracted for evaluation set.")
        return None
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= MIN_AGRI_PROB).astype(int)
    cm = confusion_matrix(y, preds, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "accuracy": accuracy_score(y, preds),
        "auc": roc_auc_score(y, probs),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        # classification report
        report = classification_report(y, preds, digits=3)
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
        roc_disp = RocCurveDisplay.from_predictions(y, probs)
        roc_disp.figure_.savefig(os.path.join(out_dir, "roc_curve.png"))
        plt.close(roc_disp.figure_)

        # metrics JSON for programmatic use
        import json
        with open(os.path.join(out_dir, "metrics.json"), "w") as jf:
            json.dump(metrics, jf, indent=2)

    return metrics
