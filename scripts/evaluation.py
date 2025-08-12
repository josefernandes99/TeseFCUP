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
)
from a3_phase1_active_learning_round import extract_features_from_label
from config import EVALUATE_FILE, MIN_AGRI_PROB

def evaluate_model(model, eval_file=EVALUATE_FILE):
    """Evaluate a trained model against the evaluation labels in
    ``evaluate.csv``.  Returns a dictionary of metrics or ``None`` if the file
    does not exist or contains no labels."""
    if not os.path.exists(eval_file):
        print("Evaluation file missing →", eval_file)
        return None
    rows = list(csv.DictReader(open(eval_file)))
    if not rows:
        print("No evaluation labels found.")
        return None
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
    return metrics
