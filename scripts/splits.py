import os
import csv
from typing import Tuple, List, Dict

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import config as cfg
from al_shared import extract_features_from_label


def load_labels(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _ensure_min_per_class(y: np.ndarray, min_per_split: int) -> bool:
    from collections import Counter
    c = Counter(y.tolist())
    return all(v >= min_per_split for v in c.values())


def stratified_train_val_test_indices(
    y: np.ndarray,
    train_frac: float = None,
    val_frac: float = None,
    test_frac: float = None,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index arrays for stratified train/val/test splits.

    For very small datasets, guarantees at least one sample per class in each
    requested split (if feasible). If a split fraction is 0, it will be empty.
    """
    n = len(y)
    if train_frac is None:
        train_frac = cfg.TRAIN_FRACTION
    if val_frac is None:
        val_frac = cfg.VAL_FRACTION
    if test_frac is None:
        test_frac = cfg.TEST_FRACTION
    # Preserve caller-provided random_state; None means use library randomness.

    # Normalize fractions to not exceed 1.0
    total = train_frac + val_frac + test_frac
    if total > 1.0:
        train_frac /= total
        val_frac /= total
        test_frac /= total

    all_idx = np.arange(n)
    train_idx, val_idx, test_idx = all_idx.copy(), np.array([], int), np.array([], int)

    # First carve out test set if requested
    if test_frac > 0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
        (train_pool_idx, test_idx), = sss.split(all_idx, y)
    else:
        train_pool_idx = all_idx

    # Then carve out validation from remaining pool
    if val_frac > 0:
        # Relative val fraction within the pool
        rel_val = val_frac / (1.0 - test_frac if test_frac < 1.0 else 1.0)
        if rel_val <= 0:
            train_idx = train_pool_idx
        else:
            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_val, random_state=random_state)
            (train_idx, val_idx), = sss2.split(train_pool_idx, y[train_pool_idx])
    else:
        train_idx = train_pool_idx

    # Ensure minimum 1 per class in each split if split not empty
    def fix_empty(split_idx: np.ndarray) -> np.ndarray:
        return split_idx if split_idx.size else np.array([], int)

    train_idx = fix_empty(train_idx)
    val_idx = fix_empty(val_idx)
    test_idx = fix_empty(test_idx)

    return train_idx, val_idx, test_idx


def build_feature_matrix(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features and labels for the given labeled rows.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    """
    X, y = [], []
    for r in rows:
        feats = extract_features_from_label(r)
        if feats is None:
            continue
        X.append(feats)
        y.append(1 if r["label"].lower() == "agricultural" else 0)
    if not X:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)
