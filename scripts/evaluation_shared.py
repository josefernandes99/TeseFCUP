from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve

import config as cfg
import retrospective_evaluate as retro


LabelRecords = List[retro.LabelRecord]
PixelLookup = Dict[str, Dict[Tuple[int, int], List[int]]]

__all__ = [
    "load_holdout_records",
    "evaluate_holdout_predictions",
]


def load_holdout_records(csv_path: Path) -> Tuple[LabelRecords, PixelLookup]:
    """Load testing labels into LabelRecord objects and a pixel lookup map."""
    if not csv_path.exists():
        return [], defaultdict(lambda: defaultdict(list))

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = cfg.filter_label_rows(list(reader))

    records: LabelRecords = []
    pixel_lookup: PixelLookup = defaultdict(lambda: defaultdict(list))

    for row in rows:
        tile = row.get("tile", "")
        if not tile:
            continue
        row_field = row.get("row_idx") or row.get("row")
        col_field = row.get("col_idx") or row.get("col")
        if not row_field or not col_field:
            continue
        try:
            r_idx = int(float(row_field))
            c_idx = int(float(col_field))
            lat = float(row.get("lat"))
            lon = float(row.get("lon"))
        except (TypeError, ValueError):
            continue
        label_text = (row.get("label") or "").strip().lower()
        is_agri = 1 if label_text == "agricultural" else 0
        idx = len(records)
        rec = retro.LabelRecord(
            idx=idx,
            label_id=row.get("id", ""),
            tile=tile,
            lat=lat,
            lon=lon,
            row=r_idx,
            col=c_idx,
            is_agri=is_agri,
            notes=row.get("notes", ""),
        )
        records.append(rec)
        pixel_lookup[tile][(r_idx, c_idx)].append(idx)

    return records, pixel_lookup


def _write_metrics_repeated(out_dir: Path, metrics: Dict[str, float]) -> None:
    payload = {"repeats": 1, "metrics": [metrics]}
    with (out_dir / "metrics_repeated.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _plot_curves(y_true: np.ndarray, probs: np.ndarray, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    prec, rec, _ = precision_recall_curve(y_true, probs)
    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
    ax_pr.plot(rec, prec, color="#1f77b4", label="PR curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.grid(True, alpha=0.2)
    ax_pr.legend(loc="best")
    fig_pr.tight_layout()
    fig_pr.savefig(out_dir / "pr_curve.png", dpi=180)
    plt.close(fig_pr)

    fpr, tpr, _ = roc_curve(y_true, probs)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, color="#1f77b4", label="ROC curve")
    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.grid(True, alpha=0.2)
    ax_roc.legend(loc="lower right")
    fig_roc.tight_layout()
    fig_roc.savefig(out_dir / "roc_curve.png", dpi=180)
    plt.close(fig_roc)


def evaluate_holdout_predictions(
    pred_csv_path: Path | str,
    out_dir: Path | str,
    testing_labels_csv: Path | str | None = None,
    verbose: bool = False,
    label: str | None = None,
) -> Dict[str, float] | None:
    """Evaluate predictions against the frozen testing labels."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    testing_path = Path(testing_labels_csv or cfg.TESTING_LABELS_FILE)
    if not testing_path.exists():
        return None

    records, pixel_lookup = load_holdout_records(testing_path)
    if not records:
        return None

    pred_path = Path(pred_csv_path)
    if not pred_path.exists():
        return None

    logger = retro.Logger(verbose=verbose)

    probs = retro.collect_probs_from_predictions(pred_path, records, pixel_lookup)
    missing = np.isnan(probs)
    if missing.any():
        logger.warn(
            f"Missing predictions for {int(missing.sum())} of {len(probs)} testing labels"
        )
        probs = np.nan_to_num(probs, nan=0.0)

    threshold = float(getattr(cfg, "MIN_AGRI_PROB", 0.35))
    y_true = np.array([rec.is_agri for rec in records], dtype=np.int32)
    y_pred = (probs >= threshold).astype(int)

    metrics = retro.compute_metrics(records, probs)

    best, best_metrics, sweep = retro.compute_best_threshold_weighted(y_true, probs)
    best_threshold = float(best.get("threshold", threshold))
    metrics["best_threshold"] = best_threshold
    metrics["best_score"] = float(best.get("score", 0.0))
    metrics["best_precision"] = float(best_metrics.get("precision", 0.0))
    metrics["best_recall"] = float(best_metrics.get("recall", 0.0))
    metrics["best_f1"] = float(best_metrics.get("f1", 0.0))

    retro.write_metrics(out_path, metrics)
    _write_metrics_repeated(out_path, metrics)

    report = classification_report(y_true, y_pred, digits=3, zero_division=0)
    with (out_path / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    retro.plot_confusion_matrix(y_true, y_pred, out_path / "confusion_matrix.png")
    retro.plot_prob_hist(y_true, probs, out_path / "prob_hist_by_class.png")

    lats = np.array([rec.lat for rec in records], dtype=float)
    lons = np.array([rec.lon for rec in records], dtype=float)
    retro.plot_false_positive_hotspots(lats, lons, y_true, y_pred, out_path / "false_positive_hotspots.png")
    retro.plot_confusion_map(lats, lons, y_true, y_pred, probs, out_path / "confusion_map.png")

    island_label = cfg.get_selected_island() or (label or "All Islands")
    retro.plot_per_island_bars(island_label, y_true, y_pred, out_path / "per_island_precision_recall.png")

    retro.plot_threshold_sweep(
        y_true,
        probs,
        sweep,
        out_path / "threshold_sweep.png",
        selected_th=threshold,
        best_th=best_threshold,
    )
    retro.plot_pr_roc_combined(
        y_true,
        probs,
        threshold,
        best_threshold,
        out_path / "pr_roc_combined.png",
    )
    retro.plot_calibration(y_true, probs, out_path / "calibration.png")
    _plot_curves(y_true, probs, out_path)

    best_payload = dict(best_metrics)
    best_payload.update({"threshold": best_threshold, "score": metrics["best_score"]})
    with (out_path / "best_threshold_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    return metrics
