#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from functools import lru_cache

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from concurrent.futures import ThreadPoolExecutor, as_completed

from progress_utils import new_progress
from memory_watcher import free_unused_memory
import config as cfg

MIN_THRESHOLD = float(getattr(cfg, "MIN_AGRI_PROB", 0.35))
DEFAULT_THRESHOLDS = np.linspace(0.0, 1.0, 101)

ALLOWED_LABEL_ID_SUBSTRINGS = ("manual", "highscore", "probableagri")

STAT_FILES_GENERATED = {
    "metrics.json",
    "metrics_summary.csv",
    "metrics_repeated.json",
    "classification_report.txt",
    "confusion_matrix.png",
    "confusion_map.png",
    "false_positive_hotspots.png",
    "pr_curve.png",
    "roc_curve.png",
    "pr_roc_combined.png",
    "prob_hist_by_class.png",
    "threshold_sweep.png",
    "calibration.png",
    "per_island_precision_recall.png",
    "round_metric_trends.png",
    "best_threshold_metrics.json",
    "old_vs_new_metrics.csv",
    "rounds_metrics.csv",
    "rounds_metrics.png",
}

@dataclass
class LabelRecord:
    idx: int
    label_id: str
    tile: str
    lat: float
    lon: float
    row: int
    col: int
    is_agri: int
    notes: str


class Logger:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def info(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def warn(self, msg: str) -> None:
        print(f"[warn] {msg}", file=sys.stderr)


TilePredData = namedtuple("TilePredData", ["rows", "cols", "probs", "lats", "lons", "mapping"])


def _canonical_island_key(name: str | None) -> str:
    """Collapse island names to a lowercase alphanumeric key for matching."""
    if not isinstance(name, str):
        return ""
    lowered = name.strip().lower()
    return "".join(ch for ch in lowered if ch.isalnum())


@lru_cache(maxsize=8)
def _load_tile_prediction_data(csv_path: str) -> TilePredData:
    rows: List[int] = []
    cols: List[int] = []
    probs: List[float] = []
    lats: List[float] = []
    lons: List[float] = []
    mapping: Dict[Tuple[int, int], float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            try:
                r_idx = int(record["row_idx"])
                c_idx = int(record["col_idx"])
                prob = float(record["predicted_prob"])
                lat = float(record.get("center_lat", "nan"))
                lon = float(record.get("center_lon", "nan"))
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(r_idx)
            cols.append(c_idx)
            probs.append(prob)
            lats.append(lat)
            lons.append(lon)
            mapping[(r_idx, c_idx)] = prob
    return TilePredData(
        rows=np.asarray(rows, dtype=np.int32),
        cols=np.asarray(cols, dtype=np.int32),
        probs=np.asarray(probs, dtype=np.float64),
        lats=np.asarray(lats, dtype=np.float64),
        lons=np.asarray(lons, dtype=np.float64),
        mapping=mapping,
    )



def find_round_dirs(island_dir: Path) -> List[Tuple[int, Path]]:
    rounds: List[Tuple[int, Path]] = []
    for child in island_dir.iterdir():
        if child.is_dir() and child.name.startswith("round_"):
            try:
                num = int(child.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            rounds.append((num, child))
    rounds.sort(key=lambda x: x[0])
    return rounds


def find_combo_dirs(round_dir: Path) -> List[Path]:
    combos: List[Path] = []
    for child in round_dir.iterdir():
        if child.is_dir():
            if (child / "statistics").is_dir() and (child / "model_round_{}.pkl".format(round_dir.name.split("_")[1])).exists():
                combos.append(child)
            elif (child / "statistics").is_dir():
                combos.append(child)
    combos.sort()
    return combos


def find_latest_tile_preds(island_dir: Path) -> Path | None:
    rounds = find_round_dirs(island_dir)
    for num, rnd in reversed(rounds):
        for combo in find_combo_dirs(rnd):
            candidate = combo / "_tile_preds"
            if candidate.is_dir():
                return candidate
    return None


def list_tiles(tile_dir: Path) -> List[str]:
    tiles: List[str] = []
    for file in tile_dir.glob("*.csv"):
        name = file.name
        if name.endswith(".csv"):
            tiles.append(name[:-4] + ".tif")
    return tiles


def read_existing_frozen(labels_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with labels_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_testing_labels(logger: Logger) -> List[Dict[str, str]]:
    # Prefer the master labels file so evaluations use the unified phase 1 labels.csv data.
    configured_path = getattr(cfg, "LABELS_FILE", None) or getattr(cfg, "MASTER_LABELS_FILE", "")
    path = Path(configured_path) if configured_path else Path(cfg.LABELS_DIR) / "labels.csv"
    if not path or not path.exists():
        logger.warn(f"Testing labels file missing => {path}")
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            label_id = (row.get("id") or "").lower()
            if not any(substr in label_id for substr in ALLOWED_LABEL_ID_SUBSTRINGS):
                continue
            rows.append(row)
        return rows


def build_label_rows(logger: Logger, final_labels_path: Path, tile_dir: Path) -> List[Dict[str, str]]:
    src = final_labels_path
    if not src.exists():
        logger.warn(f"Missing labels file: {src}")
        return []
    tiles = set(list_tiles(tile_dir))
    if not tiles:
        logger.warn(f"No tile predictions found under {tile_dir}")
        return []
    rows: List[Dict[str, str]] = []
    seen = set()
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_id = (row.get("id", "") or "")
            label_id_lower = label_id.lower()
            if not any(sub in label_id_lower for sub in ALLOWED_LABEL_ID_SUBSTRINGS):
                continue
            tile = row.get("tile", "").strip()
            if tile not in tiles:
                continue
            try:
                lat = float(row.get("lat", "nan"))
                lon = float(row.get("lon", "nan"))
            except ValueError:
                continue
            key = (tile, round(lat, 7), round(lon, 7))
            if key in seen:
                continue
            seen.add(key)
            label_name = row.get("label", "").strip().lower()
            if label_name not in {"agricultural", "non-agricultural"}:
                continue
            rows.append({
                "id": label_id,
                "lat": f"{lat:.8f}",
                "lon": f"{lon:.8f}",
                "tile": tile,
                "label": "Agricultural" if label_name == "agricultural" else "Non-Agricultural",
                "notes": row.get("notes", ""),
            })
    return rows


def prepare_testing_records(
    logger: Logger,
    island_name: str,
    tile_dir: Path,
    base_rows: Sequence[Dict[str, str]],
    tiles: Sequence[str],
) -> Tuple[List[LabelRecord], Dict[str, Dict[Tuple[int, int], List[int]]]]:
    tile_set = {t for t in tiles}
    rows = [dict(row) for row in base_rows if row.get("tile") in tile_set]
    if not rows:
        logger.warn(f"    No testing labels aligned with tiles for {island_name}")
        return [], defaultdict(lambda: defaultdict(list))

    for row in rows:
        row.setdefault("notes", "")
        row.setdefault("row_idx", row.get("row", ""))
        row.setdefault("col_idx", row.get("col", ""))

    missing = [idx for idx, row in enumerate(rows) if not row.get("row_idx") or not row.get("col_idx")]
    if missing:
        labels_by_tile: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(rows):
            labels_by_tile[row.get("tile", "")].append(idx)
        for tile, indices in labels_by_tile.items():
            if not indices:
                continue
            tile_csv = tile_dir / (Path(tile).stem + ".csv")
            if not tile_csv.exists():
                logger.warn(f"    Tile predictions missing for {tile}; cannot snap row/col")
                continue
            match_labels_for_tile(logger, tile_csv, rows, indices)

    records: List[LabelRecord] = []
    pixel_lookup: Dict[str, Dict[Tuple[int, int], List[int]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        tile = row.get("tile", "")
        try:
            lat = float(row.get("lat", "nan"))
            lon = float(row.get("lon", "nan"))
            r_idx = int(float(row.get("row_idx", "nan")))
            c_idx = int(float(row.get("col_idx", "nan")))
        except (TypeError, ValueError):
            logger.warn(f"    Skipping label without valid geometry => {row.get('id')}")
            continue
        label_text = (row.get("label") or "").strip().lower()
        is_agri = 1 if label_text == "agricultural" else 0
        rec_idx = len(records)
        record = LabelRecord(
            idx=rec_idx,
            label_id=row.get("id", ""),
            tile=tile,
            lat=lat,
            lon=lon,
            row=r_idx,
            col=c_idx,
            is_agri=is_agri,
            notes=row.get("notes", ""),
        )
        records.append(record)
        pixel_lookup[tile][(r_idx, c_idx)].append(rec_idx)

    if not records:
        logger.warn(f"    No usable testing labels after processing for {island_name}")
    return records, pixel_lookup


def load_tile_predictions(tile_csv: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = _load_tile_prediction_data(str(tile_csv))
    return data.rows, data.cols, data.lats, data.lons


def match_labels_for_tile(logger: Logger, tile_csv: Path, labels: List[Dict[str, str]], indices: List[int]) -> None:
    if not indices:
        return
    data = _load_tile_prediction_data(str(tile_csv))
    rows_arr = data.rows
    cols_arr = data.cols
    lats_arr = data.lats
    lons_arr = data.lons
    precisions = [7, 6, 5]
    pending = set(indices)
    cache: Dict[int, Tuple[int, int, float, float]] = {}
    for precision in precisions:
        if not pending:
            break
        key_to_source: Dict[Tuple[float, float], int] = {}
        if lats_arr.size:
            rounded_lats = np.round(lats_arr.astype(np.float64), precision)
            rounded_lons = np.round(lons_arr.astype(np.float64), precision)
            for src_idx, key in enumerate(zip(rounded_lats, rounded_lons)):
                if np.isnan(key[0]) or np.isnan(key[1]):
                    continue
                if key not in key_to_source:
                    key_to_source[key] = src_idx
        for label_idx in list(pending):
            try:
                lat = float(labels[label_idx]["lat"])
                lon = float(labels[label_idx]["lon"])
            except (KeyError, ValueError, TypeError):
                continue
            key = (round(lat, precision), round(lon, precision))
            src_idx = key_to_source.get(key)
            if src_idx is None:
                continue
            cache[label_idx] = (
                int(rows_arr[src_idx]),
                int(cols_arr[src_idx]),
                float(lats_arr[src_idx]),
                float(lons_arr[src_idx]),
            )
            pending.remove(label_idx)
    if pending and lats_arr.size:
        valid_mask = np.isfinite(lats_arr) & np.isfinite(lons_arr)
        ref_lats = lats_arr[valid_mask]
        ref_lons = lons_arr[valid_mask]
        ref_rows = rows_arr[valid_mask]
        ref_cols = cols_arr[valid_mask]
        if ref_lats.size:
            ref_coords = np.column_stack((ref_lats, ref_lons))
            for label_idx in list(pending):
                try:
                    lat = float(labels[label_idx]["lat"])
                    lon = float(labels[label_idx]["lon"])
                except (KeyError, ValueError, TypeError):
                    continue
                dists = np.hypot(ref_coords[:, 0] - lat, ref_coords[:, 1] - lon)
                if not np.isfinite(dists).any():
                    continue
                min_pos = int(np.nanargmin(dists))
                if np.isfinite(dists[min_pos]) and dists[min_pos] <= 1e-4:
                    cache[label_idx] = (
                        int(ref_rows[min_pos]),
                        int(ref_cols[min_pos]),
                        float(ref_lats[min_pos]),
                        float(ref_lons[min_pos]),
                    )
                    pending.remove(label_idx)
    for idx in indices:
        if idx in cache:
            r_idx, c_idx, lat, lon = cache[idx]
            labels[idx]["row_idx"] = str(r_idx)
            labels[idx]["col_idx"] = str(c_idx)
            labels[idx]["lat"] = f"{lat:.8f}"
            labels[idx]["lon"] = f"{lon:.8f}"
        else:
            labels[idx]["row_idx"] = ""
            labels[idx]["col_idx"] = ""
            logger.warn(f"Failed to match label {labels[idx]['id']} for tile {labels[idx]['tile']}")


def ensure_frozen_labels(logger: Logger, island_dir: Path, tile_dir: Path, final_labels_path: Path) -> Path:
    frozen = island_dir / "finalLabels_frozen.csv"
    allowed_substrings = ("manual", "highscore", "probableagri")
    def _filter_rows(lines):
        filtered = []
        for row in lines:
            label_id = (row.get("id", "") or "")
            if not any(sub in label_id.lower() for sub in allowed_substrings):
                continue
            filtered.append(row)
        return filtered
    if frozen.exists():
        rows = _filter_rows(read_existing_frozen(frozen))
        missing_rowcol = [row for row in rows if not row.get("row_idx") or not row.get("col_idx")]
        if not missing_rowcol and rows:
            return frozen
        logger.info(f"Completing missing row/col entries in {frozen}")
        labels = rows
    else:
        if not final_labels_path.exists():
            logger.warn(f"    Missing finalLabels.csv at {final_labels_path}")
            return frozen
        labels = build_label_rows(logger, final_labels_path, tile_dir)
        if not labels:
            return frozen
    labels_by_tile: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(labels):
        tile = row.get("tile", "")
        if tile:
            labels_by_tile[tile].append(idx)
    for tile, indices in labels_by_tile.items():
        src = tile_dir / (Path(tile).stem + ".csv")
        if not src.exists():
            logger.warn(f"Missing tile predictions for {tile}: {src}")
            for idx in indices:
                labels[idx]["row_idx"] = ""
                labels[idx]["col_idx"] = ""
            continue
        match_labels_for_tile(logger, src, labels, indices)
    valid_labels = [row for row in labels if row.get("row_idx") and row.get("col_idx")]
    if not valid_labels:
        logger.warn(f"No valid labels found after snapping for {island_dir.name}")
        return frozen
    with frozen.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "lat", "lon", "tile", "label", "notes", "row_idx", "col_idx"],
        )
        writer.writeheader()
        for row in valid_labels:
            writer.writerow(row)
    logger.info(f"Wrote {len(valid_labels)} frozen labels to {frozen}")
    return frozen


def load_label_dataset(logger: Logger, frozen: Path, tiles: Iterable[str]) -> Tuple[List[LabelRecord], Dict[str, Dict[Tuple[int, int], List[int]]]]:
    tile_set = set(tiles)
    records: List[LabelRecord] = []
    pixel_lookup: Dict[str, Dict[Tuple[int, int], List[int]]] = defaultdict(lambda: defaultdict(list))
    with frozen.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        idx = 0
        for row in reader:
            tile = row.get("tile", "")
            if tile not in tile_set:
                continue
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                r_idx = int(row["row_idx"])
                c_idx = int(row["col_idx"])
            except (KeyError, ValueError, TypeError):
                continue
            label_text = row.get("label", "")
            is_agri = 1 if label_text.strip().lower() == "agricultural" else 0
            rec = LabelRecord(
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
            idx += 1
    if not records:
        logger.warn(f"No records loaded from {frozen}")
    return records, pixel_lookup


def collect_probs_from_predictions(pred_path: Path, records: Sequence[LabelRecord], pixel_lookup: Dict[str, Dict[Tuple[int, int], List[int]]]) -> np.ndarray:
    probs = np.full(len(records), np.nan, dtype=np.float64)
    if not pred_path.exists():
        return probs
    remaining: Dict[str, set[Tuple[int, int]]] = {tile: set(mapping.keys()) for tile, mapping in pixel_lookup.items()}
    with pred_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile = row.get("tile")
            mapping = pixel_lookup.get(tile)
            if not mapping:
                continue
            try:
                key = (int(row["row_idx"]), int(row["col_idx"]))
                prob = float(row["predicted_prob"])
            except (KeyError, ValueError):
                continue
            indices = mapping.get(key)
            if not indices:
                continue
            for idx in indices:
                probs[idx] = prob
            remaining_tile = remaining.get(tile)
            if remaining_tile is not None:
                remaining_tile.discard(key)
                if not remaining_tile:
                    del remaining[tile]
                    if not remaining:
                        break
    return probs


def collect_probs_from_tile_dir(tile_dir: Path, records: Sequence[LabelRecord], pixel_lookup: Dict[str, Dict[Tuple[int, int], List[int]]]) -> np.ndarray:
    probs = np.full(len(records), np.nan, dtype=np.float64)
    for tile, index_map in pixel_lookup.items():
        csv_path = tile_dir / (Path(tile).stem + ".csv")
        if not csv_path.exists():
            continue
        data = _load_tile_prediction_data(str(csv_path))
        tile_mapping = data.mapping
        for key, indices in index_map.items():
            prob = tile_mapping.get(key)
            if prob is None:
                continue
            for idx in indices:
                probs[idx] = prob
    return probs


def _metrics_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = 1.0 - specificity
    fnr = 1.0 - recall
    macro_precision = (precision + (tn / (tn + fn) if (tn + fn) else 0.0)) / 2.0
    macro_recall = (recall + specificity) / 2.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) else 0.0
    recall_neg = specificity
    f1_neg = 2.0 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) else 0.0
    macro_f1 = (f1 + f1_neg) / 2.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    if (tp + tn + fp + fn):
        po = accuracy
        pe = (((tp + fp) / (tp + tn + fp + fn)) * ((tp + fn) / (tp + tn + fp + fn))) + (((tn + fp) / (tp + tn + fp + fn)) * ((tn + fn) / (tp + tn + fp + fn)))
        kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) else 0.0
    else:
        kappa = 0.0
    brier = brier_score_loss(y_true, probs)
    return {
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
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
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "tpr": float(recall),
    }


def compute_best_threshold_weighted(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray | None = None) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]]]:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    metrics_grid: Dict[str, List[float]] = {}
    score_series: List[float] = []
    local_pr_series: List[float] = []
    weights = {
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
    ap = float(average_precision_score(y_true, probs)) if y_true.size else 0.0
    roc_auc = float(roc_auc_score(y_true, probs)) if y_true.size and len(np.unique(y_true)) > 1 else 0.0
    brier_prob = float(np.mean((probs - y_true) ** 2)) if y_true.size else 0.0
    best = {"threshold": float(thresholds[0]) if thresholds.size else 0.0, "score": -np.inf}
    best_metrics: Dict[str, float] | None = None
    for th in thresholds:
        stats = _metrics_at_threshold(y_true, probs, float(th))
        if not metrics_grid:
            metrics_grid = {key: [] for key in stats.keys()}
        for key, value in stats.items():
            metrics_grid[key].append(float(value))
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
        local_pr = precision * recall
        one_minus_brier = 1.0 - brier
        score = (
            weights["precision"] * precision
            + weights["recall"] * recall
            + weights["f1"] * f1
            + weights["macro_f1"] * macro_f1
            + weights["balanced_accuracy"] * balanced_accuracy
            + weights["mcc"] * mcc
            + weights["cohen_kappa"] * kappa
            + weights["accuracy"] * accuracy
            + weights["one_minus_brier"] * one_minus_brier
            + weights["local_pr_area"] * local_pr
            + weights["pr_auc"] * ap
            - weights["fpr_penalty"] * fpr
        )
        score_series.append(float(score))
        local_pr_series.append(float(local_pr))
        if score > best["score"] + 1e-6:
            best = {"threshold": float(th), "score": float(score)}
            best_metrics = {**stats}
        elif math.isclose(score, best["score"], abs_tol=1e-6) and best_metrics is not None:
            prev = best_metrics
            better = False
            if precision > prev.get("precision", 0.0) + 1e-6:
                better = True
            elif math.isclose(precision, prev.get("precision", 0.0), abs_tol=1e-6) and fpr < prev.get("fpr", 1.0) - 1e-6:
                better = True
            elif (
                math.isclose(precision, prev.get("precision", 0.0), abs_tol=1e-6)
                and math.isclose(fpr, prev.get("fpr", 1.0), abs_tol=1e-6)
                and recall > prev.get("recall", 0.0) + 1e-6
            ):
                better = True
            elif (
                math.isclose(precision, prev.get("precision", 0.0), abs_tol=1e-6)
                and math.isclose(fpr, prev.get("fpr", 1.0), abs_tol=1e-6)
                and math.isclose(recall, prev.get("recall", 0.0), abs_tol=1e-6)
                and float(th) > best.get("threshold", 0.0)
            ):
                better = True
            if better:
                best = {"threshold": float(th), "score": float(score)}
                best_metrics = {**stats}
    if best_metrics is None:
        best_metrics = {key: values[-1] for key, values in metrics_grid.items()}
    metrics_grid["pr_auc"] = [ap] * len(thresholds)
    metrics_grid["roc_auc"] = [roc_auc] * len(thresholds)
    metrics_grid["brier_prob"] = [brier_prob] * len(thresholds)
    metrics_grid["local_pr_area"] = local_pr_series
    metrics_grid["score"] = score_series
    return best, best_metrics, {"thresholds": list(map(float, thresholds)), "metrics": metrics_grid}


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(all="ignore"):
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cmn, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Confusion Matrix (normalised)")
    ax.set_xticks([0, 1], ["NonAgri", "Agri"])
    ax.set_yticks([0, 1], ["NonAgri", "Agri"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    fmt = ".2f"
    thresh = cmn.max() / 2.0
    for i in range(cmn.shape[0]):
        for j in range(cmn.shape[1]):
            ax.text(j, i, format(cmn[i, j], fmt), ha="center", va="center", color="white" if cmn[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prob_hist(y_true: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(neg, bins=bins, density=True, alpha=0.5, color="#1f77b4", label="NonAgri")
    ax.hist(pos, bins=bins, density=True, alpha=0.5, color="#d62728", label="Agri")
    ax.set_xlabel("Predicted probability (Agri)")
    ax.set_ylabel("Density")
    ax.set_title("Probability histograms by class")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_false_positive_hotspots(lats: np.ndarray, lons: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    mask_valid = np.isfinite(lats) & np.isfinite(lons)
    if not mask_valid.any():
        return
    lats = lats[mask_valid]
    lons = lons[mask_valid]
    y_true = np.asarray(y_true)[mask_valid]
    y_pred = np.asarray(y_pred)[mask_valid]
    mask_fp = (y_pred == 1) & (y_true == 0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(lons, lats, s=8, color="#d3d3d3", alpha=0.4, label="All samples")
    if mask_fp.any():
        coords_rad = np.deg2rad(np.column_stack((lats[mask_fp], lons[mask_fp])))
        try:
            clustering = DBSCAN(eps=1.0 / 6371.0088, min_samples=3, metric="haversine").fit(coords_rad)
            labels = clustering.labels_
        except Exception:
            labels = np.full(mask_fp.sum(), -1)
        unique = np.unique(labels)
        cmap = matplotlib.colormaps.get_cmap("tab10")
        fp_lats = lats[mask_fp]
        fp_lons = lons[mask_fp]
        for idx, lab in enumerate(unique):
            sel = labels == lab
            if lab == -1:
                ax.scatter(fp_lons[sel], fp_lats[sel], s=25, facecolors="none", edgecolors="red", linewidths=1.2, label="FP (noise)" if idx == 0 else None)
            else:
                color = cmap(idx % cmap.N)
                ax.scatter(fp_lons[sel], fp_lats[sel], s=35, color=color, alpha=0.85, label=f"FP cluster {lab + 1}")
    else:
        ax.text(0.5, 0.5, "No false positives", transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("False-positive hotspots")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_confusion_map(lats: np.ndarray, lons: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, out_path: Path) -> None:
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    probs = np.asarray(probs, dtype=float)
    mask = np.isfinite(lats) & np.isfinite(lons)
    if not mask.any():
        return
    lats = lats[mask]
    lons = lons[mask]
    y_true = np.asarray(y_true)[mask]
    y_pred = np.asarray(y_pred)[mask]
    outcome = np.empty_like(y_true, dtype="<U3")
    outcome[(y_true == 1) & (y_pred == 1)] = "TP"
    outcome[(y_true == 0) & (y_pred == 0)] = "TN"
    outcome[(y_true == 0) & (y_pred == 1)] = "FP"
    outcome[(y_true == 1) & (y_pred == 0)] = "FN"
    colors = {"TP": "#2ca02c", "TN": "#1f77b4", "FP": "#d62728", "FN": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in ["TP", "FP", "FN", "TN"]:
        sel = outcome == label
        if sel.any():
            ax.scatter(lons[sel], lats[sel], s=20, color=colors[label], alpha=0.7, label=label)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial confusion map")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_island_bars(island_name: str, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    metrics = [("Precision", precision, "#1f77b4"), ("Recall", recall, "#ff7f0e")]
    positions = np.arange(len(metrics))
    values = [m[1] for m in metrics]
    colors = [m[2] for m in metrics]
    labels = [m[0] for m in metrics]
    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.bar(positions, values, color=colors)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Precision/Recall ({island_name})")
    ax.grid(True, axis='y', alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_threshold_sweep(y_true: np.ndarray, probs: np.ndarray, sweep: Dict[str, List[float]], out_path: Path, selected_th: float, best_th: float) -> None:
    thresholds = np.asarray(sweep.get("thresholds", []), dtype=float)
    metrics_map = sweep.get("metrics", {})
    specs = [
        ("precision", "Precision", True),
        ("recall", "Recall", True),
        ("f1", "F1", True),
        ("macro_f1", "Macro F1", True),
        ("accuracy", "Accuracy", True),
        ("balanced_accuracy", "Balanced Accuracy", True),
        ("mcc", "MCC", True),
        ("cohen_kappa", "Cohen's κ", True),
        ("specificity", "Specificity", True),
        ("fpr", "False Positive Rate", False),
        ("local_pr_area", "Precision×Recall", True),
        ("brier_score", "1 − Brier", True),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (metric, title, higher_is_better) in zip(axes, specs):
        series = np.asarray(metrics_map.get(metric, []), dtype=float)
        if metric == "brier_score":
            series = 1.0 - series
        ax.plot(thresholds, series, color="#1f77b4")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.2)
        if not higher_is_better and series.size:
            ax.set_ylim(0.0, min(1.0, max(series) * 1.05))
        else:
            ax.set_ylim(0.0, 1.0)
        ax.axvline(selected_th, color="k", linestyle="--", linewidth=1.3)
        ax.axvline(best_th, color="#9467bd", linestyle=":", linewidth=1.3)
        ax.set_xlim(0.0, 1.0)
    for ax in axes[len(specs):]:
        ax.axis("off")
    pr_auc = metrics_map.get("pr_auc", [np.nan])[0]
    roc_auc = metrics_map.get("roc_auc", [np.nan])[0]
    brier_prob = metrics_map.get("brier_prob", [np.nan])[0]
    fig.suptitle(f"PR-AUC={pr_auc:.3f} | ROC-AUC={roc_auc:.3f} | Prob.Brier={brier_prob:.3f}", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.94))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pr_roc_combined(y_true: np.ndarray, probs: np.ndarray, selected_th: float, best_th: float, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    prec, rec, _ = precision_recall_curve(y_true, probs)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.2))
    ax_roc.plot(fpr, tpr, color="#1f77b4")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC")
    ax_roc.grid(True, alpha=0.2)
    ax_pr.plot(rec, prec, color="#2ca02c")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–Recall")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.grid(True, alpha=0.2)
    def mark(th, color, label):
        stats = _metrics_at_threshold(y_true, probs, th)
        ax_roc.scatter([stats["fpr"]], [stats["tpr"]], color=color, s=45, edgecolor="black", label=label)
        ax_pr.scatter([stats["recall"]], [stats["precision"]], color=color, s=45, edgecolor="black")
    mark(selected_th, "#d62728", "Selected threshold")
    if not math.isclose(selected_th, best_th, abs_tol=1e-6):
        mark(best_th, "#9467bd", "Best threshold")
    handles = [Line2D([0], [0], color="#1f77b4", label="ROC"), Line2D([0], [0], color="#2ca02c", label="PR"), Line2D([0], [0], marker='o', color='w', markerfacecolor="#d62728", markeredgecolor='black', label='Selected threshold')]
    if not math.isclose(selected_th, best_th, abs_tol=1e-6):
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor="#9467bd", markeredgecolor='black', label='Best threshold'))
    fig.legend(handles, [h.get_label() for h in handles], loc='center right', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout(rect=(0.0, 0.0, 0.95, 1.0))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, probs: np.ndarray, out_path: Path, n_bins: int = 10) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax1.plot(mean_pred, frac_pos, marker='o', linestyle='-', label='Model')
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlabel("Predicted probability")
    ax1.set_ylabel("Observed positive rate")
    ax1.grid(True, alpha=0.2)
    ax1.legend(loc="upper left")
    ax2.hist(probs, bins=n_bins, range=(0.0, 1.0), color="#6666cc", alpha=0.8)
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.grid(True, axis='y', alpha=0.2)
    try:
        bs = brier_score_loss(y_true, probs)
        fig.suptitle(f"Calibration (Brier={bs:.3f})", y=0.98)
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_round_metrics(island_dir: Path, out_path: Path) -> None:
    csv_path = island_dir / "rounds_metrics.csv"
    if not csv_path.exists():
        return
    rounds: List[int] = []
    precision: List[float] = []
    recall: List[float] = []
    f1_vals: List[float] = []
    auc_pr_vals: List[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_round = row.get("round")
            try:
                round_val = int(raw_round) if raw_round not in (None, "") else len(rounds) + 1
            except (TypeError, ValueError):
                round_val = len(rounds) + 1

            def _parse_metric(field: str) -> float:
                value = row.get(field)
                if value in (None, ""):
                    return math.nan
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return math.nan

            precision_val = _parse_metric("precision")
            recall_val = _parse_metric("recall")
            f1_val = _parse_metric("f1")
            auc_pr_val = _parse_metric("auc_pr")

            if all(math.isnan(metric) for metric in (precision_val, recall_val, f1_val, auc_pr_val)):
                continue

            rounds.append(round_val)
            precision.append(precision_val)
            recall.append(recall_val)
            f1_vals.append(f1_val)
            auc_pr_vals.append(auc_pr_val)
    if not rounds:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(rounds, precision, marker='o', label='Precision')
    ax.plot(rounds, recall, marker='o', label='Recall')
    ax.plot(rounds, f1_vals, marker='o', label='F1')
    ax.plot(rounds, auc_pr_vals, marker='o', label='AUC-PR')
    ax.set_xlabel('Round')
    ax.set_ylabel('Score')
    vals = []
    for series in (precision, recall, f1_vals, auc_pr_vals):
        for value in series:
            if value is not None:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isnan(scalar):
                    vals.append(scalar)
    if vals:
        vmin = min(vals)
        vmax = max(vals)
        lower = max(0.0, math.floor(vmin * 10.0) / 10.0)
        upper = min(1.0, math.ceil(vmax * 10.0) / 10.0)
        if upper - lower < 0.1:
            margin = 0.05
            lower = max(0.0, vmin - margin)
            upper = min(1.0, vmax + margin)
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(right=0.80)
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.82, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_metrics(records: Sequence[LabelRecord], probs: np.ndarray) -> Dict[str, float]:
    y_true = np.array([rec.is_agri for rec in records], dtype=np.int32)
    y_pred = (probs >= MIN_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = 0.0
    auc_pr = average_precision_score(y_true, probs)
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        mcc = 0.0
    kappa = cohen_kappa_score(y_true, y_pred)
    brier = brier_score_loss(y_true, probs)
    return {
        "TP": float(tp),
        "FP": float(fp),
        "TN": float(tn),
        "FN": float(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "auc_pr": float(auc_pr),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa),
        "brier_score": float(brier),
    }


def write_metrics(out_dir: Path, metrics: Dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "metrics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    repeated = {**metrics, "repeat": 1}
    with (out_dir / "metrics_repeated.json").open("w", encoding="utf-8") as f:
        json.dump([repeated], f, indent=2)


def write_old_vs_new(out_dir: Path, original_stats: Path, metrics: Dict[str, float]) -> None:
    old: Dict[str, float] = {}
    if original_stats.exists():
        with original_stats.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for key, value in reader:
                if key == "metric":
                    continue
                try:
                    old[key] = float(value)
                except (TypeError, ValueError):
                    old[key] = value
    keys = ["precision", "recall", "f1", "balanced_accuracy", "auc_pr", "auc", "mcc", "cohen_kappa", "brier_score"]
    with (out_dir / "old_vs_new_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "old", "new"])
        for key in keys:
            writer.writerow([key, old.get(key, ""), metrics.get(key, "")])


def copy_static_assets(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    for item in src_dir.iterdir():
        if item.name in STAT_FILES_GENERATED:
            continue
        dst = dst_dir / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)


def evaluate_model(logger: Logger, island_name: str, island_dir: Path, combo_dir: Path, model_name: str, records: Sequence[LabelRecord], pixel_lookup: Dict[str, Dict[Tuple[int, int], List[int]]], probs: np.ndarray, original_stats_dir: Path, new_stats_dir: Path, summary_rows: List[Tuple[str, str, float, float]]) -> None:
    y_true = np.array([rec.is_agri for rec in records], dtype=np.int32)
    if np.isnan(probs).any():
        missing = np.isnan(probs)
        logger.warn(f"Model {model_name}: missing probabilities for {missing.sum()} labels")
        probs = np.nan_to_num(probs, nan=0.0)
    y_pred = (probs >= MIN_THRESHOLD).astype(int)
    metrics = compute_metrics(records, probs)
    # Add derived stats
    metrics.update({
        "threshold": float(MIN_THRESHOLD),
        "std_precision": 0.0,
        "std_recall": 0.0,
        "std_f1": 0.0,
        "std_macro_f1": 0.0,
        "std_accuracy": 0.0,
        "std_auc": 0.0,
        "std_auc_pr": 0.0,
        "std_balanced_accuracy": 0.0,
        "std_mcc": 0.0,
        "std_cohen_kappa": 0.0,
        "std_brier_score": 0.0,
    })
    write_metrics(new_stats_dir, metrics)
    write_old_vs_new(new_stats_dir, original_stats_dir / "metrics_summary.csv", metrics)
    with (new_stats_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, digits=3, zero_division=0))
    plot_confusion_matrix(y_true, y_pred, new_stats_dir / "confusion_matrix.png")
    plot_prob_hist(y_true, probs, new_stats_dir / "prob_hist_by_class.png")
    plot_false_positive_hotspots(
        [rec.lat for rec in records],
        [rec.lon for rec in records],
        y_true,
        y_pred,
        new_stats_dir / "false_positive_hotspots.png",
    )
    plot_confusion_map(
        [rec.lat for rec in records],
        [rec.lon for rec in records],
        y_true,
        y_pred,
        probs,
        new_stats_dir / "confusion_map.png",
    )
    plot_per_island_bars(island_name, y_true, y_pred, new_stats_dir / "per_island_precision_recall.png")
    best, best_metrics, sweep = compute_best_threshold_weighted(y_true, probs)
    best_metrics.update({"threshold": best.get("threshold", MIN_THRESHOLD), "score": best.get("score", 0.0)})
    with (new_stats_dir / "best_threshold_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    plot_threshold_sweep(y_true, probs, sweep, new_stats_dir / "threshold_sweep.png", selected_th=MIN_THRESHOLD, best_th=float(best.get("threshold", MIN_THRESHOLD)))
    plot_pr_roc_combined(y_true, probs, MIN_THRESHOLD, float(best.get("threshold", MIN_THRESHOLD)), new_stats_dir / "pr_roc_combined.png")
    plot_calibration(y_true, probs, new_stats_dir / "calibration.png")
    precision_val = metrics.get("precision", 0.0)
    recall_val = metrics.get("recall", 0.0)
    summary_rows.append((model_name, "precision", metrics.get("precision", 0.0), best_metrics.get("precision", 0.0)))
    summary_rows.append((model_name, "recall", metrics.get("recall", 0.0), best_metrics.get("recall", 0.0)))
    summary_rows.append((model_name, "f1", metrics.get("f1", 0.0), best_metrics.get("f1", 0.0)))
    prec, rec, _ = precision_recall_curve(y_true, probs)
    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
    ax_pr.plot(rec, prec, color="#1f77b4", label="PR curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.0)
    ax_pr.grid(True, alpha=0.2)
    ax_pr.legend(loc='best')
    fig_pr.tight_layout()
    fig_pr.savefig(new_stats_dir / "pr_curve.png", dpi=180)
    plt.close(fig_pr)
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, color="#1f77b4", label="ROC curve")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.0)
    ax_roc.grid(True, alpha=0.2)
    ax_roc.legend(loc='best')
    fig_roc.tight_layout()
    fig_roc.savefig(new_stats_dir / "roc_curve.png", dpi=180)
    plt.close(fig_roc)
    plot_round_metrics(island_dir, new_stats_dir / "round_metric_trends.png")
    return metrics


def copy_kml_and_predictions(src_dir: Path, dst_dir: Path) -> None:
    for src in src_dir.glob('agricultural_patches_round_*.kml'):
        shutil.copy2(src, dst_dir / src.name)
    pred_src = src_dir / 'predictions.csv'
    if pred_src.exists():
        shutil.copy2(pred_src, dst_dir / 'predictions.csv')


def process_combo(logger: Logger, island_name: str, island_dir: Path, combo_dir: Path, records: Sequence[LabelRecord], pixel_lookup: Dict[str, Dict[Tuple[int, int], List[int]]]) -> Dict[str, float] | None:
    summary_rows: List[Tuple[str, str, float, float]] = []
    tile_dir = combo_dir / "_tile_preds"
    if tile_dir.exists():
        probs_ensemble = collect_probs_from_tile_dir(tile_dir, records, pixel_lookup)
    else:
        logger.warn(f"        Missing _tile_preds for {combo_dir}; ensemble metrics will be NaN")
        probs_ensemble = np.full(len(records), np.nan, dtype=np.float64)
    logger.info("        → Evaluating ensemble")
    ensemble_metrics = process_model(logger, island_name, island_dir, combo_dir, "ensemble", records, pixel_lookup, probs_ensemble, summary_rows)
    rf_preds = combo_dir / "statistics" / "rf" / "predictions.csv"
    if rf_preds.exists():
        logger.info("        → Evaluating random forest")
        probs_rf = collect_probs_from_predictions(rf_preds, records, pixel_lookup)
        process_model(logger, island_name, island_dir, combo_dir, "rf", records, pixel_lookup, probs_rf, summary_rows)
    else:
        logger.warn(f"        RF predictions missing for {combo_dir}")
    svm_preds = combo_dir / "statistics" / "svm" / "predictions.csv"
    if svm_preds.exists():
        logger.info("        → Evaluating SVM")
        probs_svm = collect_probs_from_predictions(svm_preds, records, pixel_lookup)
        process_model(logger, island_name, island_dir, combo_dir, "svm", records, pixel_lookup, probs_svm, summary_rows)
    else:
        logger.warn(f"        SVM predictions missing for {combo_dir}")
    summary_path = combo_dir / "new_statistics" / "old_vs_new_metrics.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "metric", "threshold_metric", "best_threshold_metric"])
        for row in summary_rows:
            writer.writerow(row)
    free_unused_memory()
    return ensemble_metrics


def process_model(logger: Logger, island_name: str, island_dir: Path, combo_dir: Path, model_name: str, records: Sequence[LabelRecord], pixel_lookup: Dict[str, Dict[Tuple[int, int], List[int]]], probs: np.ndarray, summary_rows: List[Tuple[str, str, float, float]]) -> None:
    original_stats_dir = combo_dir / "statistics" / model_name
    new_stats_dir = combo_dir / "new_statistics" / model_name
    if new_stats_dir.exists():
        shutil.rmtree(new_stats_dir)
    new_stats_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(
        logger,
        island_name,
        island_dir,
        combo_dir,
        model_name,
        records,
        pixel_lookup,
        probs,
        original_stats_dir,
        new_stats_dir,
        summary_rows,
    )
    copy_static_assets(original_stats_dir, new_stats_dir)
    copy_kml_and_predictions(original_stats_dir, new_stats_dir)
    free_unused_memory()
    return metrics


def ensure_new_statistics_root(combo_dir: Path) -> Path:
    root = combo_dir / "new_statistics"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root

def cleanup_previous_outputs(year_dirs: Sequence[Path]) -> None:
    to_remove: List[Path] = []
    for year_dir in year_dirs:
        for target in year_dir.rglob("new_statistics"):
            if target.is_dir():
                to_remove.append(target)
    if not to_remove:
        return
    for target in sorted(to_remove, key=lambda p: len(str(p)), reverse=True):
        try:
            shutil.rmtree(target)
        except Exception as exc:
            print(f"[warn] Failed to remove {target}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retest models against frozen final labels.")
    parser.add_argument("--years", nargs="*", help="Optional specific years (folder names under final_results)")
    parser.add_argument("--parallelism", type=int, default=None, help="Maximum combos to evaluate concurrently (default depends on CPU cores)")
    args = parser.parse_args()
    logger = Logger(verbose=True)
    base_dir = Path(cfg.BASE_DIR)
    final_results_root = base_dir / "final_results"
    if not final_results_root.exists():
        logger.warn(f"Missing final_results root: {final_results_root}")
        sys.exit(1)
    testing_rows_global = load_testing_labels(logger)
    testing_by_island: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in testing_rows_global:
        tile = row.get("tile", "")
        island = cfg.extract_island_name(tile)
        key = _canonical_island_key(island)
        if not key:
            continue
        testing_by_island[key].append(row)
    years: List[Path] = []
    if args.years:
        for year in args.years:
            path = final_results_root / year
            if path.exists():
                years.append(path)
            else:
                logger.warn(f"Year folder not found: {path}")
    else:
        years = [p for p in final_results_root.iterdir() if p.is_dir()]
    if not years:
        logger.warn("No year directories found to process.")
        sys.exit(1)
    logger.info("Cleaning previous new_statistics outputs…")
    cleanup_previous_outputs(years)
    cpu_total = os.cpu_count() or 1
    default_parallelism = max(1, min(4, max(1, cpu_total // 2)))
    combo_parallelism = args.parallelism if (args.parallelism and args.parallelism > 0) else default_parallelism
    logger.info(f"Using combo parallelism={combo_parallelism}")
    for year_dir in sorted(years):

        logger.info(f"Processing year {year_dir.name}")
        island_dirs = sorted([p for p in year_dir.iterdir() if p.is_dir()])
        if not island_dirs:
            logger.warn(f"No island folders under {year_dir}")
            continue
        with new_progress() as year_prog:
            year_task = year_prog.add_task(f"Islands ({year_dir.name})", total=len(island_dirs))
            for island_dir in island_dirs:
                island_name = island_dir.name
                year_prog.console.print(f"[bold cyan]▶ {year_dir.name} · {island_name}[/]", highlight=False)
                logger.info(f"  Island: {island_name}")
                tile_dir = find_latest_tile_preds(island_dir)
                if tile_dir is None:
                    logger.warn(f"    No _tile_preds available for {island_name}; skipping")
                    year_prog.advance(year_task)
                    continue
                island_key = _canonical_island_key(island_name)
                base_rows = testing_by_island.get(island_key)
                if not base_rows:
                    local_csv = island_dir / "testingLabels.csv"
                    if local_csv.exists():
                        logger.info(f"    Loading island-local testingLabels.csv => {local_csv}")
                        with local_csv.open(newline="", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            base_rows = list(reader)
                    if not base_rows:
                        logger.warn(f"    No testing labels for island {island_name}; skipping")
                        year_prog.advance(year_task)
                        continue
                tiles = list_tiles(tile_dir)
                records, pixel_lookup = prepare_testing_records(logger, island_name, tile_dir, base_rows, tiles)
                if not records:
                    logger.warn(f"    No labels available after filtering for {island_name}")
                    year_prog.advance(year_task)
                    continue
                island_summary_dir = island_dir / "new_statistics"
                if island_summary_dir.exists():
                    shutil.rmtree(island_summary_dir)
                island_summary_dir.mkdir(parents=True, exist_ok=True)
                round_metrics: Dict[int, Dict[str, float]] = {}
                rounds = find_round_dirs(island_dir)
                if not rounds:
                    logger.warn(f"    No round folders for {island_name}")
                    year_prog.advance(year_task)
                    continue
                with new_progress() as island_prog:
                    round_task = island_prog.add_task(f"Rounds ({island_name})", total=len(rounds))
                    for round_num, round_dir in rounds:
                        island_prog.console.print(f"  [green]Round {round_num}[/]")
                        combos = find_combo_dirs(round_dir)
                        if not combos:
                            logger.warn(f"      No combo folders in {round_dir}")
                            island_prog.advance(round_task)
                            continue
                        combo_task = island_prog.add_task(f"Combos (round {round_num})", total=len(combos))
                        for combo in combos:
                            ensure_new_statistics_root(combo)
                        worker_count = min(combo_parallelism, len(combos)) if combos else 1
                        if worker_count > 1:
                            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                                future_map = {}
                                for combo in combos:
                                    island_prog.console.print(f"    • {combo.name}")
                                    logger.info(f"      Combo: {combo.name}")
                                    future = executor.submit(process_combo, logger, island_name, island_dir, combo, records, pixel_lookup)
                                    future_map[future] = combo
                                for future in as_completed(future_map):
                                    combo = future_map[future]
                                    ensemble_metrics = future.result()
                                    if ensemble_metrics:
                                        prev = round_metrics.get(round_num)
                                        if (prev is None) or (ensemble_metrics.get("f1", 0.0) > prev.get("f1", 0.0)):
                                            round_metrics[round_num] = ensemble_metrics
                                    island_prog.advance(combo_task)
                        else:
                            for combo in combos:
                                island_prog.console.print(f"    • {combo.name}")
                                logger.info(f"      Combo: {combo.name}")
                                ensemble_metrics = process_combo(logger, island_name, island_dir, combo, records, pixel_lookup)
                                if ensemble_metrics:
                                    prev = round_metrics.get(round_num)
                                    if (prev is None) or (ensemble_metrics.get("f1", 0.0) > prev.get("f1", 0.0)):
                                        round_metrics[round_num] = ensemble_metrics
                                island_prog.advance(combo_task)
                        island_prog.remove_task(combo_task)
                        island_prog.advance(round_task)

                if round_metrics:
                    rows_sorted = sorted(round_metrics.items())
                    csv_path = island_summary_dir / "rounds_metrics.csv"
                    with csv_path.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["round", "precision", "recall", "f1", "accuracy", "auc_pr"])
                        for round_id, metrics in rows_sorted:
                            writer.writerow([round_id, metrics.get("precision", 0.0), metrics.get("recall", 0.0), metrics.get("f1", 0.0), metrics.get("accuracy", 0.0), metrics.get("auc_pr", 0.0)])
                    plot_round_metrics(island_summary_dir, island_summary_dir / "rounds_metrics.png")
                else:
                    logger.warn(f"    No ensemble metrics captured for {island_name}")
                year_prog.advance(year_task)

if __name__ == "__main__":
    main()
