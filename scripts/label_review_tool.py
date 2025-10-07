#!/usr/bin/env python3
"""Interactive label review and replacement helper.

This utility iterates over every label in the training and testing CSVs,
generates a per-pixel KML preview, prompts the reviewer to confirm or adjust
the class, and records any changes. Whenever an agricultural label is flipped
to non-agricultural, the tool tracks the loss per island and later guides the
user through adding the same number of replacement agricultural labels (with
snap-to-grid and duplicate checks) so class balance can be maintained.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import config as cfg

from a2_phase1_initial_labeling import (
    generate_kml_for_pixel,
    get_tile_for_coordinate,
)
from al_shared import snap_to_pixel_center


REVIEW_KML_PATH = os.path.join(cfg.LABELS_DIR, "review_candidate.kml")


class LabelTracker:
    """Track existing label locations to avoid duplicates."""

    def __init__(self) -> None:
        self._keys: set[Tuple[str, Tuple[int, int] | Tuple[float, float]]] = set()

    @staticmethod
    def _normalise_tile(tile: Optional[str]) -> Optional[str]:
        if not tile:
            return None
        return tile.strip().lower()

    @staticmethod
    def _try_int(value: Optional[str]) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _try_float(value: Optional[str]) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return round(float(value), 7)
        except (TypeError, ValueError):
            return None

    def _key_from_parts(
        self,
        tile: Optional[str],
        row: Optional[int] = None,
        col: Optional[int] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> Optional[Tuple[str, Tuple[int, int] | Tuple[float, float]]]:
        norm_tile = self._normalise_tile(tile)
        if not norm_tile:
            return None
        if row is not None and col is not None:
            return norm_tile, (int(row), int(col))
        if lat is not None and lon is not None:
            return norm_tile, (float(round(lat, 7)), float(round(lon, 7)))
        return None

    def key_from_row(self, row: Dict[str, str]) -> Optional[Tuple[str, Tuple[int, int] | Tuple[float, float]]]:
        row_idx = self._try_int(row.get("row")) or self._try_int(row.get("row_idx"))
        col_idx = self._try_int(row.get("col")) or self._try_int(row.get("col_idx"))
        if row_idx is not None and col_idx is not None:
            return self._key_from_parts(row.get("tile"), row=row_idx, col=col_idx)
        lat = self._try_float(row.get("lat"))
        lon = self._try_float(row.get("lon"))
        if lat is not None and lon is not None:
            return self._key_from_parts(row.get("tile"), lat=lat, lon=lon)
        return None

    def add_row(self, row: Dict[str, str]) -> None:
        key = self.key_from_row(row)
        if key:
            self._keys.add(key)

    def exists(self, tile: str, row: Optional[int] = None, col: Optional[int] = None,
               lat: Optional[float] = None, lon: Optional[float] = None) -> bool:
        key = self._key_from_parts(tile, row=row, col=col, lat=lat, lon=lon)
        return key in self._keys if key else False


def load_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing labels file: {path}")
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def write_csv(path: str, fieldnames: Iterable[str], rows: Iterable[Dict[str, str]]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp_path, path)


def prompt_label(current_label: str) -> Optional[str]:
    prompt = "Enter label ([1] Agri, [2] Non, [3] Skip, [4] Quit dataset) => "
    while True:
        choice = input(prompt).strip().lower()
        if choice in ("1", "a", "agri", "agricultural"):
            return "Agricultural"
        if choice in ("2", "n", "non", "non-agricultural", "nonagricultural"):
            return "Non-Agricultural"
        if choice in ("3", "s", "skip"):
            return None
        if choice in ("4", "q", "quit"):
            return "__quit__"
        print("Invalid option. Type A, N, S, or Q.")


def review_dataset(
    dataset_name: str,
    rows: List[Dict[str, str]],
) -> Dict[str, int]:
    """Iterate over labels, confirm/relabel, and return per-island agri losses."""

    losses: Dict[str, int] = defaultdict(int)
    total = len(rows)
    if total == 0:
        print(f"[{dataset_name}] No labels to review.")
        return losses

    print(f"[{dataset_name}] Reviewing {total} labels. KML previews written to {REVIEW_KML_PATH}.")
    for idx, row in enumerate(rows, start=1):
        tile = row.get("tile") or ""
        label = (row.get("label") or "").strip()
        island = cfg.extract_island_name(tile) or "unknown"

        row_idx = row.get("row") or row.get("row_idx")
        col_idx = row.get("col") or row.get("col_idx")
        lat_val = row.get("lat")
        lon_val = row.get("lon")

        # Ensure row/col exist by snapping if necessary.
        if not row_idx or not col_idx:
            try:
                snapped = snap_to_pixel_center(tile, float(lat_val), float(lon_val))
            except Exception:
                snapped = None
            if snapped:
                lat_val, lon_val, r_s, c_s = snapped
                row_idx, col_idx = str(int(r_s)), str(int(c_s))
                row["lat"], row["lon"] = f"{lat_val:.7f}", f"{lon_val:.7f}"
                row["row"], row["col"] = str(int(r_s)), str(int(c_s))

        if row_idx and col_idx:
            try:
                generate_kml_for_pixel(tile, int(float(row_idx)), int(float(col_idx)), out_path=REVIEW_KML_PATH)
            except Exception as exc:
                print(f"  [warn] Failed to generate KML for {tile} r={row_idx} c={col_idx}: {exc}")
        else:
            print(f"  [warn] Missing row/col for {tile}; KML skipped.")

        print(f"[{dataset_name}] {idx}/{total} › ID={row.get('id','')} | Tile={tile} | Island={island} | Current={label}")
        choice = prompt_label(label)
        if choice == "__quit__":
            print(f"[{dataset_name}] Review aborted by user.")
            break
        if choice is None or choice == label:
            continue

        row["label"] = choice
        if label.lower().startswith("agri") and not choice.lower().startswith("agri"):
            losses[island] += 1
        print(f"  Updated label => {choice}")

    return losses


def prompt_coordinate(message: str) -> Optional[Tuple[float, float]]:
    try:
        lat = float(input(f"{message} latitude => ").strip())
        lon = float(input(f"{message} longitude => ").strip())
        return lat, lon
    except ValueError:
        print("  Invalid coordinate. Try again.")
        return None


def add_replacement_labels(
    dataset_name: str,
    path: str,
    rows: List[Dict[str, str]],
    fieldnames: List[str],
    deficits: Dict[str, int],
    tracker: LabelTracker,
) -> None:
    if not deficits:
        return

    total_needed = sum(deficits.values())
    if total_needed == 0:
        return

    print(f"[{dataset_name}] Need {total_needed} replacement agricultural labels across {len(deficits)} island(s).")
    for island, needed in deficits.items():
        if needed <= 0:
            continue
        print(f"\n[{dataset_name}] Island {island} requires {needed} new agricultural label(s).")
        added = 0
        while added < needed:
            coord = prompt_coordinate("  Enter point for new label")
            if coord is None:
                continue
            lat, lon = coord
            tile = get_tile_for_coordinate(lat, lon)
            if not tile:
                print("  No tile found for this coordinate; try again.")
                continue
            tile_island = cfg.extract_island_name(tile) or ""
            if tile_island.lower() != island.lower():
                print(f"  Coordinate maps to island '{tile_island}', expected '{island}'. Try again.")
                continue
            snapped = snap_to_pixel_center(tile, lat, lon)
            if not snapped:
                print("  Could not snap to pixel centre; try a nearby point.")
                continue
            snap_lat, snap_lon, row_idx, col_idx = snapped
            if tracker.exists(tile, row=row_idx, col=col_idx):
                print("  Duplicate pixel detected; choose a different location.")
                continue
            try:
                generate_kml_for_pixel(tile, int(row_idx), int(col_idx), out_path=REVIEW_KML_PATH)
            except Exception as exc:
                print(f"  [warn] Failed to generate KML preview: {exc}")
            confirm = input("  Confirm add as Agricultural? [Y/N] => ").strip().lower()
            if confirm not in ("y", "yes"):
                print("  Skipped.")
                continue

            new_id = f"manualrev_{random.randint(100000, 999999)}"
            new_row = {key: "" for key in fieldnames}
            if "id" in new_row:
                new_row["id"] = new_id
            if "lat" in new_row:
                new_row["lat"] = f"{snap_lat:.7f}"
            if "lon" in new_row:
                new_row["lon"] = f"{snap_lon:.7f}"
            if "tile" in new_row:
                new_row["tile"] = tile
            if "label" in new_row:
                new_row["label"] = "Agricultural"
            if "notes" in new_row:
                new_row["notes"] = ""
            if "row" in new_row:
                new_row["row"] = str(int(row_idx))
            if "col" in new_row:
                new_row["col"] = str(int(col_idx))
            if "row_idx" in new_row:
                new_row["row_idx"] = str(int(row_idx))
            if "col_idx" in new_row:
                new_row["col_idx"] = str(int(col_idx))
            rows.append(new_row)
            tracker.add_row(new_row)
            added += 1
            print(f"  Added Agricultural label at ({snap_lat:.7f}, {snap_lon:.7f}) tile={tile} r={int(row_idx)} c={int(col_idx)}")

        print(f"[{dataset_name}] Completed replacements for island {island}.")

    write_csv(path, fieldnames, rows)
    print(f"[{dataset_name}] Wrote updated file with replacements => {path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Review and refresh training/testing labels with KML previews.")
    parser.add_argument("--skip-training", action="store_true", help="Skip reviewing the trainingLabels.csv file")
    parser.add_argument("--skip-testing", action="store_true", help="Skip reviewing the testingLabels.csv file")
    args = parser.parse_args(argv)

    datasets: List[Tuple[str, str]] = []
    if not args.skip_training:
        datasets.append(("training", cfg.TRAINING_LABELS_FILE))
    if not args.skip_testing:
        datasets.append(("testing", cfg.TESTING_LABELS_FILE))

    if not datasets:
        print("Nothing to do; all datasets skipped.")
        return 0

    fieldnames_map: Dict[str, List[str]] = {}
    rows_map: Dict[str, List[Dict[str, str]]] = {}

    for name, path in datasets:
        fieldnames, rows = load_csv(path)
        fieldnames_map[name] = fieldnames
        rows_map[name] = rows

    losses_map: Dict[str, Dict[str, int]] = {}
    for name, path in datasets:
        losses = review_dataset(name, rows_map[name])
        losses_map[name] = losses
        write_csv(path, fieldnames_map[name], rows_map[name])
        print(f"[{name}] Review complete. File updated => {path}")

    tracker = LabelTracker()
    for rows in rows_map.values():
        for row in rows:
            tracker.add_row(row)

    for name, path in datasets:
        deficits = {island: count for island, count in losses_map.get(name, {}).items() if count > 0}
        if deficits:
            add_replacement_labels(name, path, rows_map[name], fieldnames_map[name], deficits, tracker)
        else:
            print(f"[{name}] No agricultural losses recorded; no replacements required.")

    print("All done. Remember to rerun downstream evaluations if required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
