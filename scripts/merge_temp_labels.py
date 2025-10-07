#!/usr/bin/env python3
"""Merge the latest temp_labels into labels.csv without dropping existing rows.

Usage
-----
python3 scripts/merge_temp_labels.py [--labels PATH] [--temp PATH] [--dry-run] [--clear-temp]

The script keeps existing entries in labels.csv, skips duplicates, and appends
any new labels found in the temporary file. Duplicates are identified by label
ID and by the (tile,row,col) key (falling back to tile+lat+lon when row/col are
missing)."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Iterable, Optional, Set

import config as cfg


DEFAULT_FIELDS = ["id", "lat", "lon", "tile", "label", "notes", "row", "col"]


def _normalise_row(row: Dict[str, str], fields: Iterable[str]) -> Dict[str, str]:
    """Return a copy restricted to the known header order."""
    out = {}
    for field in fields:
        value = row.get(field, "") if isinstance(row, dict) else ""
        out[field] = value.strip() if isinstance(value, str) else value
    return out


def _pixel_key(row: Dict[str, str]) -> Optional[str]:
    tile = row.get("tile")
    if not tile:
        return None
    row_idx = row.get("row")
    col_idx = row.get("col")
    if row_idx and col_idx:
        try:
            return f"{tile}:{int(float(row_idx))}:{int(float(col_idx))}"
        except ValueError:
            pass
    lat = row.get("lat")
    lon = row.get("lon")
    if lat and lon:
        try:
            return f"{tile}:{float(lat):.7f}:{float(lon):.7f}"
        except ValueError:
            return None
    return None


def _load_existing(path: str, fields: Iterable[str]) -> tuple[list[Dict[str, str]], Set[str], Set[str]]:
    rows: list[Dict[str, str]] = []
    ids: Set[str] = set()
    keys: Set[str] = set()
    if not os.path.exists(path):
        return rows, ids, keys
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            norm = _normalise_row(row, fields)
            rows.append(norm)
            rid = norm.get("id")
            if rid:
                ids.add(rid)
            pk = _pixel_key(norm)
            if pk:
                keys.add(pk)
    return rows, ids, keys


def _detect_fields(labels_path: str) -> list[str]:
    if not os.path.exists(labels_path):
        return list(DEFAULT_FIELDS)
    with open(labels_path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return list(DEFAULT_FIELDS)
        return header if header else list(DEFAULT_FIELDS)


def merge(labels_path: str, temp_path: str, dry_run: bool = False, clear_temp: bool = False) -> int:
    fields = _detect_fields(labels_path)
    existing_rows, existing_ids, existing_keys = _load_existing(labels_path, fields)
    if not os.path.exists(temp_path):
        print(f"Temp labels not found at {temp_path}; nothing to merge.")
        return 0

    with open(temp_path, newline="") as fh:
        reader = csv.DictReader(fh)
        incoming = [_normalise_row(row, fields) for row in reader if any(row.values())]

    added = []
    seen_ids = set(existing_ids)
    seen_keys = set(existing_keys)
    for row in incoming:
        row_id = row.get("id")
        if row_id and row_id in seen_ids:
            continue
        pk = _pixel_key(row)
        if pk and pk in seen_keys:
            continue
        if not row.get("label"):
            continue
        added.append(row)
        if row_id:
            seen_ids.add(row_id)
        if pk:
            seen_keys.add(pk)

    if not added:
        print("No new labels to merge; labels.csv unchanged.")
        return 0

    if dry_run:
        print(f"Dry run: {len(added)} rows would be appended to {labels_path}.")
        return len(added)

    write_header = not os.path.exists(labels_path) or os.stat(labels_path).st_size == 0
    with open(labels_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for row in added:
            writer.writerow(row)

    if clear_temp:
        os.remove(temp_path)
    print(f"Merged {len(added)} new labels into {labels_path}.")
    return len(added)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge temp_labels into labels.csv without overwriting existing entries.")
    parser.add_argument("--labels", default=cfg.LABELS_FILE, help="Path to labels.csv (default: config.LABELS_FILE)")
    parser.add_argument("--temp", default=cfg.TEMP_LABELS_FILE, help="Path to temp_labels.csv (default: config.TEMP_LABELS_FILE)")
    parser.add_argument("--dry-run", action="store_true", help="Show how many rows would be added without writing.")
    parser.add_argument("--clear-temp", action="store_true", help="Delete temp_labels.csv after a successful merge.")
    args = parser.parse_args()

    merge(args.labels, args.temp, dry_run=args.dry_run, clear_temp=args.clear_temp)


if __name__ == "__main__":
    main()
