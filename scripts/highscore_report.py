#!/usr/bin/env python3
import csv
import os
import sys

from config import HIGHSCORE_FILE, PROBABLE_AGRI_FILE
import config as cfg


def preview_csv(path, top=20):
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return
    with open(path) as f:
        rows = list(csv.DictReader(f))
    print(f"\nTop {min(top, len(rows))} of {len(rows)} from {path}:")
    for i, r in enumerate(rows[:top], 1):
        print(f"{i:3d}. tile={r.get('tile')} rc=({r.get('row')},{r.get('col')}) lat={r.get('lat')} lon={r.get('lon')} score={r.get('score','')} prob={r.get('prob','')}")


def main():
    hs_on = bool(getattr(cfg, 'HIGHSCORE_LIST_ENABLED', True))
    pa_on = bool(getattr(cfg, 'PROBABLE_AGRI_LIST_ENABLED', False))
    if not (hs_on or pa_on):
        print("Persistent lists disabled in config; report skipped.")
        return
    top = 20
    if len(sys.argv) > 1:
        try:
            top = int(sys.argv[1])
        except Exception:
            pass
    if hs_on:
        preview_csv(HIGHSCORE_FILE, top)
    if pa_on:
        preview_csv(PROBABLE_AGRI_FILE, top)


if __name__ == "__main__":
    main()
