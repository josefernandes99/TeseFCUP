#!/usr/bin/env python3
# scripts/generatePersistents.py

import os
import sys
import glob
import shutil
from typing import List, Tuple

import joblib
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

import config as cfg
from splits import load_labels
from al_shared import extract_features_from_label
from a3_phase1_active_learning_round import (
    predict_entire_tile,
    _write_tile_predictions_csv,
    _merge_tile_prediction_csvs,
    refresh_global_lists_full,
)
from progress_utils import new_progress


def _round_dirs() -> List[Tuple[int, str]]:
    base = cfg.ROUNDS_DIR
    out = []
    for p in glob.glob(os.path.join(base, "round_*")):
        name = os.path.basename(p)
        if not os.path.isdir(p):
            continue
        if name == "final_round":
            continue
        try:
            rnum = int(name.split("_")[-1])
        except Exception:
            continue
        out.append((rnum, p))
    out.sort(key=lambda t: t[0])
    return out


def _find_model_path(rdir: str) -> str:
    cand = glob.glob(os.path.join(rdir, "**", "model_round_*.pkl"), recursive=True)
    if not cand:
        return ""
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def _rebuild_predictions_for_round(round_num: int, round_dir: str, model_path: str, console: Console) -> str:
    # Fast path: reuse existing per-tile shards if present
    tile_dir = os.path.join(round_dir, "_tile_preds")
    if os.path.isdir(tile_dir):
        shards = sorted([p for p in glob.glob(os.path.join(tile_dir, '*.csv'))])
        if shards:
            console.print(f"[cyan]Reusing existing per-tile shards ({len(shards)} files).[/cyan]")
            merged = _merge_tile_prediction_csvs(round_dir)
            return merged or ""

    model = joblib.load(model_path)
    # Inference across tiles (streamed per tile to CSV shards)
    tifs = [tp for tp in cfg.list_raw_tiles()
            if ("_th" not in os.path.basename(tp))]
    if not tifs:
        console.print("[yellow]No raw tiles found; skipping predictions.[/yellow]")
        return ""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    max_workers = int(getattr(cfg, 'INFER_TILE_THREADS', 4)) or 4
    with Progress("[bold cyan]{task.description}", BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(), transient=True) as prog:
        task = prog.add_task(f"Round {round_num}: inference on tiles", total=len(tifs))
        def run_tile(tp):
            preds = predict_entire_tile(tp, model)
            tile_name = os.path.basename(tp)
            _ = _write_tile_predictions_csv(round_dir, tile_name, preds)
            return tile_name
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_tile, tp) for tp in tifs]
            for _ in as_completed(futs):
                prog.update(task, advance=1)
    merged = _merge_tile_prediction_csvs(round_dir)
    return merged or ""


def _collect_training_rows(console: Console):
    rows = []
    if os.path.exists(cfg.LABELS_FILE):
        rows.extend(load_labels(cfg.LABELS_FILE))
    if os.path.exists(cfg.TEMP_LABELS_FILE):
        rows.extend(load_labels(cfg.TEMP_LABELS_FILE))
    # dedup simple on (tile,lat,lon)
    seen = set(); out = []
    for r in rows:
        k = f"{r.get('tile')}:{r.get('lat')}:{r.get('lon')}"
        if k in seen: continue
        seen.add(k); out.append(r)
    console.print(f"Collected [green]{len(out)}[/green] training rows (labels + temp).")
    return out


def _build_feature_matrix_with_progress(rows):
    import numpy as _np
    if not rows:
        return _np.empty((0,)), _np.empty((0,))
    X, y = [], []
    with Progress("[bold cyan]{task.description}", BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(), transient=True) as prog:
        t = prog.add_task("Extract training features", total=len(rows))
        for r in rows:
            try:
                f = extract_features_from_label(r)
            except Exception:
                f = None
            if f is not None:
                X.append(f)
                y.append(1 if (r.get("label","" ).lower()=="agricultural") else 0)
            prog.update(t, advance=1)
    if not X:
        return _np.empty((0,)), _np.empty((0,))
    return _np.array(X, dtype=_np.float32), _np.array(y, dtype=_np.int64)


def _maybe_delete(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def main():
    console = Console()
    console.print(Panel.fit("Persistent Lists Rebuilder", subtitle="Highscore / ProbableAgri", border_style="cyan"))
    # Menu (print each line explicitly to avoid buffering quirks)
    console.print("Select what to generate:")
    console.print("  1) Highscore")
    console.print("  2) ProbableAgri")
    console.print("  3) Both")
    choice = input("=> ").strip()
    do_hs = (choice == "1" or choice == "3")
    do_pa = (choice == "2" or choice == "3")
    if not (do_hs or do_pa):
        console.print("[yellow]Nothing selected; exiting.[/yellow]")
        sys.exit(0)

    # Non-destructive by default: do NOT delete existing persistent lists.
    # If a reset is truly desired, the user can delete files manually.
    # We keep KMLs too; they will be regenerated later from merged CSVs.

    rounds = _round_dirs()
    if not rounds:
        console.print("[yellow]No round_* folders found; nothing to do.[/yellow]")
        return

    rows = _collect_training_rows(console)
    X, y = _build_feature_matrix_with_progress(rows)
    if getattr(X, 'size', 0) == 0:
        console.print("[yellow]No usable training features; continuing without representativeness.[/yellow]")

    # Overall progress across rounds
    with Progress(
        "[bold cyan]{task.description}",
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(), transient=False
    ) as prog:
        task = prog.add_task("Process rounds", total=len(rounds))
        for rnum, rdir in rounds:
            console.rule(f"Round {rnum}")
            mp = _find_model_path(rdir)
            if not mp:
                console.print(f"[yellow]No model found in {rdir}; skipping.[/yellow]")
                prog.update(task, advance=1)
                continue
            console.print(f"Model: [green]{mp}[/green]")
            pred_csv = _rebuild_predictions_for_round(rnum, rdir, mp, console)
            if not pred_csv:
                console.print(f"[yellow]predictions.csv not generated; skipping refresh.[/yellow]")
                prog.update(task, advance=1)
                continue
            console.print(f"predictions.csv => [green]{pred_csv}[/green]")
            # Run global refresh constrained by toggles
            hs_save = bool(getattr(cfg, 'HIGHSCORE_LIST_ENABLED', True))
            pa_save = bool(getattr(cfg, 'PROBABLE_AGRI_LIST_ENABLED', False))
            setattr(cfg, 'HIGHSCORE_LIST_ENABLED', do_hs)
            setattr(cfg, 'PROBABLE_AGRI_LIST_ENABLED', do_pa)
            try:
                console.print("[cyan]Refreshing persistent lists...[/cyan]")
                refresh_global_lists_full(pred_csv=pred_csv,
                                          round_dir=rdir,
                                          round_num=rnum,
                                          train_rows=rows,
                                          X_train=X,
                                          y_train=y)
            finally:
                setattr(cfg, 'HIGHSCORE_LIST_ENABLED', hs_save)
                setattr(cfg, 'PROBABLE_AGRI_LIST_ENABLED', pa_save)
            console.print("[cyan]Cleaning intermediates...[/cyan]")
            _maybe_delete(pred_csv)
            _maybe_delete(os.path.join(rdir, "_tile_preds"))
            _maybe_delete(os.path.join(rdir, "_global_refresh"))
            prog.update(task, advance=1)

    # Summary table
    tbl = Table(title="Persistent Lists Summary", show_lines=False)
    tbl.add_column("List")
    tbl.add_column("Path")
    if do_hs:
        tbl.add_row("Highscore CSV", cfg.HIGHSCORE_FILE)
        tbl.add_row("Highscore KML", cfg.HIGHSCORE_KML_GLOBAL)
    if do_pa:
        tbl.add_row("ProbableAgri CSV", cfg.PROBABLE_AGRI_FILE)
        tbl.add_row("ProbableAgri KML", cfg.PROBABLE_AGRI_KML_GLOBAL)
    console.print(tbl)
    console.print(Panel.fit("Done. Persistent lists are under labels/phase1/", border_style="green"))


if __name__ == "__main__":
    main()
