#!/usr/bin/env python3
# Aggregate per-round metrics and plot lines for Precision, Recall, F1, Accuracy

import os
import csv
import json
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config as cfg


def _find_round_metric_files() -> List[Tuple[int, str]]:
    root = cfg.ROUNDS_DIR
    if not os.path.exists(root):
        return []
    items = []
    for name in os.listdir(root):
        if not name.startswith('round_'):
            continue
        try:
            rnum = int(name.split('_', 1)[1])
        except Exception:
            continue
        rdir = os.path.join(root, name)
        # search recursively for statistics/metrics_summary.csv; fall back to metrics.json
        candidates = []
        for dirpath, _dirnames, filenames in os.walk(rdir):
            if 'statistics' in dirpath:
                if 'metrics_summary.csv' in filenames:
                    fp = os.path.join(dirpath, 'metrics_summary.csv')
                    candidates.append(fp)
                elif 'metrics.json' in filenames:
                    fp = os.path.join(dirpath, 'metrics.json')
                    candidates.append(fp)
        if candidates:
            # choose the newest by mtime
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            items.append((rnum, candidates[0]))
    # keep unique round numbers (latest metrics for that round)
    dedup = {}
    for rnum, path in items:
        if rnum not in dedup:
            dedup[rnum] = path
        else:
            # keep the newer path
            if os.path.getmtime(path) > os.path.getmtime(dedup[rnum]):
                dedup[rnum] = path
    out = sorted([(r, p) for r, p in dedup.items()], key=lambda t: t[0])
    return out


def _read_metrics(path: str) -> Dict[str, float]:
    base = os.path.basename(path)
    if base == 'metrics_summary.csv':
        # already key-value pairs
        rows = {}
        with open(path, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                k = row.get('metric')
                v = row.get('value')
                if k is None:
                    continue
                try:
                    rows[k] = float(v)
                except Exception:
                    # skip non-numeric
                    pass
        return rows
    elif base == 'metrics.json':
        with open(path, 'r') as f:
            obj = json.load(f)
        out = {}
        for k, v in obj.items():
            try:
                out[k] = float(v)
            except Exception:
                pass
        return out
    else:
        return {}


def _write_aggregate_csv(rows: List[Dict[str, float]], rounds: List[int], out_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fields = ['round', 'precision', 'recall', 'f1', 'accuracy', 'auc_pr']
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for rnum, met in zip(rounds, rows):
            w.writerow([
                rnum,
                met.get('precision', ''),
                met.get('recall', ''),
                met.get('f1', ''),
                met.get('accuracy', ''),
                met.get('auc_pr', met.get('average_precision', met.get('average_precision_score', ''))),
            ])


def _plot_lines(rounds: List[int], rows: List[Dict[str, float]], out_png: str):
    if not rounds:
        return
    xs = rounds
    def g(key: str):
        return [r.get(key) if r.get(key) is not None else None for r in rows]
    P = g('precision')
    R = g('recall')
    F1 = g('f1')
    ACC = g('accuracy')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, P, '-o', label='Precision')
    ax.plot(xs, R, '-o', label='Recall')
    ax.plot(xs, F1, '-o', label='F1')
    ax.plot(xs, ACC, '-o', label='Accuracy')
    ax.set_xlabel('Round')
    ax.set_ylabel('Metric value')
    # Ensure integer x-axis ticks (1, 2, 3, ...), not floats
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Dynamic y-range: one-decimal below min to one-decimal above max across plotted metrics
    vals = []
    for arr in (P, R, F1, ACC):
        for v in arr:
            try:
                if v is not None:
                    vals.append(float(v))
            except Exception:
                pass
    if vals:
        vmin = min(vals)
        vmax = max(vals)
        import math as _m
        y0 = max(0.0, _m.floor(vmin * 10.0) / 10.0)
        y1 = min(1.0, _m.ceil(vmax * 10.0) / 10.0)
        if y1 <= y0:
            # Fallback to default when degenerate
            y0, y1 = 0.0, 1.0
        ax.set_ylim(y0, y1)
    else:
        ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    # Reserve 20% of figure width for legend on the right (80/20 layout)
    fig.subplots_adjust(right=0.80)
    # Create a single figure-level legend occupying the reserved area
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center left', bbox_to_anchor=(0.82, 0.5), borderaxespad=0.)
    # Tight layout within the left 80% so labels don't clash
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    items = _find_round_metric_files()
    if not items:
        print("No per-round metrics found under rounds/.")
        return
    rounds, paths = zip(*items)
    rows = [_read_metrics(p) for p in paths]
    agg_dir = cfg.ROUNDS_DIR
    _write_aggregate_csv(rows, list(rounds), os.path.join(agg_dir, 'rounds_metrics.csv'))
    _plot_lines(list(rounds), rows, os.path.join(agg_dir, 'rounds_metrics.png'))
    print(f"Aggregated rounds metrics => {os.path.join(agg_dir, 'rounds_metrics.csv')} and rounds_metrics.png")


if __name__ == '__main__':
    main()
