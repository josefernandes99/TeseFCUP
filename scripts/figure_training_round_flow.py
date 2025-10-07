#!/usr/bin/env python3
"""Generate Figure 4.2: Training round pipeline flowchart."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUTPUT_DIR = Path("writtenThesis/images")
PDF_PATH = OUTPUT_DIR / "figure_4_2.pdf"
PNG_PATH = OUTPUT_DIR / "figure_4_2.png"

FIGSIZE = (16, 9)
BACKGROUND = "#F7FBFF"
HEADER_COLOURS = [
    ("Configuration", "#0B3C5D"),
    ("Labels & Features", "#145374"),
    ("Training & Calibration", "#1F6F8B"),
    ("Inference & Scoring", "#2E8B57"),
    ("Archive & Exports", "#A66F00"),
]
COLUMN_X = [0.1, 0.3, 0.5, 0.7, 0.9]
BOX_WIDTH = 0.16
BOX_HEIGHT = 0.1
VERTICAL_SPACING = 0.14
START_Y = 0.87
OUTLINE_COLOUR = "#0B3C5D"
TEXT_COLOUR = "#1F2D3D"

STAGE_STEPS = [
    [
        ("Load configuration", "Read run config, resume checkpoint, freeze thread limits and GPU flags."),
        ("Initial sanity checks", "Confirm data directories, disk space, and cached artefacts before proceeding."),
    ],
    [
        ("Sync labels", "Merge labels.csv with temp_labels.csv, snap to tile grid, apply island filters."),
        ("Prepare features", "Validate cached stacks (npz) and compute missing tiles with current settings."),
    ],
    [
        ("Split & auto-tune", "Produce 70/30 stratified folds and score curated hyperparameters in-memory."),
        ("Train base models", "Fit SVM and RandomForest, record training diagnostics, keep best combination."),
        ("Calibrate ensemble", "Stack calibrated probabilities, apply isotonic calibration when folds permit."),
        ("Report validation", "Export metrics.json, ROC/PR curves, calibration plots, threshold sweep, runtime log."),
    ],
    [
        ("Stream tile inference", "Batch tiles through selected model to write _tile_preds/ shards (CSV/NPY)."),
        ("Merge predictions", "Combine shards into predictions.csv and optional probability rasters; drop temps."),
        ("Score candidates", "Compute uncertainty, DBSCAN diversity, hard-negative boosts; rank Highscore & ProbableAgri."),
        ("Polygonise (optional)", "Convert confident masks into single-style KML overlays when enabled."),
    ],
    [
        ("Persist round folder", "Update models/, statistics/, config_snapshot.json, runtime_metrics.csv, logs."),
        ("Publish exports", "Write candidate CSV/KML bundles and refresh Highscore/ProbableAgri listings."),
        ("Advisory artefacts", "Store best-threshold statistics/ overlays alongside baseline outputs."),
        ("Stage resume state", "Rewrite checkpoint.txt so the next launch resumes after inference with synced metadata."),
    ],
]


def wrap(text: str, width: int = 36) -> str:
    return "\n".join(textwrap.fill(part, width=width) for part in text.split("\n"))


def draw_header(ax, centre_x: float, title: str, colour: str) -> None:
    rect = Rectangle((centre_x - BOX_WIDTH / 2, 0.92), BOX_WIDTH, 0.06, facecolor=colour, edgecolor="none", zorder=0)
    ax.add_patch(rect)
    ax.text(centre_x, 0.95, title, ha="center", va="center", fontsize=13.5, fontweight="bold", color="white")


def draw_box(ax, centre_x: float, base_y: float, title: str, body: str) -> None:
    lower_left = (centre_x - BOX_WIDTH / 2, base_y)
    box = FancyBboxPatch(
        lower_left,
        BOX_WIDTH,
        BOX_HEIGHT,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=OUTLINE_COLOUR,
        facecolor="white",
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(centre_x, base_y + BOX_HEIGHT * 0.7, title, ha="center", va="center", fontsize=12.2, fontweight="bold", color=OUTLINE_COLOUR)
    ax.text(
        centre_x,
        base_y + BOX_HEIGHT * 0.32,
        wrap(body),
        ha="center",
        va="center",
        fontsize=10.2,
        color=TEXT_COLOUR,
    )


def connect_down(ax, centre_x: float, top_y: float) -> None:
    ax.annotate(
        "",
        xy=(centre_x, top_y - VERTICAL_SPACING + 0.02),
        xytext=(centre_x, top_y - 0.02),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color=OUTLINE_COLOUR),
        zorder=0,
    )


def connect_across(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color=OUTLINE_COLOUR),
        zorder=0,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Headers
    for (title, colour), centre_x in zip(HEADER_COLOURS, COLUMN_X):
        draw_header(ax, centre_x, title, colour)

    # Boxes per column
    column_bottom_y = []
    for centre_x, steps in zip(COLUMN_X, STAGE_STEPS):
        y = START_Y
        for idx, (title, body) in enumerate(steps):
            draw_box(ax, centre_x, y - BOX_HEIGHT, title, body)
            if idx < len(steps) - 1:
                connect_down(ax, centre_x, y - BOX_HEIGHT)
            y -= VERTICAL_SPACING
        column_bottom_y.append(y + VERTICAL_SPACING)

    # Cross-stage connectors (bottom of previous to top of next)
    for idx in range(len(COLUMN_X) - 1):
        start_x = COLUMN_X[idx] + BOX_WIDTH / 2
        end_x = COLUMN_X[idx + 1] - BOX_WIDTH / 2
        start_y = column_bottom_y[idx]
        end_y = START_Y
        connect_across(ax, (start_x, start_y), (end_x, end_y - 0.02))

    ax.text(
        0.5,
        0.05,
        wrap(
            "Each round loads explicit configuration, prepares labels and features, calibrates compact models, "
            "runs tile-level inference to refresh probabilities, ranks candidates for review, and persists artefacts "
            "so subsequent launches resume with identical state."
        , 80),
        ha="center",
        va="center",
        fontsize=11,
        color=TEXT_COLOUR,
    )

    fig.tight_layout(pad=0.6)
    for path in (PDF_PATH, PNG_PATH):
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=BACKGROUND)
    plt.close(fig)

    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {PNG_PATH}")


if __name__ == "__main__":
    main()
