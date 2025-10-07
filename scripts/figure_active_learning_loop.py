#!/usr/bin/env python3
"""Generate the active-learning loop diagram used in writtenThesis/images.

The script recreates the circular flow requested for Figure 3.3. It only depends on
Matplotlib, so run it from an environment where Matplotlib is installed, e.g. the
local `.venv` created for figure generation.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath

# Output locations (relative to the repository root)
OUTPUT_DIR = Path("writtenThesis/images")
PDF_PATH = OUTPUT_DIR / "active_learning_loop.pdf"
PNG_PATH = OUTPUT_DIR / "active_learning_loop.png"

# Diagram configuration
OUTER_RADIUS = 1.0
INNER_RADIUS = 0.78  # thinner arrows so the inner text fits comfortably
SEGMENT_ANGLE = 90
ARROW_TIP_ANGLE = 24
BASE_ANGLE = 135  # start in the North-West quadrant and proceed clockwise
COLORS = ["#0B3C5D", "#1876C1", "#0B3C5D", "#1876C1"]
STEPS = [
    {
        "direction": "North-West",
        "title": "Train & calibrate ensemble",
        "description": (
            "Update the SVM, RandomForest, and their calibrated ensemble with the "
            "latest labels before scoring."
        ),
    },
    {
        "direction": "North-East",
        "title": "Infer & rank candidates",
        "description": (
            "Immediately after training, run tile inference to refresh probabilities, "
            "then rank pixels by uncertainty and spatial diversity."
        ),
    },
    {
        "direction": "South-East",
        "title": "Label review",
        "description": (
            "Users accept, reject, or skip each pixel through the CLI while inspecting "
            "matching KML context."
        ),
    },
    {
        "direction": "South-West",
        "title": "Persist state",
        "description": (
            "Archive models, ensemble checkpoints, probability shards, and updated "
            "CSVs before the next round."
        ),
    },
]


def polar_point(angle_deg: float, radius: float) -> tuple[float, float]:
    """Convert polar coordinates to Cartesian coordinates."""
    radians = math.radians(angle_deg)
    return radius * math.cos(radians), radius * math.sin(radians)


def add_arrow(ax, start_angle: float, color: str) -> None:
    """Add a filled annular arrow segment with a pronounced head."""
    base_angle = start_angle + SEGMENT_ANGLE - ARROW_TIP_ANGLE
    tip_angle = start_angle + SEGMENT_ANGLE

    outer_angles = [start_angle + (base_angle - start_angle) * t / 70 for t in range(71)]
    inner_angles = [base_angle - (base_angle - start_angle) * t / 70 for t in range(71)]

    outer_points = [polar_point(angle, OUTER_RADIUS) for angle in outer_angles]
    tip_outer = polar_point(tip_angle, OUTER_RADIUS + 0.12)
    tip_inner = polar_point(tip_angle, INNER_RADIUS + 0.02)
    inner_points = [polar_point(angle, INNER_RADIUS) for angle in inner_angles]

    vertices = outer_points + [tip_outer, tip_inner] + inner_points + [outer_points[0]]
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(vertices) - 2) + [MplPath.CLOSEPOLY]

    patch = PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", zorder=1)
    ax.add_patch(patch)


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")
ax.axis("off")

for index, step in enumerate(STEPS):
    start_angle = BASE_ANGLE - index * SEGMENT_ANGLE
    add_arrow(ax, start_angle, COLORS[index % len(COLORS)])

center_circle = Circle((0, 0), 0.52, facecolor="#F4F8FC", edgecolor="#0B3C5D", linewidth=1.2, zorder=2)
ax.add_patch(center_circle)
ax.text(0, 0.1, "Active Learning\nRound", ha="center", va="center", fontsize=17, fontweight="bold", color="#0B3C5D")
ax.text(0, -0.18, "Phase 1 pipeline loop", ha="center", va="center", fontsize=11, color="#1f4e79")

connector_radius = OUTER_RADIUS + 0.1
text_radius = OUTER_RADIUS + 0.4

for index, step in enumerate(STEPS):
    start_angle = BASE_ANGLE - index * SEGMENT_ANGLE
    mid_angle = start_angle + SEGMENT_ANGLE / 2

    line_start = polar_point(mid_angle, connector_radius)
    line_end = polar_point(mid_angle, text_radius - 0.08)
    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], color="#1F4E79", linewidth=1.1)

    text_x, text_y = polar_point(mid_angle, text_radius)
    angle_mod = (mid_angle + 360) % 360

    if angle_mod >= 315 or angle_mod <= 45:
        ha, va, dx, dy = "left", "center", 0.05, -0.02
    elif 135 <= angle_mod <= 225:
        ha, va, dx, dy = "right", "center", -0.05, -0.02
    elif 45 < angle_mod < 135:
        ha, va, dx, dy = "center", "bottom", 0.0, 0.06
    else:
        ha, va, dx, dy = "center", "top", 0.0, -0.08

    heading = f"{step['direction']}: {step['title']}"
    ax.text(text_x + dx, text_y + dy + 0.06, heading, ha=ha, va="bottom", fontsize=13, fontweight="bold", color="#0B3C5D")
    ax.text(
        text_x + dx,
        text_y + dy - 0.02,
        step["description"],
        ha=ha,
        va="top",
        fontsize=11,
        color="#1A5276",
        wrap=True,
    )

ax.set_xlim(-1.65, 1.65)
ax.set_ylim(-1.65, 1.65)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PDF_PATH, bbox_inches="tight", facecolor="white")
fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"Wrote {PDF_PATH}")
print(f"Wrote {PNG_PATH}")
