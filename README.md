# CropSegProject

**CropSegProject** provides a streamlined pipeline to detect potential agricultural areas using multi-temporal Sentinel-2 data from Google Earth Engine.  The current code focuses on an iterative active‑learning approach where you label a few pixels, train a simple model, and progressively improve the predictions.

The repository also contains an older, experimental segmentation workflow under `scripts/backup (old)` which is kept for reference.

## Features

- Automated download of Sentinel‑2 tiles covering your region of interest
- Interactive labeling of image patches to build an initial training set
- Active learning loop that trains a model (ResNet, SVM or RandomForest) and suggests new candidate patches for labeling
- Post‑processing step that classifies all tiles and produces cleaned overlay rasters along with CSV and text summaries
- Automatic memory watcher that frees unused resources during processing
## Repository Layout

```
TeseFCUP/
├── data/
│   └── phase1/
│       └── raw/         # Downloaded Sentinel-2 tiles
├── labels/
│   └── phase1/          # CSV file of labeled points
├── scripts/
│   ├── a0_setup_check.py
│   ├── a1_phase1_data_download.py
│   ├── a2_phase1_initial_labeling.py
│   ├── a3_phase1_active_learning_round.py
│   ├── a4_phase1_active_learning_loop.py
│   ├── a6_phase1_postprocessing.py
│   └── config.py
│   └── backup (old)/    # Archived phase2/phase3 code
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository and install the dependencies (Python ≥3.9):

```bash
pip install -r requirements.txt
```

2. Authenticate with Google Earth Engine:

```bash
earthengine authenticate
```

## Quick Start

Run the end‑to‑end phase 1 pipeline:

```bash
python scripts/ready_to_run_phase1.py
```

The script performs the following steps:

1. **Setup check** – ensures your Earth Engine credentials work and cleans leftovers from previous runs.
2. **Data download** – exports Sentinel‑2 tiles for the configured islands (see `scripts/a1_phase1_data_download.py`).
3. **Initial labeling** – guides you through labeling a handful of pixels until a balanced set is reached.
4. **Active learning** – repeatedly trains a model and proposes uncertain patches for further labeling.
5. **Post‑processing** – uses the final model to classify each tile and saves overlay rasters plus a CSV and text summary with overall statistics.
All intermediate results are stored under `data/phase1` and `labels/phase1`.

If the pipeline is interrupted, a checkpoint file is written under
`data/phase1/checkpoint.txt`. Running `ready_to_run_phase1.py` again will prompt
you to resume from that step (and round if mid-way through active learning) or
start from the beginning. The checkpoint is removed automatically once the
pipeline finishes.

## Notes

- The archived `scripts/backup (old)` folder contains a previous, more complex pipeline that handled high‑resolution imagery and segmentation. It is not actively maintained but may be useful for reference.
- The repository comes with a small example tile in `data/phase1/raw/` so you can try the code without downloading large datasets first.