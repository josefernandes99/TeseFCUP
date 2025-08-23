# Repository Guidelines

This guide aligns contributors on structure, tooling, and workflow for this project (phase‑1 active learning over Sentinel‑2) and reflects the current implementation and outputs.

## Project Structure & Module Organization
- `scripts/`: main pipeline modules (`a0_...` → `a6_...`, `ready_to_run_phase1.py`, `config.py`). Key roles:
  - `a3_phase1_active_learning_round.py`: trains a model, runs inference across tiles, updates AL artifacts, evaluates, and optionally prompts interactive labeling for selected candidates.
  - `a4_phase1_active_learning_loop.py`: orchestrates multiple AL rounds; supports grid search or manual hyper‑params; resume‑aware.
  - `a6_phase1_postprocessing.py`: runs final classification with the last round model; writes overlay rasters and a final summary.
  - `evaluation.py`: builds a validation split from labels; writes `statistics/` artifacts.
  - `al_shared.py`, `features.py`, `grid_search.py`, `splits.py`: shared utilities for features, search, and data splits.
- `data/phase1/`: working artifacts (tiles, checkpoints, outputs).
- `labels/phase1/`: labeled points and metadata.
- `scripts/backup (old)/`: archived experiments; do not modify.

## Project Architecture (Compact)
```
.
├─ scripts/
│  ├─ backup (old)/             # Archived experiments (read-only)
│  ├─ ready_to_run_phase1.py    # Orchestrator; resume support
│  ├─ a0_setup_check.py         # Env + EE auth validation
│  ├─ a1_phase1_data_download.py# Fetch Sentinel‑2 tiles
│  ├─ a2_phase1_initial_labeling.py  # Seed labels interactively
│  ├─ a3_phase1_active_learning_round.py  # One AL iteration
│  ├─ a4_phase1_active_learning_loop.py   # Iterative AL loop
│  ├─ a6_phase1_postprocessing.py    # Final classify + reports
│  ├─ al_shared.py              # Shared AL utilities (sampling, I/O)
│  ├─ evaluation.py             # Metrics and report generation
│  ├─ features.py               # Feature engineering for pixels/patches
│  ├─ grid_search.py            # SVM/RF hyperparameter tuning helpers
│  ├─ memory_watcher.py         # RAM monitor/cleanup helpers
│  └─ config.py                 # Paths, ROI, model parameters
├─ data/
│  └─ phase1/
│     ├─ raw/                  # Input tiles (*.tif)
│     ├─ rounds/               # Per-round, per-hyperparam folders with `statistics/` and `temporary/`
│     └─ checkpoint.txt        # Pipeline resume marker
├─ labels/
│  └─ phase1/
│     ├─ labels.csv            # Training labels (+ optional notes)
│     ├─ temp_labels.csv       # Working labels accumulated across AL rounds
│     ├─ labels.kml            # Quick map preview (training)
│     ├─ grids/                # Sampling grids per ROI (optional)
│     ├─ probableAgri.csv      # Assisted labeling list (optional, auto-maintained)
│     ├─ probableAgri_top.kml  # Global KML preview for probable-agri list
│     ├─ highscore.csv         # Persistent AL highscore (top‑K by entropy)
│     └─ highscore_top.kml     # Global KML preview for highscore pixels
├─ README.md                   # Usage and layout
├─ requirements.txt            # Python dependencies
├─ package.json                # Dev tooling (Codex)
├─ node_modules/               # Local JS deps (dev-only)
├─ .venv/                      # Optional local venv
├─ .idea/                      # IDE metadata
└─ .gitattributes              # Git settings
```

Usability notes:
- `ready_to_run_phase1.py`: preferred entrypoint; runs full pipeline with resume.
- `a0`→`a6`: see comments above for when to run each; `a3` is best for rapid debugging.
- `data/phase1/raw`: keep read‑only; seed with small tiles for tests.
- `data/phase1/rounds`: contains per‑round results under `<round>/<hyperparam>/statistics/` (classification report, metrics.json, ROC/CM plots) and derived KML/CSVs.
— `labels/phase1/labels.csv`: authoritative ground truth; back it up.

## Active Learning (AL)
- Candidate selection: entropy scoring with haversine DBSCAN clustering and tile‑balanced round‑robin; optional hard‑negative quota (look‑alikes) to reduce false positives.
- Assisted labeling (a2):
  - “Highscore pixels” feeds from `data/phase1/highscore.csv` (top‑K by entropy).
  - “Probable agri” feeds from `labels/phase1/probableAgri.csv` (top‑K by probability, de‑duped vs labels and highscore). Optional.
  - Initial labeling menu includes quick review options (Highscore/ProbableAgri) to label top‑N entries; labeled items are removed from all persistence lists and temp labels.
- Core knobs (config.py):
  - `NUM_CANDIDATES_PER_ROUND`, `CANDIDATE_DBSCAN_EPS_KM` (km)
  - `CANDIDATE_NEGATIVE_QUOTA`, `NEG_LIKE_PROB_RANGE`, `NEG_LIKE_NDVI_RANGE`
  - `CANDIDATE_PROB_LOWER` (candidate band), `MIN_AGRI_PROB` (decision threshold)

## Evaluation & Reports
- Source: union of `labels.csv` + `temp_labels.csv` (de‑duplicated) with validation splits from `splits.py`.
- Repeated validation (if enabled via `VAL_REPEATS > 1`):
  - `metrics.json` contains mean metrics across repeats plus `std_*` fields.
  - `metrics_repeats.json` stores per‑repeat metrics; `split_seeds.json` records the seeds used.
  - `classification_report.txt`, `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png` correspond to the first split for interpretability.
  - `validation_split_rows.csv` lists rows included in the first validation split.
- Note: older runs may have files at the hyperparam folder root; current code writes under `statistics/`.

## Models & Features
- SVM with calibration via `CalibratedClassifierCV` (`CALIBRATION_METHOD` = `sigmoid`/`isotonic`, `CALIBRATION_FOLDS`).
- RandomForest and a lightweight tabular ResNet also supported.
- Feature sets (`config.py`):
  - `FEATURE_SET` menu exposes `base`, `temporal_only`, `textures_only`, `temporal_textures`, `full`.
  - Current implementation supports only `base`; other options are marked “currently unsupported” in the menu until their schemas are implemented.
- Robust feature naming: names are derived from the on‑disk stacked raster schema (seasons + terrain). `ASPECT` is replaced by `ASPECT_SIN` and `ASPECT_COS`. This guarantees feature‑importance labels match model inputs.

## Grid Search
- Entry: `python scripts/grid_search.py` or via `ready_to_run_phase1.py` (choose “grid search”).
- Outputs: `data/phase1/rounds/grid_<MODEL>_summary/results.json` (+ optional KML).
- Uses `StratifiedKFold` over labels; reports F1/accuracy/AUC; optional permutation importance.

## Postprocessing
- Final classify uses the last round’s model; writes per‑tile overlays under:
  - `data/phase1/rounds/final_round/<combo>/overlays/*_overlay.tif`
- Per‑combo statistics under `data/phase1/rounds/final_round/<combo>/statistics/`:
  - `classification_report.txt`, `metrics.json`, `metrics_summary.csv`
  - `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`
  - `feature_importance.txt` and `feature_importance.png`
- Per‑combo artifacts:
  - `agricultural_patches_final_<combo>.kml`, `config_snapshot.json`, `final_summary.txt`
- Overall comparison:
  - `data/phase1/rounds/final_round/summary/final_comparison.csv`
  - `data/phase1/rounds/final_round/summary/best_choice.txt`
  - Best chosen by macro‑F1 (tie‑break AUC).

## Build, Test, and Development Commands
- Create env and install: `pip install -r requirements.txt` (Python 3.12).
- Earth Engine auth: `earthengine authenticate`.
- Run end‑to‑end: `python scripts/ready_to_run_phase1.py`.
- Run a single step (example): `python scripts/a3_phase1_active_learning_round.py`.
- Optional format (if available): `black scripts` and `isort scripts`.
- Note: running the pipeline can take time. For quick validation, limit to 1 round.
- If you need to test multi‑round labeling, ask the user to perform the interactions; state the exact steps and goals.

## Coding Style & Naming Conventions
- PEP 8, 4‑space indentation; line length 88 if using Black.
- Names: files/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Configuration in `scripts/config.py`; avoid hard‑coded paths; prefer `data/phase1` and `labels/phase1`.
- Prefer small, pure helpers in `scripts/*.py`; keep I/O and Earth Engine calls isolated.

## Testing Guidelines
- No formal suite yet; use smoke tests:
  - `python scripts/a0_setup_check.py` to verify env and EE auth.
  - Use only the `.tif` files inside `data/phase1/raw` as data and `labels/phase1/labels.csv` as annotated pixels.
- If adding tests, place `tests/test_*.py` with `pytest`; target helpers with mocks; aim for >80% on new code.

## Run Modes & Resume
- Preferred entrypoint: `ready_to_run_phase1.py` with checkpoint resume.
- AL loop supports a numeric round count or `infinite`.
  - Infinite mode: per round, the pipeline trains and predicts first, then asks whether to proceed with candidate labeling for that round; it does not ask again after labeling. If you answer “N” before labeling, the loop ends cleanly.

## Commit & Pull Request Guidelines
- Zero permission to do commits, pushes, and pulls. If needed, ask the user; assume local‑only work.

## Security & Configuration Tips
- Never commit credentials, tokens, or large rasters; keep them ignored.
- Review `scripts/config.py` before runs; quotas apply to Earth Engine exports.
- Long jobs: monitor memory via `scripts/memory_watcher.py` when experimenting.

## Console Logs
- All pipeline console logs are mirrored to `data/phase1/consoleLogs.txt`.
- When you need to check the latest run’s logs, open that file directly. No need to scroll back in the terminal.
- Progress bars use Rich and remain interactive in the terminal while also logging to the file.
- Live line updates in the terminal (carriage‑return) are persisted as full sequential lines in `consoleLogs.txt` for post‑hoc analysis (CR→NL conversion in the log file).

## Implementation Notes (Current)
- Logging tee: `ready_to_run_phase1.py` tees stdout/stderr to `data/phase1/consoleLogs.txt`. Carriage returns used for live progress are translated to newlines in the log, so each live update is preserved as a separate line. A signal handler ensures the log file is flushed/closed on SIGINT/SIGTERM.
- Signal handling: Signals are registered via a typed handler (`_sig_handler(signum, frame)`), and registration is wrapped with `contextlib.suppress` for environments that disallow signal registration.
- Feature cache format: Per‑tile feature arrays are cached under `data/phase1/cache/<tile>.npz` with keys: `arr` (np.ndarray, features), `transform` (first 6 coefficients of the affine), and `crs_wkt` (WKT string). On load, the transform is reconstructed via `affine.Affine(*transform)` and the CRS via `rasterio.crs.CRS.from_wkt(wkt=crs_wkt)`.
- Feature names alignment: `features.current_feature_names()` reads a tile to infer exact feature names based on channel count (seasons + terrain). `ASPECT` is replaced by `ASPECT_SIN`/`ASPECT_COS` deterministically, keeping model inputs and importance labels aligned.
- NDVI reporting: a representative seasonal NDVI (prefers `NDVI_s2`) is used for predictions CSVs and candidate justifications to keep reporting consistent across tiles.
- Postprocessing sweep: Final classification supports a sweep over thresholds/sieve/morphology defined in `config.py` (`FINAL_THRESHOLDS`, `FINAL_SIEVE_SIZES`, `FINAL_MORPH_*`). Each combo is tagged like `th0.45_s5` plus `_m3` if morphology applies. Overlays are written to `data/phase1/rounds/final_round/<combo>/overlays/`. Per‑combo stats are saved under `statistics/`, a concise `final_summary.txt` is produced, and `config_snapshot.json` records the settings. Optional permutation importance with named features is produced when enabled.
- Final summaries: After building per‑combo metrics, the pipeline writes `final_round/summary/final_comparison.csv` aggregating macro‑F1, F1, accuracy, AUC, AUC‑PR, and overall map percentage. `best_choice.txt` records the best combo (macro‑F1, tie‑break AUC).
- AL loop (infinite mode): Each round performs train/infer/eval, then prompts exactly once whether to proceed to candidate labeling. If declined, the loop ends immediately. Candidate selection can be driven from the just‑generated `predictions.csv`.
- Type/lint stability: Common issues addressed include casting sieve `size` to `int`, ensuring figure dimensions are floats (`max(3.0, ...)`), using `CRS.from_wkt(wkt=...)`, defining variables before `try/finally` to satisfy analysis, and ensuring `os.path.join` receives string path components (`str(out_dir)` when needed).
- Cleanup routine: `_clean_previous_outputs()` removes overlays in `raw/` (`*_overlay.tif`, `*_th*.tif`), clears `data/phase1/rounds/`, deletes `labels/phase1/finalLabels.csv` and `labels/phase1/temp_labels.csv`, and prunes legacy `final_predictions*.csv`/`final_summary*.txt` at the data root. The checkpoint file is preserved unless the user opts not to resume.

## Agent Interaction Rule
- If a user refuses a proposed patch, ask what was wrong with it before proceeding. Clarify whether the refusal concerns the approach/content or specific details (filenames, paths, wording).
- Summarize their feedback and confirm the intended adjustments before resubmitting an updated patch.
- When proposing changes to the same file within a single query, bundle all edits into one patch for that file; avoid multiple separate patches to the same file in the same round.

## Standing Rules
- At the start of every session and before any substantive patching, read and absorb:
  - `AGENTS.md` (this file) to align on base rules and workflow.
  - `HISTORY.md` to understand what has been implemented and decided so far.
- If either file is missing or unreadable, ask the user how to proceed.
 - After any action you take (code changes, config edits, runs, or analysis), add a concise entry to `HISTORY.md` capturing what changed, why, and where (files/paths). This preserves decisions and context for future sessions.

## File Access
- You can alter Python files inside the `scripts/` folder (except the `backup (old)` sub‑folder), `README.md`, and `requirements.txt`.
- You can also add content to the project (folders, files, libraries, tools, ...) if needed.
- Everything else is Read‑Only Mode, unless you ask the user for write rights (with a clear explanation) and the user consents.

## Migration Note
- Older round folders may contain `classification_report.txt`, `metrics.json`, and plots directly at the hyperparam folder root. New runs write these under a `statistics/` subfolder for clarity. Either layout is supported by the tooling.
