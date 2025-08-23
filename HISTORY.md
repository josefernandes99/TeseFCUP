# Project History (Session Log)

This file summarizes key changes and conclusions from recent Codex CLI sessions.

## 2025‑08‑18
- Grid Search: Focused SVM/RF grids; removed sieve from combinations (sieve is post‑cleanup, not predictive).
  - SVM grid: C=[2,3,5,7,10], gamma=['scale','auto',0.01,0.1], class_weight=[None,'balanced'], threshold=[0.50,0.55,0.60,0.65].
  - RF grid: n_estimators=[100,200], max_depth=[6,8,10], min_samples_leaf=[1,2,4], class_weight=[None,'balanced'], threshold=[0.50,0.55,0.60,0.65].
  - File: scripts/grid_search.py
- Candidate Selection: Added haversine DBSCAN clustering + tile‑balanced selection.
  - New config knob: `CANDIDATE_DBSCAN_EPS_KM=1.5` (typical 0.5–3.0 km).
  - File: scripts/a3_phase1_active_learning_round.py; scripts/config.py
- Requirements: Cleaned duplicates; added missing deps used by code.
  - Added: pyproj, scipy. File: requirements.txt
- AGENTS.md: Added full project architecture, interaction rules, and compact diagram.

- Data Splitting: Removed evaluate.csv; stratified splits from labels.csv via scripts/splits.py. Config: TRAIN_FRACTION/VAL_FRACTION/TEST_FRACTION/SPLIT_RANDOM_SEED.
- Training & Eval: a3 trains on train-split; evaluation.py validates on val-split; adds macro_f1.
- Calibration: SVC wrapped with CalibratedClassifierCV (cv=CALIBRATION_FOLDS, sigmoid).
- Grid Search/AL: Use StratifiedKFold CV (CV_FOLDS) over labels.csv; score by mean F1 (+acc, AUC). Summary JSON at data/phase1/rounds/grid_<model>_summary/results.json.
- Inference/Memory: Automatic chunked inference (INFER_CHUNKING_ENABLED, INFER_MAX_PIXELS_PER_BATCH) and per-tile feature cache (al_shared.get_tile_features) used in a3/a6.
- Thresholding: Single MIN_AGRI_PROB used for rounds and final masks.
- Orchestrator: Removed ensure_evaluate_file from ready_to_run.
- Small dataset safety: Auto-reduce CV folds when per-class counts are small.

## Last Run (Findings)
- Best F1/Accuracy observed: SVM C=3, gamma=auto, th=0.55 (F1≈0.833, acc≈0.84) on balanced eval.
- Best AUC observed: SVM C=1, gamma=auto (AUC≈0.885) with lower F1.
- Gamma=0.001 underfit badly; sieve had minimal impact on F1.

## Recommendations Agreed
- Use calibrated SVM (CalibratedClassifierCV) and auto thresholding by F1 on evaluate.csv.
- Introduce StratifiedKFold for hyperparam selection; keep evaluate.csv as holdout.
- Keep candidate diversity via haversine DBSCAN and tile‑balanced picks.
- Consider splitting thresholds: candidate selection vs final classification.

## 2025‑08‑19
- Candidate Justification: Added explicit motives for AL candidate proposals.
  - Printed per candidate (method, entropy, margin, cluster id/rank, tile pick order).
  - Stored as extra columns in per‑round `temp_labels.csv`.
  - Files: scripts/a3_phase1_active_learning_round.py, README.md.
- Persistent Highscore: Implemented cross‑round accumulation of most informative (hard) pixels.
  - Config: `HIGHSCORE_FILE` (data/phase1/highscore.csv), `HIGHSCORE_TOP_K` (default 200; enforced each round).
  - Scoring: entropy(p) per pixel; keeps top‑K globally, updates counts and rounds.
  - Artifacts: global KML `data/phase1/highscore_top.kml` and per‑round copy `rounds/round_*/highscore_top.kml`.
  - KML styling: label text hidden; placemark name format "#<rank> <tile> score=<score>"; added polygon border of exact pixel perimeter.
  - Utility: `scripts/highscore_report.py` to inspect/export top picks (CSV→console, optional KML).
  - Hook: Highscore updated automatically after predictions are generated in each AL round.

## 2025‑08‑21
- KML/UX: Highscore icons hidden (IconStyle scale=0) to remove pins; keep only red pixel perimeter. Single KML at `data/phase1/rounds/highscore_top.kml`.
- Labeling Menu: Added assisted modes in initial labeling
  - "Highscore pixels": pipeline feeds unlabeled entries from `highscore.csv` one‑by‑one; removes each labeled entry.
  - "Probable agri": pipeline feeds from `labels/phase1/probableAgri.csv` (top‑K by probability); removes entries as labeled.
  - Both enforce requested count ≤ available and avoid duplicates with `labels.csv`/`temp_labels.csv` and each other.
- Label Snapping: All `labels.csv` coordinates normalized to exact pixel centers (per tile) at startup and after manual/global labels.
- Probable Agri Maintenance: New persistent `labels/phase1/probableAgri.csv` updated each round from high‑p pixels; excludes any pixels present in master/temp labels or `highscore.csv`.
- Infinite Rounds: AL loop accepts `infinite`; asks after each round “Do you want to try another Active Learning round? [Y/N]”.
- Output Layout: Per‑round hyperparam folder now contains
  - `statistics/`: `classification_report.txt`, `confusion_matrix.png`, `metrics.json`, `roc_curve.png`.
  - `temporary/`: `temp_candidate.kml`, `temp_labels.csv`.
- Evaluation Data: Uses union of `labels.csv` + `temp_labels.csv` (de‑duplicated) to stabilize metrics; stats written under `statistics/`.
- End‑of‑Run Labels: Prompt to save merged labels (labels + temp) into `labels/phase1/<hyperparam‑name>/labels.csv`.
- Bug Fixes:
  - Permutation importance: Sklearn wrapper now provides a no‑op `fit` so `permutation_importance` accepts it.
  - Raster sieve warning: Replaced `rasterio.features.sieve` with connected‑component area filtering (ndimage) to avoid `NotGeoreferencedWarning`.

### Last Run Analysis
- Config used: SVM C=0.3, gamma=0.05, class_weight=None, threshold=0.4, features=base, calibration=sigmoid.
- Validation metrics by round (N≈25):
  - R1: f1=0.857, acc=0.84, AUC=0.974; R2: f1=0.889, acc=0.88, AUC=0.962
  - R3: f1=0.960, acc=0.96, AUC=0.974 (peak); R4: f1=0.917, acc=0.92, AUC=0.968
  - R5–R6: f1≈0.870, acc=0.88, AUC≈0.955/0.923; R7–R8: f1=0.909, acc=0.92, AUC≈0.93/0.981
- Final overlay (45 tiles): overall agricultural ≈0.32% of pixels (strong class imbalance; many tiles 0.0%).
- Highscore table values cluster near 0.693 entropy (p≈0.5), expected under entropy‑only ranking with many near‑ties; limited differentiation between such candidates.

### Recommendations
- Thresholding: Optimize decision threshold per round on validation (maximize F1 or desired trade‑off). Optionally use a looser threshold for candidate generation vs final map.
- Model/Calibration: Prefer `class_weight=balanced`; try `gamma=auto` and higher C (≈3–5). Consider isotonic calibration; then tune threshold on calibrated scores.
- Features: Compare "full" and "temporal_textures" feature sets against "base"; temporal NDVI/NDMI aggregates and textures usually improve agri discrimination.
- AL Scoring: Add committee disagreement (e.g., SVM+RF or bagged variants) to break ~0.693 ties; mix uncertainty and top‑probability picks (e.g., 70/30) to grow positives and refine boundaries.
- Validation: Grow validation via labels+temp (implemented) to reduce volatility; target ≥100 examples before trusting small deltas.
- Postprocessing: Revisit `SIEVE_MIN_SIZE` for tiles with small true fields to prevent over‑sieving; confirm with visual checks.
- Tile Focus: Adapt per‑tile candidate quotas toward tiles with most errors; inspect 0.0% tiles for domain shift or under‑sensitivity.

## 2025‑08‑21 (Status Review)
- Code/Config parity: Repository reflects the features documented above.
  - SVM with `CalibratedClassifierCV` (configurable `CALIBRATION_METHOD`/folds) and `class_weight` supported.
  - Feature toggles present (`ENABLE_INDICES_EXTRAS`, `ENABLE_TEXTURES`, `ENABLE_TEMPORAL`) plus A/B sets in `FEATURE_AB_TEST_SETS`.
  - AL candidate diversity implemented: entropy scoring + haversine DBSCAN + tile‑balanced round‑robin; optional hard‑negative quota.
  - Persistent artifacts exist: `data/phase1/highscore.csv` + `rounds/highscore_top.kml` and assisted labeling modes in a2.
  - Inference chunking and per‑tile feature cache wired in a3/a6; resume checkpointing available in orchestrator.
- Results recap (validation, per round; single combo `C-0.3_gamma-0.05_cw-None_th-0.4_sieve-5_feats-base_cal-sigmoid_cf-3`):
  - R1: f1=0.857, acc=0.84, AUC=0.974; R2: f1=0.889, acc=0.88, AUC=0.962
  - R3: f1=0.960, acc=0.96, AUC=0.974 (peak); R4: f1=0.917, acc=0.92, AUC=0.968
  - R5–R6: f1≈0.870, acc=0.88, AUC≈0.955/0.923; R7–R8: f1=0.909, acc=0.92, AUC≈0.929/0.981
  - Classification reports and metrics are stored per round; note: existing rounds have files at the combo root, while current code writes under `statistics/`.
- Final classify (a6) summary over 45 tiles:
  - Overall agri ≈0.32% of pixels (35698 / 11,037,763). Many tiles at 0.00%; a few up to 3.31%.
  - Overlays saved per tile as `_overlay.tif`; per‑tile percentages in `data/phase1/final_predictions.csv`.
- Observations:
  - Entropy tie‑plateau: many top entries around 0.693 (p≈0.5), limiting ranking differentiation; assisted “probable‑agri” list is not yet present on disk (optional), but highscore exists and is updated.
  - The console logs show a past permutation‑importance failure that is now resolved by adding `fit()` to the wrapper.
  - Strong class imbalance in scene outputs suggests the global threshold may be conservative for sparse positives; evaluation uses a small val set (≈25), so round‑to‑round volatility is expected.
- What to try next (targeted):
  - Thresholding: Optimize per‑round decision threshold on the validation split (maximize F1) instead of fixed `MIN_AGRI_PROB`; keep a looser band for candidate mining.
  - Model: Try `class_weight='balanced'` with SVM and scan `C≈3–5`, `gamma='auto'`; keep calibration (`sigmoid` first, compare with `isotonic`).
  - Features: Compare `temporal_textures` and `full` sets vs `base` in grid search; temporal NDVI/NDMI aggregates + textures often help.
  - AL scoring: Add a simple committee (SVM+RF) disagreement metric to break entropy ties; mix 70% uncertainty + 30% top‑prob positives to grow the positive set.
  - Focused sampling: Shift per‑tile candidate quotas toward tiles with highest current error or under‑represented positives.
  - Reporting: Standardize new rounds to write reports under `statistics/` (as in `evaluation.py`), keeping root clean; older rounds can be migrated if desired.

## 2025‑08‑21 (Late) — AL lists, training union, prompts, logging, and dedup
- Highscore/ProbableAgri rules: Allow overlap with temp_labels and between each other; only exclude pixels present in master `labels.csv`. When a pixel is labeled from either list, it is removed from both files.
  - Files: scripts/a3_phase1_active_learning_round.py (update_probable_agri, _labels_to_excluded_keys); scripts/a2_phase1_initial_labeling.py (cross-deletion in highscore_labeling/probable_agri_labeling).
- Training data union: Active learning training now uses the union of `labels.csv` + `temp_labels.csv` (de-duplicated), matching evaluation behavior.
  - File: scripts/a3_phase1_active_learning_round.py (active_learning_round union of rows).
- Infinite-mode prompt: Robust Yes/No handling (y/n/yes/no, case-insensitive) with re-prompt on invalid input.
  - File: scripts/a4_phase1_active_learning_loop.py.
- Recenter warnings: Detailed console logs whenever label coordinates are snapped to pixel centers (tile, original vs snapped lat/lon, row/col).
  - File: scripts/a2_phase1_initial_labeling.py (ensure_labels_snapped, manual/global/highscore/probable paths).
- Startup dedup/conflict check: On pipeline start, remove duplicate rows in `labels.csv`, `highscore.csv`, and `probableAgri.csv`; detect conflicting labels for the same pixel and abort with a descriptive error.
  - File: scripts/ready_to_run_phase1.py (validate_and_dedup_all invoked at startup).

## 2025‑08‑22 — Finalized Phase‑1 AL Enhancements Plan (to implement)

Context and goals
- Consolidate and extend Active Learning (AL) for Sentinel‑2 phase‑1 with: composite candidate ranking, persistent highscore/probableAgri lists, stratified splits, calibrated models, scalable inference, richer statistics, and better KML UX. This entry captures the agreed design in full detail before implementation.

Core policies
- Label snapping + dedup: Always snap coordinates to exact pixel centers first; dedup by (tile,row,col) derived from snapped coordinates. Log near‑duplicate raw coords that collapse to the same pixel. Apply to labels.csv, temp_labels.csv, highscore.csv, probableAgri.csv.
- Training data union: Build train/val from the union of labels.csv and the accumulated labels/phase1/temp_labels.csv (all prior rounds), de‑duplicated.
- Seeds: Config allows random or fixed seed. Defaults: SPLIT_SEED_MODE="random"; if fixed, use SPLIT_RANDOM_SEED and record the effective seed in run metadata.
- NDVI requirement: NDVI must exist for all pixels used in any step relying on NDVI (hard‑negatives, justifications, etc.). If missing for any pixel/tile, abort with explicit error and counts.

Composite ranking (Highscore)
- Objective: “Most important pixels” for fastest model improvement using a single combined method.
- Components (each normalized to [0,1]):
  - Uncertainty U: entropy H(p) = −p log p − (1−p) log(1−p).
  - Representativeness R: core‑set distance in feature space × geographic diversity penalty.
  - Consistency C: persistence of uncertainty across rounds.
- Final score: Additive default with weights wu, wr, wc (configurable). Default weights:
  - HIGHSCORE_COMPONENT_WEIGHTS = {"uncertainty": 0.5, "representativeness": 0.3, "consistency": 0.2}
- Normalization: rank‑based scaling per component per round for stability across tiles.
- Consistency definition: track per‑pixel rounds_uncertain_count where pixel considered “uncertain” (|p − MIN_AGRI_PROB| < δ or entropy above band). Default δ = 0.05; configurable.

Representativeness mechanics
- Feature‑space core‑set distance: Use standardized full feature vectors (no PCA reduction) to compute the distance to the nearest labeled sample. Larger distance ⇒ more novel information.
- Geographic crowding penalty: Haversine DBSCAN clustering over candidate points; penalize crowded clusters via 1/sqrt(cluster_size) to encourage geographic diversity in rankings.
- DBSCAN eps default: CANDIDATE_DBSCAN_EPS_KM = 1.5, with recommended range 0.5–3.0 km documented.

ProbableAgri ranking (confidence × representativeness)
- Score: confidence P (predicted probability) × representativeness R_pos computed as above (core‑set distance with cluster penalty). Weights configurable via PROBABLE_AGRI_COMPONENT_WEIGHTS = {"confidence": 0.7, "representativeness": 0.3}.

Candidate selection per AL round
- Candidate pool: filter by probability band (CANDIDATE_PROB_LOWER ≤ p ≤ 1.0).
- Uncertainty scoring: entropy H(p).
- Diversity: run haversine DBSCAN to form clusters; perform tile‑balanced round‑robin pick across clusters/tiles until NUM_CANDIDATES_PER_ROUND met.
- Hard‑negative quota (enabled): include up to CANDIDATE_NEGATIVE_QUOTA pixels that are agri‑look‑alikes (p in NEG_LIKE_PROB_RANGE and NDVI in NEG_LIKE_NDVI_RANGE), balanced across tiles. Purpose: reduce false positives by teaching near‑miss negatives.
- Justifications: per‑candidate rich rationale with fields: prob, entropy, margin, cluster_id/size/rank, tile_rank, local_density_k, dist_to_threshold, ndvi, nearest_label_dist/class, score_components (JSON), and a human‑readable reason sentence.

Persistent lists & KML
- Highscore: data/phase1/highscore.csv (global, persistent across rounds/runs; never deleted, only updated) and a per‑round copy under round folder; KMLs mirrored for both.
- ProbableAgri: labels/phase1/probableAgri.csv (global persistent) and per‑round copy; KMLs for both.
- Overlap rules: pixels may exist simultaneously in probableAgri, highscore, and tempLabels; must never include anything present in master labels.csv. Once promoted to labels.csv, remove from the other lists.
- Grid‑search mode: do not update highscore/probableAgri; updates occur only when running a selected combination.

Splits & evaluation
- Stratified splits from union labels according to TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION; test optional.
- CV folds: CV_FOLDS target with auto‑reduction based on minority class count (floor 2), controlled by CV_AUTO_REDUCE (default True). Log reductions.
- Seed policy applied to splits and CV.
- Training uses train; evaluation uses val in a3; evaluation module writes under statistics/.

Models & calibration
- SVM with CalibratedClassifierCV (method from CALIBRATION_METHOD, folds CALIBRATION_FOLDS).
- RandomForest supported.
- Tabular ResNet (PyTorch): compact residual MLP with scikit‑learn‑style wrapper; outputs calibrated post‑hoc. Optimized for tabular features. If torch unavailable, log warning and skip.

Feature importance
- Always emit model‑specific feature importance artifacts for the selected combo:
  - RF: native feature_importances_ + optional permutation importance.
  - SVM: permutation importance; for linear kernel, also log absolute coefficients as a reference.
  - ResNet: permutation importance.
- Files under <round>/<hyperparam>/statistics/: feature_importance.txt, feature_importance.png. Permutation importance controlled by RUN_PERMUTATION_IMPORTANCE (default True).

Inference scalability
- Chunked inference: INFER_CHUNKING_ENABLED with INFER_MAX_PIXELS_PER_BATCH; tuned for stability.
- Feature cache: hybrid strategy aiming for fastest pipeline without quality loss (in‑memory for hot reuse, on‑disk cache under data/phase1/cache with size cap FEATURE_CACHE_MAX_GB and eviction). Detailed hit/miss and eviction logs.

Thresholding, sieve, morphology, and overlays
- Single MIN_AGRI_PROB governs base masks during rounds.
- Sieve filtering with conditional keep:
  - SIEVE_KEEP_MODE = "component" (default): if a small component would be removed but any pixel in it has p ≥ SIEVE_KEEP_PROB (default MIN_AGRI_PROB or higher), keep the whole component.
  - Alternative "pixel" mode keeps only the high‑prob pixels and sieves the rest of the component.
- Final sweep in a6: evaluate combinations over FINAL_THRESHOLDS × FINAL_SIEVE_SIZES and optional FINAL_MORPH_OPEN (True/False) with its kernel sizes. For each combo, write overlays, full statistics/, and include in final_sweep_summary.csv. Select a “recommended” combo (macro‑F1 primary, AUC‑PR secondary) but keep all results.
- KML overlay colors:
  - Orange area: MIN_AGRI_PROB ≤ p < SIEVE_KEEP_PROB (borderline/confident‑enough)
  - Red pixels: p ≥ SIEVE_KEEP_PROB (high confidence)

Statistics & reproducibility
- Expanded statistics: macro‑F1, balanced accuracy, MCC, Cohen’s kappa, Brier score, ROC/AUC, PR/AUC‑PR, calibration reliability diagram, confusion matrix, per‑class PR.
- Threshold sweep table (F1/Precision/Recall vs threshold) written under statistics/.
- Config snapshot: write a comprehensive JSON (and human‑readable text) describing all hyperparameters, thresholds, seeds (including actual seed used), sieve/morph settings, model params, feature toggles, versions, and paths in each run/combo statistics/ folder.

Config surface (additions/highlights)
- Ranking weights: HIGHSCORE_COMPONENT_WEIGHTS, PROBABLE_AGRI_COMPONENT_WEIGHTS, UNCERTAINTY_BAND_DELTA (δ).
- DBSCAN eps: CANDIDATE_DBSCAN_EPS_KM (default 1.5; doc 0.5–3.0).
- Hard negatives: CANDIDATE_NEGATIVE_QUOTA, NEG_LIKE_PROB_RANGE, NEG_LIKE_NDVI_RANGE.
- Splits: TRAIN_FRACTION, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED_MODE, SPLIT_RANDOM_SEED.
- CV: CV_FOLDS, CV_AUTO_REDUCE.
- Calibration: CALIBRATION_METHOD, CALIBRATION_FOLDS.
- Thresholding/sieve/morph: MIN_AGRI_PROB, SIEVE_SIZE_PX, SIEVE_KEEP_PROB, SIEVE_KEEP_MODE, FINAL_THRESHOLDS, FINAL_SIEVE_SIZES, FINAL_MORPH_OPEN, FINAL_MORPH_KERNEL_SIZES.
- Inference/caching: INFER_CHUNKING_ENABLED, INFER_MAX_PIXELS_PER_BATCH, FEATURE_CACHE_ENABLED, FEATURE_CACHE_DIR, FEATURE_CACHE_MAX_GB.
- Importance: RUN_PERMUTATION_IMPORTANCE (default True).
- Files: HIGHSCORE_FILE, HIGHSCORE_TOP_K, PROBABLE_AGRI_FILE, PROBABLE_AGRI_TOP_K.

Operational notes
- Highscore/probableAgri are updated only when running the selected combination, never during grid‑search. Global files persist across runs; per‑round copies are saved for traceability.
- Assisted labeling flows: stream from highscore/probableAgri; on promotion to labels.csv, remove from both lists; allow overlaps otherwise.
- Candidate justifications are printed and stored into per‑round temporary/temp_labels.csv with all rationale fields for user interpretability.
- Final label export path: labels/phase1/finalLabels.csv (overwrite each end‑of‑run export for the selected combo).

Rationales for key choices
- Composite ranking (uncertainty + representativeness + consistency) approximates expected model improvement by combining where the model is unsure, where coverage is lacking, and where ambiguity persists.
- No PCA in distances preserves full feature fidelity as requested; DBSCAN adds geographic diversity control.
- Component‑level sieve keep maintains coherent field shapes; pixel‑level optional mode available for stricter pruning.
- Hard‑negative quota reduces false positives by teaching near‑miss negatives; kept small and tile‑balanced.

Pending implementation status
- This entry records the agreed plan. Next steps: wire config keys, implement ranking and persistence, update a3/a4/a6/al_shared/evaluation/grid_search, add ResNet wrapper, caching, KML styles, and statistics snapshot generation.

## 2025‑08‑22 (Later) — Implementation updates, threshold = 0.4, SIEVE_KEEP_PROB high, docs, and grid CV

Summary
- Implemented chunked inference in a3/a6 and a disk feature cache in al_shared to speed up repeated access without changing results.
- Completed candidate justifications: method, prob, entropy, margin, cluster id/size/rank, tile pick order, local density (DBSCAN-radius neighbors), distance to threshold, NDVI, nearest-labeled distance/class, and human-readable reason, saved to `<round>/temporary/temp_labels.csv`.
- Updated KML overlays: Orange pixels are those in `[MIN_AGRI_PROB, SIEVE_KEEP_PROB)`, Red pixels are `≥ SIEVE_KEEP_PROB`.
- Set decision threshold to 0.4 and aligned dependent knobs:
  - `MIN_AGRI_PROB=0.4`
  - `CANDIDATE_PROB_LOWER=0.3` (wider candidate band below threshold)
  - `NEG_LIKE_PROB_RANGE=(0.35, 0.45)` (just below/around threshold for hard negatives)
  - `SIEVE_KEEP_PROB=0.85` (very certain red pixels only)
  - Final sweep thresholds now `[0.35, 0.4, 0.45, 0.5]` and sieve sizes `[0,5,10,20]`.
- Wrote detailed comments in `config.py` for each configuration: expected value ranges, relationships (e.g., `SIEVE_KEEP_PROB > MIN_AGRI_PROB`, `CANDIDATE_PROB_LOWER ≤ MIN_AGRI_PROB`), and usability tips to prevent nonsensical combinations.
- Added true StratifiedKFold CV grid search in `scripts/grid_search.py` over union labels, with auto fold reduction and summary JSON at `data/phase1/rounds/grid_<MODEL>_summary/results.json`.
- Kept persistent highscore/probableAgri logic: composite scoring (uncertainty + representativeness + consistency), global CSV+KML updated only for selected-combo runs.
- Evaluation uses stratified val split; metrics expanded (macro F1, MCC, kappa, Brier, AUC‑PR, PR curve). Config snapshot JSON stored per round.

Key file changes
- `scripts/config.py`: Set `MIN_AGRI_PROB=0.4`, `SIEVE_KEEP_PROB=0.85`, expanded final sweep sieve sizes; added guidance comments for all config keys.
- `scripts/a3_phase1_active_learning_round.py`: chunked inference, NDVI enforcement, rich candidate justifications, updated temp labels path, KML orange/red masks, config snapshot, permutation importance, persistent lists merge logic.
- `scripts/a6_phase1_postprocessing.py`: chunked inference, conditional sieve keep modes, final sweep across thresholds/sieves/(optional morph open).
- `scripts/al_shared.py`: disk feature cache and pixel snapping/keys.
- `scripts/evaluation.py`: stratified val, expanded metrics, summary CSV.
- `scripts/grid_search.py`: true CV grid search + summary JSON.
- `scripts/a2_phase1_initial_labeling.py`: assisted labeling from highscore/probableAgri with cross-deletions.
- `scripts/ready_to_run_phase1.py`: startup snap/dedup validator, finalLabels export.
- `scripts/highscore_report.py`: preview tool for persistent lists.

Operational implications
- Orange vs Red KML: With `MIN_AGRI_PROB=0.4` and `SIEVE_KEEP_PROB=0.85`, orange shows borderline-to-strong (0.4–0.85), red shows only very certain (≥0.85). This helps visually verify confidence separation.
- Hard negatives: Enabled via `CANDIDATE_NEGATIVE_QUOTA > 0`. With the lower threshold, recommended to keep a modest quota (10–30% of candidates) to reduce false positives.
- Grid search: Now robust against small data via `CV_AUTO_REDUCE`. No changes to persistence or AL lists during grid runs.

Recommended first specific-combo (quality-first with threshold 0.4)
- Model: SVM with calibration
  - `SVM_PARAMS = {C: 3.0, kernel: 'rbf', gamma: 'scale', class_weight: 'balanced'}`
  - `CALIBRATION_METHOD='sigmoid'`, `CALIBRATION_FOLDS=3`
- Splits: `TRAIN_FRACTION=0.7`, `VAL_FRACTION=0.3`, `SPLIT_SEED_MODE='random'`
- AL selection: `NUM_CANDIDATES_PER_ROUND=30`, `CANDIDATE_DBSCAN_EPS_KM=1.5`, `CANDIDATE_PROB_LOWER=0.3`, `CANDIDATE_NEGATIVE_QUOTA=6`, `NEG_LIKE_PROB_RANGE=(0.35,0.45)`, `NEG_LIKE_NDVI_RANGE=(0.2,0.5)`
- Persistence lists: `HIGHSCORE_TOP_K=300`, `PROBABLE_AGRI_TOP_K=300`, `UNCERTAINTY_BAND_DELTA=0.05`
- Thresholding & sieve: `MIN_AGRI_PROB=0.4`, `SIEVE_KEEP_PROB=0.85`, `SIEVE_MIN_SIZE=5`, `SIEVE_KEEP_MODE='component'`
- Final sweep: `FINAL_THRESHOLDS=[0.35,0.4,0.45,0.5]`, `FINAL_SIEVE_SIZES=[0,5,10,20]`, `FINAL_MORPH_OPEN=True`, `FINAL_MORPH_KERNEL_SIZES=[3,5]`
- Performance: `INFER_CHUNKING_ENABLED=True`, `INFER_MAX_PIXELS_PER_BATCH=200_000`, `FEATURE_CACHE_ENABLED=True`
- Reporting: `RUN_PERMUTATION_IMPORTANCE=True`

Notes and future tweaks
- If red areas appear too sparse, decrease `SIEVE_KEEP_PROB` slightly (e.g., 0.8). If orange is too noisy, increase `SIEVE_MIN_SIZE` or use morph opening in sweep.
- `CANDIDATE_DBSCAN_EPS_KM` may need per-region tuning (dense small fields vs sparse large fields).
    - Disk cache eviction is not enforced yet; if cache grows large, consider adding a simple LRU by directory size.

## 2025‑08‑22 (Maint) — Cleanups and bug fixes

- Removed legacy evaluate.csv paths and functions from initial labeling.
  - a2: dropped `EVALUATE_FILE`/`EVALUATE_KML` references; removed `ensure_evaluate_file`, `export_evaluate_kml`, and the interactive evaluation menu. Validation now exclusively uses stratified splits over `labels.csv` + `temp_labels.csv`.
- Fixed unresolved refs and typing issues in a3:
  - Candidate selection now accepts `train_rows`, `X_train`, `y_train` parameters; call site updated.
  - Replaced `CANDIDATE_PROB_UPPER` with `MIN_AGRI_PROB` for the candidate band upper bound.
  - Guarded `.labels_` usage for DBSCAN; removed `deque` to avoid type warnings; used lists with `pop(0)`.
  - Adjusted `rasterio.transform.xy` calls to pass Python lists instead of ndarrays.
  - Trimmed unused imports (`pixel_key`, `ndlabel`, `math`, duplicate defaultdict) and minor lints.
- al_shared: simplified feature cache serialization
  - Store CRS as `crs_wkt` plain string (np scalar) and load via `CRS.from_wkt` (no `np.string_`).
- a6: removed unused `SIEVE_KEEP_PROB` import.
- evaluation: removed unused top-level `csv` import.
- Orchestrator: removed unused memory watcher `stop()` call path.
- Follow-ups: fixed CRS `from_wkt` call (keyword arg) in `al_shared.py` and added a guard `except` around main pipeline in `ready_to_run_phase1.py`; extended startup validator to also snap/dedup `temp_labels.csv`.

## 2025‑08‑22 (UX/Features) — Feature set menu, KML tweaks, notes normalization, and logs tee

- Feature sets: Added manual-run feature set selector (base, temporal_only, textures_only, temporal_textures, full). New config key `FEATURE_SET`; lightweight “textures” implemented as NDVI local mean/std (window `TEXTURE_WINDOW_SIZE`).
- KML:
  - Highscore/permanent KML polygons now have no fill (perimeter only) for clearer view.
  - Grid KML now draws 5×5 pixel patch lines only, with 50% transparency, reducing complexity.
- Notes: Reduced `NOTE_OPTIONS` and automatic normalization of old notes to the new scheme at startup (labels + temp_labels).
- Console logs: Orchestrator now tees stdout/stderr to `data/phase1/consoleLogs.txt` with UTF‑8 encoding; logs are preserved on completion or crash (SIGINT/SIGTERM handlers + atexit).

Files touched: scripts/a2_phase1_initial_labeling.py, scripts/a3_phase1_active_learning_round.py,
scripts/al_shared.py, scripts/a6_phase1_postprocessing.py, scripts/evaluation.py,
scripts/ready_to_run_phase1.py.
## 2025‑08‑23 — Stability + UX fixes; roadmap captured

Changes
- Progress bars: preserved Rich live bars while logging to file by improving the stdout/stderr tee (adds isatty/encoding/fileno passthrough). Console logs still saved to `data/phase1/consoleLogs.txt`.
- Infinite rounds: prompt appears before candidate labeling.
  - Manual (non-grid) flow now trains + predicts first, then asks if the user wants to label candidates for that round.
  - Fixed previous "no candidate labeling" in infinite mode and a crash when attempting to append labels from a metrics dict.
  - Guarded `append_temp_labels` to run only when a real CSV path is produced.
- Paths: highscore/probable artifacts now persist in `labels/phase1/` (CSV + KML), per request; code uses the new config paths consistently.
- Initial labeling: added quick review options to label top-N from Highscore/ProbableAgri directly in the menu; labeled entries are removed across highscore/probableAgri/temp_labels.
- KML cosmetics: ensured highscore polygons are perimeter-only (no fill); agricultural patch KML uses cyan-like borderline color with 25% opacity and no red/orange overlap.

Planned (not implemented now; revisit after current pipeline is stable)
- Temporal features from the 3 timeframes: per-pixel NDVI/NDMI/EVI deltas and aggregates (min/max/mean/std/ranges), plus simple phenology proxies.
- Textures beyond mean/std: GLCM features (contrast, homogeneity, entropy) for NDVI/B8 at small scales (e.g., 5×5, 9×9).
- Spatial context: light patch-level context (small CNN or neighbor pooling for tabular ResNet).
- AL committee uncertainty: SVM + RF (and/or bagged variants) disagreement; sample 70% by uncertainty + 30% high-confidence positives.
- Candidate diversity: add feature-space clustering to the geographic DBSCAN + tile balancing already in place.
- Thresholds: tune per-round decision threshold on validation; consider different thresholds for AL vs final maps.
- Postprocessing: component-level shape filters (size, elongation, compactness) in addition to sieve/morph.
- Calibration: compare `sigmoid` vs `isotonic` when validation data is sufficient.
- Validation: tile-aware splitting to reduce spatial leakage.
- OOD/shift detection: steer AL towards tiles with out-of-distribution features.
- Semi-supervised augmentation: cautious self-training between rounds.
- Models: evaluate LightGBM/XGBoost as tabular baselines alongside calibrated SVM.

Notes
- We intentionally postponed the above to avoid destabilizing the current pipeline (prior attempts with full GLCM/committee broke flows). We will re-introduce these gradually behind toggles once the core is stable.
## 2025‑08‑23 — Follow‑ups (AL prompt, final outputs, KML opacity)
- AL Loop UX: Ask only before labeling (removed post‑round prompt). If labeling is declined, the AL loop ends without repeating any steps. File: scripts/a4_phase1_active_learning_loop.py
- Persistent lists: Ensure Highscore and ProbableAgri update whenever predictions are saved (even if labeling is skipped). File: scripts/a3_phase1_active_learning_round.py
- Final outputs layout: Save final sweep results under data/phase1/rounds/final_round/<combo>/ with a statistics/ subfolder (classification_report.txt, metrics.json, metrics_summary.csv, confusion_matrix.png, roc_curve.png, pr_curve.png), plus config_snapshot.json and a basic agricultural_patches_final_<combo>.kml. Overlays are now written to per‑combo overlays/ folders (not into raw/). Files: scripts/a6_phase1_postprocessing.py
- Grid KML opacity: Reduced to 15% to improve clarity when overlaid. File: scripts/a2_phase1_initial_labeling.py
 - Grid KML duplication fix: Prevent generating grid KML for overlay/final‑sweep TIFFs by filtering them out; only base raw tiles generate a grid. Files: scripts/a2_phase1_initial_labeling.py
 - Inference robustness: Ignore overlay/final‑sweep TIFFs when scanning RAW_DATA_DIR for tiles to infer, preventing “index X out of bounds” crashes when overlays leak into raw/. Files: scripts/a3_phase1_active_learning_round.py
## 2025‑08‑23 — Checkpoint (Logs tidy, names in FI, zoom, cleanup, final summary)
- Console logs: Updated the tee so carriage‑return live updates render as sequential lines in `data/phase1/consoleLogs.txt` (clean analysis) while updating a single line in the console. File: scripts/ready_to_run_phase1.py
- Postprocessing live print: “Classifying => …” now uses a live update in console (full sequence retained in logs). File: scripts/a6_phase1_postprocessing.py
- Feature importance names: Both per‑round and final reports align permutation importances to the real feature names from `features.current_feature_names()`. Files: scripts/a3_phase1_active_learning_round.py, scripts/a6_phase1_postprocessing.py
- Candidate KML zoom: Added a KML LookAt centered on the candidate; ~2× closer view to save zooming time. File: scripts/a3_phase1_active_learning_round.py
- Heavy CSV cleanup: Delete `predictions.csv` immediately after it’s consumed (round KML and/or candidate selection). Delete per‑combo `final_predictions.csv` after computing `final_summary.txt` and `statistics/`. Files: scripts/a3_phase1_active_learning_round.py, scripts/a6_phase1_postprocessing.py
- Final comparison: Added `final_round/summary/final_comparison.csv` and `best_choice.txt` picking the best combo by macro‑F1 (tie‑break AUC). File: scripts/a6_phase1_postprocessing.py
- Overlay segregation: Final overlays moved out of `raw/` into `final_round/<combo>/overlays/` to prevent downstream confusion and crashes. File: scripts/a6_phase1_postprocessing.py
## 2025‑08‑23 — UI polish + strict FI
- Live updates extended: Converted repetitive prints to single-line live updates in console (with full sequential lines in consoleLogs.txt): memory watcher frees, postprocessing geotiff saves, labeling progress in initial/assisted labeling and AL candidate labeling. Files: scripts/memory_watcher.py, scripts/a6_phase1_postprocessing.py, scripts/a2_phase1_initial_labeling.py, scripts/a3_phase1_active_learning_round.py
- Labeling session summaries: Added per-session summaries (added/skipped and class breakdown) to manual, global, assisted, and AL candidate labeling. Files: scripts/a2_phase1_initial_labeling.py, scripts/a3_phase1_active_learning_round.py
- Strict permutation importance: Removed name fallback. If the number of feature names does not match importances, the pipeline raises a detailed error and halts. Files: scripts/a3_phase1_active_learning_round.py, scripts/a6_phase1_postprocessing.py
## 2025‑08‑23 — Checkpoint handling and startup cleanup
- Startup cleanup revised: The checkpoint file is no longer deleted at startup. If a checkpoint exists, the user is prompted; on “no”, the workspace is cleaned and the checkpoint is reset. If the user resumes, no cleanup is performed. On successful pipeline completion, the checkpoint is cleared as before. File: scripts/ready_to_run_phase1.py
- Cleanup scope: Removes overlays in raw (`*_overlay.tif`, `*_th*.tif`), all contents under `data/phase1/rounds/`, `labels/phase1/finalLabels.csv`, `labels/phase1/temp_labels.csv`, and legacy `final_predictions*.csv`/`final_summary*.txt` under `data/phase1/`. Checkpoint preservation follows the resume decision.
## 2025‑08‑23 — Static/type fixes sweep
- a6_phase1_postprocessing.py: add `shapes` import; remove unused `DATA_DIR`; ensure `cfg` and threshold variable are defined safely across try/finally; cast sieve size to `int`; fix figure size type with `max(3.0, ...)`; use module‑level `cfg`; apply parsed threshold consistently (metrics, KML, snapshot).
- a4_phase1_active_learning_loop.py: ensure `os.path.join(str(out_dir), "predictions.csv")` to satisfy path type checker.
- al_shared.py: switch to `Affine` from `affine`; correct `CRS.from_wkt(crs_wkt)` usage; update type hints to `Affine`; fixes “Parameter 'self' unfilled” warning.
- ready_to_run_phase1.py: replace signal lambdas with a proper handler (`_sig_handler(signum, frame)`); remove duplicate/invalid `except`; tidy log setup try/except.
- Intent: resolve IDE warnings/errors without changing functional behavior; improve robustness of post‑processing, logging, and path handling.
### Detailed notes
- a6_phase1_postprocessing.py
  - Imports: `from rasterio.features import sieve, shapes` to enable polygonization without re‑reading predictions; removed unused `DATA_DIR`.
  - Threshold handling: Initialize `old_th` and `th_local` before the `try/finally` block to guarantee they exist for any control path; restore `cfg.MIN_AGRI_PROB` in `finally`.
  - Type hygiene: ensure `sieve(size=int(ssz))` uses an `int`; plotting uses `max(3.0, topk*0.3)` to avoid int/float generic mismatch flagged by some type checkers.
  - Consistent threshold: use `th_local` (parsed from tag like `th0.45_s5`) for evaluation stats, KML generation, and `config_snapshot.json`.
  - KML: Geometry built from `shapes()` over the cleaned mask using the source transform; on‑the‑fly CRS transform to EPSG:4326 for coordinates.
  - Stats/cleanup: Write `final_predictions.csv` per combo, then compute `final_summary.txt` and delete the heavy CSV; save metrics to `statistics/` via `evaluation.evaluate_model`.
  - Feature importance: optional permutation importance on the validation split (as per config); names validated against `features.current_feature_names()`; bar chart saved as `feature_importance.png`.
  - Final comparison: build `final_round/summary/final_comparison.csv` and `best_choice.txt` selecting best combo by macro‑F1 (tie‑break AUC).
- a4_phase1_active_learning_loop.py
  - Path join: cast `out_dir` to `str` for `os.path.join` when passing the predictions path to `candidate_selection_from_csv` in infinite mode; prevents strict type checker complaints where pathlikes may be mixed.
- al_shared.py
  - Affine: use `from affine import Affine` and update cache type hints to `Tuple[np.ndarray, Affine, rasterio.crs.CRS]`.
  - CRS de/serialization: `CRS.from_wkt(wkt=str(crs_wkt))` to satisfy kw‑only signature; NPZ cache stores `crs_wkt` as a NumPy string scalar and `transform` as first 6 coefficients of the affine.
  - Disk feature cache: attempted load path `FEATURE_CACHE_DIR/<tile>.npz`; if present and enabled, loads `arr`, rebuilds `Affine`, and restores `CRS`, otherwise computes features and persists compressed NPZ.
- ready_to_run_phase1.py
  - Logger tee: capture stdout/stderr to `data/phase1/consoleLogs.txt` while preserving carriage‑return updates in console and converting CR→NL in the file for clear sequential logs.
  - Signals: replace inline lambdas with a proper `_sig_handler(signum, frame)`; register with `contextlib.suppress` to support constrained environments gracefully.
  - Error handling: condensed `try/except` around log setup; removed the unreachable extra `except` branch flagged by the parser.
## 2025‑08‑23 — Robust feature names, NDVI reporting, repeated validation

- Robust feature alignment: features now infer band schema from the stacked raster on disk (seasons + terrain), replacing ASPECT with ASPECT_SIN/COS deterministically.
  - Files: scripts/features.py
  - `current_feature_names()` opens a tile and derives names from channel count, avoiding config/season-count drift.
  - NDVI reporting: a3 selects a representative NDVI band (prefers NDVI_s2) for predictions.csv and candidate justifications.
- FI naming mismatch resolved: permutation importance uses the robust names; pipeline halts only if lengths truly diverge.
  - Files: scripts/a3_phase1_active_learning_round.py, scripts/a6_phase1_postprocessing.py
- Validation improvements:
  - True random splits in SPLIT_SEED_MODE="random" (no silent override).
  - Repeated validation implemented with seed recording and per-repeat metrics; summary (mean/std) written to metrics.json and metrics_summary.csv; per-repeat details in metrics_repeats.json and split_seeds.json. First split drives plots and classification report; validation_split_rows.csv saved for traceability.
  - Files: scripts/splits.py, scripts/evaluation.py
  - Config: `VAL_REPEATS` added (default 1).
- Feature set menu clarity: mark non-base options as "currently unsupported" so users avoid inactive combinations.
  - File: scripts/a4_phase1_active_learning_loop.py

Notes
- Temporal/textural feature modes remain disabled in feature extraction; menu hints reflect this to avoid confusion. Future work can re-enable with precise schemas.
