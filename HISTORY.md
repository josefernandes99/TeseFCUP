# Project History (Session Log)

This file summarizes key changes and conclusions from recent Codex CLI sessions.

## 2025‑09‑11 — Cache cleanup, script audit, re‑org plan, and final‑run guidance
- Round cache auto‑cleanup added (disk space):
  - Issue: rounds 5–8 left `_global_refresh/` and `_tile_preds/` behind because cleanup only existed in an alternate path.
  - Change: added final cleanup inside `refresh_global_lists_full(...)` so both folders are deleted at the end of a round refresh; they are caches and will be regenerated if needed.
  - File: `scripts/a3_phase1_active_learning_round.py` (function `refresh_global_lists_full`). Safe to delete manually.

- Scripts audit and necessity:
  - Essential to run Phase‑1: `ready_to_run_phase1.py`, `a0_setup_check.py`, `a1_phase1_data_download.py`, `a2_phase1_initial_labeling.py`, `a3_phase1_active_learning_round.py`, `a4_phase1_active_learning_loop.py`, `a6_phase1_postprocessing.py`, `config.py`, `features.py`, `al_shared.py`, `splits.py`, `evaluation.py`, `refresh_lists.py`, plus small utils.
  - Optional/diagnostic: `grid_search.py`, `analyze_grid_results.py`, `highscore_report.py`, `plot_round_metrics.py`, `analyze_nan_bands.py`, `gee/*`.
  - Note: there is no `detect_nan_pixels.py` at repo root; `scripts/analyze_nan_bands.py` generates `detected_nan_pixels.kml`.

- Postprocessing warning explained:
  - `NotGeoreferencedWarning` during final sweep comes from calling raster operations (e.g., `sieve`) on an in‑memory array without a transform; the algorithm still runs. It does not imply raw tiles lack CRS.
  - Action: ignore unless saved overlays fail to align in GIS. To verify a file: `gdalinfo <tile>.tif` should show “Coordinate System” and non‑identity pixel size.

- Results snapshot (full dataset, latest):
  - Rounds trend (`data/phase1/rounds/rounds_metrics.csv`): last rounds ≈ stable. Round 9: F1≈0.8705, AUC‑PR≈0.9471. Round 10: F1≈0.8711, AUC‑PR≈0.9537.
  - Final sweep (th0.35_s10): F1≈0.8719 (similar to round 10) but visually more conservative due to `keep‑prob=0.85` with pixel mode.
  - Feature importance (round 10): red‑edge bands (B6/B7), B8A/B5, and indices (NDVI/EVI2/GNDVI/NDWI/NDRE) dominate.

- Labels expansion strategy to 1–2k (per year 2019–2024):
  - 40% Uncertain near threshold (|p−0.5| ≤ 0.05), stratified by island and season.
  - 30% Hard negatives (trees/riparian/urban greens) to cut FP trees.
  - 20% Hard positives (missed small fields, early/late crops) to reduce FN crops.
  - 10% Spatially stratified random per island; enforce per‑island minima.
  - Keep de‑dup on (tile,row,col); record “skips” for ambiguous pixels.

- Fixed hyperparameters for final thesis runs (use across years for comparability):
  - Common: `MIN_AGRI_PROB=0.35`, `SIEVE_MIN_SIZE=5`; isotonic calibration, 3 folds.
  - SVM (RBF): `C=3.0`, `gamma='scale'`, `class_weight='balanced'`.
  - RandomForest: `n_estimators=500`, `max_depth=12`, `min_samples_leaf=2`, `class_weight='balanced'`.
  - XGBoost: `n_estimators=700`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=1.0` (GPU if available; else CPU hist). Calibrate probabilities after training.
  - ResNet (tabular): `epochs=20`, `learning_rate=5e‑4`, `batch_size=128` (reduce if memory constrained).

- Final sweep guidance:
  - For thesis maps prefer last‑round polygons at the selected threshold+sieve. If running final sweep for GeoTIFF overlays, use only the chosen threshold (0.35) and sieve (5). The current `keep‑prob=0.85` in pixel mode trims fringes; consider a lower gate (e.g., 0.80) or component mode if a more inclusive overlay is desired (discussion only; no code changes made).

- Thesis figures to include (per year):
  - PR and ROC curves with the chosen operating point; threshold sweep (Precision/Recall/F1 vs threshold) with the selected threshold marked.
  - Calibration plot (reliability + histogram) and Brier score.
  - Normalised confusion matrix; macro‑F1, AUC‑PR, MCC reported.
  - Feature importance bar chart and feature‑family contributions.
  - Round‑by‑round metrics trend; probability histograms by class.
  - Spatial panels (before/after labels expansion) illustrating reduced FP trees and recovered FN crops.

- Project re‑organisation plan (no refactors performed yet):
  - Proposed structure: `src/` (modules), `cli/` (entrypoints), `data/` (raw/labels/persistent), `artifacts/` (caches like `_tile_preds`), `runs/` (per‑round outputs), `reports/` (figures/KML/tables), `logs/`, `configs/`, `docs/`.
  - Module map and CLI subcommands outlined to separate orchestration from reusable code; import‑side effects to be removed in a future refactor (pending approval).

Notes
- Old island‑only run (saved under `oldRuns/run020925-12h (parcial, saoNicolau)`) visually looked tighter (fewer FP trees, fewer FN crops). Differences are consistent with the full dataset’s added variability. Metrics JSONs were not present in that archived run; comparison was KML‑based.
- Next steps for each year (2019–2024): expand labels per the 40/30/20/10 plan with per‑island quotas, train with the fixed hyperparameters above, pick threshold via the sweep (default 0.35), run 3–5 AL rounds, and export the recommended figures.

## 2025‑09‑10 — Thesis front‑matter, passive voice, acronyms, refs, and plot legend
- Front‑matter order and blanks (preamble edits approved by user):
  - Dedication on roman IV (page number hidden), Acknowledgements on V, Resumo (PT‑PT) on VI, Abstract on VII; lists follow contiguously without extra blanks; chapters still open on right pages.
  - Ensured a single blank page between cover p.1 (I) and cover p.2 (III), with no page number on the blank.
  - Implemented local `\cleardoublepage→\clearpage` wrappers around preface and lists; manual cover insertion to control blanks.
- Names of lists and additions (style edits in upthesis.sty approved by user):
  - “Contents”→“Table of Contents”; “Listings”→“List of Listings”; “Acronyms”→“List of Abbreviations”. Added “List of Tables” after “List of Figures”.
- Abstract/Resumo:
  - Rewrote Resumo in PT‑PT, enabled Portuguese hyphenation only for the Resumo block with high penalties (minimal hyphenation), normalised “pixel” and “verdade de terreno”.
  - Tightened Resumo by ~5–6 lines without losing content; then expanded Abstract wording by ~3–4 lines (no new content) to balance lengths.
  - Removed roman (i)…(v) enumeration style; replaced with parallel, semicolon‑separated phrasing.
  - Keywords aligned across languages; include all four models: Random Forest, Support Vector Machine, XGBoost, ResNet; added “Change detection / Deteção de alterações”.
  - PDF metadata updated to “Cape Verde” and to the new keyword list.
- “Cape Verde” harmonisation:
  - Replaced “Cabo Verde” with “Cape Verde” across English content files; Resumo keeps “Cabo Verde” as per PT‑PT.
- Acronyms and usage:
  - Added ROC, XGBoost, ResNet to `acros.tex`; used short‑form `\acs{…}` across chapters after the lists so full expansions don’t repeat.
- State of the Art + Intro related work:
  - Replaced prior generic cites with user‑provided references only; added “Applied Studies and Regional Context” (chap‑art) and a “Brief Related Work” (intro) section with focused citations:
    - Pereira et al., 2022 (cashew orchards, Guinea‑Bissau) — supports indices + seasonal compositing for crop classes.
    - Pereira, 2020 MSc (vegetation monitoring tools) — aligns with S2, indices, GEE, reproducibility.
    - Miguel Pereira, 2024 MSc (active learning for land‑cover) — motivates the uncertainty+diversity acquisition loop.
  - `refs.bib` now contains exactly those three entries.
- Tone and tense:
  - Converted first‑person plural (“we”) to impersonal/passive throughout content files. Past tense is used for completed work (e.g., “products were generated”), present for general facts. Acknowledgements remain personal.
- LaTeX hygiene and overflow fixes:
  - Long listings in `chap‑meth.tex` use `breaklines=true` etc.; long path tokens in `chap‑results.tex` wrapped with `\path{…}`.
  - Removed backticks from ASCII project tree; kept consistent listing style.
- Plot legend cropping fixed:
  - `scripts/plot_round_metrics.py`: reserved 20% figure width for a figure‑level legend (80/20 layout) and enforced integer x‑axis ticks with `MaxNLocator(integer=True)`.

Notes
- All template/preamble/style changes were made under explicit user approval despite the format‑lock rule.
- Build order reminder: pdflatex → biber → pdflatex → pdflatex.

## 2025‑09‑09 — Thesis content, LaTeX hygiene, UK English, and pipeline doc
- Added a permanent “writtenThesis format lock” to AGENTS.md: content‑only edits; no style/prelude changes without approval.
- Wrote and structured thesis content (Abstract, Executive Summary, Introduction, State of the Art, Methodology, Implementation, Results & Discussion, Conclusion, Appendices):
  - Set thesis metadata (title/author/subject/keywords) and dedication placeholder.
  - UK‑English harmonisation across chapters (labelling/labelled, minimise/normalise, etc.).
  - Detailed GEE acquisition methodology, seasonal windows, island ROI tiling, indices and terrain per season, and reproducibility notes.
  - Added Sentinel‑2 band reference table (centre nm, resolution, typical use) and concise GEE index‑computation listing.
  - Inserted placeholders for pipeline diagram, learning curves, ROC/PR, qualitative overlays, island‑by‑year panels.
  - Kept results narrative open; wired plan to insert pipeline‑produced plots/metrics later.
- LaTeX compile fixes and hygiene:
  - Replaced Unicode symbols (≥, ndash/mdash/nbhyphen) with TeX/math equivalents; normalised dashes ("--/---").
  - Removed stray tab in `linkcolor` and consolidated colour handling (kept template’s colour; removed duplicate xcolor in thesis.tex).
  - Fixed xcolor warnings and overfull boxes by narrowing placeholder figure boxes.
  - Cleaned config lstlisting to avoid listings escapechar collisions; removed inline `# ...` comments inside listings.
  - Project tree: converted to ASCII and wrapped in a robust `lstlisting` (escapechar override; keepspaces/breaklines/etc.) to avoid UTF‑8 and “Missing $” errors.
  - Removed Portuguese “Resumo”; added English Executive Summary only.
- Thesis cross‑refs and citations:
  - Ensured chapters carry labels (`chap:devel`, `chap:results`, `chap:conc`).
  - Populated refs.bib with all cited keys; build order documented (pdflatex → biber → pdflatex → pdflatex).

## 2025‑09‑09 — Persistent lists crash fix
- Fixed `_npb2().update(...)` misuse in highscore merge; use active progress instance `_prog2.update(...)`.
  - File: `scripts/a3_phase1_active_learning_round.py`

## 2025‑08‑30 — Remove terrain bands from pipeline
- Dropped ELEVATION, SLOPE, and ASPECT from the data pipeline to avoid DEM coverage gaps causing NaNs.
  - Downloader no longer fetches or appends USGS/SRTMGL1_003 terrain bands.
  - Config `INDICES` now includes only spectral indices: NDVI, EVI, EVI2, NBR, NDMI.
  - Feature builder no longer expects or transforms ASPECT; it only appends light NDVI texture features.
  - Analysis helpers referencing terrain remain unchanged and are not used by the pipeline.

## 2025‑08‑28 — Session Reminder
- Reminder for next session: "what we are going to do with ELEVATION, SLOPE and ASPECT? is there any fix? can we fix it in GEE? study the case with chatpgt. images of the NaN's are saved on Google Earth Pro."
- Next session Codex CLI instructions: "first thing, remind the user of the reminder. then, delete the reminder from HISTORY.md and these instructions".

## 2025‑08‑27 — Log hygiene, seeds, KML color, predictions.csv, and final‑sweep perf
- Startup log: removed the outdated note "pipeline now runs without GLCM textures or global normalization" from the orchestrator.
  - File: `scripts/ready_to_run_phase1.py`
- Grid KML logging: removed per‑tile "Grid KML generated" spam; now prints a single summary: "all grid kml generated" or an error summary with failed tile names.
  - File: `scripts/a2_phase1_initial_labeling.py`
- Feature set menu: removed the interactive feature‑set chooser; pipeline always uses all available features.
  - Files: `scripts/a4_phase1_active_learning_loop.py`, `scripts/features.py`
- Features always‑on: textures appended unconditionally; `current_feature_names()` reflects this.
  - File: `scripts/features.py`
- Memory watcher: silenced `[MemoryWatcher]` prints to reduce console noise; functionality retained.
  - File: `scripts/memory_watcher.py`
- Random seed policy: fixed stratified splits so that SPLIT_SEED_MODE="random" truly produces non‑reproducible splits (passes `random_state=None` through to sklearn).
  - File: `scripts/splits.py`; evaluation already respects the mode.
- KML colors: consolidated polygon output to a single blue style over `MIN_AGRI_PROB` (no orange/red split) in round KMLs and final postprocessing KMLs.
  - Files: `scripts/a3_phase1_active_learning_round.py`, `scripts/a6_phase1_postprocessing.py`
- Predictions.csv footprint: candidate labeling now consumes predictions in‑memory; no predictions.csv is written for labeling flows. Infinite‑mode prompt uses in‑memory predictions as well. Round KMLs are generated from memory.
  - Files: `scripts/a3_phase1_active_learning_round.py`, `scripts/a4_phase1_active_learning_loop.py`
- Final sweep efficiency: cached per‑tile probabilities within a postprocessing run, avoiding re‑inference for each threshold/sieve combo.
  - File: `scripts/a6_phase1_postprocessing.py`
## 2025‑08‑31 — Final-round overlays off, combo sweep tightened, lists fixed, and progress bars
- Final round overlays: stop writing per-combo overlay GeoTIFFs under `rounds/final_round/*/overlays`. Only statistics (`metrics.json`, `final_summary.txt`), KML and comparison CSVs are produced. Console spam from per-tile "Classifying =>" and "Saved GeoTIFF =>" is silenced during the final sweep.
- Combo sweep scope: restrict final sweep combinations to threshold (`th`) and sieve size (`s`) only; remove morphology (`m`) variants from final-round tags and runs. Config `FINAL_MORPH_OPEN=False` for clarity.
- Persistent lists: make `refresh_lists.py` and `refresh_global_lists_full()` robust when no labels/features are available. Highscore and probableAgri now populate with all pixels (ranked by uncertainty and probability respectively) and generate their KMLs even without representativeness distances.
- Post-sweep UX: add progress bars for per-combo summarization/evaluation and the final comparison build, so there is visible progress after the "Final sweep" bar completes.
## 2025‑09‑01 — Re‑add DEM terrain bands and SVM cache size
- Terrain bands back in pipeline: ELEVATION, SLOPE, ASPECT replicated per season (static DEM applied to each timeframe) and included end‑to‑end.
  - Export: Added NASADEM elevation with `ee.Terrain.products` slope/aspect and suffixed `_s#` per timeframe.
    - File: `scripts/a1_phase1_data_download.py`
  - Features: Expect terrain bands in season feature names and include them when constructing derived features (NDVI textures unchanged).
    - File: `scripts/features.py`
  - Verification: Feature stack check updated to include terrain bands in expected raw export channels.
    - File: `scripts/ready_to_run_phase1.py`
- New SVM hyperparameter: `cache_size` (default 2048 MB) configurable in `config.py` and passed through to `sklearn.svm.SVC`.
  - File: `scripts/config.py`
