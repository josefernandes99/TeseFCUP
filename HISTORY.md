# Project History (Session Log)

This file summarizes key changes and conclusions from recent Codex CLI sessions.

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
