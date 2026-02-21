# Backend Architecture

## Goal
Provide a modular, maintainable backend for buyer-exporter matching with clear separation between:
- data preparation
- feature engineering
- ranking logic
- online learning
- API/transport layer

For setup/training/evaluation commands on a new machine, see `backend/QUICKSTART_COMMANDS.md`.

## Package Layout

### `app/main.py`
- FastAPI entrypoint and route handlers only.
- Coordinates startup loading, model initialization, and request flow.

### `app/db.py`
- Database access for swipes and update logs.
- No ranking logic.

### `app/pipeline/`
- `data_loader.py`: loads cleaned CSV datasets.
- `feature_engineering.py`: buyer/exporter feature computation.
- `dynamic_weights.py`: adaptive match-weight policy (data quality + risk aware).
- `risk.py`: news risk penalty model.
- `checklist.py`: verification checklist generation.
- `legacy_ranker.py`: old heuristic ranker (fallback/debug).
- `helpers.py`: numeric/text helper utilities.

### `app/industry_map.py`
- Canonicalizes raw industry labels.
- Provides buyer-exporter/news industry similarity scoring.
- Enables related-industry matching (not only exact string matches).

### `app/ml/`
- `feature_builder.py`: pairwise candidate feature construction.
- `supervised.py`: online supervised ranking model.
- `collaborative.py`: interaction embedding model (SVD).
- `ltr.py`: learning-to-rank model (LightGBM/XGBoost with fallback).
  - Tries GPU first when enabled, then safely falls back to CPU.
- `time_decay.py`: recency weighting for interactions.
 - Full end-to-end GPU requires CUDA-enabled XGBoost and CuPy (for collaborative SVD + GPU-side prediction input).
- `hybrid_ranker.py`: orchestrates supervised + collaborative blend.
- `constants.py`: shared ML feature column contract.
- `common.py`: shared ML utility functions and sklearn/scipy availability.

### Compatibility Layers
- `app/scoring.py`: facade that re-exports pipeline APIs for older imports.
- `app/ml_pipeline.py`: facade that re-exports `HybridRanker`.

## Runtime Flow
1. Startup loads cleaned data and computes engineered features.
2. Historical swipes are loaded from DB.
3. Hybrid model trains:
- supervised learner from interactions (or bootstrap labels)
- collaborative SVD embeddings from swipe matrix
- learning-to-rank model for final ordering refinement
4. Candidate generation uses industry canonicalization + related-industry similarity thresholding.
5. Match scoring uses dynamic weights that adapt by missing buyer capacity data and macro-risk intensity.
6. Scores are blended with adaptive collaborative weight based on per-buyer interaction depth.
7. Contextual exploration injects high-potential unseen exporters into top results occasionally.
8. `/feed` builds candidate features, scores via hybrid model, returns ranked cards.
9. `/swipe` persists action and updates model online.
10. `/simulate/update` mutates news and refreshes ranker risk cache.
