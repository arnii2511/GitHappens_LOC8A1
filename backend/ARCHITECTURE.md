# Backend Architecture

## Goal
Provide a modular, maintainable backend for buyer-exporter matching with clear separation between:
- data preparation
- feature engineering
- ranking logic
- online learning
- API/transport layer

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
- `risk.py`: news risk penalty model.
- `checklist.py`: verification checklist generation.
- `legacy_ranker.py`: old heuristic ranker (fallback/debug).
- `helpers.py`: numeric/text helper utilities.

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
4. Scores are blended with adaptive collaborative weight based on per-buyer interaction depth.
5. Contextual exploration injects high-potential unseen exporters into top results occasionally.
6. `/feed` builds candidate features, scores via hybrid model, returns ranked cards.
7. `/swipe` persists action and updates model online.
8. `/simulate/update` mutates news and refreshes ranker risk cache.
