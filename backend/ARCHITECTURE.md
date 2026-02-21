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
4. `/feed` builds candidate features, scores via hybrid model, returns ranked cards.
5. `/swipe` persists action and updates model online.
6. `/simulate/update` mutates news and refreshes ranker risk cache.
