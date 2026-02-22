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
  - Adds behavior history features (buyer/exporter/pair swipe rates, recency, interaction depth).
  - Builds hybrid candidate pool: strong industry-similarity core + controlled diverse exploration pool.
  - Adds `teacher_score` (cross-encoder distillation target) and `graph_sim` (graph embedding similarity).
- `graph_features.py`: graph-derived buyer/exporter embeddings and `graph_sim` feature.
- `supervised.py`: online supervised ranking model.
  - Uses class-imbalance handling (sample/class weighting).
  - Applies PU-style denoising weights for implicit/noisy negatives.
- `collaborative.py`: interaction embedding model (SVD).
- `ltr.py`: learning-to-rank model (LightGBM/XGBoost with fallback).
  - Runs small parameter search and picks best trial by ranking quality.
  - Tries GPU first when enabled, then safely falls back to CPU.
  - Applies PU-style denoising weights for implicit/noisy negatives.
- `time_decay.py`: recency weighting for interactions.
 - Full end-to-end GPU requires CUDA-enabled XGBoost and CuPy (for collaborative SVD + GPU-side prediction input).
- `hybrid_ranker.py`: orchestrates supervised + collaborative blend.
- `constants.py`: shared ML feature column contract.
- `common.py`: shared ML utility functions and sklearn/scipy availability.

### `app/retrieval/`
- `text_encoder.py`: semantic text embeddings (SentenceTransformer with TF-IDF fallback).
- `text_encoder.py`: optional cross-encoder teacher scoring for distillation (`teacher_score`).
- `two_tower.py`: dual-encoder retrieval model (buyer tower + exporter tower).
  - Includes hard-negative sampling, logQ-style correction, and teacher distillation loss.
- `ann_index.py`: ANN candidate search over exporter embeddings.
- `industry_rules.py`: cross-industry association mining (support/confidence/lift) for candidate expansion.

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
4. Two-tower retrieval model learns buyer/exporter latent embeddings from interactions.
5. Industry association rules expand candidate industries beyond strict direct matching.
6. Two-tower retrieval scores expanded candidates and returns top-N.
7. Candidate generation applies industry canonicalization + related-industry similarity and dynamic feature construction.
8. Match scoring uses dynamic weights (risk/data-quality aware) + behavior-history features + text similarity.
9. LTR and supervised models rerank retrieved candidates, blended with collaborative signals.
10. Contextual exploration injects high-potential unseen exporters into top results occasionally.
11. `/feed` returns final ranked cards.
12. `/swipe` persists action and updates online learners; retrieval refreshes periodically.
13. `/simulate/update` mutates news and refreshes risk cache.
