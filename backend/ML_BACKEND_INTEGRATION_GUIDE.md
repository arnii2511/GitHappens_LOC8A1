# ML Backend Integration Guide

This document explains how the ML backend parts connect to each other. The first document explained the theory and design ideas. This one explains integration. It shows which file talks to which file, what kind of data is passed between them, and how one action moves through the full system. It is written in simple English so a new reader can trace the whole flow without reading every file first.

## 1. What "Integration" Means Here

- Integration means one file depends on another file.
- Integration also means one module gives data, features, scores, or updates to another module.
- In this backend, integration happens in three big directions.
- The first direction is data flow.
- The second direction is model flow.
- The third direction is API and feedback flow.

The backend is not a set of isolated files. It works like a chain. Raw data becomes clean data. Clean data becomes engineered features. Features become retrieval and ranking inputs. API requests call the ranker. Swipe feedback goes back into the database and the in-memory models. Because of this, understanding integration is the fastest way to understand the whole backend.

## 2. Main Integration Layers

- `data/raw` integrates into `scripts/clean_data.py`.
- `scripts/clean_data.py` integrates into `data/clean`.
- `app/pipeline/data_loader.py` integrates clean CSVs into the runtime.
- `app/pipeline/feature_engineering.py` integrates raw clean tables into buyer and exporter feature tables.
- `app/ml/feature_builder.py` integrates business features, interaction history, retrieval output, text similarity, and risk logic into pairwise ML features.
- `app/retrieval/*` integrates semantic retrieval into the ranking system.
- `app/ml/*` integrates learned ranking models into the final score.
- `app/main.py` integrates the FastAPI layer with the ranker and the database.
- `app/db.py` integrates the backend with Postgres for swipes, summaries, and logs.
- `scripts/*` integrate the offline training and evaluation cycle with the same runtime modules.

The project uses the same core logic in both live serving and offline training. This is a strong design choice. It means the offline scripts do not invent a separate fake pipeline. They reuse the same loaders, feature engineering, pair feature builder, and ranker logic. That reduces mismatch between "training world" and "serving world."

## 3. Top-Level Dependency Map

- `backend/app/main.py`
  connects to `db.py`, `pipeline`, and `ml`.
- `backend/app/db.py`
  connects to Postgres and returns swipe history to the ranker.
- `backend/app/pipeline/data_loader.py`
  connects to `data/clean/*.csv`.
- `backend/app/pipeline/feature_engineering.py`
  connects to `helpers.py`.
- `backend/app/pipeline/risk.py`
  connects to `industry_map.py` and `helpers.py`.
- `backend/app/pipeline/legacy_ranker.py`
  connects to `risk.py`, `dynamic_weights.py`, `checklist.py`, `helpers.py`, and `industry_map.py`.
- `backend/app/ml/hybrid_ranker.py`
  connects to `feature_builder.py`, `supervised.py`, `ltr.py`, `collaborative.py`, `ncf.py`, `retrieval`, and `checklist.py`.
- `backend/app/ml/feature_builder.py`
  connects to `industry_map.py`, `dynamic_weights.py`, `risk.py`, `helpers.py`, and optional retrieval/text/graph scorers.
- `backend/app/retrieval/two_tower.py`
  connects to `text_encoder.py`, `ann_index.py`, `industry_rules.py`, and time decay logic.
- `backend/app/ml/model_explainer.py`
  connects to the supervised model and feature columns.
- `backend/scripts/train_ranker.py`
  connects to `pipeline` and `HybridRanker`.
- `backend/scripts/evaluate_ranker.py`
  connects to `pipeline`, `HybridRanker`, and `metrics_store.py`.
- `backend/scripts/retrain_cycle.py`
  connects all major scripts into one repeatable workflow.

This map shows the real center of gravity in the backend. The center is not the API file. The center is the feature and ranking layer, especially `feature_builder.py` and `hybrid_ranker.py`. Most other parts either prepare data for them or consume their results.

## 4. Startup Integration Flow

- `app/main.py` runs the startup event.
- Startup calls `init_db()` from `app/db.py`.
- Startup calls `load_data_clean()` from `app/pipeline/data_loader.py`.
- Startup calls `engineer_buyer_features()` and `engineer_exporter_features()` from `app/pipeline/feature_engineering.py`.
- Startup creates `HybridRanker(...)`.
- Startup fetches swipe history from `fetch_swipes()` in `app/db.py`.
- Startup calls `ranker.fit(swipes)`.
- The finished ranker is stored in `STATE["ranker"]`.

This startup flow is the first major integration point. It joins storage, data preparation, model building, and serving into one live app state. After startup finishes, the API can answer `/feed` calls because all the needed pieces are already in memory.

## 5. Inside `HybridRanker.fit()`

- It sanitizes interactions.
- It rebuilds buyer memory and seen-exporter memory.
- It gives the full interaction table to `PairFeatureBuilder.update_interaction_stats()`.
- It trains the retriever through `TwoTowerRetriever.fit()`.
- It injects retriever scoring into the feature builder with `set_retrieval_scorer(...)`.
- It currently disables industry association lookup in the live path with `set_industry_assoc_lookup(None)`.
- It currently disables graph scoring in the live path with `set_graph_scorer(None)`.
- It trains collaborative SVD through `CollaborativeModel.fit()`.
- It trains NCF through `NeuralCollaborativeFilteringModel.fit()`.
- It trains the supervised model through `OnlineSupervisedModel.fit()`.
- It trains the ranking model through `LearningToRankModel.fit()`.
- It clears candidate cache and marks the ranker as trained.

This fit process is the main integration hub of the ML side. It shows how one shared interaction table is reused by many subsystems. Retrieval uses it. Collaborative models use it. Supervised models use it. The same history powers many views of the same buyer-exporter behavior.

## 6. File-by-File Integration Details

### `app/main.py`

- Integrates HTTP routes with the ML system.
- Reads from `STATE`.
- Sends `buyer_id` into the ranker.
- Sends swipe payloads into both the database and the in-memory ranker.
- Refreshes news in the ranker when simulated updates happen.

This file is the transport layer. It does not build features itself. It does not train models directly. It coordinates other modules. That separation is good because it keeps API code thin.

### `app/db.py`

- Integrates backend code with the database.
- Stores swipes.
- Reads swipe history for training.
- Builds buyer profile summary records after inserts.
- Logs signal updates.
- Can also support pipeline metrics persistence through a separate metrics table path.

This file is the memory of the system. Without it, the ML models would always restart from zero. Its main job is not scoring. Its job is persistence.

### `app/pipeline/data_loader.py`

- Integrates cleaned CSV files into the Python runtime.
- Converts date columns.
- Normalizes industry columns.

This file is small but important. If it fails, the full backend fails because almost every later step depends on clean buyers, exporters, and news tables.

### `app/pipeline/feature_engineering.py`

- Integrates helper functions with buyer and exporter raw fields.
- Converts trust-related columns into trust scores.
- Converts growth and activity signals into intent scores.

This file creates the base features that many later modules reuse. It is the bridge between table columns and model-ready business signals.

### `app/pipeline/risk.py`

- Integrates news records with industries and dates.
- Produces risk penalties and warning text.

This file connects external market context to matching logic. It is the main place where news becomes a ranking input.

### `app/pipeline/dynamic_weights.py`

- Integrates buyer profile quality and risk into match weight allocation.
- Returns weights for capacity fit, intent, and communication.

This file is a control layer. It does not score a match directly. It changes how the score should be built.

### `app/pipeline/checklist.py`

- Integrates card-level trust and risk with buyer communication preference.
- Produces document, payment, and verification advice.

This file is a downstream explainer and action layer. It connects model output to practical user guidance.

### `app/ml/feature_builder.py`

- Integrates business features from buyers and exporters.
- Integrates news risk through `risk.py`.
- Integrates dynamic business weighting through `dynamic_weights.py`.
- Integrates interaction history through internal history stores.
- Integrates retrieval scores if a retriever is attached.
- Integrates text similarity if a text encoder is attached.
- Integrates teacher scores when teacher mode is on.
- Integrates graph scores when a graph scorer is attached.

This is the most important integration file in the ML stack. It is where many separate streams become one row of pairwise features.

### `app/retrieval/text_encoder.py`

- Integrates text fields from buyers and exporters into embeddings.
- Integrates sentence-transformer or TF-IDF backends.
- Integrates optional teacher cross-encoder scoring.
- Integrates teacher cache files from disk.

This file links raw descriptive text with semantic matching. It gives the rest of the system a way to compare buyers and exporters beyond exact field equality.

### `app/retrieval/two_tower.py`

- Integrates buyer features, exporter features, optional text vectors, interactions, ANN search, and industry rule mining.
- Produces retriever embeddings and top candidate lists.

This file turns training history into fast candidate search. It is the bridge between representation learning and practical retrieval.

### `app/retrieval/ann_index.py`

- Integrates exporter embeddings into nearest-neighbor search.
- Returns top matching exporter IDs and scores.

This file is a search accelerator. It allows the retrieval layer to be used efficiently at serving time.

### `app/retrieval/industry_rules.py`

- Integrates positive buyer-exporter history into cross-industry rules.
- Can backfill sparse history with semantic relations from the manual industry map.

This file is a bridge between learned behavior and business taxonomy. It tries to say, "buyers in industry A sometimes like exporters in related industry B."

### `app/ml/collaborative.py`

- Integrates swipe interactions into matrix factorization embeddings.
- Returns collaborative scores for buyer-exporter pairs.

This file adds behavioral taste signals that business features may miss.

### `app/ml/ncf.py`

- Integrates interaction data into a neural collaborative model.
- Returns a neural feedback score.

This file adds a deeper behavioral view than simple matrix factorization.

### `app/ml/supervised.py`

- Integrates pairwise features from the feature builder with labels from swipes or crossed datasets.
- Produces a probability-like score.

This file connects supervised classification logic to the shared pair feature space.

### `app/ml/ltr.py`

- Integrates pairwise features and buyer-level grouping into a ranking objective.
- Produces relative ordering signals rather than only binary probability.

This file focuses on ordering quality, not just classification quality. That matters because recommendation is mainly a ranking problem.

### `app/ml/hybrid_ranker.py`

- Integrates retrieval, supervised learning, LTR, collaborative models, NCF, rules, and confidence logic.
- Produces final ranked cards.

This file is the system integrator on the ML side. It is where all model parts are blended into one final answer.

## 7. Runtime `/feed` Workflow Example

- A client calls `/feed?buyer_id=...`.
- `app/main.py` checks that buyers are loaded and that the ranker exists.
- It finds the buyer row from the engineered buyer table.
- It calls `rank_for_buyer(...)` on the in-memory `HybridRanker`.
- The ranker asks the retriever for multi-source candidates.
- The candidate list may contain two-tower, SVD, NCF, and popularity-based sources.
- The candidate list is passed into `PairFeatureBuilder.candidate_features_for_buyer(...)`.
- The feature builder joins business, risk, history, retrieval, and text signals into a pairwise feature table.
- The hybrid ranker scores the table with supervised, LTR, collaborative, NCF, and wide-rule logic.
- The ranker builds card objects with reasons, confidence, source labels, penalties, and a verification checklist.
- `app/main.py` returns those cards as JSON.

This is the most important live workflow in the system. It shows how one request turns into a chain of integrations across many files. The API does not guess. The retriever narrows the list. The feature builder assembles the evidence. The hybrid ranker blends multiple models. Then the response is packaged for the user.

## 8. Runtime `/swipe` Workflow Example

- A client sends buyer ID, exporter ID, action, and context fields to `/swipe`.
- `app/main.py` validates the payload with Pydantic.
- `insert_swipe(...)` in `app/db.py` writes the row to the database.
- The DB layer updates buyer summary features.
- The DB layer returns the inserted row back to the API.
- `app/main.py` calls `ranker.ingest_swipe(...)`.
- The ranker appends the event to in-memory interaction history.
- The feature builder updates buyer, exporter, and pair behavior memory.
- The supervised model tries a single online update if it is using the SGD backend.
- After enough new swipes, the ranker runs a fuller refresh of retriever, collaborative, NCF, supervised, and LTR models.

This workflow is how the system learns after deployment. It is not fully online for every model, but it is partly online and partly batch-refresh. That is a practical compromise. Fully online training for every component would be more complex and more fragile.

## 9. News Update Workflow Example

- A client calls `/simulate/update`.
- `app/main.py` creates a fake high-risk news row.
- The row is appended to the in-memory news table.
- `ranker.refresh_news(news)` is called.
- The feature builder clears cached news penalties and starts using the new news table.
- The update is logged through `log_update(...)` in `app/db.py`.

This integration is simple but useful. It proves that risk is not frozen. News can change the live recommendation behavior without rebuilding the full ranker object.

## 10. Offline Training Workflow Example

- `scripts/train_ranker.py` loads cleaned data through the pipeline.
- It engineers buyer and exporter features.
- It loads swipe labels from CSV.
- It optionally loads crossed feature rows from CSV.
- It creates `HybridRanker(...)`.
- It calls `ranker.fit(swipes, crossed_features=...)`.
- It saves the trained ranker with pickle into `backend/models`.

This workflow reuses the same main ranker class as the live API. That is a strong integration choice because it keeps the training path close to the serving path.

## 11. Offline Evaluation Workflow Example

- `scripts/evaluate_ranker.py` loads swipe history.
- It splits the data by buyer into train and test parts.
- It can also align crossed features to the train portion.
- It trains a new `HybridRanker` on the train split.
- It builds ranking metrics like precision, recall, hit rate, MAP, and NDCG.
- It builds classification metrics like accuracy, balanced accuracy, and AUC.
- It also builds retrieval recall metrics at different candidate sizes.
- It saves the result through `metrics_store.py`.

This evaluation flow integrates model training with reporting. It does not only test one score type. It checks both retrieval quality and final ranking quality. That is important because a recommender can fail either at finding candidates or at ordering them.

## 12. Retrain Cycle Workflow Example

- `scripts/retrain_cycle.py` can ingest new swipe files.
- It merges them into the master swipe label file.
- If the master file is empty, it can auto-generate imitation labels.
- It runs `train_ranker.py`.
- It can run `evaluate_ranker.py`.
- It can rebuild crossed features through `build_crossed_dataset.py`.
- It can run `suggest_top_exporters.py` for a sample buyer after retraining.

This script is the best example of system-wide integration in one place. It ties together ingestion, training, evaluation, feature rebuilding, and preview generation. For a new developer, reading this script gives a fast picture of the full backend lifecycle.

## 13. What Data Is Getting Integrated at Each Stage

- Raw CSV stage:
  buyer company records, exporter company records, and news records.
- Clean data stage:
  normalized text, parsed dates, numeric fields, and filled missing values.
- Base feature stage:
  trust, intent, stability, certification, and basic business indicators.
- Pair feature stage:
  buyer and exporter base features plus industry fit, HS fit, country complement, capacity fit, communication score, risk, retrieval score, and behavior history.
- Retrieval stage:
  text vectors, interaction patterns, learned embeddings, and candidate-source signals.
- Ranking stage:
  supervised outputs, LTR outputs, collaborative outputs, NCF outputs, and rule-based wide signals.
- Feedback stage:
  swipe action, shown rank, dwell time, session ID, source, device, region, and recommendation version.
- Metrics stage:
  ranking metrics, classification metrics, retrieval metrics, and weight-tuning outputs.

This staged integration is one of the backend’s best qualities. Each stage adds a new kind of evidence. The final decision is not based on one source of truth. It is based on many small truths combined carefully.

## 14. Compatibility and Facade Integration

- `app/scoring.py` re-exports pipeline functions for older imports.
- `app/ml_pipeline.py` re-exports `HybridRanker`.
- These files help older code keep working even after the package was reorganized.

This is a small but important integration detail. The team did not break all old imports when the structure changed. They used facades to preserve compatibility. That is a sign of maintainability thinking.

## 15. Current Live vs Available Integration

- Live path uses text embeddings, two-tower retrieval, collaborative SVD, NCF, supervised learning, and LTR.
- Live path does not currently turn on teacher scoring.
- Live path does not currently attach graph scores.
- Live path does not currently attach industry association lookup to the feature builder.
- Live path uses retrieval blending from multiple candidate sources.
- Offline scripts and ablation artifacts show that richer paths were tested.

This difference matters. There is a gap between "implemented in code" and "active in live ranking." A new reader should not assume every advanced module is currently affecting production-like results. Some modules are available for experiments, future improvements, or controlled offline tests.

## 16. Critical Integration Notes

- Strong point: the same core ranker is reused in API serving and offline training.
- Strong point: pairwise feature logic is centralized.
- Strong point: retrieval and ranking are cleanly separated.
- Strong point: feedback storage is richer than a simple left/right table.
- Risk: startup training inside the API tightly couples serving with model initialization cost.
- Risk: pickled models are easy for prototypes but weak for large-scale deployment control.
- Risk: some useful advanced modules are disconnected from the live path.
- Risk: `config.py` is empty, so integration settings are scattered across constructors and scripts.

The integration quality is mostly good, but it still reflects prototype-stage priorities. The architecture is thoughtful, yet some wiring is still manual and spread out. That is normal in an MVP. It becomes a real issue only when the team wants stronger scaling, cleaner deployment, or more controlled experiments.

## 17. Simple End-to-End Example

Imagine a buyer opens the app and asks for recommendations. The frontend sends the buyer ID to `/feed`. The API finds that buyer inside the engineered buyer table. The ranker asks the retriever for likely exporters. The feature builder then joins the buyer row, exporter rows, recent news, swipe history, and retrieval outputs into one feature table. The ranker scores those rows with several models and rule layers. It returns ranked cards with safety notes and explanations.

Now imagine the buyer swipes right on one exporter. The frontend sends the swipe event to `/swipe`. The backend saves the event in Postgres, updates buyer summary stats, and pushes the same event into the in-memory ranker. That event updates pair history immediately and can improve later scoring. After enough new swipes, the system refreshes its learned models in a fuller way. This is the full loop: request, recommendation, feedback, learning, refresh.

## 18. Final Reading Order for a New Developer

- Start with `app/main.py` to see the live API flow.
- Read `app/db.py` to understand stored feedback.
- Read `app/pipeline/data_loader.py` and `feature_engineering.py` to understand the base data.
- Read `app/ml/feature_builder.py` because it is the main feature integration layer.
- Read `app/retrieval/text_encoder.py` and `two_tower.py` to understand candidate generation.
- Read `app/ml/supervised.py`, `ltr.py`, `collaborative.py`, and `ncf.py` to understand the ranking stack.
- Read `app/ml/hybrid_ranker.py` last to see how everything is blended.
- Read `scripts/retrain_cycle.py` to understand the whole offline lifecycle.

This reading order works because it follows the actual integration path. It starts at the system edge, moves into data and features, then moves into models, and ends at the orchestration layer. That is the easiest way to make the whole backend feel connected instead of confusing.
