# ML and ML-Related Backend Theory Document

This document explains the machine learning system and the ML-related backend in this project. It is written for someone who wants to understand what the system is trying to do, how the parts connect, and why the team chose some strategies while avoiding others. It uses simple English and focuses on ideas, flow, intent, strengths, limits, and reasoning.

## 1. Main Goal

- The project tries to match buyers with exporters.
- The system does not trust one signal alone.
- It mixes business rules, learned behavior, retrieval, ranking, and risk control.
- The final aim is to show buyer-specific exporter suggestions that feel relevant, safer, and easier to explain.
- The backend is designed to support both offline model training and live recommendation serving.

The project follows a layered design. Clean data is loaded first, then static business features are built, then interaction history is used to train retrieval and ranking models, and finally a FastAPI app serves ranked exporter cards. This is important because each layer solves a different problem. Data cleaning improves trust in the inputs. Retrieval narrows a very large search space. Ranking decides the final order. The API turns all of this into a usable service.

## 2. High-Level Architecture

- `data/raw` stores the original hackathon CSV files.
- `data/clean` stores cleaned copies and the data quality report.
- `backend/app/pipeline` handles data loading, feature engineering, risk logic, helper logic, dynamic weighting, and the older rule-based ranking path.
- `backend/app/ml` handles the learned ranking stack.
- `backend/app/retrieval` handles text embeddings, dual-encoder retrieval, ANN search, and industry association mining.
- `backend/app/main.py` is the online API entry point.
- `backend/app/db.py` stores swipe history and update logs in Postgres.
- `backend/scripts` runs the offline workflow for cleaning, label generation, training, evaluation, retraining, inspection, and utility checks.
- `backend/models` stores trained model artifacts.
- `backend/data/labels`, `backend/data/cache`, and `backend/data/metrics` store training labels, teacher cache, and metric outputs.

The architecture is hybrid by design. A pure rule engine would be easy to explain, but it would not adapt well to user behavior. A pure deep learning system would sound modern, but it would be harder to trust with noisy hackathon-style data and much harder to debug. This project sits in the middle. It keeps business logic where business logic is useful, and it uses learned models where patterns are too complex to write by hand.

## 3. Data Layer and Input Assets

- Buyers come from `buyers_clean.csv`.
- Exporters come from `exporters_clean.csv`.
- News signals come from `news_clean.csv`.
- The cleaning report lives in `data_quality_report.json`.
- Swipe labels for training usually live in `backend/data/labels/swipes_labeled.csv`.
- Crossed pairwise feature training data usually lives in `backend/data/labels/cross_swipes_features.csv`.
- Cached teacher scores live in `backend/data/cache/teacher_scores_v1.pkl`.
- Latest evaluation output lives in `backend/data/metrics/latest_metrics.json`.

The data shows why the system uses careful fallback logic. The quality report says buyers have missing values in fields like average order tons and preferred channel. Exporters also have missing values in fields like certification and shipment value. There are many duplicate IDs in buyers and exporters. Because of this, the system cannot act like the raw data is complete or perfectly clean. It must normalize text, coerce numbers, fill some missing values, and avoid brittle logic that breaks when a column is weak.

## 4. Why the Cleaning Strategy Looks Conservative

- The cleaning step normalizes text instead of dropping large parts of the data.
- Industry text is lowered and cleaned because many later steps depend on industry matching.
- Dates are parsed early so both training and live scoring can use time-based logic.
- Numeric fields are coerced carefully so later features do not crash on bad text values.
- Buyers keep median-based filling for `Response_Probability` instead of forcing everything to zero.
- Exporter capacity uses industry median filling because capacity is important for matching.
- The cleaning script avoids heavy row deletion because the dataset is valuable even when some fields are missing.
- A report is saved after cleaning so the team can inspect data health instead of guessing.

This is a practical choice. The project is built around tabular trade data, not around perfect enterprise data pipelines. If the team had dropped every weak row, they would have lost too much coverage. If they had filled everything with aggressive fake values, they would have created false certainty. The current cleaning style tries to keep as much signal as possible while staying cautious.

## 5. Static Feature Engineering

- Buyer features include a certification score, a stability proxy, buyer trust, and buyer intent.
- Exporter features include a certification score, a stability proxy, exporter trust, and exporter intent.
- Trust features mainly use payment behavior, response quality, certification, and size stability.
- Intent features mainly use explicit intent score, growth signals, response signals, and activity proxies.
- Trust values are clipped into a bounded range so one noisy column does not explode the score.
- Intent values are also clipped so the system stays stable.

These features are simple on purpose. They are not meant to be the full intelligence of the system. They are stable baseline signals that can support later models. This is a good design for a small project because it gives the ML layers a cleaner starting point. It also makes explanations easier. A human can understand why payment history, response quality, and hiring activity matter in trade matching.

## 6. Industry Logic and Risk Logic

- `industry_map.py` turns raw industry names into canonical groups like healthcare, engineering, automotive, electronics, IT, energy, textiles, and chemicals.
- The industry logic supports exact matches and related matches.
- Related matches are weighted, not treated as equal.
- This lets the system say that engineering and automotive are close, but not identical.
- `risk.py` reads recent news events and builds a penalty from tariff change, stock shock, war, natural calamity, and currency shift.
- The news penalty is limited so risk can influence ranking without fully destroying every candidate.
- Exporters also get their own risk penalty from exporter-side risk columns.

This part shows a strong business-first design. Trade matching is not only about similarity. It is also about context. A buyer may look like a good fit for an exporter on paper, but live news can raise market risk. That is why the project keeps risk as a visible penalty instead of hiding it inside a black-box model. This improves explainability and makes the system safer for a domain where outside events matter.

## 7. Dynamic Weights and Verification

- Dynamic weights adjust the importance of capacity fit, intent, and communication score.
- If buyer order quantity is missing, the model leans less on capacity fit.
- If risk is high, the model shifts a little weight away from intent and toward safer practical checks.
- Verification checklists create action items for documents, payment safety, quality checks, and must-do checks.
- High risk adds stricter document and payment requirements.
- Low trust adds stronger verification steps.
- WhatsApp preference triggers a warning to move important terms into email.

This is a smart choice for explainable product behavior. Instead of only saying "the score is lower," the system also suggests what a user should verify. That makes the ranking output more useful in the real world. It turns the system from a simple recommendation engine into a light decision-support tool.

## 8. Online Backend Flow

- On startup, the API initializes the database.
- It loads the cleaned buyers, exporters, and news files.
- It engineers buyer and exporter features.
- It creates a `HybridRanker`.
- It fetches up to 250,000 recent swipe events from the database.
- It trains the ranker in memory.
- `/health` checks service health.
- `/buyers` lists buyer records with optional search.
- `/feed` returns ranked exporter cards for a buyer.
- `/swipe` saves a swipe event and updates the in-memory learning state.
- `/simulate/update` adds a fake high-risk news event and refreshes news-based risk.

The startup training pattern is easy to understand and easy to demo, which fits a prototype well. It avoids needing a separate model-serving system. The downside is that startup can become heavy as the project grows. This means the current design is good for a hackathon or early prototype, but it may need to change if the user base or history size becomes much larger.

## 9. Database Design and Feedback Storage

- Swipe events are stored in a `swipes` table.
- The database keeps fields beyond left and right, such as session ID, shown rank, source, dwell time, device, region, and recommendation version.
- A unique index on buyer, exporter, and session helps block accidental duplicates.
- Buyer profile summary features are refreshed after insert.
- A signal update log tracks simulated updates.
- A pipeline metrics table can store evaluation metrics in JSON form.

This design shows that the team was thinking ahead. They did not only store the swipe label. They also stored context around the swipe. That matters because recommendation systems improve when they know where a card came from, how long it was viewed, and which session it belonged to. Even if all of those fields are not fully used today, they create room for better training later.

## 10. Pair Feature Builder

- The pair feature builder is the heart of the ranking stack.
- It prepares buyer and exporter tables and indexes them by ID.
- It extracts HS code tokens when available.
- It computes industry similarity, HS match score, country complement score, capacity fit, intent fit, pair trust, communication score, risk penalties, and behavioral features.
- It tracks buyer history, exporter history, and pair history.
- It remembers right-swipe rates, recency, dwell behavior, and last shown rank.
- It can ingest new interactions one by one.
- It can build features for a single buyer-exporter pair or for a whole candidate set.

The builder is important because it creates a shared language for the rest of the ML stack. Supervised models, ranking models, and explainers all need a consistent feature space. Without this layer, each model would end up inventing its own data logic, and the system would become hard to maintain. Centralizing pairwise feature construction is one of the strongest design decisions in the backend.

## 11. Candidate Generation Strategy

- The system does not rank all exporters for every buyer.
- It first creates a candidate pool.
- It builds a strong core using industry similarity above a threshold.
- It also keeps an exploration pool so the system does not become too narrow.
- A quick pre-score uses industry similarity, capacity fit, exporter intent, and exporter trust.
- The core gets most of the budget.
- A smaller share of the budget is saved for exploration.
- Retrieval candidates can also be injected into the feature builder.
- Candidate source labels such as `core`, `explore`, and retrieval-based sources are tracked.

This is a strong compromise between precision and discovery. If the system only used safe industry matches, it would miss new opportunities. If it explored too much, users would see weak suggestions. The chosen approach says: keep a reliable center, but reserve room for controlled exploration. That is a common and practical recommendation-system pattern.

## 12. Retrieval Layer

- The text encoder creates buyer and exporter text representations.
- It prefers sentence-transformer embeddings when available.
- It falls back to TF-IDF when that stack is unavailable.
- A teacher cross-encoder is also implemented for stronger pair scoring.
- Teacher scores can be cached to disk.
- The two-tower retriever learns buyer and exporter embeddings.
- An ANN index searches the exporter embeddings quickly.
- An industry association miner can learn cross-industry links from positive history.

The retrieval layer exists because ranking every exporter for every buyer is expensive and unnecessary. Retrieval answers the question, "Which exporters are even worth thinking about?" Ranking answers the question, "Which of those should be shown first?" This two-stage setup is a smart design because it scales better and lets the project mix fast search with richer reranking.

## 13. Why Retrieval Uses More Than One Idea

- Text embeddings help with semantic similarity.
- Two-tower retrieval helps learn from interaction behavior.
- ANN search helps speed up lookup after embeddings are learned.
- Industry association mining helps when strict industry matching is too narrow.
- Hard negatives are implemented to teach the retriever to reject confusing exporters.
- LogQ correction is implemented to reduce popularity bias from sampled negatives.
- Distillation is implemented so a stronger teacher model can guide a cheaper retriever.

This layered retrieval design is ambitious. It shows the team wanted to move beyond old directory-style matching. At the same time, the live `HybridRanker` currently turns off several advanced retrieval options. In the live constructor, teacher mode is disabled, and the two-tower retriever is created with hard negatives, LogQ correction, and distillation turned off. This tells us the team values stability and speed in the current path more than full feature richness.

## 14. Collaborative and Deep Feedback Models

- The collaborative model uses an SVD-style factorization on the buyer-exporter interaction matrix.
- Right swipes act like positive signal.
- Left swipes still count, but as weaker negative signal.
- Time decay reduces the weight of older interactions.
- The neural collaborative filtering model uses learned buyer and exporter embeddings with a small neural network.
- NCF trains only when enough interactions exist.
- Graph feature learning is implemented in a separate service using interaction graphs and industry smoothing.

These models exist because user behavior can reveal patterns that static business features cannot see. Some buyers simply prefer certain exporter profiles in ways that a hand-built rule never fully captures. Collaborative models are useful for this. The problem is that collaborative methods also become unstable when data is sparse. That is why this project does not trust collaborative signals alone.

## 15. Supervised and Learning-to-Rank Layers

- The supervised model can use GPU XGBoost or CPU SGD.
- The LTR model can use XGBoost or LightGBM with ranking objectives.
- Both models use a shared feature list.
- Both models use class-balance weighting because swipe data is imbalanced.
- Both models use time decay because old actions matter less.
- Both models use PU-style denoising to reduce the damage from noisy negative labels.
- If real labels are weak, both models can bootstrap from rule-based content scores.
- The LTR model runs a small parameter search instead of using only one fixed setup.

This is one of the clearest examples of practical engineering. The project accepts that a left swipe is not always a true dislike. Sometimes it is noise, timing, or weak exposure. PU-style weighting is used to reflect that. The project also keeps multiple training backends so it can still work on weaker machines. That is a mature choice for a prototype that may run in many environments.

## 16. Final Hybrid Ranking Logic

- The final ranker mixes deep signals, wide signals, sequence signals, and text signals.
- Deep signals include supervised score, LTR score, retrieval score, collaborative score, NCF score, and a little text score.
- Wide signals include match-after-risk, trust, intent, industry similarity, HS match, communication score, and some collaborative signals.
- Sequence signals summarize recent pair and buyer behavior.
- A popularity penalty reduces the chance that globally popular exporters dominate the results.
- A cold-start boost uses HS and country compatibility when buyer history is weak.
- Confidence is estimated from model agreement and score margin.
- The system returns a card with scores, risk, reasons, dynamic weights, behavioral features, source labels, and a verification checklist.

This hybrid ranker is the center of the project’s design philosophy. It assumes no single model is good enough on its own. The result is more complex than a simple baseline, but the complexity has a clear reason. Every extra component fixes a real weakness: retrieval finds candidates, ranking orders them, collaborative models add taste, text adds semantic similarity, and business rules keep the output grounded.

## 17. Why Some Advanced Features Were Not Fully Used

- The graph feature service exists, but the live ranker currently disables graph scoring.
- The teacher cross-encoder exists, but the live ranker currently disables teacher mode in the text service.
- The live retriever turns off hard negatives, LogQ correction, and distillation even though the retriever code supports them.
- Industry association lookup is set to `None` in the live ranker flow.
- Adaptive collaborative weighting is present as a method, but currently returns zero.
- `backend/app/config.py` is empty, so configuration is mostly handled directly in constructors and environment lookups.

These choices likely came from prototype pressure. Advanced features increase compute cost, code paths, and debugging effort. They can also make training slower and harder to trust when labels are synthetic or sparse. In a hackathon or MVP setting, it is reasonable to keep the richer logic available in the codebase but disable it in the live path until the team has enough evidence and enough clean feedback data.

## 18. Label Strategy and Offline Workflow

- The project supports manual labels through an interactive script.
- It supports simple synthetic labels.
- It also supports imitation labels that are more realistic because they use match, trust, risk, industry fit, and behavior history.
- Crossed datasets can be built so each historical buyer-exporter action gets a full feature row.
- Separate scripts exist for ranker training, retrieval training, evaluation, candidate inspection, and retraining cycles.
- A single retrain cycle can ingest new swipe files, train, evaluate, rebuild crossed data, and print fresh suggestions.

This is a thoughtful training workflow for a small project. Real labeled data is often scarce at the start, so the system uses synthetic and imitation labels to avoid having no training data at all. That is not perfect, but it is practical. The critical point is that the team did not pretend synthetic labels were equal to real labels. They built manual labeling and retraining paths so the system can improve later.

## 19. Why These Labeling Choices Make Sense

- Manual labels are highest quality but slow and expensive.
- Synthetic labels are fast but may teach the model the same bias as the rule logic.
- Imitation labels are a middle path because they use more realistic pair scoring and some behavior simulation.
- Crossed feature datasets help the supervised and LTR models learn from richer pair context.
- The retrain cycle exists because recommendation systems improve when new feedback is folded back in often.

The team chose breadth over purity here. That is understandable. A recommendation system without labels cannot really learn. A project with only manual labels may never get enough volume. So the system starts with bootstrapping tools and leaves room for gradual improvement. This is a common and reasonable early-stage ML strategy.

## 20. Current Measured Behavior

- The latest saved evaluation run is dated February 25, 2026.
- The evaluated backend used GPU XGBoost for supervised learning and LTR.
- Collaborative SVD, NCF, and two-tower retrieval were also GPU-ready in that run.
- The text encoder used sentence-transformer embeddings.
- Teacher mode and graph mode were not active in that saved latest run.
- Ranking metrics at top 10 were still modest.
- Retrieval recall was much stronger than final ranking precision.

This result is very informative. The retrieval layer seems much better at finding relevant exporters somewhere in a larger set than the final ranker is at pushing the best ones into the top few positions. That means the biggest remaining challenge is not only search. It is better reranking, better labels, and better alignment between training targets and real user value.

## 21. Key Metric Snapshot

- Precision@10 in the latest metrics file is about 0.0223.
- Recall@10 is about 0.1291.
- Hit rate@10 is about 0.2069.
- NDCG@10 is about 0.0504.
- Retrieval recall@50 is about 0.4157.
- Retrieval recall@100 is about 0.4936.
- Retrieval recall@500 is about 0.6974.
- Retrieval recall@1000 is about 0.7831.
- Classification accuracy is about 0.6125.
- Balanced accuracy is about 0.5792.
- AUC is about 0.5827.
- Weight tuning did run and selected a slightly deep-heavy blend.

The numbers support a clear reading. The system can often pull relevant items into the candidate set, but the very top results still need improvement. In simple words, the project is better at finding possible matches than it is at choosing the best ten. This matches the general behavior of many early recommendation systems.

## 22. What the Ablation File Suggests

- A plain baseline performed weakly.
- Hard negatives alone did not change much.
- Adding LogQ correction helped ranking a little.
- Adding teacher distillation helped a little more.
- Adding graph features helped the most among the tested changes.
- Even with those gains, the absolute ranking numbers stayed modest.

This tells us the project’s direction is not random. The advanced ideas did help in controlled tests, especially graph-aware signals. But the gains were not so large that they forced immediate production use. That likely explains why the codebase contains richer research-style features while the live constructor still uses a safer setup.

## 23. Critical Analysis of the Design

The design is strongest when it behaves like a careful system of systems. It respects business context, user feedback, and compute limits at the same time. The project avoids the beginner mistake of trusting a single model. It also avoids the opposite beginner mistake of staying fully rule-based forever.

- Good choice: using a two-stage system with retrieval plus reranking.
- Good choice: keeping explainable business features alongside learned signals.
- Good choice: using time decay for stale interactions.
- Good choice: storing rich swipe context like dwell time and shown rank.
- Good choice: using fallback backends so the project can still run without full GPU support.
- Good choice: using verification checklists because trade matching involves safety and fraud concerns.
- Weak point: the live system disables some advanced features that seem promising in the ablation results.
- Weak point: the final ranking quality is still low for top-10 usefulness.
- Weak point: synthetic and imitation labels can teach the model the same biases already built into the rules.
- Weak point: startup training inside the API will become expensive at larger scale.
- Weak point: an empty `config.py` means configuration is spread out instead of centralized.

## 24. Why Some Simpler Alternatives Were Not Used

- A pure keyword search system was not used because trade matching needs more than text overlap.
- A pure industry exact-match system was not used because related industries can still be valid.
- A pure heuristic ranking system was not used because user behavior should change future ranking.
- A pure collaborative filtering system was not used because new buyers and sparse data would hurt it badly.
- A pure deep end-to-end system was not used because data quality, label quality, explainability, and debugging cost would all become harder.
- A full graph-first production system was not used because it adds complexity and its gains still need stronger validation.

The rejected alternatives make sense. Each one solves one problem well but leaves another problem exposed. The chosen hybrid design is not the simplest design, but it is the most balanced design for the dataset, team stage, and product goal shown in this project.

## 25. What a New Reader Should Understand First

- This is not one model. It is a pipeline with many parts.
- The pair feature builder is the central shared layer.
- Retrieval is for narrowing the search space.
- Ranking is for ordering the final cards.
- Business rules still matter because trade data is noisy and risk-sensitive.
- Feedback loops matter because buyer behavior changes the system over time.
- Some advanced modules exist in the codebase but are not fully active in the live path.
- Evaluation files matter because they reveal where the system is strong and weak.

If someone new opens this backend, they should read it as an evolving recommendation platform, not as a finished single-model product. The current codebase already contains a research path, a production-like path, and a prototype path at the same time. That is why this document focuses so much on reasoning. The reasoning is what keeps the project understandable even when the implementation is broad.

## 26. Final Summary

- The project uses clean trade data, live news risk, and swipe history to recommend exporters to buyers.
- The backend combines API serving, database logging, feature engineering, retrieval, ranking, and retraining support.
- The strongest architectural idea is the hybrid design.
- The biggest current weakness is low top-rank quality even when candidate retrieval is decent.
- The most useful next improvements would likely come from better real labels, better production use of proven advanced features, and cleaner configuration management.
- The codebase already shows clear thought about scale, explainability, and fallback behavior.

This project is a serious prototype. It is not a toy rules script, and it is not a fully mature production recommender either. It sits in the middle. That middle position explains almost every design choice in the backend: use strong theory, keep practical fallbacks, measure often, and do not trust any one signal too much.
