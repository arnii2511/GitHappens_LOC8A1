# Quickstart Commands (New PC)

Run these from project root: `D:\swipe-to-export`

## 1. Create and activate virtual environment

```bash
python -m venv backend\.venv
backend\.venv\Scripts\activate
```

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
```

## 3. Optional: full GPU setup hint

```bash
pip install cupy-cuda12x
```

Use the CuPy package that matches your CUDA version (`cuda11x`, `cuda12x`, etc.).

Quick GPU readiness check:

```bash
python backend\scripts\check_gpu_stack.py
```

## 4A. Create synthetic swipe labels (if you do not have real swipe history)

```bash
python backend\scripts\generate_synthetic_swipes.py --rows 12000 --buyers-sample 1200 --per-buyer-min 8 --per-buyer-max 20 --days-back 700 --seed 42 --right-bias 0.1 --out-csv data/labels/swipes_labeled.csv
```

More realistic imitation (uses match/trust/risk/industry similarity + behavior history):

```bash
python backend\scripts\generate_imitation_swipes.py --rows 12000 --buyers-sample 1000 --per-buyer-min 8 --per-buyer-max 20 --days-back 700 --seed 42 --right-bias 0.0 --out-csv data/labels/swipes_labeled.csv
```

## 4B. Or create real labels manually (interactive)

```bash
python backend\scripts\collect_labels.py --target 300 --buyers 80 --cands-per-buyer 8 --gpu
```

## 5. Train the model

```bash
python backend\scripts\train_ranker.py --gpu --swipes-csv data/labels/swipes_labeled.csv --model-out models/ranker.pkl
```

Optional: train retrieval stack separately

```bash
python backend\scripts\train_retrieval.py --gpu --swipes-csv data/labels/swipes_labeled.csv --model-out models/retrieval.pkl
```

If you want strict GPU-only training check:

```bash
python backend\scripts\train_ranker.py --gpu --strict-gpu --swipes-csv data/labels/swipes_labeled.csv --model-out models/ranker.pkl
```

## 6. Evaluate model (accuracy + ranking metrics)

```bash
python backend\scripts\evaluate_ranker.py --swipes-csv data/labels/swipes_labeled.csv --top-k 10 --test-ratio 0.2 --gpu
```

Main metrics to read:
- `classification_metrics.accuracy`
- `classification_metrics.auc`
- `ranking_metrics.ndcg_at_k`
- `ranking_metrics.hit_rate_at_k`

## 7. Get top exporter suggestions for a buyer

Direct buyer id:

```bash
python backend\scripts\suggest_top_exporters.py --buyer-id BUY_69687 --top-k 10 --model-in models/ranker.pkl
```

Inspect only retrieval candidates (before reranking):

```bash
python backend\scripts\retrieve_candidates.py --buyer-id BUY_69687 --top-k 20 --model-in models/ranker.pkl
```

Inspect learned cross-industry association rules:

```bash
python backend\scripts\inspect_industry_rules.py --model-in models/ranker.pkl --top 8
```

Prompt mode (asks buyer id):

```bash
python backend\scripts\suggest_top_exporters.py --top-k 10 --model-in models/ranker.pkl
```

## 8. Profile dataset quality (optional)

```bash
python backend\scripts\profile_dataset.py --swipes-csv backend/data/labels/swipes_labeled.csv
```

## 9. Single-command retrain cycle (recommended for periodic updates)

First time (auto-generate imitation labels if empty, then train + evaluate + rebuild crossed features):

```bash
python backend\scripts\retrain_cycle.py --gpu --generate-if-empty --model-out models/ranker.pkl --metrics-out data/metrics/latest_metrics.json
```

When new swipe data arrives (append and retrain):

```bash
python backend\scripts\retrain_cycle.py --gpu --new-swipes-csv data/labels/new_swipes_batch.csv --model-out models/ranker.pkl --metrics-out data/metrics/latest_metrics.json
```

Use previously built crossed features during retrain (recommended):

```bash
python backend\scripts\retrain_cycle.py --gpu --new-swipes-csv data/labels/new_swipes_batch.csv --crossed-train-csv data/labels/cross_swipes_features.csv --model-out models/ranker.pkl --metrics-out data/metrics/latest_metrics.json
```

Optional: include a buyer id to immediately print latest top-K suggestions after retrain:

```bash
python backend\scripts\retrain_cycle.py --gpu --new-swipes-csv data/labels/new_swipes_batch.csv --buyer-id BUY_69687 --top-k 10
```
