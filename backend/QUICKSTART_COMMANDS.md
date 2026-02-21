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

## 4A. Create synthetic swipe labels (if you do not have real swipe history)

```bash
python backend\scripts\generate_synthetic_swipes.py --rows 12000 --buyers-sample 1200 --per-buyer-min 8 --per-buyer-max 20 --days-back 700 --seed 42 --right-bias 0.1 --out-csv data/labels/swipes_labeled.csv
```

## 4B. Or create real labels manually (interactive)

```bash
python backend\scripts\collect_labels.py --target 300 --buyers 80 --cands-per-buyer 8 --gpu
```

## 5. Train the model

```bash
python backend\scripts\train_ranker.py --gpu --swipes-csv data/labels/swipes_labeled.csv --model-out models/ranker.pkl
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

Prompt mode (asks buyer id):

```bash
python backend\scripts\suggest_top_exporters.py --top-k 10 --model-in models/ranker.pkl
```

