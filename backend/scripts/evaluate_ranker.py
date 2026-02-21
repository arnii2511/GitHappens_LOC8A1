import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(root: str, path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _load_swipes(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Swipe CSV not found: {path}")
    df = pd.read_csv(path, engine="python")
    for col in ("buyer_id", "exporter_id", "action"):
        if col not in df.columns:
            raise ValueError(f"Missing required column in swipes CSV: {col}")
    if "ts" not in df.columns:
        df["ts"] = pd.Timestamp.utcnow()
    df = df[["buyer_id", "exporter_id", "action", "ts"]].copy()
    df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
    df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
    df["action"] = df["action"].astype(str).str.strip().str.lower()
    df = df[df["action"].isin(["left", "right"])]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    return df


def _split_train_test_by_buyer(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    ratio = float(np.clip(test_ratio, 0.05, 0.5))

    for _, g in df.groupby("buyer_id", sort=False):
        g = g.sort_values("ts")
        n = len(g)
        if n < 3:
            train_parts.append(g)
            continue
        n_test = max(1, int(round(n * ratio)))
        if n_test >= n:
            n_test = n - 1
        split_idx = n - n_test
        train_parts.append(g.iloc[:split_idx])
        test_parts.append(g.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def _precision_at_k(rels: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = rels[:k]
    if not top:
        return 0.0
    return float(np.mean(top))


def _recall_at_k(rels: List[int], positives_total: int, k: int) -> float:
    if positives_total <= 0:
        return 0.0
    return float(sum(rels[:k]) / positives_total)


def _average_precision_at_k(rels: List[int], k: int) -> float:
    top = rels[:k]
    hits = 0
    sum_prec = 0.0
    for i, r in enumerate(top, start=1):
        if r:
            hits += 1
            sum_prec += hits / i
    if hits == 0:
        return 0.0
    return float(sum_prec / hits)


def _ndcg_at_k(rels: List[int], positives_total: int, k: int) -> float:
    top = rels[:k]
    if not top:
        return 0.0

    dcg = 0.0
    for i, r in enumerate(top, start=1):
        if r:
            dcg += 1.0 / np.log2(i + 1.0)

    ideal_hits = min(int(positives_total), int(k))
    if ideal_hits <= 0:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 1.0) for i in range(1, ideal_hits + 1))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _score_specific_pair(ranker, buyer_id: str, exporter_id: str) -> float | None:
    if buyer_id not in ranker.builder.buyers_idx.index:
        return None
    buyer_row = ranker.builder.buyers_idx.loc[buyer_id]
    feature_df, _ = ranker.builder.candidate_features_for_buyer(buyer_row)
    if feature_df.empty:
        return None
    row = feature_df[feature_df["exporter_id"].astype(str) == str(exporter_id)]
    if row.empty:
        return None

    model_p = float(ranker.supervised.predict_proba(row)[0])
    collab_p = float(ranker.collaborative.score(str(buyer_id), np.array([str(exporter_id)], dtype=object))[0])
    collab_weight = float(ranker._adaptive_collab_weight(str(buyer_id)))  # noqa: SLF001
    blend = (1.0 - collab_weight) * model_p + collab_weight * collab_p

    ltr_p = float(ranker.ltr.score(row)[0])
    ltr_weight = float(ranker.ltr_weight if ranker.ltr.ready else 0.0)
    final_p = (1.0 - ltr_weight) * blend + ltr_weight * ltr_p
    return float(np.clip(final_p, 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranker metrics on swipe holdout data.")
    parser.add_argument("--swipes-csv", required=True, help="Swipe history CSV with buyer_id/exporter_id/action/ts.")
    parser.add_argument("--top-k", type=int, default=10, help="K for ranking metrics.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Holdout ratio per buyer.")
    parser.add_argument("--gpu", action="store_true", help="Prefer GPU.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--out-json", default=None, help="Optional JSON output path for metrics.")
    args = parser.parse_args()

    if args.gpu and args.cpu:
        raise ValueError("Use only one of --gpu or --cpu.")
    prefer_gpu = not args.cpu
    if args.gpu:
        prefer_gpu = True

    root = _project_root()
    sys.path.insert(0, root)
    swipes_path = _resolve_path(root, args.swipes_csv)

    from app.ml import HybridRanker
    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    swipes = _load_swipes(swipes_path)
    if swipes.empty:
        raise RuntimeError("Swipe CSV has no valid rows after cleaning.")

    train_df, test_df = _split_train_test_by_buyer(swipes, test_ratio=args.test_ratio)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Could not create a valid train/test split. Need more swipe history per buyer.")

    buyers_raw, exporters_raw, news = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)

    ranker = HybridRanker(buyers, exporters, news, prefer_gpu=prefer_gpu)
    ranker.fit(train_df)
    if not ranker.is_trained:
        raise RuntimeError("Training failed; ranker is not trained.")

    top_k = int(max(1, args.top_k))

    positives_by_buyer: Dict[str, set[str]] = (
        test_df[test_df["action"] == "right"]
        .groupby("buyer_id")["exporter_id"]
        .agg(lambda x: set(x.astype(str).tolist()))
        .to_dict()
    )

    ranking_rows = []
    for buyer_id, positives in positives_by_buyer.items():
        if not positives:
            continue
        if buyer_id not in ranker.builder.buyers_idx.index:
            continue
        buyer_row = ranker.builder.buyers_idx.loc[buyer_id]
        cards = ranker.rank_for_buyer(buyer_row, top_k=top_k)
        rec_ids = [str(c["exporter_id"]) for c in cards]
        rels = [1 if eid in positives else 0 for eid in rec_ids]

        ranking_rows.append(
            {
                "precision_at_k": _precision_at_k(rels, top_k),
                "recall_at_k": _recall_at_k(rels, len(positives), top_k),
                "hit_rate_at_k": 1.0 if any(rels) else 0.0,
                "map_at_k": _average_precision_at_k(rels, top_k),
                "ndcg_at_k": _ndcg_at_k(rels, len(positives), top_k),
            }
        )

    if not ranking_rows:
        raise RuntimeError("No evaluable buyers in holdout set (need right-swipe positives).")

    ranking_df = pd.DataFrame(ranking_rows)

    y_true: List[int] = []
    y_score: List[float] = []
    for _, r in test_df.iterrows():
        s = _score_specific_pair(ranker, str(r["buyer_id"]), str(r["exporter_id"]))
        if s is None:
            continue
        y_true.append(1 if str(r["action"]).lower() == "right" else 0)
        y_score.append(float(s))

    classification_metrics = {}
    if len(y_true) >= 2 and len(set(y_true)) == 2:
        y_pred = [1 if s >= 0.5 else 0 for s in y_score]
        classification_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_score)),
            "n_eval_pairs": int(len(y_true)),
        }
    else:
        classification_metrics = {
            "accuracy": None,
            "auc": None,
            "n_eval_pairs": int(len(y_true)),
        }

    metrics = {
        "split": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_ratio": float(args.test_ratio),
        },
        "model_backend": {
            "supervised": {"backend": ranker.supervised.backend, "device": ranker.supervised.device},
            "collaborative": {"backend": "svd", "device": ranker.collaborative.device, "ready": ranker.collaborative.ready},
            "ltr": {"backend": ranker.ltr.backend, "device": ranker.ltr.device},
        },
        "ranking_metrics": {
            "k": int(top_k),
            "precision_at_k": float(ranking_df["precision_at_k"].mean()),
            "recall_at_k": float(ranking_df["recall_at_k"].mean()),
            "hit_rate_at_k": float(ranking_df["hit_rate_at_k"].mean()),
            "map_at_k": float(ranking_df["map_at_k"].mean()),
            "ndcg_at_k": float(ranking_df["ndcg_at_k"].mean()),
            "n_eval_buyers": int(len(ranking_df)),
        },
        "classification_metrics": classification_metrics,
    }

    print(json.dumps(metrics, indent=2))

    if args.out_json:
        out_path = _resolve_path(root, args.out_json)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=True)
        print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
