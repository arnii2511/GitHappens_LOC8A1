import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


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


def _load_crossed(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Crossed CSV not found: {path}")
    df = pd.read_csv(path, engine="python")
    if "buyer_id" not in df.columns or "exporter_id" not in df.columns:
        raise ValueError("Crossed CSV must include buyer_id and exporter_id.")
    if "action" not in df.columns:
        if "label" in df.columns:
            df["action"] = np.where(pd.to_numeric(df["label"], errors="coerce").fillna(0) > 0, "right", "left")
        else:
            raise ValueError("Crossed CSV must include action or label.")
    if "ts" not in df.columns:
        df["ts"] = pd.Timestamp.utcnow()
    df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
    df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
    df["action"] = df["action"].astype(str).str.strip().str.lower()
    df = df[df["action"].isin(["left", "right"])]
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    return df


def _subset_crossed_by_swipes(crossed: pd.DataFrame, swipes: pd.DataFrame) -> pd.DataFrame:
    if crossed is None or crossed.empty or swipes is None or swipes.empty:
        return pd.DataFrame()

    c = crossed.copy()
    s = swipes.copy()

    c["ts_key"] = c["ts"].dt.floor("s").astype(str)
    s["ts_key"] = s["ts"].dt.floor("s").astype(str)
    c["k_full"] = c["buyer_id"].astype(str) + "|" + c["exporter_id"].astype(str) + "|" + c["action"].astype(str) + "|" + c["ts_key"]
    s["k_full"] = s["buyer_id"].astype(str) + "|" + s["exporter_id"].astype(str) + "|" + s["action"].astype(str) + "|" + s["ts_key"]
    keys_full = set(s["k_full"].tolist())
    take = c[c["k_full"].isin(keys_full)].copy()
    if not take.empty:
        return take.drop(columns=["ts_key", "k_full"], errors="ignore").reset_index(drop=True)

    # Fallback when timestamp formats differ: match without ts.
    c["k_pair"] = c["buyer_id"].astype(str) + "|" + c["exporter_id"].astype(str) + "|" + c["action"].astype(str)
    s["k_pair"] = s["buyer_id"].astype(str) + "|" + s["exporter_id"].astype(str) + "|" + s["action"].astype(str)
    keys_pair = set(s["k_pair"].tolist())
    take = c[c["k_pair"].isin(keys_pair)].copy()
    return take.drop(columns=["ts_key", "k_full", "k_pair"], errors="ignore").reset_index(drop=True)


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


def _score_specific_pair(ranker, buyer_id: str, exporter_id: str, cache: Dict[str, pd.DataFrame] | None = None) -> float | None:
    if buyer_id not in ranker.builder.buyers_idx.index:
        return None
    scored_df = None
    if cache is not None:
        scored_df = cache.get(str(buyer_id))
    if scored_df is None:
        buyer_row = ranker.builder.buyers_idx.loc[buyer_id]
        retrieval_candidates = None
        if hasattr(ranker, "multi_source_candidates_for_buyer"):
            retrieval_candidates = ranker.multi_source_candidates_for_buyer(str(buyer_id), top_k=600)
        elif getattr(ranker, "retriever", None) is not None and ranker.retriever.ready:
            retrieval_candidates = ranker.retriever.retrieve_for_buyer(str(buyer_id), top_k=400)
        feature_df, _ = ranker.builder.candidate_features_for_buyer(buyer_row, retrieval_candidates=retrieval_candidates)
        if feature_df is None or feature_df.empty:
            return None
        if hasattr(ranker, "score_feature_df"):
            final = np.asarray(ranker.score_feature_df(feature_df), dtype=np.float64)
        else:
            final = np.asarray(ranker.supervised.predict_proba(feature_df), dtype=np.float64)
        scored_df = feature_df[["exporter_id"]].copy()
        scored_df["final_score"] = np.clip(final, 0.0, 1.0)
        if cache is not None:
            cache[str(buyer_id)] = scored_df
    if scored_df.empty:
        return None
    row = scored_df[scored_df["exporter_id"].astype(str) == str(exporter_id)]
    if row.empty:
        return None

    final_p = float(pd.to_numeric(row.iloc[0].get("final_score", np.nan), errors="coerce"))
    if not np.isfinite(final_p):
        return None
    return float(np.clip(final_p, 0.0, 1.0))


def _collect_pair_scores(ranker, df: pd.DataFrame, max_rows: int = 5000) -> Tuple[List[int], List[float]]:
    if df.empty:
        return [], []
    part = df
    if len(part) > max_rows:
        part = part.sample(max_rows, random_state=42)
    y_true: List[int] = []
    y_score: List[float] = []
    cache: Dict[str, pd.DataFrame] = {}
    for _, r in part.iterrows():
        s = _score_specific_pair(ranker, str(r["buyer_id"]), str(r["exporter_id"]), cache=cache)
        if s is None:
            continue
        y_true.append(1 if str(r["action"]).lower() == "right" else 0)
        y_score.append(float(s))
    return y_true, y_score


def _find_best_threshold(y_true: List[int], y_score: List[float]) -> float:
    if len(y_true) < 20 or len(set(y_true)) < 2:
        return 0.5
    best_thr = 0.5
    best_metric = -1.0
    for thr in np.linspace(0.25, 0.75, num=51):
        y_pred = [1 if s >= float(thr) else 0 for s in y_score]
        metric = float(balanced_accuracy_score(y_true, y_pred))
        if metric > best_metric:
            best_metric = metric
            best_thr = float(thr)
    return best_thr


def _retrieval_metrics(ranker, positives_by_buyer: Dict[str, set[str]]) -> Dict[str, object]:
    ks = [50, 100, 500, 1000]
    has_multi = hasattr(ranker, "multi_source_candidates_for_buyer")
    if (not has_multi) and (getattr(ranker, "retriever", None) is None or not ranker.retriever.ready):
        return {
            "k_values": ks,
            "n_eval_buyers": 0,
            "avg_candidate_count": 0.0,
            "recall_at_50": 0.0,
            "recall_at_100": 0.0,
            "recall_at_500": 0.0,
            "recall_at_1000": 0.0,
            "candidate_source_mix": {},
        }

    max_k = max(ks)
    rows = []
    source_counts: Dict[str, int] = {}
    for buyer_id, positives in positives_by_buyer.items():
        if not positives:
            continue
        if buyer_id not in ranker.builder.buyers_idx.index:
            continue
        if has_multi:
            rec = ranker.multi_source_candidates_for_buyer(str(buyer_id), top_k=max_k)
        else:
            rec = ranker.retriever.retrieve_for_buyer(str(buyer_id), top_k=max_k)
        if rec is None or rec.empty:
            continue
        ids = rec["exporter_id"].astype(str).tolist()
        pos = set(str(x) for x in positives)
        row = {
            "candidate_count": float(len(ids)),
        }
        for k in ks:
            top = ids[:k]
            hit = len(set(top) & pos)
            row[f"recall_at_{k}"] = float(hit / max(1, len(pos)))
        rows.append(row)

        if "candidate_source" in rec.columns:
            vc = rec["candidate_source"].astype(str).value_counts()
            for src, cnt in vc.items():
                source_counts[str(src)] = int(source_counts.get(str(src), 0) + int(cnt))

    if not rows:
        return {
            "k_values": ks,
            "n_eval_buyers": 0,
            "avg_candidate_count": 0.0,
            "recall_at_50": 0.0,
            "recall_at_100": 0.0,
            "recall_at_500": 0.0,
            "recall_at_1000": 0.0,
            "candidate_source_mix": {},
        }

    rdf = pd.DataFrame(rows)
    total_src = float(sum(source_counts.values()))
    src_mix = {}
    if total_src > 0:
        src_mix = {k: float(v / total_src) for k, v in sorted(source_counts.items(), key=lambda x: (-x[1], x[0]))}

    return {
        "k_values": ks,
        "n_eval_buyers": int(len(rdf)),
        "avg_candidate_count": float(rdf["candidate_count"].mean()),
        "recall_at_50": float(rdf["recall_at_50"].mean()),
        "recall_at_100": float(rdf["recall_at_100"].mean()),
        "recall_at_500": float(rdf["recall_at_500"].mean()),
        "recall_at_1000": float(rdf["recall_at_1000"].mean()),
        "candidate_source_mix": src_mix,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranker metrics on swipe holdout data.")
    parser.add_argument("--swipes-csv", required=True, help="Swipe history CSV with buyer_id/exporter_id/action/ts.")
    parser.add_argument("--crossed-csv", default=None, help="Optional crossed feature CSV used only for train split.")
    parser.add_argument("--top-k", type=int, default=10, help="K for ranking metrics.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Holdout ratio per buyer.")
    parser.add_argument("--disable-weight-tuning", action="store_true", help="Disable precision@10 weight tuning.")
    parser.add_argument("--tune-eval-buyers", type=int, default=120, help="Buyers sampled for weight tuning.")
    parser.add_argument(
        "--online-refresh-every",
        type=int,
        default=120,
        help="After these many new swipes, run full refresh of retrieval/collab/LTR models.",
    )
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
    crossed_path = _resolve_path(root, args.crossed_csv)

    from app.ml import HybridRanker
    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    swipes = _load_swipes(swipes_path)
    if swipes.empty:
        raise RuntimeError("Swipe CSV has no valid rows after cleaning.")

    train_df, test_df = _split_train_test_by_buyer(swipes, test_ratio=args.test_ratio)
    if train_df.empty or test_df.empty:
        raise RuntimeError("Could not create a valid train/test split. Need more swipe history per buyer.")
    crossed_df = _load_crossed(crossed_path)
    train_crossed = _subset_crossed_by_swipes(crossed_df, train_df)

    buyers_raw, exporters_raw, news = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)

    ranker = HybridRanker(
        buyers,
        exporters,
        news,
        prefer_gpu=prefer_gpu,
        auto_tune_weights=(not args.disable_weight_tuning),
        tune_eval_buyers=int(max(30, args.tune_eval_buyers)),
        online_full_refresh_every=int(max(20, args.online_refresh_every)),
    )
    ranker.fit(train_df, crossed_features=train_crossed)
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
    retrieval_metrics = _retrieval_metrics(ranker, positives_by_buyer)

    train_y_true, train_y_score = _collect_pair_scores(ranker, train_df, max_rows=3000)
    best_threshold = _find_best_threshold(train_y_true, train_y_score)

    y_true, y_score = _collect_pair_scores(ranker, test_df, max_rows=8000)

    classification_metrics = {}
    if len(y_true) >= 2 and len(set(y_true)) == 2:
        y_pred = [1 if s >= best_threshold else 0 for s in y_score]
        classification_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "auc": float(roc_auc_score(y_true, y_score)),
            "threshold": float(best_threshold),
            "n_eval_pairs": int(len(y_true)),
        }
    else:
        classification_metrics = {
            "accuracy": None,
            "balanced_accuracy": None,
            "auc": None,
            "threshold": float(best_threshold),
            "n_eval_pairs": int(len(y_true)),
        }

    metrics = {
        "split": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_ratio": float(args.test_ratio),
            "crossed_train_rows": int(len(train_crossed)),
        },
        "interaction_metrics": {
            "train_right_swipe_rate": float((train_df["action"] == "right").mean()) if not train_df.empty else 0.0,
            "test_right_swipe_rate": float((test_df["action"] == "right").mean()) if not test_df.empty else 0.0,
        },
        "model_backend": {
            "supervised": {"backend": ranker.supervised.backend, "device": ranker.supervised.device},
            "collaborative": {
                "backend": getattr(ranker.collaborative, "backend", "none"),
                "device": ranker.collaborative.device,
                "ready": ranker.collaborative.ready,
            },
            "ncf": {
                "backend": getattr(ranker, "ncf", None).backend if hasattr(ranker, "ncf") else "none",
                "device": getattr(ranker, "ncf", None).device if hasattr(ranker, "ncf") else "none",
                "ready": bool(getattr(ranker, "ncf", None).ready) if hasattr(ranker, "ncf") else False,
            },
            "ltr": {"backend": ranker.ltr.backend, "device": ranker.ltr.device},
            "retrieval": {"backend": ranker.retriever.backend, "device": ranker.retriever.device, "ready": ranker.retriever.ready},
            "text_encoder": {
                "backend": ranker.text_encoder.backend,
                "ready": ranker.text_encoder.ready,
                "teacher_backend": getattr(ranker.text_encoder, "teacher_backend", "none"),
                "teacher_ready": bool(getattr(ranker.text_encoder, "teacher_ready", False)),
                "teacher_cache_loaded_entries": int(getattr(ranker.text_encoder, "teacher_cache_loaded_entries", 0)),
            },
            "graph": {
                "backend": getattr(ranker.graph, "backend", "none"),
                "ready": bool(getattr(ranker.graph, "ready", False)),
            },
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
        "retrieval_metrics": retrieval_metrics,
        "classification_metrics": classification_metrics,
        "weight_tuning": {
            "ran": bool(getattr(ranker, "tuning_info", {}).get("ran", False)),
            "best_precision_at_10": getattr(ranker, "tuning_info", {}).get("best_precision_at_10", None),
            "n_eval_buyers": int(getattr(ranker, "tuning_info", {}).get("n_eval_buyers", 0)),
            "source_weights": getattr(ranker, "source_weights", {}),
            "blend_weights": getattr(ranker, "blend_weights", {}),
        },
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
