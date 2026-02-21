import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_existing_labels(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
    df = pd.read_csv(path, engine="python")
    for col in ("buyer_id", "exporter_id", "action"):
        if col not in df.columns:
            raise ValueError(f"Existing labels CSV missing required column: {col}")
    if "ts" not in df.columns:
        df["ts"] = ""
    df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
    df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
    df["action"] = df["action"].astype(str).str.strip().str.lower()
    return df[["buyer_id", "exporter_id", "action", "ts"]]


def _build_candidate_queue(ranker, buyers_df: pd.DataFrame, buyers_to_sample: int, cands_per_buyer: int, seed: int):
    rng = np.random.default_rng(seed)
    buyers = buyers_df.dropna(subset=["Buyer_ID"]).copy()
    buyers["Buyer_ID"] = buyers["Buyer_ID"].astype(str)
    if buyers.empty:
        return []

    sample_n = min(max(1, buyers_to_sample), len(buyers))
    sampled = buyers.sample(sample_n, random_state=seed)

    queue = []
    for _, buyer_row in sampled.iterrows():
        cards = ranker.rank_for_buyer(buyer_row, top_k=max(1, cands_per_buyer))
        for card in cards:
            queue.append(
                {
                    "buyer_id": str(card["buyer_id"]),
                    "exporter_id": str(card["exporter_id"]),
                    "final_rank": card.get("final_rank"),
                    "ml_score": card.get("ml_score"),
                    "collab_score": card.get("collab_score"),
                    "ltr_score": card.get("ltr_score"),
                    "match_score": card.get("match_score"),
                    "risk_penalty": card.get("risk_penalty"),
                    "reasons": card.get("reasons", []),
                }
            )
    rng.shuffle(queue)
    return queue


def _save_labels(path: str, labels: pd.DataFrame):
    _ensure_parent_dir(path)
    labels.to_csv(path, index=False)


def _prompt_action(item: dict, idx: int, total: int) -> str:
    print(f"\n[{idx}/{total}] Buyer {item['buyer_id']} -> Exporter {item['exporter_id']}")
    print(
        f"rank={item.get('final_rank')} | ml={item.get('ml_score')} | "
        f"collab={item.get('collab_score')} | ltr={item.get('ltr_score')} | "
        f"match={item.get('match_score')} | risk={item.get('risk_penalty')}"
    )
    reasons = item.get("reasons") or []
    if reasons:
        print("reasons:", " | ".join(reasons[:3]))
    print("label? [r=right, l=left, s=skip, q=quit]")
    return input("> ").strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Interactively create a labeled swipe dataset (200-500 rows).")
    parser.add_argument("--out-csv", default="data/labels/swipes_labeled.csv", help="Output labels CSV path.")
    parser.add_argument("--target", type=int, default=300, help="Target number of total labeled rows.")
    parser.add_argument("--buyers", type=int, default=80, help="How many buyers to sample for labeling queue.")
    parser.add_argument("--cands-per-buyer", type=int, default=8, help="Candidates per sampled buyer.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--gpu", action="store_true", help="Prefer GPU during candidate generation.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU during candidate generation.")
    args = parser.parse_args()

    if args.gpu and args.cpu:
        raise ValueError("Use only one of --gpu or --cpu.")

    root = _project_root()
    sys.path.insert(0, root)
    out_csv = _resolve_path(root, args.out_csv)

    from app.ml import HybridRanker
    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    existing = _load_existing_labels(out_csv)
    existing_pairs = set(zip(existing["buyer_id"].astype(str), existing["exporter_id"].astype(str)))

    buyers_raw, exporters_raw, news = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)

    prefer_gpu = not args.cpu
    if args.gpu:
        prefer_gpu = True

    # Bootstrap model so queue quality is decent even without historical swipes.
    ranker = HybridRanker(buyers, exporters, news, prefer_gpu=prefer_gpu)
    ranker.fit(existing if not existing.empty else pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"]))
    if not ranker.is_trained:
        raise RuntimeError("Could not initialize ranker for labeling queue.")

    queue = _build_candidate_queue(
        ranker,
        buyers_df=buyers,
        buyers_to_sample=max(1, args.buyers),
        cands_per_buyer=max(1, args.cands_per_buyer),
        seed=args.seed,
    )
    queue = [q for q in queue if (q["buyer_id"], q["exporter_id"]) not in existing_pairs]
    if not queue:
        print("No new pairs to label with current settings.")
        return

    labels = existing.copy()
    total_target = max(1, int(args.target))
    print(
        f"Starting labeling session. Existing labels={len(existing)} | "
        f"Target total={total_target} | New queue={len(queue)}"
    )

    labeled_now = 0
    for i, item in enumerate(queue, start=1):
        if len(labels) >= total_target:
            break

        action = _prompt_action(item, i, len(queue))
        if action in {"q", "quit"}:
            break
        if action in {"s", "skip", ""}:
            continue
        if action not in {"r", "right", "l", "left"}:
            print("Invalid input. Use r/l/s/q.")
            continue

        swipe = "right" if action.startswith("r") else "left"
        labels = pd.concat(
            [
                labels,
                pd.DataFrame(
                    [
                        {
                            "buyer_id": item["buyer_id"],
                            "exporter_id": item["exporter_id"],
                            "action": swipe,
                            "ts": datetime.now(timezone.utc).isoformat(),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        labeled_now += 1
        if labeled_now % 10 == 0:
            _save_labels(out_csv, labels)
            print(f"Saved progress: {len(labels)} total labels")

    _save_labels(out_csv, labels)
    print(f"\nSaved labels to: {out_csv}")
    print(f"Total labeled rows: {len(labels)}")
    print("\nNext commands:")
    print(f"python backend\\scripts\\train_ranker.py --gpu --swipes-csv \"{out_csv}\" --model-out models\\ranker.pkl")
    print(f"python backend\\scripts\\evaluate_ranker.py --swipes-csv \"{out_csv}\" --top-k 10 --test-ratio 0.2 --gpu")


if __name__ == "__main__":
    main()
