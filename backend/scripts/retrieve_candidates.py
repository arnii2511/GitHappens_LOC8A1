import argparse
import os
import pickle
import sys

import pandas as pd


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(root: str, path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def main():
    parser = argparse.ArgumentParser(description="Inspect top retrieval candidates from trained HybridRanker.")
    parser.add_argument("--buyer-id", required=True, help="Buyer_ID")
    parser.add_argument("--top-k", type=int, default=20, help="Retrieved candidates")
    parser.add_argument("--model-in", default="models/ranker.pkl", help="Trained ranker path")
    args = parser.parse_args()

    root = _project_root()
    sys.path.insert(0, root)
    model_path = _resolve_path(root, args.model_in)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        ranker = pickle.load(f)
    if not getattr(ranker, "is_trained", False):
        raise RuntimeError("Ranker not trained.")
    if not getattr(ranker, "retriever", None) or not ranker.retriever.ready:
        raise RuntimeError("Retriever not ready in this model.")

    cands = ranker.retriever.retrieve_for_buyer(args.buyer_id, top_k=max(1, int(args.top_k)))
    if cands.empty:
        print("No candidates.")
        return
    print(cands.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
