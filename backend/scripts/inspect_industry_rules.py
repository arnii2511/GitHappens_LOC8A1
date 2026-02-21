import argparse
import os
import pickle
import sys


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def main():
    parser = argparse.ArgumentParser(description="Print learned cross-industry association rules from trained ranker.")
    parser.add_argument("--model-in", default="models/ranker.pkl", help="Trained ranker path.")
    parser.add_argument("--top", type=int, default=8, help="Top rules per buyer industry.")
    args = parser.parse_args()

    root = _project_root()
    sys.path.insert(0, root)
    model_path = _resolve_path(root, args.model_in)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, "rb") as f:
        ranker = pickle.load(f)

    assoc = getattr(getattr(ranker, "retriever", None), "_assoc", None)
    if assoc is None or not assoc.ready:
        print("No learned industry association rules in this model.")
        return

    top_n = int(max(1, args.top))
    for buyer_cluster in sorted(assoc.rules.keys()):
        rules = assoc.rules.get(buyer_cluster, [])[:top_n]
        if not rules:
            continue
        print(f"\n[{buyer_cluster}]")
        for r in rules:
            print(
                f" -> {r.exporter_cluster} | "
                f"conf={r.confidence:.3f} | lift={r.lift:.3f} | "
                f"support={r.support:.3f} | score={r.score:.3f}"
            )


if __name__ == "__main__":
    main()
