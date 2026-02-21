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


def _load_swipes(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
    df = pd.read_csv(path, engine="python")
    for col in ("buyer_id", "exporter_id", "action"):
        if col not in df.columns:
            raise ValueError(f"Missing required column in swipes CSV: {col}")
    if "ts" not in df.columns:
        df["ts"] = pd.Timestamp.utcnow()
    return df[["buyer_id", "exporter_id", "action", "ts"]]


def main():
    parser = argparse.ArgumentParser(description="Train only retrieval stack (Text + Two-Tower + ANN).")
    parser.add_argument("--swipes-csv", required=True, help="Swipe history CSV.")
    parser.add_argument("--model-out", default="models/retrieval.pkl", help="Output artifact path.")
    parser.add_argument("--gpu", action="store_true", help="Prefer GPU.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    args = parser.parse_args()

    if args.gpu and args.cpu:
        raise ValueError("Use only one of --gpu or --cpu.")
    prefer_gpu = not args.cpu
    if args.gpu:
        prefer_gpu = True

    root = _project_root()
    sys.path.insert(0, root)

    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean
    from app.retrieval import TextEmbeddingService, TwoTowerRetriever

    swipes = _load_swipes(_resolve_path(root, args.swipes_csv))
    buyers_raw, exporters_raw, _ = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)

    text = TextEmbeddingService(prefer_gpu=prefer_gpu)
    text.fit(buyers, exporters)
    retrieval = TwoTowerRetriever(buyers, exporters, text_encoder=text, prefer_gpu=prefer_gpu)
    retrieval.fit(swipes)

    out_path = _resolve_path(root, args.model_out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"text": text, "retrieval": retrieval}, f)

    print(
        f"Text: {text.backend} | ready={text.ready} | "
        f"Retrieval: {retrieval.backend} ({retrieval.device}) | ready={retrieval.ready}"
    )
    print(f"Saved retrieval artifact: {out_path}")


if __name__ == "__main__":
    main()
