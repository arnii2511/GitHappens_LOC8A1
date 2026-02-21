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
    parser = argparse.ArgumentParser(description="Train ranker and save model artifact.")
    parser.add_argument("--swipes-csv", default=None, help="Optional swipe history CSV.")
    parser.add_argument("--model-out", default="models/ranker.pkl", help="Output model path.")
    parser.add_argument("--gpu", action="store_true", help="Prefer GPU.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--strict-gpu", action="store_true", help="Fail if any stage is not GPU.")
    args = parser.parse_args()

    if args.gpu and args.cpu:
        raise ValueError("Use only one of --gpu or --cpu.")
    if args.strict_gpu and args.cpu:
        raise ValueError("--strict-gpu cannot be used with --cpu.")

    prefer_gpu = True
    if args.cpu:
        prefer_gpu = False
    if args.gpu or args.strict_gpu:
        prefer_gpu = True

    root = _project_root()
    sys.path.insert(0, root)

    from app.ml import HybridRanker
    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    buyers_raw, exporters_raw, news = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)
    swipes_path = _resolve_path(root, args.swipes_csv)
    swipes = _load_swipes(swipes_path)

    ranker = HybridRanker(buyers, exporters, news, prefer_gpu=prefer_gpu)
    ranker.fit(swipes)
    if not ranker.is_trained:
        raise RuntimeError("Model training did not complete successfully.")

    print(
        f"Supervised: {ranker.supervised.backend or 'none'} ({ranker.supervised.device}) | "
        f"Collaborative: svd ({ranker.collaborative.device}) | "
        f"LTR: {ranker.ltr.backend or 'none'} ({ranker.ltr.device})"
    )

    if args.strict_gpu:
        devices = [ranker.supervised.device, ranker.collaborative.device, ranker.ltr.device]
        if any(d != "gpu" for d in devices):
            raise RuntimeError(
                "Strict GPU mode failed. Devices: "
                f"supervised={ranker.supervised.device}, "
                f"collaborative={ranker.collaborative.device}, "
                f"ltr={ranker.ltr.device}"
            )

    out_path = _resolve_path(root, args.model_out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(ranker, f)
    print(f"Saved trained model to: {out_path}")


if __name__ == "__main__":
    main()
