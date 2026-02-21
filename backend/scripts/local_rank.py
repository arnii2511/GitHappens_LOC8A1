import argparse
import json
import os
import sys

import pandas as pd


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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
    parser = argparse.ArgumentParser(description="Train local ranker and return top exporters for a buyer.")
    parser.add_argument("--buyer-id", required=False, help="Buyer_ID to rank exporters for.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of exporters to return (default: 10).")
    parser.add_argument(
        "--swipes-csv",
        default=None,
        help="Optional path to local swipe history CSV with columns: buyer_id, exporter_id, action, ts",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output file path to save ranked cards as JSON.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Prefer GPU for learning-to-rank training (falls back to CPU if unavailable).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode for learning-to-rank training.",
    )
    parser.add_argument(
        "--strict-gpu",
        action="store_true",
        help="Fail if any model stage falls back to CPU.",
    )
    args = parser.parse_args()
    if args.gpu and args.cpu:
        raise ValueError("Use only one of --gpu or --cpu.")
    if args.strict_gpu and args.cpu:
        raise ValueError("--strict-gpu cannot be used with --cpu.")

    prefer_gpu = True
    if args.cpu:
        prefer_gpu = False
    if args.gpu:
        prefer_gpu = True
    if args.strict_gpu:
        prefer_gpu = True

    buyer_id = args.buyer_id
    if not buyer_id:
        buyer_id = input("Enter Buyer_ID: ").strip()
    if not buyer_id:
        raise ValueError("Buyer_ID is required.")

    root = _project_root()
    sys.path.insert(0, root)

    from app.ml import HybridRanker
    from app.ml.common import CUPY_AVAILABLE
    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    buyers_raw, exporters_raw, news = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)

    swipes = _load_swipes(args.swipes_csv)
    ranker = HybridRanker(buyers, exporters, news, prefer_gpu=prefer_gpu)
    ranker.fit(swipes)
    if not ranker.is_trained:
        raise RuntimeError(
            "Model training did not complete successfully. "
            "Provide valid data/swipe history and retry."
        )
    print(
        f"Supervised backend: {ranker.supervised.backend or 'none'} | "
        f"device: {ranker.supervised.device} | ready: {ranker.supervised.ready}"
    )
    print(
        f"Collaborative backend: svd | "
        f"device: {ranker.collaborative.device} | ready: {ranker.collaborative.ready}"
    )
    print(
        f"LTR backend: {ranker.ltr.backend or 'none'} | "
        f"device: {ranker.ltr.device} | "
        f"ready: {ranker.ltr.ready}"
    )
    if prefer_gpu and ranker.ltr.device != "gpu":
        print("GPU request was enabled, but LTR fell back to CPU on this machine.")
    if prefer_gpu and ranker.ltr.device == "gpu" and not CUPY_AVAILABLE:
        print("Note: install cupy for fully GPU-side prediction and to avoid XGBoost device-mismatch warnings.")
    if args.strict_gpu:
        devices = [ranker.supervised.device, ranker.collaborative.device, ranker.ltr.device]
        if any(d != "gpu" for d in devices):
            raise RuntimeError(
                "Strict GPU mode failed. Devices: "
                f"supervised={ranker.supervised.device}, "
                f"collaborative={ranker.collaborative.device}, "
                f"ltr={ranker.ltr.device}"
            )

    row = buyers[buyers["Buyer_ID"].astype(str) == str(buyer_id)]
    if row.empty:
        available = buyers["Buyer_ID"].dropna().astype(str).head(20).tolist()
        raise ValueError(
            f"Buyer_ID '{buyer_id}' not found. Sample Buyer_ID values: {available}"
        )

    cards = ranker.rank_for_buyer(row.iloc[0], top_k=max(1, int(args.top_k)))

    if not cards:
        print("No exporters found for this buyer.")
        return

    preview_cols = [
        "exporter_id",
        "final_rank",
        "ml_score",
        "collab_score",
        "ltr_score",
        "confidence",
        "is_exploration",
        "match_score",
        "risk_penalty",
    ]
    preview = pd.DataFrame(cards)[preview_cols]
    print(preview.to_string(index=False))

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2, ensure_ascii=True)
        print(f"\nSaved ranked cards to: {args.out_json}")


if __name__ == "__main__":
    main()
