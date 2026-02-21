import argparse
import os
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


def _load_swipes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    for col in ("buyer_id", "exporter_id", "action"):
        if col not in df.columns:
            raise ValueError(f"Missing required column in swipes CSV: {col}")
    if "ts" not in df.columns:
        df["ts"] = pd.Timestamp.utcnow()
    df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
    df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
    df["action"] = df["action"].astype(str).str.strip().str.lower()
    df = df[df["action"].isin(["left", "right"])]
    df = df[(df["buyer_id"] != "") & (df["exporter_id"] != "")]
    return df[["buyer_id", "exporter_id", "action", "ts"]].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build crossed swipe dataset with full ML features for each buyer-exporter interaction."
    )
    parser.add_argument(
        "--swipes-csv",
        default="data/labels/swipes_labeled.csv",
        help="Input swipe CSV with buyer_id/exporter_id/action/ts.",
    )
    parser.add_argument(
        "--out-csv",
        default="data/labels/cross_swipes_features.csv",
        help="Output crossed-feature dataset CSV.",
    )
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

    from app.ml import HybridRanker
    from app.ml.constants import FEATURE_COLUMNS
    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    swipes_path = _resolve_path(root, args.swipes_csv)
    out_path = _resolve_path(root, args.out_csv)
    if swipes_path is None or out_path is None:
        raise ValueError("Invalid paths.")

    buyers_raw, exporters_raw, news = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)
    swipes = _load_swipes(swipes_path)
    if swipes.empty:
        raise RuntimeError("Swipe CSV is empty after cleaning.")

    ranker = HybridRanker(buyers, exporters, news, prefer_gpu=prefer_gpu)
    ranker.fit(swipes)
    if not ranker.is_trained:
        raise RuntimeError("Ranker training failed while building crossed dataset.")

    records = []
    for _, row in swipes.iterrows():
        b = str(row["buyer_id"])
        e = str(row["exporter_id"])
        feat = ranker.builder.single_pair_features(b, e)
        if feat is None or feat.empty:
            continue
        rec = feat.iloc[0].to_dict()
        rec["buyer_id"] = b
        rec["exporter_id"] = e
        rec["action"] = str(row["action"]).lower()
        rec["label"] = 1 if rec["action"] == "right" else 0
        rec["ts"] = row.get("ts")
        records.append(rec)

    if not records:
        raise RuntimeError("No feature rows were generated. Check swipe IDs against clean data IDs.")

    out = pd.DataFrame(records)
    front = ["buyer_id", "exporter_id", "action", "label", "ts"]
    core = [c for c in FEATURE_COLUMNS if c in out.columns]
    tail = [c for c in out.columns if c not in front and c not in core]
    out = out[front + core + tail]

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(out_path, index=False)

    right_ratio = float((out["label"] == 1).mean()) if len(out) > 0 else 0.0
    print(f"Saved crossed dataset to: {out_path}")
    print(f"Rows: {len(out)} | Cols: {len(out.columns)} | right_ratio: {right_ratio:.4f}")
    print(
        f"Backend: supervised={ranker.supervised.device}, collaborative={ranker.collaborative.device}, "
        f"ltr={ranker.ltr.device}, retrieval={ranker.retriever.device}, text={ranker.text_encoder.backend}"
    )


if __name__ == "__main__":
    main()
