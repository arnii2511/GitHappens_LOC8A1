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


def _load_crossed(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path, engine="python")
    req = {"buyer_id", "exporter_id"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required crossed feature columns: {miss}")
    if "label" not in df.columns and "action" not in df.columns:
        raise ValueError("Crossed features must include at least one of: label/action.")
    if "ts" not in df.columns:
        df["ts"] = pd.Timestamp.utcnow()
    df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
    df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
    return df


def main():
    parser = argparse.ArgumentParser(description="Train ranker and save model artifact.")
    parser.add_argument("--swipes-csv", default=None, help="Optional swipe history CSV.")
    parser.add_argument("--crossed-csv", default=None, help="Optional crossed feature CSV with labels/actions.")
    parser.add_argument("--model-out", default="models/ranker.pkl", help="Output model path.")
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
    from app.ml.common import CUPY_AVAILABLE
    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    buyers_raw, exporters_raw, news = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)
    swipes_path = _resolve_path(root, args.swipes_csv)
    swipes = _load_swipes(swipes_path)
    crossed_path = _resolve_path(root, args.crossed_csv)
    crossed = _load_crossed(crossed_path)

    ranker = HybridRanker(
        buyers,
        exporters,
        news,
        prefer_gpu=prefer_gpu,
        auto_tune_weights=(not args.disable_weight_tuning),
        tune_eval_buyers=int(max(30, args.tune_eval_buyers)),
        online_full_refresh_every=int(max(20, args.online_refresh_every)),
    )
    ranker.fit(swipes, crossed_features=crossed)
    if not ranker.is_trained:
        raise RuntimeError("Model training did not complete successfully.")

    print(
        f"Supervised: {ranker.supervised.backend or 'none'} ({ranker.supervised.device}) | "
        f"Collaborative: {getattr(ranker.collaborative, 'backend', 'none') or 'none'} ({ranker.collaborative.device}) | "
        f"NCF: {getattr(ranker.ncf, 'backend', 'none') or 'none'} ({getattr(ranker.ncf, 'device', 'none')}) | "
        f"LTR: {ranker.ltr.backend or 'none'} ({ranker.ltr.device}) | "
        f"Retrieval: {ranker.retriever.backend or 'none'} ({ranker.retriever.device}) | "
        f"Text: {ranker.text_encoder.backend or 'none'} | "
        f"Teacher: {getattr(ranker.text_encoder, 'teacher_backend', 'none')} | "
        f"Graph: {getattr(ranker.graph, 'backend', 'none')} | "
        f"TeacherCacheLoaded: {int(getattr(ranker.text_encoder, 'teacher_cache_loaded_entries', 0))} | "
        f"CrossRows: {int(len(crossed))} | "
        f"TuningRan: {bool(getattr(ranker, 'tuning_info', {}).get('ran', False))} | "
        f"BestP@10: {getattr(ranker, 'tuning_info', {}).get('best_precision_at_10', None)} | "
        f"TuneBuyers: {int(getattr(ranker, 'tuning_info', {}).get('n_eval_buyers', 0))}"
    )
    print(f"SourceWeights: {getattr(ranker, 'source_weights', {})}")
    print(f"BlendWeights: {getattr(ranker, 'blend_weights', {})}")
    if prefer_gpu and getattr(ranker.collaborative, "backend", "none") not in {"none", None} and ranker.collaborative.device != "gpu":
        if not CUPY_AVAILABLE:
            print("GPU note: collaborative stage stayed on CPU because CuPy is not installed.")
        else:
            print("GPU note: collaborative stage stayed on CPU due runtime fallback.")

    if args.strict_gpu:
        stage_devices = {
            "supervised": ranker.supervised.device if (ranker.supervised.backend or "none") != "none" else None,
            "collaborative": ranker.collaborative.device if getattr(ranker.collaborative, "ready", False) else None,
            "ncf": getattr(ranker.ncf, "device", None) if getattr(ranker.ncf, "ready", False) else None,
            "ltr": ranker.ltr.device if (ranker.ltr.backend or "none") != "none" else None,
            "retrieval": ranker.retriever.device if (ranker.retriever.backend or "none") != "none" else None,
        }
        active_devices = [d for d in stage_devices.values() if d is not None]
        if any(d != "gpu" for d in active_devices):
            raise RuntimeError(
                "Strict GPU mode failed. Devices: "
                + ", ".join([f"{k}={v}" for k, v in stage_devices.items() if v is not None])
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
