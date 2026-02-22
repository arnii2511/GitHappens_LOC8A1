import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd


REQUIRED_COLS = ("buyer_id", "exporter_id", "action")


def _backend_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _repo_root() -> str:
    return os.path.abspath(os.path.join(_backend_root(), ".."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(_backend_root(), path)


def _canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    cols = {str(c).strip().lower(): c for c in df.columns}

    def _map(src: list[str], dst: str) -> None:
        for s in src:
            if s in cols:
                rename[cols[s]] = dst
                return

    _map(["buyer_id", "buyerid", "buyer"], "buyer_id")
    _map(["buyer_id ", "buyer id", "buyerid"], "buyer_id")
    _map(["exporter_id", "exporterid", "exporter"], "exporter_id")
    _map(["exporter id", "exporter_id "], "exporter_id")
    _map(["action", "swipe", "label_text"], "action")
    _map(["ts", "timestamp", "time", "datetime"], "ts")
    _map(["label", "target", "y"], "label")

    out = df.rename(columns=rename).copy()
    return out


def _normalize_action(v: object) -> str | None:
    if pd.isna(v):
        return None
    t = str(v).strip().lower()
    if t in {"right", "r", "1", "true", "yes", "swipe_right", "accept"}:
        return "right"
    if t in {"left", "l", "0", "false", "no", "swipe_left", "reject"}:
        return "left"
    return None


def _load_swipes_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, engine="python")
    df = _canonical_columns(df)

    if "action" not in df.columns and "label" in df.columns:
        df["action"] = df["label"].apply(lambda x: "right" if str(x).strip() == "1" else "left")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in: {path}")

    df["buyer_id"] = df["buyer_id"].astype(str).str.strip()
    df["exporter_id"] = df["exporter_id"].astype(str).str.strip()
    df["action"] = df["action"].map(_normalize_action)

    if "ts" not in df.columns:
        df["ts"] = datetime.now(timezone.utc).isoformat()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)

    df = df.dropna(subset=["buyer_id", "exporter_id", "action", "ts"]).copy()
    df = df[(df["buyer_id"] != "") & (df["exporter_id"] != "")]

    out = df[["buyer_id", "exporter_id", "action", "ts"]].copy()
    out["ts"] = out["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out.reset_index(drop=True)


def _merge_into_master(master_path: str, new_frames: list[pd.DataFrame]) -> pd.DataFrame:
    parts = []
    if os.path.exists(master_path):
        parts.append(_load_swipes_csv(master_path))
    parts.extend(new_frames)

    if not parts:
        return pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["buyer_id", "exporter_id", "action", "ts"], keep="last")
    merged = merged.sort_values(["buyer_id", "ts"]).reset_index(drop=True)

    out_dir = os.path.dirname(master_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(master_path, index=False)
    return merged


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=_repo_root(), check=True, env=env)


def _default_new_swipes() -> str:
    return _resolve_path("data/labels/new_swipes_batch.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end periodic retrain cycle: ingest new swipes, train, evaluate, rebuild crossed features, and suggest."
    )
    parser.add_argument("--master-swipes-csv", default="data/labels/swipes_labeled.csv", help="Master swipe history CSV.")
    parser.add_argument("--new-swipes-csv", action="append", default=[], help="New swipe CSV to ingest (repeatable).")
    parser.add_argument(
        "--new-cross-csv",
        action="append",
        default=[],
        help="Optional crossed-feature CSV to ingest (must include buyer_id/exporter_id/action[/ts]).",
    )
    parser.add_argument("--generate-if-empty", action="store_true", help="Generate imitation swipes if master is missing/empty.")
    parser.add_argument("--generate-rows", type=int, default=12000, help="Rows for imitation generation when needed.")
    parser.add_argument("--model-out", default="models/ranker.pkl", help="Trained model output path.")
    parser.add_argument("--metrics-out", default="data/metrics/latest_metrics.json", help="Evaluation JSON output path.")
    parser.add_argument(
        "--crossed-train-csv",
        default=None,
        help="Optional crossed-feature CSV to use during train/eval (history-aware learning).",
    )
    parser.add_argument("--crossed-out", default="data/labels/cross_swipes_features.csv", help="Crossed-feature CSV output path.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K suggestions.")
    parser.add_argument("--buyer-id", default=None, help="Optional buyer ID for immediate suggestions after retrain.")
    parser.add_argument("--gpu", action="store_true", help="Prefer GPU.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--strict-gpu", action="store_true", help="Fail if any stage is not on GPU.")
    parser.add_argument("--disable-weight-tuning", action="store_true", help="Disable precision@10 weight tuning.")
    parser.add_argument("--tune-eval-buyers", type=int, default=120, help="Buyers sampled for weight tuning.")
    parser.add_argument(
        "--online-refresh-every",
        type=int,
        default=120,
        help="After these many new swipes, run full refresh of retrieval/collab/LTR models.",
    )
    parser.add_argument("--teacher-cache-only", action="store_true", help="Use teacher cache only; skip fresh teacher inference on misses.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step.")
    parser.add_argument("--skip-crossed", action="store_true", help="Skip crossed dataset rebuild step.")
    args = parser.parse_args()

    if args.gpu and args.cpu:
        raise ValueError("Use only one of --gpu or --cpu.")
    if args.strict_gpu and args.cpu:
        raise ValueError("--strict-gpu cannot be used with --cpu.")

    master_path = _resolve_path(args.master_swipes_csv)
    model_out = _resolve_path(args.model_out)
    metrics_out = _resolve_path(args.metrics_out)
    crossed_train_path = _resolve_path(args.crossed_train_csv) if args.crossed_train_csv else None
    crossed_out = _resolve_path(args.crossed_out)

    if crossed_train_path is None:
        default_cross = _resolve_path("data/labels/cross_swipes_features.csv")
        if os.path.exists(default_cross):
            crossed_train_path = default_cross

    env = os.environ.copy()
    if args.teacher_cache_only:
        env["TEACHER_CACHE_ONLY"] = "1"

    # Step 1: ingest new swipes/cross rows
    new_frames: list[pd.DataFrame] = []
    ingest_paths = [p for p in (args.new_swipes_csv or []) if str(p).strip()]
    ingest_paths.extend([p for p in (args.new_cross_csv or []) if str(p).strip()])

    for p in ingest_paths:
        resolved = _resolve_path(p)
        frame = _load_swipes_csv(resolved)
        new_frames.append(frame)
        print(f"Ingested: {resolved} | rows={len(frame)}")

    if new_frames or os.path.exists(master_path):
        merged = _merge_into_master(master_path, new_frames)
        print(f"Master swipe file: {master_path} | total_rows={len(merged)}")
    else:
        merged = pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])

    # Step 2: generate imitation data if requested and master is empty
    if args.generate_if_empty and (merged.empty):
        gen_cmd = [
            sys.executable,
            os.path.join(_backend_root(), "scripts", "generate_imitation_swipes.py"),
            "--rows",
            str(int(max(200, args.generate_rows))),
            "--out-csv",
            master_path,
        ]
        _run(gen_cmd, env=env)
        merged = _load_swipes_csv(master_path)
        print(f"Generated master swipe file: {master_path} | total_rows={len(merged)}")

    if merged.empty:
        hint = _default_new_swipes()
        raise RuntimeError(
            "No swipe rows available to train. "
            f"Add --new-swipes-csv <path> (example: {hint}) or use --generate-if-empty."
        )

    # Step 3: train
    train_cmd = [
        sys.executable,
        os.path.join(_backend_root(), "scripts", "train_ranker.py"),
        "--swipes-csv",
        master_path,
        "--model-out",
        model_out,
    ]
    if args.cpu:
        train_cmd.append("--cpu")
    else:
        train_cmd.append("--gpu")
    if args.strict_gpu:
        train_cmd.append("--strict-gpu")
    if args.disable_weight_tuning:
        train_cmd.append("--disable-weight-tuning")
    train_cmd.extend(["--tune-eval-buyers", str(int(max(30, args.tune_eval_buyers)))])
    train_cmd.extend(["--online-refresh-every", str(int(max(20, args.online_refresh_every)))])
    if crossed_train_path and os.path.exists(crossed_train_path):
        train_cmd.extend(["--crossed-csv", crossed_train_path])
    _run(train_cmd, env=env)

    # Step 4: evaluate
    if not args.skip_eval:
        eval_cmd = [
            sys.executable,
            os.path.join(_backend_root(), "scripts", "evaluate_ranker.py"),
            "--swipes-csv",
            master_path,
            "--top-k",
            str(int(max(1, args.top_k))),
            "--test-ratio",
            "0.2",
            "--out-json",
            metrics_out,
        ]
        if args.cpu:
            eval_cmd.append("--cpu")
        else:
            eval_cmd.append("--gpu")
        if args.disable_weight_tuning:
            eval_cmd.append("--disable-weight-tuning")
        eval_cmd.extend(["--tune-eval-buyers", str(int(max(30, args.tune_eval_buyers)))])
        eval_cmd.extend(["--online-refresh-every", str(int(max(20, args.online_refresh_every)))])
        if crossed_train_path and os.path.exists(crossed_train_path):
            eval_cmd.extend(["--crossed-csv", crossed_train_path])
        _run(eval_cmd, env=env)

    # Step 5: rebuild crossed-feature dataset
    if not args.skip_crossed:
        crossed_cmd = [
            sys.executable,
            os.path.join(_backend_root(), "scripts", "build_crossed_dataset.py"),
            "--swipes-csv",
            master_path,
            "--out-csv",
            crossed_out,
        ]
        if args.cpu:
            crossed_cmd.append("--cpu")
        else:
            crossed_cmd.append("--gpu")
        _run(crossed_cmd, env=env)

    # Step 6: optional sample suggestion
    if args.buyer_id:
        suggest_cmd = [
            sys.executable,
            os.path.join(_backend_root(), "scripts", "suggest_top_exporters.py"),
            "--buyer-id",
            str(args.buyer_id),
            "--top-k",
            str(int(max(1, args.top_k))),
            "--model-in",
            model_out,
        ]
        _run(suggest_cmd, env=env)

    summary = {
        "master_swipes_csv": master_path,
        "model_out": model_out,
        "metrics_out": metrics_out if not args.skip_eval else None,
        "crossed_train_csv": crossed_train_path if (crossed_train_path and os.path.exists(crossed_train_path)) else None,
        "crossed_out": crossed_out if not args.skip_crossed else None,
        "buyer_id": args.buyer_id,
    }
    print("\nCycle complete:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
