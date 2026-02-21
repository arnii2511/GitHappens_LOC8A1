import argparse
import json
import os
from typing import Optional

import pandas as pd


def _summarize_csv(path: str, max_cols: int = 40) -> dict:
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    df = pd.read_csv(path, engine="python")
    null_ratio = (df.isna().mean() * 100.0).sort_values(ascending=False)
    top_null = [{"column": c, "null_pct": round(float(v), 2)} for c, v in null_ratio.head(max_cols).items()]
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    return {
        "path": path,
        "exists": True,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": [str(c) for c in df.columns.tolist()],
        "dtypes": dtypes,
        "top_null_columns": top_null,
    }


def _swipe_quality(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, engine="python")
    out = {"rows": int(len(df))}
    for col in ("buyer_id", "exporter_id", "action", "ts"):
        out[f"has_{col}"] = bool(col in df.columns)
    if {"buyer_id", "exporter_id", "action"}.issubset(df.columns):
        a = df["action"].astype(str).str.lower().str.strip()
        out["action_counts"] = a.value_counts(dropna=False).to_dict()
        out["unique_buyers"] = int(df["buyer_id"].astype(str).nunique())
        out["unique_exporters"] = int(df["exporter_id"].astype(str).nunique())
        out["duplicate_pairs"] = int(df.duplicated(subset=["buyer_id", "exporter_id"]).sum())
    return out


def main():
    parser = argparse.ArgumentParser(description="Profile clean CSV datasets and optional swipe labels.")
    parser.add_argument("--swipes-csv", default="backend/data/labels/swipes_labeled.csv", help="Optional swipe labels csv.")
    parser.add_argument("--out-json", default=None, help="Optional output report path.")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    buyers = os.path.join(root, "data", "clean", "buyers_clean.csv")
    exporters = os.path.join(root, "data", "clean", "exporters_clean.csv")
    news = os.path.join(root, "data", "clean", "news_clean.csv")
    swipes = args.swipes_csv
    if not os.path.isabs(swipes):
        swipes = os.path.join(root, swipes)

    report = {
        "buyers_clean": _summarize_csv(buyers),
        "exporters_clean": _summarize_csv(exporters),
        "news_clean": _summarize_csv(news),
        "swipes_quality": _swipe_quality(swipes),
    }
    print(json.dumps(report, indent=2))

    if args.out_json:
        out = args.out_json
        if not os.path.isabs(out):
            out = os.path.join(root, out)
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)
        print(f"\nSaved profile report: {out}")


if __name__ == "__main__":
    main()
