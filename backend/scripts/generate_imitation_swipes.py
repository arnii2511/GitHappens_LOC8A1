import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def _random_timestamp(days_back: int, rng: random.Random) -> str:
    now = datetime.now(timezone.utc)
    delta_days = rng.randint(0, max(1, days_back))
    delta_seconds = rng.randint(0, 24 * 3600 - 1)
    ts = now - timedelta(days=delta_days, seconds=delta_seconds)
    return ts.isoformat()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _pair_right_probability(row: pd.Series, buyer_style: float, right_bias: float, rng: random.Random) -> float:
    match_score = float(row.get("match_after_risk", 0.0))
    trust_score = float(row.get("pair_trust", 0.0))
    ind_sim = float(row.get("industry_similarity", row.get("industry_match", 0.0)))
    risk_penalty = float(row.get("total_risk_penalty", 0.0))
    pair_seen = float(row.get("pair_interaction_count_norm", 0.0))
    pair_right_rate = float(row.get("pair_right_rate", 0.5))

    z = (
        -1.7
        + 0.042 * match_score
        + 0.010 * trust_score
        + 0.75 * ind_sim
        - 0.030 * risk_penalty
        + 0.35 * pair_seen
        + 0.90 * (pair_right_rate - 0.5)
        + float(buyer_style)
        + float(right_bias)
        + rng.uniform(-0.22, 0.22)
    )
    return float(np.clip(_sigmoid(z), 0.02, 0.98))


def main():
    parser = argparse.ArgumentParser(description="Generate realistic synthetic swipe labels from match/trust/risk signals.")
    parser.add_argument("--rows", type=int, default=12000, help="Target labels.")
    parser.add_argument("--buyers-sample", type=int, default=1000, help="Sampled buyers.")
    parser.add_argument("--per-buyer-min", type=int, default=8, help="Min swipes per buyer.")
    parser.add_argument("--per-buyer-max", type=int, default=20, help="Max swipes per buyer.")
    parser.add_argument("--days-back", type=int, default=700, help="Max timestamp lookback.")
    parser.add_argument("--right-bias", type=float, default=0.0, help="Global right-swipe shift.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-csv", default="data/labels/swipes_labeled.csv", help="Output labels path.")
    args = parser.parse_args()

    if args.per_buyer_min <= 0 or args.per_buyer_max < args.per_buyer_min:
        raise ValueError("Invalid per-buyer range.")

    root = _project_root()
    sys.path.insert(0, root)
    out_csv = _resolve_path(root, args.out_csv)

    from app.ml.feature_builder import PairFeatureBuilder
    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    buyers_raw, exporters_raw, news = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)

    buyers = buyers.dropna(subset=["Buyer_ID"]).copy()
    exporters = exporters.dropna(subset=["Exporter_ID"]).copy()
    buyers["Buyer_ID"] = buyers["Buyer_ID"].astype(str).str.strip()
    exporters["Exporter_ID"] = exporters["Exporter_ID"].astype(str).str.strip()
    buyers = buyers[buyers["Buyer_ID"] != ""].reset_index(drop=True)
    exporters = exporters[exporters["Exporter_ID"] != ""].reset_index(drop=True)
    if buyers.empty or exporters.empty:
        raise RuntimeError("No valid buyer/exporter rows.")

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))
    builder = PairFeatureBuilder(buyers, exporters, news)
    builder.update_interaction_stats(pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"]))

    buyer_n = min(max(1, int(args.buyers_sample)), len(buyers))
    sampled_buyers = buyers.sample(n=buyer_n, random_state=int(args.seed)).reset_index(drop=True)

    buyer_style = {
        str(row["Buyer_ID"]): float(np.clip(np.random.normal(loc=0.0, scale=0.25), -0.6, 0.6))
        for _, row in sampled_buyers.iterrows()
    }

    records: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()

    while len(records) < int(args.rows):
        made_progress = False
        for _, buyer in sampled_buyers.iterrows():
            if len(records) >= int(args.rows):
                break
            buyer_id = str(buyer["Buyer_ID"])
            cands, _ = builder.candidate_features_for_buyer(buyer)
            if cands.empty:
                continue

            n_swipes = rng.randint(int(args.per_buyer_min), int(args.per_buyer_max))
            n_swipes = min(n_swipes, len(cands))
            if n_swipes <= 0:
                continue

            focus_n = max(1, int(round(0.65 * n_swipes)))
            focus = cands.nlargest(min(focus_n, len(cands)), "match_after_risk")
            explore_pool = cands[cands.get("candidate_source", "core").astype(str) == "explore"]
            remaining_n = n_swipes - len(focus)
            if remaining_n > 0:
                if explore_pool.empty:
                    tail = cands[~cands["exporter_id"].isin(focus["exporter_id"])]
                    extra = tail.sample(n=min(remaining_n, len(tail)), random_state=rng.randint(1, 10_000_000))
                else:
                    extra = explore_pool.sample(
                        n=min(remaining_n, len(explore_pool)),
                        random_state=rng.randint(1, 10_000_000),
                    )
                picks = pd.concat([focus, extra], ignore_index=True)
            else:
                picks = focus

            for _, row in picks.iterrows():
                if len(records) >= int(args.rows):
                    break
                ex_id = str(row["exporter_id"])
                pair = (buyer_id, ex_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                p_right = _pair_right_probability(
                    row=row,
                    buyer_style=buyer_style.get(buyer_id, 0.0),
                    right_bias=float(args.right_bias),
                    rng=rng,
                )
                action = "right" if rng.random() < p_right else "left"
                ts = _random_timestamp(int(args.days_back), rng)
                records.append(
                    {
                        "buyer_id": buyer_id,
                        "exporter_id": ex_id,
                        "action": action,
                        "ts": ts,
                    }
                )
                builder.ingest_interaction(buyer_id, ex_id, action, ts)
                made_progress = True

        if not made_progress:
            break

    labels = pd.DataFrame(records, columns=["buyer_id", "exporter_id", "action", "ts"])
    if labels.empty:
        raise RuntimeError("Could not generate imitation labels.")

    labels = labels.drop_duplicates(subset=["buyer_id", "exporter_id"], keep="last")
    labels = labels.sort_values("ts").reset_index(drop=True)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    labels.to_csv(out_csv, index=False)

    counts = labels["action"].value_counts().to_dict()
    right_n = int(counts.get("right", 0))
    left_n = int(counts.get("left", 0))
    ratio = round(right_n / max(1, len(labels)), 4)
    print(f"Saved imitation swipe labels: {out_csv}")
    print(f"Rows: {len(labels)} | right: {right_n} | left: {left_n} | right_ratio: {ratio}")


if __name__ == "__main__":
    main()
