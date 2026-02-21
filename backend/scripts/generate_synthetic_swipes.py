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


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _ensure_numeric(df: pd.DataFrame, cols: list[str], fill: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out.get(c), errors="coerce").fillna(fill)
    return out


def _compute_buyer_trust_intent(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_numeric(
        df,
        ["Good_Payment_History", "Prompt_Response", "Response_Probability", "Intent_Score", "Hiring_Growth", "Engagement_Spike"],
    )
    rev = pd.to_numeric(out.get("Revenue_Size_USD"), errors="coerce").fillna(0.0)
    team = pd.to_numeric(out.get("Team_Size"), errors="coerce").fillna(0.0)
    rev_norm = (rev - rev.min()) / (max(1e-9, rev.max() - rev.min()))
    team_norm = (team - team.min()) / (max(1e-9, team.max() - team.min()))
    stability = 0.6 * rev_norm + 0.4 * team_norm

    out["buyer_trust"] = 100.0 * (
        0.35 * out["Good_Payment_History"]
        + 0.25 * out["Prompt_Response"]
        + 0.20 * out["Response_Probability"]
        + 0.20 * stability
    )
    out["buyer_intent"] = 100.0 * (
        0.65 * out["Intent_Score"]
        + 0.20 * out["Hiring_Growth"]
        + 0.15 * out["Engagement_Spike"]
    )
    return out


def _compute_exporter_trust_intent(df: pd.DataFrame) -> pd.DataFrame:
    out = _ensure_numeric(
        df,
        ["Good_Payment_Terms", "Prompt_Response_Score", "Intent_Score", "Hiring_Signal", "LinkedIn_Activity"],
    )
    rev = pd.to_numeric(out.get("Revenue_Size_USD"), errors="coerce").fillna(0.0)
    team = pd.to_numeric(out.get("Team_Size"), errors="coerce").fillna(0.0)
    rev_norm = (rev - rev.min()) / (max(1e-9, rev.max() - rev.min()))
    team_norm = (team - team.min()) / (max(1e-9, team.max() - team.min()))
    stability = 0.6 * rev_norm + 0.4 * team_norm

    li = pd.to_numeric(out.get("LinkedIn_Activity"), errors="coerce").fillna(0.0)
    li_norm = (li - li.min()) / (max(1e-9, li.max() - li.min()))

    out["exporter_trust"] = 100.0 * (
        0.35 * out["Good_Payment_Terms"]
        + 0.25 * out["Prompt_Response_Score"]
        + 0.40 * stability
    )
    out["exporter_intent"] = 100.0 * (
        0.65 * out["Intent_Score"]
        + 0.20 * out["Hiring_Signal"]
        + 0.15 * li_norm
    )
    return out


def _sample_action(
    buyer: pd.Series,
    exporter: pd.Series,
    rng: random.Random,
    right_bias: float,
) -> str:
    buyer_industry = str(buyer.get("Industry", "")).strip().lower()
    exporter_industry = str(exporter.get("Industry", "")).strip().lower()
    industry_match = 1.0 if buyer_industry and buyer_industry == exporter_industry else 0.0

    buyer_avg = pd.to_numeric(pd.Series([buyer.get("Avg_Order_Tons")]), errors="coerce").fillna(0.0).iloc[0]
    exporter_qty = pd.to_numeric(pd.Series([exporter.get("Quantity_Tons")]), errors="coerce").fillna(0.0).iloc[0]
    if buyer_avg <= 0 or exporter_qty <= 0:
        cap_fit = 0.6
    else:
        ratio = max(1e-9, float(exporter_qty) / float(buyer_avg))
        cap_fit = float(np.clip(np.exp(-abs(np.log(ratio))), 0.0, 1.0))

    trust_pair = 0.5 * (float(buyer.get("buyer_trust", 0.0)) + float(exporter.get("exporter_trust", 0.0))) / 100.0
    intent_pair = (float(buyer.get("buyer_intent", 0.0)) / 100.0) * (float(exporter.get("exporter_intent", 0.0)) / 100.0)

    tariff = abs(float(pd.to_numeric(pd.Series([exporter.get("Tariff_Impact")]), errors="coerce").fillna(0.0).iloc[0]))
    stock = abs(float(pd.to_numeric(pd.Series([exporter.get("StockMarket_Impact")]), errors="coerce").fillna(0.0).iloc[0]))
    war = float(pd.to_numeric(pd.Series([exporter.get("War_Risk")]), errors="coerce").fillna(0.0).iloc[0])
    calamity = float(pd.to_numeric(pd.Series([exporter.get("Natural_Calamity_Risk")]), errors="coerce").fillna(0.0).iloc[0])
    fx = abs(float(pd.to_numeric(pd.Series([exporter.get("Currency_Shift")]), errors="coerce").fillna(0.0).iloc[0]))
    risk = float(np.clip(0.30 * tariff + 0.25 * stock + 0.20 * war + 0.15 * calamity + 0.10 * fx, 0.0, 1.0))

    z = (
        -1.2
        + 2.2 * industry_match
        + 1.6 * cap_fit
        + 1.4 * trust_pair
        + 1.6 * intent_pair
        - 1.2 * risk
        + right_bias
        + rng.uniform(-0.25, 0.25)
    )
    p_right = float(np.clip(_sigmoid(z), 0.02, 0.98))
    return "right" if rng.random() < p_right else "left"


def _random_timestamp(days_back: int, rng: random.Random) -> str:
    now = datetime.now(timezone.utc)
    delta_days = rng.randint(0, max(1, days_back))
    delta_seconds = rng.randint(0, 24 * 3600 - 1)
    ts = now - timedelta(days=delta_days, seconds=delta_seconds)
    return ts.isoformat()


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic labeled swipe dataset.")
    parser.add_argument("--rows", type=int, default=1200, help="Target labeled rows.")
    parser.add_argument("--buyers-sample", type=int, default=200, help="How many buyers to sample.")
    parser.add_argument("--per-buyer-min", type=int, default=4, help="Min swipes per buyer.")
    parser.add_argument("--per-buyer-max", type=int, default=12, help="Max swipes per buyer.")
    parser.add_argument("--days-back", type=int, default=540, help="Timestamp lookback window.")
    parser.add_argument("--right-bias", type=float, default=0.0, help="Shift right/left balance.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-csv", default="data/labels/swipes_labeled.csv", help="Output CSV path.")
    args = parser.parse_args()

    if args.per_buyer_min <= 0 or args.per_buyer_max < args.per_buyer_min:
        raise ValueError("Invalid per-buyer range.")

    root = _project_root()
    sys.path.insert(0, root)
    out_csv = _resolve_path(root, args.out_csv)

    from app.pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

    buyers_raw, exporters_raw, _ = load_data_clean()
    buyers = engineer_buyer_features(buyers_raw)
    exporters = engineer_exporter_features(exporters_raw)

    buyers = _compute_buyer_trust_intent(buyers)
    exporters = _compute_exporter_trust_intent(exporters)

    buyers = buyers.dropna(subset=["Buyer_ID"]).copy()
    exporters = exporters.dropna(subset=["Exporter_ID"]).copy()
    buyers["Buyer_ID"] = buyers["Buyer_ID"].astype(str).str.strip()
    exporters["Exporter_ID"] = exporters["Exporter_ID"].astype(str).str.strip()
    buyers = buyers[buyers["Buyer_ID"] != ""]
    exporters = exporters[exporters["Exporter_ID"] != ""]
    if buyers.empty or exporters.empty:
        raise RuntimeError("No valid buyers or exporters found.")

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    buyer_n = min(max(1, int(args.buyers_sample)), len(buyers))
    sampled_buyers = buyers.sample(n=buyer_n, random_state=int(args.seed)).reset_index(drop=True)

    exporters_by_industry = {
        str(ind).strip().lower(): grp.reset_index(drop=True)
        for ind, grp in exporters.groupby(exporters.get("Industry", "").astype(str).str.strip().str.lower(), sort=False)
    }

    records: list[dict] = []
    pair_seen: set[tuple[str, str]] = set()
    while len(records) < int(args.rows):
        for _, buyer in sampled_buyers.iterrows():
            if len(records) >= int(args.rows):
                break

            buyer_id = str(buyer["Buyer_ID"])
            ind = str(buyer.get("Industry", "")).strip().lower()
            industry_pool = exporters_by_industry.get(ind, exporters)
            if industry_pool.empty:
                continue

            n_swipes = rng.randint(int(args.per_buyer_min), int(args.per_buyer_max))
            n_swipes = min(n_swipes, len(industry_pool), len(exporters))
            if n_swipes <= 0:
                continue

            candidate_ids = set()
            ideal_ids = industry_pool["Exporter_ID"].sample(
                n=min(max(1, n_swipes // 2), len(industry_pool)),
                random_state=rng.randint(1, 10_000_000),
            ).astype(str).tolist()
            candidate_ids.update(ideal_ids)

            if len(candidate_ids) < n_swipes:
                extra = exporters["Exporter_ID"].sample(
                    n=min(n_swipes - len(candidate_ids), len(exporters)),
                    random_state=rng.randint(1, 10_000_000),
                ).astype(str).tolist()
                candidate_ids.update(extra)

            for exporter_id in list(candidate_ids)[:n_swipes]:
                if len(records) >= int(args.rows):
                    break
                pair = (buyer_id, str(exporter_id))
                if pair in pair_seen:
                    continue
                pair_seen.add(pair)

                exporter_row = exporters[exporters["Exporter_ID"].astype(str) == str(exporter_id)]
                if exporter_row.empty:
                    continue
                exporter = exporter_row.iloc[0]
                action = _sample_action(
                    buyer=buyer,
                    exporter=exporter,
                    rng=rng,
                    right_bias=float(args.right_bias),
                )
                records.append(
                    {
                        "buyer_id": buyer_id,
                        "exporter_id": str(exporter_id),
                        "action": action,
                        "ts": _random_timestamp(int(args.days_back), rng),
                    }
                )

        if not records:
            break

    labels = pd.DataFrame(records, columns=["buyer_id", "exporter_id", "action", "ts"])
    if labels.empty:
        raise RuntimeError("Could not generate any swipe labels.")

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
    print(f"Saved synthetic swipe labels: {out_csv}")
    print(f"Rows: {len(labels)} | right: {right_n} | left: {left_n} | right_ratio: {ratio}")


if __name__ == "__main__":
    main()
