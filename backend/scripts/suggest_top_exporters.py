import argparse
import json
import os
import pickle
import sys
import pandas as pd


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(root, path)


def main():
    parser = argparse.ArgumentParser(description="Load trained model and suggest top exporters for a buyer.")
    parser.add_argument("--buyer-id", required=False, help="Buyer_ID to rank exporters for.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of exporters to return.")
    parser.add_argument("--model-in", default="models/ranker.pkl", help="Trained model path.")
    parser.add_argument("--out-json", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    buyer_id = args.buyer_id
    if not buyer_id:
        buyer_id = input("Enter Buyer_ID: ").strip()
    if not buyer_id:
        raise ValueError("Buyer_ID is required.")

    root = _project_root()
    sys.path.insert(0, root)
    from app.ml.model_explainer import ModelExplainer

    model_path = _resolve_path(root, args.model_in)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run training first.")

    with open(model_path, "rb") as f:
        ranker = pickle.load(f)

    if not getattr(ranker, "is_trained", False):
        raise RuntimeError("Loaded model is not marked trained. Retrain and save again.")

    buyers = ranker.builder.buyers
    row = buyers[buyers["Buyer_ID"].astype(str) == str(buyer_id)]
    if row.empty:
        sample = buyers["Buyer_ID"].dropna().astype(str).head(20).tolist()
        raise ValueError(f"Buyer_ID '{buyer_id}' not found in trained model. Sample IDs: {sample}")

    cards = ranker.rank_for_buyer(row.iloc[0], top_k=max(1, int(args.top_k)))

    if not cards:
        print("No exporters found for this buyer.")
        return

    explainer = ModelExplainer(ranker.supervised)

    feature_df, _ = ranker.builder.candidate_features_for_buyer(row.iloc[0])

    for i, card in enumerate(cards):
        if i < len(feature_df):
            feature_row = feature_df.iloc[i]
            proba = card["ml_score"] / 100.0
            explanation = explainer.explain_prediction(feature_row, proba)
            card["model_explanation"] = explanation

    preview_cols = [
        "exporter_id",
        "final_rank",
        "ml_score",
        "collab_score",
        "ltr_score",
        "confidence",
        "is_exploration",
        "industry_similarity",
        "match_score",
        "risk_penalty",
    ]
    preview = pd.DataFrame(cards)[preview_cols]
    print(preview.to_string(index=False))

    print("\n================ EXPLANATION PREVIEW ================\n")

    for card in cards[:3]:  # show first 3 exporters only
        print(f"Exporter: {card['exporter_id']}")
        print(f"Final Rank: {card['final_rank']}")
        print(f"Confidence: {card['confidence']}")
    
        explanation = card.get("model_explanation", {})
    
        print("Model Type:", explanation.get("model_type"))
        print("Prediction Direction:", explanation.get("prediction_direction"))
        print("Summary:", explanation.get("summary"))
    
        print("Top Drivers:")
        if "top_positive_drivers" in explanation:
            print("  Positive:", explanation["top_positive_drivers"][:2])
        if "top_negative_drivers" in explanation:
            print("  Negative:", explanation["top_negative_drivers"][:2])
        if "top_important_features" in explanation:
            print("  Important:", explanation["top_important_features"][:3])

        print("--------------------------------------------------")
    if args.out_json:
        out_json = _resolve_path(root, args.out_json)
        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(cards, f, indent=2, ensure_ascii=True)
        print(f"\nSaved ranked cards to: {out_json}")


if __name__ == "__main__":
    main()
