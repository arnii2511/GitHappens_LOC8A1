import os, json, re
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# -------- Paths --------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_CLEAN = os.path.join(PROJECT_ROOT, "data", "clean")
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_CLEAN, exist_ok=True)

FILES = {
    "buyers": "EXIM_DatasetAlgo_Hackathon(Importer_LiveSignals_v5_Updated).csv",
    "exporters": "EXIM_DatasetAlgo_Hackathon(Exporter_LiveSignals_v5_Updated).csv",
    "news": "EXIM_DatasetAlgo_Hackathon(Global_News_LiveSignals_Updated).csv",
}

# -------- Mandatory fields (confirmed) --------
MANDATORY = {
    "buyers": ["Buyer_ID", "Industry", "Country", "Intent_Score", "Response_Probability"],
    "exporters": ["Exporter_ID", "Industry", "Manufacturing_Capacity_Tons", "Revenue_Size_USD", "Intent_Score"],
    "news": ["News_ID", "Date", "Region", "Event_Type", "Impact_Level", "Affected_Industry"]
}

# -------- Helpers --------
def _norm_text(s):
    if pd.isna(s):
        return None
    s = str(s).strip()
    return s if s != "" else None

def _norm_industry(s):
    s = _norm_text(s)
    return s.lower() if s else None

def _norm_channel(s):
    s = _norm_text(s)
    if not s:
        return None
    s2 = s.lower()
    if "mail" in s2: return "email"
    if "linkedin" in s2: return "linkedin"
    if "whatsapp" in s2 or "wa" == s2: return "whatsapp"
    return s2

def _parse_date(col):
    return pd.to_datetime(col, errors="coerce")

def _coerce_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _replace_inf(df):
    return df.replace([np.inf, -np.inf], np.nan)

def _missing_pct(df):
    return (df.isna().mean() * 100).round(2).to_dict()

def _dtype_map(df):
    return {c: str(t) for c, t in df.dtypes.items()}

def _count_duplicates(df, key):
    if key not in df.columns:
        return None
    return int(df[key].duplicated().sum())

def _validate_mandatory(df, name):
    missing_cols = [c for c in MANDATORY[name] if c not in df.columns]
    return missing_cols

# -------- Cleaning functions --------
def clean_buyers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _replace_inf(df)

    # text cleanup
    for c in ["Buyer_ID", "Country", "Industry", "Certification", "Preferred_Channel"]:
        if c in df.columns:
            df[c] = df[c].apply(_norm_text)

    if "Industry" in df.columns:
        df["Industry"] = df["Industry"].apply(_norm_industry)

    if "Preferred_Channel" in df.columns:
        df["Preferred_Channel"] = df["Preferred_Channel"].apply(_norm_channel)

    if "Date" in df.columns:
        df["Date"] = _parse_date(df["Date"])

    # numeric cleanup
    num_cols = [
        "Avg_Order_Tons","Revenue_Size_USD","Team_Size","Good_Payment_History","Prompt_Response",
        "Hiring_Growth","Engagement_Spike","SalesNav_ProfileVisits","DecisionMaker_Change",
        "Intent_Score","Response_Probability","Tariff_News","StockMarket_Shock","War_Event",
        "Natural_Calamity","Currency_Fluctuation"
    ]
    df = _coerce_num(df, num_cols)

    # safe imputations (NO row drop)
    if "Intent_Score" in df.columns:
        df["Intent_Score"] = df["Intent_Score"].fillna(0)

    if "Response_Probability" in df.columns:
        # fill with median (more realistic than 0)
        med = float(df["Response_Probability"].median(skipna=True)) if df["Response_Probability"].notna().any() else 0.35
        df["Response_Probability"] = df["Response_Probability"].fillna(med)

    return df

def clean_exporters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _replace_inf(df)

    for c in ["Exporter_ID", "State", "Industry", "Certification"]:
        if c in df.columns:
            df[c] = df[c].apply(_norm_text)

    if "Industry" in df.columns:
        df["Industry"] = df["Industry"].apply(_norm_industry)

    if "Date" in df.columns:
        df["Date"] = _parse_date(df["Date"])

    num_cols = [
        "MSME_Udyam","Manufacturing_Capacity_Tons","Revenue_Size_USD","Team_Size",
        "Good_Payment_Terms","Prompt_Response_Score","Hiring_Signal","LinkedIn_Activity",
        "SalesNav_ProfileViews","SalesNav_JobChange","Intent_Score","Shipment_Value_USD",
        "Quantity_Tons","Tariff_Impact","StockMarket_Impact","War_Risk","Natural_Calamity_Risk",
        "Currency_Shift"
    ]
    df = _coerce_num(df, num_cols)

    # imputations
    if "Intent_Score" in df.columns:
        df["Intent_Score"] = df["Intent_Score"].fillna(0)

    # capacity: fill by industry median if possible
    if "Manufacturing_Capacity_Tons" in df.columns and "Industry" in df.columns:
        df["Manufacturing_Capacity_Tons"] = df.groupby("Industry")["Manufacturing_Capacity_Tons"]\
            .transform(lambda x: x.fillna(x.median()))\
            .fillna(df["Manufacturing_Capacity_Tons"].median())

    return df

def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _replace_inf(df)

    for c in ["Region", "Event_Type", "Impact_Level", "Affected_Industry"]:
        if c in df.columns:
            df[c] = df[c].apply(_norm_text)

    if "Affected_Industry" in df.columns:
        df["Affected_Industry"] = df["Affected_Industry"].apply(_norm_industry)

    if "Date" in df.columns:
        df["Date"] = _parse_date(df["Date"])

    num_cols = ["Tariff_Change","StockMarket_Shock","War_Flag","Natural_Calamity_Flag","Currency_Shift"]
    df = _coerce_num(df, num_cols)

    return df

# -------- Main pipeline --------
def main():
    report = {"paths": {"raw": DATA_RAW, "clean": DATA_CLEAN}, "tables": {}}

    # Load
    buyers = pd.read_csv(os.path.join(DATA_RAW, FILES["buyers"]), engine="python", encoding="utf-8")
    exporters = pd.read_csv(os.path.join(DATA_RAW, FILES["exporters"]), engine="python", encoding="utf-8")
    news = pd.read_csv(os.path.join(DATA_RAW, FILES["news"]), engine="python", encoding="utf-8")

    # Validate mandatory columns exist
    report["tables"]["buyers"] = {"missing_mandatory_columns": _validate_mandatory(buyers, "buyers")}
    report["tables"]["exporters"] = {"missing_mandatory_columns": _validate_mandatory(exporters, "exporters")}
    report["tables"]["news"] = {"missing_mandatory_columns": _validate_mandatory(news, "news")}

    # Clean
    buyers_c = clean_buyers(buyers)
    exporters_c = clean_exporters(exporters)
    news_c = clean_news(news)

    # Report stats
    report["tables"]["buyers"].update({
        "rows_before": int(len(buyers)),
        "rows_after": int(len(buyers_c)),
        "duplicate_ids": _count_duplicates(buyers_c, "Buyer_ID"),
        "missing_pct_after": _missing_pct(buyers_c),
        "dtypes_after": _dtype_map(buyers_c),
    })
    report["tables"]["exporters"].update({
        "rows_before": int(len(exporters)),
        "rows_after": int(len(exporters_c)),
        "duplicate_ids": _count_duplicates(exporters_c, "Exporter_ID"),
        "missing_pct_after": _missing_pct(exporters_c),
        "dtypes_after": _dtype_map(exporters_c),
    })
    report["tables"]["news"].update({
        "rows_before": int(len(news)),
        "rows_after": int(len(news_c)),
        "duplicate_ids": _count_duplicates(news_c, "News_ID"),
        "missing_pct_after": _missing_pct(news_c),
        "dtypes_after": _dtype_map(news_c),
    })

    # Save clean copies
    buyers_out = os.path.join(DATA_CLEAN, "buyers_clean.csv")
    exporters_out = os.path.join(DATA_CLEAN, "exporters_clean.csv")
    news_out = os.path.join(DATA_CLEAN, "news_clean.csv")
    buyers_c.to_csv(buyers_out, index=False)
    exporters_c.to_csv(exporters_out, index=False)
    news_c.to_csv(news_out, index=False)

    # Save report
    report_path = os.path.join(DATA_CLEAN, "data_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Cleaning done.")
    print("Saved:", buyers_out, exporters_out, news_out)
    print("Report:", report_path)

if __name__ == "__main__":
    main()