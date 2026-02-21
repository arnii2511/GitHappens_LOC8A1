import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "clean"))


def load_data_clean():
    buyers = pd.read_csv(os.path.join(DATA_DIR, "buyers_clean.csv"), encoding="utf-8", engine="python")
    exporters = pd.read_csv(os.path.join(DATA_DIR, "exporters_clean.csv"), encoding="utf-8", engine="python")
    news = pd.read_csv(os.path.join(DATA_DIR, "news_clean.csv"), encoding="utf-8", engine="python")

    for df in (buyers, exporters, news):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if "Industry" in buyers.columns:
        buyers["Industry"] = buyers["Industry"].astype(str).str.strip().str.lower()
    if "Industry" in exporters.columns:
        exporters["Industry"] = exporters["Industry"].astype(str).str.strip().str.lower()
    if "Affected_Industry" in news.columns:
        news["Affected_Industry"] = news["Affected_Industry"].astype(str).str.strip().str.lower()

    return buyers, exporters, news
