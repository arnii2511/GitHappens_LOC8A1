from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import init_db, insert_swipe, log_update
from .scoring import load_data, compute_buyer_scores, compute_exporter_scores, build_feed_for_buyer
import random

app = FastAPI(title="Swipe to Export MVP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"buyers": None, "exporters": None, "news": None}

@app.on_event("startup")
def on_startup():
    init_db()  # Supabase tables
    buyers, exporters, news = load_data()
    STATE["buyers"] = compute_buyer_scores(buyers)
    STATE["exporters"] = compute_exporter_scores(exporters)
    STATE["news"] = news

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/buyers")
def buyers():
    b = STATE["buyers"]
    if b is None:
        raise HTTPException(500, "Data not loaded")
    out = b[["Buyer_ID", "Country", "Industry", "Date"]].copy()
    out["Date"] = out["Date"].astype(str)
    return out.to_dict(orient="records")

@app.get("/feed")
def feed(buyer_id: str, limit: int = 10):
    b = STATE["buyers"]
    e = STATE["exporters"]
    n = STATE["news"]

    if b is None or e is None or n is None:
        raise HTTPException(500, "Data not loaded")

    row = b[b["Buyer_ID"] == buyer_id]
    if row.empty:
        raise HTTPException(404, "Buyer not found")

    cards = build_feed_for_buyer(row.iloc[0], e, n, top_k=limit)
    return {"buyer_id": buyer_id, "cards": cards}

class SwipeIn(BaseModel):
    buyer_id: str
    exporter_id: str
    action: str  # left/right

@app.post("/swipe")
def swipe(payload: SwipeIn):
    if payload.action not in {"left", "right"}:
        raise HTTPException(400, "action must be left/right")
    insert_swipe(payload.buyer_id, payload.exporter_id, payload.action)
    return {"saved": True}

@app.post("/simulate/update")
def simulate_update(industry: str | None = None):
    """
    Adds a simulated high-impact news shock for an industry to show real-time re-ranking.
    """
    news = STATE["news"].copy()
    buyers_df = STATE["buyers"]

    if industry is None:
        industry = random.choice(list(buyers_df["Industry"].dropna().unique()))

    new_row = {
        "News_ID": int(news["News_ID"].max() + 1) if "News_ID" in news.columns else 9999,
        "Date": buyers_df["Date"].max(),
        "Region": "Global",
        "Event_Type": "Tariff Update (Simulated)",
        "Impact_Level": "High",
        "Affected_Industry": industry,
        "Tariff_Change": 0.9,
        "StockMarket_Shock": -0.6,
        "War_Flag": 0,
        "Natural_Calamity_Flag": 1,
        "Currency_Shift": 0.5
    }

    news = news._append(new_row, ignore_index=True)
    STATE["news"] = news
    log_update("news_simulation", {"industry": industry, "row": new_row})

    return {"updated": True, "industry": industry}