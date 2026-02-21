from fastapi.responses import ORJSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from .db import init_db, insert_swipe, log_update, fetch_swipes
from .ml import HybridRanker
from .pipeline import (
    engineer_buyer_features,
    engineer_exporter_features,
    load_data_clean,
)
import random

app = FastAPI(title="Swipe to Export MVP", default_response_class=ORJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"buyers": None, "exporters": None, "news": None, "ranker": None}

@app.on_event("startup")
def on_startup():
    init_db()  # Supabase tables
    buyers, exporters, news = load_data_clean()
    STATE["buyers"] = engineer_buyer_features(buyers)
    STATE["exporters"] = engineer_exporter_features(exporters)
    STATE["news"] = news
    ranker = HybridRanker(STATE["buyers"], STATE["exporters"], STATE["news"])
    try:
        swipes = fetch_swipes(limit=250_000)
    except Exception:
        swipes = pd.DataFrame(columns=["buyer_id", "exporter_id", "action", "ts"])
    ranker.fit(swipes)
    STATE["ranker"] = ranker

@app.get("/health")
def health():
    return {"ok": True}



@app.get("/buyers", response_class=ORJSONResponse)
def buyers(limit: int = 50, offset: int = 0, q: str | None = None):
    b = STATE["buyers"]
    if b is None:
        raise HTTPException(500, "Data not loaded")

    out = b[["Buyer_ID", "Country", "Industry", "Date"]].copy()
    out["Date"] = out["Date"].astype(str)

    # optional search (Buyer_ID / Country / Industry)
    if q and q.strip():
        qq = q.strip().lower()
        mask = (
            out["Buyer_ID"].astype(str).str.lower().str.contains(qq, na=False) |
            out["Country"].astype(str).str.lower().str.contains(qq, na=False) |
            out["Industry"].astype(str).str.lower().str.contains(qq, na=False)
        )
        out = out[mask]

    total = int(len(out))

    # paginate so Swagger doesn't freeze
    out = out.iloc[offset: offset + limit]

    records = out.where(out.notna(), None).to_dict(orient="records")
    return {"total": total, "limit": limit, "offset": offset, "items": records}

@app.get("/feed")
def feed(buyer_id: str, limit: int = 10):
    b = STATE["buyers"]
    e = STATE["exporters"]
    n = STATE["news"]
    ranker = STATE["ranker"]

    if b is None or e is None or n is None:
        raise HTTPException(500, "Data not loaded")

    row = b[b["Buyer_ID"] == buyer_id]
    if row.empty:
        raise HTTPException(404, "Buyer not found")

    if ranker is None:
        raise HTTPException(503, "Ranker unavailable. Train the model before requesting feed.")
    if not ranker.is_trained:
        raise HTTPException(503, "Ranker not trained. Train first, then request feed.")
    cards = ranker.rank_for_buyer(row.iloc[0], top_k=limit)
    return jsonable_encoder({"buyer_id": buyer_id, "cards": cards})

class SwipeIn(BaseModel):
    buyer_id: str
    exporter_id: str
    action: str  # left/right

@app.post("/swipe")
def swipe(payload: SwipeIn):
    if payload.action not in {"left", "right"}:
        raise HTTPException(400, "action must be left/right")
    insert_swipe(payload.buyer_id, payload.exporter_id, payload.action)
    if STATE["ranker"] is not None:
        STATE["ranker"].ingest_swipe(payload.buyer_id, payload.exporter_id, payload.action)
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
    if STATE["ranker"] is not None:
        STATE["ranker"].refresh_news(news)
    log_update("news_simulation", {"industry": industry, "row": new_row})

    return {"updated": True, "industry": industry}
