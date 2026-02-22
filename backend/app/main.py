import os
import random
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    import orjson  # noqa: F401
    from fastapi.responses import ORJSONResponse as DefaultJSONResponse
except Exception:
    from fastapi.responses import JSONResponse as DefaultJSONResponse

from .db import fetch_swipes, init_db, insert_swipe, log_update
from .ml import HybridRanker
from .pipeline import engineer_buyer_features, engineer_exporter_features, load_data_clean

app = FastAPI(title="Swipe to Export MVP", default_response_class=DefaultJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_RECOMMENDATION_VERSION = os.getenv("RECOMMENDATION_VERSION", "hybrid-v1")
STATE = {"buyers": None, "exporters": None, "news": None, "ranker": None}


@app.on_event("startup")
def on_startup():
    init_db()
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


@app.get("/buyers", response_class=DefaultJSONResponse)
def buyers(limit: int = 50, offset: int = 0, q: str | None = None):
    b = STATE["buyers"]
    if b is None:
        raise HTTPException(500, "Data not loaded")

    out = b[["Buyer_ID", "Country", "Industry", "Date"]].copy()
    out["Date"] = out["Date"].astype(str)

    if q and q.strip():
        qq = q.strip().lower()
        mask = (
            out["Buyer_ID"].astype(str).str.lower().str.contains(qq, na=False)
            | out["Country"].astype(str).str.lower().str.contains(qq, na=False)
            | out["Industry"].astype(str).str.lower().str.contains(qq, na=False)
        )
        out = out[mask]

    total = int(len(out))
    out = out.iloc[offset : offset + limit]
    records = out.where(out.notna(), None).to_dict(orient="records")
    return {"total": total, "limit": limit, "offset": offset, "items": records}


@app.get("/feed")
def feed(buyer_id: str, limit: int = 10):
    b = STATE["buyers"]
    ranker = STATE["ranker"]

    if b is None:
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
    action: Literal["left", "right"]
    session_id: str | None = Field(default=None, max_length=128)
    shown_rank: int | None = Field(default=None, ge=1, le=10000)
    source: Literal["recommended", "search", "filter", "core", "explore", "retrieval", "fallback", "unknown"] | None = (
        None
    )
    dwell_ms: int | None = Field(default=None, ge=0, le=7_200_000)
    device: str | None = Field(default=None, max_length=32)
    region: str | None = Field(default=None, max_length=32)
    recommendation_version: str | None = Field(default=None, max_length=64)


@app.post("/swipe")
def swipe(payload: SwipeIn):
    result = insert_swipe(
        payload.buyer_id,
        payload.exporter_id,
        payload.action,
        session_id=payload.session_id,
        shown_rank=payload.shown_rank,
        source=payload.source,
        dwell_ms=payload.dwell_ms,
        device=payload.device,
        region=payload.region,
        recommendation_version=payload.recommendation_version or DEFAULT_RECOMMENDATION_VERSION,
    )

    if result.get("saved") and STATE["ranker"] is not None:
        row = result.get("row") or {}
        STATE["ranker"].ingest_swipe(
            payload.buyer_id,
            payload.exporter_id,
            payload.action,
            ts=row.get("ts"),
            session_id=row.get("session_id"),
            shown_rank=row.get("shown_rank"),
            source=row.get("source"),
            dwell_ms=row.get("dwell_ms"),
            device=row.get("device"),
            region=row.get("region"),
            recommendation_version=row.get("recommendation_version"),
        )

    return {
        "saved": bool(result.get("saved")),
        "duplicate": bool(result.get("duplicate")),
        "recommendation_version": payload.recommendation_version or DEFAULT_RECOMMENDATION_VERSION,
    }


@app.post("/simulate/update")
def simulate_update(industry: str | None = None):
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
        "Currency_Shift": 0.5,
    }

    news = news._append(new_row, ignore_index=True)
    STATE["news"] = news
    if STATE["ranker"] is not None:
        STATE["ranker"].refresh_news(news)
    log_update("news_simulation", {"industry": industry, "row": new_row})
    return {"updated": True, "industry": industry}
