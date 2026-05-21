import os
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    # Supabase requires SSL
    return psycopg2.connect(DATABASE_URL, sslmode="require")


SWIPE_COLUMNS = [
    "id",
    "buyer_id",
    "exporter_id",
    "action",
    "ts",
    "session_id",
    "shown_rank",
    "source",
    "dwell_ms",
    "device",
    "region",
    "recommendation_version",
]


def _normalize_session_id(value: str | None) -> str:
    s = str(value or "").strip()
    return s[:128]


def _normalize_source(value: str | None) -> str:
    s = str(value or "").strip().lower()
    return s[:48] if s else "unknown"


def _normalize_text(value: str | None, max_len: int = 64) -> str | None:
    s = str(value or "").strip()
    if not s:
        return None
    return s[:max_len]


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS swipes (
      id BIGSERIAL PRIMARY KEY,
      buyer_id TEXT NOT NULL,
      exporter_id TEXT NOT NULL,
      action TEXT NOT NULL CHECK (action IN ('left','right')),
      ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      session_id TEXT NOT NULL DEFAULT '',
      shown_rank INTEGER,
      source TEXT NOT NULL DEFAULT 'unknown',
      dwell_ms INTEGER,
      device TEXT,
      region TEXT,
      recommendation_version TEXT NOT NULL DEFAULT 'hybrid-v1'
    );
    """
    )

    # Migration-safe upgrades for existing tables.
    cur.execute("ALTER TABLE swipes ADD COLUMN IF NOT EXISTS session_id TEXT NOT NULL DEFAULT '';")
    cur.execute("ALTER TABLE swipes ADD COLUMN IF NOT EXISTS shown_rank INTEGER;")
    cur.execute("ALTER TABLE swipes ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'unknown';")
    cur.execute("ALTER TABLE swipes ADD COLUMN IF NOT EXISTS dwell_ms INTEGER;")
    cur.execute("ALTER TABLE swipes ADD COLUMN IF NOT EXISTS device TEXT;")
    cur.execute("ALTER TABLE swipes ADD COLUMN IF NOT EXISTS region TEXT;")
    cur.execute(
        "ALTER TABLE swipes ADD COLUMN IF NOT EXISTS recommendation_version TEXT NOT NULL DEFAULT 'hybrid-v1';"
    )

    cur.execute("UPDATE swipes SET session_id = '' WHERE session_id IS NULL;")
    cur.execute("UPDATE swipes SET source = 'unknown' WHERE source IS NULL OR btrim(source) = '';")

    # Keep latest row per idempotency key before adding unique index.
    cur.execute(
        """
        DELETE FROM swipes a
        USING swipes b
        WHERE a.id < b.id
          AND a.buyer_id = b.buyer_id
          AND a.exporter_id = b.exporter_id
          AND COALESCE(a.session_id, '') = COALESCE(b.session_id, '')
        """
    )
    cur.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_swipes_buyer_exporter_session
        ON swipes (buyer_id, exporter_id, session_id)
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_swipes_ts ON swipes (ts DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_swipes_buyer_ts ON swipes (buyer_id, ts DESC);")

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS signal_updates_log (
      id BIGSERIAL PRIMARY KEY,
      update_type TEXT NOT NULL,
      payload JSONB NOT NULL,
      ts TIMESTAMPTZ DEFAULT NOW()
    );
    """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS buyer_profile_features (
          buyer_id TEXT PRIMARY KEY,
          total_swipes INTEGER NOT NULL DEFAULT 0,
          right_swipes INTEGER NOT NULL DEFAULT 0,
          left_swipes INTEGER NOT NULL DEFAULT 0,
          right_rate DOUBLE PRECISION NOT NULL DEFAULT 0.5,
          recent_right_rate_20 DOUBLE PRECISION NOT NULL DEFAULT 0.5,
          top_sources JSONB NOT NULL DEFAULT '{}'::jsonb,
          last_session_id TEXT,
          last_swipe_ts TIMESTAMPTZ,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
    )

    conn.commit()
    cur.close()
    conn.close()


def refresh_buyer_profile_features(cur, buyer_id: str, last_session_id: str | None = None):
    cur.execute(
        """
        SELECT
          COUNT(*) AS total_swipes,
          SUM(CASE WHEN action = 'right' THEN 1 ELSE 0 END) AS right_swipes,
          SUM(CASE WHEN action = 'left' THEN 1 ELSE 0 END) AS left_swipes,
          MAX(ts) AS last_swipe_ts
        FROM swipes
        WHERE buyer_id = %s
        """,
        (buyer_id,),
    )
    row = cur.fetchone()
    total_swipes = int((row or [0])[0] or 0)
    right_swipes = int((row or [0, 0])[1] or 0)
    left_swipes = int((row or [0, 0, 0])[2] or 0)
    last_swipe_ts = row[3] if row else None
    right_rate = float(right_swipes / max(1, total_swipes))

    cur.execute(
        """
        SELECT AVG(CASE WHEN action = 'right' THEN 1.0 ELSE 0.0 END)
        FROM (
          SELECT action
          FROM swipes
          WHERE buyer_id = %s
          ORDER BY ts DESC
          LIMIT 20
        ) recent
        """,
        (buyer_id,),
    )
    recent_row = cur.fetchone()
    recent_right_rate_20 = float((recent_row[0] if recent_row else None) or right_rate or 0.5)

    cur.execute(
        """
        SELECT source, COUNT(*)
        FROM swipes
        WHERE buyer_id = %s
          AND source IS NOT NULL
          AND btrim(source) <> ''
        GROUP BY source
        ORDER BY COUNT(*) DESC, source ASC
        LIMIT 12
        """,
        (buyer_id,),
    )
    source_rows = cur.fetchall() or []
    top_sources = {str(src): int(cnt) for src, cnt in source_rows}

    cur.execute(
        """
        INSERT INTO buyer_profile_features (
          buyer_id,
          total_swipes,
          right_swipes,
          left_swipes,
          right_rate,
          recent_right_rate_20,
          top_sources,
          last_session_id,
          last_swipe_ts,
          updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW())
        ON CONFLICT (buyer_id) DO UPDATE SET
          total_swipes = EXCLUDED.total_swipes,
          right_swipes = EXCLUDED.right_swipes,
          left_swipes = EXCLUDED.left_swipes,
          right_rate = EXCLUDED.right_rate,
          recent_right_rate_20 = EXCLUDED.recent_right_rate_20,
          top_sources = EXCLUDED.top_sources,
          last_session_id = EXCLUDED.last_session_id,
          last_swipe_ts = EXCLUDED.last_swipe_ts,
          updated_at = NOW()
        """,
        (
            buyer_id,
            total_swipes,
            right_swipes,
            left_swipes,
            right_rate,
            recent_right_rate_20,
            Json(top_sources),
            _normalize_session_id(last_session_id),
            last_swipe_ts,
        ),
    )


def insert_swipe(
    buyer_id: str,
    exporter_id: str,
    action: str,
    *,
    session_id: str | None = None,
    shown_rank: int | None = None,
    source: str | None = None,
    dwell_ms: int | None = None,
    device: str | None = None,
    region: str | None = None,
    recommendation_version: str | None = None,
) -> dict[str, Any]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO swipes(
          buyer_id,
          exporter_id,
          action,
          session_id,
          shown_rank,
          source,
          dwell_ms,
          device,
          region,
          recommendation_version
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (buyer_id, exporter_id, session_id) DO NOTHING
        RETURNING id, buyer_id, exporter_id, action, ts, session_id, shown_rank, source, dwell_ms, device, region, recommendation_version
        """,
        (
            str(buyer_id).strip(),
            str(exporter_id).strip(),
            str(action).strip().lower(),
            _normalize_session_id(session_id),
            int(shown_rank) if shown_rank is not None else None,
            _normalize_source(source),
            int(dwell_ms) if dwell_ms is not None else None,
            _normalize_text(device, max_len=32),
            _normalize_text(region, max_len=32),
            _normalize_text(recommendation_version, max_len=64) or "hybrid-v1",
        ),
    )
    inserted_row = cur.fetchone()
    inserted = inserted_row is not None

    if inserted:
        refresh_buyer_profile_features(cur, str(buyer_id).strip(), last_session_id=session_id)

    conn.commit()
    cur.close()
    conn.close()

    if inserted:
        row_dict = dict(zip(SWIPE_COLUMNS, inserted_row))
        return {"saved": True, "duplicate": False, "row": row_dict}
    return {"saved": False, "duplicate": True, "row": None}


def log_update(update_type: str, payload: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO signal_updates_log(update_type, payload) VALUES (%s, %s)",
        (update_type, Json(payload)),
    )
    conn.commit()
    cur.close()
    conn.close()


def fetch_swipes(limit: int = 250_000) -> pd.DataFrame:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT buyer_id, exporter_id, action, ts, session_id, shown_rank, source, dwell_ms, device, region, recommendation_version
        FROM swipes
        ORDER BY ts DESC
        LIMIT %s
        """,
        (int(max(1, limit)),),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    if not rows:
        return pd.DataFrame(
            columns=[
                "buyer_id",
                "exporter_id",
                "action",
                "ts",
                "session_id",
                "shown_rank",
                "source",
                "dwell_ms",
                "device",
                "region",
                "recommendation_version",
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "buyer_id",
            "exporter_id",
            "action",
            "ts",
            "session_id",
            "shown_rank",
            "source",
            "dwell_ms",
            "device",
            "region",
            "recommendation_version",
        ],
    )
