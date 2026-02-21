import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    # Supabase requires SSL
    return psycopg2.connect(DATABASE_URL, sslmode="require")

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS swipes (
      id BIGSERIAL PRIMARY KEY,
      buyer_id TEXT NOT NULL,
      exporter_id TEXT NOT NULL,
      action TEXT NOT NULL CHECK (action IN ('left','right')),
      ts TIMESTAMPTZ DEFAULT NOW()
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signal_updates_log (
      id BIGSERIAL PRIMARY KEY,
      update_type TEXT NOT NULL,
      payload JSONB NOT NULL,
      ts TIMESTAMPTZ DEFAULT NOW()
    );
    """)

    conn.commit()
    cur.close()
    conn.close()

def insert_swipe(buyer_id: str, exporter_id: str, action: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO swipes(buyer_id, exporter_id, action) VALUES (%s, %s, %s)",
        (buyer_id, exporter_id, action)
    )
    conn.commit()
    cur.close()
    conn.close()

def log_update(update_type: str, payload: dict):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO signal_updates_log(update_type, payload) VALUES (%s, %s)",
        (update_type, Json(payload))
    )
    conn.commit()
    cur.close()
    conn.close()