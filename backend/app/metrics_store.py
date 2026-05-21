from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg2
    from psycopg2.extras import Json
except Exception:  # pragma: no cover - optional dependency/runtime
    psycopg2 = None
    Json = None


def save_pipeline_metrics(
    metrics: dict[str, Any],
    *,
    run_name: str,
    latest_path: str | None,
    history_path: str | None,
    persist_db: bool = True,
) -> dict[str, str]:
    """Persist pipeline metrics to disk and optionally to Postgres JSONB."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {"run_name": run_name, "timestamp_utc": ts, "metrics": metrics}
    saved: dict[str, str] = {}

    if latest_path:
        latest = Path(latest_path)
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps(record, indent=2, ensure_ascii=True), encoding="utf-8")
        saved["latest"] = str(latest)

    if history_path:
        history = Path(history_path)
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        saved["history"] = str(history)

    if persist_db and psycopg2 is not None and Json is not None:
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            conn = psycopg2.connect(db_url, sslmode="require")
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_metrics (
                  id BIGSERIAL PRIMARY KEY,
                  run_name TEXT NOT NULL,
                  timestamp_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                  metrics JSONB NOT NULL
                );
                """
            )
            cur.execute(
                "INSERT INTO pipeline_metrics(run_name, metrics) VALUES (%s, %s)",
                (run_name, Json(metrics)),
            )
            conn.commit()
            cur.close()
            conn.close()
            saved["db"] = "pipeline_metrics"

    return saved
