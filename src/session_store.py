from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SQLITE_DIR = PROJECT_ROOT / "data" / "SQLite_storage"
SQLITE_DB_FILE = SQLITE_DIR / "retention_sessions.db"
SESSION_TTL_MINUTES = 45


def _connect() -> sqlite3.Connection:
    SQLITE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(SQLITE_DB_FILE)
    connection.execute("PRAGMA journal_mode=WAL;")
    return connection


def _rename_column_if_needed(connection: sqlite3.Connection, table_name: str, old_name: str, new_name: str) -> None:
    columns = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    column_names = {str(column[1]) for column in columns}
    if old_name in column_names and new_name not in column_names:
        connection.execute(f'ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name}')


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            last_activity TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            status TEXT NOT NULL,
            close_reason TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS session_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            event_time TEXT NOT NULL,
            source TEXT NOT NULL,
            step TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            session_jsonl_file TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS output_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            category TEXT NOT NULL,
            file_path TEXT NOT NULL,
            record_time TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_events_session_id ON session_events(session_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_output_records_session_id ON output_records(session_id)"
    )
    _rename_column_if_needed(connection, "sessions", "started_at_utc", "started_at")
    _rename_column_if_needed(connection, "sessions", "last_activity_atc", "last_activity")
    _rename_column_if_needed(connection, "sessions", "last_activity_at", "last_activity")
    _rename_column_if_needed(connection, "sessions", "expires_at", "expires_at")
    _rename_column_if_needed(connection, "session_events", "event_time", "event_time")


def _iso(value: datetime | None = None) -> str:
    now = value or datetime.now(timezone.utc)
    return now.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)

    text = str(value).strip()
    if not text:
        return datetime.now(timezone.utc)

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return datetime.now(timezone.utc)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone(timedelta(hours=5, minutes=30)))
    return parsed.astimezone(timezone.utc)


def _extract_session_id(payload: dict[str, Any]) -> str:
    direct_session_id = payload.get("session_id")
    if isinstance(direct_session_id, str) and direct_session_id.strip():
        return direct_session_id.strip()

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        metadata_session_id = metadata.get("session_id")
        if isinstance(metadata_session_id, str) and metadata_session_id.strip():
            return metadata_session_id.strip()

    context = payload.get("context")
    if isinstance(context, dict):
        context_session_id = context.get("session_id")
        if isinstance(context_session_id, str) and context_session_id.strip():
            return context_session_id.strip()

    return ""


def _upsert_session(
    connection: sqlite3.Connection,
    session_id: str,
    activity_time: datetime,
    ttl_minutes: int = SESSION_TTL_MINUTES,
) -> None:
    expires_at = activity_time + timedelta(minutes=ttl_minutes)
    activity_iso = _iso(activity_time)
    expires_iso = _iso(expires_at)

    connection.execute(
        """
        INSERT INTO sessions (
            session_id,
            started_at,
            last_activity,
            expires_at,
            status,
            close_reason
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            last_activity = excluded.last_activity,
            expires_at = excluded.expires_at,
            status = ?,
            close_reason = NULL
        """,
        (
            session_id,
            activity_iso,
            activity_iso,
            expires_iso,
            "active",
            None,
            "active",
        ),
    )


def _expire_inactive_sessions(connection: sqlite3.Connection, reference_time: datetime) -> None:
    connection.execute(
        """
        UPDATE sessions
        SET status = ?, close_reason = ?
        WHERE status = ?
                    AND expires_at <= ?
        """,
        ("expired", "inactive_ttl_expired", "active", _iso(reference_time)),
    )


def log_session_event(
    session_id: str,
    source: str,
    step: str,
    payload: dict[str, Any],
    event_timestamp: str | None,
    session_jsonl_file: str | None,
) -> None:
    if not session_id:
        return

    event_time = _parse_timestamp(event_timestamp)
    created_at = _iso()
    payload_json = json.dumps(payload, ensure_ascii=True)

    try:
        with _connect() as connection:
            _ensure_schema(connection)
            _upsert_session(connection, session_id, event_time)
            _expire_inactive_sessions(connection, event_time)
            connection.execute(
                """
                INSERT INTO session_events (
                    session_id,
                    event_time,
                    source,
                    step,
                    payload_json,
                    session_jsonl_file,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    _iso(event_time),
                    source,
                    step,
                    payload_json,
                    session_jsonl_file,
                    created_at,
                ),
            )
    except Exception:
        return


def set_session_status(session_id: str, status: str, reason: str | None = None) -> None:
    if not session_id:
        return
    if status not in {"active", "closed", "expired"}:
        return

    updated_at = datetime.now(timezone.utc)

    try:
        with _connect() as connection:
            _ensure_schema(connection)
            _upsert_session(connection, session_id, updated_at)
            connection.execute(
                """
                UPDATE sessions
                SET status = ?, close_reason = ?
                WHERE session_id = ?
                """,
                (status, reason, session_id),
            )
    except Exception:
        return


def log_output_record(file_path: Path, payload: dict[str, Any], category: str | None = None) -> None:
    record_time = _parse_timestamp(str(payload.get("timestamp") or ""))
    created_at = _iso()
    session_id = _extract_session_id(payload)

    if category and category.strip():
        normalized_category = category.strip().lower()
    else:
        normalized_category = file_path.stem.lower()

    try:
        with _connect() as connection:
            _ensure_schema(connection)
            if session_id:
                _upsert_session(connection, session_id, record_time)
                _expire_inactive_sessions(connection, record_time)
            connection.execute(
                """
                INSERT INTO output_records (
                    session_id,
                    category,
                    file_path,
                    record_time,
                    payload_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id or None,
                    normalized_category,
                    str(file_path),
                    _iso(record_time),
                    json.dumps(payload, ensure_ascii=True),
                    created_at,
                ),
            )
    except Exception:
        return


def _fetch_rows(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    try:
        with _connect() as connection:
            _ensure_schema(connection)
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception:
        return []


def get_session_status_counts() -> dict[str, int]:
    rows = _fetch_rows(
        """
        SELECT status, COUNT(*) AS count
        FROM sessions
        GROUP BY status
        """
    )

    counts = {"active": 0, "closed": 0, "expired": 0}
    total = 0
    for row in rows:
        status = str(row.get("status") or "").lower()
        count = int(row.get("count") or 0)
        if status in counts:
            counts[status] = count
        total += count

    counts["total"] = total
    return counts


def fetch_recent_sessions(limit: int = 20) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 200))
    return _fetch_rows(
        """
        SELECT
            session_id,
            status,
            started_at,
            last_activity AS last_activity_at,
            expires_at,
            close_reason
        FROM sessions
        ORDER BY last_activity DESC
        LIMIT ?
        """,
        (safe_limit,),
    )


def fetch_recent_session_events(limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 500))
    return _fetch_rows(
        """
        SELECT
            event_time,
            session_id,
            source,
            step,
            session_jsonl_file
        FROM session_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (safe_limit,),
    )
