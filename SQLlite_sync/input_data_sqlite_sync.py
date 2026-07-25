from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_FILE_PATTERN
from src.data_loader import list_available_files, load_single_file


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "Val_data"
DEFAULT_SQLITE_DB = PROJECT_ROOT / "data" / "SQLite_storage" / "retention_sessions.db"
INDIA_TZ = ZoneInfo("Asia/Kolkata")


def to_time_text(value):
    if pd.isna(value):
        return value

    timestamp = pd.to_datetime(value)
    if getattr(timestamp, "tzinfo", None) is None:
        timestamp = timestamp.tz_localize(INDIA_TZ)
    else:
        timestamp = timestamp.tz_convert(INDIA_TZ)

    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def IST_now_iso() -> str:
    return to_time_text(datetime.now(tz=INDIA_TZ))


def file_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_validation_data (
            customer_id TEXT PRIMARY KEY,
            source_file TEXT NOT NULL,
            source_modified_time TEXT,
            ingested_at_time TEXT NOT NULL,
            record_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS input_file_registry (
            file_path TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            modified_time TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            processed_at_time TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_validation_source_file ON customer_validation_data(source_file)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_registry_processed_at ON input_file_registry(processed_at_time DESC)"
    )


def parse_file_meta(file_path: Path) -> dict[str, Any]:
    stats = file_path.stat()
    modified_time = to_time_text(datetime.fromtimestamp(stats.st_mtime))
    return {
        "file_path": str(file_path.resolve()),
        "file_name": file_path.name,
        "file_size": int(stats.st_size),
        "modified_time": modified_time,
        "sha256": file_sha256(file_path),
    }


def is_file_changed(conn: sqlite3.Connection, meta: dict[str, Any]) -> bool:
    row = conn.execute(
        """
        SELECT file_size, modified_time, sha256, status
        FROM input_file_registry
        WHERE file_path = ?
        """,
        (meta["file_path"],),
    ).fetchone()

    if row is None:
        return True

    same_signature = (
        int(row["file_size"]) == int(meta["file_size"])
        and str(row["modified_time"]) == str(meta["modified_time"])
        and str(row["sha256"]) == str(meta["sha256"])
    )
    if same_signature and str(row["status"]) == "ok":
        return False
    return True


def upsert_registry(
    conn: sqlite3.Connection,
    *,
    meta: dict[str, Any],
    row_count: int,
    status: str,
    error_message: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO input_file_registry (
            file_path,
            file_name,
            file_size,
            modified_time,
            sha256,
            row_count,
            processed_at_time,
            status,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_path) DO UPDATE SET
            file_name = excluded.file_name,
            file_size = excluded.file_size,
            modified_time = excluded.modified_time,
            sha256 = excluded.sha256,
            row_count = excluded.row_count,
            processed_at_time = excluded.processed_at_time,
            status = excluded.status,
            error_message = excluded.error_message
        """,
        (
            meta["file_path"],
            meta["file_name"],
            int(meta["file_size"]),
            meta["modified_time"],
            meta["sha256"],
            int(row_count),
            IST_now_iso(),
            status,
            error_message,
        ),
    )


def ingest_one_file(conn: sqlite3.Connection, file_path: Path) -> dict[str, Any]:
    meta = parse_file_meta(file_path)

    try:
        frame = load_single_file(file_path)
        if frame.empty:
            upsert_registry(conn, meta=meta, row_count=0, status="skipped_empty", error_message=None)
            return {"file": str(file_path), "status": "skipped_empty", "rows": 0, "inserted": 0, "updated": 0}

        if "customer_id" not in frame.columns:
            upsert_registry(
                conn,
                meta=meta,
                row_count=0,
                status="error",
                error_message="customer_id column missing after normalization",
            )
            return {
                "file": str(file_path),
                "status": "error",
                "rows": 0,
                "inserted": 0,
                "updated": 0,
                "error": "customer_id column missing",
            }

        existing_ids = {
            str(row[0])
            for row in conn.execute("SELECT customer_id FROM customer_validation_data").fetchall()
        }

        inserted = 0
        updated = 0

        for _, row in frame.iterrows():
            customer_id = str(row.get("customer_id", "")).strip()
            if not customer_id:
                continue

            record_payload = {col: row[col] for col in frame.columns}
            record_json = json.dumps(record_payload, ensure_ascii=True, default=str)

            if customer_id in existing_ids:
                updated += 1
            else:
                inserted += 1
                existing_ids.add(customer_id)

            conn.execute(
                """
                INSERT INTO customer_validation_data (
                    customer_id,
                    source_file,
                    source_modified_time,
                    ingested_at_time,
                    record_json
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(customer_id) DO UPDATE SET
                    source_file = excluded.source_file,
                    source_modified_time = excluded.source_modified_time,
                    ingested_at_time = excluded.ingested_at_time,
                    record_json = excluded.record_json
                """,
                (
                    customer_id,
                    meta["file_name"],
                    meta["modified_time"],
                    IST_now_iso(),
                    record_json,
                ),
            )

        upsert_registry(conn, meta=meta, row_count=int(len(frame)), status="ok", error_message=None)
        return {
            "file": str(file_path),
            "status": "ok",
            "rows": int(len(frame)),
            "inserted": inserted,
            "updated": updated,
        }
    except Exception as exc:
        upsert_registry(conn, meta=meta, row_count=0, status="error", error_message=str(exc))
        return {
            "file": str(file_path),
            "status": "error",
            "rows": 0,
            "inserted": 0,
            "updated": 0,
            "error": str(exc),
        }


def ingest_input_data(input_dir: Path, db_path: Path, force: bool) -> dict[str, Any]:
    dataset_files = list_available_files(input_dir, DATA_FILE_PATTERN)
    summary = {
        "db_path": str(db_path),
        "input_dir": str(input_dir),
        "files_found": len(dataset_files),
        "files_processed": 0,
        "files_skipped_unchanged": 0,
        "rows_total": 0,
        "inserted_total": 0,
        "updated_total": 0,
        "results": [],
    }

    with connect(db_path) as conn:
        ensure_schema(conn)

        for file_path in dataset_files:
            meta = parse_file_meta(file_path)
            if not force and not is_file_changed(conn, meta):
                summary["files_skipped_unchanged"] += 1
                summary["results"].append(
                    {
                        "file": str(file_path),
                        "status": "skipped_unchanged",
                        "rows": 0,
                        "inserted": 0,
                        "updated": 0,
                    }
                )
                continue

            result = ingest_one_file(conn, file_path)
            summary["files_processed"] += 1
            summary["rows_total"] += int(result.get("rows", 0))
            summary["inserted_total"] += int(result.get("inserted", 0))
            summary["updated_total"] += int(result.get("updated", 0))
            summary["results"].append(result)

    return summary


def fetch_customer_json(db_path: Path, customer_id: str) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        ensure_schema(conn)
        row = conn.execute(
            """
            SELECT customer_id, source_file, source_modified_time, ingested_at_time, record_json
            FROM customer_validation_data
            WHERE lower(customer_id) = lower(?)
            """,
            (customer_id.strip(),),
        ).fetchone()

        if row is None:
            return None

        payload = json.loads(str(row["record_json"]))
        return {
            "customer_id": row["customer_id"],
            "source_file": row["source_file"],
            "source_modified_time": row["source_modified_time"],
            "ingested_at_time": row["ingested_at_time"],
            "record": payload,
        }


def fetch_validation_json(db_path: Path, limit: int = 100) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 5000))
    with connect(db_path) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT customer_id, source_file, source_modified_time, ingested_at_time, record_json
            FROM customer_validation_data
            ORDER BY ingested_at_time DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "customer_id": row["customer_id"],
                "source_file": row["source_file"],
                "source_modified_time": row["source_modified_time"],
                "ingested_at_time": row["ingested_at_time"],
                "record": json.loads(str(row["record_json"])),
            }
        )
    return result


def fetch_registry(db_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 5000))
    with connect(db_path) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT file_name, file_path, modified_time, row_count, processed_at_time, status, error_message
            FROM input_file_registry
            ORDER BY processed_at_time DESC
            LIMIT ?
            """,
            (safe_limit,),
        ).fetchall()

    return [dict(row) for row in rows]


def fetch_tables(db_path: Path) -> list[dict[str, Any]]:
    with connect(db_path) as conn:
        ensure_schema(conn)
        table_rows = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()

        tables: list[dict[str, Any]] = []
        for table_row in table_rows:
            table_name = str(table_row["name"])
            count_row = conn.execute(f"SELECT COUNT(*) AS row_count FROM {table_name}").fetchone()
            schema_rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            columns = [
                {
                    "name": str(schema_row["name"]),
                    "type": str(schema_row["type"]),
                    "not_null": bool(schema_row["notnull"]),
                    "primary_key": bool(schema_row["pk"]),
                }
                for schema_row in schema_rows
            ]
            tables.append(
                {
                    "table": table_name,
                    "row_count": int(count_row["row_count"]) if count_row else 0,
                    "columns": columns,
                }
            )

    return tables


def write_json_output(payload: Any, output_file: Path | None) -> None:
    data = json.dumps(payload, ensure_ascii=True, indent=2)
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(data, encoding="utf-8")
    else:
        print(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sync data/Val_data into SQLite by customer_id and query JSON validation data."
    )
    parser.add_argument("--db", default=str(DEFAULT_SQLITE_DB), help="SQLite DB path")

    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest new/changed files from Val_data")
    ingest_parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Input data directory")
    ingest_parser.add_argument("--force", action="store_true", help="Reprocess all files")
    ingest_parser.add_argument("--output", help="Optional JSON output file path")

    customer_parser = subparsers.add_parser("customer", help="Fetch one customer record as JSON")
    customer_parser.add_argument("--customer-id", required=True, help="Customer ID to fetch")
    customer_parser.add_argument("--output", help="Optional JSON output file path")

    export_parser = subparsers.add_parser("export-json", help="Export validation data from SQLite as JSON")
    export_parser.add_argument("--limit", type=int, default=100, help="Max rows to export")
    export_parser.add_argument("--output", help="Optional JSON output file path")

    registry_parser = subparsers.add_parser("registry", help="Show file ingestion registry")
    registry_parser.add_argument("--limit", type=int, default=50, help="Max registry rows")
    registry_parser.add_argument("--output", help="Optional JSON output file path")

    tables_parser = subparsers.add_parser("tables", help="Show SQLite tables with row counts and schema")
    tables_parser.add_argument("--output", help="Optional JSON output file path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not getattr(args, "command", None):
        parser.print_help()
        print("\nExamples:")
        print("  ingest      : python SQLlite_sync/input_data_sqlite_sync.py ingest")
        print("  customer    : python SQLlite_sync/input_data_sqlite_sync.py customer --customer-id TC-000171")
        print("  export-json : python SQLlite_sync/input_data_sqlite_sync.py export-json --limit 100")
        print("  registry    : python SQLlite_sync/input_data_sqlite_sync.py registry --limit 20")
        print("  tables      : python SQLlite_sync/input_data_sqlite_sync.py tables")
        return

    db_path = Path(args.db)

    if args.command == "ingest":
        summary = ingest_input_data(
            input_dir=Path(args.input_dir),
            db_path=db_path,
            force=bool(args.force),
        )
        write_json_output(summary, Path(args.output) if args.output else None)
        return

    if args.command == "customer":
        payload = fetch_customer_json(db_path=db_path, customer_id=str(args.customer_id))
        if payload is None:
            write_json_output({"status": "not_found", "customer_id": args.customer_id}, Path(args.output) if args.output else None)
        else:
            write_json_output(payload, Path(args.output) if args.output else None)
        return

    if args.command == "export-json":
        payload = fetch_validation_json(db_path=db_path, limit=int(args.limit))
        write_json_output(payload, Path(args.output) if args.output else None)
        return

    if args.command == "registry":
        payload = fetch_registry(db_path=db_path, limit=int(args.limit))
        write_json_output(payload, Path(args.output) if args.output else None)
        return

    if args.command == "tables":
        payload = fetch_tables(db_path=db_path)
        write_json_output(payload, Path(args.output) if args.output else None)
        return


if __name__ == "__main__":
    main()
