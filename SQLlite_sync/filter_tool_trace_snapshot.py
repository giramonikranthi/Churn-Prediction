from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _iter_session_events(log_file: Path):
    with log_file.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield line_no, event


def _summarize_tool(trace_item: dict[str, Any]) -> str:
    tool_name = str(trace_item.get("tool") or "unknown")
    result = trace_item.get("result")
    if not isinstance(result, dict):
        return f"{tool_name}: result=unavailable"

    status = str(result.get("status") or "unknown")
    model_source = result.get("model_source")
    if model_source:
        return f"{tool_name}: {status} (model_source={model_source})"
    return f"{tool_name}: {status}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Print only tool_trace_snapshot events for a given retention session id."
        )
    )
    parser.add_argument("session_id", help="Session id, e.g. RET-C960567C43")
    parser.add_argument(
        "--logs-dir",
        default=str(Path(__file__).resolve().parents[1] / "logs"),
        help="Directory containing per-session JSONL logs (default: <project>/logs)",
    )
    parser.add_argument(
        "--as-json",
        action="store_true",
        help="Print matched snapshot payloads as pretty JSON",
    )

    args = parser.parse_args()
    session_id = str(args.session_id).strip()
    logs_dir = Path(args.logs_dir).resolve()

    if not session_id:
        print("Error: session_id cannot be empty.", file=sys.stderr)
        return 2

    if not logs_dir.exists() or not logs_dir.is_dir():
        print(f"Error: logs directory not found: {logs_dir}", file=sys.stderr)
        return 2

    log_file = logs_dir / f"{session_id}.jsonl"
    if not log_file.exists():
        print(f"No session log file found for {session_id}: {log_file}")
        return 0

    matched: list[tuple[int, dict[str, Any]]] = []
    for line_no, event in _iter_session_events(log_file):
        if event.get("step") == "tool_trace_snapshot":
            matched.append((line_no, event))

    print(f"Session: {session_id}")
    print(f"File: {log_file}")
    print(f"tool_trace_snapshot events found: {len(matched)}")

    if not matched:
        return 0

    for idx, (line_no, event) in enumerate(matched, start=1):
        timestamp = str(event.get("timestamp_ist") or "")
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        status = str(payload.get("status") or "unknown")
        trace = payload.get("tool_trace") if isinstance(payload.get("tool_trace"), list) else []

        print("")
        print(f"[{idx}] line={line_no} timestamp_ist={timestamp} status={status} tools={len(trace)}")

        if args.as_json:
            print(json.dumps(event, ensure_ascii=True, indent=2))
            continue

        for trace_item in trace:
            if not isinstance(trace_item, dict):
                continue
            print(f"  - {_summarize_tool(trace_item)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
